from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import median

from evaluate_pitch_tracking import parse_midi
from align_score_to_audio import (
    build_anchors,
    infer_time_scale_octave_and_offset,
    map_time,
    monotonic_matches,
    scale_notes,
)
from initial_score_alignment import align_first_vocal_note, apply_constant_offset
from pianoflow_server import ANALYSIS_VERSION, find_ffmpeg


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset"
DEFAULT_WORK = ROOT / "dataset-work"
SUPPORTED_AUDIO = {".mov", ".mp4", ".m4a", ".mp3", ".wav", ".flac", ".ogg"}


def normalized_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_variable(data: bytes, index: int) -> tuple[int, int]:
    value = 0
    while True:
        if index >= len(data):
            raise ValueError("MIDI可変長値が途中で終了しています")
        byte = data[index]
        index += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, index


def midi_meta(path: Path) -> dict:
    data = path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"標準MIDIではありません: {path.name}")
    header_size = int.from_bytes(data[4:8], "big")
    format_number = int.from_bytes(data[8:10], "big")
    track_count = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    if division & 0x8000:
        raise ValueError(f"SMPTE形式のMIDI時間軸には未対応です: {path.name}")

    position = 8 + header_size
    tempos: list[dict] = []
    signatures: list[dict] = []
    for track_index in range(track_count):
        if data[position:position + 4] != b"MTrk":
            raise ValueError(f"MIDIトラックが壊れています: {path.name}")
        size = int.from_bytes(data[position + 4:position + 8], "big")
        track = data[position + 8:position + 8 + size]
        position += 8 + size
        index = 0
        tick = 0
        running_status: int | None = None
        while index < len(track):
            delta, index = read_variable(track, index)
            tick += delta
            status = track[index]
            if status < 0x80:
                if running_status is None:
                    raise ValueError(f"MIDI running statusが不正です: {path.name}")
                status = running_status
            else:
                index += 1
                if status < 0xF0:
                    running_status = status
            if status == 0xFF:
                event_type = track[index]
                index += 1
                length, index = read_variable(track, index)
                payload = track[index:index + length]
                index += length
                if event_type == 0x51 and length == 3:
                    microseconds = int.from_bytes(payload, "big")
                    tempos.append({
                        "track": track_index,
                        "tick": tick,
                        "microsecondsPerQuarter": microseconds,
                        "bpm": round(60_000_000 / microseconds, 6),
                    })
                elif event_type == 0x58 and length >= 2:
                    signatures.append({
                        "track": track_index,
                        "tick": tick,
                        "numerator": payload[0],
                        "denominator": 2 ** payload[1],
                    })
                running_status = None
                continue
            if status in (0xF0, 0xF7):
                length, index = read_variable(track, index)
                index += length
                running_status = None
                continue
            event_type = status & 0xF0
            index += 1 if event_type in (0xC0, 0xD0) else 2
    return {
        "format": format_number,
        "tracks": track_count,
        "ticksPerQuarter": division,
        "tempoEvents": tempos,
        "timeSignatures": signatures,
    }


def audit_midi(path: Path) -> dict:
    notes, bpm = parse_midi(path)
    notes = sorted(notes, key=lambda row: (float(row["start"]), float(row["end"])))
    durations = [float(note["end"]) - float(note["start"]) for note in notes]
    maximum_end = -math.inf
    overlap_indices: list[int] = []
    for index, note in enumerate(notes):
        if float(note["start"]) < maximum_end - 1e-6:
            overlap_indices.append(index)
        maximum_end = max(maximum_end, float(note["end"]))
    same_pitch_reattacks = [
        index for index in range(1, len(notes))
        if int(notes[index]["midi"]) == int(notes[index - 1]["midi"])
    ]
    meta = midi_meta(path)
    warnings: list[str] = []
    if overlap_indices:
        warnings.append(f"重複音符が{len(overlap_indices)}か所あります")
    if not meta["tempoEvents"]:
        warnings.append("テンポイベントがなく、120 BPM既定値として解釈されます")
    if len(meta["tempoEvents"]) > 1:
        warnings.append(f"テンポイベントが{len(meta['tempoEvents'])}個あります")
    if any(duration < 0.04 for duration in durations):
        warnings.append("40 ms未満の非常に短い音符があります")
    return {
        **meta,
        "initialBpm": round(bpm, 6),
        "noteCount": len(notes),
        "durationSeconds": round(maximum_end if notes else 0.0, 6),
        "pitchMinimum": min((int(note["midi"]) for note in notes), default=None),
        "pitchMaximum": max((int(note["midi"]) for note in notes), default=None),
        "overlapCount": len(overlap_indices),
        "overlapNoteIndices": overlap_indices,
        "samePitchReattackCount": len(same_pitch_reattacks),
        "samePitchReattackIndices": same_pitch_reattacks,
        "shorterThan40msCount": sum(duration < 0.04 for duration in durations),
        "warnings": warnings,
    }


def parse_combined_lyrics(path: Path, song_names: list[str]) -> dict[str, list[str]]:
    lookup = {normalized_name(name): name for name in song_names}
    result = {name: [] for name in song_names}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        key = normalized_name(line)
        if not line or key == normalized_name("歌詞"):
            continue
        if key in lookup:
            current = lookup[key]
            continue
        if current is not None:
            result[current].append(line)
    return result


def command_prepare(args: argparse.Namespace) -> int:
    dataset = args.dataset.resolve()
    work = args.work.resolve()
    if not dataset.is_dir():
        raise ValueError(f"datasetフォルダーがありません: {dataset}")
    midi_by_name = {normalized_name(path.stem): path for path in dataset.glob("*.mid")}
    audio_by_name: dict[str, Path] = {}
    for path in dataset.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO:
            key = normalized_name(path.stem)
            if key in audio_by_name:
                raise ValueError(f"同じ曲名の音源が複数あります: {path.stem}")
            audio_by_name[key] = path
    all_keys = sorted(set(midi_by_name) | set(audio_by_name))
    missing = [
        key for key in all_keys
        if key not in midi_by_name or key not in audio_by_name
    ]
    if missing:
        raise ValueError(f"MOV/MIDIの組が不足しています: {', '.join(missing)}")

    display_names = [audio_by_name[key].stem for key in all_keys]
    lyrics_source = dataset / args.lyrics
    if not lyrics_source.exists():
        raise ValueError(f"歌詞ファイルがありません: {lyrics_source}")
    lyrics = parse_combined_lyrics(lyrics_source, display_names)
    work.mkdir(parents=True, exist_ok=True)
    songs_root = work / "songs"
    songs_root.mkdir(parents=True, exist_ok=True)
    songs: list[dict] = []
    for index, key in enumerate(all_keys, start=1):
        audio = audio_by_name[key]
        midi = midi_by_name[key]
        name = audio.stem
        song_id = f"{index:02d}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]}"
        song_dir = songs_root / song_id
        song_dir.mkdir(parents=True, exist_ok=True)
        lyric_lines = lyrics.get(name, [])
        lyric_path = song_dir / "lyrics.txt"
        lyric_path.write_text("\n".join(lyric_lines).strip() + "\n", encoding="utf-8")
        audit = audit_midi(midi)
        reference = {
            "schemaVersion": 1,
            "id": song_id,
            "name": name,
            "audio": relative_to_root(audio),
            "midi": relative_to_root(midi),
            "lyrics": lyric_lines,
            "lyricsFile": relative_to_root(lyric_path),
            "bpm": audit["initialBpm"],
            "midiAudit": audit,
        }
        reference_path = song_dir / "reference.json"
        reference_path.write_text(
            json.dumps(reference, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        songs.append({**reference, "referenceFile": relative_to_root(reference_path)})

    missing_lyrics = [song["name"] for song in songs if not song["lyrics"]]
    manifest = {
        "schemaVersion": 1,
        "analysisVersion": ANALYSIS_VERSION,
        "dataset": relative_to_root(dataset),
        "work": relative_to_root(work),
        "lyricsSource": relative_to_root(lyrics_source),
        "songCount": len(songs),
        "missingLyrics": missing_lyrics,
        "songs": songs,
    }
    manifest_path = work / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# PianoFlowデータセット検査",
        "",
        f"- 曲数: {len(songs)}",
        f"- 解析基準: ver{ANALYSIS_VERSION}",
        f"- 歌詞なし: {', '.join(missing_lyrics) if missing_lyrics else 'なし'}",
        "",
        "| 曲名 | BPM | 音符 | 長さ(秒) | 音域 | 重複 | 同音再発音 | 警告 |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for song in songs:
        audit = song["midiAudit"]
        warnings = " / ".join(audit["warnings"]) or "なし"
        lines.append(
            f"| {song['name']} | {audit['initialBpm']:.3f} | {audit['noteCount']} | "
            f"{audit['durationSeconds']:.3f} | {audit['pitchMinimum']}–{audit['pitchMaximum']} | "
            f"{audit['overlapCount']} | {audit['samePitchReattackCount']} | {warnings} |"
        )
    (work / "dataset-audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "songCount": len(songs),
        "missingLyrics": missing_lyrics,
        "warnings": {
            song["name"]: song["midiAudit"]["warnings"]
            for song in songs if song["midiAudit"]["warnings"]
        },
    }, ensure_ascii=False, indent=2))
    return 0


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schemaVersion", 0)) != 1:
        raise ValueError("未対応のmanifest形式です")
    return payload


def selected_songs(manifest: dict, requested: list[str]) -> list[dict]:
    if not requested:
        return list(manifest["songs"])
    keys = {normalized_name(value) for value in requested}
    songs = [
        song for song in manifest["songs"]
        if normalized_name(song["name"]) in keys or normalized_name(song["id"]) in keys
    ]
    found = {normalized_name(song["name"]) for song in songs} | {
        normalized_name(song["id"]) for song in songs
    }
    missing = [value for value in requested if normalized_name(value) not in found]
    if missing:
        raise ValueError(f"曲が見つかりません: {', '.join(missing)}")
    return songs


def analysis_mode(args: argparse.Namespace) -> str:
    parts = ["audio-only"]
    if args.use_lyrics:
        parts.append("lyrics")
    if args.use_midi_bpm:
        parts.append("midi-bpm")
    return "+".join(parts)


def command_analyze(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest.resolve())
    songs = selected_songs(manifest, args.song)
    ffmpeg = args.ffmpeg or find_ffmpeg()
    if not ffmpeg:
        raise ValueError("FFmpegが見つかりません")
    ai_python = args.python or str(ROOT / ".venv-pianoflow" / "Scripts" / "python.exe")
    if not Path(ai_python).exists():
        raise ValueError(f"PianoFlow Pythonが見つかりません: {ai_python}")
    mode = analysis_mode(args)
    failures: list[dict] = []
    print(f"対象={len(songs)}曲 mode={mode} analysisVersion={ANALYSIS_VERSION}", flush=True)
    for index, song in enumerate(songs, start=1):
        reference_file = resolve_root_path(song["referenceFile"])
        output = reference_file.parent / mode
        analysis_path = output / "analysis.json"
        if analysis_path.exists() and not args.force:
            existing = json.loads(analysis_path.read_text(encoding="utf-8"))
            if int(existing.get("analysisVersion", 0)) == ANALYSIS_VERSION:
                print(f"[{index}/{len(songs)}] 再利用: {song['name']}", flush=True)
                continue
        output.mkdir(parents=True, exist_ok=True)
        wav_path = output / "analysis-input.wav"
        print(f"[{index}/{len(songs)}] 音声変換: {song['name']}", flush=True)
        conversion = subprocess.run(
            [
                ffmpeg, "-y", "-i", str(resolve_root_path(song["audio"])),
                "-vn", "-ac", "2", "-ar", "44100", str(wav_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if conversion.returncode != 0:
            failure = {
                "name": song["name"],
                "stage": "ffmpeg",
                "error": conversion.stderr.decode("utf-8", errors="replace")[-2000:],
            }
            failures.append(failure)
            print(f"  失敗: {failure['error'][-240:]}", flush=True)
            if not args.continue_on_error:
                break
            continue
        command = [
            ai_python,
            str(ROOT / "ai_melody.py"),
            str(wav_path),
            "--output-dir",
            str(output),
        ]
        if args.use_lyrics:
            command.extend(["--lyrics-file", str(resolve_root_path(song["lyricsFile"]))])
        if args.use_midi_bpm:
            command.extend(["--bpm", str(song["bpm"])])
        print(f"[{index}/{len(songs)}] ver{ANALYSIS_VERSION}解析: {song['name']}", flush=True)
        stderr_path = output / "analysis-error.log"
        with stderr_path.open("w", encoding="utf-8") as error_log:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=error_log,
            )
        if result.returncode != 0 or not analysis_path.exists():
            failure = {
                "name": song["name"],
                "stage": "analysis",
                "error": stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:],
            }
            failures.append(failure)
            print(f"  失敗: {failure['error'][-240:]}", flush=True)
            if not args.continue_on_error:
                break
        else:
            stderr_path.unlink(missing_ok=True)
            if not args.keep_wav:
                wav_path.unlink(missing_ok=True)
            print(f"  完了: {analysis_path}", flush=True)
    summary = {
        "mode": mode,
        "requestedSongs": len(songs),
        "failureCount": len(failures),
        "failures": failures,
    }
    work = resolve_root_path(manifest["work"])
    (work / f"analysis-run-{mode}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def pair_by_onset(reference: list[dict], predicted: list[dict], tolerance: float) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for reference_index, expected in enumerate(reference):
        for predicted_index, actual in enumerate(predicted):
            error = abs(float(actual["start"]) - float(expected["start"]))
            if error <= tolerance:
                candidates.append((error, reference_index, predicted_index))
    result: list[tuple[int, int, float]] = []
    used_reference: set[int] = set()
    used_predicted: set[int] = set()
    for error, reference_index, predicted_index in sorted(candidates):
        if reference_index in used_reference or predicted_index in used_predicted:
            continue
        used_reference.add(reference_index)
        used_predicted.add(predicted_index)
        result.append((reference_index, predicted_index, error))
    return sorted(result)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    return ordered[left] * (right - position) + ordered[right] * (position - left)


def fixed_alignment_report(
    reference: list[dict],
    predicted: list[dict],
    *,
    tolerance: float,
    alignment: dict,
    predicted_to_reference_transpose: int,
) -> dict:
    pairs = pair_by_onset(reference, predicted, tolerance)
    used_reference = {row[0] for row in pairs}
    used_predicted = {row[1] for row in pairs}
    rows: list[dict] = []
    for reference_index, predicted_index, _ in pairs:
        expected = reference[reference_index]
        actual = predicted[predicted_index]
        onset_error_ms = (float(actual["start"]) - float(expected["start"])) * 1000.0
        duration_error_ms = (
            (float(actual["end"]) - float(actual["start"]))
            - (float(expected["end"]) - float(expected["start"]))
        ) * 1000.0
        raw_predicted_midi = int(actual["midi"])
        normalized_predicted_midi = raw_predicted_midi + predicted_to_reference_transpose
        rows.append({
            "referenceIndex": reference_index,
            "predictedIndex": predicted_index,
            "time": round(float(expected["start"]), 6),
            "expectedMidi": int(expected["midi"]),
            "rawPredictedMidi": raw_predicted_midi,
            "predictedMidi": normalized_predicted_midi,
            "rawPitchErrorSemitones": raw_predicted_midi - int(expected["midi"]),
            "pitchErrorSemitones": normalized_predicted_midi - int(expected["midi"]),
            "onsetErrorMs": round(onset_error_ms, 3),
            "durationErrorMs": round(duration_error_ms, 3),
        })
    correct = sum(row["pitchErrorSemitones"] == 0 for row in rows)
    octave = sum(abs(row["pitchErrorSemitones"]) in (12, 24) for row in rows)
    wrong = len(rows) - correct - octave
    missing = len(reference) - len(pairs)
    extra = len(predicted) - len(pairs)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    onset_errors = [abs(float(row["onsetErrorMs"])) for row in rows]
    signed_errors = [float(row["onsetErrorMs"]) for row in rows]
    same_pitch_reference = [
        index for index in range(1, len(reference))
        if int(reference[index]["midi"]) == int(reference[index - 1]["midi"])
    ]
    paired_reference = {row["referenceIndex"] for row in rows}
    detected_same_pitch = sum(index in paired_reference for index in same_pitch_reference)
    quartiles: list[dict] = []
    if rows:
        start = min(float(row["time"]) for row in rows)
        end = max(float(row["time"]) for row in rows)
        width = max(1e-6, (end - start) / 4.0)
        for index in range(4):
            left = start + width * index
            right = end + 1e-9 if index == 3 else start + width * (index + 1)
            region = [
                float(row["onsetErrorMs"]) for row in rows
                if left <= float(row["time"]) < right
            ]
            quartiles.append({
                "part": index + 1,
                "fromSeconds": round(left, 4),
                "toSeconds": round(right, 4),
                "pairedNotes": len(region),
                "medianSignedOnsetErrorMs": round(median(region), 3) if region else None,
            })
    return {
        "synchronization": alignment,
        "predictedToReferenceTransposeSemitones": predicted_to_reference_transpose,
        "pitchComparison": (
            "PianoFlow vocal-register MIDI is shifted to the reference piano-score register "
            "before pitch scoring; raw values remain in noteComparisons"
        ),
        "onsetToleranceMs": round(tolerance * 1000.0),
        "referenceNotes": len(reference),
        "predictedNotes": len(predicted),
        "pairedByOnset": len(pairs),
        "correctPitch": correct,
        "wrongPitch": wrong,
        "octaveErrors": octave,
        "missingNotes": missing,
        "extraNotes": extra,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "medianAbsoluteOnsetErrorMs": round(median(onset_errors), 3) if onset_errors else None,
        "p90AbsoluteOnsetErrorMs": round(percentile(onset_errors, 0.9), 3) if onset_errors else None,
        "medianSignedOnsetErrorMs": round(median(signed_errors), 3) if signed_errors else None,
        "samePitchReattacks": len(same_pitch_reference),
        "detectedSamePitchReattacks": detected_same_pitch,
        "samePitchReattackRecall": round(
            detected_same_pitch / len(same_pitch_reference), 4
        ) if same_pitch_reference else None,
        "timingBySongQuarter": quartiles,
        "noteComparisons": rows,
        "unmatchedReferenceIndices": sorted(set(range(len(reference))) - used_reference),
        "unmatchedPredictedIndices": sorted(set(range(len(predicted))) - used_predicted),
    }


def command_evaluate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest.resolve())
    songs = selected_songs(manifest, args.song)
    mode = args.mode
    work = resolve_root_path(manifest["work"])
    alignment_rules_path = ROOT / "dataset-alignment-rules.json"
    alignment_rules = (
        json.loads(alignment_rules_path.read_text(encoding="utf-8"))
        if alignment_rules_path.exists()
        else {}
    )
    overrides_path = work / "alignment-overrides.json"
    overrides = (
        json.loads(overrides_path.read_text(encoding="utf-8"))
        if overrides_path.exists() and not args.ignore_overrides
        else {}
    )
    summaries: list[dict] = []
    failures: list[dict] = []
    for index, song in enumerate(songs, start=1):
        reference_file = resolve_root_path(song["referenceFile"])
        output = reference_file.parent / mode
        analysis_path = output / "analysis.json"
        if not analysis_path.exists():
            failures.append({"name": song["name"], "error": f"解析結果なし: {analysis_path}"})
            print(f"[{index}/{len(songs)}] 未解析: {song['name']}", flush=True)
            if not args.continue_on_error:
                break
            continue
        reference, midi_bpm = parse_midi(resolve_root_path(song["midi"]))
        song_alignment_rule = alignment_rules.get(
            song["id"],
            alignment_rules.get(song["name"], {}),
        )
        ignored_opening_notes = max(
            0,
            int(song_alignment_rule.get("ignoreOpeningMidiNotes", 0)),
        )
        if ignored_opening_notes >= len(reference):
            failures.append({
                "name": song["name"],
                "error": "同期規則によってMIDI音符がすべて除外されます",
            })
            if not args.continue_on_error:
                break
            continue
        evaluation_reference = reference[ignored_opening_notes:]
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        predicted = [
            note for note in payload.get("notes", [])
            if note.get("role", "melody") == "melody"
        ]
        try:
            initial = align_first_vocal_note(
                evaluation_reference,
                predicted,
                syllable_segments=payload.get("syllableSegments", []),
            )
        except ValueError as error:
            failures.append({"name": song["name"], "error": str(error)})
            if not args.continue_on_error:
                break
            continue
        automatic_offset = initial.offset_seconds
        override_value = overrides.get(song["id"])
        selected_offset = (
            float(override_value["offsetSeconds"])
            if isinstance(override_value, dict) and override_value.get("offsetSeconds") is not None
            else automatic_offset
        )
        analysis_bpm = payload.get("bpm")
        reference_span = max(
            0.001,
            max((float(note["end"]) for note in evaluation_reference), default=0.001)
            - min((float(note["start"]) for note in evaluation_reference), default=0.0),
        )
        predicted_span = max(
            0.001,
            max((float(note["end"]) for note in predicted), default=0.001)
            - min((float(note["start"]) for note in predicted), default=0.0),
        )
        duration_ratio = predicted_span / reference_span
        try:
            expected_scale = float(midi_bpm) / max(1.0, float(analysis_bpm))
        except (TypeError, ValueError):
            expected_scale = duration_ratio

        same_recording_clock = (
            (analysis_bpm is not None and abs(expected_scale - 1.0) <= 0.03)
            or abs(duration_ratio - 1.0) <= 0.07
        )
        if same_recording_clock:
            # The MIDI and clip already use the same real-time clock. BPM
            # detectors commonly return 2:3 or 1:2 aliases, while matching clip
            # duration exposes that no global rescaling is actually needed.
            time_scale = 1.0
            diagnostic_transpose = int(initial.diagnostic_transpose)
        else:
            time_scale, diagnostic_transpose, _estimated_offset = infer_time_scale_octave_and_offset(
                evaluation_reference,
                predicted,
                expected_scale,
            )
        scaled_reference = scale_notes(evaluation_reference, time_scale)
        pairs = monotonic_matches(
            scaled_reference,
            predicted,
            diagnostic_transpose,
            selected_offset,
        )
        anchors = build_anchors(
            scaled_reference,
            predicted,
            pairs,
            diagnostic_transpose,
            selected_offset,
        )
        # The reviewed/automatic first sung onset remains the hard start anchor.
        # Later anchors are coarse (four-second score regions), so MIDI is never
        # snapped note-by-note to the current prediction.
        first_evaluated_score_time = min(
            (float(note["start"]) for note in scaled_reference),
            default=0.0,
        )
        hard_start_anchors = [{
            "scoreTime": 0.0,
            "audioTime": round(selected_offset, 6),
            "offset": round(selected_offset, 6),
            "support": int(initial.matched_sequence_notes),
            "kind": "midi-zero",
        }]
        if first_evaluated_score_time > 0.001:
            hard_start_anchors.append({
                "scoreTime": round(first_evaluated_score_time, 6),
                "audioTime": round(first_evaluated_score_time + selected_offset, 6),
                "offset": round(selected_offset, 6),
                "support": int(initial.matched_sequence_notes),
                "kind": "first-evaluated-midi-note",
            })
        anchors = [
            *hard_start_anchors,
            *[
                {**anchor, "kind": "coarse-performance-anchor"}
                for anchor in anchors
                if float(anchor["scoreTime"]) > first_evaluated_score_time + 0.25
            ],
        ]
        aligned = [
            {
                **note,
                "start": round(map_time(float(note["start"]), anchors), 6),
                "end": round(map_time(float(note["end"]), anchors), 6),
            }
            for note in scaled_reference
        ]
        alignment = {
            **initial.as_dict(),
            "referenceAnchorIndex": ignored_opening_notes,
            "ignoredOpeningMidiNotes": ignored_opening_notes,
            "alignmentRuleReason": song_alignment_rule.get("reason"),
            "automaticInitialNoteOffsetSeconds": round(automatic_offset, 6),
            "audioTimeAtMidiZero": round(selected_offset, 6),
            "initialNoteOffsetSeconds": round(selected_offset, 6),
            "manualOverrideApplied": selected_offset != automatic_offset,
            "midiBpm": round(midi_bpm, 6),
            "analysisBpm": analysis_bpm,
            "timeScale": round(time_scale, 8),
            "diagnosticTransposeSemitones": diagnostic_transpose,
            "fixedMidiClock": False,
            "localWarpApplied": len(anchors) > 1,
            "alignmentGranularitySeconds": 4.0,
            "performanceAnchors": anchors,
            "alignmentUsesFinalBoundaries": True,
            "warning": (
                "Only the reference MIDI time axis is mapped to the audio performance. "
                "The analyzed notes are not moved. Coarse anchors avoid note-level "
                "snapping, so local boundary errors remain measurable."
            ),
        }
        report = fixed_alignment_report(
            aligned,
            predicted,
            tolerance=args.tolerance_ms / 1000.0,
            alignment=alignment,
            predicted_to_reference_transpose=args.predicted_to_midi_transpose,
        )
        alignment_payload = {
            "name": song["name"],
            "groundTruth": (
                "MIDI mapped to the audio performance with a global time scale and "
                "coarse phrase-level anchors; analyzed-note timing is unchanged"
            ),
            "ignoredOpeningMidiNotes": ignored_opening_notes,
            "alignment": alignment,
            "timeScale": round(time_scale, 8),
            "anchors": anchors,
            "alignedNotes": aligned,
        }
        (output / "initial-alignment.json").write_text(
            json.dumps(alignment_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "baseline-evaluation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        compact = {
            key: value for key, value in report.items()
            if key not in {"noteComparisons", "unmatchedReferenceIndices", "unmatchedPredictedIndices"}
        }
        compact["name"] = song["name"]
        summaries.append(compact)
        print(
            f"[{index}/{len(songs)}] {song['name']}: "
            f"F1={report['f1']:.4f} onset中央値={report['medianAbsoluteOnsetErrorMs']}ms "
            f"欠落={report['missingNotes']} 余計={report['extraNotes']}",
            flush=True,
        )

    if args.song:
        # A one-song refresh must not collapse the review UI to that song.
        refreshed_by_name = {row["name"]: row for row in summaries}
        merged_summaries = []
        for manifest_song in manifest["songs"]:
            if manifest_song["name"] in refreshed_by_name:
                merged_summaries.append(refreshed_by_name[manifest_song["name"]])
                continue
            reference_file = resolve_root_path(manifest_song["referenceFile"])
            existing_report_path = reference_file.parent / mode / "baseline-evaluation.json"
            if not existing_report_path.exists():
                continue
            existing_report = json.loads(existing_report_path.read_text(encoding="utf-8"))
            compact = {
                key: value for key, value in existing_report.items()
                if key not in {
                    "noteComparisons",
                    "unmatchedReferenceIndices",
                    "unmatchedPredictedIndices",
                }
            }
            compact["name"] = manifest_song["name"]
            merged_summaries.append(compact)
        summaries = merged_summaries

    aggregate = {
        "mode": mode,
        "songCount": len(summaries),
        "failureCount": len(failures),
        "medianSongF1": round(median([row["f1"] for row in summaries]), 4) if summaries else None,
        "medianSongOnsetErrorMs": round(median([
            row["medianAbsoluteOnsetErrorMs"]
            for row in summaries if row["medianAbsoluteOnsetErrorMs"] is not None
        ]), 3) if any(row["medianAbsoluteOnsetErrorMs"] is not None for row in summaries) else None,
        "totals": {
            key: sum(int(row[key]) for row in summaries)
            for key in (
                "referenceNotes", "predictedNotes", "correctPitch", "wrongPitch",
                "octaveErrors", "missingNotes", "extraNotes",
            )
        },
        "songs": summaries,
        "failures": failures,
    }
    output_path = work / f"baseline-summary-{mode}.json"
    output_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"集計: {output_path}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PianoFlow ver39データセット準備・重い解析・冒頭同期・基準評価"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="台帳・歌詞分割・MIDI検査（軽い処理）")
    prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--work", type=Path, default=DEFAULT_WORK)
    prepare.add_argument("--lyrics", default="歌詞.txt")
    prepare.set_defaults(handler=command_prepare)

    analyze = subparsers.add_parser("analyze", help="MOVをver39で解析（重い処理）")
    analyze.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "manifest.json")
    analyze.add_argument("--song", action="append", default=[], help="曲名またはID。省略時は全曲")
    analyze.add_argument("--use-lyrics", action="store_true", help="実験用: 正解歌詞を解析へ渡す")
    analyze.add_argument("--use-midi-bpm", action="store_true", help="実験用: MIDI BPMを解析へ渡す")
    analyze.add_argument("--force", action="store_true")
    analyze.add_argument("--continue-on-error", action="store_true")
    analyze.add_argument("--keep-wav", action="store_true")
    analyze.add_argument("--ffmpeg")
    analyze.add_argument("--python")
    analyze.set_defaults(handler=command_analyze)

    evaluate = subparsers.add_parser("evaluate", help="冒頭同期してver39基準誤差を集計（軽い処理）")
    evaluate.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "manifest.json")
    evaluate.add_argument("--mode", default="audio-only")
    evaluate.add_argument("--song", action="append", default=[], help="曲名またはID。省略時は全曲")
    evaluate.add_argument("--tolerance-ms", type=float, default=120.0)
    evaluate.add_argument(
        "--predicted-to-midi-transpose",
        type=int,
        default=12,
        help="PianoFlowの声域MIDIを正解ピアノ譜の音域へ移す値（現在のデータは+12）",
    )
    evaluate.add_argument("--continue-on-error", action="store_true")
    evaluate.add_argument("--ignore-overrides", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
