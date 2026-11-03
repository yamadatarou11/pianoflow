from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt


BOUNDARY_FIRST_SEGMENTATION = os.environ.get("PIANOFLOW_BOUNDARY_FIRST", "1").strip().lower() not in {"0", "false", "off"}


def _separate_demucs(source: Path, output_root: Path) -> tuple[Path, dict[str, Path], str]:
    # Demucs is only needed when there is no reusable high-quality stem set.
    # Keep cached-stem inference usable in environments that do not install it.
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    from demucs.pretrained import get_model

    output_root.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(source, always_2d=True, dtype="float32")
    mix = torch.from_numpy(audio.T)
    model = get_model("mdx_q")
    model.eval()
    mix = convert_audio(mix, sample_rate, model.samplerate, model.audio_channels)
    reference = mix.mean(0)
    mean = reference.mean()
    std = reference.std().clamp_min(1e-6)
    normalized = (mix - mean) / std
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        separated = apply_model(
            model,
            normalized[None],
            device=device,
            shifts=0,
            split=True,
            overlap=0.1,
            progress=False,
            num_workers=0,
        )[0]
    separated = separated.cpu() * std + mean
    stem_paths: dict[str, Path] = {}
    for index, stem_name in enumerate(model.sources):
        stem_path = output_root / f"{stem_name}.wav"
        sf.write(stem_path, separated[index].T.numpy(), model.samplerate, subtype="PCM_16")
        stem_paths[stem_name] = stem_path
    instrumental_audio = sum(
        separated[index] for index, stem_name in enumerate(model.sources) if stem_name != "vocals"
    )
    instrumental_path = output_root / "instrumental.wav"
    sf.write(instrumental_path, instrumental_audio.T.numpy(), model.samplerate, subtype="PCM_16")
    stem_paths["instrumental"] = instrumental_path

    vocals_audio = separated[model.sources.index("vocals")].mean(0).numpy()
    analysis_rate = 22050
    centered = librosa.resample(vocals_audio, orig_sr=model.samplerate, target_sr=analysis_rate)
    sos = butter(5, [100, 3800], btype="bandpass", fs=analysis_rate, output="sos")
    centered = sosfiltfilt(sos, centered)
    harmonic, _ = librosa.effects.hpss(centered, margin=(1.0, 3.0))
    peak = float(np.max(np.abs(harmonic))) if harmonic.size else 0.0
    if peak > 0:
        harmonic = harmonic / max(1.0, peak * 1.05)
    lead_path = output_root / "lead-vocal-candidate.wav"
    sf.write(lead_path, harmonic, analysis_rate, subtype="PCM_16")
    backing = centered - harmonic
    backing_path = output_root / "backing-vocal-candidate.wav"
    sf.write(backing_path, backing, analysis_rate, subtype="PCM_16")
    stem_paths["lead"] = lead_path
    stem_paths["backing"] = backing_path
    return lead_path, stem_paths, "Demucs MDX fallback"


def separate_audio(source: Path, output_root: Path) -> tuple[Path, dict[str, Path], str]:
    """Prefer the maximum-quality lead-vocal chain installed for Windows.

    The high-quality environment is separate because current RoFormer releases
    require Python 3.10+ and PyTorch 2.x, while the lightweight fallback remains
    compatible with the original PianoFlow environment.
    """

    # Re-analysis changes the pitch decoder far more often than the expensive
    # source separation.  Reuse a complete stem set from the same source/job.
    reusable = {
        "vocals": output_root / "vocals.wav",
        "instrumental": output_root / "instrumental.wav",
        "lead": output_root / "lead-vocal.wav",
        "backing": output_root / "backing-vocals.wav",
        "dryLead": output_root / "lead-vocal-dry.wav",
        "reverb": output_root / "lead-vocal-reverb.wav",
    }
    required = ("vocals", "instrumental", "lead", "dryLead")
    if all(reusable[name].exists() and reusable[name].stat().st_size > 44 for name in required):
        return reusable["dryLead"], {name: path for name, path in reusable.items() if path.exists()}, "cached high-quality lead-vocal stems"

    root = Path(__file__).resolve().parent
    hq_python = root / ".venv-pianoflow-hq" / "Scripts" / "python.exe"
    hq_script = root / "hq_vocal_separator.py"
    if not hq_python.exists() or not hq_script.exists():
        return _separate_demucs(source, output_root)

    process = subprocess.run(
        [
            str(hq_python),
            str(hq_script),
            str(source.resolve()),
            "--output-dir",
            str(output_root.resolve()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        detail = process.stderr[-3000:] or process.stdout[-3000:]
        raise RuntimeError(f"High-quality lead-vocal separation failed:\n{detail}")
    payload = None
    for line in reversed(process.stdout.splitlines()):
        if line.strip().startswith("{"):
            payload = json.loads(line)
            break
    if not isinstance(payload, dict):
        raise RuntimeError("High-quality separator returned no result")

    canonical_names = {
        "vocals": "vocals.wav",
        "instrumental": "instrumental.wav",
        "lead": "lead-vocal.wav",
        "backing": "backing-vocals.wav",
        "dryLead": "lead-vocal-dry.wav",
        "reverb": "lead-vocal-reverb.wav",
    }
    stem_paths: dict[str, Path] = {}
    for stem, filename in canonical_names.items():
        incoming = payload.get(stem)
        if not incoming:
            continue
        incoming_path = Path(incoming)
        if not incoming_path.exists():
            continue
        destination = output_root / filename
        shutil.copy2(incoming_path, destination)
        stem_paths[stem] = destination
    dry_lead = stem_paths.get("dryLead")
    if not dry_lead:
        raise RuntimeError("High-quality separator did not produce a dry lead vocal")
    return dry_lead, stem_paths, "RoFormer 6-model lead-vocal ensemble + de-reverb"


def estimate_tempo(source: Path, requested_bpm: float | None = None) -> tuple[int, list[float], float, list[dict[str, float]]]:
    audio, sample_rate = librosa.load(source, sr=22050, mono=True)
    if len(audio) < 4096:
        return 120, [], 0.0, []
    # Coarse 512-sample tempo bins quantised this excerpt to 143.55 BPM.
    # Fuse two finer resolutions with a continuously interpolated
    # autocorrelation peak so the final integer is not dictated by one bin.
    fine_hop, ultra_hop = 128, 64
    fine_onset = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=fine_hop)
    ultra_onset = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=ultra_hop)
    fine_tempo_values = librosa.feature.tempo(
        onset_envelope=fine_onset,
        sr=sample_rate,
        hop_length=fine_hop,
        aggregate=np.median,
        max_tempo=200,
    )
    ultra_tempo_values = librosa.feature.tempo(
        onset_envelope=ultra_onset,
        sr=sample_rate,
        hop_length=ultra_hop,
        aggregate=np.median,
        max_tempo=200,
    )
    fine_tempo = float(np.asarray(fine_tempo_values).reshape(-1)[0]) if np.size(fine_tempo_values) else 120.0
    ultra_tempo = float(np.asarray(ultra_tempo_values).reshape(-1)[0]) if np.size(ultra_tempo_values) else fine_tempo
    seed_tempo = float(np.median([fine_tempo, ultra_tempo]))
    while seed_tempo < 70:
        seed_tempo *= 2
    while seed_tempo > 180:
        seed_tempo /= 2
    max_lag = max(2, int(np.ceil(60.0 * sample_rate / fine_hop / max(55.0, seed_tempo * 0.92))))
    autocorrelation = librosa.autocorrelate(fine_onset, max_size=max_lag + 2)
    candidates = np.arange(max(55.0, seed_tempo * 0.94), min(200.0, seed_tempo * 1.06) + 0.001, 0.05)
    candidate_scores: list[float] = []
    for candidate in candidates:
        lag = 60.0 * sample_rate / (fine_hop * candidate)
        lower = int(np.floor(lag))
        fraction = lag - lower
        if lower < 1 or lower + 1 >= len(autocorrelation):
            candidate_scores.append(float("-inf"))
        else:
            candidate_scores.append(float((1.0 - fraction) * autocorrelation[lower] + fraction * autocorrelation[lower + 1]))
    periodic_tempo = float(candidates[int(np.argmax(candidate_scores))]) if len(candidates) else seed_tempo
    bpm = 0.25 * fine_tempo + 0.45 * ultra_tempo + 0.30 * periodic_tempo
    while bpm < 70:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    if requested_bpm is not None and 40.0 <= requested_bpm <= 240.0:
        bpm = float(requested_bpm)
    onset = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=512)
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=sample_rate,
        hop_length=512,
        bpm=bpm,
        units="frames",
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=512)
    # Beat trackers occasionally insert a short run of compressed beats in
    # dense vocals.  Preserve the measured first/last beat while winsorising
    # implausible local intervals; otherwise one bad run becomes a false
    # 160-BPM island and corrupts tuplets and score positions.
    if len(beat_times) >= 3:
        original_start = float(beat_times[0])
        global_interval = 60.0 / bpm
        intervals = np.diff(beat_times)
        corrected = np.clip(intervals, global_interval * 0.90, global_interval * 1.10)
        # Retain local tempo movement but remove the beat tracker's global bin
        # bias so the beat map cannot drift away from the established BPM.
        corrected *= global_interval / max(1e-9, float(np.median(corrected)))
        beat_times = np.concatenate(([original_start], original_start + np.cumsum(corrected)))
    times = [round(float(value), 5) for value in beat_times]
    phase = times[0] if times else 0.0
    tempo_map: list[dict[str, float]] = []
    if len(beat_times) >= 2:
        intervals = np.diff(beat_times)
        local_bpms = 60.0 / np.maximum(intervals, 1e-4)
        local_median = float(np.median(local_bpms)) if len(local_bpms) else bpm
        if local_median > 0:
            local_bpms *= bpm / local_median
        for index, value in enumerate(local_bpms):
            left, right = max(0, index - 4), min(len(local_bpms), index + 5)
            smoothed = float(np.median(local_bpms[left:right]))
            smoothed = float(np.clip(smoothed, bpm * 0.92, bpm * 1.08))
            if 40.0 <= smoothed <= 260.0:
                tempo_map.append({"time": round(float(beat_times[index]), 5), "bpm": round(smoothed, 3)})
    return int(round(bpm)), times, phase, tempo_map


def estimate_bpm(source: Path) -> int:
    return estimate_tempo(source)[0]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    midpoint = float(np.sum(ordered_weights)) / 2
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), midpoint)])


def _harmonic_support(spectrum: np.ndarray, midi: int, start_frame: int, end_frame: int, sample_rate: int, n_fft: int) -> float:
    """Score a fundamental from its own bin and the first five harmonics."""

    frequency = float(librosa.midi_to_hz(midi))
    frame_slice = spectrum[:, max(0, start_frame):max(start_frame + 1, end_frame)]
    if not frame_slice.size:
        return 0.0
    total = float(np.mean(frame_slice)) + 1e-9
    support = 0.0
    for harmonic in range(1, 6):
        harmonic_frequency = frequency * harmonic
        if harmonic_frequency >= sample_rate / 2:
            break
        bin_index = int(round(harmonic_frequency * n_fft / sample_rate))
        if bin_index <= 0 or bin_index >= frame_slice.shape[0]:
            continue
        local = frame_slice[max(0, bin_index - 1):min(frame_slice.shape[0], bin_index + 2)]
        support += float(np.mean(local)) / harmonic
    fundamental_bin = int(round(frequency * n_fft / sample_rate))
    fundamental = float(np.mean(frame_slice[max(0, fundamental_bin - 1):min(frame_slice.shape[0], fundamental_bin + 2)])) if 0 < fundamental_bin < frame_slice.shape[0] else 0.0
    return float(np.log1p((support + 0.8 * fundamental) / total))


def _tempo_snap_info(time: float, bpm: int, phase: float = 0.0) -> tuple[float, int, float]:
    beat = 60.0 / max(1, bpm)
    candidates: list[tuple[float, int, float, float]] = [(time, 0, 0.0, 0.018)]
    # Evaluate the score grid generically.  This covers quintuplets, sextuplets,
    # septuplets and other practical subdivisions without a triplet-only rule.
    for divisions in range(1, 17):
        grid = beat / divisions
        snapped = phase + round((time - phase) / grid) * grid
        error = abs(snapped - time)
        tolerance = min(0.032, max(0.010, grid * 0.24))
        if error <= tolerance:
            # Prefer conventional binary divisions when fits are effectively tied.
            complexity = 0.0 if divisions in (1, 2, 4, 8, 16) else 0.0003 * divisions
            candidates.append((snapped, divisions, error, error + complexity))
    snapped, division, error, _ = min(candidates, key=lambda candidate: candidate[3])
    return snapped, division, error


def _tempo_snap(time: float, bpm: int, phase: float = 0.0) -> float:
    return _tempo_snap_info(time, bpm, phase)[0]


def _tempo_map_snap_info(time: float, beat_times: list[float], bpm: int, phase: float = 0.0) -> tuple[float, int, float]:
    """Snap against actual neighbouring beats, preserving local tempo drift."""

    if len(beat_times) < 2:
        return _tempo_snap_info(time, bpm, phase)
    beats = np.asarray(beat_times, dtype=float)
    position = int(np.searchsorted(beats, time, side="right") - 1)
    position = int(np.clip(position, 0, len(beats) - 2))
    left, right = float(beats[position]), float(beats[position + 1])
    beat_length = max(0.08, right - left)
    candidates: list[tuple[float, int, float, float]] = [(time, 0, 0.0, 0.018)]
    for divisions in range(1, 17):
        grid = beat_length / divisions
        snapped = left + round((time - left) / grid) * grid
        error = abs(snapped - time)
        tolerance = min(0.032, max(0.008, grid * 0.24))
        if error <= tolerance:
            complexity = 0.0 if divisions in (1, 2, 4, 8, 16) else 0.0003 * divisions
            candidates.append((snapped, divisions, error, error + complexity))
    snapped, division, error, _ = min(candidates, key=lambda candidate: candidate[3])
    return snapped, division, error


def _tempo_map_note_span_info(
    start: float,
    end: float,
    beat_times: list[float],
    bpm: int,
    phase: float = 0.0,
) -> tuple[float, float, int, int, float, float]:
    """Choose a score onset and duration together without moving playback.

    Independent endpoint snapping can create impossible short values.  This
    decoder evaluates subdivisions 1..16 and durations up to four local beats,
    while keeping an unsnapped candidate when the performance is expressive.
    """

    score_start, start_division, start_error = _tempo_map_snap_info(start, beat_times, bpm, phase)
    raw_duration = max(0.0, end - start)
    local_beat = 60.0 / max(1, bpm)
    if len(beat_times) >= 2:
        beats = np.asarray(beat_times, dtype=float)
        position = int(np.clip(np.searchsorted(beats, start, side="right") - 1, 0, len(beats) - 2))
        local_beat = max(0.08, float(beats[position + 1] - beats[position]))

    duration_candidates: list[tuple[float, int, float, float]] = [(raw_duration, 0, 0.0, 0.020)]
    for division in range(1, 17):
        unit = local_beat / division
        for units in range(1, division * 4 + 1):
            duration = unit * units
            error = abs(duration - raw_duration)
            tolerance = min(0.050, max(0.010, unit * 0.26))
            if error > tolerance:
                continue
            complexity = 0.0 if division in (1, 2, 4, 8, 16) else 0.00035 * division
            duration_candidates.append((duration, division, error, error + complexity))
    score_duration, duration_division, duration_error, _ = min(duration_candidates, key=lambda item: item[3])
    return score_start, score_start + score_duration, start_division, duration_division, start_error, duration_error


def _tuplet_ratio(division: int) -> tuple[int, int] | None:
    if division <= 0 or division in (1, 2, 4, 8, 16):
        return None
    normal = 1
    while normal * 2 < division:
        normal *= 2
    return division, normal


def _transcribe_vocals(vocals: Path, lyrics_path: Path | None = None) -> dict:
    """Run local Japanese singing transcription without making analysis fail."""

    root = Path(__file__).resolve().parent
    python = root / ".venv-pianoflow-hq" / "Scripts" / "python.exe"
    script = root / "hq_transcriber.py"
    unavailable = {"available": False, "text": "", "words": [], "segments": []}
    if not python.exists() or not script.exists():
        return unavailable
    command = [str(python), str(script), str(vocals.resolve()), "--output-dir", str(vocals.parent.resolve())]
    if lyrics_path and lyrics_path.exists():
        command.extend(["--lyrics-file", str(lyrics_path.resolve())])
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        unavailable["error"] = (process.stderr or process.stdout)[-1200:]
        return unavailable
    for line in reversed(process.stdout.splitlines()):
        if line.strip().startswith("{"):
            try:
                result = json.loads(line)
                return result if isinstance(result, dict) else unavailable
            except json.JSONDecodeError:
                break
    return unavailable


def _extract_hq_pitch(
    vocals: Path,
    target_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run CREPE, RMVPE and WORLD and align all estimates to pYIN frames."""

    root = Path(__file__).resolve().parent
    python = root / ".venv-pianoflow-hq" / "Scripts" / "python.exe"
    script = root / "hq_pitch_tracker.py"
    empty = np.full(len(target_times), np.nan, dtype=float)
    if not python.exists() or not script.exists() or not len(target_times):
        zero = np.zeros(len(target_times), dtype=float)
        return empty, zero, empty.copy(), zero.copy(), empty.copy(), zero.copy(), np.full((len(target_times), 3), np.nan), np.zeros((len(target_times), 3))
    output = vocals.with_name("pitch-consensus-v3.json")
    if not output.exists():
        process = subprocess.run(
            [str(python), str(script), str(vocals.resolve()), "--output", str(output)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.returncode != 0 or not output.exists():
            zero = np.zeros(len(target_times), dtype=float)
            return empty, zero, empty.copy(), zero.copy(), empty.copy(), zero.copy(), np.full((len(target_times), 3), np.nan), np.zeros((len(target_times), 3))
    payload = json.loads(output.read_text(encoding="utf-8"))
    hop_seconds = float(payload["hopLength"]) / float(payload["sampleRate"])

    def align(hz_key: str, confidence_key: str) -> tuple[np.ndarray, np.ndarray]:
        raw_hz = np.array([np.nan if value is None else float(value) for value in payload.get(hz_key, [])], dtype=float)
        raw_confidence = np.asarray(payload.get(confidence_key, []), dtype=float)
        aligned_hz = empty.copy()
        aligned_confidence = np.zeros(len(target_times), dtype=float)
        if not len(raw_hz) or not len(raw_confidence):
            return aligned_hz, aligned_confidence
        count = min(len(raw_hz), len(raw_confidence))
        raw_hz, raw_confidence = raw_hz[:count], raw_confidence[:count]
        source_times = np.arange(count, dtype=float) * hop_seconds
        valid = np.isfinite(raw_hz)
        if np.count_nonzero(valid) < 2:
            return aligned_hz, aligned_confidence
        aligned_hz = np.interp(target_times, source_times[valid], raw_hz[valid], left=np.nan, right=np.nan)
        aligned_confidence = np.interp(target_times, source_times, raw_confidence, left=0.0, right=0.0)
        # Never interpolate a pitch through a genuinely unvoiced region.
        nearest = np.clip(np.searchsorted(source_times, target_times, side="left"), 0, count - 1)
        locally_voiced = np.zeros(len(target_times), dtype=bool)
        for offset in (-1, 0, 1):
            locally_voiced |= valid[np.clip(nearest + offset, 0, count - 1)]
        aligned_hz[~locally_voiced] = np.nan
        aligned_confidence[~locally_voiced] = 0.0
        return aligned_hz, aligned_confidence

    crepe_hz, crepe_confidence = align("hz", "confidence")
    rmvpe_hz, rmvpe_confidence = align("rmvpeHz", "rmvpeConfidence")
    world_hz, world_confidence = align("worldHz", "worldConfidence")
    candidate_hz = np.full((len(target_times), 3), np.nan, dtype=float)
    candidate_confidence = np.zeros((len(target_times), 3), dtype=float)
    raw_candidates = payload.get("rmvpeCandidateHz", [])
    raw_candidate_confidence = payload.get("rmvpeCandidateConfidence", [])
    if raw_candidates and raw_candidate_confidence:
        source_count = min(len(raw_candidates), len(raw_candidate_confidence))
        source_times = np.arange(source_count, dtype=float) * hop_seconds
        for rank in range(3):
            raw_hz = np.array([
                np.nan if rank >= len(row) or row[rank] is None else float(row[rank])
                for row in raw_candidates[:source_count]
            ])
            raw_conf = np.array([
                0.0 if rank >= len(row) else float(row[rank])
                for row in raw_candidate_confidence[:source_count]
            ])
            valid_rank = np.isfinite(raw_hz) & (raw_conf > 0)
            if np.count_nonzero(valid_rank) >= 2:
                candidate_hz[:, rank] = np.interp(target_times, source_times[valid_rank], raw_hz[valid_rank], left=np.nan, right=np.nan)
                candidate_confidence[:, rank] = np.interp(target_times, source_times, raw_conf, left=0.0, right=0.0)
    return crepe_hz, crepe_confidence, rmvpe_hz, rmvpe_confidence, world_hz, world_confidence, candidate_hz, candidate_confidence


def _fill_short_false_runs(mask: np.ndarray, maximum: int, protected: np.ndarray | None = None) -> np.ndarray:
    result = mask.copy()
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        end = index
        while end < len(result) and not result[end]:
            end += 1
        crosses_boundary = bool(
            protected is not None
            and np.any(protected[max(0, index - 1):min(len(result), end + 1)])
        )
        if index > 0 and end < len(result) and end - index <= maximum and not crosses_boundary:
            result[index:end] = True
        index = end
    return result


def _bridge_supported_false_runs(
    mask: np.ndarray,
    support: np.ndarray,
    maximum: int,
    protected: np.ndarray | None = None,
) -> np.ndarray:
    """Recover short missing F0 regions only when the full-vocal stem has energy."""
    result = mask.copy()
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        end = index
        while end < len(result) and not result[end]:
            end += 1
        bounded = index > 0 and end < len(result)
        crosses_boundary = bool(
            protected is not None
            and np.any(protected[max(0, index - 1):min(len(result), end + 1)])
        )
        if bounded and end - index <= maximum and float(np.mean(support[index:end])) >= 0.22 and not crosses_boundary:
            result[index:end] = True
        index = end
    return result


def _normalise_feature(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.percentile(values, percentile)) if values.size else 0.0
    return np.clip(values / max(scale, 1e-9), 0.0, 1.0)


def _boundary_features(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Find syllable/reattack boundaries before any temporal pitch smoothing.

    A vocal note can restart without changing pitch, so pitch transitions alone
    are insufficient. The detector combines broadband onset, spectral flux,
    high-frequency consonant flux and the positive RMS derivative.
    """

    onset = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=hop_length,
        n_mels=64,
        fmin=70,
        fmax=min(8000, sample_rate // 2),
        power=1.0,
    )
    log_mel = np.log1p(20.0 * mel)
    positive_flux = np.maximum(0.0, np.diff(log_mel, axis=1, prepend=log_mel[:, :1]))
    spectral_flux = np.mean(positive_flux, axis=0)
    consonant_flux = np.mean(positive_flux[40:], axis=0)
    short_rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop_length, center=True)[0]
    rms_rise = np.maximum(0.0, np.diff(short_rms, prepend=short_rms[:1]))
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=1024, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=1024, hop_length=hop_length)[0]

    def fit(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values[:frame_count], dtype=float)
        return np.pad(values, (0, max(0, frame_count - len(values))))

    onset = _normalise_feature(fit(onset))
    spectral_flux = _normalise_feature(fit(spectral_flux))
    consonant_flux = _normalise_feature(fit(consonant_flux))
    rms_rise = _normalise_feature(fit(rms_rise))
    energy = _normalise_feature(fit(short_rms), 90.0)
    centroid = _normalise_feature(fit(centroid), 90.0)
    flatness = _normalise_feature(fit(flatness), 90.0)
    strength = np.clip(0.42 * onset + 0.30 * spectral_flux + 0.18 * consonant_flux + 0.10 * rms_rise, 0.0, 1.0)
    # Breath and residual separation noise can have large high-frequency flux
    # without a pitched vocal body. Down-weight only low-energy, flat, bright
    # events so genuine consonant attacks remain available for segmentation.
    noise_only = np.clip(flatness * centroid * (1.0 - energy), 0.0, 1.0)
    strength *= 1.0 - 0.22 * noise_only
    # A repeated note often contains a brief closure or breath followed by a
    # fresh attack. Measure that valley independently from the positive RMS
    # derivative so quiet consonants can still restart an unchanged pitch.
    valley = np.zeros(frame_count, dtype=float)
    radius = max(2, int(round(0.040 * sample_rate / hop_length)))
    centre = max(1, int(round(0.010 * sample_rate / hop_length)))
    for index in range(centre, frame_count - centre):
        before = energy[max(0, index - radius):max(0, index - centre)]
        middle = energy[max(0, index - centre):min(frame_count, index + centre + 1)]
        after = energy[min(frame_count, index + centre + 1):min(frame_count, index + radius + 1)]
        if before.size and middle.size and after.size:
            shoulder = min(float(np.mean(before)), float(np.mean(after)))
            valley[index] = max(0.0, shoulder - float(np.mean(middle)))

    peaks = np.zeros(frame_count, dtype=bool)
    minimum_spacing = max(2, int(round(0.050 * sample_rate / hop_length)))
    last = -minimum_spacing
    for index in range(1, frame_count - 1):
        if strength[index] < 0.50 or strength[index] < strength[index - 1] or strength[index] < strength[index + 1]:
            continue
        if index - last < minimum_spacing:
            if strength[index] > strength[last]:
                peaks[last] = False
            else:
                continue
        peaks[index] = True
        last = index
    vowel_probability = np.clip(energy * (1.0 - 0.55 * centroid) * (1.0 - 0.45 * flatness), 0.0, 1.0)
    vowel_change = np.zeros(frame_count, dtype=float)
    timbre_radius = max(2, int(round(0.030 * sample_rate / hop_length)))
    vowel_bands = min(40, log_mel.shape[0])
    for index in range(timbre_radius, frame_count - timbre_radius):
        before = np.mean(log_mel[:vowel_bands, index - timbre_radius:index], axis=1)
        after = np.mean(log_mel[:vowel_bands, index + 1:index + timbre_radius + 1], axis=1)
        before /= max(float(np.linalg.norm(before)), 1e-9)
        after /= max(float(np.linalg.norm(after)), 1e-9)
        vowel_change[index] = max(0.0, 1.0 - float(np.dot(before, after)))
    vowel_change = _normalise_feature(vowel_change, 95.0)
    evidence = {
        "onset": onset,
        "spectralFlux": spectral_flux,
        "consonantFlux": consonant_flux,
        "rmsRise": rms_rise,
        "energy": energy,
        "energyValley": np.clip(valley, 0.0, 1.0),
        "vowelTimbreChange": vowel_change,
    }
    return strength, peaks, vowel_probability, evidence


def _detect_same_pitch_reattacks(
    decoded_pitch: np.ndarray,
    raw_pitch: np.ndarray,
    valid: np.ndarray,
    boundary_features: dict[str, np.ndarray],
    boundary_strength: np.ndarray,
    text_candidates: np.ndarray,
    vowel_probability: np.ndarray,
    absolute_times: np.ndarray,
    beat_times: list[float],
    bpm: float,
    beat_phase: float,
    frame_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect a fresh vocal attack whose written pitch does not change.

    Pitch continuity is required on both sides. A split needs multiple
    independent attack cues; notation/grid proximity only adds weak support
    and never creates a boundary by itself.
    """

    count = len(decoded_pitch)
    detected = np.zeros(count, dtype=bool)
    evidence_scores = np.zeros(count, dtype=float)
    phonetic_starts = np.full(count, np.nan, dtype=float)
    onset = boundary_features["onset"]
    spectral = boundary_features["spectralFlux"]
    consonant = boundary_features["consonantFlux"]
    rms_rise = boundary_features["rmsRise"]
    valley = boundary_features["energyValley"]
    vowel_change = boundary_features["vowelTimbreChange"]
    before_outer = max(4, int(round(0.075 / frame_seconds)))
    before_inner = max(1, int(round(0.012 / frame_seconds)))
    after_inner = max(1, int(round(0.014 / frame_seconds)))
    after_outer = max(4, int(round(0.080 / frame_seconds)))
    local_radius = max(1, int(round(0.012 / frame_seconds)))
    minimum_spacing = max(3, int(round(0.045 / frame_seconds)))

    for index in range(before_outer, count - after_outer):
        if not valid[index] or not np.isfinite(decoded_pitch[index]):
            continue
        before_slice = slice(index - before_outer, index - before_inner)
        after_slice = slice(index + after_inner, index + after_outer)
        before_decoded = decoded_pitch[before_slice]
        after_decoded = decoded_pitch[after_slice]
        before_valid = valid[before_slice] & np.isfinite(before_decoded)
        after_valid = valid[after_slice] & np.isfinite(after_decoded)
        if np.count_nonzero(before_valid) < 3 or np.count_nonzero(after_valid) < 3:
            continue
        before_note = int(round(float(np.median(before_decoded[before_valid]))))
        after_note = int(round(float(np.median(after_decoded[after_valid]))))
        if before_note != after_note:
            continue
        before_raw = raw_pitch[before_slice][before_valid]
        after_raw = raw_pitch[after_slice][after_valid]
        before_raw = before_raw[np.isfinite(before_raw)]
        after_raw = after_raw[np.isfinite(after_raw)]
        if len(before_raw) < 3 or len(after_raw) < 3:
            continue
        before_median, after_median = float(np.median(before_raw)), float(np.median(after_raw))
        before_spread = float(np.median(np.abs(before_raw - before_median)))
        after_spread = float(np.median(np.abs(after_raw - after_median)))
        # This rejects scoops, falls, portamento and wide vibrato: both sides
        # must be stable versions of the same equal-tempered note.
        if (
            abs(before_median - after_median) > 0.60
            or abs(after_median - after_note) > 0.72
            or before_spread > 0.48
            or after_spread > 0.48
        ):
            continue

        score_supported = False
        _, division, grid_error = _tempo_map_snap_info(
            float(absolute_times[index]), beat_times, bpm, beat_phase
        )
        if division > 0:
            score_supported = grid_error <= max(0.012, min(0.030, 60.0 / max(1.0, bpm) / 16.0))
        vowel_reset = bool(
            float(np.mean(vowel_probability[before_slice])) >= 0.18
            and float(np.mean(vowel_probability[after_slice])) >= 0.18
            and float(vowel_probability[index])
            <= 0.84 * min(
                float(np.mean(vowel_probability[before_slice])),
                float(np.mean(vowel_probability[after_slice])),
            )
        )
        text_support = bool(text_candidates[index])
        consonant_cue = bool(consonant[index] >= 0.30)
        vowel_change_cue = bool(vowel_change[index] >= 0.34)
        valley_cue = bool(valley[index] >= 0.10)
        onset_cue = bool(onset[index] >= 0.38 or rms_rise[index] >= 0.34)
        physical_cue_count = sum((consonant_cue, vowel_change_cue, valley_cue, vowel_reset, onset_cue))
        score = (
            2.10 * min(1.0, float(consonant[index]) / 0.50)
            + 1.85 * min(1.0, float(vowel_change[index]) / 0.55)
            + 1.45 * min(1.0, float(valley[index]) / 0.24)
            + (1.35 if vowel_reset else 0.0)
            + 0.75 * min(1.0, float(onset[index]) / 0.55)
            + 0.65 * min(1.0, float(spectral[index]) / 0.55)
            + 0.60 * min(1.0, float(rms_rise[index]) / 0.50)
            + (0.65 if text_support else 0.0)
            + (0.20 if score_supported else 0.0)
        )
        evidence_scores[index] = score
        strong_phonetic_exception = bool(
            consonant[index] >= 0.55
            and (valley[index] >= 0.16 or vowel_change[index] >= 0.55)
        )
        if (
            physical_cue_count < 2
            or score < 4.15
            or (boundary_strength[index] < 0.45 and not strong_phonetic_exception)
        ):
            continue
        left, right = max(0, index - local_radius), min(count, index + local_radius + 1)
        if float(boundary_strength[index]) + 0.05 < float(np.max(boundary_strength[left:right])):
            continue
        vowel_search = max(2, int(round(0.100 / frame_seconds)))
        vowel_index = index
        for candidate in range(index, min(count - 2, index + vowel_search)):
            stable_target = (
                valid[candidate]
                and np.isfinite(decoded_pitch[candidate])
                and int(round(float(decoded_pitch[candidate]))) == after_note
                and float(vowel_probability[candidate]) >= 0.30
            )
            if stable_target and np.count_nonzero(valid[candidate:candidate + 3]) >= 2:
                vowel_index = candidate
                break
        detected[vowel_index] = True
        evidence_scores[vowel_index] = max(evidence_scores[vowel_index], score)
        phonetic_starts[vowel_index] = float(absolute_times[index])

    # Keep only the strongest candidate when several adjacent frames describe
    # one attack. This spacing is acoustic (45 ms), not tied to a particular
    # quarter/eighth/triplet/quintuplet value.
    selected = np.zeros(count, dtype=bool)
    for index in np.flatnonzero(detected):
        nearby = np.flatnonzero(selected[max(0, index - minimum_spacing):index + 1])
        if nearby.size:
            previous = max(0, index - minimum_spacing) + int(nearby[-1])
            if evidence_scores[index] > evidence_scores[previous]:
                selected[previous] = False
                selected[index] = True
        else:
            selected[index] = True
    selected_phonetic_starts = np.full(count, np.nan, dtype=float)
    selected_phonetic_starts[selected] = phonetic_starts[selected]
    return selected, evidence_scores, selected_phonetic_starts


def _edge_preserving_pitch_smooth(
    values: np.ndarray,
    confidence: np.ndarray,
    voiced: np.ndarray,
    protected_boundaries: np.ndarray,
) -> np.ndarray:
    """Smooth only inside pre-segmented note regions; never across a boundary."""

    result = np.asarray(values, dtype=float).copy()
    cut = np.asarray(protected_boundaries, dtype=bool) | ~np.asarray(voiced, dtype=bool)
    starts = [0] + [int(index) for index in np.flatnonzero(cut)]
    ends = [int(index) for index in np.flatnonzero(cut)] + [len(result)]
    for start, end in zip(starts, ends):
        if end - start < 5:
            continue
        original = result[start:end].copy()
        weights = np.asarray(confidence[start:end], dtype=float)
        for local in range(len(original)):
            left, right = max(0, local - 2), min(len(original), local + 3)
            region = original[left:right]
            valid = np.isfinite(region)
            if np.count_nonzero(valid) < 2:
                continue
            region_weights = np.maximum(weights[left:right][valid], 1e-4)
            median = _weighted_median(region[valid], region_weights)
            # Preserve a supported fast change; suppress only sub-semitone
            # wobble and isolated excursions inside the already fixed region.
            if abs(original[local] - median) < 1.35 or weights[local] < 0.30:
                result[start + local] = median
    return result


def _acoustic_segments(
    times: np.ndarray,
    voiced: np.ndarray,
    boundaries: np.ndarray,
    boundary_strength: np.ndarray,
    vowel_probability: np.ndarray,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], list[dict[str, float | str]]]:
    """Return word-like vocal groups and their acoustic syllable segments.

    These are language-independent acoustic boundaries. They remain useful
    when lyrics are unavailable and do not force melismas into one note.
    """

    syllables: list[dict[str, float | str]] = []
    word_groups: list[dict[str, float | str]] = []
    vowels: list[dict[str, float | str]] = []
    voiced_indices = np.flatnonzero(voiced)
    if not len(voiced_indices):
        return word_groups, syllables, vowels
    frame_seconds = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    phrase_gap_frames = max(2, int(round(0.18 / max(frame_seconds, 1e-6))))
    phrase_start = int(voiced_indices[0])
    previous = phrase_start
    phrases: list[tuple[int, int]] = []
    for frame in voiced_indices[1:]:
        frame = int(frame)
        if frame - previous > phrase_gap_frames:
            phrases.append((phrase_start, previous + 1))
            phrase_start = frame
        previous = frame
    phrases.append((phrase_start, previous + 1))

    for group_index, (start, end) in enumerate(phrases):
        cuts = [start]
        cuts.extend(int(index) for index in np.flatnonzero(boundaries[start + 1:end]) + start + 1)
        cuts.append(end)
        cuts = sorted(set(cuts))
        word_cuts = [start]
        for boundary in cuts[1:-1]:
            if boundary_strength[boundary] >= 0.80 and (boundary - word_cuts[-1]) * frame_seconds >= 0.22:
                word_cuts.append(boundary)
        word_cuts.append(end)
        for word_index, (word_left, word_right) in enumerate(zip(word_cuts[:-1], word_cuts[1:])):
            word_groups.append({
                "kind": "word-like",
                "start": round(float(times[word_left]), 4),
                "end": round(float(times[min(word_right - 1, len(times) - 1)] + frame_seconds), 4),
                "label": f"{group_index + 1}-word-{word_index + 1}",
                "confidence": round(float(boundary_strength[word_left]) if word_left < len(boundary_strength) else 0.0, 4),
            })
        for syllable_index, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
            if right <= left:
                continue
            syllables.append({
                "kind": "syllable",
                "start": round(float(times[left]), 4),
                "end": round(float(times[min(right - 1, len(times) - 1)] + frame_seconds), 4),
                "confidence": round(float(np.max(boundary_strength[left:min(right + 1, len(boundary_strength))])), 4),
                "label": f"{group_index + 1}-{syllable_index + 1}",
            })
    vowel_mask = voiced & (vowel_probability >= 0.32)
    vowel_start: int | None = None
    minimum_vowel_frames = max(2, int(round(0.045 / max(frame_seconds, 1e-6))))
    for index in range(len(vowel_mask) + 1):
        active = index < len(vowel_mask) and bool(vowel_mask[index])
        if active and vowel_start is None:
            vowel_start = index
        elif not active and vowel_start is not None:
            if index - vowel_start >= minimum_vowel_frames:
                vowels.append({
                    "kind": "vowel",
                    "start": round(float(times[vowel_start]), 4),
                    "end": round(float(times[min(index - 1, len(times) - 1)] + frame_seconds), 4),
                    "confidence": round(float(np.mean(vowel_probability[vowel_start:index])), 4),
                })
            vowel_start = None
    return word_groups, syllables, vowels


def _align_vocal_timing(reference: np.ndarray, candidate: np.ndarray, sample_rate: int, hop_length: int) -> tuple[np.ndarray, float, float]:
    """Align a separated onset stem to the dry pitch stem within +/-250 ms."""

    if len(reference) < 2048 or len(candidate) < 2048:
        return candidate, 0.0, 0.0
    reference_env = librosa.feature.rms(y=reference, frame_length=1024, hop_length=hop_length, center=True)[0]
    candidate_env = librosa.feature.rms(y=candidate, frame_length=1024, hop_length=hop_length, center=True)[0]
    count = min(len(reference_env), len(candidate_env))
    reference_env = reference_env[:count] - np.mean(reference_env[:count])
    candidate_env = candidate_env[:count] - np.mean(candidate_env[:count])
    maximum = max(1, int(round(0.25 * sample_rate / hop_length)))
    best_shift, best_score = 0, -1.0
    for shift in range(-maximum, maximum + 1):
        if shift < 0:
            left, right = reference_env[:shift], candidate_env[-shift:]
        elif shift > 0:
            left, right = reference_env[shift:], candidate_env[:-shift]
        else:
            left, right = reference_env, candidate_env
        if len(left) < 20:
            continue
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        score = float(np.dot(left, right) / denominator) if denominator > 1e-9 else -1.0
        if score > best_score:
            best_shift, best_score = shift, score
    if best_score < 0.18 or best_shift == 0:
        return candidate, 0.0, max(0.0, best_score)
    sample_shift = int(best_shift * hop_length)
    if sample_shift < 0:
        aligned = np.pad(candidate[-sample_shift:], (0, -sample_shift))[:len(candidate)]
    else:
        aligned = np.pad(candidate, (sample_shift, 0))[:len(candidate)]
    return aligned, sample_shift / sample_rate, best_score


def _fold_same_syllable_pitch_approaches(notes: list[dict], beat_seconds: float) -> tuple[list[dict], int]:
    """Fold a short scoop/fall approach into its stable target piano note.

    Singing may spend a sizable fraction of a syllable approaching the written
    pitch.  A piano score should start the target pitch at the consonant rather
    than render every intermediate F0 plateau as a separate key.
    """

    result = [dict(note) for note in notes]
    folded = 0
    index = 0
    while index + 1 < len(result):
        current, following = result[index], result[index + 1]
        gap = float(following["start"]) - float(current["end"])
        duration = float(current["end"]) - float(current["start"])
        following_duration = float(following["end"]) - float(following["start"])
        interval = int(following["midi"]) - int(current["midi"])
        ornaments = set(current.get("ornaments", []))
        low_consensus = float(current.get("sourceAgreement", 1.0)) < 0.55 or float(current.get("pitchSpread", 0.0)) > 1.25
        explicit_glide = bool(ornaments.intersection({"scoop", "portamento"}))
        attacked_expression = (
            current.get("boundaryType") == "reattack"
            and bool(ornaments.intersection({"scoop", "fall", "portamento", "passing-tone"}))
        )
        stable_upward_approach = (
            1 <= interval <= 3
            and duration <= min(0.150, max(0.090, beat_seconds * 0.38))
            and following_duration >= 0.135
            and following.get("boundaryType") != "reattack"
            and (float(current.get("sourceAgreement", 1.0)) < 0.75 or float(current.get("pitchSpread", 0.0)) > 0.55)
        )
        long_unstable_glide = (
            1 <= interval <= 3
            and duration <= 0.55
            and following_duration >= 0.135
            and following.get("boundaryType") != "reattack"
            and explicit_glide
            and low_consensus
        )
        continuous_gap_limit = 0.100 if (stable_upward_approach or long_unstable_glide) else 0.035
        # Stable pitch-or-syllable passing tones are genuine melisma notes.
        # Do not remove them merely because they happen to lead by a step.
        transition_evidence = bool(low_consensus or explicit_glide or attacked_expression or stable_upward_approach)
        same_syllable_target = following.get("boundaryType") != "reattack"
        short_approach = duration <= min(0.185, max(0.090, beat_seconds * 0.46))
        expressive_reattack = current.get("boundaryType") == "reattack" and bool(ornaments.intersection({"scoop", "fall", "portamento", "passing-tone"}))
        target_floor = 0.075 if expressive_reattack else 0.105
        target_ratio = 0.25 if long_unstable_glide else (0.65 if expressive_reattack else 0.82)
        target_is_sustained = following_duration >= max(target_floor, duration * target_ratio)
        if (
            -0.030 <= gap <= continuous_gap_limit
            and 1 <= abs(interval) <= 5
            and same_syllable_target
            and (short_approach or long_unstable_glide)
            and target_is_sustained
            and transition_evidence
        ):
            direction = "scoop" if interval > 0 else "fall"
            following["start"] = current["start"]
            following["rawStart"] = current.get("rawStart", current["start"])
            following["scoreStart"] = current.get("scoreStart", current["start"])
            following["boundaryType"] = current.get("boundaryType", following.get("boundaryType"))
            following["onset"] = max(float(current.get("onset", 0.0)), float(following.get("onset", 0.0)))
            following["ornaments"] = sorted(set(following.get("ornaments", [])) | {f"folded-{direction}"})
            following["foldedApproachPitches"] = list(current.get("foldedApproachPitches", [])) + [int(current["midi"])]
            del result[index]
            folded += 1
            index = max(0, index - 1)
            continue
        index += 1
    return result, folded


def _consolidate_post_ornament_duplicates(notes: list[dict]) -> tuple[list[dict], int]:
    """Remove only duplicate fragments created by ornament folding.

    A real repeated note must retain its second attack.  Consolidation therefore
    requires an ornament-folding marker plus nearly identical pitch statistics;
    adjacency and equal MIDI alone are deliberately insufficient.
    """

    consolidated: list[dict] = []
    removed = 0
    for original in notes:
        note = dict(original)
        if not consolidated:
            consolidated.append(note)
            continue
        previous = consolidated[-1]
        touching = -0.020 <= float(note["start"]) - float(previous["end"]) <= 0.020
        same_pitch = int(note["midi"]) == int(previous["midi"])
        folding_marker = bool(
            previous.get("foldedApproachPitches")
            or note.get("foldedApproachPitches")
            or any(str(value).startswith("folded-") for value in previous.get("ornaments", []))
            or any(str(value).startswith("folded-") for value in note.get("ornaments", []))
        )
        identical_signature = (
            abs(float(note.get("pitchMidi", note["midi"])) - float(previous.get("pitchMidi", previous["midi"]))) <= 0.08
            and abs(float(note.get("pitchSpread", 0.0)) - float(previous.get("pitchSpread", 0.0))) <= 0.08
            and abs(float(note.get("sourceAgreement", 0.0)) - float(previous.get("sourceAgreement", 0.0))) <= 0.08
        )
        independent_second_attack = (
            bool(note.get("samePitchReattack"))
            and float(note.get("onset", 0.0)) >= 0.32
        )
        if touching and same_pitch and folding_marker and identical_signature and not independent_second_attack:
            # The later onset is the stable target; retaining the earlier start
            # would recreate the ornamental fragment that was just folded.
            replacement = note
            replacement["end"] = max(float(previous["end"]), float(note["end"]))
            replacement["rawEnd"] = max(float(previous.get("rawEnd", previous["end"])), float(note.get("rawEnd", note["end"])))
            replacement["scoreEnd"] = max(float(previous.get("scoreEnd", previous["end"])), float(note.get("scoreEnd", note["end"])))
            replacement["vowelEnd"] = max(float(previous.get("vowelEnd", previous["end"])), float(note.get("vowelEnd", note["end"])))
            replacement["ornaments"] = sorted(set(previous.get("ornaments", [])) | set(note.get("ornaments", [])) | {"post-ornament-duplicate-fold"})
            replacement["foldedApproachPitches"] = list(previous.get("foldedApproachPitches", [])) + list(note.get("foldedApproachPitches", []))
            replacement["postOrnamentDuplicateSuppressed"] = True
            consolidated[-1] = replacement
            removed += 1
            continue
        consolidated.append(note)
    return consolidated, removed


def _recover_internal_note_boundaries(notes: list[dict], pitch_curve: list[dict], beat_seconds: float) -> tuple[list[dict], int]:
    """Recover attacks that survived acoustically but were merged into a long note."""

    recovered: list[dict] = []
    recovery_count = 0
    minimum_piece = max(0.065, min(0.095, beat_seconds / 5.0))
    for original in notes:
        note = dict(original)
        start, end = float(note["start"]), float(note["end"])
        if end - start < minimum_piece * 2.4:
            recovered.append(note)
            continue
        region = [point for point in pitch_curve if start + minimum_piece <= float(point["time"]) <= end - minimum_piece]
        candidates: list[dict] = []
        for index in range(1, len(region) - 1):
            point = region[index]
            strength = float(point.get("boundaryStrength", 0.0))
            reattack_score = float(point.get("samePitchReattackScore", 0.0))
            strength_peak = (
                strength >= float(region[index - 1].get("boundaryStrength", 0.0))
                and strength >= float(region[index + 1].get("boundaryStrength", 0.0))
            )
            score_peak = (
                reattack_score >= float(region[index - 1].get("samePitchReattackScore", 0.0))
                and reattack_score >= float(region[index + 1].get("samePitchReattackScore", 0.0))
            )
            if not strength_peak and not score_peak:
                continue
            time = float(point["time"])
            before = [
                float(row["midi"]) for row in pitch_curve
                if row.get("voiced") and row.get("midi") is not None and time - 0.075 <= float(row["time"]) <= time - 0.018
            ]
            after = [
                float(row["midi"]) for row in pitch_curve
                if row.get("voiced") and row.get("midi") is not None and time + 0.018 <= float(row["time"]) <= time + 0.085
            ]
            if len(before) < 2 or len(after) < 2:
                continue
            before_pitch, after_pitch = float(np.median(before)), float(np.median(after))
            pitch_change = abs(after_pitch - before_pitch)
            before_stability = float(np.median(np.abs(np.asarray(before) - before_pitch)))
            after_stability = float(np.median(np.abs(np.asarray(after) - after_pitch)))
            conservative_phonetic_reattack = (
                score_peak
                and end - start >= 0.36
                and reattack_score >= 3.58
                and strength >= 0.30
                and float(point.get("confidence", 0.0)) >= 0.55
                and pitch_change <= 0.35
                and before_stability <= 0.34
                and after_stability <= 0.34
            )
            same_pitch_reattack = (
                (
                    bool(point.get("samePitchReattack"))
                    and float(point.get("samePitchReattackScore", 0.0)) >= 3.65
                )
                or strength >= 0.50
                or conservative_phonetic_reattack
            ) and abs(after_pitch - int(note["midi"])) < 0.70
            # Recover a changed-pitch short note only with a strong independent
            # attack and stable pitch on both sides. This catches rapid notes
            # without turning a continuous scoop or vibrato into piano keys.
            changed_pitch_attack = (
                strength >= 0.72
                and 0.75 <= pitch_change <= 7.0
                and before_stability <= 0.34
                and after_stability <= 0.30
                and abs(after_pitch - round(after_pitch)) <= 0.38
            )
            stable_internal_pitch = (
                strength_peak
                and strength >= 0.38
                and float(point.get("confidence", 0.0)) >= 0.75
                and 0.75 <= pitch_change <= 3.0
                and before_stability <= 0.25
                and after_stability <= 0.25
                and abs(after_pitch - round(after_pitch)) <= 0.22
            )
            changed_pitch_attack = bool(changed_pitch_attack or stable_internal_pitch)
            text_attack = bool(point.get("textBoundary")) and strength >= 0.62
            if same_pitch_reattack or changed_pitch_attack or text_attack:
                candidate = dict(point)
                candidate["targetMidi"] = int(np.clip(round(after_pitch), 21, 108)) if changed_pitch_attack else int(note["midi"])
                candidate["changedPitch"] = bool(changed_pitch_attack)
                candidate["conservativePhoneticReattack"] = bool(conservative_phonetic_reattack)
                candidate["stableInternalPitch"] = bool(stable_internal_pitch)
                candidates.append(candidate)
        selected: list[dict] = []
        for candidate in sorted(candidates, key=lambda row: float(row["time"])):
            if selected and float(candidate["time"]) - float(selected[-1]["time"]) < minimum_piece:
                candidate_rank = float(candidate.get("boundaryStrength", 0.0))
                selected_rank = float(selected[-1].get("boundaryStrength", 0.0))
                if candidate_rank > selected_rank:
                    selected[-1] = candidate
                continue
            selected.append(candidate)
        cursor = start
        current_pitch = int(note["midi"])
        for candidate in selected:
            boundary = float(candidate["time"])
            if boundary - cursor < minimum_piece or end - boundary < minimum_piece:
                continue
            left = dict(note)
            left["start"], left["rawStart"] = cursor, cursor
            left["end"], left["rawEnd"] = boundary, boundary
            left["vowelEnd"] = boundary
            left["scoreStart"], left["scoreEnd"] = cursor, boundary
            left["midi"] = current_pitch
            left["pitchMidi"] = float(current_pitch)
            left["pitchCents"] = 0.0
            left["internalBoundaryRecovered"] = True
            recovered.append(left)
            cursor = boundary
            current_pitch = int(candidate["targetMidi"])
            note["samePitchReattack"] = bool(candidate.get("samePitchReattack"))
            if bool(candidate.get("samePitchReattack")):
                phonetic_start = float(candidate.get("phoneticStart", boundary))
                note["phoneticStart"] = phonetic_start
                note["musicalOnset"] = boundary
                note["consonantStart"] = phonetic_start
                note["vowelStart"] = boundary
            if bool(candidate.get("conservativePhoneticReattack")):
                note["conservativePhoneticReattack"] = True
            if bool(candidate.get("stableInternalPitch")):
                note["stableInternalPitchRecovered"] = True
            recovery_count += 1
        tail = dict(note)
        tail["start"], tail["rawStart"] = cursor, cursor
        tail["scoreStart"] = cursor
        tail["vowelStart"] = cursor
        tail["vowelEnd"] = end
        tail["midi"] = current_pitch
        tail["pitchMidi"] = float(current_pitch)
        tail["pitchCents"] = 0.0
        if cursor > start:
            tail["boundaryType"] = "reattack"
            tail["internalBoundaryRecovered"] = True
            tail["changedPitchRecovered"] = current_pitch != int(note["midi"])
        recovered.append(tail)
    recovered.sort(key=lambda row: (float(row["start"]), float(row["end"])))
    return recovered, recovery_count


def _score_decode(
    pyin_midi: np.ndarray,
    pyin_confidence: np.ndarray,
    crepe_midi: np.ndarray,
    crepe_confidence: np.ndarray,
    rmvpe_midi: np.ndarray,
    rmvpe_confidence: np.ndarray,
    world_midi: np.ndarray,
    world_confidence: np.ndarray,
    voiced: np.ndarray,
    onset_strength: np.ndarray,
    beat_nearness: np.ndarray,
) -> np.ndarray:
    """Decode framewise F0 into playable equal-tempered notes.

    The octave-tolerant observation cost combines four independent trackers. A Viterbi
    transition cost requires a new pitch to persist, while vocal onsets and a
    nearby score grid make genuine note changes slightly easier.
    """

    states = np.arange(36, 97, dtype=float)
    decoded = np.full(len(voiced), np.nan, dtype=float)
    intervals = np.abs(states[:, None] - states[None, :])
    base_transition = np.where(intervals == 0, 0.0, 9.0 + 0.24 * intervals + 2.0 * (intervals >= 8))

    starts = np.flatnonzero(voiced & ~np.r_[False, voiced[:-1]])
    ends = np.flatnonzero(voiced & ~np.r_[voiced[1:], False]) + 1
    observed = np.concatenate([
        pyin_midi[np.isfinite(pyin_midi)],
        crepe_midi[np.isfinite(crepe_midi)],
        rmvpe_midi[np.isfinite(rmvpe_midi)],
        world_midi[np.isfinite(world_midi)],
    ])
    center = float(np.median(observed)) if observed.size else 66.0

    for start, end in zip(starts, ends):
        length = end - start
        costs = np.full((length, len(states)), np.inf, dtype=np.float32)
        back = np.zeros((length, len(states)), dtype=np.int16)
        for local, frame in enumerate(range(start, end)):
            emission = 0.018 * np.square(states - center)
            for observation, confidence, weight in (
                (pyin_midi[frame], pyin_confidence[frame], 0.65),
                (crepe_midi[frame], crepe_confidence[frame], 1.05),
                (rmvpe_midi[frame], rmvpe_confidence[frame], 1.60),
                (world_midi[frame], world_confidence[frame], 0.75),
            ):
                if not np.isfinite(observation) or confidence <= 0:
                    continue
                direct = np.abs(states - observation)
                # A half/double-frequency estimate remains possible but costs
                # more than agreement at the correct octave.
                distance = np.minimum.reduce((direct, 0.95 + np.abs(states - (observation - 12)), 0.95 + np.abs(states - (observation + 12))))
                emission += weight * (0.35 + confidence) * np.minimum(distance, 5.0) * 1.8
            if local == 0:
                costs[local] = emission
                continue
            change_scale = 1.0 - 0.42 * onset_strength[frame] - 0.12 * beat_nearness[frame]
            transition = base_transition * max(0.42, change_scale)
            candidates = costs[local - 1][:, None] + transition
            back[local] = np.argmin(candidates, axis=0)
            costs[local] = emission + candidates[back[local], np.arange(len(states))]
        state = int(np.argmin(costs[-1]))
        decoded[end - 1] = states[state]
        for local in range(length - 1, 0, -1):
            state = int(back[local, state])
            decoded[start + local - 1] = states[state]
    return decoded


def extract_notes(
    vocals: Path,
    bpm: int,
    beat_phase: float = 0.0,
    beat_times: list[float] | None = None,
    onset_vocals: Path | None = None,
    existence_vocals: Path | None = None,
    transcript_words: list[dict] | None = None,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | bool]],
    float,
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    list[dict[str, float | str]],
    dict[str, float | int | str],
]:
    sample_rate = 22050
    beat_times = beat_times or []
    transcript_words = transcript_words or []
    # A 5 ms master grid preserves rapid attacks. Neural estimators running at
    # 10 ms are interpolated onto it, while 23/46/93 ms analysis windows vote
    # independently for rapid, normal and sustained singing.
    hop_length = 110
    audio, _ = librosa.load(vocals, sr=sample_rate, mono=True)
    audio, trim_index = librosa.effects.trim(audio, top_db=48)
    time_offset = float(trim_index[0]) / sample_rate
    f0, voiced_flag, voiced_probability = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=1024,
        hop_length=hop_length,
        resolution=0.05,
        fill_na=np.nan,
    )
    # A shorter independent window preserves consonant-adjacent and rapid notes
    # that the stable 2048-sample pYIN window can smear together.
    fast_f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=512,
        hop_length=hop_length,
    )
    stable_f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=2048,
        hop_length=hop_length,
    )
    spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop_length, center=True))
    for index in range(min(len(f0), spectrum.shape[1])):
        frequency = float(f0[index])
        if not np.isfinite(frequency):
            continue
        base_bin = int(round(frequency * 2048 / sample_rate))
        octave_bin = int(round(frequency * 2 * 2048 / sample_rate))
        if base_bin < 2 or octave_bin >= spectrum.shape[0] - 2:
            continue
        base_energy = float(np.max(spectrum[base_bin - 1:base_bin + 2, index]))
        octave_energy = float(np.max(spectrum[octave_bin - 1:octave_bin + 2, index]))
        if octave_energy > base_energy * 2.4 and frequency * 2 <= librosa.note_to_hz("C7"):
            f0[index] *= 2
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop_length, center=True)[0]
    frame_count = min(len(f0), len(rms))
    f0, rms = f0[:frame_count], rms[:frame_count]
    noise_floor = max(1e-5, float(np.percentile(rms, 25)) * 0.7)
    energy_probability = np.clip((rms - noise_floor) / (float(np.percentile(rms, 90)) - noise_floor + 1e-9), 0.0, 1.0)
    voiced_probability = np.asarray(voiced_probability[:frame_count], dtype=float)
    voiced_flag = np.asarray(voiced_flag[:frame_count], dtype=bool)
    pyin_confidence = voiced_probability * (0.35 + 0.65 * energy_probability)
    pyin_midi = librosa.hz_to_midi(f0)
    fast_midi = librosa.hz_to_midi(fast_f0[:frame_count])
    stable_midi = librosa.hz_to_midi(stable_f0[:frame_count])
    pyin_valid = np.isfinite(pyin_midi) & voiced_flag & (pyin_confidence >= 0.20)

    absolute_times = time_offset + np.arange(frame_count, dtype=float) * hop_length / sample_rate
    crepe_hz, crepe_confidence, rmvpe_hz, rmvpe_confidence, world_hz, world_confidence, rmvpe_candidate_hz, rmvpe_candidate_confidence = _extract_hq_pitch(vocals, absolute_times)
    crepe_midi = librosa.hz_to_midi(crepe_hz)
    rmvpe_midi = librosa.hz_to_midi(rmvpe_hz)
    world_midi = librosa.hz_to_midi(world_hz)
    rmvpe_candidate_midi = librosa.hz_to_midi(rmvpe_candidate_hz)
    crepe_valid = np.isfinite(crepe_midi) & (crepe_confidence >= 0.16)
    rmvpe_valid = np.isfinite(rmvpe_midi) & (rmvpe_confidence >= 0.055)
    world_valid = np.isfinite(world_midi) & (world_confidence >= 0.38)

    # Correct global tuning (for example a recording tuned to A=442 Hz) before
    # converting the contour to equal-tempered piano notes.
    tuning_values = np.concatenate((
        pyin_midi[pyin_valid & (pyin_confidence >= 0.45)],
        crepe_midi[crepe_valid & (crepe_confidence >= 0.35)],
        rmvpe_midi[rmvpe_valid & (rmvpe_confidence >= 0.18)],
        world_midi[world_valid & (world_confidence >= 0.48)],
    ))
    tuning_weights = np.concatenate((
        pyin_confidence[pyin_valid & (pyin_confidence >= 0.45)],
        crepe_confidence[crepe_valid & (crepe_confidence >= 0.35)],
        1.35 * rmvpe_confidence[rmvpe_valid & (rmvpe_confidence >= 0.18)],
        0.7 * world_confidence[world_valid & (world_confidence >= 0.48)],
    ))
    if tuning_values.size:
        residual = tuning_values - np.rint(tuning_values)
        tuning_semitones = float(np.clip(_weighted_median(residual, tuning_weights), -0.49, 0.49))
    else:
        tuning_semitones = 0.0
    pyin_midi = pyin_midi - tuning_semitones
    crepe_midi = crepe_midi - tuning_semitones
    rmvpe_midi = rmvpe_midi - tuning_semitones
    world_midi = world_midi - tuning_semitones
    fast_midi = fast_midi - tuning_semitones
    stable_midi = stable_midi - tuning_semitones
    rmvpe_candidate_midi = rmvpe_candidate_midi - tuning_semitones

    # Top-k RMVPE recovery: choose a lower-ranked salience peak only when it
    # agrees materially better with independent estimators. This specifically
    # targets fundamental/harmonic swaps without turning the decoder into a
    # simple majority vote.
    for frame in range(frame_count):
        independent = [value for value in (pyin_midi[frame], crepe_midi[frame], world_midi[frame], stable_midi[frame]) if np.isfinite(value)]
        if not independent:
            continue
        target = float(np.median(independent))
        best_value, best_cost = rmvpe_midi[frame], np.inf
        if np.isfinite(best_value):
            direct = abs(float(best_value) - target)
            best_cost = min(direct, abs(direct - 12.0))
        for rank in range(3):
            candidate = rmvpe_candidate_midi[frame, rank]
            candidate_conf = rmvpe_candidate_confidence[frame, rank]
            if not np.isfinite(candidate) or candidate_conf < 0.04:
                continue
            direct = abs(float(candidate) - target)
            cost = min(direct, abs(direct - 12.0)) + 0.10 * rank
            if cost + 0.28 < best_cost:
                best_value, best_cost = candidate, cost
        if np.isfinite(best_value):
            rmvpe_midi[frame] = best_value

    onset_audio = audio
    separation_delay_seconds = 0.0
    alignment_confidence = 0.0
    if onset_vocals and onset_vocals.exists() and onset_vocals != vocals:
        onset_full, _ = librosa.load(onset_vocals, sr=sample_rate, mono=True)
        start_sample = int(round(time_offset * sample_rate))
        onset_audio = onset_full[start_sample:start_sample + len(audio)]
        if len(onset_audio) < len(audio):
            onset_audio = np.pad(onset_audio, (0, len(audio) - len(onset_audio)))
        onset_audio, separation_delay_seconds, alignment_confidence = _align_vocal_timing(
            audio,
            onset_audio,
            sample_rate,
            hop_length,
        )
    boundary_strength, acoustic_boundaries, vowel_probability, boundary_evidence = _boundary_features(
        onset_audio, sample_rate, hop_length, frame_count
    )
    text_boundaries = np.zeros(frame_count, dtype=bool)
    mora_candidate_boundaries = np.zeros(frame_count, dtype=bool)
    character_candidate_boundaries = np.zeros(frame_count, dtype=bool)
    silent_text_boundaries = np.zeros(frame_count, dtype=bool)
    text_boundary_labels: dict[int, str] = {}
    transcription_boundary_count = 0
    mora_candidate_count = 0
    character_boundary_candidate_count = 0
    silent_mora_boundary_count = 0
    boundary_frame_seconds = hop_length / sample_rate
    refinement_radius = max(2, int(round(0.060 / boundary_frame_seconds)))
    for word in transcript_words:
        try:
            word_start = float(word.get("start", -1.0))
            word_text = str(word.get("text", "")).strip()
            word_confidence = float(word.get("confidence", 0.0))
        except (TypeError, ValueError, AttributeError):
            continue
        source_name = str(word.get("source", ""))
        forced_mora = "forced alignment" in source_name.lower() or "supplied-lyrics" in source_name.lower()
        character_candidate = bool(word.get("isCharacterBoundaryCandidate"))
        lyric_candidate = forced_mora or character_candidate
        if word_start < 0 or not word_text or (word_confidence < 0.12 and not lyric_candidate):
            continue
        estimated = int(round((word_start - time_offset) / boundary_frame_seconds))
        if not 0 <= estimated < frame_count:
            continue
        left, right = max(0, estimated - refinement_radius), min(frame_count, estimated + refinement_radius + 1)
        acoustic_candidates = np.flatnonzero(acoustic_boundaries[left:right]) + left
        if acoustic_candidates.size:
            refined = int(acoustic_candidates[np.argmax(boundary_strength[acoustic_candidates])])
        else:
            refined = estimated
        # Supplied Japanese lyrics are aligned as morae.  Every ordinary mora
        # is a mandatory split candidate even when its consonant is weak. Small
        # kana and long vowels have already been joined by _moras(). A sokuon
        # marks silence/closure but must never create a pitched piano note.
        if word_text == "っ":
            silent_text_boundaries[refined] = True
            text_boundary_labels[refined] = word_text
            silent_mora_boundary_count += 1
        elif lyric_candidate:
            # A supplied lyric mora is only a boundary candidate. Text alone
            # must not create a piano reattack without acoustic/pitch support.
            mora_candidate_boundaries[refined] = True
            if character_candidate:
                character_candidate_boundaries[refined] = True
                character_boundary_candidate_count += 1
            text_boundary_labels[refined] = word_text
            mora_candidate_count += 1
        elif float(boundary_strength[refined]) >= 0.32:
            text_boundaries[refined] = True
            text_boundary_labels[refined] = word_text
            transcription_boundary_count += 1
    onset = boundary_strength.copy()
    rapid_candidate = np.isfinite(fast_midi) & (energy_probability >= 0.16) & (onset >= 0.24)
    # Prefer the short-window estimate only at a transient and only when it is
    # supported by CREPE or pYIN is absent. This avoids making sustained notes noisy.
    neural_support = (
        np.isfinite(rmvpe_midi) & (np.minimum(np.abs(fast_midi - rmvpe_midi), np.abs(np.abs(fast_midi - rmvpe_midi) - 12)) < 0.9)
    ) | (
        np.isfinite(crepe_midi) & (np.minimum(np.abs(fast_midi - crepe_midi), np.abs(np.abs(fast_midi - crepe_midi) - 12)) < 0.9)
    )
    fast_supported = ~pyin_valid | neural_support
    use_fast = rapid_candidate & fast_supported
    pyin_midi[use_fast] = fast_midi[use_fast]
    pyin_confidence[use_fast] = np.maximum(pyin_confidence[use_fast], 0.42 + 0.28 * onset[use_fast])
    pyin_valid[use_fast] = True
    beat_seconds = 60.0 / max(1, bpm)
    sixteenth = beat_seconds / 4
    grid_distance = np.abs(((absolute_times - beat_phase + sixteenth / 2) % sixteenth) - sixteenth / 2)
    beat_nearness = np.exp(-np.square(grid_distance / max(0.018, sixteenth * 0.18)))

    frame_seconds = hop_length / sample_rate
    gap_frames = max(2, int(round(min(0.10, beat_seconds / 8) / frame_seconds)))

    def character_boundary_evidence(index: int) -> tuple[bool, bool]:
        absolute_time = float(time_offset + index * frame_seconds)
        _, score_division, score_error = _tempo_map_snap_info(absolute_time, beat_times, bpm, beat_phase)
        score_tolerance = max(0.012, min(0.028, beat_seconds / 18.0))
        score_supported = score_division > 0 and score_error <= score_tolerance
        vowel_radius = max(2, int(round(0.045 / frame_seconds)))
        centre_radius = max(1, int(round(0.010 / frame_seconds)))
        before_vowels = vowel_probability[max(0, index - vowel_radius):max(0, index - centre_radius)]
        after_vowels = vowel_probability[min(frame_count, index + centre_radius + 1):min(frame_count, index + vowel_radius + 1)]
        centre_vowels = vowel_probability[max(0, index - centre_radius):min(frame_count, index + centre_radius + 1)]
        vowel_reset = bool(
            before_vowels.size
            and after_vowels.size
            and centre_vowels.size
            and float(np.mean(before_vowels)) >= 0.20
            and float(np.mean(after_vowels)) >= 0.20
            and float(np.mean(centre_vowels)) <= 0.82 * min(float(np.mean(before_vowels)), float(np.mean(after_vowels)))
        )
        return score_supported, vowel_reset

    # Confirm the strongest character/acoustic boundaries before any gap fill
    # or pitch smoothing. Later stages may add pitch-supported boundaries, but
    # they are never allowed to erase these hard segmentation cuts.
    pre_smoothing_boundaries = silent_text_boundaries.copy()
    if BOUNDARY_FIRST_SEGMENTATION:
        pre_smoothing_boundaries |= acoustic_boundaries & (boundary_strength >= 0.62)
        for index in np.flatnonzero(character_candidate_boundaries):
            score_supported, vowel_reset = character_boundary_evidence(int(index))
            if float(boundary_strength[index]) >= 0.46 and score_supported and vowel_reset:
                text_boundaries[index] = True
                pre_smoothing_boundaries[index] = True
                transcription_boundary_count += 1

    # RMVPE leads, while any two independent trackers can sustain a frame.
    vote_count = pyin_valid.astype(np.int8) + crepe_valid.astype(np.int8) + rmvpe_valid.astype(np.int8) + world_valid.astype(np.int8)
    valid = rmvpe_valid | (vote_count >= 2)
    valid = _fill_short_false_runs(
        valid,
        gap_frames,
        pre_smoothing_boundaries if BOUNDARY_FIRST_SEGMENTATION else None,
    )
    if existence_vocals and existence_vocals.exists() and existence_vocals != vocals:
        existence_full, _ = librosa.load(existence_vocals, sr=sample_rate, mono=True)
        start_sample = int(round(time_offset * sample_rate))
        existence_audio = existence_full[start_sample:start_sample + len(audio)]
        if len(existence_audio) < len(audio):
            existence_audio = np.pad(existence_audio, (0, len(audio) - len(existence_audio)))
        existence_rms = librosa.feature.rms(y=existence_audio, frame_length=1024, hop_length=hop_length, center=True)[0][:frame_count]
        existence_rms = np.pad(existence_rms, (0, max(0, frame_count - len(existence_rms))))
        existence_support = np.clip(existence_rms / (float(np.percentile(existence_rms, 90)) + 1e-9), 0.0, 1.0)
        valid = _bridge_supported_false_runs(
            valid,
            existence_support,
            max(gap_frames, int(round(beat_seconds / 16 / frame_seconds))),
            pre_smoothing_boundaries if BOUNDARY_FIRST_SEGMENTATION else None,
        )
    rounded = _score_decode(
        pyin_midi,
        pyin_confidence,
        crepe_midi,
        crepe_confidence,
        rmvpe_midi,
        rmvpe_confidence,
        world_midi,
        world_confidence,
        valid,
        onset,
        beat_nearness,
    )

    # Protect persistent pitch changes and acoustic reattacks before applying
    # any temporal smoothing. This keeps melismas and rapid notes separable.
    pitch_boundaries = np.zeros(frame_count, dtype=bool)
    for index in range(1, frame_count - 2):
        if not np.isfinite(rounded[index]) or not np.isfinite(rounded[index - 1]):
            continue
        if rounded[index] == rounded[index - 1]:
            continue
        persistence = rounded[index:min(frame_count, index + 3)]
        if len(persistence) >= 2 and np.count_nonzero(persistence == rounded[index]) >= 2:
            pitch_boundaries[index] = True

    boundary_raw_pitch = np.where(
        np.isfinite(rmvpe_midi),
        rmvpe_midi,
        np.where(
            np.isfinite(crepe_midi),
            crepe_midi,
            np.where(np.isfinite(pyin_midi), pyin_midi, stable_midi),
        ),
    )
    same_pitch_reattacks, same_pitch_reattack_scores, same_pitch_phonetic_starts = _detect_same_pitch_reattacks(
        rounded,
        boundary_raw_pitch,
        valid,
        boundary_evidence,
        boundary_strength,
        mora_candidate_boundaries | character_candidate_boundaries | text_boundaries,
        vowel_probability,
        absolute_times,
        beat_times,
        bpm,
        beat_phase,
        frame_seconds,
    )

    # Confirm lyric mora candidates only with independent evidence. Strong
    # acoustic attacks are sufficient; weaker attacks also need a plausible
    # tempo-map position or a nearby persistent pitch change.
    candidate_indices = np.flatnonzero(mora_candidate_boundaries)
    pitch_radius = max(1, int(round(0.030 / frame_seconds)))
    for index in candidate_indices:
        strength = float(boundary_strength[index])
        left = max(0, index - pitch_radius)
        right = min(frame_count, index + pitch_radius + 1)
        near_pitch_change = bool(np.any(pitch_boundaries[left:right]))
        score_supported, vowel_reset = character_boundary_evidence(int(index))
        # Lyrics may confirm a boundary, but must never lower the acoustic
        # threshold unless a character candidate also coincides with a
        # consonant/vowel reset and a valid score position. This specifically
        # recovers same-pitch reattacks without splitting every character.
        character_reattack = (
            bool(character_candidate_boundaries[index])
            and strength >= 0.46
            and score_supported
            and vowel_reset
        )
        confirmed = (
            strength >= 0.62
            and (bool(acoustic_boundaries[index]) or score_supported or near_pitch_change)
        ) or character_reattack
        if confirmed and not text_boundaries[index]:
            text_boundaries[index] = True
            transcription_boundary_count += 1
    protected_boundaries = (
        acoustic_boundaries
        | pitch_boundaries
        | text_boundaries
        | silent_text_boundaries
        | same_pitch_reattacks
    )

    # Keep a continuous curve for visual inspection, but fold octave mistakes
    # toward the score-decoded state before combining the two estimators.
    raw_consensus_midi = np.full(frame_count, np.nan, dtype=float)
    confidence = np.maximum.reduce((pyin_confidence, crepe_confidence, rmvpe_confidence, world_confidence))
    for index in range(frame_count):
        if not np.isfinite(rounded[index]):
            continue
        values: list[float] = []
        weights: list[float] = []
        for value, weight, reliability in (
            (pyin_midi[index], pyin_confidence[index], 0.65),
            (crepe_midi[index], crepe_confidence[index], 1.05),
            (rmvpe_midi[index], rmvpe_confidence[index], 1.60),
            (world_midi[index], world_confidence[index], 0.75),
            (stable_midi[index], energy_probability[index], 0.42),
            (fast_midi[index], onset[index], 0.38),
        ):
            if not np.isfinite(value) or weight <= 0:
                continue
            candidates = value + 12 * np.arange(-2, 3)
            values.append(float(candidates[np.argmin(np.abs(candidates - rounded[index]))]))
            weights.append(float(weight * reliability))
        raw_consensus_midi[index] = float(np.average(values, weights=weights)) if values else float(rounded[index])
    smoothed_midi = _edge_preserving_pitch_smooth(
        raw_consensus_midi,
        confidence,
        valid,
        protected_boundaries,
    )
    word_segments, syllable_segments, vowel_segments = _acoustic_segments(
        absolute_times,
        valid,
        acoustic_boundaries | same_pitch_reattacks,
        boundary_strength,
        vowel_probability,
    )

    # The lower bound follows the tempo and the finest supported (1/16 beat)
    # score unit. It is deliberately not the former fixed 80 ms filter.
    finest_grid = beat_seconds / 16
    # Below 40 ms, a vocal F0 change is overwhelmingly more likely to be a
    # transition or tracker flicker than a separately playable piano note.
    playable_note_seconds = float(np.clip(finest_grid * 0.68, 0.040, 0.070))
    onset_change_frames = max(2, int(round(0.025 / frame_seconds)))
    stable_change_frames = max(onset_change_frames, int(round(min(0.070, beat_seconds / 6) / frame_seconds)))
    notes: list[dict[str, float | int | str]] = []
    current: dict[str, float | int | str] | None = None
    unvoiced_frames = 0
    pending_pitch: int | None = None
    pending_frames = 0
    boundary_reentry = False

    for index, value in enumerate(rounded):
        time = time_offset + index * frame_seconds
        if not np.isfinite(value):
            unvoiced_frames += 1
            pending_pitch, pending_frames = None, 0
            protected_gap = bool(
                BOUNDARY_FIRST_SEGMENTATION
                and (text_boundaries[index] or silent_text_boundaries[index])
            )
            if current is not None and (protected_gap or unvoiced_frames >= gap_frames):
                current["end"] = time - (unvoiced_frames - 1) * frame_seconds
                notes.append(current)
                current = None
                boundary_reentry = protected_gap
            continue

        unvoiced_frames = 0
        pitch = int(value)
        strength = float(np.clip(confidence[index], 0.0, 1.0))
        if current is None:
            current = {"midi": pitch, "start": time, "end": time + frame_seconds, "strength": strength, "onset": float(onset[index]), "role": "melody"}
            if boundary_reentry:
                current["reattack"] = True
                current["boundaryFirstReentry"] = True
            boundary_reentry = False
        elif int(current["midi"]) == pitch:
            pending_pitch, pending_frames = None, 0
            reattack_spacing = max(0.045, min(0.070, beat_seconds / 8))
            local_peak = (
                (bool(acoustic_boundaries[index]) and float(boundary_strength[index]) >= 0.62)
                or bool(text_boundaries[index])
                or bool(same_pitch_reattacks[index])
            )
            if local_peak and time - float(current["start"]) >= reattack_spacing:
                phonetic_start = (
                    float(same_pitch_phonetic_starts[index])
                    if bool(same_pitch_reattacks[index]) and np.isfinite(same_pitch_phonetic_starts[index])
                    else time
                )
                current["end"] = max(float(current["start"]) + frame_seconds, phonetic_start)
                notes.append(current)
                current = {
                    "midi": pitch,
                    "start": time,
                    "end": time + frame_seconds,
                    "strength": strength,
                    "onset": float(onset[index]),
                    "reattack": True,
                    "samePitchReattack": bool(same_pitch_reattacks[index]),
                    "phoneticStart": phonetic_start,
                    "musicalOnset": time,
                    "consonantStart": phonetic_start,
                    "vowelStart": time,
                    "role": "melody",
                }
                continue
            current["end"] = time + frame_seconds
            current["strength"] = max(float(current["strength"]), strength)
        else:
            if pending_pitch == pitch:
                pending_frames += 1
            else:
                pending_pitch, pending_frames = pitch, 1
            pending_start = max(0, index - pending_frames + 1)
            # A pitch bend also produces spectral flux.  Only a detected local
            # acoustic peak may shorten the pitch-persistence requirement;
            # the former loose 0.30 strength test split scoops/falls into a
            # chain of 45–70 ms piano notes.
            local_boundary_mask = acoustic_boundaries[pending_start:index + 1]
            local_boundary_strength = boundary_strength[pending_start:index + 1]
            strong_onset = bool(np.any(local_boundary_mask & (local_boundary_strength >= 0.50))) or bool(np.any(text_boundaries[pending_start:index + 1]))
            interval = abs(pitch - int(current["midi"]))
            required_frames = onset_change_frames if strong_onset else stable_change_frames + (1 if interval <= 2 else 0)
            if pending_frames >= required_frames:
                boundary = time - (pending_frames - 1) * frame_seconds
                current["end"] = boundary
                notes.append(current)
                current = {
                    "midi": pitch,
                    "start": boundary,
                    "end": time + frame_seconds,
                    "strength": strength,
                    "onset": float(onset[max(0, index - pending_frames + 1)]),
                    "reattack": strong_onset,
                    "role": "melody",
                }
                pending_pitch, pending_frames = None, 0

    if current is not None:
        notes.append(current)

    # Syllable-aware scoop folding.  A scoop is a short low lead-in after a
    # consonant, followed by one continuous vowel rising into a stable target.
    # It must not become a separate piano note.  Conversely, a new consonant
    # at the pitch boundary means a genuine new note and is never folded.
    scoop_merge_count = 0
    note_index = 0
    while note_index + 1 < len(notes):
        lead_in, target = notes[note_index], notes[note_index + 1]
        lead_duration = float(lead_in["end"]) - float(lead_in["start"])
        target_duration = float(target["end"]) - float(target["start"])
        interval_up = int(target["midi"]) - int(lead_in["midi"])
        touching = float(target["start"]) - float(lead_in["end"]) <= frame_seconds * 2.0
        start_frame_index = int(np.clip(round((float(lead_in["start"]) - time_offset) / frame_seconds), 0, frame_count - 1))
        join_frame_index = int(np.clip(round((float(target["start"]) - time_offset) / frame_seconds), 0, frame_count - 1))
        consonant_radius = max(2, int(round(0.045 / frame_seconds)))
        consonant_left = max(0, start_frame_index - consonant_radius)
        consonant_right = min(frame_count, start_frame_index + consonant_radius + 1)
        consonant_start = bool(np.any(acoustic_boundaries[consonant_left:consonant_right])) and float(np.max(boundary_strength[consonant_left:consonant_right])) >= 0.48
        join_radius = max(1, int(round(0.025 / frame_seconds)))
        join_left, join_right = max(0, join_frame_index - join_radius), min(frame_count, join_frame_index + join_radius + 1)
        internal_reattack = bool(
            np.any(acoustic_boundaries[join_left:join_right] & (boundary_strength[join_left:join_right] >= 0.58))
            or np.any(same_pitch_reattacks[join_left:join_right])
        )
        vowel_left = max(start_frame_index, join_frame_index - max(2, int(round(0.055 / frame_seconds))))
        vowel_right = min(frame_count, join_frame_index + max(3, int(round(0.080 / frame_seconds))))
        continuous_vowel = vowel_right > vowel_left and float(np.mean(vowel_probability[vowel_left:vowel_right])) >= 0.30
        onset_vowel_right = min(frame_count, start_frame_index + max(2, int(round(0.030 / frame_seconds))))
        onset_vowel = float(np.mean(vowel_probability[start_frame_index:onset_vowel_right])) if onset_vowel_right > start_frame_index else 0.0
        body_vowel = float(np.mean(vowel_probability[join_left:vowel_right])) if vowel_right > join_left else 0.0
        consonant_to_vowel = consonant_start and body_vowel >= onset_vowel + 0.04
        scoop_shape = 1 <= interval_up <= 4 and 0.040 <= lead_duration <= min(0.220, beat_seconds * 0.55) and target_duration >= 0.090
        if touching and scoop_shape and consonant_to_vowel and continuous_vowel and not internal_reattack:
            target["start"] = lead_in["start"]
            target["strength"] = max(float(target["strength"]), float(lead_in["strength"]))
            target["scoopMerged"] = True
            del notes[note_index]
            scoop_merge_count += 1
            note_index = max(0, note_index - 1)
            continue
        note_index += 1

    # For true attacks, use the acoustic consonant/onset peak rather than the
    # centre of a pitch-analysis window.  This corrects audible 20–80 ms note
    # lag without quantising legato pitch changes to the beat grid.
    timing_adjustment_count = 0
    previous_end = 0.0
    for note_index, note in enumerate(notes):
        start = float(note["start"])
        separated_from_previous = note_index == 0 or start - previous_end >= 0.055
        if bool(note.get("reattack")) or separated_from_previous:
            start_frame_index = int(np.clip(round((start - time_offset) / frame_seconds), 0, frame_count - 1))
            before = max(1, int(round(0.080 / frame_seconds)))
            after = max(1, int(round(0.030 / frame_seconds)))
            left, right = max(0, start_frame_index - before), min(frame_count, start_frame_index + after + 1)
            if bool(note.get("samePitchReattack")):
                timing_boundaries = same_pitch_reattacks[left:right]
            else:
                timing_boundaries = acoustic_boundaries[left:right] & (boundary_strength[left:right] >= 0.52)
            candidates = np.flatnonzero(timing_boundaries) + left
            if candidates.size:
                timing_scores = boundary_strength[candidates] + 0.12 * np.minimum(
                    same_pitch_reattack_scores[candidates], 6.0
                )
                best = int(candidates[np.argmax(timing_scores)])
                corrected = time_offset + best * frame_seconds
                corrected = max(previous_end, corrected)
                if abs(corrected - start) <= 0.085 and corrected < float(note["end"]) - playable_note_seconds:
                    note["timingAdjustmentMs"] = round((corrected - start) * 1000.0, 2)
                    note["start"] = corrected
                    timing_adjustment_count += 1
        previous_end = float(note["end"])

    # A short segment is not automatically an ornament. Keep it when it has a
    # clear onset or a good score-grid fit; this preserves rapid real notes.
    ornament_seconds = float(np.clip(beat_seconds / 8, 0.045, 0.095))
    index = 0
    while index < len(notes):
        note = notes[index]
        duration = float(note["end"]) - float(note["start"])
        previous = notes[index - 1] if index > 0 else None
        following = notes[index + 1] if index + 1 < len(notes) else None
        octave_flash = any(neighbour is not None and abs(int(note["midi"]) - int(neighbour["midi"])) >= 8 for neighbour in (previous, following))
        _, score_division, score_error = _tempo_map_snap_info(float(note["start"]), beat_times, bpm, beat_phase)
        score_fit = score_division > 0 and score_error <= max(0.010, beat_seconds / score_division * 0.18)
        note_start_frame = max(0, int(round((float(note["start"]) - time_offset) / frame_seconds)))
        note_end_frame = min(frame_count, max(note_start_frame + 1, int(round((float(note["end"]) - time_offset) / frame_seconds))))
        note_support = float(np.mean(vote_count[note_start_frame:note_end_frame]))
        weak_ornament = (
            duration < ornament_seconds
            and not bool(note.get("reattack"))
            and (not score_fit or note_support < 2.8)
        )
        weak_ornament = weak_ornament or (octave_flash and duration < playable_note_seconds and not bool(note.get("reattack")))
        if not weak_ornament:
            index += 1
            continue
        if previous is not None and following is not None and previous["midi"] == following["midi"]:
            previous["end"] = following["end"]
            previous["strength"] = max(float(previous["strength"]), float(following["strength"]))
            del notes[index:index + 2]
        elif following is not None and float(following["start"]) - float(note["end"]) <= frame_seconds * 1.5:
            following["start"] = note["start"]
            del notes[index]
        elif previous is not None and float(note["start"]) - float(previous["end"]) <= frame_seconds * 1.5:
            previous["end"] = note["end"]
            del notes[index]
            index = max(0, index - 1)
        else:
            index += 1

    notes = [note for note in notes if float(note["end"]) - float(note["start"]) >= playable_note_seconds]
    merged: list[dict[str, float | int | str]] = []
    for note in notes:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous["midi"] == note["midi"]
            and not bool(note.get("reattack"))
            and float(note["start"]) - float(previous["end"]) < min(0.055, beat_seconds / 12)
        ):
            previous["end"] = note["end"]
            previous["strength"] = max(float(previous["strength"]), float(note["strength"]))
        else:
            merged.append(note)

    # Keep performance timing for piano playback/video. Score-grid timing is
    # stored separately for notation so quantisation cannot move the piano
    # against the singer.
    detailed_notes: list[dict[str, float | int | str]] = []
    short_terminal_pitch_correction_count = 0
    for merged_index, note in enumerate(merged):
        raw_start, raw_end = float(note["start"]), float(note["end"])
        score_start, score_end, score_division, duration_division, score_error, duration_error = _tempo_map_note_span_info(
            raw_start,
            raw_end,
            beat_times,
            bpm,
            beat_phase,
        )
        start = max(0.0, raw_start)
        end = max(start + playable_note_seconds, raw_end)
        score_start = max(0.0, score_start)
        score_end = max(score_start + playable_note_seconds, score_end)
        start_frame = max(0, int(round((raw_start - time_offset) / frame_seconds)))
        end_frame = min(frame_count, max(start_frame + 1, int(round((raw_end - time_offset) / frame_seconds))))
        # Long notes discard more of their expressive edges. Very short notes
        # keep more frames so a real rapid note is not decided from one sample.
        note_frame_count = end_frame - start_frame
        margin_ratio = 0.22 if raw_end - raw_start >= 0.16 else 0.10
        margin = int(note_frame_count * margin_ratio)
        stable_start = min(end_frame - 1, start_frame + margin)
        stable_end = max(stable_start + 1, end_frame - margin)
        region_valid = valid[stable_start:stable_end]
        region_pitch = smoothed_midi[stable_start:stable_end][region_valid]
        region_confidence = confidence[stable_start:stable_end][region_valid]
        if region_pitch.size:
            stable_pitch = _weighted_median(region_pitch, np.maximum(region_confidence, 1e-4))
            note_confidence = float(np.mean(region_confidence))
        else:
            stable_pitch = float(note["midi"])
            note_confidence = float(note["strength"])
        # Prefer the dominant score-decoded pitch when it owns a clear majority
        # of the stable region. A weighted median can otherwise land between
        # two pitches during a fast transition and round to a nonexistent note.
        # The ornament-first curve is a boundary/segmentation aid only.  It may
        # intentionally replace a sung transition with a neighbouring nucleus,
        # so it must never vote on the final piano pitch.  Final pitch keeps the
        # original multi-model score decoder that proved substantially more
        # accurate in the v35 reference evaluation.
        decoded_region = rounded[stable_start:stable_end][region_valid]
        midi_note = int(np.clip(round(stable_pitch), 21, 108))
        decoded_weights: dict[int, float] = {}
        for value, weight in zip(decoded_region, np.maximum(region_confidence, 1e-4)):
            if np.isfinite(value):
                decoded_weights[int(round(float(value)))] = decoded_weights.get(int(round(float(value))), 0.0) + float(weight)
        dominant_pitch_support = 0.0
        if decoded_weights:
            dominant_pitch = max(decoded_weights, key=decoded_weights.get)
            dominant_pitch_support = decoded_weights[dominant_pitch] / max(1e-9, sum(decoded_weights.values()))
            if dominant_pitch_support >= 0.54:
                midi_note = int(np.clip(dominant_pitch, 21, 108))
        # A 60–150 ms sung note can spend most of its centre in a scoop/fall.
        # Correct only this narrow case, and only when the terminal nucleus is
        # stable, independently supported, and improves continuity to the next
        # already-segmented note. Longer notes are never changed here.
        short_terminal_pitch_corrected = False
        note_duration = raw_end - raw_start
        preliminary_spread = (
            float(np.nanpercentile(region_pitch, 90) - np.nanpercentile(region_pitch, 10))
            if region_pitch.size >= 3 else 0.0
        )
        if (
            0.045 <= note_duration <= 0.151
            and preliminary_spread >= 1.15
            and merged_index + 1 < len(merged)
        ):
            terminal_frames = max(4, int(round(0.040 / frame_seconds)))
            terminal_start = max(start_frame, end_frame - terminal_frames)
            terminal_values = raw_consensus_midi[terminal_start:end_frame]
            terminal_values = terminal_values[np.isfinite(terminal_values)]
            if terminal_values.size >= 4:
                terminal_value = float(np.median(terminal_values))
                terminal_mad = float(np.median(np.abs(terminal_values - terminal_value)))
                terminal_pitch = int(np.clip(np.floor(terminal_value + 0.5), 21, 108))
                terminal_models = np.vstack((
                    pyin_midi[terminal_start:end_frame],
                    crepe_midi[terminal_start:end_frame],
                    rmvpe_midi[terminal_start:end_frame],
                    world_midi[terminal_start:end_frame],
                ))
                finite_terminal_models = np.isfinite(terminal_models)
                terminal_model_support = float(
                    np.count_nonzero(finite_terminal_models & (np.abs(terminal_models - terminal_pitch) < 0.75))
                    / max(1, np.count_nonzero(finite_terminal_models))
                )
                next_pitch = int(merged[merged_index + 1]["midi"])
                improves_continuity = abs(terminal_pitch - next_pitch) + 0.25 < abs(midi_note - next_pitch)
                if (
                    1 <= abs(terminal_pitch - midi_note) <= 3
                    and terminal_mad <= 0.35
                    and terminal_model_support >= 0.55
                    and improves_continuity
                ):
                    midi_note = terminal_pitch
                    stable_pitch = terminal_value
                    short_terminal_pitch_corrected = True
                    short_terminal_pitch_correction_count += 1
        octave_candidates = [candidate for candidate in (midi_note - 12, midi_note, midi_note + 12) if 21 <= candidate <= 108]
        candidate_evidence: dict[int, dict[str, float]] = {}
        estimator_matrix = np.vstack((
            pyin_midi[start_frame:end_frame],
            crepe_midi[start_frame:end_frame],
            rmvpe_midi[start_frame:end_frame],
            world_midi[start_frame:end_frame],
            stable_midi[start_frame:end_frame],
        ))
        for candidate in octave_candidates:
            finite = np.isfinite(estimator_matrix)
            model_support = float(np.sum(finite & (np.abs(estimator_matrix - candidate) < 0.75)) / max(1, end_frame - start_frame))
            harmonic_support = _harmonic_support(spectrum, candidate, start_frame, end_frame, sample_rate, 2048)
            candidate_evidence[candidate] = {
                "model": model_support,
                "harmonic": harmonic_support,
                "combined": model_support + 0.30 * harmonic_support,
            }
        original_midi_note = midi_note
        best_octave = max(octave_candidates, key=lambda candidate: candidate_evidence[candidate]["combined"])
        original_evidence = candidate_evidence[original_midi_note]
        best_evidence = candidate_evidence[best_octave]
        octave_corrected = False
        octave_ambiguous = False
        if best_octave != original_midi_note:
            evidence_margin = best_evidence["combined"] - original_evidence["combined"]
            if evidence_margin >= 0.85 and best_evidence["model"] >= original_evidence["model"] + 0.55:
                midi_note = best_octave
                octave_corrected = True
            elif evidence_margin >= 0.30:
                octave_ambiguous = True
        agreement_samples: list[float] = []
        support_samples: list[int] = []
        for frame in range(start_frame, end_frame):
            estimates = [
                pyin_midi[frame],
                crepe_midi[frame],
                rmvpe_midi[frame],
                world_midi[frame],
            ]
            available = [float(value) for value in estimates if np.isfinite(value)]
            if not available:
                continue
            support = sum(abs(value - midi_note) < 0.75 for value in available)
            support_samples.append(support)
            # One strong tracker is useful, but it is not independent consensus.
            agreement_samples.append(support / len(available) if len(available) >= 2 else 0.45 * support)
        source_agreement = float(np.mean(agreement_samples)) if agreement_samples else 0.0
        estimator_support = float(np.mean(support_samples)) if support_samples else 0.0
        rmvpe_region = rmvpe_confidence[start_frame:end_frame]
        rmvpe_note_confidence = float(np.mean(rmvpe_region[np.isfinite(rmvpe_midi[start_frame:end_frame])])) if np.any(np.isfinite(rmvpe_midi[start_frame:end_frame])) else 0.0
        pitch_spread = float(np.nanpercentile(region_pitch, 90) - np.nanpercentile(region_pitch, 10)) if region_pitch.size >= 3 else 0.0
        stable = pitch_spread <= 1.40
        classification = "note"
        ornaments: list[str] = []
        if bool(note.get("scoopMerged")):
            ornaments.append("scoop")
        full_region = raw_consensus_midi[start_frame:end_frame]
        full_region = full_region[np.isfinite(full_region)]
        if full_region.size >= 5:
            edge = max(1, int(round(full_region.size * 0.22)))
            start_delta = float(np.median(full_region[:edge]) - stable_pitch)
            end_delta = float(np.median(full_region[-edge:]) - stable_pitch)
            if start_delta <= -0.55 and abs(end_delta) < 0.75:
                ornaments.append("scoop")
            if end_delta <= -0.55 and abs(start_delta) < 0.75:
                ornaments.append("fall")
            deviation = full_region - float(np.median(full_region))
            crossings = int(np.count_nonzero(np.diff(np.signbit(deviation))))
            if raw_end - raw_start >= 0.24 and crossings >= 4 and 0.22 <= float(np.std(deviation)) <= 1.25:
                ornaments.append("vibrato")
            slope = float(np.polyfit(np.arange(full_region.size), full_region, 1)[0])
            if pitch_spread >= 1.8 and abs(slope) * full_region.size >= 1.4:
                ornaments.append("portamento")
        if not stable and note_confidence < 0.38:
            classification = "transition"
        elif raw_end - raw_start < ornament_seconds and float(note.get("onset", 0.0)) < 0.34 and score_division == 0:
            classification = "ornament-transition"
        ornaments = sorted(set(ornaments))
        review_status = "ok"
        if note_confidence < 0.24 or (estimator_support < 1.6 and rmvpe_note_confidence < 0.32) or source_agreement < 0.34 or pitch_spread > 1.80:
            review_status = "uncertain"
        ratio = _tuplet_ratio(score_division)
        detailed_note = {
            "midi": midi_note,
            "start": start,
            "end": end,
            "rawStart": raw_start,
            "rawEnd": raw_end,
            "scoreStart": score_start,
            "scoreEnd": score_end,
            "pitchMidi": round(stable_pitch, 4),
            "pitchCents": round((stable_pitch - midi_note) * 100, 2),
            "strength": round(note_confidence, 4),
            "sourceAgreement": round(source_agreement, 4),
            "estimatorSupport": round(estimator_support, 3),
            "rmvpeConfidence": round(rmvpe_note_confidence, 4),
            "octaveEvidence": {
                str(candidate): {name: round(value, 4) for name, value in evidence.items()}
                for candidate, evidence in candidate_evidence.items()
            },
            "octaveCorrected": octave_corrected,
            "shortTerminalPitchCorrected": short_terminal_pitch_corrected,
            "onset": round(float(note.get("onset", 0.0)), 4),
            "boundaryType": "reattack" if bool(note.get("reattack")) else "pitch-or-syllable",
            "samePitchReattack": bool(note.get("samePitchReattack")),
            "phoneticStart": round(float(note.get("phoneticStart", raw_start)), 6),
            "musicalOnset": round(float(note.get("musicalOnset", raw_start)), 6),
            "consonantStart": round(float(note.get("consonantStart", note.get("phoneticStart", raw_start))), 6),
            "vowelStart": round(float(note.get("vowelStart", note.get("musicalOnset", raw_start))), 6),
            "vowelEnd": round(raw_end, 6),
            "timingAdjustmentMs": float(note.get("timingAdjustmentMs", 0.0)),
            "pitchSpread": round(pitch_spread, 4),
            "scoreDivision": int(score_division),
            "scoreDurationDivision": int(duration_division),
            "scoreErrorMs": round(score_error * 1000, 2),
            "scoreDurationErrorMs": round(duration_error * 1000, 2),
            "dominantPitchSupport": round(dominant_pitch_support, 4),
            "classification": classification,
            "ornaments": ornaments,
            "reviewStatus": review_status,
            "role": "melody",
        }
        if ratio:
            detailed_note["tupletActual"], detailed_note["tupletNormal"] = ratio
        if octave_ambiguous:
            detailed_note["pitchAlternatives"] = octave_candidates
            detailed_note["reviewStatus"] = "octave-review"
        detailed_notes.append(detailed_note)

    # Reject extremely dispersed portamento fragments.  Their pitch range is
    # wider than a musical note and independent trackers do not agree; keeping
    # them creates isolated high keys between otherwise stable phrases.
    before_extreme_filter = len(detailed_notes)
    filtered_extreme_notes: list[dict] = []
    for index, note in enumerate(detailed_notes):
        neighbours = [
            int(detailed_notes[other]["midi"])
            for other in (index - 1, index + 1)
            if 0 <= other < len(detailed_notes)
        ]
        isolated_leap = len(neighbours) == 2 and min(abs(int(note["midi"]) - value) for value in neighbours) >= 8
        extreme_fragment = (
            float(note.get("pitchSpread", 0.0)) > 6.0
            and "portamento" in note.get("ornaments", [])
            and (float(note.get("sourceAgreement", 1.0)) < 0.25 or isolated_leap)
        )
        if not extreme_fragment:
            filtered_extreme_notes.append(note)
    detailed_notes = filtered_extreme_notes
    suppressed_extreme_transition_count = before_extreme_filter - len(detailed_notes)

    # Preserve genuinely rapid notes only when a vocal attack and independent
    # pitch consensus both support them. This removes vibrato fragments without
    # imposing a triplet/quintuplet ceiling.
    rapid_limit = float(np.clip(beat_seconds / 4, 0.060, 0.140))
    detailed_notes = [
        note for note in detailed_notes
        if float(note["end"]) - float(note["start"]) >= rapid_limit
        or (
            float(note.get("onset", 0.0)) >= 0.32
            and float(note.get("estimatorSupport", 0.0)) >= 1.8
            and float(note.get("sourceAgreement", 0.0)) >= 0.42
        )
    ]

    # A short, stepwise pitch between two stable neighbours is retained as a
    # playable passing note only when it has an attack; otherwise it is marked
    # as ornamental for review rather than silently becoming a piano key.
    for index, note in enumerate(detailed_notes):
        if index == 0 or index + 1 >= len(detailed_notes):
            continue
        duration_value = float(note["end"]) - float(note["start"])
        previous_note, following_note = detailed_notes[index - 1], detailed_notes[index + 1]
        between = min(int(previous_note["midi"]), int(following_note["midi"])) <= int(note["midi"]) <= max(int(previous_note["midi"]), int(following_note["midi"]))
        stepwise = abs(int(note["midi"]) - int(previous_note["midi"])) <= 3 and abs(int(following_note["midi"]) - int(note["midi"])) <= 3
        if duration_value < min(0.16, beat_seconds / 3) and between and stepwise:
            ornaments_value = list(note.get("ornaments", []))
            ornaments_value.append("passing-tone")
            note["ornaments"] = sorted(set(ornaments_value))
            has_clear_reattack = note.get("boundaryType") == "reattack"
            if not has_clear_reattack and duration_value < 0.075:
                note["classification"] = "ornament-transition"
                note["reviewStatus"] = "uncertain"

    # Isolated sub-65 ms pitch islands between touching notes are tracker
    # flicker/ornamental motion unless an independent acoustic reattack exists.
    for index, note in enumerate(detailed_notes):
        if index == 0 or index + 1 >= len(detailed_notes):
            continue
        duration_value = float(note["end"]) - float(note["start"])
        previous_note, following_note = detailed_notes[index - 1], detailed_notes[index + 1]
        touching = (
            float(note["start"]) - float(previous_note["end"]) <= 0.030
            and float(following_note["start"]) - float(note["end"]) <= 0.030
        )
        near_neighbours = min(
            abs(int(note["midi"]) - int(previous_note["midi"])),
            abs(int(note["midi"]) - int(following_note["midi"])),
        ) <= 3
        if duration_value < 0.065 and touching and near_neighbours and note.get("boundaryType") != "reattack":
            note["classification"] = "ornament-transition"
            note["reviewStatus"] = "uncertain"

    # Convert same-syllable scoops/falls into one playable target note.  This
    # operates after ornament classification so rapid genuine notes remain.
    detailed_notes, folded_approach_count = _fold_same_syllable_pitch_approaches(detailed_notes, beat_seconds)

    # "Suppressed" ornament transitions must not remain in the playable piano
    # list. Keep any fragment with a real reattack or strong multi-model pitch
    # evidence; remove only weak, short, continuous-vowel/noise transitions.
    before_ornament_suppression = len(detailed_notes)
    playable_details: list[dict] = []
    for note in detailed_notes:
        duration_value = float(note["end"]) - float(note["start"])
        weak_transition = (
            note.get("classification") == "ornament-transition"
            and note.get("boundaryType") != "reattack"
            and duration_value <= max(0.085, beat_seconds / 5.0)
            and float(note.get("onset", 0.0)) < 0.36
            and (
                float(note.get("sourceAgreement", 0.0)) < 0.46
                or float(note.get("estimatorSupport", 0.0)) < 1.8
            )
        )
        breath_or_noise = (
            note.get("classification") == "transition"
            and duration_value < max(0.075, beat_seconds / 6.0)
            and float(note.get("sourceAgreement", 0.0)) < 0.25
            and float(note.get("pitchSpread", 0.0)) > 2.5
        )
        if not weak_transition and not breath_or_noise:
            playable_details.append(note)
    detailed_notes = playable_details
    suppressed_unpitched_ornament_count = before_ornament_suppression - len(detailed_notes)
    # Run exact-duplicate removal only after every ornament/noise filter. Before
    # this point a soon-to-be-suppressed fragment may sit between two identical
    # folded notes and prevent them from being compared.
    detailed_notes, post_ornament_duplicate_count = _consolidate_post_ornament_duplicates(detailed_notes)

    # Group-level tuplet interpretation resolves positions such as 2/6 that
    # are individually indistinguishable from 1/3. A nearly even non-binary
    # run inside one beat is labelled as a whole (5, 6, 7, 9 ... up to 16).
    beat_groups: dict[int, list[dict[str, float | int | str]]] = {}
    for note in detailed_notes:
        beat_index = int(np.floor((float(note["rawStart"]) - beat_phase) / beat_seconds + 1e-6))
        beat_groups.setdefault(beat_index, []).append(note)
    for group in beat_groups.values():
        count = len(group)
        if count < 3 or count > 16 or count in (4, 8, 16):
            continue
        starts_in_beat = np.array(sorted(float(note["rawStart"]) for note in group), dtype=float)
        if len(starts_in_beat) >= 3:
            spacing = np.diff(starts_in_beat)
            spacing_variation = float(np.std(spacing) / (np.mean(spacing) + 1e-9))
            span = float(starts_in_beat[-1] - starts_in_beat[0])
            if spacing_variation > 0.14 or (count >= 5 and span < beat_seconds * 0.58):
                continue
        ratio = _tuplet_ratio(count)
        if ratio:
            for note in group:
                note["tupletActual"], note["tupletNormal"] = ratio

    # Recheck suspicious octave assignments against both estimators and melodic
    # continuity. Only change when an octave alternative is materially better.
    for index, note in enumerate(detailed_notes):
        neighbours = [int(detailed_notes[j]["midi"]) for j in (index - 1, index + 1) if 0 <= j < len(detailed_notes)]
        if not neighbours:
            continue
        current_pitch = int(note["midi"])
        alternatives = [candidate for candidate in (current_pitch - 12, current_pitch, current_pitch + 12) if 21 <= candidate <= 108]
        continuity = {candidate: float(np.mean([abs(candidate - neighbour) for neighbour in neighbours])) for candidate in alternatives}
        best = min(alternatives, key=continuity.get)
        if best != current_pitch and continuity[current_pitch] - continuity[best] >= 7 and float(note["sourceAgreement"]) < 0.58:
            note["pitchAlternatives"] = alternatives
            # Continuity alone is not evidence that an octave is wrong. Keep the
            # estimator-backed pitch and mark it for the editor instead of
            # silently moving a correct note by 12 semitones.
            note["reviewStatus"] = "octave-review"

    # Score-consistency audit. It never rewrites performance timing; it marks
    # notation contradictions and sends only doubtful locations to the editor.
    crowded_beats: set[int] = set()
    beat_note_counts: dict[int, int] = {}
    for note in detailed_notes:
        if beat_times:
            beat_index = int(np.searchsorted(np.asarray(beat_times), float(note["rawStart"]), side="right") - 1)
        else:
            beat_index = int(np.floor((float(note["rawStart"]) - beat_phase) / beat_seconds))
        beat_note_counts[beat_index] = beat_note_counts.get(beat_index, 0) + 1
    crowded_beats = {beat_index for beat_index, count_value in beat_note_counts.items() if count_value > 8}
    score_issue_count = 0
    for index, note in enumerate(detailed_notes):
        issues: list[str] = []
        duration_value = float(note["rawEnd"]) - float(note["rawStart"])
        if duration_value < playable_note_seconds:
            issues.append("too-short")
        if index > 0 and float(note["rawStart"]) < float(detailed_notes[index - 1]["rawEnd"]) - 0.012:
            issues.append("overlap")
        if index > 0 and int(note["midi"]) == int(detailed_notes[index - 1]["midi"]) and float(note.get("onset", 0.0)) < 0.24:
            issues.append("same-note-without-attack")
        beat_index = int(np.searchsorted(np.asarray(beat_times), float(note["rawStart"]), side="right") - 1) if beat_times else int(np.floor((float(note["rawStart"]) - beat_phase) / beat_seconds))
        if beat_index in crowded_beats:
            issues.append("crowded-beat")
        measure_length = beat_seconds * 4
        if int(np.floor(float(note["scoreStart"]) / measure_length)) != int(np.floor(max(float(note["scoreStart"]), float(note["scoreEnd"]) - 1e-6) / measure_length)):
            issues.append("tie-required")
        note["scoreIssues"] = issues
        score_issue_count += len(issues)
        if any(issue in issues for issue in ("too-short", "overlap", "same-note-without-attack", "crowded-beat")) and note["reviewStatus"] == "ok":
            note["reviewStatus"] = "uncertain"

    # Resolve every doubtful state automatically. Keep the former state and
    # resolution method as metadata, but never require a yellow/cyan manual
    # decision. Octaves use model+harmonic evidence; the remaining cases keep
    # the consensus pitch and already-corrected acoustic timing.
    auto_resolved_count = 0
    for note in detailed_notes:
        previous_status = str(note.get("reviewStatus", "ok"))
        if previous_status == "ok":
            continue
        resolution = "consensus-pitch-and-acoustic-timing"
        if previous_status == "octave-review":
            evidence = note.get("octaveEvidence", {})
            valid_candidates: list[tuple[int, float]] = []
            if isinstance(evidence, dict):
                for candidate_text, values in evidence.items():
                    if not isinstance(values, dict):
                        continue
                    try:
                        valid_candidates.append((int(candidate_text), float(values.get("combined", -1e9))))
                    except (TypeError, ValueError):
                        pass
            if valid_candidates:
                current_pitch = int(note["midi"])
                chosen_pitch, chosen_score = max(valid_candidates, key=lambda item: item[1])
                current_values = evidence.get(str(current_pitch), {}) if isinstance(evidence, dict) else {}
                chosen_values = evidence.get(str(chosen_pitch), {}) if isinstance(evidence, dict) else {}
                current_score = float(current_values.get("combined", -1e9)) if isinstance(current_values, dict) else -1e9
                current_model = float(current_values.get("model", 0.0)) if isinstance(current_values, dict) else 0.0
                chosen_model = float(chosen_values.get("model", 0.0)) if isinstance(chosen_values, dict) else 0.0
                # Harmonic energy alone often favours the first overtone.  An
                # octave move is allowed only when independent pitch trackers
                # also provide a materially stronger vote for that octave.
                if (
                    chosen_pitch != current_pitch
                    and chosen_score - current_score >= 0.85
                    and chosen_model - current_model >= 0.55
                ):
                    note["midi"] = int(np.clip(chosen_pitch, 21, 108))
                    note["pitchCents"] = round((float(note.get("pitchMidi", chosen_pitch)) - chosen_pitch) * 100, 2)
                    resolution = "strong-model-and-harmonic-octave-choice"
                else:
                    resolution = "kept-consensus-octave-insufficient-independent-support"
        elif note.get("classification") == "ornament-transition":
            resolution = "suppressed-ornament-transition"
        elif note.get("scoreIssues"):
            resolution = "tempo-map-and-acoustic-timing-choice"
        note["autoResolvedFrom"] = previous_status
        note["autoResolution"] = resolution
        note["reviewStatus"] = "auto-resolved"
        auto_resolved_count += 1

    pitch_curve: list[dict[str, float | bool]] = []
    for index in range(0, frame_count, 2):
        pair_end = min(frame_count, index + 2)
        is_voiced = bool(valid[index])
        point: dict[str, float | bool] = {
            "time": round(time_offset + index * frame_seconds, 4),
            "confidence": round(float(confidence[index]), 4),
            "voiced": is_voiced,
        }
        if is_voiced:
            point["hz"] = round(float(librosa.midi_to_hz(smoothed_midi[index])), 3)
            point["midi"] = round(float(smoothed_midi[index]), 4)
            point["rawMidi"] = round(float(raw_consensus_midi[index]), 4)
            point["cents"] = round(float((smoothed_midi[index] - round(smoothed_midi[index])) * 100), 2)
            point["scoreMidi"] = int(rounded[index])
        point["boundary"] = bool(np.any(protected_boundaries[index:pair_end]))
        point["boundaryStrength"] = round(float(np.max(boundary_strength[index:pair_end])), 4)
        point["textBoundary"] = bool(np.any(text_boundaries[index:pair_end]))
        point["moraCandidate"] = bool(np.any(mora_candidate_boundaries[index:pair_end]))
        point["characterCandidate"] = bool(np.any(character_candidate_boundaries[index:pair_end]))
        point["silentTextBoundary"] = bool(np.any(silent_text_boundaries[index:pair_end]))
        point["samePitchReattack"] = bool(np.any(same_pitch_reattacks[index:pair_end]))
        point["samePitchReattackScore"] = round(float(np.max(same_pitch_reattack_scores[index:pair_end])), 4)
        phonetic_values = same_pitch_phonetic_starts[index:pair_end]
        finite_phonetic = phonetic_values[np.isfinite(phonetic_values)]
        if finite_phonetic.size:
            point["phoneticStart"] = round(float(np.min(finite_phonetic)), 4)
        if index in text_boundary_labels:
            point["text"] = text_boundary_labels[index]
        pitch_curve.append(point)
    detailed_notes, recovered_internal_boundary_count = _recover_internal_note_boundaries(
        detailed_notes,
        pitch_curve,
        beat_seconds,
    )
    diagnostics: dict[str, float | int | str] = {
        "separationDelayMs": round(separation_delay_seconds * 1000.0, 2),
        "alignmentConfidence": round(alignment_confidence, 4),
        "masterResolutionMs": round(frame_seconds * 1000.0, 3),
        "analysisWindowsMs": "5ms master / 10ms neural / 23ms fast / 46ms pYIN / 93ms stable",
        "wordGroupCount": len(word_segments),
        "syllableCount": len(syllable_segments),
        "vowelCount": len(vowel_segments),
        "scoreIssueCount": score_issue_count,
        "uncertainNoteCount": 0,
        "autoResolvedNoteCount": auto_resolved_count,
        "syllableScoopMergeCount": scoop_merge_count,
        "attackTimingAdjustmentCount": timing_adjustment_count,
        "transcriptionBoundaryCount": transcription_boundary_count,
        "moraCandidateCount": mora_candidate_count,
        "characterBoundaryCandidateCount": character_boundary_candidate_count,
        "confirmedMoraBoundaryCount": int(np.count_nonzero(text_boundaries & mora_candidate_boundaries)),
        "silentMoraBoundaryCount": silent_mora_boundary_count,
        "foldedPitchApproachCount": folded_approach_count,
        "postOrnamentDuplicateCount": post_ornament_duplicate_count,
        "shortTerminalPitchCorrectionCount": short_terminal_pitch_correction_count,
        "recoveredInternalBoundaryCount": recovered_internal_boundary_count,
        "conservativePhoneticReattackCount": sum(bool(note.get("conservativePhoneticReattack")) for note in detailed_notes),
        "stableInternalPitchRecoveryCount": sum(bool(note.get("stableInternalPitchRecovered")) for note in detailed_notes),
        "samePitchReattackCount": int(np.count_nonzero(same_pitch_reattacks)),
        "suppressedExtremeTransitionCount": suppressed_extreme_transition_count,
        "suppressedUnpitchedOrnamentCount": suppressed_unpitched_ornament_count,
        "boundaryFirstSegmentation": int(BOUNDARY_FIRST_SEGMENTATION),
    }
    return detailed_notes, pitch_curve, round(tuning_semitones * 100, 2), word_segments, syllable_segments, vowel_segments, diagnostics


def _validate_transcription_with_melody(transcription: dict, notes: list[dict]) -> dict:
    """Validate decoded words with independent sung-pitch evidence."""

    if not transcription.get("available"):
        return transcription
    if transcription.get("forcedAlignment") and transcription.get("alignmentWords"):
        validated = dict(transcription)
        validated["acceptedWordCount"] = len(transcription["alignmentWords"])
        validated["rejectedWordCount"] = 0
        validated["validationMethod"] = "supplied-lyrics forced alignment kept separate from audio-only transcription accuracy"
        return validated
    melody_intervals = sorted(
        (float(note["start"]), float(note["end"]))
        for note in notes
        if note.get("role", "melody") == "melody" and float(note.get("end", 0.0)) > float(note.get("start", 0.0))
    )

    def acoustic_support(start: float, end: float, pad: float = 0.18) -> tuple[float, float]:
        duration = max(0.04, end - start)
        overlap = 0.0
        nearest = float("inf")
        for note_start, note_end in melody_intervals:
            overlap += max(0.0, min(end, note_end) - max(start, note_start))
            if note_end < start:
                nearest = min(nearest, start - note_end)
            elif note_start > end:
                nearest = min(nearest, note_start - end)
            else:
                nearest = 0.0
        coverage = min(1.0, overlap / duration)
        proximity = max(0.0, 1.0 - nearest / pad) if nearest != float("inf") else 0.0
        return coverage, proximity

    accepted_words: list[dict] = []
    rejected_words: list[dict] = []
    original_words = list(transcription.get("words", []))
    for index, original in enumerate(original_words):
        row = dict(original)
        start, end = float(row.get("start", 0.0)), float(row.get("end", 0.0))
        coverage, proximity = acoustic_support(start, end)
        decoder_score = float(row.get("contextConfidence", row.get("confidence", 0.0)))
        neighbours: list[float] = []
        for neighbour_index in (index - 1, index + 1):
            if not 0 <= neighbour_index < len(original_words):
                continue
            neighbour = original_words[neighbour_index]
            if neighbour_index < index:
                gap = max(0.0, start - float(neighbour.get("end", start)))
            else:
                gap = max(0.0, float(neighbour.get("start", end)) - end)
            neighbours.append(max(0.0, 1.0 - gap / 1.2))
        neighbour_score = sum(neighbours) / len(neighbours) if neighbours else 0.5
        combined = 0.42 * decoder_score + 0.38 * max(coverage, proximity) + 0.20 * neighbour_score
        row["pitchSupport"] = round(coverage, 4)
        row["neighbourConsistency"] = round(neighbour_score, 4)
        row["combinedConfidence"] = round(combined, 4)
        # Language context can validate a word, but it can never invent singing.
        accepted = bool((coverage >= 0.025 or proximity >= 0.20) and combined >= 0.25)
        row["accepted"] = accepted
        (accepted_words if accepted else rejected_words).append(row)

    accepted_segment_ids = {int(row.get("segmentId", -1)) for row in accepted_words}
    accepted_segments: list[dict] = []
    rejected_segments: list[dict] = []
    for original in transcription.get("segments", []):
        row = dict(original)
        start, end = float(row.get("start", 0.0)), float(row.get("end", 0.0))
        coverage, proximity = acoustic_support(start, end, pad=0.15)
        accepted = int(row.get("id", -1)) in accepted_segment_ids and (coverage >= 0.01 or proximity >= 0.15)
        row["pitchSupport"] = round(coverage, 4)
        row["accepted"] = accepted
        (accepted_segments if accepted else rejected_segments).append(row)

    validated = dict(transcription)
    validated["words"] = accepted_words
    validated["segments"] = accepted_segments
    validated["rejectedWords"] = rejected_words
    validated["rejectedSegments"] = rejected_segments
    validated["text"] = "".join(str(row.get("text", "")) for row in accepted_segments)
    validated["validationMethod"] = "Whisper phrase context + word confidence + neighbouring timing + independent sung-pitch support"
    validated["acceptedWordCount"] = len(accepted_words)
    validated["rejectedWordCount"] = len(rejected_words)
    return validated


def _mora_units(value: str) -> list[str]:
    units: list[str] = []
    for character in value:
        if character in "ゃゅょぁぃぅぇぉゎー" and units:
            units[-1] += character
        else:
            units.append(character)
    return units


def _expand_character_boundary_candidates(words: list[dict]) -> list[dict]:
    """Expand lexical ASR words into visible Japanese character/mora boundaries.

    Whisper normally returns lexical words, which hides same-pitch attacks
    inside a word. Estimated character times are candidates only; extract_notes
    still requires an acoustic vowel reset and score-compatible position.
    """

    expanded: list[dict] = []
    for original in words:
        word = dict(original)
        try:
            start = float(word.get("start", -1.0))
            end = float(word.get("end", start))
        except (TypeError, ValueError):
            continue
        text = str(word.get("hiragana") or word.get("text", "")).strip()
        units = [
            unit for unit in _mora_units(text)
            if any(character.isalnum() or "ぁ" <= character <= "ゖ" or character == "ー" for character in unit)
        ]
        if start < 0 or not units:
            continue
        span = max(0.001, end - start)
        original_source = str(word.get("source", "audio-only ASR"))
        for index, unit in enumerate(units):
            row = dict(word)
            row.update({
                "start": round(start + span * index / len(units), 4),
                "end": round(start + span * (index + 1) / len(units), 4),
                "text": unit,
                "hiragana": unit,
                "parentText": str(word.get("text", "")),
                "characterIndex": index,
                "characterCount": len(units),
                "isCharacterBoundaryCandidate": True,
                "source": f"{original_source} + character/mora candidate",
            })
            expanded.append(row)
    return expanded


def _build_acoustic_lyric_boundaries(transcription: dict, syllable_segments: list[dict]) -> list[dict]:
    """Fallback forced alignment when singing ASR and supplied lyrics disagree."""
    moras = _mora_units(str(transcription.get("suppliedLyricsHiragana", "")))
    segments = sorted(
        [row for row in syllable_segments if float(row.get("end", 0.0)) > float(row.get("start", 0.0))],
        key=lambda row: float(row["start"]),
    )
    if len(moras) < 2 or len(segments) < 4:
        return []
    # Each selected timestamp remains an independently observed acoustic
    # boundary.  When counts differ, adjacent lyric moras are grouped instead
    # of inventing boundaries at evenly spaced clock times.
    unit_count = min(len(moras), len(segments))
    segment_indices = np.round(np.linspace(0, len(segments) - 1, unit_count)).astype(int)
    mora_edges = np.round(np.linspace(0, len(moras), unit_count + 1)).astype(int)
    aligned: list[dict] = []
    for index in range(unit_count):
        segment = segments[int(segment_indices[index])]
        text = "".join(moras[int(mora_edges[index]):int(mora_edges[index + 1])])
        if not text:
            continue
        aligned.append({
            "start": round(float(segment["start"]), 4),
            "end": round(float(segment["end"]), 4),
            "text": text,
            "hiragana": text,
            "confidence": round(float(max(0.35, min(0.82, segment.get("confidence", 0.5)))), 4),
            "lyricsMatchConfidence": 0.0,
            "source": "supplied-lyrics-acoustic-forced-alignment",
        })
    return aligned


def _apply_hierarchical_audio_timing(
    notes: list[dict],
    pitch_curve: list[dict],
    transcription: dict,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Determine performance timing from audio in phrase -> mora -> onset order.

    Text supplies only the expected sequence. Every final timestamp is selected
    from an acoustic boundary/voicing observation, while the original
    segmentation and score-grid timestamps remain available for comparison.
    """

    if not notes or not pitch_curve:
        return notes, [], [], {
            "hierarchicalAudioTimingApplied": 0,
            "hierarchicalTimingAdjustmentCount": 0,
        }

    curve = sorted(pitch_curve, key=lambda row: float(row.get("time", 0.0)))
    curve_times = np.asarray([float(row.get("time", 0.0)) for row in curve], dtype=float)
    acoustic_scores = np.asarray([
        float(row.get("boundaryStrength", 0.0))
        + 0.055 * min(6.0, float(row.get("samePitchReattackScore", 0.0)))
        + (0.12 if bool(row.get("boundary")) else 0.0)
        for row in curve
    ], dtype=float)
    voiced = np.asarray([bool(row.get("voiced")) for row in curve], dtype=bool)
    local_peaks = np.zeros(len(curve), dtype=bool)
    if len(curve) >= 3:
        local_peaks[1:-1] = (
            (acoustic_scores[1:-1] >= acoustic_scores[:-2])
            & (acoustic_scores[1:-1] >= acoustic_scores[2:])
            & (acoustic_scores[1:-1] >= 0.24)
        )
    peak_indices = np.flatnonzero(local_peaks)

    def acoustic_peak_near(
        target: float,
        radius: float,
        minimum_score: float = 0.24,
        lower_bound: float = -float("inf"),
    ) -> tuple[float, float] | None:
        if peak_indices.size == 0:
            return None
        candidates = peak_indices[
            (curve_times[peak_indices] >= max(lower_bound, target - radius))
            & (curve_times[peak_indices] <= target + radius)
            & (acoustic_scores[peak_indices] >= minimum_score)
        ]
        if candidates.size == 0:
            return None
        distance = np.abs(curve_times[candidates] - target) / max(radius, 1e-6)
        quality = acoustic_scores[candidates]
        chosen = int(candidates[np.argmin(distance - 0.20 * quality)])
        return float(curve_times[chosen]), float(acoustic_scores[chosen])

    original_notes = [dict(note) for note in sorted(notes, key=lambda row: float(row["start"]))]
    audio_start = float(original_notes[0]["start"])
    audio_end = float(original_notes[-1]["end"])
    transcript_segments = sorted(
        [
            dict(row) for row in transcription.get("segments", [])
            if bool(row.get("accepted", True))
            and float(row.get("end", 0.0)) > float(row.get("start", 0.0))
        ],
        key=lambda row: float(row["start"]),
    )
    if not transcript_segments:
        transcript_segments = [{
            "id": 0,
            "start": audio_start,
            "end": audio_end,
            "text": str(transcription.get("text", "")),
        }]

    # Anchor each phrase/ASR sentence boundary to an observed acoustic event.
    phrase_starts: list[float] = []
    previous_anchor = audio_start - 0.10
    for index, segment in enumerate(transcript_segments):
        estimated = float(segment["start"])
        if index == 0:
            estimated = audio_start
        peak = acoustic_peak_near(
            estimated,
            0.55 if index > 0 else 0.18,
            minimum_score=0.30,
            lower_bound=previous_anchor + 0.08,
        )
        anchor = peak[0] if peak else max(estimated, previous_anchor + 0.08)
        if index == 0 and abs(anchor - audio_start) > 0.18:
            anchor = audio_start
        anchor = float(np.clip(anchor, audio_start, audio_end))
        phrase_starts.append(anchor)
        previous_anchor = anchor
    phrase_starts = list(np.maximum.accumulate(np.asarray(phrase_starts, dtype=float)))
    phrase_ends = phrase_starts[1:] + [audio_end]
    phrase_timing: list[dict] = []
    for index, segment in enumerate(transcript_segments):
        phrase_timing.append({
            "index": index,
            "segmentId": int(segment.get("id", index)),
            "text": str(segment.get("text", "")),
            "transcriptStart": round(float(segment["start"]), 4),
            "transcriptEnd": round(float(segment["end"]), 4),
            "audioStart": round(float(phrase_starts[index]), 4),
            "audioEnd": round(float(phrase_ends[index]), 4),
        })

    # Warp each word/mora estimate inside its phrase, then snap only within a
    # small neighbourhood to an acoustic boundary. This prevents phrase drift.
    character_rows = sorted(
        [dict(row) for row in transcription.get("characterBoundaries", [])],
        key=lambda row: float(row.get("start", 0.0)),
    )
    segment_by_id = {
        int(segment.get("id", index)): (index, segment)
        for index, segment in enumerate(transcript_segments)
    }
    mora_timing: list[dict] = []
    previous_mora = audio_start - 0.03
    for row in character_rows:
        segment_id = int(row.get("segmentId", -1))
        mapping = segment_by_id.get(segment_id)
        if mapping is None:
            transcript_time = float(row.get("start", 0.0))
            phrase_index = int(np.clip(
                np.searchsorted(
                    np.asarray([float(segment["start"]) for segment in transcript_segments]),
                    transcript_time,
                    side="right",
                ) - 1,
                0,
                len(transcript_segments) - 1,
            ))
            segment = transcript_segments[phrase_index]
        else:
            phrase_index, segment = mapping
        segment_start = float(segment["start"])
        segment_end = max(segment_start + 1e-3, float(segment["end"]))
        fraction = float(np.clip(
            (float(row.get("start", segment_start)) - segment_start) / (segment_end - segment_start),
            0.0,
            1.0,
        ))
        warped = float(phrase_starts[phrase_index] + fraction * (phrase_ends[phrase_index] - phrase_starts[phrase_index]))
        peak = acoustic_peak_near(
            warped,
            0.085,
            minimum_score=0.22,
            lower_bound=previous_mora + 0.022,
        )
        refined = peak[0] if peak else warped
        refined = max(refined, previous_mora + 0.022)
        refined = min(refined, float(phrase_ends[phrase_index]) - 0.005)
        mora_timing.append({
            "text": str(row.get("hiragana") or row.get("text", "")),
            "parentText": str(row.get("parentText", "")),
            "phraseIndex": phrase_index,
            "transcriptStart": round(float(row.get("start", warped)), 4),
            "warpedStart": round(warped, 4),
            "audioStart": round(refined, 4),
            "acousticScore": round(float(peak[1] if peak else 0.0), 4),
        })
        previous_mora = refined

    adjusted = [dict(note) for note in original_notes]
    adjustment_count = 0
    previous_start = -float("inf")
    for index, note in enumerate(adjusted):
        original_start = float(note["start"])
        original_end = float(note["end"])
        phrase_index = int(np.clip(
            np.searchsorted(np.asarray(phrase_ends), original_start, side="right"),
            0,
            len(phrase_timing) - 1,
        ))
        candidates = [
            row for row in mora_timing
            if int(row["phraseIndex"]) == phrase_index
            and abs(float(row["audioStart"]) - original_start) <= 0.120
        ]
        selected_time = original_start
        timing_source = "provisional-boundary"
        selected_mora = ""
        if candidates:
            best = min(
                candidates,
                key=lambda row: (
                    abs(float(row["audioStart"]) - original_start)
                    - 0.018 * float(row.get("acousticScore", 0.0))
                ),
            )
            candidate_time = float(best["audioStart"])
            point_index = int(np.argmin(np.abs(curve_times - candidate_time)))
            acoustic_support = float(acoustic_scores[point_index])
            phonetic_note = bool(
                note.get("boundaryType") == "reattack"
                or note.get("samePitchReattack")
                or acoustic_support >= 0.32
            )
            if phonetic_note:
                selected_time = candidate_time
                timing_source = "phrase-warped-mora-acoustic"
                selected_mora = str(best.get("text", ""))
        if timing_source == "provisional-boundary":
            peak = acoustic_peak_near(original_start, 0.055, minimum_score=0.42)
            if peak and abs(peak[0] - original_start) <= 0.055:
                selected_time = peak[0]
                timing_source = "local-audio-onset"
        selected_time = float(np.clip(selected_time, original_start - 0.090, original_start + 0.090))
        selected_time = max(selected_time, previous_start + 0.035)
        selected_time = min(selected_time, original_end - 0.035)
        note["boundaryStart"] = round(original_start, 6)
        note["boundaryEnd"] = round(original_end, 6)
        note["start"] = selected_time
        note["rawStart"] = selected_time
        note["audioTimingSource"] = timing_source
        note["audioTimingMora"] = selected_mora
        note["audioTimingPhraseIndex"] = phrase_index
        note["audioTimingAdjustmentMs"] = round((selected_time - original_start) * 1000.0, 3)
        if abs(selected_time - original_start) >= 0.001:
            adjustment_count += 1
        previous_start = selected_time

    # End timing comes from the last voiced frame before the next audio onset.
    # It is intentionally independent of the score grid.
    for index, note in enumerate(adjusted):
        start = float(note["start"])
        original_end = float(note["boundaryEnd"])
        search_end = float(adjusted[index + 1]["start"]) if index + 1 < len(adjusted) else original_end + 0.10
        region = np.flatnonzero(
            (curve_times >= start)
            & (curve_times <= search_end)
            & voiced
        )
        if region.size:
            frame_step = float(np.median(np.diff(curve_times))) if len(curve_times) > 1 else 0.01
            voiced_end = min(search_end, float(curve_times[int(region[-1])]) + frame_step)
        else:
            voiced_end = min(search_end, original_end)
        if index + 1 < len(adjusted) and search_end - original_end <= 0.12:
            final_end = min(search_end, max(start + 0.040, voiced_end))
        else:
            final_end = float(np.clip(voiced_end, original_end - 0.090, original_end + 0.090))
        final_end = max(start + 0.040, final_end)
        note["end"] = final_end
        note["rawEnd"] = final_end
        note["vowelEnd"] = final_end
        note["audioEndAdjustmentMs"] = round((final_end - original_end) * 1000.0, 3)

    diagnostics = {
        "hierarchicalAudioTimingApplied": 1,
        "hierarchicalTimingAdjustmentCount": adjustment_count,
        "audioPhraseCount": len(phrase_timing),
        "audioMoraTimingCount": len(mora_timing),
        "timingMethod": "audio phrase anchors -> phrase-warped mora candidates -> local acoustic onset -> voiced release",
    }
    return adjusted, phrase_timing, mora_timing, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lyrics-file", type=Path)
    parser.add_argument("--bpm", type=float)
    parser.add_argument(
        "--pitch-model",
        choices=("v39", "learning-v431"),
        default="v39",
        help="Pitch decoder; learning-v431 requires the trained checkpoint bundle.",
    )
    parser.add_argument("--learning-device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    vocals, stem_paths, separation_engine = separate_audio(args.source, args.output_dir)
    transcription_source = stem_paths.get("dryLead") or stem_paths.get("lead", vocals)
    transcription = _transcribe_vocals(transcription_source, args.lyrics_file)
    character_boundaries = _expand_character_boundary_candidates(
        (transcription.get("alignmentWords") or transcription.get("words", []))
        if transcription.get("available")
        else []
    )
    transcription["characterBoundaries"] = character_boundaries
    bpm, beat_times, beat_phase, tempo_map = estimate_tempo(args.source, args.bpm)
    if args.pitch_model == "learning-v431":
        from apply_learning_v431_dataset import infer_audio_pitch

        notes, learned_diagnostics = infer_audio_pitch(
            Path(transcription_source), args.learning_device
        )
        diagnostics = {
            "pitchModel": learned_diagnostics,
            "legacyPitchDecoderSkipped": True,
            "note": "GPU mode uses the trained v4.3.1 decoder directly; legacy v39 pitch ensembles are not run.",
        }
        tuning_cents = 0.0
        word_segments = []
        syllable_segments = []
        vowel_segments = []
        pitch_curve = [
            {"time": round(float(note["start"]), 4), "midi": int(note["midi"]), "voiced": True}
            for note in notes
        ]
        pitch_curve.extend(
            {"time": round(float(note["end"]), 4), "midi": int(note["midi"]), "voiced": False}
            for note in notes
        )
        pitch_curve.sort(key=lambda point: point["time"])
    else:
        notes, pitch_curve, tuning_cents, word_segments, syllable_segments, vowel_segments, diagnostics = extract_notes(
            vocals,
            bpm,
            beat_phase,
            beat_times=beat_times,
            onset_vocals=stem_paths.get("lead"),
            existence_vocals=stem_paths.get("vocals"),
            transcript_words=character_boundaries,
        )
        if transcription.get("suppliedLyricsHiragana") and not transcription.get("lyricsAlignmentAccepted", False):
            acoustic_lyrics = _build_acoustic_lyric_boundaries(transcription, syllable_segments)
            if acoustic_lyrics:
                character_boundaries = _expand_character_boundary_candidates(acoustic_lyrics)
                notes, pitch_curve, tuning_cents, word_segments, syllable_segments, vowel_segments, diagnostics = extract_notes(
                    vocals,
                    bpm,
                    beat_phase,
                    beat_times=beat_times,
                    onset_vocals=stem_paths.get("lead"),
                    existence_vocals=stem_paths.get("vocals"),
                    transcript_words=character_boundaries,
                )
                transcription["alignmentWords"] = acoustic_lyrics
                transcription["characterBoundaries"] = character_boundaries
                transcription["lyricsAlignmentAccepted"] = True
                transcription["lyricsAlignmentMethod"] = "supplied hiragana moras aligned monotonically to independently detected acoustic syllable boundaries"
                transcription["asrLyricsRejected"] = True
                diagnostics["acousticForcedLyricBoundaryCount"] = len(acoustic_lyrics)
    transcription = _validate_transcription_with_melody(transcription, notes)
    diagnostics["acceptedTranscriptionWordCount"] = int(transcription.get("acceptedWordCount", 0))
    diagnostics["rejectedTranscriptionWordCount"] = int(transcription.get("rejectedWordCount", 0))
    result = {
        "analysisVersion": 431 if args.pitch_model == "learning-v431" else 39,
        "engine": (
            f"{separation_engine} + trained learning-v4.3.1 pitch/ornament/boundary models"
            if args.pitch_model == "learning-v431"
            else f"{separation_engine} + conservative short-terminal pitch correction + multi-evidence same-pitch reattack recovery + stable internal short-note recovery + final strict duplicate suppression + top-k RMVPE/CREPE/WORLD/multi-resolution pYIN fusion"
        ),
        "bpm": bpm,
        "beatTimes": beat_times,
        "beatPhase": beat_phase,
        "tempoMap": tempo_map,
        "analysisResolutionMs": round(110 / 22050 * 1000, 3),
        "tuningOffsetCents": tuning_cents,
        "pitchCurve": pitch_curve,
        "wordSegments": word_segments,
        "syllableSegments": syllable_segments,
        "vowelSegments": vowel_segments,
        "transcription": transcription,
        "transcriptText": transcription.get("text", ""),
        "segmentation": {
            "wordMethod": "lexical words expanded to character/mora candidates, then confirmed by acoustic vowel reset and tempo-map evidence",
            "syllableMethod": "consonant/spectral-flux/reattack/pitch-change/vowel boundaries",
            "smoothingMethod": "hard boundaries fixed before gap fill; edge-preserving smoothing restricted to each segment",
            "noteBoundaryMethod": "strong consonant/vowel reattacks plus tightly-gated phonetic recovery, stable internal changed-pitch recovery, conservative ornament folding and final strict duplicate suppression",
        },
        "diagnostics": diagnostics,
        "notes": notes,
        "stemFiles": {name: path.name for name, path in stem_paths.items()},
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    # Keep stdout ASCII-safe because Windows university PCs often use CP932;
    # the server decodes JSON escapes back to the original Japanese text.
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
