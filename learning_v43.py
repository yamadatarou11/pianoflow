from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch

import learning_v41 as v41
import learning_v42 as v42


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "dataset-work" / "learning-v43"
MODEL_ROOT = ROOT / "models" / "learning-v43"
STATE_PATH = WORK / "pipeline-state.json"
REPORT_PATH = WORK / "report.json"
RUNTIME_PATH = WORK / "runtime.json"
NORMALIZATION_PATH = WORK / "normalization.npz"
CURTAIN_ROOT = WORK / "curtain"

SCHEMA_VERSION = 43
MODEL_NAME = "learning-v4.3"
MODEL_MODE = "B-first high-quality theory chorus active learning"
DEFAULT_CHANNELS = 384
DEFAULT_TRANSFORMER_LAYERS = 6
PITCH_PRIOR_COUNT = v41.PITCH_CLASSES
V4_PRIOR_FEATURES = PITCH_PRIOR_COUNT + 3 + 4
FORCE_INITIAL_B = True
ALLOW_DEFERRED_ADVANCE = False

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
MAJOR_PROGRESSIONS = (
    (0, 4, 5, 3, 0, 4, 3, 4),
    (0, 5, 3, 4, 0, 5, 1, 4),
    (5, 3, 0, 4, 5, 3, 1, 4),
    (0, 3, 4, 5, 1, 4, 0, 4),
)
MINOR_PROGRESSIONS = (
    (0, 5, 2, 6, 0, 5, 3, 4),
    (0, 3, 6, 2, 5, 3, 4, 0),
    (5, 2, 6, 0, 5, 3, 4, 0),
)
MORAS = (
    "あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ",
    "さ", "し", "す", "せ", "そ", "た", "ち", "つ", "て", "と",
    "な", "に", "ぬ", "ね", "の", "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も", "や", "ゆ", "よ", "ら", "り",
    "る", "れ", "ろ", "わ",
)
VOWEL_FORMANTS = (
    (800.0, 1200.0, 2500.0),
    (350.0, 2200.0, 3000.0),
    (325.0, 700.0, 2530.0),
    (500.0, 1900.0, 2600.0),
    (450.0, 800.0, 2830.0),
)


def json_read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def prune_training_checkpoints() -> int:
    root = MODEL_ROOT / "checkpoints"
    if not root.exists():
        return 0
    removed = 0
    for entry in root.iterdir():
        if entry.is_dir():
            removed += sum(
                path.stat().st_size for path in entry.rglob("*") if path.is_file()
            )
            shutil.rmtree(entry)
        else:
            removed += entry.stat().st_size
            entry.unlink()
    return removed


def nearest_pitch(pitch_class: int, center: int, minimum: int = 64, maximum: int = 88) -> int:
    candidates = [
        midi
        for midi in range(minimum, maximum + 1)
        if midi % 12 == pitch_class % 12
    ]
    return min(candidates, key=lambda value: abs(value - center))


def rhythmic_pattern(category: str, bar: int, rng: random.Random) -> List[float]:
    normal = (
        (1.0, 1.0, 1.0, 1.0),
        (0.5, 0.5, 1.0, 0.5, 0.5, 1.0),
        (0.5, 1.0, 0.5, 1.0, 1.0),
        (1.5, 0.5, 1.0, 1.0),
        (0.75, 0.25, 1.0, 1.0, 1.0),
    )
    if category == "short_note" and bar in (2, 5, 6):
        return list(rng.choice((
            (0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0),
            (0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 1.0),
        )))
    if category == "tuplet" and bar in (1, 5):
        if rng.random() < 0.5:
            third = 1.0 / 3.0
            return [third, third, third, 0.5, 0.5, 1.0, 1.0]
        fifth = 0.2
        return [fifth] * 5 + [0.5, 0.5, 1.0, 1.0]
    if category == "same_pitch" and bar in (1, 2, 5, 6):
        return [0.5] * 6 + [1.0]
    if category == "ornament" and bar in (2, 6):
        return [0.5, 0.5, 1.5, 0.5, 1.0]
    return list(rng.choice(normal))


class TheoryChorusComposer:
    """Hierarchical, weakness-conditioned eight-bar chorus composer."""

    def __init__(self, training: Sequence[dict], seed: int):
        self.rng = random.Random(seed)
        bpms = [float(item.get("bpm") or 0) for item in training]
        self.bpms = [value for value in bpms if 65 <= value <= 220]
        self.pitch_centers = []
        for item in training:
            pitches = [int(note["midi"]) for note in item.get("notes", [])]
            if pitches:
                self.pitch_centers.append(int(round(float(np.median(pitches)))))

    def compose(
        self, category: str, index: int
    ) -> Tuple[List[dict], float, List[dict], List[str], dict]:
        rng = random.Random(self.rng.randrange(2**31) + index * 7919)
        mode = rng.choice(("major", "major", "minor"))
        scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE
        tonic_pc = rng.randrange(12)
        progression = rng.choice(
            MAJOR_PROGRESSIONS if mode == "major" else MINOR_PROGRESSIONS
        )
        bpm = float(
            rng.choice(self.bpms)
            if self.bpms and rng.random() < 0.65
            else rng.choice((92, 104, 118, 128, 138, 146, 160, 176))
        )
        center = rng.choice(self.pitch_centers) if self.pitch_centers else 74
        center = max(69, min(81, center))
        beat_seconds = 60.0 / bpm
        cursor_beats = 0.35
        notes: List[dict] = []
        ornaments: List[dict] = []
        syllables: List[str] = []
        motif_degrees: List[int] = []
        previous_pitch = nearest_pitch((tonic_pc + scale[0]) % 12, center)
        note_index = 0

        for bar, chord_degree in enumerate(progression):
            pattern = rhythmic_pattern(category, bar, rng)
            chord_scale_indices = (
                chord_degree % 7,
                (chord_degree + 2) % 7,
                (chord_degree + 4) % 7,
            )
            chord_pcs = {
                (tonic_pc + scale[degree]) % 12 for degree in chord_scale_indices
            }
            position = 0.0
            bar_degrees: List[int] = []
            for local_index, duration_beats in enumerate(pattern):
                strong = abs(position - round(position)) < 1e-5
                if bar < 2:
                    if strong:
                        degree = rng.choice(chord_scale_indices)
                    else:
                        degree = max(0, min(6, chord_degree + rng.choice((-1, 1, 2))))
                    motif_degrees.append(degree)
                else:
                    source = motif_degrees[(note_index + bar) % max(1, len(motif_degrees))]
                    variation = 0
                    if bar in (3, 6) and rng.random() < 0.35:
                        variation = rng.choice((-1, 1))
                    degree = max(0, min(6, source + variation))

                pitch_class = (tonic_pc + scale[degree]) % 12
                target_center = center + (2 if bar >= 4 else 0) + (3 if bar == 6 else 0)
                pitch = nearest_pitch(pitch_class, target_center)
                candidates = (pitch - 12, pitch, pitch + 12)
                pitch = min(
                    (value for value in candidates if 64 <= value <= 88),
                    key=lambda value: abs(value - previous_pitch),
                )

                if category == "same_pitch" and bar in (1, 5) and local_index in (1, 2, 3):
                    pitch = previous_pitch
                if category == "octave" and bar in (2, 6) and local_index == 1:
                    octave = previous_pitch + (12 if previous_pitch <= 74 else -12)
                    if 64 <= octave <= 88:
                        pitch = octave
                elif category == "octave" and len(notes) >= 2:
                    previous_jump = previous_pitch - int(notes[-2]["midi"])
                    if abs(previous_jump) >= 11:
                        resolution_direction = -1 if previous_jump > 0 else 1
                        resolution = [
                            value
                            for value in range(64, 89)
                            if value % 12 in {
                                (tonic_pc + scale_degree) % 12
                                for scale_degree in scale
                            }
                            and 1 <= (value - previous_pitch) * resolution_direction <= 3
                        ]
                        if resolution:
                            pitch = min(
                                resolution,
                                key=lambda value: abs(value - previous_pitch),
                            )
                if abs(pitch - previous_pitch) > 9 and not (
                    category == "octave" and bar in (2, 6)
                ):
                    pitch = nearest_pitch(pitch_class, previous_pitch)

                start = cursor_beats * beat_seconds
                duration_seconds = duration_beats * beat_seconds
                articulation = 0.91 if duration_beats <= 0.5 else 0.95
                end = start + max(0.035, duration_seconds * articulation)
                notes.append(
                    {
                        "midi": int(pitch),
                        "start": round(start, 6),
                        "end": round(end, 6),
                        "bar": bar,
                        "beat": round(position, 6),
                        "durationBeats": round(duration_beats, 8),
                        "chordPitchClasses": sorted(chord_pcs),
                    }
                )
                ornament_type = "stable"
                if category == "ornament" and bar in (2, 6) and local_index in (0, 2):
                    ornament_type = rng.choice(("scoop", "fall", "transition"))
                elif rng.random() < 0.035 and duration_beats >= 0.75:
                    ornament_type = rng.choice(("scoop", "fall"))
                ornaments.append(
                    {
                        "noteIndex": note_index,
                        "type": ornament_type,
                        "shift": rng.choice((-2.0, -1.5, 1.5, 2.0)),
                        "vibratoDepth": rng.uniform(0.035, 0.13),
                    }
                )
                syllables.append(rng.choice(MORAS))
                bar_degrees.append(degree)
                previous_pitch = pitch
                note_index += 1
                position += duration_beats
                cursor_beats += duration_beats

            if abs(position - 4.0) > 1e-5:
                raise AssertionError("bar duration is not four beats")

        # Resolve the last note to the tonic or dominant and lengthen it.
        last = notes[-1]
        cadence_pc = (tonic_pc + scale[rng.choice((0, 4))]) % 12
        last["midi"] = nearest_pitch(cadence_pc, center + 2)
        last["end"] = round(
            float(last["start"]) + max(0.4, 1.35 * beat_seconds), 6
        )
        theory = {
            "bars": 8,
            "beatsPerBar": 4,
            "mode": mode,
            "tonicPitchClass": tonic_pc,
            "scalePitchClasses": sorted((tonic_pc + value) % 12 for value in scale),
            "progressionDegrees": list(progression),
            "category": category,
            "form": "A-A'-B-A'' chorus",
            "peakBar": 7,
        }
        return notes, bpm, ornaments, syllables, theory


def theory_quality(notes: Sequence[dict], theory: dict, category: str) -> dict:
    reasons: List[str] = []
    scale = set(theory["scalePitchClasses"])
    pitches = [int(note["midi"]) for note in notes]
    intervals = [second - first for first, second in zip(pitches, pitches[1:])]
    in_scale = float(np.mean([pitch % 12 in scale for pitch in pitches])) if pitches else 0.0
    strong_notes = [
        note for note in notes
        if abs(float(note.get("beat", 0)) - round(float(note.get("beat", 0)))) < 1e-4
    ]
    chord_fit = float(
        np.mean(
            [
                int(note["midi"]) % 12 in set(note.get("chordPitchClasses", []))
                for note in strong_notes
            ]
        )
    ) if strong_notes else 0.0
    large_leaps = [index for index, value in enumerate(intervals) if abs(value) > 9]
    unresolved = 0
    for index in large_leaps:
        if index + 1 >= len(intervals) or abs(intervals[index + 1]) > 3:
            unresolved += 1
    durations = [float(note.get("durationBeats", 0)) for note in notes]
    same_pitch = sum(first == second for first, second in zip(pitches, pitches[1:]))
    short_count = sum(value <= 0.25 + 1e-5 for value in durations)
    tuplet_count = sum(
        abs(value - 1 / 3) < 0.02 or abs(value - 0.2) < 0.02
        for value in durations
    )
    ornament_count = sum(
        row.get("type") != "stable" for row in theory.get("ornaments", [])
    )
    octave_count = sum(abs(value) >= 11 for value in intervals)
    coverage = {
        "short_note": short_count >= 4,
        "same_pitch": same_pitch >= 3,
        "ornament": ornament_count >= 2,
        "tuplet": tuplet_count >= 3,
        "octave": octave_count >= 1,
    }
    if not 32 <= len(notes) <= 96:
        reasons.append("note_count")
    if max(pitches, default=0) - min(pitches, default=0) > 24:
        reasons.append("range")
    if in_scale < 0.96:
        reasons.append("scale")
    if chord_fit < 0.45:
        reasons.append("strong_beat_chord_fit")
    if unresolved > 1:
        reasons.append("unresolved_leap")
    if not coverage.get(category, True):
        reasons.append("weakness_not_covered")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "inScaleRatio": round(in_scale, 4),
        "strongBeatChordToneRatio": round(chord_fit, 4),
        "unresolvedLargeLeaps": unresolved,
        "pitchRange": max(pitches, default=0) - min(pitches, default=0),
        "coverage": coverage,
    }


def synthesize_chorus_vocal(
    path: Path,
    notes: Sequence[dict],
    ornaments: Sequence[dict],
    syllables: Sequence[str],
    seed: int,
) -> None:
    rng = np.random.RandomState(seed)
    duration = max(float(note["end"]) for note in notes) + 0.25
    count = int(math.ceil(duration * v41.SAMPLE_RATE))
    output = np.zeros(count, np.float64)
    ornament_by_index = {int(row["noteIndex"]): row for row in ornaments}
    phase = rng.uniform(0, 2 * np.pi)
    for index, note in enumerate(notes):
        start = int(round(float(note["start"]) * v41.SAMPLE_RATE))
        end = min(count, int(round(float(note["end"]) * v41.SAMPLE_RATE)))
        if end - start < 16:
            continue
        local_count = end - start
        local_time = np.arange(local_count, dtype=np.float64) / v41.SAMPLE_RATE
        target_midi = float(note["midi"]) - 12.0
        curve = np.full(local_count, target_midi, np.float64)
        ornament = ornament_by_index.get(index, {"type": "stable"})
        edge = min(local_count, int(0.115 * v41.SAMPLE_RATE))
        if ornament["type"] == "scoop" and edge:
            curve[:edge] -= np.linspace(2.2, 0.0, edge)
        elif ornament["type"] == "fall" and edge:
            curve[-edge:] -= np.linspace(0.0, 2.4, edge)
        elif ornament["type"] == "transition" and edge:
            curve[-edge:] += np.linspace(
                0.0, float(ornament.get("shift", 1.5)), edge
            )
        if local_count / v41.SAMPLE_RATE > 0.34:
            vibrato_start = int(local_count * 0.42)
            curve[vibrato_start:] += float(
                ornament.get("vibratoDepth", 0.07)
            ) * np.sin(2 * np.pi * 5.3 * local_time[vibrato_start:])
        frequency = 440.0 * np.power(2.0, (curve - 69.0) / 12.0)
        phases = phase + 2 * np.pi * np.cumsum(frequency) / v41.SAMPLE_RATE
        phase = float(phases[-1] % (2 * np.pi))
        vowel_index = index % len(VOWEL_FORMANTS)
        formants = VOWEL_FORMANTS[vowel_index]
        source = np.zeros(local_count, np.float64)
        median_f0 = float(np.median(frequency))
        for harmonic in range(1, 25):
            harmonic_frequency = median_f0 * harmonic
            formant_gain = 0.08 + sum(
                math.exp(-0.5 * ((harmonic_frequency - formant) / 190.0) ** 2)
                for formant in formants
            )
            source += (
                formant_gain
                * np.sin(phases * harmonic + harmonic * 0.02)
                / (harmonic ** 1.18)
            )
        attack = min(local_count, int((0.018 + 0.012 * (index % 3)) * v41.SAMPLE_RATE))
        release = min(local_count, int(0.045 * v41.SAMPLE_RATE))
        envelope = np.ones(local_count, np.float64)
        envelope[:attack] *= np.linspace(0.0, 1.0, attack)
        envelope[-release:] *= np.linspace(1.0, 0.0, release)
        consonant_count = min(local_count, int(0.032 * v41.SAMPLE_RATE))
        consonant = np.zeros(local_count, np.float64)
        if consonant_count:
            noise = rng.normal(0, 1, consonant_count)
            noise = np.r_[noise[0], np.diff(noise)]
            consonant[:consonant_count] = noise * np.linspace(1.0, 0.0, consonant_count)
        breath = rng.normal(0, 0.012, local_count)
        syllable_gain = 0.92 + 0.08 * (len(syllables[index]) if index < len(syllables) else 1)
        output[start:end] += (
            0.155 * source * envelope * syllable_gain
            + 0.045 * consonant
            + breath * envelope
        )
    peak = max(1e-8, float(np.max(np.abs(output))))
    output = np.clip(output / peak * 0.84, -0.98, 0.98)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(v41.SAMPLE_RATE)
        handle.writeframes((output * 32767).astype("<i2").tobytes())


def audio_midi_quality(
    audio_path: Path,
    notes: Sequence[dict],
    category: str,
) -> dict:
    audio, _ = librosa.load(audio_path, sr=v41.SAMPLE_RATE, mono=True)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    clipping = float(np.mean(np.abs(audio) >= 0.985)) if len(audio) else 1.0
    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=v41.SAMPLE_RATE,
        frame_length=1024,
        hop_length=v41.HOP_LENGTH,
    )
    f0_times = librosa.frames_to_time(
        np.arange(len(f0)), sr=v41.SAMPLE_RATE, hop_length=v41.HOP_LENGTH
    )
    f0_midi = librosa.hz_to_midi(np.maximum(f0, 1e-6))
    correct = 0
    missing = 0
    pitch_errors: List[float] = []
    for note in notes:
        start = float(note["start"])
        end = float(note["end"])
        margin = min(0.055, max(0.0, (end - start) * 0.22))
        frames = np.flatnonzero(
            (f0_times >= start + margin) & (f0_times < end - margin)
        )
        if not len(frames):
            missing += 1
            continue
        predicted = float(np.median(f0_midi[frames]))
        error = abs(predicted - (int(note["midi"]) - 12))
        pitch_errors.append(error)
        correct += int(error <= 0.65)
    onset = librosa.onset.onset_strength(
        y=audio, sr=v41.SAMPLE_RATE, hop_length=v41.HOP_LENGTH
    )
    onset_times = librosa.frames_to_time(
        np.arange(len(onset)), sr=v41.SAMPLE_RATE, hop_length=v41.HOP_LENGTH
    )
    scale = max(1e-8, float(np.percentile(onset, 90)))
    normalized_onset = onset / scale
    supported = 0
    for note in notes:
        frames = np.flatnonzero(
            np.abs(onset_times - float(note["start"])) <= 0.065
        )
        if len(frames) and float(np.max(normalized_onset[frames])) >= 0.12:
            supported += 1
    note_accuracy = correct / max(1, len(notes))
    onset_support = supported / max(1, len(notes))
    reasons = []
    if peak < 0.15 or rms < 0.015:
        reasons.append("audio_too_quiet")
    if clipping > 0.0005:
        reasons.append("clipping")
    if note_accuracy < 0.88:
        reasons.append("midi_pitch_mismatch")
    if missing / max(1, len(notes)) > 0.04:
        reasons.append("missing_pitch_frames")
    onset_minimum = 0.64 if category == "same_pitch" else 0.52
    if onset_support < onset_minimum:
        reasons.append("weak_note_boundaries")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "notePitchAccuracy": round(note_accuracy, 4),
        "missingPitchRatio": round(missing / max(1, len(notes)), 4),
        "medianPitchErrorSemitones": round(
            float(np.median(pitch_errors)) if pitch_errors else 99.0, 4
        ),
        "onsetSupportRatio": round(onset_support, 4),
        "peak": round(peak, 5),
        "rms": round(rms, 5),
        "clippingRatio": round(clipping, 7),
    }


def melody_fingerprint(notes: Sequence[dict]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    pitches = [int(note["midi"]) for note in notes]
    intervals = tuple(
        max(-12, min(12, second - first))
        for first, second in zip(pitches, pitches[1:])
    )
    rhythm = tuple(
        int(round(float(note.get("durationBeats", 0.5)) * 60))
        for note in notes
    )
    return intervals, rhythm


def fingerprint_similarity(
    left: Tuple[Tuple[int, ...], Tuple[int, ...]],
    right: Tuple[Tuple[int, ...], Tuple[int, ...]],
) -> float:
    count = min(len(left[0]), len(right[0]), 32)
    if count <= 0:
        return 0.0
    interval = np.mean(
        np.asarray(left[0][:count]) == np.asarray(right[0][:count])
    )
    rhythm = np.mean(
        np.asarray(left[1][:count]) == np.asarray(right[1][:count])
    )
    return float(0.65 * interval + 0.35 * rhythm)


def generate_theory_synthetic(
    training: Sequence[dict],
    plan: dict,
    target: int,
    seed: int,
    budget: v41.Budget,
    output_root: Path,
    source_song: str,
    minimum_yield: float,
) -> Tuple[List[dict], dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [
        v42.enhance_synthetic_item(item)
        for item in v41.load_items(output_root)
    ]
    if len(existing) >= target:
        audit = json_read(output_root / "quality-report.json", {})
        return existing[:target], audit
    categories = []
    for category, count in plan.get("requestedSamples", {}).items():
        categories.extend([category] * max(0, int(count)))
    if not categories:
        categories = ["short_note", "same_pitch", "ornament", "tuplet", "octave"]
    composer = TheoryChorusComposer(training, seed)
    accepted_raw = []
    fingerprints = [
        melody_fingerprint(item.get("notes", []))
        for item in existing
        if item.get("notes")
    ]
    existing_indices = []
    for directory in output_root.glob("chorus-*"):
        try:
            existing_indices.append(int(directory.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    attempt_offset = max(existing_indices, default=0)
    attempts = 0
    rejected: Dict[str, int] = {}
    maximum_attempts = max(target * 5, target + 20)
    while len(existing) + len(accepted_raw) < target and attempts < maximum_attempts:
        if not budget.can_start(45):
            break
        attempts += 1
        generation_index = attempt_offset + attempts
        category = categories[(attempts - 1) % len(categories)]
        notes, bpm, ornaments, syllables, theory = composer.compose(
            category, generation_index
        )
        theory["ornaments"] = ornaments
        theory_audit = theory_quality(notes, theory, category)
        fingerprint = melody_fingerprint(notes)
        duplicate = max(
            (fingerprint_similarity(fingerprint, prior) for prior in fingerprints),
            default=0.0,
        )
        if duplicate >= 0.92:
            theory_audit["accepted"] = False
            theory_audit["reasons"].append("near_duplicate")
        sample_id = "chorus-{:05d}".format(generation_index)
        directory = output_root / sample_id
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        midi_path = directory / "melody.mid"
        audio_path = directory / "vocal.wav"
        v41.write_midi(midi_path, notes, bpm)
        synthesize_chorus_vocal(
            audio_path, notes, ornaments, syllables, seed + generation_index
        )
        audio_audit = audio_midi_quality(audio_path, notes, category)
        accepted = bool(theory_audit["accepted"] and audio_audit["accepted"])
        reasons = theory_audit["reasons"] + audio_audit["reasons"]
        if accepted:
            features, times = v41.extract_features(audio_path)
            labels = v41.synthetic_labels(notes, ornaments, times)
            base_quality, base_reasons = v41.quality_check(features, labels, notes)
            accepted = bool(base_quality)
            reasons.extend(base_reasons)
        metadata = {
            "schemaVersion": SCHEMA_VERSION,
            "id": "{}-{:04x}".format(sample_id, seed & 0xFFFF),
            "name": sample_id,
            "kind": "synthetic",
            "generator": "PianoFlow v4.3 theory-conditioned chorus composer",
            "sourceSong": source_song,
            "category": category,
            "bpm": bpm,
            "audio": str(audio_path),
            "midi": str(midi_path),
            "notes": notes,
            "ornaments": ornaments,
            "syllables": syllables,
            "theory": theory,
            "qualityAccepted": accepted,
            "qualityReasons": reasons,
            "theoryQuality": theory_audit,
            "audioMidiQuality": audio_audit,
            "seed": seed + generation_index,
        }
        if accepted:
            metadata["featureCount"] = int(features.shape[1])
            metadata["frameCount"] = int(len(times))
            json_write(directory / "metadata.json", metadata)
            np.savez_compressed(
                directory / "data.npz",
                features=features,
                times=times,
                **labels,
            )
            raw = {
                **metadata,
                "features": features,
                "times": times,
                **labels,
            }
            accepted_raw.append(raw)
            fingerprints.append(fingerprint)
        else:
            for reason in reasons:
                rejected[reason] = rejected.get(reason, 0) + 1
            shutil.rmtree(directory)
        total_accepted = len(existing) + len(accepted_raw)
        if attempts % 10 == 0:
            print(
                "v4.3 quality generation: accepted={}/{} attempts={}".format(
                    total_accepted, target, attempts
                ),
                flush=True,
            )
    raw_all = [
        {key: value for key, value in item.items() if not key.startswith("v4")}
        for item in existing
    ] + accepted_raw
    enhanced = [
        item if item["features"].shape[1] > 134 else v42.enhance_synthetic_item(item)
        for item in raw_all
    ]
    yield_ratio = (len(enhanced) / max(1, attempts + len(existing)))
    audit = {
        "schemaVersion": SCHEMA_VERSION,
        "target": target,
        "attemptsThisRun": attempts,
        "accepted": len(enhanced),
        "yieldRatio": round(yield_ratio, 4),
        "minimumYield": minimum_yield,
        "qualityGatePassed": bool(len(enhanced) > 0 and yield_ratio >= minimum_yield),
        "rejectedReasons": rejected,
        "sourceSong": source_song,
    }
    json_write(output_root / "quality-report.json", audit)
    return enhanced[:target], audit


def slice_item(
    item: dict,
    note_start: int,
    note_end: int,
    suffix: str,
) -> dict:
    source_notes = item.get("notes", [])
    selected = list(enumerate(source_notes[note_start:note_end], note_start))
    if not selected:
        raise ValueError("cannot slice an item without notes")

    times = np.asarray(item.get("times", []), dtype=np.float32)
    if not len(times):
        raise ValueError("{} has no acoustic frames".format(item["name"]))
    acoustic_start = float(times[0])
    acoustic_end = float(times[-1])

    # A reference MIDI may be longer than the supplied vocal clip.  Keep only
    # notes that have real acoustic support, and clip edge notes to that support.
    supported = []
    for original_index, note in selected:
        clipped_start = max(acoustic_start, float(note["start"]))
        clipped_end = min(acoustic_end, float(note["end"]))
        if clipped_end <= clipped_start:
            continue
        clipped_note = dict(note)
        clipped_note["start"] = clipped_start
        clipped_note["end"] = clipped_end
        supported.append((original_index, clipped_note))
    if not supported:
        raise ValueError(
            "{} phrase {}:{} is outside acoustic range {:.3f}:{:.3f}".format(
                item["name"], note_start, note_end, acoustic_start, acoustic_end
            )
        )

    notes = [note for _, note in supported]
    actual_note_start = supported[0][0]
    actual_note_end = supported[-1][0] + 1
    start = max(acoustic_start, float(notes[0]["start"]) - 0.18)
    end = min(acoustic_end, float(notes[-1]["end"]) + 0.18)
    frames = np.flatnonzero((item["times"] >= start) & (item["times"] <= end))
    if not len(frames):
        raise ValueError(
            "{} phrase {}:{} has no acoustic frames".format(
                item["name"], actual_note_start, actual_note_end
            )
        )
    first, last = int(frames[0]), int(frames[-1]) + 1
    result = {
        **item,
        "id": "{}-{}".format(item["id"], suffix),
        "name": "{} [{}]".format(item["name"], suffix),
        "notes": notes,
        "features": item["features"][first:last],
        "times": item["times"][first:last],
        "pitchY": item["pitchY"][first:last],
        "pitchWeight": item["pitchWeight"][first:last],
        "ornamentY": item["ornamentY"][first:last],
        "ornamentWeight": item["ornamentWeight"][first:last],
        "boundaryY": item["boundaryY"][first:last],
        "phraseSourceId": item["id"],
        "phraseNoteRange": [actual_note_start, actual_note_end],
    }
    if "boundaryWeight" in item:
        result["boundaryWeight"] = item["boundaryWeight"][first:last]
    for key in ("v4PitchPrior", "v4OrnamentPrior", "v4BoundaryPrior"):
        if key in item:
            result[key] = item[key][first:last]
    return result


def phrase_split(item: dict, fold: int, blocks: int = 5) -> Tuple[List[dict], dict]:
    notes = item.get("notes", [])
    times = np.asarray(item.get("times", []), dtype=np.float32)
    if not len(times):
        raise ValueError("{} has no acoustic frames".format(item["name"]))
    acoustic_start = float(times[0])
    acoustic_end = float(times[-1])
    supported_indices = [
        index
        for index, note in enumerate(notes)
        if min(acoustic_end, float(note["end"]))
        > max(acoustic_start, float(note["start"]))
    ]
    supported_count = len(supported_indices)
    effective_blocks = min(int(blocks), supported_count // 2)
    if effective_blocks < 2:
        raise ValueError(
            "{} has too few acoustically supported notes for phrase holdout "
            "({}/{} notes within {:.3f}:{:.3f})".format(
                item["name"],
                supported_count,
                len(notes),
                acoustic_start,
                acoustic_end,
            )
        )
    chunks = np.array_split(np.asarray(supported_indices, dtype=np.int64), effective_blocks)
    holdout_index = int(fold % effective_blocks)
    training = []
    holdout = None
    for index, chunk in enumerate(chunks):
        start, end = int(chunk[0]), int(chunk[-1]) + 1
        phrase = slice_item(item, start, end, "phrase-{}".format(index + 1))
        phrase["sourceNoteCount"] = len(notes)
        phrase["acousticallySupportedNoteCount"] = supported_count
        phrase["discardedOutsideAcousticRange"] = len(notes) - supported_count
        if index == holdout_index:
            holdout = phrase
        else:
            training.append(phrase)
    if holdout is None:
        raise AssertionError("holdout phrase was not selected")
    return training, holdout


def acoustic_priors(
    audio_path: Path,
    features: np.ndarray,
    times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio, _ = librosa.load(audio_path, sr=v41.SAMPLE_RATE, mono=True)
    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=v41.SAMPLE_RATE,
        frame_length=1024,
        hop_length=v41.HOP_LENGTH,
    )
    f0_times = librosa.frames_to_time(
        np.arange(len(f0)), sr=v41.SAMPLE_RATE, hop_length=v41.HOP_LENGTH
    ).astype(np.float32)
    indices = v42.nearest_indices(f0_times, times)
    midi = librosa.hz_to_midi(np.maximum(f0[indices], 1e-6))
    rms = np.clip(features[:, 128], 0, 2)
    voiced = rms > max(0.06, float(np.percentile(rms, 35)))
    pitch_prior = np.full((len(times), PITCH_PRIOR_COUNT), 1e-5, np.float32)
    for frame, value in enumerate(midi):
        if not voiced[frame] or not np.isfinite(value):
            pitch_prior[frame, 0] = 0.92
            continue
        center = int(round(float(value)))
        for shift, probability in ((0, 0.72), (-1, 0.10), (1, 0.10), (-12, 0.025), (12, 0.025)):
            candidate = center + shift
            if v41.MIDI_MIN <= candidate <= v41.MIDI_MAX:
                pitch_prior[frame, candidate - v41.MIDI_MIN + 1] += probability
        pitch_prior[frame, 0] += 0.02
    pitch_prior /= np.maximum(pitch_prior.sum(1, keepdims=True), 1e-7)
    ornament_prior = np.tile(
        np.asarray((0.82, 0.09, 0.09), np.float32), (len(times), 1)
    )
    onset = np.clip(features[:, 133], 0, 2)
    rise = np.clip(features[:, 129], 0, 2)
    fall = np.clip(features[:, 130], 0, 2)
    midi_delta = np.r_[0.0, np.abs(np.diff(midi))]
    boundary_prior = np.column_stack(
        (
            np.clip(0.03 + 0.62 * onset + 0.18 * rise, 0.01, 0.99),
            np.clip(0.03 + 0.62 * fall, 0.01, 0.99),
            np.clip(0.02 + 0.45 * onset * (midi_delta < 0.65), 0.01, 0.99),
            np.clip(0.02 + 0.35 * onset + 0.25 * (midi_delta >= 0.65), 0.01, 0.99),
        )
    ).astype(np.float32)
    return pitch_prior, ornament_prior, boundary_prior


def load_curtain_item() -> dict:
    base = CURTAIN_ROOT / "audio-only"
    alignment_path = base / "initial-alignment.json"
    analysis_path = base / "analysis.json"
    if not alignment_path.exists() or not analysis_path.exists():
        raise FileNotFoundError(
            "カーテンコールの評価準備がありません。RUN_LEARNING_V43.batを実行してください。"
        )
    alignment = json_read(alignment_path)
    analysis = json_read(analysis_path)
    dry_name = analysis.get("stemFiles", {}).get("dryLead", "lead-vocal-dry.wav")
    audio_path = base / dry_name
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    features, times = v41.extract_features(audio_path)
    notes = alignment["alignedNotes"]
    labels = v41.note_labels(notes, times)
    pitch_prior, ornament_prior, boundary_prior = acoustic_priors(
        audio_path, features, times
    )
    enhanced = np.column_stack(
        (features, pitch_prior, ornament_prior, boundary_prior)
    ).astype(np.float32)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": "curtain-call-heldout",
        "name": "カーテンコール",
        "kind": "heldout-evaluation",
        "audio": str(audio_path),
        "bpm": 141.0,
        "notes": notes,
        "features": enhanced,
        "times": times,
        **labels,
        "v4PitchPrior": pitch_prior,
        "v4OrnamentPrior": ornament_prior,
        "v4BoundaryPrior": boundary_prior,
        "featureCount": int(enhanced.shape[1]),
        "priorSource": "audio-only pYIN/onset prior; reference MIDI never used for features",
        "neverUseForTrainingOrGeneration": True,
    }


def model_paths() -> Dict[str, Path]:
    return {
        task: MODEL_ROOT / "current-{}.pt".format(task)
        for task in v41.TASKS
    }


def save_models(
    models: Dict[str, torch.nn.Module],
    mean: np.ndarray,
    scale: np.ndarray,
    args,
    completed_step: int,
    branch: str,
) -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for task, model in models.items():
        v41.save_checkpoint(
            model_paths()[task],
            {
                "schemaVersion": SCHEMA_VERSION,
                "complete": True,
                "task": task,
                "architecture": type(model).__name__,
                "featureCount": int(len(mean)),
                "channels": args.channels,
                "transformerLayers": args.transformer_layers,
                "outputCount": v41.task_output_count(task),
                "parameterCount": v41.parameter_count(model),
                "mean": mean,
                "scale": scale,
                "bestState": model.state_dict(),
                "completedStep": completed_step,
                "chosenBranch": branch,
            },
        )


def load_models(feature_count: int, args, device: torch.device):
    paths = model_paths()
    if not all(path.exists() for path in paths.values()):
        return None
    models = {}
    for task, path in paths.items():
        payload = torch.load(path, map_location=device, weights_only=False)
        if int(payload["featureCount"]) != feature_count:
            raise ValueError("v4.3 model feature count mismatch")
        model = v41.create_model(
            task,
            feature_count,
            v41.task_output_count(task),
            args.channels,
            args.transformer_layers,
        ).to(device)
        model.load_state_dict(payload["bestState"])
        models[task] = model
    return models


def report_for_models(models, items, mean, scale, device):
    if not items:
        return v42.aggregate_report([])
    if models is None:
        return v42.prior_report(items)
    return v41.evaluate_models(models, items, mean, scale, device)


def combined_health(
    baseline_current: dict,
    candidate_current: dict,
    baseline_past: dict,
    candidate_past: dict,
    args,
) -> dict:
    pitch_delta = (
        candidate_current["meanNoteAccuracy"]
        - baseline_current["meanNoteAccuracy"]
    )
    ornament_delta = (
        candidate_current["meanOrnamentMacroF1"]
        - baseline_current["meanOrnamentMacroF1"]
    )
    boundary_delta = (
        candidate_current["meanBoundaryF1"]
        - baseline_current["meanBoundaryF1"]
    )
    regression = v42.regression_check(
        baseline_past,
        candidate_past,
        args.mean_regression,
        args.song_regression,
    )
    passed = (
        pitch_delta >= -args.current_pitch_tolerance
        and ornament_delta >= -args.current_ornament_tolerance
        and boundary_delta >= -args.current_boundary_tolerance
        and regression["passed"]
    )
    return {
        "passed": bool(passed),
        "currentPitchDelta": round(pitch_delta, 4),
        "currentOrnamentDelta": round(ornament_delta, 4),
        "currentBoundaryDelta": round(boundary_delta, 4),
        "pastRegression": regression,
    }


def generated_target(step: int, report: dict, maximum: int) -> int:
    pitch_error = 1.0 - float(report["meanNoteAccuracy"])
    boundary_error = 1.0 - float(report["meanBoundaryF1"])
    severity = float(np.clip(0.65 * pitch_error + 0.35 * boundary_error, 0, 1))
    minimum, upper = ((100, 180) if step == 0 else (45, 120))
    return min(maximum, int(round(minimum + (upper - minimum) * severity)))


def state_default(real_items: Sequence[dict]) -> dict:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "complete": False,
        "nextSongIndex": 0,
        "songOrder": [item["name"] for item in real_items],
        "acceptedRealIds": [],
        "acceptedSyntheticDirectories": [],
        "totalAcceptedSynthetic": 0,
        "steps": [],
        "curtainCallPolicy": "evaluation only; never training/generation",
    }


def runtime_write(started: float, args, state: dict, stage: str) -> None:
    json_write(
        RUNTIME_PATH,
        {
            "model": MODEL_NAME,
            "complete": bool(state.get("complete")),
            "stage": stage,
            "nextSongIndex": state.get("nextSongIndex", 0),
            "songCount": len(state.get("songOrder", [])),
            "device": args.device,
            "maximumHoursThisLaunch": args.max_hours,
            "launchSeconds": round(time.time() - started, 3),
            "launchDisplay": v41.duration_text(time.time() - started),
        },
    )


def load_accepted_synthetic(state: dict) -> List[dict]:
    result = []
    for relative in state.get("acceptedSyntheticDirectories", []):
        for item in v41.load_items(ROOT / relative):
            result.append(v42.enhance_synthetic_item(item))
    return result


def run(args) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    v41.MODEL_ROOT = MODEL_ROOT
    v41.KEEP_CHECKPOINT_PREVIOUS = False
    removed = prune_training_checkpoints()
    if removed:
        print("Released restartable checkpoints: {:.2f} GB".format(removed / 2**30))
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    if free_gb < args.minimum_free_gb:
        raise OSError(
            "Cドライブの空き容量が不足しています: {:.2f} GB（必要 {:.2f} GB）".format(
                free_gb, args.minimum_free_gb
            )
        )
    started = time.time()
    budget = v41.Budget.create(args.max_hours)
    device = v41.choose_device(args.device)
    real_items = v42.load_real_items(args.manifest, args.rebuild_features)
    curtain = load_curtain_item()
    if any(item["features"].shape[1] != real_items[0]["features"].shape[1] for item in real_items + [curtain]):
        raise ValueError("v4.3 feature counts do not match")
    feature_count = real_items[0]["features"].shape[1]
    if NORMALIZATION_PATH.exists():
        norm = np.load(NORMALIZATION_PATH)
        mean, scale = norm["mean"], norm["scale"]
    else:
        # Only the twelve development songs contribute. Curtain Call is excluded.
        mean, scale = v41.feature_normalization(real_items)
        np.savez_compressed(NORMALIZATION_PATH, mean=mean, scale=scale)

    state = json_read(STATE_PATH, state_default(real_items))
    if state["songOrder"] != [item["name"] for item in real_items]:
        raise ValueError("dataset song order changed after v4.3 started")
    if state.get("complete"):
        print("PianoFlow v4.3 is already complete.")
        return 0
    models = load_models(feature_count, args, device)
    accepted_real = [
        item for item in real_items if item["id"] in state["acceptedRealIds"]
    ]
    accepted_synthetic = load_accepted_synthetic(state)
    print(
        "{} B-first active learning: device={} start={}/{} model={}x{}".format(
            MODEL_NAME,
            device,
            state["nextSongIndex"] + 1,
            len(real_items),
            args.channels,
            args.transformer_layers,
        ),
        flush=True,
    )

    for step in range(state["nextSongIndex"], len(real_items)):
        if budget.expired():
            raise TimeoutError("safe execution budget reached")
        current = real_items[step]
        train_phrases, holdout = phrase_split(current, args.seed + step)
        baseline_current = report_for_models(
            models, [holdout], mean, scale, device
        )
        past_holdouts = [
            phrase_split(item, args.seed + index)[1]
            for index, item in enumerate(accepted_real)
        ]
        baseline_past = report_for_models(
            models, past_holdouts, mean, scale, device
        )
        remaining = max(
            0, args.maximum_synthetic - state["totalAcceptedSynthetic"]
        )
        requested = generated_target(step, baseline_current, remaining)
        step_root = WORK / "rounds" / "{:02d}-{}".format(step + 1, current["id"])
        plan = v41.weakness_plan(baseline_current, requested) if requested else {
            "schemaVersion": SCHEMA_VERSION,
            "requestedSamples": {},
            "maximumSamples": 0,
        }
        json_write(step_root / "augmentation-plan.json", plan)
        generated, quality = generate_theory_synthetic(
            accepted_real + [current],
            plan,
            requested,
            args.seed + step * 1000,
            budget,
            step_root / "synthetic",
            current["name"],
            args.minimum_quality_yield,
        ) if requested else ([], {"qualityGatePassed": False, "accepted": 0})

        common_training = accepted_real + train_phrases + accepted_synthetic
        b_training_items = common_training + generated
        branch_seed = args.seed + 10000 + step * 37
        b_models, _, _, b_training = v41.train_bundle(
            "step-{:02d}-B-primary".format(step + 1),
            b_training_items,
            [holdout],
            device,
            budget,
            args.epochs_per_song,
            args.channels,
            args.transformer_layers,
            args.window,
            args.batch_size,
            branch_seed,
            initial_models=models,
            normalization_values=(mean, scale),
        )
        b_current = report_for_models(
            b_models, [holdout], mean, scale, device
        )
        b_past = report_for_models(
            b_models, past_holdouts, mean, scale, device
        )
        b_health = combined_health(
            baseline_current, b_current, baseline_past, b_past, args
        )
        b_health["qualityGatePassed"] = bool(quality.get("qualityGatePassed"))
        b_passed = bool(
            b_health["passed"]
            and generated
            and quality.get("qualityGatePassed")
        )
        prune_training_checkpoints()

        a_models = None
        a_training = None
        a_current = None
        a_past = None
        a_health = None
        if b_passed or (models is None and FORCE_INITIAL_B):
            chosen = "B-primary-synthetic"
            chosen_models = b_models
            chosen_current = b_current
            chosen_past = b_past
            accepted_synthetic.extend(generated)
            relative = (step_root / "synthetic").relative_to(ROOT).as_posix()
            state["acceptedSyntheticDirectories"].append(relative)
            state["totalAcceptedSynthetic"] += len(generated)
        else:
            print(
                "B failed safety gates; training real-only A fallback.",
                flush=True,
            )
            a_models, _, _, a_training = v41.train_bundle(
                "step-{:02d}-A-fallback".format(step + 1),
                common_training,
                [holdout],
                device,
                budget,
                args.epochs_per_song,
                args.channels,
                args.transformer_layers,
                args.window,
                args.batch_size,
                branch_seed,
                initial_models=models,
                normalization_values=(mean, scale),
            )
            a_current = report_for_models(
                a_models, [holdout], mean, scale, device
            )
            a_past = report_for_models(
                a_models, past_holdouts, mean, scale, device
            )
            a_health = combined_health(
                baseline_current, a_current, baseline_past, a_past, args
            )
            prune_training_checkpoints()
            if a_health["passed"]:
                chosen = "A-real-only-fallback"
                chosen_models = a_models
                chosen_current = a_current
                chosen_past = a_past
            elif ALLOW_DEFERRED_ADVANCE:
                chosen = "deferred-real-data"
                chosen_models = models
                chosen_current = baseline_current
                chosen_past = baseline_past
            else:
                chosen = "retained-previous"
                chosen_models = models
                chosen_current = baseline_current
                chosen_past = baseline_past

        step_report = {
            "step": step + 1,
            "song": current["name"],
            "holdout": {
                "phrase": holdout["name"],
                "noteRange": holdout["phraseNoteRange"],
                "neverUsedForBranchTraining": True,
            },
            "preLearningHoldout": baseline_current,
            "pastBefore": baseline_past,
            "requestedSynthetic": requested,
            "generatedSynthetic": len(generated),
            "quality": quality,
            "BPrimary": {
                "training": b_training,
                "currentHoldout": b_current,
                "pastHoldouts": b_past,
                "health": b_health,
            },
            "AFallback": None if a_models is None else {
                "training": a_training,
                "currentHoldout": a_current,
                "pastHoldouts": a_past,
                "health": a_health,
            },
            "chosenBranch": chosen,
            "postLearningHoldout": chosen_current,
            "postLearningPast": chosen_past,
        }
        json_write(step_root / "report.json", step_report)
        state["steps"].append(step_report)
        if chosen == "retained-previous":
            state["blockedSong"] = {
                "index": step,
                "name": current["name"],
                "report": (step_root / "report.json").relative_to(ROOT).as_posix(),
                "reason": "B failed quality/regression gates and A also regressed.",
            }
            state["nextSongIndex"] = step
            json_write(STATE_PATH, state)
            runtime_write(started, args, state, "needs-review")
            return 76

        models = chosen_models
        accepted_real.append(current)
        state["acceptedRealIds"].append(current["id"])
        state.pop("blockedSong", None)
        state["nextSongIndex"] = step + 1
        if models is not None:
            save_models(models, mean, scale, args, step + 1, chosen)
        json_write(STATE_PATH, state)
        runtime_write(started, args, state, "completed-song-{}".format(step + 1))
        print(
            "[{}/{}] {} holdout pre={:.4f} B={:.4f} choose={} quality={}/{}".format(
                step + 1,
                len(real_items),
                current["name"],
                baseline_current["meanNoteAccuracy"],
                b_current["meanNoteAccuracy"],
                chosen,
                len(generated),
                requested,
            ),
            flush=True,
        )

    # Final from-scratch model uses phrase-level holdouts rather than validating
    # on the same frames it learns.
    final_training_phrases = []
    final_holdouts = []
    for index, item in enumerate(real_items):
        training, holdout = phrase_split(item, args.seed + index)
        final_training_phrases.extend(training)
        final_holdouts.append(holdout)
    final_models, _, _, final_training = v41.train_bundle(
        "final-v43-from-scratch",
        final_training_phrases + accepted_synthetic,
        final_holdouts,
        device,
        budget,
        args.final_epochs,
        args.channels,
        args.transformer_layers,
        args.window,
        args.batch_size,
        args.seed + 50000,
        initial_models=None,
        normalization_values=(mean, scale),
    )
    prune_training_checkpoints()
    internal_holdout_report = report_for_models(
        final_models, final_holdouts, mean, scale, device
    )
    curtain_report = report_for_models(
        final_models, [curtain], mean, scale, device
    )
    save_models(
        final_models,
        mean,
        scale,
        args,
        len(real_items),
        "final-from-scratch-phrase-holdout",
    )
    state["complete"] = True
    state["internalPhraseHoldout"] = internal_holdout_report
    state["curtainCallHeldoutEvaluation"] = curtain_report
    json_write(STATE_PATH, state)
    report = {
        "model": MODEL_NAME,
        "mode": MODEL_MODE,
        "complete": True,
        "device": str(device),
        "modelConfiguration": {
            "featureCount": feature_count,
            "channels": args.channels,
            "transformerLayers": args.transformer_layers,
            "parameterCounts": {
                task: v41.parameter_count(model)
                for task, model in final_models.items()
            },
            "epochsPerSong": args.epochs_per_song,
            "finalEpochs": args.final_epochs,
        },
        "composer": {
            "type": "hierarchical music-theory chorus composer",
            "bars": 8,
            "conditioning": [
                "short_note",
                "same_pitch",
                "ornament",
                "tuplet",
                "octave",
            ],
            "qualityGates": [
                "music theory",
                "weakness coverage",
                "novelty",
                "audio-MIDI pitch",
                "note boundary support",
                "signal integrity",
            ],
        },
        "selectionPolicy": {
            "primary": "B real plus quality-approved synthetic",
            "fallback": "train A only when B fails safety gates",
            "currentEvaluation": "one held-out contiguous phrase",
        },
        "acceptedSynthetic": state["totalAcceptedSynthetic"],
        "steps": state["steps"],
        "finalTraining": final_training,
        "internalPhraseHoldout": internal_holdout_report,
        "curtainCallHeldoutEvaluation": curtain_report,
        "curtainCallLeakagePolicy": (
            "Curtain Call reference MIDI was used only after final inference for "
            "evaluation; it was never used for normalization, generation, training, "
            "early stopping, branch selection, or model selection."
        ),
    }
    json_write(REPORT_PATH, report)
    runtime_write(started, args, state, "complete")
    print("COMPLETE: {}".format(REPORT_PATH), flush=True)
    return 0


def self_test() -> int:
    rng_item = {
        "id": "dummy",
        "name": "dummy",
        "kind": "real",
        "bpm": 140,
        "notes": [
            {
                "midi": 72 + (index % 5),
                "start": index * 0.2,
                "end": index * 0.2 + 0.16,
            }
            for index in range(50)
        ],
    }
    composer = TheoryChorusComposer([rng_item], 4301)
    summaries = {}
    for index, category in enumerate(
        ("short_note", "same_pitch", "ornament", "tuplet", "octave")
    ):
        notes, bpm, ornaments, syllables, theory = composer.compose(category, index)
        theory["ornaments"] = ornaments
        audit = theory_quality(notes, theory, category)
        assert audit["accepted"], (category, audit)
        assert len(notes) == len(ornaments) == len(syllables)
        summaries[category] = {
            "notes": len(notes),
            "bpm": bpm,
            "coverage": audit["coverage"][category],
        }
    frame_count = 1000
    dummy = {
        **rng_item,
        "features": np.zeros((frame_count, 134 + V4_PRIOR_FEATURES), np.float32),
        "times": np.arange(frame_count, dtype=np.float32) * 0.01,
        "pitchY": np.zeros(frame_count, np.int64),
        "pitchWeight": np.ones(frame_count, np.float32),
        "ornamentY": np.zeros(frame_count, np.int64),
        "ornamentWeight": np.ones(frame_count, np.float32),
        "boundaryY": np.zeros((frame_count, 4), np.float32),
    }
    training, holdout = phrase_split(dummy, 2)
    assert len(training) == 4
    assert holdout["phraseNoteRange"][1] > holdout["phraseNoteRange"][0]
    truncated = {
        **dummy,
        "id": "dummy-truncated",
        "name": "dummy-truncated",
        "features": dummy["features"][:500],
        "times": dummy["times"][:500],
        "pitchY": dummy["pitchY"][:500],
        "pitchWeight": dummy["pitchWeight"][:500],
        "ornamentY": dummy["ornamentY"][:500],
        "ornamentWeight": dummy["ornamentWeight"][:500],
        "boundaryY": dummy["boundaryY"][:500],
    }
    truncated_training, truncated_holdout = phrase_split(truncated, 2)
    assert len(truncated_training) == 4
    assert truncated_holdout["discardedOutsideAcousticRange"] > 0
    assert all(
        phrase["times"][-1] <= truncated["times"][-1]
        for phrase in truncated_training + [truncated_holdout]
    )
    model = v41.HeavyAcousticModel(
        dummy["features"].shape[1],
        v41.PITCH_CLASSES,
        channels=64,
        transformer_layers=1,
    )
    output = model(torch.zeros((1, 64, dummy["features"].shape[1])))
    assert output.shape == (1, 64, v41.PITCH_CLASSES)
    print(
        json.dumps(
            {
                "composer": summaries,
                "trainingPhrases": len(training),
                "holdout": holdout["phraseNoteRange"],
                "truncatedAudioSupportedNotes": truncated_holdout[
                    "acousticallySupportedNoteCount"
                ],
                "truncatedAudioDiscardedNotes": truncated_holdout[
                    "discardedOutsideAcousticRange"
                ],
                "smallModelParameters": v41.parameter_count(model),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PianoFlow v4.3 theory-chorus B-first active learning"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "dataset-work" / "manifest.json",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-hours", type=float, default=71.5)
    parser.add_argument("--epochs-per-song", type=int, default=4)
    parser.add_argument("--final-epochs", type=int, default=8)
    parser.add_argument("--maximum-synthetic", type=int, default=900)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument(
        "--transformer-layers", type=int, default=DEFAULT_TRANSFORMER_LAYERS
    )
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--minimum-quality-yield", type=float, default=0.20)
    parser.add_argument("--minimum-free-gb", type=float, default=2.0)
    parser.add_argument("--current-pitch-tolerance", type=float, default=0.005)
    parser.add_argument("--current-ornament-tolerance", type=float, default=0.02)
    parser.add_argument("--current-boundary-tolerance", type=float, default=0.02)
    parser.add_argument("--mean-regression", type=float, default=0.01)
    parser.add_argument("--song-regression", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=4301)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        return run(args)
    except TimeoutError as error:
        state = json_read(STATE_PATH, {})
        runtime_write(time.time(), args, state, "time-budget")
        print("TIME BUDGET: {}".format(error), flush=True)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
