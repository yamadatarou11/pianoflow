from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_pitch_tracking import estimate_offset, parse_midi
from evaluate_synchronized_timing import build_report
from initial_score_alignment import align_first_vocal_note, apply_constant_offset


def infer_octave_and_offset(reference: list[dict], predicted: list[dict]) -> tuple[int, float]:
    best: tuple[float, int, float] | None = None
    for transpose in (-24, -12, 0, 12, 24):
        shifted = [{**note, "midi": int(note["midi"]) + transpose} for note in reference]
        offset = estimate_offset(shifted, predicted)
        score = 0.0
        for expected in shifted:
            errors = [
                abs(float(actual["start"]) - (float(expected["start"]) + offset))
                for actual in predicted
                if int(actual["midi"]) == int(expected["midi"])
            ]
            if errors and min(errors) <= 0.22:
                score += 1.0 - min(errors) / 0.22
        candidate = (score, transpose, offset)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2]


def scale_notes(reference: list[dict], scale: float) -> list[dict]:
    return [
        {**note, "start": float(note["start"]) * scale, "end": float(note["end"]) * scale}
        for note in reference
    ]


def infer_time_scale_octave_and_offset(
    reference: list[dict], predicted: list[dict], expected_scale: float
) -> tuple[float, int, float]:
    """Estimate tempo/time-base mismatch before local piecewise alignment."""
    coarse = set(np.linspace(0.55, 1.65, 45).tolist())
    coarse.update(np.linspace(max(0.45, expected_scale * 0.78), min(1.9, expected_scale * 1.22), 25).tolist())
    best: tuple[float, float, int, float] | None = None
    for scale in sorted(coarse):
        scaled = scale_notes(reference, float(scale))
        transpose, offset = infer_octave_and_offset(scaled, predicted)
        score = 0.0
        for expected in scaled:
            errors = [
                abs(float(actual["start"]) - (float(expected["start"]) + offset))
                for actual in predicted
                if int(actual["midi"]) == int(expected["midi"]) + transpose
            ]
            if errors and min(errors) <= 0.18:
                score += 1.0 - min(errors) / 0.18
        candidate = (score, float(scale), transpose, offset)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    for scale in np.linspace(max(0.4, best[1] - 0.035), min(2.0, best[1] + 0.035), 29):
        scaled = scale_notes(reference, float(scale))
        transpose, offset = infer_octave_and_offset(scaled, predicted)
        score = 0.0
        for expected in scaled:
            errors = [abs(float(actual["start"]) - (float(expected["start"]) + offset)) for actual in predicted if int(actual["midi"]) == int(expected["midi"]) + transpose]
            if errors and min(errors) <= 0.16:
                score += 1.0 - min(errors) / 0.16
        if score > best[0]:
            best = (score, float(scale), transpose, offset)
    return best[1], best[2], best[3]


def monotonic_matches(reference: list[dict], predicted: list[dict], transpose: int, offset: float) -> list[tuple[int, int]]:
    rows, columns = len(reference), len(predicted)
    cost = np.full((rows + 1, columns + 1), np.inf)
    back = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    cost[0, 0] = 0.0
    for i in range(rows + 1):
        for j in range(columns + 1):
            if i < rows and cost[i, j] + 1.25 < cost[i + 1, j]:
                cost[i + 1, j], back[i + 1, j] = cost[i, j] + 1.25, 1
            if j < columns and cost[i, j] + 1.25 < cost[i, j + 1]:
                cost[i, j + 1], back[i, j + 1] = cost[i, j] + 1.25, 2
            if i < rows and j < columns:
                pitch_error = abs((int(reference[i]["midi"]) + transpose) - int(predicted[j]["midi"]))
                time_error = abs((float(reference[i]["start"]) + offset) - float(predicted[j]["start"]))
                pitch_cost = 0.0 if pitch_error == 0 else 0.65 + 0.42 * min(pitch_error, 5)
                match_cost = pitch_cost + min(3.0, time_error / 0.11)
                if cost[i, j] + match_cost < cost[i + 1, j + 1]:
                    cost[i + 1, j + 1], back[i + 1, j + 1] = cost[i, j] + match_cost, 3
    pairs: list[tuple[int, int]] = []
    i, j = rows, columns
    while i or j:
        action = int(back[i, j])
        if action == 3:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif action == 1:
            i -= 1
        elif action == 2:
            j -= 1
        else:
            break
    return list(reversed(pairs))


def build_anchors(reference: list[dict], predicted: list[dict], pairs: list[tuple[int, int]], transpose: int, offset: float) -> list[dict]:
    reliable = []
    for reference_index, predicted_index in pairs:
        expected, actual = reference[reference_index], predicted[predicted_index]
        pitch_error = abs((int(expected["midi"]) + transpose) - int(actual["midi"]))
        residual = float(actual["start"]) - (float(expected["start"]) + offset)
        if pitch_error <= 1 and abs(residual) <= 0.24:
            reliable.append((float(expected["start"]), float(actual["start"]), pitch_error))
    anchors: list[dict] = []
    maximum = max((float(note["end"]) for note in reference), default=0.0)
    edges = np.arange(0.0, maximum + 4.0, 4.0)
    for left, right in zip(edges[:-1], edges[1:]):
        region = [row for row in reliable if left <= row[0] < right]
        if not region:
            continue
        score_time = float(np.median([row[0] for row in region]))
        local_offset = float(np.median([row[1] - row[0] for row in region]))
        anchors.append({
            "scoreTime": round(score_time, 6),
            "audioTime": round(score_time + local_offset, 6),
            "offset": round(local_offset, 6),
            "support": len(region),
        })
    if not anchors:
        anchors = [{"scoreTime": 0.0, "audioTime": round(offset, 6), "offset": round(offset, 6), "support": 0}]
    return anchors


def map_time(value: float, anchors: list[dict]) -> float:
    score = np.asarray([row["scoreTime"] for row in anchors], dtype=float)
    offsets = np.asarray([row["offset"] for row in anchors], dtype=float)
    if len(score) == 1:
        return value + float(offsets[0])
    local_offset = float(np.interp(value, score, offsets, left=offsets[0], right=offsets[-1]))
    return value + local_offset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.reference.read_text(encoding="utf-8"))
    midi_path = args.reference.parent / config["midi"]
    reference_notes, midi_bpm = parse_midi(midi_path)
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    predicted = [note for note in analysis.get("notes", []) if note.get("role", "melody") == "melody"]
    requested_bpm = float(config.get("bpm", midi_bpm) or midi_bpm)
    expected_scale = float(midi_bpm) / max(1.0, requested_bpm)
    timed_reference = scale_notes(reference_notes, expected_scale)
    initial = align_first_vocal_note(
        timed_reference,
        predicted,
        syllable_segments=analysis.get("syllableSegments", []),
    )
    synchronized_reference = apply_constant_offset(timed_reference, initial.offset_seconds)
    synchronization = {
        **initial.as_dict(),
        "fixedScoreBpm": round(float(requested_bpm), 4),
        "fixedMidiClock": True,
        "alignmentUsesFinalBoundaries": True,
    }
    diagnostic_report = build_report(synchronized_reference, predicted, 0.12, synchronization)
    diagnostic_transpose = int(diagnostic_report["referenceTransposeSemitones"])
    time_scale = expected_scale
    synchronized_offset = float(synchronization["audioTimeAtMidiZero"])
    anchors = [{
        "scoreTime": 0.0,
        "audioTime": round(synchronized_offset, 6),
        "offset": round(synchronized_offset, 6),
        "support": int(synchronization.get("matchedSequenceNotes", 0)),
    }]
    aligned_notes = [
        {
            "midi": int(note["midi"]),
            "start": round(float(note["start"]), 6),
            "end": round(float(note["end"]), 6),
            "scoreStart": round(float(note.get("scoreStart", note["start"])), 6),
            "scoreEnd": round(float(note.get("scoreEnd", note["end"])), 6),
        }
        for note in synchronized_reference
    ]
    result = {
        "name": config["name"],
        "groundTruth": "MIDI score; pitch is never transposed in alignedNotes",
        "bpm": round(requested_bpm, 6),
        "midiBpm": round(midi_bpm, 6),
        "timeScale": round(time_scale, 8),
        "lyrics": config.get("lyrics", []),
        "lyricsText": "\n".join(config.get("lyrics", [])),
        "diagnosticAnalysisToScoreTranspose": -diagnostic_transpose,
        "initialGlobalOffsetSeconds": round(initial.offset_seconds, 6),
        "synchronizedGlobalOffsetSeconds": round(synchronized_offset, 6),
        "synchronization": synchronization,
        "anchors": anchors,
        "matchedPairs": diagnostic_report["correctNotes"],
        "meanAnchorResidualMs": diagnostic_report["meanAbsoluteOnsetErrorMs"],
        "errorDetection": {
            key: value for key, value in diagnostic_report.items()
            if key != "synchronization"
        },
        "alignedNotes": aligned_notes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "alignedNotes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
