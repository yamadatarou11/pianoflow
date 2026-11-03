from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import learning_v41 as v41
import learning_v43 as v43
import learning_v431 as v431
from pianoflow_dataset import fixed_alignment_report


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models" / "learning-v431"
REPORT_PATH = ROOT / "dataset-work" / "learning-v431" / "report.json"
DEFAULT_STATUS = ROOT / "dataset-work" / "v431-batch-status.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def update_status(path: Path, **values) -> None:
    current = read_json(path) if path.exists() else {}
    current.update(values)
    current["updatedAt"] = time.time()
    write_json(path, current)


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAを利用できません")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_models(device: torch.device):
    v41.MODEL_FACTORY = v431.model_factory
    models = {}
    mean = scale = None
    for task in v41.TASKS:
        path = MODEL_ROOT / f"current-{task}.pt"
        if not path.exists():
            raise FileNotFoundError(f"学習済みモデルがありません: {path}")
        payload = torch.load(path, map_location=device, weights_only=False)
        model = v431.model_factory(
            task,
            int(payload["featureCount"]),
            int(payload["outputCount"]),
            int(payload["channels"]),
            int(payload["transformerLayers"]),
        ).to(device)
        model.load_state_dict(payload["bestState"])
        model.eval()
        models[task] = model
        if mean is None:
            mean = np.asarray(payload["mean"], dtype=np.float32)
            scale = np.asarray(payload["scale"], dtype=np.float32)
    return models, mean, scale


def predict_with_gpu_worker(features: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    """Run only the trained v4.3.1 networks in the isolated CUDA environment."""
    python = os.environ.get("PIANOFLOW_LEARNING_GPU_WORKER", "").strip()
    model_root = os.environ.get("PIANOFLOW_LEARNING_GPU_MODEL_ROOT", "").strip()
    worker = ROOT / "pianoflow_gpu_worker.py"
    if not python or not Path(python).is_file():
        raise RuntimeError("GPU版Python環境が見つかりません。GPU版の起動BATから起動してください")
    if not model_root or not all(
        (Path(model_root) / ("current-" + task + ".pt")).is_file()
        for task in v41.TASKS
    ):
        raise RuntimeError("Torch 1.3互換の学習済みモデルがありません。GPU版を再起動してください")
    if not worker.is_file():
        raise RuntimeError("GPU推論ワーカーが見つかりません: " + str(worker))
    with tempfile.TemporaryDirectory(prefix="pianoflow-gpu-infer-") as temporary:
        request_path = Path(temporary) / "request.npz"
        output_path = Path(temporary) / "probabilities.npz"
        np.savez(request_path, features=features, mean=mean, scale=scale)
        command = [
            python, str(worker),
            "--request", str(request_path),
            "--output", str(output_path),
            "--model-dir", model_root,
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )
        if result.returncode != 0 or not output_path.is_file():
            detail = (result.stderr or result.stdout or "CUDA推論に失敗しました")[-2400:]
            raise RuntimeError("GPU推論に失敗しました。CPUへ自動切替せず停止しました。\n" + detail)
        with np.load(output_path) as values:
            probabilities = {task: np.asarray(values[task], dtype=np.float32) for task in v41.TASKS}
        if any(len(probabilities[task]) != len(features) for task in v41.TASKS):
            raise RuntimeError("GPU推論の出力フレーム数が一致しません")
        gpu_name = next(
            (line.split("=", 1)[1].strip() for line in result.stdout.splitlines() if line.startswith("GPU_DEVICE=")),
            "CUDA GPU",
        )
        return probabilities, gpu_name


def infer_audio_pitch(audio_path: Path, device_name: str = "auto") -> tuple[list[dict], dict]:
    """Run the trained v4.3.1 audio-only pitch model on one dry vocal stem.

    This intentionally does not read a reference MIDI or supplied lyrics. The
    model's training target is one octave below the aligned piano score, so the
    returned piano notes are shifted up by one octave after decoding.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"解析用ボーカル音源がありません: {audio_path}")
    worker_python = os.environ.get("PIANOFLOW_LEARNING_GPU_WORKER", "").strip()
    use_gpu_worker = device_name == "cuda" and bool(worker_python)
    # Feature extraction stays on the main CPU environment. Only neural model
    # inference crosses into the legacy CUDA-compatible worker.
    device = torch.device("cpu") if use_gpu_worker else choose_device(device_name)
    if use_gpu_worker:
        checkpoint = torch.load(MODEL_ROOT / "current-pitch.pt", map_location="cpu", weights_only=False)
        mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        scale = np.asarray(checkpoint["scale"], dtype=np.float32)
        del checkpoint
        models = None
    else:
        models, mean, scale = load_models(device)
    raw_features, times = v41.extract_features(audio_path)
    pitch_prior, ornament_prior, boundary_prior = v43.acoustic_priors(
        audio_path, raw_features, times
    )
    features = np.column_stack(
        (raw_features, pitch_prior, ornament_prior, boundary_prior)
    ).astype(np.float32)
    if features.shape[1] != len(mean):
        raise ValueError(
            f"学習モデルと特徴量が一致しません: {features.shape[1]} != {len(mean)}"
        )
    if use_gpu_worker:
        probabilities, gpu_name = predict_with_gpu_worker(features, mean, scale)
        inference_device = gpu_name + " (learning-v4.3.1 inference)"
    else:
        item = {"features": features}
        probabilities = {
            task: v41.predict_item(models[task], item, mean, scale, task, device)
            for task in v41.TASKS
        }
        inference_device = str(device)
    notes, diagnostics = decode_notes(times, probabilities, validation_thresholds())
    for note in notes:
        note["vocalMidi"] = int(note["midi"])
        note["midi"] = int(note["midi"]) + 12
        note["pianoOctaveShift"] = 12
    return notes, {
        "model": "learning-v4.3.1",
        "device": inference_device,
        "referenceMidiUsed": False,
        "suppliedLyricsUsedForPitch": False,
        "featureFrames": int(len(times)),
        **diagnostics,
    }


def validation_thresholds() -> dict:
    defaults = {"onset": 0.45, "offset": 0.40, "samePitch": 0.45, "pitchChange": 0.45}
    if not REPORT_PATH.exists():
        return defaults
    report = read_json(REPORT_PATH)
    rows = report.get("internalPhraseHoldout", {}).get("songs", [])
    for name in defaults:
        values = [
            float(row.get("boundaries", {}).get(name, {}).get("threshold"))
            for row in rows
            if row.get("boundaries", {}).get(name, {}).get("threshold") is not None
        ]
        if values:
            defaults[name] = float(statistics.median(values))
    return defaults


def local_peaks(values: np.ndarray, threshold: float, spacing: int) -> list[int]:
    candidates = []
    for index in range(1, len(values) - 1):
        if values[index] < threshold or values[index] < values[index - 1] or values[index] < values[index + 1]:
            continue
        if candidates and index - candidates[-1] < spacing:
            if values[index] > values[candidates[-1]]:
                candidates[-1] = index
        else:
            candidates.append(index)
    return candidates


def representative_pitch(
    pitch_probability: np.ndarray,
    ornament_probability: np.ndarray,
    start: int,
    end: int,
) -> tuple[int | None, float, float]:
    if end <= start:
        return None, 0.0, 0.0
    edge = min(4, max(0, (end - start) // 5))
    indices = np.arange(start + edge, max(start + edge + 1, end - edge))
    indices = indices[indices < len(pitch_probability)]
    if not len(indices):
        indices = np.arange(start, min(end, len(pitch_probability)))
    voiced = 1.0 - pitch_probability[indices, 0]
    stable = ornament_probability[indices, 0]
    weights = np.maximum(0.02, voiced * (0.35 + 0.65 * stable))
    scores = (pitch_probability[indices, 1:] * weights[:, None]).sum(0)
    selected = int(np.argmax(scores))
    confidence = float(scores[selected] / max(1e-7, weights.sum()))
    voiced_mean = float(np.mean(voiced))
    if voiced_mean < 0.30 or confidence < 0.055:
        return None, confidence, voiced_mean
    return selected + v41.MIDI_MIN, confidence, voiced_mean


def decode_notes(times: np.ndarray, probabilities: dict, thresholds: dict) -> tuple[list[dict], dict]:
    pitch = probabilities["pitch"]
    ornament = probabilities["ornament"]
    boundary = probabilities["boundary"]
    voiced = 1.0 - pitch[:, 0]
    frame_pitch = np.argmax(pitch, axis=1)
    frame_seconds = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    spacing = max(2, int(round(0.045 / max(frame_seconds, 1e-4))))
    candidates = set(local_peaks(boundary[:, 0], thresholds["onset"], spacing))
    candidates.update(local_peaks(boundary[:, 2], thresholds["samePitch"], spacing))
    candidates.update(local_peaks(boundary[:, 3], thresholds["pitchChange"], spacing))
    for index in range(1, len(voiced)):
        if voiced[index] >= 0.43 and voiced[max(0, index - 1)] < 0.30:
            candidates.add(index)
    persistence = max(2, int(round(0.035 / max(frame_seconds, 1e-4))))
    for index in range(persistence, len(frame_pitch) - persistence):
        before = frame_pitch[index - persistence]
        after = frame_pitch[index + persistence]
        if before > 0 and after > 0 and before != after and boundary[index, 3] >= thresholds["pitchChange"] * 0.72:
            candidates.add(index)
    voiced_indices = np.flatnonzero(voiced >= 0.40)
    if len(voiced_indices):
        candidates.add(int(voiced_indices[0]))
    starts = sorted(index for index in candidates if 0 <= index < len(times) and voiced[index] >= 0.22)
    filtered = []
    for index in starts:
        if filtered and index - filtered[-1] < spacing:
            old = filtered[-1]
            old_strength = max(boundary[old, 0], boundary[old, 2], boundary[old, 3])
            new_strength = max(boundary[index, 0], boundary[index, 2], boundary[index, 3])
            if new_strength > old_strength:
                filtered[-1] = index
        else:
            filtered.append(index)
    notes = []
    rejected = 0
    for position, start in enumerate(filtered):
        next_start = filtered[position + 1] if position + 1 < len(filtered) else len(times) - 1
        if next_start <= start:
            continue
        segment_voiced = np.flatnonzero(voiced[start:next_start] >= 0.25)
        if not len(segment_voiced):
            rejected += 1
            continue
        end = start + int(segment_voiced[-1]) + 1
        offset_candidates = local_peaks(boundary[start:end, 1], thresholds["offset"], spacing)
        if offset_candidates:
            late = [value for value in offset_candidates if value >= spacing]
            if late:
                end = min(end, start + late[-1] + 1)
        midi, confidence, voiced_mean = representative_pitch(pitch, ornament, start, end)
        note_start = float(times[start])
        note_end = float(times[min(len(times) - 1, max(start + 1, end))])
        if midi is None or note_end - note_start < 0.028:
            rejected += 1
            continue
        strength = float(max(boundary[start, 0], boundary[start, 2], boundary[start, 3]))
        notes.append({
            "midi": int(midi), "start": note_start, "end": max(note_start + 0.03, note_end),
            "role": "melody", "strength": round(strength, 5),
            "classification": "learning-v431-audio-only",
            "learningV431PitchConfidence": round(confidence, 6),
            "learningV431VoicedProbability": round(voiced_mean, 6),
            "learningV431Ornament": ("stable", "scoop_or_fall", "transition")[int(np.argmax(ornament[start:end].mean(0)))],
        })
    merged = []
    for note in notes:
        if merged and note["midi"] == merged[-1]["midi"] and note["start"] - merged[-1]["end"] <= 0.045:
            frame = int(np.clip(np.searchsorted(times, note["start"]), 0, len(times) - 1))
            if boundary[frame, 2] < thresholds["samePitch"]:
                merged[-1]["end"] = note["end"]
                continue
        merged.append(note)
    return merged, {
        "frameSeconds": round(frame_seconds, 6), "candidateBoundaries": len(filtered),
        "rejectedSegments": rejected, "decodedNotes": len(merged), "thresholds": thresholds,
    }


def process_song(song: dict, models, mean, scale, device, thresholds) -> dict:
    base = (ROOT / song["referenceFile"]).parent / "audio-only"
    analysis_path = base / "analysis.json"
    alignment_path = base / "initial-alignment.json"
    if not analysis_path.exists() or not alignment_path.exists():
        raise FileNotFoundError(f"基礎解析がありません: {song['name']}")
    analysis = read_json(analysis_path)
    alignment = read_json(alignment_path)
    dry_name = analysis.get("stemFiles", {}).get("dryLead") or analysis.get("stemFiles", {}).get("lead") or "lead-vocal-dry.wav"
    audio_path = base / dry_name
    raw_features, times = v41.extract_features(audio_path)
    pitch_prior, ornament_prior, boundary_prior = v43.acoustic_priors(audio_path, raw_features, times)
    features = np.column_stack((raw_features, pitch_prior, ornament_prior, boundary_prior)).astype(np.float32)
    if features.shape[1] != len(mean):
        raise ValueError(f"特徴量数がモデルと一致しません: {song['name']} {features.shape[1]} != {len(mean)}")
    item = {"features": features}
    probabilities = {task: v41.predict_item(models[task], item, mean, scale, task, device) for task in v41.TASKS}
    notes, diagnostics = decode_notes(times, probabilities, thresholds)
    evaluation = fixed_alignment_report(
        alignment["alignedNotes"], notes, tolerance=0.120,
        alignment=alignment.get("alignment"), predicted_to_reference_transpose=12,
    )
    result = {
        **analysis,
        "analysisVersion": "learning-v4.3.1-audio-only",
        "notes": notes,
        "learningV431Diagnostics": diagnostics,
        "learningV431ReferenceMidiUsedForInference": False,
        "learningV431ModelPaths": {task: str(MODEL_ROOT / f"current-{task}.pt") for task in v41.TASKS},
    }
    write_json(base / "learning-v431-analysis.json", result)
    write_json(base / "learning-v431-evaluation.json", evaluation)
    return {
        "id": song["id"], "name": song["name"], "notes": len(notes),
        "correctPitch": evaluation.get("correctPitch", 0), "wrongPitch": evaluation.get("wrongPitch", 0),
        "missingNotes": evaluation.get("missingNotes", 0), "extraNotes": evaluation.get("extraNotes", 0),
        "f1": evaluation.get("f1", 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PianoFlow learning v4.3.1 全曲試聴用推論")
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset-work" / "manifest.json")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    started = time.time()
    try:
        manifest = read_json(args.manifest)
        device = choose_device(args.device)
        models, mean, scale = load_models(device)
        thresholds = validation_thresholds()
        songs = manifest.get("songs", [])
        update_status(args.status, state="running", stage="v431-inference", current=0, total=len(songs), device=str(device), message="v4.3.1モデルを読み込みました", error="")
        rows = []
        for index, song in enumerate(songs, 1):
            update_status(args.status, state="running", stage="v431-inference", current=index - 1, total=len(songs), song=song["name"], message=f"{index}/{len(songs)} {song['name']}を解析中")
            rows.append(process_song(song, models, mean, scale, device, thresholds))
            update_status(args.status, state="running", stage="v431-inference", current=index, total=len(songs), song=song["name"], message=f"{index}/{len(songs)} 完了")
        summary = {"model": "learning-v4.3.1", "complete": True, "device": str(device), "seconds": round(time.time() - started, 2), "songs": rows}
        write_json(ROOT / "dataset-work" / "learning-v431-listening-summary.json", summary)
        update_status(args.status, state="complete", stage="complete", current=len(songs), total=len(songs), message="全曲のv4.3.1解析が完了しました", summary=summary)
        return 0
    except Exception as error:
        update_status(args.status, state="failed", stage="failed", message="一括解析に失敗しました", error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
