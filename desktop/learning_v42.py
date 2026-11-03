from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import learning_v41 as v41


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "dataset-work" / "learning-v42"
MODEL_ROOT = ROOT / "models" / "learning-v42"
STATE_PATH = WORK / "pipeline-state.json"
REPORT_PATH = WORK / "report.json"
RUNTIME_PATH = WORK / "runtime.json"
NORMALIZATION_PATH = WORK / "normalization.npz"

SCHEMA_VERSION = 42
PITCH_PRIOR_COUNT = v41.PITCH_CLASSES
V4_PRIOR_FEATURES = PITCH_PRIOR_COUNT + 3 + 4
DEFAULT_CHANNELS = 256
DEFAULT_TRANSFORMER_LAYERS = 4


def prune_training_checkpoints() -> int:
    """Remove restartable A/B optimizer states while preserving accepted models."""
    checkpoint_root = MODEL_ROOT / "checkpoints"
    if not checkpoint_root.exists():
        return 0
    removed_bytes = 0
    for entry in checkpoint_root.iterdir():
        if entry.is_dir():
            removed_bytes += sum(
                path.stat().st_size
                for path in entry.rglob("*")
                if path.is_file()
            )
            shutil.rmtree(entry)
        elif entry.is_file():
            removed_bytes += entry.stat().st_size
            entry.unlink()
    return removed_bytes


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(source_times, target_times)
    indices = np.clip(indices, 0, len(source_times) - 1)
    previous = np.maximum(0, indices - 1)
    use_previous = (
        np.abs(source_times[previous] - target_times)
        < np.abs(source_times[indices] - target_times)
    )
    indices[use_previous] = previous[use_previous]
    return indices


def candidate_to_pitch_prior(
    candidate_midi: np.ndarray,
    probability: np.ndarray,
    voiced: np.ndarray,
) -> np.ndarray:
    prior = np.full(
        (len(candidate_midi), PITCH_PRIOR_COUNT), 1e-5, dtype=np.float32
    )
    for frame in range(len(candidate_midi)):
        for slot, midi in enumerate(candidate_midi[frame]):
            midi = int(midi)
            if v41.MIDI_MIN <= midi <= v41.MIDI_MAX:
                pitch_class = midi - v41.MIDI_MIN + 1
                prior[frame, pitch_class] += float(probability[frame, slot])
        prior[frame, 0] += 0.80 if voiced[frame] < 0.5 else 0.015
    prior /= np.maximum(prior.sum(1, keepdims=True), 1e-7)
    return prior


def enhance_real_item(item: dict) -> dict:
    v4_path = ROOT / "dataset-work" / "learning-v4" / item["id"] / "learning-v4-data.npz"
    oof_path = ROOT / "dataset-work" / "learning-v4-oof" / item["id"] / "probabilities.npz"
    if not v4_path.exists() or not oof_path.exists():
        raise FileNotFoundError(
            "v4のOOF出力がありません: {}。先にRUN_LEARNING_V4.batを完了してください。".format(
                item["name"]
            )
        )
    v4_data = np.load(v4_path)
    probabilities = np.load(oof_path)
    source_times = v4_data["frameTimes"].astype(np.float32)
    indices = nearest_indices(source_times, item["times"])
    pitch_prior_source = candidate_to_pitch_prior(
        v4_data["candidateMidi"].astype(np.int16),
        probabilities["pitch"].astype(np.float32),
        v4_data["frameFeatures"][:, 0].astype(np.float32),
    )
    pitch_prior = pitch_prior_source[indices]
    ornament_prior = probabilities["ornament"].astype(np.float32)[indices]
    boundary_prior = probabilities["boundary"].astype(np.float32)[indices]
    features = np.column_stack(
        (item["features"], pitch_prior, ornament_prior, boundary_prior)
    ).astype(np.float32)
    return {
        **item,
        "features": features,
        "v4PitchPrior": pitch_prior,
        "v4OrnamentPrior": ornament_prior,
        "v4BoundaryPrior": boundary_prior,
        "featureCount": int(features.shape[1]),
        "priorSource": "strict v4 OOF; this song excluded from v4 training fold",
    }


def synthetic_priors(item: dict, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    pitch_prior = np.full(
        (len(item["times"]), PITCH_PRIOR_COUNT), 1e-5, np.float32
    )
    for frame, target in enumerate(item["pitchY"]):
        target = int(target)
        if target <= 0:
            pitch_prior[frame, 0] = 0.94
            continue
        missing_probability = 0.20 if item.get("category") in ("octave", "ornament") else 0.08
        if rng.rand() < missing_probability:
            wrong = target + rng.choice((-12, -1, 1, 12))
            wrong = max(1, min(PITCH_PRIOR_COUNT - 1, wrong))
            pitch_prior[frame, wrong] = 0.72
            pitch_prior[frame, 0] = 0.12
        else:
            pitch_prior[frame, target] = 0.76
            for shift, value in ((-1, 0.06), (1, 0.06), (-12, 0.025), (12, 0.025)):
                candidate = target + shift
                if 1 <= candidate < PITCH_PRIOR_COUNT:
                    pitch_prior[frame, candidate] += value
    pitch_prior /= np.maximum(pitch_prior.sum(1, keepdims=True), 1e-7)

    ornament_prior = np.full((len(item["times"]), 3), 0.08, np.float32)
    for frame, target in enumerate(item["ornamentY"]):
        if int(target) >= 0:
            ornament_prior[frame, int(target)] = 0.78
    ornament_prior += rng.normal(0, 0.025, ornament_prior.shape).astype(np.float32)
    ornament_prior = np.clip(ornament_prior, 1e-4, None)
    ornament_prior /= ornament_prior.sum(1, keepdims=True)

    boundary_prior = (
        0.08 + 0.70 * item["boundaryY"] + rng.normal(
            0, 0.06, item["boundaryY"].shape
        )
    ).astype(np.float32)
    boundary_prior = np.clip(boundary_prior, 0.01, 0.99)
    return pitch_prior, ornament_prior, boundary_prior


def enhance_synthetic_item(item: dict) -> dict:
    pitch_prior, ornament_prior, boundary_prior = synthetic_priors(
        item, int(item.get("seed", 4200))
    )
    features = np.column_stack(
        (item["features"], pitch_prior, ornament_prior, boundary_prior)
    ).astype(np.float32)
    return {
        **item,
        "features": features,
        "v4PitchPrior": pitch_prior,
        "v4OrnamentPrior": ornament_prior,
        "v4BoundaryPrior": boundary_prior,
        "featureCount": int(features.shape[1]),
        "priorSource": "controlled noisy v4-like prior generated from exact synthetic MIDI",
    }


def load_real_items(manifest: Path, rebuild: bool) -> List[dict]:
    v41.WORK = ROOT / "dataset-work" / "learning-v41"
    v41.build_real_dataset(manifest, rebuild)
    return [
        enhance_real_item(item)
        for item in v41.load_items(v41.WORK / "real")
    ]


def prior_report(items: Sequence[dict]) -> dict:
    rows = []
    for item in items:
        pitch = v41.note_pitch_metrics(item, item["v4PitchPrior"])
        ornament_valid = item["ornamentY"] >= 0
        ornament_prediction = np.argmax(item["v4OrnamentPrior"], axis=1)
        if np.any(ornament_valid):
            from sklearn.metrics import f1_score

            ornament_f1 = float(
                f1_score(
                    item["ornamentY"][ornament_valid],
                    ornament_prediction[ornament_valid],
                    average="macro",
                    zero_division=0,
                )
            )
        else:
            ornament_f1 = 0.0
        boundaries = {}
        for column, name in enumerate(v41.BOUNDARY_NAMES):
            boundaries[name] = v41.event_f1(
                item["boundaryY"][:, column],
                item["v4BoundaryPrior"][:, column],
                5 if name == "samePitch" else 4,
            )
        rows.append(
            {
                "id": item["id"],
                "name": item["name"],
                "pitch": pitch,
                "ornamentMacroF1": round(ornament_f1, 4),
                "boundaries": boundaries,
            }
        )
    return aggregate_report(rows)


def aggregate_report(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "songs": [],
            "meanNoteAccuracy": 0.0,
            "meanOrnamentMacroF1": 0.0,
            "meanBoundaryF1": 0.0,
        }
    return {
        "songs": list(rows),
        "meanNoteAccuracy": round(
            float(np.mean([row["pitch"]["noteAccuracy"] for row in rows])), 4
        ),
        "meanOrnamentMacroF1": round(
            float(np.mean([row["ornamentMacroF1"] for row in rows])), 4
        ),
        "meanBoundaryF1": round(
            float(
                np.mean(
                    [
                        metric["f1"]
                        for row in rows
                        for metric in row["boundaries"].values()
                    ]
                )
            ),
            4,
        ),
    }


def generated_count(step_index: int, pre_report: dict, remaining: int) -> int:
    if remaining <= 0:
        return 0
    accuracy = float(pre_report.get("meanNoteAccuracy", 0.0))
    boundary = float(pre_report.get("meanBoundaryF1", 0.0))
    severity = np.clip(0.70 * (1 - accuracy) + 0.30 * (1 - boundary), 0, 1)
    if step_index == 0:
        # The first accepted pool has no earlier synthetic coverage.  Give it a
        # larger but still bounded bootstrap allocation.
        requested = max(150, min(250, int(round(150 + 100 * severity))))
    else:
        requested = max(40, min(150, int(round(40 + 110 * severity))))
    return min(remaining, requested)


def regression_check(
    baseline: dict,
    candidate: dict,
    mean_tolerance: float,
    per_song_tolerance: float,
) -> dict:
    if not baseline["songs"]:
        return {
            "passed": True,
            "meanPitchDelta": 0.0,
            "worstSongDelta": 0.0,
            "songDeltas": [],
        }
    baseline_lookup = {
        row["id"]: row["pitch"]["noteAccuracy"] for row in baseline["songs"]
    }
    deltas = []
    for row in candidate["songs"]:
        if row["id"] in baseline_lookup:
            deltas.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "delta": round(
                        row["pitch"]["noteAccuracy"] - baseline_lookup[row["id"]],
                        4,
                    ),
                }
            )
    mean_delta = float(np.mean([row["delta"] for row in deltas])) if deltas else 0.0
    worst_delta = min((row["delta"] for row in deltas), default=0.0)
    return {
        "passed": (
            mean_delta >= -mean_tolerance
            and worst_delta >= -per_song_tolerance
        ),
        "meanPitchDelta": round(mean_delta, 4),
        "worstSongDelta": round(worst_delta, 4),
        "songDeltas": deltas,
    }


def current_model_paths() -> Dict[str, Path]:
    return {
        task: MODEL_ROOT / "current-{}.pt".format(task)
        for task in v41.TASKS
    }


def save_current_models(
    models: Dict[str, torch.nn.Module],
    mean: np.ndarray,
    scale: np.ndarray,
    args,
    step_index: int,
    branch: str,
) -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for task, model in models.items():
        v41.save_checkpoint(
            current_model_paths()[task],
            {
                "schemaVersion": SCHEMA_VERSION,
                "complete": True,
                "task": task,
                "architecture": "HeavyAcousticModelWithV4OOFPriors",
                "featureCount": int(len(mean)),
                "channels": args.channels,
                "transformerLayers": args.transformer_layers,
                "outputCount": v41.task_output_count(task),
                "parameterCount": v41.parameter_count(model),
                "mean": mean,
                "scale": scale,
                "bestState": model.state_dict(),
                "completedStep": step_index,
                "chosenBranch": branch,
            },
        )


def load_current_models(
    feature_count: int,
    args,
    device: torch.device,
) -> Optional[Dict[str, torch.nn.Module]]:
    paths = current_model_paths()
    if not all(path.exists() for path in paths.values()):
        return None
    models = {}
    for task, path in paths.items():
        payload = torch.load(path, map_location=device, weights_only=False)
        model = v41.HeavyAcousticModel(
            feature_count,
            v41.task_output_count(task),
            args.channels,
            args.transformer_layers,
        ).to(device)
        model.load_state_dict(payload["bestState"])
        models[task] = model
    return models


def load_accepted_synthetic(state: dict) -> List[dict]:
    items = []
    for relative in state.get("acceptedSyntheticDirectories", []):
        directory = ROOT / relative
        for item in v41.load_items(directory):
            items.append(enhance_synthetic_item(item))
    return items


def report_for_models(
    models: Optional[Dict[str, torch.nn.Module]],
    items: Sequence[dict],
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> dict:
    if not items:
        return aggregate_report([])
    if models is None:
        return prior_report(items)
    return v41.evaluate_models(models, items, mean, scale, device)


def state_default(order: Sequence[dict]) -> dict:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "complete": False,
        "nextSongIndex": 0,
        "songOrder": [item["name"] for item in order],
        "acceptedRealIds": [],
        "acceptedSyntheticDirectories": [],
        "totalAcceptedSynthetic": 0,
        "steps": [],
    }


def runtime_write(started: float, args, state: dict, stage: str) -> None:
    v41.json_write(
        RUNTIME_PATH,
        {
            "model": "learning-v4.2",
            "complete": bool(state.get("complete")),
            "stage": stage,
            "nextSongIndex": state.get("nextSongIndex", 0),
            "songCount": len(state.get("songOrder", [])),
            "deviceRequest": args.device,
            "maximumHoursThisLaunch": args.max_hours,
            "launchSeconds": round(time.time() - started, 3),
            "launchDisplay": v41.duration_text(time.time() - started),
        },
    )


def run(args) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    v41.MODEL_ROOT = MODEL_ROOT
    v41.KEEP_CHECKPOINT_PREVIOUS = False
    removed_bytes = prune_training_checkpoints()
    if removed_bytes:
        print(
            "Cleaned restartable v4.2 checkpoints: {:.2f} GB".format(
                removed_bytes / (1024 ** 3)
            ),
            flush=True,
        )
    started = time.time()
    budget = v41.Budget.create(args.max_hours)
    device = v41.choose_device(args.device)
    real_items = load_real_items(args.manifest, args.rebuild_features)
    feature_count = real_items[0]["features"].shape[1]
    if any(item["features"].shape[1] != feature_count for item in real_items):
        raise ValueError("曲ごとの特徴量次元が一致しません")
    if NORMALIZATION_PATH.exists():
        norm = np.load(NORMALIZATION_PATH)
        mean, scale = norm["mean"], norm["scale"]
    else:
        # Future-song labels are never used.  Global unsupervised acoustic
        # normalization is fixed once so sequential checkpoints remain valid.
        mean, scale = v41.feature_normalization(real_items)
        np.savez_compressed(NORMALIZATION_PATH, mean=mean, scale=scale)

    state = v41.json_read(STATE_PATH, state_default(real_items))
    if state.get("songOrder") != [item["name"] for item in real_items]:
        raise ValueError("データセットの曲順が前回実行時と異なります")
    if state.get("complete"):
        print("ver4.2はすでに完了しています。", flush=True)
        return 0
    models = load_current_models(feature_count, args, device)
    accepted_real = [
        item for item in real_items if item["id"] in state["acceptedRealIds"]
    ]
    accepted_synthetic = load_accepted_synthetic(state)
    total_limit = args.maximum_synthetic

    print(
        "PianoFlow ver4.2 sequential active learning: device={} start={}/{}".format(
            device, state["nextSongIndex"] + 1, len(real_items)
        ),
        flush=True,
    )
    for step_index in range(state["nextSongIndex"], len(real_items)):
        if budget.expired():
            raise TimeoutError("安全な実行時間上限へ到達しました")
        current = real_items[step_index]
        past_report = report_for_models(
            models, accepted_real, mean, scale, device
        )
        pre_report = report_for_models(
            models, [current], mean, scale, device
        )
        remaining = max(0, total_limit - state["totalAcceptedSynthetic"])
        requested = generated_count(step_index, pre_report, remaining)
        plan = v41.weakness_plan(pre_report, requested) if requested else {
            "schemaVersion": SCHEMA_VERSION,
            "requestedSamples": {},
            "maximumSamples": 0,
            "reason": "synthetic total limit reached",
        }
        step_root = WORK / "rounds" / "{:02d}-{}".format(
            step_index + 1, current["id"]
        )
        v41.json_write(step_root / "augmentation-plan.json", plan)
        generated_raw = []
        if requested:
            generated_raw = v41.generate_synthetic(
                accepted_real + [current],
                plan,
                requested,
                args.seed + step_index * 1000,
                budget,
                output_root=step_root / "synthetic",
                source_song=current["name"],
            )
        generated = [enhance_synthetic_item(item) for item in generated_raw]
        common_training = accepted_real + [current] + accepted_synthetic
        branch_seed = args.seed + 10000 + step_index * 20
        branch_a_models, _, _, branch_a_training = v41.train_bundle(
            "step-{:02d}-real-only".format(step_index + 1),
            common_training,
            [current],
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
        branch_b_models, _, _, branch_b_training = v41.train_bundle(
            "step-{:02d}-real-plus-synthetic".format(step_index + 1),
            common_training + generated,
            [current],
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
        current_a = report_for_models(
            branch_a_models, [current], mean, scale, device
        )
        current_b = report_for_models(
            branch_b_models, [current], mean, scale, device
        )
        past_a = report_for_models(
            branch_a_models, accepted_real, mean, scale, device
        )
        past_b = report_for_models(
            branch_b_models, accepted_real, mean, scale, device
        )
        regression_a = regression_check(
            past_report, past_a, args.mean_regression, args.song_regression
        )
        regression_b = regression_check(
            past_report, past_b, args.mean_regression, args.song_regression
        )
        score_a = v41.bundle_score(current_a)
        score_b = v41.bundle_score(current_b)
        b_improves = (
            bool(generated)
            and regression_b["passed"]
            and score_b >= score_a + args.synthetic_minimum_gain
            and current_b["meanNoteAccuracy"]
            >= pre_report["meanNoteAccuracy"] - 1e-6
        )
        a_improves = (
            regression_a["passed"]
            and current_a["meanNoteAccuracy"]
            >= pre_report["meanNoteAccuracy"] - 1e-6
        )
        if b_improves:
            chosen = "real-plus-synthetic"
            models = branch_b_models
            chosen_current = current_b
            chosen_past = past_b
            accepted_synthetic.extend(generated)
            relative = (step_root / "synthetic").relative_to(ROOT).as_posix()
            state["acceptedSyntheticDirectories"].append(relative)
            state["totalAcceptedSynthetic"] += len(generated)
        elif a_improves or models is None:
            chosen = "real-only"
            models = branch_a_models
            chosen_current = current_a
            chosen_past = past_a
        else:
            chosen = "retained-previous"
            chosen_current = pre_report
            chosen_past = past_report
        if chosen != "retained-previous":
            accepted_real.append(current)
            state["acceptedRealIds"].append(current["id"])
            save_current_models(
                models, mean, scale, args, step_index + 1, chosen
            )
        step_report = {
            "step": step_index + 1,
            "song": current["name"],
            "preLearning": pre_report,
            "pastBefore": past_report,
            "requestedSynthetic": requested,
            "generatedSynthetic": len(generated),
            "augmentationPlan": plan,
            "branches": {
                "realOnly": {
                    "training": branch_a_training,
                    "currentSong": current_a,
                    "pastSongs": past_a,
                    "regression": regression_a,
                    "score": round(score_a, 6),
                },
                "realPlusSynthetic": {
                    "training": branch_b_training,
                    "currentSong": current_b,
                    "pastSongs": past_b,
                    "regression": regression_b,
                    "score": round(score_b, 6),
                },
            },
            "chosenBranch": chosen,
            "postLearningCurrent": chosen_current,
            "postLearningPast": chosen_past,
        }
        v41.json_write(step_root / "report.json", step_report)
        state["steps"].append(step_report)
        if chosen == "retained-previous":
            state["blockedSong"] = {
                "index": step_index,
                "name": current["name"],
                "reason": (
                    "Neither branch improved the current song while satisfying "
                    "past-song regression limits."
                ),
                "report": (step_root / "report.json").relative_to(ROOT).as_posix(),
            }
            state["nextSongIndex"] = step_index
            v41.json_write(STATE_PATH, state)
            runtime_write(started, args, state, "needs-more-data")
            prune_training_checkpoints()
            print(
                "NEEDS MORE DATA: {}。次の曲へは進みません。".format(
                    current["name"]
                ),
                flush=True,
            )
            return 76
        state.pop("blockedSong", None)
        state["nextSongIndex"] = step_index + 1
        v41.json_write(STATE_PATH, state)
        runtime_write(started, args, state, "completed-song-{}".format(step_index + 1))
        removed_bytes = prune_training_checkpoints()
        if removed_bytes:
            print(
                "Released completed A/B checkpoints: {:.2f} GB".format(
                    removed_bytes / (1024 ** 3)
                ),
                flush=True,
            )
        print(
            "[{}/{}] {}: pre={:.4f} A={:.4f} B={:.4f} choose={} synthetic={}".format(
                step_index + 1,
                len(real_items),
                current["name"],
                pre_report["meanNoteAccuracy"],
                current_a["meanNoteAccuracy"],
                current_b["meanNoteAccuracy"],
                chosen,
                len(generated),
            ),
            flush=True,
        )

    sequential_report = report_for_models(
        models, real_items, mean, scale, device
    )
    final_training_items = accepted_real + accepted_synthetic
    if not final_training_items:
        raise RuntimeError("最終学習に採用されたデータがありません")
    final_models, _, _, final_training = v41.train_bundle(
        "final-from-scratch",
        final_training_items,
        accepted_real[-1:] if accepted_real else [],
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
    final_report = report_for_models(
        final_models, real_items, mean, scale, device
    )
    final_selected = (
        v41.bundle_score(final_report)
        >= v41.bundle_score(sequential_report) - args.final_allowed_delta
    )
    if final_selected:
        models = final_models
        save_current_models(
            models, mean, scale, args, len(real_items), "final-from-scratch"
        )
    state["complete"] = True
    state["sequentialDiagnosticAllSongs"] = sequential_report
    state["finalFromScratchDiagnosticAllSongs"] = final_report
    state["finalFromScratchSelected"] = final_selected
    v41.json_write(STATE_PATH, state)
    report = {
        "model": "learning-v4.2",
        "mode": "one-song-at-a-time prequential active data growth",
        "complete": True,
        "device": str(device),
        "modelConfiguration": {
            "featureCount": feature_count,
            "v4PriorFeatureCount": V4_PRIOR_FEATURES,
            "channels": args.channels,
            "transformerLayers": args.transformer_layers,
            "epochsPerSong": args.epochs_per_song,
        },
        "firstSongSyntheticRule": {
            "minimum": 150,
            "maximum": 250,
            "reason": "no previously accepted synthetic coverage",
        },
        "maximumSynthetic": total_limit,
        "acceptedSynthetic": state["totalAcceptedSynthetic"],
        "songOrder": state["songOrder"],
        "steps": state["steps"],
        "finalFromScratchTraining": final_training,
        "sequentialDiagnosticAllSongs": sequential_report,
        "finalFromScratchDiagnosticAllSongs": final_report,
        "finalFromScratchSelected": final_selected,
        "warning": (
            "Post-learning and final metrics include songs already used for training. "
            "Use each step's preLearning result or new songs for generalization."
        ),
    }
    v41.json_write(REPORT_PATH, report)
    runtime_write(started, args, state, "complete")
    print("COMPLETE: {}".format(REPORT_PATH), flush=True)
    return 0


def self_test() -> int:
    frame_count = 120
    item = {
        "id": "dummy",
        "name": "dummy",
        "kind": "synthetic",
        "seed": 42,
        "category": "octave",
        "times": np.arange(frame_count, dtype=np.float32) * 0.01,
        "features": np.zeros((frame_count, 134), np.float32),
        "pitchY": np.r_[np.zeros(20, np.int64), np.full(100, 38, np.int64)],
        "pitchWeight": np.ones(frame_count, np.float32),
        "ornamentY": np.zeros(frame_count, np.int64),
        "ornamentWeight": np.ones(frame_count, np.float32),
        "boundaryY": np.zeros((frame_count, 4), np.float32),
        "notes": [
            {"midi": 73, "start": 0.2, "end": 1.1}
        ],
    }
    enhanced = enhance_synthetic_item(item)
    assert enhanced["features"].shape == (frame_count, 134 + V4_PRIOR_FEATURES)
    first_count = generated_count(
        0,
        {"meanNoteAccuracy": 0.4, "meanBoundaryF1": 0.3},
        1000,
    )
    later_count = generated_count(
        1,
        {"meanNoteAccuracy": 0.4, "meanBoundaryF1": 0.3},
        1000,
    )
    assert 150 <= first_count <= 250
    assert 40 <= later_count <= 150
    model = v41.HeavyAcousticModel(
        enhanced["features"].shape[1],
        v41.PITCH_CLASSES,
        channels=64,
        transformer_layers=1,
    )
    output = model(torch.zeros((2, 64, enhanced["features"].shape[1])))
    assert output.shape == (2, 64, v41.PITCH_CLASSES)
    print(
        json.dumps(
            {
                "featureShape": list(enhanced["features"].shape),
                "firstSongSynthetic": first_count,
                "laterSongSynthetic": later_count,
                "modelOutput": list(output.shape),
                "smallModelParameters": v41.parameter_count(model),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PianoFlow v4.2 sequential active dataset growth"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "dataset-work" / "manifest.json",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-hours", type=float, default=71.5)
    parser.add_argument("--epochs-per-song", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=6)
    parser.add_argument("--maximum-synthetic", type=int, default=1000)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument(
        "--transformer-layers", type=int, default=DEFAULT_TRANSFORMER_LAYERS
    )
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--synthetic-minimum-gain", type=float, default=0.005)
    parser.add_argument("--mean-regression", type=float, default=0.01)
    parser.add_argument("--song-regression", type=float, default=0.03)
    parser.add_argument("--final-allowed-delta", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        return run(args)
    except TimeoutError as error:
        state = v41.json_read(STATE_PATH, {"schemaVersion": SCHEMA_VERSION})
        state["complete"] = False
        state["stoppedForTimeBudget"] = True
        state["message"] = str(error)
        v41.json_write(STATE_PATH, state)
        print("TIME BUDGET STOP: {}".format(error), flush=True)
        print("同じBATで自動再開できます。", flush=True)
        return 75
    except KeyboardInterrupt:
        state = v41.json_read(STATE_PATH, {"schemaVersion": SCHEMA_VERSION})
        state["complete"] = False
        state["interrupted"] = True
        v41.json_write(STATE_PATH, state)
        print("中断しました。チェックポイントから再開できます。", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
