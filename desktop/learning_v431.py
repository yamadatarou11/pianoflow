from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

import learning_v41 as v41
import learning_v42 as v42
import learning_v43 as v43


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "dataset-work" / "learning-v431"
MODEL_ROOT = ROOT / "models" / "learning-v431"
STATE_PATH = WORK / "pipeline-state.json"
REPORT_PATH = WORK / "report.json"
RUNTIME_PATH = WORK / "runtime.json"
NORMALIZATION_PATH = WORK / "normalization.npz"
CURTAIN_ROOT = ROOT / "dataset-work" / "learning-v43" / "curtain"
SCHEMA_VERSION = 431

BASE_GENERATE = v43.generate_theory_synthetic
BASE_NORMALIZATION = v41.feature_normalization
REALISM_PROFILE: Dict[str, object] = {}


class PriorResidualAcousticModel(nn.Module):
    """Large correction network whose zero correction exactly equals the prior."""

    def __init__(
        self,
        task: str,
        feature_count: int,
        output_count: int,
        channels: int,
        transformer_layers: int,
    ):
        super().__init__()
        if feature_count < 134 + v43.V4_PRIOR_FEATURES:
            raise ValueError("v4.3.1 requires raw acoustic and prior features")
        self.task = task
        self.residual = v41.HeavyAcousticModel(
            feature_count,
            output_count,
            channels,
            transformer_layers,
        )
        # The final residual layer starts at zero. The initial prediction is
        # therefore exactly the incoming v4 prior, not a random large model.
        final = self.residual.head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.gate_logit = nn.Parameter(torch.tensor(-2.20))
        self.maximum_logit_correction = 8.0

    def prior(self, features: torch.Tensor) -> torch.Tensor:
        if self.task == "pitch":
            return features[..., 134 : 134 + v41.PITCH_CLASSES]
        if self.task == "ornament":
            start = 134 + v41.PITCH_CLASSES
            return features[..., start : start + 3]
        start = 134 + v41.PITCH_CLASSES + 3
        return features[..., start : start + 4]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        probability = self.prior(features).clamp(1e-5, 1 - 1e-5)
        if self.task == "boundary":
            base_logits = torch.logit(probability)
        else:
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-7)
            base_logits = torch.log(probability)
        residual = self.residual(features)
        bounded = self.maximum_logit_correction * torch.tanh(
            residual / self.maximum_logit_correction
        )
        gate = torch.sigmoid(self.gate_logit)
        return base_logits + gate * bounded


def model_factory(
    task: str,
    feature_count: int,
    output_count: int,
    channels: int,
    transformer_layers: int,
) -> nn.Module:
    return PriorResidualAcousticModel(
        task,
        feature_count,
        output_count,
        channels,
        transformer_layers,
    )


def prior_preserving_normalization(items: Sequence[dict]):
    mean, scale = BASE_NORMALIZATION(items)
    # Residual models read these columns as probabilities inside forward().
    mean[134:] = 0.0
    scale[134:] = 1.0
    return mean, scale


def acoustic_distribution_score(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> float:
    raw = np.asarray(features[:, :134], dtype=np.float32)
    item_mean = raw.mean(0)
    item_std = raw.std(0)
    mean_distance = np.mean(np.abs((item_mean - mean) / scale))
    spread_distance = np.mean(
        np.abs(np.log((item_std + 0.02) / (scale + 0.02)))
    )
    return float(mean_distance + 0.35 * spread_distance)


def build_realism_profile(real_items: Sequence[dict]) -> dict:
    values = np.concatenate(
        [item["features"][:, :134] for item in real_items], axis=0
    ).astype(np.float32)
    mean = values.mean(0).astype(np.float32)
    scale = np.maximum(values.std(0).astype(np.float32), 0.02)
    real_scores = [
        acoustic_distribution_score(item["features"], mean, scale)
        for item in real_items
    ]
    # Allow natural singer-to-singer variation plus a small synthesis margin.
    threshold = max(real_scores) + 0.08
    return {
        "mean": mean,
        "scale": scale,
        "realScores": real_scores,
        "threshold": float(threshold),
        "maximumRealScore": float(max(real_scores)),
    }


def realism_filtered_generate(
    training: Sequence[dict],
    plan: dict,
    target: int,
    seed: int,
    budget: v41.Budget,
    output_root: Path,
    source_song: str,
    minimum_yield: float,
):
    generated, audit = BASE_GENERATE(
        training,
        plan,
        target,
        seed,
        budget,
        output_root,
        source_song,
        minimum_yield,
    )
    if not REALISM_PROFILE:
        raise RuntimeError("real-vocal acoustic profile was not initialized")
    kept: List[dict] = []
    rejected_scores: List[float] = []
    threshold = float(REALISM_PROFILE["threshold"])
    mean = REALISM_PROFILE["mean"]
    scale = REALISM_PROFILE["scale"]
    root_resolved = output_root.resolve()
    for item in generated:
        score = acoustic_distribution_score(item["features"], mean, scale)
        item["realVocalDistributionScore"] = round(score, 5)
        item["realVocalDistributionThreshold"] = round(threshold, 5)
        directory = Path(item["audio"]).resolve().parent
        if score <= threshold:
            metadata_path = directory / "metadata.json"
            if metadata_path.exists():
                metadata = v43.json_read(metadata_path)
                metadata["realVocalDistribution"] = {
                    "accepted": True,
                    "score": round(score, 5),
                    "threshold": round(threshold, 5),
                }
                v43.json_write(metadata_path, metadata)
            kept.append(item)
        else:
            rejected_scores.append(score)
            try:
                directory.relative_to(root_resolved)
            except ValueError:
                raise RuntimeError(
                    "refusing to remove generated data outside its round directory"
                )
            shutil.rmtree(directory)
    attempts = max(1, int(audit.get("attemptsThisRun", len(generated))))
    yield_ratio = len(kept) / attempts
    audit = {
        **audit,
        "acceptedBeforeRealVocalGate": len(generated),
        "accepted": len(kept),
        "yieldRatio": round(yield_ratio, 4),
        "qualityGatePassed": bool(kept and yield_ratio >= minimum_yield),
        "realVocalDistributionGate": {
            "threshold": round(threshold, 5),
            "maximumRealSongScore": round(
                float(REALISM_PROFILE["maximumRealScore"]), 5
            ),
            "rejected": len(rejected_scores),
            "rejectedMedianScore": (
                None
                if not rejected_scores
                else round(float(np.median(rejected_scores)), 5)
            ),
        },
    }
    v43.json_write(output_root / "quality-report.json", audit)
    return kept, audit


def configure(args) -> None:
    global REALISM_PROFILE
    v43.WORK = WORK
    v43.MODEL_ROOT = MODEL_ROOT
    v43.STATE_PATH = STATE_PATH
    v43.REPORT_PATH = REPORT_PATH
    v43.RUNTIME_PATH = RUNTIME_PATH
    v43.NORMALIZATION_PATH = NORMALIZATION_PATH
    v43.CURTAIN_ROOT = CURTAIN_ROOT
    v43.SCHEMA_VERSION = SCHEMA_VERSION
    v43.MODEL_NAME = "learning-v4.3.1"
    v43.MODEL_MODE = (
        "prior-preserving residual B-first theory chorus active learning"
    )
    v43.FORCE_INITIAL_B = False
    v43.ALLOW_DEFERRED_ADVANCE = True
    v43.generate_theory_synthetic = realism_filtered_generate
    v41.MODEL_FACTORY = model_factory
    v41.TRAINING_SAMPLES_PER_EPOCH = int(args.samples_per_epoch)
    v41.feature_normalization = prior_preserving_normalization
    real_items = v42.load_real_items(args.manifest, args.rebuild_features)
    REALISM_PROFILE = build_realism_profile(real_items)
    WORK.mkdir(parents=True, exist_ok=True)
    v43.json_write(
        WORK / "real-vocal-distribution-profile.json",
        {
            "schemaVersion": SCHEMA_VERSION,
            "songCount": len(real_items),
            "realScores": [
                round(float(value), 5)
                for value in REALISM_PROFILE["realScores"]
            ],
            "maximumRealScore": round(
                float(REALISM_PROFILE["maximumRealScore"]), 5
            ),
            "syntheticAcceptanceThreshold": round(
                float(REALISM_PROFILE["threshold"]), 5
            ),
            "curtainCallExcluded": True,
        },
    )


def self_test() -> int:
    frame_count = 80
    feature_count = 134 + v43.V4_PRIOR_FEATURES
    features = torch.zeros((2, frame_count, feature_count), dtype=torch.float32)
    pitch = torch.rand((2, frame_count, v41.PITCH_CLASSES))
    pitch /= pitch.sum(-1, keepdim=True)
    ornament = torch.rand((2, frame_count, 3))
    ornament /= ornament.sum(-1, keepdim=True)
    boundary = 0.02 + 0.96 * torch.rand((2, frame_count, 4))
    features[..., 134 : 134 + v41.PITCH_CLASSES] = pitch
    features[
        ..., 134 + v41.PITCH_CLASSES : 134 + v41.PITCH_CLASSES + 3
    ] = ornament
    features[..., 134 + v41.PITCH_CLASSES + 3 :] = boundary
    maximum_errors = {}
    parameter_counts = {}
    for task, expected in (
        ("pitch", pitch),
        ("ornament", ornament),
        ("boundary", boundary),
    ):
        model = PriorResidualAcousticModel(
            task,
            feature_count,
            v41.task_output_count(task),
            channels=64,
            transformer_layers=1,
        )
        model.eval()
        with torch.no_grad():
            logits = model(features)
            actual = (
                torch.sigmoid(logits)
                if task == "boundary"
                else torch.softmax(logits, -1)
            )
        if task != "boundary":
            expected = expected.clamp(1e-5, 1 - 1e-5)
            expected = expected / expected.sum(-1, keepdim=True)
        error = float(torch.max(torch.abs(actual - expected)))
        maximum_errors[task] = error
        parameter_counts[task] = v41.parameter_count(model)
        assert error < 2e-6, (task, error)
    print(
        json.dumps(
            {
                "initialOutputMaximumError": maximum_errors,
                "smallResidualModelParameters": parameter_counts,
                "samplesPerEpochDefault": 768,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PianoFlow v4.3.1 prior-preserving residual learning"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "dataset-work" / "manifest.json",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-hours", type=float, default=71.5)
    parser.add_argument("--epochs-per-song", type=int, default=6)
    parser.add_argument("--final-epochs", type=int, default=10)
    parser.add_argument("--maximum-synthetic", type=int, default=900)
    parser.add_argument("--channels", type=int, default=384)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-epoch", type=int, default=768)
    parser.add_argument("--minimum-quality-yield", type=float, default=0.20)
    parser.add_argument("--minimum-free-gb", type=float, default=2.0)
    parser.add_argument("--current-pitch-tolerance", type=float, default=0.005)
    parser.add_argument("--current-ornament-tolerance", type=float, default=0.02)
    parser.add_argument("--current-boundary-tolerance", type=float, default=0.02)
    parser.add_argument("--mean-regression", type=float, default=0.01)
    parser.add_argument("--song-regression", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=4311)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    configure(args)
    try:
        return v43.run(args)
    except TimeoutError as error:
        state = v43.json_read(STATE_PATH, {})
        state["complete"] = False
        state["stoppedForTimeBudget"] = True
        state["message"] = str(error)
        v43.json_write(STATE_PATH, state)
        print("TIME BUDGET: {}".format(error), flush=True)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
