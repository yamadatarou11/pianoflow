from __future__ import annotations

import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "models" / "learning-v431"
DEFAULT_DESTINATION = ROOT / "models" / "learning-v431-gpu"


def export(destination: Path, force: bool = False) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    for task in ("pitch", "ornament", "boundary"):
        source = SOURCE / ("current-" + task + ".pt")
        target = destination / source.name
        if not source.is_file():
            raise FileNotFoundError("学習済みモデルがありません: " + str(source))
        if target.is_file() and not force and target.stat().st_mtime >= source.stat().st_mtime:
            print("再利用: " + target.name)
            continue
        print("Torch 1.3互換形式へ変換: " + source.name)
        payload = torch.load(source, map_location="cpu", weights_only=False)
        # PyTorch 1.3 cannot read the zip serialization emitted by current
        # PyTorch. Preserve the trained weights; only the container format changes.
        compatible = {
            key: payload[key]
            for key in (
                "task", "featureCount", "outputCount", "channels",
                "transformerLayers", "bestState", "mean", "scale",
            )
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            torch.save(compatible, str(temporary), _use_new_zipfile_serialization=False)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print("保存: " + target.name)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return export(args.output, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
