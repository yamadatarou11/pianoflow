from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "audio-separator"


def _ensure_ffmpeg() -> None:
    # WindowsApps may expose a non-executable ffmpeg alias. Prefer the real
    # WinGet binary so audio-separator's own `ffmpeg -version` check cannot
    # fail with WinError 5 on managed university PCs.
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = list(
        local.glob(
            "Microsoft/WinGet/Packages/Gyan.FFmpeg_*/**/bin/ffmpeg.exe"
        )
    )
    if candidates:
        os.environ["PATH"] = str(candidates[0].parent) + os.pathsep + os.environ.get("PATH", "")
        return
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg executable was not found")


def _run_separator(
    source: Path,
    output_dir: Path,
    output_names: dict[str, str],
    *,
    preset: str | None = None,
    model: str | None = None,
) -> dict[str, Path]:
    from audio_separator.separator import Separator

    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "output_dir": str(output_dir),
        "model_file_dir": str(MODEL_DIR),
        "output_format": "WAV",
        "sample_rate": 44100,
        "normalization_threshold": 0.95,
        "amplification_threshold": 0.0,
        "use_soundfile": True,
        "log_level": 30,
        # High overlap reduces boundary artifacts. Batch size one keeps memory
        # bounded while the large RoFormer models run on CPU.
        "mdxc_params": {
            "segment_size": 256,
            "override_model_segment_size": False,
            "batch_size": 1,
            "overlap": 8,
            "pitch_shift": 0,
        },
    }
    if preset:
        kwargs["ensemble_preset"] = preset
    separator = Separator(**kwargs)
    separator.load_model(model_filename=model) if model else separator.load_model()
    produced = separator.separate(str(source), output_names)
    result: dict[str, Path] = {}
    for path_string in produced:
        path = Path(path_string)
        if not path.is_absolute():
            path = output_dir / path.name
        lower = path.stem.lower()
        for stem, requested_name in output_names.items():
            if lower == requested_name.lower() or requested_name.lower() in lower:
                result[stem] = path
                break
    del separator
    gc.collect()
    return result


def separate_lead_vocal(source: Path, output_root: Path) -> dict[str, Path]:
    """Maximum-quality lead-vocal path for melody transcription.

    1. Two-model clean-vocal ensemble removes instruments.
    2. Three-model karaoke ensemble estimates backing vocals/harmonies; its
       residual is the lead vocal.
    3. A dedicated RoFormer removes reverb and echo from that lead residual.
    """

    _ensure_ffmpeg()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stage1 = output_root / "hq-stage-1"
    stage2 = output_root / "hq-stage-2"
    stage3 = output_root / "hq-stage-3"

    clean = _run_separator(
        source,
        stage1,
        {"Vocals": "vocals-clean", "Instrumental": "instrumental-clean"},
        preset="vocal_clean",
    )
    vocals = clean.get("Vocals")
    instrumental = clean.get("Instrumental")
    if not vocals or not vocals.exists():
        raise RuntimeError("RoFormer did not create a vocal stem")

    karaoke = _run_separator(
        vocals,
        stage2,
        {"Vocals": "lead-vocal", "Instrumental": "backing-vocals"},
        preset="karaoke",
    )
    lead = karaoke.get("Vocals")
    backing = karaoke.get("Instrumental")
    if not lead or not lead.exists():
        raise RuntimeError("Karaoke ensemble did not create a lead-vocal residual")

    dry = _run_separator(
        lead,
        stage3,
        {"Dry": "lead-vocal-dry", "No Dry": "lead-vocal-reverb"},
        model="dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
    )
    dry_lead = dry.get("Dry")
    reverb = dry.get("No Dry")
    if not dry_lead or not dry_lead.exists():
        raise RuntimeError("De-reverb model did not create a dry lead-vocal stem")

    final_paths = {
        "vocals": vocals,
        "lead": lead,
        "dryLead": dry_lead,
    }
    if instrumental and instrumental.exists():
        final_paths["instrumental"] = instrumental
    if backing and backing.exists():
        final_paths["backing"] = backing
    if reverb and reverb.exists():
        final_paths["reverb"] = reverb
    return final_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = separate_lead_vocal(args.source.resolve(), args.output_dir.resolve())
    print(json.dumps({name: str(path.resolve()) for name, path in paths.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
