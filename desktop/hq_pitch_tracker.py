from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile as sf
import torch
import torchcrepe

from hq_rmvpe import RMVPE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Use an NVIDIA GPU automatically when the installed PyTorch build supports
    # CUDA. Keeping the CPU fallback makes the same project folder usable on
    # machines without a compatible GPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sample_rate = 16000
    hop_length = 160
    audio, source_rate = sf.read(args.source, always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1)
    if source_rate != sample_rate:
        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=sample_rate)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1e-6:
        mono = mono / max(peak, 0.25)

    tensor = torch.from_numpy(np.asarray(mono, dtype=np.float32))[None]
    with torch.inference_mode():
        pitch, periodicity = torchcrepe.predict(
            tensor,
            sample_rate,
            hop_length,
            float(librosa.note_to_hz("C2")),
            float(librosa.note_to_hz("C7")),
            model="full",
            batch_size=512,
            device=device,
            return_periodicity=True,
            decoder=torchcrepe.decode.viterbi,
        )
        periodicity = torchcrepe.filter.median(periodicity, 3)
        pitch = torchcrepe.filter.mean(pitch, 3)

    hz = pitch[0].cpu().numpy().astype(float)
    confidence = periodicity[0].cpu().numpy().astype(float)
    rms = librosa.feature.rms(
        y=np.asarray(mono, dtype=np.float32),
        frame_length=1024,
        hop_length=hop_length,
        center=True,
    )[0]
    count = min(len(hz), len(confidence), len(rms))
    hz, confidence, rms = hz[:count], confidence[:count], rms[:count]
    floor = max(1e-6, float(np.percentile(rms, 20)) * 1.4)
    ceiling = max(floor + 1e-6, float(np.percentile(rms, 90)))
    energy = np.clip((rms - floor) / (ceiling - floor), 0.0, 1.0)
    confidence *= 0.3 + 0.7 * energy
    voiced = (confidence >= 0.16) & (rms >= floor)
    hz = np.where(voiced, hz, np.nan)

    # RMVPE is the primary singing-voice estimator. Unlike a continuity-only
    # octave heuristic, its salience map supplies independent evidence for the
    # fundamental frequency on every 10 ms frame.
    rmvpe_hz = np.full(count, np.nan, dtype=float)
    rmvpe_confidence = np.zeros(count, dtype=float)
    rmvpe_candidate_hz = np.full((count, 3), np.nan, dtype=float)
    rmvpe_candidate_confidence = np.zeros((count, 3), dtype=float)
    model_path = Path(__file__).resolve().parent / "models" / "rmvpe.pt"
    if model_path.exists() and mono.size:
        model = RMVPE(str(model_path), is_half=False, device=device)
        with torch.inference_mode():
            rmvpe_audio = torch.from_numpy(np.asarray(mono, dtype=np.float32))
            mel = model.mel_extractor(rmvpe_audio.unsqueeze(0), center=True)
            hidden = model.mel2hidden(mel).squeeze(0).cpu().numpy()
        raw_rmvpe_hz = model.decode(hidden, thred=0.03)
        raw_rmvpe_confidence = np.max(hidden, axis=1)
        rmvpe_count = min(count, len(raw_rmvpe_hz), len(raw_rmvpe_confidence))
        rmvpe_hz[:rmvpe_count] = raw_rmvpe_hz[:rmvpe_count]
        rmvpe_confidence[:rmvpe_count] = raw_rmvpe_confidence[:rmvpe_count]
        rmvpe_confidence[:rmvpe_count] *= 0.35 + 0.65 * energy[:rmvpe_count]
        rmvpe_voiced = (rmvpe_hz > 0) & (rmvpe_confidence >= 0.055) & (rms >= floor)
        rmvpe_hz[~rmvpe_voiced] = np.nan
        # Preserve the three strongest distinct salience peaks. Downstream note
        # decoding can then recover a fundamental that lost the top-1 vote to
        # an octave harmonic instead of being locked to one RMVPE answer.
        candidate_count = min(count, hidden.shape[0])
        cents_mapping = 20.0 * np.arange(hidden.shape[1]) + 1997.3794084376191
        for frame in range(candidate_count):
            salience = hidden[frame].copy()
            for rank in range(3):
                peak = int(np.argmax(salience))
                peak_confidence = float(salience[peak])
                if peak_confidence < 0.025:
                    break
                left, right = max(0, peak - 4), min(len(salience), peak + 5)
                weights = hidden[frame, left:right]
                cents = float(np.sum(weights * cents_mapping[left:right]) / max(np.sum(weights), 1e-9))
                rmvpe_candidate_hz[frame, rank] = 10.0 * (2.0 ** (cents / 1200.0))
                rmvpe_candidate_confidence[frame, rank] = peak_confidence * (0.35 + 0.65 * energy[min(frame, len(energy) - 1)])
                salience[left:right] = -np.inf

    # WORLD/Harvest is intentionally an independent DSP vote. It is most
    # useful for confirming voicing and rejecting isolated neural octave errors.
    world_hz, world_times = pyworld.harvest(
        np.asarray(mono, dtype=np.float64),
        sample_rate,
        f0_floor=float(librosa.note_to_hz("C2")),
        f0_ceil=float(librosa.note_to_hz("C7")),
        frame_period=1000.0 * hop_length / sample_rate,
    )
    world_hz = pyworld.stonemask(np.asarray(mono, dtype=np.float64), world_hz, world_times, sample_rate)
    world_output = np.full(count, np.nan, dtype=float)
    world_confidence = np.zeros(count, dtype=float)
    world_count = min(count, len(world_hz))
    world_output[:world_count] = world_hz[:world_count]
    world_voiced = (world_output > 0) & (rms >= floor)
    world_confidence[world_voiced] = 0.38 + 0.42 * energy[world_voiced]
    world_output[~world_voiced] = np.nan

    payload = {
        "sampleRate": sample_rate,
        "hopLength": hop_length,
        "hz": [None if not np.isfinite(value) else round(float(value), 4) for value in hz],
        "confidence": [round(float(value), 5) for value in confidence],
        "rmvpeHz": [None if not np.isfinite(value) else round(float(value), 4) for value in rmvpe_hz],
        "rmvpeConfidence": [round(float(value), 5) for value in rmvpe_confidence],
        "rmvpeCandidateHz": [
            [None if not np.isfinite(value) else round(float(value), 4) for value in row]
            for row in rmvpe_candidate_hz
        ],
        "rmvpeCandidateConfidence": [
            [round(float(value), 5) for value in row]
            for row in rmvpe_candidate_confidence
        ],
        "worldHz": [None if not np.isfinite(value) else round(float(value), 4) for value in world_output],
        "worldConfidence": [round(float(value), 5) for value in world_confidence],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
