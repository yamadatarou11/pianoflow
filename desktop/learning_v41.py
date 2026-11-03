from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import struct
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "dataset-work" / "learning-v41"
MODEL_ROOT = ROOT / "models" / "learning-v41"
STATE_PATH = WORK / "pipeline-state.json"
REPORT_PATH = WORK / "report.json"
RUNTIME_PATH = WORK / "runtime.json"

SCHEMA_VERSION = 41
SAMPLE_RATE = 22050
HOP_LENGTH = 220
MIDI_MIN = 24
MIDI_MAX = 108
PITCH_CLASSES = MIDI_MAX - MIDI_MIN + 2  # 0=rest, 1..85=MIDI 24..108
ORNAMENT_NAMES = ("stable", "scoop_or_fall", "transition")
BOUNDARY_NAMES = ("onset", "offset", "samePitch", "pitchChange")
TASKS = ("pitch", "ornament", "boundary")
# v4.1 keeps one previous optimizer checkpoint for crash recovery. Sequential
# v4.2 disables this because A/B branches can restart from accepted models.
KEEP_CHECKPOINT_PREVIOUS = True
# Newer learning experiments can replace the acoustic architecture and increase
# the amount of training without duplicating the stable v4.1 training loop.
MODEL_FACTORY = None
TRAINING_SAMPLES_PER_EPOCH = 192
DEFAULT_VALIDATION = "ウタカタララバイ"
DEFAULT_TEST = "逆夢"


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


def duration_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return "{}時間{}分{}秒".format(hours, minutes, seconds)
    if minutes:
        return "{}分{}秒".format(minutes, seconds)
    return "{}秒".format(seconds)


@dataclass
class Budget:
    started: float
    deadline: float
    reserve_seconds: float = 900.0

    @classmethod
    def create(cls, hours: float) -> "Budget":
        now = time.time()
        return cls(now, now + max(0.25, hours) * 3600.0)

    def remaining(self) -> float:
        return self.deadline - time.time()

    def can_start(self, estimated_seconds: float = 0.0) -> bool:
        return self.remaining() > self.reserve_seconds + estimated_seconds

    def expired(self) -> bool:
        return self.remaining() <= self.reserve_seconds


def gpu_temperature() -> Optional[float]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return max(float(value) for value in completed.stdout.split())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def thermal_guard(limit: float = 85.0) -> None:
    temperature = gpu_temperature()
    if temperature is None or temperature < limit:
        return
    print(
        "GPU温度 {:.0f}℃: チェックポイントを保持して冷却を待ちます。".format(
            temperature
        ),
        flush=True,
    )
    for _ in range(10):
        time.sleep(15)
        temperature = gpu_temperature()
        if temperature is None or temperature < limit - 3:
            return
    raise RuntimeError("GPU温度が安全範囲まで下がらないため停止しました")


def choose_device(request: str) -> torch.device:
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA版PyTorchが利用できません。--device auto または cpu を使用してください。"
        )
    if request == "cuda" or (request == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def extract_features(audio_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if not len(audio):
        raise ValueError("音声が空です: {}".format(audio_path))
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=HOP_LENGTH,
        n_mels=64,
        fmin=55,
        fmax=9000,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T / 80.0 + 1.0
    delta = librosa.feature.delta(log_mel.T).T
    rms = librosa.feature.rms(
        y=audio, frame_length=1024, hop_length=HOP_LENGTH
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=1024, hop_length=HOP_LENGTH
    )[0]
    centroid = (
        librosa.feature.spectral_centroid(
            y=audio, sr=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH
        )[0]
        / (SAMPLE_RATE / 2)
    )
    onset = librosa.onset.onset_strength(
        y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    count = min(len(log_mel), len(rms), len(zcr), len(centroid), len(onset))
    rms = rms[:count] / max(1e-6, float(np.percentile(rms[:count], 95)))
    rms_delta = np.r_[0.0, np.diff(rms)]
    auxiliary = np.column_stack(
        (
            np.clip(rms, 0, 2),
            np.clip(np.maximum(0, rms_delta) * 8, 0, 2),
            np.clip(np.maximum(0, -rms_delta) * 8, 0, 2),
            np.clip(zcr[:count] * 4, 0, 2),
            np.clip(centroid[:count], 0, 1),
            np.clip(onset[:count] / max(1e-6, np.percentile(onset[:count], 95)), 0, 2),
        )
    )
    features = np.column_stack(
        (log_mel[:count], delta[:count], auxiliary)
    ).astype(np.float32)
    times = librosa.frames_to_time(
        np.arange(count), sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    ).astype(np.float32)
    return features, times


def note_labels(
    notes: Sequence[dict],
    times: np.ndarray,
    weak_ornament: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    count = len(times)
    pitch = np.zeros(count, np.int64)
    pitch_weight = np.full(count, 0.20, np.float32)
    ornament = np.full(count, -1, np.int64)
    ornament_weight = np.zeros(count, np.float32)
    boundary = np.zeros((count, 4), np.float32)
    for index, note in enumerate(notes):
        start, end = float(note["start"]), float(note["end"])
        frames = np.flatnonzero((times >= start) & (times < end))
        if not len(frames):
            continue
        vocal_midi = int(note["midi"]) - 12
        vocal_midi = max(MIDI_MIN, min(MIDI_MAX, vocal_midi))
        pitch[frames] = vocal_midi - MIDI_MIN + 1
        relative = np.linspace(0, 1, len(frames), dtype=np.float32)
        central = np.minimum(relative / 0.15, (1 - relative) / 0.15)
        pitch_weight[frames] = np.clip(central, 0.25, 1.0)
        onset_index = int(np.argmin(np.abs(times - start)))
        offset_index = int(np.argmin(np.abs(times - end)))
        boundary[onset_index, 0] = 1
        boundary[offset_index, 1] = 1
        if index:
            column = (
                2
                if int(note["midi"]) == int(notes[index - 1]["midi"])
                else 3
            )
            boundary[onset_index, column] = 1
    if weak_ornament is not None:
        source_times, source_labels = weak_ornament
        nearest = np.searchsorted(source_times, times)
        nearest = np.clip(nearest, 0, len(source_times) - 1)
        previous = np.maximum(0, nearest - 1)
        use_previous = (
            np.abs(source_times[previous] - times)
            < np.abs(source_times[nearest] - times)
        )
        nearest[use_previous] = previous[use_previous]
        copied = source_labels[nearest]
        valid = copied >= 0
        ornament[valid] = copied[valid]
        ornament_weight[valid] = 0.45
    return {
        "pitchY": pitch,
        "pitchWeight": pitch_weight,
        "ornamentY": ornament,
        "ornamentWeight": ornament_weight,
        "boundaryY": boundary,
    }


def build_real_dataset(manifest_path: Path, force: bool = False) -> List[dict]:
    output_root = WORK / "real"
    manifest = json_read(manifest_path)
    rows = []
    for song in manifest["songs"]:
        output = output_root / song["id"]
        npz_path = output / "data.npz"
        metadata_path = output / "metadata.json"
        if npz_path.exists() and metadata_path.exists() and not force:
            rows.append(json_read(metadata_path))
            continue
        base = (ROOT / song["referenceFile"]).parent / "audio-only"
        alignment = json_read(base / "initial-alignment.json")
        analysis = json_read(base / "analysis.json")
        dry_name = analysis.get("stemFiles", {}).get(
            "dryLead", "lead-vocal-dry.wav"
        )
        audio_path = base / dry_name
        features, times = extract_features(audio_path)
        weak = None
        v4_npz = ROOT / "dataset-work" / "learning-v4" / song["id"] / "learning-v4-data.npz"
        if v4_npz.exists():
            v4 = np.load(v4_npz)
            weak = (
                v4["frameTimes"].astype(np.float32),
                v4["ornamentLabels"].astype(np.int64),
            )
        notes = alignment["alignedNotes"]
        labels = note_labels(notes, times, weak)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            features=features,
            times=times,
            **labels
        )
        metadata = {
            "schemaVersion": SCHEMA_VERSION,
            "id": song["id"],
            "name": song["name"],
            "kind": "real",
            "audio": str(audio_path),
            "bpm": song.get("bpm"),
            "notes": notes,
            "featureCount": int(features.shape[1]),
            "frameCount": int(len(times)),
        }
        json_write(metadata_path, metadata)
        rows.append(metadata)
        print("実音源特徴: {} ({} frames)".format(song["name"], len(times)), flush=True)
    json_write(
        output_root / "build-report.json",
        {"schemaVersion": SCHEMA_VERSION, "songs": rows},
    )
    return rows


def load_items(root: Path) -> List[dict]:
    items = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        metadata = json_read(metadata_path)
        arrays = np.load(metadata_path.with_name("data.npz"))
        items.append(
            {
                **metadata,
                "features": arrays["features"].astype(np.float32),
                "times": arrays["times"].astype(np.float32),
                "pitchY": arrays["pitchY"].astype(np.int64),
                "pitchWeight": arrays["pitchWeight"].astype(np.float32),
                "ornamentY": arrays["ornamentY"].astype(np.int64),
                "ornamentWeight": arrays["ornamentWeight"].astype(np.float32),
                "boundaryY": arrays["boundaryY"].astype(np.float32),
            }
        )
    return items


def variable_length(value: int) -> bytes:
    buffer = value & 0x7F
    result = bytearray()
    while value >> 7:
        value >>= 7
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80
    while True:
        result.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(result)


def write_midi(path: Path, notes: Sequence[dict], bpm: float) -> None:
    ticks = 480
    seconds_per_tick = 60.0 / bpm / ticks
    events = []
    for note in notes:
        start_tick = max(0, int(round(float(note["start"]) / seconds_per_tick)))
        end_tick = max(start_tick + 1, int(round(float(note["end"]) / seconds_per_tick)))
        midi = int(note["midi"])
        events.append((start_tick, 1, bytes((0x90, midi, 96))))
        events.append((end_tick, 0, bytes((0x80, midi, 0))))
    events.sort(key=lambda row: (row[0], row[1]))
    tempo = int(round(60_000_000 / bpm))
    track = bytearray()
    track.extend(variable_length(0))
    track.extend(b"\xff\x51\x03" + tempo.to_bytes(3, "big"))
    track.extend(variable_length(0))
    track.extend(b"\xff\x58\x04\x04\x02\x18\x08")
    previous = 0
    for tick, _, payload in events:
        track.extend(variable_length(tick - previous))
        track.extend(payload)
        previous = tick
    track.extend(b"\x00\xff\x2f\x00")
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks)
    path.write_bytes(header + b"MTrk" + struct.pack(">I", len(track)) + bytes(track))


def synthesize_vocal(
    path: Path,
    notes: Sequence[dict],
    ornaments: Sequence[dict],
    seed: int,
) -> None:
    rng = np.random.RandomState(seed)
    duration = max((float(note["end"]) for note in notes), default=1.0) + 0.25
    count = int(math.ceil(duration * SAMPLE_RATE))
    output = np.zeros(count, np.float64)
    phase = 0.0
    formant_sets = (
        (800.0, 1200.0, 2500.0),
        (500.0, 1700.0, 2500.0),
        (350.0, 900.0, 2200.0),
        (450.0, 800.0, 2800.0),
        (550.0, 1000.0, 2400.0),
    )
    ornament_by_index = {int(row["noteIndex"]): row for row in ornaments}
    for index, note in enumerate(notes):
        start = int(round(float(note["start"]) * SAMPLE_RATE))
        end = min(count, int(round(float(note["end"]) * SAMPLE_RATE)))
        if end <= start:
            continue
        local_count = end - start
        local_time = np.arange(local_count, dtype=np.float64) / SAMPLE_RATE
        base_midi = float(note["midi"] - 12)
        ornament = ornament_by_index.get(index, {"type": "stable"})
        curve = np.full(local_count, base_midi, np.float64)
        ornament_type = ornament.get("type", "stable")
        if ornament_type == "scoop":
            length = min(local_count, int(0.12 * SAMPLE_RATE))
            curve[:length] -= np.linspace(2.5, 0.0, length)
        elif ornament_type == "fall":
            length = min(local_count, int(0.13 * SAMPLE_RATE))
            curve[-length:] -= np.linspace(0.0, 2.8, length)
        elif ornament_type == "transition":
            length = min(local_count, int(0.10 * SAMPLE_RATE))
            curve[-length:] += np.linspace(0.0, float(ornament.get("shift", 1.5)), length)
        vibrato_depth = float(ornament.get("vibratoDepth", 0.08))
        curve += vibrato_depth * np.sin(2 * np.pi * 5.2 * local_time)
        frequency = 440.0 * np.power(2.0, (curve - 69.0) / 12.0)
        phases = phase + 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
        phase = float(phases[-1] % (2 * np.pi))
        formants = formant_sets[index % len(formant_sets)]
        source = np.zeros(local_count, np.float64)
        for harmonic in range(1, 19):
            harmonic_frequency = np.median(frequency) * harmonic
            envelope = 0.10
            for formant in formants:
                envelope += math.exp(
                    -0.5 * ((harmonic_frequency - formant) / 180.0) ** 2
                )
            source += envelope * np.sin(phases * harmonic) / (harmonic ** 1.22)
        attack = min(local_count, int(0.035 * SAMPLE_RATE))
        release = min(local_count, int(0.055 * SAMPLE_RATE))
        amplitude = np.ones(local_count)
        if attack:
            amplitude[:attack] *= np.linspace(0, 1, attack)
        if release:
            amplitude[-release:] *= np.linspace(1, 0, release)
        breath = rng.normal(0, 0.015, local_count)
        output[start:end] += (0.18 * source + breath) * amplitude
    peak = max(1e-7, float(np.max(np.abs(output))))
    output = np.clip(output / peak * 0.82, -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((output * 32767).astype("<i2").tobytes())


class MarkovComposer:
    def __init__(self, training: Sequence[dict], seed: int):
        self.rng = random.Random(seed)
        self.intervals = []
        self.duration_beats = []
        self.pitches = []
        for item in training:
            bpm = float(item.get("bpm") or 120)
            notes = item.get("notes", [])
            self.pitches.extend(int(note["midi"]) - 12 for note in notes)
            for first, second in zip(notes, notes[1:]):
                self.intervals.append(
                    max(-12, min(12, int(second["midi"]) - int(first["midi"])))
                )
            self.duration_beats.extend(
                max(
                    0.125,
                    min(
                        4.0,
                        (float(note["end"]) - float(note["start"])) * bpm / 60.0,
                    ),
                )
                for note in notes
            )
        if not self.intervals:
            self.intervals = [-2, -1, 0, 0, 1, 2]
        if not self.duration_beats:
            self.duration_beats = [0.25, 0.5, 1.0]
        if not self.pitches:
            self.pitches = [60, 62, 64, 65, 67, 69]

    def compose(self, category: str, index: int) -> Tuple[List[dict], float, List[dict]]:
        bpm = self.rng.choice((80, 90, 100, 120, 132, 140, 160, 180, 208))
        note_count = self.rng.randint(12, 26)
        pitch = self.rng.choice(self.pitches)
        time_cursor = 0.20
        notes = []
        ornaments = []
        for note_index in range(note_count):
            interval = self.rng.choice(self.intervals)
            if category == "octave" and note_index % 7 == 4:
                interval = self.rng.choice((-12, 12))
            if category == "same_pitch" and note_index % 3:
                interval = 0
            pitch = max(MIDI_MIN, min(MIDI_MAX, pitch + interval))
            beat = self.rng.choice(self.duration_beats)
            if category == "short_note":
                beat = self.rng.choice((0.2, 0.25, 1 / 3, 0.5))
            elif category == "tuplet":
                beat = self.rng.choice((1 / 3, 0.2, 1 / 7))
            beat = max(0.12, min(1.5, beat))
            seconds = beat * 60.0 / bpm
            gap = min(0.018, seconds * 0.08)
            notes.append(
                {
                    "midi": int(pitch + 12),
                    "start": round(time_cursor, 6),
                    "end": round(time_cursor + seconds - gap, 6),
                }
            )
            ornament_type = "stable"
            if category == "ornament" and note_index % 4 == 1:
                ornament_type = self.rng.choice(("scoop", "fall", "transition"))
            ornaments.append(
                {
                    "noteIndex": note_index,
                    "type": ornament_type,
                    "shift": self.rng.choice((-2.0, -1.5, 1.5, 2.0)),
                    "vibratoDepth": self.rng.uniform(0.04, 0.18),
                }
            )
            time_cursor += seconds
        return notes, float(bpm), ornaments


def synthetic_labels(
    notes: Sequence[dict],
    ornaments: Sequence[dict],
    times: np.ndarray,
) -> Dict[str, np.ndarray]:
    result = note_labels(notes, times)
    result["ornamentY"][:] = -1
    result["ornamentWeight"][:] = 0
    for ornament in ornaments:
        index = int(ornament["noteIndex"])
        note = notes[index]
        frames = np.flatnonzero(
            (times >= float(note["start"])) & (times < float(note["end"]))
        )
        if not len(frames):
            continue
        ornament_type = ornament["type"]
        result["ornamentY"][frames] = 0
        result["ornamentWeight"][frames] = 1.0
        edge = max(1, min(len(frames) // 3, int(0.13 * SAMPLE_RATE / HOP_LENGTH)))
        if ornament_type == "scoop":
            result["ornamentY"][frames[:edge]] = 1
        elif ornament_type == "fall":
            result["ornamentY"][frames[-edge:]] = 1
        elif ornament_type == "transition":
            result["ornamentY"][frames[-edge:]] = 2
    return result


def quality_check(
    features: np.ndarray,
    labels: Dict[str, np.ndarray],
    notes: Sequence[dict],
) -> Tuple[bool, List[str]]:
    reasons = []
    if not len(features) or not np.all(np.isfinite(features)):
        reasons.append("特徴量が空または非有限")
    if len(notes) < 8:
        reasons.append("音符数不足")
    voiced = np.mean(labels["pitchY"] > 0) if len(labels["pitchY"]) else 0
    if voiced < 0.35:
        reasons.append("有声音比率不足")
    if features.shape[1] != 134:
        reasons.append("特徴量次元不一致")
    return not reasons, reasons


def generate_synthetic(
    training: Sequence[dict],
    plan: dict,
    maximum: int,
    seed: int,
    budget: Budget,
    output_root: Optional[Path] = None,
    source_song: Optional[str] = None,
) -> List[dict]:
    output_root = output_root or (WORK / "synthetic")
    output_root.mkdir(parents=True, exist_ok=True)
    existing = load_items(output_root)
    if len(existing) >= maximum:
        return existing[:maximum]
    categories = []
    for category, count in plan.get("requestedSamples", {}).items():
        categories.extend([category] * int(count))
    if not categories:
        categories = ["short_note", "same_pitch", "ornament", "tuplet", "octave"]
    composer = MarkovComposer(training, seed)
    rng = random.Random(seed)
    while len(existing) < maximum and budget.can_start(30):
        sample_index = len(existing)
        category = categories[sample_index % len(categories)]
        notes, bpm, ornaments = composer.compose(category, sample_index)
        sample_id = "synthetic-{:04d}".format(sample_index + 1)
        output = output_root / sample_id
        output.mkdir(parents=True, exist_ok=True)
        midi_path = output / "melody.mid"
        audio_path = output / "vocal.wav"
        write_midi(midi_path, notes, bpm)
        synthesize_vocal(audio_path, notes, ornaments, seed + sample_index)
        features, times = extract_features(audio_path)
        labels = synthetic_labels(notes, ornaments, times)
        accepted, reasons = quality_check(features, labels, notes)
        metadata = {
            "schemaVersion": SCHEMA_VERSION,
            "id": sample_id,
            "name": sample_id,
            "kind": "synthetic",
            "generator": "PianoFlow Markov melody + deterministic source-filter vocal pilot",
            "sourceSong": source_song,
            "category": category,
            "bpm": bpm,
            "audio": str(audio_path),
            "midi": str(midi_path),
            "notes": notes,
            "ornaments": ornaments,
            "lyrics": ["あ", "い", "う", "え", "お"],
            "qualityAccepted": accepted,
            "qualityReasons": reasons,
            "featureCount": int(features.shape[1]),
            "frameCount": int(len(times)),
            "seed": seed + sample_index,
        }
        json_write(output / "metadata.json", metadata)
        if accepted:
            np.savez_compressed(
                output / "data.npz",
                features=features,
                times=times,
                **labels
            )
            existing.append(
                {
                    **metadata,
                    "features": features,
                    "times": times,
                    **labels,
                }
            )
        else:
            shutil.rmtree(output)
        if len(existing) % 10 == 0:
            print(
                "合成データ: {}/{}（{}）".format(len(existing), maximum, category),
                flush=True,
            )
        rng.random()
    return existing


class TrainingWindows(Dataset):
    def __init__(
        self,
        items: Sequence[dict],
        task: str,
        mean: np.ndarray,
        scale: np.ndarray,
        window: int,
        samples_per_epoch: int,
        seed: int,
    ):
        self.items = list(items)
        self.task = task
        self.mean = mean
        self.scale = scale
        self.window = window
        real_indices = [
            index for index, value in enumerate(items)
            if value.get("kind") != "synthetic"
        ]
        synthetic_indices = [
            index for index, value in enumerate(items)
            if value.get("kind") == "synthetic"
        ]
        chooser = random.Random(seed + 1701)
        self.index = []
        for sample in range(samples_per_epoch):
            use_synthetic = bool(synthetic_indices) and chooser.random() < 0.35
            pool = synthetic_indices if use_synthetic else real_indices
            if not pool:
                pool = list(range(len(items)))
            self.index.append((pool[chooser.randrange(len(pool))], sample))
        self.seed = seed

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item_index):
        source_index, repeat = self.index[item_index]
        item = self.items[source_index]
        rng = random.Random(self.seed + item_index * 997 + repeat)
        features = item["features"]
        length = len(features)
        if self.task == "pitch":
            interesting = np.flatnonzero(item["pitchY"] > 0)
        elif self.task == "ornament":
            interesting = np.flatnonzero(item["ornamentY"] > 0)
        else:
            interesting = np.flatnonzero(item["boundaryY"].max(1) > 0)
        if length <= self.window:
            start = 0
        elif len(interesting) and rng.random() < 0.8:
            center = int(interesting[rng.randrange(len(interesting))])
            start = max(0, min(length - self.window, center - self.window // 2))
        else:
            start = rng.randrange(length - self.window)
        end = min(length, start + self.window)
        count = end - start
        x = np.zeros((self.window, features.shape[1]), np.float32)
        x[:count] = (features[start:end] - self.mean) / self.scale
        if self.task == "boundary":
            y = np.full((self.window, 4), -1, np.float32)
            weight = np.zeros((self.window, 4), np.float32)
            y[:count] = item["boundaryY"][start:end]
            boundary_weight = item.get("boundaryWeight")
            if boundary_weight is None:
                weight[:count] = 1
            else:
                selected_weight = np.asarray(
                    boundary_weight[start:end], dtype=np.float32
                )
                if selected_weight.ndim == 1:
                    selected_weight = selected_weight[:, None]
                weight[:count] = np.broadcast_to(
                    selected_weight, (count, 4)
                )
        else:
            key = "pitchY" if self.task == "pitch" else "ornamentY"
            weight_key = (
                "pitchWeight" if self.task == "pitch" else "ornamentWeight"
            )
            y = np.full(self.window, -1, np.int64)
            weight = np.zeros(self.window, np.float32)
            y[:count] = item[key][start:end]
            weight[:count] = item[weight_key][start:end]
        if rng.random() < 0.8:
            noise = np.random.RandomState(self.seed + item_index).normal(
                0, 0.008, x[:count].shape
            )
            x[:count] += noise.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weight)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels * 3, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 3, channels, 1),
        )

    def forward(self, x):
        return x + self.pointwise(self.depthwise(self.norm(x)))


class HeavyAcousticModel(nn.Module):
    def __init__(
        self,
        feature_count: int,
        output_count: int,
        channels: int = 256,
        transformer_layers: int = 4,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_count, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            *[
                TemporalBlock(channels, dilation, 0.18)
                for dilation in (1, 2, 4, 8, 16, 32, 1, 2)
            ]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dim_feedforward=channels * 4,
            dropout=0.16,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, transformer_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(channels, output_count),
        )

    def forward(self, features):
        encoded = self.input(features)
        encoded = self.temporal(encoded.transpose(1, 2)).transpose(1, 2)
        return self.head(self.transformer(encoded))


def create_model(
    task: str,
    feature_count: int,
    output_count: int,
    channels: int,
    transformer_layers: int,
) -> nn.Module:
    if MODEL_FACTORY is not None:
        return MODEL_FACTORY(
            task,
            feature_count,
            output_count,
            channels,
            transformer_layers,
        )
    return HeavyAcousticModel(
        feature_count,
        output_count,
        channels,
        transformer_layers,
    )


def feature_normalization(items: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([item["features"] for item in items], axis=0)
    mean = values.mean(0).astype(np.float32)
    scale = values.std(0).astype(np.float32)
    scale[scale < 1e-5] = 1
    return mean, scale


def task_output_count(task: str) -> int:
    if task == "pitch":
        return PITCH_CLASSES
    if task == "ornament":
        return len(ORNAMENT_NAMES)
    return len(BOUNDARY_NAMES)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def class_weights(task: str, items: Sequence[dict], device: torch.device):
    if task == "ornament":
        labels = np.concatenate([item["ornamentY"] for item in items])
        labels = labels[labels >= 0]
        counts = np.bincount(labels, minlength=3).astype(float)
        weights = np.sqrt(max(1, counts.max()) / np.maximum(1, counts))
        return torch.tensor(np.clip(weights, 1, 8), dtype=torch.float32, device=device)
    if task == "boundary":
        labels = np.concatenate([item["boundaryY"] for item in items])
        positive = np.sum(labels > 0, axis=0)
        negative = np.sum(labels == 0, axis=0)
        values = np.clip((negative / np.maximum(1, positive)) ** 0.55, 1, 20)
        return torch.tensor(values, dtype=torch.float32, device=device)
    labels = np.concatenate([item["pitchY"] for item in items])
    counts = np.bincount(labels, minlength=PITCH_CLASSES).astype(float)
    nonzero = counts[counts > 0]
    maximum = max(1, np.percentile(nonzero, 75) if len(nonzero) else 1)
    values = np.sqrt(maximum / np.maximum(1, counts))
    values[0] *= 0.45
    return torch.tensor(np.clip(values, 0.25, 5), dtype=torch.float32, device=device)


def loss_for_task(task, logits, target, weight, weights):
    if task in ("pitch", "ornament"):
        valid = target >= 0
        base = nn.functional.cross_entropy(
            logits.transpose(1, 2),
            target.clamp_min(0),
            weight=weights,
            reduction="none",
            label_smoothing=0.015,
        )
    else:
        valid = target >= 0
        base = nn.functional.binary_cross_entropy_with_logits(
            logits,
            target.clamp(0, 1),
            pos_weight=weights,
            reduction="none",
        )
        probability = torch.sigmoid(logits)
        agreement = probability * target.clamp(0, 1) + (1 - probability) * (
            1 - target.clamp(0, 1)
        )
        base = base * (1 - agreement).pow(1.2)
    weighted = base * weight * valid
    return weighted.sum() / (weight * valid).sum().clamp_min(1)


def predict_item(
    model: nn.Module,
    item: dict,
    mean: np.ndarray,
    scale: np.ndarray,
    task: str,
    device: torch.device,
    chunk: int = 1400,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(item["features"]), chunk):
            values = (item["features"][start : start + chunk] - mean) / scale
            tensor = torch.from_numpy(values[None]).to(device)
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(tensor)
            probability = (
                torch.sigmoid(logits)
                if task == "boundary"
                else torch.softmax(logits, -1)
            )
            outputs.append(probability[0].cpu().numpy())
    return np.concatenate(outputs, axis=0)


def event_f1(labels: np.ndarray, probabilities: np.ndarray, tolerance: int) -> dict:
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    truth = np.flatnonzero(labels > 0)
    for threshold in np.arange(0.2, 0.81, 0.05):
        candidates = np.flatnonzero(probabilities >= threshold)
        selected = []
        for candidate in candidates:
            if (
                candidate == 0
                or candidate == len(probabilities) - 1
                or probabilities[candidate] >= probabilities[candidate - 1]
                and probabilities[candidate] >= probabilities[candidate + 1]
            ):
                selected.append(int(candidate))
        used = set()
        matched = 0
        for prediction in selected:
            possible = [
                (abs(prediction - int(value)), index)
                for index, value in enumerate(truth)
                if index not in used and abs(prediction - int(value)) <= tolerance
            ]
            if possible:
                _, index = min(possible)
                used.add(index)
                matched += 1
        precision = matched / max(1, len(selected))
        recall = matched / max(1, len(truth))
        score = 2 * precision * recall / max(1e-8, precision + recall)
        if score > best["f1"]:
            best = {
                "threshold": round(float(threshold), 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(score, 4),
            }
    return best


def note_pitch_metrics(item: dict, probability: np.ndarray) -> dict:
    frame_prediction = np.argmax(probability, axis=1)
    comparisons = []
    for index, note in enumerate(item.get("notes", [])):
        frames = np.flatnonzero(
            (item["times"] >= float(note["start"]))
            & (item["times"] < float(note["end"]))
        )
        expected_midi = int(note["midi"]) - 12
        if len(frames):
            edge = min(5, len(frames) // 4)
            central = frames[edge : len(frames) - edge] if edge else frames
            if not len(central):
                central = frames
            scores = probability[central].mean(0)
            selected_class = int(np.argmax(scores))
            predicted_midi = (
                None if selected_class == 0 else selected_class + MIDI_MIN - 1
            )
        else:
            predicted_midi = None
        comparisons.append(
            {
                "referenceIndex": index,
                "time": round(float(note["start"]), 5),
                "expectedMidi": expected_midi,
                "predictedMidi": predicted_midi,
                "correct": predicted_midi == expected_midi,
                "octaveError": (
                    predicted_midi is not None
                    and abs(predicted_midi - expected_midi) in (12, 24)
                ),
                "duration": round(float(note["end"]) - float(note["start"]), 5),
                "samePitch": bool(
                    index
                    and int(note["midi"]) == int(item["notes"][index - 1]["midi"])
                ),
            }
        )
    correct = sum(row["correct"] for row in comparisons)
    missing = sum(row["predictedMidi"] is None for row in comparisons)
    wrong = len(comparisons) - correct - missing
    voiced = item["pitchY"] > 0
    frame_accuracy = float(
        np.mean(frame_prediction[voiced] == item["pitchY"][voiced])
    ) if np.any(voiced) else 0.0
    return {
        "referenceNotes": len(comparisons),
        "correctPitch": correct,
        "wrongPitch": wrong,
        "missingNotes": missing,
        "octaveErrors": sum(row["octaveError"] for row in comparisons),
        "noteAccuracy": round(correct / max(1, len(comparisons)), 4),
        "voicedFrameAccuracy": round(frame_accuracy, 4),
        "comparisons": comparisons,
    }


def evaluate_models(
    models: Dict[str, nn.Module],
    items: Sequence[dict],
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> dict:
    rows = []
    for item in items:
        probabilities = {
            task: predict_item(models[task], item, mean, scale, task, device)
            for task in TASKS
        }
        pitch = note_pitch_metrics(item, probabilities["pitch"])
        ornament_valid = item["ornamentY"] >= 0
        ornament_prediction = np.argmax(probabilities["ornament"], axis=1)
        ornament_f1 = (
            f1_score(
                item["ornamentY"][ornament_valid],
                ornament_prediction[ornament_valid],
                average="macro",
                zero_division=0,
            )
            if np.any(ornament_valid)
            else 0.0
        )
        boundaries = {}
        for column, name in enumerate(BOUNDARY_NAMES):
            boundaries[name] = event_f1(
                item["boundaryY"][:, column],
                probabilities["boundary"][:, column],
                5 if name == "samePitch" else 4,
            )
        rows.append(
            {
                "id": item["id"],
                "name": item["name"],
                "pitch": pitch,
                "ornamentMacroF1": round(float(ornament_f1), 4),
                "boundaries": boundaries,
            }
        )
    return {
        "songs": rows,
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


def weakness_plan(validation_report: dict, maximum: int) -> dict:
    comparisons = [
        comparison
        for song in validation_report["songs"]
        for comparison in song["pitch"]["comparisons"]
        if not comparison["correct"]
    ]
    counts = {
        "short_note": sum(row["duration"] < 0.18 for row in comparisons),
        "same_pitch": sum(row["samePitch"] for row in comparisons),
        "octave": sum(row["octaveError"] for row in comparisons),
        "ornament": max(
            1,
            int(
                round(
                    (1 - validation_report["meanOrnamentMacroF1"])
                    * max(1, len(comparisons))
                )
            ),
        ),
        "tuplet": max(1, len(comparisons) // 5),
    }
    total_weight = sum(max(1, value) for value in counts.values())
    requested = {
        key: max(5, int(round(maximum * max(1, value) / total_weight)))
        for key, value in counts.items()
    }
    difference = maximum - sum(requested.values())
    requested["short_note"] = max(5, requested["short_note"] + difference)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "source": "validation MIDI error analysis only",
        "errorCounts": counts,
        "requestedSamples": requested,
        "maximumSamples": maximum,
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.with_suffix(path.suffix + ".previous")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() and KEEP_CHECKPOINT_PREVIOUS:
        shutil.copy2(path, previous)
    elif previous.exists() and not KEEP_CHECKPOINT_PREVIOUS:
        previous.unlink()
    torch.save(payload, temporary)
    temporary.replace(path)


def train_task(
    task: str,
    training: Sequence[dict],
    validation: Sequence[dict],
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    budget: Budget,
    checkpoint: Path,
    epochs: int,
    channels: int,
    transformer_layers: int,
    window: int,
    batch_size: int,
    seed: int,
    initial_state: Optional[dict] = None,
) -> Tuple[nn.Module, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = create_model(
        task,
        training[0]["features"].shape[1],
        task_output_count(task),
        channels,
        transformer_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1.5e-3
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )
    start_epoch = 0
    best_loss = float("inf")
    best_state = None
    history = []
    if checkpoint.exists():
        try:
            saved = torch.load(
                checkpoint, map_location=device, weights_only=False
            )
            if saved.get("complete"):
                model.load_state_dict(saved["bestState"])
                return model, saved["report"]
            model.load_state_dict(saved["modelState"])
            optimizer.load_state_dict(saved["optimizerState"])
            start_epoch = int(saved["epoch"])
            best_loss = float(saved["bestLoss"])
            best_state = saved.get("bestState")
            history = saved.get("history", [])
            print("再開: {} epoch {}".format(task, start_epoch + 1), flush=True)
        except Exception:
            previous = checkpoint.with_suffix(checkpoint.suffix + ".previous")
            if previous.exists():
                saved = torch.load(
                    previous, map_location=device, weights_only=False
                )
                model.load_state_dict(saved["modelState"])
                optimizer.load_state_dict(saved["optimizerState"])
                start_epoch = int(saved["epoch"])
                best_loss = float(saved["bestLoss"])
                best_state = saved.get("bestState")
                history = saved.get("history", [])
    elif initial_state is not None:
        model.load_state_dict(copy.deepcopy(initial_state))
    dataset = TrainingWindows(
        training,
        task,
        mean,
        scale,
        window,
        int(TRAINING_SAMPLES_PER_EPOCH),
        seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    weights = class_weights(task, training, device)
    stale = 0
    for epoch in range(start_epoch, epochs):
        if budget.expired():
            break
        thermal_guard()
        model.train()
        losses = []
        for features, target, weight in loader:
            features = features.to(device)
            target = target.to(device)
            weight = weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(features)
                loss = loss_for_task(task, logits, target, weight, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 3.5)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        training_loss = float(np.mean(losses))
        validation_loss = []
        for item in validation:
            probability = predict_item(model, item, mean, scale, task, device)
            if task in ("pitch", "ornament"):
                labels = item["pitchY"] if task == "pitch" else item["ornamentY"]
                valid = labels >= 0
                if task == "pitch":
                    valid &= labels > 0
                if np.any(valid):
                    validation_loss.append(
                        float(
                            np.mean(
                                -np.log(
                                    np.clip(
                                        probability[np.arange(len(labels))[valid], labels[valid]],
                                        1e-7,
                                        1,
                                    )
                                )
                            )
                        )
                    )
            else:
                labels = item["boundaryY"]
                validation_loss.append(
                    float(
                        np.mean(
                            -labels * np.log(np.clip(probability, 1e-7, 1))
                            - (1 - labels)
                            * np.log(np.clip(1 - probability, 1e-7, 1))
                        )
                    )
                )
        selected_loss = (
            float(np.mean(validation_loss)) if validation_loss else training_loss
        )
        row = {
            "epoch": epoch + 1,
            "trainingLoss": round(training_loss, 6),
            "validationLoss": round(selected_loss, 6),
        }
        history.append(row)
        if selected_loss < best_loss - 1e-4:
            best_loss = selected_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        report = {
            "task": task,
            "parameterCount": parameter_count(model),
            "epochsRequested": epochs,
            "epochsCompleted": epoch + 1,
            "bestValidationLoss": round(best_loss, 6),
            "history": history,
        }
        save_checkpoint(
            checkpoint,
            {
                "schemaVersion": SCHEMA_VERSION,
                "complete": False,
                "task": task,
                "epoch": epoch + 1,
                "modelState": model.state_dict(),
                "optimizerState": optimizer.state_dict(),
                "bestState": best_state,
                "bestLoss": best_loss,
                "history": history,
                "report": report,
            },
        )
        print(
            "{} epoch {}/{} loss={:.4f} val={:.4f}".format(
                task, epoch + 1, epochs, training_loss, selected_loss
            ),
            flush=True,
        )
        if stale >= 4:
            break
    stopped_for_budget = budget.expired()
    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    report = {
        "task": task,
        "parameterCount": parameter_count(model),
        "epochsRequested": epochs,
        "epochsCompleted": len(history),
        "bestValidationLoss": round(best_loss, 6),
        "history": history,
        "stoppedForTimeBudget": stopped_for_budget,
    }
    save_checkpoint(
        checkpoint,
        {
            "schemaVersion": SCHEMA_VERSION,
            "complete": not stopped_for_budget,
            "task": task,
            "epoch": len(history),
            "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(),
            "bestState": best_state,
            "bestLoss": best_loss,
            "history": history,
            "report": report,
        },
    )
    if stopped_for_budget:
        raise TimeoutError(
            "{}の学習中に安全な実行時間上限へ到達しました".format(task)
        )
    return model, report


def train_bundle(
    name: str,
    training: Sequence[dict],
    validation: Sequence[dict],
    device: torch.device,
    budget: Budget,
    epochs: int,
    channels: int,
    transformer_layers: int,
    window: int,
    batch_size: int,
    seed: int,
    initial_models: Optional[Dict[str, nn.Module]] = None,
    normalization_values: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Dict[str, nn.Module], np.ndarray, np.ndarray, dict]:
    mean, scale = (
        normalization_values
        if normalization_values is not None
        else feature_normalization(training)
    )
    models = {}
    reports = {}
    for task_index, task in enumerate(TASKS):
        if budget.expired():
            raise TimeoutError("10時間の実行予算に到達しました")
        initial_state = (
            initial_models[task].state_dict()
            if initial_models is not None
            else None
        )
        model, report = train_task(
            task,
            training,
            validation,
            mean,
            scale,
            device,
            budget,
            MODEL_ROOT / "checkpoints" / name / "{}.pt".format(task),
            epochs,
            channels,
            transformer_layers,
            window,
            batch_size,
            seed + task_index,
            initial_state,
        )
        models[task] = model
        reports[task] = report
    return models, mean, scale, reports


def bundle_score(report: dict) -> float:
    return (
        report["meanNoteAccuracy"]
        + 0.15 * report["meanOrnamentMacroF1"]
        + 0.10 * report["meanBoundaryF1"]
    )


def save_final_models(
    models: Dict[str, nn.Module],
    mean: np.ndarray,
    scale: np.ndarray,
    args,
    chosen_branch: str,
) -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for task, model in models.items():
        save_checkpoint(
            MODEL_ROOT / "{}-heavy-v41.pt".format(task),
            {
                "schemaVersion": SCHEMA_VERSION,
                "complete": True,
                "task": task,
                "architecture": "HeavyAcousticModel",
                "featureCount": int(len(mean)),
                "channels": args.channels,
                "transformerLayers": args.transformer_layers,
                "outputCount": task_output_count(task),
                "parameterCount": parameter_count(model),
                "mean": mean,
                "scale": scale,
                "bestState": model.state_dict(),
                "chosenBranch": chosen_branch,
            },
        )


def resolve_split(
    items: Sequence[dict], validation_name: str, test_name: str
) -> Tuple[List[dict], List[dict], List[dict]]:
    validation = [item for item in items if item["name"] == validation_name]
    test = [item for item in items if item["name"] == test_name]
    if len(validation) != 1 or len(test) != 1:
        raise ValueError("検証曲またはテスト曲が見つかりません")
    training = [
        item
        for item in items
        if item["name"] not in (validation_name, test_name)
    ]
    return training, validation, test


def runtime_update(started: float, args, stage: str, complete: bool, extra=None):
    payload = {
        "model": "learning-v4.1",
        "complete": complete,
        "stage": stage,
        "deviceRequest": args.device,
        "maximumHours": args.max_hours,
        "totalSeconds": round(time.time() - started, 3),
        "totalDisplay": duration_text(time.time() - started),
    }
    if extra:
        payload.update(extra)
    json_write(RUNTIME_PATH, payload)


def run(args) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    budget = Budget.create(args.max_hours)
    device = choose_device(args.device)
    print("PianoFlow ver4.1 pilot", flush=True)
    print("device={} 最大実行={:.2f}時間".format(device, args.max_hours), flush=True)
    runtime_update(started, args, "build-real", False)
    build_real_dataset(args.manifest, args.rebuild_features)
    real_items = load_items(WORK / "real")
    training, validation, test = resolve_split(
        real_items, args.validation_song, args.test_song
    )
    split = {
        "training": [item["name"] for item in training],
        "validation": validation[0]["name"],
        "test": test[0]["name"],
        "testMidiNeverUsedForTrainingOrGeneration": True,
    }
    json_write(WORK / "split.json", split)

    runtime_update(started, args, "base-training", False, {"split": split})
    base_models, mean, scale, base_training = train_bundle(
        "base",
        training,
        validation,
        device,
        budget,
        args.base_epochs,
        args.channels,
        args.transformer_layers,
        args.window,
        args.batch_size,
        args.seed,
    )
    base_validation = evaluate_models(
        base_models, validation, mean, scale, device
    )
    plan = weakness_plan(base_validation, args.synthetic_samples)
    json_write(WORK / "augmentation-plan.json", plan)

    runtime_update(started, args, "synthetic-generation", False)
    synthetic = generate_synthetic(
        training, plan, args.synthetic_samples, args.seed + 1000, budget
    )
    if not synthetic:
        raise RuntimeError("合成データを生成できませんでした")

    runtime_update(started, args, "ab-training", False)
    branch_a_models, _, _, branch_a_training = train_bundle(
        "branch-real-only",
        training,
        validation,
        device,
        budget,
        args.branch_epochs,
        args.channels,
        args.transformer_layers,
        args.window,
        args.batch_size,
        args.seed + 2000,
        initial_models=base_models,
        normalization_values=(mean, scale),
    )
    branch_a_validation = evaluate_models(
        branch_a_models, validation, mean, scale, device
    )

    # TrainingWindows samples a fixed number of windows per epoch and keeps
    # real/synthetic windows at 65/35.  A/B branches therefore receive the
    # same number of optimizer updates.
    mixed = list(training) + list(synthetic)
    branch_b_models, _, _, branch_b_training = train_bundle(
        "branch-real-plus-synthetic",
        mixed,
        validation,
        device,
        budget,
        args.branch_epochs,
        args.channels,
        args.transformer_layers,
        args.window,
        args.batch_size,
        args.seed + 3000,
        initial_models=base_models,
        normalization_values=(mean, scale),
    )
    branch_b_validation = evaluate_models(
        branch_b_models, validation, mean, scale, device
    )
    score_a = bundle_score(branch_a_validation)
    score_b = bundle_score(branch_b_validation)
    chosen = (
        "real-plus-synthetic"
        if score_b > score_a + args.minimum_gain
        else "real-only"
    )
    chosen_models = (
        branch_b_models if chosen == "real-plus-synthetic" else branch_a_models
    )
    chosen_validation = (
        branch_b_validation
        if chosen == "real-plus-synthetic"
        else branch_a_validation
    )
    runtime_update(started, args, "held-out-test", False)
    test_report = evaluate_models(chosen_models, test, mean, scale, device)
    save_final_models(chosen_models, mean, scale, args, chosen)
    report = {
        "model": "learning-v4.1",
        "mode": "maximum-10-hour active synthetic-data pilot",
        "device": str(device),
        "complete": True,
        "split": split,
        "modelConfiguration": {
            "channels": args.channels,
            "transformerLayers": args.transformer_layers,
            "window": args.window,
            "batchSize": args.batch_size,
            "parameterCounts": {
                task: parameter_count(model)
                for task, model in chosen_models.items()
            },
        },
        "syntheticData": {
            "requested": args.synthetic_samples,
            "generated": len(synthetic),
            "format": "paired melody.mid + vocal.wav + exact labels",
            "pilotVocalRenderer": "deterministic source-filter; replaceable adapter for licensed SVS in v4.2",
            "plan": plan,
        },
        "training": {
            "base": base_training,
            "realOnly": branch_a_training,
            "realPlusSynthetic": branch_b_training,
        },
        "validation": {
            "base": base_validation,
            "realOnly": branch_a_validation,
            "realPlusSynthetic": branch_b_validation,
            "scoreRealOnly": round(score_a, 6),
            "scoreRealPlusSynthetic": round(score_b, 6),
            "minimumGainToAcceptSynthetic": args.minimum_gain,
            "chosenBranch": chosen,
            "chosen": chosen_validation,
        },
        "heldOutTest": test_report,
        "elapsedSeconds": round(time.time() - started, 3),
        "elapsedDisplay": duration_text(time.time() - started),
    }
    json_write(REPORT_PATH, report)
    json_write(
        STATE_PATH,
        {
            "schemaVersion": SCHEMA_VERSION,
            "complete": True,
            "chosenBranch": chosen,
            "report": str(REPORT_PATH),
        },
    )
    runtime_update(
        started,
        args,
        "complete",
        True,
        {"chosenBranch": chosen, "report": str(REPORT_PATH)},
    )
    print("\nCOMPLETE", flush=True)
    print("採用: {}".format(chosen), flush=True)
    print("評価: {}".format(REPORT_PATH), flush=True)
    return 0


def self_test() -> int:
    model = HeavyAcousticModel(134, PITCH_CLASSES, channels=64, transformer_layers=1)
    values = torch.zeros((2, 64, 134), dtype=torch.float32)
    output = model(values)
    assert output.shape == (2, 64, PITCH_CLASSES)
    temporary = WORK / "self-test"
    temporary.mkdir(parents=True, exist_ok=True)
    notes = [
        {"midi": 72, "start": 0.2, "end": 0.5},
        {"midi": 72, "start": 0.5, "end": 0.8},
        {"midi": 74, "start": 0.8, "end": 1.2},
    ]
    ornaments = [
        {"noteIndex": 0, "type": "scoop", "vibratoDepth": 0.1},
        {"noteIndex": 1, "type": "stable", "vibratoDepth": 0.1},
        {
            "noteIndex": 2,
            "type": "transition",
            "shift": 1.5,
            "vibratoDepth": 0.1,
        },
    ]
    write_midi(temporary / "test.mid", notes, 120)
    from evaluate_pitch_tracking import parse_midi

    parsed_notes, parsed_bpm = parse_midi(temporary / "test.mid")
    assert len(parsed_notes) == 3
    assert abs(parsed_bpm - 120.0) < 0.01
    synthesize_vocal(temporary / "test.wav", notes, ornaments, 41)
    features, times = extract_features(temporary / "test.wav")
    labels = synthetic_labels(notes, ornaments, times)
    accepted, reasons = quality_check(features, labels, notes * 3)
    dummy_real = {
        "kind": "real",
        "features": features,
        "pitchY": labels["pitchY"],
        "pitchWeight": labels["pitchWeight"],
        "ornamentY": labels["ornamentY"],
        "ornamentWeight": labels["ornamentWeight"],
        "boundaryY": labels["boundaryY"],
    }
    dummy_synthetic = {**dummy_real, "kind": "synthetic"}
    mean, scale = feature_normalization([dummy_real])
    windows = TrainingWindows(
        [dummy_real, dummy_synthetic],
        "pitch",
        mean,
        scale,
        64,
        200,
        41,
    )
    synthetic_windows = sum(
        windows.items[source].get("kind") == "synthetic"
        for source, _ in windows.index
    )
    assert 50 <= synthetic_windows <= 90
    checkpoint = temporary / "checkpoint.pt"
    save_checkpoint(checkpoint, {"generation": 1})
    save_checkpoint(checkpoint, {"generation": 2})
    assert checkpoint.exists()
    assert checkpoint.with_suffix(checkpoint.suffix + ".previous").exists()
    print(
        json.dumps(
            {
                "modelOutput": list(output.shape),
                "parameterCount": parameter_count(model),
                "featureShape": list(features.shape),
                "midiNotes": len(parsed_notes),
                "midiBpm": parsed_bpm,
                "syntheticWindowsOf200": synthetic_windows,
                "checkpointGenerations": 2,
                "accepted": accepted,
                "reasons": reasons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    shutil.rmtree(temporary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PianoFlow learning v4.1 time-bounded active data pilot"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "dataset-work" / "manifest.json",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-hours", type=float, default=9.75)
    parser.add_argument("--base-epochs", type=int, default=6)
    parser.add_argument("--branch-epochs", type=int, default=3)
    parser.add_argument("--synthetic-samples", type=int, default=120)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-song", default=DEFAULT_VALIDATION)
    parser.add_argument("--test-song", default=DEFAULT_TEST)
    parser.add_argument("--minimum-gain", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        return run(args)
    except TimeoutError as error:
        state = {
            "schemaVersion": SCHEMA_VERSION,
            "complete": False,
            "stoppedForTimeBudget": True,
            "message": str(error),
        }
        json_write(STATE_PATH, state)
        print("\nTIME BUDGET STOP: {}".format(error), flush=True)
        print("同じBATを再実行するとチェックポイントから再開します。", flush=True)
        return 75
    except KeyboardInterrupt:
        json_write(
            STATE_PATH,
            {
                "schemaVersion": SCHEMA_VERSION,
                "complete": False,
                "interrupted": True,
            },
        )
        print("\n中断しました。保存済みチェックポイントから再開できます。", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
