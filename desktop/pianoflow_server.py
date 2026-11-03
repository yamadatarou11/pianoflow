from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import http.server
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
JOBS_ROOT = ROOT / "work" / "jobs"
ANALYSIS_VERSION = 39
FFMPEG_OVERRIDE: str | None = None
DATASET_BATCH_PROCESS: subprocess.Popen | None = None
DATASET_BATCH_LOCK = threading.Lock()
AI_ANALYSIS_LOCK = threading.Lock()
DATASET_STATUS_PATH = ROOT / "dataset-work" / "v431-batch-status.json"
DATASET_UPLOAD_EXTENSIONS = {".mov", ".mp4", ".wav", ".mp3", ".m4a", ".mid", ".midi", ".txt"}
VIDEO_SETTINGS_PATH = ROOT / "work" / "pianoflow-video-settings.json"


def boundary_first_enabled() -> bool:
    return os.environ.get("PIANOFLOW_BOUNDARY_FIRST", "1").strip().lower() not in {"0", "false", "off"}


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    base = os.environ.get("LOCALAPPDATA", "")
    exact = os.path.join(
        base, "Microsoft", "WinGet", "Packages",
        "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
        "ffmpeg-8.1.2-full_build", "bin", "ffmpeg.exe",
    )
    if os.path.isfile(exact):
        return exact
    matches = glob.glob(os.path.join(base, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg_*", "**", "bin", "ffmpeg.exe"), recursive=True)
    return matches[0] if matches else None


def find_ffprobe(ffmpeg: str | None = None) -> str | None:
    found = shutil.which("ffprobe")
    if found:
        return found
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if sibling.exists():
            return str(sibling)
    return None


def rational_fps(value: str) -> float:
    try:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / max(float(denominator), 1e-9)
    except (AttributeError, ValueError, ZeroDivisionError):
        return 0.0


def validate_shorts_mp4(path: Path, ffprobe: str, width: int, height: int, fps: int) -> dict:
    probe = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries",
            "format=format_name,duration:stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,field_order,sample_rate,channels,duration",
            "-of", "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if probe.returncode != 0:
        raise ValueError("ffprobe failed: " + probe.stderr.decode("utf-8", "replace")[-4000:])
    data = json.loads(probe.stdout.decode("utf-8"))
    streams = data.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    format_info = data.get("format", {})
    video_fps = rational_fps(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
    video_duration = float(video.get("duration") or format_info.get("duration") or 0)
    audio_duration = float(audio.get("duration") or format_info.get("duration") or 0)
    header = path.read_bytes()
    moov_position, mdat_position = header.find(b"moov"), header.find(b"mdat")
    checks = {
        "mp4Container": "mp4" in str(format_info.get("format_name", "")),
        "h264Video": video.get("codec_name") == "h264",
        "aacAudio": audio.get("codec_name") == "aac",
        "resolution": int(video.get("width") or 0) == width and int(video.get("height") or 0) == height,
        "aspectRatio9x16": width * 16 == height * 9,
        "constantFrameRate": abs(video_fps - fps) < 0.01 and fps in (30, 60),
        "pixelFormat": video.get("pix_fmt") == "yuv420p",
        "progressive": str(video.get("field_order", "progressive")) in {"progressive", "unknown"},
        "audioSampleRate": int(audio.get("sample_rate") or 0) == 48000,
        "audioStereo": int(audio.get("channels") or 0) == 2,
        "durationDifferenceUnder100ms": abs(video_duration - audio_duration) <= 0.1,
        "fastStart": 0 <= moov_position < mdat_position,
        "mp4Extension": path.suffix.lower() == ".mp4",
        "mime": "video/mp4",
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "format": format_info,
        "video": video,
        "audio": audio,
        "measured": {
            "fps": round(video_fps, 6),
            "videoDuration": round(video_duration, 6),
            "audioDuration": round(audio_duration, 6),
            "durationDifference": round(abs(video_duration - audio_duration), 6),
            "moovPosition": moov_position,
            "mdatPosition": mdat_position,
        },
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, *_args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename, X-Lyrics-Base64, X-BPM, X-Response-Mode, X-Video-Length, X-Frame-Rate, X-Video-Format, X-Video-Width, X-Video-Height, X-Video-Duration, X-Audio-Duration, X-Audio-Start, X-Output-Duration, X-Audio-Present")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        request_path = urlparse(self.path).path
        if request_path == "/download-latest-mp4":
            self.download_latest_mp4()
            return
        if request_path == "/video-settings":
            self.send_video_settings()
            return
        if request_path == "/dataset-v431-status":
            self.send_dataset_batch_status()
            return
        if request_path == "/capabilities":
            ai_ready = (ROOT / ".venv-pianoflow" / "Scripts" / "python.exe").exists()
            gpu_worker = os.environ.get("PIANOFLOW_LEARNING_GPU_WORKER", "").strip()
            gpu_model_root = Path(os.environ.get("PIANOFLOW_LEARNING_GPU_MODEL_ROOT", ""))
            gpu_ready = (
                os.environ.get("PIANOFLOW_GPU_READY", "") == "1"
                and bool(gpu_worker)
                and Path(gpu_worker).is_file()
                and (ROOT / "pianoflow_gpu_worker.py").is_file()
                and all((gpu_model_root / ("current-" + task + ".pt")).is_file()
                        for task in ("pitch", "ornament", "boundary"))
            )
            hq_ready = all(path.exists() for path in (
                ROOT / ".venv-pianoflow-hq" / "Scripts" / "python.exe",
                ROOT / "models" / "rmvpe.pt",
                ROOT / "hq_rmvpe.py",
            ))
            learned_model_ready = all(
                (ROOT / "models" / "learning-v431" / f"current-{task}.pt").exists()
                for task in ("pitch", "ornament", "boundary")
            )
            data = json.dumps({
                "mp4Export": True,
                "mp4Only": True,
                "version": 126,
                "shortsExport": "1080x1920 H.264 High + AAC-LC, validated with ffprobe",
                "analysisVersion": ANALYSIS_VERSION,
                "aiOnlyEndpoint": "/analyze-ai",
                "trainedMelodyModel": learned_model_ready,
                "aiMelody": ai_ready,
                "highQualityLeadVocal": hq_ready,
                "aiEngine": "RoFormer lead-vocal ensemble + RMVPE/CREPE/WORLD score-aware pitch decoder" if hq_ready else "Demucs MDX + pYIN score-aware pitch decoder",
                "pitchCurve": True,
                "pitchConsensus": ["RMVPE", "CREPE", "WORLD", "pYIN"],
                "tempoAwareSegmentation": True,
                "boundaryFirstSegmentation": boundary_first_enabled(),
                "edgePreservingPitchSmoothing": True,
                "acousticWordAndSyllableSegments": True,
                "localJapaneseTranscription": (ROOT / "hq_transcriber.py").exists(),
                "suppliedLyricsHiraganaAlignment": True,
                "lyricsForcedAlignment": "MMS multilingual CTC",
                "scorePreview": True,
                "generalTuplets": "1-16 subdivisions",
                "separateVideoAudioMux": True,
                "stemSeparation": ["vocals", "instrumental", "lead", "backing", "dryLead", "reverb"],
                "noteEditing": True,
                "dedicatedPitchEditor": True,
                "referenceMidiAlignment": "first reliable sung-note onset + fixed MIDI clock; no local warp",
                "gpu": gpu_ready,
                "gpuInference": "learning-v4.3.1 pitch/ornament/boundary TCN inference" if gpu_ready else None,
                "gpuName": "NVIDIA GeForce GTX TITAN" if gpu_ready else None,
                "gpuCpuStages": ["ボーカル分離", "文字起こし", "特徴量抽出"] if gpu_ready else [],
                "gpuMode": gpu_ready,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def download_latest_mp4(self):
        path = ROOT / "work" / "shorts-validation" / "pianoflow-shorts-latest.mp4"
        if not path.exists() or path.stat().st_size <= 0:
            self.send_error(404, "Completed MP4 not found")
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Disposition", 'attachment; filename="PianoFlow-video.mp4"')
        self.send_header("Content-Length", str(size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with path.open("rb") as source:
            shutil.copyfileobj(source, self.wfile, length=1024 * 1024)

    def send_video_settings(self):
        if not VIDEO_SETTINGS_PATH.exists():
            self.send_json(404, {"error": "saved settings not found"})
            return
        try:
            self.send_json(200, json.loads(VIDEO_SETTINGS_PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            self.send_json(500, {"error": str(error)})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/video-settings":
            self.save_video_settings()
            return
        if path == "/dataset-batch-file":
            self.save_dataset_batch_file()
            return
        if path == "/dataset-v431-start":
            self.start_dataset_batch()
            return
        if path == "/dataset-review-override":
            self.save_dataset_review_override()
            return
        if path == "/dataset-review-auto-confirm-all":
            self.auto_confirm_all_dataset_offsets()
            return
        if path == "/analyze-melody":
            self.analyze_melody()
            return
        if path == "/analyze-ai":
            if not AI_ANALYSIS_LOCK.acquire(blocking=False):
                self.send_error(429, "別のAI解析が実行中です。完了後にもう一度お試しください")
                return
            try:
                self.analyze_melody(pitch_model="learning-v431")
            finally:
                AI_ANALYSIS_LOCK.release()
            return
        if path == "/align-reference":
            self.align_reference()
            return
        if path == "/mux-piano-mp4":
            self.mux_piano_mp4()
            return
        if path != "/convert-to-mp4":
            self.send_error(404)
            return
        ffmpeg = FFMPEG_OVERRIDE or find_ffmpeg()
        if not ffmpeg:
            self.send_error(500, "FFmpeg not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 4 * 1024 * 1024 * 1024:
            self.send_error(400, "Invalid video size")
            return

        with tempfile.TemporaryDirectory(prefix="pianoflow-") as temp:
            webm = Path(temp) / "recording.webm"
            mp4 = Path(temp) / "pianoflow.mp4"
            remaining = length
            with webm.open("wb") as out:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)
            result = subprocess.run(
                [ffmpeg, "-y", "-i", str(webm), "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(mp4)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode != 0 or not mp4.exists():
                self.send_error(500, "MP4 conversion failed")
                return
            size = mp4.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Disposition", 'attachment; filename="pianoflow.mp4"')
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with mp4.open("rb") as source:
                shutil.copyfileobj(source, self.wfile, length=1024 * 1024)

    def save_video_settings(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 512 * 1024:
                raise ValueError("設定データのサイズが不正です")
            settings = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(settings, dict) or not isinstance(settings.get("controls"), dict):
                raise ValueError("設定データの形式が不正です")
            settings["schemaVersion"] = 1
            settings["savedAt"] = int(settings.get("savedAt") or time.time() * 1000)
            VIDEO_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = VIDEO_SETTINGS_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(VIDEO_SETTINGS_PATH)
            self.send_json(200, {"ok": True, "savedAt": settings["savedAt"]})
        except Exception as error:
            self.send_json(400, {"ok": False, "error": str(error)})

    def send_json(self, status: int, value: dict):
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_dataset_batch_status(self):
        global DATASET_BATCH_PROCESS
        try:
            status = json.loads(DATASET_STATUS_PATH.read_text(encoding="utf-8")) if DATASET_STATUS_PATH.exists() else {"state": "idle", "message": "一括解析はまだ開始されていません"}
        except (OSError, json.JSONDecodeError):
            status = {"state": "idle", "message": "進捗情報はありません"}
        with DATASET_BATCH_LOCK:
            process = DATASET_BATCH_PROCESS
            if process is not None and process.poll() is None:
                status["state"] = "running"
            elif process is not None and process.poll() is not None:
                DATASET_BATCH_PROCESS = None
        self.send_json(200, status)

    def save_dataset_batch_file(self):
        try:
            filename = Path(unquote(self.headers.get("X-Filename", ""))).name
            if not filename or Path(filename).suffix.lower() not in DATASET_UPLOAD_EXTENSIONS:
                raise ValueError("対応する音声・動画・MIDI・歌詞ファイルを選択してください")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4 * 1024 * 1024 * 1024:
                raise ValueError("ファイルサイズが不正です")
            destination = ROOT / "dataset" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            remaining = length
            with destination.open("wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("ファイル転送が途中で終了しました")
                    output.write(chunk)
                    remaining -= len(chunk)
            self.send_json(200, {"ok": True, "filename": filename, "size": length})
        except Exception as error:
            self.send_json(400, {"ok": False, "error": str(error)})

    def start_dataset_batch(self):
        global DATASET_BATCH_PROCESS
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            with DATASET_BATCH_LOCK:
                if DATASET_BATCH_PROCESS is not None and DATASET_BATCH_PROCESS.poll() is None:
                    self.send_json(409, {"ok": False, "error": "一括解析はすでに実行中です"})
                    return
                # Use the current interpreter by default. Users who keep the
                # training environment separate can override it without
                # embedding a machine-specific path in the public source.
                configured_python = os.environ.get("PIANOFLOW_DATASET_PYTHON", "").strip()
                python = Path(configured_python) if configured_python else Path(sys.executable)
                command = [str(python), str(ROOT / "run_dataset_v431_review.py")]
                if bool(request.get("refreshBase")):
                    command.append("--refresh-base")
                DATASET_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
                DATASET_STATUS_PATH.write_text(json.dumps({"state": "running", "stage": "starting", "current": 0, "total": 0, "message": "一括解析を開始しています", "error": "", "updatedAt": time.time()}, ensure_ascii=False, indent=2), encoding="utf-8")
                log = (ROOT / "dataset-work" / "v431-batch.log").open("ab")
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                try:
                    DATASET_BATCH_PROCESS = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
                finally:
                    log.close()
            self.send_json(202, {"ok": True, "pid": DATASET_BATCH_PROCESS.pid})
        except Exception as error:
            self.send_json(400, {"ok": False, "error": str(error)})

    def save_dataset_review_override(self):
        """Persist a reviewed first-note offset and refresh lightweight metrics."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 64 * 1024:
            self.send_error(400, "Invalid dataset override payload")
            return
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            song_id = str(request.get("songId", "")).strip()
            if (
                not song_id
                or len(song_id) > 80
                or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in song_id)
            ):
                raise ValueError("曲IDが不正です")
            offset = float(request.get("offsetSeconds"))
            if not math.isfinite(offset) or not -10.0 <= offset <= 300.0:
                raise ValueError("同期位置が範囲外です")
            manifest_path = ROOT / "dataset-work" / "manifest.json"
            if not manifest_path.exists():
                raise ValueError("データセット台帳がありません")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            song = next((item for item in manifest.get("songs", []) if item.get("id") == song_id), None)
            if song is None:
                raise ValueError("対象曲が見つかりません")
            overrides_path = ROOT / "dataset-work" / "alignment-overrides.json"
            overrides = (
                json.loads(overrides_path.read_text(encoding="utf-8"))
                if overrides_path.exists()
                else {}
            )
            overrides[song_id] = {
                "offsetSeconds": round(offset, 6),
                "updatedAtUnix": round(time.time(), 3),
                "method": "manual first intended vocal-syllable onset review",
            }
            temporary = overrides_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(overrides_path)
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pianoflow_dataset.py"),
                    "evaluate",
                    "--continue-on-error",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=flags,
            )
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace")[-1600:]
                raise RuntimeError(detail or "同期後の再評価に失敗しました")
            reference_file = ROOT / str(song["referenceFile"])
            evaluation_path = reference_file.parent / "audio-only" / "baseline-evaluation.json"
            alignment_path = reference_file.parent / "audio-only" / "initial-alignment.json"
            payload = json.dumps({
                "ok": True,
                "songId": song_id,
                "offsetSeconds": round(offset, 6),
                "evaluation": json.loads(evaluation_path.read_text(encoding="utf-8")),
                "alignment": json.loads(alignment_path.read_text(encoding="utf-8")),
            }, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
        except Exception as error:
            payload = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")
            self.send_response(400)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def auto_confirm_all_dataset_offsets(self):
        """Fill every unreviewed song with its automatic first-vocal-onset offset."""
        try:
            manifest_path = ROOT / "dataset-work" / "manifest.json"
            if not manifest_path.exists():
                raise ValueError("データセット台帳がありません")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            songs = manifest.get("songs", [])
            if not songs:
                raise ValueError("対象曲がありません")

            overrides_path = ROOT / "dataset-work" / "alignment-overrides.json"
            overrides = (
                json.loads(overrides_path.read_text(encoding="utf-8"))
                if overrides_path.exists()
                else {}
            )
            # Refresh automatic anchors with the current alignment algorithm
            # before reading them. Otherwise a stale baseline file could
            # preserve an older repeated-phrase mistake (for example Phony).
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            refresh = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pianoflow_dataset.py"),
                    "evaluate",
                    "--ignore-overrides",
                    "--continue-on-error",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=flags,
            )
            if refresh.returncode != 0:
                detail = refresh.stderr.decode("utf-8", errors="replace")[-1600:]
                raise RuntimeError(detail or "自動同期位置の更新に失敗しました")
            now = round(time.time(), 3)
            saved = []
            preserved_manual = []
            low_confidence = []
            for song in songs:
                song_id = str(song["id"])
                existing = overrides.get(song_id, {})
                if str(existing.get("method", "")).startswith("manual"):
                    preserved_manual.append(str(song.get("name", song_id)))
                    continue
                reference_file = ROOT / str(song["referenceFile"])
                evaluation_path = reference_file.parent / "audio-only" / "baseline-evaluation.json"
                if not evaluation_path.exists():
                    raise ValueError(f"解析結果がありません: {song.get('name', song_id)}")
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
                synchronization = evaluation.get("synchronization", {})
                offset = float(
                    synchronization.get(
                        "automaticInitialNoteOffsetSeconds",
                        synchronization.get("audioTimeAtMidiZero"),
                    )
                )
                confidence = float(synchronization.get("confidence", 0.0))
                if not math.isfinite(offset) or not -10.0 <= offset <= 300.0:
                    raise ValueError(f"自動同期位置が不正です: {song.get('name', song_id)}")
                overrides[song_id] = {
                    "offsetSeconds": round(offset, 6),
                    "updatedAtUnix": now,
                    "method": "automatic first intended vocal-syllable onset batch confirmation",
                    "confidence": round(confidence, 6),
                    "reviewStatus": "auto-provisional",
                }
                saved.append(str(song.get("name", song_id)))
                if confidence < 0.6:
                    low_confidence.append({
                        "name": str(song.get("name", song_id)),
                        "confidence": round(confidence, 4),
                    })

            temporary = overrides_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(overrides_path)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "pianoflow_dataset.py"),
                    "evaluate",
                    "--continue-on-error",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=flags,
            )
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace")[-1600:]
                raise RuntimeError(detail or "一括同期後の再評価に失敗しました")
            payload = json.dumps({
                "ok": True,
                "confirmedCount": len(overrides),
                "totalCount": len(songs),
                "autoSaved": saved,
                "preservedManual": preserved_manual,
                "lowConfidence": low_confidence,
            }, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
        except Exception as error:
            payload = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")
            self.send_response(400)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def mux_piano_mp4(self):
        """Create and validate a YouTube Shorts-safe H.264/AAC MP4."""
        ffmpeg = FFMPEG_OVERRIDE or find_ffmpeg()
        ffprobe = find_ffprobe(ffmpeg)
        if not ffmpeg or not ffprobe:
            self.send_mp4_error(500, "FFmpeg or ffprobe not found")
            return
        try:
            total = int(self.headers.get("Content-Length", "0"))
            video_length = int(self.headers.get("X-Video-Length", "0"))
            frame_rate = int(self.headers.get("X-Frame-Rate", "30"))
            width = int(self.headers.get("X-Video-Width", "1080"))
            height = int(self.headers.get("X-Video-Height", "1920"))
            video_duration = float(self.headers.get("X-Video-Duration", "0"))
            audio_duration = float(self.headers.get("X-Audio-Duration", "0"))
            audio_start = max(0.0, float(self.headers.get("X-Audio-Start", "0")))
            output_duration = float(self.headers.get("X-Output-Duration", "0"))
            audio_present = self.headers.get("X-Audio-Present", "0") == "1"
            video_format = self.headers.get("X-Video-Format", "unknown").lower()
        except ValueError:
            total = video_length = 0
            frame_rate = 30
            width, height = 1080, 1920
            video_duration = audio_duration = audio_start = output_duration = 0.0
            audio_present = False
            video_format = "unknown"
        frame_rate = 60 if frame_rate >= 45 else 30
        diagnostics = {
            "inputVideoFormat": video_format,
            "inputResolution": f"{width}x{height}",
            "inputFps": frame_rate,
            "inputVideoDuration": video_duration,
            "inputAudioDuration": audio_duration,
            "audioPresent": audio_present,
        }
        if (
            total <= 0
            or video_length <= 0
            or video_length >= total
            or total > 4 * 1024 * 1024 * 1024
            or not audio_present
            or width <= 0
            or height <= 0
            or output_duration <= 0
        ):
            self.send_mp4_error(400, "Invalid mux payload", diagnostics=diagnostics)
            return
        with tempfile.TemporaryDirectory(prefix="pianoflow-mux-") as temp:
            extension = {"h264": ".h264", "ivf": ".ivf", "webm": ".webm"}.get(video_format, ".bin")
            visual, wav, mp4 = Path(temp) / f"visual{extension}", Path(temp) / "piano.wav", Path(temp) / "pianoflow-shorts.mp4"
            remaining_video, remaining_total = video_length, total
            with visual.open("wb") as output:
                while remaining_video:
                    chunk = self.rfile.read(min(1024 * 1024, remaining_video))
                    if not chunk:
                        break
                    output.write(chunk)
                    remaining_video -= len(chunk)
                    remaining_total -= len(chunk)
            with wav.open("wb") as output:
                while remaining_total:
                    chunk = self.rfile.read(min(1024 * 1024, remaining_total))
                    if not chunk:
                        break
                    output.write(chunk)
                    remaining_total -= len(chunk)
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            gop = 30 if frame_rate == 60 else 15
            bitrate = "16M" if width <= 1080 else ("45M" if width <= 2160 else "100M")
            result = subprocess.run(
                [
                    ffmpeg, "-y",
                    "-r", str(frame_rate), "-i", str(visual),
                    "-ss", f"{audio_start:.6f}", "-i", str(wav),
                    "-t", f"{output_duration:.6f}",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-vf", f"setpts=N/({frame_rate}*TB),fps={frame_rate},scale={width}:{height}:flags=lanczos,format=yuv420p",
                    "-af", "volume=10,acompressor=threshold=0.18:ratio=3.5:attack=8:release=100:makeup=1,loudnorm=I=-13:TP=-1:LRA=6",
                    "-c:v", "libx264", "-profile:v", "high", "-preset", "medium",
                    "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", str(int(bitrate[:-1]) * 2) + "M",
                    "-r", str(frame_rate), "-fps_mode", "cfr", "-g", str(gop),
                    "-keyint_min", str(gop), "-sc_threshold", "0",
                    "-pix_fmt", "yuv420p", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "320k",
                    "-ar", "48000", "-ac", "2", "-movflags", "+faststart",
                    str(mp4),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=flags,
            )
            if result.returncode != 0 or not mp4.exists():
                (ROOT / "work").mkdir(parents=True, exist_ok=True)
                (ROOT / "work" / "mux-last-error.log").write_bytes(result.stderr[-12000:])
                self.send_mp4_error(
                    500,
                    "MP4 mux failed",
                    result.stderr.decode("utf-8", "replace")[-12000:],
                    diagnostics,
                )
                return
            try:
                validation = validate_shorts_mp4(mp4, ffprobe, width, height, frame_rate)
            except Exception as error:
                self.send_mp4_error(500, f"MP4 validation failed: {error}", diagnostics=diagnostics)
                return
            if not validation["ok"]:
                self.send_mp4_error(500, "MP4 validation failed", diagnostics={**diagnostics, "validation": validation})
                return
            artifact_root = ROOT / "work" / "shorts-validation"
            artifact_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mp4, artifact_root / "pianoflow-shorts-latest.mp4")
            (artifact_root / "ffprobe-validation.json").write_text(
                json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if self.headers.get("X-Response-Mode", "").lower() == "json":
                payload = json.dumps(
                    {
                        "ok": True,
                        "downloadUrl": f"/download-latest-mp4?t={int(time.time())}",
                        "validationUrl": "/work/shorts-validation/ffprobe-validation.json",
                        "validation": validation,
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("X-PianoFlow-Validated", "1")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            size = mp4.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Disposition", 'attachment; filename="pianoflow-shorts.mp4"')
            self.send_header("X-PianoFlow-Validated", "1")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with mp4.open("rb") as source:
                shutil.copyfileobj(source, self.wfile, length=1024 * 1024)

    def send_mp4_error(self, status: int, message: str, ffmpeg_stderr: str = "", diagnostics: dict | None = None):
        payload = json.dumps(
            {
                "error": message,
                "ffmpegStderr": ffmpeg_stderr,
                **(diagnostics or {}),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def analyze_melody(self, pitch_model: str = "v39"):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        maximum_upload = 512 * 1024 * 1024 if pitch_model == "learning-v431" else 2 * 1024 * 1024 * 1024
        if length <= 0 or length > maximum_upload:
            self.send_error(400, "Invalid audio size")
            return
        filename = Path(unquote(self.headers.get("X-Filename", "source.mp3"))).name
        lyrics_text = ""
        encoded_lyrics = self.headers.get("X-Lyrics-Base64", "")
        if encoded_lyrics:
            try:
                lyrics_text = base64.b64decode(encoded_lyrics, validate=True).decode("utf-8").strip()
            except (ValueError, UnicodeDecodeError):
                self.send_error(400, "Invalid lyrics encoding")
                return
        try:
            requested_bpm = float(self.headers.get("X-BPM", "") or 0)
        except ValueError:
            requested_bpm = 0.0
        if not 40.0 <= requested_bpm <= 240.0:
            requested_bpm = 0.0
        boundary_mode = "boundary-first" if boundary_first_enabled() else "legacy"
        input_signature = hashlib.sha256((
            lyrics_text
            + "\0"
            + (f"{requested_bpm:.6f}" if requested_bpm else "auto")
            + "\0"
            + pitch_model
            + "\0"
            + boundary_mode
            + ("\0GPU-TITAN-v431" if os.environ.get("PIANOFLOW_GPU_READY", "") == "1" else "")
        ).encode("utf-8")).hexdigest()
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mov"}:
            suffix = ".bin"
        ai_python = ROOT / ".venv-pianoflow" / "Scripts" / "python.exe"
        if not ai_python.exists():
            self.send_error(503, "PianoFlow AI environment is not installed")
            return
        if pitch_model == "learning-v431" and not all(
            (ROOT / "models" / "learning-v431" / f"current-{task}.pt").exists()
            for task in ("pitch", "ornament", "boundary")
        ):
            self.send_error(503, "学習済み v4.3.1 音程モデル一式が見つかりません")
            return
        JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - 24 * 60 * 60
        for old_job in JOBS_ROOT.iterdir():
            try:
                if old_job.is_dir() and old_job.stat().st_mtime < cutoff:
                    shutil.rmtree(old_job, ignore_errors=True)
            except OSError:
                pass
        job_id = uuid.uuid4().hex
        job_dir = JOBS_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        try:
            source = job_dir / f"source{suffix}"
            analysis_wav = job_dir / "analysis.wav"
            remaining = length
            source_hash = hashlib.sha256()
            with source.open("wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    output.write(chunk)
                    source_hash.update(chunk)
                    remaining -= len(chunk)
            digest = source_hash.hexdigest()
            (job_dir / "source.sha256").write_text(digest, encoding="ascii")
            (job_dir / "analysis-input.sha256").write_text(input_signature, encoding="ascii")
            lyrics_path = job_dir / "supplied-lyrics.txt"
            if lyrics_text:
                lyrics_path.write_text(lyrics_text, encoding="utf-8")

            # A browser disconnect near the end must not force another 15-minute
            # analysis. Reuse a completed result for the exact same input bytes.
            cached_dirs = sorted(JOBS_ROOT.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
            for cached_dir in cached_dirs:
                if cached_dir == job_dir or not (cached_dir / "analysis.json").exists():
                    continue
                marker = cached_dir / "source.sha256"
                cached_digest = marker.read_text(encoding="ascii").strip() if marker.exists() else ""
                if not cached_digest:
                    cached_sources = list(cached_dir.glob("source.*"))
                    if not cached_sources or cached_sources[0].stat().st_size != source.stat().st_size:
                        continue
                    hasher = hashlib.sha256()
                    with cached_sources[0].open("rb") as cached_source:
                        for chunk in iter(lambda: cached_source.read(1024 * 1024), b""):
                            hasher.update(chunk)
                    cached_digest = hasher.hexdigest()
                    marker.write_text(cached_digest, encoding="ascii")
                if cached_digest != digest:
                    continue
                cached_input = cached_dir / "analysis-input.sha256"
                result_data = json.loads((cached_dir / "analysis.json").read_text(encoding="utf-8"))
                expected_version = 431 if pitch_model == "learning-v431" else ANALYSIS_VERSION
                if int(result_data.get("analysisVersion", 0)) != expected_version:
                    continue
                if not bool(result_data.get("transcription", {}).get("available")):
                    continue
                exact_settings = bool(
                    cached_input.exists()
                    and cached_input.read_text(encoding="ascii").strip() == input_signature
                )
                # Empty lyrics/BPM is the normal audio-only mode; supplied
                # values are an experimental mode. Never mix their caches.
                if not exact_settings:
                    continue
                stem_files = result_data.pop("stemFiles", {})
                result_data["jobId"] = cached_dir.name
                result_data["cached"] = True
                result_data["cacheMatch"] = "exact-settings"
                result_data["stems"] = {
                    name: f"/work/jobs/{cached_dir.name}/{stem_filename}"
                    for name, stem_filename in stem_files.items()
                }
                payload = json.dumps(result_data, ensure_ascii=False).encode("utf-8")
                shutil.rmtree(job_dir, ignore_errors=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            ffmpeg = FFMPEG_OVERRIDE or find_ffmpeg()
            if not ffmpeg:
                self.send_error(500, "FFmpeg not found")
                return
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            conversion = subprocess.run(
                [ffmpeg, "-y", "-i", str(source), "-vn", "-ac", "2", "-ar", "44100", str(analysis_wav)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=flags,
            )
            if conversion.returncode != 0:
                self.send_error(500, "Audio conversion failed")
                return
            command = [
                str(ai_python),
                str(ROOT / "ai_melody.py"),
                str(analysis_wav),
                "--output-dir",
                str(job_dir),
            ]
            if lyrics_text:
                command.extend(["--lyrics-file", str(lyrics_path)])
            if requested_bpm:
                command.extend(["--bpm", str(requested_bpm)])
            if pitch_model == "learning-v431":
                command.extend([
                    "--pitch-model", "learning-v431",
                    "--learning-device", os.environ.get("PIANOFLOW_LEARNING_DEVICE", "auto"),
                ])
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=flags,
            )
            if result.returncode != 0:
                message = result.stderr.decode("utf-8", errors="replace")[-1600:]
                (ROOT / "work" / "analysis-last-error.log").write_text(message, encoding="utf-8")
                payload = json.dumps({"error": message or "AI melody analysis failed"}, ensure_ascii=False).encode("utf-8")
                self.send_response(500)
            else:
                result_data = json.loads(result.stdout.decode("utf-8"))
                stem_files = result_data.pop("stemFiles", {})
                result_data["jobId"] = job_id
                result_data["stems"] = {
                    name: f"/work/jobs/{job_id}/{filename}"
                    for name, filename in stem_files.items()
                }
                payload = json.dumps(result_data, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as error:
            (ROOT / "work").mkdir(parents=True, exist_ok=True)
            (ROOT / "work" / "analysis-last-error.log").write_text(str(error), encoding="utf-8")
            payload = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def align_reference(self):
        """Align a user-supplied score MIDI to an existing audio analysis job."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 32 * 1024 * 1024:
            self.send_error(400, "Invalid reference payload")
            return
        try:
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            job_id = str(request.get("jobId", ""))
            if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
                raise ValueError("解析ジョブが見つかりません。先に音源を解析してください")
            job_dir = JOBS_ROOT / job_id
            analysis_path = job_dir / "analysis.json"
            if not analysis_path.exists():
                raise ValueError("解析結果の保存期限が切れています。音源を再解析してください")
            encoded = str(request.get("midiBase64", ""))
            midi_bytes = base64.b64decode(encoded, validate=True)
            if not midi_bytes.startswith(b"MThd") or len(midi_bytes) > 16 * 1024 * 1024:
                raise ValueError("有効なMIDIファイルを選択してください")
            midi_path = job_dir / "reference.mid"
            config_path = job_dir / "reference.json"
            alignment_path = job_dir / "reference-alignment.json"
            midi_path.write_bytes(midi_bytes)
            lyrics = request.get("lyrics", [])
            if isinstance(lyrics, str):
                lyrics = [line.strip() for line in lyrics.splitlines() if line.strip()]
            raw_bpm = request.get("bpm")
            supplied_bpm = (
                max(20.0, min(400.0, float(raw_bpm)))
                if raw_bpm not in (None, "")
                else None
            )
            config = {
                "name": str(request.get("name") or "dataset-reference")[:120],
                "midi": midi_path.name,
                # With no explicit experimental BPM, preserve the tempo embedded
                # in the reference MIDI instead of silently forcing 120 BPM.
                "bpm": supplied_bpm,
                "lyrics": lyrics if isinstance(lyrics, list) else [],
            }
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            ai_python = ROOT / ".venv-pianoflow" / "Scripts" / "python.exe"
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            result = subprocess.run(
                [str(ai_python), str(ROOT / "align_score_to_audio.py"), str(config_path), str(analysis_path), "--output", str(alignment_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=flags,
            )
            if result.returncode != 0 or not alignment_path.exists():
                raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-1600:] or "MIDI同期に失敗しました")
            alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
            alignment["alignmentUrl"] = f"/work/jobs/{job_id}/{alignment_path.name}"
            payload = json.dumps(alignment, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
        except Exception as error:
            payload = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")
            self.send_response(400)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    global FFMPEG_OVERRIDE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4176)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    FFMPEG_OVERRIDE = args.ffmpeg
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
