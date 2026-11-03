from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "dataset-work"
STATUS = WORK / "v431-batch-status.json"
MANIFEST = WORK / "manifest.json"


def write_status(**values) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if STATUS.exists():
        try:
            current = json.loads(STATUS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current.update(values)
    current["updatedAt"] = time.time()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def run_step(stage: str, message: str, command: list[str]) -> None:
    write_status(state="running", stage=stage, message=message, error="")
    process = subprocess.Popen(
        command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        match = re.search(r"\[(\d+)/(\d+)\]\s*(?:[^:]+:\s*)?(.+)", line.strip())
        if match:
            current, total, song = int(match.group(1)), int(match.group(2)), match.group(3).strip()
            write_status(state="running", stage=stage, current=max(0, current - 1), total=total, song=song, message=f"{current}/{total} {song}")
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{stage} failed (exit={return_code})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and run the PianoFlow v4.3.1 listening batch")
    parser.add_argument("--refresh-base", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    started = time.time()
    try:
        if args.refresh_base:
            run_step("prepare", "データセット台帳を作成しています", [python, str(ROOT / "pianoflow_dataset.py"), "prepare"])
            run_step("base-analysis", "全曲の基礎音源解析を実行しています", [python, str(ROOT / "pianoflow_dataset.py"), "analyze", "--continue-on-error"])
            run_step("base-evaluation", "正解MIDIとの基準評価を作成しています", [python, str(ROOT / "pianoflow_dataset.py"), "evaluate", "--continue-on-error"])
        elif not MANIFEST.exists():
            raise FileNotFoundError("dataset-work/manifest.json がありません。フォルダ登録から開始してください")
        run_step(
            "v431-inference", "v4.3.1で全曲を解析しています",
            [python, str(ROOT / "apply_learning_v431_dataset.py"), "--manifest", str(MANIFEST), "--status", str(STATUS)],
        )
        return 0
    except Exception as error:
        write_status(state="failed", stage="failed", message="一括解析に失敗しました", error=str(error), seconds=round(time.time() - started, 2))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
