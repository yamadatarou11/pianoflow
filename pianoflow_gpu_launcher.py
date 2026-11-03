from __future__ import annotations

import argparse
import http.client
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local PianoFlow GPU server and open its page.")
    parser.add_argument("--port", type=int, default=4177)
    args = parser.parse_args()
    server_script = ROOT / "pianoflow_server.py"
    if not server_script.is_file():
        print("サーバーファイルが見つかりません: " + str(server_script))
        return 1

    print("PianoFlow GPUサーバーを起動しています。")
    try:
        server = subprocess.Popen(
            [sys.executable, str(server_script), "--port", str(args.port)],
            cwd=str(ROOT),
            env=os.environ.copy(),
        )
    except OSError as error:
        print("サーバーを起動できませんでした: " + str(error))
        return 1

    # Do not treat a live Python process as proof that the HTTP server is ready.
    # Confirm that the requested page is actually reachable before opening the
    # browser. This also catches bind errors and slow startup on older PCs.
    deadline = time.monotonic() + 30.0
    last_error = "まだ接続できていません"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            print("サーバーが起動直後に終了しました。上に表示されたエラーを確認してください。")
            return int(server.returncode or 1)
        connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=1.0)
        try:
            connection.request("GET", "/PianoFlow_AI_GPU.html")
            response = connection.getresponse()
            response.read(128)
            if response.status == 200:
                break
            last_error = "HTTP {}".format(response.status)
        except OSError as error:
            last_error = "{}: {}".format(type(error).__name__, error)
        finally:
            connection.close()
        time.sleep(0.5)
    else:
        print("サーバーの待受を確認できませんでした。ブラウザーは開きません。")
        print("確認先: http://127.0.0.1:{}/PianoFlow_AI_GPU.html".format(args.port))
        print("最終エラー: " + last_error)
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        return 1

    url = "http://127.0.0.1:{}/PianoFlow_AI_GPU.html".format(args.port)
    print("サーバーを起動しました。ブラウザーを開きます: " + url)
    try:
        if not webbrowser.open(url, new=2):
            print("自動でブラウザーを開けませんでした。上記URLをブラウザーへ入力してください。")
        print("利用中はこの黒い画面を閉じないでください。終了するには Ctrl+C を押します。")
        server.wait()
        return int(server.returncode or 0)
    except KeyboardInterrupt:
        print("PianoFlow GPUサーバーを終了しています。")
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
