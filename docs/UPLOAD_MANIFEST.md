# 配布ファイル一覧

リポジトリのルートには公開デモへの入口と案内を置き、ローカル版を `desktop/` にまとめています。以下のパスはリポジトリのルートを基準とします。

## 含めるもの

- `index.html`: `./demo/?sample=1` への転送ページ
- `README.md`: 概要、操作動画、起動方法、構成、音源クレジット
- `.gitignore`、`.nojekyll`: ローカルデータの除外、GitHub Pagesの配信設定
- `demo/`: 静的MIDIデモのHTML、JavaScript、CSS、説明、埋め込みピアノ音源
- `desktop/PianoFlow_v126.html`、`desktop/piano_samples_embedded.js`: MIDI編集・動画作成
- `desktop/PianoFlow_AI*.html`、`desktop/pianoflow_server.py`、`desktop/ai_melody.py`、`desktop/hq_*.py`: 音声解析・ローカル処理
- `desktop/learning_v*.py`、`desktop/apply_learning_v431_dataset.py`、`desktop/pianoflow_gpu_*.py`: 学習モデル・推論
- `desktop/export_learning_v431_gpu_checkpoints.py`: GPU互換用チェックポイントの書き出し
- `desktop/pianoflow_dataset.py`、`desktop/align_score_to_audio.py`、`desktop/run_dataset_v431_review.py`: データセット管理・位置合わせ・評価コード
- `desktop/PianoFlow*.bat`、`desktop/pianoflow_windows.ps1`: Windows起動用
- `desktop/piano-samples/`: Salamander Grand Piano MP3サンプル30個
- `desktop/models/README.md`: 別途用意する学習済み重みの配置説明
- `assets/PianoFlow_使い方_ゆっくり操作ガイド.mp4`: 操作ガイド動画
- `assets/styles.css`: 旧紹介ページのスタイル（現在の転送ページでは未使用）
- `THIRD_PARTY_NOTICES.md`: ピアノ音源の出典・ライセンス
- `docs/UPLOAD_MANIFEST.md`: この一覧

## 配布時の注意

- 学習データ、正解MIDI、解析出力、仮想環境、キャッシュ、モデル重みは含めません。
- ローカル版の仮想環境は `desktop/.venv-pianoflow/`、`desktop/.venv-pianoflow-hq/` に作成します。生成される学習データや解析出力も `desktop/` 内で管理されます。
- ピアノ音源はAlexander Holm作「Salamander Grand Piano V3」（CC BY 3.0）のTone.js配布サンプルです。動画・サイト・サンプルデータの配布時も作者名・出典・ライセンス表示を保持してください。
- 操作動画の紹介用デモMIDIと、動画内のSalamander Grand Piano音源については `THIRD_PARTY_NOTICES.md` を参照してください。
- `desktop/piano_samples_embedded.js` と `demo/piano_samples_embedded.js` は、各アプリが相対パスで独立して読み込むための意図的な同一コピーです。

## 既存の不足ファイル

学習・評価・位置合わせコードは `evaluate_pitch_tracking.py`、`evaluate_synchronized_timing.py`、`initial_score_alignment.py` を参照しますが、元の提出物にこれらは含まれていません。該当処理を使うには別途用意する必要があります。今回の構成整理では、新規実装や代替ファイルの追加は行っていません。
