# GitHubアップロード用ファイル

このフォルダー内をリポジトリのルートとして使用します。

## 含めたもの

- `PianoFlow_v126.html` と `piano_samples_embedded.js`: MIDI編集・ピアノ動画作成
- `PianoFlow_AI*.html`、`pianoflow_server.py`、`ai_melody.py`、`hq_*.py`: 音声解析
- `learning_v*.py`、`apply_learning_v431_dataset.py`、`pianoflow_gpu_*.py`: TCN推論
- `PianoFlow*.bat`、`pianoflow_windows.ps1`: Windows起動用
- `index.html`、`demo/`、`assets/`: 紹介ページ、操作デモ、操作動画
- `piano-samples/`: CC BY 3.0のSalamander Grand Pianoサンプル音源
- `THIRD_PARTY_NOTICES.md`: サンプル音源の出典・ライセンス表示

## 公開前に確認すること

- `models/README.md` に記載した学習済み重みは容量のため含めていません。
- 学習データ、正解MIDI、解析出力、仮想環境、キャッシュは含めていません。
- ピアノサンプルはSalamander Grand Piano V3（Alexander Holm、CC BY 3.0）として出典を特定し、クレジットを追加しました。クレジットを削除せず、動画・サイト・サンプルデータをまとめて配布してください。
- 操作動画はPianoFlowの画面録画、字幕、紹介用にCodexが作成したデモMIDI、アプリ内ピアノ再生で構成されています。制作フォルダー内に外部MIDIや外部映像素材を取得した記録は見つかりませんでした。ピアノ音源のクレジットは動画・サイト・サンプルデータと一緒に保持してください。
- このフォルダーはアップロード用に整理したものです。GitHubへの送信・公開はまだ行っていません。
