# PianoFlow — MIDIから、ピアノ動画へ

## 概要

PianoFlowは、MIDI音源を入力し演奏を見やすいピアノ動画にし、色や歌詞を画面上で編集できるものです。音声からメロディMIDIを作り、自動でピアノ動画に変換するAI解析機能も搭載されています。

## 操作デモ

**操作デモ:** [GitHub Pages版を開く](https://yamadatarou11.github.io/pianoflow/demo/?sample=1)

## 操作動画

https://github.com/user-attachments/assets/21d18149-cc3b-440e-841d-f3dacbfa32a5

[操作ガイド動画（MP4）](assets/PianoFlow_使い方_ゆっくり操作ガイド.mp4)

動画では、MIDIの読み込みと再生、バーの色変更、歌詞の割り当て、発光タイミングの調整、動画保存を紹介しています。

## 音源クレジット

ピアノ音源には、Alexander Holm作「Salamander Grand Piano V3」（CC BY 3.0）のTone.js配布サンプルを使用しています。サンプルはMP3およびオフライン再生用の埋め込みデータとして同梱し、再生時に音程を移調します。詳しい出典・表示は[第三者素材のクレジット](THIRD_PARTY_NOTICES.md)を参照してください。

## 主な機能

- MIDIファイルの読み込みとピアノロール再生
- バー・鍵盤の色、ドレミ表示などの編集
- 曲名・歌詞の表示と、歌詞のバーへの割り当て
- 文字単位の歌詞発光タイミングの調整
- 編集設定の保存と、音声付きピアノ動画の出力
- 音声からのボーカル分離、歌詞文字起こし、メロディMIDI推定

## 技術選定

| 処理 | 技術 | 用途 | 選定理由 |
| --- | --- | --- | --- |
| MIDI動画 | HTML・CSS・JavaScript | 画面と編集操作 | ブラウザーだけで動かすため。 |
| MIDI動画 | Standard MIDI File読み込み | 音符・テンポの取得 | MIDIの時刻と音高を保つため。 |
| MIDI動画 | Canvas 2D | バー・鍵盤・歌詞の描画 | 再生に合わせて画面を描くため。 |
| MIDI動画 | Web Audio API・ピアノ音源 | 演奏音の再生 | MIDIと音を同期するため。 |
| MIDI動画 | `MediaRecorder`・`canvas.captureStream()` | 映像・音声の記録 | 再生内容を動画にするため。 |
| 共通 | Python HTTPサーバー・FFmpeg | 音声処理・MP4出力 | 長時間処理と動画変換を行うため。 |
| MIDI動画 | `localStorage` | 編集設定の保存 | 同じブラウザーで編集を再開するため。 |
| 音声AI | `audio-separator`（RoFormer）・Demucs | ボーカル分離 | 歌声を音程・歌詞解析に使うため。 |
| 音声AI | faster-whisper `large-v3-turbo` | 歌詞・単語時刻の推定 | 日本語音声を文字起こしするため。 |
| 音声AI | RMVPE・CREPE・WORLD・pYIN | 音程候補の抽出 | 複数候補を比較するため。 |
| 音声AI | 音量・スペクトル・子音／母音特徴 | 発声区間・音符境界の推定 | 音声の変化から区切りを判断するため。 |
| 音声AI | TCN・PyTorch | 音程・装飾・境界の学習と推論 | 前後フレームを使って時系列を判定するため。 |

### MIDIからピアノ動画を作る流れ

MIDIの音符とテンポを読み、バー・鍵盤を描画して音と同期した動画にします。歌詞は音符へ割り当て、バーから調整できます。

```text
MIDI → 音符・テンポを読み込む → バー・鍵盤・歌詞を描画
     → Web Audio APIで演奏 → 映像と音声を記録 → MP4を出力
```

### 音声からメロディMIDIを作る流れ

歌詞認識で文字と時刻を得て、音響特徴から音程を推定します。歌詞の時刻は境界候補に使い、音程は音声から判断します。

```text
音声入力 → FFmpegで解析用音声へ変換 → ボーカルを分離
         → 歌詞と単語時刻を認識 → 音響特徴から音程候補を抽出
         → 歌詞境界・母音・子音・音量・スペクトル変化から音符境界を調整
         → ノート列をMIDI出力 → PianoFlowで試聴・動画化
```

通常解析（v39）は複数の音程推定器を組み合わせます。TCN解析（v4.3.1）は別経路で、v39の後段に追加する構成ではありません。

### TCNの学習方法

12曲分のボーカル音源と正解MIDIを使い、音程・装飾・境界を段階的に学習・評価しました。重みを引き継ぎ、過去曲の再学習と弱点を補う合成データを加えています。

GPUは必須ではありません。MIDI編集とAI解析はCPUで動作します。GPU版ではTCN推論をCUDAで行います。GTX TITANには専用の互換環境が必要です。

## 動かし方

PianoFlowはGitHubからソースコードを取得し、自分のWindows PC上で実行します。音声AI解析を行うPCにはPython環境、必要なモデル、FFmpegを用意します。

動作環境はWindows、ChromeまたはEdge、FFmpegです。現在の構成では、アプリ側にPython 3.8系、音源分離・文字起こし・高品質音程推定側にPython 3.10系の仮想環境を使います。GPUは必須ではありません。MIDI編集はGPUなしで動き、AI解析もCPUで実行できますが、音声分離や認識には時間がかかります。

### MIDIの読み込み・編集・動画出力

1. リポジトリを取得します。

   ```powershell
   git clone https://github.com/yamadatarou11/pianoflow.git
   cd pianoflow
   ```

2. MIDIを読み込んで編集・試聴する場合は、`PianoFlow_v126.html`をChromeまたはEdgeで開きます。
3. Windowsランチャーから一連の機能を使う場合は、`PianoFlow Windows版.bat`を起動します。MP4出力にはローカルPython環境とFFmpegが必要です。

### 音声からのAI解析

1. Python 3.8系と3.10系でAI解析用の仮想環境を用意し、依存パッケージと音源分離・音程推定モデルを準備します。起動スクリプトは`.venv-pianoflow`と`.venv-pianoflow-hq`を参照します。これらの仮想環境自体はGitHubへ登録せず、PC上に構築します。
2. FFmpegをインストールし、コマンドから実行できるようにします。
3. CPUで解析する場合は`PianoFlow AI解析版.bat`を起動し、音声を選択して解析します。faster-whisperはCUDAが使えない環境ではCPUへ切り替わります。
4. GPU推論を使う場合は、対応するNVIDIA GPU・CUDA対応PyTorch環境とTCNの学習済み重みを準備して`PianoFlow AI GPU版.bat`を起動します。スクリプトがCUDA演算とモデルを事前確認し、確認に失敗した場合は解析を開始しません。

初回実行時は、音声認識や音源分離のモデルを取得するためインターネット接続が必要な場合があります。取得後の音声解析はローカルPCで行います。学習用音源、正解MIDI、解析キャッシュ、Python仮想環境は、サイズや権利の都合上、ソースコードと分けて管理します。TCN重みが必要なモードでは、`models/learning-v431/`に音程・装飾・境界の3つのチェックポイントを配置します。

## リポジトリ構成

```text
.
├── index.html                 デモへの転送ページ
├── PianoFlow_v126.html        MIDI編集・動画作成
├── PianoFlow_AI*.html         音声AI解析画面
├── pianoflow_server.py        ローカル処理・動画出力
├── ai_melody.py・hq_*.py      音声解析処理
├── learning_v*.py             学習・評価コード
├── apply_learning_v431_dataset.py・pianoflow_gpu_*.py  TCN推論
├── piano_samples_embedded.js  MIDIアプリ用の埋め込みピアノ音源
├── piano-samples/             ピアノサンプルMP3（30ファイル）
├── assets/                    操作ガイド動画
├── demo/                      GitHub Pages用MIDIデモ
│   └── piano_samples_embedded.js  デモ単体で動かすための同一コピー
├── THIRD_PARTY_NOTICES.md     ピアノ音源の出典・ライセンス
├── PianoFlow*.bat・*.ps1      Windows起動用
└── models/README.md           学習済み重みの配置説明（重み本体は別途準備）
```

`piano_samples_embedded.js` は、ルートのMIDIアプリ用と `demo/` の静的デモ用に同じ内容を置いています。デモを別パスから開いても相対パスで音源を読み込めるようにするための意図的な重複です。
