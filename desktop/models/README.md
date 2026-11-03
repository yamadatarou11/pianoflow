# 学習済みモデル

学習済み重みは容量が大きいため、このリポジトリには含めていません。

TCN推論を使う場合は、別途用意した重みをリポジトリのルートを基準に、次の場所へ配置してください。

```text
desktop/models/learning-v431/current-pitch.pt
desktop/models/learning-v431/current-ornament.pt
desktop/models/learning-v431/current-boundary.pt
```

音程推定・音源分離・文字起こしの外部モデルは、実行時に各ライブラリの仕様に従って取得します。モデルごとの配布条件を確認してください。
