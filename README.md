# Deep_CNN

Deep_CNN は、斎藤康毅 著『ゼロから作る Deep Learning』を読み進めながら手を動かして実装した、MNIST データセットなどの画像認識タスク向けに構築したシンプルな畳み込みニューラルネットワーク（CNN）実装です。  
Python でレイヤーや活性化関数、損失関数、最適化アルゴリズム一つずつ自作し、理論と実装の理解を深めることを目的としています。

---
## 🔨 ネットワーク構成

本実装の畳み込みニューラルネットワーク（DeepConvNet）は、以下のような層構成になっています。  

- **Conv ブロック ×3**  
  各ブロック内で 2 回の畳み込み＋バッチ正規化＋ReLU → プーリング
  - [Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → Pool] × 3
    
- **全結合層 ×2**  
  - Affine → BatchNorm → ReLU → Dropout  
  - Affine → Dropout → Softmax  

詳細は `models/deep_CNN.py` のクラス定義コメントをご覧ください。

---

## 🚀 特徴

- NumPy ベースで外部フレームワークに依存しない  
- 自作のレイヤー（畳み込み、全結合、プーリングなど）を実装  
- 活性化関数（ReLU, Softmax）・損失関数（クロスエントロピー）を自作  
- 最適化アルゴリズム（SGD, Momentum）をサポート  
- モジュール構造で拡張性あり

---

## 📝 ディレクトリ構成
```
Deep_CNN/
├── common/ # 共有モジュール
│ ├── init.py
│ ├── function.py # 活性化関数・損失関数定義
│ ├── layers.py # 自作レイヤー群
│ ├── mnist.pkl # MNIST データセット（pickle 形式）
│ ├── mnist.py # MNIST 取り込み・前処理用スクリプト
│ ├── optimization.py # 最適化アルゴリズム（SGD, Momentum 等）
│ └── trainer.py # 訓練ループ・評価ロジック
├── models/
│ ├── init.py
│ └── deep_CNN.py # ネットワーク構築＆順伝播・逆伝播
├── train_deep_CNN.py # 訓練ループ＆評価スクリプト
├── deep_convnet_params.pkl # 学習済みモデルパラメータ（サンプル）
├── README.md # このファイル
└── requirements.txt # 必要パッケージ
```

---

## 🔧 必要環境

- Python 3.6 以上  
- NumPy  
