# Deep_CNN

Deep_CNN は、MNIST データセットなどの画像認識タスク向けに構築したシンプルな畳み込みニューラルネットワーク（CNN）実装です。  
Python でゼロからレイヤーや活性化関数、損失関数、最適化アルゴリズムを実装し、動作を学ぶことを目的としています。

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
