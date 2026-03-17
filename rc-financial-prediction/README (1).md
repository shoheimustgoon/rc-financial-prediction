# 🧠 Reservoir Computing for Financial Market Prediction

> **[🇯🇵 日本語の説明はこちら (Click here for Japanese Description)](#japanese-description)**

## 📖 Overview

A financial market prediction system built on **Reservoir Computing (Echo State Networks)**. Supports multi-asset prediction with automated walk-forward backtesting.

### What is Reservoir Computing?

Reservoir Computing (RC) is a computationally efficient recurrent neural network framework:

- **Reservoir** (hidden layer): Fixed random weights — NOT trained
- **Output layer**: Trained via simple linear regression
- **Result**: Orders of magnitude faster than LSTM/GRU

```
Input → [Fixed Random Reservoir] → Trained Readout → Prediction
```

### Supported Assets

| Asset | Description |
|---|---|
| USD/JPY | FX |
| EUR/JPY | FX |
| Nikkei 225 | Equity Index |
| Bitcoin/JPY | Cryptocurrency |

---

## 🔬 Approach

- Multiple RC variants with ensemble averaging
- Automated hyperparameter optimization
- Walk-forward backtesting (expanding window, no future leakage)
- Feature engineering: technical indicators, fractional differentiation, Hurst exponent

---

## 📦 Key Libraries

```
reservoirpy, numpy, pandas, scikit-learn, yfinance, matplotlib
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. NOT financial advice. Past performance does not guarantee future results.

---

## 📚 References

1. Jaeger (2001) "The echo state approach" GMD Report 148
2. Lukoševičius & Jaeger (2009) "Reservoir computing approaches" Computer Science Review
3. Gallicchio & Micheli (2017) "Deep Echo State Network" Information Sciences
4. López de Prado (2018) "Advances in Financial Machine Learning" Wiley

---

## 👨‍💻 Author

**Go Sato** — Data Scientist

---

---

<a name="japanese-description"></a>

# 🧠 リザバーコンピューティングによる金融市場予測

## 📖 概要

**リザバーコンピューティング（Echo State Network）** を用いた金融市場予測システムです。複数アセットの予測とウォークフォワードバックテストに対応しています。

### リザバーコンピューティングとは？

- **リザバー**（隠れ層）：固定ランダム重み（学習しない）
- **出力層のみ**を線形回帰で学習
- LSTM/GRUと比べ**桁違いに高速**

### 対応アセット

| アセット | 種別 |
|---|---|
| ドル/円 | FX |
| ユーロ/円 | FX |
| 日経225 | 株価指数 |
| ビットコイン/円 | 暗号通貨 |

---

## 🔬 アプローチ

- 複数のRC手法をアンサンブル
- ハイパーパラメータ自動最適化
- ウォークフォワードバックテスト（拡張ウィンドウ方式）
- 特徴量: テクニカル指標、分数階差分、ハースト指数

---

## ⚠️ 免責事項

**教育・研究目的のみ**です。投資助言ではありません。

---

## 👨‍💻 Author

**佐藤 剛 (Go Sato)** — データサイエンティスト
