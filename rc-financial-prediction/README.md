# 📈 RC Financial Prediction (Reservoir Computing)

> **[🇯🇵 日本語の説明はこちら](#japanese-description)**

## 📖 Overview

A Python-based **Reservoir Computing (Echo State Network)** system for financial market prediction, featuring:
- **yfinance** data acquisition (FX, stocks, indices)
- Walk-forward backtesting with spread cost modeling
- Tomorrow's direction prediction with confidence score
- Mock data mode for offline testing

---

## 📊 Architecture

```
yfinance API (or Mock CSV)
    ↓
Technical Features (RSI, ATR, Bollinger, MA ratios, Volatility)
    ↓
Echo State Network (Reservoir Computing)
    ↓
Direction Prediction (UP / DOWN) + Confidence
    ↓
Walk-Forward Backtest → Performance Metrics → Excel Report
```

### Key Features
- **ESN Model:** 200-unit reservoir with leaky integrator neurons
- **Features:** Return, Volatility, MA ratios, RSI, ATR, Bollinger position
- **Backtest:** Walk-forward with 250-bar training window, 20-bar step
- **Spread Cost:** Configurable pips (default 3.0 for USDJPY)
- **Fallback:** Built-in SimpleESN when reservoirpy unavailable

---

## 🛠 Scripts

| Script | Purpose |
|---|---|
| `rc_financial_prediction.py` | Main: fetch data → ESN → backtest → predict tomorrow |
| `generate_mock_data.py` | Generate synthetic OHLCV data for offline testing |

### Supported Tickers
- **USDJPY** (`USDJPY=X`)
- **EURUSD** (`EURUSD=X`)
- **Nikkei 225** (`^N225`)

---

## 💻 Usage

### Live Mode (with internet)
```bash
pip install numpy pandas scipy yfinance openpyxl

python rc_financial_prediction.py
```

### Mock Mode (offline testing)
```bash
python generate_mock_data.py            # → ./data/mock_*.csv
python rc_financial_prediction.py --mock  # → ./output/*.xlsx
```

### Output Files (in `./output/`)
| File | Description |
|---|---|
| `rc_prediction_results.xlsx` | Full results (predictions, backtest, performance) |
| `tomorrow_predictions.csv` | Tomorrow's direction predictions |
| `performance_summary.csv` | Backtest performance metrics |

### Performance Metrics
| Metric | Description |
|---|---|
| Accuracy | Directional hit rate |
| Sharpe Ratio | Risk-adjusted return |
| Profit Factor | Gross profit / Gross loss |
| Max Drawdown | Largest peak-to-trough |

---

## 📦 Requirements

```
numpy, pandas, scipy, yfinance, openpyxl
# Optional: reservoirpy (enhanced ESN)
```

## 📚 References

- Jaeger (2001) *The "Echo State" Approach to Analysing and Training Recurrent Neural Networks*
- Lopez de Prado (2018) *Advances in Financial Machine Learning* — Fractional Differentiation

## 👨‍💻 Author

**Go Sato** — Data Scientist | Reservoir Computing, Algorithmic Trading, Financial Engineering

---

<a name="japanese-description"></a>

# 📈 RC金融予測（リザーバコンピューティング）

yfinanceからデータを取得し、Echo State Network（ESN）で翌日の方向を予測、ウォークフォワードバックテストを実行するPythonシステム。

```bash
# ライブモード（インターネット接続時）
python rc_financial_prediction.py

# モックモード（オフラインテスト）
python generate_mock_data.py
python rc_financial_prediction.py --mock
```

**佐藤 剛 (Go Sato)** — データサイエンティスト
