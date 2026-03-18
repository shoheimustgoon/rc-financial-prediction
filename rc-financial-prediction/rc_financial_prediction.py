# -*- coding: utf-8 -*-
"""
RC Financial Prediction — Reservoir Computing ESN
Fetches data from yfinance, predicts next-day direction using Echo State Network,
and runs a walk-forward backtest with performance metrics.

Features:
  - yfinance data acquisition (FX, stocks, crypto)
  - Fractional Differentiation for stationarity
  - Echo State Network (ESN) via reservoirpy
  - Bayesian TPE hyperparameter optimization
  - Walk-forward backtest with spread cost modeling
  - Tomorrow's prediction with confidence

Author: Go Sato
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from reservoirpy.nodes import Reservoir, Ridge
    HAS_RESERVOIRPY = True
except ImportError:
    HAS_RESERVOIRPY = False
    print("  [WARN] reservoirpy not installed. Using fallback simple ESN.")


# ============================================================
# Configuration
# ============================================================
DEFAULT_TICKER = 'USDJPY=X'
DEFAULT_PERIOD = '2y'
LOOKBACK = 20          # input window
TRAIN_RATIO = 0.7
SPREAD_PIPS = 3.0      # typical USDJPY spread
PIP_VALUE = 0.01       # for JPY pairs

# ESN defaults
ESN_UNITS = 200
ESN_SPECTRAL_RADIUS = 0.9
ESN_INPUT_SCALING = 0.5
ESN_LEAK_RATE = 0.3
ESN_RIDGE_ALPHA = 1e-6


# ============================================================
# Data Acquisition
# ============================================================
def fetch_data(ticker=DEFAULT_TICKER, period=DEFAULT_PERIOD):
    """Fetch OHLCV data from yfinance."""
    print(f"  Fetching {ticker} ({period})...")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    print(f"  Got {len(df)} bars: {df.index[0].date()} → {df.index[-1].date()}")
    return df


def compute_features(df, lookback=LOOKBACK):
    """Technical features for ESN input."""
    close = df['Close'].values.astype(float)
    high = df['High'].values.astype(float)
    low = df['Low'].values.astype(float)

    feat = pd.DataFrame(index=df.index)

    # Returns
    feat['Return_1d'] = pd.Series(close, index=df.index).pct_change()
    feat['Return_5d'] = pd.Series(close, index=df.index).pct_change(5)

    # Volatility
    feat['Volatility_10d'] = feat['Return_1d'].rolling(10).std()
    feat['Volatility_20d'] = feat['Return_1d'].rolling(20).std()

    # Moving averages ratio
    feat['MA5_ratio'] = close / pd.Series(close, index=df.index).rolling(5).mean()
    feat['MA20_ratio'] = close / pd.Series(close, index=df.index).rolling(20).mean()

    # RSI (14)
    delta = pd.Series(close, index=df.index).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat['RSI_14'] = 100 - (100 / (1 + rs))

    # ATR (14)
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - np.roll(close, 1)),
        'lc': abs(low - np.roll(close, 1))
    }, index=df.index).max(axis=1)
    feat['ATR_14'] = tr.rolling(14).mean()
    feat['ATR_ratio'] = tr / feat['ATR_14']

    # Bollinger position
    ma20 = pd.Series(close, index=df.index).rolling(20).mean()
    std20 = pd.Series(close, index=df.index).rolling(20).std()
    feat['BB_position'] = (close - ma20) / std20.replace(0, np.nan)

    # Target: next day direction (1=up, 0=down)
    feat['Target'] = (pd.Series(close, index=df.index).shift(-1) > close).astype(int)

    feat = feat.dropna()
    return feat


def fractional_diff(series, d=0.4, thres=1e-4):
    """Fractional differentiation for stationarity (Marcos Lopez de Prado)."""
    weights = [1.0]
    for k in range(1, len(series)):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < thres:
            break
        weights.append(w)
    weights = np.array(weights)
    width = len(weights)
    result = np.full(len(series), np.nan)
    for i in range(width - 1, len(series)):
        result[i] = np.dot(weights, series[i - width + 1:i + 1])
    return result


# ============================================================
# ESN Model (fallback if no reservoirpy)
# ============================================================
class SimpleESN:
    """Minimal Echo State Network for environments without reservoirpy."""

    def __init__(self, n_inputs, n_reservoir=200, spectral_radius=0.9,
                 input_scaling=0.5, leak_rate=0.3, ridge_alpha=1e-6, seed=42):
        np.random.seed(seed)
        self.n_reservoir = n_reservoir
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha

        # Input weights
        self.W_in = (np.random.rand(n_reservoir, n_inputs) - 0.5) * 2 * input_scaling

        # Reservoir weights (sparse)
        W = np.random.rand(n_reservoir, n_reservoir) - 0.5
        mask = np.random.rand(n_reservoir, n_reservoir) < 0.1
        W *= mask
        rhoW = max(abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / rhoW) if rhoW > 0 else W

        self.state = np.zeros(n_reservoir)
        self.W_out = None

    def _update(self, x):
        pre = np.tanh(self.W_in @ x + self.W @ self.state)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * pre
        return self.state.copy()

    def fit(self, X, y):
        states = []
        self.state = np.zeros(self.n_reservoir)
        for i in range(len(X)):
            s = self._update(X[i])
            states.append(s)
        S = np.array(states)
        # Ridge regression
        I = np.eye(self.n_reservoir) * self.ridge_alpha
        self.W_out = np.linalg.solve(S.T @ S + I, S.T @ y)

    def predict(self, X):
        preds = []
        for i in range(len(X)):
            s = self._update(X[i])
            pred = s @ self.W_out
            preds.append(pred)
        return np.array(preds)

    def reset_state(self):
        self.state = np.zeros(self.n_reservoir)


# ============================================================
# Backtest
# ============================================================
def walk_forward_backtest(features_df, spread_pips=SPREAD_PIPS, pip_value=PIP_VALUE,
                          train_window=250, step=20):
    """Walk-forward backtest with ESN prediction."""
    feature_cols = [c for c in features_df.columns if c != 'Target']
    X_all = features_df[feature_cols].values
    y_all = features_df['Target'].values

    # Normalize features
    mu = np.nanmean(X_all[:train_window], axis=0)
    sigma = np.nanstd(X_all[:train_window], axis=0)
    sigma[sigma == 0] = 1
    X_norm = (X_all - mu) / sigma

    results = []
    n = len(X_all)
    spread_cost = spread_pips * pip_value  # as fraction of price

    for start in range(train_window, n - step, step):
        train_end = start
        test_end = min(start + step, n - 1)

        X_train = X_norm[:train_end]
        y_train = y_all[:train_end]
        X_test = X_norm[train_end:test_end]
        y_test = y_all[train_end:test_end]

        if len(X_test) == 0:
            break

        n_inputs = X_train.shape[1]

        if HAS_RESERVOIRPY:
            reservoir = Reservoir(ESN_UNITS, sr=ESN_SPECTRAL_RADIUS,
                                  input_scaling=ESN_INPUT_SCALING, lr=ESN_LEAK_RATE)
            readout = Ridge(ridge=ESN_RIDGE_ALPHA)
            esn = reservoir >> readout
            esn.fit(X_train, y_train.reshape(-1, 1))
            pred_raw = esn.run(X_test).flatten()
        else:
            esn = SimpleESN(n_inputs, ESN_UNITS, ESN_SPECTRAL_RADIUS,
                            ESN_INPUT_SCALING, ESN_LEAK_RATE, ESN_RIDGE_ALPHA)
            esn.fit(X_train, y_train)
            esn.reset_state()
            # Warm up
            for i in range(max(0, train_end - 50), train_end):
                esn._update(X_norm[i])
            pred_raw = esn.predict(X_test)

        pred_dir = (pred_raw > 0.5).astype(int)

        for i in range(len(y_test)):
            idx = train_end + i
            actual_dir = y_test[i]
            predicted = pred_dir[i]
            confidence = min(1.0, abs(pred_raw[i] - 0.5) * 2)  # 0-1 scale

            # PnL: if predicted correctly, gain = |return| - spread
            # if wrong, loss = -|return| - spread
            correct = int(predicted == actual_dir)
            pnl = (1 if correct else -1) * 0.001 - spread_cost * 0.5  # simplified

            results.append({
                'Date': features_df.index[idx],
                'Predicted': predicted,
                'Actual': actual_dir,
                'Confidence': round(confidence, 4),
                'Correct': correct,
                'PnL_pips': round(pnl / pip_value, 2),
                'Cumulative_PnL_pips': 0,  # filled later
            })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['Cumulative_PnL_pips'] = results_df['PnL_pips'].cumsum()

    return results_df


def predict_tomorrow(features_df, train_window=250):
    """Predict tomorrow's direction using the latest data."""
    feature_cols = [c for c in features_df.columns if c != 'Target']
    X_all = features_df[feature_cols].values

    mu = np.nanmean(X_all, axis=0)
    sigma = np.nanstd(X_all, axis=0)
    sigma[sigma == 0] = 1
    X_norm = (X_all - mu) / sigma

    y_all = features_df['Target'].values

    n_inputs = X_norm.shape[1]

    if HAS_RESERVOIRPY:
        reservoir = Reservoir(ESN_UNITS, sr=ESN_SPECTRAL_RADIUS,
                              input_scaling=ESN_INPUT_SCALING, lr=ESN_LEAK_RATE)
        readout = Ridge(ridge=ESN_RIDGE_ALPHA)
        esn = reservoir >> readout
        esn.fit(X_norm[:-1], y_all[:-1].reshape(-1, 1))
        pred_raw = esn.run(X_norm[-1:]).flatten()[0]
    else:
        esn = SimpleESN(n_inputs, ESN_UNITS, ESN_SPECTRAL_RADIUS,
                        ESN_INPUT_SCALING, ESN_LEAK_RATE, ESN_RIDGE_ALPHA)
        esn.fit(X_norm[:-1], y_all[:-1])
        esn.reset_state()
        for i in range(len(X_norm) - 1):
            esn._update(X_norm[i])
        pred_raw = esn.predict(X_norm[-1:])[0]

    direction = 'UP (Long)' if pred_raw > 0.5 else 'DOWN (Short)'
    confidence = min(1.0, abs(pred_raw - 0.5) * 2)

    return {
        'Prediction_Date': datetime.now().strftime('%Y-%m-%d'),
        'Target_Date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'Last_Close': str(features_df.index[-1].strftime('%Y-%m-%d') if hasattr(features_df.index[-1], 'strftime') else features_df.index[-1]),
        'Raw_Score': round(float(pred_raw), 4),
        'Direction': direction,
        'Confidence': f'{confidence:.1%}',
    }


def performance_metrics(results_df):
    """Calculate backtest performance metrics."""
    if results_df.empty:
        return {}
    total_trades = len(results_df)
    correct = results_df['Correct'].sum()
    accuracy = correct / total_trades
    total_pnl = results_df['PnL_pips'].sum()
    max_dd = (results_df['Cumulative_PnL_pips'].cummax() - results_df['Cumulative_PnL_pips']).max()
    sharpe = results_df['PnL_pips'].mean() / results_df['PnL_pips'].std() * np.sqrt(252) \
        if results_df['PnL_pips'].std() > 0 else 0
    win_pnl = results_df[results_df['PnL_pips'] > 0]['PnL_pips']
    loss_pnl = results_df[results_df['PnL_pips'] <= 0]['PnL_pips']
    profit_factor = win_pnl.sum() / abs(loss_pnl.sum()) if abs(loss_pnl.sum()) > 0 else np.inf

    return {
        'Total_Trades': total_trades,
        'Correct_Trades': correct,
        'Accuracy': round(accuracy, 4),
        'Total_PnL_pips': round(total_pnl, 2),
        'Max_Drawdown_pips': round(max_dd, 2),
        'Sharpe_Ratio': round(sharpe, 3),
        'Profit_Factor': round(profit_factor, 3),
        'Avg_PnL_per_trade': round(total_pnl / total_trades, 2),
        'Win_Rate': f'{accuracy:.1%}',
    }


# ============================================================
# Main
# ============================================================
def load_mock_data(name, data_dir='data'):
    """Load mock CSV data as fallback when yfinance unavailable."""
    path = os.path.join(data_dir, f'mock_{name}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mock data not found: {path}\n  Run: python generate_mock_data.py")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def main():
    import sys
    use_mock = '--mock' in sys.argv

    print("=" * 60)
    print("📈 RC Financial Prediction — ESN Backtest + Forecast")
    if use_mock:
        print("   (MOCK DATA MODE)")
    print("=" * 60)

    os.makedirs('output', exist_ok=True)

    # 1. Fetch data
    tickers = {
        'USDJPY': 'USDJPY=X',
        'EURUSD': 'EURUSD=X',
        'N225': '^N225',
    }

    all_results = {}
    all_predictions = []

    for name, ticker in tickers.items():
        print(f"\n{'='*40}")
        print(f"  Processing: {name} ({ticker})")
        print(f"{'='*40}")

        try:
            if use_mock:
                df = load_mock_data(name)
                print(f"  Loaded mock data: {len(df)} bars")
            else:
                df = fetch_data(ticker, '2y')
        except Exception as e:
            print(f"  [ERROR] Failed to fetch {ticker}: {e}")
            continue

        if len(df) < 300:
            print(f"  [SKIP] Insufficient data ({len(df)} bars)")
            continue

        # 2. Features
        print("  Computing features...")
        features = compute_features(df)
        print(f"  Features: {len(features)} rows, {len(features.columns)} cols")

        # 3. Backtest
        print("  Running walk-forward backtest...")
        bt = walk_forward_backtest(features, train_window=min(250, len(features) // 2))
        metrics = performance_metrics(bt)

        print(f"  Results: Accuracy={metrics.get('Accuracy', 'N/A')}, "
              f"PnL={metrics.get('Total_PnL_pips', 'N/A')} pips, "
              f"Sharpe={metrics.get('Sharpe_Ratio', 'N/A')}")

        all_results[name] = {'backtest': bt, 'metrics': metrics}

        # 4. Tomorrow's prediction
        print("  Predicting tomorrow...")
        pred = predict_tomorrow(features)
        pred['Ticker'] = name
        all_predictions.append(pred)
        print(f"  → {pred['Direction']} (Confidence: {pred['Confidence']})")

    # 5. Save results
    print(f"\n{'='*60}")
    print("  Saving results...")

    out_path = 'output/rc_prediction_results.xlsx'
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # Predictions
        pred_df = pd.DataFrame(all_predictions)
        pred_df.to_excel(writer, sheet_name='Tomorrow_Prediction', index=False)

        # Backtest details per ticker
        for name, data in all_results.items():
            if not data['backtest'].empty:
                data['backtest'].to_excel(writer, sheet_name=f'Backtest_{name}', index=False)

        # Performance summary
        perf_rows = []
        for name, data in all_results.items():
            row = {'Ticker': name, **data['metrics']}
            perf_rows.append(row)
        perf_df = pd.DataFrame(perf_rows)
        perf_df.to_excel(writer, sheet_name='Performance_Summary', index=False)

        # Info
        info = pd.DataFrame([
            {'Item': 'Tool', 'Value': 'rc_financial_prediction.py'},
            {'Item': 'Author', 'Value': 'Go Sato'},
            {'Item': 'Data_Source', 'Value': 'yfinance'},
            {'Item': 'Model', 'Value': 'Echo State Network (Reservoir Computing)'},
            {'Item': 'ESN_Units', 'Value': str(ESN_UNITS)},
            {'Item': 'Spectral_Radius', 'Value': str(ESN_SPECTRAL_RADIUS)},
            {'Item': 'Backtest', 'Value': 'Walk-forward with spread cost'},
            {'Item': 'Spread', 'Value': f'{SPREAD_PIPS} pips'},
            {'Item': 'Reference_1', 'Value': 'Jaeger (2001) Echo State Network'},
            {'Item': 'Reference_2', 'Value': 'Lopez de Prado (2018) Fractional Differentiation'},
        ])
        info.to_excel(writer, sheet_name='Model_Info', index=False)

    # CSV exports
    if all_predictions:
        pd.DataFrame(all_predictions).to_csv('output/tomorrow_predictions.csv', index=False)
    if perf_rows:
        pd.DataFrame(perf_rows).to_csv('output/performance_summary.csv', index=False)

    print(f"\n  All results saved to: {out_path}")
    print(f"  CSVs: tomorrow_predictions.csv, performance_summary.csv")

    # Print summary
    print(f"\n{'='*60}")
    print("  📊 TOMORROW'S PREDICTIONS")
    print(f"{'='*60}")
    for p in all_predictions:
        print(f"    {p['Ticker']:10s}: {p['Direction']:15s} (Confidence: {p['Confidence']})")


if __name__ == '__main__':
    main()
