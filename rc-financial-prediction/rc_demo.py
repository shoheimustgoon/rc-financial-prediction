# -*- coding: utf-8 -*-
"""
Reservoir Computing Financial Prediction — Concept Demo
Demonstrates the core RC pipeline on publicly available data.

This is a simplified educational demo. The full production system
includes 9 RC variants, multi-method ensembles, and automated
hyperparameter optimization (11,800+ lines).

Author: Go Sato
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Data Acquisition
# ============================================================
def fetch_market_data(symbol='USD', lookback=400):
    """Fetch OHLCV data from Yahoo Finance.

    Symbol mapping:
      USD → JPY=X (USD/JPY)
      EUR → EURJPY=X
      N225 → ^N225
      BTC  → BTC-JPY
    """
    import yfinance as yf

    ticker_map = {
        'USD': 'JPY=X',
        'EUR': 'EURJPY=X',
        'N225': '^N225',
        'BTC': 'BTC-JPY',
    }
    ticker = ticker_map.get(symbol, symbol)

    df = yf.download(ticker, period=f'{lookback + 50}d', auto_adjust=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df.tail(lookback)


# ============================================================
# 2. Feature Engineering
# ============================================================
def compute_features(df):
    """Compute technical indicators as RC input features.

    Returns normalized feature matrix X and target y (next day's close direction).
    """
    close = df['Close'].values.flatten()
    high = df['High'].values.flatten()
    low = df['Low'].values.flatten()

    features = {}

    # Trend: Moving averages
    for w in [5, 20, 60]:
        if len(close) > w:
            sma = pd.Series(close).rolling(w).mean().values
            features[f'SMA_{w}_ratio'] = close / (sma + 1e-10) - 1

    # Momentum: RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    features['RSI'] = 100 - 100 / (1 + rs)

    # Volatility: ATR-based
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    features['ATR_ratio'] = pd.Series(tr).rolling(14).mean().values / (close + 1e-10)

    # Returns at multiple horizons
    for lag in [1, 5, 20]:
        features[f'Return_{lag}d'] = np.concatenate([[0]*lag, np.diff(np.log(close + 1e-10), n=lag)])

    # Combine
    feat_df = pd.DataFrame(features, index=df.index)
    feat_df = feat_df.dropna()

    # Normalize (z-score)
    X = (feat_df - feat_df.mean()) / (feat_df.std() + 1e-10)

    # Target: next day's close direction (1 = up, 0 = down)
    future_return = np.diff(np.log(close + 1e-10))
    y = (future_return > 0).astype(float)
    y = y[len(close) - len(X) - 1: len(close) - 1]

    # Align lengths
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y[:min_len]

    return X.values, y, feat_df.columns.tolist()


# ============================================================
# 3. Echo State Network (Simplified)
# ============================================================
class SimpleESN:
    """Minimal Echo State Network implementation for demonstration.

    The production system uses ReservoirPy with advanced features:
    - Deep ESN (multi-layer)
    - Adaptive spectral radius
    - SOGWO hyperparameter optimization
    - SINDy-based nonlinear feature extraction

    Reference: Jaeger (2001) "The echo state approach"
    """

    def __init__(self, input_dim, reservoir_size=500, spectral_radius=0.95,
                 input_scaling=0.1, leak_rate=0.3, ridge_alpha=1e-6, seed=42):
        np.random.seed(seed)
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha

        # Input weights (fixed, random)
        self.W_in = np.random.randn(reservoir_size, input_dim) * input_scaling

        # Reservoir weights (fixed, random, scaled to spectral radius)
        W = np.random.randn(reservoir_size, reservoir_size) * 0.1
        eigenvalues = np.abs(np.linalg.eigvals(W))
        W = W * (spectral_radius / (max(eigenvalues) + 1e-10))
        self.W = W

        # Output weights (to be trained)
        self.W_out = None
        self.states = None

    def _run_reservoir(self, X):
        """Drive the reservoir with input data and collect states."""
        T = len(X)
        states = np.zeros((T, self.reservoir_size))
        x = np.zeros(self.reservoir_size)

        for t in range(T):
            # Leaky integration: x(t) = (1-α)x(t-1) + α·tanh(W_in·u(t) + W·x(t-1))
            pre_activation = self.W_in @ X[t] + self.W @ x
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(pre_activation)
            states[t] = x

        return states

    def fit(self, X, y):
        """Train the ESN: run reservoir, then solve Ridge regression for output."""
        self.states = self._run_reservoir(X)

        # Ridge regression: W_out = (S^T S + αI)^(-1) S^T y
        S = self.states
        reg = self.ridge_alpha * np.eye(self.reservoir_size)
        self.W_out = np.linalg.solve(S.T @ S + reg, S.T @ y)

        return self

    def predict(self, X):
        """Predict using trained ESN."""
        states = self._run_reservoir(X)
        return states @ self.W_out


# ============================================================
# 4. Walk-Forward Backtest
# ============================================================
def walk_forward_backtest(X, y, train_ratio=0.6, n_steps=10, reservoir_size=300):
    """Simple expanding-window walk-forward backtest.

    The production system includes:
    - Bayesian TPE hyperparameter optimization per window
    - Multi-method ensemble (9 RC variants)
    - Spread cost modeling
    - Japanese holiday handling
    """
    total = len(X)
    initial_train = int(total * train_ratio)
    step_size = (total - initial_train) // n_steps

    results = []

    for i in range(n_steps):
        train_end = initial_train + i * step_size
        test_end = min(train_end + step_size, total)

        if train_end >= total or test_end > total:
            break

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]

        if len(X_test) == 0:
            continue

        # Train ESN
        esn = SimpleESN(input_dim=X.shape[1], reservoir_size=reservoir_size)
        esn.fit(X_train, y_train)

        # Predict
        y_pred = esn.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(float)

        # Accuracy
        accuracy = np.mean(y_pred_binary == y_test)

        results.append({
            'window': i + 1,
            'train_size': train_end,
            'test_size': len(X_test),
            'accuracy': accuracy,
        })

    return pd.DataFrame(results)


# ============================================================
# 5. Main Demo
# ============================================================
def main():
    print("=" * 60)
    print("🧠 Reservoir Computing — Financial Prediction Demo")
    print("=" * 60)

    symbols = ['USD', 'N225', 'BTC']

    for symbol in symbols:
        print(f"\n{'='*40}")
        print(f"  Asset: {symbol}")
        print(f"{'='*40}")

        try:
            # Fetch data
            df = fetch_market_data(symbol, lookback=400)
            print(f"  Data: {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")

            # Features
            X, y, feat_names = compute_features(df)
            print(f"  Features: {len(feat_names)} ({', '.join(feat_names[:5])}...)")
            print(f"  Samples: {len(X)}")

            # Walk-forward backtest
            results = walk_forward_backtest(X, y, n_steps=8)
            mean_acc = results['accuracy'].mean()
            print(f"\n  Walk-Forward Results ({len(results)} windows):")
            print(f"  Mean Accuracy: {mean_acc:.1%}")
            print(f"  Best Window:   {results['accuracy'].max():.1%}")
            print(f"  Worst Window:  {results['accuracy'].min():.1%}")

            # Note about production system
            print(f"\n  [Note] Production system uses:")
            print(f"    - 9 RC methods (Deep ESN, HAR-ESN, SINDy-RC, ...)")
            print(f"    - SOGWO hyperparameter optimization")
            print(f"    - Fractional differentiation + Hurst exponent")
            print(f"    - Multi-method ensemble averaging")

        except Exception as e:
            print(f"  Error: {e}")
            print(f"  (Install yfinance: pip install yfinance)")

    print(f"\n{'='*60}")
    print("Demo complete. See README.md for full architecture details.")


if __name__ == '__main__':
    main()
