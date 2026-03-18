# -*- coding: utf-8 -*-
"""
RC Financial Prediction — Mock Data Generator
Creates realistic synthetic FX/stock data for testing when yfinance is unavailable.

Author: Go Sato
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_mock_ohlcv(ticker_name='USDJPY', n_days=500, base_price=150.0,
                        volatility=0.008, seed=42):
    """Generate realistic OHLCV data using geometric Brownian motion."""
    np.random.seed(seed)

    dates = pd.bdate_range(end=datetime.now(), periods=n_days)
    returns = np.random.normal(0.0001, volatility, n_days)

    close = np.zeros(n_days)
    close[0] = base_price
    for i in range(1, n_days):
        close[i] = close[i-1] * (1 + returns[i])

    # Generate OHLV from close
    daily_range = np.abs(np.random.normal(0, volatility * 0.7, n_days))
    high = close * (1 + daily_range)
    low = close * (1 - daily_range)
    open_price = close * (1 + np.random.normal(0, volatility * 0.3, n_days))

    # Ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = np.random.randint(50000, 500000, n_days)

    df = pd.DataFrame({
        'Open': np.round(open_price, 3),
        'High': np.round(high, 3),
        'Low': np.round(low, 3),
        'Close': np.round(close, 3),
        'Volume': volume,
    }, index=dates)

    return df


def main():
    os.makedirs('data', exist_ok=True)

    tickers = {
        'USDJPY': {'base': 150.0, 'vol': 0.008, 'seed': 42},
        'EURUSD': {'base': 1.08, 'vol': 0.006, 'seed': 123},
        'N225': {'base': 38000.0, 'vol': 0.012, 'seed': 456},
    }

    print("=" * 60)
    print("📈 RC Financial — Mock Data Generator")
    print("=" * 60)

    for name, params in tickers.items():
        df = generate_mock_ohlcv(name, n_days=500,
                                 base_price=params['base'],
                                 volatility=params['vol'],
                                 seed=params['seed'])
        path = f'data/mock_{name}.csv'
        df.to_csv(path)
        print(f"  {name}: {len(df)} bars → {path}")
        print(f"    Range: {df.index[0].date()} → {df.index[-1].date()}")
        print(f"    Close: {df['Close'].min():.2f} – {df['Close'].max():.2f}")

    print(f"\n  Files saved to ./data/")
    print(f"\n  To run with mock data:")
    print(f"    python rc_financial_prediction.py --mock")


if __name__ == '__main__':
    main()
