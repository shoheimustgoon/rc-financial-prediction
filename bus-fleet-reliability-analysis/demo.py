# -*- coding: utf-8 -*-
"""
Bus Fleet Reliability Analysis — Concept Demo

Demonstrates mileage-adjusted MTBF and bathtub curve classification
on synthetic bus fleet failure data.

The full production system (not published) includes:
- Multi-tier monthly KPI reports (Count/Rate/MTBF/RF-MTBF)
- Cumulative wafer/mileage vs MTBF aging analysis
- Automated breakpoint detection for phase transitions
- Cox PH / Weibull AFT with right-censoring
- Intervention-linked survival data generation
- Excel reports with conditional formatting

Author: Go Sato
"""

import numpy as np
import pandas as pd


def generate_bus_data(n_buses=50, n_months=36, seed=42):
    """Generate synthetic bus failure data with varying mileage."""
    np.random.seed(seed)
    rows = []

    for b in range(n_buses):
        bus_id = f'BUS_{b+1:03d}'
        depot = np.random.choice(['Depot_A', 'Depot_B', 'Depot_C'])
        monthly_km = np.random.uniform(1000, 5000)  # varying utilization
        base_failure_rate = np.random.uniform(0.001, 0.003)  # per km
        cumulative_km = 0

        for m in range(1, n_months + 1):
            km = monthly_km * np.random.uniform(0.8, 1.2)
            cumulative_km += km

            # Bathtub: higher rate early and late
            age_factor = 1.0
            if cumulative_km < 10000:
                age_factor = 1.5  # early failures (DFR)
            elif cumulative_km > 100000:
                age_factor = 1.0 + (cumulative_km - 100000) / 50000  # wear-out (IFR)

            failures = np.random.poisson(base_failure_rate * km * age_factor)
            rows.append({
                'BusID': bus_id, 'Depot': depot, 'Month': m,
                'Monthly_KM': round(km), 'Cumulative_KM': round(cumulative_km),
                'Failures': failures,
            })

    return pd.DataFrame(rows)


def demo_mtbf():
    """Demonstrate mileage-adjusted MTBF."""
    print("=" * 50)
    print("🚌 Mileage-Adjusted MTBF Analysis")
    print("=" * 50)

    df = generate_bus_data()

    # Calendar MTBF (naive)
    total_months = df.groupby('BusID')['Month'].count().sum()
    total_failures = df['Failures'].sum()
    calendar_mtbf = total_months / (total_failures + 1)

    # Mileage-adjusted MTBF
    total_km = df['Monthly_KM'].sum()
    mileage_mtbf = total_km / (total_failures + 1)

    print(f"\n  Buses: {df['BusID'].nunique()}")
    print(f"  Total failures: {total_failures}")
    print(f"  Calendar MTBF: {calendar_mtbf:.1f} months (naive)")
    print(f"  Mileage MTBF: {mileage_mtbf:.0f} km (exposure-adjusted)")
    print(f"\n  [Note] Production system computes per-bus MTBF")
    print(f"  with RF-time/wafer-count exposure normalization.")


def demo_bathtub():
    """Demonstrate bathtub curve classification."""
    print(f"\n{'='*50}")
    print("🚌 Bathtub Curve — Aging Phase Classification")
    print("=" * 50)

    df = generate_bus_data()
    df['Rate_per_KM'] = df['Failures'] / (df['Monthly_KM'] + 1)

    # Classify by cumulative km
    bins = [0, 10000, 100000, np.inf]
    labels = ['Early (DFR)', 'Stable (CFR)', 'Wear-out (IFR)']
    df['Phase'] = pd.cut(df['Cumulative_KM'], bins=bins, labels=labels)

    for phase in labels:
        sub = df[df['Phase'] == phase]
        rate = sub['Rate_per_KM'].mean() * 1000
        print(f"  {phase:20s}: Rate={rate:.3f} per 1000km (n={len(sub)})")

    print(f"\n  [Note] Production system uses Weibull shape parameter β")
    print(f"  and automated breakpoint detection for phase boundaries.")


if __name__ == '__main__':
    demo_mtbf()
    demo_bathtub()
    print(f"\nDemo complete.")
