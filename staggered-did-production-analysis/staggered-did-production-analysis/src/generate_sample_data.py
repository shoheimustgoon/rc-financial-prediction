#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample Data Generator for Bakery Production Analysis
=====================================================

Generates realistic synthetic data for bakery manufacturing equipment,
including:
- Equipment failure patterns (bathtub curve)
- Intervention effects
- Right-censored observations
- Multiple production lines

Usage:
    python generate_sample_data.py --output data/sample/bakery_data.xlsx
    python generate_sample_data.py --n-equipment 50 --n-months 36
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List


# ============================================================
# Constants
# ============================================================

# Equipment types in a bakery
EQUIPMENT_TYPES = ['Oven', 'Mixer', 'Proofer', 'Conveyor', 'Slicer', 'Packager']

# Production lines
PRODUCTION_LINES = ['Line_A', 'Line_B', 'Line_C']

# Error types
ERROR_TYPES = [
    'Temperature_Deviation',
    'Motor_Failure',
    'Sensor_Error',
    'Belt_Wear',
    'Calibration_Drift',
    'Overload_Trip',
    'Timing_Fault',
    'Lubrication_Issue'
]

# Default parameters
DEFAULT_N_EQUIPMENT = 30
DEFAULT_N_MONTHS = 24
DEFAULT_TREATMENT_RATE = 0.5
DEFAULT_TREATMENT_EFFECT = 50  # MTBF improvement in hours


# ============================================================
# Data Generation Functions
# ============================================================

def generate_bathtub_mtbf(production: float,
                          equipment_type: str,
                          base_mtbf: float = 150,
                          dfr_end: float = 2000,
                          cfr_end: float = 50000) -> float:
    """
    Generate MTBF following bathtub curve pattern.
    
    Parameters
    ----------
    production : float
        Cumulative production count
    equipment_type : str
        Type of equipment (affects parameters)
    base_mtbf : float
        Base MTBF in stable period
    dfr_end : float
        Production count where DFR ends
    cfr_end : float
        Production count where CFR ends
        
    Returns
    -------
    float
        MTBF value
    """
    # Equipment-specific adjustments
    type_factors = {
        'Oven': 1.2,
        'Mixer': 1.0,
        'Proofer': 0.9,
        'Conveyor': 1.1,
        'Slicer': 0.95,
        'Packager': 0.85
    }
    
    factor = type_factors.get(equipment_type, 1.0)
    
    if production < dfr_end:
        # DFR phase: high initial failures, decreasing
        # MTBF starts low and increases
        progress = production / dfr_end
        mtbf = base_mtbf * factor * (0.5 + 0.5 * progress)
    elif production < cfr_end:
        # CFR phase: stable
        mtbf = base_mtbf * factor
    else:
        # IFR phase: increasing failures
        # MTBF decreases
        excess = production - cfr_end
        decay = 1.0 - min(0.5, excess / 100000)
        mtbf = base_mtbf * factor * decay
    
    # Add noise
    noise = np.random.normal(0, base_mtbf * 0.15)
    mtbf = max(10, mtbf + noise)
    
    return mtbf


def generate_equipment_data(equipment_id: str,
                            equipment_type: str,
                            line: str,
                            start_date: datetime,
                            n_months: int,
                            is_treated: bool,
                            treatment_month: int = None,
                            treatment_effect: float = DEFAULT_TREATMENT_EFFECT) -> pd.DataFrame:
    """
    Generate failure data for a single equipment.
    
    Parameters
    ----------
    equipment_id : str
        Unique equipment identifier
    equipment_type : str
        Type of equipment
    line : str
        Production line
    start_date : datetime
        Start of observation period
    n_months : int
        Number of months to simulate
    is_treated : bool
        Whether equipment receives intervention
    treatment_month : int
        Month when treatment is applied
    treatment_effect : float
        MTBF improvement from treatment
        
    Returns
    -------
    pd.DataFrame
        Equipment failure records
    """
    records = []
    
    # Initialize
    current_date = start_date
    cumulative_production = np.random.uniform(500, 2000)  # Starting production
    production_rate = np.random.uniform(80, 150)  # Units per day
    
    # Generate intervention date
    if is_treated and treatment_month:
        intervention_date = start_date + timedelta(days=treatment_month * 30)
    else:
        intervention_date = None
    
    # Simulate failures over time
    observation_end = start_date + timedelta(days=n_months * 30)
    
    while current_date < observation_end:
        # Is this after treatment?
        post_treatment = is_treated and intervention_date and current_date >= intervention_date
        
        # Calculate MTBF
        base_mtbf = generate_bathtub_mtbf(cumulative_production, equipment_type)
        
        # Apply treatment effect
        if post_treatment:
            base_mtbf += treatment_effect
        
        # Generate time to next failure (exponential distribution)
        time_to_failure = np.random.exponential(base_mtbf)
        
        # Update state
        hours_to_failure = max(1, time_to_failure)
        days_to_failure = hours_to_failure / 24
        
        next_date = current_date + timedelta(days=days_to_failure)
        
        # Check if within observation window
        if next_date >= observation_end:
            # Right-censored observation
            censored_duration = (observation_end - current_date).total_seconds() / 3600
            
            records.append({
                'Equipment_ID': equipment_id,
                'Equipment_Type': equipment_type,
                'Line': line,
                'Error_DateTime': observation_end,
                'Error_Type': 'Censored',
                'MTBF': censored_duration,
                'Cumulative_Production': cumulative_production,
                'Operating_Hours': cumulative_production / (production_rate / 24),
                'Event': 0,  # Censored
                'Treatment': 1 if is_treated else 0,
                'Post': 1 if post_treatment else 0,
                'Intervention_Date': intervention_date
            })
            break
        
        # Record failure
        error_type = np.random.choice(ERROR_TYPES)
        
        records.append({
            'Equipment_ID': equipment_id,
            'Equipment_Type': equipment_type,
            'Line': line,
            'Error_DateTime': next_date,
            'Error_Type': error_type,
            'MTBF': hours_to_failure,
            'Cumulative_Production': cumulative_production,
            'Operating_Hours': cumulative_production / (production_rate / 24),
            'Event': 1,  # Failure
            'Treatment': 1 if is_treated else 0,
            'Post': 1 if post_treatment else 0,
            'Intervention_Date': intervention_date
        })
        
        # Update for next iteration
        current_date = next_date
        cumulative_production += production_rate * days_to_failure
    
    return pd.DataFrame(records)


def generate_sample_dataset(n_equipment: int = DEFAULT_N_EQUIPMENT,
                            n_months: int = DEFAULT_N_MONTHS,
                            treatment_rate: float = DEFAULT_TREATMENT_RATE,
                            treatment_effect: float = DEFAULT_TREATMENT_EFFECT,
                            random_seed: int = 42) -> pd.DataFrame:
    """
    Generate complete sample dataset.
    
    Parameters
    ----------
    n_equipment : int
        Number of equipment to simulate
    n_months : int
        Observation period in months
    treatment_rate : float
        Fraction of equipment receiving treatment
    treatment_effect : float
        MTBF improvement from treatment
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Complete sample dataset
    """
    np.random.seed(random_seed)
    
    start_date = datetime(2023, 1, 1)
    treatment_month = n_months // 2  # Treatment at midpoint
    
    all_records = []
    
    for i in range(n_equipment):
        # Assign equipment attributes
        equipment_id = f"BK-{i+1:03d}"
        equipment_type = np.random.choice(EQUIPMENT_TYPES)
        line = np.random.choice(PRODUCTION_LINES)
        is_treated = np.random.random() < treatment_rate
        
        # Staggered treatment timing (within 3 months of midpoint)
        if is_treated:
            eq_treatment_month = treatment_month + np.random.randint(-2, 3)
        else:
            eq_treatment_month = None
        
        # Generate data for this equipment
        eq_df = generate_equipment_data(
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            line=line,
            start_date=start_date,
            n_months=n_months,
            is_treated=is_treated,
            treatment_month=eq_treatment_month,
            treatment_effect=treatment_effect
        )
        
        all_records.append(eq_df)
    
    # Combine all equipment
    df = pd.concat(all_records, ignore_index=True)
    
    # Sort by date
    df = df.sort_values('Error_DateTime').reset_index(drop=True)
    
    # Add derived columns
    df['Year_Month'] = df['Error_DateTime'].dt.to_period('M')
    df['Treatment_Post'] = df['Treatment'] * df['Post']
    
    # Calculate RF-MTBF (simplified: assume 70% utilization)
    df['RF_MTBF'] = df['MTBF'] * 0.7
    
    # Calculate WBF (wafers between failures - here, units between failures)
    df['WBF'] = df['MTBF'] * np.random.uniform(3, 5, len(df))
    
    return df


def create_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create survival analysis ready dataset.
    
    Ensures proper structure for survival analysis including
    right-censored observations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw failure data
        
    Returns
    -------
    pd.DataFrame
        Survival-ready dataset
    """
    survival_df = df.copy()
    
    # Ensure required columns
    survival_df['duration'] = survival_df['MTBF']
    survival_df['event'] = survival_df['Event']
    
    # Add ChamberGroup equivalent (equipment grouping)
    survival_df['ChamberGroup'] = survival_df['Equipment_Type'] + '_' + survival_df['Line']
    
    # Add phase information (simplified)
    def assign_phase(row):
        prod = row['Cumulative_Production']
        if prod < 2000:
            return 'Initial'
        elif prod < 50000:
            return 'Stable'
        else:
            return 'Wearout'
    
    survival_df['Phase'] = survival_df.apply(assign_phase, axis=1)
    
    return survival_df


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate sample data for bakery production analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output', '-o', default='data/sample/bakery_equipment_data.xlsx',
                       help='Output file path')
    parser.add_argument('--n-equipment', type=int, default=DEFAULT_N_EQUIPMENT,
                       help=f'Number of equipment (default: {DEFAULT_N_EQUIPMENT})')
    parser.add_argument('--n-months', type=int, default=DEFAULT_N_MONTHS,
                       help=f'Observation months (default: {DEFAULT_N_MONTHS})')
    parser.add_argument('--treatment-rate', type=float, default=DEFAULT_TREATMENT_RATE,
                       help=f'Treatment rate (default: {DEFAULT_TREATMENT_RATE})')
    parser.add_argument('--treatment-effect', type=float, default=DEFAULT_TREATMENT_EFFECT,
                       help=f'Treatment effect on MTBF (default: {DEFAULT_TREATMENT_EFFECT})')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating Sample Bakery Equipment Data")
    print("=" * 60)
    print(f"  Equipment: {args.n_equipment}")
    print(f"  Months: {args.n_months}")
    print(f"  Treatment rate: {args.treatment_rate:.0%}")
    print(f"  Treatment effect: {args.treatment_effect} hours")
    print()
    
    # Generate data
    df = generate_sample_dataset(
        n_equipment=args.n_equipment,
        n_months=args.n_months,
        treatment_rate=args.treatment_rate,
        treatment_effect=args.treatment_effect,
        random_seed=args.seed
    )
    
    survival_df = create_survival_data(df)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='Equipment_Data', index=False)
        
        # Survival data sheet
        survival_df.to_excel(writer, sheet_name='Survival_Data', index=False)
        
        # Summary statistics
        summary = pd.DataFrame({
            'Metric': [
                'Total Records',
                'Total Equipment',
                'Failure Events',
                'Censored Events',
                'Treated Equipment',
                'Control Equipment',
                'Mean MTBF (hours)',
                'Median MTBF (hours)',
                'Mean Production'
            ],
            'Value': [
                len(df),
                df['Equipment_ID'].nunique(),
                (df['Event'] == 1).sum(),
                (df['Event'] == 0).sum(),
                df[df['Treatment'] == 1]['Equipment_ID'].nunique(),
                df[df['Treatment'] == 0]['Equipment_ID'].nunique(),
                df['MTBF'].mean(),
                df['MTBF'].median(),
                df['Cumulative_Production'].mean()
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Data saved to: {args.output}")
    print()
    print("Summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Equipment: {df['Equipment_ID'].nunique()}")
    print(f"  Failures: {(df['Event'] == 1).sum()}")
    print(f"  Censored: {(df['Event'] == 0).sum()}")
    print(f"  Treated: {df[df['Treatment'] == 1]['Equipment_ID'].nunique()}")
    print(f"  Control: {df[df['Treatment'] == 0]['Equipment_ID'].nunique()}")
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
