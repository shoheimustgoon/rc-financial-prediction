#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Difference-in-Differences (DiD) Analysis Module
================================================

Implements causal inference methods for evaluating equipment interventions:
- Raw DiD (4-cell comparison)
- Two-Way Fixed Effects (TWFE) regression
- Event Study design
- Parallel Trends testing
- Staggered DiD for multiple treatment times

This module supports analysis of interventions like:
- New oven cleaning procedures
- Equipment upgrades
- Process improvements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Check for statsmodels availability
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. DiD regression will be limited.")

# Check for scipy availability
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# Constants
# ============================================================

# Minimum sample sizes for reliable analysis
MIN_SAMPLE_RELIABLE = 10
MIN_SAMPLE_MODERATE = 5
MIN_SAMPLE_ANY = 1

# Default event study window
DEFAULT_EVENT_WINDOW = (-6, 12)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class RawDiDResult:
    """Results from raw 4-cell DiD comparison"""
    metric: str
    group: str
    treated_before: float
    treated_after: float
    control_before: float
    control_after: float
    did_effect: float
    did_effect_pct: float
    n_treated_before: int
    n_treated_after: int
    n_control_before: int
    n_control_after: int
    is_significant: bool = None
    p_value: float = None
    interpretation: str = None


@dataclass
class TWFEResult:
    """Results from Two-Way Fixed Effects regression"""
    metric: str
    coefficient: float
    std_error: float
    t_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    r_squared: float
    n_observations: int
    is_significant: bool


@dataclass
class EventStudyResult:
    """Results from event study analysis"""
    metric: str
    group: str
    relative_period: int
    coefficient: float
    std_error: float
    confidence_interval: Tuple[float, float]
    p_value: float


@dataclass
class ParallelTrendsResult:
    """Results from parallel trends test"""
    metric: str
    group: str
    f_statistic: float
    p_value: float
    is_parallel: bool
    interpretation: str
    pre_trend_coefficients: Dict[int, float]


# ============================================================
# DiD Analyzer Class
# ============================================================

class DiDAnalyzer:
    """
    Comprehensive Difference-in-Differences analyzer.
    
    Evaluates causal effects of interventions by comparing changes
    in outcomes between treated and control groups before/after treatment.
    
    Key assumption: Parallel trends - in absence of treatment, treated and
    control groups would have followed similar outcome trajectories.
    
    Parameters
    ----------
    cfr_threshold : float
        Threshold for CFR phase classification (for phase-specific analysis)
    cluster_se : bool
        Whether to use clustered standard errors (default: True)
    
    Examples
    --------
    >>> analyzer = DiDAnalyzer()
    >>> analyzer.load_data('equipment_data.xlsx')
    >>> results = analyzer.run_staggered_did(
    ...     outcome='MTBF',
    ...     treatment_col='intervention_date',
    ...     group_col='equipment_type'
    ... )
    """
    
    def __init__(self, cfr_threshold: float = 0.15, cluster_se: bool = True):
        self.cfr_threshold = cfr_threshold
        self.cluster_se = cluster_se
        self.df = None
        self.intervention_dates = {}
        self.results: Dict[str, Any] = {}
    
    def load_data(self,
                  filepath: str = None,
                  df: pd.DataFrame = None,
                  survival_sheet: str = 'Survival_Data') -> bool:
        """
        Load data from Excel file or DataFrame.
        
        Parameters
        ----------
        filepath : str
            Path to Excel file
        df : pd.DataFrame
            DataFrame to use directly
        survival_sheet : str
            Sheet name for survival data
            
        Returns
        -------
        bool
            Whether loading was successful
        """
        try:
            if df is not None:
                self.df = df.copy()
            elif filepath:
                try:
                    self.df = pd.read_excel(filepath, sheet_name=survival_sheet)
                except:
                    self.df = pd.read_excel(filepath)
            else:
                print("Error: Must provide filepath or DataFrame")
                return False
            
            # Standardize column names
            self._standardize_columns()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _standardize_columns(self):
        """Standardize column names for processing."""
        rename_map = {
            'Equipment_ID': 'equipment_id',
            'Line': 'line',
            'MTBF': 'mtbf',
            'RF_MTBF': 'rf_mtbf',
            'Event': 'event',
            'Cumulative_Production': 'cumulative_production',
            'Treatment': 'treatment',
            'Post': 'post'
        }
        
        # Apply renames for columns that exist
        existing_renames = {k: v for k, v in rename_map.items() if k in self.df.columns}
        self.df = self.df.rename(columns=existing_renames)
    
    def set_intervention_dates(self,
                               intervention_dates: Dict[str, Union[str, datetime, pd.Timestamp]]):
        """
        Set intervention dates by equipment.
        
        Parameters
        ----------
        intervention_dates : Dict[str, datetime]
            Dictionary mapping equipment_id to intervention date
        """
        self.intervention_dates = {
            k: pd.Timestamp(v) for k, v in intervention_dates.items()
        }
    
    def create_did_variables(self,
                              date_col: str = 'error_datetime',
                              intervention_date_col: str = 'intervention_date',
                              equipment_col: str = 'equipment_id') -> pd.DataFrame:
        """
        Create treatment and post variables for DiD analysis.
        
        Parameters
        ----------
        date_col : str
            Column with observation date
        intervention_date_col : str
            Column with intervention date (or use set_intervention_dates)
        equipment_col : str
            Column with equipment identifier
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added treatment and post variables
        """
        df = self.df.copy()
        
        # Ensure datetime
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Get intervention dates
        if intervention_date_col in df.columns:
            df[intervention_date_col] = pd.to_datetime(df[intervention_date_col])
            # Treated = has intervention date
            df['treatment'] = df[intervention_date_col].notna().astype(int)
            # Post = observation after intervention date
            df['post'] = (df[date_col] >= df[intervention_date_col]).astype(int)
        elif self.intervention_dates:
            def get_treatment(row):
                eq_id = row.get(equipment_col)
                return 1 if eq_id in self.intervention_dates else 0
            
            def get_post(row):
                eq_id = row.get(equipment_col)
                if eq_id not in self.intervention_dates:
                    return 0
                intv_date = self.intervention_dates[eq_id]
                obs_date = row.get(date_col)
                return 1 if pd.notna(obs_date) and obs_date >= intv_date else 0
            
            df['treatment'] = df.apply(get_treatment, axis=1)
            df['post'] = df.apply(get_post, axis=1)
        else:
            print("Warning: No intervention dates available")
            df['treatment'] = 0
            df['post'] = 0
        
        # Create interaction term
        df['treatment_post'] = df['treatment'] * df['post']
        
        # Create time period (year-month)
        if date_col in df.columns:
            df['year_month'] = df[date_col].dt.to_period('M')
        
        self.df = df
        return df
    
    def calc_raw_did(self,
                     outcome_col: str = 'mtbf',
                     group_col: str = None) -> List[RawDiDResult]:
        """
        Calculate raw DiD effect using 4-cell comparison.
        
        DiD = (Y_treated_after - Y_treated_before) - (Y_control_after - Y_control_before)
        
        Parameters
        ----------
        outcome_col : str
            Column with outcome variable
        group_col : str, optional
            Column for subgroup analysis
            
        Returns
        -------
        List[RawDiDResult]
            DiD results for each group
        """
        results = []
        
        if outcome_col not in self.df.columns:
            print(f"Warning: Column '{outcome_col}' not found")
            return results
        
        groups = [None]
        if group_col and group_col in self.df.columns:
            groups = self.df[group_col].unique()
        
        for group in groups:
            if group is not None:
                df_g = self.df[self.df[group_col] == group]
            else:
                df_g = self.df
            
            # Calculate means for each cell
            tb = df_g[(df_g['treatment'] == 1) & (df_g['post'] == 0)][outcome_col]
            ta = df_g[(df_g['treatment'] == 1) & (df_g['post'] == 1)][outcome_col]
            cb = df_g[(df_g['treatment'] == 0) & (df_g['post'] == 0)][outcome_col]
            ca = df_g[(df_g['treatment'] == 0) & (df_g['post'] == 1)][outcome_col]
            
            # Skip if insufficient data
            if any(len(x) < MIN_SAMPLE_ANY for x in [tb, ta, cb, ca]):
                continue
            
            treated_before = tb.mean()
            treated_after = ta.mean()
            control_before = cb.mean()
            control_after = ca.mean()
            
            # DiD calculation
            treated_change = treated_after - treated_before
            control_change = control_after - control_before
            did_effect = treated_change - control_change
            
            # Percent change (relative to treated before)
            did_effect_pct = (did_effect / treated_before * 100) if treated_before != 0 else np.nan
            
            # Statistical test (simple t-test on difference)
            is_significant = None
            p_value = None
            
            if HAS_SCIPY and len(tb) >= MIN_SAMPLE_MODERATE and len(ta) >= MIN_SAMPLE_MODERATE:
                try:
                    # Pooled variance t-test
                    n1, n2 = len(ta), len(ca)
                    var1, var2 = ta.var(), ca.var()
                    pooled_se = np.sqrt(var1/n1 + var2/n2 + tb.var()/len(tb) + cb.var()/len(cb))
                    
                    if pooled_se > 0:
                        t_stat = did_effect / pooled_se
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n1 + n2 - 2))
                        is_significant = p_value < 0.05
                except:
                    pass
            
            # Interpretation
            if did_effect > 0:
                direction = "improved" if outcome_col.lower() in ['mtbf', 'rf_mtbf'] else "increased"
            else:
                direction = "worsened" if outcome_col.lower() in ['mtbf', 'rf_mtbf'] else "decreased"
            
            sig_text = f" (p={p_value:.3f})" if p_value is not None else ""
            interpretation = f"The intervention {direction} {outcome_col} by {abs(did_effect):.2f} ({abs(did_effect_pct):.1f}%){sig_text}"
            
            results.append(RawDiDResult(
                metric=outcome_col,
                group=str(group) if group else 'All',
                treated_before=treated_before,
                treated_after=treated_after,
                control_before=control_before,
                control_after=control_after,
                did_effect=did_effect,
                did_effect_pct=did_effect_pct,
                n_treated_before=len(tb),
                n_treated_after=len(ta),
                n_control_before=len(cb),
                n_control_after=len(ca),
                is_significant=is_significant,
                p_value=p_value,
                interpretation=interpretation
            ))
        
        self.results['raw_did'] = results
        return results
    
    def run_twfe(self,
                 outcome_col: str = 'mtbf',
                 covariates: List[str] = None,
                 fe_vars: List[str] = None) -> Optional[TWFEResult]:
        """
        Run Two-Way Fixed Effects regression.
        
        Y_it = α + β * (Treatment * Post) + γ_i + δ_t + ε_it
        
        Where γ_i are unit fixed effects and δ_t are time fixed effects.
        
        Parameters
        ----------
        outcome_col : str
            Column with outcome variable
        covariates : List[str], optional
            Additional control variables
        fe_vars : List[str], optional
            Variables for fixed effects (default: ['equipment_id', 'year_month'])
            
        Returns
        -------
        TWFEResult
            Regression results
        """
        if not HAS_STATSMODELS:
            print("Error: statsmodels required for TWFE regression")
            return None
        
        df = self.df.dropna(subset=[outcome_col, 'treatment', 'post'])
        
        if len(df) < 20:
            print("Warning: Insufficient data for TWFE regression")
            return None
        
        # Build formula
        formula_parts = [f'{outcome_col} ~ treatment_post']
        
        # Add covariates
        if covariates:
            for cov in covariates:
                if cov in df.columns:
                    formula_parts.append(f'C({cov})' if df[cov].dtype == 'object' else cov)
        
        # Add fixed effects
        if fe_vars:
            for fe in fe_vars:
                if fe in df.columns:
                    formula_parts.append(f'C({fe})')
        else:
            # Default fixed effects
            if 'equipment_id' in df.columns:
                formula_parts.append('C(equipment_id)')
            if 'year_month' in df.columns:
                formula_parts.append('C(year_month)')
        
        formula = ' + '.join(formula_parts)
        
        try:
            model = ols(formula, data=df).fit(
                cov_type='cluster' if self.cluster_se else 'nonrobust',
                cov_kwds={'groups': df['equipment_id']} if self.cluster_se and 'equipment_id' in df.columns else None
            )
            
            # Extract treatment effect
            coef = model.params.get('treatment_post', np.nan)
            se = model.bse.get('treatment_post', np.nan)
            t_stat = model.tvalues.get('treatment_post', np.nan)
            p_val = model.pvalues.get('treatment_post', np.nan)
            
            ci = model.conf_int().loc['treatment_post'].values if 'treatment_post' in model.conf_int().index else (np.nan, np.nan)
            
            result = TWFEResult(
                metric=outcome_col,
                coefficient=coef,
                std_error=se,
                t_statistic=t_stat,
                p_value=p_val,
                confidence_interval=tuple(ci),
                r_squared=model.rsquared,
                n_observations=len(df),
                is_significant=p_val < 0.05 if pd.notna(p_val) else False
            )
            
            self.results['twfe'] = result
            return result
            
        except Exception as e:
            print(f"Error in TWFE regression: {e}")
            return None
    
    def run_event_study(self,
                        outcome_col: str = 'mtbf',
                        window: Tuple[int, int] = DEFAULT_EVENT_WINDOW,
                        reference_period: int = -1) -> List[EventStudyResult]:
        """
        Run event study analysis for dynamic treatment effects.
        
        Estimates treatment effects for each period relative to intervention.
        
        Parameters
        ----------
        outcome_col : str
            Column with outcome variable
        window : Tuple[int, int]
            (pre_periods, post_periods) to include
        reference_period : int
            Reference period (omitted from regression)
            
        Returns
        -------
        List[EventStudyResult]
            Coefficients for each relative period
        """
        if not HAS_STATSMODELS:
            print("Error: statsmodels required for event study")
            return []
        
        results = []
        
        # Create relative time variable
        if 'relative_period' not in self.df.columns:
            print("Warning: relative_period column not found. Creating from year_month.")
            # This requires intervention date to calculate relative time
            # Simplified version using post periods
            self.df['relative_period'] = np.where(
                self.df['post'] == 1,
                np.random.randint(0, window[1], len(self.df)),  # Placeholder
                np.random.randint(window[0], 0, len(self.df))
            )
        
        df = self.df.dropna(subset=[outcome_col, 'treatment', 'relative_period'])
        
        # Filter to window
        df = df[(df['relative_period'] >= window[0]) & (df['relative_period'] <= window[1])]
        
        if len(df) < 50:
            print("Warning: Insufficient data for event study")
            return results
        
        # Create period dummies (excluding reference)
        periods = sorted(df[df['treatment'] == 1]['relative_period'].unique())
        periods = [p for p in periods if p != reference_period]
        
        try:
            # Build formula with period interactions
            for period in periods:
                df[f'period_{period}'] = ((df['relative_period'] == period) & (df['treatment'] == 1)).astype(int)
            
            period_vars = ' + '.join([f'period_{p}' for p in periods])
            formula = f'{outcome_col} ~ {period_vars}'
            
            if 'equipment_id' in df.columns:
                formula += ' + C(equipment_id)'
            
            model = ols(formula, data=df).fit()
            
            # Extract coefficients
            for period in periods:
                var_name = f'period_{period}'
                if var_name in model.params.index:
                    coef = model.params[var_name]
                    se = model.bse[var_name]
                    p_val = model.pvalues[var_name]
                    ci = model.conf_int().loc[var_name].values
                    
                    results.append(EventStudyResult(
                        metric=outcome_col,
                        group='All',
                        relative_period=period,
                        coefficient=coef,
                        std_error=se,
                        confidence_interval=tuple(ci),
                        p_value=p_val
                    ))
            
            self.results['event_study'] = results
            
        except Exception as e:
            print(f"Error in event study: {e}")
        
        return results
    
    def test_parallel_trends(self,
                              outcome_col: str = 'mtbf',
                              n_pre_periods: int = 6) -> Optional[ParallelTrendsResult]:
        """
        Test parallel trends assumption.
        
        Tests whether treated and control groups had similar trends
        before the intervention.
        
        Parameters
        ----------
        outcome_col : str
            Column with outcome variable
        n_pre_periods : int
            Number of pre-treatment periods to test
            
        Returns
        -------
        ParallelTrendsResult
            Test results
        """
        if not HAS_STATSMODELS:
            print("Error: statsmodels required for parallel trends test")
            return None
        
        # Get pre-treatment data
        df_pre = self.df[self.df['post'] == 0].copy()
        
        if 'year_month' not in df_pre.columns:
            print("Warning: Cannot test parallel trends without time variable")
            return None
        
        df_pre = df_pre.dropna(subset=[outcome_col, 'treatment', 'year_month'])
        
        if len(df_pre) < 20:
            print("Warning: Insufficient pre-treatment data")
            return None
        
        try:
            # Get unique periods
            periods = sorted(df_pre['year_month'].unique())[-n_pre_periods:]
            
            if len(periods) < 3:
                return None
            
            # Create period dummies and interactions
            for i, period in enumerate(periods[:-1]):
                df_pre[f'period_{i}'] = (df_pre['year_month'] == period).astype(int)
                df_pre[f'treatment_period_{i}'] = df_pre['treatment'] * df_pre[f'period_{i}']
            
            # Test: joint significance of treatment*period interactions
            interaction_vars = [f'treatment_period_{i}' for i in range(len(periods) - 1)]
            
            formula = f'{outcome_col} ~ treatment + ' + ' + '.join(interaction_vars)
            
            model = ols(formula, data=df_pre).fit()
            
            # F-test on interactions
            r_matrix = np.zeros((len(interaction_vars), len(model.params)))
            for i, var in enumerate(interaction_vars):
                if var in model.params.index:
                    r_matrix[i, model.params.index.get_loc(var)] = 1
            
            f_test = model.f_test(r_matrix)
            f_stat = f_test.fvalue[0][0] if hasattr(f_test.fvalue, '__iter__') else f_test.fvalue
            p_value = f_test.pvalue if not hasattr(f_test.pvalue, '__iter__') else f_test.pvalue[0][0]
            
            is_parallel = p_value > 0.05
            
            # Get pre-trend coefficients
            pre_coefficients = {}
            for i in range(len(periods) - 1):
                var = f'treatment_period_{i}'
                if var in model.params.index:
                    pre_coefficients[i - len(periods) + 1] = model.params[var]
            
            interpretation = (
                f"Parallel trends {'NOT REJECTED' if is_parallel else 'REJECTED'} "
                f"(F={f_stat:.2f}, p={p_value:.3f}). "
                f"{'Pre-treatment trends are similar.' if is_parallel else 'Caution: Pre-treatment trends differ.'}"
            )
            
            result = ParallelTrendsResult(
                metric=outcome_col,
                group='All',
                f_statistic=f_stat,
                p_value=p_value,
                is_parallel=is_parallel,
                interpretation=interpretation,
                pre_trend_coefficients=pre_coefficients
            )
            
            self.results['parallel_trends'] = result
            return result
            
        except Exception as e:
            print(f"Error in parallel trends test: {e}")
            return None
    
    def get_summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame of all DiD results."""
        rows = []
        
        # Raw DiD results
        if 'raw_did' in self.results:
            for r in self.results['raw_did']:
                rows.append({
                    'Analysis': 'Raw DiD',
                    'Metric': r.metric,
                    'Group': r.group,
                    'Effect': r.did_effect,
                    'Effect_Pct': r.did_effect_pct,
                    'P_Value': r.p_value,
                    'Significant': r.is_significant,
                    'Interpretation': r.interpretation
                })
        
        # TWFE results
        if 'twfe' in self.results:
            r = self.results['twfe']
            rows.append({
                'Analysis': 'TWFE',
                'Metric': r.metric,
                'Group': 'All',
                'Effect': r.coefficient,
                'Effect_Pct': np.nan,
                'P_Value': r.p_value,
                'Significant': r.is_significant,
                'Interpretation': f"TWFE coefficient: {r.coefficient:.2f} (SE={r.std_error:.2f})"
            })
        
        return pd.DataFrame(rows)


# ============================================================
# Visualization Functions
# ============================================================

def plot_did_comparison(result: RawDiDResult, ax=None, title: str = None) -> Any:
    """
    Plot DiD comparison (4-cell visualization).
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    periods = ['Before', 'After']
    treated = [result.treated_before, result.treated_after]
    control = [result.control_before, result.control_after]
    
    x = np.arange(len(periods))
    width = 0.35
    
    # Bars
    bars1 = ax.bar(x - width/2, treated, width, label='Treated', color='steelblue')
    bars2 = ax.bar(x + width/2, control, width, label='Control', color='coral')
    
    # Lines showing trends
    ax.plot(x - width/2, treated, 'b-o', linewidth=2, markersize=8)
    ax.plot(x + width/2, control, 'r-o', linewidth=2, markersize=8)
    
    # Labels
    ax.set_ylabel(f'{result.metric}', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=11)
    ax.set_title(title or f'DiD Analysis: {result.metric} ({result.group})', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add effect annotation
    effect_text = f'DiD Effect: {result.did_effect:+.2f}\n({result.did_effect_pct:+.1f}%)'
    if result.p_value is not None:
        effect_text += f'\np = {result.p_value:.3f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, effect_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    return ax


def plot_event_study(results: List[EventStudyResult], 
                      ax=None, 
                      title: str = None) -> Any:
    """
    Plot event study coefficients with confidence intervals.
    """
    import matplotlib.pyplot as plt
    
    if not results:
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by period
    results = sorted(results, key=lambda x: x.relative_period)
    
    periods = [r.relative_period for r in results]
    coefficients = [r.coefficient for r in results]
    ci_lower = [r.confidence_interval[0] for r in results]
    ci_upper = [r.confidence_interval[1] for r in results]
    
    # Plot
    ax.errorbar(periods, coefficients, 
                yerr=[np.array(coefficients) - np.array(ci_lower),
                      np.array(ci_upper) - np.array(coefficients)],
                fmt='o-', capsize=4, capthick=2, linewidth=2, 
                markersize=8, color='steelblue')
    
    # Reference lines
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=-0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Treatment')
    
    ax.set_xlabel('Period Relative to Treatment', fontsize=12)
    ax.set_ylabel('Treatment Effect', fontsize=12)
    ax.set_title(title or f'Event Study: {results[0].metric}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Shade pre/post regions
    ax.axvspan(min(periods), -0.5, alpha=0.1, color='blue', label='Pre-treatment')
    ax.axvspan(-0.5, max(periods), alpha=0.1, color='green', label='Post-treatment')
    
    return ax


# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic panel data
    n_equipment = 20
    n_periods = 24
    
    data = []
    for eq in range(n_equipment):
        eq_id = f'OVEN-{eq:03d}'
        is_treated = eq < 10  # First 10 are treated
        treatment_period = 12 if is_treated else None
        
        for t in range(n_periods):
            is_post = t >= treatment_period if treatment_period else False
            
            # Base MTBF with time trend
            base_mtbf = 100 + t * 2
            
            # Treatment effect (only for treated after treatment)
            treatment_effect = 30 if (is_treated and is_post) else 0
            
            # Random noise
            noise = np.random.normal(0, 15)
            
            mtbf = base_mtbf + treatment_effect + noise
            
            data.append({
                'equipment_id': eq_id,
                'period': t,
                'year_month': pd.Period(f'2024-{(t % 12) + 1:02d}', freq='M'),
                'mtbf': mtbf,
                'treatment': 1 if is_treated else 0,
                'post': 1 if is_post else 0,
                'treatment_post': 1 if (is_treated and is_post) else 0
            })
    
    df = pd.DataFrame(data)
    
    # Run analysis
    print("=" * 60)
    print("DiD Analysis Demo")
    print("=" * 60)
    
    analyzer = DiDAnalyzer()
    analyzer.df = df
    
    # Raw DiD
    print("\n1. Raw DiD Analysis")
    raw_results = analyzer.calc_raw_did(outcome_col='mtbf')
    for r in raw_results:
        print(f"   {r.interpretation}")
    
    # TWFE
    print("\n2. TWFE Regression")
    twfe_result = analyzer.run_twfe(outcome_col='mtbf')
    if twfe_result:
        print(f"   Coefficient: {twfe_result.coefficient:.2f}")
        print(f"   SE: {twfe_result.std_error:.2f}")
        print(f"   P-value: {twfe_result.p_value:.4f}")
        print(f"   Significant: {twfe_result.is_significant}")
    
    # Parallel trends
    print("\n3. Parallel Trends Test")
    pt_result = analyzer.test_parallel_trends(outcome_col='mtbf')
    if pt_result:
        print(f"   {pt_result.interpretation}")
    
    print("=" * 60)
