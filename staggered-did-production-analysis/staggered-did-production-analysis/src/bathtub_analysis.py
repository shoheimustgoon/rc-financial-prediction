#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bathtub Curve Analysis Module
=============================

Equipment failure rate analysis following the bathtub curve pattern:
- DFR (Decreasing Failure Rate): Initial/burn-in period
- CFR (Constant Failure Rate): Stable/useful life period  
- IFR (Increasing Failure Rate): Wear-out period

This module provides tools to:
1. Detect lifecycle phases from failure data
2. Fit Weibull distribution for shape parameter (beta)
3. Calculate phase boundaries
4. Visualize the bathtub curve
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# Constants
# ============================================================

# Weibull beta thresholds for phase classification
BETA_DFR_THRESHOLD = 0.85   # beta < 0.85 indicates DFR
BETA_IFR_THRESHOLD = 1.15   # beta > 1.15 indicates IFR

# Slope-based phase classification threshold
SLOPE_CFR_THRESHOLD = 0.15  # |normalized_slope| < 0.15 indicates CFR

# Default breakpoints (can be overridden per equipment type)
DEFAULT_BREAKPOINTS = {
    'Oven': {'DFR_end': 2000, 'CFR_end': 50000},
    'Mixer': {'DFR_end': 1500, 'CFR_end': 40000},
    'Proofer': {'DFR_end': 1800, 'CFR_end': 45000},
    'Conveyor': {'DFR_end': 3000, 'CFR_end': 60000},
    'Default': {'DFR_end': 2000, 'CFR_end': 50000}
}


# ============================================================
# Data Classes
# ============================================================

@dataclass
class PhaseInfo:
    """Information about a detected lifecycle phase"""
    phase_name: str          # 'DFR', 'CFR', or 'IFR'
    start_production: float  # Production count at phase start
    end_production: float    # Production count at phase end
    weibull_beta: float      # Weibull shape parameter
    mean_mtbf: float         # Mean time between failures in this phase
    slope: float             # Hazard rate slope
    n_failures: int          # Number of failures in this phase


@dataclass  
class BathtubResult:
    """Complete bathtub curve analysis result"""
    equipment_id: str
    equipment_type: str
    dfr_end: Optional[float]
    cfr_end: Optional[float]
    overall_beta: float
    overall_phase: str
    phases: List[PhaseInfo]
    bin_centers: np.ndarray
    bin_hazard_rates: np.ndarray
    interpretation: str


# ============================================================
# Utility Functions
# ============================================================

def classify_by_weibull_beta(beta: float, 
                              beta_low: float = BETA_DFR_THRESHOLD,
                              beta_high: float = BETA_IFR_THRESHOLD) -> Tuple[str, str]:
    """
    Classify failure pattern based on Weibull beta parameter.
    
    Parameters
    ----------
    beta : float
        Weibull shape parameter
    beta_low : float
        Lower threshold for CFR classification
    beta_high : float
        Upper threshold for CFR classification
        
    Returns
    -------
    Tuple[str, str]
        (phase_name, interpretation_string)
    """
    if pd.isna(beta):
        return 'Unknown', 'β not available'
    
    if beta < beta_low:
        return 'DFR', f'β={beta:.2f} < {beta_low} (Decreasing Failure Rate)'
    elif beta > beta_high:
        return 'IFR', f'β={beta:.2f} > {beta_high} (Increasing Failure Rate)'
    else:
        return 'CFR', f'{beta_low} ≤ β={beta:.2f} ≤ {beta_high} (Constant Failure Rate)'


def classify_by_slope(normalized_slope: float, 
                      threshold: float = SLOPE_CFR_THRESHOLD) -> str:
    """
    Classify failure pattern based on hazard rate slope.
    
    Parameters
    ----------
    normalized_slope : float
        Normalized slope of hazard rate
    threshold : float
        Threshold for CFR classification
        
    Returns
    -------
    str
        Phase name: 'DFR', 'CFR', or 'IFR'
    """
    if pd.isna(normalized_slope):
        return 'Unknown'
    
    if normalized_slope < -threshold:
        return 'DFR'
    elif normalized_slope > threshold:
        return 'IFR'
    else:
        return 'CFR'


def calculate_hazard_rate(time_to_failure: np.ndarray, 
                          censored: np.ndarray = None,
                          n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate empirical hazard rate using life table method.
    
    Parameters
    ----------
    time_to_failure : np.ndarray
        Array of times to failure (or censoring)
    censored : np.ndarray, optional
        Array indicating censored observations (0=failure, 1=censored)
    n_bins : int
        Number of bins for hazard rate calculation
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (bin_centers, hazard_rates)
    """
    if censored is None:
        censored = np.zeros(len(time_to_failure))
    
    # Create bins
    min_val, max_val = time_to_failure.min(), time_to_failure.max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    bin_width = (max_val - min_val) / n_bins
    
    hazard_rates = []
    bin_centers = []
    
    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        center = (left + right) / 2
        
        # Number at risk at start of interval
        at_risk = np.sum(time_to_failure >= left)
        
        # Number of failures in interval
        failures = np.sum((time_to_failure >= left) & 
                         (time_to_failure < right) & 
                         (censored == 0))
        
        if at_risk > 0:
            hazard = failures / (at_risk * bin_width)
            hazard_rates.append(hazard)
            bin_centers.append(center)
    
    return np.array(bin_centers), np.array(hazard_rates)


# ============================================================
# Main Bathtub Analyzer Class
# ============================================================

class BathtubAnalyzer:
    """
    Analyzer for equipment failure patterns following the bathtub curve.
    
    The bathtub curve describes three distinct phases in equipment lifecycle:
    1. DFR (Decreasing Failure Rate): Early failures due to manufacturing defects
    2. CFR (Constant Failure Rate): Stable period with random failures
    3. IFR (Increasing Failure Rate): Wear-out failures due to aging
    
    Parameters
    ----------
    cfr_threshold : float
        Threshold for CFR phase classification (default: 0.15)
    n_bins : int
        Number of bins for hazard rate calculation (default: 20)
    
    Examples
    --------
    >>> analyzer = BathtubAnalyzer()
    >>> result = analyzer.analyze(
    ...     production=df['Cumulative_Production'],
    ...     mtbf=df['MTBF'],
    ...     event=df['Event'],
    ...     equipment_id='OVEN-001',
    ...     equipment_type='Oven'
    ... )
    >>> print(f"DFR ends at: {result.dfr_end}")
    >>> print(f"Overall phase: {result.overall_phase}")
    """
    
    def __init__(self, cfr_threshold: float = SLOPE_CFR_THRESHOLD, 
                 n_bins: int = 20):
        self.cfr_threshold = cfr_threshold
        self.n_bins = n_bins
        self.results: Dict[str, BathtubResult] = {}
    
    def analyze(self,
                production: np.ndarray,
                mtbf: np.ndarray,
                event: np.ndarray = None,
                equipment_id: str = 'Unknown',
                equipment_type: str = 'Default') -> Optional[BathtubResult]:
        """
        Analyze failure data to detect bathtub curve phases.
        
        Parameters
        ----------
        production : np.ndarray
            Cumulative production count at each failure
        mtbf : np.ndarray
            Mean time between failures (or time to failure)
        event : np.ndarray, optional
            Event indicator (1=failure, 0=censored)
        equipment_id : str
            Equipment identifier
        equipment_type : str
            Type of equipment for default breakpoints
            
        Returns
        -------
        BathtubResult
            Complete analysis result including phase boundaries
        """
        # Validate input
        valid_mask = ~(pd.isna(production) | pd.isna(mtbf))
        if valid_mask.sum() < self.n_bins:
            return None
        
        prod_v = np.array(production)[valid_mask]
        mtbf_v = np.array(mtbf)[valid_mask]
        
        if event is not None:
            event_v = np.array(event)[valid_mask]
        else:
            event_v = np.ones(len(prod_v))
        
        # Sort by production
        sort_idx = np.argsort(prod_v)
        prod_sorted = prod_v[sort_idx]
        mtbf_sorted = mtbf_v[sort_idx]
        event_sorted = event_v[sort_idx]
        
        # Calculate binned statistics
        bin_edges = np.percentile(prod_sorted, np.linspace(0, 100, self.n_bins + 1))
        bin_centers = []
        bin_means = []
        
        for i in range(self.n_bins):
            mask = (prod_sorted >= bin_edges[i]) & (prod_sorted < bin_edges[i + 1])
            if i == self.n_bins - 1:
                mask = (prod_sorted >= bin_edges[i]) & (prod_sorted <= bin_edges[i + 1])
            
            if mask.sum() > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_means.append(mtbf_sorted[mask].mean())
        
        if len(bin_centers) < 5:
            return None
        
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        
        # Calculate slopes between bins
        slopes = np.zeros(len(bin_means) - 1)
        for i in range(len(slopes)):
            dx = bin_centers[i + 1] - bin_centers[i]
            dy = bin_means[i + 1] - bin_means[i]
            slopes[i] = dy / dx if dx > 0 else 0
        
        # Normalize slopes for phase classification
        slope_std = np.std(slopes)
        normalized_slopes = slopes / slope_std if slope_std > 0 else slopes
        
        # Classify each segment
        phase_labels = [classify_by_slope(s, self.cfr_threshold) 
                       for s in normalized_slopes]
        
        # Detect phase boundaries
        dfr_end, cfr_end = self._detect_boundaries(phase_labels, bin_centers)
        
        # Fit Weibull distribution
        weibull_beta, weibull_phase = self._fit_weibull(mtbf_v, event_v)
        
        # Build phase info list
        phases = self._build_phases(prod_sorted, mtbf_sorted, event_sorted,
                                    dfr_end, cfr_end, bin_centers, bin_means)
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            weibull_beta, weibull_phase, dfr_end, cfr_end, phases
        )
        
        result = BathtubResult(
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            dfr_end=dfr_end,
            cfr_end=cfr_end,
            overall_beta=weibull_beta,
            overall_phase=weibull_phase,
            phases=phases,
            bin_centers=bin_centers,
            bin_hazard_rates=1 / bin_means,  # Hazard rate = 1/MTBF
            interpretation=interpretation
        )
        
        self.results[equipment_id] = result
        return result
    
    def _detect_boundaries(self, 
                           phase_labels: List[str],
                           bin_centers: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Detect DFR->CFR and CFR->IFR transition points."""
        dfr_end = None
        cfr_end = None
        
        valid_labels = [p for p in phase_labels if p != 'Unknown']
        
        if len(valid_labels) == 0:
            return dfr_end, cfr_end
        
        # Find DFR to CFR transition
        for i in range(1, len(valid_labels)):
            if valid_labels[i - 1] == 'DFR' and valid_labels[i] != 'DFR':
                if i < len(bin_centers):
                    dfr_end = bin_centers[i]
                break
        
        # Find CFR to IFR transition
        for i in range(1, len(valid_labels)):
            if valid_labels[i - 1] == 'CFR' and valid_labels[i] == 'IFR':
                if i < len(bin_centers):
                    cfr_end = bin_centers[i]
                break
        
        return dfr_end, cfr_end
    
    def _fit_weibull(self, 
                     mtbf: np.ndarray,
                     event: np.ndarray) -> Tuple[float, str]:
        """Fit Weibull distribution and return shape parameter."""
        try:
            # Use only failure events for fitting
            valid_mtbf = mtbf[event == 1]
            
            if len(valid_mtbf) < 5:
                return np.nan, 'Unknown'
            
            # Fit Weibull distribution
            shape, loc, scale = stats.weibull_min.fit(valid_mtbf, floc=0)
            weibull_beta = shape
            weibull_phase, _ = classify_by_weibull_beta(weibull_beta)
            
            return weibull_beta, weibull_phase
        except Exception:
            return np.nan, 'Unknown'
    
    def _build_phases(self,
                      prod: np.ndarray,
                      mtbf: np.ndarray,
                      event: np.ndarray,
                      dfr_end: Optional[float],
                      cfr_end: Optional[float],
                      bin_centers: np.ndarray,
                      bin_means: np.ndarray) -> List[PhaseInfo]:
        """Build list of phase information."""
        phases = []
        
        # Define phase boundaries
        min_prod, max_prod = prod.min(), prod.max()
        
        boundaries = [
            ('DFR', min_prod, dfr_end if dfr_end else min_prod * 1.1),
            ('CFR', dfr_end if dfr_end else min_prod, cfr_end if cfr_end else max_prod * 0.9),
            ('IFR', cfr_end if cfr_end else max_prod * 0.9, max_prod)
        ]
        
        for phase_name, start, end in boundaries:
            if start >= end:
                continue
            
            mask = (prod >= start) & (prod < end)
            if mask.sum() == 0:
                continue
            
            phase_mtbf = mtbf[mask]
            phase_event = event[mask]
            
            # Fit Weibull for this phase
            phase_beta, _ = self._fit_weibull(phase_mtbf, phase_event)
            
            # Calculate slope for this phase
            phase_bin_mask = (bin_centers >= start) & (bin_centers < end)
            if phase_bin_mask.sum() >= 2:
                phase_bins = bin_centers[phase_bin_mask]
                phase_means = bin_means[phase_bin_mask]
                slope = (phase_means[-1] - phase_means[0]) / (phase_bins[-1] - phase_bins[0])
            else:
                slope = 0.0
            
            phases.append(PhaseInfo(
                phase_name=phase_name,
                start_production=start,
                end_production=end,
                weibull_beta=phase_beta,
                mean_mtbf=np.nanmean(phase_mtbf),
                slope=slope,
                n_failures=int(np.sum(phase_event == 1))
            ))
        
        return phases
    
    def _generate_interpretation(self,
                                  weibull_beta: float,
                                  weibull_phase: str,
                                  dfr_end: Optional[float],
                                  cfr_end: Optional[float],
                                  phases: List[PhaseInfo]) -> str:
        """Generate human-readable interpretation of results."""
        lines = []
        
        # Overall assessment
        lines.append(f"Overall Weibull β = {weibull_beta:.2f}" if not pd.isna(weibull_beta) 
                    else "Weibull β: Not available")
        lines.append(f"Dominant failure pattern: {weibull_phase}")
        
        # Phase boundaries
        if dfr_end:
            lines.append(f"Initial period (DFR) ends at: {dfr_end:,.0f} units")
        if cfr_end:
            lines.append(f"Stable period (CFR) ends at: {cfr_end:,.0f} units")
        
        # Phase details
        for phase in phases:
            lines.append(f"\n{phase.phase_name} Phase:")
            lines.append(f"  - Production range: {phase.start_production:,.0f} - {phase.end_production:,.0f}")
            lines.append(f"  - Failures: {phase.n_failures}")
            lines.append(f"  - Mean MTBF: {phase.mean_mtbf:.1f} hours")
            if not pd.isna(phase.weibull_beta):
                lines.append(f"  - Phase β: {phase.weibull_beta:.2f}")
        
        return '\n'.join(lines)
    
    def analyze_dataframe(self,
                          df: pd.DataFrame,
                          production_col: str = 'Cumulative_Production',
                          mtbf_col: str = 'MTBF',
                          event_col: str = 'Event',
                          equipment_col: str = 'Equipment_ID',
                          type_col: str = 'Equipment_Type') -> pd.DataFrame:
        """
        Analyze multiple equipment from a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with failure data
        production_col : str
            Column name for cumulative production
        mtbf_col : str
            Column name for MTBF values
        event_col : str
            Column name for event indicator
        equipment_col : str
            Column name for equipment identifier
        type_col : str
            Column name for equipment type
            
        Returns
        -------
        pd.DataFrame
            Summary of bathtub analysis for each equipment
        """
        results_list = []
        
        for eq_id in df[equipment_col].unique():
            eq_df = df[df[equipment_col] == eq_id]
            
            eq_type = eq_df[type_col].iloc[0] if type_col in eq_df.columns else 'Default'
            
            result = self.analyze(
                production=eq_df[production_col].values,
                mtbf=eq_df[mtbf_col].values,
                event=eq_df[event_col].values if event_col in eq_df.columns else None,
                equipment_id=eq_id,
                equipment_type=eq_type
            )
            
            if result:
                results_list.append({
                    'Equipment_ID': result.equipment_id,
                    'Equipment_Type': result.equipment_type,
                    'DFR_End': result.dfr_end,
                    'CFR_End': result.cfr_end,
                    'Weibull_Beta': result.overall_beta,
                    'Overall_Phase': result.overall_phase,
                    'N_Phases': len(result.phases),
                    'Interpretation': result.interpretation
                })
        
        return pd.DataFrame(results_list)
    
    def get_phase_for_observation(self,
                                   production: float,
                                   equipment_id: str = None,
                                   equipment_type: str = 'Default') -> str:
        """
        Determine which phase a single observation belongs to.
        
        Parameters
        ----------
        production : float
            Cumulative production count
        equipment_id : str, optional
            Equipment ID to use previously analyzed boundaries
        equipment_type : str
            Equipment type for default boundaries
            
        Returns
        -------
        str
            Phase name: 'DFR', 'CFR', or 'IFR'
        """
        if pd.isna(production):
            return 'Unknown'
        
        # Use analyzed results if available
        if equipment_id and equipment_id in self.results:
            result = self.results[equipment_id]
            dfr_end = result.dfr_end
            cfr_end = result.cfr_end
        else:
            # Use default breakpoints
            bp = DEFAULT_BREAKPOINTS.get(equipment_type, DEFAULT_BREAKPOINTS['Default'])
            dfr_end = bp['DFR_end']
            cfr_end = bp['CFR_end']
        
        if dfr_end and production < dfr_end:
            return 'DFR'
        elif cfr_end and production >= cfr_end:
            return 'IFR'
        else:
            return 'CFR'


# ============================================================
# Visualization Functions
# ============================================================

def plot_bathtub_curve(result: BathtubResult,
                       ax=None,
                       show_phases: bool = True,
                       title: str = None) -> Any:
    """
    Plot the bathtub curve for an equipment.
    
    Parameters
    ----------
    result : BathtubResult
        Analysis result from BathtubAnalyzer
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    show_phases : bool
        Whether to show phase boundaries
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot hazard rate
    ax.plot(result.bin_centers, result.bin_hazard_rates, 
            'b-o', linewidth=2, markersize=6, label='Observed Hazard Rate')
    
    # Show phase boundaries
    if show_phases:
        if result.dfr_end:
            ax.axvline(x=result.dfr_end, color='orange', linestyle='--', 
                      linewidth=2, label=f'DFR→CFR ({result.dfr_end:,.0f})')
        if result.cfr_end:
            ax.axvline(x=result.cfr_end, color='red', linestyle='--', 
                      linewidth=2, label=f'CFR→IFR ({result.cfr_end:,.0f})')
        
        # Shade phases
        x_min, x_max = result.bin_centers.min(), result.bin_centers.max()
        y_max = result.bin_hazard_rates.max() * 1.1
        
        if result.dfr_end:
            ax.axvspan(x_min, result.dfr_end, alpha=0.2, color='yellow', label='DFR')
        
        if result.dfr_end and result.cfr_end:
            ax.axvspan(result.dfr_end, result.cfr_end, alpha=0.2, color='green', label='CFR')
        elif result.dfr_end:
            ax.axvspan(result.dfr_end, x_max, alpha=0.2, color='green', label='CFR')
        
        if result.cfr_end:
            ax.axvspan(result.cfr_end, x_max, alpha=0.2, color='red', label='IFR')
    
    ax.set_xlabel('Cumulative Production', fontsize=12)
    ax.set_ylabel('Hazard Rate (failures/unit time)', fontsize=12)
    ax.set_title(title or f'Bathtub Curve - {result.equipment_id}', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add Weibull info
    info_text = f'Weibull β = {result.overall_beta:.2f}\nPhase: {result.overall_phase}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    return ax


# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic bathtub curve data
    n_obs = 500
    production = np.sort(np.random.exponential(20000, n_obs).cumsum() / 10)
    
    # DFR phase (high initial hazard)
    dfr_mask = production < 2000
    # CFR phase (constant hazard)
    cfr_mask = (production >= 2000) & (production < 45000)
    # IFR phase (increasing hazard)
    ifr_mask = production >= 45000
    
    mtbf = np.zeros(n_obs)
    mtbf[dfr_mask] = np.random.weibull(0.7, dfr_mask.sum()) * 100 + 50
    mtbf[cfr_mask] = np.random.weibull(1.0, cfr_mask.sum()) * 200 + 100
    mtbf[ifr_mask] = np.random.weibull(1.5, ifr_mask.sum()) * 150 + 50
    
    event = np.ones(n_obs)
    
    # Run analysis
    analyzer = BathtubAnalyzer()
    result = analyzer.analyze(
        production=production,
        mtbf=mtbf,
        event=event,
        equipment_id='OVEN-001',
        equipment_type='Oven'
    )
    
    if result:
        print("=" * 60)
        print("Bathtub Curve Analysis Results")
        print("=" * 60)
        print(result.interpretation)
        print("=" * 60)
