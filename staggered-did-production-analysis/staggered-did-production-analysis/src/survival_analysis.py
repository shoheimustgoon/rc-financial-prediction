#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Survival Analysis Module
========================

Implements survival analysis methods for equipment reliability:
- Cox Proportional Hazards Model
- Accelerated Failure Time (AFT) Models
- Right-censoring handling
- Kaplan-Meier estimation

This module provides tools to:
1. Fit Cox PH models with covariates
2. Fit AFT models (Weibull, Log-Normal, Log-Logistic)
3. Handle right-censored observations
4. Generate survival curves and hazard functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

# Check for lifelines availability
try:
    from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
    from lifelines import KaplanMeierFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("Warning: lifelines not installed. Survival analysis will be limited.")

# Check for scipy availability
try:
    from scipy import stats
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# Data Classes
# ============================================================

@dataclass
class CoxResult:
    """Results from Cox Proportional Hazards model"""
    coefficients: Dict[str, float]
    hazard_ratios: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    p_values: Dict[str, float]
    concordance_index: float
    log_likelihood: float
    aic: float
    n_observations: int
    n_events: int
    summary_df: pd.DataFrame = None
    

@dataclass
class AFTResult:
    """Results from Accelerated Failure Time model"""
    distribution: str
    coefficients: Dict[str, float]
    acceleration_factors: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    p_values: Dict[str, float]
    aic: float
    bic: float
    log_likelihood: float
    n_observations: int
    n_events: int
    median_survival: float
    summary_df: pd.DataFrame = None


@dataclass
class SurvivalData:
    """Prepared data for survival analysis"""
    duration: np.ndarray
    event: np.ndarray
    covariates: pd.DataFrame
    equipment_ids: np.ndarray
    groups: np.ndarray = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for lifelines"""
        df = self.covariates.copy()
        df['duration'] = self.duration
        df['event'] = self.event
        df['equipment_id'] = self.equipment_ids
        if self.groups is not None:
            df['group'] = self.groups
        return df


# ============================================================
# Right-Censoring Handler
# ============================================================

class RightCensoringHandler:
    """
    Handles right-censored observations in survival data.
    
    Right-censoring occurs when:
    - Equipment is still operational at observation end
    - Equipment is removed from service before failure
    - Follow-up is lost
    
    Parameters
    ----------
    observation_end_date : str or datetime
        End date of observation period
    """
    
    def __init__(self, observation_end_date: Union[str, pd.Timestamp] = None):
        self.observation_end_date = pd.Timestamp(observation_end_date) if observation_end_date else None
    
    def create_censored_data(self,
                              df: pd.DataFrame,
                              last_event_col: str = 'Last_Error_DateTime',
                              duration_col: str = 'MTBF',
                              equipment_col: str = 'Equipment_ID',
                              group_col: str = None) -> pd.DataFrame:
        """
        Create dataset with right-censored observations.
        
        For each equipment, adds a censored observation from the last
        failure to the observation end date.
        
        Parameters
        ----------
        df : pd.DataFrame
            Original failure data
        last_event_col : str
            Column with last event timestamp
        duration_col : str
            Column with failure duration
        equipment_col : str
            Column with equipment identifier
        group_col : str, optional
            Column for group (treatment) indicator
            
        Returns
        -------
        pd.DataFrame
            Data with added censored observations
        """
        if self.observation_end_date is None:
            # Use max date from data + end of month
            max_date = pd.to_datetime(df[last_event_col]).max()
            self.observation_end_date = max_date + pd.offsets.MonthEnd(0)
        
        censored_records = []
        
        for eq_id in df[equipment_col].unique():
            eq_df = df[df[equipment_col] == eq_id]
            
            # Find last event for this equipment
            last_event = pd.to_datetime(eq_df[last_event_col]).max()
            
            # Calculate time from last event to observation end
            censored_duration = (self.observation_end_date - last_event).total_seconds() / 3600
            
            if censored_duration > 0:
                record = {
                    equipment_col: eq_id,
                    'duration': censored_duration,
                    'event': 0,  # Censored
                    'Last_Error_Date': last_event,
                    'Censored_End_Date': self.observation_end_date
                }
                
                # Copy group info if available
                if group_col and group_col in eq_df.columns:
                    record[group_col] = eq_df[group_col].iloc[0]
                
                # Copy other relevant columns
                for col in ['Equipment_Type', 'Line', 'ChamberGroup']:
                    if col in eq_df.columns:
                        record[col] = eq_df[col].iloc[0]
                
                censored_records.append(record)
        
        # Create censored DataFrame
        censored_df = pd.DataFrame(censored_records)
        
        # Prepare original data
        original_df = df.copy()
        original_df['duration'] = original_df[duration_col]
        original_df['event'] = 1  # All original records are failures
        
        # Combine
        combined = pd.concat([original_df, censored_df], ignore_index=True)
        
        return combined


# ============================================================
# Main Survival Analyzer Class
# ============================================================

class SurvivalAnalyzer:
    """
    Comprehensive survival analysis for equipment reliability.
    
    Implements:
    - Cox Proportional Hazards model
    - Weibull AFT model
    - Log-Normal AFT model
    - Log-Logistic AFT model
    - Kaplan-Meier estimation
    
    Parameters
    ----------
    penalizer : float
        L2 penalizer for regularization (default: 0.01)
        
    Examples
    --------
    >>> analyzer = SurvivalAnalyzer()
    >>> cox_result = analyzer.fit_cox(
    ...     duration=df['duration'],
    ...     event=df['event'],
    ...     covariates=df[['treatment', 'equipment_type', 'line']]
    ... )
    >>> print(f"Concordance: {cox_result.concordance_index:.3f}")
    """
    
    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.cox_model = None
        self.aft_models: Dict[str, Any] = {}
        self.km_fitter = None
        self.results: Dict[str, Any] = {}
    
    def prepare_data(self,
                     df: pd.DataFrame,
                     duration_col: str = 'duration',
                     event_col: str = 'event',
                     covariate_cols: List[str] = None,
                     equipment_col: str = 'Equipment_ID',
                     group_col: str = None) -> SurvivalData:
        """
        Prepare data for survival analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
        duration_col : str
            Column with time to event
        event_col : str
            Column with event indicator (1=failure, 0=censored)
        covariate_cols : List[str]
            Columns to use as covariates
        equipment_col : str
            Column with equipment identifier
        group_col : str, optional
            Column for group/treatment indicator
            
        Returns
        -------
        SurvivalData
            Prepared survival data object
        """
        # Clean data
        clean_df = df.dropna(subset=[duration_col, event_col]).copy()
        
        # Remove invalid durations
        clean_df = clean_df[clean_df[duration_col] > 0]
        
        # Prepare covariates
        if covariate_cols:
            # One-hot encode categorical variables
            covariates = pd.get_dummies(clean_df[covariate_cols], drop_first=True)
        else:
            covariates = pd.DataFrame()
        
        return SurvivalData(
            duration=clean_df[duration_col].values,
            event=clean_df[event_col].values.astype(int),
            covariates=covariates,
            equipment_ids=clean_df[equipment_col].values if equipment_col in clean_df.columns else np.arange(len(clean_df)),
            groups=clean_df[group_col].values if group_col and group_col in clean_df.columns else None
        )
    
    def fit_cox(self,
                duration: np.ndarray,
                event: np.ndarray,
                covariates: pd.DataFrame = None) -> Optional[CoxResult]:
        """
        Fit Cox Proportional Hazards model.
        
        The Cox PH model estimates hazard ratios for covariates without
        specifying the baseline hazard function.
        
        h(t|X) = h0(t) * exp(β'X)
        
        Parameters
        ----------
        duration : np.ndarray
            Time to event (failure or censoring)
        event : np.ndarray
            Event indicator (1=failure, 0=censored)
        covariates : pd.DataFrame, optional
            Covariate data
            
        Returns
        -------
        CoxResult
            Model results including coefficients and hazard ratios
        """
        if not HAS_LIFELINES:
            print("Error: lifelines not installed")
            return None
        
        # Prepare DataFrame
        df = pd.DataFrame({
            'duration': duration,
            'event': event
        })
        
        if covariates is not None and len(covariates.columns) > 0:
            df = pd.concat([df, covariates.reset_index(drop=True)], axis=1)
        
        # Remove any NaN or infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) < 10:
            print("Warning: Insufficient data for Cox model")
            return None
        
        try:
            # Fit model
            self.cox_model = CoxPHFitter(penalizer=self.penalizer)
            self.cox_model.fit(df, duration_col='duration', event_col='event')
            
            # Extract results
            summary = self.cox_model.summary
            
            coefficients = {}
            hazard_ratios = {}
            confidence_intervals = {}
            p_values = {}
            
            for covar in summary.index:
                coefficients[covar] = summary.loc[covar, 'coef']
                hazard_ratios[covar] = summary.loc[covar, 'exp(coef)']
                confidence_intervals[covar] = (
                    summary.loc[covar, 'exp(coef) lower 95%'],
                    summary.loc[covar, 'exp(coef) upper 95%']
                )
                p_values[covar] = summary.loc[covar, 'p']
            
            result = CoxResult(
                coefficients=coefficients,
                hazard_ratios=hazard_ratios,
                confidence_intervals=confidence_intervals,
                p_values=p_values,
                concordance_index=self.cox_model.concordance_index_,
                log_likelihood=self.cox_model.log_likelihood_,
                aic=self.cox_model.AIC_partial_,
                n_observations=len(df),
                n_events=int(df['event'].sum()),
                summary_df=summary
            )
            
            self.results['cox'] = result
            return result
            
        except Exception as e:
            print(f"Error fitting Cox model: {e}")
            return None
    
    def fit_aft(self,
                duration: np.ndarray,
                event: np.ndarray,
                covariates: pd.DataFrame = None,
                distribution: str = 'weibull') -> Optional[AFTResult]:
        """
        Fit Accelerated Failure Time model.
        
        AFT models parameterize survival time directly:
        log(T) = μ + β'X + σW
        
        Where W follows a specified distribution.
        
        Parameters
        ----------
        duration : np.ndarray
            Time to event
        event : np.ndarray
            Event indicator (1=failure, 0=censored)
        covariates : pd.DataFrame, optional
            Covariate data
        distribution : str
            Distribution type: 'weibull', 'lognormal', 'loglogistic'
            
        Returns
        -------
        AFTResult
            Model results including coefficients and acceleration factors
        """
        if not HAS_LIFELINES:
            print("Error: lifelines not installed")
            return None
        
        # Prepare DataFrame
        df = pd.DataFrame({
            'duration': duration,
            'event': event
        })
        
        if covariates is not None and len(covariates.columns) > 0:
            df = pd.concat([df, covariates.reset_index(drop=True)], axis=1)
        
        # Remove invalid values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df[df['duration'] > 0]
        
        if len(df) < 10:
            print("Warning: Insufficient data for AFT model")
            return None
        
        # Select model based on distribution
        model_map = {
            'weibull': WeibullAFTFitter,
            'lognormal': LogNormalAFTFitter,
            'loglogistic': LogLogisticAFTFitter
        }
        
        if distribution not in model_map:
            print(f"Error: Unknown distribution '{distribution}'")
            return None
        
        try:
            # Fit model
            model = model_map[distribution](penalizer=self.penalizer)
            model.fit(df, duration_col='duration', event_col='event')
            self.aft_models[distribution] = model
            
            # Extract results
            summary = model.summary
            
            coefficients = {}
            acceleration_factors = {}
            confidence_intervals = {}
            p_values = {}
            
            for idx in summary.index:
                covar = idx[1] if isinstance(idx, tuple) else idx
                coefficients[str(idx)] = summary.loc[idx, 'coef']
                acceleration_factors[str(idx)] = np.exp(summary.loc[idx, 'coef'])
                confidence_intervals[str(idx)] = (
                    summary.loc[idx, 'coef lower 95%'],
                    summary.loc[idx, 'coef upper 95%']
                )
                p_values[str(idx)] = summary.loc[idx, 'p']
            
            result = AFTResult(
                distribution=distribution,
                coefficients=coefficients,
                acceleration_factors=acceleration_factors,
                confidence_intervals=confidence_intervals,
                p_values=p_values,
                aic=model.AIC_,
                bic=model.BIC_,
                log_likelihood=model.log_likelihood_,
                n_observations=len(df),
                n_events=int(df['event'].sum()),
                median_survival=model.median_survival_time_,
                summary_df=summary
            )
            
            self.results[f'aft_{distribution}'] = result
            return result
            
        except Exception as e:
            print(f"Error fitting {distribution} AFT model: {e}")
            return None
    
    def fit_kaplan_meier(self,
                          duration: np.ndarray,
                          event: np.ndarray,
                          group: np.ndarray = None,
                          group_labels: Dict[Any, str] = None) -> Dict[str, Any]:
        """
        Fit Kaplan-Meier survival estimator.
        
        Non-parametric estimate of the survival function:
        S(t) = P(T > t)
        
        Parameters
        ----------
        duration : np.ndarray
            Time to event
        event : np.ndarray
            Event indicator
        group : np.ndarray, optional
            Group labels for stratified analysis
        group_labels : Dict, optional
            Labels for groups
            
        Returns
        -------
        Dict[str, Any]
            Kaplan-Meier results by group
        """
        if not HAS_LIFELINES:
            print("Error: lifelines not installed")
            return {}
        
        results = {}
        
        if group is None:
            # Single group
            self.km_fitter = KaplanMeierFitter()
            self.km_fitter.fit(duration, event)
            
            results['Overall'] = {
                'survival_function': self.km_fitter.survival_function_,
                'median_survival': self.km_fitter.median_survival_time_,
                'confidence_interval': self.km_fitter.confidence_interval_survival_function_
            }
        else:
            # Multiple groups
            for g in np.unique(group):
                mask = group == g
                label = group_labels.get(g, str(g)) if group_labels else str(g)
                
                km = KaplanMeierFitter()
                km.fit(duration[mask], event[mask], label=label)
                
                results[label] = {
                    'survival_function': km.survival_function_,
                    'median_survival': km.median_survival_time_,
                    'confidence_interval': km.confidence_interval_survival_function_
                }
        
        self.results['kaplan_meier'] = results
        return results
    
    def compare_models(self,
                       duration: np.ndarray,
                       event: np.ndarray,
                       covariates: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compare multiple AFT models using AIC/BIC.
        
        Parameters
        ----------
        duration : np.ndarray
            Time to event
        event : np.ndarray
            Event indicator
        covariates : pd.DataFrame, optional
            Covariate data
            
        Returns
        -------
        pd.DataFrame
            Comparison table with AIC, BIC for each model
        """
        distributions = ['weibull', 'lognormal', 'loglogistic']
        comparison = []
        
        for dist in distributions:
            result = self.fit_aft(duration, event, covariates, distribution=dist)
            
            if result:
                comparison.append({
                    'Distribution': dist,
                    'AIC': result.aic,
                    'BIC': result.bic,
                    'Log-Likelihood': result.log_likelihood,
                    'Median_Survival': result.median_survival
                })
        
        df = pd.DataFrame(comparison)
        
        if len(df) > 0:
            # Rank by AIC
            df['AIC_Rank'] = df['AIC'].rank()
            df = df.sort_values('AIC')
        
        return df
    
    def predict_survival(self,
                          covariates: pd.DataFrame,
                          times: np.ndarray = None,
                          model_type: str = 'cox') -> pd.DataFrame:
        """
        Predict survival probabilities for new observations.
        
        Parameters
        ----------
        covariates : pd.DataFrame
            Covariate values for prediction
        times : np.ndarray, optional
            Times at which to predict survival
        model_type : str
            Model to use: 'cox' or AFT distribution name
            
        Returns
        -------
        pd.DataFrame
            Survival probabilities at each time point
        """
        if model_type == 'cox' and self.cox_model:
            return self.cox_model.predict_survival_function(covariates, times=times)
        elif model_type in self.aft_models:
            return self.aft_models[model_type].predict_survival_function(covariates, times=times)
        else:
            print(f"Error: Model '{model_type}' not fitted")
            return pd.DataFrame()
    
    def get_hazard_ratios_summary(self) -> pd.DataFrame:
        """Get summary of hazard ratios from Cox model."""
        if 'cox' not in self.results:
            return pd.DataFrame()
        
        cox_result = self.results['cox']
        
        data = []
        for covar in cox_result.hazard_ratios:
            data.append({
                'Covariate': covar,
                'Hazard_Ratio': cox_result.hazard_ratios[covar],
                'HR_Lower_95': cox_result.confidence_intervals[covar][0],
                'HR_Upper_95': cox_result.confidence_intervals[covar][1],
                'P_Value': cox_result.p_values[covar],
                'Significant': cox_result.p_values[covar] < 0.05
            })
        
        return pd.DataFrame(data)


# ============================================================
# Visualization Functions
# ============================================================

def plot_survival_curves(analyzer: SurvivalAnalyzer,
                          ax=None,
                          show_ci: bool = True,
                          title: str = None) -> Any:
    """
    Plot Kaplan-Meier survival curves.
    
    Parameters
    ----------
    analyzer : SurvivalAnalyzer
        Analyzer with fitted Kaplan-Meier model
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    show_ci : bool
        Whether to show confidence intervals
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    
    if 'kaplan_meier' not in analyzer.results:
        print("Error: Kaplan-Meier not fitted")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    km_results = analyzer.results['kaplan_meier']
    colors = plt.cm.tab10(np.linspace(0, 1, len(km_results)))
    
    for i, (label, data) in enumerate(km_results.items()):
        sf = data['survival_function']
        ax.step(sf.index, sf.values, where='post', linewidth=2, 
                color=colors[i], label=label)
        
        if show_ci:
            ci = data['confidence_interval']
            ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                           alpha=0.2, color=colors[i], step='post')
        
        # Add median line
        median = data['median_survival']
        if pd.notna(median):
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=median, color=colors[i], linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(title or 'Kaplan-Meier Survival Curves', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    return ax


def plot_hazard_ratios(cox_result: CoxResult,
                        ax=None,
                        title: str = None) -> Any:
    """
    Plot forest plot of hazard ratios from Cox model.
    
    Parameters
    ----------
    cox_result : CoxResult
        Results from Cox model
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(6, len(cox_result.hazard_ratios) * 0.5)))
    
    covariates = list(cox_result.hazard_ratios.keys())
    y_pos = np.arange(len(covariates))
    
    hrs = [cox_result.hazard_ratios[c] for c in covariates]
    ci_low = [cox_result.confidence_intervals[c][0] for c in covariates]
    ci_high = [cox_result.confidence_intervals[c][1] for c in covariates]
    p_vals = [cox_result.p_values[c] for c in covariates]
    
    # Error bars
    xerr_low = [hr - low for hr, low in zip(hrs, ci_low)]
    xerr_high = [high - hr for hr, high in zip(hrs, ci_high)]
    
    # Colors based on significance
    colors = ['green' if p < 0.05 else 'gray' for p in p_vals]
    
    ax.errorbar(hrs, y_pos, xerr=[xerr_low, xerr_high], fmt='o', 
                capsize=4, capthick=2, markersize=8, color='black')
    
    for i, (hr, c) in enumerate(zip(hrs, colors)):
        ax.scatter([hr], [y_pos[i]], c=c, s=100, zorder=10)
    
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='HR=1 (No effect)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
    ax.set_title(title or 'Cox Model - Hazard Ratios', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add significance markers
    for i, p in enumerate(p_vals):
        marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        if marker:
            ax.text(max(hrs) * 1.1, y_pos[i], marker, va='center', fontsize=12)
    
    return ax


# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic survival data
    n_obs = 200
    
    # Covariates
    treatment = np.random.binomial(1, 0.5, n_obs)
    equipment_type = np.random.choice(['Oven', 'Mixer', 'Proofer'], n_obs)
    
    # Survival times (treatment improves survival)
    baseline_hazard = 0.01
    treatment_effect = -0.5  # Reduces hazard
    
    hazard = baseline_hazard * np.exp(treatment * treatment_effect)
    duration = np.random.exponential(1/hazard)
    
    # Censoring (20% censored)
    censoring_time = np.random.exponential(150, n_obs)
    observed_time = np.minimum(duration, censoring_time)
    event = (duration <= censoring_time).astype(int)
    
    # Create covariates DataFrame
    covariates = pd.DataFrame({
        'treatment': treatment,
        'equipment_type': equipment_type
    })
    covariates = pd.get_dummies(covariates, drop_first=True)
    
    # Run analysis
    analyzer = SurvivalAnalyzer()
    
    print("=" * 60)
    print("Survival Analysis Results")
    print("=" * 60)
    
    # Cox model
    cox_result = analyzer.fit_cox(observed_time, event, covariates)
    if cox_result:
        print("\nCox Proportional Hazards Model:")
        print(f"  Concordance Index: {cox_result.concordance_index:.3f}")
        print(f"  N Observations: {cox_result.n_observations}")
        print(f"  N Events: {cox_result.n_events}")
        print("\n  Hazard Ratios:")
        for covar, hr in cox_result.hazard_ratios.items():
            p = cox_result.p_values[covar]
            sig = '*' if p < 0.05 else ''
            print(f"    {covar}: HR={hr:.3f} (p={p:.4f}) {sig}")
    
    # AFT model comparison
    print("\n" + "-" * 40)
    print("AFT Model Comparison:")
    comparison = analyzer.compare_models(observed_time, event, covariates)
    print(comparison.to_string(index=False))
    
    print("=" * 60)
