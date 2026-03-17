#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reservoir Computing Module
==========================

Implements Echo State Network (ESN) for time series analysis:
- Right-censoring data imputation
- Time series prediction
- Missing value handling

The Echo State Network is a recurrent neural network with:
- Fixed random input weights
- Fixed random reservoir weights (with controlled spectral radius)
- Trained output weights (linear regression)

This provides efficient training while capturing temporal dynamics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# Data Classes
# ============================================================

@dataclass
class ESNResult:
    """Results from Echo State Network training"""
    r2_score: float
    rmse: float
    predictions: np.ndarray
    states: np.ndarray
    n_reservoir: int
    spectral_radius: float


@dataclass
class ImputationResult:
    """Results from right-censoring imputation"""
    original_values: np.ndarray
    imputed_values: np.ndarray
    imputation_mask: np.ndarray  # True where imputation was performed
    confidence: np.ndarray  # Confidence score for imputed values
    method: str


# ============================================================
# Echo State Network Class
# ============================================================

class EchoStateNetwork:
    """
    Echo State Network (Reservoir Computing) implementation.
    
    ESN is a type of recurrent neural network where:
    - Input-to-reservoir weights (W_in) are random and fixed
    - Reservoir weights (W) are random, sparse, and scaled to spectral_radius
    - Output weights (W_out) are trained using ridge regression
    
    Parameters
    ----------
    n_reservoir : int
        Number of reservoir neurons (default: 100)
    spectral_radius : float
        Spectral radius of reservoir weight matrix (default: 0.95)
        Controls the "memory" of the network. Should be < 1 for stability.
    sparsity : float
        Fraction of non-zero weights in reservoir (default: 0.1)
    noise : float
        Noise level added to reservoir states (default: 0.001)
    input_scaling : float
        Scaling factor for input weights (default: 0.1)
    ridge : float
        Ridge regression regularization parameter (default: 1e-6)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Examples
    --------
    >>> esn = EchoStateNetwork(n_reservoir=100, spectral_radius=0.95)
    >>> esn.fit(X_train, y_train)
    >>> predictions = esn.predict(X_test)
    """
    
    def __init__(self,
                 n_reservoir: int = 100,
                 spectral_radius: float = 0.95,
                 sparsity: float = 0.1,
                 noise: float = 0.001,
                 input_scaling: float = 0.1,
                 ridge: float = 1e-6,
                 random_state: int = 42):
        
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_scaling = input_scaling
        self.ridge = ridge
        self.random_state = random_state
        
        # Initialize random state
        np.random.seed(random_state)
        
        # Weights (initialized on first fit)
        self.W_in = None  # Input weights
        self.W = None     # Reservoir weights
        self.W_out = None # Output weights
        
        # Training state
        self.is_fitted = False
        self.last_state = None
    
    def _initialize_reservoir(self, n_inputs: int):
        """
        Initialize reservoir weights.
        
        Parameters
        ----------
        n_inputs : int
            Number of input features
        """
        # Input weights: random values scaled by input_scaling
        self.W_in = np.random.randn(self.n_reservoir, n_inputs) * self.input_scaling
        
        # Reservoir weights: sparse random matrix
        W = np.random.randn(self.n_reservoir, self.n_reservoir)
        
        # Apply sparsity mask
        mask = np.random.rand(self.n_reservoir, self.n_reservoir) < self.sparsity
        W = W * mask
        
        # Scale to desired spectral radius
        current_radius = np.max(np.abs(np.linalg.eigvals(W)))
        if current_radius > 0:
            W = W * (self.spectral_radius / current_radius)
        
        self.W = W
    
    def _compute_states(self, X: np.ndarray, 
                        initial_state: np.ndarray = None) -> np.ndarray:
        """
        Compute reservoir states for input sequence.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequence, shape (n_samples, n_inputs)
        initial_state : np.ndarray, optional
            Initial reservoir state
            
        Returns
        -------
        np.ndarray
            Reservoir states, shape (n_samples, n_reservoir)
        """
        n_samples, n_inputs = X.shape
        
        # Initialize weights if needed
        if self.W_in is None:
            self._initialize_reservoir(n_inputs)
        
        # Initialize states
        states = np.zeros((n_samples, self.n_reservoir))
        
        if initial_state is not None:
            state = initial_state.copy()
        else:
            state = np.zeros(self.n_reservoir)
        
        # Compute states sequentially
        for t in range(n_samples):
            # Update state: tanh(W_in @ input + W @ previous_state)
            state = np.tanh(self.W_in @ X[t] + self.W @ state)
            
            # Add noise
            state += self.noise * np.random.randn(self.n_reservoir)
            
            states[t] = state
        
        # Save final state for continuation
        self.last_state = state.copy()
        
        return states
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EchoStateNetwork':
        """
        Train the ESN on input-output pairs.
        
        Uses ridge regression to learn output weights from reservoir states.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequence, shape (n_samples,) or (n_samples, n_inputs)
        y : np.ndarray
            Target values, shape (n_samples,)
            
        Returns
        -------
        EchoStateNetwork
            Self (for method chaining)
        """
        # Ensure 2D input
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and len(X.shape) == 2:
            X = X.T
        
        # Ensure 1D output
        y = np.array(y).flatten()
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
        
        # Compute reservoir states
        states = self._compute_states(X)
        
        # Extend states with input (direct input-output connection)
        extended_states = np.hstack([states, X])
        
        # Train output weights using ridge regression
        # W_out = (X'X + ridge*I)^-1 X'y
        n_features = extended_states.shape[1]
        reg_matrix = extended_states.T @ extended_states + self.ridge * np.eye(n_features)
        
        self.W_out = np.linalg.solve(reg_matrix, extended_states.T @ y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, 
                continue_from_last: bool = False) -> np.ndarray:
        """
        Predict output for new input sequence.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequence
        continue_from_last : bool
            Whether to continue from last training state
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Ensure 2D input
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and len(X.shape) == 2:
            X = X.T
        
        # Compute states
        initial_state = self.last_state if continue_from_last else None
        states = self._compute_states(X, initial_state)
        
        # Extend states with input
        extended_states = np.hstack([states, X])
        
        # Predict
        predictions = extended_states @ self.W_out
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate prediction metrics.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequence
        y : np.ndarray
            True values
            
        Returns
        -------
        Dict[str, float]
            Dictionary with R2 score and RMSE
        """
        predictions = self.predict(X)
        y = np.array(y).flatten()
        
        # R2 score
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # RMSE
        rmse = np.sqrt(np.mean((y - predictions) ** 2))
        
        return {'r2': r2, 'rmse': rmse}
    
    def get_states(self, X: np.ndarray) -> np.ndarray:
        """Get reservoir states for analysis."""
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        return self._compute_states(X)


# ============================================================
# Right-Censoring Imputation
# ============================================================

class RightCensoringImputer:
    """
    Imputes right-censored observations using Reservoir Computing.
    
    For equipment still operational at observation end, we don't observe
    the actual failure time. This class uses ESN to predict the expected
    MTBF for censored observations based on historical patterns.
    
    Parameters
    ----------
    n_reservoir : int
        Number of reservoir neurons
    spectral_radius : float
        Spectral radius of reservoir
    lookback : int
        Number of historical observations to use for prediction
    
    Examples
    --------
    >>> imputer = RightCensoringImputer()
    >>> result = imputer.fit_transform(
    ...     mtbf=df['MTBF'],
    ...     event=df['event'],
    ...     production=df['Cumulative_Production']
    ... )
    """
    
    def __init__(self,
                 n_reservoir: int = 50,
                 spectral_radius: float = 0.9,
                 lookback: int = 10):
        
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.lookback = lookback
        self.esn = None
        self.group_models: Dict[str, EchoStateNetwork] = {}
    
    def fit(self,
            mtbf: np.ndarray,
            event: np.ndarray,
            production: np.ndarray,
            group: np.ndarray = None) -> 'RightCensoringImputer':
        """
        Fit imputation models on observed (non-censored) data.
        
        Parameters
        ----------
        mtbf : np.ndarray
            MTBF values (some may be censored)
        event : np.ndarray
            Event indicator (1=failure, 0=censored)
        production : np.ndarray
            Cumulative production for temporal ordering
        group : np.ndarray, optional
            Group labels for stratified modeling
            
        Returns
        -------
        RightCensoringImputer
            Self
        """
        # Filter to observed events
        observed_mask = event == 1
        
        if observed_mask.sum() < self.lookback + 5:
            print("Warning: Insufficient observed data for ESN training")
            return self
        
        # Create features from production sequence
        X_obs = production[observed_mask].reshape(-1, 1)
        y_obs = mtbf[observed_mask]
        
        # Sort by production
        sort_idx = np.argsort(X_obs.flatten())
        X_obs = X_obs[sort_idx]
        y_obs = y_obs[sort_idx]
        
        if group is None:
            # Single model
            self.esn = EchoStateNetwork(
                n_reservoir=self.n_reservoir,
                spectral_radius=self.spectral_radius
            )
            self.esn.fit(X_obs, y_obs)
        else:
            # Group-specific models
            group_obs = group[observed_mask][sort_idx]
            
            for g in np.unique(group_obs):
                g_mask = group_obs == g
                if g_mask.sum() >= self.lookback + 5:
                    esn = EchoStateNetwork(
                        n_reservoir=self.n_reservoir,
                        spectral_radius=self.spectral_radius
                    )
                    esn.fit(X_obs[g_mask], y_obs[g_mask])
                    self.group_models[g] = esn
        
        return self
    
    def transform(self,
                  mtbf: np.ndarray,
                  event: np.ndarray,
                  production: np.ndarray,
                  group: np.ndarray = None) -> ImputationResult:
        """
        Impute censored MTBF values.
        
        Parameters
        ----------
        mtbf : np.ndarray
            MTBF values
        event : np.ndarray
            Event indicator
        production : np.ndarray
            Cumulative production
        group : np.ndarray, optional
            Group labels
            
        Returns
        -------
        ImputationResult
            Original and imputed values with confidence scores
        """
        imputed = mtbf.copy()
        confidence = np.ones(len(mtbf))
        imputation_mask = event == 0
        
        if group is None and self.esn is not None:
            # Use single model
            X_cens = production[imputation_mask].reshape(-1, 1)
            if len(X_cens) > 0:
                predictions = self.esn.predict(X_cens)
                
                # Imputed value should be at least the observed duration
                # (since failure hasn't occurred yet)
                imputed[imputation_mask] = np.maximum(
                    predictions,
                    mtbf[imputation_mask]
                )
                
                # Lower confidence for imputed values
                confidence[imputation_mask] = 0.7
                
        elif group is not None:
            # Use group-specific models
            for g, esn in self.group_models.items():
                g_mask = (group == g) & imputation_mask
                if g_mask.sum() > 0:
                    X_g = production[g_mask].reshape(-1, 1)
                    predictions = esn.predict(X_g)
                    
                    imputed[g_mask] = np.maximum(predictions, mtbf[g_mask])
                    confidence[g_mask] = 0.7
        
        return ImputationResult(
            original_values=mtbf,
            imputed_values=imputed,
            imputation_mask=imputation_mask,
            confidence=confidence,
            method='RC_ESN'
        )
    
    def fit_transform(self,
                      mtbf: np.ndarray,
                      event: np.ndarray,
                      production: np.ndarray,
                      group: np.ndarray = None) -> ImputationResult:
        """
        Fit models and transform in one step.
        """
        self.fit(mtbf, event, production, group)
        return self.transform(mtbf, event, production, group)


# ============================================================
# Time Series Prediction
# ============================================================

class MTBFPredictor:
    """
    Predict future MTBF trends using Reservoir Computing.
    
    Uses historical MTBF patterns to forecast expected reliability
    at future production levels.
    
    Parameters
    ----------
    n_reservoir : int
        Number of reservoir neurons
    spectral_radius : float
        Spectral radius
    horizon : int
        Default prediction horizon
    """
    
    def __init__(self,
                 n_reservoir: int = 100,
                 spectral_radius: float = 0.95,
                 horizon: int = 10):
        
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.horizon = horizon
        self.model = None
        self.feature_stats = {}
    
    def fit(self,
            production: np.ndarray,
            mtbf: np.ndarray,
            additional_features: pd.DataFrame = None) -> 'MTBFPredictor':
        """
        Fit prediction model.
        
        Parameters
        ----------
        production : np.ndarray
            Cumulative production values
        mtbf : np.ndarray
            MTBF values
        additional_features : pd.DataFrame, optional
            Additional features for prediction
            
        Returns
        -------
        MTBFPredictor
            Self
        """
        # Sort by production
        sort_idx = np.argsort(production)
        production = production[sort_idx]
        mtbf = mtbf[sort_idx]
        
        # Create feature matrix
        # Features: production, log(production), production^2
        X = np.column_stack([
            production,
            np.log1p(production),
            production ** 2 / 1e6
        ])
        
        if additional_features is not None:
            X = np.hstack([X, additional_features.iloc[sort_idx].values])
        
        # Normalize features
        self.feature_stats['mean'] = X.mean(axis=0)
        self.feature_stats['std'] = X.std(axis=0) + 1e-10
        X_norm = (X - self.feature_stats['mean']) / self.feature_stats['std']
        
        # Normalize target
        self.feature_stats['y_mean'] = mtbf.mean()
        self.feature_stats['y_std'] = mtbf.std() + 1e-10
        y_norm = (mtbf - self.feature_stats['y_mean']) / self.feature_stats['y_std']
        
        # Train ESN
        self.model = EchoStateNetwork(
            n_reservoir=self.n_reservoir,
            spectral_radius=self.spectral_radius
        )
        self.model.fit(X_norm, y_norm)
        
        return self
    
    def predict(self,
                production: np.ndarray,
                additional_features: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict MTBF for given production levels.
        
        Parameters
        ----------
        production : np.ndarray
            Production levels to predict for
        additional_features : pd.DataFrame, optional
            Additional features
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (predictions, confidence_intervals)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted")
        
        # Create features
        X = np.column_stack([
            production,
            np.log1p(production),
            production ** 2 / 1e6
        ])
        
        if additional_features is not None:
            X = np.hstack([X, additional_features.values])
        
        # Normalize
        X_norm = (X - self.feature_stats['mean']) / self.feature_stats['std']
        
        # Predict
        y_pred_norm = self.model.predict(X_norm)
        
        # Denormalize
        predictions = y_pred_norm * self.feature_stats['y_std'] + self.feature_stats['y_mean']
        
        # Simple confidence interval (based on training variance)
        ci_width = 1.96 * self.feature_stats['y_std']
        ci_lower = predictions - ci_width
        ci_upper = predictions + ci_width
        
        return predictions, np.column_stack([ci_lower, ci_upper])


# ============================================================
# Utility Functions
# ============================================================

def apply_rc_correction(df: pd.DataFrame,
                         mtbf_col: str = 'MTBF',
                         event_col: str = 'event',
                         production_col: str = 'Cumulative_Production',
                         group_col: str = None,
                         output_col: str = 'MTBF_RC') -> pd.DataFrame:
    """
    Apply RC-based right-censoring correction to DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    mtbf_col : str
        Column with MTBF values
    event_col : str
        Column with event indicator
    production_col : str
        Column with cumulative production
    group_col : str, optional
        Column for grouping
    output_col : str
        Name for output column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with RC-corrected MTBF column
    """
    df = df.copy()
    
    imputer = RightCensoringImputer()
    
    group = df[group_col].values if group_col and group_col in df.columns else None
    
    result = imputer.fit_transform(
        mtbf=df[mtbf_col].values,
        event=df[event_col].values,
        production=df[production_col].values,
        group=group
    )
    
    df[output_col] = result.imputed_values
    df[f'{output_col}_confidence'] = result.confidence
    df[f'{output_col}_imputed'] = result.imputation_mask
    
    return df


# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic data
    n_obs = 150
    production = np.sort(np.cumsum(np.random.exponential(1000, n_obs)))
    
    # MTBF with production-dependent trend
    true_mtbf = 100 + 0.005 * production + 20 * np.sin(production / 10000)
    noise = np.random.normal(0, 15, n_obs)
    mtbf = true_mtbf + noise
    mtbf = np.maximum(mtbf, 10)  # Ensure positive
    
    # Some censored observations (20%)
    event = np.random.binomial(1, 0.8, n_obs)
    
    print("=" * 60)
    print("Reservoir Computing Demo")
    print("=" * 60)
    
    # Test ESN
    print("\n1. Echo State Network Training")
    esn = EchoStateNetwork(n_reservoir=50, spectral_radius=0.9)
    
    # Split data
    train_size = int(0.8 * n_obs)
    X_train = production[:train_size].reshape(-1, 1)
    y_train = mtbf[:train_size]
    X_test = production[train_size:].reshape(-1, 1)
    y_test = mtbf[train_size:]
    
    esn.fit(X_train, y_train)
    scores = esn.score(X_test, y_test)
    print(f"   R2 Score: {scores['r2']:.4f}")
    print(f"   RMSE: {scores['rmse']:.2f}")
    
    # Test right-censoring imputation
    print("\n2. Right-Censoring Imputation")
    imputer = RightCensoringImputer()
    result = imputer.fit_transform(mtbf, event, production)
    
    n_imputed = result.imputation_mask.sum()
    print(f"   Observations imputed: {n_imputed}")
    print(f"   Average imputation adjustment: {(result.imputed_values - result.original_values)[result.imputation_mask].mean():.2f}")
    
    # Test MTBF predictor
    print("\n3. MTBF Prediction")
    predictor = MTBFPredictor(n_reservoir=100)
    predictor.fit(production, mtbf)
    
    future_production = np.linspace(production.max(), production.max() * 1.5, 20)
    predictions, ci = predictor.predict(future_production)
    
    print(f"   Current max production: {production.max():,.0f}")
    print(f"   Predicted MTBF at 150% production: {predictions[-1]:.1f} hours")
    print(f"   95% CI: [{ci[-1, 0]:.1f}, {ci[-1, 1]:.1f}]")
    
    print("=" * 60)
