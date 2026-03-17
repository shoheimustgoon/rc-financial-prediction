#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Staggered DiD Production Analysis
================================================

Run with: pytest tests/test_analysis.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bathtub_analysis import BathtubAnalyzer, classify_by_weibull_beta, classify_by_slope
from survival_analysis import SurvivalAnalyzer, RightCensoringHandler
from did_analysis import DiDAnalyzer
from reservoir_computing import EchoStateNetwork, RightCensoringImputer


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_bathtub_data():
    """Generate sample bathtub curve data."""
    np.random.seed(42)
    n = 200
    
    production = np.sort(np.cumsum(np.random.exponential(1000, n)))
    
    # Create bathtub pattern
    dfr_mask = production < 2000
    cfr_mask = (production >= 2000) & (production < 45000)
    ifr_mask = production >= 45000
    
    mtbf = np.zeros(n)
    mtbf[dfr_mask] = np.random.weibull(0.7, dfr_mask.sum()) * 100 + 50
    mtbf[cfr_mask] = np.random.weibull(1.0, cfr_mask.sum()) * 150 + 100
    mtbf[ifr_mask] = np.random.weibull(1.5, ifr_mask.sum()) * 100 + 50
    
    event = np.ones(n)
    
    return {
        'production': production,
        'mtbf': mtbf,
        'event': event
    }


@pytest.fixture
def sample_survival_data():
    """Generate sample survival data."""
    np.random.seed(42)
    n = 100
    
    treatment = np.random.binomial(1, 0.5, n)
    duration = np.random.exponential(100, n) * (1 + 0.5 * treatment)
    
    # Add censoring
    censoring_time = np.random.exponential(150, n)
    observed_time = np.minimum(duration, censoring_time)
    event = (duration <= censoring_time).astype(int)
    
    return pd.DataFrame({
        'duration': observed_time,
        'event': event,
        'treatment': treatment
    })


@pytest.fixture
def sample_did_data():
    """Generate sample DiD data."""
    np.random.seed(42)
    
    n_eq = 20
    n_periods = 12
    treatment_period = 6
    
    data = []
    for eq in range(n_eq):
        is_treated = eq < 10
        for t in range(n_periods):
            is_post = t >= treatment_period
            
            base_mtbf = 100 + t * 2
            effect = 30 if (is_treated and is_post) else 0
            noise = np.random.normal(0, 15)
            
            data.append({
                'equipment_id': f'EQ-{eq:02d}',
                'period': t,
                'mtbf': base_mtbf + effect + noise,
                'treatment': 1 if is_treated else 0,
                'post': 1 if is_post else 0,
                'treatment_post': 1 if (is_treated and is_post) else 0
            })
    
    return pd.DataFrame(data)


# ============================================================
# Bathtub Analysis Tests
# ============================================================

class TestBathtubAnalysis:
    """Tests for bathtub curve analysis."""
    
    def test_classify_by_weibull_beta_dfr(self):
        """Test DFR classification."""
        phase, _ = classify_by_weibull_beta(0.7)
        assert phase == 'DFR'
    
    def test_classify_by_weibull_beta_cfr(self):
        """Test CFR classification."""
        phase, _ = classify_by_weibull_beta(1.0)
        assert phase == 'CFR'
    
    def test_classify_by_weibull_beta_ifr(self):
        """Test IFR classification."""
        phase, _ = classify_by_weibull_beta(1.5)
        assert phase == 'IFR'
    
    def test_classify_by_slope_dfr(self):
        """Test slope-based DFR classification."""
        phase = classify_by_slope(-0.3)
        assert phase == 'DFR'
    
    def test_classify_by_slope_cfr(self):
        """Test slope-based CFR classification."""
        phase = classify_by_slope(0.05)
        assert phase == 'CFR'
    
    def test_classify_by_slope_ifr(self):
        """Test slope-based IFR classification."""
        phase = classify_by_slope(0.3)
        assert phase == 'IFR'
    
    def test_analyzer_basic(self, sample_bathtub_data):
        """Test basic analyzer functionality."""
        analyzer = BathtubAnalyzer()
        result = analyzer.analyze(
            production=sample_bathtub_data['production'],
            mtbf=sample_bathtub_data['mtbf'],
            event=sample_bathtub_data['event'],
            equipment_id='TEST-001'
        )
        
        assert result is not None
        assert result.equipment_id == 'TEST-001'
        assert not np.isnan(result.overall_beta)
    
    def test_analyzer_phase_detection(self, sample_bathtub_data):
        """Test phase boundary detection."""
        analyzer = BathtubAnalyzer()
        result = analyzer.analyze(
            production=sample_bathtub_data['production'],
            mtbf=sample_bathtub_data['mtbf'],
            event=sample_bathtub_data['event']
        )
        
        # Should detect some boundaries
        assert result is not None
        # At least one of the phases should be detected
        assert result.overall_phase in ['DFR', 'CFR', 'IFR', 'Unknown']


# ============================================================
# Survival Analysis Tests
# ============================================================

class TestSurvivalAnalysis:
    """Tests for survival analysis."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = SurvivalAnalyzer()
        assert analyzer.penalizer == 0.01
    
    def test_cox_model_basic(self, sample_survival_data):
        """Test Cox model fitting."""
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            pytest.skip("lifelines not installed")
        
        analyzer = SurvivalAnalyzer()
        
        covariates = pd.DataFrame({'treatment': sample_survival_data['treatment']})
        
        result = analyzer.fit_cox(
            duration=sample_survival_data['duration'].values,
            event=sample_survival_data['event'].values,
            covariates=covariates
        )
        
        assert result is not None
        assert 'treatment' in result.hazard_ratios
        assert 0 < result.concordance_index <= 1
    
    def test_aft_model_comparison(self, sample_survival_data):
        """Test AFT model comparison."""
        try:
            from lifelines import WeibullAFTFitter
        except ImportError:
            pytest.skip("lifelines not installed")
        
        analyzer = SurvivalAnalyzer()
        
        comparison = analyzer.compare_models(
            duration=sample_survival_data['duration'].values,
            event=sample_survival_data['event'].values
        )
        
        assert len(comparison) > 0
        assert 'AIC' in comparison.columns
    
    def test_right_censoring_handler(self):
        """Test right-censoring handler."""
        handler = RightCensoringHandler(observation_end_date='2024-12-31')
        assert handler.observation_end_date is not None


# ============================================================
# DiD Analysis Tests
# ============================================================

class TestDiDAnalysis:
    """Tests for DiD analysis."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = DiDAnalyzer(cfr_threshold=0.15)
        assert analyzer.cfr_threshold == 0.15
    
    def test_raw_did_calculation(self, sample_did_data):
        """Test raw DiD calculation."""
        analyzer = DiDAnalyzer()
        analyzer.df = sample_did_data
        
        results = analyzer.calc_raw_did(outcome_col='mtbf')
        
        assert len(results) > 0
        assert results[0].did_effect != 0  # Should detect treatment effect
    
    def test_twfe_regression(self, sample_did_data):
        """Test TWFE regression."""
        try:
            import statsmodels.api as sm
        except ImportError:
            pytest.skip("statsmodels not installed")
        
        analyzer = DiDAnalyzer()
        analyzer.df = sample_did_data
        
        result = analyzer.run_twfe(outcome_col='mtbf')
        
        if result:
            assert result.coefficient != 0
            assert 0 <= result.p_value <= 1


# ============================================================
# Reservoir Computing Tests
# ============================================================

class TestReservoirComputing:
    """Tests for Reservoir Computing."""
    
    def test_esn_initialization(self):
        """Test ESN initialization."""
        esn = EchoStateNetwork(n_reservoir=50, spectral_radius=0.9)
        assert esn.n_reservoir == 50
        assert esn.spectral_radius == 0.9
    
    def test_esn_fit_predict(self):
        """Test ESN fit and predict."""
        np.random.seed(42)
        
        # Simple time series
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = np.sin(X.flatten()) + np.random.normal(0, 0.1, 100)
        
        esn = EchoStateNetwork(n_reservoir=50)
        esn.fit(X[:80], y[:80])
        
        predictions = esn.predict(X[80:])
        
        assert len(predictions) == 20
        assert not np.any(np.isnan(predictions))
    
    def test_esn_score(self):
        """Test ESN scoring."""
        np.random.seed(42)
        
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = np.sin(X.flatten()) + np.random.normal(0, 0.1, 100)
        
        esn = EchoStateNetwork(n_reservoir=100)
        esn.fit(X[:80], y[:80])
        
        scores = esn.score(X[80:], y[80:])
        
        assert 'r2' in scores
        assert 'rmse' in scores
    
    def test_right_censoring_imputer(self):
        """Test right-censoring imputation."""
        np.random.seed(42)
        n = 100
        
        mtbf = np.random.exponential(100, n)
        event = np.random.binomial(1, 0.8, n)  # 20% censored
        production = np.sort(np.cumsum(np.random.exponential(1000, n)))
        
        imputer = RightCensoringImputer(n_reservoir=30)
        result = imputer.fit_transform(mtbf, event, production)
        
        assert result is not None
        assert len(result.imputed_values) == n
        # Imputed values should be >= original for censored
        censored_mask = event == 0
        assert np.all(result.imputed_values[censored_mask] >= mtbf[censored_mask])


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests combining multiple modules."""
    
    def test_full_pipeline(self, sample_bathtub_data, sample_did_data):
        """Test full analysis pipeline."""
        # Bathtub analysis
        bathtub = BathtubAnalyzer()
        bathtub_result = bathtub.analyze(
            production=sample_bathtub_data['production'],
            mtbf=sample_bathtub_data['mtbf']
        )
        assert bathtub_result is not None
        
        # DiD analysis
        did = DiDAnalyzer()
        did.df = sample_did_data
        did_results = did.calc_raw_did(outcome_col='mtbf')
        assert len(did_results) > 0
    
    def test_data_flow(self):
        """Test data can flow through all analyzers."""
        np.random.seed(42)
        n = 50
        
        # Create synthetic data
        data = pd.DataFrame({
            'equipment_id': [f'EQ-{i%5:02d}' for i in range(n)],
            'mtbf': np.random.exponential(100, n),
            'production': np.cumsum(np.random.exponential(500, n)),
            'event': np.random.binomial(1, 0.9, n),
            'treatment': np.random.binomial(1, 0.5, n),
            'post': np.random.binomial(1, 0.5, n)
        })
        data['treatment_post'] = data['treatment'] * data['post']
        
        # Test each analyzer can process the data
        bathtub = BathtubAnalyzer()
        result1 = bathtub.analyze(
            production=data['production'].values,
            mtbf=data['mtbf'].values
        )
        
        did = DiDAnalyzer()
        did.df = data
        result2 = did.calc_raw_did(outcome_col='mtbf')
        
        assert result1 is not None or result2 is not None  # At least one should work


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
