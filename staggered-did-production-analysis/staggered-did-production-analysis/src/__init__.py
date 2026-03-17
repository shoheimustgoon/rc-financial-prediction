"""
Staggered DiD Production Analysis
=================================

A comprehensive analysis toolkit for bakery manufacturing equipment reliability.

Modules:
- bathtub_analysis: Equipment failure lifecycle analysis (DFR/CFR/IFR)
- survival_analysis: Cox/AFT survival models with right-censoring
- did_analysis: Difference-in-Differences causal inference
- reservoir_computing: Echo State Network for time series

Example:
    >>> from src.bathtub_analysis import BathtubAnalyzer
    >>> from src.survival_analysis import SurvivalAnalyzer
    >>> from src.did_analysis import DiDAnalyzer
    >>> 
    >>> analyzer = BathtubAnalyzer()
    >>> result = analyzer.analyze(production, mtbf, event)
"""

__version__ = '1.0.0'
__author__ = 'shoheimustgoon'

from .bathtub_analysis import BathtubAnalyzer, BathtubResult, PhaseInfo
from .survival_analysis import SurvivalAnalyzer, CoxResult, AFTResult
from .did_analysis import DiDAnalyzer, RawDiDResult, TWFEResult
from .reservoir_computing import EchoStateNetwork, RightCensoringImputer

__all__ = [
    'BathtubAnalyzer',
    'BathtubResult',
    'PhaseInfo',
    'SurvivalAnalyzer',
    'CoxResult',
    'AFTResult',
    'DiDAnalyzer',
    'RawDiDResult',
    'TWFEResult',
    'EchoStateNetwork',
    'RightCensoringImputer'
]
