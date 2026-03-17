# Staggered DiD Production Analysis for Bakery Manufacturing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A comprehensive statistical analysis toolkit for evaluating equipment interventions and production quality in bakery manufacturing environments. This tool implements advanced causal inference methods including Staggered Difference-in-Differences (DiD), Survival Analysis (Cox/AFT), and Reservoir Computing for time series analysis.

![Bathtub Curve Analysis](docs/images/bathtub_curve.png)

## Key Features

### 1. Equipment Failure Analysis (Bathtub Curve)
- **Initial Failure Period (DFR)**: Early-stage equipment failures due to burn-in issues
- **Stable Period (CFR)**: Constant failure rate during normal operation
- **Wear-out Period (IFR)**: Increasing failures due to equipment aging

### 2. Survival Analysis
- **Cox Proportional Hazards Model**: Semi-parametric survival analysis
- **Accelerated Failure Time (AFT) Models**: Weibull, Log-Normal, Log-Logistic distributions
- **Right-Censoring Handling**: Properly accounts for equipment still operational at observation end

### 3. Staggered DiD Analysis
- **Two-Way Fixed Effects (TWFE)**: Controls for time and unit fixed effects
- **Event Study Design**: Examines dynamic treatment effects over time
- **Parallel Trends Testing**: Validates the key DiD assumption

### 4. Reservoir Computing (Echo State Network)
- **Time Series Imputation**: Handles missing data in equipment metrics
- **Prediction**: Forecasts future failure patterns
- **Anomaly Detection**: Identifies unusual equipment behavior

## Installation

```bash
# Clone the repository
git clone https://github.com/shoheimustgoon/staggered-did-production-analysis.git
cd staggered-did-production-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### GUI Mode
```bash
python src/main_gui.py
```

### Command Line Mode
```bash
# Basic analysis
python src/main.py data/sample/bakery_equipment_data.xlsx --output outputs/

# With specific analysis type
python src/main.py data/sample/bakery_equipment_data.xlsx --output outputs/ --analysis survival

# Full analysis with all options
python src/main.py data/sample/bakery_equipment_data.xlsx \
    --output outputs/ \
    --analysis all \
    --cfr-threshold 0.15 \
    --reservoir-nodes 100
```

## Data Format

### Input Excel File Structure

The input file should contain equipment failure/error records with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| Equipment_ID | Unique equipment identifier | OVEN-001 |
| Line | Production line | Line_A |
| Error_DateTime | Error occurrence timestamp | 2024-01-15 08:30:00 |
| Error_Type | Type of error/failure | Temperature_Deviation |
| Cumulative_Production | Cumulative production count | 15000 |
| Operating_Hours | Total operating hours | 2400 |
| Intervention_Date | Date of intervention (if applicable) | 2024-02-01 |

### Sample Data Generation
```bash
python src/generate_sample_data.py --output data/sample/bakery_equipment_data.xlsx
```

## Analysis Modules

### 1. Bathtub Curve Analysis (`src/bathtub_analysis.py`)

Analyzes equipment failure patterns across lifecycle phases:

```python
from src.bathtub_analysis import BathtubAnalyzer

analyzer = BathtubAnalyzer(data)
phases = analyzer.detect_phases()
# Returns: {'DFR_end': 1500, 'CFR_end': 45000, 'weibull_beta': 1.05}
```

**Key Metrics:**
- **Weibull β (Shape Parameter)**: β < 0.85 indicates DFR, 0.85-1.15 indicates CFR, β > 1.15 indicates IFR
- **Hazard Rate Slope**: Normalized slope of failure rate over production volume

### 2. Survival Analysis (`src/survival_analysis.py`)

Implements multiple survival models for equipment reliability:

```python
from src.survival_analysis import SurvivalAnalyzer

analyzer = SurvivalAnalyzer(data)
cox_results = analyzer.fit_cox_model(covariates=['line', 'equipment_type'])
aft_results = analyzer.fit_aft_model(distribution='weibull')
```

**Models Available:**
- Cox Proportional Hazards
- Weibull AFT
- Log-Normal AFT
- Log-Logistic AFT

### 3. DiD Analysis (`src/did_analysis.py`)

Evaluates causal effects of equipment interventions:

```python
from src.did_analysis import DiDAnalyzer

analyzer = DiDAnalyzer(data, intervention_date_col='Intervention_Date')
results = analyzer.run_staggered_did(
    outcome='MTBF',
    group_col='Line',
    time_col='Month'
)
```

**Analysis Types:**
- Raw DiD (4-cell comparison)
- Two-Way Fixed Effects (TWFE)
- Event Study
- Parallel Trends Test

### 4. Reservoir Computing (`src/reservoir_computing.py`)

Echo State Network for time series analysis:

```python
from src.reservoir_computing import EchoStateNetwork

esn = EchoStateNetwork(n_reservoir=100, spectral_radius=0.95)
esn.fit(X_train, y_train)
predictions = esn.predict(X_test)
```

## Output Structure

```
outputs/
├── analysis_results_YYYYMMDD_HHMMSS.xlsx
│   ├── Summary                    # Overall analysis summary
│   ├── Bathtub_Phases             # Equipment lifecycle phases
│   ├── Survival_Data              # Survival analysis dataset
│   ├── Cox_Results                # Cox model coefficients
│   ├── AFT_Results                # AFT model parameters
│   ├── DiD_Results                # DiD effect estimates
│   ├── Event_Study                # Dynamic treatment effects
│   ├── Parallel_Trends            # Pre-trend test results
│   └── RC_Predictions             # Reservoir Computing output
├── plots/
│   ├── bathtub_curve_*.png        # Failure rate curves
│   ├── survival_curve_*.png       # Kaplan-Meier curves
│   ├── counterfactual_*.png       # DiD counterfactual plots
│   └── event_study_*.png          # Treatment effect dynamics
└── logs/
    └── analysis_log_YYYYMMDD.txt  # Detailed analysis log
```

## Methodology

### Bathtub Curve Detection

The bathtub curve represents the typical failure rate pattern of equipment:

1. **DFR (Decreasing Failure Rate)**: Initial period with high but decreasing failures
   - Caused by manufacturing defects, installation errors
   - Weibull β < 0.85

2. **CFR (Constant Failure Rate)**: Stable operational period
   - Random failures with constant probability
   - Weibull β ≈ 1.0 (0.85-1.15)

3. **IFR (Increasing Failure Rate)**: Wear-out period
   - Failures increase due to equipment degradation
   - Weibull β > 1.15

### Right-Censoring in Survival Analysis

Equipment still operational at observation end are "right-censored":

```
Event = 1: Failure observed
Event = 0: Censored (still operational / observation ended)
```

The Cox and AFT models properly handle censored observations to avoid bias.

### Staggered DiD

For interventions implemented at different times across equipment:

```
ATT = E[Y(1) - Y(0) | Treated] 
    = (Y_treated_after - Y_treated_before) - (Y_control_after - Y_control_before)
```

## Configuration

### config.yaml
```yaml
analysis:
  cfr_threshold: 0.15          # CFR classification threshold
  min_sample_size: 10          # Minimum observations for reliable analysis
  
reservoir_computing:
  n_reservoir: 100             # Reservoir size
  spectral_radius: 0.95        # ESN spectral radius
  sparsity: 0.1                # Reservoir sparsity
  noise: 0.001                 # Noise level

survival:
  distributions:
    - weibull
    - lognormal
    - loglogistic
  
did:
  event_window: [-6, 12]       # Event study window (months)
  cluster_se: true             # Cluster standard errors
```

## Dependencies

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- lifelines >= 0.27.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- openpyxl >= 3.0.0

Optional:
- emcee >= 3.1.0 (for Bayesian analysis)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{staggered_did_production,
  author = {shoheimustgoon},
  title = {Staggered DiD Production Analysis},
  year = {2025},
  url = {https://github.com/shoheimustgoon/staggered-did-production-analysis}
}
```

## Acknowledgments

- Inspired by causal inference methods in manufacturing quality control
- Cox model implementation based on lifelines library
- Echo State Network architecture follows Jaeger (2001)

## Contact

For questions or suggestions, please open an issue on GitHub.
