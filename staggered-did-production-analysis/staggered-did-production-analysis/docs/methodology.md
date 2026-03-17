# Methodology Guide
## Staggered DiD Production Analysis

This document explains the statistical methods implemented in this toolkit.

---

## 1. Bathtub Curve Analysis

### Overview

The bathtub curve is a fundamental concept in reliability engineering that describes the typical failure rate pattern of equipment over its lifecycle.

```
Failure Rate
    │
    │\                            /
    │ \                          /
    │  \                        /
    │   \_______________________/
    │   DFR      CFR       IFR
    └────────────────────────────────→ Production/Time
```

### Three Phases

#### 1.1 DFR (Decreasing Failure Rate) - Initial Period

**Characteristics:**
- High initial failure rate that decreases over time
- Also called "infant mortality" or "burn-in" period
- Weibull shape parameter β < 0.85

**Causes in Bakery Equipment:**
- Manufacturing defects
- Installation errors
- Calibration issues
- Material flaws
- Initial operator errors

**Detection Method:**
```python
# Weibull distribution fitting
from scipy.stats import weibull_min
shape, loc, scale = weibull_min.fit(mtbf_data, floc=0)
if shape < 0.85:
    phase = 'DFR'
```

#### 1.2 CFR (Constant Failure Rate) - Stable Period

**Characteristics:**
- Constant, relatively low failure rate
- Random failures due to external factors
- Weibull shape parameter β ≈ 1.0 (0.85 to 1.15)

**Causes in Bakery Equipment:**
- Random external events
- Operational variations
- Environmental factors
- Human errors during normal operation

**Interpretation:**
Equipment is performing as expected. Failures are random and unpredictable, which is normal for well-maintained equipment.

#### 1.3 IFR (Increasing Failure Rate) - Wear-out Period

**Characteristics:**
- Failure rate increases over time
- Equipment degradation becomes dominant
- Weibull shape parameter β > 1.15

**Causes in Bakery Equipment:**
- Component wear
- Fatigue
- Corrosion
- Accumulated stress
- End-of-life approaching

**Action Required:**
Consider preventive replacement or major overhaul.

### Phase Boundary Detection

We use two complementary methods:

**Method 1: Weibull Beta Analysis**
```python
beta < 0.85  → DFR
0.85 ≤ beta ≤ 1.15  → CFR  
beta > 1.15  → IFR
```

**Method 2: Hazard Rate Slope Analysis**
```python
normalized_slope < -0.15  → DFR (decreasing)
|normalized_slope| ≤ 0.15  → CFR (flat)
normalized_slope > 0.15  → IFR (increasing)
```

---

## 2. Survival Analysis

### 2.1 Right-Censoring

**The Problem:**
Not all equipment failures are observed. Some equipment is still running at the end of observation, removed from service, or lost to follow-up.

**Handling:**
```
Event = 1: Failure observed (exact MTBF known)
Event = 0: Censored (only know MTBF > observed time)
```

Proper handling of censored data is critical for unbiased estimates.

### 2.2 Cox Proportional Hazards Model

**Formula:**
```
h(t|X) = h₀(t) × exp(β'X)
```

Where:
- h(t|X): Hazard rate at time t given covariates X
- h₀(t): Baseline hazard (unspecified)
- β: Coefficients to estimate
- X: Covariates (treatment, equipment type, etc.)

**Interpretation:**
- **Hazard Ratio (HR) > 1**: Increased failure risk
- **Hazard Ratio (HR) < 1**: Decreased failure risk
- **HR = 1**: No effect

**Example Output:**
```
Covariate      HR      95% CI        p-value
treatment      0.65    [0.52, 0.81]  0.0002 *
equipment_Oven 1.12    [0.89, 1.41]  0.3401
```

Treatment reduces failure hazard by 35% (HR = 0.65).

### 2.3 AFT (Accelerated Failure Time) Models

**Formula:**
```
log(T) = μ + β'X + σW
```

Where:
- T: Survival time
- W: Error term following specified distribution
- σ: Scale parameter

**Distributions Available:**
- Weibull: Flexible for all bathtub phases
- Log-Normal: When failures accumulate around a mean
- Log-Logistic: Heavy-tailed distributions

**Model Selection:**
Use AIC (Akaike Information Criterion) to select best distribution:
```
AIC = 2k - 2ln(L)
```
Lower AIC = better fit.

---

## 3. Difference-in-Differences (DiD) Analysis

### 3.1 Basic Concept

DiD estimates causal effects by comparing changes over time between treated and control groups.

**The Key Equation:**
```
DiD = (Y_treated_after - Y_treated_before) - (Y_control_after - Y_control_before)
```

**Visual Representation:**
```
MTBF
  │      Treated (actual)
  │           ●────────●
  │          /         △ Treatment Effect
  │         /     ●----● Counterfactual
  │        /     /
  │       ●────●
  │      Control
  └──────┬───────┬────────→ Time
       Before  After
```

### 3.2 Parallel Trends Assumption

**Critical Assumption:**
In the absence of treatment, treated and control groups would have followed the same trend.

**Testing:**
1. Visual inspection of pre-treatment trends
2. Statistical test for differential pre-trends

```python
# Pre-treatment periods regression
Y ~ treatment + time + treatment*time (pre-period only)

# F-test on treatment*time interactions
# H0: All pre-treatment interactions = 0
# If p > 0.05: Parallel trends not rejected
```

### 3.3 Two-Way Fixed Effects (TWFE)

**Model:**
```
Y_it = α + β × (Treatment × Post) + γ_i + δ_t + ε_it
```

Where:
- γ_i: Unit (equipment) fixed effects
- δ_t: Time fixed effects
- β: Treatment effect (parameter of interest)

**Advantages:**
- Controls for time-invariant unit characteristics
- Controls for common time shocks
- More robust than simple DiD

### 3.4 Staggered DiD

When treatment is implemented at different times:

**Challenge:**
Standard TWFE can be biased when:
- Treatment timing varies across units
- Treatment effects are heterogeneous

**Solution:**
Event study design showing dynamic treatment effects:
```
Y_it = α + Σ_k β_k × D_it^k + γ_i + δ_t + ε_it
```

Where D_it^k = 1 if unit i is k periods from treatment at time t.

---

## 4. Reservoir Computing (Echo State Network)

### 4.1 Overview

ESN is a recurrent neural network approach for time series analysis with efficient training.

**Architecture:**
```
Input → [Fixed Random Reservoir] → Trained Output
         (recurrent dynamics)
```

### 4.2 Mathematical Formulation

**State Update:**
```
x(t+1) = tanh(W_in × u(t) + W × x(t))
```

Where:
- x(t): Reservoir state at time t
- u(t): Input at time t
- W_in: Input weights (fixed, random)
- W: Reservoir weights (fixed, random, sparse)

**Output:**
```
y(t) = W_out × [x(t); u(t)]
```

W_out is trained via ridge regression.

### 4.3 Key Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| n_reservoir | 100 | More nodes = more capacity |
| spectral_radius | 0.95 | < 1 for stability, higher = longer memory |
| sparsity | 0.1 | Fraction of non-zero weights |
| ridge | 1e-6 | Regularization strength |

### 4.4 Application: Right-Censoring Imputation

For censored observations (equipment still running), we predict expected MTBF:

1. Train ESN on observed failures
2. Input: cumulative production
3. Output: expected MTBF

**Constraint:**
Imputed MTBF ≥ observed time (since failure hasn't occurred yet).

---

## 5. Interpretation Guidelines

### Reading Results

#### Bathtub Analysis
```
DFR_End: 2000 units → Equipment stabilizes after 2000 units
CFR_End: 45000 units → Wear-out begins after 45000 units
Weibull_Beta: 1.05 → Currently in CFR phase (stable)
```

#### Survival Analysis
```
Cox HR for treatment: 0.65 → 35% reduction in failure hazard
AFT acceleration factor: 1.4 → 40% longer time to failure
```

#### DiD Analysis
```
DiD Effect: +30 hours → Intervention improves MTBF by 30 hours
P-value: 0.02 → Statistically significant at 5% level
```

### Confidence Assessment

| Sample Size | Confidence | Recommendation |
|-------------|------------|----------------|
| < 10 | Low | Use with caution |
| 10-30 | Moderate | Reasonable estimates |
| 30-100 | Good | Reliable inference |
| > 100 | High | Robust conclusions |

---

## References

1. Meeker, W. Q., & Escobar, L. A. (1998). Statistical Methods for Reliability Data.
2. Cox, D. R. (1972). Regression Models and Life-Tables. Journal of the Royal Statistical Society.
3. Angrist, J. D., & Pischke, J. S. (2009). Mostly Harmless Econometrics.
4. Jaeger, H. (2001). The "Echo State" Approach to Analysing and Training Recurrent Neural Networks.
5. Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing.
