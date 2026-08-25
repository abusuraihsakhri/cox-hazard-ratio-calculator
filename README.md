# Cox Proportional Hazards Calculator

A pure-Python (stdlib-only) implementation of the Cox Proportional Hazards model with hazard ratio estimation, Wald tests, confidence intervals, and proportional hazards assumption checking.

## Features

- **Cox PH model fitting**: Newton-Raphson optimization of partial log-likelihood
- **Hazard ratio**: HR = exp(β)
- **Wald test**: z = β / SE(β), with p-value
- **95% Confidence intervals**: exp(β ± 1.96 × SE)
- **Proportional hazards check**: Schoenfeld residual correlation test
- **Forest plot data generation**: HR, CI, p-values for each covariate
- **Multivariate support**: Multiple covariates in a single model
- **CSV batch processing**: Auto-detect covariate columns

## Formulas

### Partial Log-Likelihood
ℓ(β) = Σ[event times] [Xi·β − d·log(Σ exp(Xj·β))]

Where:
- Xi = covariate of event subject
- d = number of events at that time
- Sum over risk set for the log term

### Hazard Ratio
HR = exp(β)

### Wald Test
z = β / SE(β)
p-value = 2 × (1 − Φ(|z|))

### Confidence Interval for HR
CI = [exp(β − 1.96×SE), exp(β + 1.96×SE)]

## Usage

### Command Line

```bash
# Fit univariate Cox model
python cli.py fit --time 1 2 3 4 5 6 7 8 --event 1 0 1 1 0 1 0 1 \
                  --covariate 0.5 1.2 0.8 1.5 0.3 1.1 0.9 1.4

# Multivariate model
python cli.py multivariate --time 1 2 3 4 5 --event 1 0 1 1 0 \
    --covariates '[[0.5,65],[1.2,70],[0.8,55],[1.5,60],[0.3,72]]' \
    --labels treatment age

# Check proportional hazards assumption
python cli.py ph-check --time 1 2 3 4 5 --event 1 0 1 1 0 \
                       --covariate 0.5 1.2 0.8 1.5 0.3

# Batch CSV processing
python cli.py batch --input sample.csv --output results.csv
```

### Python API

```python
from cox_hr import cox_ph, hazard_ratio, forest_plot_data, check_proportional_hazards

# Fit model
result = cox_ph(
    times=[1, 2, 3, 4, 5],
    events=[1, 0, 1, 1, 0],
    covariates=[[0.5], [1.2], [0.8], [1.5], [0.3]]
)
print(f"HR: {result['hazard_ratios'][0]:.4f}")
print(f"95% CI: [{result['ci_lower'][0]:.4f}, {result['ci_upper'][0]:.4f}]")
print(f"p-value: {result['p_values'][0]:.6f}")

# Forest plot data
fp = forest_plot_data(times, events, covariates, labels=["treatment"])
```

## CSV Format

Input CSV should have `time`, `event`, and covariate columns:

| time | event | treatment | age |
|------|-------|-----------|-----|
| 1    | 1     | 1         | 65  |
| 2    | 0     | 0         | 70  |

## Testing

```bash
python -m pytest test_cox_hr.py -v
```

## License

MIT
