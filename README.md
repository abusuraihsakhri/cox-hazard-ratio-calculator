# Cox Hazard Ratio Calculator

> **Domain:** Clinical Decision Support & Biomedical Computing  
> **Standards:** CAP / CLSI / ISO Quality Frameworks

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![Audit Trail](https://img.shields.io/badge/Audit-HMAC--SHA256_Tamper--Evident-brightgreen.svg)
![Zero-PHI Guard](https://img.shields.io/badge/Guard-Zero--PHI_Outbound-blue.svg)

</div>

---

## What It Does

A pure Python implementation of the Cox Proportional Hazards model for survival analysis. Implements:

- **Hazard ratio**: HR = exp(β)
- **Partial likelihood estimation** via Newton-Raphson (simplified Breslow method for ties)
- **Wald test**: z = β / SE(β)
- **Confidence intervals** for HR: exp(β ± z_α/2 × SE)
- **P-values** from standard normal distribution
- **Proportional hazards assumption check** (Schoenfeld residuals with correlation test)
- **Forest plot data generation** for visualization

The core algorithm (`cox_hr.py`) uses only Python stdlib — no external dependencies required.

---

## Installation

No installation required for core functionality. Simply clone and run:

```bash
git clone https://github.com/abusuraihsakhri/cox-hazard-ratio-calculator.git
cd cox-hazard-ratio-calculator
```

### Optional Dependencies

For the FastAPI server and agent framework features:

```bash
pip install fastapi uvicorn pydantic pytest
```

Or create a `requirements.txt`:

```
fastapi>=0.110
uvicorn>=0.27
pydantic>=2.0
pytest>=8.0
```

---

## Quick Start

### CLI Usage

```bash
# Fit a univariate Cox PH model
python cli.py fit --time 1 2 3 4 5 6 7 8 9 10 \
                   --event 1 0 1 1 0 1 0 1 0 1 \
                   --covariate 0.5 1.2 0.8 1.5 0.3 1.1 0.9 1.4 0.6 1.0

# Fit multivariate model with JSON covariates
python cli.py multivariate --time 1 2 3 4 5 6 7 8 9 10 \
                           --event 1 0 1 1 0 1 0 1 0 1 \
                           --covariates '[[0.5,65],[1.2,70],[0.8,55],[1.5,60],[0.3,72],[1.1,68],[0.9,75],[1.4,58],[0.6,80],[1.0,62]]' \
                           --labels treatment age

# Check proportional hazards assumption
python cli.py ph-check --time 1 2 3 4 5 6 7 8 \
                        --event 1 0 1 1 0 1 0 1 \
                        --covariate 0.5 1.2 0.8 1.5 0.3 1.1 0.9 1.4

# Process a CSV file
python cli.py batch --input sample.csv --output results.csv

# Get JSON output
python cli.py fit --time 1 2 3 4 5 --event 1 0 1 1 0 --covariate 0.5 1.2 0.8 1.5 0.3 --json
```

### Python API

```python
from cox_hr import cox_ph, forest_plot_data, check_proportional_hazards

times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
events = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1], [0.9], [1.4], [0.6], [1.0]]

result = cox_ph(times, events, covariates)
print(f"Hazard Ratio: {result['hazard_ratios'][0]:.4f}")
print(f"95% CI: [{result['ci_lower'][0]:.4f}, {result['ci_upper'][0]:.4f}]")
print(f"P-value: {result['p_values'][0]:.6f}")

# Forest plot data
fp = forest_plot_data(times, events, covariates, labels=["treatment"])
```

### CSV Batch Processing

Input CSV format:
```csv
time,event,treatment,age
1,1,1,65
2,0,0,70
3,1,1,55
```

Output CSV contains: covariate, hazard_ratio, ci_lower, ci_upper, p_value, significant

---

## API Reference

### `cox_ph(times, events, covariates, max_iter=50, tol=1e-8)`

Fit a Cox Proportional Hazards model.

**Parameters:**
- `times`: List of observed times (non-negative floats)
- `events`: List of event indicators (1=event, 0=censored)
- `covariates`: List of covariate vectors per subject
- `max_iter`: Maximum Newton-Raphson iterations (default: 50)
- `tol`: Convergence tolerance (default: 1e-8)

**Returns:** Dict with keys: coefficients, hazard_ratios, se, z_scores, p_values, ci_lower, ci_upper, log_likelihood, iterations, n_subjects, n_events, schoenfeld

### `forest_plot_data(times, events, covariates, labels=None)`

Generate forest plot data.

**Returns:** List of dicts with: label, hr, ci_lower, ci_upper, p_value, significant

### `check_proportional_hazards(times, events, covariates)`

Check PH assumption using Schoenfeld residuals.

**Returns:** Dict with key `checks` containing per-covariate correlation, p_value, and assumption_holds

### `summary(times, events, covariates, labels=None)`

Return a formatted summary string of the analysis.

### `process_csv(input_path, output_path)`

Process a CSV file for Cox PH analysis. Validates paths for security.

---

## Security Features

### Zero-PHI Outbound Interceptor

The `agents/base.py` module includes PHI pattern detection blocking SSNs, MRNs, phone numbers, emails, and patient names from outbound data.

```python
from agents.base import PHIGuard, SecurityException

try:
    PHIGuard.assert_no_phi("Patient MRN-12345")  # Raises SecurityException
except SecurityException as e:
    print(f"PHI blocked: {e}")
```

### HMAC-SHA256 Audit Trail

Cryptographic, tamper-evident audit logging for all operations.

```python
from agents.base import AuditLogger

# Logging is automatic; verify integrity:
assert AuditLogger.verify_integrity() is True
```

**Important:** Set `AUDIT_SECRET_KEY` environment variable in production. Without it, a random key is generated (development only).

### Path Traversal Protection

CSV processing validates file paths to prevent directory traversal attacks.

---

## Testing

Run the full test suite:

```bash
pytest -v
```

Run with coverage:

```bash
pytest -v --cov=. --cov-report=html
```

Execute benchmarks:

```bash
python simulator.py 1000
```

---

## Docker Deployment

Build and run with Docker:

```bash
docker build -t cox-hazard-ratio-calculator .
docker run -p 8000:8000 cox-hazard-ratio-calculator fit --time 1 2 3 --event 1 0 1 --covariate 0.5 1.2 0.8
```

With docker-compose (create `.env` from `.env.example` first):

```bash
cp .env.example .env
# Edit .env to set AUDIT_SECRET_KEY
docker-compose up
```

---

## Project Structure

```
cox-hazard-ratio-calculator/
├── cox_hr.py          # Core Cox PH algorithm (pure Python stdlib)
├── cli.py             # Command-line interface
├── test_cox_hr.py     # Core algorithm tests
├── agents/            # Enterprise agent framework
│   ├── base.py        # PHI guard, audit trail, security
│   ├── models.py      # Pydantic data models
│   ├── supervisor.py  # Multi-agent orchestration
│   ├── workers.py     # Specialized audit workers
│   ├── api.py         # FastAPI REST endpoints
│   └── ...
├── tests/             # Integration tests
├── web/               # Web operations console
├── simulator.py       # High-throughput stress testing
├── sample.csv         # Example input data
└── Dockerfile         # Container configuration
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 Dr. Abu Suraih
