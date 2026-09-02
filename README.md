# Cox Hazard Ratio Calculator

> **Domain:** Clinical Decision Support & Biomedical Computing  
> **Reference Guidelines & Standards:** `Standard Clinical Formulations & ISO/IEC Quality Frameworks`

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![Audit Trail](https://img.shields.io/badge/Audit-HMAC--SHA256_Tamper--Evident-brightgreen.svg)
![Zero-PHI Guard](https://img.shields.io/badge/Guard-Zero--PHI_Outbound-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)

</div>

---

## 📖 What It Does

Cox Proportional Hazards Model

Real implementation of:
  - Hazard ratio: HR = exp(β)
  - Partial likelihood estimation via Newton-Raphson (simplified Breslow method)
  - Wald test: z = β / SE(β)
  - Confidence interval for HR: exp(β ± z_α/2 × SE)
  - P-value from chi-square distribution
  - Proportional hazards assumption check (Schoenfeld residuals)
  - Forest plot data generation

Pure Python stdlib — no external dependencies.

---

## ⚙️ Key Capabilities & Algorithmic Modules

### 🔬 Analytical Functions

- **`cox_ph()`**: Fit a Cox Proportional Hazards model using Newton-Raphson.

Parameters
----------
times : list of float
    Observed times.
events : list of int
    Event indicators (1=event, 0=censored).
covariates : list of list of float
    Each element is a list of covariate values for one subject.
    For univariate: [[x1], [x2], ...]
    For multivariate: [[x1a, x1b], [x2a, x2b], ...]
max_iter : int
    Maximum Newton-Raphson iterations.
tol : float
    Convergence tolerance.

Returns
-------
dict with keys:
    coefficients   – estimated β coefficients
    hazard_ratios  – exp(β) for each covariate
    se             – standard errors
    z_scores       – Wald z-scores
    p_values       – p-values for each coefficient
    ci_lower       – lower 95% CI for HR
    ci_upper       – upper 95% CI for HR
    log_likelihood – partial log-likelihood at convergence
    iterations     – number of iterations
    n_subjects     – number of subjects
    n_events       – number of events
    schoenfeld     – Schoenfeld residuals for PH check
- **`hazard_ratio()`**: Convenience function: fit Cox model and return hazard ratio summary.
- **`forest_plot_data()`**: Generate data suitable for a forest plot.

Returns list of dicts with: label, hr, ci_lower, ci_upper, p_value, significant.
- **`check_proportional_hazards()`**: Check the proportional hazards assumption using Schoenfeld residuals.

Tests whether Schoenfeld residuals are correlated with time.
A significant correlation suggests violation of PH assumption.

Returns dict with:
    correlation : correlation between Schoenfeld residuals and time
    p_value     : p-value for the correlation test
    assumption_holds : True if p > 0.05
- **`process_csv()`**: Process a CSV file for Cox PH analysis.

Expected columns: time, event, and one or more covariate columns.
Column names auto-detected.

---

## 💻 CLI Quickstart & Usage

### 1. Guided Interactive Mode
```bash
python cli.py
```

### 2. Direct Parameterized Evaluation
```bash
python cli.py --time <value> --event <value> --covariate <value> --input <value>
```

### Parameter Reference
- `--time`: Specifies input measurement or parameter value.
- `--event`: Specifies input measurement or parameter value.
- `--covariate`: Specifies input measurement or parameter value.
- `--input`: Specifies input measurement or parameter value.
- `--output`: Specifies input measurement or parameter value.
- `---`: Specifies input measurement or parameter value.
- `--json`: Specifies input measurement or parameter value.
- `--covariates`: Specifies input measurement or parameter value.
- `--labels`: Specifies input measurement or parameter value.

### Input Data Schema

| Field | Description | Requirement |
|:------|:------------|:------------|
| `time` | Parameter / observation metric | Required |
| `event` | Parameter / observation metric | Required |
| `treatment` | Parameter / observation metric | Required |
| `age` | Parameter / observation metric | Required |

---

## 🛡️ Security & Enterprise Architecture

* **Zero-PHI Outbound Interceptor:** Active AST and regex inspection blocking SSNs, MRNs, phone numbers, and patient identifiers.
* **Tamper-Evident HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs for every evaluation and state transition.
* **Air-Gapped LLM Reasoning Adapter:** Agnostic integration for local Ollama instances (`llama3`, `mistral`), Claude 3.5 Sonnet, GPT-4o, and deterministic test mocks.
* **Active Learning Bayesian Calibration:** Dynamic tracker updating worker reliability weights and monitoring Brier calibration drift.
* **FastAPI & Prometheus Telemetry:** Exposes OpenAPI 3.1 REST endpoints and operational Prometheus metrics (`/metrics`).

---

## 🧪 Testing & Verification

Run the automated test suite:

```bash
pytest -v
```

Execute high-throughput batch simulation benchmarks:

```bash
python simulator.py --tasks 1000 --concurrency 8
```

---

## 🐳 Container Deployment

```bash
docker build -t cox-hazard-ratio-calculator .
docker run -p 8000:8000 cox-hazard-ratio-calculator
```
