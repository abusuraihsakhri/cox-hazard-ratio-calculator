# Cox Hazard Ratio Calculator

### [Open the Live Application →](https://abusuraihsakhri.github.io/cox-hazard-ratio-calculator/)

A compact Cox proportional hazards regression utility with a pure-Python core, command-line interface, FastAPI service, and browser application. The browser version runs the repository's Python implementation locally through Pyodide.

## Features

- Cox partial-likelihood estimation with Newton-Raphson optimization.
- Breslow approximation for tied event times.
- Multivariable coefficients, hazard ratios, standard errors, Wald z statistics, two-sided p-values, and 95% confidence intervals.
- Schoenfeld residual generation and a simple residual-versus-event-time correlation check.
- CSV batch input and forest-plot-ready summary output.
- Browser execution through Pyodide with no application backend required.
- JSON API and Docker deployment for server-side use.
- Input validation, convergence reporting, numerically stabilized risk-set exponentials, and regression tests for tied events.

The core estimator in `cox_hr.py` uses only the Python standard library.

> **Statistical scope:** the proportional-hazards check is a simple correlation test on Schoenfeld residuals. It is not a full Grambsch-Therneau diagnostic suite. This project is a statistical utility, not a validated clinical decision-support system.

## Browser application

Open the live application above, enter observed times, event indicators, and one covariate row per subject, then select **Analyze**. Multivariable input is supported by placing multiple values on each covariate row.

The page loads Pyodide from jsDelivr and runs `cox_hr.py` in WebAssembly. The application code does not upload entered analysis data to a server. Normal browser requests are still made to GitHub Pages and the Pyodide CDN to load the application resources.

A modern browser with JavaScript and WebAssembly enabled is required. The initial load is larger than a conventional static calculator because the Python runtime must be downloaded.

## Command line

No package installation is required for the core CLI:

```bash
python cli.py fit \
  --time 1 2 3 4 5 6 7 8 \
  --event 1 0 1 1 0 1 0 1 \
  --covariate 0.5 1.2 0.8 1.5 0.3 1.1 0.9 1.4
```

Multivariable model:

```bash
python cli.py multivariate \
  --time 1 2 3 4 5 \
  --event 1 0 1 1 0 \
  --covariates '[[0.5,65],[1.2,70],[0.8,55],[1.5,60],[0.3,72]]' \
  --labels treatment age
```

CSV batch processing:

```bash
python cli.py batch --input sample.csv --output results.csv
```

Expected input columns are a recognized time column, a recognized event column, and at least one numeric covariate column. Missing or non-numeric covariate values are rejected rather than silently imputed.

## Python API

```python
from cox_hr import cox_ph

times = [1, 2, 3, 4, 5, 6]
events = [1, 0, 1, 1, 0, 1]
covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1]]

result = cox_ph(times, events, covariates)

print(result["hazard_ratios"])
print(result["ci_lower"], result["ci_upper"])
print(result["p_values"])
print(result["converged"])
```

Important result fields include `coefficients`, `hazard_ratios`, `se`, `z_scores`, `p_values`, `ci_lower`, `ci_upper`, `log_likelihood`, `iterations`, `converged`, `n_subjects`, `n_events`, and `schoenfeld`.

## FastAPI server

Install runtime dependencies and start the service:

```bash
python -m pip install -r requirements.txt
uvicorn agents.api:app --host 127.0.0.1 --port 8000
```

The web application is served at `/`, interactive API documentation at `/docs`, and the Cox endpoints at `/api/cox` and `/api/ph-check`.

When using the API, submitted data are processed by the server rather than exclusively in the browser.

## Docker

```bash
docker build -t cox-hazard-ratio-calculator .
docker run --rm -p 8000:8000 \
  -e AUDIT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  cox-hazard-ratio-calculator
```

Then open `http://localhost:8000/`.

The image runs as a non-root user and includes a healthcheck against `/health`.

## Testing and security checks

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q .
pytest -v
python -m pip_audit -r requirements.txt
```

GitHub Actions runs these checks on Python 3.10, 3.11, and 3.12, builds and smoke-tests the Docker image, launches the assembled browser app in headless Chrome, initializes Pyodide, executes the sample Cox analysis, deploys Pages, and performs an HTTP smoke test of the published page and Python module.

The optional `agents/` supervisor demo includes an HMAC-SHA256 in-memory audit chain and a regex-based sensitive-identifier guard. The guard is heuristic and must not be treated as complete de-identification or HIPAA compliance. Only the deterministic local `mock` model adapter is implemented.

## Project layout

```text
cox_hr.py              Core Cox PH implementation
cli.py                 Command-line interface
agents/api.py          FastAPI application
agents/                Optional supervisor/audit demo
web/index.html         Pyodide browser UI
tests/                 Integration/security tests
test_cox_hr.py         Core statistical tests
sample.csv             Example data
requirements.txt       API runtime dependencies
requirements-dev.txt   Test and dependency-audit tools
Dockerfile             Container deployment
.github/workflows/     CI and GitHub Pages deployment
```

## License

MIT License. See [LICENSE](LICENSE).
