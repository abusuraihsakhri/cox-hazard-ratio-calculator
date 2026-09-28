#!/usr/bin/env python3
"""
Cox proportional hazards regression implemented with the Python standard library.

The model uses Newton-Raphson optimization of the Cox partial likelihood,
Breslow handling of tied event times, Wald inference, Schoenfeld residuals,
and CSV batch processing.
"""

import csv
import math
import os
from typing import Any, Dict, List, Optional


def cox_ph(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """Fit a Cox proportional hazards model using Breslow ties."""
    _validate_inputs(times, events, covariates, max_iter=max_iter, tol=tol)

    n = len(times)
    p = len(covariates[0])
    indices = sorted(range(n), key=lambda i: times[i])
    sorted_times = [float(times[i]) for i in indices]
    sorted_events = [int(events[i]) for i in indices]
    sorted_cov = [[float(v) for v in covariates[i]] for i in indices]
    event_times = sorted({t for t, e in zip(sorted_times, sorted_events) if e == 1})

    beta = [0.0] * p
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        _, grad, hess = _partial_likelihood_components(
            sorted_times, sorted_events, sorted_cov, beta, event_times
        )
        try:
            delta = _solve_linear_system(hess, [-g for g in grad])
        except ValueError:
            # A singular information matrix can occur with collinearity or
            # complete/quasi-complete separation. Preserve the best iterate and
            # expose convergence state instead of fabricating standard errors.
            break

        beta = [beta[k] + delta[k] for k in range(p)]
        if max(abs(d) for d in delta) < tol:
            converged = True
            break

    ll, _, hess = _partial_likelihood_components(
        sorted_times, sorted_events, sorted_cov, beta, event_times
    )

    neg_hess = [[-hess[i][j] for j in range(p)] for i in range(p)]
    try:
        var_cov = _invert_matrix(neg_hess)
        se = [
            math.sqrt(v) if math.isfinite(v) and v >= 0.0 else float("nan")
            for v in (var_cov[k][k] for k in range(p))
        ]
    except ValueError:
        se = [float("nan")] * p

    z_val = 1.959963984540054
    hazard_ratios: List[float] = []
    z_scores: List[float] = []
    p_values: List[float] = []
    ci_lower: List[float] = []
    ci_upper: List[float] = []

    for k in range(p):
        hr = _safe_exp(beta[k])
        se_k = se[k]
        if math.isfinite(se_k) and se_k > 0:
            z = beta[k] / se_k
            p_value = _z_to_p(z)
            lo = _safe_exp(beta[k] - z_val * se_k)
            hi = _safe_exp(beta[k] + z_val * se_k)
        else:
            z = float("nan")
            p_value = float("nan")
            lo = float("nan")
            hi = float("nan")

        hazard_ratios.append(hr)
        z_scores.append(z)
        p_values.append(p_value)
        ci_lower.append(lo)
        ci_upper.append(hi)

    schoenfeld = _schoenfeld_residuals(
        sorted_times, sorted_events, sorted_cov, beta, event_times
    )

    return {
        "coefficients": beta,
        "hazard_ratios": hazard_ratios,
        "se": se,
        "z_scores": z_scores,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "log_likelihood": ll,
        "iterations": iteration,
        "converged": converged,
        "n_subjects": n,
        "n_events": sum(sorted_events),
        "schoenfeld": schoenfeld,
    }


def hazard_ratio(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
) -> Dict[str, Any]:
    """Fit a Cox model and return the hazard-ratio summary."""
    result = cox_ph(times, events, covariates)
    return {
        "hazard_ratios": result["hazard_ratios"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "p_values": result["p_values"],
        "coefficients": result["coefficients"],
        "se": result["se"],
        "converged": result["converged"],
    }


def forest_plot_data(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
    labels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return model estimates formatted for a forest plot."""
    result = cox_ph(times, events, covariates)
    p = len(result["coefficients"])

    if labels is None:
        labels = [f"Covariate_{i + 1}" for i in range(p)]
    elif len(labels) != p:
        raise ValueError(f"labels must contain exactly {p} values")

    rows: List[Dict[str, Any]] = []
    for k in range(p):
        p_value = result["p_values"][k]
        rows.append(
            {
                "label": labels[k],
                "hr": _round_or_nan(result["hazard_ratios"][k], 4),
                "ci_lower": _round_or_nan(result["ci_lower"][k], 4),
                "ci_upper": _round_or_nan(result["ci_upper"][k], 4),
                "p_value": _round_or_nan(p_value, 6),
                "significant": bool(math.isfinite(p_value) and p_value < 0.05),
            }
        )
    return rows


def check_proportional_hazards(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
) -> Dict[str, Any]:
    """Check correlation of Schoenfeld residuals with event time."""
    result = cox_ph(times, events, covariates)
    schoenfeld = result["schoenfeld"]
    if not schoenfeld:
        return {"checks": []}

    checks: List[Dict[str, Any]] = []
    p = len(result["coefficients"])
    for k in range(p):
        residuals = [s["residual"][k] for s in schoenfeld]
        times_at_events = [s["time"] for s in schoenfeld]

        if len(residuals) < 3:
            checks.append(
                {
                    "covariate": k,
                    "correlation": 0.0,
                    "p_value": 1.0,
                    "assumption_holds": True,
                }
            )
            continue

        r = _pearson_correlation(times_at_events, residuals)
        df = len(residuals) - 2
        if abs(r) >= 1.0:
            p_value = 0.0
        else:
            t_stat = abs(r) * math.sqrt(df / max(1e-15, 1.0 - r * r))
            p_value = min(1.0, 2.0 * _t_sf(t_stat, df))

        checks.append(
            {
                "covariate": k,
                "correlation": round(r, 4),
                "p_value": round(p_value, 6),
                "assumption_holds": p_value > 0.05,
            }
        )

    return {"checks": checks}


def process_csv(input_path: str, output_path: str) -> Dict[str, Any]:
    """Fit a Cox model from a CSV and write a forest-plot summary CSV."""
    input_path = _validate_safe_path(input_path)
    output_path = _validate_safe_path(output_path)

    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError("Input CSV is missing a header row")
        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV contains no data rows")

    time_col = _find_col(fieldnames, ["time", "survival_time", "days", "months", "t"])
    event_col = _find_col(fieldnames, ["event", "status", "dead", "censored", "e"])
    if time_col == event_col:
        raise ValueError("Time and event columns must be distinct")

    covariate_cols = [c for c in fieldnames if c not in (time_col, event_col)]
    if not covariate_cols:
        raise ValueError("At least one covariate column is required")

    times: List[float] = []
    events: List[int] = []
    covariates: List[List[float]] = []

    for row_number, row in enumerate(rows, start=2):
        try:
            times.append(float(row[time_col]))
            events.append(int(float(row[event_col])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid time/event value on CSV row {row_number}"
            ) from exc

        cov_row: List[float] = []
        for column in covariate_cols:
            raw = row.get(column)
            if raw is None or str(raw).strip() == "":
                raise ValueError(
                    f"Missing covariate value in column '{column}' on CSV row {row_number}"
                )
            try:
                cov_row.append(float(raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid covariate value in column '{column}' on CSV row {row_number}"
                ) from exc
        covariates.append(cov_row)

    result = cox_ph(times, events, covariates)
    forest = forest_plot_data(times, events, covariates, labels=covariate_cols)

    out_fields = [
        "covariate",
        "hazard_ratio",
        "ci_lower",
        "ci_upper",
        "p_value",
        "significant",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for item in forest:
            writer.writerow(
                {
                    "covariate": item["label"],
                    "hazard_ratio": item["hr"],
                    "ci_lower": item["ci_lower"],
                    "ci_upper": item["ci_upper"],
                    "p_value": item["p_value"],
                    "significant": item["significant"],
                }
            )

    return result


def _validate_inputs(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
    *,
    max_iter: int,
    tol: float,
) -> None:
    n = len(times)
    if n == 0:
        raise ValueError("At least one subject is required")
    if len(events) != n:
        raise ValueError("times and events must have the same length")
    if len(covariates) != n:
        raise ValueError("covariates must have the same length as times")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite number")

    p = len(covariates[0]) if covariates else 0
    if p == 0:
        raise ValueError("At least one covariate is required")

    for i, t in enumerate(times):
        if not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError(f"Time at index {i} must be a finite number, got {t}")
        if t < 0:
            raise ValueError(f"Time at index {i} must be non-negative, got {t}")

    for i, event in enumerate(events):
        if event not in (0, 1):
            raise ValueError(f"Event at index {i} must be 0 or 1, got {event}")

    for i, cov in enumerate(covariates):
        if len(cov) != p:
            raise ValueError(
                f"Covariate vector at index {i} has {len(cov)} elements, expected {p}"
            )
        for j, value in enumerate(cov):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(
                    f"Covariate value at subject {i}, covariate {j} must be finite, got {value}"
                )

    if not any(events):
        raise ValueError("No events observed — cannot fit Cox model")


def _partial_likelihood_components(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
    beta: List[float],
    event_times: List[float],
):
    """Return Breslow partial log-likelihood, score, and Hessian."""
    n = len(times)
    p = len(beta)
    grad = [0.0] * p
    hess = [[0.0] * p for _ in range(p)]
    ll = 0.0

    for event_time in event_times:
        risk_indices = [i for i in range(n) if times[i] >= event_time]
        event_indices = [
            i for i in risk_indices if times[i] == event_time and events[i] == 1
        ]
        d = len(event_indices)
        if d == 0:
            continue

        xb = [
            sum(covariates[i][k] * beta[k] for k in range(p))
            for i in risk_indices
        ]
        shift = max(xb)
        weights = [math.exp(value - shift) for value in xb]
        denominator = sum(weights)
        log_risk_sum = shift + math.log(denominator)

        mean = [
            sum(weights[j] * covariates[risk_indices[j]][k] for j in range(len(risk_indices)))
            / denominator
            for k in range(p)
        ]
        mean_xx = [
            [
                sum(
                    weights[j]
                    * covariates[risk_indices[j]][k]
                    * covariates[risk_indices[j]][l]
                    for j in range(len(risk_indices))
                )
                / denominator
                for l in range(p)
            ]
            for k in range(p)
        ]

        ll += sum(
            sum(covariates[i][k] * beta[k] for k in range(p))
            for i in event_indices
        ) - d * log_risk_sum

        for k in range(p):
            grad[k] += sum(covariates[i][k] for i in event_indices) - d * mean[k]
            for l in range(p):
                hess[k][l] -= d * (mean_xx[k][l] - mean[k] * mean[l])

    return ll, grad, hess


def _find_col(fieldnames: List[str], candidates: List[str]) -> str:
    lower_map = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    raise ValueError(
        "Required CSV column not found; expected one of: " + ", ".join(candidates)
    )


def _validate_safe_path(path: str) -> str:
    """Reject NULs and relative parent traversal while allowing explicit absolute paths."""
    if not isinstance(path, str) or not path:
        raise ValueError("Path must be a non-empty string")
    if "\x00" in path:
        raise ValueError("Path contains null bytes")

    normalized = os.path.normpath(path)
    absolute = os.path.abspath(path)
    if not os.path.isabs(path) and ".." in normalized.split(os.sep):
        cwd = os.path.abspath(os.getcwd())
        try:
            inside_cwd = os.path.commonpath([cwd, absolute]) == cwd
        except ValueError:
            inside_cwd = False
        if not inside_cwd:
            raise ValueError(f"Path traversal detected: {path}")
    return absolute


def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    if len(A) != n or any(len(row) != n for row in A):
        raise ValueError("A must be a square matrix matching b")

    matrix = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(matrix[row][col]))
        matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
        if abs(matrix[col][col]) < 1e-15:
            raise ValueError("Singular matrix")

        for row in range(col + 1, n):
            factor = matrix[row][col] / matrix[col][col]
            for j in range(col, n + 1):
                matrix[row][j] -= factor * matrix[col][j]

    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        rhs = matrix[i][n] - sum(
            matrix[i][j] * solution[j] for j in range(i + 1, n)
        )
        solution[i] = rhs / matrix[i][i]
    return solution


def _invert_matrix(A: List[List[float]]) -> List[List[float]]:
    """Invert a square matrix using Gauss-Jordan elimination."""
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("A must be a non-empty square matrix")

    matrix = [
        A[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(matrix[row][col]))
        matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
        pivot = matrix[col][col]
        if abs(pivot) < 1e-15:
            raise ValueError("Singular matrix")

        for j in range(2 * n):
            matrix[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = matrix[row][col]
            for j in range(2 * n):
                matrix[row][j] -= factor * matrix[col][j]

    return [matrix[i][n:] for i in range(n)]


def _schoenfeld_residuals(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
    beta: List[float],
    event_times: List[float],
) -> List[Dict[str, Any]]:
    """Compute Schoenfeld residuals for each observed event."""
    n = len(times)
    p = len(beta)
    residuals: List[Dict[str, Any]] = []

    for event_time in event_times:
        risk_indices = [i for i in range(n) if times[i] >= event_time]
        event_indices = [
            i for i in risk_indices if times[i] == event_time and events[i] == 1
        ]
        xb = [
            sum(covariates[i][k] * beta[k] for k in range(p))
            for i in risk_indices
        ]
        shift = max(xb)
        weights = [math.exp(value - shift) for value in xb]
        denominator = sum(weights)
        expected = [
            sum(weights[j] * covariates[risk_indices[j]][k] for j in range(len(risk_indices)))
            / denominator
            for k in range(p)
        ]

        for i in event_indices:
            residuals.append(
                {
                    "time": event_time,
                    "residual": [
                        covariates[i][k] - expected[k] for k in range(p)
                    ],
                }
            )

    return residuals


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute the Pearson correlation coefficient."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((value - mean_x) ** 2 for value in x)
    var_y = sum((value - mean_y) ** 2 for value in y)
    denominator = math.sqrt(var_x * var_y)
    return 0.0 if denominator == 0.0 else covariance / denominator


def _z_to_p(z: float) -> float:
    """Two-tailed p-value from a standard-normal z statistic."""
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_sf(t: float, df: int) -> float:
    """One-tailed survival function P(T > |t|) for Student's t."""
    if df <= 0:
        raise ValueError("df must be positive")
    t = abs(float(t))
    if df > 30:
        return 1.0 - _norm_cdf(t)
    x = df / (df + t * t)
    # I_x(df/2, 1/2) is the two-tailed probability for |T| >= t.
    return 0.5 * _inc_beta(df / 2.0, 0.5, x)


def _inc_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x < 0.0 or x > 1.0:
        raise ValueError("x must lie in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    max_iter = 200
    eps = 1e-12
    log_beta = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta)

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(a, b, x, max_iter, eps) / a
    return 1.0 - front * _beta_cf(b, a, 1.0 - x, max_iter, eps) / b


def _beta_cf(a: float, b: float, x: float, max_iter: int, eps: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _log_gamma(x: float) -> float:
    """Log gamma using the Lanczos approximation."""
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - _log_gamma(1.0 - x)

    coefficients = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    x -= 1.0
    total = coefficients[0]
    for i, coefficient in enumerate(coefficients[1:], start=1):
        total += coefficient / (x + i)
    g = 7.0
    t = x + g + 0.5
    return (
        0.5 * math.log(2.0 * math.pi)
        + (x + 0.5) * math.log(t)
        - t
        + math.log(total)
    )


def _safe_exp(value: float) -> float:
    if value > 709.0:
        return float("inf")
    if value < -745.0:
        return 0.0
    return math.exp(value)


def _round_or_nan(value: float, digits: int):
    return round(value, digits) if math.isfinite(value) else value


def summary(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
    labels: Optional[List[str]] = None,
) -> str:
    """Return a formatted text summary of the fitted model."""
    result = cox_ph(times, events, covariates)
    p = len(result["coefficients"])
    if labels is None:
        labels = [f"Covariate_{i + 1}" for i in range(p)]
    elif len(labels) != p:
        raise ValueError(f"labels must contain exactly {p} values")

    lines = [
        "Cox Proportional Hazards Model",
        "=" * 70,
        (
            f"  Subjects: {result['n_subjects']}   Events: {result['n_events']}   "
            f"Iterations: {result['iterations']}   Converged: {result['converged']}"
        ),
        f"  Log-likelihood: {result['log_likelihood']:.4f}",
        "",
        f"  {'Covariate':<20} {'Coef':>8} {'SE':>8} {'HR':>8} {'95% CI':>18} {'z':>8} {'p':>10}",
        "  " + "-" * 82,
    ]

    for k in range(p):
        lo = result["ci_lower"][k]
        hi = result["ci_upper"][k]
        ci = (
            f"[{lo:.4f}, {hi:.4f}]"
            if math.isfinite(lo) and math.isfinite(hi)
            else "[nan, nan]"
        )
        p_value = result["p_values"][k]
        significant = "*" if math.isfinite(p_value) and p_value < 0.05 else ""
        lines.append(
            f"  {labels[k]:<20} {result['coefficients'][k]:>8.4f} "
            f"{result['se'][k]:>8.4f} {result['hazard_ratios'][k]:>8.4f} "
            f"{ci:>18s} {result['z_scores'][k]:>8.4f} "
            f"{result['p_values'][k]:>10.6f}{significant}"
        )

    lines.extend(["", "  * p < 0.05"])
    return "\n".join(lines)
