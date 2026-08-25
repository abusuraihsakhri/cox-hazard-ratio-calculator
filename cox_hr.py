#!/usr/bin/env python3
"""
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
"""

import math
import csv
import json
import sys
from typing import List, Tuple, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Core Cox PH model
# ---------------------------------------------------------------------------

def cox_ph(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Fit a Cox Proportional Hazards model using Newton-Raphson.

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
    """
    n = len(times)
    p = len(covariates[0]) if covariates else 0

    if len(events) != n:
        raise ValueError("times and events must have the same length")
    if len(covariates) != n:
        raise ValueError("covariates must have the same length as times")
    if p == 0:
        raise ValueError("At least one covariate is required")

    # Sort by time (ascending)
    indices = sorted(range(n), key=lambda i: times[i])
    sorted_times = [times[i] for i in indices]
    sorted_events = [events[i] for i in indices]
    sorted_cov = [covariates[i] for i in indices]

    # Identify unique event times
    event_times = sorted(set(t for t, e in zip(sorted_times, sorted_events) if e == 1))

    if not event_times:
        raise ValueError("No events observed — cannot fit Cox model")

    # Newton-Raphson
    beta = [0.0] * p  # initial coefficients

    for iteration in range(max_iter):
        # Compute gradient and Hessian of partial log-likelihood
        grad = [0.0] * p
        hess = [[0.0] * p for _ in range(p)]
        ll = 0.0

        for ti in event_times:
            # Risk set: subjects with time >= ti
            risk_indices = [j for j in range(n) if sorted_times[j] >= ti]

            # Compute exp(Xj * beta) for risk set
            exp_xb = []
            for j in risk_indices:
                xb = sum(sorted_cov[j][k] * beta[k] for k in range(p))
                exp_xb.append(math.exp(xb))

            # Subjects with event at ti
            event_indices = [j for j in risk_indices if sorted_times[j] == ti and sorted_events[j] == 1]

            # For Breslow method with ties:
            # Weighted sums
            sum_exp_xb = sum(exp_xb)
            sum_exp_xb_x = [sum(exp_xb[i] * sorted_cov[risk_indices[i]][k] for i in range(len(risk_indices))) for k in range(p)]
            sum_exp_xb_xx = [
                [sum(exp_xb[i] * sorted_cov[risk_indices[i]][k] * sorted_cov[risk_indices[i]][l]
                     for i in range(len(risk_indices)))
                 for l in range(p)]
                for k in range(p)
            ]

            # Number of events at this time
            d = len(event_indices)

            # For each event subject, add to gradient
            for j in event_indices:
                # Log-likelihood contribution
                xb = sum(sorted_cov[j][k] * beta[k] for k in range(p))
                ll += xb - d * math.log(sum_exp_xb)

                # Gradient
                for k in range(p):
                    grad[k] += sorted_cov[j][k] - d * sum_exp_xb_x[k] / sum_exp_xb

                # Hessian
                for k in range(p):
                    for l in range(p):
                        mean_x = sum_exp_xb_x[k] / sum_exp_xb
                        mean_xl = sum_exp_xb_x[l] / sum_exp_xb
                        mean_xxl = sum_exp_xb_xx[k][l] / sum_exp_xb
                        hess[k][l] -= d * (mean_xxl - mean_x * mean_xl)

        # Newton-Raphson update: beta_new = beta - H^{-1} * g
        # Solve H * delta = -g
        neg_grad = [-g for g in grad]
        try:
            delta = _solve_linear_system(hess, neg_grad)
        except Exception:
            break

        # Update beta
        beta_new = [beta[k] + delta[k] for k in range(p)]

        # Check convergence
        if all(abs(delta[k]) < tol for k in range(p)):
            beta = beta_new
            break
        beta = beta_new

    # Compute standard errors from inverse Hessian
    # Recompute Hessian at final beta
    hess = [[0.0] * p for _ in range(p)]
    ll = 0.0

    for ti in event_times:
        risk_indices = [j for j in range(n) if sorted_times[j] >= ti]
        exp_xb = []
        for j in risk_indices:
            xb = sum(sorted_cov[j][k] * beta[k] for k in range(p))
            exp_xb.append(math.exp(xb))

        event_indices = [j for j in risk_indices if sorted_times[j] == ti and sorted_events[j] == 1]
        d = len(event_indices)

        sum_exp_xb = sum(exp_xb)
        sum_exp_xb_x = [sum(exp_xb[i] * sorted_cov[risk_indices[i]][k] for i in range(len(risk_indices))) for k in range(p)]
        sum_exp_xb_xx = [
            [sum(exp_xb[i] * sorted_cov[risk_indices[i]][k] * sorted_cov[risk_indices[i]][l]
                 for i in range(len(risk_indices)))
             for l in range(p)]
            for k in range(p)
        ]

        for j in event_indices:
            xb = sum(sorted_cov[j][k] * beta[k] for k in range(p))
            ll += xb - d * math.log(sum_exp_xb)

        for k in range(p):
            for l in range(p):
                mean_x = sum_exp_xb_x[k] / sum_exp_xb
                mean_xl = sum_exp_xb_x[l] / sum_exp_xb
                mean_xxl = sum_exp_xb_xx[k][l] / sum_exp_xb
                hess[k][l] -= d * (mean_xxl - mean_x * mean_xl)

    # Variance-covariance matrix = (-H)^{-1}
    neg_hess = [[-hess[i][j] for j in range(p)] for i in range(p)]
    try:
        var_cov = _invert_matrix(neg_hess)
    except Exception:
        var_cov = [[float('nan')] * p for _ in range(p)]

    se = [math.sqrt(max(0, var_cov[k][k])) for k in range(p)]

    # Wald test, HR, CI, p-values
    z_val = 1.96
    hr = []
    z_scores = []
    p_values = []
    ci_lo = []
    ci_hi = []

    for k in range(p):
        hr_k = math.exp(beta[k])
        z_k = beta[k] / se[k] if se[k] > 0 else 0.0
        p_k = _z_to_p(z_k)

        hr.append(hr_k)
        z_scores.append(z_k)
        p_values.append(p_k)
        ci_lo.append(math.exp(beta[k] - z_val * se[k]))
        ci_hi.append(math.exp(beta[k] + z_val * se[k]))

    # Schoenfeld residuals for PH assumption check
    schoenfeld = _schoenfeld_residuals(sorted_times, sorted_events, sorted_cov, beta, event_times)

    return {
        "coefficients": beta,
        "hazard_ratios": hr,
        "se": se,
        "z_scores": z_scores,
        "p_values": p_values,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "log_likelihood": ll,
        "iterations": iteration + 1,
        "n_subjects": n,
        "n_events": sum(events),
        "schoenfeld": schoenfeld,
    }


def hazard_ratio(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
) -> Dict[str, Any]:
    """
    Convenience function: fit Cox model and return hazard ratio summary.
    """
    result = cox_ph(times, events, covariates)
    return {
        "hazard_ratios": result["hazard_ratios"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "p_values": result["p_values"],
        "coefficients": result["coefficients"],
        "se": result["se"],
    }


def forest_plot_data(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
    labels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate data suitable for a forest plot.

    Returns list of dicts with: label, hr, ci_lower, ci_upper, p_value, significant.
    """
    result = cox_ph(times, events, covariates)
    p = len(result["coefficients"])

    if labels is None:
        labels = [f"Covariate_{i+1}" for i in range(p)]

    data = []
    for k in range(p):
        data.append({
            "label": labels[k],
            "hr": round(result["hazard_ratios"][k], 4),
            "ci_lower": round(result["ci_lower"][k], 4),
            "ci_upper": round(result["ci_upper"][k], 4),
            "p_value": round(result["p_values"][k], 6),
            "significant": result["p_values"][k] < 0.05,
        })
    return data


def check_proportional_hazards(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
) -> Dict[str, Any]:
    """
    Check the proportional hazards assumption using Schoenfeld residuals.

    Tests whether Schoenfeld residuals are correlated with time.
    A significant correlation suggests violation of PH assumption.

    Returns dict with:
        correlation : correlation between Schoenfeld residuals and time
        p_value     : p-value for the correlation test
        assumption_holds : True if p > 0.05
    """
    result = cox_ph(times, events, covariates)
    schoenfeld = result["schoenfeld"]

    if not schoenfeld:
        return {"correlation": [], "p_value": [], "assumption_holds": []}

    p = len(result["coefficients"])
    checks = []

    for k in range(p):
        if isinstance(schoenfeld[0], dict):
            residuals = [s["residual"][k] for s in schoenfeld]
            times_at_events = [s["time"] for s in schoenfeld]
        else:
            residuals = [s[k] for s in schoenfeld]
            times_at_events = list(range(len(schoenfeld)))

        # Compute correlation
        if len(residuals) < 3:
            checks.append({
                "covariate": k,
                "correlation": 0.0,
                "p_value": 1.0,
                "assumption_holds": True,
            })
            continue

        r = _pearson_correlation(times_at_events, residuals)
        n_r = len(residuals)
        # t-test for correlation
        if abs(r) >= 1.0:
            t_stat = float('inf')
            p_val = 0.0
        else:
            t_stat = r * math.sqrt((n_r - 2) / (1 - r * r))
            p_val = 2.0 * _t_sf(abs(t_stat), n_r - 2)

        checks.append({
            "covariate": k,
            "correlation": round(r, 4),
            "p_value": round(p_val, 6),
            "assumption_holds": p_val > 0.05,
        })

    return {"checks": checks}


# ---------------------------------------------------------------------------
# CSV batch processing
# ---------------------------------------------------------------------------

def process_csv(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Process a CSV file for Cox PH analysis.

    Expected columns: time, event, and one or more covariate columns.
    Column names auto-detected.
    """
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    time_col = _find_col(fieldnames, ["time", "survival_time", "days", "months", "t"])
    event_col = _find_col(fieldnames, ["event", "status", "dead", "censored", "e"])

    # All other numeric columns are covariates
    covariate_cols = [c for c in fieldnames if c not in (time_col, event_col)]

    times = [float(r[time_col]) for r in rows]
    events = [int(float(r[event_col])) for r in rows]
    covariates = []
    for r in rows:
        cov = []
        for c in covariate_cols:
            try:
                cov.append(float(r[c]))
            except (ValueError, KeyError):
                cov.append(0.0)
        covariates.append(cov)

    result = cox_ph(times, events, covariates)
    fp = forest_plot_data(times, events, covariates, labels=covariate_cols)

    # Write forest plot data
    out_fields = ["covariate", "hazard_ratio", "ci_lower", "ci_upper", "p_value", "significant"]
    out_rows = []
    for item in fp:
        out_rows.append({
            "covariate": item["label"],
            "hazard_ratio": str(item["hr"]),
            "ci_lower": str(item["ci_lower"]),
            "ci_upper": str(item["ci_upper"]),
            "p_value": str(item["p_value"]),
            "significant": str(item["significant"]),
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(fieldnames: List[str], candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return fieldnames[0]


def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Partial pivoting
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]

        if abs(M[col][col]) < 1e-15:
            raise ValueError("Singular matrix")

        # Eliminate
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x


def _invert_matrix(A: List[List[float]]) -> List[List[float]]:
    """Invert a matrix using Gauss-Jordan elimination."""
    n = len(A)
    # Augmented matrix [A | I]
    M = [A[i][:] + [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        # Partial pivoting
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]

        pivot = M[col][col]
        if abs(pivot) < 1e-15:
            raise ValueError("Singular matrix")

        # Scale pivot row
        for j in range(2 * n):
            M[col][j] /= pivot

        # Eliminate column
        for row in range(n):
            if row == col:
                continue
            factor = M[row][col]
            for j in range(2 * n):
                M[row][j] -= factor * M[col][j]

    # Extract inverse
    return [M[i][n:] for i in range(n)]


def _schoenfeld_residuals(
    times: List[float],
    events: List[int],
    covariates: List[List[float]],
    beta: List[float],
    event_times: List[float],
) -> List[Dict[str, Any]]:
    """
    Compute Schoenfeld residuals at each event time.
    r_k(ti) = x_k(event_subject) - E[x_k | ti]
    where E[x_k | ti] = Σ x_k(j) * exp(xj*β) / Σ exp(xj*β) over risk set.
    """
    n = len(times)
    p = len(beta)
    residuals = []

    for ti in event_times:
        risk_indices = [j for j in range(n) if times[j] >= ti]
        event_indices = [j for j in risk_indices if times[j] == ti and events[j] == 1]

        # Compute weighted mean of covariates in risk set
        exp_xb = []
        for j in risk_indices:
            xb = sum(covariates[j][k] * beta[k] for k in range(p))
            exp_xb.append(math.exp(xb))

        sum_exp_xb = sum(exp_xb)
        expected = []
        for k in range(p):
            ex = sum(exp_xb[i] * covariates[risk_indices[i]][k] for i in range(len(risk_indices))) / sum_exp_xb
            expected.append(ex)

        # Schoenfeld residual for each event subject
        for j in event_indices:
            res = [covariates[j][k] - expected[k] for k in range(p)]
            residuals.append({"time": ti, "residual": res})

    return residuals


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


def _z_to_p(z: float) -> float:
    """Two-tailed p-value from z-score."""
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_sf(t: float, df: int) -> float:
    """Survival function of t-distribution (approximation for large df)."""
    if df > 30:
        return 2.0 * (1.0 - _norm_cdf(abs(t)))
    # Use approximation: for small df, use the incomplete beta function
    x = df / (df + t * t)
    return _inc_beta(df / 2.0, 0.5, x)


def _inc_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction."""
    if x < 0 or x > 1:
        return 0.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Continued fraction (Lentz's method)
    max_iter = 200
    eps = 1e-12

    lbeta = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)

    # Use continued fraction for I_x(a, b)
    if x < (a + 1) / (a + b + 2):
        cf = _beta_cf(a, b, x, max_iter, eps)
        return front * cf / a
    else:
        cf = _beta_cf(b, a, 1.0 - x, max_iter, eps)
        return 1.0 - front * cf / b


def _beta_cf(a: float, b: float, x: float, max_iter: int, eps: float) -> float:
    """Continued fraction for incomplete beta."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return h


def _log_gamma(x: float) -> float:
    """Log of gamma function using Lanczos approximation."""
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - _log_gamma(1.0 - x)
    x -= 1.0
    g = 7
    c = [
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
    total = c[0]
    for i in range(1, g + 2):
        total += c[i] / (x + i)
    st = x + g + 0.5
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(st) - st + math.log(total)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summary(times: List[float], events: List[int], covariates: List[List[float]], labels: Optional[List[str]] = None) -> str:
    """Return a formatted summary of the Cox PH analysis."""
    result = cox_ph(times, events, covariates)
    p = len(result["coefficients"])

    if labels is None:
        labels = [f"Covariate_{i+1}" for i in range(p)]

    lines = []
    lines.append("Cox Proportional Hazards Model")
    lines.append("=" * 70)
    lines.append(f"  Subjects: {result['n_subjects']}   Events: {result['n_events']}   Iterations: {result['iterations']}")
    lines.append(f"  Log-likelihood: {result['log_likelihood']:.4f}")
    lines.append("")
    lines.append(f"  {'Covariate':<20} {'Coef':>8} {'SE':>8} {'HR':>8} {'95% CI':>18} {'z':>8} {'p':>10}")
    lines.append("  " + "-" * 82)
    for k in range(p):
        ci = f"[{result['ci_lower'][k]:.4f}, {result['ci_upper'][k]:.4f}]"
        sig = "*" if result["p_values"][k] < 0.05 else ""
        lines.append(
            f"  {labels[k]:<20} {result['coefficients'][k]:>8.4f} {result['se'][k]:>8.4f} "
            f"{result['hazard_ratios'][k]:>8.4f} {ci:>18s} {result['z_scores'][k]:>8.4f} "
            f"{result['p_values'][k]:>10.6f}{sig}"
        )
    lines.append("")
    lines.append("  * p < 0.05")
    return "\n".join(lines)
