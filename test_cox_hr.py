#!/usr/bin/env python3
"""
Tests for Cox Proportional Hazards Model.
"""
import math
import os
import sys
import tempfile

import pytest

from cox_hr import (
    cox_ph,
    hazard_ratio,
    forest_plot_data,
    check_proportional_hazards,
    summary,
    process_csv,
    _solve_linear_system,
    _invert_matrix,
    _pearson_correlation,
    _norm_cdf,
    _z_to_p,
)


# ---------------------------------------------------------------------------
# Basic model fitting
# ---------------------------------------------------------------------------

class TestCoxPHBasic:
    def test_simple_fit(self):
        """Basic univariate Cox model."""
        times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        events = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1], [0.9], [1.4], [0.6], [1.0]]
        result = cox_ph(times, events, covariates)
        assert "coefficients" in result
        assert "hazard_ratios" in result
        assert len(result["coefficients"]) == 1
        assert result["n_subjects"] == 10
        assert result["n_events"] == 6

    def test_hazard_ratio_exp_beta(self):
        """HR should equal exp(beta)."""
        times = [1, 2, 3, 4, 5]
        events = [1, 1, 1, 1, 1]
        covariates = [[0.5], [1.0], [1.5], [2.0], [2.5]]
        result = cox_ph(times, events, covariates)
        for k in range(len(result["coefficients"])):
            expected_hr = math.exp(result["coefficients"][k])
            assert abs(result["hazard_ratios"][k] - expected_hr) < 1e-10

    def test_hr_positive(self):
        """Hazard ratios should be positive."""
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        events = [1, 0, 1, 1, 0, 1, 0, 1]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1], [0.9], [1.4]]
        result = cox_ph(times, events, covariates)
        for hr in result["hazard_ratios"]:
            assert hr > 0

    def test_ci_contains_hr(self):
        """CI should contain the HR estimate."""
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        events = [1, 0, 1, 1, 0, 1, 0, 1]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1], [0.9], [1.4]]
        result = cox_ph(times, events, covariates)
        for k in range(len(result["hazard_ratios"])):
            assert result["ci_lower"][k] <= result["hazard_ratios"][k]
            assert result["ci_upper"][k] >= result["hazard_ratios"][k]

    def test_p_value_range(self):
        """p-values should be in [0, 1]."""
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        events = [1, 0, 1, 1, 0, 1, 0, 1]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1], [0.9], [1.4]]
        result = cox_ph(times, events, covariates)
        for p in result["p_values"]:
            assert 0.0 <= p <= 1.0

    def test_no_events_raises(self):
        """No events should raise ValueError."""
        with pytest.raises(ValueError):
            cox_ph([1, 2, 3], [0, 0, 0], [[1], [2], [3]])

    def test_mismatched_lengths_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError):
            cox_ph([1, 2], [1], [[1], [2]])

    def test_no_covariates_raises(self):
        """Empty covariates should raise ValueError."""
        with pytest.raises(ValueError):
            cox_ph([1, 2], [1, 0], [[], []])


# ---------------------------------------------------------------------------
# Multivariate model
# ---------------------------------------------------------------------------

class TestMultivariate:
    def test_multivariate_fit(self):
        """Multivariate Cox model with two covariates."""
        times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        events = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        covariates = [
            [0.5, 65], [1.2, 70], [0.8, 55], [1.5, 60], [0.3, 72],
            [1.1, 68], [0.9, 75], [1.4, 58], [0.6, 80], [1.0, 62]
        ]
        result = cox_ph(times, events, covariates)
        assert len(result["coefficients"]) == 2
        assert len(result["hazard_ratios"]) == 2
        assert len(result["ci_lower"]) == 2

    def test_multivariate_convergence(self):
        """Model should converge in reasonable iterations."""
        times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        events = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        covariates = [
            [0.5, 65], [1.2, 70], [0.8, 55], [1.5, 60], [0.3, 72],
            [1.1, 68], [0.9, 75], [1.4, 58], [0.6, 80], [1.0, 62]
        ]
        result = cox_ph(times, events, covariates)
        assert result["iterations"] <= 50


# ---------------------------------------------------------------------------
# Wald test
# ---------------------------------------------------------------------------

class TestWaldTest:
    def test_z_score_sign(self):
        """z-score should have same sign as coefficient."""
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        events = [1, 0, 1, 1, 0, 1, 0, 1]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1], [0.9], [1.4]]
        result = cox_ph(times, events, covariates)
        for k in range(len(result["coefficients"])):
            if result["coefficients"][k] != 0:
                assert (result["z_scores"][k] > 0) == (result["coefficients"][k] > 0)


# ---------------------------------------------------------------------------
# Forest plot data
# ---------------------------------------------------------------------------

class TestForestPlot:
    def test_forest_plot_structure(self):
        """Forest plot data should have correct structure."""
        times = [1, 2, 3, 4, 5]
        events = [1, 0, 1, 1, 0]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3]]
        fp = forest_plot_data(times, events, covariates, labels=["treatment"])
        assert len(fp) == 1
        assert fp[0]["label"] == "treatment"
        assert "hr" in fp[0]
        assert "ci_lower" in fp[0]
        assert "ci_upper" in fp[0]
        assert "p_value" in fp[0]
        assert "significant" in fp[0]

    def test_forest_plot_multivariate(self):
        """Forest plot with multiple covariates."""
        times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        events = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        covariates = [
            [0.5, 65], [1.2, 70], [0.8, 55], [1.5, 60], [0.3, 72],
            [1.1, 68], [0.9, 75], [1.4, 58], [0.6, 80], [1.0, 62]
        ]
        fp = forest_plot_data(times, events, covariates, labels=["treatment", "age"])
        assert len(fp) == 2


# ---------------------------------------------------------------------------
# Proportional hazards check
# ---------------------------------------------------------------------------

class TestPHCheck:
    def test_ph_check_structure(self):
        """PH check should return proper structure."""
        times = [1, 2, 3, 4, 5, 6, 7, 8]
        events = [1, 0, 1, 1, 0, 1, 0, 1]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3], [1.1], [0.9], [1.4]]
        result = check_proportional_hazards(times, events, covariates)
        assert "checks" in result
        assert len(result["checks"]) == 1
        assert "correlation" in result["checks"][0]
        assert "p_value" in result["checks"][0]
        assert "assumption_holds" in result["checks"][0]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_output(self):
        """Summary should return formatted string."""
        times = [1, 2, 3, 4, 5]
        events = [1, 0, 1, 1, 0]
        covariates = [[0.5], [1.2], [0.8], [1.5], [0.3]]
        s = summary(times, events, covariates)
        assert isinstance(s, str)
        assert "Cox" in s
        assert "Hazard" in s or "HR" in s


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

class TestCSVProcessing:
    def test_batch_process(self, tmp_path):
        """Batch CSV processing should produce output."""
        csv_in = tmp_path / "input.csv"
        csv_out = tmp_path / "output.csv"
        csv_in.write_text(
            "time,event,treatment\n1,1,1\n2,0,0\n3,1,1\n4,1,0\n5,0,1\n6,1,0\n7,1,1\n8,0,0\n",
            encoding="utf-8"
        )
        result = process_csv(str(csv_in), str(csv_out))
        assert csv_out.exists()
        assert result["n_subjects"] == 8


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------

class TestLinearAlgebra:
    def test_solve_linear_system(self):
        """Test solving a simple linear system."""
        A = [[2, 1], [1, 3]]
        b = [5, 7]
        x = _solve_linear_system(A, b)
        # 2*1 + 1*2 = 4... let me check: 2x + y = 5, x + 3y = 7
        # x = 8/5, y = 9/5
        assert abs(x[0] - 8.0 / 5.0) < 1e-10
        assert abs(x[1] - 9.0 / 5.0) < 1e-10

    def test_invert_matrix(self):
        """Test matrix inversion."""
        A = [[4, 7], [2, 6]]
        A_inv = _invert_matrix(A)
        # A * A_inv should be identity
        n = len(A)
        for i in range(n):
            for j in range(n):
                prod = sum(A[i][k] * A_inv[k][j] for k in range(n))
                expected = 1.0 if i == j else 0.0
                assert abs(prod - expected) < 1e-10

    def test_invert_identity(self):
        """Inverse of identity is identity."""
        I = [[1, 0], [0, 1]]
        I_inv = _invert_matrix(I)
        assert abs(I_inv[0][0] - 1.0) < 1e-10
        assert abs(I_inv[0][1]) < 1e-10
        assert abs(I_inv[1][0]) < 1e-10
        assert abs(I_inv[1][1] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

class TestStatHelpers:
    def test_pearson_correlation_perfect(self):
        """Perfect positive correlation."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert abs(_pearson_correlation(x, y) - 1.0) < 1e-10

    def test_pearson_correlation_negative(self):
        """Perfect negative correlation."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        assert abs(_pearson_correlation(x, y) - (-1.0)) < 1e-10

    def test_norm_cdf(self):
        """Standard normal CDF at known values."""
        assert abs(_norm_cdf(0) - 0.5) < 1e-10
        assert abs(_norm_cdf(1.96) - 0.975) < 0.001

    def test_z_to_p(self):
        """Two-tailed p-value from z."""
        # z=1.96 -> p≈0.05
        assert abs(_z_to_p(1.96) - 0.05) < 0.005
        # z=0 -> p=1
        assert abs(_z_to_p(0) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_strong_effect(self):
        """Strong covariate effect should give extreme HR."""
        # Group with high covariate dies early, low covariate survives
        times = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14]
        events = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        covariates = [[10], [10], [10], [10], [10], [0], [0], [0], [0], [0]]
        result = cox_ph(times, events, covariates)
        # HR should be > 1 (high covariate increases hazard)
        assert result["hazard_ratios"][0] > 1.0

    def test_large_dataset(self):
        """Performance test with larger dataset."""
        import random
        random.seed(42)
        n = 100
        times = [random.uniform(0.1, 50) for _ in range(n)]
        events = [random.choice([0, 1]) for _ in range(n)]
        covariates = [[random.uniform(0, 1)] for _ in range(n)]
        result = cox_ph(times, events, covariates)
        assert result["n_subjects"] == n


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_inconsistent_covariate_dimensions(self):
        """Inconsistent covariate dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="Covariate vector at index"):
            cox_ph([1, 2, 3], [1, 0, 1], [[1], [2, 3], [4]])

    def test_negative_time_raises(self):
        """Negative times should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            cox_ph([-1, 2, 3], [1, 0, 1], [[1], [2], [3]])

    def test_nan_time_raises(self):
        """NaN times should raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            cox_ph([float('nan'), 2, 3], [1, 0, 1], [[1], [2], [3]])

    def test_inf_time_raises(self):
        """Infinite times should raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            cox_ph([float('inf'), 2, 3], [1, 0, 1], [[1], [2], [3]])

    def test_invalid_event_value_raises(self):
        """Non-binary event values should raise ValueError."""
        with pytest.raises(ValueError, match="must be 0 or 1"):
            cox_ph([1, 2, 3], [2, 0, 1], [[1], [2], [3]])

    def test_nan_covariate_raises(self):
        """NaN covariate values should raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            cox_ph([1, 2, 3], [1, 0, 1], [[float('nan')], [2], [3]])

    def test_inf_covariate_raises(self):
        """Infinite covariate values should raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            cox_ph([1, 2, 3], [1, 0, 1], [[float('inf')], [2], [3]])


# ---------------------------------------------------------------------------
# Path traversal protection tests
# ---------------------------------------------------------------------------

class TestPathSecurity:
    def test_path_traversal_blocked(self):
        """Path traversal attempts should be blocked."""
        with pytest.raises(ValueError, match="Path traversal"):
            process_csv("../../../etc/passwd", "output.csv")

    def test_null_byte_blocked(self):
        """Null bytes in path should be blocked."""
        with pytest.raises(ValueError, match="null bytes"):
            process_csv("file\x00name.csv", "output.csv")
