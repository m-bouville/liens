"""
Tests for evaluation/check_perturbation.py's linear_fit. Pure numpy --
no torch dependency -- so this actually runs here and is checked
directly.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_check_perturbation.py -v
"""
import numpy as np
import pytest

from evaluation.check_perturbation import linear_fit


def test_linear_fit_perfect_line():
    """delta = 2*eps + 0.5 exactly, no noise -- should recover the
    exact slope/intercept and R^2 == 1."""
    eps_values = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4])
    delta = 2.0 * eps_values + 0.5

    dz, c, r_squared = linear_fit(eps_values, delta)

    assert dz == pytest.approx(2.0, abs=1e-9)
    assert c == pytest.approx(0.5, abs=1e-9)
    assert r_squared == pytest.approx(1.0, abs=1e-9)


def test_linear_fit_with_noise_gives_partial_r_squared():
    """Real noisy data should recover a slope close to the true one,
    with R^2 meaningfully less than 1 (but still positive, since the
    linear trend dominates)."""
    rng = np.random.default_rng(0)
    eps_values = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4])
    true_dz, true_c = 1.0, 0.0
    noise = rng.normal(scale=0.02, size=eps_values.shape)
    delta = true_dz * eps_values + true_c + noise

    dz, c, r_squared = linear_fit(eps_values, delta)

    assert dz == pytest.approx(true_dz, abs=0.1)
    assert 0.0 < r_squared < 1.0


def test_linear_fit_constant_delta_gives_nan_r_squared():
    """ss_tot == 0 (delta is perfectly constant, no variance to explain
    at all) should give R^2 = nan, not a divide-by-zero error or a
    misleadingly perfect 1.0."""
    eps_values = np.array([0.1, 0.2, 0.3])
    delta = np.array([0.5, 0.5, 0.5])

    dz, c, r_squared = linear_fit(eps_values, delta)

    assert np.isnan(r_squared)


def test_linear_fit_pairwise_scaling_property():
    """A sanity property used elsewhere in this script's own output:
    for a clean linear response through the origin, delta(0.3)/delta(0.1)
    should equal eps2/eps1 = 3.0 -- confirms linear_fit's recovered
    slope reproduces this ratio correctly."""
    eps_values = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4])
    dz_true = 0.7
    delta = dz_true * eps_values  # exactly through the origin

    dz, c, r_squared = linear_fit(eps_values, delta)
    delta_at_01 = dz * 0.1 + c
    delta_at_03 = dz * 0.3 + c
    assert (delta_at_03 / delta_at_01) == pytest.approx(3.0, abs=1e-6)
