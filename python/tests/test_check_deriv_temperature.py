"""
Tests for evaluation/check_deriv_temperature.py -- specifically, that it
shares evaluation._fits.robust_polynomial_fit rather than carrying its
own copy (as it used to). See _fits.py's own module docstring for why
that duplication was a real problem, not just untidy: the two copies
were used to fit the SAME underlying quantity (eps/eps') against
DIFFERENT bases while reporting the shared coefficients under the same
names, producing directly-contradictory numbers from the same checkpoint.
"""
import numpy as np
import pytest

import evaluation.check_deriv_temperature as cdt
from evaluation._fits import robust_polynomial_fit


def test_shares_fits_module_rather_than_redefining_it():
    """
    REGRESSION: check_deriv_temperature.py must import
    robust_polynomial_fit from evaluation._fits, not define its own
    module-level copy again. `is`, not just numeric equality -- the
    whole point is ONE implementation, not two that happen to agree
    right now and can silently drift apart later (exactly how this
    duplication originally happened).
    """
    assert cdt.robust_polynomial_fit is robust_polynomial_fit


def test_fit_still_recovers_known_coefficients_through_the_shared_import():
    """Exercises the IRLS/Huber iteration (not just import wiring) via
    the path check_deriv_temperature.py actually calls it through --
    same shape of problem this module fits: y = eps/dt + eps', with
    outliers the Huber reweighting must down-weight to recover the
    true coefficients."""
    rng = np.random.default_rng(0)
    dt = rng.uniform(10, 5000, 300)
    true_eps, true_eps_prime = 3.1e-3, -8e-5
    y = true_eps / dt + true_eps_prime + rng.normal(0, 1e-4, 300)
    y[::20] += rng.normal(0, 5e-2, 15)  # outliers

    basis = [lambda d: 1.0 / d, lambda d: np.ones_like(d)]
    coefs, stderr = cdt.robust_polynomial_fit(dt, y, basis)

    assert coefs[0] == pytest.approx(true_eps, rel=0.1)
    assert coefs[1] == pytest.approx(true_eps_prime, abs=5e-5)
    assert np.all(np.isfinite(stderr))
