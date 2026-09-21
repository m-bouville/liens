"""
Tests for the torch-free numeric helpers behind check_parameter_dependence's
figures and reports: the fit family (fit_power_law, fit_exponential,
fit_saturating_exponential, robust_polynomial_fit, fit_taylor_residual_
coefficients -- all in utils.fits), max_autocorr_dist, and the binned/grouped
mean-curve + y-range helpers (_mean_curves_by_unique_value, _mean_curves_by_bin,
_size_by_count, _symmetric_left_zero_right_ylim, _ylim_from_below_cutoff).

These import the SHIPPED implementations directly rather than testing verbatim
copies pasted into this file (which an earlier version did, on a now-false "the
module can't be imported without torch" rationale -- utils.fits is numpy-only,
and evaluation.check_parameter_dependence imports fine in the test env, as the
sibling end-to-end tests and test_evaluation_reconstruction_integration already
rely on). Testing copies let the shipped code regress with every test still
green, and the copies had already drifted (e.g. _size_by_count's empty-array
guard, _mean_curves_by_bin's non-positive-x handling) -- the whole point of the
fixup was to make these guard the real functions.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_check_parameter_dependence.py -v
"""
import numpy as np
import pytest

from utils.fits import (
    fit_exponential, fit_power_law, fit_saturating_exponential,
    fit_taylor_residual_coefficients, robust_polynomial_fit,
)
from evaluation.check_parameter_dependence import (
    max_autocorr_dist, _mean_curves_by_unique_value, _mean_curves_by_bin,
    _size_by_count, _symmetric_left_zero_right_ylim, _ylim_from_below_cutoff,
)

# NOTE on aggregation coverage: the per-(temperature,noise) vs per-run
# aggregation behind panel [0,2] and the "worst runs" report is NOT a
# standalone function -- it lives inline in
# check_parameter_dependence._print_summary_statistics (built there as `per_point`
# and `per_run` dict comprehensions). A previous version of this file tested
# local copies named `_aggregate_per_point`/`_aggregate_per_run` that do not exist
# in the module, so those tests guarded nothing and were removed. The inline
# aggregation is exercised end-to-end by
# test_evaluation_reconstruction_integration.test_check_parameter_dependence_non_default_spatial_size
# (which asserts the populated SUMMARY block). Extracting it into a real helper
# here would let it be unit-tested directly -- a worthwhile follow-up, but a
# change to the module, not this test file.


# ---------------------------------------------------------------------
# max_autocorr_dist
# ---------------------------------------------------------------------

def test_max_autocorr_dist_matches_cpp_formula():
    """The C++ side computes int max_dist = std::min(Nx*2/3, Ny*2/3) --
    integer division. Python's // matches C++'s truncating int division
    for non-negative operands, so this must reproduce the exact same
    sentinel value the simulation actually wrote out, not an
    approximation (e.g. round() would give 43 for Nx=64, not 42)."""
    assert max_autocorr_dist(64, 64) == 42  # 64*2=128, 128//3=42 (not 42.67 rounded)
    assert max_autocorr_dist(128, 128) == 85


def test_max_autocorr_dist_takes_the_smaller_axis():
    assert max_autocorr_dist(64, 32) == 21  # min(42, 21) -> limited by the shorter axis


# ---------------------------------------------------------------------
# fit_power_law / fit_exponential / fit_saturating_exponential
# ---------------------------------------------------------------------

def test_fit_power_law_recovers_known_exponent():
    dt = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    true_a, true_b = 0.7, -2.0
    error = np.exp(true_b) * dt ** true_a  # exact power law, no noise
    a, b, r2_log, sse_real, pred_real = fit_power_law(dt, error)
    assert a == pytest.approx(true_a, abs=1e-6)
    assert b == pytest.approx(true_b, abs=1e-6)
    assert r2_log == pytest.approx(1.0, abs=1e-9)
    assert sse_real < 1e-9


def test_fit_exponential_recovers_known_params():
    """The semi-log analogue of the power-law test above -- x itself
    (not log(x)) is linear in log(error), appropriate for a panel like
    length_scale's (linear x-axis, log-scaled error), where fit_power_law
    would fit a curve rather than the straight line this is built for."""
    x = np.linspace(0, 40, 50)
    true_a, true_b = 0.05, -3.0
    error = np.exp(true_a * x + true_b)  # exact exponential, no noise
    a, b, r2_log, sse_real, pred_real = fit_exponential(x, error)
    assert a == pytest.approx(true_a, abs=1e-6)
    assert b == pytest.approx(true_b, abs=1e-6)
    assert r2_log == pytest.approx(1.0, abs=1e-9)
    assert sse_real < 1e-9


def test_fit_saturating_exponential_recovers_known_params():
    dt = np.linspace(1, 500, 50)
    true_c, true_tau = 3.5, 80.0
    error = true_c * (1 - np.exp(-dt / true_tau))  # exact, no noise
    c, tau, r2_real, sse, pred_real = fit_saturating_exponential(dt, error)
    assert c == pytest.approx(true_c, rel=0.05)
    assert tau == pytest.approx(true_tau, rel=0.15)  # coarser -- grid search, not continuous optimization
    assert r2_real == pytest.approx(1.0, abs=1e-3)


def test_model_comparison_prefers_the_true_generating_model():
    """If the data really is a saturating exponential, the fit
    comparison (lower SSE wins) should say so -- and vice versa for a
    true power law. This is the actual decision logic the script prints
    to the user, so it's worth checking both directions explicitly."""
    dt = np.linspace(1, 500, 50)

    true_c, true_tau = 3.5, 80.0
    error_sat = true_c * (1 - np.exp(-dt / true_tau))
    _, _, _, sse_power_1, _ = fit_power_law(dt, error_sat)
    _, _, _, sse_sat_1, _ = fit_saturating_exponential(dt, error_sat)
    assert sse_sat_1 < sse_power_1

    error_power = 0.1 * dt ** 0.7
    _, _, _, sse_power_2, _ = fit_power_law(dt, error_power)
    _, _, _, sse_sat_2, _ = fit_saturating_exponential(dt, error_power)
    assert sse_power_2 < sse_sat_2


# ---------------------------------------------------------------------
# robust_polynomial_fit
# ---------------------------------------------------------------------

def test_robust_polynomial_fit_recovers_a_known_clean_line():
    """Baseline sanity check with the SAME 2-term [1, x] basis
    robust_linear_fit used to hard-code, and no outliers at all --
    confirms the IRLS mechanism itself doesn't introduce bias when
    there's nothing to be robust AGAINST, before testing the actual
    robustness claim below."""
    x = np.linspace(1, 100, 50)
    true_intercept, true_slope = -1.5, 0.3
    y = true_slope * x + true_intercept  # exact line, no noise
    basis_funcs = [lambda xx: np.ones_like(xx), lambda xx: xx]
    coefs, stderr = robust_polynomial_fit(x, y, basis_funcs)
    assert coefs[0] == pytest.approx(true_intercept, abs=1e-6)
    assert coefs[1] == pytest.approx(true_slope, abs=1e-6)


def test_robust_polynomial_fit_resists_a_planted_outlier():
    """The actual claim this function exists for: with a single,
    extreme outlier planted among otherwise-clean points on a known
    line, plain OLS gets measurably pulled toward it, while the robust
    fit stays much closer to the TRUE line -- verified directly by
    comparing both fits' own distance from the known ground truth."""
    x = np.linspace(1, 100, 50)
    true_intercept, true_slope = -1.5, 0.3
    y = true_slope * x + true_intercept
    y_with_outlier = y.copy()
    y_with_outlier[0] += 500.0  # concentrated at small x, mirrors small-dt windows dominating a fit

    ols_intercept, ols_slope = np.polyfit(x, y_with_outlier, deg=1)[::-1]
    basis_funcs = [lambda xx: np.ones_like(xx), lambda xx: xx]
    coefs, stderr = robust_polynomial_fit(x, y_with_outlier, basis_funcs)
    robust_intercept, robust_slope = coefs

    ols_error = abs(ols_slope - true_slope) + abs(ols_intercept - true_intercept)
    robust_error = abs(robust_slope - true_slope) + abs(robust_intercept - true_intercept)
    assert robust_error < ols_error / 5, (
        f"robust fit should be MUCH closer to the true line than OLS given a single planted "
        f"outlier -- OLS error={ols_error:.4f}, robust error={robust_error:.4f}"
    )
    assert robust_slope == pytest.approx(true_slope, abs=0.05)
    assert robust_intercept == pytest.approx(true_intercept, abs=5.0)


def test_robust_polynomial_fit_recovers_a_known_cubic():
    """The actual generalization robust_linear_fit couldn't do at all --
    an arbitrary basis, not just [1, x]. A clean (no-noise) cubic with a
    4-term basis [1, x, x^2, x^3] should be recovered essentially
    exactly, confirming the basis generalization itself is correct, not
    just the outlier-resistance mechanism it wraps (already covered
    above, unchanged from the linear case)."""
    x = np.linspace(1, 50, 60)
    true_coefs = np.array([0.5, -0.02, 0.001, -0.00001])
    basis_funcs = [lambda xx: np.ones_like(xx), lambda xx: xx, lambda xx: xx ** 2, lambda xx: xx ** 3]
    y = sum(c * f(x) for c, f in zip(true_coefs, basis_funcs))
    coefs, stderr = robust_polynomial_fit(x, y, basis_funcs)
    np.testing.assert_allclose(coefs, true_coefs, atol=1e-6)


def test_robust_polynomial_fit_stderr_grows_with_noise():
    """The returned standard errors should actually reflect how noisy
    the fit is -- a sanity check that they're not just placeholder
    zeros or some fixed value, by comparing a clean fit's stderr
    against a noisy fit's stderr on the SAME underlying line."""
    x = np.linspace(1, 100, 200)
    true_intercept, true_slope = -1.5, 0.3
    y_clean = true_slope * x + true_intercept
    rng = np.random.RandomState(0)
    y_noisy = y_clean + rng.normal(0, 5.0, size=len(x))
    basis_funcs = [lambda xx: np.ones_like(xx), lambda xx: xx]
    _, stderr_clean = robust_polynomial_fit(x, y_clean, basis_funcs)
    _, stderr_noisy = robust_polynomial_fit(x, y_noisy, basis_funcs)
    assert stderr_noisy[1] > stderr_clean[1] * 10  # slope's own stderr should be MUCH larger under real noise


# ---------------------------------------------------------------------
# fit_taylor_residual_coefficients
# ---------------------------------------------------------------------

def test_fit_taylor_residual_coefficients_joint_mode_recovers_known_params():
    """The joint (euler_only=False) fit shares eps/eps'/A across BOTH
    residual types, but lets C and D differ independently -- construct
    synthetic data with genuinely different C/D so a bug that
    accidentally forced them equal (or accidentally shared A/eps/eps'
    incorrectly) would be caught."""
    rng = np.random.RandomState(0)
    n = 4000
    dts = np.exp(rng.uniform(np.log(20), np.log(30000), n))
    true_eps, true_eps_prime, true_C, true_D, true_A = 2e-3, -8e-5, -1.2e-10, 1.9e-7, 3e-15

    euler_signed = true_eps + true_eps_prime * dts + true_C * dts ** 2 - true_A * dts ** 3
    full_signed = true_eps + true_eps_prime * dts + true_D * dts ** 2 - true_A * dts ** 3

    result = fit_taylor_residual_coefficients(dts, euler_signed, full_signed, label="TEST")
    assert result["eps"] == pytest.approx(true_eps, rel=0.05)
    assert result["eps_prime"] == pytest.approx(true_eps_prime, rel=0.05)
    assert result["C"] == pytest.approx(true_C, rel=0.1)
    assert result["D"] == pytest.approx(true_D, rel=0.1)
    assert result["A"] == pytest.approx(true_A, rel=0.2)
    assert "mean_z0_ddot" in result
    assert "mean_f_theta_minus_z0_ddot" in result


def test_fit_taylor_residual_coefficients_euler_only_mode_ignores_second_array():
    """euler_only=True must fit ONLY against euler_losses_signed -- the
    second (latent_losses_signed) array passed in should be completely
    irrelevant, even if it's garbage, and the returned dict should have
    NO 'D' key at all (a real, earlier bug: check_parameter_dependence's
    own panel [1,3] used to KeyError on this exact thing when it forgot
    to guard a 'D' lookup behind euler_only)."""
    rng = np.random.RandomState(1)
    n = 3000
    dts = np.exp(rng.uniform(np.log(20), np.log(30000), n))
    true_eps, true_eps_prime, true_C, true_A = 2e-3, -5e-5, -1.5e-10, 2e-15
    euler_signed = true_eps + true_eps_prime * dts + true_C * dts ** 2 - true_A * dts ** 3
    garbage = rng.normal(0, 1000, n)  # deliberately unrelated, large-magnitude

    result = fit_taylor_residual_coefficients(dts, euler_signed, garbage, euler_only=True, label="TEST")
    assert result["eps"] == pytest.approx(true_eps, rel=0.05)
    assert result["eps_prime"] == pytest.approx(true_eps_prime, rel=0.05)
    assert result["C"] == pytest.approx(true_C, rel=0.1)
    assert "D" not in result
    assert result["euler_only"] is True


# ---------------------------------------------------------------------
# _mean_curves_by_unique_value / _mean_curves_by_bin
# ---------------------------------------------------------------------

def test_mean_curves_by_unique_value_groups_and_averages_correctly():
    x = np.array([0.55, 0.55, 0.60, 0.60, 0.60])
    y_signed = np.array([1.0, 3.0, -2.0, -4.0, 0.0])
    y_abs = np.array([1.0, 3.0, 2.0, 4.0, 0.0])
    unique_x, mean_signed, mean_abs, n_windows = _mean_curves_by_unique_value(x, y_signed, y_abs)
    np.testing.assert_allclose(unique_x, [0.55, 0.60])
    np.testing.assert_allclose(mean_signed, [2.0, -2.0])
    np.testing.assert_allclose(mean_abs, [2.0, 2.0])
    np.testing.assert_array_equal(n_windows, [2, 3])


def test_mean_curves_by_unique_value_merges_near_duplicate_floats():
    """Same rounding rationale as _boxplot_by_x: float round-trip
    through a text metadata file can turn one intended sweep value into
    several bit-distinct floats -- these must still merge into one
    group, not silently multiply the apparent number of sweep points."""
    x = np.array([0.55, 0.5500000001, 0.5499999998, 0.60])
    y_signed = np.array([1.0, 2.0, 3.0, 5.0])
    y_abs = np.abs(y_signed)
    unique_x, mean_signed, mean_abs, n_windows = _mean_curves_by_unique_value(x, y_signed, y_abs)
    assert len(unique_x) == 2
    np.testing.assert_array_equal(n_windows, [3, 1])


def test_mean_curves_by_bin_covers_every_point_exactly_once():
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 40, 500)
    y_signed = rng.normal(0, 1, 500)
    y_abs = np.abs(y_signed)
    _, _, _, n_windows = _mean_curves_by_bin(x, y_signed, y_abs, n_bins=8)
    assert n_windows.sum() == 500  # every point counted, none double-counted, none dropped


def test_mean_curves_by_bin_includes_max_value():
    """The last bin uses <= specifically so the maximum observed value
    isn't silently dropped."""
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([1.0, 2.0, 3.0])
    centers, mean_signed, mean_abs, n_windows = _mean_curves_by_bin(x, y, np.abs(y), n_bins=2)
    assert n_windows.sum() == 3


def test_mean_curves_by_bin_log_bins_spaces_edges_geometrically():
    """log_bins=True (used for dt, which spans several orders of
    magnitude) should place bin edges geometrically, not linearly --
    checked directly via each bin's own center, which for log bins is a
    geometric mean (sqrt(lo*hi)), not an arithmetic one."""
    rng = np.random.RandomState(0)
    x = np.exp(rng.uniform(np.log(10), np.log(100000), 2000))  # log-uniform, like dt
    y = rng.normal(0, 1, 2000)
    centers_log, _, _, n_log = _mean_curves_by_bin(x, y, np.abs(y), n_bins=5, log_bins=True)
    centers_lin, _, _, n_lin = _mean_curves_by_bin(x, y, np.abs(y), n_bins=5, log_bins=False)
    # Log bins should be roughly EVENLY spread in count across 5 decades
    # of data; linear bins on the same log-uniform data pile almost
    # everything into the first (smallest-x) bin instead.
    assert n_log.std() < n_lin.std()
    # Successive log-bin centers should have a roughly CONSTANT ratio
    # (geometric spacing), not a constant difference (arithmetic).
    ratios = centers_log[1:] / centers_log[:-1]
    assert ratios.std() < 0.5  # roughly constant ratio -- loose bound, just confirms geometric-ish spacing


# ---------------------------------------------------------------------
# _size_by_count
# ---------------------------------------------------------------------

def test_size_by_count_spans_the_requested_range():
    n_windows = np.array([5, 50, 500])
    sizes = _size_by_count(n_windows, min_size=20.0, max_size=150.0)
    assert sizes.min() == pytest.approx(20.0)
    assert sizes.max() == pytest.approx(150.0)
    assert sizes[0] < sizes[1] < sizes[2]  # monotonic in window count


def test_size_by_count_handles_all_equal_counts_without_dividing_by_zero():
    """hi <= lo (every group has the same window count) must not raise
    a ZeroDivisionError -- falls back to the midpoint size for every
    point instead."""
    n_windows = np.array([10, 10, 10])
    sizes = _size_by_count(n_windows, min_size=20.0, max_size=150.0)
    np.testing.assert_allclose(sizes, 85.0)  # midpoint of [20, 150]


def test_size_by_count_handles_an_empty_curve():
    """A curve emptied by run-coverage filtering is a legitimate outcome;
    .min() on an empty array raises, so the shipped helper guards it and
    returns the (empty) input unchanged rather than crashing."""
    sizes = _size_by_count(np.array([]), min_size=20.0, max_size=150.0)
    assert sizes.size == 0


# ---------------------------------------------------------------------
# _symmetric_left_zero_right_ylim
# ---------------------------------------------------------------------

def test_symmetric_left_zero_right_ylim():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2)
    twin1, twin2 = ax1.twinx(), ax2.twinx()
    # margins(0) on every axis -- matplotlib pads autoscaled limits by
    # ~5% by default, which would make an exact-value comparison below
    # fail for reasons having nothing to do with the function under
    # test. Not needed by _symmetric_left_zero_right_ylim itself (it
    # just reads whatever get_ylim() already returns), only by this
    # test wanting an EXACT expected value to assert against.
    for ax in (ax1, ax2, twin1, twin2):
        ax.margins(0)
    ax1.plot([0, 1], [-3.0, 2.0])   # left range: [-3, 2]
    ax2.plot([0, 1], [-1.0, 5.0])   # left range: [-1, 5]
    twin1.plot([0, 1], [0.5, 4.0])  # right range: [0.5, 4]
    twin2.plot([0, 1], [1.0, 2.0])  # right range: [1, 2]

    left_ylim, right_ylim = _symmetric_left_zero_right_ylim([ax1, ax2], [twin1, twin2])
    plt.close(fig)

    # Left: symmetric about 0, magnitude = the largest |value| seen
    # across BOTH left axes (5.0, from ax2's own upper limit).
    assert left_ylim[0] == pytest.approx(-left_ylim[1])
    assert left_ylim[1] == pytest.approx(5.0)
    # Right: floored at 0 regardless of what was actually plotted,
    # extending to the largest value seen across BOTH right axes (4.0).
    assert right_ylim[0] == pytest.approx(0.0)
    assert right_ylim[1] == pytest.approx(4.0)


# ---------------------------------------------------------------------
# _ylim_from_below_cutoff
# ---------------------------------------------------------------------

def test_ylim_from_below_cutoff_ignores_the_converged_regime():
    """The dt-dependence y-range must come ONLY from points below the
    convergence cutoff -- points at/above it (dz0->0, error/dt meaningless)
    would blow the range up. This logic caused a three-iteration y-range saga
    when it lived inline in _build_and_save_figures; now a module-level unit."""
    fb = (-9.0, 9.0)
    # below-cutoff points [0,10] set the range (padded 5%); x=100 (converged) excluded
    lo, hi = _ylim_from_below_cutoff([([1.0, 2.0, 100.0], [0.0, 10.0, 999.0])],
                                     dt_cutoff=50.0, fallback=fb)
    assert (round(lo, 9), round(hi, 9)) == (-0.5, 10.5)
    # nothing below the cutoff -> fall back (e.g. cutoff=inf, no convergence)
    assert _ylim_from_below_cutoff([([1.0, 2.0], [0.0, 10.0])],
                                   dt_cutoff=0.5, fallback=fb) == fb
    # degenerate (all equal) -> fallback, not a zero-height range
    assert _ylim_from_below_cutoff([([1.0, 2.0], [5.0, 5.0])],
                                   dt_cutoff=10.0, fallback=fb) == fb
    # non-finite points ignored
    lo, hi = _ylim_from_below_cutoff([([1.0, float("nan"), 3.0], [2.0, 100.0, 4.0])],
                                     dt_cutoff=10.0, fallback=fb)
    assert (round(lo, 9), round(hi, 9)) == (1.9, 4.1)
