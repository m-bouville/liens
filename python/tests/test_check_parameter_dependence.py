"""
Tests for evaluation/check_parameter_dependence.py's torch-free logic
(fit_power_law, fit_saturating_exponential, and the binning/per-run
aggregation logic inlined in check_parameter_dependence() itself,
extracted here for testing since the module can't be imported directly
without torch). Actually run in this environment -- no torch needed.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_check_parameter_dependence.py -v
"""
import numpy as np
import pytest

# Extracted verbatim from evaluation/check_parameter_dependence.py,
# since that module imports torch at the top level and can't be
# imported directly in a torch-free environment.


def fit_power_law(dt, error):
    log_dt = np.log(dt)
    log_err = np.log(np.clip(error, 1e-12, None))
    a, b = np.polyfit(log_dt, log_err, 1)
    pred_log = a * log_dt + b
    ss_res_log = np.sum((log_err - pred_log) ** 2)
    ss_tot_log = np.sum((log_err - log_err.mean()) ** 2)
    r2_log = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else float("nan")
    pred_real = np.exp(pred_log)
    sse_real = np.sum((error - pred_real) ** 2)
    return a, b, r2_log, sse_real, pred_real


def robust_linear_fit(x, y, n_iter=10, huber_delta_scale=1.345):
    slope, intercept = np.polyfit(x, y, deg=1)
    for _ in range(n_iter):
        residuals = y - (slope * x + intercept)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = 1.4826 * mad if mad > 0 else np.std(residuals) + 1e-12
        huber_delta = huber_delta_scale * scale
        abs_resid = np.abs(residuals)
        weights = np.where(abs_resid <= huber_delta, 1.0, huber_delta / np.maximum(abs_resid, 1e-12))
        slope, intercept = np.polyfit(x, y, deg=1, w=weights)
    return slope, intercept


def fit_exponential(x, error):
    log_err = np.log(np.clip(error, 1e-12, None))
    a, b = np.polyfit(x, log_err, 1)
    pred_log = a * x + b
    ss_res_log = np.sum((log_err - pred_log) ** 2)
    ss_tot_log = np.sum((log_err - log_err.mean()) ** 2)
    r2_log = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else float("nan")
    pred_real = np.exp(pred_log)
    sse_real = np.sum((error - pred_real) ** 2)
    return a, b, r2_log, sse_real, pred_real


def fit_saturating_exponential(dt, error, n_grid=200):
    tau_grid = np.logspace(np.log10(dt.min() / 10), np.log10(dt.max() * 10), n_grid)
    best_sse, best_tau, best_c = np.inf, None, None
    for tau in tau_grid:
        basis = 1 - np.exp(-dt / tau)
        denom = np.sum(basis ** 2)
        if denom < 1e-12:
            continue
        c = np.sum(error * basis) / denom
        pred = c * basis
        sse = np.sum((error - pred) ** 2)
        if sse < best_sse:
            best_sse, best_tau, best_c = sse, tau, c
    pred_real = best_c * (1 - np.exp(-dt / best_tau))
    ss_tot = np.sum((error - error.mean()) ** 2)
    r2_real = 1 - best_sse / ss_tot if ss_tot > 0 else float("nan")
    return best_c, best_tau, r2_real, best_sse, pred_real


def _make_bin_masks(values, n_bins):
    edges = np.linspace(values.min(), values.max(), n_bins + 1)
    masks = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (values >= lo) & (values <= hi if i == n_bins - 1 else values < hi)
        masks.append((lo, hi, mask))
    return masks


def _aggregate_per_run(run_dirs, temperatures, noises, latent_losses):
    per_run = {}
    for run_dir, t, n, ll in zip(run_dirs, temperatures, noises, latent_losses):
        entry = per_run.setdefault(run_dir, {"temperature": t, "noise": n, "losses": []})
        entry["losses"].append(ll)
    return per_run


def max_autocorr_dist(nx, ny):
    return min(nx * 2 // 3, ny * 2 // 3)


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


def test_fit_power_law_recovers_known_exponent():
    dt = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    true_a, true_b = 0.7, -2.0
    error = np.exp(true_b) * dt ** true_a  # exact power law, no noise
    a, b, r2_log, sse_real, pred_real = fit_power_law(dt, error)
    assert a == pytest.approx(true_a, abs=1e-6)
    assert b == pytest.approx(true_b, abs=1e-6)
    assert r2_log == pytest.approx(1.0, abs=1e-9)
    assert sse_real < 1e-9


def test_robust_linear_fit_recovers_a_known_clean_line():
    """Baseline sanity check with no outliers at all -- confirms the
    IRLS mechanism itself doesn't introduce bias when there's nothing
    to be robust AGAINST, before testing the actual robustness claim
    below."""
    x = np.linspace(1, 100, 50)
    true_slope, true_intercept = 0.3, -1.5
    y = true_slope * x + true_intercept  # exact line, no noise
    slope, intercept = robust_linear_fit(x, y)
    assert slope == pytest.approx(true_slope, abs=1e-6)
    assert intercept == pytest.approx(true_intercept, abs=1e-6)


def test_robust_linear_fit_resists_a_planted_outlier_better_than_ols():
    """The actual claim this function exists for: with a single,
    extreme outlier planted among otherwise-clean points on a known
    line, plain OLS (np.polyfit) gets measurably pulled toward it,
    while the robust fit stays much closer to the TRUE line -- verified
    directly by comparing both fits' own distance from the known
    ground truth, not just asserting the robust fit "looks reasonable"
    in isolation."""
    x = np.linspace(1, 100, 50)
    true_slope, true_intercept = 0.3, -1.5
    y = true_slope * x + true_intercept
    # One extreme outlier, concentrated at a SMALL x -- mirrors the
    # actual concern raised (small-dt windows with unusually large
    # residuals dominating the fit).
    y_with_outlier = y.copy()
    y_with_outlier[0] += 500.0

    ols_slope, ols_intercept = np.polyfit(x, y_with_outlier, deg=1)
    robust_slope, robust_intercept = robust_linear_fit(x, y_with_outlier)

    ols_error = abs(ols_slope - true_slope) + abs(ols_intercept - true_intercept)
    robust_error = abs(robust_slope - true_slope) + abs(robust_intercept - true_intercept)
    assert robust_error < ols_error / 5, (
        f"robust fit should be MUCH closer to the true line than OLS given a single planted "
        f"outlier -- OLS error={ols_error:.4f}, robust error={robust_error:.4f}"
    )
    # The robust fit shouldn't just be "less wrong" -- it should be
    # genuinely close to the true line, not merely better than a badly
    # wrong OLS fit.
    assert robust_slope == pytest.approx(true_slope, abs=0.05)
    assert robust_intercept == pytest.approx(true_intercept, abs=5.0)


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


def test_binned_summary_covers_every_value_exactly_once():
    values = np.linspace(0, 10, 11)
    masks = _make_bin_masks(values, n_bins=5)

    covered = np.zeros(len(values), dtype=bool)
    for lo, hi, mask in masks:
        assert not np.any(covered & mask), "bins overlap"
        covered |= mask
    assert covered.all(), "some values not covered by any bin"


def test_binned_summary_includes_max_value_in_last_bin():
    """The last bin uses <= specifically so the maximum observed value
    isn't silently dropped (every other bin uses < to avoid
    double-counting the shared edge with the next bin)."""
    values = np.array([0.0, 5.0, 10.0])
    masks = _make_bin_masks(values, n_bins=2)
    _, _, last_mask = masks[-1]
    assert last_mask[-1] == True  # noqa: E712 -- explicit bool check reads clearer here


def test_per_run_aggregation_groups_and_averages_correctly():
    from pathlib import Path
    run_dirs = [Path("runA"), Path("runA"), Path("runB"), Path("runA"), Path("runB")]
    temperatures = [0.8, 0.8, 0.9, 0.8, 0.9]
    noises = [0.01, 0.01, 0.02, 0.01, 0.02]
    latent_losses = [0.1, 0.3, 0.2, 0.5, 0.7]  # runA: [0.1,0.3,0.5]->0.3, runB: [0.2,0.7]->0.45

    per_run = _aggregate_per_run(run_dirs, temperatures, noises, latent_losses)

    assert set(per_run.keys()) == {Path("runA"), Path("runB")}
    assert np.mean(per_run[Path("runA")]["losses"]) == pytest.approx(0.3)
    assert len(per_run[Path("runA")]["losses"]) == 3
    assert np.mean(per_run[Path("runB")]["losses"]) == pytest.approx(0.45)
    assert len(per_run[Path("runB")]["losses"]) == 2
    assert per_run[Path("runA")]["temperature"] == 0.8
    assert per_run[Path("runB")]["noise"] == 0.02
