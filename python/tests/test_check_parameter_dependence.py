"""
Tests for evaluation/check_parameter_dependence.py's torch-free logic
(fit_power_law, fit_saturating_exponential, fit_exponential,
robust_polynomial_fit, fit_taylor_residual_coefficients, the binned/
grouped mean-curve helpers behind the [1,0]/[1,1]/[0,3] panels, and the
per-(temperature,noise)/per-run aggregation logic behind panel [0,2] and
the "highest error" console report) -- extracted verbatim below, since
the module can't be imported directly without torch. Actually run in
this environment -- no torch needed.

Regenerated from scratch against the CURRENT module (the previous
version of this file predated: robust_polynomial_fit replacing
robust_linear_fit, fit_taylor_residual_coefficients, the
_mean_curves_by_unique_value/_mean_curves_by_bin/_size_by_count/
_symmetric_left_zero_right_ylim helpers behind the current [1,0]/[1,1]/
[0,3] panels, and -- most importantly -- panel [0,2]'s aggregation
switching from per-run to per-(temperature,noise), which is exactly the
kind of behavior change most worth having a real regression test for).

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_check_parameter_dependence.py -v
"""
import numpy as np
import pytest

# Extracted verbatim from evaluation/check_parameter_dependence.py,
# since that module imports torch at the top level and can't be
# imported directly in a torch-free environment.


def max_autocorr_dist(nx: int, ny: int) -> int:
    """
    The C++ simulation caps autocorr_length's search at
    min(Nx*2/3, Ny*2/3) (integer division) -- distances beyond that are
    deemed artifacts of the periodic-boundary autocorrelation wrapping
    around on itself, not a genuine length scale. Any window whose
    autocorrelation never decays within that search range gets this
    exact value back as a SENTINEL, not a real measurement -- and it's
    common enough (near-critical/smooth microstructures in particular)
    to distort both the plot and any regression fit through it if left
    in as if it were real data. Mirrors the C++ integer-division
    formula exactly (Python's // matches C++'s truncating int division
    for non-negative operands), so this returns the same sentinel value
    the simulation actually produced, not an approximation of it.
    """
    return min(nx * 2 // 3, ny * 2 // 3)


def robust_polynomial_fit(x: np.ndarray, y: np.ndarray, basis_funcs: list,
                           n_iter: int = 10, huber_delta_scale: float = 1.345):
    """
    Generalizes robust_linear_fit (see its own docstring for the IRLS/
    Huber mechanism itself, unchanged here) from a fixed 2-term model
    (slope*x + intercept) to an ARBITRARY set of basis functions of x --
    y = sum_j(coef_j * basis_funcs[j](x)). Needed specifically for the
    Taylor-residual decomposition below: that model has a genuine 1/dt
    term (see fit_taylor_residual_coefficients' own docstring for why),
    which robust_linear_fit's fixed [x, 1] basis has no way to
    represent at all -- fitting a straight line to data that actually
    contains a 1/dt term doesn't approximate it, it silently absorbs
    that term's effect into a BIASED estimate of the intercept instead.

    Returns (coefs, coef_stderr): coefs in the SAME order as
    basis_funcs; coef_stderr are the weighted-least-squares standard
    errors from the FINAL IRLS iteration's own weights (sigma^2 *
    (X^T W X)^-1 diagonal) -- lets a caller judge whether a given
    coefficient (e.g. the 1/dt term's own coefficient) is actually
    distinguishable from zero, not just report a point estimate with
    no sense of its own uncertainty.
    """
    X = np.column_stack([f(x) for f in basis_funcs])
    n, p = X.shape
    weights = np.ones(n)

    def _weighted_lstsq(w):
        sqrt_w = np.sqrt(w)
        coefs, *_ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)
        return coefs

    coefs = _weighted_lstsq(weights)
    for _ in range(n_iter):
        residuals = y - X @ coefs
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = 1.4826 * mad if mad > 0 else np.std(residuals) + 1e-12
        huber_delta = huber_delta_scale * scale
        abs_resid = np.abs(residuals)
        weights = np.where(abs_resid <= huber_delta, 1.0, huber_delta / np.maximum(abs_resid, 1e-12))
        coefs = _weighted_lstsq(weights)

    # Standard errors from the FINAL weights -- same weighted normal-
    # equations matrix the last _weighted_lstsq call itself solved,
    # reused here rather than recomputed independently, so these are
    # guaranteed consistent with the coefficients actually returned.
    residuals = y - X @ coefs
    XtWX = X.T @ (weights[:, None] * X)
    dof = max(n - p, 1)
    sigma2 = np.sum(weights * residuals ** 2) / dof
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
        coef_stderr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        coef_stderr = np.full(p, np.nan)  # near-singular design (e.g. dt range too narrow) -- be honest, not silent
    return coefs, coef_stderr


def fit_taylor_residual_coefficients(dts: np.ndarray, euler_losses_signed: np.ndarray,
                                      latent_losses_signed: np.ndarray,
                                      n_iter: int = 10, huber_delta_scale: float = 1.345,
                                      label: str = "", euler_only: bool = False) -> dict:
    """
    Fits the FULL Taylor-residual model (not the plain-linear
    approximation robust_linear_fit's own 2-term model reduces to),
    separating out the divergent 1/dt term explicitly rather than
    letting it silently bias a straight-line fit's intercept.

    SIGN CONVENTION: predicted minus true, throughout -- matching
    _per_sample_signed_mean(pred, true) = (pred-true).mean(), which is
    where euler_losses_signed/latent_losses_signed themselves come from
    (unchanged by this function), and matching the derivation's own
    numerator (z0_tilde(t+dt) - z0(t+dt), i.e. predicted minus true).
    Not something this function chose independently -- it's inherited
    directly from how those two arrays were already computed.

    Derivation (z1(t) = [z0(t+dt)-z0(t)+eps]/dt + eps' -- z1's own
    error, split into a piece that scales with 1/dt and a piece that
    doesn't; z0_ddot(t) the TRUE curvature; A*dt^3 the next real Taylor
    term beyond what either the Euler-only or f_theta-corrected
    prediction can represent). Written here in terms of the UNDIVIDED
    residual (euler_losses_signed/latent_losses_signed themselves, NOT
    divided by dt) -- fit AGAINST this form, not R=residual/dt:
        euler_losses_signed = eps + eps'*dt - (z0_ddot/2)*dt^2 - A*dt^3
        latent_losses_signed = eps + eps'*dt + ((f_theta-z0_ddot)/2)*dt^2 - A*dt^3
    (multiply either of the R(dt) formulas from this function's own
    module docstring by dt to get these directly). Fitting the
    UNDIVIDED residual, not R=residual/dt, avoids a real problem: if
    the residual's own measurement noise is roughly dt-INDEPENDENT
    (plausible -- e.g. floating-point/encoder-level noise with no
    reason to scale with dt itself), then dividing by dt makes that
    same fixed-size noise LOOK like it grows as 1/dt at small dt --
    heteroscedastic (dt-dependent-variance) noise that violates plain
    least-squares' own equal-variance assumption, and which Huber/IRLS
    reweighting does NOT fix (it downweights OUTLIERS, not a smoothly
    varying noise scale). Fitting the undivided residual sidesteps this
    entirely. The R=residual/dt representation is still what gets
    PLOTTED (see the panel below) -- it's the quantity with the direct
    theoretical interpretation (-> eps as dt->0) -- just not what the
    regression itself is run against.

    eps, eps' (BOTH the same physical quantity -- z1's own error --
    regardless of which model it's measured through) and A (a property
    of z0's own TRUE dynamics, likewise independent of which
    approximation is being compared against it) are fit JOINTLY,
    constrained EQUAL across the euler-only and full residuals, in a
    SINGLE regression over both datasets stacked together -- not two
    independent per-model fits. An earlier version of this function did
    fit them independently, and reported eps'_euler and eps'_full
    differing by ~1.8x despite representing the literal same underlying
    quantity in both models: that's not evidence eps' is unstable in
    reality, it's evidence that estimating the SAME parameter twice,
    independently, from two correlated-but-distinct fits, needlessly
    discards the constraint that they must agree -- each fit only gets
    to use HALF the available data to pin down a parameter both halves
    actually inform, and near-degenerate basis functions (1/dt-ish and
    dt^2-ish terms trading off against each other under noise) are then
    free to resolve that ambiguity differently in each independent fit.
    The joint fit uses ALL the data for eps/eps'/A (both residual types
    inform the same three shared parameters at once) and only lets the
    dt^2-coefficient itself differ between euler-only (C, ~ -z0_ddot/2)
    and full (D, ~ (f_theta-z0_ddot)/2) -- the one place they SHOULD
    differ, since that's exactly where f_theta's own contribution
    enters.

    dt is rescaled by its own geometric mean before building the
    polynomial design matrix (converted back to physical units in the
    returned/printed coefficients) -- raw dt spans several orders of
    magnitude in this project's own data, and unscaled dt/dt^2/dt^3
    columns in the SAME design matrix then differ from each other by
    many further orders of magnitude on top of that: a classical
    ill-conditioning trap for polynomial regression, and a second,
    independent likely contributor (alongside the fully-independent-fit
    issue above) to that same eps' instability.

    Returns a dict with the joint fit's own coefficients+stderrs, the
    EARLIER independent-fit numbers too (kept as a diagnostic showing
    the disagreement the joint fit resolves, not as the recommended
    estimate), and the two derived quantities: mean_z0_ddot (from C)
    and mean_f_theta_minus_z0_ddot (from D) -- f_theta's own average
    signed bias relative to the true curvature, directly.

    label: purely cosmetic -- included in the printed report's own
    header (e.g. "T < 0.9 SUBSET") so console output from multiple
    calls (e.g. comparing a temperature-restricted subset against the
    full dataset, to check whether a specific region of parameter space
    is driving a large stderr) doesn't read as one undifferentiated
    block.

    euler_only: False (default) does the full joint fit described
    above. True skips it entirely and fits a single, simpler 4-term
    model (eps, eps', C, -A -- no D, no joint stacking, no independent-
    fit diagnostic) against euler_losses_signed ALONE -- for exactly
    the situation where latent_losses_signed IS euler_losses_signed
    (the same array passed twice; see check_parameter_dependence's own
    euler_only substitution), where the joint/independent-fit machinery
    above would otherwise silently fit a perfectly degenerate model
    (D == C by construction, mean_f_theta_minus_z0_ddot == 0.0 always)
    and print it as if it were a real finding.
    """
    n = len(dts)
    if n < 50:
        print(f"\n  WARNING: fit_taylor_residual_coefficients called with only {n} windows"
              f"{f' [{label}]' if label else ''} -- the joint model has 5 free parameters "
              f"shared across 2*{n} stacked rows; a small subset can make coefficients "
              f"(especially C/D, the dt^2 terms) genuinely poorly determined rather than "
              f"revealing a real difference from the full-data fit. Treat stderr-vs-estimate "
              f"comparisons on a small subset with real caution.")
    # Geometric mean -- appropriate for log-uniformly-sampled dt (this
    # project's own dt distribution spans decades, not a linear range),
    # matching how the rest of this module already treats dt on a log
    # axis everywhere else.
    dt_scale = float(np.exp(np.mean(np.log(dts))))
    u = dts / dt_scale
    basis_funcs = [lambda uu: np.ones_like(uu), lambda uu: uu, lambda uu: uu ** 2, lambda uu: uu ** 3]
    unscale4 = np.array([1.0, dt_scale, dt_scale ** 2, dt_scale ** 3])

    if euler_only:
        # Single 4-term fit against euler_losses_signed ALONE -- no
        # joint stacking, no independent-fit diagnostic (there's only
        # one model here, nothing to compare against or reconcile). See
        # this function's own docstring for why this branch exists
        # instead of just letting the joint fit below run on
        # euler_losses_signed passed in twice.
        coefs_u, stderr_u = robust_polynomial_fit(u, euler_losses_signed, basis_funcs,
                                                    n_iter=n_iter, huber_delta_scale=huber_delta_scale)
        coefs_phys = coefs_u / unscale4
        stderr_phys = stderr_u / unscale4
        eps, eps_prime, C, neg_A = coefs_phys
        eps_se, eps_prime_se, C_se, neg_A_se = stderr_phys
        mean_z0_ddot = -2 * C
        A = -neg_A
        param_names = ["eps", "eps'", "C (dt^2, ~-z0_ddot/2)", "-A (dt^3)"]

        result = {
            "param_names": param_names, "euler_only": True,
            "joint_coefs": coefs_phys, "joint_stderr": stderr_phys,
            "eps": eps, "eps_stderr": eps_se,
            "eps_prime": eps_prime, "eps_prime_stderr": eps_prime_se,
            "C": C, "C_stderr": C_se, "A": A, "A_stderr": neg_A_se,
            "mean_z0_ddot": mean_z0_ddot,
            "dt_scale": dt_scale,
        }
        print("\n" + "=" * 70)
        print(f"Taylor-residual coefficient fit (euler-only)" + (f"  [{label}]" if label else ""))
        print("=" * 70)
        for name, c, se in zip(param_names, coefs_phys, stderr_phys):
            print(f"    {name:<26} = {c: .6e}  (stderr {se:.2e})")
        print(f"\n  derived quantity:")
        print(f"    mean(z0_ddot) [true curvature, from C] = {mean_z0_ddot:.4e}")
        print("=" * 70)
        return result

    # ---- Independent per-model fits (diagnostic only -- see docstring
    # for why these are NOT the recommended estimate, kept here purely
    # to report the disagreement the joint fit below resolves). Fit
    # against the UNDIVIDED residual now too (basis [1, dt, dt^2, dt^3]
    # in RESCALED u, not [1/dt, 1, dt, dt^2] against R=residual/dt as
    # an earlier version of this function did) -- both the
    # heteroscedasticity and conditioning fixes apply here as much as
    # to the joint fit.
    indep_euler_u, indep_euler_se_u = robust_polynomial_fit(u, euler_losses_signed, basis_funcs,
                                                              n_iter=n_iter, huber_delta_scale=huber_delta_scale)
    indep_full_u, indep_full_se_u = robust_polynomial_fit(u, latent_losses_signed, basis_funcs,
                                                            n_iter=n_iter, huber_delta_scale=huber_delta_scale)
    indep_euler = indep_euler_u / unscale4
    indep_full = indep_full_u / unscale4
    eps_euler_indep, eps_prime_euler_indep = indep_euler[0], indep_euler[1]
    eps_full_indep, eps_prime_full_indep = indep_full[0], indep_full[1]

    # ---- Joint fit: eps, eps', A shared; only the dt^2 coefficient
    # (C vs D) differs by residual type. Stacked design: row 0..n-1 are
    # euler-only residuals, row n..2n-1 are full residuals.
    y = np.concatenate([euler_losses_signed, latent_losses_signed])
    u_stack = np.concatenate([u, u])
    is_euler = np.concatenate([np.ones(n, dtype=bool), np.zeros(n, dtype=bool)])
    X = np.column_stack([
        np.ones_like(u_stack),                    # eps            (shared)
        u_stack,                                   # eps'           (shared)
        np.where(is_euler, u_stack ** 2, 0.0),      # C (euler-only dt^2 coefficient)
        np.where(~is_euler, u_stack ** 2, 0.0),     # D (full-only dt^2 coefficient)
        u_stack ** 3,                               # -A             (shared)
    ])
    param_names = ["eps", "eps'", "C (euler dt^2, ~-z0_ddot/2)", "D (full dt^2, ~(f_theta-z0_ddot)/2)", "-A (dt^3)"]

    weights = np.ones(len(y))

    def _weighted_lstsq(w):
        sqrt_w = np.sqrt(w)
        coefs, *_ = np.linalg.lstsq(X * sqrt_w[:, None], y * sqrt_w, rcond=None)
        return coefs

    coefs = _weighted_lstsq(weights)
    for _ in range(n_iter):
        residuals = y - X @ coefs
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = 1.4826 * mad if mad > 0 else np.std(residuals) + 1e-12
        huber_delta = huber_delta_scale * scale
        abs_resid = np.abs(residuals)
        weights = np.where(abs_resid <= huber_delta, 1.0, huber_delta / np.maximum(abs_resid, 1e-12))
        coefs = _weighted_lstsq(weights)

    residuals = y - X @ coefs
    XtWX = X.T @ (weights[:, None] * X)
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = np.sum(weights * residuals ** 2) / dof
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
        coef_stderr_u = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        coef_stderr_u = np.full(X.shape[1], np.nan)

    unscale5 = np.array([1.0, dt_scale, dt_scale ** 2, dt_scale ** 2, dt_scale ** 3])
    coefs_phys = coefs / unscale5
    stderr_phys = coef_stderr_u / unscale5
    eps, eps_prime, C, D, neg_A = coefs_phys
    eps_se, eps_prime_se, C_se, D_se, neg_A_se = stderr_phys

    mean_z0_ddot = -2 * C
    mean_f_theta_minus_z0_ddot = 2 * D
    A = -neg_A

    result = {
        "param_names": param_names,
        "joint_coefs": coefs_phys, "joint_stderr": stderr_phys,
        "eps": eps, "eps_stderr": eps_se,
        "eps_prime": eps_prime, "eps_prime_stderr": eps_prime_se,
        "C": C, "C_stderr": C_se, "D": D, "D_stderr": D_se,
        "A": A, "A_stderr": neg_A_se,
        "mean_z0_ddot": mean_z0_ddot, "mean_f_theta_minus_z0_ddot": mean_f_theta_minus_z0_ddot,
        "dt_scale": dt_scale,
        # kept as a diagnostic, not the recommended estimate -- see docstring
        "independent_euler_coefs": indep_euler, "independent_full_coefs": indep_full,
    }

    print("\n" + "=" * 70)
    print("Taylor-residual coefficient decomposition" + (f"  [{label}]" if label else ""))
    print("(joint fit: eps, eps', A shared across euler-only and full residuals)")
    print("=" * 70)
    print("  independent per-model fits (DIAGNOSTIC ONLY -- see this function's own "
          "docstring for why these should NOT be trusted as the final estimate):")
    print(f"    eps_euler={eps_euler_indep:.4e}   eps_full={eps_full_indep:.4e}   "
          f"|diff|={abs(eps_euler_indep - eps_full_indep):.4e}")
    print(f"    eps'_euler={eps_prime_euler_indep:.4e}  eps'_full={eps_prime_full_indep:.4e}  "
          f"|diff|={abs(eps_prime_euler_indep - eps_prime_full_indep):.4e}")
    print("\n  joint fit (recommended):")
    for name, c, se in zip(param_names, coefs_phys, stderr_phys):
        print(f"    {name:<38} = {c: .6e}  (stderr {se:.2e})")
    print(f"\n  derived quantities:")
    print(f"    mean(z0_ddot) [true curvature, from C]              = {mean_z0_ddot:.4e}")
    print(f"    mean(f_theta - z0_ddot) [f_theta's own signed bias] = {mean_f_theta_minus_z0_ddot:.4e}")
    if mean_z0_ddot != 0:
        rel_bias = mean_f_theta_minus_z0_ddot / abs(mean_z0_ddot)
        print(f"    relative bias = mean(f_theta-z0_ddot) / |mean(z0_ddot)| = {rel_bias:.2%}")
    print("=" * 70)
    return result


def fit_power_law(dt: np.ndarray, error: np.ndarray):
    """
    log(error) = a*log(dt) + b via least squares. Returns (a, b, r2_log,
    sse_real, pred_real) -- sse_real is the fit's error IN REAL (non-log)
    space, so it can be compared directly against fit_saturating_exponential's
    sse, which is fit in real space to begin with. Comparing R^2 values
    computed in DIFFERENT spaces (log vs real) would not be a fair comparison.
    """
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


def fit_exponential(x: np.ndarray, error: np.ndarray):
    """
    log(error) = a*x + b via least squares (x itself, NOT log(x)) --
    i.e. error = exp(b) * exp(a*x). The semi-log analogue of
    fit_power_law: appropriate for a panel with a LINEAR x-axis and
    log-scaled error axis (like length_scale's), where fit_power_law's
    form would plot as a curve rather than a straight line and so
    wouldn't give the same at-a-glance visual fit-quality check that it
    does on a genuinely log-log panel (like dt's). Same
    sse_real/pred_real convention as fit_power_law, for direct SSE
    comparison against fit_saturating_exponential.
    """
    log_err = np.log(np.clip(error, 1e-12, None))
    a, b = np.polyfit(x, log_err, 1)
    pred_log = a * x + b
    ss_res_log = np.sum((log_err - pred_log) ** 2)
    ss_tot_log = np.sum((log_err - log_err.mean()) ** 2)
    r2_log = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else float("nan")
    pred_real = np.exp(pred_log)
    sse_real = np.sum((error - pred_real) ** 2)
    return a, b, r2_log, sse_real, pred_real


def fit_saturating_exponential(dt: np.ndarray, error: np.ndarray, n_grid: int = 200):
    """
    error = c*(1 - exp(-dt/tau)) -- a smooth, fully DETERMINISTIC
    relaxation toward an asymptote c, with timescale tau. This is a
    genuinely different mechanism from "error grows without bound" or
    "irreducible unpredictability": it's ordinary exponential relaxation,
    which can look deceptively like a decelerating power law in a log-log
    plot over a limited dt range -- exactly why this needs an explicit
    fit-and-compare rather than eyeballing curvature in binned means.

    Fit via a tau grid search (log-spaced across the observed dt range)
    with closed-form c at each tau -- error is LINEAR in c for fixed tau,
    so c has a direct least-squares solution, avoiding a scipy dependency.
    """
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


def _mean_curves_by_unique_value(x_values: np.ndarray, y_signed: np.ndarray, y_abs: np.ndarray,
                                  round_decimals: int = 6):
    """
    For DISCRETE x (temperature/noise -- a handful of fixed sweep
    values, not a continuous range): groups by each unique (rounded)
    x value -- same rounding convention as _boxplot_by_x, for the same
    reason (float round-trip through a text metadata file can turn one
    intended sweep value into many bit-distinct floats) -- and returns
    (sorted unique x values, mean(y_signed) per value, mean(y_abs) per
    value, window count per value). Two curves instead of
    _boxplot_by_x's full per-value distribution: simpler to read at a
    glance, at the cost of not showing spread -- an intentional trade a
    boxplot-per-value grid doesn't make.
    """
    rounded = np.round(x_values, round_decimals)
    unique_x = np.unique(rounded)
    masks = [rounded == v for v in unique_x]
    mean_signed = np.array([y_signed[m].mean() for m in masks])
    mean_abs = np.array([y_abs[m].mean() for m in masks])
    n_windows = np.array([int(m.sum()) for m in masks])
    return unique_x, mean_signed, mean_abs, n_windows


def _mean_curves_by_bin(x_values: np.ndarray, y_signed: np.ndarray, y_abs: np.ndarray,
                         n_bins: int = 8, log_bins: bool = False):
    """
    For CONTINUOUS x (length_scale -- computed per window, a different
    value nearly every time; or dt, which additionally spans several
    orders of magnitude): n_bins bins -- equal-width in LINEAR space by
    default (matching _print_binned_summary's own console table, for
    temperature/noise-like ranges), or equal-width in LOG space when
    log_bins=True (appropriate for dt, which is naturally log-uniformly
    sampled -- see dt_scale's own geometric-mean choice in
    fit_taylor_residual_coefficients for the same reasoning). Returns
    (bin centers, mean(y_signed) per bin, mean(y_abs) per bin, window
    count per bin), skipping any bin with no points in it. Bin centers
    are geometric-mean centers under log_bins=True (consistent with the
    log-spaced edges), arithmetic-mean otherwise.
    """
    if log_bins:
        edges = np.geomspace(x_values.min(), x_values.max(), n_bins + 1)
    else:
        edges = np.linspace(x_values.min(), x_values.max(), n_bins + 1)
    centers, mean_signed, mean_abs, n_windows = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (x_values >= lo) & (x_values <= hi if i == n_bins - 1 else x_values < hi)
        if mask.sum() == 0:
            continue
        centers.append(np.sqrt(lo * hi) if log_bins else (lo + hi) / 2)
        mean_signed.append(y_signed[mask].mean())
        mean_abs.append(y_abs[mask].mean())
        n_windows.append(int(mask.sum()))
    return np.array(centers), np.array(mean_signed), np.array(mean_abs), np.array(n_windows)


def _size_by_count(n_windows: np.ndarray, min_size: float = 20.0, max_size: float = 150.0) -> np.ndarray:
    """
    Marker AREA (matplotlib scatter's own s= convention -- area, not
    diameter; see [0,2]'s own comment on this) scaled linearly between
    min_size and max_size across THIS call's own observed range of
    window counts. Proportional in spirit to [0,2]'s own window-count
    sizing, but re-normalized per panel here rather than reusing that
    panel's literal additive formula (30 + 10*n) -- these panels' window
    counts (pooled per temperature/noise VALUE, or per length_scale
    BIN) are typically on a very different scale than [0,2]'s own
    (pooled per sweep point), and reusing a formula tuned for one scale
    on the other would produce either imperceptibly-similar dots or
    absurdly oversized ones.
    """
    n_windows = np.asarray(n_windows, dtype=float)
    lo, hi = n_windows.min(), n_windows.max()
    if hi <= lo:
        return np.full_like(n_windows, (min_size + max_size) / 2)
    return min_size + (max_size - min_size) * (n_windows - lo) / (hi - lo)


def _symmetric_left_zero_right_ylim(left_axes, right_axes):
    """
    For a group of twin-axis panels sharing one signed quantity (left,
    "mean(error)"-style) and one non-negative quantity (right,
    "mean|error|"-style): computes ONE shared y-range per side, from
    each axis's own current (already-autoscaled) limits.

    Left: symmetric about 0 -- (-a, +a), a = the largest magnitude seen
    across every left axis in the group. Not just "start below the most
    negative point" -- a symmetric range makes it possible to judge "is
    this curve mostly positive or mostly negative" at a glance (equal
    visual weight either side of the y=0 reference line), which an
    asymmetric range skews toward whichever sign happens to have the
    larger excursion.

    Right: floored at 0 (never negative -- this axis's own quantity
    never is either), extending up to the largest value actually seen.

    Returns (left_ylim, right_ylim); does not itself call set_ylim --
    the caller applies these to every axis in the group.
    """
    left_los, left_his = zip(*(ax.get_ylim() for ax in left_axes))
    right_los, right_his = zip(*(ax.get_ylim() for ax in right_axes))
    a = max(abs(min(left_los)), abs(max(left_his)))
    left_ylim = (-a, a)
    right_ylim = (0.0, max(right_his))
    return left_ylim, right_ylim

def _aggregate_per_point(temperatures, noises, latent_losses):
    """
    Extracted verbatim from check_parameter_dependence()'s own
    per-(temperature, noise) aggregation (panel [0,2]) -- keyed by
    (rounded temperature, rounded noise), NOT run_dir, specifically so
    multiple SEEDS sharing one sweep point pool into a single bubble
    rather than each getting its own overlapping one. See that
    function's own inline comment for the full rationale.
    """
    per_point = {}
    for t, n, ll in zip(temperatures, noises, latent_losses):
        key = (round(float(t), 6), round(float(n), 6))
        entry = per_point.setdefault(key, {"temperature": key[0], "noise": key[1], "losses": []})
        entry["losses"].append(ll)
    return per_point


def _aggregate_per_run(run_dirs, temperatures, noises, latent_losses):
    """
    Extracted verbatim from check_parameter_dependence()'s own per-RUN
    aggregation -- kept SEPARATE from _aggregate_per_point above,
    specifically for the "which specific run/seed performs worst"
    console report, which needs each seed identified individually.
    """
    per_run = {}
    for run_dir, t, n, ll in zip(run_dirs, temperatures, noises, latent_losses):
        entry = per_run.setdefault(run_dir, {"temperature": t, "noise": n, "losses": []})
        entry["losses"].append(ll)
    return per_run


# ---------------------------------------------------------------------
# max_autocorr_dist -- unchanged since the previous version of this file
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
# fit_power_law / fit_exponential / fit_saturating_exponential --
# unchanged since the previous version of this file
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
# robust_polynomial_fit -- replaces the previous version's
# robust_linear_fit tests (that function no longer exists: superseded
# by this one, which generalizes it to an arbitrary basis, needed for
# fit_taylor_residual_coefficients's own 1/dt term)
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
# fit_taylor_residual_coefficients -- new since the previous version of
# this file
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
# _mean_curves_by_unique_value / _mean_curves_by_bin -- new since the
# previous version of this file (replace the old, plain
# _make_bin_masks helper -- panels [1,0]/[1,1] no longer bin
# temperature/noise at all, they group by exact unique value instead)
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
# _size_by_count -- new since the previous version of this file
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


# ---------------------------------------------------------------------
# _symmetric_left_zero_right_ylim -- new since the previous version of
# this file
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
# per-(temperature, noise) vs per-run aggregation -- panel [0,2] used
# to be keyed by run_dir, which meant several SEEDS sharing one
# (temperature, noise) sweep point each got their own overlapping
# bubble at the same (x, y) location. This is the single most important
# regression test in this file: it locks in the actual bug fix, not
# just the surrounding machinery.
# ---------------------------------------------------------------------

def test_per_point_aggregation_pools_across_seeds():
    from pathlib import Path
    # Three runs: two are DIFFERENT SEEDS at the SAME (temperature,
    # noise) sweep point; one is a genuinely different sweep point.
    run_dirs = [Path("T900_n020_s79"), Path("T900_n020_s79"),
                Path("T900_n020_s599"), Path("T900_n020_s599"),
                Path("T850_n020_s79")]
    temperatures = [0.9, 0.9, 0.9, 0.9, 0.85]
    noises = [0.02, 0.02, 0.02, 0.02, 0.02]
    latent_losses = [1.0, 2.0, 3.0, 4.0, 10.0]  # seeds s79/s599 together: mean([1,2,3,4])=2.5

    per_point = _aggregate_per_point(temperatures, noises, latent_losses)

    # ONE entry for (0.9, 0.02) pooling BOTH seeds' windows together,
    # not two separate entries that would show up as two overlapping
    # bubbles at the identical (x, y) location.
    assert len(per_point) == 2
    key_pooled = (0.9, 0.02)
    assert key_pooled in per_point
    assert len(per_point[key_pooled]["losses"]) == 4  # all 4 windows from BOTH seeds
    assert np.mean(per_point[key_pooled]["losses"]) == pytest.approx(2.5)


def test_per_run_aggregation_keeps_seeds_separate():
    """The OTHER aggregation (kept deliberately separate from
    per_point above) -- for the "which specific run/seed performs
    worst" report, which needs seeds identified individually, unlike
    panel [0,2]'s own pooled view. Same input as the pooling test
    above, opposite expectation."""
    from pathlib import Path
    run_dirs = [Path("T900_n020_s79"), Path("T900_n020_s79"),
                Path("T900_n020_s599"), Path("T900_n020_s599"),
                Path("T850_n020_s79")]
    temperatures = [0.9, 0.9, 0.9, 0.9, 0.85]
    noises = [0.02, 0.02, 0.02, 0.02, 0.02]
    latent_losses = [1.0, 2.0, 3.0, 4.0, 10.0]

    per_run = _aggregate_per_run(run_dirs, temperatures, noises, latent_losses)

    assert len(per_run) == 3  # three DISTINCT run_dirs, seeds NOT pooled
    assert np.mean(per_run[Path("T900_n020_s79")]["losses"]) == pytest.approx(1.5)
    assert np.mean(per_run[Path("T900_n020_s599")]["losses"]) == pytest.approx(3.5)
    assert np.mean(per_run[Path("T850_n020_s79")]["losses"]) == pytest.approx(10.0)


def test_per_point_aggregation_merges_near_duplicate_floats():
    """Same float round-trip concern as _mean_curves_by_unique_value --
    two runs meant to be the SAME sweep point, differing only in the
    17th decimal digit after a metadata-file round-trip, must still
    pool into one bubble."""
    from pathlib import Path
    run_dirs = [Path("runA"), Path("runB")]
    temperatures = [0.9, 0.9000000001]
    noises = [0.02, 0.0199999998]
    latent_losses = [1.0, 3.0]

    per_point = _aggregate_per_point(temperatures, noises, latent_losses)
    assert len(per_point) == 1
    assert len(list(per_point.values())[0]["losses"]) == 2
