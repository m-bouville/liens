"""
Pure-numeric fitting helpers extracted from check_parameter_dependence.py
(which had grown to 2566 lines, 42% of evaluation/'s total). Depends on
numpy ALONE -- no torch, no dataset, no checkpoint -- so it is trivially
unit-testable and reusable by any other diagnostic.

Extracted specifically because check_deriv_temperature.py had grown its
OWN second copy of the same robust-IRLS weighted-least-squares algorithm,
and the two then fitted the same underlying residual against DIFFERENT
bases ([1/dt, 1] vs [1, u, u^2, u^3]) while reporting the shared
coefficients under the same names, eps and eps'. On one real checkpoint
that produced eps' = +2.46e-05 from one script and -3.87e-06 from the
other -- the same quantity, same data, incomparable numbers. One shared
fitter and one declared basis is the fix; this module is where it lives.
"""
import numpy as np


def _stderr_from_normal_equations(XtWX: np.ndarray, sigma2: float, n_params: int,
                                   label: str = "") -> np.ndarray:
    """Coefficient standard errors from a weighted normal-equations matrix.

    Shared by BOTH robust fits in this module. They had independent copies of
    this block and only one was ever hardened -- the second still emitted
    "RuntimeWarning: invalid value encountered in sqrt" from the joint Taylor
    fit. Same defect, same file, one fix away from each other.

    Two failure modes, and the obvious try/except only catches the rarer one:

      * EXACTLY singular -> np.linalg.inv raises. pinv instead: the least-norm
        solution's covariance is finite and honestly enormous along the
        unidentifiable directions, which says more than an all-NaN row.
      * NEAR singular -> inv() SUCCEEDS and returns a matrix that is not
        positive semi-definite, so a variance comes back negative. sqrt then
        warns and yields NaN: an error bar that quietly vanishes from the
        report while the coefficient beside it still looks authoritative.

    Checked explicitly so a NaN is a decision rather than a by-product, and so
    the condition number is reported -- the usual cause is a dt range too
    narrow to separate the basis terms.
    """
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(XtWX)
    variances = np.diag(cov)
    stderr = np.where(variances >= 0, np.sqrt(np.abs(variances)), np.nan)
    if not np.all(np.isfinite(stderr)):
        where = f" [{label}]" if label else ""
        print(f"    NOTE{where}: {int(np.sum(~np.isfinite(stderr)))}/{n_params} coefficient "
              f"standard error(s) are undefined -- the weighted design matrix is "
              f"near-singular (condition number {np.linalg.cond(XtWX):.2e}). The "
              f"COEFFICIENTS are still the least-squares solution, but nothing here says "
              f"how well determined they are; usually the dt range is too narrow to "
              f"separate the basis terms.")
    return stderr


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
    coef_stderr = _stderr_from_normal_equations(XtWX, sigma2, p)
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
    coef_stderr_u = _stderr_from_normal_equations(XtWX, sigma2, X.shape[1],
                                                   label=label or "joint fit")

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




def fit_broken_power_law(x: np.ndarray, y: np.ndarray, n_candidates: int = 60,
                          min_side: int = 8, min_side_fraction: float = 0.15):
    """Continuous two-slope fit in log-log: y ~ x^p1 below x_knee, x^p2 above.

    WHY NOT A SPLINE, which is the obvious reach for "flat then sloped". A
    spline is a smoothing device: it would render the bend faithfully and
    report nothing about it, leaving "there seems to be a regime change" a
    visual impression. The bend is the physics -- delta_t/t against t is the
    question of whether the step follows coarsening's own slowdown -- so the
    fit should NAME the knee and the two exponents rather than draw through
    them. It also cannot wiggle: a spline through noisy binned medians
    invents structure at the ends, exactly where the data is thinnest.

    Continuity is imposed rather than fitted (two independent segments would
    jump at the knee, which no physical crossover does), via the hinge basis
    [1, u, max(u - k, 0)] with u = log x. The second coefficient is then the
    CHANGE in slope, so p2 = p1 + c2 by construction.

    The knee is chosen by exhaustive search over candidate positions rather
    than by optimisation: the SSE-vs-knee curve is not convex and a local
    method lands wherever it starts.

    Each side must hold at least `min_side` points AND `min_side_fraction` of
    them. The absolute floor alone is not enough: at 1800 points it let a
    "regime" be eight of them, and on real data that produced a first segment
    of slope +3.76 -- fitted to a handful of early windows -- with a 30%
    apparent improvement, which is exactly the spurious knee a reader would
    have believed. A regime that covers under a sixth of the range is a tail
    artefact, whatever it does to the SSE.

    Returns (x_knee, p1, p2, sse_broken, sse_single). Comparing the two SSEs
    is how the caller decides whether the bend is worth reporting at all --
    a broken fit has two extra parameters and will always fit at least as
    well, so a marginal improvement means "one power law, drawn on noisy
    data", not "a regime change".
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    u, v = np.log(x[ok]), np.log(y[ok])
    min_side = max(min_side, int(np.ceil(min_side_fraction * u.size)))
    if u.size < 2 * min_side + 1:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    single = np.polyfit(u, v, 1)
    sse_single = float(np.sum((v - np.polyval(single, u)) ** 2))

    order = np.argsort(u)
    u, v = u[order], v[order]
    lo, hi = u[min_side], u[-min_side - 1]
    if not (hi > lo):
        return float("nan"), float(single[0]), float(single[0]), sse_single, sse_single

    best = None
    for k in np.linspace(lo, hi, n_candidates):
        basis = np.column_stack([np.ones_like(u), u, np.maximum(u - k, 0.0)])
        coef, *_ = np.linalg.lstsq(basis, v, rcond=None)
        sse = float(np.sum((v - basis @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, k, coef)
    sse, k, coef = best
    return float(np.exp(k)), float(coef[1]), float(coef[1] + coef[2]), sse, sse_single
