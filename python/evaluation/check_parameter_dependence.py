"""
Scatter one-step LDS prediction error against dt, temperature, and
noise, across every window in the test set (not just a handful) -- to
check whether error systematically depends on any of these (as a few
examples in check_rollout.py suggested for dt), and specifically to
help decide WHERE in (temperature, noise) space more simulation data
would help most. Motivation: a rare, hard case like hourglass-shaped
grain snapping is exactly the kind of thing under-represented in some
region of parameter space, not something more training epochs would
fix -- if error is concentrated in an identifiable (temperature, noise)
region, that's a direct, actionable signal for where to run more
simulations, rather than a diffuse "the model needs to be better"
conclusion.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_parameter_dependence \
        --lds-checkpoint ../checkpoints/stage3/64x64.pt
"""

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.losses import centered_deriv_target

# GENERAL POLICY (matches training/train_refinement.py's own
# _PYTHON_ROOT): every default checkpoint/output path is built from
# THIS anchor, never from a bare relative string like "../../output/...".
# Relative strings resolve against the process's CWD at invocation
# time, which silently differs across bare CLI, `python -m`, and being
# imported and called from another module (e.g. main.py calling this
# function) -- exactly the recurring "output ended up in the wrong
# place" bug hit repeatedly on this project. Path(__file__) is anchored
# to THIS FILE's own on-disk location instead, which is invariant
# regardless of how/from-where the process was launched.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/


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


from evaluation._fits import (
    fit_exponential, fit_power_law, fit_saturating_exponential,
    fit_taylor_residual_coefficients,
)
from evaluation._latent_eval import (
    _DerivedStats, _EvaluationResults, _evaluate_windows, _load_models_and_dataset,
)

def _print_oracle_z1_attribution(dataset, results, device, max_dist: int | None = None):
    """
    THE stage-1-vs-stage-2 attribution test: is the euler-only error a
    property of z1 (stage 2's own output), or is it already baked into
    z0's own trajectory (stage 1) before z1 is even consulted?

    Method -- substitute an ORACLE z1 and re-measure:

        err_actual = || z0(t) + z1(t)*dt      - z0(t+dt) ||
        err_oracle = || z0(t) + z0_dot_c*dt   - z0(t+dt) ||

    where z0_dot_c is the second-order-accurate CENTERED derivative of
    z0's OWN trajectory at t (see losses.centered_deriv_target), built
    from z0(t-dt_minus), z0(t), z0(t+dt_plus). That is the best
    first-order-update accuracy obtainable from z0's trajectory at all:
    what remains is pure z0_ddot*dt^2/2 truncation, PLUS whatever noise
    z0's own encoding carries (the centered estimate inherits it).

    Reading the result:
      err_oracle ~= err_actual  -> z1 is already as good as z0's own
        trajectory permits. The floor lives in STAGE 1 (z0), and
        further stage-2 work on z1 is capped -- no z1 could do much
        better against this z0.
      err_oracle << err_actual  -> z1 is leaving real accuracy on the
        table that z0's trajectory would have supported. The problem is
        STAGE 2, and improving z1 is worth doing.

    Costs no extra encoding: this dataset is in cached-latent mode, so
    every run's full latent sequence is already resident. The oracle
    needs one frame BEFORE each window's own start, which exists for
    every window except those starting at a run's first kept step --
    those are reported and skipped, not silently dropped.

    Deliberately reported per dt decade as well as overall, for the
    same reason the saturation cross-tab is: error tracks log dt at
    ~94% here, so any aggregate comparison that doesn't hold dt roughly
    fixed mostly measures the dt distribution of whichever subset
    happens to be included.
    """
    err_actual, err_oracle, err_causal, dts_ok = [], [], [], []
    n_skipped = 0
    # NaN-padded to len(dataset._index) so these align 1:1 with every
    # results.* array (same dataset order, loader is unshuffled) -- the
    # figure code needs that alignment; the console summary below only
    # needs the compacted lists. NaN marks "no frame before this window".
    n_all = len(dataset._index)
    per_window = {k: np.full(n_all, np.nan, dtype=float)
                  for k in ("causal_abs", "causal_signed", "oracle_abs", "oracle_signed")}
    with torch.no_grad():
        for w_idx, (run_idx, start) in enumerate(dataset._index):
            if start == 0:  # no frame before this window -- no centered estimate possible
                n_skipped += 1
                continue
            run_state = dataset._run_data[run_idx]
            run_deriv = dataset._run_data_deriv[run_idx]
            steps = dataset._run_steps[run_idx]
            scale = dataset._run_dt_scale[run_idx]

            z0_before, z0_t, z0_next = run_state[start - 1], run_state[start], run_state[start + 1]
            z1_t = run_deriv[start]
            dt_minus = (steps[start] - steps[start - 1]) * scale
            dt_plus = (steps[start + 1] - steps[start]) * scale
            if dt_minus <= 0 or dt_plus <= 0:
                n_skipped += 1
                continue

            dtm = torch.tensor(dt_minus, dtype=z0_t.dtype)
            dtp = torch.tensor(dt_plus, dtype=z0_t.dtype)
            z0_dot_c = centered_deriv_target(z0_before, z0_t, z0_next, dtm, dtp)

            # CAUSAL baseline: backward difference, built ONLY from
            # z0(t-dt_minus) and z0(t) -- both available at prediction
            # time. This is what z1 must actually be compared against
            # (see this function's own docstring on the centered
            # oracle's future-information leak).
            z0_dot_back = (z0_t - z0_before) / dt_minus

            resid_causal = z0_t + z0_dot_back * dt_plus - z0_next
            resid_oracle = z0_t + z0_dot_c * dt_plus - z0_next
            err_actual.append((z0_t + z1_t * dt_plus - z0_next).abs().mean().item())
            err_oracle.append(resid_oracle.abs().mean().item())
            err_causal.append(resid_causal.abs().mean().item())
            dts_ok.append(dt_plus)

            per_window["causal_abs"][w_idx] = resid_causal.abs().mean().item()
            per_window["causal_signed"][w_idx] = resid_causal.mean().item()
            per_window["oracle_abs"][w_idx] = resid_oracle.abs().mean().item()
            per_window["oracle_signed"][w_idx] = resid_oracle.mean().item()

    if not err_actual:
        print("\n(oracle-z1 attribution: no window has a preceding frame -- nothing to compare)")
        return per_window

    a = np.array(err_actual); o = np.array(err_oracle); d = np.array(dts_ok)
    c = np.array(err_causal)
    print(f"\n{'='*70}")
    print("Oracle-z1 attribution: is the euler error z1's (stage 2) or z0's (stage 1)?")
    print(f"{'='*70}")
    print(f"  {len(a)} windows usable ({n_skipped} skipped: no frame before the window's own start)")
    print(f"  err_actual (real z1)                       = {a.mean():.6e}")
    print(f"  err_causal (backward dz0/dt, PAST ONLY)    = {c.mean():.6e}   "
          f"ratio {c.mean()/max(a.mean(), 1e-30):.3f}")
    print(f"  err_oracle (centered dz0/dt, SEES FUTURE)  = {o.mean():.6e}   "
          f"ratio {o.mean()/max(a.mean(), 1e-30):.3f}")

    print("\n  per dt decade (error tracks log dt strongly -- aggregates alone mislead):")
    print("  dt decade         n     err_actual     err_causal     err_oracle  causal/act  oracle/act")
    finite = d > 0
    if finite.any():
        for e in range(int(np.floor(np.log10(d[finite].min()))),
                        int(np.ceil(np.log10(d[finite].max())))):
            m = finite & (d >= 10.0**e) & (d < 10.0**(e + 1))
            if not m.any():
                continue
            print(f"  1e{e:<2d}- 1e{e+1:<3d} {m.sum():6d}   {a[m].mean():12.6e}   {c[m].mean():12.6e}   "
                  f"{o[m].mean():12.6e}  {c[m].mean()/max(a[m].mean(), 1e-30):10.3f}  "
                  f"{o[m].mean()/max(a[m].mean(), 1e-30):10.3f}")

    ratio = c.mean() / max(a.mean(), 1e-30)  # CAUSAL ratio -- the fair one
    if ratio > 0.7:
        print("\n  -> oracle is close to actual: z1 is already near the best any first-order")
        print("     update could do against THIS z0. The floor is in STAGE 1 (z0's own")
        print("     trajectory), and further z1/stage-2 work is capped.")
    elif ratio < 0.3:
        print("\n  -> oracle is far below actual: z0's own trajectory supports a much more")
        print("     accurate first-order update than z1 currently delivers. The problem is")
        print("     in STAGE 2 (z1), and improving it is worth doing.")
    else:
        print("\n  -> intermediate: z1 leaves some accuracy on the table, but a real floor")
        print("     from z0's own trajectory remains too. Both stages contribute.")
    print("  (verdict is keyed on err_CAUSAL, not err_oracle: the centered oracle is built")
    print("   from z0(t+dt) and then scored against that same z0(t+dt), so it absorbs part of")
    print("   an error no causal predictor could avoid -- measured at ~50% on a deliberately")
    print("   unpredictable trajectory. It is a smoothness probe of z0, NOT an achievable")
    print("   target for z1, which sees only frame t. Both estimates also inherit z0's own")
    print("   encoding noise, so neither is an absolute floor.)")
    return per_window


def _print_saturation_cross_tab(temperatures: np.ndarray, length_scales: np.ndarray,
                                 abs_steps: np.ndarray, latent_losses: np.ndarray,
                                 dts: np.ndarray, max_dist: int, n_bins: int = 8):
    """
    Saturation fraction of autocorr_length, cross-tabulated against
    temperature -- the direct test of whether this run's own
    "error grows with temperature" finding is really a FINITE-SIZE
    artifact of the domain rather than a property of the model.

    Why this matters, and why temperature specifically: the C++ search
    cap is min(Nx*2/3, Ny*2/3) -- 42 px on a 64x64 domain -- and on
    real 64x64 data ~86% of windows hit it. At the same time the
    encoder's own theoretical receptive field is 43 px and the typical
    feature is >= 42 px, so THREE candidate explanations for
    "error grows with feature size" (receptive field too small,
    measurement saturating, box smaller than the physics) are
    numerically indistinguishable at this one domain size. Since the
    correlation length diverges near T0, all three ALSO predict
    "error grows with temperature" -- so the temperature finding cannot
    be attributed from 64x64 data alone. If saturation turns out to be
    strongly temperature-correlated, the temperature effect and the
    length-scale effect are substantially the SAME effect, and the
    finite-size component of it should largely vanish on a bigger
    domain (the cap scales with the box, while a fixed-dx feature does
    not).

    Saturation is NOT one phenomenon, which is why the "no clear peak"
    column below exists: autocorr_length saturates whenever the
    correlation never decays within range, and that happens both when
    structure is LARGER than the cap AND when there is effectively no
    structure to decorrelate at all -- early, unpatterned noise (which
    persists at ANY domain size) or late, fully-coarsened
    near-single-phase states (which become very unlikely in a larger
    box). Those scale in OPPOSITE directions, so a bare saturation
    fraction can't be read as "features are large". abs_steps is
    reported alongside as the cheapest available discriminator: early
    saturated windows point at the unpatterned-noise cause, late ones
    at the coarsened-out cause, and a saturated population concentrated
    at NEITHER extreme is the genuinely-large-feature case.
    """
    saturated = length_scales >= max_dist
    edges = np.linspace(temperatures.min(), temperatures.max(), n_bins + 1)
    print(f"\nautocorr_length saturation (>= {max_dist} px) vs temperature -- "
          f"finite-size confound check:")
    print("temperature bin          n   sat%   median step (sat)  "
          "mean loss (sat)  mean loss (unsat)")
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (temperatures >= lo) & (temperatures <= hi if i == n_bins - 1 else temperatures < hi)
        if mask.sum() == 0:
            continue
        sat_here = saturated & mask
        uns_here = (~saturated) & mask
        frac = 100.0 * sat_here.sum() / mask.sum()
        med_step = f"{np.median(abs_steps[sat_here]):11.0f}" if sat_here.sum() else f"{'--':>11s}"
        loss_sat = f"{latent_losses[sat_here].mean():13.6f}" if sat_here.sum() else f"{'--':>13s}"
        loss_uns = f"{latent_losses[uns_here].mean():15.6f}" if uns_here.sum() else f"{'--':>15s}"
        print(f"{lo:7.4f} - {hi:7.4f} {mask.sum():5d} {frac:5.1f}  {med_step}      "
              f"{loss_sat}  {loss_uns}")

    if not (saturated.sum() and (~saturated).sum()):
        print("\n(all or no windows saturated -- cross-tab uninformative at this domain size)")
        return

    corr_sat_temp = np.corrcoef(temperatures, saturated.astype(float))[0, 1]
    print(f"\ncorr(temperature, is_saturated) = {corr_sat_temp*100:.0f}%")
    print(f"  overall saturated mean loss = {latent_losses[saturated].mean():.6f}  vs  "
          f"unsaturated = {latent_losses[~saturated].mean():.6f}  "
          f"(x{latent_losses[saturated].mean()/max(latent_losses[~saturated].mean(), 1e-12):.1f})")

    # THE decisive statistic -- not the corr(temperature, is_saturated)
    # above, which turned out to be the least informative number here.
    # What actually settles the attribution is whether the temperature
    # trend survives INSIDE each saturation class. If loss rises with
    # temperature only among SATURATED windows and is flat/falling among
    # unsaturated ones, then "error grows with temperature" is not a
    # temperature property of the model at all -- it's a property of the
    # saturated subpopulation, which is exactly the population a larger
    # domain changes.
    corr_within = {}
    for label, mask in (("saturated", saturated), ("unsaturated", ~saturated)):
        if mask.sum() >= 3 and np.ptp(temperatures[mask]) > 0:
            corr_within[label] = np.corrcoef(temperatures[mask], latent_losses[mask])[0, 1]
    if len(corr_within) == 2:
        cs, cu = corr_within["saturated"], corr_within["unsaturated"]
        print(f"\ncorr(temperature, loss) WITHIN each class:  "
              f"saturated {cs*100:+.0f}%   unsaturated {cu*100:+.0f}%")
        if cs > 0.2 and cu < 0.1:
            print("  -> the temperature trend exists ONLY among saturated windows; among windows "
                  "with genuinely resolvable structure it is absent or reversed. So this run's own "
                  "temperature finding is a property of the SATURATED subpopulation, not of "
                  "temperature per se -- re-check on a larger domain before spending anything on "
                  "temperature-specific modelling changes.")
        elif cu > 0.2:
            print("  -> the temperature trend survives among UNSATURATED windows too, so it is NOT "
                  "purely a saturation/finite-size artifact and is worth treating as real.")

    # The remaining confound: saturated windows here are LATE windows,
    # and late windows have large dt on a geometric save schedule --
    # while error already correlates ~94% with log dt. So a raw
    # saturated-vs-unsaturated loss gap may be mostly a dt gap. Compare
    # them WITHIN each dt decade, where dt is held roughly fixed.
    print("\nsaturated vs unsaturated WITHIN each dt decade (controls for the dt confound --")
    print("late windows have large dt, and error already tracks log dt strongly):")
    print("dt decade        n_sat  n_unsat   mean loss (sat)  mean loss (unsat)   ratio")
    finite = dts > 0
    if finite.sum():
        lo_exp = int(np.floor(np.log10(dts[finite].min())))
        hi_exp = int(np.ceil(np.log10(dts[finite].max())))
        for e in range(lo_exp, hi_exp):
            in_dec = finite & (dts >= 10.0**e) & (dts < 10.0**(e + 1))
            ns, nu = (in_dec & saturated).sum(), (in_dec & ~saturated).sum()
            if ns == 0 and nu == 0:
                continue
            ms = f"{latent_losses[in_dec & saturated].mean():15.6f}" if ns else f"{'--':>15s}"
            mu = f"{latent_losses[in_dec & ~saturated].mean():17.6f}" if nu else f"{'--':>17s}"
            ratio = (f"{latent_losses[in_dec & saturated].mean() / latent_losses[in_dec & ~saturated].mean():7.2f}"
                     if ns and nu and latent_losses[in_dec & ~saturated].mean() > 0 else f"{'--':>7s}")
            print(f"1e{e:<2d}- 1e{e+1:<3d}    {ns:5d}    {nu:5d}   {ms}  {mu} {ratio}")
        print("  (ratios near 1.0 across decades -> the saturated/unsaturated gap was mostly a dt")
        print("   effect; ratios staying well above 1.0 -> saturation carries real extra error)")


def _print_binned_summary(name: str, values: np.ndarray, latent_losses: np.ndarray,
                           pixel_losses: np.ndarray | None = None, n_bins: int = 8):
    """Linear-space binned summary -- unlike dt (which spans orders of
    magnitude and gets log-decade bins below), temperature/noise are
    each a narrow, bounded range in a typical sweep, so linear bins are
    the more natural choice here.

    pixel_losses=None (e.g. when check_parameter_dependence's own
    decode=False) skips that column entirely, rather than requiring
    the caller to always compute pixel-space losses just to print
    this summary."""
    edges = np.linspace(values.min(), values.max(), n_bins + 1)
    header = f"\n{name} bin              n       mean latent_loss"
    header += "   mean pixel_loss" if pixel_losses is not None else ""
    print(header)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (values >= lo) & (values <= hi if i == n_bins - 1 else values < hi)
        if mask.sum() == 0:
            continue
        row = f"{lo:8.4f} - {hi:8.4f}   {mask.sum():4d}   {latent_losses[mask].mean():.6f}"
        row += f"         {pixel_losses[mask].mean():.6f}" if pixel_losses is not None else ""
        print(row)


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
        # geomspace cannot include zero or negative values (no
        # meaningful log) -- excluded here, with a warning, rather than
        # letting np.geomspace raise on them. A real, hit-in-practice
        # case: x_values here is sometimes an absolute simulation step
        # number, and min_step=0 is a legitimate value (e.g.
        # ensure_lds_checkpoint's own fallback default when no min_step
        # is given) -- meaning a step of exactly 0 can genuinely appear
        # if the underlying simulation saved one, silently crashing this
        # function long after the point where the caller could easily
        # tell why.
        _positive = x_values > 0
        if not _positive.all():
            warnings.warn(
                f"_mean_curves_by_bin(log_bins=True): {(~_positive).sum()} of {len(x_values)} "
                f"x_values are <= 0 and have no meaningful log -- excluded from binning "
                f"entirely (their y_signed/y_abs are dropped too, not folded into another bin)."
            )
            x_values, y_signed, y_abs = x_values[_positive], y_signed[_positive], y_abs[_positive]
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


def _boxplot_by_x(ax, x_values: np.ndarray, y_values: np.ndarray, log_x: bool = False,
                   round_decimals: int = 6):
    """
    Boxplot of y_values grouped by each unique value in x_values, with
    boxes positioned at their LITERAL value -- honest, in both the
    log_x=True (dt) and log_x=False (temperature/noise) cases. An
    earlier version of this positioned temperature/noise boxes at
    evenly-spaced CATEGORICAL indices instead, discarding their real
    value spacing entirely to avoid crowding -- that's misleading, not
    a readability trade-off: it makes two boxes close in value and two
    boxes far apart in value look identically spaced, which is a false
    impression of the underlying sweep, not just a cosmetic
    simplification. Reverted.

    dt (log_x=True) was never the problem: its sweep is naturally close
    to geometric (roughly x2 between consecutive values), so an honest
    log axis at literal values already looks close to evenly spaced --
    nothing special is done for it, it just doesn't need anything
    special. temperature/noise are swept roughly linearly, so a linear
    axis at literal values will only look evenly spread if the sweep
    itself actually is -- if it isn't (or if the ACTUAL unique-value
    count is much larger than the intended sweep grid; see
    round_decimals below), no amount of axis trickery makes that
    honest AND uniformly readable at the same time. The two things
    below address the two real, non-misleading ways to reduce crowding:

    1. round_decimals: values are rounded to this many decimals before
       computing uniqueness. Temperature/noise round-trip through a
       text metadata file -- float parsing can turn what's meant to be
       the same nominal sweep value (e.g. 0.55) into many
       bit-distinct floats (0.5500000001, 0.5499999998, ...) across
       different runs, which np.unique would count as genuinely
       different x positions, silently exploding the apparent sweep
       size far beyond the real number of distinct settings. Rounding
       merges those back into one honest value rather than plotting
       dozens of near-duplicate boxes that were never meant to be
       distinguishable. This does NOT round away real distinctions --
       a human-designed sweep grid isn't going to have two genuinely
       different settings within 1e-6 of each other.
    2. Per-box width from that box's own LOCAL neighbor gaps, not a
       single global minimum gap. The old width = min(all gaps) * 0.6
       meant one closely-spaced pair anywhere in the sweep forced every
       box everywhere to be that thin, including ones with plenty of
       room -- using each box's own local spacing gives genuinely
       sparse boxes their actual available width instead.

    Tick labels are still thinned (not every value gets text) once
    there are more than max_labeled_ticks distinct values, since even
    honestly-positioned boxes can have more values than can be legibly
    labeled -- but the BOXES themselves are never merged or
    repositioned, only which ones get a text label underneath.
    """
    x_values = np.round(x_values, round_decimals)
    unique_x = np.unique(x_values)
    groups = [y_values[x_values == x] for x in unique_x]
    print(f"  ({len(unique_x)} distinct x values after rounding to {round_decimals} "
          f"decimals -- if this looks far larger than the intended sweep grid, "
          f"floating-point round-trip noise is the likely cause)")

    if log_x:
        widths = unique_x * 0.15
    else:
        widths = _local_widths(unique_x)

    ax.boxplot(groups, positions=unique_x, widths=widths, showfliers=True,
               patch_artist=True, boxprops=dict(facecolor="tab:blue", alpha=0.4),
               medianprops=dict(color="black"),
               flierprops=dict(markersize=3, alpha=0.3, markeredgecolor="tab:blue"))

    # matplotlib's boxplot() applies its OWN default x margin -- +-0.5
    # around the position range -- regardless of what scale the
    # positions are actually on. That's a reasonable margin for
    # categorical integer positions (0, 1, 2, ...), and a wildly
    # disproportionate one for small real-valued positions: verified
    # numerically for a noise-like sweep spanning [0.005, 0.05] (range
    # 0.045), the default autoscale gives xlim=(-0.495, 0.55) -- a
    # range over 23x wider than the actual data, squeezing every box
    # into a sliver in the middle of mostly blank axis. Overriding xlim
    # explicitly, proportional to the REAL data range, fixes this for
    # any parameter's scale rather than relying on matplotlib's
    # categorical-position assumption.
    if log_x:
        ax.set_xlim(unique_x.min() / 1.5, unique_x.max() * 1.5)
    else:
        data_range = unique_x.max() - unique_x.min() if len(unique_x) > 1 else widths[0]
        pad = max(data_range * 0.1, widths.max() * 0.75)
        ax.set_xlim(unique_x.min() - pad, unique_x.max() + pad)

    if log_x:
        ax.set_xscale("log")
        return

    # Value-evenly-spaced tick TARGETS, nearest-snapped to real data --
    # only decides which of the (honestly, literally positioned) boxes
    # get a text label, never moves or merges a box itself.
    max_labeled_ticks = 15
    if len(unique_x) > max_labeled_ticks:
        targets = np.linspace(unique_x.min(), unique_x.max(), max_labeled_ticks)
        indices = np.clip(np.searchsorted(unique_x, targets), 0, len(unique_x) - 1)
        for i, t in enumerate(targets):
            idx = indices[i]
            if idx > 0 and abs(unique_x[idx - 1] - t) < abs(unique_x[idx] - t):
                indices[i] = idx - 1
        tick_positions = np.unique(unique_x[indices])
    else:
        tick_positions = unique_x
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{v:.3g}" for v in tick_positions], rotation=90, fontsize=8)


def _local_widths(unique_x: np.ndarray, factor: float = 0.6) -> np.ndarray:
    """
    Per-position box width from that position's OWN nearest-neighbor
    gap (left or right, whichever is smaller), not a single global
    min-gap applied everywhere. A box in a sparse region gets a width
    that reflects the room it actually has, instead of being forced
    down to match the tightest pair anywhere else in the sweep.
    """
    if len(unique_x) == 1:
        return np.array([max(abs(unique_x[0]) * 0.1, 1e-3)])
    gaps = np.diff(unique_x)
    left_gap = np.concatenate([[gaps[0]], gaps])
    right_gap = np.concatenate([gaps, [gaps[-1]]])
    return np.minimum(left_gap, right_gap) * factor


def _print_summary_statistics(results: _EvaluationResults, ae_config: dict, decode: bool,
                               euler_only: bool, dataset=None, device=None) -> _DerivedStats:
    """Every console-only diagnostic that doesn't touch a figure object --
    bias/variance, the dt power-law/saturating-exponential model comparison,
    the euler-vs-full direct-magnitude comparison, the dt-decade table,
    temperature/noise/length_scale correlations, and the top-10-runs-by-loss
    listing. Extracted verbatim (see _DerivedStats' own docstring for the one
    added line, a safe default for corr_dt_pixel, that this split required).
    Mostly a side effect (print), but also returns the handful of computed
    values _build_and_save_figures' own panels reuse -- see _DerivedStats."""
    # Bias vs variance in z1's own error: results.euler_losses (E[|residual|],
    # mean ABSOLUTE error) can never distinguish "z1 is wrong by the
    # same amount, in the same direction, every time" (a bias -- in
    # principle correctable, e.g. by retraining z1 differently) from
    # "z1 is wrong by that much, but in a random direction each time"
    # (variance -- an irreducible floor no retraining on the SAME kind
    # of data could remove). |E[residual]| (mean of the SIGNED
    # residual's own norm, NOT mean of the norm) answers this directly:
    # random, cancelling errors average toward zero across many
    # windows; a genuine, consistent bias does not.
    mean_signed_residual = results.signed_residual_sum / results.n_total  # (C, H, W), SIGN preserved
    bias_magnitude = mean_signed_residual.abs().mean().item()
    total_magnitude = float(results.euler_losses.mean())
    bias_fraction = bias_magnitude / total_magnitude if total_magnitude > 0 else float("nan")
    print(f"\nz1's own euler-only error: bias vs variance (n={results.n_total} windows):")
    print(f"  E[|residual|]  (total error magnitude, already reported above as euler-only): "
          f"{total_magnitude:.6e}")
    print(f"  |E[residual]|  (the part that does NOT cancel across windows -- the bias): "
          f"{bias_magnitude:.6e}")
    print(f"  bias fraction = |E[residual]| / E[|residual|] = {bias_fraction:.3f}")
    print(f"  (near 1.0 -> error is mostly a consistent, SYSTEMATIC bias in the same "
          f"direction every time -- in principle correctable by retraining z1 differently. "
          f"near 0.0 -> error is mostly VARIANCE/NOISE, cancelling across windows -- an "
          f"irreducible floor that retraining on the same KIND of data is unlikely to fix.)")

    # ---- dt: unchanged from check_dt_dependence.py -- log-log, with
    # the power-law vs saturating-exponential model comparison. dt
    # genuinely spans orders of magnitude within one sweep; temperature
    # and noise (below) typically don't, so they get linear treatment
    # instead rather than forcing the same log-space analysis on them.
    log_dt = np.log10(results.dts)
    log_latent = np.log10(np.clip(results.latent_losses, 1e-12, None))
    corr_dt_latent = np.corrcoef(log_dt, log_latent)[0, 1]
    # corr_dt_pixel/log_pixel need results.pixel_losses, which is only populated
    # when decode=True (see the main loop's own decoder-skipping note).
    # Defaulted to None here (a NEW line, not present in the original
    # in-one-function version) so _DerivedStats' own return below always
    # has a value to bundle regardless of decode -- the original code never
    # needed this default, since a single function body never references a
    # variable before its own conditional assignment; splitting the read
    # (in _build_and_save_figures, itself also decode-gated identically) into
    # a SEPARATE function from the write means Python can no longer see that
    # the two conditionals are the same one, so the return statement itself
    # needs corr_dt_pixel to unconditionally exist.
    corr_dt_pixel = None
    if decode:
        log_pixel = np.log10(np.clip(results.pixel_losses, 1e-12, None))
        corr_dt_pixel = np.corrcoef(log_dt, log_pixel)[0, 1]
        print(f"\ncorr w.r.t. log dt: log latent_loss {corr_dt_latent*100:.0f}%, "
              f"log pixel_loss {corr_dt_pixel*100:.0f}%")
    else:
        print(f"\ncorr w.r.t. log dt: log latent_loss {corr_dt_latent*100:.0f}% "
              f"(pixel-space skipped -- decode=False)")
    print("(near 0 = no dt dependence; positive = error grows with dt)")

    print(f"\nModel comparison ({'euler-only' if euler_only else 'latent_loss'} vs dt): "
          f"power law vs saturating exponential")
    a, b, r2_log, sse_power, pred_power = fit_power_law(results.dts, results.latent_losses)
    c, tau, r2_sat, sse_sat, pred_sat = fit_saturating_exponential(results.dts, results.latent_losses)
    print(f"  power law:     error ~ dt^{a:.3f}, SSE(real space)={sse_power:.6f}")
    print(f"  saturating exp: error -> {c:.4f} with timescale tau={tau:.1f}, "
          f"SSE(real space)={sse_sat:.6f}")
    better = "saturating exponential" if sse_sat < sse_power else "power law"
    print(f"  -> {better} fits better (lower SSE).")

    if euler_only:
        # results.latent_losses IS results.euler_losses here (see the substitution
        # above) -- a "euler-only vs full" comparison would just be
        # comparing that array against itself (ratio of means = 1.0,
        # "0.0% worse" every time), not a real finding. Skipped
        # entirely rather than printed as if it meant something.
        #
        # a_euler set to the ALREADY-computed a from the fit just
        # above, not recomputed via a separate fit_power_law(results.dts,
        # results.euler_losses) call -- that fit would be numerically IDENTICAL
        # anyway (same data, results.latent_losses IS results.euler_losses here), just
        # wastefully repeated. Still needed below regardless of this
        # mode: a_euler feeds the dt-decade table's own header. (The
        # power-law FIT CURVE itself, pred_power/pred_power_euler, is
        # no longer plotted anywhere -- panel [0,3] plots the same
        # Taylor-residual regression as panel [1,3] instead; only the
        # exponent a/a_euler is still reported, in text.)
        a_euler = a
    else:
        # Euler-only (hard-coded first-order term alone, no f_theta) vs the
        # full, f_theta-corrected prediction -- SAME z0_next_true, SAME dt,
        # SAME window for both, so this is a genuine, point-by-point
        # decomposition, not just two separately-fit trends.
        #
        # THEORY, if z1 were an EXACT derivative and f_theta an EXACT
        # curvature: raw euler_error ~ dt^2 (euler_error/dt ~ dt^1), raw
        # full_error ~ dt^3 (full_error/dt ~ dt^2) -- z1 being hard-coded
        # first-order means its error is entirely the curvature term
        # f_theta's own trained correction should be cancelling.
        #
        # WHAT WE ACTUALLY SEE (confirmed on real 64x64 data): euler-only
        # exponent ~1.0, not ~2.0. euler_error/dt is then ~dt^0, i.e. a
        # NON-VANISHING CONSTANT as dt->0 -- exactly z1(t) - z0_dot(t), z1's
        # own systematic bias against the true derivative. euler_error/dt
        # is measuring z1's own error directly, not curvature at all.
        #
        # A higher exponent alone does NOT mean full is actually better --
        # it only describes the asymptotic trend toward dt=0, which may lie
        # well below every dt actually observed. The direct magnitude
        # comparison below is what actually answers "which one is smaller,
        # in practice, over the range that matters."
        a_euler, _b_euler, _r2_log_euler, _sse_power_euler, _pred_power_euler = fit_power_law(results.dts, results.euler_losses)
        print(f"\nEuler-only (z0+z1*dt, no f_theta) vs full (f_theta-corrected) prediction, "
              f"same windows/dt:")
        print(f"  euler-only:  error ~ dt^{a_euler:.3f}")
        print(f"  full (f_theta): error ~ dt^{a:.3f}")
        print(f"  (a higher exponent only describes the asymptotic trend toward dt->0 -- "
              f"see the direct magnitude comparison below for which is ACTUALLY smaller "
              f"over the observed dt range)")

        ratio = np.array(results.latent_losses) / np.maximum(np.array(results.euler_losses), 1e-12)
        frac_full_worse = float((ratio > 1).mean())
        print(f"\nDirect magnitude comparison (full / euler-only), per window:")
        print(f"  mean(full)={np.mean(results.latent_losses):.6f}  mean(euler-only)={np.mean(results.euler_losses):.6f}  "
              f"ratio of means={np.mean(results.latent_losses) / np.mean(results.euler_losses):.4f}")
        print(f"  mean(full/euler-only ratio)={ratio.mean():.4f}  median={np.median(ratio):.4f}")
        print(f"  full prediction is WORSE than euler-only on {frac_full_worse:.1%} of windows")
        if frac_full_worse > 0.5:
            print(f"  -> f_theta's own trained correction is making the prediction WORSE on "
                  f"most windows, despite a higher fit exponent -- it is NOT currently adding "
                  f"value in practice, whatever its asymptotic behavior would eventually be.")

    header = f"\ndt decade      n       mean {'euler-only' if euler_only else 'latent'}_loss"
    header += "   mean pixel_loss" if decode else ""
    print(header)
    edges = np.floor(log_dt.min()), np.ceil(log_dt.max())
    for lo in np.arange(edges[0], edges[1] + 1):
        mask = (log_dt >= lo) & (log_dt < lo + 1)
        if mask.sum() == 0:
            continue
        row = f"1e{lo:.0f} - 1e{lo+1:.0f}   {mask.sum():4d}   {results.latent_losses[mask].mean():.6f}"
        row += f"         {results.pixel_losses[mask].mean():.6f}" if decode else ""
        print(row)

    # ---- temperature and noise: linear-space correlation + binned summary
    # Correlated against |error| and error DIRECTLY (results.latent_losses/
    # results.latent_losses_signed), NOT log10(|error|) -- panels [1,0]/[1,1]
    # themselves are LINEAR-scale (mean(error)/mean|error| twin axes,
    # not a log-scale boxplot the way they used to be), so a log-based
    # correlation number doesn't actually describe what those panels
    # show. Both signed and absolute reported now, not just absolute --
    # the panels themselves plot BOTH quantities (mean(error) on the
    # left, mean|error| on the right), so it's worth knowing whether
    # temperature/noise correlate with the DIRECTION of the bias too,
    # not just its magnitude.
    corr_temp_abs = np.corrcoef(results.temperatures, results.latent_losses)[0, 1]
    corr_temp_signed = np.corrcoef(results.temperatures, results.latent_losses_signed)[0, 1]
    corr_noise_abs = np.corrcoef(results.noises, results.latent_losses)[0, 1]
    corr_noise_signed = np.corrcoef(results.noises, results.latent_losses_signed)[0, 1]
    print(f"\ncorr w.r.t. temperature: error {corr_temp_signed * 100:.0f}%, |error| {corr_temp_abs * 100:.0f}%")
    print(f"corr w.r.t. noise: error {corr_noise_signed * 100:.0f}%, |error| {corr_noise_abs * 100:.0f}%")
    _print_binned_summary("temperature", results.temperatures, results.latent_losses, results.pixel_losses if decode else None)
    _print_binned_summary("noise", results.noises, results.latent_losses, results.pixel_losses if decode else None)

    # ---- length scale (first peak in autocorrelation): DIAGNOSTIC for
    # whether error tracks the microstructure's own dominant length
    # scale rather than (or in addition to) dt/temperature/noise --
    # motivated by rollout figures showing large-scale, visually "easy"
    # microstructures failing outright while a much finer-grained
    # texture reconstructs cleanly, raising the question of whether the
    # 8x8 latent bottleneck is under-resolving a specific length-scale
    # regime rather than "coarse features = easy".
    #
    # SATURATED VALUES EXCLUDED: the C++ simulation caps autocorr_length
    # at max_autocorr_dist(Nx, Ny) -- distances beyond that are an
    # artifact of periodic-boundary wraparound, not a real length scale.
    # This sentinel value is common (near-critical/smooth microstructures
    # in particular never show a decaying autocorrelation within range),
    # and left in as if real it distorts both the correlation and the
    # fit below -- a cluster of fake "very large length scale" points
    # that are actually "the true length scale is unknown/unbounded".
    # Treated as N/A: excluded from the correlation, the binned summary,
    # BOTH fits, and the scatter -- not just visually hidden.
    max_dist = max_autocorr_dist(ae_config["size"], ae_config["size"])
    saturated = results.length_scales >= max_dist
    n_saturated = int(saturated.sum())
    print(f"\n{n_saturated}/{len(results.length_scales)} windows have autocorr_length >= "
          f"{max_dist} (the C++ search cap) -- treated as N/A (not a real length "
          f"scale, just 'never decayed within range') and excluded below.")
    _print_saturation_cross_tab(results.temperatures, results.length_scales,
                                 results.abs_steps, results.latent_losses, results.dts, max_dist)

    # Oracle-z1 attribution -- placed here, alongside the other
    # console-only diagnostics, since it answers the question the
    # rest of this report can only circle around: WHICH STAGE the
    # euler-only floor actually belongs to. Needs the dataset
    # itself (not just results) because it reads one frame BEFORE
    # each window, which no per-window results array carries.
    oracle_per_window = None
    if dataset is not None:
        oracle_per_window = _print_oracle_z1_attribution(dataset, results, device)

    length_scales_valid = results.length_scales[~saturated]
    latent_losses_for_length = results.latent_losses[~saturated]
    log_latent_for_length = log_latent[~saturated]

    corr_length_latent = np.corrcoef(length_scales_valid, log_latent_for_length)[0, 1]
    print(f"corr w.r.t. length_scale: log latent_loss {corr_length_latent*100:.0f}% "
          f"(n={len(length_scales_valid)}, excluding saturated)")
    print("(if this is the dominant driver, error should track length_scale more "
          "cleanly than it tracks dt/temperature/noise individually above)")
    _print_binned_summary("length_scale", length_scales_valid, latent_losses_for_length,
                           results.pixel_losses[~saturated] if decode else None)

    # UNLIKE dt (log-log panel, where fit_power_law's straight-line form
    # is the natural visual fit check), the length_scale panel is
    # semi-log: length_scale itself is plotted on a LINEAR axis (it's a
    # continuous, per-window-computed quantity -- not a small discrete
    # sweep grid like temperature/noise -- so it's shown as a raw
    # scatter, not a boxplot), against a log-scaled error axis. A power
    # law would plot as a curve there, not a line, and wouldn't give the
    # same at-a-glance fit-quality read. fit_exponential (log(error)
    # linear in length_scale itself, not log(length_scale)) is the
    # correct semi-log analogue -- it's what actually draws straight on
    # this panel. Compared against the same saturating-exponential
    # candidate as dt (that model isn't tied to either axis convention).
    print("\nModel comparison (latent_loss vs length_scale): exponential vs saturating exponential")
    a_len, b_len, r2_len, sse_exp_len, pred_exp_len = fit_exponential(
        length_scales_valid, latent_losses_for_length
    )
    c_len, tau_len, r2_sat_len, sse_sat_len, pred_sat_len = fit_saturating_exponential(
        length_scales_valid, latent_losses_for_length
    )
    print(f"  exponential:    error ~ exp({a_len:.4f} * length_scale), SSE(real space)={sse_exp_len:.6f}")
    print(f"  saturating exp: error -> {c_len:.4f} with timescale tau={tau_len:.1f}, "
          f"SSE(real space)={sse_sat_len:.6f}")
    better_len = "saturating exponential" if sse_sat_len < sse_exp_len else "exponential"
    print(f"  -> {better_len} fits better (lower SSE).")

    # ---- per-(temperature, noise) aggregation for the 2D view -- the
    # most directly actionable panel: raw per-WINDOW points at the same
    # (temperature, noise) would just overplot (many windows share one
    # sweep point's fixed temperature/noise), so this pools first,
    # giving one point per sweep point, colored by its mean one-step
    # error.
    #
    # Keyed by (temperature, noise), NOT run_dir: this project's own
    # sweep runs several SEEDS at each (temperature, noise) grid point
    # (see the "T..._n..._s..." run-directory naming used throughout --
    # the trailing s### is the seed). Keying by run_dir would give each
    # seed its own separate bubble at the SAME (x, y) location --
    # several overlapping, indistinguishable bubbles per sweep point,
    # not one. Pooling by (temperature, noise) instead means the
    # reported mean/window-count reflect ALL windows from ALL seeds at
    # that sweep point, which is what "one bubble per sweep point"
    # should mean. Rounded before use as a dict key for the same reason
    # _boxplot_by_x/_mean_curves_by_unique_value round first: float
    # round-trip through a text metadata file can turn one intended
    # sweep value into several bit-distinct floats.
    per_point: dict[tuple[float, float], dict] = {}
    for t, n, ll in zip(results.temperatures, results.noises, results.latent_losses):
        key = (round(float(t), 6), round(float(n), 6))
        entry = per_point.setdefault(key, {"temperature": key[0], "noise": key[1], "losses": []})
        entry["losses"].append(ll)
    run_temps = np.array([v["temperature"] for v in per_point.values()])
    run_noises = np.array([v["noise"] for v in per_point.values()])
    run_mean_loss = np.array([np.mean(v["losses"]) for v in per_point.values()])
    run_n_windows = np.array([len(v["losses"]) for v in per_point.values()])

    # ---- per-RUN aggregation (kept separate from per_point above) --
    # for the "which specific run/seed performs worst" report just
    # below, which needs each seed identified individually, unlike
    # panel [0,2]'s own pooled-across-seeds view.
    per_run: dict[Path, dict] = {}
    for run_dir, t, n, ll in zip(results.run_dirs, results.temperatures, results.noises, results.latent_losses):
        entry = per_run.setdefault(run_dir, {"temperature": t, "noise": n, "losses": []})
        entry["losses"].append(ll)
    per_run_mean_loss = np.array([np.mean(v["losses"]) for v in per_run.values()])
    per_run_temps = np.array([v["temperature"] for v in per_run.values()])
    per_run_noises = np.array([v["noise"] for v in per_run.values()])
    per_run_n_windows = np.array([len(v["losses"]) for v in per_run.values()])

    print(f"\n{len(per_run)} distinct runs contributing windows. "
          f"Runs with the highest mean one-step error:")
    order = np.argsort(per_run_mean_loss)[::-1]
    run_dir_list = list(per_run.keys())
    for i in order[:10]:
        print(f"  {run_dir_list[i].name}: T={per_run_temps[i]:.3f}  noise={per_run_noises[i]:.4f}  "
              f"mean_loss={per_run_mean_loss[i]:.6f}  ({per_run_n_windows[i]} windows)")

    return _DerivedStats(
        corr_dt_pixel=corr_dt_pixel,
        oracle_per_window=oracle_per_window,
        corr_noise_abs=corr_noise_abs,
        corr_noise_signed=corr_noise_signed,
        corr_temp_abs=corr_temp_abs,
        corr_temp_signed=corr_temp_signed,
        run_mean_loss=run_mean_loss,
        run_n_windows=run_n_windows,
        run_noises=run_noises,
        run_temps=run_temps,
    )


def _build_and_save_figures(
    results: _EvaluationResults, stats: _DerivedStats, lds_checkpoint_path: Path, output_path: Path,
    dz0dt_output_path: Path, dt_dependence_output_path: Path, decode: bool, euler_only: bool,
) -> None:
    """Builds and saves all three output figures (parameter_dependence.png,
    dt_dependence.png, dz0dt.png), extracted verbatim as ONE function rather
    than split further -- the Taylor-residual decomposition (taylor_fit) is
    computed once here, right after fig/fig_dt are created, and reused by
    both the dt_dependence.png panels AND the later T<0.9/T>=0.9 console-only
    refits deep inside this same block (see that computation's own comment,
    preserved below, on why its position here is deliberate, not incidental)
    -- genuinely interleaved with panel construction, not a separable
    "compute all stats, then plot" sequence, so splitting it further would
    require REORDERING logic, not just extracting it. Pure side effect
    (writes three files), returns nothing."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    # dt_dependence.png: the 4 panels whose x-axis is dt (error,
    # |error|, pixel-space, and the combined dz0/dz0dt panel), moved
    # out of parameter_dependence.png entirely -- a separate figure,
    # not just a separate section of the same one, since they answer a
    # genuinely different question (how does error scale with the
    # PREDICTION SPAN itself) from what's left in parameter_dependence.png
    # (how does error depend on WHERE in parameter space -- temperature,
    # noise, which run -- a window sits).
    fig_dt, axes_dt = plt.subplots(2, 2, figsize=(12, 10))
    # Computed HERE (early, right after figure creation) rather than
    # right before the T<0.9/T>=0.9 console-only diagnostic calls below
    # (where it originally lived) -- the dt panels (axes[0,2]/axes[1,2])
    # plot this SAME regression (the one this function's own docstring
    # recommends), and their own position later in this function needs
    # it already available. See this call's own original location
    # (still nearby, just below) for why the T<0.9/T>=0.9 splits
    # themselves stay console-only and don't get their own variable
    # here: neither is plotted, only this ALL-DATA fit is.
    taylor_fit = fit_taylor_residual_coefficients(results.dts, results.euler_losses_signed, results.latent_losses_signed,
                                                    label="ALL DATA", euler_only=euler_only)

    # Main title parsed from the checkpoint's own filename (e.g.
    # "64x64-stage3a.pt" -> "64x64, stage 3a") rather than hardcoded --
    # stays correct automatically for whichever size/stage this actual
    # run is, without needing separate size/stage parameters threaded
    # through just for a title. Falls back to the raw stem if the name
    # doesn't match the expected pattern (e.g. an ephemeral
    # ensure_lds_checkpoint conversion, which appends its own suffix) --
    # a less pretty title beats a crash over a cosmetic-only feature.
    #
    # Also carries "(point size ~ windows contributed)" -- previously
    # repeated in every panel's own title (dt x2, temperature, noise,
    # the per-run scatter) since it's true of ALL of them here, not
    # panel-specific.
    _stem_match = re.match(r"^(\d+x\d+)-stage(\w+)", lds_checkpoint_path.stem)
    _title_text = (f"{_stem_match.group(1)}, stage {_stem_match.group(2)}" if _stem_match
                   else lds_checkpoint_path.stem)
    fig.suptitle(f"{_title_text}\n(point size ~ windows contributed)", fontsize=13)
    fig_dt.suptitle(f"{_title_text}\n(point size ~ windows contributed)", fontsize=13)

    if euler_only:
        # SEPARATE from the main title above (fig.text, not a second
        # fig.suptitle -- calling suptitle twice replaces the first
        # call rather than stacking) so the euler-only warning keeps
        # its own distinct red styling without forcing the whole main
        # title red too. Every panel not explicitly labeled otherwise
        # (temperature/noise curves, the per-run scatter) is ALSO
        # showing euler-only data here (via the results.latent_losses/
        # results.pixel_losses substitution above), even though their own
        # individual titles don't each say so -- this makes that fact
        # impossible to miss regardless of which panel someone looks at
        # first. Shown on BOTH figures now (fig and fig_dt), since the
        # dt panels moved to fig_dt are just as affected by euler_only
        # as anything still in fig.
        _euler_only_banner = ("EULER-ONLY MODE -- every panel below reports on the euler-only "
                               "(z0+z1*dt, no f_theta) prediction; f_theta is either untrained "
                               "(AE-family checkpoint) or excluded by request")
        fig.text(0.5, 0.955, _euler_only_banner, fontsize=11, color="tab:red", ha="center")
        fig_dt.text(0.5, 0.955, _euler_only_banner, fontsize=11, color="tab:red", ha="center")

    _euler_tag = " [euler-only]" if euler_only else ""
    # Moved to dt_dependence.png ([1,0] there), and now only built at
    # all when decode=True -- both the panel AND the results.pixel_losses data
    # it needs (see the main loop's own decoder-skipping note) are
    # gated on the same flag, so there's nothing here to plot when it's
    # False.
    if decode:
        # error, precisely: z0_pred(t+dt) - z0_true(t+dt) -- BOTH evaluated at
        # the same, target time t+dt, not t. z0_true(t+dt) = z0(t+dt) (the
        # encoder's own next-step z0) and z0_pred(t+dt) = z0(t) + z1(t)*dt [+
        # f_theta(...)*dt^2/2 when this is the "full", not euler-only,
        # quantity -- see z0_next_pred's own construction above]. This
        # panel is pixel-space specifically: decode(z0_pred) - decode(z0_true),
        # the SAME residual, run through the decoder, not an independently
        # computed pixel-space quantity.
        ax = axes_dt[1, 1]
        _boxplot_by_x(ax, results.dts, results.pixel_losses, log_x=True)
        ax.set_yscale("log")
        ax.set_xlabel("dt")
        ax.set_xlim(left=10)
        ax.set_ylabel(f"pixel-space{_euler_tag}: decode(z0(t+dt)): mean|pred - true|")
        ax.set_title(f"pixel-space (L1, decoded){_euler_tag}\n"
                     f"corr w.r.t. log dt: log |error| {stats.corr_dt_pixel * 100:.0f}%")
    else:
        fig_dt.delaxes(axes_dt[1, 1])

    # SAME binned-mean-curve style as the temperature/noise panels below
    # (see that loop's own comments for the full rationale on twin axes
    # / symmetric-left / window-count sizing), applied here to dt and to
    # error/dt instead
    # of error directly -- and to [1,3] as well (see that panel's own
    # code, further below): both now share this exact style, [0,3]
    # showing whatever the current mode's quantity is (full when not
    # euler_only, matching results.latent_losses' own substitution), [1,3]
    # ALWAYS showing the euler-only quantity specifically, regardless of
    # mode, for a direct side-by-side comparison between the two.
    #
    # dt is grouped by UNIQUE value (_mean_curves_by_unique_value), NOT
    # binned (_mean_curves_by_bin) -- dt is INTRINSICALLY discrete here,
    # not a continuous quantity artificially made to look discrete:
    # every dt is (later_step - earlier_step) * metadata.dt, a product
    # of an INTEGER step difference and this sweep's own fixed per-step
    # dt, so the full SET of dt values actually occurring is itself
    # finite and exact -- matching how [0,0]/[0,1] already treat it
    # (_boxplot_by_x's own per-unique-value grouping), not the
    # continuous-range binning length_scale used (back when it was
    # still plotted) or that dt itself used here before this fix.
    #
    # _DT_PANEL_SCALE (1000): these values are naturally tiny (~1e-4 to
    # 1e-5), making default tick labels either cramped scientific
    # notation or hard-to-read leading zeros -- scaling by 1000 here
    # (applied to the DATA itself, not just tick formatting, so it
    # propagates automatically through the ylim computation and the
    # regression curve below without needing separate handling) and
    # labeling the axis "[1e-3]" keeps the actual numbers readable.
    _DT_PANEL_SCALE = 1000
    dt_x, dt_signed, dt_abs, dt_n = _mean_curves_by_unique_value(
        results.dts, results.latent_losses_signed / results.dts, results.latent_losses / results.dts)
    dt_signed, dt_abs = dt_signed * _DT_PANEL_SCALE, dt_abs * _DT_PANEL_SCALE

    # Trivial baseline: the "assume nothing changes" predictor,
    # z0_pred(t+dt) = z0(t). Its error/dt is EXACTLY -dz0/dt (note the
    # sign: error = z0_pred(t+dt) - z0_true(t+dt) = z0(t) - z0(t+dt) =
    # -dz0, matching this panel's own error convention above), and its
    # |error|/dt is EXACTLY |dz0/dt| = results.dz0dt_abs -- no new
    # computation needed, dz0/dz0dt are already tracked per-window for
    # dz0dt.png and the [0,1] panel below; this just reuses them here
    # too, grouped the same way (_mean_curves_by_unique_value, by
    # unique dt) as every other curve on these two panels. A genuine,
    # exact baseline (not a rough guide) precisely because it needs no
    # model at all -- either curve beating it is the real bar for
    # "learned something dt-dependent worth having", not just "beat
    # zero".
    _, dt_trivial_signed, dt_trivial_abs, _ = _mean_curves_by_unique_value(
        results.dts, results.dz0dt_signed, results.dz0dt_abs)
    dt_trivial_signed = -dt_trivial_signed * _DT_PANEL_SCALE
    dt_trivial_abs = dt_trivial_abs * _DT_PANEL_SCALE

    # [1,3] built HERE too (not at its former, separate location later
    # in this function) -- both panels need exactly the same already-
    # available data (results.dts, taylor_fit, results.euler_losses/_signed), and
    # building them together is what makes the shared-axis alignment
    # below straightforward, rather than needing a second pass back
    # over [0,3] after [1,3] is built elsewhere.
    #
    # [1,3] ALWAYS uses results.euler_losses/results.euler_losses_signed specifically
    # (NOT the results.latent_losses/_signed substitution [0,3] and every other
    # panel use) -- regardless of the overall euler_only mode, so it's
    # always a meaningful "euler-only" reference to compare [0,3]
    # against. Under euler_only=True, [0,3] and [1,3] end up showing
    # the same thing (results.latent_losses IS results.euler_losses in that mode) --
    # an expected, harmless redundancy in that specific mode, not a
    # bug, since [1,3]'s own point is to stay meaningful specifically
    # when [0,3] is NOT already euler-only.
    dt_x_13, dt_signed_13, dt_abs_13, dt_n_13 = _mean_curves_by_unique_value(
        results.dts, results.euler_losses_signed / results.dts, results.euler_losses / results.dts)
    dt_signed_13, dt_abs_13 = dt_signed_13 * _DT_PANEL_SCALE, dt_abs_13 * _DT_PANEL_SCALE

    # Regression curve overlay: eps/eps'/A are SHARED (the joint fit's
    # own point, or the only fit there is under euler_only=True); C is
    # ALWAYS the euler-only-specific coefficient (present in both fit
    # modes), D only exists under euler_only=False (the joint fit) --
    # [0,3] uses whichever of C/D actually matches its own CURRENT
    # quantity, [1,3] always uses C, matching its own always-euler-only
    # quantity. Curve evaluated over a SMOOTH, continuous dt range (not
    # just at the discrete dt_x/dt_x_13 values above) for visual
    # smoothness, overlaid on the discrete empirical points -- the
    # regression itself was fit against ALL windows regardless of what
    # gets drawn here (see fit_taylor_residual_coefficients' own
    # docstring), this is purely a display choice.
    eps, eps_prime, A = taylor_fit["eps"], taylor_fit["eps_prime"], taylor_fit["A"]
    C = taylor_fit["C"]
    D_or_C = C if euler_only else taylor_fit["D"]
    dt_curve = np.geomspace(results.dts.min(), results.dts.max(), 400)

    # Two panels now, one per QUANTITY (signed "error", absolute
    # "|error|"), each showing BOTH euler-only and full (mode-
    # dependent) as separate curves on the SAME axis -- replacing the
    # earlier layout (one panel always-euler-only, one mode-dependent,
    # each internally split signed/absolute via twin axes). Answers a
    # more direct question this way: does full (f_theta-corrected)
    # actually beat euler-only, for signed and absolute error
    # separately -- without needing a cross-panel comparison to see it.
    #
    # No twin axes needed anymore -- both curves in a given panel are
    # the SAME kind of quantity (both signed, or both absolute), unlike
    # before where one axis's own two curves were genuinely different
    # things (signed vs absolute) needing separate scales.
    ax_signed, ax_abs = axes_dt[0, 0], axes_dt[1, 0]

    # In EULER-ONLY mode (a stage-2/AE-family checkpoint, f_theta
    # untrained) the "full" curve is a numerically IDENTICAL duplicate of
    # euler-only -- two overlapping lines carrying one line's worth of
    # information. That slot is far better spent on the causal/oracle
    # derivative baselines, which answer the question a stage-2
    # checkpoint actually raises: how much of this error is z1's own,
    # versus already present in z0's trajectory (see
    # _print_oracle_z1_attribution). In stage-3 mode "full" is the
    # informative curve and is kept as-is; the oracle curves are then
    # omitted rather than crowding four lines onto one panel.
    oracle_curves = None
    if euler_only and stats.oracle_per_window is not None:
        opw = stats.oracle_per_window
        ok = ~np.isnan(opw["causal_abs"])
        if ok.any():
            ox, o_causal_signed, o_causal_abs, _ = _mean_curves_by_unique_value(
                results.dts[ok], opw["causal_signed"][ok] / results.dts[ok],
                opw["causal_abs"][ok] / results.dts[ok])
            _, o_oracle_signed, o_oracle_abs, _ = _mean_curves_by_unique_value(
                results.dts[ok], opw["oracle_signed"][ok] / results.dts[ok],
                opw["oracle_abs"][ok] / results.dts[ok])
            oracle_curves = dict(
                x=ox,
                causal_signed=o_causal_signed * _DT_PANEL_SCALE,
                causal_abs=o_causal_abs * _DT_PANEL_SCALE,
                oracle_signed=o_oracle_signed * _DT_PANEL_SCALE,
                oracle_abs=o_oracle_abs * _DT_PANEL_SCALE,
            )
    # dt_n and dt_n_13 group the SAME dt array the same way -- only the
    # loss VALUES differ between euler-only/full, not which windows
    # landed in which dt bucket -- so one dot-sizing scheme covers both
    # curves in both panels.
    sizes = _size_by_count(dt_n)

    ax_signed.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    # Trivial baseline plotted FIRST (lower zorder, sits visually behind
    # the two real curves) -- see its own data-prep comment above for
    # why this is an exact, not approximate, reference: literally the
    # error of never updating the prediction at all. Gray/square
    # markers/dashed, deliberately distinct from euler-only's blue and
    # full's orange -- this is a reference line, not a third model
    # variant to compare against those two on equal footing.
    ax_signed.plot(dt_x, dt_trivial_signed, "--", color="gray", linewidth=1, zorder=0)
    ax_signed.scatter(dt_x, dt_trivial_signed, s=sizes, color="gray", zorder=0,
                       marker="s", edgecolors="black", linewidths=0.3,
                       label="trivial (no change)")
    ax_signed.plot(dt_x_13, dt_signed_13, "-", color="tab:blue", linewidth=1, zorder=1)
    ax_signed.scatter(dt_x_13, dt_signed_13, s=sizes, color="tab:blue", zorder=2,
                       edgecolors="black", linewidths=0.3, label="euler-only")
    if oracle_curves is None:
        ax_signed.plot(dt_x, dt_signed, "-", color="tab:orange", linewidth=1, zorder=1)
        ax_signed.scatter(dt_x, dt_signed, s=sizes, color="tab:orange", zorder=2,
                           edgecolors="black", linewidths=0.3, label="full")
    else:
        oc = oracle_curves
        ax_signed.plot(oc["x"], oc["causal_signed"], "-", color="tab:green", linewidth=1, zorder=1)
        ax_signed.scatter(oc["x"], oc["causal_signed"], s=20, color="tab:green", zorder=2,
                           marker="^", edgecolors="black", linewidths=0.3,
                           label="causal dz0/dt (past only)")
        ax_signed.plot(oc["x"], oc["oracle_signed"], ":", color="tab:purple", linewidth=1, zorder=1)
        ax_signed.scatter(oc["x"], oc["oracle_signed"], s=20, color="tab:purple", zorder=2,
                           marker="v", edgecolors="black", linewidths=0.3,
                           label="oracle dz0/dt (sees future)")
    # Lock the y-range from empirical data alone, BEFORE the regression
    # curves (below) are drawn -- same reasoning as before: the curves'
    # own 1/dt divergence near results.dts.min() would otherwise crush the
    # actual data into a thin band. The trivial baseline is real,
    # non-diverging empirical data (unlike the regression curves below),
    # so it's included in this lock deliberately -- if it's large enough
    # to widen the range, that's the actual finding, not something to
    # hide by locking the range before plotting it.
    ax_signed.set_ylim(ax_signed.get_ylim())

    # |error| is strictly a magnitude, so it belongs on a LOG y-axis --
    # the curves span 3+ decades here (a trivial-baseline floor near 1e-2
    # against euler errors above 1e0), which a linear axis flattens into
    # an unreadable band near zero. The signed panel above stays LINEAR:
    # it legitimately crosses zero, which log cannot represent.
    #
    # Truncation rule: a mean |error| of exactly zero in some dt bin is
    # not a real measurement -- it means that bin has too little usable
    # data (often a single window, or one whose own residual underflowed)
    # -- and log() of it is undefined anyway. Per the same reasoning that
    # such a bin is dubious, EVERY dt at or above the first offending one
    # is dropped too, not just the offending bin: the largest-dt bins are
    # the sparsest, so one bad bin means the tail beyond it is
    # untrustworthy rather than merely gappy.
    abs_curves = [(dt_x, dt_trivial_abs), (dt_x_13, dt_abs_13)]
    if oracle_curves is None:
        abs_curves.append((dt_x, dt_abs))
    else:
        abs_curves += [(oracle_curves["x"], oracle_curves["causal_abs"]),
                        (oracle_curves["x"], oracle_curves["oracle_abs"])]
    dt_abs_cutoff = np.inf
    for cx, cy in abs_curves:
        bad = np.asarray(cy) <= 0
        if bad.any():
            dt_abs_cutoff = min(dt_abs_cutoff, float(np.asarray(cx)[bad].min()))
    if np.isfinite(dt_abs_cutoff):
        n_dropped = int((np.asarray(dt_x) >= dt_abs_cutoff).sum())
        print(f"\n|error| panel: dropping dt >= {dt_abs_cutoff:.4g} "
              f"({n_dropped} dt value(s)) -- a mean |error| of exactly zero there means too "
              f"little usable data in that bin, so that dt and every larger one is dubious.")

    def _keep(cx, *ys):
        m = np.asarray(cx) < dt_abs_cutoff
        return (np.asarray(cx)[m],) + tuple(np.asarray(y)[m] for y in ys) + (m,)

    tx, t_triv, t_mask = _keep(dt_x, dt_trivial_abs)
    ax_abs.plot(tx, t_triv, "--", color="gray", linewidth=1, zorder=0)
    ax_abs.scatter(tx, t_triv, s=sizes[t_mask], color="gray", zorder=0,
                    marker="s", edgecolors="black", linewidths=0.3,
                    label="trivial (no change)")
    ex, e_abs, e_mask = _keep(dt_x_13, dt_abs_13)
    ax_abs.plot(ex, e_abs, "-", color="tab:blue", linewidth=1, zorder=1)
    ax_abs.scatter(ex, e_abs, s=sizes[e_mask], color="tab:blue", zorder=2,
                    edgecolors="black", linewidths=0.3, label="euler-only")
    if oracle_curves is None:
        fx, f_abs, f_mask = _keep(dt_x, dt_abs)
        ax_abs.plot(fx, f_abs, "-", color="tab:orange", linewidth=1, zorder=1)
        ax_abs.scatter(fx, f_abs, s=sizes[f_mask], color="tab:orange", zorder=2,
                        edgecolors="black", linewidths=0.3, label="full")
    else:
        oc = oracle_curves
        ox, o_c, o_o, _ = _keep(oc["x"], oc["causal_abs"], oc["oracle_abs"])
        ax_abs.plot(ox, o_c, "-", color="tab:green", linewidth=1, zorder=1)
        ax_abs.scatter(ox, o_c, s=20, color="tab:green", zorder=2,
                        marker="^", edgecolors="black", linewidths=0.3,
                        label="causal dz0/dt (past only)")
        ax_abs.plot(ox, o_o, ":", color="tab:purple", linewidth=1, zorder=1)
        ax_abs.scatter(ox, o_o, s=20, color="tab:purple", zorder=2,
                        marker="v", edgecolors="black", linewidths=0.3,
                        label="oracle dz0/dt (sees future)")
    # log scale, NOT set_ylim(bottom=0) -- 0 is not representable on a log
    # axis, and matplotlib would silently clip to its own positive floor.
    ax_abs.set_yscale("log")

    # Regression curves, SIGNED panel only -- still no equivalent shown
    # on the absolute panel; abs() of this same curve tracks E[signed
    # residual] (the bias), not E[|residual|] (what the absolute points
    # themselves show), and on real data it doesn't just undershoot --
    # it can sit flat, visibly describing nothing about the actual
    # mean|error|/dt trend. One curve per data line now, matching
    # colors: euler-only always uses C, full uses D (or C again under
    # euler_only=True, where full's own data line already IS the
    # euler-only one -- the expected, harmless redundancy this session
    # has established elsewhere for that specific mode).
    raw_curve_euler = eps + eps_prime * dt_curve + C * dt_curve ** 2 - A * dt_curve ** 3
    ax_signed.plot(dt_curve, raw_curve_euler / dt_curve * _DT_PANEL_SCALE, "--",
                   color="tab:blue", linewidth=1, label=f"fit (euler): eps={eps:.3e}")
    # "fit (full)" plotted ONLY when there is a genuinely separate full
    # model to fit. Under euler_only=True the joint fit shares eps/eps'/A
    # and D_or_C falls back to C, so this curve is numerically IDENTICAL
    # to the euler one above -- a second dashed orange line exactly on
    # top of the blue, plus a legend entry repeating the same eps. Same
    # reasoning as dropping the duplicate "full" DATA curve in that mode.
    if not euler_only:
        raw_curve_full = eps + eps_prime * dt_curve + D_or_C * dt_curve ** 2 - A * dt_curve ** 3
        ax_signed.plot(dt_curve, raw_curve_full / dt_curve * _DT_PANEL_SCALE, "--",
                       color="tab:orange", linewidth=1, label=f"fit (full): eps={eps:.3e}")

    # Precise, not just "error": z0_pred(t+dt) = z0(t) + z1(t)*dt for
    # the euler-only curve, + f_theta(...)*dt^2/2 for the full curve
    # (when NOT euler_only -- under euler_only=True the two curves and
    # their data coincide, per the redundancy note above). For the
    # euler-only curve specifically, error/dt reduces EXACTLY to
    # z1 - dz0/dt (the same target_deriv
    # evaluation.check_deriv_temperature.py measures z1 against
    # directly) -- z0_euler_pred=z0(t)+z1(t)*dt divided by dt is
    # z1(t) + [z0(t)-z0(t+dt)]/dt = z1(t) - dz0/dt, exactly.
    for ax, name, title in [(ax_signed, "error", "error (mean, signed)\nerror = (z0_pred(t+dt)-z0_true(t+dt))/dt"),
                             (ax_abs, "|error|", "|error| (mean, absolute)\nerror = (z0_pred(t+dt)-z0_true(t+dt))/dt")]:
        ax.set_xscale("log")
        ax.set_xlabel("dt")
        ax.set_xlim(left=10)
        ax.set_ylabel(f"mean({name}) [1e-3]" if name == "error" else f"mean{name} [1e-3]")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")

    # The key actionable panel: which (temperature, noise) region has
    # the highest error, aggregated per run so points don't overplot.
    # Left as a scatter, not a boxplot -- this is genuinely 2D (two
    # discrete axes at once), not a single discrete-x-vs-y relationship.
    # alpha=0.7 (not 1.0): different SEEDS at the same (temperature,
    # noise) setting are still separate points here (aggregation is
    # PER RUN, i.e. per seed -- see the comment above -- not per T/noise
    # pair), so multiple points can genuinely share a coordinate. Without
    # transparency, whichever one matplotlib happens to draw last fully
    # hides the other(s) -- alpha lets overlapping points blend visibly
    # instead. Deliberately NOT jittered: jittering would hide the same
    # problem by DISTORTING the coordinates instead of revealing it, and
    # the exact (temperature, noise) value is the actual thing this
    # panel exists to let someone read off precisely.
    # vmin=0 (not matplotlib's own auto-picked min, which lands slightly
    # above 0 since mean_latent_loss is a genuinely positive quantity
    # with no run actually AT zero error) -- 0 is the real, physically
    # meaningful floor for a loss, so the colorbar should start there,
    # not at whatever the smallest observed run happens to be. Same
    # reasoning for the y-axis (noise) below: 0 is noise's own real
    # floor, not "the smallest noise value in this sweep".
    # s is marker AREA in matplotlib (points^2), and diameter ~ sqrt(s)
    # -- reducing diameter by a quarter (new = 0.75 * old) means scaling
    # area by 0.75^2 = 0.5625, not by 0.75 itself.
    sc = axes[2].scatter(stats.run_temps, stats.run_noises, c=stats.run_mean_loss,
                          s=0.5625 * (30 + 10 * stats.run_n_windows),
                          cmap="viridis", edgecolors="black", linewidths=0.3, alpha=0.7, vmin=0)
    axes[2].set_xlabel("temperature")
    axes[2].set_ylabel("noise")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("z0(t+dt): mean|pred - true| per run")
    fig.colorbar(sc, ax=axes[2], label="z0(t+dt): mean|pred - true|")

    # Two curves per panel -- mean(error) (signed -- can be negative,
    # shows whether the bias itself has a consistent direction) and
    # mean|error| (the magnitude, same quantity _boxplot_by_x used to
    # show as a full distribution before this) -- on SEPARATE y-axes
    # (ax.twinx()), not sharing one: mean|error| is never negative and,
    # by the Jensen's-inequality relationship established earlier this
    # session, is generally LARGER in magnitude than mean(error) --
    # sharing one axis would visually flatten whichever curve has the
    # smaller range. Marker size at each point scales with how many
    # windows went into it (via _size_by_count, normalized per panel --
    # see its own docstring for why NOT [0,2]'s literal formula), same
    # idea as [0,2]'s own window-count sizing.
    #
    # Grouped by unique sweep value (temperature/noise are discrete
    # sweep inputs, not a continuous range -- see
    # _mean_curves_by_unique_value's own docstring).
    temp_x, temp_signed, temp_abs, temp_n = _mean_curves_by_unique_value(
        results.temperatures, results.latent_losses_signed, results.latent_losses)
    noise_x, noise_signed, noise_abs, noise_n = _mean_curves_by_unique_value(
        results.noises, results.latent_losses_signed, results.latent_losses)

    # [1,2] (length_scale) REMOVED entirely -- its own values are
    # computed from only the non-saturated subset (windows where
    # autocorr_length actually decayed within the search range), which
    # this project's own data shows is disproportionately the LOW-error
    # windows (near-T0/high-noise windows tend to have less-developed
    # structure, hence "never decayed", hence EXCLUDED here) -- so this
    # panel's own values are typically ~1000x smaller than
    # temperature/noise's, not because length_scale's own effect is
    # negligible, but because it's drawn from a systematically
    # different, lower-error subpopulation. Including it in the SAME
    # shared-axis alignment as temperature/noise was actively
    # misleading (forcing their own, meaningful ranges to accommodate
    # its near-zero one) rather than just uninformative on its own.
    # length_scale's own correlation/model-comparison console output
    # above is UNCHANGED -- only this plotted panel is gone.
    panel_specs = [
        (axes[0], temp_x, temp_signed, temp_abs, temp_n, "temperature",
         stats.corr_temp_signed, stats.corr_temp_abs),
        (axes[1], noise_x, noise_signed, noise_abs, noise_n, "noise",
         stats.corr_noise_signed, stats.corr_noise_abs),
    ]
    twin_axes = []
    for ax, x, mean_signed, mean_abs, n_windows, xlabel, corr_s, corr_a in panel_specs:
        twin = ax.twinx()
        twin_axes.append(twin)
        sizes = _size_by_count(n_windows)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax.plot(x, mean_signed, "-", color="tab:blue", linewidth=1, zorder=1)
        ax.scatter(x, mean_signed, s=sizes, color="tab:blue", zorder=2,
                   edgecolors="black", linewidths=0.3, label="z0(t+dt): mean(pred - true)")
        twin.plot(x, mean_abs, "-", color="tab:orange", linewidth=1, zorder=1)
        twin.scatter(x, mean_abs, s=sizes, color="tab:orange", zorder=2,
                     edgecolors="black", linewidths=0.3, label="z0(t+dt): mean|pred - true|")
        ax.set_xlabel(xlabel)
        # Precise, not just "error": pred = z0(t) + z1(t)*dt [+
        # f_theta(...)*dt^2/2 when this is the "full", not euler-only,
        # quantity], true = z0(t+dt) (the encoder's own next-step z0) --
        # see the dt panels' own comment for the exact euler-only
        # reduction (z0_pred(t+dt) - z0_true(t+dt), divided by dt, is exactly
        # z1 - dz0/dt there specifically).
        ax.set_ylabel("z0(t+dt): mean(pred - true)", color="tab:blue")
        twin.set_ylabel("z0(t+dt): mean|pred - true|", color="tab:orange")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        twin.tick_params(axis="y", labelcolor="tab:orange")
        # Condensed: corr(...) stated once, not once per value -- the
        # variable being correlated against (xlabel) is the shared part,
        # error/|error| are the two values, matching the console print's
        # own format just above.
        ax.set_title(f"corr w.r.t. {xlabel}: error {corr_s * 100:.0f}%, |error| {corr_a * 100:.0f}%")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="best")

    # noise, like length_scale, is a non-negative physical quantity --
    # starts its x-axis at 0 for the same reason [0,2]/(the former)
    # [1,2] do. temperature is NOT included here (deliberately) -- it's
    # not a quantity with a meaningful zero in this context (this
    # sweep's own temperature range starts well above 0, and 0 has no
    # special physical significance for it the way it does for a
    # magnitude like noise).
    axes[1].set_xlim(left=0)

    # y-ranges: LEFT (mean(z0_pred(t+dt)-z0_true(t+dt))) symmetric about 0, RIGHT
    # (mean|z0_pred(t+dt)-z0_true(t+dt)|) floored at 0 -- see
    # _symmetric_left_zero_right_ylim's own docstring for why. Shared
    # across these two panels only, computed from what's actually
    # autoscaled after every artist is in place.
    left_ylim, right_ylim = _symmetric_left_zero_right_ylim(
        [axes[0], axes[1]], twin_axes)
    for ax, twin in zip([axes[0], axes[1]], twin_axes):
        ax.set_ylim(left_ylim)
        twin.set_ylim(right_ylim)

    # Run TWICE -- all data, and a T<0.9 subset -- to check whether the
    # large stderr on eps/C (both consistent with zero in the all-data
    # fit) is coming disproportionately from the high-temperature
    # (T>=0.9) region specifically, rather than being spread evenly
    # across the whole dataset. The PLOTTED curve in the dt panels
    # (ax_signed, in dt_dependence.png) still uses the all-data fit (computed earlier,
    # right after figure creation) -- the T<0.9 comparison here is a
    # console-only diagnostic, not a second set of plotted points;
    # restricting the plot's own data to a temperature subset would
    # need its own dedicated panel to be meaningful, not just swapped
    # into this one.
    t_mask = results.temperatures < 0.9
    if t_mask.sum() >= 2 and (~t_mask).sum() >= 2:
        # BOTH T<0.9 and T>=0.9 run, not just the former excluding the
        # latter -- isolating T>=0.9 directly (rather than only ever
        # seeing it by omission) is what actually distinguishes "the
        # excluded region is just noisier" from "the excluded region
        # has genuinely different, e.g. opposite-signed, coefficients"
        # -- two very different findings that look identical if you
        # only ever run the T<0.9 side.
        fit_taylor_residual_coefficients(results.dts[t_mask], results.euler_losses_signed[t_mask],
                                          results.latent_losses_signed[t_mask], label="T < 0.9 SUBSET",
                                          euler_only=euler_only)
        fit_taylor_residual_coefficients(results.dts[~t_mask], results.euler_losses_signed[~t_mask],
                                          results.latent_losses_signed[~t_mask], label="T >= 0.9 SUBSET",
                                          euler_only=euler_only)
    else:
        print(f"\n  Skipping T-split regressions -- {t_mask.sum()} of {len(results.temperatures)} "
              f"windows have T<0.9 (not enough on one side of the split to fit against).")

    # New 7th panel: the two ORANGE (mean|.|) curves from dz0dt.png's own
    # [0:1, 1] (both "vs dt" panels, one per row -- mean|dz0| and
    # mean|dz0/dt|) added here too, as a single combined plot -- NOT
    # removed from dz0dt.png, which still has both in full (with their
    # own mean(.) blue curves alongside, which aren't duplicated here).
    # Grouped by unique dt value, same convention as everywhere dt
    # appears in this script. Twin axes, same reasoning as elsewhere --
    # |dz0| and |dz0/dt| have very different natural scales (|dz0| grows
    # with dt, |dz0/dt| roughly doesn't, per this session's own earlier
    # finding), so one shared axis would flatten whichever has the
    # smaller range.
    dt_x7, _, dz0_abs_by_dt, _ = _mean_curves_by_unique_value(results.dts, results.dz0_signed, results.dz0_abs)
    _, _, dz0dt_abs_by_dt, _ = _mean_curves_by_unique_value(results.dts, results.dz0dt_signed, results.dz0dt_abs)
    dz0_abs_by_dt = dz0_abs_by_dt * _DT_PANEL_SCALE
    dz0dt_abs_by_dt = dz0dt_abs_by_dt * _DT_PANEL_SCALE
    ax7 = axes_dt[0, 1]
    twin7 = ax7.twinx()
    ax7.plot(dt_x7, dz0_abs_by_dt, "o-", color="tab:orange", label="mean|dz0|")
    twin7.plot(dt_x7, dz0dt_abs_by_dt, "o-", color="tab:red", label="mean|dz0/dt|")
    ax7.set_xscale("log")
    ax7.set_xlabel("dt")
    ax7.set_xlim(left=10)
    ax7.set_ylabel("mean|dz0| [1e-3]", color="tab:orange")
    twin7.set_ylabel("mean|dz0/dt| [1e-3]", color="tab:red")
    ax7.tick_params(axis="y", labelcolor="tab:orange")
    twin7.tick_params(axis="y", labelcolor="tab:red")
    ax7.set_title("real |dz0|, |dz0/dt| vs dt\n(from dz0dt.png's own [0:1,1] panels)")
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = twin7.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.96) if euler_only else (0, 0, 1, 1))
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved figure to {output_path}")

    fig_dt.tight_layout(rect=(0, 0, 1, 0.96) if euler_only else (0, 0, 1, 1))
    fig_dt.savefig(dt_dependence_output_path, dpi=120)
    plt.close(fig_dt)
    print(f"Saved figure to {dt_dependence_output_path}")

    # ---- separate figure: ground-truth dz0 and dz0/dt vs t and dt --
    # NOT part of the main figure above, since neither is an error or a
    # model-dependent quantity at all: both are computed purely from
    # the encoder's own z0_t/z0_next_true, with no z1/f_theta/euler_only
    # distinction applying to either (unlike literally everything else
    # this function computes). Answers a different question than the
    # rest of this script -- not "how wrong is the model", but "how
    # much/fast is the underlying microstructure actually evolving, and
    # does that itself depend on when (t) you look or over what span
    # (dt)". Row 0: dz0 (the raw, un-normalized displacement). Row 1:
    # dz0/dt (the rate) -- the SAME quantity this figure showed before
    # this row was added, just now alongside its own un-normalized
    # counterpart for direct comparison.
    #
    # Two lines per panel (mean(x) and mean|x|), NOT a boxplot -- tried
    # boxplots first, but with real data they end up dominated by
    # outlier whiskers/fliers (this quantity has the same "mean >> median,
    # outlier-driven skew" character established repeatedly elsewhere in
    # this script), burying the actual central trend the mean curves
    # show cleanly. Also no scatter/dot-sizing-by-window-count the way
    # the main figure's own twin-axis panels have -- just the two lines,
    # kept deliberately simple.
    #
    # Grouped by unique value (_mean_curves_by_unique_value) for BOTH t
    # and dt, not binned -- both are naturally discrete here, drawn from
    # the same fixed, regularly-spaced save-cadence grid (see the main
    # figure's own dt panels for the same reasoning; t is no different
    # from dt in this respect, despite spanning a much wider range).
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

    panel_rows = [
        (results.dz0_signed, results.dz0_abs, "dz0", 0),
        (results.dz0dt_signed, results.dz0dt_abs, "dz0/dt", 1),
    ]
    for y_signed, y_abs, name, row in panel_rows:
        t_x, t_signed_y, t_abs_y, _ = _mean_curves_by_unique_value(results.abs_steps, y_signed, y_abs)
        dt_x, dt_signed_y, dt_abs_y, _ = _mean_curves_by_unique_value(results.dts, y_signed, y_abs)
        # x1000 for readability -- these values are naturally tiny,
        # matching the same reasoning/named-constant convention as
        # _DT_PANEL_SCALE in the main figure's own dt panels above.
        t_signed_y, t_abs_y = t_signed_y * _DT_PANEL_SCALE, t_abs_y * _DT_PANEL_SCALE
        dt_signed_y, dt_abs_y = dt_signed_y * _DT_PANEL_SCALE, dt_abs_y * _DT_PANEL_SCALE

        ax_t, ax_dt = axes2[row, 0], axes2[row, 1]
        twin_t, twin_dt = ax_t.twinx(), ax_dt.twinx()

        ax_t.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax_t.plot(t_x, t_signed_y, "o-", color="tab:blue", label=f"mean({name})")
        twin_t.plot(t_x, t_abs_y, "o-", color="tab:orange", label=f"mean|{name}|")
        ax_t.set_xlabel("t (window's own starting step)")
        ax_t.set_ylabel(f"mean({name})", color="tab:blue")
        twin_t.set_ylabel(f"mean|{name}|", color="tab:orange")
        ax_t.tick_params(axis="y", labelcolor="tab:blue")
        twin_t.tick_params(axis="y", labelcolor="tab:orange")
        ax_t.set_title(f"real {name} vs t")
        ax_t.set_xscale("log")
        # Starts at 1000, not the data's own minimum -- requested
        # explicitly, presumably to crop out the smallest-t region
        # (sparse, few windows this early relative to min_step) rather
        # than let it dominate the axis the way [0,3]'s own divergent
        # small-dt region did earlier in this session.
        ax_t.set_xlim(left=1000)
        lines1, labels1 = ax_t.get_legend_handles_labels()
        lines2, labels2 = twin_t.get_legend_handles_labels()
        ax_t.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

        ax_dt.axhline(0, color="gray", linewidth=0.7, linestyle=":")
        ax_dt.plot(dt_x, dt_signed_y, "o-", color="tab:blue", label=f"mean({name})")
        twin_dt.plot(dt_x, dt_abs_y, "o-", color="tab:orange", label=f"mean|{name}|")
        ax_dt.set_xlabel("dt")
        ax_dt.set_ylabel(f"mean({name})", color="tab:blue")
        twin_dt.set_ylabel(f"mean|{name}|", color="tab:orange")
        ax_dt.tick_params(axis="y", labelcolor="tab:blue")
        twin_dt.tick_params(axis="y", labelcolor="tab:orange")
        ax_dt.set_title(f"real {name} vs dt")
        ax_dt.set_xscale("log")
        ax_dt.set_xlim(left=10)
        lines1, labels1 = ax_dt.get_legend_handles_labels()
        lines2, labels2 = twin_dt.get_legend_handles_labels()
        ax_dt.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

        # y-ranges aligned WITHIN this row only (dz0 and dz0/dt are
        # different quantities with different natural scales -- dz0
        # grows with dt, dz0/dt does not, per this session's own
        # earlier finding -- so aligning across rows would be
        # comparing two different units on one shared scale, not a
        # fair comparison the way aligning the two columns within one
        # row is). Left (signed) symmetric about 0, right (abs) floored
        # at 0, same reasoning/helper as the main figure's own
        # twin-axis panels.
        left_ylim, right_ylim = _symmetric_left_zero_right_ylim([ax_t, ax_dt], [twin_t, twin_dt])
        ax_t.set_ylim(left_ylim)
        ax_dt.set_ylim(left_ylim)
        twin_t.set_ylim(right_ylim)
        twin_dt.set_ylim(right_ylim)

    # No EULER-ONLY MODE banner here, unlike the main figure -- neither
    # dz0 nor dz0/dt depends on that mode at all (see this section's
    # own opening comment), so showing that banner would incorrectly
    # imply this figure is also about the euler-only-vs-full
    # distinction.
    fig2.suptitle(f"{_title_text}\nground-truth dz0 and dz0/dt [1e-3]", fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(dz0dt_output_path, dpi=120)
    plt.close(fig2)
    print(f"Saved figure to {dz0dt_output_path}")



def check_parameter_dependence(
    lds_checkpoint_path: Path, min_step: int | None = None, min_stdev_phi: float | None = None,
    min_passing_steps: int | None = None,
    base_path: Path | None = None, size: int | None = None,
    ae_stats_weight: float | None = None, hidden_dim: int = 256, n_hidden_layers: int = 2,
    condition_on_theta: bool | None = None,
    euler_only: bool | None = None,
    output_path: Path | None = None, dz0dt_output_path: Path | None = None,
    dt_dependence_output_path: Path | None = None, decode: bool = False, device: str | None = None,
) -> Path:
    """Saves three figures: parameter_dependence.png (temperature, noise,
    per-run scatter), dt_dependence.png (error/|error| vs dt, and,
    when decode=True, pixel-space error vs dt), and dz0dt.png
    (ground-truth dz0/dz0dt vs t and dt, independent of euler_only).
    Returns parameter_dependence.png's own path (the other two are
    still written to disk either way; see dt_dependence_output_path/
    dz0dt_output_path to control where).

    decode=False (default) skips pixel-space entirely -- no decoder
    forward passes in the main evaluation loop, no pixel-space console
    diagnostics (dt-decade table, temperature/noise/length_scale binned
    summaries all drop their "mean pixel_loss" column), and
    dt_dependence.png's own pixel-space panel isn't built at all. Set
    True to include it, at the cost of the extra decoder calls.

    lds_checkpoint_path may ALSO be a stage-1/1b/2 (AE-family)
    checkpoint, not just a real stage-3 one -- see
    checkpoint_identification.ensure_lds_checkpoint's own docstring for
    the full mechanism (an ephemeral, UNTRAINED f_theta gets trained
    against it for epochs=0). base_path/size are REQUIRED (and
    otherwise unused) specifically for this AE-family-conversion path;
    a real stage-3 checkpoint needs neither.

    euler_only: None (default) auto-detects -- True whenever
    lds_checkpoint_path actually got converted (an AE-family checkpoint
    was given, so f_theta is untrained and "full" is meaningless), False
    otherwise (a real stage-3 checkpoint, both euler-only and full shown
    as usual). Every panel/fit/print that would otherwise report on
    "full" (f_theta-corrected) numbers reports on euler-only ones
    instead in this mode, and the full-vs-euler-only COMPARISONS
    themselves (which have nothing to compare against here) are
    skipped rather than printed as a meaningless ratio of 1.0. Force
    True/False explicitly to override the auto-detection either way
    (e.g. to see the euler-only-only report for a real stage-3
    checkpoint too, for a same-scale comparison against an AE-family
    run).
    """
    ctx = _load_models_and_dataset(
        lds_checkpoint_path, min_step, min_stdev_phi, min_passing_steps, base_path, size,
        ae_stats_weight, hidden_dim, n_hidden_layers, condition_on_theta, euler_only,
        output_path, dz0dt_output_path, dt_dependence_output_path, device,
    )
    results = _evaluate_windows(ctx.dataset, ctx.f_theta, ctx.ae_decoder, ctx.device,
                                 decode, ctx.euler_only)
    stats = _print_summary_statistics(results, ctx.ae_config, decode, ctx.euler_only,
                                       dataset=ctx.dataset, device=ctx.device)
    _build_and_save_figures(results, stats, ctx.lds_checkpoint_path, ctx.output_path,
                             ctx.dz0dt_output_path, ctx.dt_dependence_output_path,
                             decode, ctx.euler_only)
    return ctx.output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lds-checkpoint", type=Path, required=True,
            help="no default -- multiple LDS variants can now coexist under "
                 "../checkpoints/stage3/. ALSO accepts a stage-1/1b/2 checkpoint -- see "
                 "check_parameter_dependence()'s own docstring for what that does and, "
                 "importantly, which parts of the report become uninformative in that mode")
    parser.add_argument("--min-step", type=int, default=None,
                         help="default: whatever the checkpoint's own saved data_config used")
    parser.add_argument("--min-stdev-phi", type=float, default=None,
                         help="default: whatever the checkpoint's own saved data_config used")
    parser.add_argument("--min-passing-steps", type=int, default=None,
                         help="default: whatever the checkpoint's own saved data_config used "
                              "(None for checkpoints trained before this parameter existed)")
    parser.add_argument("--base-path", type=Path, default=None,
                         help="REQUIRED only if --lds-checkpoint is a stage-1/1b/2 checkpoint "
                              "(needed to build the ephemeral stage-3 wrapper). The SWEEP ROOT "
                              "(e.g. '../datasets'), NOT including the size-specific subdirectory "
                              "-- train_lds() appends '{size}x{size}' itself; including it here "
                              "too doubles it")
    parser.add_argument("--size", type=int, default=None,
                         help="only used if --lds-checkpoint is a stage-1/1b/2 checkpoint -- "
                              "auto-derived from that checkpoint's own saved config if omitted, "
                              "so only needed to override or if that checkpoint predates this field")
    parser.add_argument("--ae-stats-weight", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256,
                         help="only affects the ephemeral f_theta built for an AE-family "
                              "checkpoint -- arbitrary, since that network is never trained here")
    parser.add_argument("--n-hidden-layers", type=int, default=2, help="see --hidden-dim")
    parser.add_argument("--condition-on-theta", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--euler-only", action=argparse.BooleanOptionalAction, default=None,
                         help="default: auto-detect (True if --lds-checkpoint got converted from "
                              "an AE-family checkpoint, False for a real stage-3 one). Force "
                              "either way to override -- e.g. --euler-only on a real stage-3 "
                              "checkpoint too, for a same-scale comparison against an AE-family run")
    parser.add_argument("--output", type=Path, default=None,
            help="default: <repo root>/output/stage3/<lds checkpoint name>-parameter_dependence.png")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    check_parameter_dependence(
        lds_checkpoint_path=args.lds_checkpoint, min_step=args.min_step,
        min_stdev_phi=args.min_stdev_phi, min_passing_steps=args.min_passing_steps,
        base_path=args.base_path, size=args.size, ae_stats_weight=args.ae_stats_weight,
        hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden_layers,
        condition_on_theta=args.condition_on_theta, euler_only=args.euler_only,
        output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()