"""Measure alpha, the Taylor-validity ratio that should REPLACE n_substeps.

WHAT ALPHA IS. Every sub-step advances z0 by a linear term and a curvature
correction:

    z0 <- z0 + z1*delta_t + f_theta*delta_t^2/2

alpha is the ratio of the second term to the first:

    alpha = |f_theta|*delta_t / |z1|

i.e. THE FRACTION OF THE DISPLACEMENT THAT THE CURVATURE CORRECTION
CONTRIBUTES. Small alpha means the step is inside the regime where the Taylor
expansion the whole scheme is built on actually holds; alpha near 1 means the
"correction" is as large as the thing it corrects, which is not a correction
at all.

WHY IT REPLACES n_substeps. n_substeps sets delta_t = Delta_t/n_substeps, so
it fixes the STEP and lets alpha fall where it may -- large wherever the
dynamics are fast, small wherever they are slow, and different for every
window. Solving the same equation the other way,

    n_substeps = ceil(|f_theta|*Delta_t / (alpha*|z1|))

fixes the Taylor validity and lets the step follow. They are one equation
read in two directions; the difference is which side is held constant. Holding
alpha constant is what makes a single setting valid across a dt range instead
of needing retuning every time max_dt moves.

WHY THIS SCRIPT EXISTS. alpha is not a number to guess. Two runs have already
bracketed it in delta_t units, at n_substeps=7 (delta_t ~ 71, stable for ~115
epochs and then escalating to a deadlock) and n_substeps=14 (delta_t ~ 36,
2000+ epochs with no spike skips at all). This script converts that bracket
into alpha by measuring |f_theta| and |z1| on real windows, so the controller
is calibrated from evidence rather than from a plausible-looking constant.

READING THE OUTPUT -- AND THE TRAP IN IT. A fixed n_substeps produces a
DISTRIBUTION of alpha (most windows near the median, a few at the tail); a
fixed alpha puts EVERY window at alpha. The two are therefore not comparable
quantile-for-quantile, and the natural-seeming reading is wrong: choosing
alpha at or below the UNSTABLE configuration's tail makes the typical window
as coarse as that configuration's worst one. Measured, at real cost: alpha=0.3
chosen that way (from an unstable p99 of 0.67) deadlocked within 9 epochs,
because its implied median sub-step count sat BELOW the configuration already
known to fail.

Anchor on the STABLE configuration's MEDIAN instead. Then every window gets at
least what the stable run's median window got, and the windows that need more
get strictly more than the stable run ever gave them.

Expect to go lower still, for a reason no formula shows: alpha is computed
from |f_theta|, the curvature the MODEL believes in, not the true one. With
f_theta capturing ~13% of z0_ddot (relative bias -87%, measured), the true
second-order term is several times what this criterion sees. That factor
shrinks as f_theta improves, which is exactly why alpha must be calibrated
against runs whose stability outcome is known rather than derived.

The report also inverts the relation and prices it: for a candidate alpha it
reports the sub-step COUNT distribution (mean, p95, max) and the fixed
n_substeps of equal total cost, so a choice of alpha can be read in the
familiar units before committing to it.

WHAT IT DOES NOT MEASURE. f_theta at the states an ADAPTIVE integrator would
actually visit -- those states do not exist until the controller does. Every
number here is measured at the encoder's own (z0, z1), i.e. on-trajectory,
which is the right reference for calibration but is not a simulation of the
controller.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from evaluation._latent_eval import (
    _load_ae_f_theta_and_dataset, _stage_folder_from_checkpoint_stem,
)
from utils.fits import fit_broken_power_law
from orchestration.paths import default_latent_cache_dir
from utils.plots import log_axis_ticks

_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/

# None is a MEANINGFUL value (caching off), so it cannot double as "not
# specified" -- same convention as check_dt_vs_time/check_parameter_dependence.
_UNSET_CACHE = object()

def default_alpha_figure_path(lds_checkpoint_path: Path) -> Path:
    """`output/<stage>/<checkpoint stem>-alpha.png`.

    Same derivation _latent_eval uses for its own figures, so this diagnostic
    lands beside -parameter_dependence.png / -dt_dependence.png rather than in
    checkpoints/. The stage folder comes from the STEM, which carries it even
    for a timestamped ancestor.
    """
    return (_PYTHON_ROOT.parent / "output"
            / _stage_folder_from_checkpoint_stem(lds_checkpoint_path)
            / f"{Path(lds_checkpoint_path).stem}-alpha.png")


# ONE colour per bracket n_substeps, used by EVERY panel that shows them.
# [0,0] and [0,1] display the same two configurations, and had independent
# colour cycles -- so n=7 was red in one panel and blue in the other, which
# invites reading the two panels as being about different things when they
# are two views of the same pair.
_BRACKET_COLOURS = ("tab:red", "tab:purple", "tab:brown", "tab:olive")

# The colour of the measured (adaptive) data, everywhere it appears. Neutral
# grey against the saturated reference lines: the eye reads shared colour as
# shared identity, and the whole point is that the measurement belongs to
# neither fixed-n reference.
_MEASURED_COLOUR = "0.35"

# max_dt is a DATA-SELECTION boundary, not a model quantity, so it is drawn
# the same way wherever a dt-like axis appears: black, dotted, thin. Every
# window in the population lies on one side of it by construction, which is
# exactly what makes it worth marking -- the eye otherwise reads the edge of
# the scatter as where the physics stops rather than where the filter cut.
_MAX_DT_STYLE = dict(color="k", ls=":", lw=1.2, zorder=3)


def _mark_max_dt(ax, max_dt, axis: str = "x") -> None:
    """Dotted line at max_dt on a dt-like axis, with a label."""
    if max_dt is None or not np.isfinite(max_dt):
        return
    (ax.axvline if axis == "x" else ax.axhline)(max_dt, **_MAX_DT_STYLE)
    if axis == "x":
        ax.text(max_dt, 0.02, f" max_dt={max_dt:g}", rotation=90, fontsize=7,
                 va="bottom", ha="right", color="k", transform=ax.get_xaxis_transform())


# The two configurations that bracket stability, as measured on the 128x128
# sweep at max_dt=500. Reported side by side so the alpha they imply can be
# read against the outcome they produced, rather than in the abstract.
_BRACKET_SUBSTEPS = (7, 14)


def collect_alpha(dataset, f_theta, device, max_windows_per_run: int | None = None,
                   batch_size: int = 512) -> dict:
    """Per-frame |f_theta|, |z1| and Delta_t over the dataset's own runs.

    Norms are over the WHOLE latent tensor (C,8,8), not per element: alpha is a
    statement about the step the integrator takes, and the integrator takes one
    step for the whole state. Per-channel ratios would be dominated by whichever
    channel happens to be near zero, which is a property of that channel and not
    of the step.

    Delta_t is the transition to the NEXT kept frame, so the last frame of each
    run is skipped -- it has no transition, and f_theta there would be measured
    against a step that is never taken.
    """
    rows_f, rows_z1, rows_dt, rows_t, rows_theta = [], [], [], [], []
    n_runs = len(dataset._run_data)
    for run_idx in range(n_runs):
        state = dataset._run_data[run_idx]
        deriv = dataset._run_data_deriv[run_idx]
        steps = dataset._run_steps[run_idx]
        scale = dataset._run_dt_scale[run_idx]
        theta = dataset._run_theta[run_idx]
        n = len(steps)
        if n < 2:
            continue
        limit = n - 1 if max_windows_per_run is None else min(n - 1, max_windows_per_run)
        z0_all = state[:limit].to(device)
        z1_all = deriv[:limit].to(device)
        theta_b = theta.to(device).unsqueeze(0).expand(limit, -1)
        f_norms = []
        with torch.no_grad():
            for start in range(0, limit, batch_size):
                stop = min(start + batch_size, limit)
                # f_theta's own field, NOT forward(): forward would fold in the
                # z1*dt term and the dt_cap, and alpha is about the raw
                # curvature field the controller would query.
                f_val = f_theta.f(z0_all[start:stop], z1_all[start:stop],
                                   theta_b[start:stop])
                f_norms.append(torch.linalg.vector_norm(
                    f_val.reshape(f_val.shape[0], -1), dim=1).cpu())
        f_norm = torch.cat(f_norms).numpy()
        z1_norm = torch.linalg.vector_norm(
            z1_all.reshape(limit, -1), dim=1).cpu().numpy()
        dts = np.array([(steps[i + 1] - steps[i]) * scale for i in range(limit)],
                        dtype=float)
        ts = np.array([steps[i] * scale for i in range(limit)], dtype=float)
        rows_f.append(f_norm)
        rows_z1.append(z1_norm)
        rows_dt.append(dts)
        rows_t.append(ts)
        rows_theta.append(np.full(limit, float(theta[0]) if theta.numel() else np.nan))
    if not rows_f:
        raise ValueError("no frames collected -- every run has fewer than 2 kept steps")
    return {"f_norm": np.concatenate(rows_f), "z1_norm": np.concatenate(rows_z1),
            "dt": np.concatenate(rows_dt), "t": np.concatenate(rows_t),
            "theta0": np.concatenate(rows_theta)}


def alpha_at_substeps(data: dict, n_substeps: int) -> np.ndarray:
    """alpha = |f_theta|*delta_t/|z1| with delta_t = Delta_t/n_substeps.

    |z1| == 0 yields inf rather than nan: a state with no velocity and nonzero
    curvature has NO valid step under this criterion, which is a real (if rare)
    statement and must not be silently dropped from a tail quantile.
    """
    delta_t = data["dt"] / n_substeps
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = data["f_norm"] * delta_t / data["z1_norm"]
    # |z1|=0 with |f|>0 ALREADY gives inf from the division itself -- an
    # earlier version "handled" it with an np.where that was pure dead code,
    # and the test guarding it passed vacuously. The case that genuinely needs
    # a decision is 0/0: no velocity AND no curvature, where the division
    # yields nan. Nothing is happening at such a state, so the step is
    # unbounded (alpha=0), NOT undefined -- nan would be dropped from every
    # quantile silently, turning "this state is trivially fine" into "this
    # state was never measured".
    alpha = np.where((data["z1_norm"] == 0) & (data["f_norm"] == 0), 0.0, alpha)
    return alpha


def substeps_for_alpha(data: dict, alpha: float) -> np.ndarray:
    """n_substeps = ceil(|f_theta|*Delta_t/(alpha*|z1|)), at least 1.

    The inverse of alpha_at_substeps, and the number a run would actually pay.
    Reported as a distribution because the MAX is what a batched implementation
    costs when a batch mixes windows -- the loop runs until every sample in the
    batch has arrived.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = data["f_norm"] * data["dt"] / (alpha * data["z1_norm"])
    raw = np.where(np.isfinite(raw), raw, np.inf)
    return np.maximum(np.ceil(raw), 1.0)


def step_size_table(data: dict, alpha: float, max_substeps: int = 256,
                     n_rollout_steps: int = 2, n_decades: int = 4) -> list[dict]:
    """Per-dt-decade statistics for delta_t, delta_t/t, n_substeps and DEPTH.

    The four quantities answer four different questions, and conflating them
    is how a run ends up flying blind:

      n_substeps   -- what it COSTS. Cost per batch is the batch MAXIMUM, not
                      the mean, because the masked loop runs until the last
                      sample arrives.
      delta_t      -- whether the step is inside the band where f_theta is
                      trustworthy (measured 50-100 on this sweep). A stable
                      run and an exploding one can share an alpha and differ
                      entirely here, because alpha fixes a RATIO and delta_t
                      is what the integrator actually takes.
      delta_t/t    -- whether the step tracks the physics' own clock.
                      Coarsening slows as t grows, so the natural timescale
                      grows with t; a delta_t/t that stays roughly CONSTANT
                      across decades means the criterion is following that
                      slowdown, and the cost of reaching t is logarithmic
                      rather than linear. A delta_t/t that FALLS with t means
                      the step is being refined faster than the physics slows,
                      and full-trajectory cost grows without bound.
      depth        -- n_substeps * n_rollout_steps, the number of chained
                      Jacobians in the BACKWARD pass. This is the quantity
                      that has no bearing on inference and dominates training:
                      at max_dt=2000 every batch reported an infinite gradient
                      norm with a perfectly finite loss, and at max_dt=1000 the
                      MEDIAN norm was 1e17-1e18 against ~3e3 at max_dt=500.
                      Depth is why raising max_substeps made things worse: a
                      larger cap permits a deeper graph.

    Decades are of Delta_t, since that is what max_dt selects on and what
    every other diagnostic in this project bins by.
    """
    counts = substeps_for_alpha(data, alpha)
    clamped = counts > max_substeps
    counts_eff = np.minimum(counts, float(max_substeps))
    delta_t = data["dt"] / counts_eff
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_over_t = np.where(data["t"] > 0, delta_t / data["t"], np.nan)

    rows = []
    for decade in range(1, n_decades + 1):
        lo, hi = 10.0 ** decade, 10.0 ** (decade + 1)
        m = (data["dt"] >= lo) & (data["dt"] < hi)
        if not m.any():
            continue
        finite_ratio = delta_over_t[m][np.isfinite(delta_over_t[m])]
        rows.append({
            "decade": f"1e{decade} - 1e{decade + 1}",
            "n": int(m.sum()),
            "dt_median": float(np.median(data["dt"][m])),
            "n_sub_mean": float(counts_eff[m].mean()),
            "n_sub_p95": float(np.quantile(counts_eff[m], 0.95)),
            "n_sub_max": float(counts_eff[m].max()),
            "delta_t_median": float(np.median(delta_t[m])),
            "delta_t_min": float(delta_t[m].min()),
            "delta_over_t_median": float(np.median(finite_ratio)) if finite_ratio.size else float("nan"),
            "clamped_pct": 100.0 * float(clamped[m].mean()),
            "depth_max": float(counts_eff[m].max() * n_rollout_steps),
        })
    return rows


def _quantiles(values: np.ndarray, qs=(0.5, 0.9, 0.95, 0.99, 1.0)) -> dict:
    """Quantiles as p50/p90/p95/p99/p100 keys, plus mean and an infinity count.

    The quantile LIST and the keys the report asks for must agree: an earlier
    version omitted 0.95 while the cost table printed p95, so that column was
    silently nan on every row. Unit tests could not catch it -- they call the
    numeric functions, not the report -- so the guard is a test that renders
    the report and asserts no nan reaches it.
    """
    finite = values[np.isfinite(values)]
    out = {"n": int(values.size), "n_infinite": int(values.size - finite.size)}
    if finite.size:
        out["mean"] = float(finite.mean())
        for q in qs:
            out[f"p{int(q * 100)}"] = float(np.quantile(finite, q))
    return out


def _plot(data: dict, report_alpha: float, max_substeps: int, n_rollout_steps: int,
           candidate_alphas: tuple[float, ...], bracket_substeps: tuple[int, ...],
           output_path, max_dt: float | None = None) -> None:
    """Six panels: what the criterion DOES, not just what it costs.

    The tables answer "how many sub-steps" well enough. What they cannot show
    is the SHAPE of each distribution, and every mistake in this project's
    alpha calibration has been a shape mistake read off a summary statistic:
    alpha=0.3 was chosen from a tail quantile as though a fixed alpha produced
    a distribution, and delta_t was inferred as median(Delta_t)/median(n),
    which is not the median of the ratio and was wrong by an order of
    magnitude (36 assumed against 4-16 measured).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.minimum(substeps_for_alpha(data, report_alpha), float(max_substeps))
    delta_t = data["dt"] / counts
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # THE ZERO-FIELD CASE, guarded where it actually bites. f_theta's final
    # layer is zero-initialised, so a fresh stage 3a has f == 0 EVERYWHERE:
    # every count is 1, every alpha is 0, and matplotlib raises "Data has no
    # positive values, and therefore cannot be log-scaled" partway through --
    # leaving the caller with tables and no figure, on precisely the
    # checkpoint someone runs this against first.
    #
    # Only TWO panels can go empty: the alpha histogram (alpha == 0
    # everywhere) and the |f|/|z1| scatter (the ratio is 0 everywhere). Every
    # other axis plots Delta_t, t, delta_t or a sub-step count, all of which
    # are positive by construction whatever f_theta does. An earlier version
    # wrapped every set_*scale in a "log only if positive" helper; removing
    # that helper broke no test, because no reachable input makes it bind.
    def _nothing_to_show(ax, message: str) -> None:
        ax.text(0.5, 0.5, message, ha="center", va="center",
                 transform=ax.transAxes, fontsize=9)

    # [0,0] delta_t vs Delta_t. THE panel that corrects the intuition: a
    # constant n_substeps is a straight line here, and the criterion is not.
    ax = axes[0, 0]
    # Neutral grey for the DATA, saturated colours for the two reference
    # LINES. They were all blue: the eye reads shared colour as shared
    # identity, so the scatter looked like it belonged to the n=7 line when
    # the whole point of the panel is that it belongs to neither.
    ax.scatter(data["dt"], delta_t, s=4, alpha=0.2, color=_MEASURED_COLOUR, linewidths=0,
                label=f"alpha={report_alpha:g} (measured)")
    dts = np.array(sorted(set(np.round(data["dt"]))))
    for n, colour in zip(bracket_substeps, _BRACKET_COLOURS):
        ax.plot(dts, dts / n, "--", lw=1.4, color=colour, label=f"fixed n_substeps={n}")
    ax.axhspan(50, 100, color="tab:green", alpha=0.12, zorder=0)
    # Right-anchored in AXES coordinates: left-anchored at the data minimum it
    # ran under the legend on the real figure, where the band sits at the top
    # of the range and the legend is top-left.
    ax.text(0.98, 0.5, "f_theta trustworthy band ", fontsize=7, color="tab:green",
             ha="right", va="center", transform=ax.get_yaxis_transform(),
             clip_on=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_axis_ticks(ax.xaxis, data["dt"].min(), data["dt"].max())
    log_axis_ticks(ax.yaxis, delta_t.min(), delta_t.max())
    _mark_max_dt(ax, max_dt)
    ax.set_xlabel("Delta_t (transition)")
    ax.set_ylabel("delta_t (step actually taken)")
    ax.set_title(f"delta_t vs Delta_t at alpha={report_alpha:g}\n"
                  f"fixed n would be a straight line; this is not")
    ax.legend(fontsize=7, loc="upper left")

    # [0,1] the distribution-vs-point confusion, drawn.
    ax = axes[0, 1]
    # alpha on a LOG AXIS, not log10(alpha) as the quantity: the reader wants
    # to see "0.1", not "-1". Bins are still uniform in log space -- that is a
    # property of the binning, not something the axis label should expose.
    drew = False
    finite_alphas = [a[np.isfinite(a) & (a > 0)]
                      for a in (alpha_at_substeps(data, n) for n in bracket_substeps)]
    spread = np.concatenate([a for a in finite_alphas if a.size]) if any(
        a.size for a in finite_alphas) else np.array([])
    if spread.size:
        bins = np.logspace(np.log10(spread.min()), np.log10(spread.max()), 60)
        for n, a, colour in zip(bracket_substeps, finite_alphas, _BRACKET_COLOURS):
            if a.size:
                # SAME colour as [0,0]: these are the same two configurations,
                # seen as a step-size curve there and as a distribution here.
                ax.hist(a, bins=bins, histtype="step", lw=1.4, color=colour,
                         label=f"fixed n_substeps={n}")
                drew = True
        ax.set_xscale("log")
        log_axis_ticks(ax.xaxis, spread.min(), spread.max())
    ax.set_xlabel("alpha")
    ax.set_ylabel("windows")
    # The two curves are the SAME distribution shifted: alpha_at_n = D/n with
    # D = |f|*Delta_t/|z1|, so changing n only translates it on a log axis. I
    # read the n=7 curve as "clearly bimodal" and the n=14 one as "much less
    # so", which is impossible by construction -- an artifact of comparing two
    # translations by eye. The title now says what varies (n) and what cannot
    # (the shape), so the panel cannot be misread that way again.
    ax.set_title("alpha is a DISTRIBUTION under fixed n,\n"
                  "a single point under fixed alpha (same shape, shifted by n)")
    if drew:
        ax.axvline(report_alpha, color="k", lw=2,
                    label=f"fixed alpha={report_alpha:g} (every window)")
        ax.legend(fontsize=7)
    else:
        _nothing_to_show(ax, "alpha == 0 everywhere\n(f_theta is identically zero:\n"
                              "a fresh, zero-initialised stage 3a)")

    # [0,2] cost: mean is not what a batch pays.
    ax = axes[0, 2]
    means, p95s, maxs = [], [], []
    for a in candidate_alphas:
        c = substeps_for_alpha(data, a)
        c = c[np.isfinite(c)]
        means.append(c.mean())
        p95s.append(np.quantile(c, 0.95))
        maxs.append(c.max())
    ax.plot(candidate_alphas, means, "o-", label="mean")
    ax.plot(candidate_alphas, p95s, "s-", label="p95")
    ax.plot(candidate_alphas, maxs, "^-", label="max (what a mixed batch pays)")
    ax.axhline(max_substeps, color="tab:red", ls=":", lw=1,
                label=f"max_substeps={max_substeps}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_axis_ticks(ax.xaxis, min(candidate_alphas), max(candidate_alphas))
    log_axis_ticks(ax.yaxis, min(means), max(maxs))
    ax.invert_xaxis()   # tighter alpha to the right, i.e. cost rising rightward
    ax.set_xlabel("alpha (tighter ->)")
    ax.set_ylabel("n_substeps")
    ax.set_title("cost of alpha: the gap between mean and max\nis what batching by dt would recover")
    ax.legend(fontsize=7)

    # [1,0] the sublinearity question.
    ax = axes[1, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(data["t"] > 0, delta_t / data["t"], np.nan)
    ok = np.isfinite(ratio) & (ratio > 0)
    ax.scatter(data["t"][ok], ratio[ok], s=4, alpha=0.15, color="tab:purple", linewidths=0)
    # BINNED at 6 per decade, not 1. Decade medians gave two or three points
    # over this range -- enough to state a slope, far too few to show that the
    # slope CHANGES, which is what the scatter plainly does.
    t_ok, ratio_ok = data["t"][ok], ratio[ok]
    knee = p1 = p2 = float("nan")
    if t_ok.size:
        edges = 10.0 ** np.arange(np.floor(np.log10(t_ok.min())),
                                   np.ceil(np.log10(t_ok.max())) + 1 / 6, 1 / 6)
        mids, meds = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (t_ok >= lo) & (t_ok < hi)
            if m.sum() > 5:
                mids.append(np.sqrt(lo * hi))
                meds.append(np.median(ratio_ok[m]))
        if mids:
            ax.plot(mids, meds, "o", color="k", ms=4, label="median per 1/6 decade")
        # A BROKEN POWER LAW rather than a spline: the knee is the quantity of
        # interest (where the step stops tracking the slowdown), and a spline
        # would draw the bend beautifully while reporting nothing about it.
        # Fitted to the raw points, not the medians, so the binning affects
        # only what is drawn.
        knee, p1, p2, sse_b, sse_s = fit_broken_power_law(t_ok, ratio_ok)
        gain = 1.0 - sse_b / sse_s if (np.isfinite(knee) and sse_s > 0) else 0.0
        # A broken fit has two extra parameters and can only fit better, so a
        # small improvement means "one power law drawn on noisy data", not a
        # regime change. Below this the panel says so rather than reporting a
        # knee position the data does not support -- the failure mode being
        # that a knee, once drawn with a dotted line and a number, gets
        # believed. Validated on synthetic data: a genuine flat-then-sloped
        # break buys 92%, a pure single power law 0.3%.
        _MIN_BREAK_GAIN = 0.25
        if np.isfinite(knee) and gain >= _MIN_BREAK_GAIN:
            grid = np.logspace(np.log10(t_ok.min()), np.log10(t_ok.max()), 200)
            # Continuous by construction; anchor on the fitted knee value.
            anchor = np.interp(knee, mids, meds) if mids else np.median(ratio_ok)
            curve = np.where(grid < knee, anchor * (grid / knee) ** p1,
                              anchor * (grid / knee) ** p2)
            ax.plot(grid, curve, "-", color="tab:red", lw=1.6,
                     label=f"broken power law (knee t={knee:.3g})")
            ax.axvline(knee, color="tab:red", ls=":", lw=1)
            ax.set_title(f"does the step follow the physics' clock?\n"
                          f"delta_t/t ~ t^{p1:+.2f} then t^{p2:+.2f} at t={knee:.3g}  "
                          f"(break explains {100 * gain:.0f}% of the residual)")
        else:
            # Single power law, and say which: the exponent is the answer to
            # the panel's own question (0 = the step tracks the slowdown
            # exactly; -1 = it does not track it at all).
            single = np.polyfit(np.log(t_ok), np.log(ratio_ok), 1)
            grid = np.logspace(np.log10(t_ok.min()), np.log10(t_ok.max()), 200)
            ax.plot(grid, np.exp(single[1]) * grid ** single[0], "-",
                     color="tab:red", lw=1.6, label="single power law")
            ax.set_title(f"does the step follow the physics' clock?\n"
                          f"delta_t/t ~ t^{single[0]:+.2f}  "
                          f"(a break would add only {100 * gain:.0f}%: one regime)")
    if not t_ok.size:
        ax.set_title("does the step follow the physics' clock?\n"
                      "no window has t > 0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_axis_ticks(ax.xaxis, t_ok.min(), t_ok.max()) if t_ok.size else None
    ax.set_xlabel("t (time, = step * dt_scale)")
    ax.set_ylabel("delta_t / t")
    ax.legend(fontsize=7)

    # [1,1] what actually drives the count.
    ax = axes[1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        drive = data["f_norm"] / data["z1_norm"]
    ok = np.isfinite(drive) & (drive > 0)
    if ok.any():
        sc = ax.scatter(data["dt"][ok], drive[ok], s=4, alpha=0.2,
                         c=np.log10(np.maximum(data["t"][ok], 1.0)),
                         cmap="viridis", linewidths=0)
        fig.colorbar(sc, ax=ax, label="log10 t")
        ax.set_yscale("log")
        # The exponent is the whole content of this panel: n ~ |f|/|z1| * Delta_t,
        # so if the drive falls as Delta_t^p then the COUNT grows as
        # Delta_t^(1+p) -- and 1+p is what decides whether widening max_dt is
        # affordable. Reading it off by eye is what produced my own "-0.6"
        # guess; the fit states it.
        fit = np.polyfit(np.log(data["dt"][ok]), np.log(drive[ok]), 1)
        grid = np.logspace(np.log10(data["dt"][ok].min()),
                            np.log10(data["dt"][ok].max()), 200)
        ax.plot(grid, np.exp(fit[1]) * grid ** fit[0], "-", color="tab:red", lw=1.6,
                 label=(f"|f|/|z1| ~ {np.exp(fit[1]):.3g} Delta_t^{fit[0]:+.2f}"
                        f"  =>  n ~ Delta_t^{1 + fit[0]:+.2f}"))
        ax.legend(fontsize=7, loc="lower left")
    else:
        _nothing_to_show(ax, "|f_theta| == 0 everywhere")
    ax.set_xscale("log")
    log_axis_ticks(ax.xaxis, data["dt"].min(), data["dt"].max())
    if ok.any():
        log_axis_ticks(ax.yaxis, drive[ok].min(), drive[ok].max())
    _mark_max_dt(ax, max_dt)
    ax.set_xlabel("Delta_t")
    ax.set_ylabel("|f_theta| / |z1|   (1/time)")
    ax.set_title("the count is driven by |f|/|z1|, NOT by Delta_t\n"
                  "-- which is why short-dt windows still sub-step hard")

    # [1,2] backward-graph depth, the training-only cost.
    ax = axes[1, 2]
    depth = counts * n_rollout_steps
    ax.scatter(data["dt"], depth, s=4, alpha=0.15, color="tab:orange", linewidths=0)
    ax.axhline(264, color="k", ls="--", lw=1,
                label="depth reached at max_dt=500 (trains cleanly)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_axis_ticks(ax.xaxis, data["dt"].min(), data["dt"].max())
    log_axis_ticks(ax.yaxis, depth.min(), depth.max())
    _mark_max_dt(ax, max_dt)
    ax.set_xlabel("Delta_t")
    ax.set_ylabel(f"depth = n_substeps x {n_rollout_steps}")
    ax.set_title("chained Jacobians in the BACKWARD pass\n"
                  "inference does not pay this; training does")
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved figure to {output_path}")


def check_alpha(lds_checkpoint_path: Path, base_path: Path | None = None,
                 size: int | None = None, min_step: int | None = None,
                 min_stdev_phi: float | None = None, min_passing_steps: int | None = None,
                 max_dt: float | None = None, device: str | None = None,
                 candidate_alphas: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02),
                 bracket_substeps: tuple[int, ...] = _BRACKET_SUBSTEPS,
                 report_alpha: float | None = None, max_substeps: int | None = None,
                 n_rollout_steps: int = 2, output_path=None,
                 max_windows_per_run: int | None = None,
                 latent_cache_dir=_UNSET_CACHE) -> dict:
    """Measure alpha on a checkpoint's own test population and print the report."""
    ctx = _load_ae_f_theta_and_dataset(
        lds_checkpoint_path, min_step, min_stdev_phi, min_passing_steps,
        base_path, size, None, 256, 2, None, None, device,
        max_dt=max_dt,
        latent_cache_dir=(default_latent_cache_dir(_PYTHON_ROOT)
                           if latent_cache_dir is _UNSET_CACHE else latent_cache_dir),
    )
    # 7-tuple; positional for the same reason check_dt_vs_time is: binding
    # f_theta to `dataset` by name would fail far from here.
    resolved_device, _, _, _, dataset, _, f_theta = ctx
    f_theta.eval()

    # The alpha and max_substeps the CHECKPOINT itself was trained with, unless
    # overridden -- so the table describes the run that produced these weights
    # rather than a hypothetical. Falls back to the middle candidate for a
    # checkpoint that predates alpha.
    if report_alpha is None:
        report_alpha = getattr(f_theta, "alpha", None)
    if report_alpha is None:
        report_alpha = sorted(candidate_alphas)[len(candidate_alphas) // 2]
    if max_substeps is None:
        max_substeps = int(getattr(f_theta, "max_substeps", 256) or 256)

    data = collect_alpha(dataset, f_theta, resolved_device,
                          max_windows_per_run=max_windows_per_run)
    n = data["f_norm"].size
    print(f"\n{n} frames from {len(dataset._run_data)} runs "
          f"(each with its own transition to the next kept frame)\n")

    print("  |f_theta| and |z1| over the whole latent tensor:")
    for name, key in (("|f_theta|", "f_norm"), ("|z1|", "z1_norm"), ("Delta_t", "dt")):
        q = _quantiles(data[key])
        print(f"    {name:>10}: mean {q.get('mean', float('nan')):.4g}   "
              f"median {q.get('p50', float('nan')):.4g}   "
              f"p99 {q.get('p99', float('nan')):.4g}   "
              f"max {q.get('p100', float('nan')):.4g}")

    print("\n  alpha implied by each n_substeps (delta_t = Delta_t/n_substeps):")
    print(f"    {'n_sub':>6} {'delta_t~':>10} {'mean':>10} {'median':>10} "
          f"{'p90':>10} {'p99':>10} {'max':>10}")
    by_substeps = {}
    median_dt = float(np.median(data["dt"]))
    for n_sub in bracket_substeps:
        a = alpha_at_substeps(data, n_sub)
        q = _quantiles(a)
        by_substeps[n_sub] = q
        print(f"    {n_sub:>6} {median_dt / n_sub:>10.4g} {q.get('mean', np.nan):>10.4g} "
              f"{q.get('p50', np.nan):>10.4g} {q.get('p90', np.nan):>10.4g} "
              f"{q.get('p99', np.nan):>10.4g} {q.get('p100', np.nan):>10.4g}")
    print("    (delta_t~ uses the MEDIAN Delta_t; individual windows vary over decades)")

    print("\n  cost of each candidate alpha, in the familiar units:")
    print(f"    {'alpha':>8} {'mean n_sub':>12} {'p95':>8} {'max':>8} "
          f"{'equal-cost fixed n_substeps':>30}")
    by_alpha = {}
    for alpha in candidate_alphas:
        counts = substeps_for_alpha(data, alpha)
        q = _quantiles(counts)
        by_alpha[alpha] = q
        equal_cost = q.get("mean", float("nan"))
        print(f"    {alpha:>8.3g} {q.get('mean', np.nan):>12.2f} "
              f"{q.get('p95', np.nan):>8.0f} {q.get('p100', np.nan):>8.0f} "
              f"{equal_cost:>30.1f}")
    print("    (equal-cost = the fixed n_substeps with the same TOTAL f_theta evaluations;\n"
          "     the gap between it and max is what adaptivity buys -- and what a batched\n"
          "     implementation loses if a batch mixes windows, since the loop runs until\n"
          "     the LAST sample in the batch has arrived)")

    # THE STEP-SIZE BREAKDOWN, per dt decade, for the alpha actually in use.
    # The aggregate cost table above says what a run PAYS; this says what step
    # it TAKES, whether that step tracks the physics' own clock, and how deep
    # the backward graph gets -- the three things a run cannot otherwise see.
    print(f"\n  step size at alpha={report_alpha:g} "
          f"(max_substeps={max_substeps}, n_rollout_steps={n_rollout_steps}):")
    print(f"    {'dt decade':>12} {'n':>7} {'dt~':>8} {'n_sub':>7} {'p95':>6} "
          f"{'max':>6} {'delta_t':>9} {'min':>8} {'delta_t/t':>10} "
          f"{'clamp%':>7} {'depth':>7}")
    rows = step_size_table(data, report_alpha, max_substeps=max_substeps,
                            n_rollout_steps=n_rollout_steps)
    for r in rows:
        print(f"    {r['decade']:>12} {r['n']:>7} {r['dt_median']:>8.0f} "
              f"{r['n_sub_mean']:>7.1f} {r['n_sub_p95']:>6.0f} {r['n_sub_max']:>6.0f} "
              f"{r['delta_t_median']:>9.1f} {r['delta_t_min']:>8.2f} "
              f"{r['delta_over_t_median']:>10.2e} {r['clamped_pct']:>6.1f}% "
              f"{r['depth_max']:>7.0f}")
    print("    (delta_t = Delta_t/n_sub, the step ACTUALLY taken -- f_theta was measured\n"
          "     trustworthy over roughly 50 <= delta_t <= 100 on this sweep.\n"
          "     delta_t/t roughly CONSTANT across decades = the step follows the physics'\n"
          "     own slowdown, so reaching late t costs logarithmically many evaluations;\n"
          "     FALLING with t = the step outpaces the slowdown and the cost grows.\n"
          "     depth = max n_sub x n_rollout_steps = chained Jacobians in the BACKWARD\n"
          "     pass. Inference does not pay it; training does, and it is what overflows:\n"
          "     max_dt=2000 gave infinite grad norms on EVERY batch with finite losses,\n"
          "     max_dt=1000 a MEDIAN norm of 1e17-1e18 against ~3e3 at max_dt=500.\n"
          "     clamp% > 0 means max_substeps overrode alpha on those windows -- they ran\n"
          "     COARSER than asked, so the alpha guarantee does not hold there.)")

    if len(bracket_substeps) >= 2:
        lo_sub, hi_sub = max(bracket_substeps), min(bracket_substeps)
        stable, unstable = by_substeps[lo_sub], by_substeps[hi_sub]
        print(f"\n  -> THE BRACKET. n_substeps={lo_sub} was stable over 2000+ epochs; "
              f"n_substeps={hi_sub} escalated to a deadlock after ~115.")
        print(f"     stable   (n={lo_sub}): median alpha {stable.get('p50', np.nan):.4g}, "
              f"p99 {stable.get('p99', np.nan):.4g}")
        print(f"     unstable (n={hi_sub}): median alpha {unstable.get('p50', np.nan):.4g}, "
              f"p99 {unstable.get('p99', np.nan):.4g}")
        print(f"\n     ANCHOR ON THE STABLE RUN'S MEDIAN ({stable.get('p50', np.nan):.3g}), NOT on "
              f"either run's tail.\n"
              f"     A fixed n_substeps produces a DISTRIBUTION of alpha -- most windows near\n"
              f"     the median, a few at the tail. A fixed alpha puts EVERY window at alpha.\n"
              f"     So choosing alpha at the stable run's p99 makes the typical window as\n"
              f"     coarse as that run's WORST one, which is a different and much less stable\n"
              f"     configuration than the one that was observed to work. Measured: alpha=0.3,\n"
              f"     picked from the unstable run's tail, deadlocked within 9 epochs -- its\n"
              f"     implied median sub-step count was below the failing configuration's.\n"
              f"     At alpha = the stable median, every window gets at least what the stable\n"
              f"     run's median window got, and the hard windows get strictly more.")
        print("\n     AND EXPECT TO GO LOWER STILL. alpha is computed from |f_theta|, i.e. the\n"
              "     curvature the MODEL believes in, not the true one. Where f_theta\n"
              "     underestimates z0_ddot (measured relative bias -87% on the 128 sweep, so\n"
              "     f_theta captures ~13% of it), the true second-order term is several times\n"
              "     what this criterion sees, and the realised Taylor ratio is correspondingly\n"
              "     worse than the alpha requested. That factor shrinks as f_theta improves,\n"
              "     which is why alpha is calibrated against known-stable runs rather than\n"
              "     reasoned to from the formula.")

    if output_path is not None:
        # output/<stage>/<stem>-alpha.png, the layout every other diagnostic
        # in this package already writes to -- NOT alongside the checkpoint.
        # An earlier version defaulted next to the .pt, which scatters figures
        # through checkpoints/ where nothing else puts them and breaks the
        # one-folder-per-stage grouping the rest of output/ relies on. The
        # stage comes from the checkpoint's own stem, so a timestamped
        # ancestor (…-stage3b-20260806_18h10.pt) still lands in stage3b/.
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _plot(data, report_alpha, max_substeps, n_rollout_steps,
               tuple(candidate_alphas), tuple(bracket_substeps), output_path,
               max_dt=getattr(dataset, "max_dt", None))

    return {"data": data, "by_substeps": by_substeps, "by_alpha": by_alpha,
            "median_dt": median_dt, "step_size_rows": rows,
            "report_alpha": report_alpha, "max_substeps": max_substeps}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lds-checkpoint", type=Path, required=True)
    p.add_argument("--base-path", type=Path, default=None)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--min-step", type=int, default=None)
    p.add_argument("--min-stdev-phi", type=float, default=None)
    p.add_argument("--min-passing-steps", type=int, default=None)
    p.add_argument("--max-dt", type=float, default=None,
                    help="defaults to the checkpoint's own data_config value, so the "
                         "measurement matches the population the checkpoint was trained on")
    p.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 0.2, 0.1, 0.05, 0.02])
    p.add_argument("--bracket-substeps", type=int, nargs="+", default=list(_BRACKET_SUBSTEPS),
                    help="the n_substeps values whose stability outcome is known")
    p.add_argument("--report-alpha", type=float, default=None,
                    help="alpha for the per-decade step-size table (default: the "
                         "checkpoint's own, so the table describes the run that produced "
                         "these weights)")
    p.add_argument("--max-substeps", type=int, default=None,
                    help="default: the checkpoint's own")
    p.add_argument("--n-rollout-steps", type=int, default=2,
                    help="only used to report backward-graph DEPTH (n_sub x steps)")
    p.add_argument("--max-windows-per-run", type=int, default=None)
    p.add_argument("--output", type=Path, default=None,
                    help="figure path (default: output/<stage>/<stem>-alpha.png, "
                         "beside the other diagnostics; --no-figure to skip)")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--no-latent-cache", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    check_alpha(
        args.lds_checkpoint, base_path=args.base_path, size=args.size,
        min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        min_passing_steps=args.min_passing_steps, max_dt=args.max_dt,
        device=args.device, candidate_alphas=tuple(args.alphas),
        bracket_substeps=tuple(args.bracket_substeps),
        report_alpha=args.report_alpha, max_substeps=args.max_substeps,
        n_rollout_steps=args.n_rollout_steps,
        output_path=(None if args.no_figure else
                      (args.output or default_alpha_figure_path(args.lds_checkpoint))),
        max_windows_per_run=args.max_windows_per_run,
        latent_cache_dir=None if args.no_latent_cache else _UNSET_CACHE,
    )


if __name__ == "__main__":
    main()
