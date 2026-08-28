"""
Visualization helpers for phase-field snapshots -- displaying raw binary
files directly, without going through the solver's PNG export.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import numpy as np

from . import load_datasets as load


LOSS_FIGURE_EVERY = 10


def log_axis_ticks(axis, lo: float, hi: float, mantissas=None) -> None:
    """
    Label a log axis at {1,2,3,5}x10^k rather than at decades only.

    Matplotlib's default LogLocator labels decades, so an axis spanning
    less than one decade -- which several of this project's do (-a(T)/b
    runs 0.005 to 0.45; check_alpha's Delta_t runs 12 to 500) -- ends up
    with a SINGLE labelled tick, leaving the reader no way to read a value
    off the plot at all.

    Shared rather than duplicated: check_stdev_phi_time defined it first,
    check_alpha needed exactly the same thing, and a second caller is this
    project's bar for extraction.
    """
    if not (hi > lo > 0):
        return
    if mantissas is None:
        # THINNED BY SPAN. {1,2,3,5} per decade is right for the sub-decade
        # case this was written for, and unreadable over three decades -- the
        # alpha histogram came out with "0.00050.0010.0020.003" run together.
        # Fewer labels over a wider range keeps every one legible, which is
        # the whole point: a labelled tick nobody can read is worse than the
        # decade-only default it replaced.
        decades = math.log10(hi / lo)
        mantissas = ((1,) if decades > 2.5 else
                      (1, 3) if decades > 1.2 else
                      (1, 2, 3, 5))
    ticks = [m * 10.0 ** k
              for k in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
              for m in mantissas]
    ticks = [t for t in ticks if lo <= t <= hi]
    if len(ticks) < 2:
        return
    axis.set_ticks(ticks)
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(mticker.NullLocator())



def should_write_loss_figure(epoch: int, log_every_epoch: bool, every: int = LOSS_FIGURE_EVERY) -> bool:
    """
    Whether to regenerate the per-epoch loss figures THIS epoch.

    Keyed off log_every_epoch -- the SAME flag each stage already uses to
    decide whether to print every epoch's own console line or only the
    epochs a checkpoint was saved on (see e.g. train_stage1's own "if
    log_every_epoch or saved_this_epoch"). Reused here rather than a
    second, independent flag, since it already encodes exactly the
    distinction that matters: is this run being watched closely enough
    that a stale plot is a real cost?

    log_every_epoch=True: write EVERY epoch. Measured cost is real in
    absolute terms (~300 ms for loss_curve plus ~390 ms for
    loss_component_scatter, ~0.7 s/epoch -- see fig.savefig()/
    tight_layout() being the dominant cost, not the plotting itself:
    8 ms to build a figure, 87 ms to build AND save it) but log_every_epoch
    is what a user watching a SLOW, closely-monitored run sets (e.g. a
    real stage-1 epoch taking ~20 minutes, where 0.7s is ~0.06% overhead)
    -- throttling there would only make the plot they're actively
    watching stale, for savings too small to matter at that timescale.

    log_every_epoch=False: throttle to every `every` epochs. This is the
    quiet/automated case (short ablations, batch runs, tests) where
    nobody is watching each epoch's own plot update, and where epochs
    are often fast enough that 0.7s IS a proportionally real cost
    (measured directly: 12 unthrottled writes took ~6.8s vs ~3.1s
    throttled on a 12-epoch synthetic run).

    epoch=0 always writes regardless of log_every_epoch, so train_*()'s
    own epochs=0 ablation (which runs the loop body exactly once, at
    epoch 0) still produces its figures either way.
    """
    return log_every_epoch or epoch % every == 0


def write_loss_history(output_path: Path, epochs: list[int], train_loss: list[float],
                        val_loss: list[float], best_so_far: list[float],
                        secondary_train: list[float] | None = None,
                        secondary_val: list[float] | None = None,
                        secondary_label: str = "1step") -> Path:
    """Dump the per-epoch history beside the loss curve, as CSV.

    Until this existed the history survived ONLY as pixels: it lives in memory
    during the run, is drawn into the PNG, and is discarded. The console log is
    no substitute -- it is throttled to saved epochs, so a run that spikes and
    stops improving prints nothing for its final hundreds of epochs.

    That combination cost a real diagnosis: 128x128 stage 3a spiked at epoch
    3568, and the only evidence was a vertical line on a plot. Answering "how
    big, how long, did it recover" meant reading pixels.

    Written next to output_path with a .csv suffix, rewritten each time the
    figure is, so it is current even if the run is killed.
    """
    csv_path = Path(output_path).with_suffix(".csv")
    cols = ["epoch", "train_loss", "val_loss", "best_ema_so_far"]
    data = [epochs, train_loss, val_loss, best_so_far]
    if secondary_train is not None:
        cols.append(f"train_{secondary_label}")
        data.append(secondary_train)
    if secondary_val is not None:
        cols.append(f"val_{secondary_label}")
        data.append(secondary_val)
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for row in zip(*data):
            fh.write(",".join("" if v is None else repr(v) for v in row) + "\n")
    tmp.replace(csv_path)          # atomic: never a half-written file
    return csv_path


def loss_curve(
    epochs: list[int], train_loss: list[float], val_loss: list[float],
    best_so_far: list[float], output_path: Path, title: str = "",
    secondary_train: list[float] | None = None, secondary_val: list[float] | None = None,
    secondary_label: str = "1step",
    event_epochs: list[tuple[float, str]] | None = None,
    reference_levels: list[tuple[float, str]] | None = None,
) -> Path:
    """
    Called from every stage's epoch loop (see train_stage1.py/
    train_stage2.py/train_lds.py/train_refinement.py) so the
    visualization and its y-axis-saturation behavior stay in exactly
    one place rather than being reimplemented slightly differently per
    stage.

    train_loss/val_loss/best_so_far: one value per epoch so far, in
    order -- best_so_far is the running minimum of val_loss's OWN
    criterion (see CheckpointCriterionTracker), i.e. what actually
    decides whether a checkpoint gets saved, not just min(val_loss) up
    to that point (during a warmup phase the two can differ).

    secondary_train/secondary_val: for stages with more than one loss
    scale worth watching together (currently just train_lds() at
    n_rollout_steps>1: 1step alongside the full rollout loss -- at
    n_rollout_steps=1 the two are identical, so this is skipped, same
    condition as the console's own show_1step). Plotted as thinner,
    dashed lines so the primary loss stays visually dominant.

    Y-axis capped at 1.5x the 99th percentile of every value across ALL
    plotted curves (train, valid, best_so_far, and secondary if given --
    concatenated together, not computed per-curve), but ONLY when the
    data actually exceeds that cap -- training here is prone to large,
    transient early spikes (seen repeatedly in this project's own
    stage-3 runs) that would otherwise stretch the y-axis so far that
    the later, more informative convergence behavior gets squashed into
    an unreadable flat line near zero. When nothing in the run actually
    reaches 1.5x the 99th percentile (i.e. even the top 1% of values
    don't clear that margin -- e.g. a short or already-smooth run), the
    cap is skipped entirely, auto-scaling to the real range instead --
    always applying the cap regardless of whether it was needed just
    stretches well-behaved runs unnecessarily, hiding real (if small)
    variation in the same way an unwarranted spike-driven stretch would.
    The 1.5x margin above the raw percentile (not the percentile itself)
    exists so the plot doesn't read as if it were hard-clipped exactly
    at the 99th-percentile line -- a bit of headroom above it keeps the
    handful of points that DO exceed the percentile visible near the
    top of the plot, rather than pinned flat against the ceiling.
    Percentile-of-all-values, not a multiple of the first epoch's
    train_loss (an earlier version of this): the first epoch isn't
    always the worst offender (a spike can land anywhere in a run), and
    a fixed multiplier of it has no way to adapt to how MANY extreme
    points there are -- percentile naturally trims exactly the top slice
    regardless of where in the run it occurs or how much of it there is.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_loss, label="train", color="tab:blue", alpha=0.8)
    ax.plot(epochs, val_loss, label="valid", color="tab:orange", alpha=0.8)
    ax.plot(epochs, best_so_far, label="best EMA so far", color="tab:green", linewidth=2)

    # Mid-run events (objective switches, ramp completions, criterion resets)
    # drawn as labelled vertical lines. Without them the curve shows a
    # discontinuity that reads as a learning event: stage 2's centered-target
    # switch drops val_loss sharply in one epoch because the QUANTITY changed,
    # not the model -- and on a log-log axis that cliff is the most prominent
    # feature of the whole figure. The label goes in the legend rather than as
    # rotated text on the line, which collides with the curves at small sizes.
    for event_x, event_label in (event_epochs or []):
        ax.axvline(event_x, color="tab:red", linestyle=":", linewidth=1.2,
                    alpha=0.8, label=event_label)

    # Reference LEVELS are horizontal: a loss VALUE the curve is measured
    # against (the ancestor's val_loss under this run's own objective),
    # not an epoch at which something happened. Drawn only when the
    # measured quantity is constant across the whole run -- a level from
    # before a mid-run target switch is not a bar the post-switch curve
    # can be read against, so the caller withholds it in that case.
    for level_y, level_label in (reference_levels or []):
        ax.axhline(level_y, color="tab:purple", linestyle="--", linewidth=1.2,
                    alpha=0.8, label=level_label)

    if secondary_train is not None:
        ax.plot(epochs, secondary_train, label=f"train ({secondary_label})",
                color="tab:blue", linestyle="--", linewidth=1, alpha=0.5)
    if secondary_val is not None:
        ax.plot(epochs, secondary_val, label=f"valid ({secondary_label})",
                color="tab:orange", linestyle="--", linewidth=1, alpha=0.5)

    all_series = [train_loss, val_loss, best_so_far]
    if secondary_train is not None:
        all_series.append(secondary_train)
    if secondary_val is not None:
        all_series.append(secondary_val)
    # FINITE values only. The previous version filtered nothing and tested
    # min(all_values) > 0, which passes when every value is +inf --
    # min([inf, inf]) is inf, and inf > 0 is True -- so log scale was set on
    # data containing no finite positive value at all, and matplotlib raised
    # "Data has no positive values, and therefore cannot be log-scaled" from
    # deep inside tight_layout(). That killed a 1000-epoch training run at
    # epoch 1, after the epoch's real work was already done.
    #
    # nan behaves differently and hid the problem in testing: min() with a nan
    # is ORDER-DEPENDENT (min([2.0, nan]) is 2.0, min([nan, 2.0]) is nan), so
    # a nan sometimes tripped the guard and sometimes did not.
    #
    # A diverged loss is a real outcome, not a corrupt input -- the figure
    # should degrade to linear and keep the run alive, so the operator can see
    # WHERE it diverged instead of losing the run to its plot.
    all_values = [v for series in all_series for v in series if math.isfinite(v)]
    # Log scale needs every value strictly > 0 -- log(0) and log(negative)
    # are undefined. Losses are positive in practice, but this is a real
    # (if rare) edge case worth falling back gracefully for, not
    # crashing on (e.g. a genuinely perfect, exactly-zero-loss epoch).
    use_log = bool(all_values) and min(all_values) > 0
    if use_log:
        ax.set_yscale("log")

    if all_values:
        cap = 1.5 * float(np.percentile(all_values, 99))
        observed_max = max(all_values)
        # observed_max > cap is false whenever even the top 1% of values,
        # with the 1.5x margin included, don't clear the real max (short
        # runs, or a run with no real outliers) -- exactly when
        # auto-scaling to the real range is already the tighter, more
        # accurate choice (see docstring). cap > 0 guards the degenerate
        # all-zero/all-tiny case, where capping at 0 would make the plot
        # blank rather than doing nothing useful.
        if observed_max > cap and cap > 0:
            # A reference LEVEL must survive the cap. Feeding levels into the
            # percentile instead does not work -- one extra value among
            # hundreds barely moves p99, so a level well above the curves was
            # still clipped (measured: level 50 against a 4.65 cap). The cap
            # exists to stop a transient spike stretching the axis, not to
            # hide an annotation the caller explicitly asked for, so raise
            # the top to clear the highest level when there is one.
            top = cap
            if reference_levels:
                highest = max(y for y, _ in reference_levels)
                if highest > top:
                    top = highest * 1.05
            ax.set_ylim(top=top)
        # else: leave the top unset -- auto-scale to the real range.
    if not use_log:
        ax.set_ylim(bottom=0)
    # else: no bottom=0 on a log axis (undefined) -- matplotlib auto-
    # floors to the smallest positive value actually present instead.

    # Log scale needs every value strictly > 0 -- log(0) and log(negative)
    # are undefined. Epochs should always start at 1 in every training
    # loop in this project (range(1, epochs+1)), but this is a shared,
    # generic plotting function called from several places -- same
    # defensive fallback already applied to loss values below, not an
    # assumption this will always hold.
    # `and all_values`: with no finite y value anywhere, matplotlib registers
    # no data limits at all (non-finite points are skipped), so the X view
    # interval is degenerate too and ITS LogLocator raises the same
    # "Data has no positive values" -- even though the epoch numbers are
    # perfectly positive. Filtering y alone was not enough; the two axes have
    # to stand down together.
    use_log_x = bool(epochs) and min(epochs) > 0 and bool(all_values)
    if use_log_x:
        ax.set_xscale("log")

    ax.set_xlabel("epoch" + (" (log scale)" if use_log_x else ""))
    ax.set_ylabel("loss" + (" (log scale)" if use_log else ""))
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both" if (use_log or use_log_x) else "major")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def _proportional_limits(values: list[float], exponent: float = 0.15,
                          floor: float = 1.05) -> tuple[float, float]:
    """(lo, hi) log-axis limits padded by a fraction of the data's OWN span.

    Returns limits such that the data occupies a roughly constant fraction
    of a LOG axis regardless of how tight or wide it is -- measured at
    51-77% across spans from 1.1x to 100x, against 10-83% for a fixed
    ratio. All values must be positive (the caller checks).
    """
    lo, hi = min(values), max(values)
    pad = max((hi / lo) ** exponent, floor) if lo > 0 and hi > lo else floor
    return lo / pad, hi * pad


def rollout_vs_1step_scatter(l_1step_val, l_rollout_val, output_path, title="",
                             saved_epochs=None,
                             l_1step_train=None, l_rollout_train=None):
    """Stage-3b diagnostic: L_rollout vs L_1step at each SAVED epoch, on shared
    log-log SQUARE axes, for BOTH training and validation (tab:blue / tab:orange,
    same convention as loss_curve). Makes the rollout/per-step TRADEOFF visible:
    a series buying multi-step stability at the cost of single-step accuracy
    walks DOWN-and-RIGHT; one improving both heads to the lower-left. Comparing
    train vs valid also shows GENERALIZATION -- e.g. val bending right while
    train does not is the rollout equivalent of overfitting. Only saved epochs
    are shown, so every point is a real, reloadable checkpoint. Fewer than 2
    finite validation points -> returns None (a 1-step 3a where the two are
    identical, or a run that never saved twice)."""
    import numpy as np

    def _clean(xs, ys):
        xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
        ok = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
        return xs[ok], ys[ok], ok

    xv, yv, okv = _clean(l_1step_val, l_rollout_val)
    if len(xv) < 2:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    allx = list(xv); ally = list(yv)

    def _series(x1, lr, color, name):
        if x1 is None or lr is None:
            return
        x, y, _ = _clean(x1, lr)
        if len(x) < 1:
            return
        allx.extend(x); ally.extend(y)
        ax.plot(x, y, "-o", color=color, ms=4, lw=1, alpha=0.8, zorder=3, label=name)
        ax.scatter(x[-1], y[-1], facecolors="none", edgecolors=color,
                   s=140, lw=2, zorder=4)                 # latest saved (ring)
        _b = int(np.argmin(y))
        ax.scatter(x[_b], y[_b], marker="*", color=color, s=180, zorder=5)  # best L_rollout

    _series(l_1step_train, l_rollout_train, "tab:blue", "train")
    _series(l_1step_val, l_rollout_val, "tab:orange", "valid")

    lo = min(allx + ally) * 0.7
    hi = max(allx + ally) * 1.4
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, zorder=2,
            label="L_rollout = L_1step")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("L_1step (saved epochs)")
    ax.set_ylabel("L_rollout (saved epochs)")
    ax.set_title(title or "L_rollout vs L_1step (saved checkpoints)")
    # ring = latest saved, star = best L_rollout (per series)
    ax.plot([], [], "o", mfc="none", mec="gray", label="latest saved")
    ax.plot([], [], "*", color="gray", label="best L_rollout")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def loss_component_scatter(
    epoch_history: list[int], component_histories: dict[str, dict[str, list[float]]],
    output_path: Path, title: str = "",
    ref_components: dict[str, float] | None = None,
    ref_label: str = "ref (pre-run)",
) -> Path | None:
    """
    Companion to loss_curve(): for a COMPOSITE loss (total = sum of
    several separately-weighted terms), plots every pairwise combination
    of components against each other, one point per epoch, connected in
    epoch order -- train/valid/best-so-far, same color code as
    loss_curve() (tab:blue/tab:orange/tab:green).

    Why this is useful beyond loss_curve() itself: loss_curve() shows
    each quantity's own trajectory over epoch, but not how two components
    trade off AGAINST each other -- e.g. whether a run is genuinely
    improving both recon0 and deriv together, or buying one at the
    other's expense (see z0_from_deriv_weight's own docstring for a real
    case of exactly this trade). A 2D trajectory makes that visible
    directly: a run converging cleanly heads toward the origin; a run
    trading one term for another moves diagonally instead, roughly along
    an iso-total line.

    component_histories: {component_name: {"train": [...], "val": [...],
    "best_so_far": [...]}}, one entry per epoch in epoch_history, in the
    SAME units already used in each stage's own console breakdown and
    checkpoint criterion -- i.e. the WEIGHTED, SCALE-NORMALIZED
    contribution (e.g. stats0_weight*val_stats0/stats0_scale), not the
    raw component value. This matters for two reasons: (1) it's what
    genuinely sums to the total loss, so "(0, 0)" is the real convergence
    point regardless of each term's own arbitrary scale, and (2) it
    means an iso-TOTAL line restricted to any two axes is simply
    x + y = c (slope -1), not a differently-sloped line per pair that
    would need each term's own weight/scale threaded through here too.
    "best_so_far": see ComponentBestTracker -- the val-side component
    values co-occurring on the epoch a checkpoint was actually last
    saved, NOT each component's own independent running minimum (which
    wouldn't correspond to any single real, reloadable checkpoint).

    Fewer than 2 components (nothing to pair) -- returns None, writes
    nothing. This is a normal, expected case (e.g. stage 1 with
    include_stats=False), not an error.

    Iso-total lines: several parallel, dashed gray x + y = c lines,
    c chosen from the actual data range (see _iso_total_levels), so they
    frame the trajectory rather than being an arbitrary, possibly
    off-screen grid. Restricted to the TWO plotted components' own
    contribution to the total -- it ignores whatever other components
    exist (deliberately: a true N-dimensional iso-surface sliced through
    changing conditions, e.g. a ramping deriv_weight, would need
    threading every OTHER component's own current value through here
    too, for comparatively little extra visual clarity in what is meant
    to stay a quick, at-a-glance diagnostic).

    A run that resumes from a checkpoint trained under DIFFERENT
    component weights can genuinely start this trajectory somewhere far
    from (0, 0) and take a visibly different path than a fresh run would
    -- not a bug in the plot; see this function's own caller for exactly
    this situation (deriv_target_centered's own mid-run switch, or a
    resume with different stats0_weight/deriv_weight).
    """
    names = list(component_histories.keys())
    if len(names) < 2:
        return None

    # LOWER-TRIANGULAR corner layout, not a wrapped flat list. With four
    # components the flat 2x3 packing put (recon0,stats0) and (stats0,deriv)
    # in different rows with different axes, so nothing lined up and the
    # reader had to re-read the axis labels on all six panels. Here row r
    # is ALWAYS the y variable names[r+1] and column c is ALWAYS the x
    # variable names[c]: every panel in a row shares a y quantity, every
    # panel in a column shares an x quantity, and the empty upper triangle
    # is the redundant transpose rather than wasted space.
    n = len(names) - 1
    fig, axes = plt.subplots(n, n, figsize=(5 * n, 4.5 * n), squeeze=False)
    for r in range(n):
        for c in range(n):
            if c > r:
                axes[r][c].set_visible(False)

    pairs = [(names[c], names[r + 1], axes[r][c], r, c)
             for r in range(n) for c in range(r + 1)]

    # ONE shared range for every axis of every panel, computed over ALL
    # components at once -- max(over all scaled vars) and min(over all scaled
    # vars). The components are already the WEIGHTED, SCALE-NORMALIZED
    # contributions (they sum to the total), so they are directly comparable
    # in magnitude and belong on a common scale. Per-axis auto-scaling
    # (each axis fitted to its own variable's spread) was the bug behind the
    # near-horizontal iso-total lines: an iso-total x + y = c is only a 45-deg
    # visual line when both axes cover the same interval per unit length. With
    # recon0 spanning ~1-1.06 and deriv ~1-675 on their OWN axes, the shared
    # x+y=c line tilted flat. A single square range fixes that: the iso-lines
    # render at 45 deg and panels become directly comparable to each other.
    _all_positive = [v
                     for comp in component_histories.values()
                     for series in ("train", "val", "best_so_far")
                     for v in comp[series]
                     if math.isfinite(v) and v > 0]
    # the ref (pre-run baseline) point must be in-frame too, so its component
    # values join the shared-range computation.
    if ref_components:
        _all_positive += [v for v in ref_components.values()
                          if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
    if _all_positive:
        global_lo, global_hi = _proportional_limits(_all_positive)
    else:
        global_lo, global_hi = 0.0, 1.0

    for name_x, name_y, ax, row, col in pairs:
        cx, cy = component_histories[name_x], component_histories[name_y]

        # best_so_far is DASHED, and that is not decoration. It coincides
        # EXACTLY with val on every epoch where a checkpoint was saved (see
        # ComponentBestTracker: on a saved epoch it takes that epoch's own val
        # values), so on a run that improves every epoch -- common early, and
        # the normal case for a short run -- a solid green line of width 2.0
        # drawn last hides the orange one completely. The plot then shows two
        # curves and three legend entries, which reads as a missing series
        # rather than as two series agreeing. Dashes let the val line show
        # through the gaps, so coincidence looks like coincidence.
        for series_key, label, color, lw, ls in [
            ("train", "train", "tab:blue", 1.2, "-"),
            ("val", "valid", "tab:orange", 1.2, "-"),
            ("best_so_far", "best so far", "tab:green", 2.0, "--"),
        ]:
            x, y = cx[series_key], cy[series_key]
            ax.plot(x, y, ls, color=color, alpha=0.6, linewidth=lw, zorder=2)
            ax.scatter(x, y, s=14, color=color, alpha=0.7, zorder=3, label=label)

        # Mark the trajectory's own start/end so its DIRECTION is legible
        # without relying on line thickness or point density alone --
        # important here specifically because, unlike loss_curve()'s own
        # x-axis, epoch order isn't directly visible on a loss-vs-loss
        # plot otherwise.
        for series_key, color in [("train", "tab:blue"), ("val", "tab:orange")]:
            x, y = component_histories[name_x][series_key], component_histories[name_y][series_key]
            if x:
                ax.scatter([x[0]], [y[0]], s=70, facecolors="none", edgecolors=color,
                           linewidths=1.5, zorder=4)
                ax.scatter([x[-1]], [y[-1]], s=90, marker="*", color=color,
                           edgecolors="black", linewidths=0.5, zorder=5)

        # The REF point: the pre-run baseline (the log's "ref|" line -- the
        # ancestor's component values before this run's epoch 1). Drawn as a
        # purple circle so the run's trajectory is legible RELATIVE to where it
        # started from the resumed checkpoint, not just relative to its own
        # first epoch. Only drawn when both this panel's components are present
        # and positive (log axes).
        if ref_components is not None:
            rx = ref_components.get(name_x)
            ry = ref_components.get(name_y)
            if (rx is not None and ry is not None
                    and math.isfinite(rx) and math.isfinite(ry)
                    and rx > 0 and ry > 0):
                ax.scatter([rx], [ry], s=70, marker="o", facecolors="none",
                           edgecolors="tab:purple", linewidths=2.0, zorder=6,
                           label=ref_label)

        # FINITE values only, everywhere below. An epochs=0 ablation never
        # iterates the train set, so every train component is NaN -- and while
        # `v > 0` already excludes NaN from `positive_x`, the fallback branch
        # then read ax.get_xlim() from an axes whose only scattered points were
        # NaN, which matplotlib reports as non-finite. That reached set_xlim as
        # "Axis limits cannot be NaN or Inf" and killed the run from inside its
        # own diagnostic figure -- the same class of failure as loss_curve's,
        # which was hardened while this function was not.
        all_x = [v for v in cx["train"] + cx["val"] + cx["best_so_far"] if math.isfinite(v)]
        all_y = [v for v in cy["train"] + cy["val"] + cy["best_so_far"] if math.isfinite(v)]

        # LOG-LOG. These components span more than a decade within a single
        # run -- stage 2's deriv went 0.79 -> 0.09 while recon0 went 8.5 -> 3.2
        # -- and on linear axes the early, large values compress the whole
        # late trajectory into a corner, which is exactly the part worth
        # reading. Log axes give every factor-of-two the same visual weight.
        #
        # Non-positive values cannot be shown on a log axis. They are dropped
        # from the LIMIT calculation rather than silently clipped, and a
        # component that is legitimately 0 (an inactive term, e.g.
        # stats1_weight=0) simply has no point to draw.
        positive_x = [v for v in all_x if v > 0]
        positive_y = [v for v in all_y if v > 0]
        if positive_x and positive_y:
            ax.set_xscale("log")
            ax.set_yscale("log")
            # SHARED square range on both axes (see the global computation
            # above): same [lo, hi] for x and y so the iso-total line renders
            # at 45 deg and every panel is on the same scale. This replaces
            # the former per-axis _proportional_limits(positive_x/positive_y),
            # which fitted each axis to its own variable and tilted the
            # iso-lines flat whenever the two components differed in magnitude.
            xlo, xmax = global_lo, global_hi
            ylo, ymax = global_lo, global_hi
        else:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            xlo, ylo = 0.0, 0.0
            # (0, 1) rather than ax.get_xlim() when there is nothing finite to
            # scale to: an axes holding only NaN points reports non-finite
            # limits, and feeding those straight back into set_xlim raises.
            # `or 1.0` covers all-zero components too (every term inactive):
            # max()*1.1 is then 0, and set_xlim(0, 0) is singular. Shared
            # across x and y (a single max over BOTH) to keep the axes square
            # even on this degenerate branch, consistent with the log branch.
            _fallback_max = (max(all_x + all_y) * 1.1 if (all_x or all_y) else 0.0) or 1.0
            xmax = ymax = _fallback_max

        for c in _iso_total_levels(all_x, all_y):
            if ax.get_xscale() == "log":
                # x + y = c is a STRAIGHT line only on linear axes; on log-log
                # it curves, so it has to be sampled rather than drawn from two
                # endpoints. Sampled in log x, so the curve stays smooth across
                # the whole decade rather than bunching near the right edge.
                xs = np.geomspace(max(xlo, c * 1e-3), min(xmax, c), 200)
                ys = c - xs
                keep = ys > 0
                if keep.sum() >= 2:
                    ax.plot(xs[keep], ys[keep], "--", color="gray", alpha=0.35,
                            linewidth=0.8, zorder=1)
            else:
                segment = _clip_iso_total_segment(c, xmax, ymax)
                if segment is not None:
                    (x0, y0), (x1, y1) = segment
                    ax.plot([x0, x1], [y0, y1], "--", color="gray", alpha=0.35,
                            linewidth=0.8, zorder=1)
        ax.set_xlim(xlo if xlo else 0, xmax)
        ax.set_ylim(ylo if ylo else 0, ymax)
        # Corner-plot convention: label only the MARGINS. Every panel in a
        # column has the same x quantity and every panel in a row the same
        # y, so repeating labels and tick text in the interior is pure
        # clutter -- and on a log axis with a narrow range matplotlib
        # labels every MINOR tick, which overprinted into an unreadable
        # smear ("1.2x10 1.4x10 1.6x10..."). Interior panels keep their
        # ticks (the grid still reads) but lose the text.
        # In a lower triangle column `col` is visible for every row >= col,
        # so its bottom-most visible panel is ALWAYS the last row; and row
        # `row` starts at column 0. Hence the margins are exactly
        # row == n-1 and col == 0.
        if row == n - 1:
            ax.set_xlabel(name_x)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(name_y)
        else:
            ax.tick_params(labelleft=False)
        # Minor-tick text off on log axes: majors (decades) stay labelled.
        # BUT only when there is at least one major (decade) tick IN VIEW to
        # carry the labelling -- a sub-decade range (e.g. 0.4..0.5, common on a
        # converged run) crosses no power of ten, so LogLocator places no major
        # ticks at all, and blanking the minors too would leave the axis with
        # NO numbers whatsoever. In that case keep the minor labels (and give
        # them a readable scalar format) so the axis is still legible.
        def _has_major_in_view(axis, lo, hi):
            ticks = axis.get_majorticklocs()
            return any(lo <= t <= hi for t in ticks)
        if ax.get_xscale() == "log":
            _lo, _hi = ax.get_xlim()
            if _has_major_in_view(ax.xaxis, _lo, _hi):
                ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            else:
                ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        if ax.get_yscale() == "log":
            _lo, _hi = ax.get_ylim()
            if _has_major_in_view(ax.yaxis, _lo, _hi):
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            else:
                ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        if (row, col) == (0, 0):
            # "best", not a fixed corner. These trajectories head toward the
            # origin, so they occupy the LOWER-LEFT... except early in a run,
            # when every point is still up and to the right and a hardcoded
            # "upper right" lands the legend squarely on the data while three
            # quarters of the axes are empty. matplotlib's own overlap
            # minimisation costs a little draw time and gets it right in both
            # regimes.
            ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)

    # (the upper triangle was already hidden above -- it is the redundant
    #  transpose of the lower one, not an unused remainder.)

    if title:
        fig.suptitle(title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def _clip_iso_total_segment(
    c: float, xmax: float, ymax: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """
    The visible portion of the line x + y = c within the rectangle
    [0, xmax] x [0, ymax], as ((x0, y0), (x1, y1)) -- or None if the
    line doesn't intersect that rectangle at all (c < 0, or c so large
    the whole line passes beyond both xmax and ymax).

    The line's own two candidate endpoints inside the rectangle: where
    it crosses x=0 (at y=c, valid only if 0 <= c <= ymax) and where it
    crosses y=0 (at x=c, valid only if 0 <= c <= xmax). Between those,
    the actual visible segment runs from
    x = max(0, c - ymax) [the point where y is capped at ymax, if c is
    large enough to need it] to x = min(c, xmax) [the point where either
    x hits xmax or y hits 0, whichever comes first].
    """
    if c < 0 or xmax <= 0 or ymax <= 0:
        return None
    x0 = max(0.0, c - ymax)
    x1 = min(c, xmax)
    if x0 > x1:
        return None
    return (x0, c - x0), (x1, c - x1)


def _iso_total_levels(all_x: list[float], all_y: list[float], n_levels: int = 4) -> list[float]:
    """A handful of x+y=c levels spanning the actual plotted data, not an
    arbitrary fixed grid that might land mostly off-screen for a
    particular run's own loss scale. Returns [] if there's no data to
    span at all (e.g. every history empty), rather than raising."""
    if not all_x or not all_y:
        return []
    totals = [x + y for x, y in zip(all_x, all_y)]
    lo, hi = min(totals), max(totals)
    if hi <= lo:
        return [hi] if hi > 0 else []
    return list(np.linspace(lo, hi, n_levels))



def show_snapshot(path: str | Path, nx: int, ny: int,
                   ax=None, cmap: str = "RdBu", vmin: float | None = None,
                   vmax: float | None = None, title: str | None = None):
    """
    Display a single raw binary snapshot (as written by save_phi_half),
    without going through the solver's PNG export.

    vmin/vmax default to None, which auto-scales symmetrically around 0
    to this image's own data range. A fixed scale (e.g. the previous
    default of +-1) makes low-amplitude fields -- like early,
    noise-dominated steps, before microstructure develops -- look flat
    and blank even though real structure is there at a smaller scale.
    """
    phi = load.read_phi_half(path, nx, ny)

    if vmin is None or vmax is None:
        scale = max(abs(phi.min()), abs(phi.max()), 1e-6)
        vmin, vmax = -scale, scale

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(phi, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title or f"{Path(path).name} (scale=+-{max(abs(vmin), abs(vmax)):.3f})")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


# NOTE: called nowhere in the codebase, but kept for direct use from the
# command line / an interactive session: do not remove. (Reported not
# working as of 2026-08; works in isolation on synthetic data, failure
# mode not yet pinned down -- see session notes.)
def make_video(run_dir: str | Path, metadata: "load.RunMetadata", output_path: str | Path,
                fps: int = 10, cmap: str = "RdBu", vmin: float | None = None,
                vmax: float | None = None):
    """
    Build a video from a run's saved snapshots, in step order.
    Skips steps whose file is missing rather than failing outright --
    check_snapshots_saved should be used beforehand for a real completeness check.

    output_path must end in .mp4 (needs ffmpeg on PATH) or .gif (works
    everywhere via Pillow, no extra dependency).

    vmin/vmax default to None, which auto-scales symmetrically around 0
    using the actual range across ALL frames in this video (not
    per-frame, which would make amplitude changes over time invisible --
    the whole point of a growing-microstructure video is to see the
    field's amplitude and structure develop, so the scale must stay
    fixed across frames while still reflecting the real data range).
    """
    run_dir = Path(run_dir)
    output_path = Path(output_path)

    if output_path.suffix not in (".mp4", ".gif"):
        raise ValueError(
            f"output_path '{output_path}' must end in .mp4 or .gif "
            f"(got suffix '{output_path.suffix}')"
        )

    frames = []
    for step in metadata.save_steps:
        f = run_dir / load.snapshot_filename(step)
        if f.exists():
            frames.append((step, load.read_phi_half(f, metadata.nx, metadata.ny)))

    if not frames:
        raise ValueError(f"{run_dir}: no snapshot files found to build a video from")

    if vmin is None or vmax is None:
        scale = max(max(abs(phi.min()), abs(phi.max())) for _, phi in frames)
        scale = max(scale, 1e-6)
        vmin, vmax = -scale, scale

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0][1], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    title = ax.set_title(f"step {frames[0][0]}")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        step, phi = frames[i]
        im.set_data(phi)
        title.set_text(f"step {step}")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)

    writer = "ffmpeg" if output_path.suffix == ".mp4" else "pillow"
    anim.save(output_path, writer=writer, fps=fps)
    plt.close(fig)

    return output_path
