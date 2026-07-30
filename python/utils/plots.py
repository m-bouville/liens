"""
Visualization helpers for phase-field snapshots -- displaying raw binary
files directly, without going through the solver's PNG export.
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from . import load_datasets as load


LOSS_FIGURE_EVERY = 10


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
    return epoch % every == 0


def loss_curve(
    epochs: list[int], train_loss: list[float], val_loss: list[float],
    best_so_far: list[float], output_path: Path, title: str = "",
    secondary_train: list[float] | None = None, secondary_val: list[float] | None = None,
    secondary_label: str = "1step",
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
    all_values = [v for series in all_series for v in series]
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
            ax.set_ylim(top=cap)
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
    use_log_x = bool(epochs) and min(epochs) > 0
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


def loss_component_scatter(
    epoch_history: list[int], component_histories: dict[str, dict[str, list[float]]],
    output_path: Path, title: str = "",
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
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    if not pairs:
        return None

    n_cols = min(3, len(pairs))
    n_rows = -(-len(pairs) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)

    for idx, (name_x, name_y) in enumerate(pairs):
        ax = axes[idx // n_cols][idx % n_cols]
        cx, cy = component_histories[name_x], component_histories[name_y]

        for series_key, label, color, lw in [
            ("train", "train", "tab:blue", 1.2),
            ("val", "valid", "tab:orange", 1.2),
            ("best_so_far", "best so far", "tab:green", 2.0),
        ]:
            x, y = cx[series_key], cy[series_key]
            ax.plot(x, y, "-", color=color, alpha=0.6, linewidth=lw, zorder=2)
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

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        xmax = ax.get_xlim()[1]
        ymax = ax.get_ylim()[1]
        all_x = cx["train"] + cx["val"] + cx["best_so_far"]
        all_y = cy["train"] + cy["val"] + cy["best_so_far"]
        for c in _iso_total_levels(all_x, all_y):
            segment = _clip_iso_total_segment(c, xmax, ymax)
            if segment is not None:
                (x0, y0), (x1, y1) = segment
                ax.plot([x0, x1], [y0, y1], "--", color="gray", alpha=0.35,
                        linewidth=0.8, zorder=1)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel(name_x)
        ax.set_ylabel(name_y)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    # Unused grid cells (n_rows*n_cols > len(pairs), e.g. 4 pairs in a
    # 2x3 grid) hidden rather than left as blank axes with tick marks.
    for idx in range(len(pairs), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

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
