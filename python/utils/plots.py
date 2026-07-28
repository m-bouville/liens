"""
Visualization helpers for phase-field snapshots -- displaying raw binary
files directly, without going through the solver's PNG export.
"""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from . import load_datasets as load


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
