"""
Presentation renderer: ground-truth vs. surrogate-predicted microstructure
evolution, a few runs stacked as real/prediction row pairs.

This is the PRESENTATION sibling of compare_f_theta's diagnostic trajectory
figure. compare_f_theta compares several checkpoints (many rows, every
ancestor, dense labels) to locate WHERE a model goes wrong; this shows ONE
model tracking the ground truth on a handful of chosen runs, clean enough to
put in a talk or a post. It deliberately owns only the LAYOUT: the actual
roll-and-decode is compare_f_theta.compute_trajectory (shared verbatim, so the
u-scheme / previous_quotient handling can never drift between the two tools),
and the checkpoint loading is compare_f_theta._load_model (stage 3 and the
stage 4/5 joint format both).

Usage (one line -- PowerShell-safe):
    python -m evaluation.plot_evolution checkpoints/stage5/128x128-stage5-...pt --steps 10 --datasets ../datasets/128x128/T725_n003_s123:15000 ../datasets/128x128/T725_n003_s191:15000

--steps N means N ROLLOUT steps (matching compare_f_theta's own --steps), so
the figure has N+1 columns (start frame + N predicted). Each --datasets entry
is RUNDIR:STARTSTEP; the N+1 frames are the consecutive saved steps from
STARTSTEP onward (the irregular save schedule means a late start may not have
N+1 ahead of it -- that is an explicit error, not a silent short row).
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: save a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.load_datasets as load
from evaluation.compare_f_theta import _load_model, compute_trajectory
from utils.plot_helpers import every_other_columns

# Anchor the default output path to this file's location (python/..), the
# project's path policy, so the figure lands in output/ regardless of CWD --
# same convention compare_f_theta uses.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent


def _parse_run_spec(spec: str) -> tuple[Path, int]:
    """'RUNDIR:STARTSTEP' -> (Path(RUNDIR), STARTSTEP).

    Split on the LAST ':' so a Windows drive-letter path ('D:\\work\\...:15000')
    or any ':' inside the directory name is preserved -- the start step is
    always the final token.
    """
    if ":" not in spec:
        raise ValueError(
            f"run spec {spec!r} must be RUNDIR:STARTSTEP (e.g. "
            f"../datasets/128x128/T725_n003_s123:15000)")
    run_dir, start = spec.rsplit(":", 1)
    try:
        start_step = int(start)
    except ValueError:
        raise ValueError(
            f"start step {start!r} in {spec!r} is not an integer")
    return Path(run_dir), start_step


def _pick_steps(save_steps: list[int], start_step: int, n_frames: int) -> list[int]:
    """The n_frames consecutive saved steps beginning at start_step.

    save_steps: the run's own saved-step list (ascending), from its
    metadata.txt. Pure (no file IO) so it is unit-testable without a dataset.
    Raises with an actionable message if start_step is not a saved step, or if
    fewer than n_frames saved steps remain from there.
    """
    if start_step not in save_steps:
        near = min(save_steps, key=lambda s: abs(s - start_step)) if save_steps else None
        raise ValueError(
            f"start step {start_step} is not a saved step for this run "
            f"(nearest saved: {near}). Pass one of the run's actual saved steps.")
    i = save_steps.index(start_step)
    chosen = save_steps[i:i + n_frames]
    if len(chosen) < n_frames:
        raise ValueError(
            f"only {len(chosen)} saved step(s) at/after {start_step}, but "
            f"{n_frames} frames were requested (--steps {n_frames - 1}). Start "
            f"from an earlier step, or reduce --steps.")
    return chosen


def _saved_steps(run_dir: Path, start_step: int, n_frames: int) -> tuple[list[int], "load.RunMetadata"]:
    """(_pick_steps applied to the run's real save_steps, its metadata).

    The file-reading wrapper around _pick_steps -- kept thin so the slicing
    logic stays pure and testable.
    """
    meta = load.read_metadata(run_dir / "metadata.txt")
    return _pick_steps(list(meta.save_steps), start_step, n_frames), meta


def plot_evolution(checkpoint_path: Path, run_specs: list[str], n_steps: int,
                   device: str | None = None, output_path: Path | None = None,
                   z1_resync: bool = False, title: str | None = None) -> Path:
    """Render real/prediction row pairs for each run and save one figure."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_path)
    model = _load_model(checkpoint_path, device)          # ae, f_theta, ae_config
    n_frames = n_steps + 1

    # Roll and decode each run BEFORE laying out, so a bad run spec fails before
    # any figure is built, and so the column count is known.
    rows = []            # one (tag, real_frames, pred_frames) per run
    for spec in run_specs:
        run_dir, start_step = _parse_run_spec(spec)
        steps, meta = _saved_steps(run_dir, start_step, n_frames)
        real_frames, pred_frames, _dt = compute_trajectory(
            run_dir, steps, model["ae"], model["f_theta"], model["ae_config"],
            device, z1_resync=z1_resync)
        # Exact temperature via :g (not a fixed :.3f): the run-dir token
        # (e.g. T933) is round(T*1000) half-away-from-zero, so a fixed 3-decimal
        # print can read 0.932 next to a "933" name and look like a rounding
        # bug. :g shows the real stored value (0.725, 0.9325, ...) with no
        # spurious rounding and no trailing zeros.
        tag = f"{run_dir.name}  T={meta.temperature:g}"
        rows.append((tag, real_frames, pred_frames))

    n_runs = len(rows)
    n_plot_rows = 2 * n_runs                              # real + prediction per run
    # Thin long runs to every other column (same rule as compare_f_theta's own
    # trajectory montage -- shared via every_other_columns). Purely a DISPLAY
    # subsample: compute_trajectory already rolled through every frame; this
    # only chooses which columns are drawn, keeping the first and last.
    col_indices = every_other_columns(n_frames)
    n_plot_cols = len(col_indices)
    fig, axes = plt.subplots(
        n_plot_rows, n_plot_cols,
        figsize=(1.35 * n_plot_cols, 1.35 * n_plot_rows),
        squeeze=False)

    for r, (tag, real_frames, pred_frames) in enumerate(rows):
        # ONE symmetric colour scale per run, shared by its real AND prediction
        # rows: the prediction cannot be flattered by its own autoscaling, so a
        # genuine contrast loss (the amplitude-collapse failure mode) stays
        # visible. Different runs may differ (different T -> different physical
        # amplitude), which is honest -- only the within-pair scale must match.
        # vmax over ALL frames (not just the shown columns) so the scale is
        # stable regardless of which columns the thinning happens to keep.
        vmax = max(float(np.abs(f).max()) for f in real_frames) or 1.0
        kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")

        real_row, pred_row = 2 * r, 2 * r + 1
        for plot_col, col in enumerate(col_indices):
            axes[real_row][plot_col].imshow(real_frames[col], **kw)
            axes[pred_row][plot_col].imshow(pred_frames[col], **kw)
        for row in (real_row, pred_row):
            for plot_col in range(n_plot_cols):
                axes[row][plot_col].set_xticks([])
                axes[row][plot_col].set_yticks([])
        # Small per-row labels distinguish the two rows; the RUN identity is a
        # single label spanning the pair, added after layout below. "reality"
        # (not "truth") reads more clearly to a non-specialist audience.
        axes[real_row][0].set_ylabel("reality", fontsize=7)
        axes[pred_row][0].set_ylabel("prediction", fontsize=7)

    # A single time-direction cue on the bottom row, no per-column clutter.
    axes[-1][0].set_xlabel("time \u2192", fontsize=9, loc="left")
    if title is None:
        # Say so when the montage is thinned, so "N-step rollout" isn't mistaken
        # for N columns shown -- the model rolled through every step, the figure
        # just draws every other one (see every_other_columns).
        note = ", plotted every other step" if n_plot_cols < n_frames else ""
        title = (f"{checkpoint_path.stem}: microstructure evolution "
                 f"({n_steps}-step rollout{note})")
    fig.suptitle(title, fontsize=11)
    # A SMALL reserved left margin so the packed x0 isn't tiny (which would push
    # the label off-canvas), but no more -- an earlier 0.11 strip put the label
    # a couple of cm from the images.
    fig.tight_layout(rect=(0.05, 0, 1, 0.97))

    # Run label spanning each real/prediction PAIR: placed just LEFT of the axes
    # edge (ha="right" grows the text leftward from x_left), vertically centred
    # across the two rows. TUNING: the gap to the "reality"/"prediction" ylabels
    # is set by _LABEL_GAP -- ~1 mm per 0.01 at a typical figure width. 0.012
    # touched them, 0.05-from-a-zero-margin fell off; 0.02 with the 0.05 reserve
    # above is a small, safe gap. Nudge _LABEL_GAP if it still looks off.
    _LABEL_GAP = 0.02
    for r, (tag, _rf, _pf) in enumerate(rows):
        top = axes[2 * r][0].get_position()          # real row (upper)
        bot = axes[2 * r + 1][0].get_position()      # prediction row (lower)
        y_center = 0.5 * (top.y1 + bot.y0)
        fig.text(top.x0 - _LABEL_GAP, y_center, tag, rotation=90,
                 va="center", ha="right", fontsize=8, fontweight="medium")

    if output_path is None:
        output_path = (_PYTHON_ROOT.parent / "output" / "plot_evolution"
                       / f"{checkpoint_path.stem}-evolution.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_path}")
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path,
                   help="a stage-3/4/5 checkpoint (the model whose prediction is shown)")
    p.add_argument("--datasets", type=str, nargs="+", required=True,
                   metavar="RUNDIR:STARTSTEP",
                   help="one or more runs, each as RUNDIR:STARTSTEP")
    p.add_argument("--steps", type=int, default=10,
                   help="number of rollout steps (figure has steps+1 columns)")
    p.add_argument("--z1-resync", action="store_true",
                   help="teacher-force z1 each step (default off: the inference regime)")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    plot_evolution(args.checkpoint, args.datasets, args.steps,
                   device=args.device, output_path=args.output,
                   z1_resync=args.z1_resync, title=args.title)


if __name__ == "__main__":
    main()
