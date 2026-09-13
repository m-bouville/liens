"""
Plot the phi-normalization factor to test WHY normalize_phi hurts the downstream
latent-dynamics surrogate.

normalize_phi divides phi by the theoretical equilibrium amplitude
    A(T) = sqrt(-a(T)/b),  a(T) = a0*(T - T0)
(the same normalizer _dataset_filtering uses; it puts every temperature's plateau
at 1.0). The dynamics regression rested on WHAT this factor does along a
trajectory and across trajectories:

  - ALONG a trajectory: T is constant, so A(T) is CONSTANT in time. This rules out
    the "normalization makes the latent trajectory non-stationary" hypothesis -- the
    factor does not drift frame-to-frame. The plot shows this directly (flat lines).

  - ACROSS trajectories: A(T) shrinks like sqrt(T0 - T) and -> 0 as T -> T0. So
    near-critical runs are divided by a tiny number: their phi (and its frame-to-
    frame differences, which carry the interface-velocity signal the dynamics reads)
    are blown up by a large, T-dependent gain, while low-T runs are barely rescaled.
    Normalization thus imposes a temperature-dependent gain on the dynamics targets
    -- large and ill-conditioned near T0, mild far from it. THAT is the mechanism to
    look at: a per-trajectory-constant but wildly-across-T factor, not a within-
    trajectory drift.

Two panels:
  (left)  A(T) vs time for a sample of runs across the T range -- each run a flat
          horizontal line (constant in time), colored by T, so the flatness (within)
          and the spread (across) are both visible at once.
  (right) A(T) and 1/A(T) vs T over all runs -- the gain the dynamics sees; 1/A is
          the multiplier applied to phi differences, which explodes as T -> T0.

Usage (Windows/PowerShell, single line):
    python -m evaluation.check_normalization_factor --size 128 --device cpu
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from training.datasets import report_save_step_distribution
from utils import load_datasets as load

_PYTHON_ROOT = Path(__file__).resolve().parent.parent   # python/evaluation/X.py -> python/


def _amplitude(metadata) -> float | None:
    """A(T) = sqrt(-a0*(T-T0)/b), or None for T >= T0 (no equilibrium amplitude --
    exactly the steps normalize_phi cannot handle and excludes)."""
    a_T = metadata.a0 * (metadata.temperature - metadata.T0)
    val = -a_T / metadata.b
    return float(np.sqrt(val)) if val > 0.0 else None


def check_normalization_factor(size: int, base: str | None = None,
                               output_path: Path | None = None,
                               max_runs_time_panel: int = 40) -> None:
    base_path = base if base is not None else str(_PYTHON_ROOT.parent / "datasets")
    if output_path is None:
        output_path = _PYTHON_ROOT.parent / "output" / "normalization_factor.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = load.enumerate_run_dirs_from_metadata(base_path, size, size)
    report_save_step_distribution(run_dirs)   # sweep may be mixed -- surface it (project policy)

    # one (T, A(T), save_steps) per run
    per_run = []
    skipped_super_T0 = 0
    for run_dir in run_dirs:
        mpath = run_dir / "metadata.txt"
        if not mpath.exists():
            continue
        metadata = load.read_metadata(mpath)
        if not metadata.is_complete:
            continue
        A = _amplitude(metadata)
        if A is None:
            skipped_super_T0 += 1
            continue
        per_run.append((metadata.temperature, A, list(metadata.save_steps)))

    if not per_run:
        print(f"No usable runs under {base_path} (size {size}).")
        return
    per_run.sort(key=lambda r: r[0])
    temps = np.array([r[0] for r in per_run])
    amps = np.array([r[1] for r in per_run])
    print(f"{len(per_run)} run(s); T in [{temps.min():.4g}, {temps.max():.4g}]; "
          f"A(T) in [{amps.min():.4g}, {amps.max():.4g}] "
          f"(ratio max/min = {amps.max()/max(amps.min(), 1e-12):.1f}x)"
          + (f"; {skipped_super_T0} run(s) at T>=T0 excluded (no amplitude)"
             if skipped_super_T0 else ""))

    fig, (ax_t, ax_T) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("coolwarm")
    tmin, tmax = temps.min(), temps.max()

    def _tcolor(T):
        return cmap((T - tmin) / (tmax - tmin) if tmax > tmin else 0.5)

    # LEFT: A(T) vs time -- each run a flat line (constant in time), colored by T.
    # Subsample runs so the panel stays legible; span the T range evenly.
    idx = np.linspace(0, len(per_run) - 1, min(max_runs_time_panel, len(per_run)))
    for i in np.unique(idx.astype(int)):
        T, A, steps = per_run[i]
        if not steps:
            continue
        ax_t.plot([steps[0], steps[-1]], [A, A], "-", color=_tcolor(T),
                  alpha=0.7, linewidth=1.2)
    ax_t.set_xlabel("time step")
    ax_t.set_ylabel("normalization factor  A(T) = \u221a(-a(T)/b)")
    ax_t.set_title("A(T) along trajectories\n(flat = constant in time within a run)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(tmin, tmax))
    fig.colorbar(sm, ax=ax_t, label="temperature T")

    # RIGHT: A(T) and 1/A(T) vs T -- the gain applied to phi (and phi-differences).
    ax_T.plot(temps, amps, "o-", color="tab:blue", label="A(T)  (divisor of \u03c6)")
    ax_T.set_xlabel("temperature T")
    ax_T.set_ylabel("A(T)", color="tab:blue")
    ax_T.tick_params(axis="y", labelcolor="tab:blue")
    ax_T.set_title("A(T) and the gain 1/A(T) vs T\n(1/A explodes as T \u2192 T0 \u2192 amplifies \u03c6 & its differences)")
    ax2 = ax_T.twinx()
    ax2.plot(temps, 1.0 / np.maximum(amps, 1e-12), "s-", color="tab:red",
             label="1/A(T)  (gain on \u03c6 differences)")
    ax2.set_ylabel("1/A(T)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    lines = ax_T.get_lines() + ax2.get_lines()
    ax_T.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)

    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=110)
        print(f"Saved figure to {output_path}")
    except OSError as e:                       # locked file etc. -- warn, don't crash
        print(f"  (could not save {output_path}: {e})")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Plot the phi normalization factor A(T) along trajectories and vs T.")
    ap.add_argument("--size", type=int, required=True, help="grid size, e.g. 128")
    ap.add_argument("--base", type=str, default=None,
                    help="datasets root (default: <repo>/datasets)")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cpu",
                    help="accepted for CLI parity; this diagnostic needs no model")
    args = ap.parse_args()
    check_normalization_factor(size=args.size, base=args.base, output_path=args.output)


if __name__ == "__main__":
    main()
