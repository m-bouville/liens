"""
DISPOSABLE: how many windows survive each min_passing_steps threshold, and at
which times? min_passing_steps drops an ENTIRE run when fewer than that many of
its steps clear min_stdev_phi -- so it removes whole (short / slow-developing,
typically near-critical) runs and all their windows. This REBUILDS via
build_good_steps per swept value (frame-free) and reports the RETENTION rate vs
the baseline (lowest) value, resolved by window start-time t -- showing which
TIMES lose windows as short runs are excluded.

Companion to sweep_min_std_deriv. min_stdev_phi is fixed here (min_passing_steps
counts steps that pass it); min_passing_steps is the swept axis.

    python -m evaluation.sweep_min_passing_steps --base-path ../datasets --size 128 \
        --window-length 2 --min-step 1000 --min-stdev-phi 0.01 \
        --min-passing-steps 1 4 8 12 20 40 --max-runs 0
"""
import argparse
from pathlib import Path

import numpy as np

from training.datasets import build_good_steps, complete_run_dirs
from evaluation._sweep_filters_common import dt_by_run, window_start_times, render

_PYTHON_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-path", type=Path, default=_PYTHON_ROOT.parent / "datasets")
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--window-length", type=int, default=2)
    p.add_argument("--min-step", type=int, default=0,
                   help="0 = include all steps (default). Raise to drop early steps")
    p.add_argument("--min-stdev-phi", type=float, default=0.01,
                   help="FIXED (min_passing_steps counts steps that pass this)")
    p.add_argument("--max-dt", type=float, default=float("inf"))
    p.add_argument("--min-passing-steps", type=int, nargs="+",
                   default=[8, 12, 20, 30, 40, 50],
                   help="SWEPT (displayed rows). Retention denominator is min_passing_steps OFF, "
                        "computed separately -- so these are fraction-of-all-windows")
    p.add_argument("--current-value", type=int, default=None,
                   help="optional: value to mark '<- current' in table/plot")
    p.add_argument("--max-runs", type=int, default=0, help="0 = all runs")
    p.add_argument("--min-bin-count", type=int, default=10)
    p.add_argument("--sma", type=int, default=3)
    p.add_argument("--output", type=Path,
                   default=_PYTHON_ROOT.parent / "output" / "datasets"
                            / "128x128-min_passing_steps_sweep.png")
    a = p.parse_args()

    run_dirs = complete_run_dirs(a.base_path, a.size, a.size)
    n_total = len(run_dirs)
    if a.max_runs and len(run_dirs) > a.max_runs:
        idx = np.linspace(0, len(run_dirs) - 1, a.max_runs).round().astype(int)
        run_dirs = [run_dirs[i] for i in sorted(set(idx))]
    frac = len(run_dirs) / n_total
    scale = n_total / len(run_dirs)
    print(f"using {len(run_dirs)}/{n_total} runs ({100 * frac:.1f}%)"
          if frac < 1 else f"using all {n_total} runs")
    dts = dt_by_run(run_dirs)

    values = sorted(set(a.min_passing_steps))
    # Retention denominator: min_passing_steps OFF (None), same min_stdev_phi --
    # so retention is the fraction of all windows that survive each threshold,
    # not a fraction of the mps=1 set.
    base_gs = build_good_steps(run_dirs, min_step=a.min_step,
                               min_stdev_phi=a.min_stdev_phi, min_passing_steps=None)
    baseline_starts = window_start_times(base_gs, dts, a.window_length, a.max_dt)

    starts = {}
    for mps in values:
        gs = build_good_steps(run_dirs, min_step=a.min_step,
                              min_stdev_phi=a.min_stdev_phi, min_passing_steps=mps)
        starts[mps] = window_start_times(gs, dts, a.window_length, a.max_dt)

    render("min_passing_steps", values, starts, baseline_starts, a.current_value, scale,
           frac, f"min_stdev_phi={a.min_stdev_phi:g}", a.output,
           min_bin_count=a.min_bin_count, sma=a.sma,
           baseline_label="no min_passing_steps filter")


if __name__ == "__main__":
    main()
