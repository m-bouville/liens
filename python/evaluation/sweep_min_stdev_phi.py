"""
DISPOSABLE: how many windows survive each min_stdev_phi threshold, and at which
times? min_stdev_phi drops FRAMES whose spatial std of phi is below threshold
(near-uniform / single-domain), which changes which steps are 'good' and thus
the window population -- so this REBUILDS via build_good_steps per swept value
(frame-free: statistics.csv only) and reports the RETENTION rate vs the baseline
(lowest) value, resolved by window start-time t.

Companion to sweep_min_std_deriv. min_passing_steps is fixed here (pass the
value you train with); min_stdev_phi is the swept axis.

With --normalized the swept threshold is applied to stdev_phi DIVIDED BY the
theoretical equilibrium amplitude sqrt(-a(T)/b) (a=a0(T-T0)) -- i.e. it sweeps
min_normalized_stdev_phi instead. Because that normalizer puts every
temperature's plateau at 1.0, the swept values then live on a ~0..1 scale (a
FRACTION of the ground state), not the raw-stdev scale, so pass values like
0.1 0.2 0.4 0.6 0.8. Same retention-vs-time readout; the point of the switch
is to see whether the temperature bias in which windows survive goes away.

    python -m evaluation.sweep_min_stdev_phi --base-path ../datasets --size 128 \
        --window-length 2 --min-step 1000 --min-passing-steps 12 \
        --min-stdev-phi 0.0 0.005 0.01 0.02 0.05 0.1 --max-runs 0

    python -m evaluation.sweep_min_stdev_phi --normalized ... \
        --min-stdev-phi 0.0 0.1 0.2 0.4 0.6 0.8
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
    p.add_argument("--min-passing-steps", type=int, default=12,
                   help="FIXED; dropped automatically for the baseline min_stdev_phi=0 build")
    p.add_argument("--max-dt", type=float, default=float("inf"))
    p.add_argument("--min-stdev-phi", type=float, nargs="+", default=None,
                   help="SWEPT (displayed rows). Retention denominator is the NO-FILTER "
                        "population, computed separately, so these are fraction-of-all-windows. "
                        "Default depends on --normalized: raw-scale when off, ~0..1 scale when on")
    p.add_argument("--normalized", action="store_true",
                   help="sweep min_NORMALIZED_stdev_phi (stdev_phi / equilibrium "
                        "amplitude sqrt(-a(T)/b)) instead of raw min_stdev_phi -- "
                        "swept values then live on a ~0..1 scale")
    p.add_argument("--current-value", type=float, default=None,
                   help="optional: value to mark '<- current' in table/plot")
    p.add_argument("--max-runs", type=int, default=0, help="0 = all runs")
    p.add_argument("--min-bin-count", type=int, default=10)
    p.add_argument("--sma", type=int, default=3)
    p.add_argument("--output", type=Path,
                   default=_PYTHON_ROOT.parent / "output" / "datasets"
                            / "128x128-min_stdev_phi_sweep.png")
    a = p.parse_args()

    # Mode-aware default for the swept axis: raw stdev_phi lives on a very
    # different scale than stdev_phi/amplitude. A raw-scale default under
    # --normalized would put most rows near "no filter" (0.002 of equilibrium is
    # nothing); the normalized default spans the ~0..1 fraction-of-ground-state
    # range where the filter actually bites. Explicit --min-stdev-phi overrides.
    if a.min_stdev_phi is None:
        a.min_stdev_phi = ([0.002, 0.005, 0.01, 0.02, 0.05, 0.5, 0.8, 0.9]
                           if a.normalized
                           else [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                                 0.1, 0.2, 0.3])

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

    values = sorted(v for v in set(a.min_stdev_phi) if v > 0)   # 0 is the baseline, not a row
    # Retention denominator: the NO-FILTER population (min_stdev_phi off, and
    # therefore min_passing_steps off too -- they're coupled). Retention is then
    # "fraction of ALL windows" surviving min_stdev_phi=v WITH min_passing_steps,
    # not a fraction of some already-filtered set.
    base_gs = build_good_steps(run_dirs, min_step=a.min_step,
                               min_stdev_phi=None, min_passing_steps=None)
    baseline_starts = window_start_times(base_gs, dts, a.window_length, a.max_dt)

    _filter_key = "min_normalized_stdev_phi" if a.normalized else "min_stdev_phi"
    starts = {}
    for msp in values:
        gs = build_good_steps(run_dirs, min_step=a.min_step,
                              min_passing_steps=a.min_passing_steps,
                              **{_filter_key: msp})
        starts[msp] = window_start_times(gs, dts, a.window_length, a.max_dt)

    _name = _filter_key
    _out = a.output
    if a.normalized and _out.name == "128x128-min_stdev_phi_sweep.png":
        _out = _out.with_name("128x128-min_normalized_stdev_phi_sweep.png")
    render(_name, values, starts, baseline_starts, a.current_value, scale,
           frac, f"min_passing_steps={a.min_passing_steps}", _out,
           min_bin_count=a.min_bin_count, sma=a.sma,
           baseline_label=f"no filter, {_name}=0")


if __name__ == "__main__":
    main()
