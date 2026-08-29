"""
DISPOSABLE one-off: how many windows survive each min_std_deriv threshold?

min_std_deriv drops a candidate window when the SPATIAL std of its first
transition's raw-pixel time-derivative,

    std_space( (phi[start+1] - phi[start]) / dt ),

falls below the threshold (see MicrostructureEvolutionDataset's own filter).
This computes that value for every candidate window ONCE (at a fixed
min_stdev_phi), then reports how many windows clear a range of min_std_deriv
thresholds -- the survival curve, with the 0.15e-3 you run marked. Answers
whether raising min_std_deriv can reach the destabilising (frozen-field)
windows, or whether they sit above it -- and at what cost to the training set.

min_stdev_phi is a FIXED argument here (it filters frames, changing the window
population -- pass the value you train with). min_std_deriv is the swept axis:
it's a post-hoc threshold on a per-window value, so it sweeps for free once the
frames are loaded, no rebuild.

Reuses the dataset's OWN construction (encoder=None, min_std_deriv=None) so the
window population matches a training run's exactly. Raw frames sit in memory, so
subsample with --max-runs; fraction kept is population-stable, absolute counts
are extrapolated to the full sweep and labelled. Run from python/.

    python -m evaluation.sweep_min_std_deriv --base-path ../datasets --size 128 \
        --window-length 6 --min-step 2000 --min-stdev-phi 0.01 \
        --min-passing-steps 12 --max-runs 400
"""
import argparse
from pathlib import Path

import numpy as np

from training.datasets import MicrostructureEvolutionDataset, complete_run_dirs

_PYTHON_ROOT = Path(__file__).resolve().parent.parent


def _sma(y, w):
    """Centered, NaN-aware simple moving average over w points (w<=1 = no-op)."""
    if w <= 1:
        return y
    y = np.asarray(y, float)
    valid = ~np.isnan(y)
    num = np.convolve(np.where(valid, y, 0.0), np.ones(w), "same")
    den = np.convolve(valid.astype(float), np.ones(w), "same")
    return np.where(den > 0, num / den, np.nan)


def _window_std_deriv_values(ds):
    """Per candidate window: (the quantity min_std_deriv filters on,
    the window's START step t) -- t lets us see WHICH times get discarded."""
    vals, t_start = [], []
    for run_idx, start in ds._index:
        run_data = ds._run_data[run_idx]              # raw phi frames (raw-pixel mode)
        steps = ds._run_steps[run_idx]
        dt = ds._run_dt_scale[run_idx]
        first_dt = (steps[start + 1] - steps[start]) * dt
        first_deriv = (run_data[start + 1] - run_data[start]) / first_dt
        vals.append(first_deriv.std().item())
        t_start.append(steps[start])                  # window start time (step)
    return np.array(vals, dtype=float), np.array(t_start, dtype=float)


def sweep(base_path, size, window_length, min_step, min_stdev_phi,
          min_passing_steps, max_dt, max_runs, deriv_grid, current, output_path,
          min_bin_count=10, sma=3):
    run_dirs = complete_run_dirs(base_path, size, size)
    n_total_runs = len(run_dirs)
    if max_runs and len(run_dirs) > max_runs:
        idx = np.linspace(0, len(run_dirs) - 1, max_runs).round().astype(int)
        run_dirs = [run_dirs[i] for i in sorted(set(idx))]
    frac_runs = len(run_dirs) / n_total_runs
    scale = n_total_runs / len(run_dirs)       # extrapolate subsample -> full sweep
    print(f"using {len(run_dirs)}/{n_total_runs} runs ({100 * frac_runs:.1f}% subsample)"
          if frac_runs < 1 else f"using all {n_total_runs} runs")

    # min_passing_steps is coupled to min_stdev_phi (it counts steps that PASS
    # that threshold), so drop it when there's no frame filter.
    _msp = min_stdev_phi if (min_stdev_phi and min_stdev_phi > 0) else None
    _mps = min_passing_steps if _msp is not None else None
    ds = MicrostructureEvolutionDataset(
        run_dirs, encoder=None, window_length=window_length,
        min_step=min_step, min_stdev_phi=_msp,
        min_passing_steps=_mps, max_dt=max_dt, min_std_deriv=None)

    vals, t_start = _window_std_deriv_values(ds)
    n = len(vals)
    if n == 0:
        print("no candidate windows -- check the filter params")
        return

    # thresholds: TWO PER DECADE across a fixed 1e-8 .. 1e-4 range, plus the
    # current value -- the SAME grid drives both the table and the plot below.
    grid = (np.unique(np.concatenate([
        10.0 ** (np.arange(-16, -7) / 2.0),
        [] if current is None else [current]]))   # -8 .. -4 step 0.5
        if deriv_grid is None else np.unique(np.array(
            deriv_grid + ([] if current is None else [current]))))

    print(f"\nper-window std_space(dphi/dt) over {n} windows at min_stdev_phi="
          f"{min_stdev_phi:g}  (min={vals.min():.3e}  median={np.median(vals):.3e}  "
          f"max={vals.max():.3e})")

    # survival rate at specific times: windows in a half-decade band around each
    # target t (exact steps needn't exist on the schedule). Shows directly WHICH
    # times a threshold discards -- read a row across t=1e4..1e7.
    targets = [1e4, 1e5, 1e6, 1e7]
    root10 = np.sqrt(10.0)
    t_bands = {t: (t_start >= t / root10) & (t_start < t * root10) for t in targets}
    t_hdr = "  ".join(f"surv%@{t:.0e}".rjust(9) for t in targets)

    print(f"\n{'min_std_deriv':>13}  {'kept':>8}  {'kept%':>6}  {'full sweep':>11}   {t_hdr}")
    print("-" * (48 + len(t_hdr)))
    for thr in grid:
        kept = int((vals >= thr).sum())
        mark = "  <- current" if (current is not None and abs(thr - current) < 1e-12) else ""
        cells = []
        for t in targets:
            m = t_bands[t]
            cells.append(f"{100 * (vals[m] >= thr).mean():8.1f}%" if m.sum() else "      n/a")
        print(f"{thr:13.3e}  {kept:8d}  {100 * kept / n:5.1f}%  {int(round(kept * scale)):11d}   "
              f"{'  '.join(cells)}{mark}")

    # how many windows sit in each t-band (context for the rates above)
    print("\n  windows per t-band: " + "  ".join(
        f"{t:.0e}:{int(t_bands[t].sum())}" for t in targets))

    if output_path is not None:
        import matplotlib.pyplot as plt
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Data is discrete (~71-74 shared save-steps), so plot the SURVIVAL RATE
        # at each ACTUAL start-time -- no binning, no bin-position drift. For each
        # distinct t, rate = fraction of windows starting there that clear the
        # threshold. x-positions are the schedule's own steps, identical across
        # any subsample; only their per-point population (and thus noise) varies.
        # A curve high at small t and dipping at large t => that threshold
        # discards LATE windows; a flat curve => it discards uniformly in time.
        uniq_t, inv = np.unique(t_start, return_inverse=True)
        counts_t = np.bincount(inv)                       # windows per distinct t
        keep_t = counts_t >= min_bin_count                # blank noisy sparse steps

        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.cm.viridis(np.linspace(0, 1, len(grid)))
        for thr, col in zip(grid, cmap):
            surv = np.bincount(inv, weights=(vals >= thr).astype(float), minlength=len(uniq_t))
            rate = np.where(keep_t, surv / np.maximum(counts_t, 1), np.nan)
            rate = _sma(rate, sma)
            is_cur = current is not None and abs(thr - current) < 1e-12
            ax.plot(uniq_t, 100.0 * rate, "-o", ms=2,
                    color=("tab:red" if is_cur else col),
                    lw=(2.5 if is_cur else 1.2),
                    label=f"{thr:.1e}  ({(vals >= thr).mean():.0%} kept)"
                          + ("  <- current" if is_cur else ""))
        ax.set_xscale("log")
        ax.set_ylim(0, 102)
        ax.set_xlabel("window start time t (step)")
        ax.set_ylabel("survival rate at each t  [%]  (survivors / windows at that step)")
        sma_note = f", SMA={sma}" if sma > 1 else ""
        ax.set_title(f"which times does min_std_deriv discard?  "
                     f"(min_stdev_phi={min_stdev_phi:g}, n={n}, {100 * frac_runs:.0f}% of runs{sma_note})")
        ax.legend(title="min_std_deriv (overall kept)", fontsize=7, ncol=2, loc="lower left")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        print(f"\nsaved {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-path", type=Path, default=_PYTHON_ROOT.parent / "datasets")
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--window-length", type=int, default=2,
                   help="2 = full per-transition coverage (dataset characterisation, the "
                        "default). Set to n_rollout_steps+1 only to match a training population")
    p.add_argument("--min-step", type=int, default=0,
                   help="0 = include all steps (default). Raise to drop early steps")
    p.add_argument("--min-stdev-phi", type=float, default=0.0,
                   help="FIXED (filters frames). Default 0 = no frame filter, so the late/"
                        "frozen windows stay in the denominator. Pass 0.01 to match training")
    p.add_argument("--min-passing-steps", type=int, default=12)
    p.add_argument("--max-dt", type=float, default=float("inf"))
    p.add_argument("--min-std-deriv", type=float, nargs="+", default=None,
                   help="SWEPT threshold values (default log-spaced across the observed range)")
    p.add_argument("--max-runs", type=int, default=400,
                   help="subsample this many runs (memory holds raw frames); 0 = all")
    p.add_argument("--current-value", type=float, default=None,
                   help="the min_std_deriv you're running, marked in table/plot")
    p.add_argument("--min-bin-count", type=int, default=10,
                   help="blank save-steps with fewer than this many windows (noisy rate)")
    p.add_argument("--sma", type=int, default=3,
                   help="centered moving-average window (in x-axis / time points) applied "
                        "to each survival curve; 1 = off, raise for a choppier tail")
    p.add_argument("--output", type=Path,
                   default=_PYTHON_ROOT.parent / "output" / "datasets"
                            / "128x128-min_std_deriv_sweep.png",
                   help="survival-curve png (default output/datasets/128x128-...)")
    a = p.parse_args()
    sweep(a.base_path, a.size, a.window_length, a.min_step, a.min_stdev_phi,
          a.min_passing_steps, a.max_dt, a.max_runs, a.min_std_deriv, a.current_value, a.output,
          min_bin_count=a.min_bin_count, sma=a.sma)


if __name__ == "__main__":
    main()
