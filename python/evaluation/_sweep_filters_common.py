"""
DISPOSABLE shared helper for the filter sweeps (min_stdev_phi, min_passing_steps).

Both of those filter STEPS/RUNS, not per-window values, so unlike min_std_deriv
the window population itself changes with the threshold -- you can't post-hoc
threshold a fixed set, you rebuild via build_good_steps per value. This computes
per-value window start-times and renders the RETENTION rate (windows at each t
vs the baseline value) -- the analog of min_std_deriv's survival rate.

Frame-free: build_good_steps reads statistics.csv (stdev_phi) + metadata only.
"""
from pathlib import Path

import numpy as np

from utils import load_datasets as load


def dt_by_run(run_dirs):
    return {Path(r): load.read_metadata(Path(r) / "metadata.txt").dt for r in run_dirs}


def window_start_times(good_steps, dts, window_length, max_dt):
    """Start-step of every window formed from good_steps, applying max_dt --
    matches MicrostructureEvolutionDataset's own window loop."""
    starts = []
    finite = np.isfinite(max_dt)
    for run, kept in good_steps.items():
        dt = dts[run]
        for s in range(len(kept) - window_length + 1):
            win = kept[s:s + window_length]
            if finite and any((win[i + 1] - win[i]) * dt > max_dt
                              for i in range(window_length - 1)):
                continue
            starts.append(kept[s])
    return np.array(starts, dtype=float)


def _sma(y, w):
    if w <= 1:
        return y
    y = np.asarray(y, float)
    valid = ~np.isnan(y)
    num = np.convolve(np.where(valid, y, 0.0), np.ones(w), "same")
    den = np.convolve(valid.astype(float), np.ones(w), "same")
    return np.where(den > 0, num / den, np.nan)


def render(name, values, starts_by_value, baseline_starts, current, scale,
           frac_runs, min_stdev_phi_note, output_path, min_bin_count=10, sma=3,
           baseline_label=None):
    """Table + plot: RETENTION = windows at t under `value` / windows at t under
    the baseline value. Rows are the swept `values`; columns resolve it by t."""
    base_n = len(baseline_starts)
    targets = [1e4, 1e5, 1e6, 1e7]
    root10 = np.sqrt(10.0)
    base_band = {t: ((baseline_starts >= t / root10) & (baseline_starts < t * root10)).sum()
                 for t in targets}
    t_hdr = "  ".join(f"ret%@{t:.0e}".rjust(9) for t in targets)

    _blabel = baseline_label if baseline_label is not None else f"{values[0]:g}"
    print(f"\n{name}: baseline ({_blabel}) has {base_n} windows  [{min_stdev_phi_note}]")
    print(f"\n{name:>14}  {'windows':>8}  {'ret%':>6}  {'full sweep':>11}   {t_hdr}")
    print("-" * (46 + len(t_hdr)))
    for v in values:
        st = starts_by_value[v]
        m = "  <- current" if (current is not None and abs(v - current) < 1e-12) else ""
        cells = []
        for t in targets:
            b = base_band[t]
            cur = ((st >= t / root10) & (st < t * root10)).sum()
            cells.append(f"{100 * cur / b:8.1f}%" if b else "      n/a")
        ret = 100 * len(st) / base_n if base_n else float("nan")
        print(f"{v:14g}  {len(st):8d}  {ret:5.1f}%  {int(round(len(st) * scale)):11d}   "
              f"{'  '.join(cells)}{m}")

    if output_path is None:
        return
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    uniq_t = np.unique(baseline_starts)
    base_cnt = {t: c for t, c in zip(*np.unique(baseline_starts, return_counts=True))}
    keep_t = np.array([base_cnt.get(t, 0) >= min_bin_count for t in uniq_t])

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(values)))
    for v, col in zip(values, cmap):
        st = starts_by_value[v]
        cnt = {t: c for t, c in zip(*np.unique(st, return_counts=True))}
        ret = np.array([100.0 * cnt.get(t, 0) / base_cnt[t] if base_cnt.get(t) else np.nan
                        for t in uniq_t])
        ret = np.where(keep_t, ret, np.nan)
        ret = _sma(ret, sma)
        is_cur = current is not None and abs(v - current) < 1e-12
        ax.plot(uniq_t, ret, "-o", ms=2, color=("tab:red" if is_cur else col),
                lw=(2.5 if is_cur else 1.2),
                label=f"{v:g}  ({100 * len(st) / base_n:.0f}%)" + ("  <- current" if is_cur else ""))
    ax.set_xscale("log")
    ax.set_ylim(0, 102)
    ax.set_xlabel("window start time t (step)")
    ax.set_ylabel("retention at each t  [%]  (windows here / baseline windows here)")
    sma_note = f", SMA={sma}" if sma > 1 else ""
    ax.set_title(f"which times does {name} discard?  ({100 * frac_runs:.0f}% of runs{sma_note})")
    ax.legend(title=f"{name} (overall retained)", fontsize=7, ncol=2, loc="lower center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"\nsaved {output_path}")
