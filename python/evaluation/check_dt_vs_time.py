"""Break the t / Delta_t collinearity by pairing NON-ADJACENT frames.

THE PROBLEM. The save schedule is self-similar -- {1, 1.15, 1.3, 1.5, 1.75, 2,
2.25, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9} x 10^k per decade -- so

    corr(log t, log Delta_t) = 0.997      Delta_t / t in [0.083, 0.167]

and per-Delta_t coverage of t spans a factor of ~1.4. Consequence: EVERY
"error vs dt decade" table in this project is equally a table of error vs
TIME. The max_dt=200 ceiling, the dt=125 spike corner, the dt=25 collapse --
none of them currently distinguishes "large step" from "late time".

THE FIX, and why it is nearly free. Nothing forces a window to use CONSECUTIVE
saved steps. Taking (i, i+k) for k = 1, 2, 4, 8 gives, at fixed t, a Delta_t
sweep; and at fixed Delta_t, coverage of t across the whole range. That turns
the diagonal into a filled (t, Delta_t) grid, using only the per-run latents
already cached -- no retraining, no dataset rebuild, no re-encoding.

WHAT IT MEASURES. For each pair it reports the EULER-ONLY error
||z0(t+Delta_t) - (z0(t) + z1(t)*Delta_t)|| -- z1's own one-step error, the
quantity the oracle attribution and every dt-decade table are built on. It
deliberately does NOT run f_theta: f_theta is trained at the dataset's own
spacing, so scoring it at k>1 would confound "this dt is hard" with "f_theta
never saw this dt".

READING THE OUTPUT. The table is error binned by (t, Delta_t). Read it two
ways:

  * DOWN a Delta_t column: error vs t at FIXED step size. Varies -> time (or
    the coarsening state it proxies) matters in its own right.
  * ACROSS a t row: error vs Delta_t at FIXED time. Varies -> the step size
    matters in its own right.

If only the columns vary, max_dt is the wrong lever and the ceiling should be
on t or on a physical progress variable. If only the rows vary, max_dt is
right and the current value can be chosen on evidence rather than against a
confounded measurement.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from evaluation._latent_eval import _load_ae_f_theta_and_dataset
from orchestration.paths import default_latent_cache_dir

_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/

# None is a MEANINGFUL value (caching off), so it cannot double as "not
# specified" -- see check_parameter_dependence, which learned the same thing.
_UNSET_CACHE = object()


def collect_pairs(dataset, strides: list[int], max_pairs_per_run: int | None = None
                  ) -> list[dict]:
    """Every (i, i+k) pair available in the cached latents.

    Walks each run's own saved-step list directly rather than the dataset's
    window index: the index only contains ADJACENT windows, which is the
    collinearity being escaped.
    """
    rows: list[dict] = []
    n_runs = len(dataset._run_data)
    for run_idx in range(n_runs):
        state = dataset._run_data[run_idx]
        deriv = dataset._run_data_deriv[run_idx]
        steps = dataset._run_steps[run_idx]
        scale = dataset._run_dt_scale[run_idx]
        theta = dataset._run_theta[run_idx]
        n = len(steps)
        n_this_run = 0
        for k in strides:
            for i in range(0, n - k):
                dt = (steps[i + k] - steps[i]) * scale
                t = steps[i] * scale
                # euler-only: z0 + z1*dt, the quantity every dt-decade table
                # in this project is built on
                pred = state[i] + deriv[i] * dt
                err = float(torch.linalg.vector_norm(state[i + k] - pred))
                denom = float(torch.linalg.vector_norm(state[i + k] - state[i]))
                rows.append({
                    "run_idx": run_idx, "k": k, "t": t, "dt": dt,
                    "err": err,
                    # normalized by the actual CHANGE over the pair: an error
                    # of 0.1 across a step where nothing moved is a different
                    # statement from the same error across a large excursion
                    "rel_err": err / denom if denom > 0 else float("nan"),
                    "theta0": float(theta[0]) if theta.numel() else float("nan"),
                })
                n_this_run += 1
                if max_pairs_per_run is not None and n_this_run >= max_pairs_per_run:
                    break
            if max_pairs_per_run is not None and n_this_run >= max_pairs_per_run:
                break
    return rows


def _log_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return np.array([])
    return np.geomspace(positive.min(), positive.max(), n_bins + 1)


def grid_table(rows: list[dict], n_t_bins: int = 5, n_dt_bins: int = 5,
                field: str = "rel_err") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean `field` binned by (t, dt). Returns (t_edges, dt_edges, mean, count)."""
    t = np.array([r["t"] for r in rows])
    dt = np.array([r["dt"] for r in rows])
    v = np.array([r[field] for r in rows])
    t_edges, dt_edges = _log_bins(t, n_t_bins), _log_bins(dt, n_dt_bins)
    mean = np.full((n_t_bins, n_dt_bins), np.nan)
    count = np.zeros((n_t_bins, n_dt_bins), dtype=int)
    if t_edges.size == 0 or dt_edges.size == 0:
        return t_edges, dt_edges, mean, count
    ti = np.clip(np.digitize(t, t_edges) - 1, 0, n_t_bins - 1)
    di = np.clip(np.digitize(dt, dt_edges) - 1, 0, n_dt_bins - 1)
    for a in range(n_t_bins):
        for b in range(n_dt_bins):
            sel = (ti == a) & (di == b) & np.isfinite(v)
            count[a, b] = int(sel.sum())
            if count[a, b]:
                mean[a, b] = float(v[sel].mean())
    return t_edges, dt_edges, mean, count


def variation_per_decade(mean: np.ndarray, centers: np.ndarray, axis: int) -> float:
    """Median per-decade factor along `axis`, NOT a raw max/min ratio.

    The raw ratio is systematically biased on a triangular grid, and this grid
    IS triangular: strides give dt >= the local gap and the gap grows as ~t/8,
    so pairing can only make dt larger, never smaller. Each line therefore
    spans a DIFFERENT range of the other variable, and comparing their ratios
    compares ranges as much as effects.

    Measured: the dt=6.24e3 column reported 108x against 24x for dt=4.95e4,
    which read as a non-monotonic outlier. Per decade they are 8.74x and
    9.10x -- the same effect over different spans. The three small-dt columns
    likewise agree at 1.35-1.40x. Two clean regimes, invisible under the raw
    ratio.

    `centers` are the bin centres along the axis being traversed, in the same
    units as the bins (a log-spaced geometric sequence), so the normalisation
    is by the decades actually covered.
    """
    factors = []
    for line in (mean if axis == 1 else mean.T):
        idx = np.flatnonzero(np.isfinite(line))
        if idx.size < 2:
            continue
        lo, hi = idx[0], idx[-1]
        span = centers[hi] / centers[lo]
        if span <= 1 or line[lo] <= 0:
            continue
        factors.append(float((line[hi] / line[lo]) ** (1.0 / np.log10(span))))
    return float(np.median(factors)) if factors else float("nan")


def joint_exponents(rows: list[dict], field: str = "rel_err") -> tuple[float, float, float]:
    """Count-free log-log fit of err ~ t^a * dt^b over the raw pairs.

    The decisive summary, because it uses every pair at its own (t, dt) rather
    than a binned line, so the triangular geometry cannot bias it the way the
    per-line ratios do. On the real off-distribution run the per-line medians
    said "dt dominates 2:1" while this fit gives a=0.52, b=0.47 -- equal to
    within 10%.

    Returns (a, b, r2).
    """
    t = np.array([r["t"] for r in rows], dtype=float)
    dt = np.array([r["dt"] for r in rows], dtype=float)
    e = np.array([r[field] for r in rows], dtype=float)
    ok = np.isfinite(t) & np.isfinite(dt) & np.isfinite(e) & (t > 0) & (dt > 0) & (e > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    A = np.column_stack([np.log10(t[ok]), np.log10(dt[ok]), np.ones(int(ok.sum()))])
    y = np.log10(e[ok])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), r2


def regime_split(mean: np.ndarray, t_centers: np.ndarray, dt_centers: np.ndarray
                  ) -> list[tuple[float, float]]:
    """Per-decade t effect for each dt column -- the interaction, made visible.

    A single "does dt or t dominate" verdict cannot express what the data
    actually shows: below dt~1e3 time costs ~1.4x per decade, above it ~9x.
    The dt axis does not merely add error, it changes how much t matters.
    """
    out = []
    for j in range(mean.shape[1]):
        col = mean[:, j]
        idx = np.flatnonzero(np.isfinite(col))
        if idx.size < 2:
            continue
        lo, hi = idx[0], idx[-1]
        span = t_centers[hi] / t_centers[lo]
        if span > 1 and col[lo] > 0:
            out.append((float(dt_centers[j]),
                        float((col[hi] / col[lo]) ** (1.0 / np.log10(span)))))
    return out


def split_regimes(split: list[tuple[float, float]], min_gap: float = 3.0):
    """Cut the per-decade factors at their LARGEST GAP, or return None.

    Split at the MEDIAN and a 1.40x column lands in the HIGH group -- factors
    were [1.39, 1.40, 1.35, 8.78, 9.10], median 1.40, and "f >= median" swept
    up the 1.40. The message then read "below dt~2.22e3 ... above dt~279":
    two overlapping ranges, with a high-group median of 5.1x that had averaged
    a low column into the high ones. Reported.

    The gap is the right cut because a regime is a DISCONTINUITY: the three
    low columns agree to within 4% of each other, the two high ones to within
    4%, and the step between them is 6.3x.

    Returns (low_dt_max, low_median, high_dt_min, high_median, gap), or None
    when the factors are smooth -- calling a gradual trend a threshold would
    invent structure, the opposite error and just as misleading.
    """
    if len(split) < 2:
        return None
    factors = np.array([f for _, f in split], dtype=float)
    dts = np.array([d for d, _ in split], dtype=float)
    sorted_f = factors[np.argsort(factors)]
    gaps = sorted_f[1:] / sorted_f[:-1]
    if not gaps.size or gaps.max() <= min_gap:
        return None
    cut = float(sorted_f[int(np.argmax(gaps))])
    low, high = factors <= cut, factors > cut
    if not low.any() or not high.any():
        return None
    return (float(dts[low].max()), float(np.median(factors[low])),
            float(dts[high].min()), float(np.median(factors[high])), float(gaps.max()))


def check_dt_vs_time(lds_checkpoint_path: Path, strides: tuple[int, ...] = (1, 2, 4, 8, 16),
                      base_path: Path | None = None, size: int | None = None,
                      min_step: int | None = None, min_stdev_phi: float | None = None,
                      min_passing_steps: int | None = None, device: str | None = None,
                      n_t_bins: int = 5, n_dt_bins: int = 5,
                      max_pairs_per_run: int | None = None,
                      max_dt: float | None = None,
                      latent_cache_dir=_UNSET_CACHE) -> dict:
    """Run the (t, Delta_t) grid diagnostic and print its table."""
    ctx = _load_ae_f_theta_and_dataset(
        lds_checkpoint_path, min_step, min_stdev_phi, min_passing_steps,
        base_path, size, None, 256, 2, None, None, device,
        # max_dt=None DELIBERATELY: the ceiling is what is under test, so
        # applying it would truncate away the very pairs that break the
        # collinearity.
        max_dt=max_dt,
        # THE SHARED CACHE. This diagnostic reads a frozen encoder out of a
        # checkpoint and encodes the whole test population -- exactly the case
        # the cache exists for, and with max_dt unset there is no prefix
        # truncation, so it encodes EVERY frame of every run. Omitting it (as
        # the first version did) makes this the most expensive diagnostic in
        # the project for no reason.
        latent_cache_dir=(default_latent_cache_dir(_PYTHON_ROOT)
                           if latent_cache_dir is _UNSET_CACHE else latent_cache_dir),
    )
    # _load_ae_f_theta_and_dataset returns a 7-tuple; the dataset is index 4.
    # Positional rather than by attribute: it is a plain tuple, and getting
    # this wrong would silently bind f_theta or the decoder to `dataset`.
    dataset = ctx[4]

    rows = collect_pairs(dataset, list(strides), max_pairs_per_run=max_pairs_per_run)
    if not rows:
        raise ValueError("no pairs collected -- the dataset has no runs with enough "
                          "kept steps for the requested strides")

    t_edges, dt_edges, mean, count = grid_table(rows, n_t_bins, n_dt_bins)
    print(f"\n{len(rows)} non-adjacent pairs from {len(dataset._run_data)} runs, "
          f"strides {list(strides)}\n")
    print("mean relative euler-only error, binned by (t, dt).")
    print("  DOWN a column = error vs t at fixed dt;  ACROSS a row = error vs dt at fixed t\n")
    header = "        t \\ dt" + "".join(f"{dt_edges[j]:>11.3g}" for j in range(len(dt_edges) - 1))
    print(header)
    for a in range(mean.shape[0]):
        cells = "".join("          ." if not np.isfinite(mean[a, b])
                         else f"{mean[a, b]:>11.4f}" for b in range(mean.shape[1]))
        print(f"{t_edges[a]:>13.3g}{cells}")
    print("\n(counts)")
    for a in range(count.shape[0]):
        print(f"{t_edges[a]:>13.3g}" + "".join(f"{count[a, b]:>11d}" for b in range(count.shape[1])))

    # Bin CENTRES (geometric), not edges: the per-decade normalisation must
    # use the span actually covered by the populated cells.
    t_centers = np.sqrt(t_edges[:-1] * t_edges[1:])
    dt_centers = np.sqrt(dt_edges[:-1] * dt_edges[1:])

    down = variation_per_decade(mean, t_centers, axis=0)
    across = variation_per_decade(mean, dt_centers, axis=1)
    print(f"\n  median per-DECADE effect of t  (down a dt column):  {down:.2f}x per decade")
    print(f"  median per-DECADE effect of dt (across a t row):   {across:.2f}x per decade")
    print("  (per decade, NOT raw max/min: this grid is triangular -- strides can only "
          "make dt\n   LARGER, never smaller -- so each line spans a different range and "
          "raw ratios\n   compare ranges as much as effects)")

    a, b, r2 = joint_exponents(rows)
    print(f"\n  joint fit over all {len(rows)} pairs:  err ~ t^{a:+.3f} * dt^{b:+.3f}   "
          f"(R^2={r2:.3f})")
    print("  (every pair at its OWN (t, dt), so the triangular geometry cannot bias it)")

    split = regime_split(mean, t_centers, dt_centers)
    if split:
        print("\n  per-decade effect of t, BY dt column -- the interaction:")
        for dt_c, factor in split:
            print(f"    dt~{dt_c:>10.3g}: {factor:>7.2f}x per decade of t")
        regimes = split_regimes(split)
        if regimes is not None:
            lo_dt, lo_f, hi_dt, hi_f, gap = regimes
            print(f"\n  -> TWO REGIMES, split at the largest gap in the per-decade factors "
                  f"({gap:.1f}x, which is why this is a threshold and not a trend):\n"
                  f"       dt <= {lo_dt:>10.3g}:  t costs {lo_f:.2f}x per decade\n"
                  f"       dt >= {hi_dt:>10.3g}:  t costs {hi_f:.2f}x per decade\n"
                  f"     dt does not merely ADD error -- it changes how much t matters, so a "
                  f"ceiling on dt is really a way of staying below that threshold.")

    if np.isfinite(a) and np.isfinite(b):
        ratio = b / a if a != 0 else float("inf")
        if 0.5 <= ratio <= 2.0:
            print(f"\n  -> t and dt matter COMPARABLY (exponents within {max(ratio, 1/ratio):.1f}x). "
                  f"Neither alone explains the error, and max_dt cannot address the t term: "
                  f"at fixed small dt, t still costs ~{down:.1f}x per decade, which over the "
                  f"decades a long run spans is the part no ceiling touches.")
        elif ratio > 2.0:
            print("\n  -> dt dominates. max_dt is the right lever and can be set on evidence.")
        else:
            print("\n  -> TIME dominates. max_dt is the wrong lever: the ceiling belongs on t, "
                  "or on a physical progress variable such as stdev_phi.")

    return {"rows": rows, "t_edges": t_edges, "dt_edges": dt_edges,
            "mean": mean, "count": count, "down": down, "across": across,
            "a": a, "b": b, "r2": r2, "regimes": split}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lds-checkpoint", type=Path, required=True)
    p.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                help="k values for the (i, i+k) pairing. Up to 16 by default: on this "
                     "schedule k=4 spans only ~2.6x in dt at fixed t, inside the "
                     "schedule's own scatter, so it separates nothing")
    p.add_argument("--base-path", type=Path, default=None)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--min-step", type=int, default=None)
    p.add_argument("--min-stdev-phi", type=float, default=None)
    p.add_argument("--min-passing-steps", type=int, default=None)
    p.add_argument("--max-dt", type=float, default=None,
                    help="left UNSET by default: the ceiling is what is under test")
    p.add_argument("--n-t-bins", type=int, default=5)
    p.add_argument("--n-dt-bins", type=int, default=5)
    p.add_argument("--max-pairs-per-run", type=int, default=None)
    p.add_argument("--no-latent-cache", action="store_true",
                    help="disable the shared latent cache for this run")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    check_dt_vs_time(
        args.lds_checkpoint, strides=tuple(args.strides), base_path=args.base_path,
        size=args.size, min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        min_passing_steps=args.min_passing_steps, device=args.device,
        n_t_bins=args.n_t_bins, n_dt_bins=args.n_dt_bins,
        max_pairs_per_run=args.max_pairs_per_run, max_dt=args.max_dt,
        latent_cache_dir=None if args.no_latent_cache else _UNSET_CACHE,
    )


if __name__ == "__main__":
    main()
