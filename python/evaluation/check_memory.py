"""
Measure what ACTUALLY predicts training memory, batch by batch.

WHY THIS EXISTS. Six successive structural models of per-batch memory were
built during the truncated-BPTT work, and every one was falsified by the next
measurement: raw sub-step count, retained depth min(count, k), span-aware
depth, span bounded by batch size, the realised per-transition cost (which
TRIPLED on a run whose measured peak HALVED), and a bytes-per-unit constant
that took six different values (1700-77000) across runs. The training-side
answer was to stop modelling and feedback-control on measured bytes
(MemoryGovernor) -- which works, but explains nothing, and any deeper design
decision (e.g. homogenising counts within a batch) needs the explanation.

So this measures, per batch, on the real checkpoint and the real sampler:

  * measured peak ALLOCATED bytes for one full training step
    (forward + backward, same rollout shape as train_lds), and
  * every candidate predictor of it, side by side, with a fit and residuals
    per predictor, so which one tracks reality is a TABLE rather than an
    argument.

It also measures the MASKED-LOOP WASTE per batch: the fraction of f
evaluations spent on already-arrived samples (h zeroed). That number is the
cost of count heterogeneity within a batch, and therefore also exactly what
rounding counts up to the batch max would legitimise -- the compute half of
that design question, quantified on real batches instead of asserted.

Selection: measuring every batch would take a training epoch's worth of
compute, so a SPREAD is chosen -- the cheapest and deepest batches by
predicted cost, quantiles between, and the widest-span batch, which is where
the models disagree most.

On CPU the memory columns are n/a (CUDA peak-allocation counters are the
measurement); the batch structure, predictors and waste are still reported,
so the sampler-side numbers can be checked anywhere.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from evaluation._latent_eval import _load_ae_f_theta_and_dataset
from training.dt_bucketing import BudgetedBatchSampler, estimate_window_costs


def select_batches(batches, costs, n_probe: int = 8) -> list[int]:
    """Indices of the batches to measure: a cost spread plus the widest span.

    Quantiles of PREDICTED cost (len x max estimated count) from cheapest to
    deepest, because memory should rise along that axis under every candidate
    model -- where the models differ is off-axis, so the single widest-SPAN
    batch is always included: span is where per-sample truncation's arrival
    segments multiply and where the models were furthest apart.
    """
    if not batches:
        return []
    costs = np.asarray(costs, dtype=np.float64)
    predicted = np.array([len(b) * costs[list(b)].max() for b in batches])
    spans = np.array([costs[list(b)].max() - costs[list(b)].min() for b in batches])
    order = np.argsort(predicted, kind="stable")
    qs = np.linspace(0.0, 1.0, num=max(n_probe - 1, 2))
    picks = {int(order[int(round(q * (len(order) - 1)))]) for q in qs}
    picks.add(int(np.argmax(spans)))
    return sorted(picks)


def masked_loop_waste(counts: np.ndarray) -> float:
    """Fraction of the batch's f evaluations spent on ARRIVED samples.

    The masked loop runs every sample to the batch max, zeroing h for the
    arrived ones -- each such iteration still evaluates f on the full batch.
    waste = 1 - mean(count) / max(count). This is simultaneously:
      * the compute cost of count heterogeneity within a batch today, and
      * exactly the extra integration that rounding counts up to the batch
        max would turn from waste into (unrequested) refinement -- i.e. the
        number that decides whether that design is cheap or expensive, batch
        by batch, rather than on average.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0 or counts.max() <= 0:
        return 0.0
    return float(1.0 - counts.mean() / counts.max())


def predictor_table(counts: np.ndarray, truncate_bptt: int | None) -> dict:
    """Every candidate memory predictor for ONE batch, in sample-substeps.

    Kept deliberately including the FALSIFIED ones: the point of the
    diagnostic is to show, on the same row, which of them tracks the measured
    bytes and which do not -- deleting the losers would turn the table back
    into an assertion.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.size
    if n == 0:
        return {}
    lo, hi = float(counts.min()), float(counts.max())
    k = float(truncate_bptt) if truncate_bptt else None
    out = {
        "n_windows": n,
        "count_lo": lo,
        "count_hi": hi,
        "span": hi - lo,
        "raw": n * hi,                                  # no truncation model
        "retained": n * (min(hi, k) if k else hi),      # min(count, k)
        "span_aware": n * (min(hi, n * k, (hi - lo) + k) if k else hi),
        "arrival_segments": (min(n, int(np.ceil((hi - lo) / k)) + 1, int(np.ceil(hi / k)))
                              if k else 1),
    }
    out["segments_x_k"] = n * out["arrival_segments"] * (k or hi)
    return out


def _fit_and_residuals(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares y ~ a*x through the origin; returns (a, max relative
    residual). Through the origin because every candidate is a count of
    sample-substeps and the claim under test is proportionality."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = (x > 0) & (y > 0)
    if good.sum() < 2:
        return float("nan"), float("nan")
    a = float((x[good] * y[good]).sum() / (x[good] ** 2).sum())
    rel = np.abs(a * x[good] - y[good]) / y[good]
    return a, float(rel.max())


def fit_power_law(n: np.ndarray, hi: np.ndarray, y: np.ndarray,
                   lo_p: float = 0.3, hi_p: float = 1.5) -> tuple[float, float, float]:
    """Best n * hi**p fit to measured bytes; returns (p, bytes_per_unit, worst).

    Included because the FIXED predictors all assume memory is proportional to
    some count of sample-substeps, and measurement says otherwise: the
    per-sample-per-substep figure falls monotonically with depth (9.9 -> 6.0
    x1e-3 MiB over n_max 43 -> 470 on a real 3b checkpoint). A single fitted
    exponent captures that where p=1 cannot.

    Reported WITH its exponent, so a p far from 1 is visible as the finding it
    is rather than hidden inside a constant. Beware over-reading it: this fits
    two parameters to one batch per row, and on six rows p was only pinned to
    about +/-0.08.
    """
    best = (float("nan"), float("nan"), float("inf"))
    for p in np.arange(lo_p, hi_p + 1e-9, 0.01):
        a, worst = _fit_and_residuals(n * hi ** p, y)
        if np.isfinite(worst) and worst < best[2]:
            best = (float(p), a, worst)
    return best


def _fit_with_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares y ~ b*x + c; returns (slope, intercept, max rel residual).

    THROUGH THE ORIGIN IS THE WRONG DEFAULT and this exists to show it. Model
    parameters, optimizer state and the batch tensors cost the same whatever
    the depth, so a proportional fit has to bend the slope to absorb them and
    every model then looks worse than it is: on the fixed-n sweep, raw n*depth
    went from 78.1% through the origin to 29.6% with a 67 MiB constant.

    The intercept is also a NUMBER WORTH READING -- it is the floor below
    which no batching change can take the run.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = (x > 0) & (y > 0)
    if good.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([x[good], np.ones(good.sum())]).T
    (b, c), *_ = np.linalg.lstsq(A, y[good], rcond=None)
    rel = np.abs(b * x[good] + c - y[good]) / y[good]
    return float(b), float(c), float(rel.max())


def fixed_depth_probe_batches(costs: np.ndarray, sizes: tuple[int, ...] = (128, 256, 512, 1024),
                               quantile: float = 0.5) -> list[np.ndarray]:
    """Batches of VARYING size drawn from one narrow depth slice.

    The complement of fixed_n_probe_batches, and needed for the same reason
    the first one was: with only a depth sweep, the per-window cost and the
    per-depth cost are still entangled -- fitting A*n + B*n*d^p to the batch
    table gave A = -1088 KiB/window and predicted NEGATIVE memory at depth 8.
    Varying n while depth is held fixed identifies A directly; the two sweeps
    together span the design matrix that ordinary batches cannot, because
    there n and depth correlate at about -0.97.

    Drawn around a single quantile so every batch has nearly the same depth,
    largest first, so that if the biggest size does not fit the smaller rows
    are still collected.
    """
    costs = np.asarray(costs, dtype=np.float64)
    order = np.argsort(costs, kind="stable")
    biggest = max(sizes)
    centre = int(quantile * order.size)
    start = max(0, centre - biggest // 2)
    pool = order[start:start + biggest]
    if pool.size < biggest:
        return []
    out = []
    for size in sorted(sizes, reverse=True):
        if size > pool.size:
            continue
        # EVENLY SPACED over the same pool, endpoints included -- so every row
        # spans the IDENTICAL depth range and only n differs. Contiguous
        # slices do not: a 1024-window slice covered depths 28-88 while its
        # 128-window counterpart covered 45-53, leaving n and depth still
        # co-varying (1.66x) in the sweep meant to separate them.
        idx = np.linspace(0, pool.size - 1, size).round().astype(int)
        out.append(pool[np.unique(idx)])
    return out


def fixed_n_probe_batches(costs: np.ndarray, n_fixed: int,
                           n_levels: int = 5) -> list[np.ndarray]:
    """Batches of IDENTICAL size spanning the depth range.

    THE IDENTIFIABILITY PROBLEM. In cost-budgeted batches the deep ones are
    small by construction, so log(n) and log(depth) correlate at -0.97 on real
    data -- and a fit cannot then separate "memory grows with batch size" from
    "memory grows with depth". Measured: q=1.18/p=0.88 and q=1.00/p=0.76 fit
    the same five points equally well (13.6% vs 13.7%), so the exponent that
    looked like a finding was an artifact of how the batches were built.

    Holding n fixed while depth varies breaks it: whatever the bytes do across
    these rows is depth's doing alone. The cost is that a fixed n must be
    small enough that the DEEPEST level still fits in memory, so these probes
    are not representative batches -- they are an experiment, not a workload.
    """
    costs = np.asarray(costs, dtype=np.float64)
    order = np.argsort(costs, kind="stable")
    if order.size < n_fixed * 2:
        return []
    # Depth levels spread over the population's quantiles, each drawn from a
    # contiguous (hence narrow-span) slice, so span does not vary with depth
    # and confound the sweep in turn.
    out = []
    for q in np.linspace(0.1, 0.98, n_levels):
        end = int(q * order.size)
        start = end - n_fixed
        if start < 0:
            continue
        out.append(order[start:end])
    return out


def calibrate_cost_model(f_theta, dataset, device, n_rollout_steps: int,
                          sizes=(128, 256, 512, 1024),
                          depths=(8, 32, 128, 512)) -> dict:
    """Fit A, B, p by FORCING the depth, so no trained model is needed.

    THE CIRCULARITY THIS REMOVES. The two probe sweeps get their depth spread
    by selecting windows whose sub-step counts differ -- which needs a model
    whose alpha criterion already produces a spread, i.e. a trained one. So
    "measure the coefficients, then train" required training first.

    But the coefficients do not depend on the weights. They describe the
    ARCHITECTURE and the CARD: how many bytes one window costs at one depth,
    for this hidden_dim, latent size, rollout length and truncate_bptt. The
    weights only decide which depths the criterion asks for -- the model's
    INPUT, not its coefficients. Forcing n_substeps to a prescribed grid
    therefore measures the same thing, on any checkpoint, including an
    untrained one.

    The grid is a proper factorial design: every size at every depth, so n and
    depth vary independently by construction rather than by careful sampling.
    That is the same identifiability problem the sweeps solved, solved once
    and properly -- in cost-budgeted batches n and depth correlate at about
    -0.97 and no fit can separate them.
    """
    rows = []
    n_windows = len(dataset)
    for size in sizes:
        if size > n_windows:
            continue
        batch = np.arange(size) % n_windows
        for depth in depths:
            prev_alpha, prev_n = f_theta.alpha, f_theta.n_substeps
            try:
                f_theta.alpha = None
                f_theta.n_substeps = int(depth)
                b = measure_batch_bytes(f_theta, dataset, batch, device,
                                         n_rollout_steps)
            finally:
                f_theta.alpha, f_theta.n_substeps = prev_alpha, prev_n
            if b is None:
                return {"rows": [], "reason": "no CUDA device"}
            rows.append({"n": float(size), "depth": float(depth), "bytes": b})
    if len(rows) < 4:
        return {"rows": rows, "reason": "grid too small to fit"}
    if len({r["depth"] for r in rows}) < 3:
        # TWO depths cannot identify p. A + B*d^p has three unknowns, so with
        # two depth levels EVERY exponent admits an exact (A, B) -- the search
        # then returns whichever it tried first, with a zero residual that
        # looks like a perfect fit. Measured: a 2x2 grid on exact synthetic
        # data returned p=0.36 for a true 0.70, at 0% residual.
        return {"rows": rows,
                "reason": "need at least 3 distinct depths to identify the "
                          "exponent; with 2 every p fits exactly"}
    if len({r["n"] for r in rows}) < 2:
        return {"rows": rows,
                "reason": "need at least 2 batch sizes to separate the "
                          "per-window term from the depth term"}

    nn = np.array([r["n"] for r in rows])
    dd = np.array([r["depth"] for r in rows])
    yy = np.array([r["bytes"] for r in rows])
    best = None
    for p in np.arange(0.2, 1.31, 0.01):
        X = np.vstack([nn, nn * dd ** p]).T
        coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
        w = float((np.abs(X @ coef - yy) / yy).max())
        if best is None or w < best[3]:
            best = (float(p), float(coef[0]), float(coef[1]), w)
    p, A, B, worst = best
    return {"rows": rows, "p": p, "A": A, "B": B, "worst": worst}


def measure_batch_bytes(f_theta, dataset, batch, device,
                         n_rollout_steps: int) -> float | None:
    """Peak ALLOCATED bytes for one training-shaped step on `batch`.

    Mirrors the memory-relevant shape of train_lds's step: rollout over
    n_rollout_steps transitions, squared-error loss, backward. Optimizer
    state is deliberately excluded (it is batch-independent); so is the
    guard/telemetry machinery. None on CPU -- the CUDA counters ARE the
    measurement, and a host-side proxy would be another model of the thing
    this exists to measure rather than the thing itself.
    """
    if device.type != "cuda":
        return None
    rows = [dataset[i] for i in batch]
    # (window, window_deriv, dt_window, theta) -- the encode_both_streams
    # layout train_lds unpacks; mirroring its step exactly:
    #   z0 = window[:, 0], targets = window[:, 1:], z1 = window_deriv[:, 0]
    window = torch.stack([r[0] for r in rows]).to(device)
    window_deriv = torch.stack([r[1] for r in rows]).to(device)
    dt_window = torch.stack([r[2] for r in rows]).to(device)
    theta = torch.stack([r[3] for r in rows]).to(device)
    z0, z0_true = window[:, 0], window[:, 1:]
    z1 = window_deriv[:, 0]

    f_theta.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    z0_hat, z1_hat, f_carry = z0, z1, None
    loss = z0.new_zeros(())
    for step in range(n_rollout_steps):
        z0_hat, z1_hat, f_carry = f_theta._integrate(
            z0_hat, z1_hat, dt_window[:, step], theta, f_carry=f_carry)
        loss = loss + (z0_hat - z0_true[:, step]).square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    peak = float(torch.cuda.max_memory_allocated(device))
    f_theta.zero_grad(set_to_none=True)
    return peak


def check_memory(lds_checkpoint_path: Path, base_path: Path | None = None,
                  size: int | None = None, device: str | None = None,
                  batch_size: int = 2048, batch_cost_budget: float | None = None,
                  n_probe: int = 8, max_windows_per_run: int | None = None,
                  truncate_bptt: int | None = None, fixed_n: int = 256,
                  calibrate: bool = False,
                  latent_cache_dir: Path | str | None = None) -> dict:
    ctx = _load_ae_f_theta_and_dataset(
        lds_checkpoint_path, min_step=None, min_stdev_phi=None,
        min_passing_steps=None, base_path=base_path, size=size,
        ae_stats_weight=None, hidden_dim=256, n_hidden_layers=2,
        condition_on_theta=None, euler_only=None, device=device,
        announce_euler_only=False, latent_cache_dir=latent_cache_dir)
    resolved_device, _, _, _, dataset, _, f_theta = ctx
    f_theta.eval()

    if calibrate:
        # CALIBRATION ONLY: forces the depth grid, so it needs no trained
        # model and no representative batching. Runs before stage 3 exists.
        n_rollout = max(int(getattr(dataset, "window_length", 3)) - 1, 1)
        if truncate_bptt is not None:
            f_theta.truncate_bptt = int(truncate_bptt)
        print(f"calibrating the memory cost model by FORCED depth "
              f"(truncate_bptt={getattr(f_theta, 'truncate_bptt', None)}, "
              f"n_rollout_steps={n_rollout}). No trained weights needed: the "
              f"coefficients describe the architecture and the card, and the "
              f"weights only decide which depths get asked for.")
        fit = calibrate_cost_model(f_theta, dataset, resolved_device, n_rollout)
        if not fit.get("rows"):
            print(f"  could not calibrate: {fit.get('reason', 'unknown')}")
            return fit
        print(f"\n{'n':>7} {'depth':>7} {'MiB':>9}")
        for r in fit["rows"]:
            print(f"{r['n']:7.0f} {r['depth']:7.0f} {r['bytes'] / 2**20:9.1f}")
        if "p" not in fit:
            print(f"  {fit.get('reason', 'fit failed')}")
            return fit
        print(f"\nbytes = {fit['A']:.0f}*n + {fit['B']:.0f}*n*depth^{fit['p']:.2f}"
              f"   (worst residual {fit['worst']:.1%})")
        if fit["A"] <= 0:
            print("  A IS NOT POSITIVE, which is impossible -- do not use this "
                  "fit; it is a fitting artifact. Widen the grid.")
        else:
            print(f"    memory_cost_a_bytes = {fit['A']:.0f}")
            print(f"    memory_cost_b_bytes = {fit['B']:.0f}")
            print(f"    memory_cost_p       = {fit['p']:.2f}")
            print(f"  These describe n_rollout_steps={n_rollout}; measure again "
                  f"for a stage with a different rollout length.")
        return fit

    alpha = getattr(f_theta, "alpha", None)
    _ckpt_tb = getattr(f_theta, "truncate_bptt", None)
    if truncate_bptt is not None:
        # OVERRIDE, and applied to the MODEL too, not just the predictors --
        # otherwise the table would compare truncation-aware predictions
        # against bytes measured without truncation, which is worse than not
        # measuring at all.
        f_theta.truncate_bptt = int(truncate_bptt)
        print(f"check_memory: truncate_bptt overridden to {truncate_bptt} "
              f"(the checkpoint records "
              f"{_ckpt_tb}).")
    truncate_bptt = getattr(f_theta, "truncate_bptt", None)
    max_substeps = getattr(f_theta, "max_substeps", 256)
    n_rollout_steps = max(int(getattr(dataset, "window_length", 3)) - 1, 1)
    if alpha is None:
        raise ValueError(
            "this checkpoint has no alpha: the sampler being diagnosed only "
            "runs under adaptive sub-stepping")

    costs = estimate_window_costs(dataset, f_theta, alpha, max_substeps,
                                   resolved_device)
    sampler = BudgetedBatchSampler(costs, batch_size, budget=batch_cost_budget,
                                    shuffle=False, truncate_bptt=truncate_bptt)
    batches = sampler._batches
    picks = select_batches(batches, costs, n_probe=n_probe)

    print(f"check_memory: {len(batches)} batches; measuring {len(picks)} "
          f"(cost quantiles + the widest span). truncate_bptt={truncate_bptt}, "
          f"alpha={alpha}, n_rollout_steps={n_rollout_steps}.")
    header = (f"{'batch':>6} {'n':>6} {'counts':>13} {'span':>6} {'waste':>6} "
              f"{'raw':>10} {'retained':>10} {'span_aw':>10} {'segxk':>10} "
              f"{'MiB':>9}")
    print(header)

    rows = []
    for bi in picks:
        batch = [int(i) for i in batches[bi]]
        c = costs[batch]
        pred = predictor_table(c, truncate_bptt)
        peak = measure_batch_bytes(f_theta, dataset, batch, resolved_device,
                                    n_rollout_steps)
        rows.append({"batch": bi, "predictors": pred, "peak_bytes": peak,
                      "waste": masked_loop_waste(c)})
        mib = "n/a" if peak is None else f"{peak / 2**20:9.1f}"
        print(f"{bi:>6} {pred['n_windows']:>6} "
              f"{pred['count_lo']:6.0f}-{pred['count_hi']:<6.0f} "
              f"{pred['span']:>6.0f} {rows[-1]['waste']:>6.1%} "
              f"{pred['raw']:>10.3g} {pred['retained']:>10.3g} "
              f"{pred['span_aware']:>10.3g} {pred['segments_x_k']:>10.3g} "
              f"{mib:>9}")

    measured = np.array([r["peak_bytes"] or np.nan for r in rows])
    fits = {}
    if np.isfinite(measured).sum() >= 2:
        print("\nWhich predictor tracks the measured bytes "
              "(fit through origin; a model is only as good as its WORST batch):")
        for key in ("raw", "retained", "span_aware", "segments_x_k"):
            x = np.array([r["predictors"][key] for r in rows])
            a, worst = _fit_and_residuals(x, measured)
            _b, _c, _w_int = _fit_with_intercept(x, measured)
            fits[key] = {"bytes_per_unit": a, "worst_rel_residual": worst,
                          "worst_with_intercept": _w_int,
                          "intercept_bytes": _c}
            print(f"  {key:>14}: {a:8.0f} bytes/unit, worst residual "
                  f"{worst:6.1%}   (with a fitted constant: {_w_int:6.1%}, "
                  f"constant {_c / 2**20:6.1f} MiB)")
        _n = np.array([r["predictors"]["n_windows"] for r in rows], float)
        _hi = np.array([r["predictors"]["count_hi"] for r in rows], float)
        _p, _a, _worst = fit_power_law(_n, _hi, measured)
        fits["n_x_depth^p"] = {"bytes_per_unit": _a, "worst_rel_residual": _worst,
                                "exponent": _p}
        print(f"  {'n x depth^p':>14}: {_a:8.0f} bytes/unit at p={_p:.2f}, "
              f"worst residual {_worst:6.1%}")
        best = min(fits, key=lambda kk: fits[kk]["worst_rel_residual"])
        print(f"  -> best: {best}. A worst residual above ~30% means NO "
              f"per-batch constant exists for that predictor and sizing "
              f"anything from it will be wrong on some batch.")
        if abs(_p - 1.0) > 0.1 and _worst < fits["raw"]["worst_rel_residual"]:
            print(f"     NOTE p={_p:.2f}, not 1: memory is SUBLINEAR in depth, so "
                  f"every predictor above (all of which assume p=1) over-charges "
                  f"deep batches and under-charges shallow ones -- the direction "
                  f"of every budget discrepancy seen so far. Two parameters on "
                  f"{len(rows)} rows, so read p as a range, not a value.")
    else:
        print("\n(no CUDA device: memory columns are n/a -- run this on the "
              "training machine for the measurement half)")

    waste_all = np.array([masked_loop_waste(costs[list(b)]) for b in batches])
    print(f"\nmasked-loop waste across ALL {len(batches)} batches: "
          f"median {np.median(waste_all):.1%}, worst {waste_all.max():.1%} "
          f"-- the f evaluations already spent on arrived samples. This is "
          f"both today's cost of heterogeneity and exactly what rounding "
          f"counts up to the batch max would convert into (unrequested) "
          f"refinement.")
    sweep_rows = []
    if resolved_device.type == "cuda" and fixed_n:
        probes = fixed_n_probe_batches(costs, fixed_n)
        if probes:
            print(f"\nFIXED-n DEPTH SWEEP (n={fixed_n} for every row, so anything the "
                  f"bytes do here is DEPTH's doing).")
            print(f"{'depth':>8} {'span':>6} {'MiB':>9} {'MiB/depth':>11}")
            for b in probes:
                c = costs[list(b)]
                mb = measure_batch_bytes(f_theta, dataset, b, resolved_device,
                                          n_rollout_steps)
                if mb is None:
                    continue
                depth = float(c.max())
                sweep_rows.append({"depth": depth, "n": float(len(b)),
                                    "span": float(c.max() - c.min()),
                                    "bytes": mb})
                print(f"{depth:8.0f} {c.max()-c.min():6.0f} {mb/2**20:9.1f} "
                      f"{mb/2**20/depth:11.3f}")
            if len(sweep_rows) >= 3:
                d = np.array([r["depth"] for r in sweep_rows])
                y = np.array([r["bytes"] for r in sweep_rows])
                best_p, best_w = None, float("inf")
                for p in np.arange(0.0, 1.51, 0.01):
                    _, _, w = _fit_with_intercept(d ** p, y)
                    if np.isfinite(w) and w < best_w:
                        best_p, best_w = float(p), w
                _lin_b, _lin_c, _lin_w = _fit_with_intercept(d, y)
                print(f"  depth exponent, n held fixed, WITH a fitted constant: "
                      f"p={best_p:.2f} (worst residual {best_w:.1%}); "
                      f"forcing p=1 gives {_lin_w:.1%} on a {_lin_c / 2**20:.0f} MiB "
                      f"constant.")
                print("  THIS exponent is identifiable -- n is fixed here, whereas "
                      "in the batch table n and depth correlate at about -0.97.")
                if best_p < 0.85 and best_w < _lin_w / 1.5:
                    print("     => memory is genuinely SUBLINEAR in depth, and not "
                          "because of fixed overhead (that is the constant, fitted "
                          "separately). No proportional cost model exists, so any "
                          "budget in sample-substeps will misprice some batch; "
                          "measure-and-correct is the design, not a stopgap.")

    width_rows = []
    if resolved_device.type == "cuda" and fixed_n:
        print("\nFIXED-DEPTH WIDTH SWEEP (depth held near the median, n varied, so "
              "anything the bytes do here is BATCH SIZE's doing).")
        print(f"{'n':>8} {'depth':>7} {'MiB':>9} {'MiB/n':>9}")
        for b in fixed_depth_probe_batches(costs):
            c = costs[list(b)]
            mb = measure_batch_bytes(f_theta, dataset, b, resolved_device,
                                      n_rollout_steps)
            if mb is None:
                continue
            width_rows.append({"n": float(len(b)), "depth": float(c.max()),
                                "bytes": mb})
            print(f"{len(b):8d} {c.max():7.0f} {mb/2**20:9.1f} "
                  f"{mb/2**20/len(b):9.4f}")

    if len(sweep_rows) >= 3 and len(width_rows) >= 3:
        # JOINT FIT over both sweeps: A*n + B*n*depth^p. Identifiable only
        # because the two sweeps between them vary n and depth independently.
        # On the batch table alone the same fit returned a NEGATIVE per-window
        # cost and predicted negative memory at small depth.
        nn = np.array([r.get("n", fixed_n) for r in sweep_rows]
                       + [r["n"] for r in width_rows], float)
        dd = np.array([r["depth"] for r in sweep_rows]
                       + [r["depth"] for r in width_rows], float)
        yy = np.array([r["bytes"] for r in sweep_rows]
                       + [r["bytes"] for r in width_rows], float)
        best = None
        for p in np.arange(0.2, 1.31, 0.01):
            X = np.vstack([nn, nn * dd ** p]).T
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
            w = float((np.abs(X @ coef - yy) / yy).max())
            if best is None or w < best[3]:
                best = (float(p), float(coef[0]), float(coef[1]), w)
        p, A, B, w = best
        print(f"\nJOINT FIT over both sweeps ({len(yy)} points): "
              f"bytes = {A:.0f}*n + {B:.0f}*n*depth^{p:.2f}, worst residual {w:.1%}")
        if A < 0:
            print("  A IS NEGATIVE, which is impossible -- the per-window cost "
                  "cannot be below zero. The two sweeps still do not pin the "
                  "model, so treat every constant above as a fitting artifact.")
        else:
            print(f"  i.e. {A/1024:.1f} KiB per window regardless of depth, plus "
                  f"{B:.0f} bytes per window per depth^{p:.2f}.")
            print("\n  To use this fit, put these in the stage's params section "
                  "(they default to a fit measured on an RTX 2060 Super at "
                  "hidden_dim=256, latent 8x8x8, n_rollout_steps=2 -- a "
                  "different card, latent size or rollout length moves them):")
            print(f"    memory_cost_a_bytes = {A:.0f}")
            print(f"    memory_cost_b_bytes = {B:.0f}")
            print(f"    memory_cost_p       = {p:.2f}")
            print(f"  Measure them PER STAGE: this run's numbers describe "
                  f"n_rollout_steps={n_rollout_steps}, and a stage with a "
                  f"different rollout length pays a different multiple.")

    return {"rows": rows, "fits": fits, "waste": waste_all, "sweep": sweep_rows,
             "width": width_rows, "n_batches": len(batches)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--base-path", type=Path, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--truncate-bptt", type=int, default=None,
                         help="override the checkpoint's truncate_bptt. Needed for "
                              "checkpoints saved before the field was recorded: "
                              "without it the tool reports None, every predictor "
                              "collapses to the same number by construction, and "
                              "the comparison discriminates nothing")
    parser.add_argument("--calibrate", action="store_true",
                         help="fit the memory cost model by FORCING a depth grid, "
                              "and print the three params lines. Needs no trained "
                              "model -- run it on the stage-2 checkpoint before "
                              "stage 3 exists. The coefficients describe the "
                              "architecture and the card; the weights only decide "
                              "which depths get asked for")
    parser.add_argument("--fixed-n", type=int, default=256,
                         help="batch size for the fixed-n depth sweep, which is the "
                              "only part of this tool that can identify a depth "
                              "exponent -- in ordinary batches n and depth correlate "
                              "at about -0.97 and the two are inseparable. 0 skips it")
    parser.add_argument("--batch-cost-budget", type=float, default=None)
    parser.add_argument("--n-probe", type=int, default=8)
    args = parser.parse_args()
    check_memory(args.checkpoint, base_path=args.base_path, size=args.size,
                  device=args.device, batch_size=args.batch_size,
                  batch_cost_budget=args.batch_cost_budget,
                  truncate_bptt=args.truncate_bptt, fixed_n=args.fixed_n,
                  calibrate=args.calibrate,
                  n_probe=args.n_probe)


if __name__ == "__main__":
    main()
