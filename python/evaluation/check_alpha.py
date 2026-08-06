"""Measure alpha, the Taylor-validity ratio that should REPLACE n_substeps.

WHAT ALPHA IS. Every sub-step advances z0 by a linear term and a curvature
correction:

    z0 <- z0 + z1*delta_t + f_theta*delta_t^2/2

alpha is the ratio of the second term to the first:

    alpha = |f_theta|*delta_t / |z1|

i.e. THE FRACTION OF THE DISPLACEMENT THAT THE CURVATURE CORRECTION
CONTRIBUTES. Small alpha means the step is inside the regime where the Taylor
expansion the whole scheme is built on actually holds; alpha near 1 means the
"correction" is as large as the thing it corrects, which is not a correction
at all.

WHY IT REPLACES n_substeps. n_substeps sets delta_t = Delta_t/n_substeps, so
it fixes the STEP and lets alpha fall where it may -- large wherever the
dynamics are fast, small wherever they are slow, and different for every
window. Solving the same equation the other way,

    n_substeps = ceil(|f_theta|*Delta_t / (alpha*|z1|))

fixes the Taylor validity and lets the step follow. They are one equation
read in two directions; the difference is which side is held constant. Holding
alpha constant is what makes a single setting valid across a dt range instead
of needing retuning every time max_dt moves.

WHY THIS SCRIPT EXISTS. alpha is not a number to guess. Two runs have already
bracketed it in delta_t units, at n_substeps=7 (delta_t ~ 71, stable for ~115
epochs and then escalating to a deadlock) and n_substeps=14 (delta_t ~ 36,
2000+ epochs with no spike skips at all). This script converts that bracket
into alpha by measuring |f_theta| and |z1| on real windows, so the controller
is calibrated from evidence rather than from a plausible-looking constant.

READING THE OUTPUT -- AND THE TRAP IN IT. A fixed n_substeps produces a
DISTRIBUTION of alpha (most windows near the median, a few at the tail); a
fixed alpha puts EVERY window at alpha. The two are therefore not comparable
quantile-for-quantile, and the natural-seeming reading is wrong: choosing
alpha at or below the UNSTABLE configuration's tail makes the typical window
as coarse as that configuration's worst one. Measured, at real cost: alpha=0.3
chosen that way (from an unstable p99 of 0.67) deadlocked within 9 epochs,
because its implied median sub-step count sat BELOW the configuration already
known to fail.

Anchor on the STABLE configuration's MEDIAN instead. Then every window gets at
least what the stable run's median window got, and the windows that need more
get strictly more than the stable run ever gave them.

Expect to go lower still, for a reason no formula shows: alpha is computed
from |f_theta|, the curvature the MODEL believes in, not the true one. With
f_theta capturing ~13% of z0_ddot (relative bias -87%, measured), the true
second-order term is several times what this criterion sees. That factor
shrinks as f_theta improves, which is exactly why alpha must be calibrated
against runs whose stability outcome is known rather than derived.

The report also inverts the relation and prices it: for a candidate alpha it
reports the sub-step COUNT distribution (mean, p95, max) and the fixed
n_substeps of equal total cost, so a choice of alpha can be read in the
familiar units before committing to it.

WHAT IT DOES NOT MEASURE. f_theta at the states an ADAPTIVE integrator would
actually visit -- those states do not exist until the controller does. Every
number here is measured at the encoder's own (z0, z1), i.e. on-trajectory,
which is the right reference for calibration but is not a simulation of the
controller.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from evaluation._latent_eval import _load_ae_f_theta_and_dataset
from orchestration.paths import default_latent_cache_dir

_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/

# None is a MEANINGFUL value (caching off), so it cannot double as "not
# specified" -- same convention as check_dt_vs_time/check_parameter_dependence.
_UNSET_CACHE = object()

# The two configurations that bracket stability, as measured on the 128x128
# sweep at max_dt=500. Reported side by side so the alpha they imply can be
# read against the outcome they produced, rather than in the abstract.
_BRACKET_SUBSTEPS = (7, 14)


def collect_alpha(dataset, f_theta, device, max_windows_per_run: int | None = None,
                   batch_size: int = 512) -> dict:
    """Per-frame |f_theta|, |z1| and Delta_t over the dataset's own runs.

    Norms are over the WHOLE latent tensor (C,8,8), not per element: alpha is a
    statement about the step the integrator takes, and the integrator takes one
    step for the whole state. Per-channel ratios would be dominated by whichever
    channel happens to be near zero, which is a property of that channel and not
    of the step.

    Delta_t is the transition to the NEXT kept frame, so the last frame of each
    run is skipped -- it has no transition, and f_theta there would be measured
    against a step that is never taken.
    """
    rows_f, rows_z1, rows_dt, rows_t, rows_theta = [], [], [], [], []
    n_runs = len(dataset._run_data)
    for run_idx in range(n_runs):
        state = dataset._run_data[run_idx]
        deriv = dataset._run_data_deriv[run_idx]
        steps = dataset._run_steps[run_idx]
        scale = dataset._run_dt_scale[run_idx]
        theta = dataset._run_theta[run_idx]
        n = len(steps)
        if n < 2:
            continue
        limit = n - 1 if max_windows_per_run is None else min(n - 1, max_windows_per_run)
        z0_all = state[:limit].to(device)
        z1_all = deriv[:limit].to(device)
        theta_b = theta.to(device).unsqueeze(0).expand(limit, -1)
        f_norms = []
        with torch.no_grad():
            for start in range(0, limit, batch_size):
                stop = min(start + batch_size, limit)
                # f_theta's own field, NOT forward(): forward would fold in the
                # z1*dt term and the dt_cap, and alpha is about the raw
                # curvature field the controller would query.
                f_val = f_theta.f(z0_all[start:stop], z1_all[start:stop],
                                   theta_b[start:stop])
                f_norms.append(torch.linalg.vector_norm(
                    f_val.reshape(f_val.shape[0], -1), dim=1).cpu())
        f_norm = torch.cat(f_norms).numpy()
        z1_norm = torch.linalg.vector_norm(
            z1_all.reshape(limit, -1), dim=1).cpu().numpy()
        dts = np.array([(steps[i + 1] - steps[i]) * scale for i in range(limit)],
                        dtype=float)
        ts = np.array([steps[i] * scale for i in range(limit)], dtype=float)
        rows_f.append(f_norm)
        rows_z1.append(z1_norm)
        rows_dt.append(dts)
        rows_t.append(ts)
        rows_theta.append(np.full(limit, float(theta[0]) if theta.numel() else np.nan))
    if not rows_f:
        raise ValueError("no frames collected -- every run has fewer than 2 kept steps")
    return {"f_norm": np.concatenate(rows_f), "z1_norm": np.concatenate(rows_z1),
            "dt": np.concatenate(rows_dt), "t": np.concatenate(rows_t),
            "theta0": np.concatenate(rows_theta)}


def alpha_at_substeps(data: dict, n_substeps: int) -> np.ndarray:
    """alpha = |f_theta|*delta_t/|z1| with delta_t = Delta_t/n_substeps.

    |z1| == 0 yields inf rather than nan: a state with no velocity and nonzero
    curvature has NO valid step under this criterion, which is a real (if rare)
    statement and must not be silently dropped from a tail quantile.
    """
    delta_t = data["dt"] / n_substeps
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = data["f_norm"] * delta_t / data["z1_norm"]
    # |z1|=0 with |f|>0 ALREADY gives inf from the division itself -- an
    # earlier version "handled" it with an np.where that was pure dead code,
    # and the test guarding it passed vacuously. The case that genuinely needs
    # a decision is 0/0: no velocity AND no curvature, where the division
    # yields nan. Nothing is happening at such a state, so the step is
    # unbounded (alpha=0), NOT undefined -- nan would be dropped from every
    # quantile silently, turning "this state is trivially fine" into "this
    # state was never measured".
    alpha = np.where((data["z1_norm"] == 0) & (data["f_norm"] == 0), 0.0, alpha)
    return alpha


def substeps_for_alpha(data: dict, alpha: float) -> np.ndarray:
    """n_substeps = ceil(|f_theta|*Delta_t/(alpha*|z1|)), at least 1.

    The inverse of alpha_at_substeps, and the number a run would actually pay.
    Reported as a distribution because the MAX is what a batched implementation
    costs when a batch mixes windows -- the loop runs until every sample in the
    batch has arrived.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = data["f_norm"] * data["dt"] / (alpha * data["z1_norm"])
    raw = np.where(np.isfinite(raw), raw, np.inf)
    return np.maximum(np.ceil(raw), 1.0)


def _quantiles(values: np.ndarray, qs=(0.5, 0.9, 0.95, 0.99, 1.0)) -> dict:
    """Quantiles as p50/p90/p95/p99/p100 keys, plus mean and an infinity count.

    The quantile LIST and the keys the report asks for must agree: an earlier
    version omitted 0.95 while the cost table printed p95, so that column was
    silently nan on every row. Unit tests could not catch it -- they call the
    numeric functions, not the report -- so the guard is a test that renders
    the report and asserts no nan reaches it.
    """
    finite = values[np.isfinite(values)]
    out = {"n": int(values.size), "n_infinite": int(values.size - finite.size)}
    if finite.size:
        out["mean"] = float(finite.mean())
        for q in qs:
            out[f"p{int(q * 100)}"] = float(np.quantile(finite, q))
    return out


def check_alpha(lds_checkpoint_path: Path, base_path: Path | None = None,
                 size: int | None = None, min_step: int | None = None,
                 min_stdev_phi: float | None = None, min_passing_steps: int | None = None,
                 max_dt: float | None = None, device: str | None = None,
                 candidate_alphas: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05, 0.02),
                 bracket_substeps: tuple[int, ...] = _BRACKET_SUBSTEPS,
                 max_windows_per_run: int | None = None,
                 latent_cache_dir=_UNSET_CACHE) -> dict:
    """Measure alpha on a checkpoint's own test population and print the report."""
    ctx = _load_ae_f_theta_and_dataset(
        lds_checkpoint_path, min_step, min_stdev_phi, min_passing_steps,
        base_path, size, None, 256, 2, None, None, device,
        max_dt=max_dt,
        latent_cache_dir=(default_latent_cache_dir(_PYTHON_ROOT)
                           if latent_cache_dir is _UNSET_CACHE else latent_cache_dir),
    )
    # 7-tuple; positional for the same reason check_dt_vs_time is: binding
    # f_theta to `dataset` by name would fail far from here.
    resolved_device, _, _, _, dataset, _, f_theta = ctx
    f_theta.eval()

    data = collect_alpha(dataset, f_theta, resolved_device,
                          max_windows_per_run=max_windows_per_run)
    n = data["f_norm"].size
    print(f"\n{n} frames from {len(dataset._run_data)} runs "
          f"(each with its own transition to the next kept frame)\n")

    print("  |f_theta| and |z1| over the whole latent tensor:")
    for name, key in (("|f_theta|", "f_norm"), ("|z1|", "z1_norm"), ("Delta_t", "dt")):
        q = _quantiles(data[key])
        print(f"    {name:>10}: mean {q.get('mean', float('nan')):.4g}   "
              f"median {q.get('p50', float('nan')):.4g}   "
              f"p99 {q.get('p99', float('nan')):.4g}   "
              f"max {q.get('p100', float('nan')):.4g}")

    print("\n  alpha implied by each n_substeps (delta_t = Delta_t/n_substeps):")
    print(f"    {'n_sub':>6} {'delta_t~':>10} {'mean':>10} {'median':>10} "
          f"{'p90':>10} {'p99':>10} {'max':>10}")
    by_substeps = {}
    median_dt = float(np.median(data["dt"]))
    for n_sub in bracket_substeps:
        a = alpha_at_substeps(data, n_sub)
        q = _quantiles(a)
        by_substeps[n_sub] = q
        print(f"    {n_sub:>6} {median_dt / n_sub:>10.4g} {q.get('mean', np.nan):>10.4g} "
              f"{q.get('p50', np.nan):>10.4g} {q.get('p90', np.nan):>10.4g} "
              f"{q.get('p99', np.nan):>10.4g} {q.get('p100', np.nan):>10.4g}")
    print("    (delta_t~ uses the MEDIAN Delta_t; individual windows vary over decades)")

    print("\n  cost of each candidate alpha, in the familiar units:")
    print(f"    {'alpha':>8} {'mean n_sub':>12} {'p95':>8} {'max':>8} "
          f"{'equal-cost fixed n_substeps':>30}")
    by_alpha = {}
    for alpha in candidate_alphas:
        counts = substeps_for_alpha(data, alpha)
        q = _quantiles(counts)
        by_alpha[alpha] = q
        equal_cost = q.get("mean", float("nan"))
        print(f"    {alpha:>8.3g} {q.get('mean', np.nan):>12.2f} "
              f"{q.get('p95', np.nan):>8.0f} {q.get('p100', np.nan):>8.0f} "
              f"{equal_cost:>30.1f}")
    print("    (equal-cost = the fixed n_substeps with the same TOTAL f_theta evaluations;\n"
          "     the gap between it and max is what adaptivity buys -- and what a batched\n"
          "     implementation loses if a batch mixes windows, since the loop runs until\n"
          "     the LAST sample in the batch has arrived)")

    if len(bracket_substeps) >= 2:
        lo_sub, hi_sub = max(bracket_substeps), min(bracket_substeps)
        stable, unstable = by_substeps[lo_sub], by_substeps[hi_sub]
        print(f"\n  -> THE BRACKET. n_substeps={lo_sub} was stable over 2000+ epochs; "
              f"n_substeps={hi_sub} escalated to a deadlock after ~115.")
        print(f"     stable   (n={lo_sub}): median alpha {stable.get('p50', np.nan):.4g}, "
              f"p99 {stable.get('p99', np.nan):.4g}")
        print(f"     unstable (n={hi_sub}): median alpha {unstable.get('p50', np.nan):.4g}, "
              f"p99 {unstable.get('p99', np.nan):.4g}")
        print(f"\n     ANCHOR ON THE STABLE RUN'S MEDIAN ({stable.get('p50', np.nan):.3g}), NOT on "
              f"either run's tail.\n"
              f"     A fixed n_substeps produces a DISTRIBUTION of alpha -- most windows near\n"
              f"     the median, a few at the tail. A fixed alpha puts EVERY window at alpha.\n"
              f"     So choosing alpha at the stable run's p99 makes the typical window as\n"
              f"     coarse as that run's WORST one, which is a different and much less stable\n"
              f"     configuration than the one that was observed to work. Measured: alpha=0.3,\n"
              f"     picked from the unstable run's tail, deadlocked within 9 epochs -- its\n"
              f"     implied median sub-step count was below the failing configuration's.\n"
              f"     At alpha = the stable median, every window gets at least what the stable\n"
              f"     run's median window got, and the hard windows get strictly more.")
        print("\n     AND EXPECT TO GO LOWER STILL. alpha is computed from |f_theta|, i.e. the\n"
              "     curvature the MODEL believes in, not the true one. Where f_theta\n"
              "     underestimates z0_ddot (measured relative bias -87% on the 128 sweep, so\n"
              "     f_theta captures ~13% of it), the true second-order term is several times\n"
              "     what this criterion sees, and the realised Taylor ratio is correspondingly\n"
              "     worse than the alpha requested. That factor shrinks as f_theta improves,\n"
              "     which is why alpha is calibrated against known-stable runs rather than\n"
              "     reasoned to from the formula.")

    return {"data": data, "by_substeps": by_substeps, "by_alpha": by_alpha,
            "median_dt": median_dt}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lds-checkpoint", type=Path, required=True)
    p.add_argument("--base-path", type=Path, default=None)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--min-step", type=int, default=None)
    p.add_argument("--min-stdev-phi", type=float, default=None)
    p.add_argument("--min-passing-steps", type=int, default=None)
    p.add_argument("--max-dt", type=float, default=None,
                    help="defaults to the checkpoint's own data_config value, so the "
                         "measurement matches the population the checkpoint was trained on")
    p.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 0.2, 0.1, 0.05, 0.02])
    p.add_argument("--bracket-substeps", type=int, nargs="+", default=list(_BRACKET_SUBSTEPS),
                    help="the n_substeps values whose stability outcome is known")
    p.add_argument("--max-windows-per-run", type=int, default=None)
    p.add_argument("--no-latent-cache", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    check_alpha(
        args.lds_checkpoint, base_path=args.base_path, size=args.size,
        min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        min_passing_steps=args.min_passing_steps, max_dt=args.max_dt,
        device=args.device, candidate_alphas=tuple(args.alphas),
        bracket_substeps=tuple(args.bracket_substeps),
        max_windows_per_run=args.max_windows_per_run,
        latent_cache_dir=None if args.no_latent_cache else _UNSET_CACHE,
    )


if __name__ == "__main__":
    main()
