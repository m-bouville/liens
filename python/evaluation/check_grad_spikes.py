"""
WHICH windows produce catastrophic gradient norms, and what marks them?

The spike guard reports "skipped 3 grad (77x) of 10 @ dt_max=1000" every
epoch, indefinitely -- roughly 30% of each epoch's gradient discarded, with
ORDINARY losses. dt_max is the only feature the guard can name, because it is
the only one it happens to carry. That is not evidence that dt is the cause:
it is evidence that dt is what was printed.

This measures the rest. For each sampled window individually it records the
gradient norm and a set of candidate features, then reports which features
actually separate the outliers from the bulk.

THE DECOMPOSITION THAT MATTERS. The loss is a sum over rollout steps, and the
excursions have consistently shown a normal one-step column beside a diverging
two-step one. So the gradient is measured THREE ways per window: from the
first transition alone, from the second alone, and from the full rollout. If
the outliers live entirely in the second, the problem is the chained path --
propagating from a state the model produced -- and no amount of dt filtering
addresses it.

Features recorded per window:
    dt of each transition, |z0|, |z1|, |f_theta|, the criterion's own ratio
    |f|/(alpha*|z1|), the realised sub-step count, whether it clamped, theta,
    and the per-step losses.

Usage (from python/, so imports resolve):

    python -m evaluation.check_grad_spikes \\
        checkpoints/stage3b/128x128-stage3b.pt --size 128 --n-windows 256

One backward per window, so cost scales with --n-windows; 256 is usually
enough to characterise a tail that is 30% of every epoch.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from evaluation._latent_eval import _load_ae_f_theta_and_dataset


def _grad_norm(f_theta) -> float:
    total = 0.0
    for p in f_theta.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().square().sum())
    return total ** 0.5


def measure_window(f_theta, row, device, n_rollout_steps: int) -> dict:
    """Gradient norms and features for ONE window.

    Three gradients: first transition alone, second alone, and the full
    rollout. A window whose full-rollout gradient is an outlier while its
    first-transition gradient is ordinary tells a different story from one
    that is extreme in both.
    """
    clamped_before = getattr(f_theta, "n_substeps_clamped", 0)
    window = row[0].unsqueeze(0).to(device)
    window_deriv = row[1].unsqueeze(0).to(device)
    dt_window = row[2].unsqueeze(0).to(device)
    theta = row[3].unsqueeze(0).to(device)
    z0, z0_true = window[:, 0], window[:, 1:]
    z1 = window_deriv[:, 0]

    out = {
        "dt0": float(dt_window[0, 0]),
        "dt_max": float(dt_window[0, :n_rollout_steps].max()),
        "z0_norm": float(z0.square().mean().sqrt()),
        "z1_norm": float(z1.square().mean().sqrt()),
        "theta0": float(theta[0, 0]),
    }

    with torch.no_grad():
        f0 = f_theta.f(z0, z1, theta)
        out["f_norm"] = float(f0.square().mean().sqrt())
        # The criterion's own quantity: what it divides to pick a count.
        out["f_over_z1"] = out["f_norm"] / max(out["z1_norm"], 1e-30)

    # per-step gradients, then the full rollout
    losses = []
    for only in ("step0", "step1", "full"):
        f_theta.zero_grad(set_to_none=True)
        z0_hat, z1_hat, f_carry = z0, z1, None
        loss = z0.new_zeros(())
        for step in range(n_rollout_steps):
            z0_hat, z1_hat, f_carry = f_theta._integrate(
                z0_hat, z1_hat, dt_window[:, step], theta, f_carry=f_carry)
            term = (z0_hat - z0_true[:, step]).square().mean()
            if only == "full" or only == f"step{step}":
                loss = loss + term
            if only == "step0" and step == 0:
                break
        if not loss.requires_grad:
            out[f"grad_{only}"] = 0.0
            continue
        loss.backward()
        out[f"grad_{only}"] = _grad_norm(f_theta)
        out[f"loss_{only}"] = float(loss.detach())
        losses.append(float(loss.detach()))
    f_theta.zero_grad(set_to_none=True)

    stats = f_theta.substep_stats()
    if stats:
        out["n_substeps"] = float(stats["max"])
        # PER-WINDOW, by differencing. n_substeps_clamped is deliberately
        # never reset (a clamp that bound once in a run is worth carrying), so
        # reading it directly would give every window the run total and make
        # the last window look like the worst offender. I misread exactly this
        # counter as a per-epoch rate in a training log.
        out["clamped"] = float(stats["clamped"]) - float(clamped_before)
    return out


def separation(rows: list[dict], key: str, outlier_mask: np.ndarray) -> tuple:
    """How well `key` separates outliers from the bulk.

    Returns (ratio of medians, overlap, AUC).

    THE OVERLAP ALONE LIES IN TWO WAYS, both seen on real data:

      * A CONSTANT feature scores a perfect 0 overlap. `clamped` was
        identically zero for all 256 windows, so 0/0 gave ratio 0.00 and no
        bulk value exceeded the (zero) outlier median -- and it was ranked the
        best separator of all.

      * A feature whose outliers sit at the TOP OF ITS RANGE also scores 0,
        by construction. dt cannot exceed max_dt, so once the outlier median
        is max_dt, "no bulk window exceeds it" is arithmetic, not evidence.
        Ties are exactly what overlap cannot see.

    So the AUC is the honest number: the probability that a random outlier
    ranks above a random bulk window, counting ties as half. 0.5 is no
    separation, 1.0 is perfect, and a constant feature scores exactly 0.5
    however its medians happen to divide.
    """
    vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    good = np.isfinite(vals)
    out_v, bulk_v = vals[outlier_mask & good], vals[(~outlier_mask) & good]
    if out_v.size == 0 or bulk_v.size == 0:
        return float("nan"), float("nan"), float("nan")
    ratio = float(np.median(out_v) / max(abs(np.median(bulk_v)), 1e-30))
    overlap = float((bulk_v > np.median(out_v)).mean())
    greater = (out_v[:, None] > bulk_v[None, :]).sum()
    ties = (out_v[:, None] == bulk_v[None, :]).sum()
    auc = float((greater + 0.5 * ties) / (out_v.size * bulk_v.size))
    return ratio, overlap, auc


FEATURE_KEYS = ("dt0", "dt_max", "z0_norm", "z1_norm", "f_norm", "f_over_z1",
                 "n_substeps", "clamped", "theta0")


def rank_features(rows: list[dict], outlier_mask: np.ndarray,
                   keys=FEATURE_KEYS) -> tuple[list, list]:
    """Score every feature; return (table, ranked) with constants excluded.

    Ranked by |AUC - 0.5|, NOT by overlap. Ranking on overlap picked
    `clamped` -- identically zero across all 256 measured windows -- as the
    best separator of all, because a constant scores a perfect 0 overlap by
    vacuity. Constants are dropped from the ranking entirely and marked in the
    table, so they stay visible without being able to win.
    """
    table, ranked = [], []
    for key in keys:
        ratio, overlap, auc = separation(rows, key, outlier_mask)
        if not np.isfinite(auc):
            continue
        vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        flat = bool(np.nanstd(vals) == 0)
        table.append({"key": key, "ratio": ratio, "overlap": overlap,
                       "auc": auc, "constant": flat})
        if not flat:
            ranked.append((-abs(auc - 0.5), key, ratio, auc))
    ranked.sort()
    return table, ranked


def check_grad_spikes(lds_checkpoint_path: Path, base_path: Path | None = None,
                       size: int | None = None, device: str | None = None,
                       n_windows: int = 256, spike_factor: float = 10.0,
                       latent_cache_dir: Path | str | None = None) -> dict:
    ctx = _load_ae_f_theta_and_dataset(
        lds_checkpoint_path, min_step=None, min_stdev_phi=None,
        min_passing_steps=None, base_path=base_path, size=size,
        ae_stats_weight=None, hidden_dim=256, n_hidden_layers=2,
        condition_on_theta=None, euler_only=None, device=device,
        announce_euler_only=False, latent_cache_dir=latent_cache_dir)
    resolved_device, _, _, _, dataset, _, f_theta = ctx
    f_theta.train()

    n_rollout = max(int(getattr(dataset, "window_length", 3)) - 1, 1)
    take = min(int(n_windows), len(dataset))
    idx = np.linspace(0, len(dataset) - 1, take).round().astype(int)

    print(f"check_grad_spikes: {take} windows, n_rollout_steps={n_rollout}, "
          f"alpha={f_theta.alpha}, max_substeps={f_theta.max_substeps}. "
          f"Outlier = gradient norm above {spike_factor:g}x the median, which "
          f"is the guard's own rule (spike_skip_factor).")

    rows = [measure_window(f_theta, dataset[int(i)], resolved_device, n_rollout)
            for i in idx]

    g_full = np.array([r["grad_full"] for r in rows])
    finite = np.isfinite(g_full)
    med = float(np.median(g_full[finite])) if finite.any() else float("nan")
    outliers = finite & (g_full > spike_factor * med)
    print(f"\n{int(outliers.sum())}/{take} windows exceed {spike_factor:g}x the "
          f"median gradient norm ({med:.4g}).")
    if not outliers.any():
        print("  No outliers in this sample -- the tail is rarer than "
              "1/{take}, so raise --n-windows to catch it.")
        return {"rows": rows, "outliers": outliers}

    print("\nWHERE IN THE ROLLOUT (median gradient norm):")
    for key in ("grad_step0", "grad_step1", "grad_full"):
        o = np.array([r.get(key, np.nan) for r in rows])[outliers]
        b = np.array([r.get(key, np.nan) for r in rows])[~outliers & finite]
        print(f"  {key:>11}: outliers {np.nanmedian(o):12.4g}   "
              f"bulk {np.nanmedian(b):12.4g}   "
              f"ratio {np.nanmedian(o) / max(np.nanmedian(b), 1e-30):8.1f}x")

    print("\nWHICH FEATURE SEPARATES THEM. Ranked by AUC -- the probability a")
    print("random outlier ranks above a random bulk window, ties counted half.")
    print("0.5 is no separation. The overlap column is shown too, but it reads")
    print("0 for a CONSTANT feature and for any feature whose outliers sit at")
    print("the top of its range (dt cannot exceed max_dt), so it is not ranked on.")
    print(f"  {'feature':>14} {'ratio':>10} {'overlap':>9} {'AUC':>7}")
    table, scored = rank_features(rows, outliers)
    for r in table:
        note = "  (constant -- cannot separate)" if r["constant"] else ""
        print(f"  {r['key']:>14} {r['ratio']:10.2f} {r['overlap']:9.2f} "
              f"{r['auc']:7.2f}{note}")
    if scored:
        _, key, ratio, auc = scored[0]
        if abs(auc - 0.5) > 0.3:
            direction = "HIGH" if auc > 0.5 else "LOW"
            print(f"\n  => {key} separates best (AUC {auc:.2f}): the outliers are "
                  f"the {direction} end of it, {ratio:.2g}x the bulk median.")
        else:
            print(f"\n  => NO feature separates cleanly (best is {key} at AUC "
                  f"{auc:.2f}). The outliers are not a distinct population in "
                  f"any of these coordinates, so filtering on any one of them "
                  f"would discard mostly-healthy windows.")
    return {"rows": rows, "outliers": outliers, "median_grad": med}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--base-path", type=Path, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-windows", type=int, default=256)
    parser.add_argument("--spike-factor", type=float, default=10.0,
                         help="outlier threshold as a multiple of the median "
                              "gradient norm; matches spike_skip_factor")
    args = parser.parse_args()
    check_grad_spikes(args.checkpoint, base_path=args.base_path, size=args.size,
                       device=args.device, n_windows=args.n_windows,
                       spike_factor=args.spike_factor)


if __name__ == "__main__":
    main()
