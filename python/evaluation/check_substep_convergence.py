"""
Does f_theta behave as a VECTOR FIELD or as a one-shot CORRECTOR?

This is the measurement the whole alpha/sub-stepping design rests on, and
until now it has only been argued.

    A vector field can be integrated. Halving the step size makes the
    endpoint CONVERGE toward a limit, and the error against the truth falls
    (until floating point or the model's own bias floors it).

    A dt-averaged corrector cannot. It was fitted to absorb one whole
    transition's worth of curvature, so evaluating it at intermediate states
    applies that correction several times over: the endpoint DIVERGES, and
    more sub-steps make it monotonically worse.

Stage 3a trained at n_substeps=1 produces the second kind. Stage 3b then
integrates it at 40-100 sub-steps via the alpha criterion, which is why 3a
was re-run with alpha=0.75 -- so that f_theta is fitted at intermediate
states from the start. This script says whether that worked.

TWO measurements, because they answer different questions:

  * SELF-CONVERGENCE  ||z(N) - z(2N)||, no ground truth needed. Falling
    means the integration is approaching a limit, i.e. f_theta is being
    treated as a field. This is the property that makes sub-stepping
    meaningful at all.

  * TRUTH ERROR  ||z(N) - z_true||, against the encoder's own latent at the
    transition's end. Falling means the limit is the RIGHT one. A model can
    self-converge beautifully to a wrong answer.

Both are reported per dt decade, because the corrector/field distinction is
a statement about step size and the decades differ by 100x in it.

Usage (from python/, so imports resolve):

    python -m evaluation.check_substep_convergence \\
        checkpoints/stage3a/128x128-stage3a.pt --size 128 --n-windows 256

Runs on CPU. The sweep is a forward pass under no_grad at a handful of step
counts, so it is cheap enough to run while a training job holds the GPU --
pass --device cpu and a small --n-windows.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from evaluation._latent_eval import _load_ae_f_theta_and_dataset


def _integrate_at(f_theta, z0, z1, dt, theta, n_substeps: int):
    """Endpoint z0 after ONE transition, integrated in exactly n_substeps.

    Forces the fixed-count path: alpha is disabled for the call, so N is
    what was asked for rather than what the criterion wants. Restored after,
    since the model is shared with the caller.
    """
    prev_alpha, prev_n = f_theta.alpha, f_theta.n_substeps
    try:
        f_theta.alpha = None
        f_theta.n_substeps = int(n_substeps)
        with torch.no_grad():
            z0_out, _, _ = f_theta._integrate(z0, z1, dt, theta)
        return z0_out
    finally:
        f_theta.alpha, f_theta.n_substeps = prev_alpha, prev_n


def decade_of(dt: np.ndarray) -> np.ndarray:
    """1, 2, 3... by powers of ten, matching the training weights' banding."""
    return np.floor(np.log10(np.maximum(dt, 1e-12))).astype(int) + 1


def sweep(f_theta, z0, z1, dt, theta, z0_true, counts) -> list[dict]:
    """One row per step count: self-convergence and truth error."""
    endpoints = {n: _integrate_at(f_theta, z0, z1, dt, theta, n) for n in counts}
    rows = []
    for i, n in enumerate(counts):
        e = endpoints[n]
        truth = (e - z0_true).reshape(len(e), -1).square().mean(dim=1).sqrt()
        row = {"n": n, "truth_rms": truth.cpu().numpy()}
        if i + 1 < len(counts):
            nxt = endpoints[counts[i + 1]]
            delta = (e - nxt).reshape(len(e), -1).square().mean(dim=1).sqrt()
            row["self_rms"] = delta.cpu().numpy()
        rows.append(row)
    return rows


def verdict(rows: list[dict]) -> str:
    """What the sweep says about f_theta's nature.

    Deliberately blunt: the whole point is to replace an argument with a
    measurement, so the script states the conclusion rather than leaving a
    table for the reader to squint at.
    """
    truths = [float(np.median(r["truth_rms"])) for r in rows]
    selfs = [float(np.median(r["self_rms"])) for r in rows if "self_rms" in r]
    if not selfs:
        return "not enough step counts to say anything"

    # VACUOUS SWEEP. If the endpoint barely moves with N, every verdict below
    # is noise dressed as a finding -- and that is exactly what an untrained
    # f_theta produces: |f| ~ 0, so the state hardly evolves and all step
    # counts agree to rounding. Caught by feeding this an untrained model,
    # which confidently reported "CORRECTOR" on differences of 1e-6.
    spread = (max(truths) - min(truths)) / max(max(truths), 1e-30)
    if spread < 0.05:
        return (f"INDISTINGUISHABLE: the endpoint moves by {100 * spread:.1f}% "
                f"across the whole sweep, so the step count barely matters here "
                f"and no verdict is warranted. Either f_theta is near-zero (an "
                f"untrained model does this), or dt is far too small for the "
                f"dynamics -- try larger-dt windows.")

    converging = selfs[-1] < selfs[0] / 2
    truth_better = truths[-1] < truths[0]
    worst_is_finest = truths[-1] >= max(truths) - 1e-12

    if converging and truth_better:
        return ("f_theta behaves as a VECTOR FIELD: the endpoint converges as the "
                "step shrinks AND the error against the truth falls. Sub-stepping "
                "is doing what it is meant to, and alpha can be tightened.")
    if converging and not truth_better:
        return ("f_theta SELF-CONVERGES but toward a WORSE answer: refining the "
                "step is stable yet the limit is wrong. That is a fitting "
                "problem, not an integration one -- more sub-steps will not "
                "help, and alpha should not be tightened.")
    if worst_is_finest:
        return ("f_theta behaves as a one-shot CORRECTOR: the finest step is the "
                "WORST. It was fitted to absorb a whole transition's curvature, "
                "so evaluating it at intermediate states applies that correction "
                "repeatedly. Sub-stepping is actively harmful here; f_theta must "
                "be refitted at intermediate states (train the ancestor stage "
                "with alpha set) before any of it is meaningful.")
    return ("MIXED: neither converging nor monotonically worse. Read the per-decade "
            "table -- the decades most likely disagree, and the aggregate hides it.")


def check_substep_convergence(lds_checkpoint_path: Path,
                               base_path: Path | None = None,
                               size: int | None = None,
                               device: str | None = None,
                               n_windows: int = 256,
                               counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
                               max_dt: float | None = None,
                               window_length: int | None = None,
                               latent_cache_dir: Path | str | None = None) -> dict:
    ctx = _load_ae_f_theta_and_dataset(
        lds_checkpoint_path, min_step=None, min_stdev_phi=None,
        min_passing_steps=None, base_path=base_path, size=size,
        ae_stats_weight=None, hidden_dim=256, n_hidden_layers=2,
        condition_on_theta=None, euler_only=None, device=device,
        announce_euler_only=False, max_dt=max_dt,
        window_length_override=window_length,
        latent_cache_dir=latent_cache_dir)
    resolved_device, _, _, _, dataset, _, f_theta = ctx
    f_theta.eval()

    take = min(int(n_windows), len(dataset))
    idx = np.linspace(0, len(dataset) - 1, take).round().astype(int)
    rows = [dataset[int(i)] for i in idx]
    window = torch.stack([r[0] for r in rows]).to(resolved_device)
    window_deriv = torch.stack([r[1] for r in rows]).to(resolved_device)
    dt_window = torch.stack([r[2] for r in rows]).to(resolved_device)
    theta = torch.stack([r[3] for r in rows]).to(resolved_device)

    z0, z1 = window[:, 0], window_deriv[:, 0]
    dt = dt_window[:, 0]
    z0_true = window[:, 1]

    print(f"check_substep_convergence: {take} windows, step counts {list(counts)}, "
          f"alpha={f_theta.alpha}, max_substeps={f_theta.max_substeps}"
          + (f", max_dt OVERRIDDEN to {max_dt:g}." if max_dt else "."))
    if max_dt is None:
        print("  NOTE: max_dt comes from this checkpoint's own config. To compare "
              "two checkpoints, pass --max-dt explicitly -- otherwise each one "
              "filters the runs differently and a given decade is a different "
              "window set in each, so the floors are not comparable.")
    print("  self  = RMS ||z(N) - z(2N)||, no ground truth: is the integration "
          "approaching a limit?")
    print("  truth = RMS ||z(N) - z_true||: is that limit the right one?")

    sweep_rows = sweep(f_theta, z0, z1, dt, theta, z0_true, list(counts))
    dts = dt.cpu().numpy()
    dec = decade_of(dts)

    print(f"\n{'N':>5} {'self (median)':>15} {'truth (median)':>16}")
    for r in sweep_rows:
        s = ("" if "self_rms" not in r
             else f"{np.median(r['self_rms']):15.6g}")
        print(f"{r['n']:5d} {s:>15} {np.median(r['truth_rms']):16.6g}")

    print("\nper dt decade (truth RMS, median):")
    header = "  decade  n_windows " + " ".join(f"{('N=%d' % r['n']):>12}"
                                                for r in sweep_rows)
    print(header)
    for d in sorted(set(dec.tolist())):
        m = dec == d
        cells = " ".join(f"{np.median(r['truth_rms'][m]):12.4g}" for r in sweep_rows)
        print(f"  {d:6d}  {int(m.sum()):9d} {cells}")

    print(f"\n{verdict(sweep_rows)}")
    return {"rows": sweep_rows, "decades": dec, "dt": dts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--base-path", type=Path, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-windows", type=int, default=256)
    parser.add_argument("--counts", type=str, default="1,2,4,8,16,32,64",
                         help="comma-separated step counts, ascending")
    parser.add_argument("--window-length", type=int, default=None,
                         help="override the window length the checkpoint's "
                              "n_rollout_steps implies. ALSO REQUIRED for comparing "
                              "checkpoints across stages: a 3a checkpoint builds "
                              "window_length=2 and a 3b one builds 3, and requiring "
                              "a second valid transition biases the population "
                              "toward earlier, faster-evolving states")
    parser.add_argument("--max-dt", type=float, default=None,
                         help="override the checkpoint's own max_dt. REQUIRED for "
                              "comparing two checkpoints: each one's config filters "
                              "the runs differently, so 'decade 3' is a different "
                              "window set in each and the floors are not comparable")
    args = parser.parse_args()
    check_substep_convergence(
        args.checkpoint, base_path=args.base_path, size=args.size,
        device=args.device, n_windows=args.n_windows, max_dt=args.max_dt,
        window_length=args.window_length, 
        counts=tuple(int(x) for x in args.counts.split(",")))


if __name__ == "__main__":
    main()
