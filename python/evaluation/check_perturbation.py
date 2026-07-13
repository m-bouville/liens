"""
Stage 2 latent validation: perturbation in LATENT space, measured via
stats_head on BOTH sides -- never touching true (C++-computed) ground
truth statistics, and never using the decoder.

z_eps = z + eps*eta stays close to the real data manifold for small eps
(unlike a real-space pixel blend, which is never physically plausible
even for tiny eps). Comparing stats_head(z_eps) against stats_head(z)
directly isolates latent-space GEOMETRY: any systematic error in
stats_head's own predictions cancels out, since it's applied
identically to both sides -- comparing against ground truth instead
would conflate "is stats_head accurate" with "is the representation
smooth", and discrepancies in real space say nothing about the latent
representation itself.

For each real test frame x, z = E(x):
    stats(z) = stats_head(z)
    delta(eps) = || stats_head(z + eps*eta) - stats_head(z) ||
                 (averaged over several eta draws, per eps)

A linear regression of delta against eps, PER SAMPLE, gives both
diagnostics from one fit:
  intercept ~ 0   -> no discontinuity right at eps=0
  R^2 ~ 1         -> response is genuinely linear (proportional to eps),
                     not curved/saturating

check_perturbation() is importable -- see main.py, which calls it
automatically after stage 2 AND after stage 4/5 (skipped gracefully
there if the ancestor AE has no stats_head at all) with the checkpoint
path it already has in hand. The CLI below is for standalone use.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_perturbation --latent-channels 8
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.autoencoder import Autoencoder
from models.latent_streams import DEFAULT_STREAM_NAME
from training.stats_head import StatsHead
from utils import load_datasets as load
from utils.naming import ae_checkpoint_name

# GENERAL POLICY (matches training/train_refinement.py's own
# _PYTHON_ROOT): every default checkpoint/output path is built from
# THIS anchor, never from a bare relative string like "../../output/...".
# Relative strings resolve against the process's CWD at invocation
# time, which silently differs across bare CLI, `python -m`, and being
# imported and called from another module (e.g. main.py or train_ae.py
# calling this function) -- exactly the recurring "output ended up in
# the wrong place" bug hit repeatedly on this project. Path(__file__)
# is anchored to THIS FILE's own on-disk location instead, which is
# invariant regardless of how/from-where the process was launched.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/check_perturbation.py -> python/


def linear_fit(eps_values: np.ndarray, delta: np.ndarray) -> tuple[float, float, float]:
    """delta = dz*eps + c via least squares. Returns (dz, c, r_squared)."""
    dz, c = np.polyfit(eps_values, delta, 1)
    pred = dz * eps_values + c
    ss_res = np.sum((delta - pred) ** 2)
    ss_tot = np.sum((delta - delta.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dz, c, r_squared


def check_perturbation(
    checkpoint_path: Path, n_samples: int = 16, n_repeats: int = 16,
    eps_values: list[float] | None = None, pairwise_eps: tuple[float, float] = (0.1, 0.3),
    min_step: int = 4000, seed: int = 0,
    output_path: Path | None = None, device: str | None = None,
) -> Path:
    """Saves the perturbation-response plot and returns its path."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    eps_values = np.array(eps_values or [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4])

    if output_path is None:
        output_path = (_PYTHON_ROOT.parent / "output" / "perturbation_check_png"
                       / f"{checkpoint_path.stem}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, config={model_cfg}")

    stats_config = checkpoint.get("stats_config")
    if stats_config is None:
        raise ValueError(f"{checkpoint_path} has no stats_head (trained with --stats-weight 0) "
                          f"-- this check is built entirely around stats_head.")

    ae = Autoencoder(
        size=model_cfg["size"], channels=1,
        base_channels=model_cfg["base_channels"], latent_channels=model_cfg["latent_channels"],
    ).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()

    stats_head = StatsHead(
        latent_channels=model_cfg["latent_channels"], stat_names=stats_config["stat_names"],
    ).to(device)
    stats_head.load_state_dict(checkpoint["stats_head_state"])
    stats_head.eval()
    print(f"stats_head covers: {stats_config['stat_names']}")

    test_dirs = checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(f"{checkpoint_path} has no saved test_dirs")
    test_dirs = [Path(d) for d in test_dirs]

    nx = ny = model_cfg["size"]
    frames = []
    for run_dir in test_dirs:
        metadata = load.read_metadata(run_dir / "metadata.txt")
        check = load.check_snapshots_saved(run_dir, metadata)
        bad_steps = set(check["missing"]) | set(check["bad_size"])
        for step in metadata.save_steps:
            if step not in bad_steps and step >= min_step:
                frames.append((run_dir, step))
    if not frames:
        raise ValueError("No usable test frames found")

    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(frames), generator=generator)[:n_samples].tolist()
    chosen = [frames[i] for i in idx]
    print(f"Perturbing {len(chosen)} real test-set frames in LATENT space, "
          f"{n_repeats} eta draws each")

    all_deltas = []  # (n_samples, n_eps), stats-space distance

    with torch.no_grad():
        for run_dir, step in chosen:
            x_np = load.read_phi_half(run_dir / load.snapshot_filename(step), nx, ny)
            x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
            z = ae.encoder(x)[DEFAULT_STREAM_NAME]
            stats_z = stats_head(z)  # baseline stats(z), NOT ground truth

            deltas = []
            for eps in eps_values:
                delta_sum = 0.0
                for _ in range(n_repeats):
                    eta = torch.randn_like(z)
                    z_eps = z + eps * eta
                    stats_z_eps = stats_head(z_eps)
                    delta_sum += (stats_z_eps - stats_z).norm(dim=1).item()
                deltas.append(delta_sum / n_repeats)
            all_deltas.append(deltas)

    all_deltas = np.array(all_deltas)  # (n_samples, n_eps)
    mean_delta = all_deltas.mean(axis=0)

    # Pooled (aggregate) fit -- one headline number.
    dz, c, r_squared = linear_fit(eps_values, mean_delta)
    print(f"\nPooled fit, delta = dz*eps + c, across {len(chosen)} samples:")
    print(f"  dz = {dz:.5f}   intercept c = {c:.5f}   R^2 = {r_squared:.4f}")
    print(f"  (c far from 0 -> discontinuity; R^2 far from 1 -> curvature/saturation)")

    # Per-sample fits -- catches individual discontinuous/nonlinear samples
    # even when the pooled average looks clean.
    per_sample_c = np.zeros(len(chosen))
    per_sample_r2 = np.zeros(len(chosen))
    for i in range(len(chosen)):
        _, c_i, r2_i = linear_fit(eps_values, all_deltas[i])
        per_sample_c[i] = c_i
        per_sample_r2[i] = r2_i
    print(f"\nPer-sample fits ({len(chosen)} samples):")
    print(f"  intercept: mean={per_sample_c.mean():.5f}  std={per_sample_c.std():.5f}  "
          f"max|c|={np.abs(per_sample_c).max():.5f}")
    print(f"  R^2:       mean={per_sample_r2.mean():.4f}  min={per_sample_r2.min():.4f}")

    # Concrete illustrative pairwise check on the pooled mean, at specific
    # requested eps values (not sweep endpoints).
    eps1_target, eps2_target = pairwise_eps
    idx1 = np.argmin(np.abs(eps_values - eps1_target))
    idx2 = np.argmin(np.abs(eps_values - eps2_target))
    eps1, eps2 = eps_values[idx1], eps_values[idx2]
    d1, d2 = mean_delta[idx1], mean_delta[idx2]
    if d1 > 1e-8:
        print(f"\nPairwise check: delta({eps2})/delta({eps1}) = {d2/d1:.3f} "
              f"vs expected eps2/eps1 = {eps2/eps1:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for row in all_deltas:
        ax.plot(eps_values, row, color="tab:blue", alpha=0.15)
    ax.plot(eps_values, mean_delta, color="tab:blue", linewidth=2, marker="o", label="mean")
    ax.plot(eps_values, dz * eps_values + c, "--", color="gray",
            label=f"pooled fit (c={c:.4f}, R2={r_squared:.3f})")
    ax.set_xlabel("eps (latent space)")
    ax.set_ylabel("||stats_head(z+eps*eta) - stats_head(z)||")
    ax.set_title(f"Perturbation response across {len(chosen)} real frames")
    ax.legend()

    axes[1].hist(per_sample_c, bins=min(20, max(6, len(per_sample_c) // 3)))
    axes[1].axvline(0.0, color="red", linestyle="--", label="0 = no discontinuity")
    axes[1].set_xlabel("per-sample fitted intercept c")
    axes[1].set_ylabel("count")
    axes[1].set_title("Distribution of per-sample discontinuities")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"\nSaved plot to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True,
                         help="grid size (square only) -- config.txt is never read")
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--stats-weight", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=16,
                         help="real test-set frames to perturb and average over")
    parser.add_argument("--n-repeats", type=int, default=16,
                         help="random eta draws per (sample, eps) -- averaged, to separate "
                              "genuine eps-scaling from single-draw direction noise")
    parser.add_argument("--eps-values", type=float, nargs="+",
                         default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4],
                         help="sweep of LATENT perturbation magnitudes")
    parser.add_argument("--pairwise-eps", type=float, nargs=2, default=[0.1, 0.3],
                         metavar=("EPS1", "EPS2"),
                         help="direct illustrative check: (delta(eps2))/(delta(eps1)) "
                              "=? eps2/eps1, on the pooled mean -- the regression's R^2 "
                              "already tests this more generally, this is just a concrete "
                              "sanity-check number")
    parser.add_argument("--min-step", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
            help="default: <repo root>/output/perturbation_check_png/<checkpoint name>.png")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint is None:
        if args.latent_channels is None or args.stats_weight is None:
            raise ValueError(
                "Provide either --checkpoint directly, or --latent-channels and "
                "--stats-weight so the expected path can be reconstructed."
            )
        name = ae_checkpoint_name(args.size, args.latent_channels, args.stats_weight)
        args.checkpoint = _PYTHON_ROOT / "checkpoints" / "stage2" / f"{name}.pt"
        print(f"Reconstructed checkpoint path: {args.checkpoint}")

    if args.output is None:
        args.output = (_PYTHON_ROOT.parent / "output" / "perturbation_check_png"
                       / f"{args.checkpoint.stem}.png")

    check_perturbation(
        checkpoint_path=args.checkpoint, n_samples=args.n_samples, n_repeats=args.n_repeats,
        eps_values=args.eps_values, pairwise_eps=tuple(args.pairwise_eps),
        min_step=args.min_step, seed=args.seed, output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()
