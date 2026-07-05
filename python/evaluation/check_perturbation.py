"""
Stage 3 latent validation: perturbation in LATENT space.

Real-space perturbation (x_eps = x + eps*eta, checking stats(z_eps)) was
tried first and abandoned: stats_head predicts statistics for latents of
REAL simulation snapshots, but x_eps is synthetic, off-manifold noise --
stats_head(E(x_eps)) reflects whatever the LEARNED MODEL outputs for an
input it was never trained on, not the TRUE physical statistics of that
synthetic state (which would require running the actual C++ statistics
computation, unavailable in Python). That conflates "is stats_head smooth"
with "is the representation smooth", and there's no ground truth available
to settle it either way.

Perturbation in LATENT space avoids this entirely: z_eps = z + eps*eta
stays close to the real data manifold for small eps (unlike a noisy
pixel blend), and decoding gives x_eps = D(z_eps), a real image directly
comparable in PIXEL SPACE -- no model standing in for missing ground
truth, and no stats_head needed at all (this works even for an AE
trained without a stats loss).

For each real test frame x, z = E(x), baseline = D(z) (NOT the raw x --
comparing against D(z) isolates decoder SMOOTHNESS specifically, without
conflating it with the AE's own reconstruction error at eps=0, which
would confound the intercept check below):
    delta(eps) = || D(z + eps*eta) - D(z) ||   (real-space RMSE, averaged
                                                  over several eta draws)

A linear regression of delta against eps, PER SAMPLE, gives both
diagnostics from one fit:
  intercept ~ 0   -> no discontinuity right at eps=0
  R^2 ~ 1         -> response is genuinely linear (proportional to eps),
                     not curved/saturating

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
from utils import load_datasets as load
from utils.naming import ae_checkpoint_name


def linear_fit(eps_values: np.ndarray, delta: np.ndarray) -> tuple[float, float, float]:
    """delta = dz*eps + c via least squares. Returns (dz, c, r_squared)."""
    dz, c = np.polyfit(eps_values, delta, 1)
    pred = dz * eps_values + c
    ss_res = np.sum((delta - pred) ** 2)
    ss_tot = np.sum((delta - delta.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dz, c, r_squared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("../../config.txt"))
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=8)
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
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
            help="default: ../../output/perturbation_check_png/<checkpoint name>.png")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.size is None or args.stats_weight is None or args.min_step is None:
        config = load.read_config(args.config)
        if args.size is None:
            args.size = config.nx
        if args.stats_weight is None:
            args.stats_weight = config.stats_weight
        if args.min_step is None:
            args.min_step = config.min_step

    if args.checkpoint is None:
        if args.latent_channels is None:
            raise ValueError(
                "Provide either --checkpoint directly, or --latent-channels (--size and "
                "--stats-weight now default to config.txt's values if not given)."
            )
        name = ae_checkpoint_name(args.size, args.latent_channels, args.stats_weight)
        args.checkpoint = Path(f"../../output/ae_checkpoint_pt/{name}.pt")
        print(f"Reconstructed checkpoint path: {args.checkpoint}")

    if args.output is None:
        args.output = Path(f"../../output/perturbation_check_png/{args.checkpoint.stem}.png")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, config={model_cfg}")

    ae = Autoencoder(
        size=model_cfg["size"], channels=1,
        base_channels=model_cfg["base_channels"], latent_channels=model_cfg["latent_channels"],
    ).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()

    test_dirs = checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(f"{args.checkpoint} has no saved test_dirs")
    test_dirs = [Path(d) for d in test_dirs]

    nx = ny = model_cfg["size"]
    frames = []
    for run_dir in test_dirs:
        metadata = load.read_metadata(run_dir / "metadata.txt")
        check = load.check_snapshots_saved(run_dir, metadata)
        bad_steps = set(check["missing"]) | set(check["bad_size"])
        for step in metadata.save_steps:
            if step not in bad_steps and step >= args.min_step:
                frames.append((run_dir, step))
    if not frames:
        raise ValueError("No usable test frames found")

    generator = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(frames), generator=generator)[:args.n_samples].tolist()
    chosen = [frames[i] for i in idx]
    print(f"Perturbing {len(chosen)} real test-set frames in LATENT space, "
          f"{args.n_repeats} eta draws each")

    eps_values = np.array(args.eps_values)
    all_deltas = []  # (n_samples, n_eps), real-space RMSE

    with torch.no_grad():
        for run_dir, step in chosen:
            x_np = load.read_phi_half(run_dir / load.snapshot_filename(step), nx, ny)
            x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
            z = ae.encoder(x)
            baseline = ae.decoder(z)  # D(z), NOT raw x -- isolates decoder smoothness
                                       # specifically, without AE reconstruction error
                                       # at eps=0 confounding the intercept check

            deltas = []
            for eps in eps_values:
                delta_sum = 0.0
                for _ in range(args.n_repeats):
                    eta = torch.randn_like(z)
                    z_eps = z + eps * eta
                    x_eps = ae.decoder(z_eps)
                    delta_sum += (x_eps - baseline).pow(2).mean().sqrt().item()
                deltas.append(delta_sum / args.n_repeats)
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
    eps1_target, eps2_target = args.pairwise_eps
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
    ax.set_ylabel("||D(z+eps*eta) - D(z)|| (real space)")
    ax.set_title(f"Perturbation response across {len(chosen)} real frames")
    ax.legend()

    axes[1].hist(per_sample_c, bins=min(20, max(6, len(per_sample_c) // 3)))
    axes[1].axvline(0.0, color="red", linestyle="--", label="0 = no discontinuity")
    axes[1].set_xlabel("per-sample fitted intercept c")
    axes[1].set_ylabel("count")
    axes[1].set_title("Distribution of per-sample discontinuities")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()