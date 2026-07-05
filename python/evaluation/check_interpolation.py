"""
Stage 3 latent validation: interpolation, quantified rather than judged
visually.

Given three real, CONSECUTIVE kept steps (t1, t2, t3) from one run:
  x1 = x(t1), x2_true = x(t2) [ground truth, held out from interpolation],
  x3 = x(t3)
  z1 = E(x1), z3 = E(x3)

alpha is weighted by REAL elapsed time, NOT the midpoint -- since
save_steps are irregular, alpha=0.5 would compare against the wrong
physical instant whenever the two gaps differ:
    alpha = (t2 - t1) / (t3 - t1)   [step-count ratio == time ratio,
                                      since dt = metadata.dt * step_diff
                                      and metadata.dt cancels]

Two interpolations:
    x_alpha = (1-alpha)*x1 + alpha*x3        (real-space, trivial baseline)
    z_alpha = (1-alpha)*z1 + alpha*z3         (latent-space, the thing being tested)

Comparisons, deliberately kept separate by which network they touch:
  ENCODER-ONLY (no decoder involved):
    ||z_alpha - E(x_alpha)||   -- does linearly interpolating LATENTS match
                                  encoding the linearly-interpolated REAL
                                  state? Pure encoder-geometry/linearity check.
    ||z_alpha - E(x2_true)||   -- does the latent interpolation match the
                                  TRUE middle latent? Tests whether the
                                  latent trajectory itself is well
                                  approximated by a straight line between
                                  real encoded points.

  DECODER-INVOLVING (real space, three-way baseline comparison, same
  discipline as check_rollout.py's AE-baseline column):
    ||D(z_alpha) - x2_true||   -- the actual test: does decoding the
                                  latent interpolation recover the real
                                  physical state that occurred in between?
    ||x_alpha - x2_true||      -- trivial real-space pixel interpolation
                                  baseline. If this beats D(z_alpha), the
                                  AE+latent-interpolation pipeline isn't
                                  adding value over doing nothing.
    ||D(E(x2_true)) - x2_true||-- AE reconstruction-only baseline: how
                                  much error is just inherent AE
                                  imperfection, unrelated to interpolation
                                  at all.

  PHYSICAL PLAUSIBILITY (decoder-independent, reuses the already-trained
  stats_head rather than reimplementing statistics.csv's computation):
    stats_head(z_alpha) vs the REAL statistics.csv row at t2 -- a direct,
    quantitative, physically-interpretable plausibility score. Requires
    the checkpoint to have been trained with --stats-weight > 0.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_interpolation \
        --checkpoint ../../output/ae_checkpoint_pt/64x64-4latent-stats_weight_0p01.pt \
        --fixed-triples "../../datasets/64x64/T800_n050_s79:100000:120000:150000" ...
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.autoencoder import Autoencoder
from training.stats_head import StatsHead
from utils import load_datasets as load


def parse_fixed_triple(s: str) -> tuple[Path, int, int, int]:
    parts = s.split(":")
    if len(parts) != 4:
        raise ValueError(f"expected 'run_dir:t1:t2:t3', got '{s}'")
    run_dir, t1, t2, t3 = parts
    return Path(run_dir), int(t1), int(t2), int(t3)


def find_random_triples(test_dirs: list[Path], n: int, min_step: int,
                         generator: torch.Generator) -> list[tuple[Path, int, int, int]]:
    """Every consecutive (t1,t2,t3) triple of kept steps across all test_dirs."""
    candidates = []
    for run_dir in test_dirs:
        metadata = load.read_metadata(run_dir / "metadata.txt")
        check = load.check_snapshots_saved(run_dir, metadata)
        bad_steps = set(check["missing"]) | set(check["bad_size"])
        kept = [s for s in metadata.save_steps if s not in bad_steps and s >= min_step]
        for i in range(len(kept) - 2):
            candidates.append((run_dir, kept[i], kept[i + 1], kept[i + 2]))
    if not candidates:
        return []
    idx = torch.randperm(len(candidates), generator=generator)[:n].tolist()
    return [candidates[i] for i in idx]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-triples", type=str, nargs="+", default=None,
                         help="'run_dir:t1:t2:t3' (repeatable) for reproducible comparison "
                              "across checkpoints, same rationale as check_rollout.py's "
                              "--fixed-windows")
    parser.add_argument("--output", type=Path, default=None,
            help="default: ../../output/interpolation_check_png/<checkpoint name>.png")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"../../output/interpolation_check_png/{args.checkpoint.stem}.png")
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

    stats_head = None
    stats_config = checkpoint.get("stats_config")
    if stats_config is not None:
        stats_head = StatsHead(
            latent_channels=model_cfg["latent_channels"], stat_names=stats_config["stat_names"],
        ).to(device)
        stats_head.load_state_dict(checkpoint["stats_head_state"])
        stats_head.eval()
        print(f"stats_head available -- will report physical-plausibility scores "
              f"for: {stats_config['stat_names']}")
    else:
        print("No stats_head in this checkpoint (trained with --stats-weight 0) -- "
              "skipping physical-plausibility metric")

    if args.fixed_triples:
        triples = [parse_fixed_triple(s) for s in args.fixed_triples]
    else:
        test_dirs = checkpoint.get("test_dirs") or []
        if not test_dirs:
            raise ValueError(f"{args.checkpoint} has no saved test_dirs")
        test_dirs = [Path(d) for d in test_dirs]
        generator = torch.Generator().manual_seed(args.seed)
        triples = find_random_triples(test_dirs, args.n_samples, args.min_step, generator)
        if not triples:
            raise ValueError("No consecutive (t1,t2,t3) triples found in test_dirs")
        print("Selected triples -- reuse via --fixed-triples for reproducible comparison:")
        for run_dir, t1, t2, t3 in triples:
            print(f"  {run_dir}:{t1}:{t2}:{t3}")

    nx = ny = model_cfg["size"]
    n = len(triples)

    # Per-triple scalar results, for the summary table/plot.
    enc_vs_interp_true = np.zeros(n)   # ||z_alpha - E(x2_true)||
    enc_self_consistency = np.zeros(n)  # ||z_alpha - E(x_alpha)||
    dec_vs_true = np.zeros(n)          # ||D(z_alpha) - x2_true||           (the actual test)
    real_interp_vs_true = np.zeros(n)  # ||x_alpha - x2_true||              (trivial baseline)
    ae_recon_vs_true = np.zeros(n)     # ||D(E(x2_true)) - x2_true||        (AE-only baseline)
    stats_error = np.full(n, np.nan) if stats_head is not None else None

    with torch.no_grad():
        for i, (run_dir, t1, t2, t3) in enumerate(triples):
            alpha = (t2 - t1) / (t3 - t1)  # dt-weighted, NOT 0.5, unless gaps happen to be equal

            x1_np = load.read_phi_half(run_dir / load.snapshot_filename(t1), nx, ny)
            x2_np = load.read_phi_half(run_dir / load.snapshot_filename(t2), nx, ny)
            x3_np = load.read_phi_half(run_dir / load.snapshot_filename(t3), nx, ny)

            x1 = torch.from_numpy(x1_np).unsqueeze(0).unsqueeze(0).to(device)
            x2_true = torch.from_numpy(x2_np).unsqueeze(0).unsqueeze(0).to(device)
            x3 = torch.from_numpy(x3_np).unsqueeze(0).unsqueeze(0).to(device)

            z1 = ae.encoder(x1)
            z3 = ae.encoder(x3)
            z2_true = ae.encoder(x2_true)

            x_alpha = (1 - alpha) * x1 + alpha * x3
            z_alpha = (1 - alpha) * z1 + alpha * z3

            z_of_x_alpha = ae.encoder(x_alpha)
            x_from_z_alpha = ae.decoder(z_alpha)
            x_recon_true = ae.decoder(z2_true)

            enc_vs_interp_true[i] = rmse(z_alpha.cpu().numpy(), z2_true.cpu().numpy())
            enc_self_consistency[i] = rmse(z_alpha.cpu().numpy(), z_of_x_alpha.cpu().numpy())
            dec_vs_true[i] = rmse(x_from_z_alpha.cpu().numpy(), x2_np)
            real_interp_vs_true[i] = rmse(x_alpha.cpu().numpy(), x2_np)
            ae_recon_vs_true[i] = rmse(x_recon_true.cpu().numpy(), x2_np)

            if stats_head is not None:
                stats_pred = stats_head(z_alpha).cpu().numpy()[0]
                stats_df = load.read_statistics_csv(run_dir / "statistics.csv")
                true_stats = stats_df.loc[t2, stats_config["stat_names"]].to_numpy(dtype=float)
                mean = stats_config["stats_mean"].numpy()
                std = stats_config["stats_std"].numpy()
                # Normalized error, matching StatsLoss's own normalization,
                # so the number is comparable across statistics of very
                # different raw scale.
                stats_error[i] = rmse((stats_pred - mean) / std, (true_stats - mean) / std)

    print(f"\n{'triple':<45}{'dec_vs_true':>13}{'real_interp':>13}{'ae_baseline':>13}"
          f"{'enc_vs_true':>13}{'enc_self':>10}" + ("  stats_err" if stats_head else ""))
    for i, (run_dir, t1, t2, t3) in enumerate(triples):
        label = f"{run_dir.name}:{t1}:{t2}:{t3}"
        line = (f"{label:<45}{dec_vs_true[i]:13.5f}{real_interp_vs_true[i]:13.5f}"
                f"{ae_recon_vs_true[i]:13.5f}{enc_vs_interp_true[i]:13.5f}"
                f"{enc_self_consistency[i]:10.5f}")
        if stats_head is not None:
            line += f"{stats_error[i]:11.4f}"
        print(line)

    print(f"\nMeans across {n} triples:")
    print(f"  D(z_alpha) vs true middle frame:      {dec_vs_true.mean():.5f}  (the actual test)")
    print(f"  real pixel interpolation vs true:     {real_interp_vs_true.mean():.5f}  "
          f"(trivial baseline -- {'beats' if real_interp_vs_true.mean() < dec_vs_true.mean() else 'loses to'} "
          f"latent interpolation)")
    print(f"  AE reconstruction-only vs true:        {ae_recon_vs_true.mean():.5f}  "
          f"(inherent AE error, unrelated to interpolation)")
    print(f"  z_alpha vs E(true middle):             {enc_vs_interp_true.mean():.5f}  (latent-space)")
    print(f"  z_alpha vs E(x_alpha) self-consistency: {enc_self_consistency.mean():.5f}  (latent-space)")
    if stats_head is not None:
        print(f"  stats_head(z_alpha) normalized error:   {np.nanmean(stats_error):.4f}  "
              f"(physical plausibility, decoder-free)")

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["D(z_alpha)\nvs true", "real interp\nvs true", "AE recon\nvs true"]
    means = [dec_vs_true.mean(), real_interp_vs_true.mean(), ae_recon_vs_true.mean()]
    stds = [dec_vs_true.std(), real_interp_vs_true.std(), ae_recon_vs_true.std()]
    ax.bar(labels, means, yerr=stds, capsize=5, color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_ylabel("RMSE vs true middle frame (real space)")
    ax.set_title(f"Interpolation quality across {n} real (t1,t2,t3) triples")

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
