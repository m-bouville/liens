"""
Stage 3 latent validation: interpolation, using ||z|| := mean(stats_head(z))
throughout -- comparing stats_head's output to ITSELF (via a different z),
never mixing it with raw statistics.csv values.

This sidesteps a real bug an earlier version of this script had: stats_head
is trained (via StatsLoss) to output ALREADY-NORMALIZED values -- i.e.
stats_head(z) directly predicts (true_raw - mean) / std, not a raw
physical value. Normalizing stats_head's output a SECOND time (treating
it as if it were still raw) produces a large, spurious, roughly-constant
error dominated by the raw scale of `mean`, unrelated to actual prediction
quality. Comparing stats_head(a) to stats_head(b) directly -- both already
in the same, consistent normalized units -- has no such risk.

Given three real, CONSECUTIVE kept steps (t1 < t2 < t3) from one run:
  x1 = x(t1), x2 = x(t2), x3 = x(t3)
  z1 = E(x1), z2 = E(x2), z3 = E(x3)

alpha is weighted by REAL elapsed time, not the midpoint -- since
save_steps are irregular, alpha=0.5 would compare against the wrong
physical instant whenever the two gaps differ:
    alpha = (t2 - t1) / (t3 - t1)
    z_tilde = (1-alpha)*z1 + alpha*z3

Metric: (||z_tilde|| - ||z2||) / ||z2||, expected close to 0 for a
sensible representation -- a large deviation means interpolation doesn't
track the real trajectory, independent of overall AE reconstruction
accuracy (both z_tilde and z2 are compared through the identical
stats_head map, so this isolates representation GEOMETRY, not accuracy).

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_interpolation --latent-channels 8
    python -m evaluation.check_interpolation --latent-channels 8 \
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
from utils.naming import ae_checkpoint_name


def parse_fixed_triple(s: str) -> tuple[Path, int, int, int]:
    parts = s.split(":")
    if len(parts) != 4:
        raise ValueError(f"expected 'run_dir:t1:t2:t3', got '{s}'")
    run_dir, t1, t2, t3 = parts
    return Path(run_dir), int(t1), int(t2), int(t3)


def find_all_triples(test_dirs: list[Path], min_step: int) -> list[tuple[Path, int, int, int]]:
    """Every consecutive (t1,t2,t3) triple of kept steps across all test_dirs."""
    triples = []
    for run_dir in test_dirs:
        metadata = load.read_metadata(run_dir / "metadata.txt")
        check = load.check_snapshots_saved(run_dir, metadata)
        bad_steps = set(check["missing"]) | set(check["bad_size"])
        kept = [s for s in metadata.save_steps if s not in bad_steps and s >= min_step]
        for i in range(len(kept) - 2):
            triples.append((run_dir, kept[i], kept[i + 1], kept[i + 2]))
    return triples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("../../config.txt"),
            help="source for --size/--stats-weight/--min-step defaults")
    parser.add_argument("--size", type=int, default=None,
            help="default: read from --config's Nx/Ny")
    parser.add_argument("--latent-channels", type=int, default=4,
            help="required -- not a sweep parameter, so config.txt has no value for this")
    parser.add_argument("--stats-weight", type=float, default=None,
            help="default: read from --config's stats_weight")
    parser.add_argument("--checkpoint", type=Path, default=None,
            help="direct path override, instead of --size/--latent-channels/--stats-weight")
    parser.add_argument("--n-samples", type=int, default=None,
            help="default: use EVERY available (t1,t2,t3) triple in the test set")
    parser.add_argument("--min-step", type=int, default=None,
            help="default: read from --config's min_step")
    parser.add_argument("--seed", type=int, default=0,
                         help="only matters if --n-samples subsamples; ignored otherwise")
    parser.add_argument("--fixed-triples", type=str, nargs="+", default=None,
                         help="'run_dir:t1:t2:t3' (repeatable) for reproducible comparison")
    parser.add_argument("--output", type=Path, default=None,
            help="default: ../../output/interpolation_check_png/<checkpoint name>.png")
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
        args.output = Path(f"../../output/interpolation_check_png/{args.checkpoint.stem}.png")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, config={model_cfg}")

    stats_config = checkpoint.get("stats_config")
    if stats_config is None:
        raise ValueError(f"{args.checkpoint} has no stats_head (trained with --stats-weight 0)")

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

    if args.fixed_triples:
        triples = [parse_fixed_triple(s) for s in args.fixed_triples]
    else:
        test_dirs = checkpoint.get("test_dirs") or []
        if not test_dirs:
            raise ValueError(f"{args.checkpoint} has no saved test_dirs")
        test_dirs = [Path(d) for d in test_dirs]
        triples = find_all_triples(test_dirs, args.min_step)
        if not triples:
            raise ValueError("No consecutive (t1,t2,t3) triples found in test_dirs")
        if args.n_samples is not None and args.n_samples < len(triples):
            generator = torch.Generator().manual_seed(args.seed)
            idx = torch.randperm(len(triples), generator=generator)[:args.n_samples].tolist()
            triples = [triples[i] for i in idx]
        print(f"Using {len(triples)} (t1,t2,t3) triples from {len(test_dirs)} test dirs")

    nx = ny = model_cfg["size"]
    n = len(triples)
    relative_error = np.zeros(n)
    elapsed_span = np.zeros(n)

    with torch.no_grad():
        for i, (run_dir, t1, t2, t3) in enumerate(triples):
            alpha = (t2 - t1) / (t3 - t1)
            metadata = load.read_metadata(run_dir / "metadata.txt")
            elapsed_span[i] = (t3 - t1) * metadata.dt

            x1_np = load.read_phi_half(run_dir / load.snapshot_filename(t1), nx, ny)
            x2_np = load.read_phi_half(run_dir / load.snapshot_filename(t2), nx, ny)
            x3_np = load.read_phi_half(run_dir / load.snapshot_filename(t3), nx, ny)
            x1 = torch.from_numpy(x1_np).unsqueeze(0).unsqueeze(0).to(device)
            x2 = torch.from_numpy(x2_np).unsqueeze(0).unsqueeze(0).to(device)
            x3 = torch.from_numpy(x3_np).unsqueeze(0).unsqueeze(0).to(device)

            z1 = ae.encoder(x1)
            z2 = ae.encoder(x2)
            z3 = ae.encoder(x3)
            z_tilde = (1 - alpha) * z1 + alpha * z3

            norm_z_tilde = stats_head(z_tilde).mean().item()
            norm_z2 = stats_head(z2).mean().item()

            relative_error[i] = (norm_z_tilde - norm_z2) / norm_z2 if abs(norm_z2) > 1e-3 else np.nan

    valid = ~np.isnan(relative_error)
    n_dropped = n - valid.sum()
    if n_dropped:
        print(f"\nDropped {n_dropped}/{n} triples with ||z2|| too close to 0 (<1e-3) -- "
              f"a relative error isn't meaningful when the reference value itself is ~0")
    rel_err = relative_error[valid]
    span_valid = elapsed_span[valid]

    print(f"\n(||z_tilde|| - ||z2||) / ||z2||, across {valid.sum()} triples:")
    print(f"  mean:   {rel_err.mean():.4f}   (systematic bias if far from 0)")
    print(f"  median: {np.median(rel_err):.4f}")
    print(f"  mean |error|: {np.abs(rel_err).mean():.4f}")
    print(f"  std:    {rel_err.std():.4f}")
    if len(rel_err) >= 4:
        corr = np.corrcoef(np.log(span_valid), np.abs(rel_err))[0, 1]
        print(f"  corr(log elapsed_span, |error|) = {corr:.3f} "
              f"(positive -> worse over longer spans)")

    lo, hi = np.percentile(rel_err, [0.5, 99.5])
    n_outside = int(((rel_err < lo) | (rel_err > hi)).sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(rel_err, bins=np.linspace(lo, hi, 40), color="tab:blue")
    ax.axvline(0.0, color="red", linestyle="--", label="0 = perfect")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("(||z_tilde|| - ||z2||) / ||z2||")
    ax.set_ylabel("count")
    ax.set_title(f"Distribution across {len(rel_err)} triples "
                 f"(central 99% shown, {n_outside} outliers excluded from view)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()