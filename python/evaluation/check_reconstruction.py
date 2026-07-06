"""
Load a trained autoencoder checkpoint and visually compare its held-out
TEST set (saved in the checkpoint by train_ae.py, never touched during
training or checkpoint selection) against their reconstructions.

check_reconstruction() is importable -- see main.py, which calls it
automatically after stages 2/3 with the checkpoint path it already has
in hand. The CLI below is for standalone use.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_reconstruction --latent-channels 4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from models.autoencoder import Autoencoder
from training.datasets import MicrostructureSnapshotDataset
from training.losses import ReconLoss
from utils import load_datasets as load
from utils.naming import ae_checkpoint_name


def check_reconstruction(
    checkpoint_path: Path, n_samples: int = 6, seed: int = 0, min_step: int = 0,
    output_path: Path | None = None, device: str | None = None,
) -> Path:
    """Saves a visual comparison figure and returns its path."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if output_path is None:
        output_path = Path(f"../../output/reconstruction_check_png/{checkpoint_path.stem}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f}, config={model_cfg}")

    test_dirs = checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(
            f"{checkpoint_path} has no saved test_dirs -- it was likely trained with "
            f"--test-fraction 0, or with an older version of train_ae.py."
        )
    test_dirs = [Path(d) for d in test_dirs]

    ae = Autoencoder(
        size=model_cfg["size"], channels=1,
        base_channels=model_cfg["base_channels"], latent_channels=model_cfg["latent_channels"],
    ).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()

    # Deliberately unaugmented: we want to look at real frames, not
    # rotated/translated synthetic views. Uses the checkpoint's own
    # saved test_dirs, so this is guaranteed to be the exact same held-out
    # set that training never touched.
    dataset = MicrostructureSnapshotDataset(test_dirs, augment=False, min_step=min_step)
    if len(dataset) == 0:
        raise ValueError(f"No snapshots found in the checkpoint's {len(test_dirs)} test_dirs "
                          f"(after min_step={min_step} filtering)")

    generator = torch.Generator().manual_seed(seed)
    n_samples = min(n_samples, len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:n_samples].tolist()

    recon_loss = ReconLoss()

    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]  # keep 2D indexing uniform for a single sample

    with torch.no_grad():
        for row, idx in enumerate(indices):
            x = dataset[idx].unsqueeze(0).to(device)  # (1, 1, H, W)
            x_recon, _ = ae(x)
            loss = recon_loss(x_recon, x).item()

            x_np = x[0, 0].cpu().numpy()
            x_recon_np = x_recon[0, 0].cpu().numpy()
            diff_np = x_recon_np - x_np

            # Auto-scale symmetric around 0, from the actual data range of
            # THIS sample, floored at 0.1 (see docstring history: avoids
            # amplifying near-noise samples into looking like real signal).
            scale = max(abs(x_np.min()), abs(x_np.max()), 0.1)
            diff_scale = max(abs(diff_np.min()), abs(diff_np.max()), 1e-6)

            axes[row, 0].imshow(x_np, cmap="RdBu", vmin=-scale, vmax=scale)
            axes[row, 0].set_title(f"original (idx={idx}, scale=+-{scale:.3f})" if row == 0
                                    else f"scale=+-{scale:.3f}")
            axes[row, 1].imshow(x_recon_np, cmap="RdBu", vmin=-scale, vmax=scale)
            axes[row, 1].set_title(f"reconstruction (loss={loss:.4f})" if row == 0 else
                                    f"loss={loss:.4f}")
            im_diff = axes[row, 2].imshow(diff_np, cmap="RdBu", vmin=-diff_scale, vmax=diff_scale)
            axes[row, 2].set_title(f"difference (scale=+-{diff_scale:.3f})" if row == 0
                                    else f"scale=+-{diff_scale:.3f}")
            fig.colorbar(im_diff, ax=axes[row, 2], fraction=0.046)

            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"Saved comparison figure to {output_path} ({n_samples} samples from "
          f"{len(test_dirs)} held-out test dirs)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("../../config.txt"),
            help="source for --size/--stats-weight defaults")
    parser.add_argument("--size", type=int, default=None, help="default: read from --config")
    parser.add_argument("--latent-channels", type=int, default=None,
            help="required -- not a sweep parameter, so config.txt has no value for this")
    parser.add_argument("--stats-weight", type=float, default=None,
            help="default: read from --config")
    parser.add_argument("--checkpoint", type=Path, default=None,
            help="direct path override, instead of --size/--latent-channels/--stats-weight")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint is None:
        if args.size is None or args.stats_weight is None:
            config = load.read_config(args.config)
            if args.size is None:
                args.size = config.nx
            if args.stats_weight is None:
                args.stats_weight = config.stats_weight
        if args.latent_channels is None:
            raise ValueError("Provide either --checkpoint directly, or --latent-channels")
        name = ae_checkpoint_name(args.size, args.latent_channels, args.stats_weight)
        args.checkpoint = Path(f"../../output/ae_checkpoint_pt/{name}.pt")
        print(f"Reconstructed checkpoint path: {args.checkpoint}")

    check_reconstruction(
        checkpoint_path=args.checkpoint, n_samples=args.n_samples, seed=args.seed,
        min_step=args.min_step, output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()
