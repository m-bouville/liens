"""
Load a trained autoencoder checkpoint and visually compare real
snapshots against their reconstructions, side by side.

Usage:
    python check_reconstruction.py --checkpoint ae_checkpoint.pt \
        --config config.txt --base ../datasets
"""

import argparse
from   pathlib import Path

import matplotlib.pyplot as plt
import torch

from   models.autoencoder import Autoencoder
from   training.datasets  import MicrostructureSnapshotDataset
from   training.losses    import ReconLoss
from   utils              import load_datasets as load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=Path("ae_checkpoint.pt"))
    parser.add_argument("--config",   type=Path, default=Path("../config.txt"))
    parser.add_argument("--base",     type=Path, default=Path("../datasets"))
    parser.add_argument("--n-samples",type=int,  default=6)
    parser.add_argument("--seed",     type=int,  default=0)
    parser.add_argument("--min-step", type=int,  default=10_000,
                     help="skip snapshots earlier than this step (early steps are "
                          "near-pure noise and look flat under any fixed color scale)")
    parser.add_argument("--vmin",     type=float,default=-0.7)
    parser.add_argument("--vmax",     type=float,default= 0.7)
    parser.add_argument("--output",   type=Path, default=Path("reconstruction_check.png"))
    parser.add_argument("--device",   type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f}, config={model_cfg}")

    ae = Autoencoder(
        size=model_cfg["size"],
        channels=1,
        base_channels=model_cfg["base_channels"],
        latent_channels=model_cfg["latent_channels"],
    ).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()

    # Deliberately unaugmented: we want to look at real frames, not
    # rotated/translated synthetic views.
    config = load.read_config(args.config)
    dataset = MicrostructureSnapshotDataset.from_sweep(config, base=args.base, augment=False)
    if len(dataset) == 0:
        raise ValueError("No complete runs found -- check --config/--base paths")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:args.n_samples].tolist()

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

            axes[row, 0].imshow(x_np, cmap="RdBu", vmin=args.vmin, vmax=args.vmax)
            axes[row, 0].set_title(f"original (idx={idx})" if row == 0 else "")
            axes[row, 1].imshow(x_recon_np, cmap="RdBu", vmin=args.vmin, vmax=args.vmax)
            axes[row, 1].set_title(f"reconstruction (loss={loss:.4f})" if row == 0 else
                                    f"loss={loss:.4f}")
            im_diff = axes[row, 2].imshow(diff_np, cmap="RdBu",
                                           vmin=-args.vmax / 4, vmax=args.vmax / 4)
            axes[row, 2].set_title("difference (x4 zoom)" if row == 0 else "")

            fig.colorbar(im_diff, ax=axes[row, 2], fraction=0.046)

            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()
