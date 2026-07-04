"""
Load a trained autoencoder checkpoint and visually compare its held-out
TEST set (saved in the checkpoint by train_ae.py, never touched during
training or checkpoint selection) against their reconstructions.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_reconstruction --checkpoint ae_checkpoint.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from models.autoencoder import Autoencoder
from training.datasets import MicrostructureSnapshotDataset
from training.losses import ReconLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("../output/ae_checkpoint.pt"))
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0,
                         help="which test-set frames to display (not the train/val/test "
                              "split itself, which is fixed and loaded from the checkpoint)")
    parser.add_argument("--min-step", type=int, default=0,
                         help="skip snapshots earlier than this step (early steps are "
                              "near-pure noise and look flat under any fixed color scale)")
    parser.add_argument("--output", type=Path, default=Path("../output/reconstruction_check.png"))
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f}, config={model_cfg}")

    test_dirs = checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(
            f"{args.checkpoint} has no saved test_dirs -- it was likely trained with "
            f"--test-fraction 0, or with an older version of train_ae.py. Re-train with "
            f"--test-fraction > 0 to get a held-out test set for this script."
        )
    test_dirs = [Path(d) for d in test_dirs]

    ae = Autoencoder(
        size=model_cfg["size"],
        channels=1,
        base_channels=model_cfg["base_channels"],
        latent_channels=model_cfg["latent_channels"],
    ).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()

    # Deliberately unaugmented: we want to look at real frames, not
    # rotated/translated synthetic views. Uses the checkpoint's own
    # saved test_dirs, so this is guaranteed to be the exact same held-out
    # set that training never touched -- not re-derived from config/seed,
    # which could silently drift if the dataset on disk changes later.
    dataset = MicrostructureSnapshotDataset(test_dirs, augment=False, min_step=args.min_step)
    if len(dataset) == 0:
        raise ValueError(f"No snapshots found in the checkpoint's {len(test_dirs)} test_dirs "
                          f"(after min_step={args.min_step} filtering)")

    generator = torch.Generator().manual_seed(args.seed)
    n_samples = min(args.n_samples, len(dataset))
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
            # THIS sample -- a fixed global scale makes low-amplitude
            # fields (like early, still noise-dominated steps) look
            # flat/blank even though real structure is there.
            scale      = max(abs(x_np   .min()), abs(x_np   .max()), 0.1)
                # if the scale drops below +/-0.1, we plot noise as signal.
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
    fig.savefig(args.output, dpi=120)
    print(f"Saved comparison figure to {args.output} ({n_samples} samples from "
          f"{len(test_dirs)} held-out test dirs)")


if __name__ == "__main__":
    main()
