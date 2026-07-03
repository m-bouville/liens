"""
Stage 2: train the autoencoder on individual snapshots with L_recon only
(no L_stats yet -- stats_head.py doesn't exist).

Usage:
    python train_ae.py --config config.txt --base ../datasets
"""

import argparse
from   pathlib            import Path

import torch
from   torch.utils.data   import DataLoader

from   models.autoencoder import Autoencoder
from   training.datasets  import MicrostructureSnapshotDataset
from   training.losses    import ReconLoss
from   utils              import load_datasets as load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",       type=Path, default=Path("../config.txt"))
    parser.add_argument("--base",         type=Path, default=Path("../datasets"))
    parser.add_argument("--epochs",       type=int,  default=50)
    parser.add_argument("--batch-size",   type=int,  default=32)
                # 64 for 64x64, 32 for 128x128
    parser.add_argument("--lr",           type=float,default=1e-3)
    parser.add_argument("--base-channels",type=int,  default=32)
    parser.add_argument("--latent-channels",type=int,default=4)
                # 4 for 64x64, 6? for 128x128
    parser.add_argument("--val-fraction", type=float,default=0.1)
    parser.add_argument("--num-workers",  type=int,  default=4)
    # parser.add_argument("--augment", action="store_true",
    #             help="expand training data 32x via D4 symmetry + periodic translation")
    # parser.add_argument("--cache-in-memory", action="store_true")
    parser.add_argument("--checkpoint",   type=Path, default=Path("ae_checkpoint.pt"))
    parser.add_argument("--device",       type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.augment = True
    args.cache_in_memory = True



    # PyTorch and CUDA
    print("PyTorch version:",torch.__version__)
    if torch.cuda.is_available():
        print("CUDA version:",   torch.version.cuda)
        print("GPU count:",      torch.cuda.device_count())
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print("CUDA not available")

    device = torch.device(args.device)
    print(f"device: {device}")
    print()


    config = load.read_config(args.config)
    if config.nx != config.ny:
        raise ValueError(f"Autoencoder assumes a square grid, got {config.nx}x{config.ny}")

    dataset = MicrostructureSnapshotDataset.from_sweep(config, base=args.base,
            cache_in_memory=args.cache_in_memory, augment=args.augment)
    if len(dataset) == 0:
        raise ValueError("No complete runs found -- check --config/--base paths")
    print(f"{dataset.n_base_samples} snapshots ({config.nx} x {config.ny}) "
          + "pooled from complete runs"
          + (f" -> {len(dataset)} after augmentation" if args.augment else ""))

    # Split on real, distinct frames (base samples) before augmentation
    # expands each into 32 views -- prevents augmented views of the same
    # underlying frame from landing on both sides of the split. val is
    # always evaluated unaugmented, so val_loss stays directly comparable
    # to the earlier (pre-augmentation) runs.
    train_set, val_set = dataset.random_split_by_base(args.val_fraction)
    n_train, n_val = len(train_set), len(val_set)

    device_type = args.device
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,  # avoid respawning workers every epoch
        pin_memory=device_type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device_type == "cuda",
    )

    device = torch.device(args.device)
    ae = Autoencoder(
        size=config.nx,
        channels=1,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
    ).to(device)

    recon_loss = ReconLoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    print(f"Starting {args.epochs} epochs (batches of {args.batch_size})...")

    for epoch in range(1, args.epochs + 1):
        ae.train()
        train_loss = 0.0
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            x_recon, _ = ae(x)
            loss = recon_loss(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
        train_loss /= n_train

        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                x_recon, _ = ae(x)
                val_loss += recon_loss(x_recon, x).item() * x.size(0)
        val_loss /= n_val

        print(f"{epoch:3d}/{args.epochs:3d}  train_loss:{train_loss*1000:8.4f}e-3 "
              f" val_loss:{val_loss*1000:8.4f}e-3")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": ae.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": {
                        "size": config.nx,
                        "base_channels": args.base_channels,
                        "latent_channels": args.latent_channels,
                    },
                },
                args.checkpoint,
            )
            print(f"  -> saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()