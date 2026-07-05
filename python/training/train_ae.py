"""
Stage 2: train the autoencoder on individual snapshots with L_recon,
and (if --include-stats) L_stats via stats_head.py.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m training.train_ae --config config.txt --base ../datasets
"""

import argparse
from   pathlib import Path
import gc

import torch
from   torch.utils.data    import DataLoader

from   models.autoencoder  import Autoencoder
from   training.datasets   import MicrostructureSnapshotDataset, \
                                  complete_run_dirs, split_run_dirs
from   training.losses     import ReconLoss, StatsLoss
from   training.stats_head import StatsHead
from   utils               import load_datasets   as load
from   utils.naming        import ae_checkpoint_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",       type=Path, default=Path("../config.txt"))
    parser.add_argument("--base",         type=Path, default=Path("../datasets"))
    parser.add_argument("--epochs",       type=int,  default=20)
    parser.add_argument("--batch-size",   type=int,  default=64)
                # 64 for 64x64, 32 for 128x128
    parser.add_argument("--lr",           type=float,default=1e-3)
    parser.add_argument("--base-channels",type=int,  default=32)
    parser.add_argument("--latent-channels",type=int,default=8)
                # 4 for 64x64, 6? for 128x128
    parser.add_argument("--val-fraction", type=float,default=0.2)
    parser.add_argument("--test-fraction",type=float,default=0.1,
            help="held out entirely from training, for check_reconstruction.py "
                 "later -- unlike val (used for checkpoint selection, so not a "
                 "truly blind measure), test is never touched during training")
    parser.add_argument("--num-workers", type=int, default=4)
    # parser.add_argument("--augment", action="store_true",
    #         help="expand training data 32x via D4 symmetry + periodic translation")
    # parser.add_argument("--cache-in-memory", action="store_true")
    parser.add_argument("--min-step", type=int, default=None,
            help="skip snapshots earlier than this step (early steps are "
                 "near-pure noise, before microstructure develops). Default: read "
                 "from config.txt's min_step.")
    parser.add_argument("--min-stdev-phi", type=float, default=None,
            help="skip snapshots whose statistics.csv stdev_phi is below this "
                 "or NaN -- catches both pre-growth noise AND post-coarsening "
                 "single-grain states. Default: read from config.txt's "
                 "min_stdev_phi.")
    parser.add_argument("--stat-names", type=str, nargs="+", default=None,
            help="which statistics.csv columns to predict; default auto-detects "
                 "from the first train run (risky if runs have mixed schemas -- "
                 "e.g. some with 'trace', some without -- pass explicitly then). "
                 "'angle' is transformed to match D4 augmentation (see "
                 "training/datasets.py's _transform_angle) -- the 90-degree "
                 "rotation sign is a documented, unconfirmed assumption there.")
    parser.add_argument("--stats-weight", type=float, default=None,
            help="lambda_1: weight of L_stats relative to L_recon. 0.: statistics "
                 "not used, only L_recon. Default: read from config.txt's "
                 "stats_weight.")
    parser.add_argument("--val-ema-decay", type=float, default=0.7,
            help="EMA decay for the val loss used in checkpoint selection "
                 "(effective window ~= 1/(1-decay) epochs, so 0.7 ~ 3 epochs). "
                 "Smooths single-epoch val spikes/dips -- e.g. from a small val "
                 "set where a few hard/borderline samples can swing the whole "
                 "epoch average -- so checkpointing isn't driven by which epoch "
                 "got lucky on those samples.")
    parser.add_argument("--seed",       type=int, default=0,
            help="fixed for reproducibility across runs (data split + model init "
                 "+ shuffling); no expectation this needs changing between runs")
    parser.add_argument("--checkpoint", type=Path, default=None,
            help="default: auto-generated from size/latent-channels/stats-weight, e.g. "
                 "../output/ae_checkpoint_pt/64x64-4latent-stats_weight_0p01.pt -- "
                 "override to name it yourself")
    parser.add_argument("--device",     type=str,
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
        # print(f"memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # clear VRAM
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        print("CUDA not available")

    device_type = args.device
    device = torch.device(args.device)
    print(f"device: {device}")
    print()


    # Seeds everything downstream that draws from the global RNG: the
    # train/val/test directory split, model weight init, and DataLoader
    # shuffling order (which uses the global RNG unless a generator is
    # passed explicitly, which we don't). Must happen before any of those.
    torch.manual_seed(args.seed)

    config = load.read_config(args.config)
    if config.nx != config.ny:
        raise ValueError(f"Autoencoder assumes a square grid, got {config.nx}x{config.ny}")

    # CLI flag wins if explicitly given; otherwise use config.txt's
    # centralized value (single canonical config.txt, no backward-compat
    # fallback needed).
    if args.min_step is None:
        args.min_step = config.min_step
    if args.min_stdev_phi is None:
        args.min_stdev_phi = config.min_stdev_phi
    if args.stats_weight is None:
        args.stats_weight = config.stats_weight
    print(f"min_step={args.min_step}  min_stdev_phi={args.min_stdev_phi}  "
          f"stats_weight={args.stats_weight}")

    args.include_stats = (args.stats_weight > 1e-6)

    if args.checkpoint is None:
        name = ae_checkpoint_name(config.nx, args.latent_channels, args.stats_weight)
        args.checkpoint = Path(f"../output/ae_checkpoint_pt/{name}.pt")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {args.checkpoint}")

    run_dirs = complete_run_dirs(config, args.base)
    if not run_dirs:
        raise ValueError("No complete runs found -- check --config/--base paths")

    # Split at the DIRECTORY level, not by frame or by "distinct base
    # sample" (an earlier version of this code did the latter): two
    # frames from the same run, even at unrelated timesteps, are
    # snapshots of one continuous evolution and can look very similar,
    # so splitting below the directory level risks leaking correlated
    # frames across train/val/test.
    train_dirs, val_dirs, test_dirs = split_run_dirs(
        run_dirs, args.val_fraction, args.test_fraction, seed=args.seed,
    )
    print(f"{len(run_dirs)} complete runs ({config.nx} x {config.ny}) -> "
          f"{len(train_dirs)} train / {len(val_dirs)} val / {len(test_dirs)} test dirs")

    train_set = MicrostructureSnapshotDataset(
        train_dirs, cache_in_memory=args.cache_in_memory, augment=args.augment,
        min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        include_stats=args.include_stats, stat_names=args.stat_names,
    )
    # Lock val's stat_names to whatever train resolved (auto-detection is
    # order-dependent; without this, val could silently end up checked
    # against a different run's schema than train was).
    val_stat_names = train_set.stat_names if args.include_stats else None
    val_set = MicrostructureSnapshotDataset(
        val_dirs, cache_in_memory=args.cache_in_memory, augment=False,
        min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        include_stats=args.include_stats, stat_names=val_stat_names,
    )

    print(f"{train_set.n_base_samples} base snapshots (train)"
          + (f" -> {len(train_set)} after augmentation" if args.augment else "")
          + f", {val_set.n_base_samples} (val, unaugmented)")

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
    params = list(ae.parameters())

    stats_head = None
    stats_loss_fn = None
    if args.include_stats:
        stats_head = StatsHead(
            latent_channels=args.latent_channels, stat_names=train_set.stat_names,
        ).to(device)
        mean, std = train_set.stats_normalization()
        stats_loss_fn = StatsLoss(mean.to(device), std.to(device))
        params += list(stats_head.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    def step(batch, train: bool):
        """One forward pass (+ backward if train). Returns (total, recon, stats) losses."""
        if args.include_stats:
            x, stats_target = batch
            x = x.to(device, non_blocking=True)
            stats_target = stats_target.to(device, non_blocking=True)
        else:
            x = batch.to(device, non_blocking=True)
            stats_target = None

        x_recon, z = ae(x)
        recon = recon_loss(x_recon, x)

        if args.include_stats:
            stats_pred = stats_head(z)
            stats = stats_loss_fn(stats_pred, stats_target)
            total = recon + args.stats_weight * stats
        else:
            stats = torch.tensor(0.0)
            total = recon

        if train:
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        return total.item(), recon.item(), stats.item()

    best_val_loss = float("inf")
    val_ema = None  # EMA of val_total, used for checkpoint selection instead of
                    # the raw per-epoch value -- see --val-ema-decay help text


    print(f"Starting {args.epochs} epochs (batches of {args.batch_size})...")

    # heading for tabulated output
    heading = (f"/{args.epochs:3d} ")
    if args.include_stats:
        heading += (f"train = recon +{args.stats_weight:6.3f} stats | "
                    f"valid = recon +{args.stats_weight:6.3f} stats  (e-3)  ema")
    else:
        heading += ("train | valid  (e-3)  ema")
    print(heading)


    for epoch in range(1, args.epochs + 1):
        ae.train()
        if stats_head is not None:
            stats_head.train()
        train_total = train_recon = train_stats = 0.0
        n_train = len(train_set)
        for batch in train_loader:
            bs = batch[0].size(0) if args.include_stats else batch.size(0)
            total, recon, stats = step(batch, train=True)
            train_total += total * bs
            train_recon += recon * bs
            train_stats += stats * bs
        train_total /= n_train
        train_recon /= n_train
        train_stats /= n_train

        ae.eval()
        if stats_head is not None:
            stats_head.eval()
        val_total = val_recon = val_stats = 0.0
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0) if args.include_stats else batch.size(0)
                total, recon, stats = step(batch, train=False)
                val_total += total * bs
                val_recon += recon * bs
                val_stats += stats * bs
        val_total /= n_val
        val_recon /= n_val
        val_stats /= n_val

        # EMA update: first epoch has no history, so start from the raw value
        # rather than e.g. 0 (which would bias the whole EMA trajectory low).
        if val_ema is None:
            val_ema = val_total
        else:
            val_ema = args.val_ema_decay * val_ema + (1 - args.val_ema_decay) * val_total

        msg = (f"{epoch:4d}")
        if args.include_stats:
            msg+=(f"{train_total*1_000:7.2f} ={train_recon*1_000:7.2f} +{train_stats*1_000:7.1f} |"
                 +f"{val_total  *1_000:7.2f} ={val_recon  *1_000:7.2f} +{val_stats  *1_000:7.1f}"
                 +f"  {val_ema*1_000:7.2f}")
        else:
           msg += f"{train_total*1_000:7.2f} |{val_total*1_000:7.2f}  {val_ema*1_000:7.2f}"

        # Checkpoint on the EMA, not the raw per-epoch val_total: with a small
        # val set, a few hard/borderline samples can swing the whole-epoch
        # average enough that "best epoch" ends up picked by which epoch got
        # lucky on those samples rather than genuine improvement.
        if val_ema < best_val_loss:
            best_val_loss = val_ema
            checkpoint = {
                "model_state": ae.state_dict(),
                "stats_head_state": stats_head.state_dict()
                                if stats_head is not None else None,
                "epoch": epoch,
                "val_loss": val_total,      # raw value, for reference
                "val_loss_ema": val_ema,    # smoothed value that actually triggered this save
                "normalized": False,  # not yet implemented -- always False for now
                "test_dirs": [str(d) for d in test_dirs],
                "config": {
                    "size": config.nx,
                    "base_channels": args.base_channels,
                    "latent_channels": args.latent_channels,
                    "stats_weight": args.stats_weight,
                },
                "stats_config": {
                    "stat_names": train_set.stat_names,
                    "stats_mean": stats_loss_fn.mean.cpu()
                                if stats_loss_fn is not None else None,
                    "stats_std": stats_loss_fn.std.cpu()
                                if stats_loss_fn is not None else None,
                } if args.include_stats else None,
            }
            torch.save(checkpoint, args.checkpoint)
            msg += "  -> saved"

        print(msg)

if __name__ == "__main__":
    main()

    if torch.cuda.is_available():
        # clear VRAM
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
