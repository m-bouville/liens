"""
Stage 2 (train_autoencoder) and stage 3 (train_stage3) of the LIENS
pipeline. Both are importable functions, not just CLI scripts -- see
main.py for the orchestrated 2 -> 3 -> 4 pipeline. train_ae.py's CLI
below covers stage 2 only; stage 3 is driven by main.py, since it always
resumes from a stage-2 checkpoint (there's no "stage 3 from scratch").

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m training.train_ae --config ../config.txt --base ../datasets
"""

import argparse
from pathlib import Path
import gc

import torch
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder
from training.datasets import MicrostructureSnapshotDataset, MicrostructureTripletDataset, \
                               complete_run_dirs, split_run_dirs
from training.losses import ReconLoss, StatsLoss
from training.stats_head import StatsHead
from utils import load_datasets as load
from utils.naming import ae_checkpoint_name


def train_autoencoder(
    config_path: Path, base_path: Path,
    epochs: int = 100, batch_size: int = 64, lr: float = 1e-3,
    base_channels: int = 32, latent_channels: int = 8,
    val_fraction: float = 0.2, test_fraction: float = 0.1, num_workers: int = 4,
    min_step: int | None = None, min_stdev_phi: float | None = None,
    stat_names: list[str] | None = None, stats_weight: float | None = None,
    val_ema_decay: float = 0.7, early_stopping_patience: int | None = None,
    seed: int = 0, checkpoint_path: Path | None = None,
    resume_from: Path | None = None, device: str | None = None,
) -> Path:
    """
    Stage 2: train the AE on individual snapshots with L_recon, and (if
    stats_weight > 0) L_stats via stats_head.py. Returns the path of the
    best checkpoint saved (selected on an EMA of val_total, see
    val_ema_decay).

    resume_from: optional checkpoint to initialize model_state/
    stats_head_state from before training starts (e.g. to continue stage
    2 training itself with more epochs -- NOT how stage 3 works, which
    is a separate function with a different loss/data structure).

    early_stopping_patience: stop once val_ema hasn't improved for this
    many consecutive epochs, instead of always running the full `epochs`
    budget -- a data-driven stopping signal rather than a guessed epoch
    count for "stage 2 is done".
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    config = load.read_config(config_path)
    if config.nx != config.ny:
        raise ValueError(f"Autoencoder assumes a square grid, got {config.nx}x{config.ny}")

    if min_step is None:
        min_step = config.min_step
    if min_stdev_phi is None:
        min_stdev_phi = config.min_stdev_phi
    if stats_weight is None:
        stats_weight = config.stats_weight
    print(f"min_step={min_step}  min_stdev_phi={min_stdev_phi}  stats_weight={stats_weight}")

    include_stats = stats_weight > 1e-6

    if checkpoint_path is None:
        name = ae_checkpoint_name(config.nx, latent_channels, stats_weight)
        checkpoint_path = Path(f"../output/ae_checkpoint_pt/{name}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {checkpoint_path}")

    run_dirs = complete_run_dirs(config, base_path)
    if not run_dirs:
        raise ValueError("No complete runs found -- check config_path/base_path")

    # Split at the DIRECTORY level, not by frame: two frames from the same
    # run, even at unrelated timesteps, are snapshots of one continuous
    # evolution and can look very similar, so splitting below the
    # directory level risks leaking correlated frames across train/val/test.
    train_dirs, val_dirs, test_dirs = split_run_dirs(run_dirs, val_fraction, test_fraction, seed=seed)
    print(f"{len(run_dirs)} complete runs ({config.nx} x {config.ny}) -> "
          f"{len(train_dirs)} train / {len(val_dirs)} val / {len(test_dirs)} test dirs")

    train_set = MicrostructureSnapshotDataset(
        train_dirs, cache_in_memory=True, augment=True,
        min_step=min_step, min_stdev_phi=min_stdev_phi,
        include_stats=include_stats, stat_names=stat_names,
    )
    # Lock val's stat_names to whatever train resolved (auto-detection is
    # order-dependent; without this, val could silently end up checked
    # against a different run's schema than train was).
    val_stat_names = train_set.stat_names if include_stats else None
    val_set = MicrostructureSnapshotDataset(
        val_dirs, cache_in_memory=True, augment=False,
        min_step=min_step, min_stdev_phi=min_stdev_phi,
        include_stats=include_stats, stat_names=val_stat_names,
    )
    print(f"{train_set.n_base_samples} base snapshots (train) -> {len(train_set)} after "
          f"augmentation, {val_set.n_base_samples} (val, unaugmented)")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        persistent_workers=num_workers > 0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0, pin_memory=device.type == "cuda",
    )

    ae = Autoencoder(size=config.nx, channels=1, base_channels=base_channels,
                      latent_channels=latent_channels).to(device)

    recon_loss = ReconLoss()
    params = list(ae.parameters())

    stats_head = None
    stats_loss_fn = None
    if include_stats:
        stats_head = StatsHead(latent_channels=latent_channels, stat_names=train_set.stat_names).to(device)
        mean, std = train_set.stats_normalization()
        stats_loss_fn = StatsLoss(mean.to(device), std.to(device), stat_names=train_set.stat_names)
        params += list(stats_head.parameters())

    if resume_from is not None:
        prev = torch.load(resume_from, map_location=device, weights_only=True)
        ae.load_state_dict(prev["model_state"])
        if stats_head is not None and prev.get("stats_head_state") is not None:
            stats_head.load_state_dict(prev["stats_head_state"])
        print(f"Resumed model weights from {resume_from}")

    optimizer = torch.optim.Adam(params, lr=lr)

    def step(batch, train: bool):
        if include_stats:
            x, stats_target = batch
            x = x.to(device, non_blocking=True)
            stats_target = stats_target.to(device, non_blocking=True)
        else:
            x = batch.to(device, non_blocking=True)
            stats_target = None

        x_recon, z = ae(x)
        recon = recon_loss(x_recon, x)

        if include_stats:
            stats_pred = stats_head(z)
            stats = stats_loss_fn(stats_pred, stats_target)
            total = recon + stats_weight * stats
        else:
            stats = torch.tensor(0.0)
            total = recon

        if train:
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        return total.item(), recon.item(), stats.item()

    best_val_loss = float("inf")
    val_ema = None
    epochs_since_improvement = 0

    print(f"Starting {epochs} epochs (batches of {batch_size})...")
    heading = f"/{epochs:3d} "
    heading += (f"train = recon +{stats_weight:6.3f} stats | valid = recon +{stats_weight:6.3f} "
                f"stats  (e-3)  ema") if include_stats else "train | valid  (e-3)  ema"
    print(heading)

    for epoch in range(1, epochs + 1):
        ae.train()
        if stats_head is not None:
            stats_head.train()
        train_total = train_recon = train_stats = 0.0
        n_train = len(train_set)
        for batch in train_loader:
            bs = batch[0].size(0) if include_stats else batch.size(0)
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
                bs = batch[0].size(0) if include_stats else batch.size(0)
                total, recon, stats = step(batch, train=False)
                val_total += total * bs
                val_recon += recon * bs
                val_stats += stats * bs
        val_total /= n_val
        val_recon /= n_val
        val_stats /= n_val

        val_ema = val_total if val_ema is None else val_ema_decay * val_ema + (1 - val_ema_decay) * val_total

        msg = f"{epoch:4d}"
        if include_stats:
            msg += (f"{train_total*1_000:7.2f} ={train_recon*1_000:7.2f} +{train_stats*1_000:7.1f} |"
                    f"{val_total*1_000:7.2f} ={val_recon*1_000:7.2f} +{val_stats*1_000:7.1f}"
                    f"  {val_ema*1_000:7.2f}")
        else:
            msg += f"{train_total*1_000:7.2f} |{val_total*1_000:7.2f}  {val_ema*1_000:7.2f}"

        if val_ema < best_val_loss:
            best_val_loss = val_ema
            epochs_since_improvement = 0
            checkpoint = {
                "model_state": ae.state_dict(),
                "stats_head_state": stats_head.state_dict() if stats_head is not None else None,
                "epoch": epoch,
                "val_loss": val_total,
                "val_loss_ema": val_ema,
                "normalized": False,
                "test_dirs": [str(Path(d).resolve()) for d in test_dirs],
                "config": {
                    "size": config.nx, "base_channels": base_channels,
                    "latent_channels": latent_channels, "stats_weight": stats_weight,
                },
                "stats_config": {
                    "stat_names": train_set.stat_names,
                    "stats_mean": stats_loss_fn.mean.cpu() if stats_loss_fn is not None else None,
                    "stats_std": stats_loss_fn.std.cpu() if stats_loss_fn is not None else None,
                } if include_stats else None,
            }
            torch.save(checkpoint, checkpoint_path)
            msg += "  -> saved"
        else:
            epochs_since_improvement += 1

        print(msg)

        if early_stopping_patience is not None and epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}: no improvement for "
                  f"{early_stopping_patience} epochs")
            break

    return checkpoint_path


def train_stage3(
    config_path: Path, base_path: Path, resume_from: Path,
    interp_weight: float = 1.0,
    epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
    val_fraction: float = 0.2, test_fraction: float = 0.1, num_workers: int = 4,
    min_step: int | None = None, min_stdev_phi: float | None = None,
    val_ema_decay: float = 0.7, early_stopping_patience: int | None = None,
    seed: int = 0, checkpoint_path: Path | None = None, device: str | None = None,
) -> Path:
    """
    Stage 3: continue training the AE (encoder, decoder, AND stats_head --
    none of them frozen here) on real (t1,t2,t3) triplets, adding
    L_interp (interpolation-consistency, compared against TRUE t2
    statistics from statistics.csv, not stats_head(z2) -- grounds the
    target in real data rather than another model's prediction) to
    L_recon and L_stats, BOTH of which are computed on x2/z2 specifically
    (the real middle frame of each triplet), continuing exactly the same
    objective stage 2 used, just now alongside L_interp.

    Always resumes from a stage-2 (or another stage-3) checkpoint --
    there's no "stage 3 from scratch". stats_weight is read from that
    checkpoint's own config, not re-specified here.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    config = load.read_config(config_path)
    if min_step is None:
        min_step = config.min_step
    if min_stdev_phi is None:
        min_stdev_phi = config.min_stdev_phi

    prev = torch.load(resume_from, map_location=device, weights_only=True)
    model_cfg = prev["config"]
    stats_config = prev.get("stats_config")
    if stats_config is None:
        raise ValueError(f"{resume_from} has no stats_head (stage 2 needs stats_weight > 0 -- "
                          f"both L_stats and L_interp require it)")
    stat_names = stats_config["stat_names"]
    stats_weight = model_cfg["stats_weight"]
    print(f"Resuming from {resume_from} (stat_names={stat_names}, stats_weight={stats_weight})")

    ae = Autoencoder(size=model_cfg["size"], channels=1, base_channels=model_cfg["base_channels"],
                      latent_channels=model_cfg["latent_channels"]).to(device)
    ae.load_state_dict(prev["model_state"])

    stats_head = StatsHead(latent_channels=model_cfg["latent_channels"], stat_names=stat_names).to(device)
    stats_head.load_state_dict(prev["stats_head_state"])

    mean = stats_config["stats_mean"].to(device)
    std = stats_config["stats_std"].to(device)
    stats_loss_fn = StatsLoss(mean, std, stat_names=stat_names)
    recon_loss = ReconLoss()

    if checkpoint_path is None:
        name = ae_checkpoint_name(model_cfg["size"], model_cfg["latent_channels"], stats_weight)
        checkpoint_path = Path(f"../output/ae_checkpoint_pt/{name}-stage3.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {checkpoint_path}")

    run_dirs = complete_run_dirs(config, base_path)
    if not run_dirs:
        raise ValueError("No complete runs found -- check config_path/base_path")
    train_dirs, val_dirs, test_dirs = split_run_dirs(run_dirs, val_fraction, test_fraction, seed=seed)
    print(f"{len(run_dirs)} complete runs -> "
          f"{len(train_dirs)} train / {len(val_dirs)} val / {len(test_dirs)} test dirs")

    train_set = MicrostructureTripletDataset(train_dirs, stat_names=stat_names,
                                              min_step=min_step, min_stdev_phi=min_stdev_phi)
    val_set = MicrostructureTripletDataset(val_dirs, stat_names=stat_names,
                                            min_step=min_step, min_stdev_phi=min_stdev_phi)
    print(f"{len(train_set)} train triplets, {len(val_set)} val triplets")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")

    params = list(ae.parameters()) + list(stats_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    def step(batch, train: bool):
        x1, x2, x3, alpha, true_stats = batch
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        x3 = x3.to(device, non_blocking=True)
        alpha = alpha.to(device, non_blocking=True).view(-1, 1, 1, 1)
        true_stats = true_stats.to(device, non_blocking=True)

        z1 = ae.encoder(x1)
        z2 = ae.encoder(x2)
        z3 = ae.encoder(x3)
        z_tilde = (1 - alpha) * z1 + alpha * z3

        x2_recon = ae.decoder(z2)
        recon = recon_loss(x2_recon, x2)

        stats_z2 = stats_head(z2)
        stats_loss_val = stats_loss_fn(stats_z2, true_stats)

        stats_z_tilde = stats_head(z_tilde)
        interp_loss = stats_loss_fn(stats_z_tilde, true_stats)

        total = recon + stats_weight * stats_loss_val + interp_weight * interp_loss

        if train:
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        return total.item(), recon.item(), stats_loss_val.item(), interp_loss.item()

    best_val_loss = float("inf")
    val_ema = None
    epochs_since_improvement = 0

    print(f"Stage 3: starting {epochs} epochs (batches of {batch_size}), interp_weight={interp_weight}")
    print(f"/{epochs:3d} train=recon+stats+interp | valid=... (e-3) ema")

    for epoch in range(1, epochs + 1):
        ae.train()
        stats_head.train()
        train_total = train_recon = train_stats = train_interp = 0.0
        n_train = len(train_set)
        for batch in train_loader:
            bs = batch[0].size(0)
            total, recon, stats, interp = step(batch, train=True)
            train_total += total * bs
            train_recon += recon * bs
            train_stats += stats * bs
            train_interp += interp * bs
        train_total /= n_train
        train_recon /= n_train
        train_stats /= n_train
        train_interp /= n_train

        ae.eval()
        stats_head.eval()
        val_total = val_recon = val_stats = val_interp = 0.0
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0)
                total, recon, stats, interp = step(batch, train=False)
                val_total += total * bs
                val_recon += recon * bs
                val_stats += stats * bs
                val_interp += interp * bs
        val_total /= n_val
        val_recon /= n_val
        val_stats /= n_val
        val_interp /= n_val

        val_ema = val_total if val_ema is None else val_ema_decay * val_ema + (1 - val_ema_decay) * val_total

        msg = (f"{epoch:4d} {train_total*1_000:7.2f}=recon{train_recon*1_000:6.2f}"
               f"+stats{train_stats*1_000:6.1f}+interp{train_interp*1_000:6.1f} | "
               f"{val_total*1_000:7.2f}=recon{val_recon*1_000:6.2f}"
               f"+stats{val_stats*1_000:6.1f}+interp{val_interp*1_000:6.1f}  ema{val_ema*1_000:7.2f}")

        if val_ema < best_val_loss:
            best_val_loss = val_ema
            epochs_since_improvement = 0
            torch.save({
                "model_state": ae.state_dict(),
                "stats_head_state": stats_head.state_dict(),
                "epoch": epoch,
                "val_loss": val_total,
                "val_loss_ema": val_ema,
                "test_dirs": [str(Path(d).resolve()) for d in test_dirs],
                "config": {
                    "size": model_cfg["size"], "base_channels": model_cfg["base_channels"],
                    "latent_channels": model_cfg["latent_channels"], "stats_weight": stats_weight,
                },
                "stats_config": {"stat_names": stat_names, "stats_mean": mean.cpu(), "stats_std": std.cpu()},
                "stage3_config": {"interp_weight": interp_weight, "resumed_from": str(resume_from)},
            }, checkpoint_path)
            msg += "  -> saved"
        else:
            epochs_since_improvement += 1

        print(msg)

        if early_stopping_patience is not None and epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}: no improvement for "
                  f"{early_stopping_patience} epochs")
            break

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("../config.txt"))
    parser.add_argument("--base", type=Path, default=Path("../datasets"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--min-stdev-phi", type=float, default=None)
    parser.add_argument("--stat-names", type=str, nargs="+", default=None)
    parser.add_argument("--stats-weight", type=float, default=None)
    parser.add_argument("--val-ema-decay", type=float, default=0.7)
    parser.add_argument("--early-stopping-patience", type=int, default=None,
            help="stop once val_ema hasn't improved for this many epochs; default runs "
                 "the full --epochs budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        print("CUDA not available")

    train_autoencoder(
        config_path=args.config, base_path=args.base,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        base_channels=args.base_channels, latent_channels=args.latent_channels,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        num_workers=args.num_workers, min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        stat_names=args.stat_names, stats_weight=args.stats_weight,
        val_ema_decay=args.val_ema_decay, early_stopping_patience=args.early_stopping_patience,
        seed=args.seed, checkpoint_path=args.checkpoint, resume_from=args.resume_from,
        device=args.device,
    )

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
