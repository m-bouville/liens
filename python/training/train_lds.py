"""
Stage 4: train the Latent Dynamics Surrogate (f_theta) with L_rollout
(chained multi-step prediction, errors accumulating as they would at
real inference time -- not L_1step's ground-truth-conditioned single
transitions), on top of a FROZEN, already-trained autoencoder (its
encoder only -- the decoder is not used here at all, only for later
visual checks).

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m training.train_lds --ae-latent-channels 8 --ae-stats-weight 0.01 \
        --config ../../config.txt --base ../../datasets --n-rollout-steps 6
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder
from models.latent_dynamics import LatentDynamics
from training.datasets import MicrostructureEvolutionDataset, complete_run_dirs, split_run_dirs
from training.losses import RolloutLoss
from utils import load_datasets as load
from utils.naming import ae_checkpoint_name, lds_checkpoint_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-latent-channels", type=int, default=8,
            help="which AE to use, by the parameters you actually know -- e.g. "
                 "--ae-latent-channels 8 --ae-stats-weight 0.01. Reconstructs the "
                 "expected checkpoint path (using this script's own --config grid size "
                 "and train_ae.py's naming convention) rather than you needing to type "
                 "out a filename. Required unless --ae-checkpoint is given directly.")
    parser.add_argument("--ae-stats-weight", type=float, default=None,
            help="see --ae-latent-channels. Also used directly as this LDS checkpoint's "
                 "own auto-generated name component (below), since it's already known "
                 "from what you typed -- no need to read it back out of the AE checkpoint. "
                 "Default: read from config.txt's stats_weight.")
    parser.add_argument("--ae-checkpoint", type=Path, default=None,
            help="direct path override, if you'd rather specify the AE checkpoint this "
                 "way instead of by --ae-latent-channels/--ae-stats-weight (e.g. for a "
                 "checkpoint that doesn't follow the standard naming convention)")
    parser.add_argument("--config", type=Path, default=Path("../../config.txt"))
    parser.add_argument("--base", type=Path, default=Path("../../datasets"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1,
            help="held out entirely, mirroring train_ae.py -- for a future rollout/visual "
                 "check, never touched during LDS training or checkpoint selection")
    parser.add_argument("--num-workers", type=int, default=0,
            help="unlike train_ae.py, MicrostructureEvolutionDataset is fully cached in "
                 "memory after the one-time encoding pass -- no disk I/O left to "
                 "parallelize, so extra workers are pure overhead here, and on Windows "
                 "combined with CUDA, num_workers>0 is a common cause of the whole process "
                 "hanging (each worker is a spawned process needing its own CUDA context). "
                 "Only raise this if you have a specific reason to.")
    parser.add_argument("--n-rollout-steps", type=int, default=1,
            help="number of CHAINED predictions per training window (n_r in the docs). "
                 "1 = pure one-step training (equivalent to the old OneStepLoss exactly -- "
                 "verified). >1 trains against real compounding drift: each step's "
                 "prediction feeds into the next, rather than always resetting to the "
                 "true z(t) before predicting. window_length is n_rollout_steps+1.")
    parser.add_argument("--min-step", type=int, default=None,
            help="should match (or be a superset of) what the AE was trained/filtered "
                 "with -- see --min-stdev-phi. Default: read from config.txt's min_step.")
    parser.add_argument("--min-stdev-phi", type=float, default=None,
            help="skip snapshots whose statistics.csv stdev_phi is below this or NaN -- "
                 "same rationale as train_ae.py's filter, applied here too since a "
                 "transition into/out of the noise-dominated regime isn't a meaningful "
                 "dynamics example either. Default: read from config.txt's "
                 "min_stdev_phi.")
    parser.add_argument("--encode-batch-size", type=int, default=256,
            help="batch size for the upfront, one-time encoding pass -- unrelated to "
                 "--batch-size, which is for LDS training itself")
    parser.add_argument("--val-ema-decay", type=float, default=0.7,
            help="see train_ae.py's --val-ema-decay -- same rationale, smooths checkpoint "
                 "selection against small-val-set noise")
    parser.add_argument("--ema-warmup-epochs", type=int, default=5,
            help="don't seed/compare val_ema until after this many epochs -- rollout "
                 "training's first few epochs can be catastrophically unstable (observed: "
                 "val_loss ~1e11 at epoch 1), and since EMA decays geometrically, seeding "
                 "it from such an extreme outlier means val_ema stays huge (and therefore "
                 "'improving' on every subsequent epoch, triggering spurious saves) for "
                 "~log(target/outlier)/log(decay) epochs regardless of actual performance -- "
                 "observed to take ~70 epochs here. During warmup, checkpoint selection "
                 "uses raw val_loss directly instead.")
    parser.add_argument("--lr-warmup-steps", type=int, default=20,
            help="linearly ramp LR from 1%% of --lr up to --lr over this many OPTIMIZER "
                 "STEPS (not epochs) -- addresses the root cause of the epoch-1 blowup "
                 "(an oversized early update entering the compounding rollout feedback "
                 "loop) rather than just recovering gracefully from it after the fact. "
                 "Set to 0 to disable.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
            help="max gradient norm (torch.nn.utils.clip_grad_norm_). Guards against "
                 "large-dt batches producing oversized gradients despite the zero-init "
                 "forward pass being well-scaled -- the gradient through dz=g*dt still "
                 "carries a factor of dt via the chain rule, so a batch containing your "
                 "largest dt values can still produce a large first gradient. Set to 0 "
                 "to disable.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None,
            help="default: auto-generated from the AE's size/latent-channels/"
                 "stats-weight plus --n-rollout-steps, e.g. "
                 "../../output/lds_checkpoint_pt/64x64-4latent-weight_0p01-rollout_3.pt")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.n_rollout_steps < 1:
        raise ValueError(f"--n-rollout-steps must be >= 1 (got {args.n_rollout_steps})")
    window_length = args.n_rollout_steps + 1

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device: {device}\n")

    # Read config FIRST: needed to resolve --ae-checkpoint from parameters
    # (its grid size must match this script's own --config), before we can
    # even locate the AE checkpoint file to load.
    config = load.read_config(args.config)
    if config.nx != config.ny:
        raise ValueError(f"Autoencoder assumes a square grid, got {config.nx}x{config.ny}")

    # CLI flag wins if explicitly given; otherwise use config.txt's
    # centralized value (single canonical config.txt, no backward-compat
    # fallback needed). Same pattern as train_ae.py.
    if args.ae_stats_weight is None:
        args.ae_stats_weight = config.stats_weight
    if args.min_step is None:
        args.min_step = config.min_step
    if args.min_stdev_phi is None:
        args.min_stdev_phi = config.min_stdev_phi
    print(f"min_step={args.min_step}  min_stdev_phi={args.min_stdev_phi}  "
          f"ae_stats_weight={args.ae_stats_weight}")

    if args.ae_checkpoint is None:
        if args.ae_latent_channels is None:
            raise ValueError(
                "Provide either --ae-checkpoint directly, or --ae-latent-channels "
                "(--ae-stats-weight is no longer required to be explicit -- it now "
                "falls back to config.txt's stats_weight) so the expected path can "
                f"be reconstructed (using this script's own --config grid size, "
                f"{config.nx}x{config.nx})."
            )
        ae_name = ae_checkpoint_name(config.nx, args.ae_latent_channels, args.ae_stats_weight)
        args.ae_checkpoint = Path(f"../../output/ae_checkpoint_pt/{ae_name}.pt")
        print(f"Reconstructed AE checkpoint path: {args.ae_checkpoint}")

    # Load the frozen autoencoder. Only .encoder is ever used below --
    # the decoder is irrelevant to stage 4 training (see docs/README.md's
    # workflow table: stage 4 trains f alone, encoder frozen).
    ae_checkpoint = torch.load(args.ae_checkpoint, map_location=device, weights_only=True)
    ae_config = ae_checkpoint["config"]
    ae = Autoencoder(
        size=ae_config["size"], channels=1,
        base_channels=ae_config["base_channels"], latent_channels=ae_config["latent_channels"],
    ).to(device)
    ae.load_state_dict(ae_checkpoint["model_state"])
    encoder = ae.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen encoder from {args.ae_checkpoint} "
          f"(epoch {ae_checkpoint['epoch']}, val_loss={ae_checkpoint['val_loss']:.6f}, "
          f"latent_channels={ae_config['latent_channels']})\n")

    if config.nx != ae_config["size"]:
        raise ValueError(f"config.txt grid size ({config.nx}) doesn't match the "
                          f"autoencoder checkpoint's size ({ae_config['size']})")

    if args.checkpoint is None:
        # args.ae_stats_weight is always resolved to a real value by this
        # point (CLI, config.txt, or the hardcoded fallback above) --
        # regardless of whether --ae-checkpoint was given directly or
        # reconstructed, so no "unknown" fallback is needed here anymore.
        name = lds_checkpoint_name(ae_config["size"], ae_config["latent_channels"],
                                    args.ae_stats_weight, args.n_rollout_steps)
        args.checkpoint = Path(f"../../output/lds_checkpoint_pt/{name}.pt")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {args.checkpoint}\n")

    run_dirs = complete_run_dirs(config, args.base)
    if not run_dirs:
        raise ValueError("No complete runs found -- check --config/--base paths")

    train_dirs, val_dirs, test_dirs = split_run_dirs(
        run_dirs, args.val_fraction, args.test_fraction, seed=args.seed,
    )
    print(f"{len(run_dirs)} complete runs -> "
          f"{len(train_dirs)} train / {len(val_dirs)} val / {len(test_dirs)} test dirs")

    train_set = MicrostructureEvolutionDataset(
        train_dirs, encoder=encoder, device=device, window_length=window_length,
        min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        encode_batch_size=args.encode_batch_size,
    )
    val_set = MicrostructureEvolutionDataset(
        val_dirs, encoder=encoder, device=device, window_length=window_length,
        min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        encode_batch_size=args.encode_batch_size,
    )
    print(f"{len(train_set)} train windows, {len(val_set)} val windows "
          f"(n_rollout_steps={args.n_rollout_steps}, window_length={window_length})\n")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    f_theta = LatentDynamics(
        latent_channels=ae_config["latent_channels"], n_theta=1,
        hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden_layers,
    ).to(device)

    rollout_loss = RolloutLoss()
    optimizer = torch.optim.Adam(f_theta.parameters(), lr=args.lr)

    lr_scheduler = None
    if args.lr_warmup_steps > 0:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=args.lr_warmup_steps,
        )

    def step(batch, train: bool) -> float:
        z_window, dt_window, theta = batch
        z_window = z_window.to(device, non_blocking=True)
        dt_window = dt_window.to(device, non_blocking=True)
        theta = theta.to(device, non_blocking=True)

        z0 = z_window[:, 0]
        z_true = z_window[:, 1:]  # (B, n_r, C, H, W) -- the true continuation

        # Chained prediction: each step's output feeds into the next
        # (via LatentDynamics.rollout), so errors compound exactly as
        # they would at real inference time -- unlike always resetting
        # to the true z(t) before each individual prediction.
        z_hat_full = f_theta.rollout(z0, dt_window, theta)  # (B, n_r+1, C, H, W)
        z_hat = z_hat_full[:, 1:]  # drop the given z0, keep only predictions

        loss = rollout_loss(z_hat, z_true)

        if train:
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(f_theta.parameters(), args.grad_clip)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        return loss.item()

    best_val_loss = float("inf")
    val_ema = None

    print(f"Starting {args.epochs} epochs (batches of {args.batch_size})...")
    print(f"/{args.epochs:3d}  train    valid    ema")

    for epoch in range(1, args.epochs + 1):
        f_theta.train()
        train_loss = 0.0
        n_train = len(train_set)
        for batch in train_loader:
            bs = batch[0].size(0)
            train_loss += step(batch, train=True) * bs
        train_loss /= n_train

        f_theta.eval()
        val_loss = 0.0
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0)
                val_loss += step(batch, train=False) * bs
        val_loss /= n_val

        # EMA warmup: don't seed/track val_ema from potentially-catastrophic
        # early epochs (rollout training's compounding-error feedback loop
        # can produce extreme early losses -- see --ema-warmup-epochs help).
        # During warmup, fall back to raw val_loss as the selection
        # criterion directly; val_ema starts only once warmup ends, seeded
        # from that epoch's (by-then-reasonable) value.
        if epoch <= args.ema_warmup_epochs:
            criterion = val_loss
        else:
            val_ema = val_loss if val_ema is None else \
                args.val_ema_decay * val_ema + (1 - args.val_ema_decay) * val_loss
            criterion = val_ema

        ema_str = f"{val_ema:.6f}" if val_ema is not None else "  (warmup)"
        msg = f"{epoch:4d}  {train_loss:.6f}  {val_loss:.6f}  {ema_str}"

        if criterion < best_val_loss:
            best_val_loss = criterion
            torch.save({
                "model_state": f_theta.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_loss_ema": val_ema,  # None during warmup -- criterion used raw val_loss then
                "ae_checkpoint": str(args.ae_checkpoint),
                "test_dirs": [str(d) for d in test_dirs],
                "config": {
                    "latent_channels": ae_config["latent_channels"],
                    "n_theta": 1,
                    "hidden_dim": args.hidden_dim,
                    "n_hidden_layers": args.n_hidden_layers,
                },
                "data_config": {
                    "min_step": args.min_step,
                    "min_stdev_phi": args.min_stdev_phi,
                    "window_length": window_length,
                },
            }, args.checkpoint)
            msg += "  -> saved"

        print(msg)


if __name__ == "__main__":
    main()
