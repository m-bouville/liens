"""
Stage 3: train the Latent Dynamics Surrogate (f_theta) with L_rollout
(chained multi-step prediction, errors accumulating as they would at
real inference time -- not L_1step's ground-truth-conditioned single
transitions), on top of a FROZEN, already-trained autoencoder (its
encoder only -- the decoder is not used here at all, only for later
visual checks).

train_lds() is importable -- see main.py for the orchestrated
1 -> 2 -> 3 pipeline. The CLI below is for standalone stage-3 runs.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m training.train_lds --ae-latent-channels 8 --ae-stats-weight 0.01 \
        --size 64 --base ../../datasets --n-rollout-steps 6
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder
from models.latent_dynamics import LatentDynamics
from training.checkpoint_criterion import CheckpointCriterionTracker
from training.datasets import MicrostructureEvolutionDataset, complete_run_dirs, split_run_dirs
from training.losses import RolloutLoss
from utils.naming import ae_checkpoint_name, lds_checkpoint_name
from utils.plots import loss_curve


def train_lds(
    size: int, base_path: Path,
    ae_checkpoint_path: Path | None = None,
    ae_latent_channels: int | None = None, stats_weight: float | None = None,
    epochs: int = 100, batch_size: int = 512, lr: float = 1e-3,
    hidden_dim: int = 256, n_hidden_layers: int = 2,
    val_fraction: float = 0.2, test_fraction: float = 0.1, num_workers: int = 0,
    n_rollout_steps: int = 1, min_step: int | None = None, min_stdev_phi: float | None = None,
    encode_batch_size: int = 256, val_ema_decay: float = 0.7, ema_warmup_epochs: int = 5,
    early_stopping_patience: int | None = None,
    lr_warmup_steps: int = 20, grad_clip: float = 1.0,
    seed: int = 0, checkpoint_path: Path | None = None, device: str | None = None,
    on_checkpoint_saved: Callable[[Path, int], None] | None = None,
    resume_from: Path | None = None,
    log_every_epoch: bool = True,
    step_weights: list[float] | None = None,
    loss_curve_path: Path | None = None,
    one_step_weight: float = 0.0,
) -> Path:
    """
    Stage 3. Returns the path of the best checkpoint saved. Either give
    ae_checkpoint_path directly, or both ae_latent_channels and
    stats_weight (the AE checkpoint's own stats_weight, used only to
    reconstruct its expected filename -- must be given explicitly,
    config.txt no longer provides ML training defaults).

    resume_from: optional previous LDS checkpoint to initialize
    f_theta's weights from -- e.g. curriculum rollout, training first
    with n_rollout_steps=1 (stable, fast) then resuming here with the
    target n_rollout_steps (harder, more directly optimizes the actual
    downstream chained-prediction use case), rather than jumping
    straight to multi-step rollout training from scratch (which
    produced an epoch-1 loss blowup to ~1e15 when tried that way).
    Architecture (latent_channels/hidden_dim/n_hidden_layers) must
    match the checkpoint being resumed from exactly.

    on_checkpoint_saved: optional callback(checkpoint_path, epoch),
    called immediately after EVERY checkpoint save, not just the final
    one -- see train_autoencoder()'s docstring for the rationale.

    log_every_epoch: if False, only prints a line when a checkpoint is
    actually saved (plus the early-stopping/final message) -- most
    useful here specifically, given this stage commonly runs for
    hundreds of epochs and per-epoch console output between
    improvements is mostly noise.

    step_weights: one weight per rollout step (length must equal
    n_rollout_steps), passed straight through to RolloutLoss -- see
    that class's docstring. WORTH READING BEFORE CHOOSING A DIRECTION:
    the design intent recorded there is to weight LATER steps MORE
    (long-term stability is what actually matters for real use), but
    empirically, later steps are also where multi-step rollout training
    instability concentrates (compounding error on the model's own,
    increasingly off-manifold predictions) -- so weighting them more
    heavily can directly worsen the instability it's trying to protect
    against. None (default, uniform weighting) is a safe starting
    point; there's no settled answer here yet on which direction is
    actually better in practice, just this tension to be aware of
    before picking one.

    one_step_weight: adds epsilon*L_1step on top of the (possibly
    step-weighted) rollout loss -- L = L_rollout + one_step_weight*L_1step,
    optimized together, not just L_1step's existing diagnostic role
    (per_step[0], already computed every step for the console/loss-curve
    display regardless of this parameter). Distinct from step_weights
    above: step_weights reweights terms INSIDE the existing rollout sum;
    this adds an INDEPENDENT term on top, mirroring
    compute_stage45_loss's own primary+epsilon*secondary structure.
    Motivated by n_rollout_steps>1 runs' worst spikes looking like
    compounding failures concentrated in step 2/3 specifically -- a
    small, well-behaved 1-step anchor gives a second gradient signal
    that shouldn't blow up the same way during those episodes, without
    changing what's ultimately being optimized for (0.0, the default,
    reproduces the exact previous behavior: L_1step stays diagnostic-only).
    Should be small (epsilon), matching every other primary+epsilon*
    secondary weight in this project -- not yet validated at any
    specific value.
    """
    if n_rollout_steps < 1:
        raise ValueError(f"n_rollout_steps must be >= 1 (got {n_rollout_steps})")
    window_length = n_rollout_steps + 1

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    print(f"device: {device}\n")

    # Same rationale as train_ae.py: config.txt is simulation-only now,
    # these must be passed explicitly by the caller. min_stdev_phi is
    # NOT in this list -- it's genuinely allowed to be None (meaning no
    # stdev-based filtering at all, matching MicrostructureEvolutionDataset's
    # own float|None semantics), unlike these three, which have no
    # meaningful None value at all.
    missing = [name for name, v in [("size", size), ("stats_weight", stats_weight),
                                     ("min_step", min_step)]
               if v is None]
    if missing:
        raise ValueError(f"train_lds() requires {', '.join(missing)} to be given explicitly "
                          f"-- config.txt no longer provides ML training defaults.")
    print(f"min_step={min_step}  min_stdev_phi={min_stdev_phi}  stats_weight={stats_weight}")
    print(f"n_rollout_steps={n_rollout_steps}  one_step_weight={one_step_weight}")
    print(f"grad_clip={grad_clip}  lr_warmup_steps={lr_warmup_steps}")

    if ae_checkpoint_path is None:
        if ae_latent_channels is None:
            raise ValueError(
                "Provide either ae_checkpoint_path directly, or ae_latent_channels "
                "so the expected path can be reconstructed."
            )
        ae_name = ae_checkpoint_name(size, ae_latent_channels, stats_weight)
        ae_checkpoint_path = Path(f"../checkpoints/stage2/{ae_name}.pt")
        print(f"Reconstructed AE checkpoint path: {ae_checkpoint_path}")

    # Load the frozen autoencoder. Only .encoder is ever used below --
    # the decoder is irrelevant to stage 3 training.
    ae_checkpoint = torch.load(ae_checkpoint_path, map_location=device, weights_only=True)
    ae_config = ae_checkpoint["config"]
    ae = Autoencoder(size=ae_config["size"], channels=1, base_channels=ae_config["base_channels"],
                      latent_channels=ae_config["latent_channels"]).to(device)
    ae.load_state_dict(ae_checkpoint["model_state"])
    encoder = ae.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen encoder from {ae_checkpoint_path} "
          f"(epoch {ae_checkpoint['epoch']}, val_loss={ae_checkpoint['val_loss']:.6f}, "
          f"latent_channels={ae_config['latent_channels']})\n")

    # Still a valuable sanity check -- just against the size YOU gave,
    # not config.txt (which may describe an unrelated sweep entirely).
    if size != ae_config["size"]:
        raise ValueError(
            f"size given ({size}) doesn't match the autoencoder checkpoint's own "
            f"size ({ae_config['size']}) -- double check which checkpoint/size you meant."
        )

    if checkpoint_path is None:
        name = lds_checkpoint_name(ae_config["size"], ae_config["latent_channels"],
                                    stats_weight, n_rollout_steps)
        checkpoint_path = Path(f"../checkpoints/stage3/{name}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {checkpoint_path}\n")

    if loss_curve_path is None:
        name = lds_checkpoint_name(ae_config["size"], ae_config["latent_channels"],
                                    stats_weight, n_rollout_steps)
        loss_curve_path = Path(f"../../output/stage3/{name}-loss_curve.png")

    epoch_history: list[int] = []
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_so_far_history: list[float] = []
    train_1step_history: list[float] = []
    val_1step_history: list[float] = []

    run_dirs = complete_run_dirs(base_path, size, size)
    if not run_dirs:
        raise ValueError(f"No complete runs found under {base_path}/{size}x{size} -- "
                          f"check base_path/size, or that metadata.txt exists there")

    train_dirs, val_dirs, test_dirs = split_run_dirs(run_dirs, val_fraction, test_fraction, seed=seed)
    print(f"{len(run_dirs)} complete runs -> "
          f"{len(train_dirs)} train / {len(val_dirs)} val / {len(test_dirs)} test dirs")

    train_set = MicrostructureEvolutionDataset(
        train_dirs, encoder=encoder, device=device, window_length=window_length,
        min_step=min_step, min_stdev_phi=min_stdev_phi, encode_batch_size=encode_batch_size,
    )
    val_set = MicrostructureEvolutionDataset(
        val_dirs, encoder=encoder, device=device, window_length=window_length,
        min_step=min_step, min_stdev_phi=min_stdev_phi, encode_batch_size=encode_batch_size,
    )
    print(f"{len(train_set)} train windows, {len(val_set)} val windows "
          f"(n_rollout_steps={n_rollout_steps}, window_length={window_length})\n")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")

    f_theta = LatentDynamics(latent_channels=ae_config["latent_channels"], n_theta=1,
                              hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers).to(device)

    if resume_from is not None:
        # Curriculum rollout: train with n_rollout_steps=1 first (stable,
        # fast, avoids the epoch-1 loss blowup a from-scratch jump straight
        # to multi-step rollout produced), then resume here with the
        # target n_rollout_steps. Architecture must match exactly --
        # loading into a differently-shaped f_theta would either error
        # confusingly deep in load_state_dict or silently mismatch.
        prev_lds = torch.load(resume_from, map_location=device, weights_only=True)
        prev_config = prev_lds["config"]
        mismatch = [(k, prev_config[k], v) for k, v in
                    [("latent_channels", ae_config["latent_channels"]),
                     ("hidden_dim", hidden_dim), ("n_hidden_layers", n_hidden_layers)]
                    if prev_config[k] != v]
        if mismatch:
            raise ValueError(f"{resume_from}'s architecture doesn't match the requested one: "
                              + ", ".join(f"{k}={old} (checkpoint) vs {new} (requested)"
                                          for k, old, new in mismatch))
        f_theta.load_state_dict(prev_lds["model_state"])
        prev_n_rollout = prev_lds.get("data_config", {}).get("n_rollout_steps")
        print(f"Resumed f_theta from {resume_from} (epoch {prev_lds['epoch']}, "
              f"val_loss={prev_lds['val_loss']:.6f}, trained at n_rollout_steps="
              f"{prev_n_rollout if prev_n_rollout is not None else '?'})\n")
        if prev_n_rollout is not None and n_rollout_steps <= prev_n_rollout:
            print(f"WARNING: resuming from a checkpoint trained at n_rollout_steps="
                  f"{prev_n_rollout}, but this run asks for n_rollout_steps="
                  f"{n_rollout_steps} -- not larger, so this isn't the usual curriculum "
                  f"direction (easy -> hard). Continuing anyway, but double-check this "
                  f"is intentional.\n")

    step_weights_tensor = torch.tensor(step_weights, dtype=torch.float32, device=device) \
        if step_weights is not None else None
    if step_weights_tensor is not None and step_weights_tensor.shape != (n_rollout_steps,):
        raise ValueError(f"step_weights has {len(step_weights)} entries, but "
                          f"n_rollout_steps={n_rollout_steps} -- need exactly one weight per step.")
    rollout_loss = RolloutLoss(step_weights=step_weights_tensor)
    optimizer = torch.optim.Adam(f_theta.parameters(), lr=lr)

    lr_scheduler = None
    if lr_warmup_steps > 0:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=lr_warmup_steps,
        )

    def step(batch, train: bool) -> tuple[float, float]:
        z_window, dt_window, theta = batch
        z_window = z_window.to(device, non_blocking=True)
        dt_window = dt_window.to(device, non_blocking=True)
        theta = theta.to(device, non_blocking=True)

        z0 = z_window[:, 0]
        z_true = z_window[:, 1:]

        z_hat_full = f_theta.rollout(z0, dt_window, theta)
        z_hat = z_hat_full[:, 1:]

        # per_step[0] is L_1step -- the loss restricted to just the first
        # predicted step, directly comparable to a model trained with
        # n_rollout_steps=1 (see RolloutLoss.forward()'s docstring: this
        # is mathematically identical to computing L_1step independently
        # on the same data, not an approximation).
        loss, per_step = rollout_loss(z_hat, z_true, return_per_step=True)
        l_1step = per_step[0]
        # total is what's actually optimized -- see one_step_weight's
        # docstring. At the default one_step_weight=0.0, total is
        # exactly loss (l_1step contributes nothing, backward()
        # reproduces the prior, rollout-only behavior precisely).
        total = loss + one_step_weight * l_1step

        if train:
            optimizer.zero_grad()
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(f_theta.parameters(), grad_clip)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        return total.item(), l_1step.item()

    tracker = CheckpointCriterionTracker(ema_warmup_epochs=ema_warmup_epochs,
                                          val_ema_decay=val_ema_decay)
    epochs_since_improvement = 0

    show_1step = n_rollout_steps > 1  # at n=1, L_1step == L_rollout always -- redundant to show

    print(f"Starting {epochs} epochs (batches of {batch_size})...")
    if show_1step:
        print(f"/{epochs:3d}  train  (1step)   valid  (1step)     ema")
    else:
        print(f"/{epochs:3d}  train    valid      ema")

    for epoch in range(1, epochs + 1):
        f_theta.train()
        train_loss = train_1step = 0.0
        n_train = len(train_set)
        for batch in train_loader:
            bs = batch[0].size(0)
            loss, l_1step = step(batch, train=True)
            train_loss += loss * bs
            train_1step += l_1step * bs
        train_loss /= n_train
        train_1step /= n_train

        f_theta.eval()
        val_loss = val_1step = 0.0
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0)
                loss, l_1step = step(batch, train=False)
                val_loss += loss * bs
                val_1step += l_1step * bs
        val_loss /= n_val
        val_1step /= n_val

        criterion, saved_this_epoch = tracker.update(epoch, val_loss)

        epoch_history.append(epoch)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        best_so_far_history.append(tracker.best_val_loss)
        if show_1step:
            train_1step_history.append(train_1step)
            val_1step_history.append(val_1step)
        loss_curve(
            epoch_history, train_loss_history, val_loss_history, best_so_far_history,
            loss_curve_path, title="Stage 3 loss",
            secondary_train=train_1step_history if show_1step else None,
            secondary_val=val_1step_history if show_1step else None,
            secondary_label="1step",
        )

        ema_str = f"{tracker.val_ema:.6f}" if tracker.val_ema is not None else "  (warmup)"
        if show_1step:
            msg = (f"{epoch:4d} {train_loss:7.3f} ({train_1step:6.3f}),"
                   f"{val_loss:7.3f} ({val_1step:6.3f}) |{ema_str:>9}")
        else:
            msg = f"{epoch:4d} {train_loss:7.3f},{val_loss:7.3f} |{ema_str:>9}"

        if saved_this_epoch:
            epochs_since_improvement = 0
            torch.save({
                "model_state": f_theta.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_loss_ema": tracker.val_ema,
                "ae_checkpoint": str(Path(ae_checkpoint_path).resolve()),
                "test_dirs": [str(Path(d).resolve()) for d in test_dirs],
                "config": {
                    "latent_channels": ae_config["latent_channels"], "n_theta": 1,
                    "hidden_dim": hidden_dim, "n_hidden_layers": n_hidden_layers,
                },
                "data_config": {
                    "min_step": min_step, "min_stdev_phi": min_stdev_phi,
                    "window_length": window_length, "n_rollout_steps": n_rollout_steps,
                },
            }, checkpoint_path)
            msg += "  -> saved"
            if on_checkpoint_saved is not None:
                on_checkpoint_saved(checkpoint_path, epoch)
        else:
            epochs_since_improvement += 1

        if log_every_epoch or saved_this_epoch:
            print(msg)

        # Only counts post-warmup, since raw val_loss during warmup can be
        # wildly noisy by design (see ema_warmup_epochs) -- counting those
        # epochs toward patience could trigger a spurious early stop before
        # training has even stabilized.
        if (early_stopping_patience is not None and epoch > ema_warmup_epochs
                and epochs_since_improvement >= early_stopping_patience):
            print(f"Early stopping at epoch {epoch}: no improvement for "
                  f"{early_stopping_patience} epochs")
            break

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-latent-channels", type=int, default=None)
    parser.add_argument("--ae-stats-weight", type=float, default=None, dest="stats_weight",
                         help="the AE checkpoint's own stats_weight (used only to locate its "
                              "expected filename) -- named --ae-stats-weight here since it's "
                              "paired with --ae-latent-channels, but stored as args.stats_weight "
                              "to match train_lds()'s actual parameter name")
    parser.add_argument("--ae-checkpoint", type=Path, default=None)
    parser.add_argument("--size", type=int, required=True,
                         help="grid size (square only) -- locates base/<size>x<size>/, "
                              "reading ITS OWN metadata.txt (not config.txt)")
    parser.add_argument("--base", type=Path, default=Path("../../datasets"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-rollout-steps", type=int, default=1)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--min-stdev-phi", type=float, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--val-ema-decay", type=float, default=0.7)
    parser.add_argument("--ema-warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None,
                         help="previous LDS checkpoint to initialize f_theta from, e.g. for "
                              "curriculum rollout (train n_rollout_steps=1 first, then resume "
                              "here with the target n_rollout_steps)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quiet", action="store_true",
                         help="only print a line when a checkpoint is actually saved, instead "
                              "of every epoch -- this stage commonly runs for hundreds of "
                              "epochs, so this cuts console output down to what matters")
    args = parser.parse_args()

    train_lds(
        size=args.size, base_path=args.base,
        ae_checkpoint_path=args.ae_checkpoint, ae_latent_channels=args.ae_latent_channels,
        stats_weight=args.stats_weight,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden_layers,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        num_workers=args.num_workers, n_rollout_steps=args.n_rollout_steps,
        min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        encode_batch_size=args.encode_batch_size, val_ema_decay=args.val_ema_decay,
        ema_warmup_epochs=args.ema_warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        lr_warmup_steps=args.lr_warmup_steps, grad_clip=args.grad_clip,
        seed=args.seed, checkpoint_path=args.checkpoint, device=args.device,
        resume_from=args.resume_from, log_every_epoch=not args.quiet,
    )


if __name__ == "__main__":
    main()
