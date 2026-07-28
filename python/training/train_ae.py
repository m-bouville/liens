"""
Stage 1 (train_autoencoder) and stage 2 (train_stage2) of the LIENS
pipeline. Both are importable functions, not just CLI scripts -- see
main.py for the orchestrated 1 -> 2 -> 3 pipeline. train_ae.py's CLI
below covers stage 1 only; stage 2 is driven by main.py, since it always
resumes from a stage-1 checkpoint (there's no "stage 2 from scratch").

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m training.train_ae --size 64 --base ../datasets
"""

import argparse
from collections.abc import Callable
from pathlib import Path
import gc

import torch
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder, EncoderDecoderPair, MultiStreamAutoencoder
from models.constants import LATENT_SPATIAL_SIZE
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_streams import (
    DEFAULT_STREAM_NAME, LatentStreamMode,
    cross_check_stream_configs_against_state_dict, resolve_stream_configs_from_checkpoint_config,
)
from training.checkpoint_criterion import CheckpointCriterionTracker
from training.extend_encoder import extend_state_checkpoint_with_deriv_stream
from training.datasets import MicrostructureEvolutionDataset, MicrostructureSnapshotDataset, \
                               complete_run_dirs, split_run_dirs
from training.losses import ReconLoss, StatsLoss
from training.stats_head import StatsHead
from utils.naming import ae_checkpoint_name
from utils.plots import loss_curve
from evaluation.check_interpolation import check_interpolation
from evaluation.check_perturbation import check_perturbation

# GENERAL POLICY (matches training/train_refinement.py's own
# _PYTHON_ROOT): every checkpoint/output/dataset path is built from
# THIS anchor, never from a bare relative string like "../output/...".
# Relative strings resolve against the process's CWD at invocation
# time, which silently differs across bare CLI, `python -m`, and being
# imported and called from another module (e.g. main.py calling
# train_autoencoder) -- exactly the recurring "output ended up in the
# wrong place" bug hit repeatedly on this project. Path(__file__) is
# anchored to THIS FILE's own on-disk location instead, which is
# invariant regardless of how/from-where the process was launched.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/training/train_ae.py -> python/


class _RunningStats:
    """
    Incremental mean/std tracker -- accumulates sum, sum-of-squares,
    and count across many tensors without ever storing one, so this
    can track a quantity's actual scale across an entire epoch's worth
    of batches cheaply. Exists specifically to debug z0/z1 scale
    mismatches with real, measured numbers instead of theoretical
    reasoning about what their scale "should" be.
    """
    def __init__(self):
        self.sum = 0.0
        self.sumsq = 0.0
        self.count = 0

    def update(self, x: torch.Tensor) -> None:
        x = x.detach()
        self.sum += x.sum().item()
        self.sumsq += (x * x).sum().item()
        self.count += x.numel()

    def mean_std(self) -> tuple[float, float]:
        if self.count == 0:
            return float("nan"), float("nan")
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)  # clamp: float roundoff can make this tiny-negative
        return mean, var ** 0.5


def train_autoencoder(
    size: int, base_path: Path,
    epochs: int = 100, batch_size: int = 64, lr: float = 1e-3,
    base_channels: int = 32, latent_channels: int = 8,
    latent_spatial_size: int = LATENT_SPATIAL_SIZE,
    val_fraction: float = 0.2, test_fraction: float = 0.1, num_workers: int = 4,
    augment: bool = True,
    min_step: int | None = None, min_stdev_phi: float | None = None,
    min_passing_steps: int | None = None,
    stat_names: list[str] | None = None, stats0_weight: float | None = None,
    recon0_scale: float = 1.0, stats0_scale: float = 1.0,
    val_ema_decay: float = 0.7, early_stopping_patience: int | None = None,
    seed: int = 0, checkpoint_path: Path | None = None,
    resume_from: Path | None = None, device: str | None = None,
    on_checkpoint_saved: Callable[[Path, int], None] | None = None,
    log_every_epoch: bool = True,
    loss_curve_path: Path | None = None,
) -> Path:
    """
    Stage 1a: train a SINGLE-stream AE (state only) on individual
    snapshots with L_recon0/recon0_scale, and (if stats0_weight > 0)
    stats0_weight*L_stats0/stats0_scale via stats_head.py. Returns the
    path of the best checkpoint saved (selected on an EMA of val_total,
    see val_ema_decay).

    recon0_scale/stats0_scale are magnitude-normalization terms, NOT
    importance weights -- L_recon0 and L_stats0 sit at genuinely
    different natural orders of magnitude, and dividing each by its own
    scale (default 1.0, a no-op) lets stats0_weight express relative
    IMPORTANCE cleanly, without also having to silently absorb a
    magnitude correction it was never meant to carry. Live in the
    params file's shared preamble (same value across every stage that
    recognizes them), not re-specified per stage the way *_weight is.

    Single-stream, deliberately -- this used to also support an
    alternating C0/C1 multi-stream mode (train a 'deriv' stream here
    too, batch-by-batch alternation with C0); see
    train_ae_pre_stage1b_archive.py for that version. It was replaced
    by a dedicated stage 1b (since removed -- see
    training/extend_encoder.py's own module docstring for the full
    history: stage 1b's own training loop had been inert since it
    started running at epochs=0, and its one genuinely load-bearing
    piece, extending the encoder with a fresh deriv bottleneck, now
    happens in memory at the start of train_stage2() itself) once an
    isolation test showed C0-alone here already predicts state
    correctly, while the alternating version's z0/z1 latent scale
    never stabilized across a real run -- training C0 with ZERO
    interference from C1 is exactly what this function now does, by
    construction, not by argument. Every current caller resumes
    directly into stage 2 for the deriv stream; this function no
    longer knows that stream exists at all.

    log_every_epoch: if False, only prints a line when a checkpoint is
    actually saved (plus the early-stopping/final message) -- useful for
    long runs (esp. stage 3's hundreds of epochs) where per-epoch
    console output is mostly noise between improvements.

    on_checkpoint_saved: optional callback(checkpoint_path, epoch),
    called immediately after EVERY checkpoint save (not just the final
    one) -- e.g. so a caller can update a parameter registry as
    training progresses, rather than only after this function returns,
    which would otherwise mean a crashed/interrupted run leaves a valid
    checkpoint+log on disk with no registry entry at all.

    size: grid size (square only) -- this is what locates the dataset
    (base_path/<size>x<size>/), read from ITS OWN metadata.txt (not
    config.txt, which may describe an unrelated sweep -- see
    complete_run_dirs).

    resume_from: optional checkpoint to initialize model_state/
    stats_head_state from before training starts (e.g. to continue stage
    1a training itself with more epochs -- NOT how stage 2 works, a
    separate function with a different loss/data structure).

    early_stopping_patience: stop once val_ema hasn't improved for this
    many consecutive epochs, instead of always running the full `epochs`
    budget -- a data-driven stopping signal rather than a guessed epoch
    count for "stage 1a is done".
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    # config.txt is simulation-sweep-only now (Nx/Ny/dt/temperatures/...) --
    # min_step/stats0_weight are ML training parameters and must be passed
    # explicitly by the caller (e.g. from a stage-parameters file via
    # main.py), not silently inferred from config.txt. min_stdev_phi is
    # NOT required to be non-None -- it's genuinely allowed to be None
    # (no stdev-based filtering at all), unlike min_step/stats0_weight,
    # which have no meaningful None value (stats0_weight is compared
    # against a threshold just below; min_step feeds a plain int
    # comparison at the dataset level).
    missing = [name for name, v in [("min_step", min_step),
                                     ("stats0_weight", stats0_weight)] if v is None]
    if missing:
        raise ValueError(f"train_autoencoder() requires {', '.join(missing)} to be given "
                          f"explicitly -- config.txt no longer provides ML training defaults.")
    print(f"min_step={min_step}  min_stdev_phi={min_stdev_phi}  min_passing_steps={min_passing_steps}  "
          f"stats0_weight={stats0_weight}")

    include_stats = stats0_weight > 1e-6

    if checkpoint_path is None:
        # ae_checkpoint_name's own "stats_weight" argument name/the
        # checkpoint config's own saved "stats_weight" key are UNCHANGED
        # -- an internal, cross-cutting persistence format read by many
        # downstream consumers (every evaluation script, later stages'
        # own ancestor lookups), separate from this function's own,
        # renamed parameter name. Same precedent as ae_latent_channels
        # already not matching the checkpoint's own "latent_channels" key.
        name = ae_checkpoint_name(size, latent_channels, stats0_weight)
        checkpoint_path = _PYTHON_ROOT / "checkpoints" / "stage1" / f"{name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {checkpoint_path}")

    if loss_curve_path is None:
        name = ae_checkpoint_name(size, latent_channels, stats0_weight)
        loss_curve_path = _PYTHON_ROOT.parent / "output" / "stage1" / f"{name}-loss_curve.png"

    epoch_history: list[int] = []
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_so_far_history: list[float] = []

    run_dirs = complete_run_dirs(base_path, size, size)
    if not run_dirs:
        raise ValueError(f"No complete runs found under {base_path}/{size}x{size} -- "
                          f"check base_path/size, or that metadata.txt exists there")

    # Split at the DIRECTORY level, not by frame: two frames from the same
    # run, even at unrelated timesteps, are snapshots of one continuous
    # evolution and can look very similar, so splitting below the
    # directory level risks leaking correlated frames across train/val/test.
    train_dirs, val_dirs, test_dirs = split_run_dirs(run_dirs, val_fraction, test_fraction, seed=seed)
    print(f"{len(run_dirs)} complete runs ({size} x {size}) -> "
          f"{len(train_dirs)} train / {len(val_dirs)} val / {len(test_dirs)} test dirs")

    if epochs == 0:
        # Ablation mode: no training happens (see the epoch loop
        # below), so train_set/train_loader would never be touched.
        # MicrostructureSnapshotDataset's own construction is cheaper
        # than MicrostructureEvolutionDataset's (no min_std_deriv --
        # nothing here needs to read snapshot PAIRS to compute a
        # derivative), but cache_in_memory=True still means reading
        # every train snapshot into memory upfront, for a dataset
        # that's then never iterated -- skipped entirely instead.
        # stat_names normally locks to whatever train_set itself
        # auto-detected -- with train_set skipped, val_set's own
        # auto-detection is used directly instead, since it's the
        # only dataset being built at all in this case.
        train_set = train_loader = None
        val_set = MicrostructureSnapshotDataset(
            val_dirs, cache_in_memory=True, augment=False,
            min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
            include_stats=include_stats, stat_names=stat_names,
        )
        print(f"train_set: skipped (epochs=0 ablation -- never iterated over), "
              f"{val_set.n_base_samples} (val, unaugmented)")
    else:
        train_set = MicrostructureSnapshotDataset(
            train_dirs, cache_in_memory=True, augment=augment,
            min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
            include_stats=include_stats, stat_names=stat_names,
        )
        # Lock val's stat_names to whatever train resolved (auto-detection is
        # order-dependent; without this, val could silently end up checked
        # against a different run's schema than train was).
        val_stat_names = train_set.stat_names if include_stats else None
        val_set = MicrostructureSnapshotDataset(
            val_dirs, cache_in_memory=True, augment=False,
            min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
            include_stats=include_stats, stat_names=val_stat_names,
        )
        if augment:
            print(f"{train_set.n_base_samples} base snapshots (train) -> {len(train_set)} after "
                  f"augmentation, {val_set.n_base_samples} (val, unaugmented)")
        else:
            print(f"{len(train_set)} snapshots (train, augmentation OFF -- faster, less diverse), "
                  f"{val_set.n_base_samples} (val, unaugmented)")

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            persistent_workers=num_workers > 0, pin_memory=device.type == "cuda",
        )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0, pin_memory=device.type == "cuda",
    )

    recon_stream_name = DEFAULT_STREAM_NAME
    ae = Autoencoder(size=size, channels=1, base_channels=base_channels,
                      latent_channels=latent_channels,
                      latent_spatial_size=latent_spatial_size).to(device)

    recon_loss = ReconLoss()
    params = list(ae.parameters())

    stats_head = None
    stats_loss_fn = None
    if include_stats:
        # train_set is None in epochs=0 ablation mode (skipped entirely,
        # see above) -- val_set is the only dataset built in that case,
        # so it's the source for stat_names/normalization instead.
        stats_source = train_set if train_set is not None else val_set
        stats_head = StatsHead(latent_channels=latent_channels, stat_names=stats_source.stat_names,
                                latent_spatial=latent_spatial_size).to(device)
        mean, std = stats_source.stats_normalization()
        stats_loss_fn = StatsLoss(mean.to(device), std.to(device), stat_names=stats_source.stat_names)
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
        (z0_train_stats if train else z0_val_stats).update(z)
        recon0 = recon_loss(x_recon, x)

        if include_stats:
            stats_pred = stats_head(z)
            stats0 = stats_loss_fn(stats_pred, stats_target)
            total = recon0 / recon0_scale + stats0_weight * stats0 / stats0_scale
        else:
            stats0 = torch.tensor(0.0)
            total = recon0 / recon0_scale

        if train:
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        return total.item(), recon0.item(), stats0.item()

    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=val_ema_decay)
    epochs_since_improvement = 0

    print(f"Starting {epochs} epochs (early_stopping_patience: "
          f"{early_stopping_patience}, batches of {batch_size})...")
    heading = f"/{epochs:3d} "
    heading += (f"train = recon0/{recon0_scale} +{stats0_weight:.3g}*stats0/{stats0_scale} | "
                f"valid = ...  | ema") if include_stats else f"train = recon0/{recon0_scale} | valid | ema"
    print(heading)

    for epoch in range(0 if epochs == 0 else 1, epochs + 1):
        # Fresh each epoch -- see step() for where these get updated,
        # and the epoch-end print below for where they're reported.
        # Kept even without a second (deriv) stream to compare against
        # -- monitoring z's own scale over training is a cheap, useful,
        # general-purpose diagnostic on its own, not something that
        # only mattered for the alternation this function no longer
        # does.
        z0_train_stats, z0_val_stats = _RunningStats(), _RunningStats()

        ae.train()
        if stats_head is not None:
            stats_head.train()

        train_total = train_recon0 = train_stats0 = 0.0
        n_train = 0
        if epoch > 0:
            for batch in train_loader:
                bs = batch[0].size(0) if include_stats else batch.size(0)
                total, recon0, stats0 = step(batch, train=True)
                train_total += total * bs
                train_recon0 += recon0 * bs
                train_stats0 += stats0 * bs
                n_train += bs
            train_total /= n_train
            train_recon0 /= n_train
            train_stats0 /= n_train
        else:
            # epoch 0 (epochs=0 ablation only): no training at all --
            # NaN honestly reflects that these metrics don't apply this
            # "epoch" (n_train stays 0, so dividing would also fail),
            # rather than a misleading 0.0.
            train_total = train_recon0 = train_stats0 = float("nan")

        ae.eval()
        if stats_head is not None:
            stats_head.eval()
        val_total = val_recon0 = val_stats0 = 0.0
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0) if include_stats else batch.size(0)
                total, recon0, stats0 = step(batch, train=False)
                val_total += total * bs
                val_recon0 += recon0 * bs
                val_stats0 += stats0 * bs
        val_total /= n_val
        val_recon0 /= n_val
        val_stats0 /= n_val

        _, saved_this_epoch = tracker.update(epoch, val_total)
        val_ema = tracker.val_ema
        val_ema_str = f"{val_ema:7.4f}" if val_ema is not None else "(warmup)"

        epoch_history.append(epoch)
        train_loss_history.append(train_total)
        val_loss_history.append(val_total)
        best_so_far_history.append(tracker.best_val_loss)
        loss_curve(
            epoch_history, train_loss_history, val_loss_history, best_so_far_history,
            loss_curve_path, title="Stage 1 loss",
        )

        msg = f"{epoch:4d}|"
        if include_stats:
            msg += (f"{train_total:7.4f} ={train_recon0/recon0_scale:7.4f} "
                    f"+{stats0_weight*train_stats0/stats0_scale:7.4f} |"
                    f"{val_total:7.4f} ={val_recon0/recon0_scale:7.4f} "
                    f"+{stats0_weight*val_stats0/stats0_scale:7.4f} |"
                    f"{val_ema_str}")
        else:
            msg += f"{train_total:7.4f} |{val_total:7.4f}  {val_ema_str}"

        if saved_this_epoch:
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
                    "size": size, "base_channels": base_channels,
                    "latent_channels": latent_channels,
                    "latent_spatial_size": latent_spatial_size,
                    "stats_weight": stats0_weight,
                    # Plain dicts/strings, not the LatentStreamConfig/
                    # LatentStreamMode objects themselves -- this
                    # project loads every checkpoint with
                    # weights_only=True (see the project's own
                    # torch.load convention), which only allow-lists a
                    # fixed set of safe types; custom dataclasses/Enums
                    # aren't in it. Still written as a single-entry
                    # stream_configs (not a bare latent_channels/
                    # latent_spatial_size pair) so every downstream
                    # reader (checkpoint_components.py,
                    # resolve_stream_configs_from_checkpoint_config,
                    # train_stage2, every evaluation script) keeps
                    # working from ONE convention regardless of which
                    # stage produced a checkpoint.
                    "stream_configs": {
                        recon_stream_name: {"channels": latent_channels,
                                             "spatial_size": latent_spatial_size,
                                             "mode": LatentStreamMode.AUTOENCODER.value}
                    },
                    "recon_stream_name": recon_stream_name,
                },
                "stats_config": {
                    "stat_names": stats_source.stat_names,
                    "stats_mean": stats_loss_fn.mean.cpu() if stats_loss_fn is not None else None,
                    "stats_std": stats_loss_fn.std.cpu() if stats_loss_fn is not None else None,
                } if include_stats else None,
            }
            torch.save(checkpoint, checkpoint_path)
            msg += "  -> saved"
            if on_checkpoint_saved is not None:
                on_checkpoint_saved(checkpoint_path, epoch)
        else:
            epochs_since_improvement += 1

        if log_every_epoch or saved_this_epoch:
            print(msg)
            z0_train_mean, z0_train_std = z0_train_stats.mean_std()
            z0_val_mean, z0_val_std = z0_val_stats.mean_std()
            print(f"      z0: train mean={z0_train_mean:+.4e} std={z0_train_std:.4e} | "
                  f"val mean={z0_val_mean:+.4e} std={z0_val_std:.4e}")

        if early_stopping_patience is not None and epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}: no improvement for "
                  f"{early_stopping_patience} epochs")
            break

    return checkpoint_path


def freeze_outer_layers(ae: Autoencoder | EncoderDecoderPair | MultiStreamAutoencoder,
                         n_frozen_stages: int) -> list[torch.nn.Module]:
    """
    Freezes the OUTERMOST n_frozen_stages layers on each side (closest to
    real space, farthest from the latent bottleneck): the encoder's
    FIRST n_frozen_stages DownBlocks, and the decoder's LAST
    n_frozen_stages UpBlocks plus its final output_conv. Layers closest
    to the latent bottleneck (encoder's bottleneck 1x1 conv, decoder's
    unbottleneck 1x1 conv, and any remaining inner DownBlocks/UpBlocks)
    stay trainable, since that's where stage 2 needs room to reshape the
    latent geometry.

    NOTE, corrected after checking actual parameter counts: outer layers
    are NOT the largest in this network, despite operating at the
    largest spatial resolution (most FLOPs-expensive to run forward).
    Conv parameter count depends only on channel counts (in_channels *
    out_channels * kernel_size^2), not spatial size -- and channels
    DOUBLE going inward (e.g. 32 -> 64 -> 128), so the outermost stage
    has the FEWEST channels, hence the fewest parameters, while the
    deepest/innermost stage alone can hold the majority of the network's
    weights. Freezing outer stages therefore does NOT meaningfully speed
    up training via reduced optimizer work -- for a shallow network
    (few stages total), it barely reduces trainable parameter count at
    all unless nearly every stage is frozen, which would leave stage 2
    almost nothing to actually work with. Treat n_frozen_stages as a
    regularization/degrees-of-freedom knob, not a speed optimization.

    IMPORTANT CAVEAT, not just a footnote: this reduces the degrees of
    freedom available for the encoder/decoder to drift together, and
    makes such drift much less likely to be found by gradient descent
    (fewer trainable parameters, starting from a good stage-1
    initialization) -- but it is NOT a structural guarantee against the
    scale-collapse failure mode observed empirically. bottleneck and
    unbottleneck (both plain 1x1 convs, i.e. arbitrary linear maps) sit
    immediately adjacent to z on each side and stay trainable here --
    on their own, they are SUFFICIENT to implement an arbitrary
    rescaling of z (one scales up, the other scales back down) that
    L_recon cannot detect, regardless of how many other layers are
    frozen. Freezing narrows the search space; it doesn't close this
    loophole structurally. See compute_weight_drift() for a diagnostic
    that catches it if it happens anyway.

    Returns the frozen submodules. requires_grad_(False) alone stops
    gradient-based updates, but NOT BatchNorm's running_mean/running_var
    -- those are buffers, updated via a forward-pass EMA every time the
    module runs in train() mode, entirely independent of requires_grad.
    Since the training loop calls ae.train() every epoch (recursively
    setting EVERY submodule, frozen or not, back to train mode), callers
    must re-apply .eval() to exactly this returned list right after each
    ae.train() call, or "frozen" blocks with BatchNorm will keep
    drifting via their running stats even though their learnable weights
    are correctly held fixed.

    Freezes EVERY decoder found (ae.decoders, if the container has more
    than one -- e.g. a stage-1b-derived checkpoint's own D0/D1), not
    just one -- symmetric treatment for every decode pathway, not just
    whichever happens to be named "shared".
    """
    frozen_modules: list[torch.nn.Module] = []
    if n_frozen_stages <= 0:
        return frozen_modules
    # MultiStreamAutoencoder doesn't expose .encoder/.decoder directly
    # (only .encoders["shared"]/.decoders[...], see its own docstring
    # on why) -- Autoencoder/EncoderDecoderPair still do.
    encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
    decoders = [ae.decoder] if hasattr(ae, "decoder") else list(ae.decoders.values())
    for block in encoder.down_blocks[:n_frozen_stages]:
        for p in block.parameters():
            p.requires_grad_(False)
        frozen_modules.append(block)
    for decoder in decoders:
        for block in decoder.up_blocks[-n_frozen_stages:]:
            for p in block.parameters():
                p.requires_grad_(False)
            frozen_modules.append(block)
        for p in decoder.output_conv.parameters():
            p.requires_grad_(False)
        frozen_modules.append(decoder.output_conv)
    return frozen_modules


def _param_group(key: str) -> str:
    """Groups a state_dict/named_parameters/named_buffers key by its
    containing block. Keys come from MultiStreamAutoencoder specifically
    (the only kind of model train_stage2 ever builds), e.g.
    'encoders.shared.down_blocks.0.conv1.weight' ->
    'encoders.shared.down_blocks.0', 'decoders.shared.output_conv.weight'
    -> 'decoders.shared.output_conv', 'pathways.deriv.log_output_scale'
    -> 'pathways.deriv.log_output_scale' (its own group -- a single
    scalar, not part of a larger block to group further)."""
    parts = key.split(".")
    if len(parts) >= 4 and parts[2] in ("down_blocks", "up_blocks"):
        return ".".join(parts[:4])
    return ".".join(parts[:3])


def _drift_by_block(initial: dict, final: dict) -> dict[str, float]:
    totals: dict[str, float] = {}
    for key in initial:
        group = _param_group(key)
        diff_sq = (final[key].float() - initial[key].float()).pow(2).sum().item()
        totals[group] = totals.get(group, 0.0) + diff_sq
    return {group: total**0.5 for group, total in totals.items()}


def compute_weight_drift(
    initial_params: dict, initial_buffers: dict, final_params: dict, final_buffers: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Per-block L2 norm of the change in learnable PARAMETERS and,
    SEPARATELY, in BUFFERS (e.g. BatchNorm's running_mean/running_var)
    between initial and final state.

    Kept separate deliberately: parameter drift in a frozen block should
    be EXACTLY 0 (a red flag if it isn't -- something is wrong with the
    freeze itself). Buffer drift in a frozen block should ALSO be ~0 once
    freeze_outer_layers()'s returned modules are kept in .eval() mode
    every epoch (see that function's docstring) -- but treating the two
    as one number (as an earlier version of this function did) made it
    impossible to tell "the freeze isn't working" apart from "BatchNorm
    bookkeeping moved a little", which are very different problems.
    """
    return (_drift_by_block(initial_params, final_params),
            _drift_by_block(initial_buffers, final_buffers))


def train_stage2(
    base_path: Path, resume_from: Path,
    deriv_weight: float = 1.0, deriv_weight_warmup_epochs: int = 3, stats0_weight: float = 0.0,
    recon1_weight: float = 0.0, stats1_weight: float = 0.0, z0_from_deriv_weight: float = 0.0,
    recon0_scale: float = 1.0, recon1_scale: float = 1.0,
    stats0_scale: float = 1.0, stats1_scale: float = 1.0, deriv_scale: float = 1.0,
    epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
    val_fraction: float = 0.2, test_fraction: float = 0.1, num_workers: int = 4,
    min_step: int | None = None, min_stdev_phi: float | None = None,
    min_passing_steps: int | None = None,
    min_std_deriv: float | None = None, augment: bool = False,
    condition_on_theta: bool | None = None,
    val_ema_decay: float = 0.7, early_stopping_patience: int | None = None,
    seed: int = 0, checkpoint_path: Path | None = None, device: str | None = None,
    on_checkpoint_saved: Callable[[Path, int], None] | None = None,
    n_frozen_stages: int = 0,
    log_every_epoch: bool = True,
    loss_curve_path: Path | None = None,
) -> Path:
    """
    Stage 2 (latent-space validation, C0/C1 redesign): trains (E, D) on
    real consecutive-pair (x(t), x(t+dt)) windows with L_recon +
    lambda_deriv*L_deriv (+ stats0_weight*L_stats anchor, see below) --
    matching the project's own design doc's Stage 2 spec ("similar to
    Stage 1 + extra L_deriv").

    L_deriv is a LATENT-space consistency loss, genuinely different
    from Stage 1's C1 (which is pixel-space: decode z1, compare against
    the real pixel derivative). Here: z1(t) (the deriv stream's own
    encode of x(t)) is compared against a DETACHED
    [z0(t+dt)-z0(t)]/dt -- z0 being the recon stream's encode, at both
    t and t+dt. Detached because this is NOT derivable from pixel-space
    alone (per the design doc) -- z1 is being pulled toward what z0's
    OWN latent trajectory implies its rate of change should be, not
    the other way around; gradient from this term must not flow back
    into z0 through this path (z0 already gets its own gradient from
    L_recon), only into z1 and whatever of the shared trunk feeds it.
    This is exactly what Stage 3a's coupled integrator needs primed:
    z1 actually meaning dz0/dt in latent space, not just "some other
    thing D can decode" (which is as far as Stage 1's C1 alone gets
    it).

    deriv_weight_warmup_epochs (default 3): L_deriv's magnitude at the
    very start of stage 2 can be enormous relative to recon/stats --
    empirically ~100x recon in epoch 1, collapsing ~2000x by epoch 10
    -- because z1's encoding was never previously calibrated against a
    LATENT-space target (Stage 1's C1, if the ancestor went through it,
    only ever compares z1 in PIXEL space, after decoding). Even though
    z0 is correctly detached as L_deriv's target, that only protects
    z0's own VALUE -- the shared trunk's PARAMETERS (which also compute
    z0, via the separate recon path) are not protected from a huge
    gradient arriving via z1's path through that same trunk, and a
    transient ~100x-oversized loss term is enough to visibly disrupt
    recon quality for exactly as many epochs as the transient lasts.
    Since the problem is specifically transient (z1 calibrates fast --
    the ~2000x collapse above happens within the first several epochs
    on its own), linearly ramping deriv_weight's EFFECTIVE value from 0
    up to its full value over this many epochs (0 = no warmup, full
    weight from epoch 1) sidesteps the transient without changing the
    steady-state objective at all once warmup completes. The printed
    per-epoch deriv contribution already reflects the ramped, not the
    nominal, weight, so what's logged is always what was actually
    optimized that epoch.

    REPLACES the old L_interp entirely (not summed alongside it) --
    the design doc's own phrasing reads as substitution, and there was
    already a standing todo item questioning L_interp's usefulness
    before this redesign existed. The (t1,t2,t3)-triplet interpolation-
    consistency CHECK itself (check_interpolation.py) is UNCHANGED and
    still run as a before/after diagnostic (see below) -- only removed
    as a TRAINING loss, not as a sanity-check tool.

    Full loss: L_recon0 + recon0_weight*L_stats0 + recon1_weight*L_recon1
    + stats1_weight*L_stats1 + effective_deriv_weight*L_deriv (last three
    all default to 0 except deriv_weight -- see below). L_recon0 is the
    SAME term this function already had (state stream, unweighted --
    L_recon0's own weight is implicitly 1.0, the reference every other
    term is expressed relative to, matching every other stage in this
    project's own convention of never giving the primary recon term an
    explicit weight parameter).

    DEFAULT BEHAVIOR (recon1_weight=stats1_weight=0.0): z1 is trained
    PURELY by L_deriv -- a pure latent-space quantity from stage 2
    onward, never re-anchored to pixel space or to stats_head1 here.
    This reflects a real design decision, not a placeholder: the deriv
    bottleneck (and the shared trunk, to a lesser extent) needs to land
    in a region that already carries real, extractable directional
    signal before L_deriv alone can refine it -- see the zero-signal/
    checkerboard failure this project hit when L_deriv was once
    attempted with no such grounding at all. That grounding used to
    come from a separate stage 1b pass with its own pixel-space D1;
    this function now builds the deriv stream itself, directly from a
    stage 1a ancestor (see training/extend_encoder.py's own module
    docstring), with no equivalent pixel-space warm-start step at all.

    D1 IS confirmed unnecessary (resolves what used to be an open
    question here): recon1_weight is now ONLY meaningful when resuming
    from an already-multi-stream ancestor whose own 'deriv' stream is
    still DECODER-mode (a legacy shape -- see this function's own
    resume_from handling below for the two ancestor shapes it accepts).
    Against the new, default shape (deriv built fresh here, PURE_LATENT,
    no decoder at all), a nonzero recon1_weight raises clearly rather
    than silently doing nothing or crashing on a missing pathway.
    stats1_weight has NO such restriction -- stats_head1 is always
    available (built fresh for a stage-1a ancestor, loaded for an
    already-multi-stream one), kept "just in case" even though nothing
    trains it by default.

    recon1_weight/stats1_weight, if explicitly set nonzero against a
    legacy, still-decodable ancestor, mirror the loss terms a former
    stage 1b pass used to compute: L_recon1 is D1(z1) compared against
    the REAL pixel-space derivative via a per-sample normalized-ratio
    form (plain MSE would be wrong here -- a raw derivative's tiny
    natural scale would let whichever samples have the largest
    magnitude dominate the gradient disproportionately), and L_stats1
    predicts the SAME original stats stats_head0 predicts (not their
    derivative), via stats_head1, reusing the identical true_stats
    target L_stats0 already has. L_recon0 and L_recon1 are genuinely
    different loss FORMS at different natural scales
    (plain reconstruction MSE vs. a normalized ratio) -- summing them
    with an assumed-equal weight risked exactly the kind of scale
    domination this project already hit and fixed once, so this stays
    an explicit, independently-tunable choice rather than a silent
    assumption.

    Accepts EITHER a stage 1a (single-stream) ancestor directly -- the
    deriv stream is built fresh, in memory, right here -- OR an
    already-multi-stream ancestor (e.g. resuming a prior stage 2 run,
    or a legacy stage-1b-derived checkpoint while any still exist on
    disk). Raises clearly if the ancestor has more than one non-recon
    stream (L_deriv has no meaning without exactly one to compare z0's
    own trajectory against) -- a direct, unavoidable consequence of
    replacing the old L_interp (which worked for any single-stream
    checkpoint) with a loss that's inherently about the C0/C1 split,
    not a separate design choice made here.

    stats_head is loaded from the checkpoint but FROZEN here (not
    trained) -- used only as a fixed measuring instrument for the
    stats anchor below, never touching true (C++-computed) ground-truth
    statistics in a way that could retrain it, so any systematic error
    in stats_head's own predictions doesn't get compounded by also
    being asked to learn from this stage's own (smaller, fine-tuning)
    data.

    stats0_weight: an ANCHOR, not a second objective -- pulls E back
    toward producing z's the frozen stats_head can still correctly
    interpret against true statistics, directly countering the
    scale-collapse failure mode observed empirically once L_stats was
    removed entirely (nothing else tied z's absolute scale/distribution
    to anything external, so E and D were free to drift together
    arbitrarily while still satisfying L_recon). Since stats_head itself
    is frozen, gradient from this term flows only into E -- it cannot
    retrain stats_head, only discourage E from drifting away from what
    stats_head already understands. Defaults to 0.0 (no anchor, matching
    the original table exactly); this is an explicit, tunable deviation
    from that table, added in response to observed evidence rather than
    changing the design speculatively.

    n_frozen_stages: freezes the outermost n layers of E/D (see
    freeze_outer_layers()) -- reduces drift risk (fewer degrees of
    freedom for E/D to drift together), NOT training speed (outer
    layers are actually the smallest by parameter count in this
    architecture -- see freeze_outer_layers()'s docstring). Also NOT a
    structural guarantee against drift (see that function's docstring
    for the bottleneck/unbottleneck loophole). Per-block weight-drift is
    always printed at the end regardless, as the actual safety net --
    complementary to stats0_weight, not a substitute for it. Matches the
    design doc's own Stage 2 freeze pattern (shared trunk + D synthesis
    layers frozen, E's projection heads + D's first layer trainable)
    without needing new code -- this function already freezes outer
    (pixel-space-near) layers while keeping bottleneck-adjacent ones
    trainable, which is the same split by a different name.

    augment (default False): D4-dihedral + translation augmentation on
    the TRAINING set only (never val -- val_loss stays a clean measure
    of real, unaugmented performance). Was previously accepted by the
    params file's own parsing but silently never threaded through to
    the dataset here -- this stage always used encoder=None (raw
    pixel), so nothing about augment's own encoder=None requirement
    ever blocked it; it just wasn't wired up. Same underlying mechanism
    train_autoencoder already uses (MicrostructureEvolutionDataset's own
    augment flag) -- same (k, flip, shift) applied consistently across
    both frames of a window, so the derivative direction the window
    encodes stays geometrically consistent under the transform, not
    scrambled by it.

    z0_from_deriv_weight (default 0.0): how much of L_deriv's own
    gradient reaches z0, rather than being blocked entirely (the
    original design -- see this loss term's own comment, right where
    it's computed, for the full "optimizer could cheat" rationale
    against ever letting L_deriv shape z0 at all). 0.0 reproduces that
    original behavior exactly. A nonzero value reopens the cheating
    risk deliberately: z0's own trajectory could start drifting toward
    whatever makes z1's job easier rather than staying purely anchored
    by L_recon. If z1's own error looks bias-dominated (see
    check_parameter_dependence.py's own bias-vs-variance diagnostic) --
    consistent with z0's own trajectory being genuinely hard for a
    direct, first-order encoding to track, not just noisy -- letting
    z0 adapt even a little may be the only way to close that gap,
    since z1-only retraining (longer patience, more unfrozen encoder
    layers, augmentation) can never change what z0 itself is doing.
    Start SMALL (e.g. 0.05-0.2) and watch recon0/stats0 for
    degradation -- this is z0's own reconstruction quality being put at
    risk for z1's benefit, a real trade, not a free improvement. Also
    consider raising deriv_weight_warmup_epochs beyond its own default
    when using this, so the (now z0-reaching) L_deriv gradient doesn't
    hit z0 abruptly, at full weight, before it and z1 have had time to
    settle into a shared representation together -- a sudden, early
    shock could genuinely destabilize a z0 that Stage 1's own training
    already spent real time getting right.

    on_checkpoint_saved: see train_autoencoder(). log_every_epoch: see
    train_autoencoder() -- same behavior here.

    Before any training happens, also runs check_interpolation/
    check_perturbation on resume_from itself (stage 1's own checkpoint,
    untouched at that point) and saves the result under output/stage1/ --
    a direct before/after baseline against this same stage's own
    post-training check_interpolation/check_perturbation output, to
    answer "did stage 2 actually improve the latent representation"
    rather than assuming it did. These are diagnostic TOOLS, unrelated
    to (and unaffected by) L_interp no longer being a training loss.

    Always resumes from a stage-1 (or another stage-2) checkpoint --
    there's no "stage 2 from scratch". Grid size is read from that
    checkpoint's own config; ITS stats_weight is used only for
    checkpoint naming (see ancestor_stats_weight below), independent of
    THIS function's own stats0_weight parameter (the anchor's weight).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    # Same rationale as train_autoencoder(): config.txt is simulation-only
    # now, min_step must be passed explicitly. min_stdev_phi is NOT
    # required to be non-None -- it's genuinely allowed to be None (no
    # stdev-based filtering at all), unlike min_step, which has no
    # meaningful None value at the dataset level.
    if min_step is None:
        raise ValueError("train_stage2() requires min_step to be given explicitly -- "
                          "config.txt no longer provides ML training defaults.")

    prev = torch.load(resume_from, map_location=device, weights_only=True)
    model_cfg = prev["config"]
    stats_config = prev.get("stats_config")
    if stats_config is None:
        raise ValueError(f"{resume_from} has no stats_head (it was trained with stats_weight "
                          f"<= 0 in stage 1 -- both L_stats-as-anchor here and the "
                          f"check_interpolation/check_perturbation diagnostics require it)")
    stat_names = stats_config["stat_names"]
    # NOTE: this is the ANCESTOR checkpoint's own stats_weight (stage 1's),
    # used only to name this stage's checkpoint file consistently -- NOT
    # the same as this function's own stats0_weight parameter (the anchor
    # weight, which may be zero even though the ancestor's wasn't).
    ancestor_stats_weight = model_cfg["stats_weight"]
    size = model_cfg["size"]
    print(f"Resuming from {resume_from} (stat_names={stat_names}, "
          f"ancestor_stats_weight={ancestor_stats_weight}, this stage's stats0_weight={stats0_weight})")

    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, prev["model_state"],
    )

    if len(stream_configs) == 1:
        # A Stage 1a (single-stream, state-only) ancestor -- the deriv
        # stream is built fresh, in memory, right here, rather than
        # requiring a separate stage 1b pass and its own intermediate
        # checkpoint first (see extend_encoder.py's own module docstring
        # for the full rationale: stage 1b's own training loop has been
        # inert since it started running at epochs=0, and D1, the one
        # thing genuinely built ONLY by that loop's own surrounding
        # setup, is confirmed permanently unnecessary). condition_on_theta
        # here plays the SAME role stage 1b's own parameter of the same
        # name did -- deriv is CREATED here, once; this is the only
        # place left where that's a structural decision rather than
        # something to validate against an already-built ancestor (see
        # the OTHER branch below, where a mismatch raises instead).
        print(f"Resuming from {resume_from}: single-stream (stage 1a) checkpoint -- building "
              f"the deriv stream fresh (replaces what used to require a separate stage 1b pass).")
        ext = extend_state_checkpoint_with_deriv_stream(
            resume_from, condition_on_theta=(True if condition_on_theta is None else condition_on_theta),
            device=device,
        )
        ae = ext.ae
        stats_head = ext.stats_head0
        stats_head1 = ext.stats_head1
        stream_configs = ext.stream_configs
        recon_stream_name = ext.state_name
        recon_stream = stream_configs[recon_stream_name]
        deriv_stream_name = "deriv"
        deriv_stream = stream_configs[deriv_stream_name]
        decoder_for_stream = {recon_stream_name: "D0"}
        frozen_modules = freeze_outer_layers(ae, n_frozen_stages)
        if n_frozen_stages > 0:
            n_trainable = sum(p.numel() for p in ae.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in ae.parameters())
            print(f"Froze outermost {n_frozen_stages} stage(s) on each side: "
                  f"{n_trainable}/{n_total} AE parameters remain trainable "
                  f"({100*n_trainable/n_total:.1f}%)")
        initial_params = {k: v.clone().cpu() for k, v in ae.named_parameters()}
        initial_buffers = {k: v.clone().cpu() for k, v in ae.named_buffers()}
        # stats_head0 (extended fresh above, loaded from stage 1a's own
        # trained weights): frozen, same rationale as the OTHER branch's
        # identical freezing below -- a fixed, trustworthy measuring
        # instrument for the anchor term.
        stats_head.eval()
        for p in stats_head.parameters():
            p.requires_grad_(False)
        # stats_head1 (extended fresh above, RANDOM init -- there is no
        # prior stats_head1 to have trained it): deliberately left
        # TRAINABLE here, unlike the OTHER branch's frozen, already-
        # trained stats_head1 -- this is this stream's OWN first chance
        # to learn anything at all, exactly parallel to how stage 1b
        # itself used to leave a freshly-built stats_head1 trainable.
    else:
        # Already-extended ancestor (e.g. resuming a prior stage 2 run,
        # or -- while any such checkpoints still exist on disk -- a
        # stage 1b-derived one). Unchanged from before, except: the
        # deriv-stream check below now accepts a PURE_LATENT stream, not
        # just a DECODER-mode one (a stage 1b-derived ancestor's own
        # "deriv" used to always be DECODER-mode; a fresh, stage-1a-
        # resumed run from the OTHER branch above now saves it as
        # PURE_LATENT instead -- resuming again from THAT checkpoint
        # must not be rejected here).
        recon_stream = stream_configs[recon_stream_name]
        other_streams = [n for n in stream_configs if n != recon_stream_name]
        if len(other_streams) != 1:
            raise ValueError(
                f"train_stage2() requires the ancestor checkpoint to have exactly one "
                f"deriv-role stream -- L_deriv has no meaning without one (this is a direct "
                f"consequence of replacing L_interp with L_deriv, not a separate restriction). "
                f"Got {len(other_streams)} other stream(s): {other_streams}."
            )
        deriv_stream_name = other_streams[0]
        deriv_stream = stream_configs[deriv_stream_name]

        # condition_on_theta is NOT decided here -- deriv's theta-FiLM
        # conditioning is a structural property fixed once, when the
        # stream is CREATED (see the OTHER branch above, or stage 1b's
        # own former docstring, for the full rationale). Stage 2 only
        # ever resumes an already-built encoder in THIS branch, so there
        # is nothing to set here -- only something to VALIDATE. None
        # (default) skips this entirely, trusting whatever the resumed
        # checkpoint has. Given explicitly, a MISMATCH raises
        # immediately and clearly, rather than silently training stage 2
        # against a differently-conditioned ancestor than intended.
        if condition_on_theta is not None and deriv_stream.condition_on_theta != condition_on_theta:
            raise ValueError(
                f"condition_on_theta={condition_on_theta} was requested, but {resume_from}'s own "
                f"'{deriv_stream_name}' stream has condition_on_theta={deriv_stream.condition_on_theta} "
                f"-- this was decided when the stream was CREATED, not something stage 2 can "
                f"change by resuming with a different value. Drop this parameter here to match "
                f"whatever {resume_from} already has, or build a new ancestor with the value "
                f"you actually want."
            )

        # Always MultiStreamAutoencoder: the multi-stream requirement
        # above guarantees len(stream_configs) >= 2 by this point,
        # unlike train_autoencoder's own construction (which still needs
        # to support the single-stream case).
        #
        # decoder_for_stream: read from the ANCESTOR's own config, not
        # assumed -- a stage-1b-derived checkpoint has separate D0/D1
        # (see autoencoder.py's MultiStreamAutoencoder); a fresh,
        # stage-1a-resumed checkpoint from the OTHER branch above has
        # just D0, with no "deriv" entry at all (PURE_LATENT streams are
        # never looked up here -- see MultiStreamAutoencoder's own
        # pathway construction). An older, pre-stage-1b multi-stream
        # checkpoint (if one still exists) has no decoder_for_stream key
        # at all, in which case every stream falls back to sharing one
        # decoder, matching what such a checkpoint's own state_dict
        # actually has.
        encoder = Encoder(input_size=size, in_channels=1, base_channels=model_cfg["base_channels"],
                           stream_configs=stream_configs, n_theta=1)
        decoder_for_stream = model_cfg.get("decoder_for_stream")
        if decoder_for_stream is None:
            decoder = Decoder(output_size=size, out_channels=1, base_channels=model_cfg["base_channels"],
                               latent_channels=recon_stream.channels,
                               latent_spatial_size=recon_stream.spatial_size)
            ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                         stream_configs=stream_configs).to(device)
        else:
            decoders = {}
            for stream_name, decoder_key in decoder_for_stream.items():
                stream_cfg = stream_configs[stream_name]
                decoders[decoder_key] = Decoder(
                    output_size=size, out_channels=1, base_channels=model_cfg["base_channels"],
                    latent_channels=stream_cfg.channels, latent_spatial_size=stream_cfg.spatial_size,
                )
            ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders=decoders,
                                         stream_configs=stream_configs,
                                         decoder_for_stream=decoder_for_stream).to(device)
        ae.load_state_dict(prev["model_state"])
        frozen_modules = freeze_outer_layers(ae, n_frozen_stages)
        if n_frozen_stages > 0:
            n_trainable = sum(p.numel() for p in ae.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in ae.parameters())
            print(f"Froze outermost {n_frozen_stages} stage(s) on each side: "
                  f"{n_trainable}/{n_total} AE parameters remain trainable "
                  f"({100*n_trainable/n_total:.1f}%)")
        # Keep CPU copies of the starting parameters AND buffers
        # (separately -- see compute_weight_drift()), to report per-block
        # drift against at the end -- the actual safety net for the
        # bottleneck/unbottleneck loophole noted in freeze_outer_layers().
        initial_params = {k: v.clone().cpu() for k, v in ae.named_parameters()}
        initial_buffers = {k: v.clone().cpu() for k, v in ae.named_buffers()}

        stats_head = StatsHead(latent_channels=recon_stream.channels, stat_names=stat_names,
                                latent_spatial=recon_stream.spatial_size).to(device)
        stats_head.load_state_dict(prev["stats_head_state"])
        # FROZEN during stage 2: L_deriv is a purely latent-space z0/z1
        # comparison (see step()'s own docstring/comments) with no
        # ground-truth supervision for stats_head at all -- the anchor
        # term (stats0_weight*L_stats below) is the ONLY source of real-
        # statistics signal this stage has, and it's explicitly
        # gradient-into-E-only by design (see that term's own comment).
        # Without freezing, stats_head could drift away from actually
        # predicting real statistics while nothing holds it to the truth.
        # Freezing keeps it as a fixed, trustworthy measuring instrument
        # -- the same move as freezing the encoder in stage 3.
        stats_head.eval()
        for p in stats_head.parameters():
            p.requires_grad_(False)

        # stats_head1: the deriv stream's own analogous anchor. Same
        # freezing rationale as stats_head above -- gracefully absent
        # (None) if the ancestor itself never had one, in which case
        # L_stats1 is simply not computed regardless of what
        # stats1_weight is set to here.
        stats_head1_state = prev.get("stats_head1_state")
        if stats_head1_state is None:
            stats_head1 = None
            print("NOTE: ancestor checkpoint has no stats_head1 -- L_stats1 will not be "
                  "computed this stage, regardless of stats1_weight.")
        else:
            stats_head1 = StatsHead(latent_channels=deriv_stream.channels, stat_names=stat_names,
                                     latent_spatial=deriv_stream.spatial_size).to(device)
            stats_head1.load_state_dict(stats_head1_state)
            stats_head1.eval()
            for p in stats_head1.parameters():
                p.requires_grad_(False)


    mean = stats_config["stats_mean"].to(device)
    std = stats_config["stats_std"].to(device)
    stats_loss_fn = StatsLoss(mean, std, stat_names=stat_names)
    recon_loss = ReconLoss()

    if checkpoint_path is None:
        name = ae_checkpoint_name(size, model_cfg["latent_channels"], ancestor_stats_weight)
        checkpoint_path = _PYTHON_ROOT / "checkpoints" / "stage2" / f"{name}-stage2.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {checkpoint_path}")

    if loss_curve_path is None:
        name = ae_checkpoint_name(size, model_cfg["latent_channels"], ancestor_stats_weight)
        loss_curve_path = _PYTHON_ROOT.parent / "output" / "stage2" / f"{name}-stage2-loss_curve.png"

    epoch_history: list[int] = []
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_so_far_history: list[float] = []

    run_dirs = complete_run_dirs(base_path, size, size)
    if not run_dirs:
        raise ValueError(f"No complete runs found under {base_path}/{size}x{size} -- "
                          f"check base_path, or that metadata.txt exists there")
    train_dirs, val_dirs, test_dirs = split_run_dirs(run_dirs, val_fraction, test_fraction, seed=seed)
    print(f"{len(run_dirs)} complete runs -> "
          f"{len(train_dirs)} train / {len(val_dirs)} val / {len(test_dirs)} test dirs")

    if epochs == 0:
        # Ablation mode: no training happens (see the epoch loop
        # below), so train_set/train_loader would never be touched --
        # skipped entirely, same rationale as Stage 1a/1b's own fix.
        train_set = train_loader = None
        val_set = MicrostructureEvolutionDataset(val_dirs, encoder=None, window_length=2,
                                                  stat_names=stat_names, min_std_deriv=min_std_deriv,
                                                  min_step=min_step, min_stdev_phi=min_stdev_phi,
                                                  min_passing_steps=min_passing_steps)
        print(f"train_set: skipped (epochs=0 ablation -- never iterated over), "
              f"{len(val_set)} val consecutive-pair windows")
    else:
        train_set = MicrostructureEvolutionDataset(train_dirs, encoder=None, window_length=2,
                                                    stat_names=stat_names, min_std_deriv=min_std_deriv,
                                                    min_step=min_step, min_stdev_phi=min_stdev_phi,
                                                    min_passing_steps=min_passing_steps,
                                                    augment=augment)
        val_set = MicrostructureEvolutionDataset(val_dirs, encoder=None, window_length=2,
                                                  stat_names=stat_names, min_std_deriv=min_std_deriv,
                                                  min_step=min_step, min_stdev_phi=min_stdev_phi,
                                                  min_passing_steps=min_passing_steps)
        print(f"{len(train_set)} train / {len(val_set)} val consecutive-pair windows")
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")

    # Baseline, BEFORE any stage 2 training happens: resume_from is
    # exactly the model's current state right now (just loaded, untouched
    # by this function), so it can be used directly -- no new checkpoint
    # file needed. Stored under output/stage1/, not output/stage2/, since
    # this reflects stage 1's own checkpoint. Deliberately
    # check_interpolation/check_perturbation only, not check_reconstruction
    # -- those two are what actually test latent GEOMETRY quality (the
    # thing stage 2 exists to improve), whereas pixel-level reconstruction
    # fidelity is already covered separately by stage 1's own sanity check
    # in main.py. Comparing these against stage 2's own post-training
    # check_interpolation/check_perturbation output (same tools, run again
    # on the trained checkpoint) is the direct before/after answer to
    # "did stage 2 actually improve the latent representation".
    print("=" * 70)
    print("Baseline (pre-stage-2): latent geometry of the stage 1 checkpoint")
    print("=" * 70)
    check_interpolation(
        checkpoint_path=resume_from, min_step=min_step, device=device,
        output_path=(_PYTHON_ROOT.parent / "output" / "stage1"
                     / f"{resume_from.stem}-pre_stage2-interpolation.png"),
    )
    print()
    check_perturbation(
        checkpoint_path=resume_from, min_step=min_step, device=device,
        output_path=(_PYTHON_ROOT.parent / "output" / "stage1"
                     / f"{resume_from.stem}-pre_stage2-perturbation.png"),
    )
    print()

    # stats_head frozen (not optimized here); ae itself may also have
    # frozen outer layers (see freeze_outer_layers/n_frozen_stages) --
    # filter to only what's actually trainable.
    params = [p for p in ae.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    def step(batch, train: bool, deriv_weight_used: float):
        window, dt_window, theta, true_stats = batch
        window = window.to(device, non_blocking=True)
        dt_window = dt_window.to(device, non_blocking=True)
        theta = theta.to(device, non_blocking=True)
        true_stats = true_stats.to(device, non_blocking=True)

        x_t = window[:, 0]
        x_next = window[:, 1]
        dt = dt_window[:, 0].view(-1, 1, 1, 1)

        # theta (temperature, centered at T0) now actually used -- see
        # Encoder's own docstring for why the deriv stream specifically
        # needs it: the driving force a(T)=a0*(T-T0) genuinely vanishes
        # near T0, so a state-only encoder can get the DIRECTION of
        # change right from the image alone but has no way to know the
        # correct MAGNITUDE without T. Was previously unpacked from the
        # batch and never used at all.
        z_t = ae.encoders["shared"](x_t, theta=theta)
        z0_t = z_t[recon_stream_name]
        z1_t = z_t[deriv_stream_name]

        # Decoder access is decoder_for_stream-AWARE now (via
        # ae.pathways[...].decoder), not the old hardcoded
        # ae.decoders["shared"] -- D0/D1 may be genuinely separate
        # decoders now (a stage-1b-derived ancestor), not necessarily
        # one shared decoder.
        x_recon = (ae.pathways[recon_stream_name].decoder(z0_t)
                   * torch.exp(ae.pathways[recon_stream_name].log_output_scale))
        recon = recon_loss(x_recon, x_t)

        stats_pred = stats_head(z0_t)
        # ANCHOR (weight = stats0_weight, default 0.0 = no anchor, matching
        # the table exactly unless explicitly opted into): pulls E back
        # toward producing z's the FROZEN stats_head can still correctly
        # interpret against true statistics. Gradient flows through z0_t
        # into E only -- stats_head's own weights are frozen, so no
        # gradient reaches them regardless of this term being active.
        # This directly counters the scale-collapse failure mode observed
        # when L_stats was removed entirely (see train_stage2's
        # docstring): without ANY tie to true statistics, nothing stopped
        # E and D from drifting together arbitrarily while still
        # satisfying L_recon.
        stats_loss_val = stats_loss_fn(stats_pred, true_stats)

        # L_recon1: D1(z1) compared against the REAL pixel-space
        # derivative -- mirrors stage 1b's own former L_recon1 EXACTLY
        # (same per-sample normalized-ratio form, same rationale: a raw
        # derivative's tiny natural scale makes plain MSE let whichever
        # samples happen to have the largest magnitude dominate the
        # gradient disproportionately). Genuinely different from
        # L_deriv below -- this is PIXEL-space (decode z1, compare to
        # the real pixel derivative), L_deriv is LATENT-space (compare
        # z1 directly to what z0's own trajectory implies, never
        # decoding at all).
        #
        # Guarded here, not called unconditionally the way it used to
        # be: deriv_stream_name is NOT in ae.pathways at all whenever
        # deriv is PURE_LATENT (no D1 -- see extend_encoder.py's own
        # module docstring on why D1 is confirmed permanently
        # unnecessary) -- an unconditional call here would KeyError, not
        # silently contribute zero, once an ancestor resumed through
        # THIS function's own new Stage-1a branch above. Before this
        # guard existed, D1's own forward+backward pass ran on every
        # single batch regardless of recon1_weight's value (multiplied
        # by zero only when COMBINING into the total below) -- real,
        # wasted compute paid for a term that structurally could never
        # affect the loss whenever recon1_weight was left at its default.
        if deriv_stream_name in ae.pathways:
            pred_pixel_deriv1 = (ae.pathways[deriv_stream_name].decoder(z1_t)
                                  * torch.exp(ae.pathways[deriv_stream_name].log_output_scale))
            target_pixel_deriv1 = (x_next - x_t) / dt
            diff_norm1 = torch.linalg.vector_norm(pred_pixel_deriv1 - target_pixel_deriv1, dim=(1, 2, 3))
            target_norm1 = torch.linalg.vector_norm(target_pixel_deriv1, dim=(1, 2, 3)).clamp_min(1e-6)
            recon1 = (diff_norm1 / target_norm1).mean()
        elif recon1_weight > 0:
            raise ValueError(
                f"recon1_weight={recon1_weight} was requested, but '{deriv_stream_name}' has no "
                f"decoder at all (PURE_LATENT stream, no D1) -- L_recon1 has nothing to compute "
                f"against. This ancestor's own deriv stream was built without a decoder; "
                f"recon1_weight must be 0 when resuming from it."
            )
        else:
            recon1 = torch.tensor(0.0)

        # L_stats1: predicts the SAME original stats stats_head0
        # predicts (not their derivative) -- the SAME rationale a
        # former stage 1b pass's own identical term used: a grain
        # boundary's own motion doesn't imply the bulk statistics are
        # changing at any comparable rate, so predicting the SAME
        # target stats_head0 already uses tests whether z1 happens to
        # ALSO carry usable information about the CURRENT state's bulk
        # statistics -- a fair question even though z1's primary job
        # (L_deriv) is about motion, not state. None (ancestor never
        # had one) means this term is simply skipped, regardless of
        # stats1_weight.
        if stats_head1 is not None:
            stats_pred1 = stats_head1(z1_t)
            stats1_loss_val = stats_loss_fn(stats_pred1, true_stats)
        else:
            stats1_loss_val = torch.tensor(0.0)

        # L_deriv: z1(t) (the deriv stream's OWN encode of x(t)) pulled
        # toward what z0's own latent trajectory implies its rate of
        # change should be -- NOT derivable from pixel-space alone (per
        # the project's own design doc), which is the whole reason this
        # is a separate objective from L_recon1 above (pixel-space:
        # decode z1, compare against the real pixel derivative).
        #
        # z0_from_deriv_weight controls how much of L_deriv's own
        # gradient reaches z0 (0.0, the default, blocks it entirely --
        # today's original design: z0 already gets its own gradient
        # from L_recon above, and letting L_deriv ALSO shape z0 risked
        # the optimizer cheating by warping z0's own trajectory to make
        # z1's job easier, rather than z1 genuinely learning to predict
        # a trajectory that's independently anchored by L_recon). A
        # nonzero weight reopens that risk deliberately, in a
        # controlled, partial way -- x.detach() + weight*(x-x.detach())
        # has the exact same FORWARD value as x (detached and
        # non-detached copies are numerically identical, so the two
        # terms combine back to x's own value regardless of weight),
        # but its gradient w.r.t. x is exactly `weight`, not the full
        # 1.0 an un-detached x would give -- a genuinely controllable
        # dial, not just an on/off switch.
        if z0_from_deriv_weight > 0:
            # theta passed even though only the state stream's output is
            # kept below -- Encoder.forward computes EVERY stream in one
            # pass internally (see its own docstring), so it still needs
            # theta if ANY of its streams (deriv, here) requires it,
            # regardless of which single stream this particular call
            # site goes on to use.
            z0_next = ae.encoders["shared"](x_next, theta=theta)[recon_stream_name]
            z0_next_for_deriv = z0_next.detach() + z0_from_deriv_weight * (z0_next - z0_next.detach())
            z0_t_for_deriv = z0_t.detach() + z0_from_deriv_weight * (z0_t - z0_t.detach())
        else:
            with torch.no_grad():
                z0_next = ae.encoders["shared"](x_next, theta=theta)[recon_stream_name]
            z0_next_for_deriv = z0_next
            z0_t_for_deriv = z0_t.detach()
        target_deriv = (z0_next_for_deriv - z0_t_for_deriv) / dt
        deriv_loss = recon_loss(z1_t, target_deriv)

        total = (recon / recon0_scale + recon1_weight * recon1 / recon1_scale
                 + stats0_weight * stats_loss_val / stats0_scale
                 + stats1_weight * stats1_loss_val / stats1_scale
                 + deriv_weight_used * deriv_loss / deriv_scale)

        if train:
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        return (total.item(), recon.item(), recon1.item(),
                stats_loss_val.item(), stats1_loss_val.item(), deriv_loss.item())

    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=val_ema_decay)
    epochs_since_improvement = 0

    stats_label = "stats0" if stats0_weight > 0 else "stats0_diag"
    stats1_label = "stats1" if stats1_weight > 0 else "stats1_diag"
    # (weight, label, scale) for every OPTIONAL term -- recon0 is always
    # shown separately (the unweighted primary term). Only include a
    # term in the header/per-epoch breakdown at all if its own weight is
    # nonzero -- a term that structurally cannot contribute anything
    # ("+0.0*something", every single epoch) is noise, not information.
    active_terms = [(w, lbl, s) for w, lbl, s in [
        (recon1_weight, "recon1", recon1_scale), (stats0_weight, stats_label, stats0_scale),
        (stats1_weight, stats1_label, stats1_scale), (deriv_weight, "deriv", deriv_scale),
    ] if w > 0]

    print(f"Stage 2: starting {epochs} epochs (early_stopping_patience: "
          f"{early_stopping_patience}, batches of {batch_size}), "
          f"deriv_weight={deriv_weight}"
          f"{f' (ramped over {deriv_weight_warmup_epochs} epochs for training)' if deriv_weight_warmup_epochs > 0 else ''}"
          f", recon1_weight={recon1_weight}, stats0_weight={stats0_weight}"
          f"{' (anchor active)' if stats0_weight > 0 else ' (diagnostic only, not optimized)'}"
          f", stats1_weight={stats1_weight}"
          f"{' (anchor active)' if stats_head1 is not None and stats1_weight > 0 else ' (inactive)'}"
          f", augment={augment}"
          f", z0_from_deriv_weight={z0_from_deriv_weight}"
          f"{' (WARNING: nonzero -- z0 can now be shaped by L_deriv, not just L_recon)' if z0_from_deriv_weight > 0 else ''}")
    formula = " ".join(f"+{w}*{lbl}/{s}" for w, lbl, s in active_terms)
    print(f"/{epochs:3d} train = recon0/{recon0_scale} {formula} | valid = ...  | ema")

    for epoch in range(0 if epochs == 0 else 1, epochs + 1):
        # Linear ramp: 0 at epoch 0 (never reached, epochs are 1-indexed)
        # up to deriv_weight at epoch=deriv_weight_warmup_epochs and
        # beyond. deriv_weight_warmup_epochs=0 (opt-out) skips this
        # entirely -- full weight from epoch 1, byte-identical to before
        # this parameter existed.
        effective_deriv_weight = (
            deriv_weight if deriv_weight_warmup_epochs <= 0
            else deriv_weight * min(1.0, epoch / deriv_weight_warmup_epochs)
        )
        # val_deriv_weight is NEVER ramped, unlike effective_deriv_weight
        # above -- the ramp's own purpose (see deriv_weight_warmup_epochs'
        # docstring) is protecting the TRAINING gradient from a transient
        # shock through the shared trunk; val_loss is never backpropagated,
        # so it never needed that protection, and using the ramped weight
        # for it too was just an artifact of sharing one variable across
        # both calls below, not a deliberate design choice. Keeping val's
        # own weight constant from epoch 1 means best_val_loss comparisons
        # stay meaningful across the whole run -- no discontinuity when the
        # ramp completes, and no need for the kind of best_val_loss reset
        # CheckpointCriterionTracker already does for ITS OWN, different
        # warmup (raw-vs-EMA) -- that reset fixes a real problem there but
        # introduces a visible non-monotonic jump in best_so_far_history,
        # which this avoids by never letting the comparison target drift
        # in the first place.
        val_deriv_weight = deriv_weight

        ae.train()
        # stats_head stays in eval mode always -- frozen, never trained
        # here, so no reason to toggle its train/eval-mode-specific
        # behavior (dropout/batchnorm, if any) at all.
        for m in frozen_modules:
            # ae.train() above is recursive and would otherwise flip
            # these back to train mode, letting BatchNorm's running
            # stats keep drifting via the forward-pass EMA even though
            # requires_grad_(False) correctly stops gradient updates --
            # see freeze_outer_layers()'s docstring.
            m.eval()
        train_total = train_recon = train_recon1 = train_stats = train_stats1 = train_deriv = 0.0
        if epoch > 0:
            n_train = len(train_set)
            for batch in train_loader:
                bs = batch[0].size(0)
                total, recon, recon1, stats, stats1, deriv = step(
                    batch, train=True, deriv_weight_used=effective_deriv_weight)
                train_total += total * bs
                train_recon += recon * bs
                train_recon1 += recon1 * bs
                train_stats += stats * bs
                train_stats1 += stats1 * bs
                train_deriv += deriv * bs
            train_total /= n_train
            train_recon /= n_train
            train_recon1 /= n_train
            train_stats /= n_train
            train_stats1 /= n_train
            train_deriv /= n_train
        else:
            # epoch 0 (epochs=0 ablation only): no training at all --
            # NaN honestly reflects that these metrics don't apply this
            # "epoch", rather than a misleading 0.0.
            train_total = train_recon = train_recon1 = train_stats = train_stats1 = train_deriv = float("nan")

        ae.eval()
        val_total = val_recon = val_recon1 = val_stats = val_stats1 = val_deriv = 0.0
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0)
                total, recon, recon1, stats, stats1, deriv = step(
                    batch, train=False, deriv_weight_used=val_deriv_weight)
                val_total += total * bs
                val_recon += recon * bs
                val_recon1 += recon1 * bs
                val_stats += stats * bs
                val_stats1 += stats1 * bs
                val_deriv += deriv * bs
        val_total /= n_val
        val_recon /= n_val
        val_recon1 /= n_val
        val_stats /= n_val
        val_stats1 /= n_val
        val_deriv /= n_val

        _, saved_this_epoch = tracker.update(epoch, val_total)
        val_ema = tracker.val_ema
        val_ema_str = f"{val_ema:7.4f}" if val_ema is not None else "(warmup)"

        epoch_history.append(epoch)
        train_loss_history.append(train_total)
        val_loss_history.append(val_total)
        best_so_far_history.append(tracker.best_val_loss)
        loss_curve(
            epoch_history, train_loss_history, val_loss_history, best_so_far_history,
            loss_curve_path, title="Stage 2 loss",
        )

        # (train_weight, val_weight, scale, train_value, val_value) for
        # every entry, uniformly -- even recon1/stats0/stats1, whose two
        # weights are always equal, since they're never ramped. Only
        # "deriv" actually has train_weight != val_weight now -- giving
        # every entry the same 5-tuple shape keeps the two loops below
        # simple, rather than special-casing "deriv" alone. Necessary,
        # not just tidy: these ARE what actually reproduces train_total/
        # val_total's own deriv component (step() only returns the raw,
        # unweighted deriv_loss; the weight multiplication happens HERE,
        # in the printed breakdown, separately from step()'s own total
        # computation) -- using one shared weight for both sides here
        # would silently print a val breakdown that no longer sums to
        # the correct val_total, even though val_total itself (computed
        # inside step(), which DOES already use the right weight) stayed
        # correct.
        term_values = {
            "recon1": (recon1_weight, recon1_weight, recon1_scale, train_recon1, val_recon1),
            stats_label: (stats0_weight, stats0_weight, stats0_scale, train_stats, val_stats),
            stats1_label: (stats1_weight, stats1_weight, stats1_scale, train_stats1, val_stats1),
            "deriv": (effective_deriv_weight, val_deriv_weight, deriv_scale, train_deriv, val_deriv),
        }
        train_terms = " ".join(f"+{tw*tv/s:7.4f}" for lbl, (tw, _, s, tv, _) in term_values.items()
                                if any(lbl == l for _, l, _ in active_terms))
        val_terms = " ".join(f"+{vw*vv/s:7.4f}" for lbl, (_, vw, s, _, vv) in term_values.items()
                              if any(lbl == l for _, l, _ in active_terms))
        msg = (f"{epoch:4d}|"
               f"{train_total:7.4f} ={train_recon/recon0_scale:7.4f} {train_terms} |"
               f"{val_total:7.4f} ={val_recon/recon0_scale:7.4f} {val_terms} |"
               f"{val_ema_str}")

        if saved_this_epoch:
            epochs_since_improvement = 0
            torch.save({
                "model_state": ae.state_dict(),
                "stats_head_state": stats_head.state_dict(),
                "stats_head1_state": stats_head1.state_dict() if stats_head1 is not None else None,
                "epoch": epoch,
                "val_loss": val_total,
                "val_loss_ema": val_ema,
                "test_dirs": [str(Path(d).resolve()) for d in test_dirs],
                "config": {
                    "size": model_cfg["size"], "base_channels": model_cfg["base_channels"],
                    "latent_channels": recon_stream.channels,
                    "latent_spatial_size": recon_stream.spatial_size,
                    "stats_weight": ancestor_stats_weight,
                    "stream_configs": {
                        name: {"channels": cfg.channels, "spatial_size": cfg.spatial_size,
                               "mode": cfg.mode.value, "condition_on_theta": cfg.condition_on_theta}
                        for name, cfg in stream_configs.items()
                    },
                    "recon_stream_name": recon_stream_name,
                    "decoder_for_stream": decoder_for_stream,
                },
                "stats_config": {"stat_names": stat_names, "stats_mean": mean.cpu(), "stats_std": std.cpu()},
                "stage2_config": {"deriv_weight": deriv_weight,
                                   "deriv_weight_warmup_epochs": deriv_weight_warmup_epochs,
                                   "recon1_weight": recon1_weight,
                                   "stats0_weight": stats0_weight, "stats1_weight": stats1_weight,
                                   "n_frozen_stages": n_frozen_stages, "resumed_from": str(resume_from)},
            }, checkpoint_path)
            msg += "  -> saved"
            if on_checkpoint_saved is not None:
                on_checkpoint_saved(checkpoint_path, epoch)
        else:
            epochs_since_improvement += 1

        if log_every_epoch or saved_this_epoch:
            print(msg)

        if early_stopping_patience is not None and epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}: no improvement for "
                  f"{early_stopping_patience} epochs")
            break

    # Per-block weight drift: the actual safety net against the
    # bottleneck/unbottleneck loophole in freeze_outer_layers() -- a
    # frozen block's PARAMETERS should show exactly 0; a large,
    # unexpected drift in bottleneck/unbottleneck specifically is the
    # red flag to watch for. BUFFER drift (BatchNorm running stats) is
    # reported separately -- should also be ~0 for frozen blocks now
    # that they're kept in eval() mode every epoch, but is a distinct
    # question from parameter drift and shouldn't be conflated with it.
    # Compares the SAVED (best) checkpoint, not just the final epoch's
    # in-memory weights, since those may differ if early stopping fired.
    final_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    final_state = final_checkpoint["model_state"]
    final_params = {k: final_state[k] for k in initial_params}
    final_buffers = {k: final_state[k] for k in initial_buffers}
    param_drift, buffer_drift = compute_weight_drift(
        initial_params, initial_buffers, final_params, final_buffers)

    print("\nPer-block PARAMETER drift (L2 norm of change from stage-1 starting point):")
    frozen_groups = {group for group, value in param_drift.items() if value == 0.0}
    for group, value in sorted(param_drift.items()):
        flag = "  <- frozen, should be 0" if group in frozen_groups else ""
        print(f"  {group:<28} {value:10.4f}{flag}")
    if buffer_drift:
        print("\nPer-block BUFFER drift (e.g. BatchNorm running_mean/running_var):")
        for group, value in sorted(buffer_drift.items()):
            flag = "  <- frozen, should be ~0" if group in frozen_groups else ""
            print(f"  {group:<28} {value:10.4f}{flag}")

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True,
                         help="grid size (square only) -- locates base/<size>x<size>/, "
                              "reading ITS OWN metadata.txt (not config.txt)")
    parser.add_argument("--base", type=Path, default=_PYTHON_ROOT.parent / "datasets")
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
    parser.add_argument("--min-passing-steps", type=int, default=None,
                         help="exclude an entire run when fewer than this many of its steps clear "
                              "min-stdev-phi -- see build_good_steps' own docstring in "
                              "training/datasets.py")
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
    parser.add_argument("--quiet", action="store_true",
                         help="only print a line when a checkpoint is actually saved, instead "
                              "of every epoch")
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
        size=args.size, base_path=args.base,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        base_channels=args.base_channels, latent_channels=args.latent_channels,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        num_workers=args.num_workers, min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        min_passing_steps=args.min_passing_steps,
        stat_names=args.stat_names, stats0_weight=args.stats_weight,
        val_ema_decay=args.val_ema_decay, early_stopping_patience=args.early_stopping_patience,
        seed=args.seed, checkpoint_path=args.checkpoint, resume_from=args.resume_from,
        device=args.device, log_every_epoch=not args.quiet,
    )

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
