"""
Stage 1 (train_autoencoder) of the LIENS pipeline -- an importable
function, not just a CLI script; see main.py for the orchestrated
1 -> 2 -> 3 pipeline. Split out of what used to be a combined
train_ae.py (train_autoencoder + train_stage2 together) once stage 1b's
own removal made that combination less coherent than it once was --
see training/train_stage2.py's own module docstring for that half.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m training.train_stage1 --size 64 --base ../datasets
"""

import time
import argparse
from collections.abc import Callable
from pathlib import Path
import gc

import torch
from torch.utils.data import DataLoader

from utils.logging_utils import print_run_parameters, EpochProgress
from models.autoencoder import Autoencoder
from models.constants import LATENT_SPATIAL_SIZE
from models.latent_streams import DEFAULT_STREAM_NAME, LatentStreamMode
from training._checkpoint_criterion import (
    CheckpointCriterionTracker, ComponentBestTracker, atomic_torch_save, clamp_grace_epochs,
)
from training.datasets import MicrostructureSnapshotDataset, complete_run_dirs, split_run_dirs
from training.losses import ReconLoss, StatsLoss
from training.stats_head import StatsHead
from utils.naming import ae_checkpoint_name
from utils.plots import (loss_component_scatter, loss_curve, loss_scale_curve,
                          write_loss_history, should_write_loss_figure)

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
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/training/train_stage1.py -> python/


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



def _vram_report(tag: str) -> str:
    """
    allocated / reserved / free, as one line.

    The three together are what separates the two causes of an OOM that
    arrives WELL INTO an epoch rather than at its start:

      - allocated stays flat, reserved grows  -> FRAGMENTATION. The caching
        allocator is holding blocks it cannot coalesce into one large enough
        for the next request. Retry with the environment variable
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, which lets it grow
        segments instead of splitting them.
      - allocated itself grows                -> a genuine leak: something is
        keeping a reference per iteration.
      - both flat but `free` small            -> another process (or an
        earlier cell in the same IPython kernel) owns the rest of the card,
        and no change here will help.

    A 128x128 stage 1 at batch 64 needs ~3.1 GB steady-state, so on an 8 GB
    card any of these is survivable ONLY if it is identified.
    """
    if not torch.cuda.is_available():
        return ""
    free_b, total_b = torch.cuda.mem_get_info()
    return (f"    [vram {tag}] allocated {torch.cuda.memory_allocated() / 1024**3:.2f}"
            f" / reserved {torch.cuda.memory_reserved() / 1024**3:.2f}"
            f" / peak {torch.cuda.max_memory_allocated() / 1024**3:.2f}"
            f" / free {free_b / 1024**3:.2f} of {total_b / 1024**3:.2f} GB")


_STAGE1_PREAMBLE_PARAMS = (
    # See train_lds's _LDS_PREAMBLE_PARAMS for why these are excluded here.
    "size", "epochs", "batch_size", "min_step", "min_stdev_phi", "min_passing_steps",
    "stats0_weight", "recon0_scale", "stats0_scale", "early_stopping_patience",
)


def train_autoencoder(
    size: int, base_path: Path,
    epochs: int = 100, batch_size: int = 64, lr: float = 1e-3,
    base_channels: int = 32, latent_channels: int = 8,
    latent_spatial_size: int = LATENT_SPATIAL_SIZE,
    val_fraction: float = 0.2, test_fraction: float = 0.1, num_workers: int = 4,
    augment: bool = True, cache_in_memory: bool = True,
    min_step: int | None = None, min_stdev_phi: float | None = None,
    min_passing_steps: int | None = None,
    stat_names: list[str] | None = None, stats0_weight: float | None = None,
    recon0_scale: float = 1.0, stats0_scale: float = 1.0,
    val_ema_decay: float = 0.7, ema_warmup_epochs: int = 5, early_stopping_patience: int | None = None,
    seed: int = 0, checkpoint_path: Path | None = None,
    resume_from: Path | None = None, device: str | None = None,
    on_checkpoint_saved: Callable[[Path, int], None] | None = None,
    log_every_epoch: bool = True,
    loss_curve_path: Path | None = None, vram_log_every: int = 0,
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

    vram_log_every: print a VRAM breakdown every N training batches (0 =
    never, the default). Useful when an OOM arrives partway through an
    epoch rather than at its start -- see _vram_report for how the three
    numbers separate fragmentation from a leak from a busy card. An
    augmented 128x128 epoch is ~48k batches, so ~2000 gives about 25
    lines per epoch.

    ema_warmup_epochs: the save criterion is suppressed for this many
    epochs at the start of a run (default 5, matching train_lds), while
    the EMA accumulates real history.

    It used to be 0, which made stage 1 the ONLY stage without the
    protection, and the failure mode is the one
    CheckpointCriterionTracker's own docstring describes: with no
    warmup, epoch 1's raw, unsmoothed val_loss seeds BOTH the EMA and
    best_val_loss, so a lucky first epoch sets a bar that later,
    properly-smoothed values struggle to clear (smoothing pulls toward
    an average, which rarely beats an extreme single-epoch low).

    Observed on a real 128x128 run: epoch 1's val_loss was the minimum
    of all 11 epochs, nothing was ever saved again, and early stopping
    fired at epoch 11 while TRAIN loss was still falling (-19.6% over
    the run). Stage 2 then trained from that epoch-1 checkpoint.

    0 restores the old behaviour. clamp_grace_epochs() caps it so at
    least one epoch can always save, which matters for short runs and
    for epochs=0 ablations.

    cache_in_memory: hold every decoded snapshot in RAM (default True,
    which is what this function did unconditionally before the flag
    existed). Snapshots are float16 on disk and cached as float32, so
    the cache is 2x the on-disk bytes, and on Windows the DataLoader
    spawns rather than forks -- each of `num_workers` workers receives
    its own pickled copy of the dataset, cache included. Total is
    therefore ~(1 + num_workers) x cache size:

        size    cache    with num_workers=4
          64   0.6 GB              2.9 GB
         128   2.3 GB             11.7 GB
         256   9.4 GB             46.8 GB
         512  37.5 GB            187.4 GB

    (~38k snapshots, measured on a real sweep; scales with the number
    of runs kept, not with grid size, so only the per-snapshot bytes
    change down that column.)

    Set False from 128x128 up. It also makes STARTUP faster, which is
    the counter-intuitive part: the cache is built by a serial
    `[self._load(i) for i in ...]` in the dataset's own __init__, one
    file at a time on one thread with the GPU idle, whereas uncached
    reads happen inside the `num_workers` DataLoader workers, in
    parallel, overlapped with compute. Caching moves that I/O from
    where it is parallel to where it is not.

    Keep True only when the dataset genuinely fits with room to spare
    AND the same frames are re-read many times per epoch (augment=True
    re-reads each base frame up to 32x, which is the case it was
    originally added for).

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
    print_run_parameters(train_autoencoder, locals(), _STAGE1_PREAMBLE_PARAMS)

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
    loss_components_path = loss_curve_path.with_name(
        loss_curve_path.stem + "-components" + loss_curve_path.suffix)
    loss_scales_path = loss_curve_path.with_name(
        loss_curve_path.stem + "-scales" + loss_curve_path.suffix)

    epoch_history: list[int] = []
    train_loss_history: list[float] = []
    scale_ratio_history: dict[str, list[float]] = {}   # per component: VAL raw / scale, vs epoch
    val_loss_history: list[float] = []
    best_so_far_history: list[float] = []
    # loss_component_scatter's own bookkeeping -- only meaningful when
    # include_stats (recon0+stats0 are both real terms); with stats0
    # inactive there is only ONE component, nothing to pair, and
    # loss_component_scatter itself already returns None/writes nothing
    # for that case, but there's no point building the (empty) history
    # dict at all in that case either.
    component_histories: dict[str, dict[str, list[float]]] = (
        {name: {"train": [], "val": [], "best_so_far": []} for name in ("recon0", "stats0")}
        if include_stats else {}
    )
    component_best_tracker = ComponentBestTracker()

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
        # derivative), but with cache_in_memory left at its default it
        # still means reading every train snapshot into memory upfront,
        # for a dataset that's then never iterated -- skipped entirely
        # instead.
        # stat_names normally locks to whatever train_set itself
        # auto-detected -- with train_set skipped, val_set's own
        # auto-detection is used directly instead, since it's the
        # only dataset being built at all in this case.
        train_set = train_loader = None
        val_set = MicrostructureSnapshotDataset(
            val_dirs, cache_in_memory=cache_in_memory, augment=False,
            min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
            include_stats=include_stats, stat_names=stat_names,
            split_label="validation",
        )
        print(f"train_set: skipped (epochs=0 ablation -- never iterated over), "
              f"{val_set.n_base_samples} (val, unaugmented)")
    else:
        train_set = MicrostructureSnapshotDataset(
            train_dirs, cache_in_memory=cache_in_memory, augment=augment,
            min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
            include_stats=include_stats, stat_names=stat_names,
            split_label="training",
        )
        # Lock val's stat_names to whatever train resolved (auto-detection is
        # order-dependent; without this, val could silently end up checked
        # against a different run's schema than train was).
        val_stat_names = train_set.stat_names if include_stats else None
        val_set = MicrostructureSnapshotDataset(
            val_dirs, cache_in_memory=cache_in_memory, augment=False,
            min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
            include_stats=include_stats, stat_names=val_stat_names,
            split_label="validation",
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
        # no-op for theta-free stage-1 checkpoints; upgrades any conditioner
        # a stage-2-lineage ancestor might carry (zero-pads the new theta col)
        from models.encoder import zero_pad_theta_columns
        ae.load_state_dict(zero_pad_theta_columns(prev["model_state"], ae))
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
            # device=device explicitly -- a bare torch.tensor(0.0)
            # defaults to CPU regardless of what device training is
            # actually running on. Harmless back when this got
            # .item()'d immediately below (a plain Python float has no
            # device), but step() now returns .detach()'d tensors for
            # on-device accumulation (see the epoch loop's own
            # comment) -- without this, a CUDA run would raise a
            # device-mismatch error the moment this got added into a
            # CUDA-resident running sum, every time include_stats is
            # False.
            stats0 = torch.tensor(0.0, device=device)
            total = recon0 / recon0_scale

        if train:
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        # Returned as GPU tensors, NOT .item()'d here -- .item() blocks
        # the CPU until the GPU actually finishes and forces a full
        # round trip EVERY batch. .detach() keeps each value a scalar
        # GPU tensor, cheap to accumulate into a running sum on-device
        # (see the epoch loop below, which now does exactly ONE
        # .item() per metric per epoch, not one per metric per batch)
        # -- while dropping the autograd graph, which backward() above
        # has already consumed and which would otherwise be kept alive
        # (and keep growing) for the rest of the epoch if left
        # attached. Same fix as train_lds.py's/train_stage2.py's own
        # step() -- see either function's own identical comment.
        return total.detach(), recon0.detach(), stats0.detach()

    # clamp_grace_epochs, not the raw value: a grace period covering every
    # remaining epoch means NO checkpoint is written at all -- a missing file,
    # not a worse one, and every downstream consumer fails far from the cause.
    _grace = clamp_grace_epochs(ema_warmup_epochs, epochs)
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=_grace, val_ema_decay=val_ema_decay)
    epochs_since_improvement = 0

    print(f"Starting {epochs} epochs (early_stopping_patience: "
          f"{early_stopping_patience}, batches of {batch_size})...")
    heading = f"/{epochs:3d} "
    heading += (f"train = recon0/{recon0_scale} +{stats0_weight:.3g}*stats0/{stats0_scale} | "
                f"valid = ...  | ema") if include_stats else f"train = recon0/{recon0_scale} | valid | ema"
    print(heading)

    if epochs > 0 and resume_from is not None:
        # Epoch-0 reference: how the model that resume_from actually loaded
        # performs BEFORE this run touches it. Without it a resumed run's
        # epoch 1 has nothing to be compared against -- and after a SIZE PORT
        # that is the single number worth having, since it says what the
        # transferred weights were worth before any 128x128 training happened.
        # train_stage2 has had this for a while; stage 1 did not.
        #
        # Deliberately NOT a real epoch: it never calls tracker.update, never
        # saves, and never appends to epoch_history or the loss curve. If it
        # participated in any of those it would ALWAYS win the first save
        # (nothing beats best_val_loss=inf), writing a checkpoint labelled
        # epoch 0 before any training happened.
        #
        # RNG state is saved and restored around the whole block, for the
        # reason train_stage2's own reference row documents: the forward
        # passes below advance torch's global RNG, which would change
        # train_loader's shuffle order at epoch 1 (shuffle=True) and so
        # silently alter the training trajectory depending on whether a
        # purely diagnostic row was printed at all.
        _rng_state = torch.get_rng_state()
        _cuda_rng_state = torch.cuda.get_rng_state() if device.type == "cuda" else None
        # step() writes into these via closure, and the epoch loop rebinds them
        # per epoch -- so they must exist before the first call, and the
        # instances used here are DISCARDED when epoch 1 rebinds them. That is
        # correct: the z0 mean/std line is a per-epoch statistic and this
        # reference pass must not contribute to epoch 1's.
        z0_train_stats, z0_val_stats = _RunningStats(), _RunningStats()
        ae.eval()
        if stats_head is not None:
            stats_head.eval()
        _ref_total = torch.zeros((), device=device)
        _ref_recon0 = torch.zeros((), device=device)
        _ref_stats0 = torch.zeros((), device=device)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0) if include_stats else batch.size(0)
                total, recon0, stats0 = step(batch, train=False)
                _ref_total += total * bs
                _ref_recon0 += recon0 * bs
                _ref_stats0 += stats0 * bs
        _n = len(val_set)
        _r_total = (_ref_total / _n).item()
        # Same as stage 2's: this pass has just measured the ancestor's
        # val_loss under THIS run's objective, so hand it to the tracker as a
        # ceiling rather than printing and discarding it. Without it
        # best_val_loss starts at inf and epoch 1 always saves, which on a
        # resume can overwrite a better checkpoint with a worse one.
        tracker.reference_val_loss = _r_total
        _r_recon0 = (_ref_recon0 / _n).item()
        _r_stats0 = (_ref_stats0 / _n).item()
        # SCALED exactly as the epoch rows are (see the msg built below):
        # step() returns RAW component values and the epoch line divides them
        # by recon0_scale / stats0_scale and applies stats0_weight, because
        # that is what the heading advertises ("recon0/0.0001 +1*stats0/0.01")
        # and what actually sums to total. Printing them raw here made the ref
        # row the only line in the table whose components did NOT add up --
        # 6.9589 alongside 0.0003 + 0.0381 -- which reads as a broken total
        # rather than a units mismatch.
        if include_stats:
            print(f"{'ref':>4}|    nan =    nan +    nan "
                  f"|{_r_total:7.4f} ={_r_recon0 / recon0_scale:7.4f} "
                  f"+{stats0_weight * _r_stats0 / stats0_scale:7.4f} |(before this run)",
                  flush=True)
        else:
            print(f"{'ref':>4}|    nan |{_r_total:7.4f}  (before this run)", flush=True)
        torch.set_rng_state(_rng_state)
        if _cuda_rng_state is not None:
            torch.cuda.set_rng_state(_cuda_rng_state)

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

        # GPU-resident accumulators, not Python floats -- see step()'s
        # own docstring/comment: the ONLY host sync per phase (train/
        # val) is the batch of three .item() calls after each loop
        # ends, not three per batch. Same fix as train_lds.py's/
        # train_stage2.py's own epoch loop.
        train_total_sum = torch.zeros((), device=device)
        train_recon0_sum = torch.zeros((), device=device)
        train_stats0_sum = torch.zeros((), device=device)
        n_train = 0
        if epoch > 0:
            _epoch_progress = EpochProgress(len(train_loader))
            for batch_idx, batch in enumerate(train_loader):
                _epoch_progress.tick()
                bs = batch[0].size(0) if include_stats else batch.size(0)
                total, recon0, stats0 = step(batch, train=True)
                train_total_sum += total * bs
                train_recon0_sum += recon0 * bs
                train_stats0_sum += stats0 * bs
                n_train += bs
                if vram_log_every and batch_idx % vram_log_every == 0:
                    # Printed from INSIDE the batch loop on purpose: an epoch
                    # here can be ~48k batches, so a per-epoch report would
                    # first appear only after the point where the OOM happens.
                    print(_vram_report(f"epoch {epoch} batch {batch_idx}"), flush=True)
            _epoch_progress.close()
            train_total = (train_total_sum / n_train).item()
            train_recon0 = (train_recon0_sum / n_train).item()
            train_stats0 = (train_stats0_sum / n_train).item()
        else:
            # epoch 0 (epochs=0 ablation only): no training at all --
            # NaN honestly reflects that these metrics don't apply this
            # "epoch" (n_train stays 0, so dividing would also fail),
            # rather than a misleading 0.0.
            train_total = train_recon0 = train_stats0 = float("nan")

        ae.eval()
        if stats_head is not None:
            stats_head.eval()
        val_total_sum = torch.zeros((), device=device)
        val_recon0_sum = torch.zeros((), device=device)
        val_stats0_sum = torch.zeros((), device=device)
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0) if include_stats else batch.size(0)
                total, recon0, stats0 = step(batch, train=False)
                val_total_sum += total * bs
                val_recon0_sum += recon0 * bs
                val_stats0_sum += stats0 * bs
        val_total = (val_total_sum / n_val).item()
        val_recon0 = (val_recon0_sum / n_val).item()
        val_stats0 = (val_stats0_sum / n_val).item()

        _, saved_this_epoch = tracker.update(epoch, val_total)
        val_ema = tracker.val_ema
        val_ema_str = f"{val_ema:7.4f}" if val_ema is not None else "(warmup)"

        epoch_history.append(epoch)
        train_loss_history.append(train_total)
        val_loss_history.append(val_total)
        best_so_far_history.append(tracker.best_val_loss)
        # L_XX / XX_scale (val): recon0 always; stats0 only when it is an active term
        scale_ratio_history.setdefault("recon0", []).append(val_recon0 / recon0_scale)
        if include_stats:
            scale_ratio_history.setdefault("stats0", []).append(val_stats0 / stats0_scale)
        if should_write_loss_figure(epoch, log_every_epoch, n_points=len(epoch_history)):
            loss_curve(
                epoch_history, train_loss_history, val_loss_history, best_so_far_history,
                loss_curve_path, title="Stage 1 loss",
            )
            write_loss_history(loss_curve_path, epoch_history, train_loss_history, val_loss_history, best_so_far_history)
            loss_scale_curve(epoch_history, scale_ratio_history, loss_scales_path,
                             title="Stage 1 loss/scale ratios (val)")
        if include_stats:
            current_val_components = {
                "recon0": val_recon0 / recon0_scale,
                "stats0": stats0_weight * val_stats0 / stats0_scale,
            }
            best_components = component_best_tracker.update(current_val_components, saved_this_epoch)
            component_histories["recon0"]["train"].append(train_recon0 / recon0_scale)
            component_histories["recon0"]["val"].append(current_val_components["recon0"])
            component_histories["recon0"]["best_so_far"].append(best_components["recon0"])
            component_histories["stats0"]["train"].append(stats0_weight * train_stats0 / stats0_scale)
            component_histories["stats0"]["val"].append(current_val_components["stats0"])
            component_histories["stats0"]["best_so_far"].append(best_components["stats0"])
            if should_write_loss_figure(epoch, log_every_epoch, n_points=len(epoch_history)):
                loss_component_scatter(
                    epoch_history, component_histories, loss_components_path,
                    title="Stage 1 loss components",
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
            atomic_torch_save(checkpoint, checkpoint_path)
            msg += "  -> saved"
            msg += f" at {time.strftime('%H:%M')}"  # match the checkpoint filename timestamp
            if on_checkpoint_saved is not None:
                try:
                    on_checkpoint_saved(checkpoint_path, epoch)
                except Exception as e:
                    # Bookkeeping must never kill training: a failed registry
                    # upsert crashed one real run BETWEEN the save and the epoch
                    # line (checkpoint one epoch newer than the log). Announce
                    # and continue -- a lost registry row, never a lost run.
                    print(f"  WARNING: on_checkpoint_saved failed "
                          f"({type(e).__name__}: {e}) -- continuing training")
        else:
            epochs_since_improvement += 1

        if log_every_epoch or saved_this_epoch:
            print(msg)
            z0_train_mean, z0_train_std = z0_train_stats.mean_std()
            z0_val_mean, z0_val_std = z0_val_stats.mean_std()
            print(f"      z0: train mean={z0_train_mean:+.4e} std={z0_train_std:.4e} | "
                  f"val mean={z0_val_mean:+.4e} std={z0_val_std:.4e}")

        # `epoch > _grace`, mirroring train_lds: during the warmup window
        # should_save is unconditionally False, so every one of those epochs
        # increments epochs_since_improvement. Without this, any
        # ema_warmup_epochs >= early_stopping_patience would stop the run
        # before the criterion had even started answering -- the same
        # interaction that made train_stage2's deriv_target_centered switch
        # stop one epoch short of its own grace window.
        if (early_stopping_patience is not None and epoch > _grace
                and epochs_since_improvement >= early_stopping_patience):
            print(f"Early stopping at epoch {epoch}: no improvement for "
                  f"{early_stopping_patience} epochs")
            break

    # Unconditional final write: the in-loop calls are throttled (see
    # should_write_loss_figure), and a run can end on an epoch that was
    # skipped -- via early stopping, or simply because the last epoch
    # wasn't a multiple of the interval. Without this the figures left on
    # disk could be up to `every` epochs stale, which is exactly the
    # state a finished run gets judged from.
    loss_curve(
        epoch_history, train_loss_history, val_loss_history, best_so_far_history,
        loss_curve_path, title="Stage 1 loss",
    )
    write_loss_history(loss_curve_path, epoch_history, train_loss_history, val_loss_history, best_so_far_history)
    if scale_ratio_history:
        loss_scale_curve(epoch_history, scale_ratio_history, loss_scales_path,
                         title="Stage 1 loss/scale ratios (val)")
    if component_histories:
        loss_component_scatter(
            epoch_history, component_histories, loss_components_path,
            title="Stage 1 loss components",
        )

    if not Path(checkpoint_path).exists() and resume_from is not None:
        # NOTHING BEAT THE ANCESTOR. Keyed on resume_from, NOT on
        # tracker.reference_val_loss still being set: reset_with_grace clears
        # the reference when the objective changes mid-run (the
        # deriv_target_centered switch), so a run that switched would fall
        # through to the raise -- which is exactly the run most likely not to
        # improve, since its criterion restarted from the post-grace EMA.
        #
        # That is not a failure: the reference
        # ceiling exists precisely so a resumed run cannot save something worse
        # than it started from, and the honest outcome when nothing improves is
        # that the ANCESTOR is the best model available.
        #
        # Copying it to the output path keeps the contract every caller relies
        # on -- a returned path that exists -- without pretending a worse model
        # was an improvement. Raising instead made a short resumed run fail
        # whenever it happened not to improve, which is data-dependent: two
        # tests passed on one machine and failed on another for exactly this.
        import shutil as _shutil
        if Path(resume_from).resolve() != Path(checkpoint_path).resolve():
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(resume_from, checkpoint_path)
        print(f"\nstage 1: no epoch improved on the ancestor "
              f"(reference val_loss {tracker.reference_val_loss if tracker.reference_val_loss is not None else float('nan'):.6f}), so the ancestor "
              f"remains the best model and has been kept as {checkpoint_path}. This is a "
              f"real outcome, not an error -- the run simply found nothing better.")
    elif not Path(checkpoint_path).exists():
        # Same guard as train_stage2/train_lds. Without it this returns a path
        # that does not exist and the caller fails far from the cause -- the
        # pipeline feeds stage 1's straight to check_reconstruction, and stage
        # 4's to stage 5.
        reason = ("epochs=0, so the epoch loop never ran and nothing was saved"
                  if epochs == 0 else
                  "no epoch's val criterion beat the running best")
        raise RuntimeError(
            f"stage 1 finished without ever saving a checkpoint to "
            f"{checkpoint_path}: {reason}. An epochs=0 ablation cannot produce a "
            f"checkpoint -- remove the stage from the params file rather than "
            f"setting its epochs to 0, if anything downstream needs its output."
        )
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
    parser.add_argument("--vram-log-every", type=int, default=0,
                         help="print a VRAM breakdown every N training batches (0 = never). Use "
                              "when an OOM arrives partway through an epoch: allocated flat with "
                              "reserved growing means fragmentation, allocated growing means a "
                              "leak, both flat with little free means another process owns the card")
    parser.add_argument("--cache-in-memory", action=argparse.BooleanOptionalAction, default=True,
                         help="hold every decoded snapshot in RAM (default). --no-cache-in-memory "
                              "from 128x128 up: the cache is ~2.3 GB at 128 and ~9.4 GB at 256, "
                              "duplicated into each of --num-workers DataLoader workers on "
                              "Windows, and building it is a serial single-threaded read with the "
                              "GPU idle -- so disabling it usually makes startup FASTER as well "
                              "as leaner")
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
        num_workers=args.num_workers, cache_in_memory=args.cache_in_memory,
        vram_log_every=args.vram_log_every,
        min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
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
