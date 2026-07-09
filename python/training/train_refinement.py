"""
Stage 4/5's epoch loop -- the same shape every other stage's training
loop already has (dataset -> per-epoch train/val -> criterion tracker
-> checkpoint save), but the first one with TWO ancestor checkpoints
(stage 2's for E/D/stats_head, stage 3's for f) instead of one, and the
first one where the encoder is trainable during rollout training.

One function covers both stages (freeze_decoder selects which), same
pattern as train_lds()'s 3a/3b curriculum sharing one function.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.checkpoint_components import assemble_joint_checkpoint, load_joint_refinement_checkpoint
from training.checkpoint_criterion import CheckpointCriterionTracker
from training.datasets import MicrostructureEvolutionDataset, complete_run_dirs, split_run_dirs
from training.losses import StatsLoss
from training.model_assembly import build_models_from_components
from training.refinement_loss import compute_stage45_loss
from utils.plots import loss_curve

_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/training/train_refinement.py -> python/


def train_refinement(
    base_path: Path, freeze_decoder: bool,
    ae_checkpoint_path: Path | None = None, lds_checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
    rollout_weight: float = 1.0, recon_weight: float = 0.0, stats_weight: float = 0.0,
    epochs: int = 100, batch_size: int = 32, lr: float = 1e-4,
    val_fraction: float = 0.2, test_fraction: float = 0.1, num_workers: int = 0,
    n_rollout_steps: int = 1, min_step: int | None = None, min_stdev_phi: float | None = None,
    val_ema_decay: float = 0.7, ema_warmup_epochs: int = 0,
    early_stopping_patience: int | None = None, grad_clip: float = 1.0,
    seed: int = 0, checkpoint_path: Path | None = None, device: str | None = None,
    on_checkpoint_saved=None, log_every_epoch: bool = True,
    loss_curve_path: Path | None = None,
) -> Path:
    """
    Stage 4 (freeze_decoder=True: D stays fixed, used only as a tether
    for L_recon -- see model_assembly.build_models_from_components) or
    stage 5 (freeze_decoder=False: D trains too). Which term dominates
    the objective is entirely a weight choice (stage 4:
    rollout_weight=1, recon_weight=small; stage 5: recon_weight=1,
    rollout_weight=small), not a structural difference -- see
    compute_stage45_loss.

    TWO mutually exclusive ways to get the starting E/D/stats_head/f:
      - ae_checkpoint_path + lds_checkpoint_path (stage 4's normal path):
        merges two SEPARATE ancestors (stage 2's checkpoint, stage 3's)
        via checkpoint_components.assemble_joint_checkpoint, which
        validates they're actually compatible before anything else
        happens with them.
      - resume_from (stage 5's normal path): continues a PREVIOUS stage
        4/5 joint checkpoint -- one that already holds E, D, f (and
        stats_head) TOGETHER, produced by this same function -- via
        checkpoint_components.load_joint_refinement_checkpoint. The
        original stage-2/stage-3 ancestors' provenance is carried
        forward from that checkpoint rather than lost.
    Give exactly one of these two options, not both, not neither.

    stats_weight > 0 only has an effect if the ancestor AE actually has
    a stats_head (trained with stats_weight > 0 back in stage 1) --
    otherwise L_stats is silently skipped (with a printed warning),
    matching compute_stage45_loss's own graceful-skip behavior rather
    than raising, since "no stats_head available" is a property of the
    ancestor checkpoint the caller may not control.

    ema_warmup_epochs defaults to 0 (no separate warmup phase, EMA from
    epoch 1) -- unlike train_lds()'s default of 5. Stage 4/5 starts from
    an already-converged encoder AND an already-converged f_theta (both
    ancestors were fully trained), unlike stage 3b which resumes a
    reasonable f_theta into unfamiliar multi-step territory but had NO
    prior anchor at all before that; there's less reason to expect the
    same wild early-epoch noise a genuine warmup phase was built for.
    Raise this if empirically that assumption turns out wrong.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    if min_step is None:
        raise ValueError("train_refinement() requires min_step to be given explicitly -- "
                          "config.txt no longer provides ML training defaults. "
                          "(min_stdev_phi may legitimately be None -- that means no "
                          "stdev-based filtering at all, not 'forgotten'.)")

    has_ancestors = ae_checkpoint_path is not None and lds_checkpoint_path is not None
    has_partial_ancestors = (ae_checkpoint_path is not None) != (lds_checkpoint_path is not None)
    if has_partial_ancestors:
        raise ValueError("train_refinement() got only one of ae_checkpoint_path/"
                          "lds_checkpoint_path -- give both together, or neither (with "
                          "resume_from instead).")
    if has_ancestors == (resume_from is not None):
        raise ValueError("train_refinement() needs EXACTLY ONE of (ae_checkpoint_path + "
                          "lds_checkpoint_path) OR resume_from -- got "
                          f"{'both' if has_ancestors else 'neither'}.")

    if resume_from is not None:
        components = load_joint_refinement_checkpoint(resume_from, device=device)
        ancestor_note = f"resumed from {resume_from}"
        # Provenance carried forward from whatever checkpoint resume_from
        # itself points at, so the full chain back to stage 2/3 stays
        # traceable, not just the immediate parent.
        ae_checkpoint_str = components["lds"].provenance.get("ae_checkpoint")
        lds_checkpoint_str = components["lds"].provenance.get("lds_checkpoint")
    else:
        components = assemble_joint_checkpoint(ae_checkpoint_path, lds_checkpoint_path, device=device)
        ancestor_note = f"E/D{'/stats_head' if 'stats_head' in components else ''} from " \
                         f"{ae_checkpoint_path}, f_theta from {lds_checkpoint_path}"
        ae_checkpoint_str = str(Path(ae_checkpoint_path).resolve())
        lds_checkpoint_str = str(Path(lds_checkpoint_path).resolve())
    ae, stats_head, f_theta, frozen_modules = build_models_from_components(
        components, device=device, freeze_decoder=freeze_decoder,
    )
    size = components["encoder"].config["size"]
    print(f"Stage {'4' if freeze_decoder else '5'}: loaded {ancestor_note}")
    print(f"size={size}, latent_channels={components['encoder'].config['latent_channels']}, "
          f"freeze_decoder={freeze_decoder}")
    print(f"rollout_weight={rollout_weight}  recon_weight={recon_weight}  "
          f"stats_weight={stats_weight}")
    print(f"min_step={min_step}  min_stdev_phi={min_stdev_phi}  n_rollout_steps={n_rollout_steps}\n")

    stats_loss_fn = None
    stat_names = None
    if stats_weight > 0:
        if stats_head is None:
            print("WARNING: stats_weight > 0 but the ancestor AE has no stats_head "
                  "(trained with stats_weight <= 0 back in stage 1) -- L_stats will be "
                  "skipped entirely for this run.\n")
        else:
            sh = components["stats_head"]
            stat_names = sh.config["stat_names"]
            stats_loss_fn = StatsLoss(sh.provenance["stats_mean"], sh.provenance["stats_std"],
                                       stat_names=stat_names).to(device)

    window_length = n_rollout_steps + 1
    run_dirs = complete_run_dirs(base_path, size, size)
    if not run_dirs:
        raise ValueError(f"No complete runs found under {base_path}/{size}x{size} -- "
                          f"check base_path/size, or that metadata.txt exists there")
    train_dirs, val_dirs, test_dirs = split_run_dirs(run_dirs, val_fraction, test_fraction, seed=seed)

    train_set = MicrostructureEvolutionDataset(
        train_dirs, encoder=None, window_length=window_length,
        min_step=min_step, min_stdev_phi=min_stdev_phi, stat_names=stat_names,
    )
    val_set = MicrostructureEvolutionDataset(
        val_dirs, encoder=None, window_length=window_length,
        min_step=min_step, min_stdev_phi=min_stdev_phi, stat_names=stat_names,
    )
    print(f"{len(run_dirs)} complete runs -> {len(train_dirs)} train / {len(val_dirs)} val / "
          f"{len(test_dirs)} test dirs")
    print(f"{len(train_set)} train windows, {len(val_set)} val windows "
          f"(n_rollout_steps={n_rollout_steps}, window_length={window_length})\n")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=device.type == "cuda")

    # Trainable params: E always, f_theta always, D only if not frozen
    # (freeze_decoder already set requires_grad=False on D's parameters
    # at assembly time -- filtering here just keeps them out of the
    # optimizer's own state, not a second freezing mechanism).
    trainable_params = ([p for p in ae.parameters() if p.requires_grad]
                         + [p for p in f_theta.parameters() if p.requires_grad])
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    if checkpoint_path is None:
        stage_dir = "stage4" if freeze_decoder else "stage5"
        checkpoint_path = _PYTHON_ROOT / "checkpoints" / stage_dir / "refinement.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {checkpoint_path}")

    if loss_curve_path is None:
        stage_dir = "stage4" if freeze_decoder else "stage5"
        loss_curve_path = _PYTHON_ROOT.parent / "output" / stage_dir / "loss_curve.png"

    epoch_history: list[int] = []
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_so_far_history: list[float] = []

    def unpack(batch):
        if stat_names is not None:
            x_window, dt_window, theta, true_stats = batch
            return (x_window.to(device), dt_window.to(device), theta.to(device),
                    true_stats.to(device))
        x_window, dt_window, theta = batch
        return x_window.to(device), dt_window.to(device), theta.to(device), None

    def step(batch, train: bool) -> float:
        x_window, dt_window, theta, true_stats = unpack(batch)
        loss = compute_stage45_loss(
            ae, f_theta, stats_head, x_window, dt_window, theta,
            rollout_weight=rollout_weight, recon_weight=recon_weight, stats_weight=stats_weight,
            stats_loss_fn=stats_loss_fn, true_stats=true_stats,
        )
        if train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
        return loss.item()

    tracker = CheckpointCriterionTracker(ema_warmup_epochs=ema_warmup_epochs,
                                          val_ema_decay=val_ema_decay)
    epochs_since_improvement = 0

    print(f"Starting {epochs} epochs (early_stopping_patience: "
          f"{early_stopping_patience}, batches of {batch_size})...")
    print(f"/{epochs:3d}  train      valid        ema")

    for epoch in range(1, epochs + 1):
        ae.train()
        f_theta.train()
        # ae.train()/f_theta.train() are recursive and would otherwise
        # flip any frozen BatchNorm layers back to train mode every
        # epoch -- the exact bug fixed in stage 2's freeze_outer_layers,
        # same fix needed here for freeze_decoder.
        for m in frozen_modules:
            m.eval()

        train_loss = 0.0
        n_train = len(train_set)
        for batch in train_loader:
            bs = batch[0].size(0)
            train_loss += step(batch, train=True) * bs
        train_loss /= n_train

        ae.eval()
        f_theta.eval()
        val_loss = 0.0
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0)
                val_loss += step(batch, train=False) * bs
        val_loss /= n_val

        criterion, saved_this_epoch = tracker.update(epoch, val_loss)

        epoch_history.append(epoch)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        best_so_far_history.append(tracker.best_val_loss)
        loss_curve(
            epoch_history, train_loss_history, val_loss_history, best_so_far_history,
            loss_curve_path, title=f"Stage {'4' if freeze_decoder else '5'} loss",
        )

        ema_str = f"{tracker.val_ema:.6f}" if tracker.val_ema is not None else "  (warmup)"
        msg = f"{epoch:4d}  {train_loss:9.5f}  {val_loss:9.5f} |{ema_str:>10}"

        if saved_this_epoch:
            epochs_since_improvement = 0
            torch.save({
                "ae_state": ae.state_dict(),
                "f_theta_state": f_theta.state_dict(),
                "stats_head_state": stats_head.state_dict() if stats_head is not None else None,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_loss_ema": tracker.val_ema,
                "ae_checkpoint": ae_checkpoint_str,
                "lds_checkpoint": lds_checkpoint_str,
                **({"resumed_from": str(Path(resume_from).resolve())} if resume_from is not None else {}),
                "test_dirs": [str(Path(d).resolve()) for d in test_dirs],
                "config": dict(components["encoder"].config),
                "lds_config": dict(components["lds"].config),
                "data_config": {"min_step": min_step, "min_stdev_phi": min_stdev_phi,
                                "window_length": window_length, "n_rollout_steps": n_rollout_steps},
                "stats_config": (
                    {"stat_names": stat_names, "stats_mean": stats_loss_fn.mean.cpu(),
                     "stats_std": stats_loss_fn.std.cpu()}
                    if stats_loss_fn is not None else None
                ),
                "stage45_config": {
                    "freeze_decoder": freeze_decoder, "rollout_weight": rollout_weight,
                    "recon_weight": recon_weight, "stats_weight": stats_weight,
                    "n_rollout_steps": n_rollout_steps,
                },
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

    return checkpoint_path