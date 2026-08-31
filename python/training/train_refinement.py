"""
Stage 4/5's epoch loop -- the same shape every other stage's training
loop already has (dataset -> per-epoch train/val -> criterion tracker
-> checkpoint save), but the first one with TWO ancestor checkpoints
(stage 2's for E/D/stats_head, stage 3's for f) instead of one, and the
first one where the encoder is trainable during rollout training.

One function covers both stages (freeze_decoder selects which), same
pattern as train_lds()'s 3a/3b curriculum sharing one function.
"""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.checkpoint_components import cross_check_ancestor_config
from training.checkpoint_components import assemble_joint_checkpoint, load_joint_refinement_checkpoint
from training.spike_guard import (
    _SpikeGuard, _record_spike, difficulty_band, early_stop_message, end_epoch_pair,
    restore_running_stats, snapshot_running_stats, SkipReporter,
)
from utils.logging_utils import print_run_parameters, EpochProgress
from training.checkpoint_criterion import (
    CheckpointCriterionTracker, ComponentBestTracker, atomic_torch_save,
    clamp_grace_epochs, grace_epochs_for_ema,
)
from training.datasets import MicrostructureEvolutionDataset, complete_run_dirs, split_run_dirs
from training.losses import StatsLoss
from training.model_assembly import build_models_from_components
from training.refinement_loss import compute_stage45_loss
from utils.plots import loss_component_scatter, loss_curve, write_loss_history, should_write_loss_figure

_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/training/train_refinement.py -> python/


def linear_warmup_weight(epoch: int, full_weight: float, warmup_epochs: int,
                          start_fraction: float) -> float:
    """`full_weight` ramped LINEARLY from start_fraction to 1 over warmup_epochs.

    A module-level function rather than an expression inline in the epoch
    loop, so a test can exercise the ACTUAL formula. An inline version forced
    the test to re-implement it, and an off-by-one mutation in the production
    copy then left every endpoint assertion green -- the test was checking its
    own arithmetic.

    Linear, not geometric. This USED to be geometric, on the argument that
    L_rollout collapsed ~6e9 over the first ten epochs (1.76e9 -> 0.29,
    measured) so a linear weight ramp left epoch 1 eight decades above the
    converged contribution. That collapse was itself an artefact of the
    filter-manufactured large-dt windows (du_max=2.5e4): require_consecutive
    now excludes those at the window definition, and with the scales
    recalibrated L_rollout sits at O(1-10) from epoch 1, not O(1e9). Nothing
    left to hold flat -- so the ramp is a plain linear introduction of the
    term, and the same function serves any warmed-in weight (rollout,
    recon_predict) rather than encoding one term's obsolete transient.

    epoch is 1-based: epoch 1 gives exactly start_fraction*full_weight, epoch
    warmup_epochs and beyond give exactly full_weight.
    """
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return full_weight
    frac = (epoch - 1) / max(1, warmup_epochs - 1)
    return full_weight * (start_fraction + (1.0 - start_fraction) * frac)


_REFINEMENT_PREAMBLE_PARAMS = (
    # See _LDS_PREAMBLE_PARAMS for why these are excluded rather than repeated.
    "freeze_decoder", "size", "rollout_weight_warmup_epochs", "rollout_weight_warmup_start",
    "recon_predict_weight_warmup_epochs", "recon_predict_weight_warmup_start",
    "max_dt", "rollout_weight", "recon0_weight", "stats0_weight", "recon_predict_weight",
    "rollout_scale", "recon0_scale", "stats0_scale", "recon_predict_scale",
    "epochs", "batch_size", "n_rollout_steps",
    "min_step", "min_stdev_phi", "min_normalized_stdev_phi", "early_stopping_patience",
)


def train_refinement(
    base_path: Path, freeze_decoder: bool, size: int | None = None,
    rollout_weight_warmup_epochs: int = 0,
    rollout_weight_warmup_start: float = 1e-6,
    recon_predict_weight_warmup_epochs: int = 0,
    recon_predict_weight_warmup_start: float = 1e-6,
    max_dt: float | None = None, min_passing_steps: int | None = None,
    ae_checkpoint_path: Path | None = None, lds_checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
    rollout_weight: float = 1.0, recon0_weight: float = 0.0, stats0_weight: float = 0.0,
    recon_predict_weight: float = 0.0,
    rollout_scale: float = 1.0, recon0_scale: float = 1.0, stats0_scale: float = 1.0,
    recon_predict_scale: float = 1.0,
    epochs: int = 100, batch_size: int = 32, lr: float = 1e-4,
    val_fraction: float = 0.2, test_fraction: float = 0.1, num_workers: int = 0,
    n_rollout_steps: int | None = None, min_step: int | None = None, min_stdev_phi: float | None = None,
    min_normalized_stdev_phi: float | None = None,
    val_ema_decay: float = 0.7, ema_warmup_epochs: int = 0,
    early_stopping_patience: int | None = None, grad_clip: float = 1.0,
    spike_skip_factor: float = 10.0, grad_spike_factor: float = 10.0,
    seed: int = 0, checkpoint_path: Path | None = None, device: str | None = None,
    on_checkpoint_saved=None, log_every_epoch: bool = True,
    loss_curve_path: Path | None = None,
) -> Path:
    """
    Stage 4 (freeze_decoder=True: D stays fixed, used only as a tether
    for L_recon -- see model_assembly.build_models_from_components) or
    stage 5 (freeze_decoder=False: D trains too). Which term dominates
    the objective is entirely a weight choice (stage 4:
    rollout_weight=1, recon0_weight=small; stage 5: recon0_weight=1,
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

    stats0_weight > 0 only has an effect if the ancestor AE actually has
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
    ae, stats_head, f_theta, frozen_modules, stream_configs, recon_stream_name = build_models_from_components(
        components, device=device, freeze_decoder=freeze_decoder,
    )
    # Same hazard as stage 2's: this `size` decides both the architecture and
    # WHICH DATASET is read (complete_run_dirs(base_path, size, size) below),
    # while the output filename comes from the params file. Stage 4/5 take TWO
    # ancestors, so both are checked -- an E/D from one size and an f_theta
    # from another is a mismatch nothing else would report.
    cross_check_ancestor_config(components["encoder"].config, {"size": size},
                                 ae_checkpoint_path or resume_from, what="encoder ancestor")
    size = components["encoder"].config["size"]

    # u-scheme: log10_t f_theta is now supported -- the stage-4 dataset emits
    # per-frame physical t (return_frame_t) and compute_stage45_loss converts
    # z1->z̃1 and dt->Delta-u before the rollout (mirrors the stage-3 dataset).
    _needs_frame_t = getattr(f_theta, "time_coordinate", "t") == "log10_t"
    print(f"Stage {'4' if freeze_decoder else '5'}: loaded {ancestor_note}")
    print(f"size={size}, latent_channels={components['encoder'].config['latent_channels']}, "
          f"freeze_decoder={freeze_decoder}")
    if rollout_weight_warmup_epochs > 0:
        print(f"rollout_weight ramped LINEARLY from {rollout_weight_warmup_start:g}x to 1x "
              f"over {rollout_weight_warmup_epochs} epoch(s). This was geometric while L_rollout "
              f"collapsed ~6e9 over the first epochs (a linear ramp then left epoch 1 eight "
              f"decades hot); require_consecutive removed the large-dt windows that caused that "
              f"collapse, so L_rollout is O(1-10) from epoch 1 and a plain linear introduction "
              f"suffices. The warmup still exists because f_theta is FROZEN and the encoder "
              f"starts on exactly the distribution it was fitted to, so a full-strength rollout "
              f"gradient at epoch 1 drives the encoder off that distribution faster than "
              f"recon0/stats0 can anchor it. VAL is never ramped, so val_loss stays comparable "
              f"across epochs and against runs without a warmup.")
    if recon_predict_weight_warmup_epochs > 0:
        print(f"recon_predict_weight ramped LINEARLY from "
              f"{recon_predict_weight_warmup_start:g}x to 1x over "
              f"{recon_predict_weight_warmup_epochs} epoch(s). recon_predict decodes the "
              f"rolled-out endpoint and backprops a full-weight pixel loss through the rollout "
              f"into a decoder that (resuming from stage 4) only ever saw frame-0 latents; the "
              f"ramp lets the decoder settle on frame-0 reconstruction before being pulled "
              f"toward rendering the drifted endpoint. VAL is never ramped.")
    print(f"rollout_weight={rollout_weight}  recon0_weight={recon0_weight}  "
          f"stats0_weight={stats0_weight}  recon_predict_weight={recon_predict_weight}")
    print(f"min_step={min_step}  min_stdev_phi={min_stdev_phi}  "
          f"min_normalized_stdev_phi={min_normalized_stdev_phi}  n_rollout_steps={n_rollout_steps}")
    print_run_parameters(train_refinement, locals(), _REFINEMENT_PREAMBLE_PARAMS)
    print()

    stats_loss_fn = None
    stat_names = None
    if stats0_weight > 0:
        if stats_head is None:
            print("WARNING: stats0_weight > 0 but the ancestor AE has no stats_head "
                  "(trained with stats_weight <= 0 back in stage 1) -- L_stats will be "
                  "skipped entirely for this run.\n")
        else:
            sh = components["stats_head"]
            stat_names = sh.config["stat_names"]
            stats_loss_fn = StatsLoss(sh.provenance["stats_mean"], sh.provenance["stats_std"],
                                       stat_names=stat_names).to(device)

    run_dirs = complete_run_dirs(base_path, size, size)
    if not run_dirs:
        raise ValueError(f"No complete runs found under {base_path}/{size}x{size} -- "
                          f"check base_path/size, or that metadata.txt exists there")
    train_dirs, val_dirs, test_dirs = split_run_dirs(run_dirs, val_fraction, test_fraction, seed=seed)

    # max_dt and min_passing_steps INHERITED from the f_theta being
    # refined, unless the caller states otherwise. A stage that consumes
    # f_theta must reproduce the window population f_theta was trained on:
    # its correction goes as f*dt^2/2, so applying an f_theta fitted at
    # dt <= 200 across dt up to 25000 inflates that term 15625x, z1 then
    # propagates it through the sub-steps and the rollout chains it. The
    # first stage-4 run without this reported val_loss = 2.7e29, with
    # recon0 (0.21) and stats0 (25.1) sane beside it -- so nothing but the
    # rollout term was wrong, and nothing said why.
    #
    # Same rule already applied to check_parameter_dependence and
    # check_rollout. This was missed because stage 4 is a TRAINING stage,
    # and the test enforcing the rule only covered diagnostics.
    lds_data_config = components["lds"].provenance.get("data_config") or {}
    # The rollout REGIME, alongside max_dt above. f_theta trained with
    # z1_resync=False expects z1 to be propagated, not reset at each real
    # frame; applying it teacher-forced is the "NOT equivalent" direction.
    lds_z1_resync = components["lds"].config.get("z1_resync", True)
    # n_rollout_steps INHERITED too, and for the same reason as max_dt and
    # z1_resync: it is the regime f_theta was tuned in, not a free choice.
    #
    # It used to default to 1 no matter what the ancestor trained at. Measured
    # on a full-chain run: 3b recorded n_rollout_steps=2 and stage 4 silently
    # used 1 -- and at ONE step there is nothing to propagate, so the inherited
    # z1_resync=False went inert as well. The encoder was refined in a regime
    # f_theta was never tuned for, with no message at all.
    #
    # None (not 1) is the "unspecified" sentinel: 1 is a meaningful explicit
    # value and must stay overridable.
    _inherited_rollout = lds_data_config.get("n_rollout_steps")
    if n_rollout_steps is None:
        n_rollout_steps = _inherited_rollout if _inherited_rollout is not None else 1
        if _inherited_rollout is not None:
            print(f"n_rollout_steps={n_rollout_steps} inherited from f_theta's own training "
                  f"regime (pass n_rollout_steps explicitly to override)")
    elif _inherited_rollout is not None and n_rollout_steps != _inherited_rollout:
        print(f"NOTE: this stage runs n_rollout_steps={n_rollout_steps} but f_theta was "
              f"trained at {_inherited_rollout}. Deliberate is fine -- but at 1 step "
              f"z1_resync has nothing to propagate, so a rollout-trained f_theta is being "
              f"applied outside its own regime.")
    # Computed HERE, after n_rollout_steps is resolved -- it used to sit ~35
    # lines earlier, which made the None sentinel a TypeError.
    window_length = n_rollout_steps + 1
    max_dt = max_dt if max_dt is not None else lds_data_config.get("max_dt")
    min_passing_steps = (min_passing_steps if min_passing_steps is not None
                          else lds_data_config.get("min_passing_steps"))
    # Inherited alongside min_passing_steps: it is the filter that count is taken
    # against (raw min_stdev_phi OR this normalized one), and the encoder must be
    # refined on the SAME window population f_theta was trained on. Params may
    # still override; the mismatch report below covers it like min_stdev_phi.
    min_normalized_stdev_phi = (min_normalized_stdev_phi
                                if min_normalized_stdev_phi is not None
                                else lds_data_config.get("min_normalized_stdev_phi"))
    if max_dt is not None:
        print(f"max_dt={max_dt} inherited from f_theta's own training window "
              f"population (pass max_dt explicitly to override)")
    # min_step / min_stdev_phi are NOT inherited -- they come from the params
    # file at every stage, deliberately: unlike max_dt a mismatch is mild (a
    # different slice of eligible frames) rather than catastrophic (f_theta
    # applied where f*dt^2/2 is orders of magnitude beyond anything it saw).
    # But nothing checked them either, so a per-stage override would silently
    # refine the encoder against a different population than f_theta trained
    # on. Reported, not enforced.
    for _field, _here in (("min_step", min_step), ("min_stdev_phi", min_stdev_phi),
                          ("min_normalized_stdev_phi", min_normalized_stdev_phi)):
        _there = lds_data_config.get(_field, "<absent>")
        if _there != "<absent>" and _there != _here:
            print(f"NOTE: {_field}={_here} differs from f_theta's own {_there}. Not an error "
                  f"-- unlike max_dt this only shifts which frames are eligible, not how far "
                  f"f_theta is extrapolated -- but the encoder is being refined against a "
                  f"different window population than f_theta was trained on.")

    if epochs == 0:
        # Ablation mode: no training happens (see the epoch loop
        # below), so train_set/train_loader would never be touched --
        # skipped entirely, same rationale as the other four training
        # functions' own fix.
        train_set = train_loader = None
        val_set = MicrostructureEvolutionDataset(
            val_dirs, encoder=None, window_length=window_length,
            min_step=min_step, min_stdev_phi=min_stdev_phi, stat_names=stat_names,
            max_dt=max_dt, min_passing_steps=min_passing_steps,
            min_normalized_stdev_phi=min_normalized_stdev_phi,
            split_label="validation", return_frame_t=_needs_frame_t,
        )
        print(f"{len(run_dirs)} complete runs -> {len(train_dirs)} train / {len(val_dirs)} val / "
              f"{len(test_dirs)} test dirs")
        print(f"train_set: skipped (epochs=0 ablation -- never iterated over), "
              f"{len(val_set)} val windows (n_rollout_steps={n_rollout_steps}, "
              f"window_length={window_length})\n")
    else:
        train_set = MicrostructureEvolutionDataset(
            train_dirs, encoder=None, window_length=window_length,
            min_step=min_step, min_stdev_phi=min_stdev_phi, stat_names=stat_names,
            max_dt=max_dt, min_passing_steps=min_passing_steps,
            min_normalized_stdev_phi=min_normalized_stdev_phi,
            split_label="training", return_frame_t=_needs_frame_t,
        )
        val_set = MicrostructureEvolutionDataset(
            val_dirs, encoder=None, window_length=window_length,
            min_step=min_step, min_stdev_phi=min_stdev_phi, stat_names=stat_names,
            max_dt=max_dt, min_passing_steps=min_passing_steps,
            min_normalized_stdev_phi=min_normalized_stdev_phi,
            split_label="validation", return_frame_t=_needs_frame_t,
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
    loss_components_path = loss_curve_path.with_name(
        loss_curve_path.stem + "-components" + loss_curve_path.suffix)

    loss_curve_events = []

    epoch_history: list[int] = []
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_so_far_history: list[float] = []
    # loss_component_scatter's own bookkeeping -- rollout/recon0/stats0
    # are all always active in this stage (no conditional gating like
    # stage 1's include_stats or stage 2's active_terms).
    component_histories: dict[str, dict[str, list[float]]] = {
        name: {"train": [], "val": [], "best_so_far": []}
        for name in ("rollout", "recon0", "stats0", "recon_predict")
    }
    component_best_tracker = ComponentBestTracker()

    def unpack(batch):
        if stat_names is not None:
            if len(batch) == 5:      # return_frame_t on (u-scheme): + per-frame t
                x_window, dt_window, theta, true_stats, t_window = batch
            else:
                x_window, dt_window, theta, true_stats = batch
                t_window = None
            return (x_window.to(device), dt_window.to(device), theta.to(device),
                    true_stats.to(device),
                    t_window.to(device) if t_window is not None else None)
        x_window, dt_window, theta = batch
        return x_window.to(device), dt_window.to(device), theta.to(device), None, None

    def step(batch, train: bool, effective_rollout_weight: float | None = None,
             effective_recon_predict_weight: float | None = None):
        x_window, dt_window, theta, true_stats, t_window = unpack(batch)
        # BEFORE the forward, because the forward is what moves the buffers.
        # A skipped batch must leave the model exactly as it found it, and
        # "the optimizer step was not taken" only covers PARAMETERS -- see
        # snapshot_running_stats for the stage-4 run where 487/487 skipped
        # batches still moved val_loss by 1.2%.
        _bn_snapshot = snapshot_running_stats(ae, f_theta) if train else None
        loss, components = compute_stage45_loss(
            ae, f_theta, stats_head, x_window, dt_window, theta,
            rollout_weight=(rollout_weight if effective_rollout_weight is None
                             else effective_rollout_weight),
            recon0_weight=recon0_weight, stats0_weight=stats0_weight,
            recon_predict_weight=(recon_predict_weight
                                   if effective_recon_predict_weight is None
                                   else effective_recon_predict_weight),
            rollout_scale=rollout_scale, recon0_scale=recon0_scale, stats0_scale=stats0_scale,
            recon_predict_scale=recon_predict_scale,
            stats_loss_fn=stats_loss_fn, true_stats=true_stats,
            recon_stream_name=recon_stream_name, return_components=True,
            z1_resync=lds_z1_resync, t_window=t_window,
        )
        if train:
            # SAME MACHINERY AS STAGE 3, for the same observed failure.
            # Stages 4/5 spike too and had no protection:
            #   stage 4: epoch 37 train 0.3647 -> epoch 38 train 302,890.5
            #   stage 5: epoch 20 train 1.4757 -> epoch 21 train 2,838.0
            # with val barely moving in both -- so the weights survived and a
            # few train batches had exploded. Those runs got lucky.
            #
            # At ~135 batches per epoch (vs stage 3's 7) a single bad batch is
            # 1e5-1e8x the median, so a factor-10 threshold catches it with an
            # enormous margin over ordinary variation.
            # See train_lds: per-band comparison, so a bucketed batch is
            # judged against batches of similar difficulty rather than against
            # the whole population's median.
            _band = difficulty_band(float(dt_window.detach().max()))
            if spike_guard.should_skip(float(loss.detach()), band=_band):
                _record_spike(spike_guard, loss, dt_window, theta)
                optimizer.zero_grad()
                restore_running_stats(_bn_snapshot)
            else:
                optimizer.zero_grad()
                loss.backward()
                # The pre-clip norm, which clip_grad_norm_ returns anyway.
                # Catches the ordinary-loss/huge-gradient case the loss guard
                # structurally cannot see (measured on stage 3a).
                _gnorm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, grad_clip if grad_clip > 0 else float("inf"))
                # See train_lds: reached only when the loss guard passed,
                # so the loss is ordinary by construction.
                if grad_guard.should_skip(float(_gnorm), band=_band,
                                           loss_was_ordinary=True):
                    _record_spike(grad_guard, _gnorm, dt_window, theta)
                    optimizer.zero_grad()
                    # The gradient path ran the SAME forward, so it moved the
                    # same buffers -- both skip branches need the restore, and
                    # this one is the easier to forget because the loss looked
                    # ordinary.
                    restore_running_stats(_bn_snapshot)
                else:
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
        # attached. Same fix as train_lds.py's/train_stage1.py's/
        # train_stage2.py's own step() -- see any of those functions'
        # own identical comment.
        return (loss.detach(), components["rollout"].detach(),
                components["recon0"].detach(), components["stats0"].detach(),
                components["recon_predict"].detach())

    # Whether THIS run has written the checkpoint at least once -- train_lds
    # tracks the same thing for its no-save guard; here it feeds
    # early_stop_message, so an exit with zero saves is reported as the
    # never-improved event it is rather than as ordinary patience.
    _saved_this_run = False
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=ema_warmup_epochs,
                                          val_ema_decay=val_ema_decay)
    epochs_since_improvement = 0
    longest_gap = 0            # longest no-save stretch this run RECOVERED from
    longest_gap_range = None   # (first, last) epochs it spanned

    print(f"Starting {epochs} epochs (early_stopping_patience: "
          f"{early_stopping_patience}, batches of {batch_size})...")
    print(f"/{epochs:3d} train = {rollout_weight}*rollout/{rollout_scale} "
          f"+{recon0_weight}*recon0/{recon0_scale} "
          f"+{stats0_weight}*stats0/{stats0_scale} "
          f"+{recon_predict_weight}*recon_predict/{recon_predict_scale} | valid = ...  | ema")

    spike_guard = _SpikeGuard(spike_skip_factor)
    # Separate history: gradient norms and losses live on different scales.
    grad_guard = _SpikeGuard(grad_spike_factor)
    _spikes_reported = 0
    _grad_spikes_reported = 0
    _nonfinite_reported = 0
    # Same digesting reporter stages 3 uses, so stage 4/5 gets the SAME merged
    # single-block message when BOTH guards fire in an epoch (was two separate
    # near-identical blocks) and the compact one-line digest for routine skips.
    _skip_reporter = SkipReporter()
    _n_train_batches = 0

    _prev_val_seconds = None   # previous epoch's validation duration, added to
    #                            the training bar's ETA so it reflects the WHOLE
    #                            epoch (shown as "+ validation" on the first).
    for epoch in range(0 if epochs == 0 else 1, epochs + 1):
        ae.train()
        f_theta.train()
        # ae.train()/f_theta.train() are recursive and would otherwise
        # flip any frozen BatchNorm layers back to train mode every
        # epoch -- the exact bug fixed in stage 2's freeze_outer_layers,
        # same fix needed here for freeze_decoder.
        for m in frozen_modules:
            m.eval()

        # GPU-resident accumulators, not Python floats -- see step()'s
        # own docstring/comment: the ONLY host sync per phase (train/
        # val) is the batch of four .item() calls after each loop
        # ends, not four per batch.
        # ROLLOUT WEIGHT RAMP, mirroring stage 2's deriv_weight_warmup_epochs
        # and needed here for a sharper reason: stage 2's new stream can adapt,
        # while stage 4's f_theta is FROZEN and cannot.
        #
        # The encoder starts at exactly stage 2's optimum -- which is the
        # distribution f_theta was fitted to. A full-strength rollout gradient
        # immediately pulls it OFF that distribution, which raises the rollout
        # loss, which pulls harder: positive feedback, and f*dt^2/2 makes it
        # quadratic. Measured on a real run, the raw rollout fell 6e9 between
        # epoch 1 and epoch 10 (1.76e9 -> 0.29), so NO single rollout_scale
        # serves both ends -- calibrated on the converged value it makes epoch
        # 1 explosive, calibrated on epoch 1 it makes the term irrelevant
        # later. rollout_scale=1 stalled the run outright; 10 was the largest
        # gradient the transient tolerated.
        #
        # The ramp separates those two jobs: recon0 and stats0 hold the encoder
        # (they have the decoder and the statistics as tethers, and their
        # train/val gap stayed under 2x while rollout's hit 83x) while the
        # rollout term comes up gradually. rollout_scale can then be set from
        # the CONVERGED magnitude, which is what it is for.
        #
        # Lowering lr instead would damp the transient AND everything after it.
        # GEOMETRIC, not linear. Linear is what stage 2 uses for
        # deriv_weight, and it is wrong here because the quantity being ramped
        # behaves completely differently: L_deriv is O(1) from the start, while
        # L_rollout collapses by ~6e9 over the first ten epochs
        # (1.76e9 -> 0.29, measured). A linear ramp at epoch 1 gives
        # 0.10 * 1.76e9 = 1.76e8 -- still eight orders of magnitude above the
        # converged contribution, so it barely softens the transient it exists
        # to absorb.
        #
        # Linear from rollout_weight_warmup_start to 1 over the window
        # tracks the collapse instead: 1e-6 * 1.76e9 = 1.8e3 at epoch 1, then
        # ~0.34 at epochs 2 and 5, then 0.29 at epoch 10. The CONTRIBUTION is
        # roughly flat across the ramp, which is the actual goal -- a constant
        # gradient scale while the encoder settles, rather than a constant
        # weight applied to a wildly varying loss.
        effective_rollout_weight = linear_warmup_weight(
            epoch, rollout_weight, rollout_weight_warmup_epochs, rollout_weight_warmup_start)
        # Same linear ramp for recon_predict: it backprops a full-weight pixel
        # loss through the rollout into a decoder that (resuming from stage 4)
        # only ever saw frame-0 latents, so it warms in to let the decoder
        # settle on frame-0 reconstruction before being pulled toward rendering
        # the drifted endpoint. Default off (0 warmup epochs) -> full weight
        # from epoch 1, the weights as specified.
        effective_recon_predict_weight = linear_warmup_weight(
            epoch, recon_predict_weight, recon_predict_weight_warmup_epochs,
            recon_predict_weight_warmup_start)

        # THE END OF THE RAMP RESETS THE SAVE CRITERION.
        #
        # val_loss is deliberately never ramped (see the step() call below):
        # it must stay comparable across epochs and against runs with no
        # warmup. The consequence is that DURING the ramp the model is not yet
        # optimising the full objective, so its val_loss under that objective
        # is legitimately terrible -- and feeding those values to the tracker
        # leaves it descending from a number describing a model that no longer
        # exists.
        #
        # Observed on a real run: epoch 1's val_loss was 33.86 (a ramp
        # transient), and by epoch 13 the EMA was still at 0.656 -- still
        # mostly forgetting that transient -- while the actual val_loss had
        # stopped improving at epoch 11 and was oscillating in 0.10-0.12. Every
        # epoch cleared the descending bar and saved, so the criterion was
        # blind and early stopping could not fire either.
        #
        # Exactly the situation reset_with_grace exists for: stage 2 calls it
        # when deriv_target_centered switches, because "val_loss computed under
        # the OLD target isn't a fair bar for the NEW target's own val_loss to
        # clear". A weight ramp is the same change spread over several epochs.
        if (max(rollout_weight_warmup_epochs, recon_predict_weight_warmup_epochs) > 0
                and epoch == max(rollout_weight_warmup_epochs,
                                 recon_predict_weight_warmup_epochs)):
            # max(2, ...) not max(1, ...): a single-epoch grace is
            # mathematically identical to no grace at all, since best_val_loss
            # then ends up as that one epoch's own raw value, lucky or not.
            # Same derivation as stage 2's, from val_ema_decay's own averaging
            # window.
            grace = grace_epochs_for_ema(val_ema_decay)
            grace = clamp_grace_epochs(grace, epochs - epoch + 1)
            if grace > 0:
                print(f"  [epoch {epoch}: rollout_weight has reached full strength -- giving "
                      f"the tracker a {grace}-epoch grace period before comparing again, since "
                      f"val_loss recorded DURING the ramp described a model trained on a "
                      f"different objective and is not a fair bar for the full one to clear]")
            # Same marking as stage 2's target switch: the criterion reset
            # makes "best EMA so far" jump discontinuously, and the ramp
            # completing changes what the TRAIN column measures.
            loss_curve_events.append((epoch - 0.5, "rollout ramp complete"))
            tracker.reset_with_grace(grace)

        train_loss_sum = torch.zeros((), device=device)
        train_rollout_sum = torch.zeros((), device=device)
        train_recon0_sum = torch.zeros((), device=device)
        train_stats0_sum = torch.zeros((), device=device)
        train_recon_predict_sum = torch.zeros((), device=device)
        if epoch > 0:
            n_train = len(train_set)
            _n_train_batches = 0
            _epoch_progress = EpochProgress(
                len(train_loader),
                tail_label="validation", tail_seconds=_prev_val_seconds)
            for batch in train_loader:
                _epoch_progress.tick()
                _n_train_batches += 1
                bs = batch[0].size(0)
                loss, rollout, recon0, stats0, recon_predict = step(
                    batch, train=True, effective_rollout_weight=effective_rollout_weight,
                    effective_recon_predict_weight=effective_recon_predict_weight)
                train_loss_sum += loss * bs
                train_rollout_sum += rollout * bs
                train_recon0_sum += recon0 * bs
                train_stats0_sum += stats0 * bs
                train_recon_predict_sum += recon_predict * bs
            _epoch_progress.close()
            train_loss = (train_loss_sum / n_train).item()
            train_rollout = (train_rollout_sum / n_train).item()
            train_recon0 = (train_recon0_sum / n_train).item()
            train_stats0 = (train_stats0_sum / n_train).item()
            train_recon_predict = (train_recon_predict_sum / n_train).item()
        else:
            # epoch 0 (epochs=0 ablation only): no training at all --
            # NaN honestly reflects that these metrics don't apply this
            # "epoch", rather than a misleading 0.0.
            train_loss = train_rollout = train_recon0 = train_stats0 = float("nan")
            train_recon_predict = float("nan")

        ae.eval()
        f_theta.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_rollout_sum = torch.zeros((), device=device)
        val_recon0_sum = torch.zeros((), device=device)
        val_stats0_sum = torch.zeros((), device=device)
        val_recon_predict_sum = torch.zeros((), device=device)
        n_val = len(val_set)
        # Stage 4's validation runs a full val_loader pass after training with
        # no output -- looks hung on a large sweep. Bar it (self-gating delay)
        # and time it so the NEXT epoch's training ETA can include it.
        _val_prog = EpochProgress(len(val_loader), label="validation",
                                  unit="batches")
        _val_t0 = time.monotonic()
        with torch.no_grad():
            for batch in val_loader:
                _val_prog.tick()
                bs = batch[0].size(0)
                loss, rollout, recon0, stats0, recon_predict = step(batch, train=False)
                val_loss_sum += loss * bs
                val_rollout_sum += rollout * bs
                val_recon0_sum += recon0 * bs
                val_stats0_sum += stats0 * bs
                val_recon_predict_sum += recon_predict * bs
        _val_prog.close()
        _prev_val_seconds = time.monotonic() - _val_t0   # feeds next epoch's ETA
        val_loss = (val_loss_sum / n_val).item()
        val_rollout = (val_rollout_sum / n_val).item()
        val_recon0 = (val_recon0_sum / n_val).item()
        val_stats0 = (val_stats0_sum / n_val).item()
        val_recon_predict = (val_recon_predict_sum / n_val).item()

        # SKIPPED BATCHES ARE NEVER SILENT -- a guard that quietly drops data
        # would be worse than the crash it prevents, since the run would look
        # healthy while training on a filtered distribution.
        # BOTH counts captured before either reset -- see end_epoch_pair's own
        # docstring for the ordering bug this replaces.
        _deadlocked = end_epoch_pair(spike_guard, grad_guard, _n_train_batches)
        # ONE merged report via the shared reporter: when both guards fire in an
        # epoch it is a single block (both worst-clauses, boilerplate once), not
        # two near-identical ones; routine skips digest to a compact line. This
        # is the SAME path stage 3 uses -- stages 4/5 used to hand-roll two
        # separate blocks here.
        _newg = grad_guard.n_skipped - _grad_spikes_reported
        _new = spike_guard.n_skipped - _spikes_reported
        _grad_spikes_reported = grad_guard.n_skipped
        _spikes_reported = spike_guard.n_skipped
        _all_nonfinite = spike_guard.n_nonfinite + grad_guard.n_nonfinite
        _new_nonfinite = _all_nonfinite - _nonfinite_reported
        _nonfinite_reported = _all_nonfinite
        _line = _skip_reporter.epoch(
            epoch, _new, spike_guard.last_worst, _newg, grad_guard.last_worst,
            _n_train_batches, n_nonfinite_new=_new_nonfinite,
            dt_label=("du_max" if getattr(f_theta, "time_coordinate", "t") == "log10_t"
                      else "dt_max"))
        if _line:
            print(_line)
        if _deadlocked and spike_guard.consecutive_total_skip_epochs >= 5:
            # STOP, keeping the best checkpoint -- the stage-3 lesson. No
            # rollback here: at ~135 batches per epoch an all-skipped epoch
            # means the model is comprehensively broken, not that one window
            # tripped a threshold, and stage 4/5's checkpoint is a JOINT one
            # whose restore path would need its own testing to be trustworthy.
            print(f"\nSTOPPING at epoch {epoch}: every batch has been skipped for "
                  f"{spike_guard.consecutive_total_skip_epochs} consecutive epochs, so no "
                  f"gradient step is being taken and the weights cannot recover. Keeping "
                  f"the best checkpoint so far. LOWER lr (currently {lr:g}); raising "
                  f"spike_skip_factor only lets the damaging batches through.\n")
            break

        criterion, saved_this_epoch = tracker.update(epoch, val_loss)

        epoch_history.append(epoch)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        best_so_far_history.append(tracker.best_val_loss)
        if should_write_loss_figure(epoch, log_every_epoch, n_points=len(epoch_history)):
            loss_curve(
                epoch_history, train_loss_history, val_loss_history, best_so_far_history,
                loss_curve_path, title=f"Stage {'4' if freeze_decoder else '5'} loss",
                event_epochs=loss_curve_events,
            )
            write_loss_history(loss_curve_path, epoch_history, train_loss_history, val_loss_history, best_so_far_history)

        current_val_components = {
            "rollout": rollout_weight * val_rollout / rollout_scale,
            "recon0": recon0_weight * val_recon0 / recon0_scale,
            "stats0": stats0_weight * val_stats0 / stats0_scale,
            "recon_predict": recon_predict_weight * val_recon_predict / recon_predict_scale,
        }
        best_components = component_best_tracker.update(current_val_components, saved_this_epoch)
        current_train_components = {
            "rollout": rollout_weight * train_rollout / rollout_scale,
            "recon0": recon0_weight * train_recon0 / recon0_scale,
            "stats0": stats0_weight * train_stats0 / stats0_scale,
            "recon_predict": recon_predict_weight * train_recon_predict / recon_predict_scale,
        }
        for name in component_histories:
            component_histories[name]["train"].append(current_train_components[name])
            component_histories[name]["val"].append(current_val_components[name])
            component_histories[name]["best_so_far"].append(best_components[name])
        if should_write_loss_figure(epoch, log_every_epoch, n_points=len(epoch_history)):
            loss_component_scatter(
                epoch_history, component_histories, loss_components_path,
                title=f"Stage {'4' if freeze_decoder else '5'} loss components",
            )

        ema_str = f"{tracker.val_ema:7.4f}" if tracker.val_ema is not None else "  (warmup)"
        msg = (f"{epoch:4d}|"
               # effective_rollout_weight, not rollout_weight: during the ramp
               # the printed train component must be what was actually
               # OPTIMISED, or the columns will not sum to the train_loss
               # beside them and the discrepancy reads as a bug.
               f"{train_loss:7.4f} ={effective_rollout_weight*train_rollout/rollout_scale:7.4f} "
               f"+{recon0_weight*train_recon0/recon0_scale:7.4f} "
               f"+{stats0_weight*train_stats0/stats0_scale:7.4f} "
               f"+{effective_recon_predict_weight*train_recon_predict/recon_predict_scale:7.4f} |"
               f"{val_loss:7.4f} ={rollout_weight*val_rollout/rollout_scale:7.4f} "
               f"+{recon0_weight*val_recon0/recon0_scale:7.4f} "
               f"+{stats0_weight*val_stats0/stats0_scale:7.4f} "
               f"+{recon_predict_weight*val_recon_predict/recon_predict_scale:7.4f} |"
               f"{ema_str:>10}"
               + (f"  [rollout_weight {effective_rollout_weight:.3g}/{rollout_weight:g}]"
                  if effective_rollout_weight < rollout_weight else ""))

        # AFTER the ramp, not at epoch 1. During a warmup the imbalance is
        # deliberate -- the whole point is that rollout contributes almost
        # nothing yet -- so an epoch-1 reading describes the transient, not the
        # scales. On a real run it fired at epoch 1 reporting raw rollout=168
        # and recommending rollout_scale~168, when the CONVERGED value was
        # 0.053: acting on it would have undone the ramp that had just worked.
        if epoch == max(1, rollout_weight_warmup_epochs):
            # ONE component dominating the loss means the *_scale values are
            # mis-calibrated for THIS stage, and the weights beside them mean
            # nothing. Reported once, at the first epoch, because that is when
            # it is cheap to act on.
            #
            # rollout_scale in particular is easy to inherit wrongly: stage 3
            # FREEZES the encoder, so f_theta sees exactly the latents it was
            # fitted to and its rollout loss is ~1e-6. Stage 4 unfreezes it, so
            # f_theta is off-distribution from the first update and the same
            # quantity is ~0.7 -- 6e5 times larger. A single global
            # rollout_scale=1e-6 in a params file therefore makes stage 4's
            # loss 99.997% rollout, with recon0 and stats0 contributing 26 parts
            # per MILLION. Stage 4's entire purpose is the balance between them.
            contributions = {
                "rollout": rollout_weight * val_rollout / rollout_scale,
                "recon0": recon0_weight * val_recon0 / recon0_scale,
                "stats0": stats0_weight * val_stats0 / stats0_scale,
                "recon_predict": recon_predict_weight * val_recon_predict / recon_predict_scale,
            }
            total = sum(abs(v) for v in contributions.values())
            raw = {"rollout": val_rollout, "recon0": val_recon0, "stats0": val_stats0,
                   "recon_predict": val_recon_predict}
            weights = {"rollout": rollout_weight, "recon0": recon0_weight,
                        "stats0": stats0_weight, "recon_predict": recon_predict_weight}
            scales = {"rollout": rollout_scale, "recon0": recon0_scale,
                       "stats0": stats0_scale, "recon_predict": recon_predict_scale}
            if total > 0:
                shares = {k: abs(v) / total for k, v in contributions.items()}
                _raw_str = ", ".join(f"{k}={raw[k]:.3e}" for k in shares)
                _suggest = ", ".join(
                    f"{k}_scale~{raw[k]:.3g}" for k in shares if raw[k] > 0)

                # TWO-SIDED. The original check only caught a component
                # DOMINATING, and stage 4 has since hit the opposite end just
                # as hard: with rollout_scale=100 against a converged raw of
                # 0.04, L_rollout fell to 0.46% of the validation loss and the
                # warning stayed silent -- while the term stage 4 exists to
                # balance had effectively left the objective. Both failures are
                # the same defect (a scale that is not the raw magnitude of its
                # own component) and both deserve the same report.
                dominant = [k for k, sh in shares.items() if sh > 0.99]
                # "starved" is keyed on a NONZERO weight: a component the user
                # deliberately switched off must not be reported as a problem.
                starved = [k for k, sh in shares.items()
                            if sh < 0.01 and weights.get(k, 0.0) != 0.0]

                if dominant:
                    k = dominant[0]
                    others = ", ".join(f"{n} {100 * shares[n]:.4f}%"
                                        for n in shares if n != k)
                    print(f"\n  WARNING: '{k}' is {100 * shares[k]:.4f}% of the validation "
                          f"loss ({others}). The *_scale values are calibrated for a "
                          f"different stage -- each scale should be the RAW magnitude of its "
                          f"own component here, so the weights beside them mean what they say. "
                          f"Raw values this epoch: {_raw_str}. Suggested: {_suggest}\n")
                elif starved:
                    detail = ", ".join(
                        f"'{k}' {100 * shares[k]:.4f}% (weight {weights[k]:g}, "
                        f"scale {scales[k]:g})" for k in starved)
                    print(f"\n  WARNING: {detail} of the validation loss, despite a nonzero "
                          f"weight -- that term is effectively OUT of the objective, so this "
                          f"stage is not balancing what it exists to balance. Its scale is far "
                          f"above its own raw magnitude. Raw values this epoch: {_raw_str}. "
                          f"Suggested: {_suggest}\n")

        if saved_this_epoch:
            _saved_this_run = True
            if epochs_since_improvement > longest_gap:
                longest_gap = epochs_since_improvement
                longest_gap_range = (epoch - epochs_since_improvement, epoch)
            epochs_since_improvement = 0
            atomic_torch_save({
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
                "config": {
                    **{k: v for k, v in components["encoder"].config.items() if k != "decoder_for_stream"},
                    "stream_configs": {
                        name: {"channels": cfg.channels, "spatial_size": cfg.spatial_size,
                               "mode": cfg.mode.value, "condition_on_theta": cfg.condition_on_theta,
                               # head_kind/head_hidden MUST be recorded: a residual-head
                               # stream's Encoder has real residual_heads.<name> weights, and
                               # a reload rebuilds a plain Encoder without them (RuntimeError:
                               # unexpected residual_heads.deriv.*) if the config omits these.
                               "head_kind": cfg.head_kind, "head_hidden": cfg.head_hidden}
                        for name, cfg in stream_configs.items()
                    },
                    "recon_stream_name": recon_stream_name,
                },
                "lds_config": dict(components["lds"].config),
                # max_dt/min_passing_steps recorded too, so stage 5 (and any
                # diagnostic) inherits the same window population rather than
                # rediscovering it -- the resolved values, not the arguments,
                # since these may have come from f_theta's own data_config.
                "data_config": {"min_step": min_step, "min_stdev_phi": min_stdev_phi,
                                "min_passing_steps": min_passing_steps, "max_dt": max_dt,
                                "window_length": window_length, "n_rollout_steps": n_rollout_steps},
                "stats_config": (
                    {"stat_names": stat_names, "stats_mean": stats_loss_fn.mean.cpu(),
                     "stats_std": stats_loss_fn.std.cpu()}
                    if stats_loss_fn is not None else None
                ),
                "stage45_config": {
                    "freeze_decoder": freeze_decoder, "rollout_weight": rollout_weight,
                    "recon0_weight": recon0_weight, "stats0_weight": stats0_weight,
                    "n_rollout_steps": n_rollout_steps,
                },
            }, checkpoint_path)
            msg += "  -> saved"
            msg += f" at {time.strftime('%H:%M')}"  # match the checkpoint filename timestamp
            if on_checkpoint_saved is not None:
                try:
                    on_checkpoint_saved(checkpoint_path, epoch)
                except Exception as e:
                    # Bookkeeping must never kill training: a failed registry
                    # upsert must announce and continue, never lose the run.
                    print(f"  WARNING: on_checkpoint_saved failed "
                          f"({type(e).__name__}: {e}) -- continuing training")
        else:
            epochs_since_improvement += 1

        if log_every_epoch or saved_this_epoch:
            print(msg)

        if early_stopping_patience is not None and epochs_since_improvement >= early_stopping_patience:
            print(early_stop_message(epoch, early_stopping_patience, _saved_this_run,
                                     longest_gap=longest_gap,
                                     longest_gap_range=longest_gap_range))
            break

    # Unconditional final write: the in-loop calls are throttled (see
    # should_write_loss_figure), and a run can end on an epoch that was
    # skipped -- via early stopping, or simply because the last epoch
    # wasn't a multiple of the interval. Without this the figures left on
    # disk could be up to `every` epochs stale, which is exactly the
    # state a finished run gets judged from.
    loss_curve(
        epoch_history, train_loss_history, val_loss_history, best_so_far_history,
        loss_curve_path, title=f"Stage {'4' if freeze_decoder else '5'} loss",
        event_epochs=loss_curve_events,
    )
    write_loss_history(loss_curve_path, epoch_history, train_loss_history, val_loss_history, best_so_far_history)
    loss_component_scatter(
        epoch_history, component_histories, loss_components_path,
        title=f"Stage {'4' if freeze_decoder else '5'} loss components",
    )

    if not Path(checkpoint_path).exists():
        # Same guard as train_stage2/train_lds. Without it this returns a path
        # that does not exist and the caller fails far from the cause -- the
        # pipeline feeds stage 1's straight to check_reconstruction, and stage
        # 4's to stage 5.
        # NOT keyed on epochs==0 here, unlike stage 1: stage 4/5 DOES evaluate
        # and save at epoch 0 even with epochs=0 (observed -- a real run wrote
        # "0| ... -> saved"), so reaching this point means the epoch-0 save
        # itself failed, which epochs=0 does not explain.
        reason = "no epoch's val criterion beat the running best, and epoch 0 did not save"
        raise RuntimeError(
            f"stage 4/5 finished without ever saving a checkpoint to "
            f"{checkpoint_path}: {reason}. An epochs=0 ablation cannot produce a "
            f"checkpoint -- remove the stage from the params file rather than "
            f"setting its epochs to 0, if anything downstream needs its output."
        )
    return checkpoint_path