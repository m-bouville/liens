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

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder, MultiStreamAutoencoder
from models.constants import LATENT_SPATIAL_SIZE
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_dynamics import LatentDynamics
from models.latent_streams import (
    LatentStreamMode, cross_check_stream_configs_against_state_dict,
    resolve_stream_configs_from_checkpoint_config,
)
from training.checkpoint_criterion import CheckpointCriterionTracker
from training.datasets import MicrostructureEvolutionDataset, complete_run_dirs, split_run_dirs
from training.losses import RolloutLoss, compute_dt_decade_weights
from utils.naming import ae_checkpoint_name, lds_checkpoint_name
from utils.plots import loss_curve

# GENERAL POLICY (matches training/train_refinement.py's own
# _PYTHON_ROOT): every checkpoint/output/dataset path is built from
# THIS anchor, never from a bare relative string like "../output/...".
# Relative strings resolve against the process's CWD at invocation
# time, which silently differs across bare CLI, `python -m`, and being
# imported and called from another module (e.g. main.py calling
# train_lds) -- exactly the recurring "output ended up in the wrong
# place" bug hit repeatedly on this project. Path(__file__) is anchored
# to THIS FILE's own on-disk location instead, which is invariant
# regardless of how/from-where the process was launched.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/training/train_lds.py -> python/


def compute_euler_only_losses(
    f_theta: LatentDynamics, train_set, device: torch.device,
    batch_size: int = 512, num_workers: int = 0,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    One pass over train_set with a FRESHLY-INITIALIZED f_theta (its
    weights exactly as constructed, before any training step and
    before any resume_from checkpoint is loaded into it) -- measures
    how raw per-transition loss actually scales with dt, so
    compute_dt_decade_weights can invert the real per-decade LOSS MASS
    (see that function's own docstring for the bug this fixes) rather
    than assuming it from window count alone.

    "Euler-only" because at init, f_theta's own output is (by
    LatentDynamics' own construction) ~0, so the rollout below reduces
    to the hard-coded z0 + z1*dt term alone -- exactly the same
    well-understood, stable proxy used elsewhere in this project (see
    check_parameter_dependence.py's own euler-only baseline) for how
    raw error scales with dt BEFORE any learned correction exists.
    Deliberately reuses f_theta.rollout() itself (the actual
    training-time computation), not a separately hand-coded z0+z1*dt
    formula -- keeps this measurement mechanically identical to what
    real training will later reweight, rather than a parallel
    implementation that could quietly drift out of sync with it.

    MUST be called before f_theta is trained, and before resume_from is
    loaded into it -- calling this on a partially- or fully-trained
    model would measure THAT model's own residual error (already
    shaped by whatever it has learned), not the raw, pre-training
    dt-vs-error relationship this weighting exists to correct for.

    Returns (all_dts, all_losses): both flat 1-D numpy arrays, one
    entry per (window, rollout step) transition -- i.e. length
    len(train_set) * n_rollout_steps, matching exactly the granularity
    at which dt_decade_weights_fn(dt_window) is later looked up and
    applied during real training (per (B, n_r) element, not just once
    per window).
    """
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=device.type == "cuda")
    was_training = f_theta.training
    f_theta.eval()
    all_dts_parts = []
    all_losses_parts = []
    with torch.no_grad():
        for batch in loader:
            window0, window1, dt_window, theta = batch
            window0 = window0.to(device, non_blocking=True)
            window1 = window1.to(device, non_blocking=True)
            dt_window = dt_window.to(device, non_blocking=True)
            theta = theta.to(device, non_blocking=True)

            z0 = window0[:, 0]
            z0_true = window0[:, 1:]
            # Same teacher-forced rollout call the real training step()
            # uses (see its own docstring) -- at init this is
            # equivalent to the hard-coded Euler term, but going
            # through the real code path keeps this measurement exactly
            # consistent with what gets reweighted later, not a
            # separate approximation of it.
            z0_hat_full = f_theta.rollout(z0, window1, dt_window, theta)
            z0_hat = z0_hat_full[:, 1:]

            diff = z0_hat - z0_true
            # Same reduction RolloutLoss itself uses (spatial mean
            # only, BEFORE any weighting) -- (B, n_r).
            per_window_step = diff.pow(2).mean(dim=(2, 3, 4))

            all_dts_parts.append(dt_window.detach().cpu().numpy().reshape(-1))
            all_losses_parts.append(per_window_step.detach().cpu().numpy().reshape(-1))
    if was_training:
        f_theta.train()

    all_dts = np.concatenate(all_dts_parts)
    all_losses = np.concatenate(all_losses_parts)
    return all_dts, all_losses


def _load_frozen_encoder(
    ae_checkpoint_path: Path | None, ae_latent_channels: int | None, ae_stats_weight: float,
    size: int, condition_on_theta: bool | None, device: torch.device,
) -> tuple[torch.nn.Module, dict, dict, Path]:
    """
    Loads a frozen, already-trained autoencoder's encoder from a stage-
    1/2 checkpoint, validating condition_on_theta (if given) and the
    requested size against what the checkpoint actually has. Extracted
    verbatim from train_lds()'s own former body -- same logic, same
    order, just named and callable on its own.

    Accepts EITHER a single-stream (stage 1a) ancestor, OR an already-
    multi-stream one -- the latter in either of two decoder shapes: no
    decoder_for_stream at all (stage 2's own pre-stage-1b format, every
    stream sharing one decoder), or a real decoder_for_stream mapping
    (a legacy, stage-1b-derived checkpoint with separate per-stream
    decoders still on disk). Only the encoder is ever actually used
    here or by any caller downstream -- decoder weights, if loaded at
    all, are inert (load_state_dict still needs the right key
    structure to succeed in the first place).

    Returns (encoder, ae_checkpoint, ae_config, ae_checkpoint_path) --
    ae_checkpoint_path is returned too since it may be reconstructed
    here (from ae_latent_channels) when not given directly, and the
    caller needs the resolved path for its own printing and for the
    "ae_checkpoint" field of whatever it goes on to save.
    """
    if ae_checkpoint_path is None:
        if ae_latent_channels is None:
            raise ValueError(
                "Provide either ae_checkpoint_path directly, or ae_latent_channels "
                "so the expected path can be reconstructed."
            )
        ae_name = ae_checkpoint_name(size, ae_latent_channels, ae_stats_weight)
        ae_checkpoint_path = _PYTHON_ROOT / "checkpoints" / "stage2" / f"{ae_name}.pt"
        print(f"Reconstructed AE checkpoint path: {ae_checkpoint_path}")

    # Load the frozen autoencoder. Only .encoder is ever used below --
    # the decoder is irrelevant to stage 3 training.
    ae_checkpoint = torch.load(ae_checkpoint_path, map_location=device, weights_only=True)
    ae_config = ae_checkpoint["config"]
    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(ae_config)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, ae_checkpoint["model_state"],
    )
    recon_stream = stream_configs[recon_stream_name]

    # condition_on_theta is NOT decided here -- deriv's theta-FiLM
    # conditioning is a structural property fixed once, when the stream
    # is CREATED (see training/extend_encoder.py's own module docstring
    # -- deriv is now built directly inside train_stage2(), no separate
    # stage 1b pass). Stage 3 only ever loads an already-built, FROZEN
    # encoder, so there is nothing to set -- only something to VALIDATE,
    # same rationale and
    # same error as train_stage2's own identical check. None (default)
    # skips this entirely, trusting whatever the loaded checkpoint has.
    if condition_on_theta is not None:
        deriv_candidates = [n for n, c in stream_configs.items()
                             if n != recon_stream_name and c.mode != LatentStreamMode.PURE_LATENT]
        if len(deriv_candidates) == 1:
            deriv_stream_name = deriv_candidates[0]
            deriv_stream = stream_configs[deriv_stream_name]
            if deriv_stream.condition_on_theta != condition_on_theta:
                raise ValueError(
                    f"condition_on_theta={condition_on_theta} was requested, but "
                    f"{ae_checkpoint_path}'s own '{deriv_stream_name}' stream has "
                    f"condition_on_theta={deriv_stream.condition_on_theta} -- this was decided when "
                    f"the stream was CREATED (in stage 2, see training/extend_encoder.py), not something "
                    f"stage 3 can change by loading a different value. Rebuild the ancestor "
                    f"from stage 2 with condition_on_theta="
                    f"{condition_on_theta} if you actually want a different ancestor, or drop this "
                    f"parameter here to match whatever {ae_checkpoint_path} already has."
                )
        # len(deriv_candidates) != 1 (0 or ambiguous): nothing meaningful
        # to validate against -- silently skipped rather than raising a
        # SECOND, less specific error here; whatever actually needs a
        # deriv stream to exist (the dataset construction below) will
        # raise its own clear error shortly regardless.

    decoder_for_stream = ae_config.get("decoder_for_stream")
    if len(stream_configs) == 1:
        ae = Autoencoder(size=ae_config["size"], channels=1, base_channels=ae_config["base_channels"],
                          latent_channels=recon_stream.channels,
                          latent_spatial_size=recon_stream.spatial_size).to(device)
    elif decoder_for_stream is None:
        # Stage 2's own (pre-stage-1b) format: every stream shares ONE decoder.
        _encoder_module = Encoder(input_size=ae_config["size"], in_channels=1,
                                   base_channels=ae_config["base_channels"], stream_configs=stream_configs,
                                   n_theta=1)
        _decoder_module = Decoder(output_size=ae_config["size"], out_channels=1,
                                   base_channels=ae_config["base_channels"], latent_channels=recon_stream.channels,
                                   latent_spatial_size=recon_stream.spatial_size)
        ae = MultiStreamAutoencoder(encoders={"shared": _encoder_module}, decoders={"shared": _decoder_module},
                                     stream_configs=stream_configs).to(device)
    else:
        # Stage 1b's own format: a SEPARATE decoder per stream (see
        # autoencoder.py's MultiStreamAutoencoder). Only the encoder
        # ever gets used below, but load_state_dict still needs a
        # decoder object per unique decoder key for the checkpoint's
        # own keys to match at all -- their actual weights are inert
        # here, never read again after this load.
        _encoder_module = Encoder(input_size=ae_config["size"], in_channels=1,
                                   base_channels=ae_config["base_channels"], stream_configs=stream_configs,
                                   n_theta=1)
        _decoders = {}
        for stream_name, decoder_key in decoder_for_stream.items():
            stream_cfg = stream_configs[stream_name]
            _decoders[decoder_key] = Decoder(
                output_size=ae_config["size"], out_channels=1,
                base_channels=ae_config["base_channels"], latent_channels=stream_cfg.channels,
                latent_spatial_size=stream_cfg.spatial_size,
            )
        ae = MultiStreamAutoencoder(encoders={"shared": _encoder_module}, decoders=_decoders,
                                     stream_configs=stream_configs,
                                     decoder_for_stream=decoder_for_stream).to(device)
    ae.load_state_dict(ae_checkpoint["model_state"])
    encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
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

    return encoder, ae_checkpoint, ae_config, ae_checkpoint_path


def _resume_f_theta_from_checkpoint(
    f_theta: torch.nn.Module, resume_from: Path, ae_config: dict,
    hidden_dim: int, n_hidden_layers: int, dt_cap: float, n_rollout_steps: int,
    device: torch.device,
) -> None:
    """
    Loads f_theta's weights from a prior LDS checkpoint (curriculum
    rollout: train at n_rollout_steps=1 first, then resume here at the
    target n_rollout_steps), validating the architecture matches
    exactly first -- latent_channels, hidden_dim, n_hidden_layers,
    dt_cap, and latent_spatial_size. Mutates f_theta in place via
    load_state_dict, matching the idiom load_state_dict itself uses;
    no return value. Extracted verbatim from train_lds()'s own former
    body -- same logic, same order, just named and callable on its
    own.
    """
    if resume_from is not None:
        # Curriculum rollout: train with n_rollout_steps=1 first (stable,
        # fast, avoids the epoch-1 loss blowup a from-scratch jump straight
        # to multi-step rollout produced), then resume here with the
        # target n_rollout_steps. Architecture must match exactly --
        # loading into a differently-shaped f_theta would either error
        # confusingly deep in load_state_dict or silently mismatch.
        prev_lds = torch.load(resume_from, map_location=device, weights_only=True)
        prev_config = prev_lds["config"]
        prev_latent_spatial_size = prev_config.get("latent_spatial_size", LATENT_SPATIAL_SIZE)
        current_latent_spatial_size = ae_config.get("latent_spatial_size", LATENT_SPATIAL_SIZE)
        mismatch = [(k, prev_config[k], v) for k, v in
                    [("latent_channels", ae_config["latent_channels"]),
                     ("hidden_dim", hidden_dim), ("n_hidden_layers", n_hidden_layers)]
                    if prev_config[k] != v]
        # dt_cap checked separately, not folded into the list above --
        # unlike the other three keys (always present, any checkpoint
        # this project has produced), dt_cap may be absent from an older
        # checkpoint's own saved config, and prev_config[k] direct
        # indexing above would KeyError on those rather than falling
        # back cleanly. Not a weight-shape mismatch load_state_dict
        # would ever catch on its own (it's a plain float attribute,
        # not a learnable parameter/buffer) -- but resuming under a
        # DIFFERENT dt_cap than the loaded weights were actually trained
        # under is exactly the kind of silent, dangerous inconsistency
        # this whole check exists to catch. .get(..., inf) on the old
        # side means resuming from a pre-dt_cap checkpoint against a
        # new, also-default (inf) dt_cap doesn't spuriously flag a
        # mismatch that was never really there.
        prev_dt_cap = prev_config.get("dt_cap", float("inf"))
        if prev_dt_cap != dt_cap:
            mismatch.append(("dt_cap", prev_dt_cap, dt_cap))
        if prev_latent_spatial_size != current_latent_spatial_size:
            mismatch.append(("latent_spatial_size", prev_latent_spatial_size, current_latent_spatial_size))
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


def train_lds(
    size: int, base_path: Path,
    ae_checkpoint_path: Path | None = None,
    ae_latent_channels: int | None = None, ae_stats_weight: float | None = None,
    rollout_scale: float = 1.0,
    epochs: int = 100, batch_size: int = 512, lr: float = 1e-3,
    hidden_dim: int = 256, n_hidden_layers: int = 2,
    val_fraction: float = 0.2, test_fraction: float = 0.1, num_workers: int = 0,
    n_rollout_steps: int = 1, min_step: int | None = None, min_stdev_phi: float | None = None,
    min_passing_steps: int | None = None,
    condition_on_theta: bool | None = None,
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
    use_dt_decade_weights: bool = False,
    z0_noise_scale: float = 0.0,
    dt_cap: float = float("inf"),
) -> Path:
    """
    Stage 3. Returns the path of the best checkpoint saved. Either give
    ae_checkpoint_path directly, or both ae_latent_channels and
    ae_stats_weight (the AE checkpoint's own ae_stats_weight, used only to
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

    use_dt_decade_weights (default False): global, per-decade loss
    rebalancing (see training.losses.compute_dt_decade_weights and
    compute_euler_only_losses above) -- directly counteracts a training
    set where a small fraction of windows (extreme-dt ones) carry a
    large majority of the raw loss MASS (confirmed empirically: ~7% of
    windows carrying ~68% of the total loss, in one real measurement),
    which otherwise dominates the gradient regardless of how few
    windows produce it. Computed ONCE, before training starts, from a
    single pass over train_set's own data with a freshly-initialized
    f_theta measuring both its dt distribution AND its actual raw loss
    per decade -- weighting by dt distribution alone (an earlier,
    buggy version of this) is NOT equivalent and actively made things
    worse: see compute_dt_decade_weights' own docstring for the full
    story. Not per-batch (a per-batch version would give windows in a
    sparsely-represented decade an enormous individual weight whenever
    a given batch happens to draw few of them, reintroducing
    instability from a different angle). Applied to the TRAINING loss
    only, never validation -- val_loss stays a clean, naturally-
    distributed measure of real performance, and the EMA-based
    checkpoint criterion keeps selecting on that, not on the
    artificially-rebalanced training objective.

    z0_noise_scale (default 0.0, off -- reproduces exact prior
    behavior): trains f_theta against a z0 deliberately perturbed away
    from the true value, to fix a specific, diagnosed failure mode
    (see evaluation.check_f_theta): single-step training here NEVER
    shows f_theta anything but the exact true z0 as input (window_length
    is only 2), so it has no learned behavior for a slightly-wrong z0 --
    and empirically, feeding it one anyway (as multi-step rollout
    training necessarily does, chaining its own prior-step prediction
    into the next step) makes ||f|| blow up by 3-5 orders of magnitude
    at the SAME dt, with corr(log(z0_step1_error), log(||f2_chained||))
    exceeding even corr(log(dt2), log(||f2_chained||)) -- direct
    evidence this is an off-distribution generalization gap, not an
    intrinsic dt-dependence, and one that resuming straight into
    multi-step rollout training (n_rollout_steps>1) was NOT able to fix
    on its own: it either blew up outright, or (at a heavily-protected
    one_step_weight) quietly stopped degrading single-step accuracy
    while the rollout term stayed catastrophic regardless -- i.e. the
    rollout loss alone could not teach this robustness, only trade it
    off against something else.

    The perturbation is scaled PER WINDOW to that window's own actual
    Euler-step magnitude, |z1(t0)| * dt(t0->t1) -- the dominant term in
    z0_hat itself (see the z1*dt + f*dt^2/2 update rule) -- rather than
    a single fixed or dataset-global scale, specifically because the
    natural scale of "how wrong a real predicted z0 might plausibly be"
    is itself hugely dt-dependent (per-decade raw loss spans ~4 orders
    of magnitude in this project's own measurements -- see
    compute_dt_decade_weights). A fixed absolute noise scale would be
    wildly wrong for either the smallest or largest dt decade; scaling
    to each window's own characteristic step size keeps the injected
    perturbation the right order of magnitude everywhere.

    Applied ONLY when train=True, to z0 alone (never z0_true, window1,
    or dt_window) -- val_loss must stay computed against the exact same
    clean, unperturbed inputs as always, or it stops being the
    naturally-distributed measure of real performance the checkpoint
    criterion depends on (same rationale as use_dt_decade_weights being
    train-only, above). A small nonzero value (this project's own
    diagnostics suggest starting an order of magnitude or two below 1.0)
    is the intended range -- this reproduces a REALISTIC step-1 error,
    not an adversarial one; too large a value teaches robustness to
    inputs the model will never actually encounter, at the cost of
    fitting the real, clean data distribution well.
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
    missing = [name for name, v in [("size", size), ("ae_stats_weight", ae_stats_weight),
                                     ("min_step", min_step)]
               if v is None]
    if missing:
        raise ValueError(f"train_lds() requires {', '.join(missing)} to be given explicitly "
                          f"-- config.txt no longer provides ML training defaults.")
    print(f"min_step={min_step}  min_stdev_phi={min_stdev_phi}  min_passing_steps={min_passing_steps}  "
          f"ae_stats_weight={ae_stats_weight}")
    print(f"n_rollout_steps={n_rollout_steps}  one_step_weight={one_step_weight}")
    print(f"grad_clip={grad_clip}  lr_warmup_steps={lr_warmup_steps}  z0_noise_scale={z0_noise_scale}")
    print(f"lr={lr}  seed={seed}")

    encoder, ae_checkpoint, ae_config, ae_checkpoint_path = _load_frozen_encoder(
        ae_checkpoint_path, ae_latent_channels, ae_stats_weight, size, condition_on_theta, device,
    )

    if checkpoint_path is None:
        name = lds_checkpoint_name(ae_config["size"], ae_config["latent_channels"],
                                    ae_stats_weight, n_rollout_steps)
        checkpoint_path = _PYTHON_ROOT / "checkpoints" / "stage3" / f"{name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {checkpoint_path}\n")

    if loss_curve_path is None:
        name = lds_checkpoint_name(ae_config["size"], ae_config["latent_channels"],
                                    ae_stats_weight, n_rollout_steps)
        loss_curve_path = _PYTHON_ROOT.parent / "output" / "stage3" / f"{name}-loss_curve.png"

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

    if epochs == 0:
        # Ablation mode: no training happens (see the epoch loop
        # below), so train_set/train_loader would never be touched --
        # skipped entirely. Especially worth skipping here: encoder=
        # encoder means EVERY snapshot gets run through the frozen AE's
        # forward pass at construction time, not just read/filtered --
        # likely the most expensive dataset build in this whole
        # pipeline, for a dataset that would then never be iterated.
        train_set = train_loader = None
        val_set = MicrostructureEvolutionDataset(
            val_dirs, encoder=encoder, device=device, window_length=window_length,
            min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
            encode_batch_size=encode_batch_size,
            encode_both_streams=True,
        )
        print(f"train_set: skipped (epochs=0 ablation -- never iterated over), "
              f"{len(val_set)} val windows (n_rollout_steps={n_rollout_steps}, "
              f"window_length={window_length})\n")
    else:
        train_set = MicrostructureEvolutionDataset(
            train_dirs, encoder=encoder, device=device, window_length=window_length,
            min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
            encode_batch_size=encode_batch_size,
            encode_both_streams=True,
        )
        val_set = MicrostructureEvolutionDataset(
            val_dirs, encoder=encoder, device=device, window_length=window_length,
            min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
            encode_batch_size=encode_batch_size,
            encode_both_streams=True,
        )
        print(f"{len(train_set)} train windows, {len(val_set)} val windows "
              f"(n_rollout_steps={n_rollout_steps}, window_length={window_length})\n")
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             persistent_workers=num_workers > 0, pin_memory=device.type == "cuda")

    # f_theta must exist BEFORE the decade-weight measurement below (it
    # needs a freshly-initialized model to measure against -- see
    # compute_euler_only_losses' own docstring for why), so construction
    # is here, ahead of resume_from's own loading further down: that
    # load must stay AFTER this measurement, or it would measure an
    # already-trained model's residual error instead of the raw,
    # pre-training dt-vs-error relationship this weighting needs.
    f_theta = LatentDynamics(latent_channels=ae_config["latent_channels"], n_theta=1,
                              latent_spatial=ae_config.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
                              hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
                              dt_cap=dt_cap).to(device)

    # Global per-decade loss weights, computed ONCE from train_set's own
    # full dt distribution AND its own raw per-transition loss
    # (measured with f_theta exactly as freshly constructed above) --
    # see use_dt_decade_weights' own docstring for the full rationale,
    # and compute_dt_decade_weights' own docstring for why BOTH dt and
    # measured loss are required, not dt alone. None if disabled, or if
    # train_set itself was skipped (epochs=0 ablation) -- there's no
    # training loss to reweight in that case regardless.
    dt_decade_weights_fn = None
    if use_dt_decade_weights and train_set is not None:
        all_dts, all_losses = compute_euler_only_losses(
            f_theta, train_set, device, batch_size=batch_size, num_workers=num_workers,
        )
        dt_decade_weights_fn = compute_dt_decade_weights(all_dts, all_losses)
        print(f"use_dt_decade_weights=True: per-decade weights computed from "
              f"{len(dt_decade_weights_fn.decade_weight)} decades in train_set's own dt "
              f"distribution, weighted by each decade's own measured raw loss mass -- "
              f"{dict(sorted(dt_decade_weights_fn.decade_weight.items()))}\n")

    _resume_f_theta_from_checkpoint(
        f_theta, resume_from, ae_config, hidden_dim, n_hidden_layers, dt_cap, n_rollout_steps, device,
    )

    step_weights_tensor = torch.tensor(step_weights, dtype=torch.float32, device=device) \
        if step_weights is not None else None
    if step_weights_tensor is not None and step_weights_tensor.shape != (n_rollout_steps,):
        raise ValueError(f"step_weights has {len(step_weights)} entries, but "
                          f"n_rollout_steps={n_rollout_steps} -- need exactly one weight per step.")
    # exponent_deriv=0.0 (not the class's own default of 1.0): divides
    # the z0-space reconstruction error by dt before squaring, rather
    # than leaving it unweighted. f_theta's own architecture (self.f
    # takes z0, z1, theta -- no dt) means this changes ONLY the relative
    # weight different-dt windows get in the loss, not what f_theta
    # itself converges to -- the minimum of a dt-weighted squared error
    # sits at the same place as the unweighted one. Concretely, for
    # f_theta's dt^2/2 structure specifically (NOT the dt-independent
    # "rate error" this class's own docstring describes for q=0 --
    # that description holds for a linear, first-order model, not this
    # one): this reduces the raw loss's own dt-scaling from dt^4 (the
    # unweighted default) to dt^2, rather than eliminating dt-dependence
    # entirely. Still a substantial reduction -- it's what stops
    # use_dt_decade_weights from needing to suppress large-dt decades
    # as severely as it currently must (observed directly in Stage 3a's
    # own log: decade 4 weighted ~3000x lower than decade 1) just to
    # counteract the raw loss's own dt^4 blowup, which was leaving
    # f_theta with almost no effective gradient signal at large dt --
    # exactly where check_parameter_dependence.py's dt_dependence.png
    # shows the full (f_theta-corrected) prediction going wrong.
    rollout_loss = RolloutLoss(step_weights=step_weights_tensor, exponent_deriv=0.0)
    optimizer = torch.optim.Adam(f_theta.parameters(), lr=lr)

    lr_scheduler = None
    if lr_warmup_steps > 0:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=lr_warmup_steps,
        )

    def step(batch, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        window0, window1, dt_window, theta = batch
        window0 = window0.to(device, non_blocking=True)
        window1 = window1.to(device, non_blocking=True)
        dt_window = dt_window.to(device, non_blocking=True)
        theta = theta.to(device, non_blocking=True)

        z0 = window0[:, 0]
        z0_true = window0[:, 1:]

        # z0_noise_scale=0.0 (default) is a strict no-op -- everything
        # below this block is byte-identical to before it existed. See
        # z0_noise_scale's own docstring for the full rationale; the
        # short version: teach f_theta to behave reasonably on a z0
        # that's plausibly-wrong, at the SAME order of magnitude a real
        # chained step-1 error would be, BEFORE multi-step rollout
        # training ever has to chain it for real. Scaled per-window by
        # |z1(t0)| * dt(t0->t1) -- the dominant term in z0_hat itself --
        # not a fixed scale, since the natural error magnitude here is
        # itself hugely dt-dependent (see compute_dt_decade_weights).
        # TRAIN ONLY: val_loss must stay computed against the exact
        # same clean z0 as always (see this parameter's own docstring).
        if train and z0_noise_scale > 0:
            z1_step0 = window1[:, 0]
            dt_step0 = dt_window[:, 0].view(-1, *([1] * (z0.dim() - 1)))
            euler_step_scale = (z1_step0 * dt_step0).abs()
            z0 = z0 + z0_noise_scale * euler_step_scale * torch.randn_like(z0)

        # window1 passed WHOLE (not just window1[:, 0]) -- rollout()
        # teacher-forces the REAL z1 at every step, not a predicted
        # one (see LatentDynamics' own class docstring: this is what
        # makes f_theta directly testable without g_theta existing).
        z0_hat_full = f_theta.rollout(z0, window1, dt_window, theta)
        z0_hat = z0_hat_full[:, 1:]

        # exponent_deriv=0.0 (set at RolloutLoss's own construction
        # above) DOES apply here now, via dt=dt_window below -- this
        # reverses an earlier decision, made on reasoning that turned
        # out to be incomplete: the claim was that reweighting "no
        # longer holds cleanly" once the update rule mixes dt and dt^2
        # terms, since f_theta's own architecture (self.f takes z0, z1,
        # theta -- no dt) means diff's dt-dependence comes entirely
        # from the explicit (dt^2/2) multiplier, not from any implicit
        # diff=dt*err relationship -- but that's exactly why the
        # reweighting still works: dividing diff by dt before squaring
        # rescales this window's own loss MAGNITUDE without moving
        # where f_theta's output has to sit to minimize it (still the
        # same dt^2/2-normalized curvature target either way). It just
        # doesn't produce a fully dt-independent "rate error" the way
        # RolloutLoss's own docstring describes for q=0 -- that
        # description holds for a linear, first-order model, not this
        # dt^2/2 one -- it reduces the raw loss's dt-scaling from dt^4
        # to dt^2, a real and substantial reduction, not a complete
        # elimination. dt_decade_weights_fn is STILL applied too, on
        # top of this -- a genuinely different, composable mechanism
        # (see its own docstring): not rescaling each element's own
        # loss magnitude by its dt (exponent_deriv's own job), but
        # rescaling each WINDOW's relative contribution to the batch
        # average, so a small number of extreme-dt windows can't
        # dominate the gradient the way they otherwise do. TRAIN
        # only -- val_loss must stay an unweighted, naturally-
        # distributed measure of real performance.
        weights = dt_decade_weights_fn(dt_window) if (train and dt_decade_weights_fn is not None) else None
        z0_loss, z0_per_step = rollout_loss(z0_hat, z0_true, dt=dt_window, weights=weights, return_per_step=True)
        # per_step[0] is L_1step -- the loss restricted to just the first
        # predicted step, directly comparable to a model trained with
        # n_rollout_steps=1 (see RolloutLoss.forward()'s docstring: this
        # is mathematically identical to computing L_1step independently
        # on the same data, not an approximation).
        l_1step = z0_per_step[0]

        # total is what's actually optimized -- see one_step_weight's
        # docstring. At the default one_step_weight=0.0, total is
        # exactly z0_loss/rollout_scale (l_1step contributes nothing,
        # backward() reproduces the prior, rollout-only behavior
        # precisely). rollout_scale applies to BOTH terms -- they're
        # the same underlying per-step loss, just different
        # aggregations (l_1step is loss restricted to the first step
        # only).
        total = (z0_loss + one_step_weight * l_1step) / rollout_scale
        # l_1step ITSELF must also be scaled before being returned --
        # it's displayed/plotted directly (the "(1step)" figure and the
        # loss_curve.png secondary line), not folded into total, so if
        # only total got the /rollout_scale treatment, l_1step would be
        # on a genuinely different scale than train_loss/val_loss
        # whenever rollout_scale != 1 -- not actually comparable at
        # all, despite being shown side by side every single epoch.
        l_1step_scaled = l_1step / rollout_scale

        if train:
            optimizer.zero_grad()
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(f_theta.parameters(), grad_clip)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

        # Returned as GPU tensors, NOT .item()'d here -- .item() blocks
        # the CPU until the GPU actually finishes and forces a full
        # round trip EVERY batch. With a model this small (a few
        # hundred latent values, hidden_dim=256), that per-batch sync
        # latency dwarfs the actual compute -- exactly why GPU
        # utilization stays low even while wall-clock time doesn't:
        # the GPU is repeatedly finishing near-instantly, then sitting
        # idle while the CPU waits on the sync before it can even
        # launch the NEXT batch's kernels. .detach() (not .item()) --
        # keeps the value as a scalar GPU tensor, cheap to accumulate
        # into a running sum on-device (see the epoch loop below,
        # which now does exactly ONE .item() per epoch, not one per
        # batch) -- while dropping the autograd graph, which backward()
        # above has already consumed and which would otherwise be kept
        # alive (and keep growing) for the rest of the epoch if left
        # attached.
        return total.detach(), l_1step_scaled.detach()

    tracker = CheckpointCriterionTracker(ema_warmup_epochs=ema_warmup_epochs,
                                          val_ema_decay=val_ema_decay)
    epochs_since_improvement = 0

    show_1step = n_rollout_steps > 1  # at n=1, L_1step == L_rollout always -- redundant to show

    print(f"Starting {epochs} epochs (early_stopping_patience: "
          f"{early_stopping_patience}, batches of {batch_size})...")
    if show_1step:
        print(f"/{epochs:3d}  train  (1step)   valid  (1step)     ema")
    else:
        print(f"/{epochs:3d}  train    valid      ema")

    for epoch in range(0 if epochs == 0 else 1, epochs + 1):
        f_theta.train()
        # GPU-resident accumulators, not Python floats -- see step()'s
        # own docstring/comment: `loss * bs` below stays a GPU tensor
        # op (no sync), so the ONLY host sync in this whole loop is the
        # single .item() call after it ends, not one per batch.
        train_loss_sum = torch.zeros((), device=device)
        train_1step_sum = torch.zeros((), device=device)
        if epoch > 0:
            n_train = len(train_set)
            for batch in train_loader:
                bs = batch[0].size(0)
                loss, l_1step = step(batch, train=True)
                train_loss_sum += loss * bs
                train_1step_sum += l_1step * bs
            train_loss = (train_loss_sum / n_train).item()
            train_1step = (train_1step_sum / n_train).item()
        else:
            # epoch 0 (epochs=0 ablation only): no training at all --
            # NaN honestly reflects that these metrics don't apply this
            # "epoch", rather than a misleading 0.0.
            train_loss = train_1step = float("nan")

        f_theta.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_1step_sum = torch.zeros((), device=device)
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0)
                loss, l_1step = step(batch, train=False)
                val_loss_sum += loss * bs
                val_1step_sum += l_1step * bs
        val_loss = (val_loss_sum / n_val).item()
        val_1step = (val_1step_sum / n_val).item()

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
                    "latent_spatial_size": ae_config.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
                    "hidden_dim": hidden_dim, "n_hidden_layers": n_hidden_layers,
                    "dt_cap": dt_cap,
                },
                "data_config": {
                    "min_step": min_step, "min_stdev_phi": min_stdev_phi,
                    "min_passing_steps": min_passing_steps,
                    "window_length": window_length, "n_rollout_steps": n_rollout_steps,
                    "z0_noise_scale": z0_noise_scale,
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
    parser.add_argument("--ae-stats-weight", type=float, default=None, dest="ae_stats_weight",
                         help="the AE checkpoint's own ae_stats_weight (used only to locate its "
                              "expected filename) -- named --ae-stats-weight here since it's "
                              "paired with --ae-latent-channels, but stored as args.ae_stats_weight "
                              "to match train_lds()'s actual parameter name")
    parser.add_argument("--ae-checkpoint", type=Path, default=None)
    parser.add_argument("--size", type=int, required=True,
                         help="grid size (square only) -- locates base/<size>x<size>/, "
                              "reading ITS OWN metadata.txt (not config.txt)")
    parser.add_argument("--base", type=Path, default=_PYTHON_ROOT.parent / "datasets")
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
    parser.add_argument("--min-passing-steps", type=int, default=None,
                         help="exclude an entire run when fewer than this many of its steps clear "
                              "min-stdev-phi -- see build_good_steps' own docstring in "
                              "training/datasets.py")
    parser.add_argument("--condition-on-theta", action=argparse.BooleanOptionalAction, default=None,
                         help="VALIDATES against the loaded AE checkpoint's own deriv stream -- "
                              "stage 3 cannot change this (decided in stage 2); omit to skip "
                              "validation and trust whatever the checkpoint already has")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--val-ema-decay", type=float, default=0.7)
    parser.add_argument("--ema-warmup-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--z0-noise-scale", type=float, default=0.0,
                         help="perturb z0 during training only, scaled per-window to that "
                              "window's own |z1(t0)|*dt Euler-step magnitude -- see "
                              "train_lds()'s own docstring for the full rationale (default "
                              "0.0: off, exact prior behavior)")
    parser.add_argument("--dt-cap", type=float, default=float("inf"),
                         help="caps dt INSIDE f_theta's own second-order correction term only "
                              "(f_val*(min(dt,dt_cap)^2/2)), not the first-order z1*dt term -- "
                              "see LatentDynamics.__init__'s own docstring for why this, rather "
                              "than saturating f_val's own output, is what actually guarantees "
                              "the first-order term dominates again at large dt instead of just "
                              "delaying when the second-order term overtakes it (default inf: "
                              "off, exact prior behavior)")
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
        ae_stats_weight=args.ae_stats_weight,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        hidden_dim=args.hidden_dim, n_hidden_layers=args.n_hidden_layers,
        dt_cap=args.dt_cap,
        val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        num_workers=args.num_workers, n_rollout_steps=args.n_rollout_steps,
        min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        min_passing_steps=args.min_passing_steps,
        condition_on_theta=args.condition_on_theta,
        encode_batch_size=args.encode_batch_size, val_ema_decay=args.val_ema_decay,
        ema_warmup_epochs=args.ema_warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        lr_warmup_steps=args.lr_warmup_steps, grad_clip=args.grad_clip,
        z0_noise_scale=args.z0_noise_scale,
        seed=args.seed, checkpoint_path=args.checkpoint, device=args.device,
        resume_from=args.resume_from, log_every_epoch=not args.quiet,
    )


if __name__ == "__main__":
    main()
