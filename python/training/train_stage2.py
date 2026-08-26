"""
Stage 2 (train_stage2) of the LIENS pipeline -- an importable function,
not just a CLI script; driven by main.py, since it always resumes from
a stage-1 checkpoint (there's no "stage 2 from scratch") and so has no
CLI entry point of its own (see training/train_stage1.py's own module
docstring for that half). Split out of what used to be a combined
train_ae.py once stage 1b's own removal made that combination less
coherent than it once was: train_stage2 now also builds the deriv
stream itself, directly from a stage 1a ancestor (see
training/extend_encoder.py's own module docstring for the full
rationale), a piece of real complexity train_autoencoder has no reason
to know anything about.
"""

import time
from collections.abc import Callable
from models.constants import N_THETA
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.logging_utils import print_run_parameters, EpochProgress
from models.autoencoder import MultiStreamAutoencoder
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_streams import cross_check_stream_configs_against_state_dict, \
                                   resolve_stream_configs_from_checkpoint_config
from training.checkpoint_criterion import (
    CheckpointCriterionTracker, ComponentBestTracker, atomic_torch_save, clamp_grace_epochs,
    grace_epochs_for_ema,
)
from training.checkpoint_components import cross_check_ancestor_config
from training.extend_encoder import extend_state_checkpoint_with_deriv_stream
from training.datasets import VAL_DECORRELATED_AUG_INDICES, MicrostructureEvolutionDataset, complete_run_dirs, split_run_dirs
from training.losses import (ReconLoss, StatsLoss, InterpLoss, centered_deriv_target,
                              dt_weighted_deriv_loss)
from training.stats_head import StatsHead
from training.spike_guard import _SpikeGuard, _record_spike, difficulty_band
from training.train_ae_common import freeze_outer_layers, compute_weight_drift
from utils.naming import ae_checkpoint_name
from utils.plots import loss_component_scatter, loss_curve, write_loss_history, should_write_loss_figure
from evaluation.check_interpolation import check_interpolation
from evaluation.check_perturbation import check_perturbation

# GENERAL POLICY (matches training/train_stage1.py's own identical
# comment, and training/train_refinement.py's own _PYTHON_ROOT): every
# checkpoint/output/dataset path is built from THIS anchor, never from
# a bare relative string -- see train_stage1.py's own copy of this
# comment for the full rationale. Each module that needs this gets its
# own copy (Path(__file__) is necessarily per-file), not a shared
# import -- see conftest.py's own isolated_project_root fixture, which
# redirects every such module's own copy independently during tests.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/training/train_stage2.py -> python/


def _compact_loss(value: float, width: int = 7, precision: int = 4) -> str:
    """`f"{value:7.4f}"`, unless that overflows the field -- then 4 SIGNIFICANT
    figures instead.

    The epoch rows are all small numbers and fixed-point suits them. The
    reference row is not: it reports an UNTRAINED component, and a freshly
    built deriv stream starts around 4e4, which `:7.4f` renders as
    "40613.3085" -- ten characters in a seven-character field, and nine
    significant figures for a number whose leading digit is the only one that
    means anything. It broke the column alignment and implied a precision the
    quantity does not have.

    Identical output for anything that fits, so the epoch rows are unchanged.
    """
    fixed = f"{value:{width}.{precision}f}"
    return fixed if len(fixed) <= width else f"{value:{width}.{precision}g}"


_STAGE2_PREAMBLE_PARAMS = (
    # See train_lds's _LDS_PREAMBLE_PARAMS for why these are excluded here.
    "deriv_weight", "deriv_weight_warmup_epochs", "stats0_weight", "stats1_weight",
    "z0_from_deriv_weight",
    "deriv_dt_weight_exponent", "deriv_target_centered",
    "val_aug_averaging", "recon0_scale", "epochs", "batch_size", "augment",
    "early_stopping_patience", "n_frozen_stages",
)


def train_stage2(
    base_path: Path, resume_from: Path, size: int | None = None,
    deriv_weight: float = 1.0, deriv_weight_warmup_epochs: int = 3, stats0_weight: float = 0.0,
    stats1_weight: float = 0.0, z0_from_deriv_weight: float = 0.0,
    trunk_from_deriv_weight: float = 1.0, stage2a: bool = False,
    deriv_head_hidden: int = 0,
    deriv_dt_weight_exponent: float = 0.0, deriv_target_centered: bool = False,
    interp_weight: float = 0.0, interp_scale: float = 1.0,
    val_aug_averaging: bool = False,
    recon0_scale: float = 1.0,
    stats0_scale: float = 1.0, stats1_scale: float = 1.0, deriv_scale: float = 1.0,
    epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
    val_fraction: float = 0.2, test_fraction: float = 0.1, num_workers: int = 4,
    min_step: int | None = None, min_stdev_phi: float | None = None,
    min_passing_steps: int | None = None,
    min_std_deriv: float | None = None, augment: bool = False,
    condition_on_theta: bool | None = None,
    val_ema_decay: float = 0.7, early_stopping_patience: int | None = None,
    spike_skip_factor: float = 0.0, grad_clip: float = 0.0,
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

    Originally REPLACED the old L_interp entirely (not summed alongside
    it), the design doc's phrasing reading as substitution. That is no
    longer true: L_interp is back as an OPTIONAL term (interp_weight,
    default 0.0 = the old behaviour exactly) and the two now SUM. It
    returned because check_z2_measurability found the latent trajectory
    carries frame-scale roughness -- curvature estimates from disjoint
    stencils anti-correlate (-0.076, 67% of windows negative) and
    E|z2|*dt^2 is flat across five decades of E|z2|, the signature of
    pure encoding noise -- which is exactly the representation artifact
    L_interp penalises and L_deriv does not.

    Both terms share ONE activation epoch (deriv_switch_epoch, from
    deriv_weight_warmup_epochs): L_interp needs the same 3-frame window
    the centered target does, so gating them together means there is no
    way to configure one active without the other's data.

    The (t1,t2,t3)-triplet interpolation-consistency CHECK
    (check_interpolation.py) was UNCHANGED throughout and is still run
    as a before/after diagnostic -- it now measures a quantity that is
    once again also trained on, so it stops being an independent probe
    when interp_weight > 0.

    Full loss: L_recon0 + recon0_weight*L_stats0 + stats1_weight*L_stats1
    + effective_deriv_weight*L_deriv (last two default to 0 except
    deriv_weight -- see below). L_recon0 is the SAME term this function
    already had (state stream, unweighted -- L_recon0's own weight is
    implicitly 1.0, the reference every other term is expressed
    relative to, matching every other stage in this project's own
    convention of never giving the primary recon term an explicit
    weight parameter).

    DEFAULT BEHAVIOR (stats1_weight=0.0): z1 is trained PURELY by
    L_deriv -- a pure latent-space quantity from stage 2 onward, never
    re-anchored to pixel space or to stats_head1 here. This reflects a
    real design decision, not a placeholder: the deriv bottleneck (and
    the shared trunk, to a lesser extent) needs to land in a region
    that already carries real, extractable directional signal before
    L_deriv alone can refine it -- see the zero-signal/checkerboard
    failure this project hit when L_deriv was once attempted with no
    such grounding at all. That grounding used to come from a separate
    stage 1b pass with its own pixel-space D1; this function now builds
    the deriv stream itself, directly from a stage 1a ancestor (see
    training/extend_encoder.py's own module docstring), with no
    equivalent pixel-space warm-start step at all.

    D1/L_recon1 are GONE entirely, not just defaulted off -- D1 is
    confirmed permanently unnecessary (see extend_encoder.py's own
    module docstring), and this function no longer has any way to
    train against it, at any weight. Backward compatibility for a
    LEGACY, already-multi-stream ancestor that still happens to have a
    real D1 decoder in it (see this function's own resume_from
    handling below) is preserved at the LOADING level only -- such a
    checkpoint's own D1 still loads correctly and is carried forward
    unchanged into whatever gets saved, but is never called in this
    function's own forward pass at all, exactly parallel to how D0
    already sits inert in train_lds() (loaded because load_state_dict
    needs the right key structure to succeed, never actually used).
    stats1_weight has NO such restriction -- stats_head1 is always
    available (built fresh for a stage-1a ancestor, loaded for an
    already-multi-stream one), kept "just in case" even though nothing
    trains it by default.

    stats1_weight, if explicitly set nonzero, mirrors the loss term a
    former stage 1b pass used to compute: L_stats1 predicts the SAME
    original stats stats_head0 predicts (not their derivative), via
    stats_head1, reusing the identical true_stats target L_stats0
    already has.

    Accepts EITHER a stage 1a (single-stream) ancestor directly -- the
    deriv stream is built fresh, in memory, right here -- OR an
    already-multi-stream ancestor (e.g. resuming a prior stage 2 run,
    or a legacy stage-1b-derived checkpoint while any still exist on
    disk). Raises clearly if the ancestor has more than one non-recon
    stream (L_deriv has no meaning without exactly one to compare z0's
    own trajectory against) -- a direct, unavoidable consequence of
    L_DERIV being inherently about the C0/C1 split, unlike the old
    L_interp (which worked for any single-stream checkpoint) -- not a
    separate design choice made here. Note L_interp is once again
    available alongside it (interp_weight), and still carries no such
    restriction itself; the requirement comes from L_deriv alone.

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

    deriv_dt_weight_exponent (default 0.0): see dt_weighted_deriv_loss's
    own docstring (losses.py) for the mechanism. 0.0 (default)
    reproduces today's plain, uniform-weight L_deriv exactly.

    CORRECTION to this parameter's own original motivation: it was
    first added believing z1, having no dt input, was forced to
    compromise across MANY different dt values seen by the SAME frame.
    That's wrong -- MicrostructureEvolutionDataset's own sliding-window
    construction gives each frame exactly ONE window and hence ONE dt,
    so there is no such per-frame conflict. What IS real, and worth
    testing instead: L_deriv's one-sided target has an O(dt) truncation
    bias (see deriv_target_centered below, which fixes this directly
    and is likely the better lever) and dt varies systematically with
    a frame's OWN coarsening state across the dataset (see dz0dt.png's
    "vs t" panels), which is a real, if different, dt-correlated
    effect. Left in place as an orthogonal knob, but the evidence
    motivating it needs re-establishing before trusting a nonzero
    value here -- see check_deriv_temperature.py's own eps/eps' split
    for how to check what's actually driving any given dt-dependence
    before reaching for this parameter specifically.

    deriv_target_centered (default False): builds L_deriv's own target
    from a THREE-frame (t-dt_minus, t, t+dt_plus) window instead of
    today's two-frame, one-sided (t, t+dt) one -- see
    centered_deriv_target's own docstring (losses.py) for the full
    derivation. This is the FIX for the O(dt) truncation bias just
    described: the one-sided target is only first-order accurate in
    dt, and check_deriv_temperature.py's own eps' term (a real,
    dt-independent bias, not just noise -- see its own >40 sigma
    significance on real 64x64 data) is exactly what that bias looks
    like once measured. The centered target is second-order accurate
    instead, WITHOUT assuming even spacing (this project's own
    save_steps schedules are not uniform). Changes window_length from
    2 to 3 for both train_set and val_set -- a strictly larger, not
    filtered-differently, set of interior windows (a run's own first
    and last kept step still can't have both neighbors, so those
    windows drop out, same as MicrostructureTripletDataset's own,
    already-existing window_length=3-equivalent construction already
    does for check_interpolation.py).

    val_aug_averaging (default False): evaluate every VALIDATION window
    under datasets.VAL_DECORRELATED_AUG_INDICES' own 4 exact-symmetry
    variants and average, instead of the single untransformed view --
    see that constant's own comment, and fixed_aug_indices' entry in
    MicrostructureEvolutionDataset's own docstring, for the full
    reasoning and for why those specific 4 variants.

    Motivation, measured not assumed: on a real 64x64 run, val_total's
    own per-epoch std was ~0.12 while the deriv term (the one stage 2
    exists to improve) was improving at ~0.0014/epoch -- an ~85:1
    noise-to-signal ratio that makes the save criterion and early
    stopping fire on noise rather than on real progress. The val set is
    the reason: augment applies to TRAIN only, so val ends up ~100x
    smaller in window count (555872 vs 5232 in that run), and
    min_std_deriv had already discarded roughly two thirds of the val
    candidates on top of that.

    Costs 4x the val-set forward passes, which is negligible against an
    augmented train set two orders of magnitude larger. Note this
    CHANGES WHAT val_loss MEANS (a symmetry-averaged mean rather than a
    single-view one), so values aren't directly comparable against runs
    trained without it -- and it does NOT touch training at all, only
    the metric that decides saving/early stopping. Also strictly weaker
    than simply enlarging val_fraction, since these 4 views are
    symmetry-related rather than genuinely independent samples.

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
    # BEFORE adopting the ancestor's size: if the caller stated one, they must
    # agree. The ancestor's size decides both the architecture AND which
    # dataset is read, while the output filename comes from the params file --
    # so a mistyped resume_from trains the wrong model into the right name,
    # silently. See cross_check_ancestor_config for the incident.
    cross_check_ancestor_config(model_cfg, {"size": size}, resume_from,
                                 what="stage-2 ancestor")
    size = model_cfg["size"]
    print(f"Resuming from {resume_from} (stat_names={stat_names}, "
          f"ancestor_stats_weight={ancestor_stats_weight}, this stage's stats0_weight={stats0_weight})")

    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, prev["model_state"],
    )

    # Captured BEFORE the branch below (which reassigns stream_configs
    # in both cases, so by the time either branch finishes this can no
    # longer be told apart from stream_configs alone) -- needed by
    # deriv_target_centered's own epoch-scheduling logic further down,
    # to tell a genuinely-fresh stage-1a ancestor (whose own "epoch" is
    # STAGE 1's, unrelated to any stage-2 progress at all) apart from
    # resuming a PRIOR stage-2 run (whose own "epoch" says exactly how
    # far that prior run got).
    resumed_from_stage2 = len(stream_configs) > 1

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
            device=device, deriv_head_hidden=deriv_head_hidden,
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

        # Upgrade the deriv stream to a residual head (z1 = B y + H(y)) when
        # deriv_head_hidden > 0. The resumed checkpoint's B weights load as
        # before; the new H tensors are zero-initialised and absent from the
        # checkpoint, which the strict=False load below tolerates -- so the
        # encoder starts byte-identical to the ancestor and training grows H.
        # dataclasses.replace keeps every other field of the resumed config.
        if deriv_head_hidden > 0:
            import dataclasses
            deriv_stream = dataclasses.replace(
                deriv_stream, head_kind="residual", head_hidden=deriv_head_hidden)
            stream_configs = {**stream_configs, deriv_stream_name: deriv_stream}
            print(f"upgrading the '{deriv_stream_name}' stream to a residual "
                  f"head (head_hidden={deriv_head_hidden}); H is zero-init, so "
                  f"the encoder starts identical to the ancestor.")

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
                           stream_configs=stream_configs, n_theta=N_THETA)
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
        # Tolerate a checkpoint that predates residual heads: a "residual"
        # stream adds residual_heads.* keys the old (B-only) state_dict lacks,
        # and those are zero-initialised (H(y)=0), so the encoder is
        # byte-identical to the checkpoint until training. Accept ONLY missing
        # residual_heads.* keys; any OTHER missing key, or any unexpected key,
        # is still a real mismatch and must raise.
        # Upgrade any pre-2-feature-theta checkpoint: zero-pad the new theta
        # input column of each conditioner's first Linear so the loaded model
        # is bit-identical in function (the log(T0-T) coordinate starts silent
        # and training grows it). A same-n_theta checkpoint passes through
        # untouched.
        from models.encoder import zero_pad_theta_columns
        _prev_state = zero_pad_theta_columns(prev["model_state"], ae)
        _missing, _unexpected = ae.load_state_dict(_prev_state, strict=False)
        # The AE exposes the encoder under MORE THAN ONE state_dict path
        # (encoders.<name>.* and the per-stream pathways.<stream>.encoder.*
        # aliases of the same shared module), so match the H tensors by the
        # ".residual_heads." segment wherever it appears, not by one prefix.
        _bad_missing = [k for k in _missing if ".residual_heads." not in k]
        if _bad_missing or _unexpected:
            raise RuntimeError(
                f"checkpoint does not match the model: "
                f"missing {_bad_missing}, unexpected {list(_unexpected)}")
        if _missing:
            print(f"resuming a pre-residual-head checkpoint: {len(_missing)} "
                  f"zero-initialised residual-head tensor(s) added "
                  f"(H(y)=0, so the encoder starts identical to the ancestor).")
        # Isolate z0's shared trunk from L_deriv's gradient when asked. The
        # deriv stream still reads the trunk forward (z1 is still computed
        # from it); only the gradient L_deriv sends BACK into the trunk is
        # scaled. 1.0 keeps every prior run bit-identical; 0.0 stops L_deriv
        # from reshaping z0 (the fix for z0's velocity-coherence collapse in
        # stage 2). The deriv stream is the one NOT used for reconstruction.
        if trunk_from_deriv_weight != 1.0:
            # Only touch the encoder when the knob is off its default, so
            # every existing single-stream / default run is bit-identical.
            # NB: reuse deriv_stream_name (the str, bound above); do NOT
            # rebind deriv_stream, which holds the stream CONFIG object and
            # is read later for stats_head1 (.channels/.spatial_size).
            isolatable = [n for n in stream_configs if n != recon_stream_name]
            if not isolatable:
                raise ValueError(
                    "trunk_from_deriv_weight is only meaningful with a deriv "
                    "stream to isolate; this checkpoint has only the recon "
                    f"stream {recon_stream_name!r}.")
            ae.encoders["shared"].set_trunk_grad_scale(
                deriv_stream_name, trunk_from_deriv_weight)
            print(f"trunk_from_deriv_weight={trunk_from_deriv_weight}: L_deriv's "
                  f"gradient into the shared trunk via the '{deriv_stream_name}' "
                  f"stream is scaled by {trunk_from_deriv_weight} "
                  f"(1.0=full, 0.0=z0 trunk frozen against L_deriv).")
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
    interp_loss_fn = InterpLoss()
    if interp_weight > 0 and not deriv_target_centered:
        # L_interp needs the (t-, t, t+) triplet, which only the centered
        # branch encodes; the one-sided path loads window_length=2 and has
        # no middle frame to interpolate TO. Refusing loudly beats silently
        # contributing nothing to the objective the caller asked for.
        raise ValueError(
            f"interp_weight={interp_weight} requires deriv_target_centered=True "
            f"(L_interp needs the 3-frame window the centered target already "
            f"loads; the one-sided path has window_length=2)")

    if checkpoint_path is None:
        name = ae_checkpoint_name(size, model_cfg["latent_channels"], ancestor_stats_weight)
        checkpoint_path = _PYTHON_ROOT / "checkpoints" / "stage2" / f"{name}-stage2.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint: {checkpoint_path}")
    print_run_parameters(train_stage2, locals(), _STAGE2_PREAMBLE_PARAMS)

    if loss_curve_path is None:
        name = ae_checkpoint_name(size, model_cfg["latent_channels"], ancestor_stats_weight)
        loss_curve_path = _PYTHON_ROOT.parent / "output" / "stage2" / f"{name}-stage2-loss_curve.png"
    # Derived from loss_curve_path's own filename (not a new parameter)
    # so every existing call site keeps working unchanged -- same
    # "override just the one path you care about" pattern already used
    # elsewhere in this project (see _latent_eval.py's own
    # dz0dt_output_path/dt_dependence_output_path).
    loss_components_path = loss_curve_path.with_name(
        loss_curve_path.stem + "-components" + loss_curve_path.suffix)

    epoch_history: list[int] = []
    loss_curve_events: list[tuple[float, str]] = []
    loss_curve_levels: list[tuple[float, str]] = []
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_so_far_history: list[float] = []

    # deriv_target_centered's own epoch schedule: conflated with
    # deriv_weight_warmup_epochs deliberately, not a separate knob --
    # while L_deriv's own gradient is still ramping up (a transient
    # shock to the shared trunk is exactly what the ramp exists to
    # avoid), there's little reason to also pay for the more accurate,
    # more expensive target; once the ramp completes and L_deriv is at
    # full strength, that's exactly when using the better target
    # starts to matter. deriv_switch_epoch mirrors
    # effective_deriv_weight's own ramp-completion point EXACTLY (see
    # that formula below): deriv_weight_warmup_epochs itself if
    # positive, else 1 (immediately, matching "no ramp" meaning "full
    # weight from epoch 1" there too).
    #
    # prior_stage2_epochs: 0 for a fresh run OR one resuming a stage-1a
    # ancestor (see resumed_from_stage2 above); a PRIOR stage-2 run's
    # own saved epoch otherwise -- this is what lets resuming an
    # already-fully-trained stage-2 checkpoint (e.g. one trained for
    # 226 epochs, deriv_target_centered=False throughout) start
    # DIRECTLY in the centered phase from this run's own epoch 1,
    # rather than re-running a now-pointless warmup: 226 +
    # (whatever this run's own epoch 1 is) is already far past any
    # reasonable deriv_weight_warmup_epochs.
    deriv_switch_epoch = (
        (deriv_weight_warmup_epochs if deriv_weight_warmup_epochs > 0 else 1)
        if deriv_target_centered else None
    )
    prior_stage2_epochs = prev["epoch"] if resumed_from_stage2 else 0
    if deriv_target_centered:
        already_past_switch = prior_stage2_epochs + 1 >= deriv_switch_epoch
        # deriv_switch_epoch counts CUMULATIVE epochs (the condition below is
        # `prior_stage2_epochs + epoch >= deriv_switch_epoch`), so it is not
        # this run's own epoch number when resuming. Printing it as if it were
        # contradicted the very next clause, which correctly derived the count
        # of one-sided epochs: with switch=10 and prior=2 the message read
        # "at epoch 10 ... spends its first 7 epoch(s)", where 7+1 = 8.
        own_switch_epoch = max(1, deriv_switch_epoch - prior_stage2_epochs)
        print(f"deriv_target_centered=True: switching to the centered L_deriv target at "
              f"epoch {own_switch_epoch} of this run's own numbering "
              f"(cumulative epoch {deriv_switch_epoch}) "
              f"(prior_stage2_epochs={prior_stage2_epochs}, so this run "
              f"{'starts already past that point -- centered from epoch 1' if already_past_switch else f'spends its first {min(deriv_switch_epoch - prior_stage2_epochs - 1, epochs)} epoch(s) in the cheaper, one-sided phase'}).")

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
        val_set = MicrostructureEvolutionDataset(val_dirs, encoder=None,
                                                  window_length=3 if deriv_target_centered else 2,
                                                  stats_frame_index=1 if deriv_target_centered else 0,
                                                  stat_names=stat_names, min_std_deriv=min_std_deriv,
                                                  min_step=min_step, min_stdev_phi=min_stdev_phi,
                                                  min_passing_steps=min_passing_steps,
                                                  fixed_aug_indices=(VAL_DECORRELATED_AUG_INDICES
                                                                      if val_aug_averaging else None),
                                                  split_label="validation")
        print(f"train_set: skipped (epochs=0 ablation -- never iterated over), "
              f"{len(val_set)} val "
              f"{'centered (t-dt_minus, t, t+dt_plus)' if deriv_target_centered else 'consecutive-pair'} windows")
    else:
        train_set = MicrostructureEvolutionDataset(train_dirs, encoder=None,
                                                    window_length=3 if deriv_target_centered else 2,
                                                    stats_frame_index=1 if deriv_target_centered else 0,
                                                    stat_names=stat_names, min_std_deriv=min_std_deriv,
                                                    min_step=min_step, min_stdev_phi=min_stdev_phi,
                                                    min_passing_steps=min_passing_steps,
                                                    augment=augment, split_label="training")
        val_set = MicrostructureEvolutionDataset(val_dirs, encoder=None,
                                                  window_length=3 if deriv_target_centered else 2,
                                                  stats_frame_index=1 if deriv_target_centered else 0,
                                                  stat_names=stat_names, min_std_deriv=min_std_deriv,
                                                  min_step=min_step, min_stdev_phi=min_stdev_phi,
                                                  min_passing_steps=min_passing_steps,
                                                  fixed_aug_indices=(VAL_DECORRELATED_AUG_INDICES
                                                                      if val_aug_averaging else None),
                                                  split_label="validation")
        print(f"{len(train_set)} train / {len(val_set)} val "
              f"{'centered (t-dt_minus, t, t+dt_plus)' if deriv_target_centered else 'consecutive-pair'} windows")
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
    if stage2a and interp_weight == 0:
        # These measure z0-space geometry (stats of interpolated z0 vs encoded
        # z2). stage2a freezes trunk/decoder/z0 and the stats head, so with no
        # L_interp the measured quantities CANNOT change this run -- the two
        # recent restarts printed bit-identical numbers (mean 0.0376). Minutes
        # of repeated cost per restart, zero information: skip with a note.
        print("  (skipping check_interpolation/check_perturbation: stage2a freezes"
              " trunk/decoder/z0\n   and interp_weight=0, so the latent geometry"
              " these measure cannot change this run;\n   the ancestor's numbers"
              " stand)")
        print()
    else:
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
    # filter to only what's actually trainable. stats_head1 is a
    # SEPARATE nn.Module from ae (not one of its submodules), so its
    # own parameters need adding explicitly -- a real, confirmed bug
    # otherwise: stats_head1.requires_grad_(False) is never called in
    # the Stage-1a-direct branch above (deliberately left trainable),
    # but without this line its gradients, though genuinely computed
    # by total.backward(), were never actually applied by
    # optimizer.step() at all -- confirmed directly by comparing
    # stats_head1's own true initial state_dict against its state after
    # real training with stats1_weight=1.0: byte-identical, despite the
    # printed stats1 loss term visibly decreasing epoch over epoch (that
    # decrease came entirely from the shared trunk moving via L_deriv/
    # L_recon, not from stats_head1 itself learning anything at all).
    if stage2a:
        # A frozen trunk with only a LINEAR (1x1) deriv head is the exact
        # configuration already shown to fail: the head cannot express the
        # derivative from fixed features, so L_deriv climbs instead of
        # falling. stage2a is only meaningful with a residual head to train,
        # so require one rather than silently running the futile case.
        if deriv_stream_name not in ae.encoders["shared"].residual_heads:
            raise ValueError(
                "stage2a=True trains ONLY the deriv head with the trunk "
                "frozen, but the deriv stream has a linear (1x1) head, which "
                "cannot fit the derivative from fixed features. Set "
                "deriv_head_hidden>0 so there is a residual head to train, or "
                "run joint (stage2a=False) if you intend to move the trunk.")
        # STAGE 2a: train ONLY the deriv stream's own head (residual_heads +
        # bottleneck + FiLM), everything else -- shared trunk, decoder, the
        # recon stream's head -- frozen. z0 is then bit-unchanged, so its
        # velocity coherence is preserved by construction (no gate needed on
        # it), and L_recon0/L_stats0 drop out of the objective since nothing
        # they touch is trainable. This is the fast, cache-friendly pass that
        # fits z1 against a fixed z0 before any joint (2b) re-training.
        for prm in ae.parameters():
            prm.requires_grad_(False)
        head_prefixes = tuple(
            f"encoders.shared.{mod}.{deriv_stream_name}"
            for mod in ("residual_heads", "bottlenecks", "theta_conditioners"))
        n_head = 0
        for pname, prm in ae.named_parameters():
            if pname.startswith(head_prefixes):
                prm.requires_grad_(True)
                n_head += prm.numel()
        # BatchNorm etc. in the frozen trunk must not update running stats.
        ae.eval()
        for mod in ae.encoders["shared"].bottlenecks[deriv_stream_name].modules():
            mod.train()
        if deriv_stream_name in ae.encoders["shared"].residual_heads:
            ae.encoders["shared"].residual_heads[deriv_stream_name].train()
        print(f"STAGE 2a: head-only training of the '{deriv_stream_name}' "
              f"stream ({n_head} trainable params); trunk, decoder and the "
              f"recon head are frozen and in eval mode. z0 is unchanged.")

    params = [p for p in ae.parameters() if p.requires_grad]
    if stats_head1 is not None:
        params += [p for p in stats_head1.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    # Spike guard: stage 2 had NO spike protection, unlike stage 3 (train_lds)
    # and stages 4/5. When the trunk is trainable (trunk_from_deriv_weight>0,
    # stage2a=False) L_deriv's gradient reaches the shared trunk and stage 2
    # spikes like the others -- observed: train deriv 0.545 -> 11.75 in one
    # epoch, jolting z0 (val recon0 bounced +-40%). The guard skips a batch
    # whose loss exceeds spike_skip_factor x a running median WITHIN its
    # difficulty band (dt_max), the same banded logic train_lds uses so
    # legitimately-hard long-dt batches are not skipped as a block. Off by
    # default (factor 0.0 -> no guard), matching prior stage-2 behavior; a
    # trunk-moving run should set it (10.0 is train_lds's default).
    #
    # CAVEAT specific to stage 2 (not present in the stage-3 caller): when the
    # trunk trains (2b), its BatchNorm running buffers ADVANCE on a skipped
    # batch, because the forward pass has already run by the time the loss --
    # the skip signal -- is known (see spike_guard.snapshot_running_stats' own
    # docstring). No optimizer STEP is taken, but the buffers drift by one
    # batch's momentum. For occasional spikes (1-2 batches/epoch, the design
    # case) this is second-order versus the spike it prevents. If skipping
    # becomes frequent it compounds -- and frequent skipping itself signals lr
    # too high; lower lr rather than relying on the guard as a crutch. (Stage 3
    # is immune: its encoder is frozen and LatentDynamics has no BatchNorm.)
    spike_guard = _SpikeGuard(spike_skip_factor) if spike_skip_factor > 0 else None

    def step(batch, train: bool, deriv_weight_used: float, use_centered: bool = False):
        window, dt_window, theta, true_stats = batch
        window = window.to(device, non_blocking=True)
        dt_window = dt_window.to(device, non_blocking=True)
        theta = theta.to(device, non_blocking=True)
        true_stats = true_stats.to(device, non_blocking=True)

        if deriv_target_centered and use_centered:
            # window_length=3: (t-dt_minus, t, t+dt_plus). z1 -- and
            # everything else in this step (recon, stats, stats1) --
            # is still evaluated at the SAME frame as the other cases
            # below: the one z1 itself encodes. Here that's the MIDDLE
            # frame, not window[:, 0] -- x_before/x_after exist ONLY to
            # build target_deriv below, and contribute no gradient of
            # their own to recon/stats/stats1 at all.
            x_before, x_t, x_after = window[:, 0], window[:, 1], window[:, 2]
            dt_minus = dt_window[:, 0].view(-1, 1, 1, 1)
            dt_plus = dt_window[:, 1].view(-1, 1, 1, 1)
        elif deriv_target_centered:
            # Cheap phase (use_centered=False, only possible when
            # deriv_target_centered=True -- see deriv_switch_epoch's
            # own comment above): the SAME window_length=3 dataset as
            # the centered case, but x_before/dt_minus are simply never
            # touched -- 2 encodes here (x_t, x_after), not 3, exactly
            # matching the OLD one-sided cost. x_t is STILL window[:,1]
            # (the middle frame), NOT window[:,0] -- deliberately kept
            # identical to the centered case above, so stats_frame_index=1
            # (see the dataset construction above) stays correct
            # regardless of which phase is active THIS epoch, rather
            # than needing two different dataset configurations for one
            # run.
            x_t = window[:, 1]
            x_next = window[:, 2]
            dt = dt_window[:, 1].view(-1, 1, 1, 1)
        else:
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
            # device=device explicitly -- a bare torch.tensor(0.0)
            # defaults to CPU regardless of what device training is
            # actually running on. Harmless back when this got
            # .item()'d immediately below (a plain Python float has no
            # device), but step() now returns .detach()'d tensors for
            # on-device accumulation (see the epoch loop's own
            # comment) -- without this, a CUDA run would raise a
            # device-mismatch error the moment this got added into a
            # CUDA-resident running sum, every time stats_head1 is None.
            stats1_loss_val = torch.tensor(0.0, device=device)

        # L_deriv: z1(t) (the deriv stream's OWN encode of x(t)) pulled
        # toward what z0's own latent trajectory implies its rate of
        # change should be -- NOT derivable from pixel-space alone (per
        # the project's own design doc): a purely LATENT-space
        # comparison, never decoding z1 at all.
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
        # Defined on BOTH branches: the one-sided path has no middle frame,
        # so L_interp is structurally unavailable there. interp_weight > 0
        # with a one-sided target is refused at construction, so this zero
        # is only ever reached with interp_weight == 0.
        interp = torch.tensor(0.0, device=device)
        if deriv_target_centered and use_centered:
            if z0_from_deriv_weight > 0:
                z0_before = ae.encoders["shared"](x_before, theta=theta)[recon_stream_name]
                z0_after = ae.encoders["shared"](x_after, theta=theta)[recon_stream_name]
                z0_before_for_deriv = z0_before.detach() + z0_from_deriv_weight * (z0_before - z0_before.detach())
                z0_after_for_deriv = z0_after.detach() + z0_from_deriv_weight * (z0_after - z0_after.detach())
                z0_t_for_deriv = z0_t.detach() + z0_from_deriv_weight * (z0_t - z0_t.detach())
            else:
                with torch.no_grad():
                    z0_before = ae.encoders["shared"](x_before, theta=theta)[recon_stream_name]
                    z0_after = ae.encoders["shared"](x_after, theta=theta)[recon_stream_name]
                z0_before_for_deriv = z0_before
                z0_after_for_deriv = z0_after
                z0_t_for_deriv = z0_t.detach()
            # See centered_deriv_target's own docstring (losses.py) for
            # the full derivation -- second-order accurate even under
            # the UNEVEN spacing this project's own save_steps
            # schedules actually have, unlike a plain symmetric
            # (z0_after-z0_before)/(dt_minus+dt_plus) which silently
            # assumes dt_minus == dt_plus.
            target_deriv = centered_deriv_target(z0_before_for_deriv, z0_t_for_deriv, z0_after_for_deriv,
                                                  dt_minus, dt_plus)
            # L_interp, on the SAME triplet the centered target already
            # encoded -- no second loader, no extra encode, and no RNG
            # perturbation when disabled. It belongs here rather than in
            # stage 1 for exactly that reason: stage 1 iterates single
            # snapshots and would need a parallel triplet loader.
            #
            # UNLIKE the deriv target, this uses the UNDETACHED z0s: the
            # whole point is to shape z0's own geometry through time, so
            # routing it through z0_*_for_deriv (which detaches unless
            # z0_from_deriv_weight > 0) would silence it by default.
            #
            # alpha = dt_minus / (dt_minus + dt_plus) -- where t sits
            # BETWEEN its neighbours. Not 0.5: the save schedule is
            # geometric, so a midpoint blend would ask the encoding to be
            # wrong by exactly the spacing asymmetry.
            if interp_weight > 0:
                alpha_interp = dt_minus / (dt_minus + dt_plus)
                interp = interp_loss_fn(z0_before, z0_t, z0_after, alpha_interp)
            # dt_minus+dt_plus (the TOTAL span this target draws on),
            # not either half alone -- dt_weighted_deriv_loss's own
            # weighting is about how much a WINDOW should count
            # relative to others, and this window's real extent in
            # time is the full span, matching how the non-centered
            # path's own dt already means exactly that (there, the
            # window's only span IS the one gap).
            dt_for_weighting = dt_minus + dt_plus
        else:
            # theta passed even though only the state stream's output is
            # kept below -- Encoder.forward computes EVERY stream in one
            # pass internally (see its own docstring), so it still needs
            # theta if ANY of its streams (deriv, here) requires it,
            # regardless of which single stream this particular call
            # site goes on to use.
            if z0_from_deriv_weight > 0:
                z0_next = ae.encoders["shared"](x_next, theta=theta)[recon_stream_name]
                z0_next_for_deriv = z0_next.detach() + z0_from_deriv_weight * (z0_next - z0_next.detach())
                z0_t_for_deriv = z0_t.detach() + z0_from_deriv_weight * (z0_t - z0_t.detach())
            else:
                with torch.no_grad():
                    z0_next = ae.encoders["shared"](x_next, theta=theta)[recon_stream_name]
                z0_next_for_deriv = z0_next
                z0_t_for_deriv = z0_t.detach()
            target_deriv = (z0_next_for_deriv - z0_t_for_deriv) / dt
            dt_for_weighting = dt
        deriv_loss = dt_weighted_deriv_loss(recon_loss, z1_t, target_deriv, dt_for_weighting,
                                             deriv_dt_weight_exponent)

        total = (recon / recon0_scale
                 + stats0_weight * stats_loss_val / stats0_scale
                 + stats1_weight * stats1_loss_val / stats1_scale
                 + deriv_weight_used * deriv_loss / deriv_scale
                 + interp_weight * interp / interp_scale)

        if train:
            # Guard reads the loss BEFORE backward(): a spiked (or non-finite)
            # loss must never reach backward()/optimizer.step(). Banded by the
            # batch's own dt_max so hard long-dt batches are judged against
            # their own population, not skipped as a block (see train_lds and
            # difficulty_band). Off entirely when spike_guard is None.
            if spike_guard is not None:
                _band = difficulty_band(float(dt_window.detach().max()))
                if spike_guard.should_skip(float(total.detach()), band=_band):
                    _record_spike(spike_guard, total, dt_window, theta)
                    optimizer.zero_grad()
                    return (total.detach(), recon.detach(),
                            stats_loss_val.detach(), stats1_loss_val.detach(),
                            deriv_loss.detach(), interp.detach())
            optimizer.zero_grad()
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in ae.parameters() if p.requires_grad), grad_clip)
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
        # attached. Same fix as train_lds.py's own step() -- see that
        # function's own identical comment for the full rationale.
        return (total.detach(), recon.detach(),
                stats_loss_val.detach(), stats1_loss_val.detach(),
                deriv_loss.detach(), interp.detach())

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
        (stats0_weight, stats_label, stats0_scale),
        (stats1_weight, stats1_label, stats1_scale), (deriv_weight, "deriv", deriv_scale),
        (interp_weight, "interp", interp_scale),
        # interp belongs here, not just in the total: EVERYTHING that
        # reports the loss breakdown derives from this list -- the console
        # formula line, the per-epoch component columns, component_names,
        # and loss_component_scatter. Omitting it made a run with
        # interp_weight=0.5 print a three-term breakdown whose parts could
        # not sum to the total it printed beside them.
    ] if w > 0]

    # loss_component_scatter's own bookkeeping -- recon0 (always present)
    # plus whichever of stats0/stats1/deriv are actually active THIS run
    # (active_terms, computed just above, is what decides that set).
    component_names = ["recon0"] + [lbl for _, lbl, _ in active_terms]
    component_histories: dict[str, dict[str, list[float]]] = {
        name: {"train": [], "val": [], "best_so_far": []} for name in component_names
    }
    component_best_tracker = ComponentBestTracker()

    print(f"Stage 2: starting {epochs} epochs (early_stopping_patience: "
          f"{early_stopping_patience}, batches of {batch_size}), "
          f"deriv_weight={deriv_weight}"
          f"{f' (ramped over {deriv_weight_warmup_epochs} epochs for training)' if deriv_weight_warmup_epochs > 0 else ''}"
          f", stats0_weight={stats0_weight}"
          f"{' (anchor active)' if stats0_weight > 0 else ' (diagnostic only, not optimized)'}"
          f", stats1_weight={stats1_weight}"
          f"{' (anchor active)' if stats_head1 is not None and stats1_weight > 0 else ' (inactive)'}"
          f", augment={augment}"
          f", z0_from_deriv_weight={z0_from_deriv_weight}"
          f"{' (WARNING: nonzero -- z0 can now be shaped by L_deriv, not just L_recon)' if z0_from_deriv_weight > 0 else ''}"
          f", deriv_dt_weight_exponent={deriv_dt_weight_exponent}"
          f"{' (favoring small-dt windows in L_deriv)' if deriv_dt_weight_exponent > 0 else ''}"
          f", deriv_target_centered={deriv_target_centered}"
          f"{' (second-order accurate L_deriv target, window_length=3)' if deriv_target_centered else ''}"
          f", val_aug_averaging={val_aug_averaging}"
          f"{f' (val_loss averaged over {len(VAL_DECORRELATED_AUG_INDICES)} symmetry variants -- NOT comparable to runs without it)' if val_aug_averaging else ''}")
    formula = " ".join(f"+{w}*{lbl}/{s}" for w, lbl, s in active_terms)
    print(f"/{epochs:3d} train = recon0/{recon0_scale} {formula} | valid = ...  | ema")

    _prev_use_centered = False  # see just_switched's own comment inside the loop below

    ref_components_for_scatter = None  # populated by the pre-run baseline eval below
    if epochs > 0:
        # RNG state saved/restored around this ENTIRE block -- confirmed
        # directly (via a real A/B run) that without this, the reference
        # evaluation's own forward passes shift torch's global RNG state
        # enough to change train_loader's own shuffle order at epoch 1
        # (shuffle=True), silently changing the actual training
        # trajectory depending on whether this purely-diagnostic
        # reference row is present at all -- exactly the kind of
        # behavior-changing side effect a "just for reference, printed
        # only" feature must never have.
        _rng_state = torch.get_rng_state()
        _cuda_rng_state = torch.cuda.get_rng_state() if device.type == "cuda" else None
        # epoch-0 reference: how does the model resume_from actually
        # loaded perform, BEFORE this run's own training touches it at
        # all -- printed for comparison against epoch 1 onward, same
        # format, but deliberately NOT a real epoch: never calls
        # tracker.update, never saves a checkpoint, never touches
        # epoch_history/the loss curve. If it DID participate in any
        # of those, it would ALWAYS "win" the very first save (nothing
        # beats CheckpointCriterionTracker's own starting
        # best_val_loss=inf), silently saving a checkpoint labeled
        # "epoch 0" before any real training happened -- and a LATER
        # resume's own prior_stage2_epochs logic (see
        # _resolve_stage_specific_ancestor's own docstring) would then
        # misread that as "this model has had zero epochs of training
        # since this checkpoint's own ancestor", even if real training
        # this run then failed to beat it.
        #
        # use_centered computed as epoch 1 itself would (not epoch 0)
        # -- this reference is only a fair "before" snapshot if it's
        # evaluated under the SAME target definition epoch 1 is about
        # to optimize against, not whichever phase would apply at
        # epoch 0 specifically (which can genuinely differ right at a
        # deriv_target_centered switch boundary).
        reference_use_centered = bool(deriv_target_centered) and (
            prior_stage2_epochs + 1 >= deriv_switch_epoch)
        ae.eval()
        ref_total_sum = torch.zeros((), device=device)
        ref_recon_sum = torch.zeros((), device=device)
        ref_stats_sum = torch.zeros((), device=device)
        ref_stats1_sum = torch.zeros((), device=device)
        ref_deriv_sum = torch.zeros((), device=device)
        ref_interp_sum = torch.zeros((), device=device)
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0)
                total, recon, stats, stats1, deriv, interp_val = step(
                    batch, train=False, deriv_weight_used=deriv_weight,
                    use_centered=reference_use_centered)
                ref_total_sum += total * bs
                ref_recon_sum += recon * bs
                ref_stats_sum += stats * bs
                ref_stats1_sum += stats1 * bs
                ref_deriv_sum += deriv * bs
                ref_interp_sum += interp_val * bs
        ref_total = (ref_total_sum / n_val).item()
        # The reference IS the ancestor's val_loss under this run's own
        # objective, and this pass has just measured it -- previously it was
        # printed and thrown away. Handing it to the tracker as a ceiling means
        # a resumed run can never save something WORSE than it started from.
        #
        # Observed without it: reference 3.9054, then epoch 1 saved at 4.0149
        # and epochs 2-3 were worse still (4.35, 4.80) before epoch 4 finally
        # beat the ancestor. The better checkpoint had already been overwritten
        # and survived only because the backup fired.
        #
        # Note this bar is superseded at the deriv_target_centered switch,
        # which calls reset_with_grace -- correctly, since a reference measured
        # under the one-sided target is not a fair bar for the centered one.
        # ONLY when the ancestor is itself a stage-2 checkpoint. The ceiling
        # exists to stop a resumed run overwriting a BETTER checkpoint with a
        # worse one -- and when the ancestor is a single-stream stage-1a
        # checkpoint there is nothing to overwrite: the output is a new file in
        # a different format ("encoders.shared.*" vs "encoder.*"), so no
        # earlier stage-2 result is at risk.
        #
        # Applying it there anyway blocked saving without providing any
        # fallback, so a short run that did not improve produced NO checkpoint
        # at all -- observed as a stage-2 RuntimeError in a test whose ancestor
        # was a stage-1a checkpoint.
        if resumed_from_stage2:
            tracker.reference_val_loss = ref_total
        # The same number as a HORIZONTAL line on the loss curve: the bar the
        # run starts from, measured under this run's own objective. Withheld
        # when a mid-run target switch is coming, for the reason the ceiling
        # itself is superseded there -- a reference measured under the
        # one-sided target is not a fair bar for the centered one, so a flat
        # line drawn across the switch would invite exactly the false
        # comparison. In that case the vertical switch marker is the
        # informative annotation instead; the two are mutually exclusive by
        # construction, which is what makes the already-centered run (no
        # marker possible at epoch 1) get a level instead of nothing.
        switch_lands_mid_run = (
            bool(deriv_target_centered)
            and prior_stage2_epochs + 1 < deriv_switch_epoch <= prior_stage2_epochs + epochs
        )
        if not switch_lands_mid_run:
            loss_curve_levels.append(
                (ref_total, "reference (ancestor, this run's objective)"))
        ref_recon = (ref_recon_sum / n_val).item()
        ref_stats = (ref_stats_sum / n_val).item()
        ref_stats1 = (ref_stats1_sum / n_val).item()
        ref_deriv = (ref_deriv_sum / n_val).item()
        ref_interp = (ref_interp_sum / n_val).item()
        ref_term_values = {
            stats_label: (stats0_weight, stats0_scale, ref_stats),
            stats1_label: (stats1_weight, stats1_scale, ref_stats1),
            "deriv": (deriv_weight, deriv_scale, ref_deriv),
            "interp": (interp_weight, interp_scale, ref_interp),
        }
        ref_terms = " ".join(f"+{_compact_loss(w * v / s)}" for lbl, (w, s, v) in ref_term_values.items()
                              if any(lbl == l for _, l, _ in active_terms))
        # Retain the ref (pre-run baseline) as WEIGHTED, SCALE-NORMALIZED
        # contributions -- the same quantity loss_component_scatter plots for
        # every epoch -- so the scatter can mark where the run started from the
        # resumed checkpoint (a purple circle) alongside its own trajectory.
        ref_components_for_scatter = {"recon0": ref_recon / recon0_scale}
        for lbl, (w, s, v) in ref_term_values.items():
            if any(lbl == l for _, l, _ in active_terms):
                ref_components_for_scatter[lbl] = w * v / s
        nan_terms = " ".join(f"+{_compact_loss(float('nan'))}" for lbl, (_, _, _) in ref_term_values.items()
                              if any(lbl == l for _, l, _ in active_terms))
        print(f"{'ref':>4}|"
              f"{_compact_loss(float('nan'))} ={_compact_loss(float('nan'))} {nan_terms} |"
              f"{_compact_loss(ref_total)} ={_compact_loss(ref_recon / recon0_scale)} "
              f"{ref_terms} |"
              f"{'(before this run)':>9}")
        torch.set_rng_state(_rng_state)
        if _cuda_rng_state is not None:
            torch.cuda.set_rng_state(_cuda_rng_state)

    for epoch in range(0 if epochs == 0 else 1, epochs + 1):
        # Linear ramp: 0 at epoch 0 (never reached, epochs are 1-indexed)
        # up to deriv_weight at epoch=deriv_weight_warmup_epochs and
        # beyond. deriv_weight_warmup_epochs=0 (opt-out) skips this
        # entirely -- full weight from epoch 1, byte-identical to before
        # this parameter existed.
        #
        # ramp_epoch: epoch itself when deriv_target_centered=False --
        # EVERY existing resume_from use case (unrelated to this
        # feature) keeps its own long-standing behavior exactly,
        # including the ramp restarting on resume, unchanged. Only
        # when deriv_target_centered=True does this become
        # prior_stage2_epochs-aware, so the weight ramp and
        # use_centered_this_epoch's own switch (below) move together --
        # a run resuming a stage-2 ancestor already well past
        # deriv_weight_warmup_epochs starts at BOTH full deriv_weight
        # AND the centered target from its own epoch 1, rather than
        # re-ramping the weight while use_centered has already jumped
        # ahead of it.
        ramp_epoch = (prior_stage2_epochs + epoch) if deriv_target_centered else epoch
        effective_deriv_weight = (
            deriv_weight if deriv_weight_warmup_epochs <= 0
            else deriv_weight * min(1.0, ramp_epoch / deriv_weight_warmup_epochs)
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

        # use_centered_this_epoch: see deriv_switch_epoch's own comment
        # above for the full schedule. False for every epoch when
        # deriv_target_centered=False (step()'s own "and use_centered"
        # checks then never matter, but keeping this explicit rather
        # than None avoids a separate falsy-vs-None branch everywhere
        # else below).
        use_centered_this_epoch = bool(deriv_target_centered) and (
            prior_stage2_epochs + epoch >= deriv_switch_epoch)
        # just_switched: True on the FIRST epoch this run itself
        # transitions from the cheap to the centered phase (not true
        # for a run that starts ALREADY past the switch point --
        # tracker.best_val_loss already starts at its own fresh
        # float("inf") every call, per CheckpointCriterionTracker's own
        # construction below, so a run that's centered from epoch 1 has
        # nothing stale to reset in the first place).
        just_switched = use_centered_this_epoch and not _prev_use_centered
        _prev_use_centered = use_centered_this_epoch
        if just_switched:
            # Remembered for the loss curve: the switch drops val_loss sharply
            # in ONE epoch because the measured QUANTITY changed, not the
            # model. Unmarked, that cliff is the most prominent feature of the
            # figure and reads as a learning event. x at epoch-0.5 so the line
            # sits BETWEEN the last old-target point and the first new one.
            #
            # BUT only when epoch > 1: at epoch 1 there IS no earlier point on
            # THIS run's own curve for the marker to sit between (the epoch-0
            # reference above is deliberately never added to epoch_history),
            # regardless of whether the switch is real relative to the loaded
            # checkpoint. Without this guard, a run that is already past the
            # switch point at its own epoch 1 -- deriv_target_centered=True
            # from a fresh run, or an ancestor resumed well past
            # deriv_switch_epoch -- got a vertical line at epoch 0.5 with
            # nothing on either side of it: a discontinuity marker for a
            # discontinuity that isn't in this figure. The print message and
            # the grace-period reset below are UNCHANGED by this guard -- both
            # are about comparability against the loaded checkpoint's own
            # val_loss, which is a real concern at epoch 1 too and stays
            # correct; only the plotted marker is curve-local.
            if epoch > 1:
                # Names BOTH terms when L_interp is on: they share this
                # switch epoch, so the cliff at it is the sum of two
                # objective changes and attributing it to the target
                # change alone would understate what moved.
                label = ("centered L_deriv + L_interp" if interp_weight > 0
                          else "centered L_deriv target")
                loss_curve_events.append((epoch - 0.5, label))
            # grace_epochs derived from val_ema_decay's own effective
            # averaging window (1/(1-decay)) -- max(2, ...) specifically,
            # not max(1, ...): confirmed directly that a single-epoch
            # grace period is mathematically IDENTICAL to the original
            # bug (see reset_with_grace's own docstring) -- with only one
            # epoch, there's no second value for the ema to blend with,
            # so best_val_loss at the moment grace ends is exactly that
            # one epoch's own raw value, lucky or not.
            grace_epochs = grace_epochs_for_ema(val_ema_decay)
            # Clamped against the epochs ACTUALLY remaining after this
            # one -- see clamp_grace_epochs' own docstring. Without
            # this, a switch late in a short run (or any run with fewer
            # epochs left than grace_epochs) covers every remaining
            # epoch, so NO checkpoint is ever saved and the run
            # produces no output file at all.
            grace_epochs = clamp_grace_epochs(grace_epochs, epochs - epoch + 1)
            if grace_epochs > 0:
                print(f"  [epoch {epoch}: switching to the centered L_deriv target now -- "
                      f"giving the tracker a {grace_epochs}-epoch grace period before comparing "
                      f"again, since val_loss computed under the OLD target isn't a fair bar for "
                      f"the NEW target's own val_loss to clear]")
            else:
                print(f"  [epoch {epoch}: switching to the centered L_deriv target now -- "
                      f"resetting the save criterion with NO grace period, since too few epochs "
                      f"remain ({epochs - epoch + 1}) to spend any on grace while still leaving "
                      f"one that can save]")
            tracker.reset_with_grace(grace_epochs)

        ae.train()
        if stage2a:
            # ae.train() above is recursive and just flipped the WHOLE model
            # -- including the stage2a-frozen trunk and decoder -- back to
            # train mode, which lets BatchNorm running stats drift via the
            # forward-pass EMA even though requires_grad_(False) stops the
            # gradient updates (measured: buffer drift 21.2 on a "frozen"
            # trunk before this guard). Re-apply eval to everything, then
            # train mode ONLY on the deriv head's own modules.
            ae.eval()
            ae.encoders["shared"].bottlenecks[deriv_stream_name].train()
            if deriv_stream_name in ae.encoders["shared"].theta_conditioners:
                ae.encoders["shared"].theta_conditioners[deriv_stream_name].train()
            if deriv_stream_name in ae.encoders["shared"].residual_heads:
                ae.encoders["shared"].residual_heads[deriv_stream_name].train()
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
        # GPU-resident accumulators, not Python floats -- see step()'s
        # own docstring/comment: `x.detach() * bs` below stays a GPU
        # tensor op (no sync), so the ONLY host sync per phase (train/
        # val) is the batch of five .item() calls after each loop ends,
        # not five per batch. Same fix as train_lds.py's own epoch
        # loop -- see that function's own identical comment.
        train_total_sum = torch.zeros((), device=device)
        train_recon_sum = torch.zeros((), device=device)
        train_stats_sum = torch.zeros((), device=device)
        train_stats1_sum = torch.zeros((), device=device)
        train_deriv_sum = torch.zeros((), device=device)
        train_interp_sum = torch.zeros((), device=device)
        if epoch > 0:
            n_train = len(train_set)
            _n_train_batches = 0
            _epoch_progress = EpochProgress(len(train_loader))
            for batch in train_loader:
                _epoch_progress.tick()
                bs = batch[0].size(0)
                _n_train_batches += 1
                total, recon, stats, stats1, deriv, interp_val = step(
                    batch, train=True, deriv_weight_used=effective_deriv_weight,
                    use_centered=use_centered_this_epoch)
                train_total_sum += total * bs
                train_recon_sum += recon * bs
                train_stats_sum += stats * bs
                train_stats1_sum += stats1 * bs
                train_deriv_sum += deriv * bs
                train_interp_sum += interp_val * bs
            _epoch_progress.close()
            if spike_guard is not None:
                _n_skipped = spike_guard.n_skipped_this_epoch
                _worst = spike_guard.worst
                _deadlocked = spike_guard.end_epoch(_n_train_batches)
                if _n_skipped:
                    _w = (f" worst loss {_SpikeGuard.median_display(_worst[0])} "
                          f"(band median {_SpikeGuard.median_display(_worst[1])}, "
                          f"dt_max={_worst[2]:.0f}, theta0={_worst[3]:.4f})"
                          if _worst else "")
                    print(f"  spike guard: skipped {_n_skipped}/{_n_train_batches} "
                          f"train batch(es) this epoch (loss > {spike_skip_factor}x "
                          f"band median).{_w}")
                if _deadlocked:
                    print(f"  spike guard: WARNING every train batch skipped this "
                          f"epoch -- weights took no step. If this persists the run "
                          f"is deadlocked (broken weights make every batch an "
                          f"outlier); lower lr or raise spike_skip_factor.")
            train_total = (train_total_sum / n_train).item()
            train_recon = (train_recon_sum / n_train).item()
            train_stats = (train_stats_sum / n_train).item()
            train_stats1 = (train_stats1_sum / n_train).item()
            train_deriv = (train_deriv_sum / n_train).item()
            train_interp = (train_interp_sum / n_train).item()
        else:
            # epoch 0 (epochs=0 ablation only): no training at all --
            # NaN honestly reflects that these metrics don't apply this
            # "epoch", rather than a misleading 0.0.
            train_total = train_recon = train_stats = train_stats1 = train_deriv = float("nan")
            train_interp = float("nan")

        ae.eval()
        val_total_sum = torch.zeros((), device=device)
        val_recon_sum = torch.zeros((), device=device)
        val_stats_sum = torch.zeros((), device=device)
        val_stats1_sum = torch.zeros((), device=device)
        val_deriv_sum = torch.zeros((), device=device)
        val_interp_sum = torch.zeros((), device=device)
        n_val = len(val_set)
        with torch.no_grad():
            for batch in val_loader:
                bs = batch[0].size(0)
                total, recon, stats, stats1, deriv, interp_val = step(
                    batch, train=False, deriv_weight_used=val_deriv_weight,
                    use_centered=use_centered_this_epoch)
                val_total_sum += total * bs
                val_recon_sum += recon * bs
                val_stats_sum += stats * bs
                val_stats1_sum += stats1 * bs
                val_deriv_sum += deriv * bs
                val_interp_sum += interp_val * bs
        val_total = (val_total_sum / n_val).item()
        val_recon = (val_recon_sum / n_val).item()
        val_stats = (val_stats_sum / n_val).item()
        val_stats1 = (val_stats1_sum / n_val).item()
        val_deriv = (val_deriv_sum / n_val).item()
        val_interp = (val_interp_sum / n_val).item()

        # Captured BEFORE update(), which decrements the grace counter and
        # flips in_grace_period to False on the FINAL grace epoch -- an epoch
        # that still could not save. Reading it afterwards would count that one
        # as a real non-improvement.
        was_in_grace_period = tracker.in_grace_period
        _, saved_this_epoch = tracker.update(epoch, val_total)
        val_ema = tracker.val_ema
        val_ema_str = f"{val_ema:7.4f}" if val_ema is not None else "(warmup)"

        # (train_weight, val_weight, scale, train_value, val_value) for
        # every entry, uniformly -- even stats0/stats1, whose two
        # weights are always equal, since they're never ramped. Only
        # "deriv" actually has train_weight != val_weight now -- giving
        # every entry the same 5-tuple shape keeps the two loops below
        # simple, rather than special-casing "deriv" alone. Necessary,
        # not just tidy: these ARE what actually reproduces train_total/
        # val_total's own deriv component (step() only returns the raw,
        # unweighted deriv_loss; the weight multiplication happens HERE,
        # separately from step()'s own total computation) -- using one
        # shared weight for both sides here would silently print a val
        # breakdown that no longer sums to the correct val_total, even
        # though val_total itself (computed inside step(), which DOES
        # already use the right weight) stayed correct.
        #
        # Moved ABOVE loss_curve()/loss_component_scatter() (used to sit
        # just below) so the same weighted, scale-normalized values feed
        # BOTH the console breakdown below AND the per-component history
        # that loss_component_scatter needs -- computing it twice would
        # risk the two silently drifting apart from each other.
        term_values = {
            stats_label: (stats0_weight, stats0_weight, stats0_scale, train_stats, val_stats),
            stats1_label: (stats1_weight, stats1_weight, stats1_scale, train_stats1, val_stats1),
            "deriv": (effective_deriv_weight, val_deriv_weight, deriv_scale, train_deriv, val_deriv),
            "interp": (interp_weight, interp_weight, interp_scale, train_interp, val_interp),
        }

        epoch_history.append(epoch)
        train_loss_history.append(train_total)
        val_loss_history.append(val_total)
        best_so_far_history.append(tracker.best_val_loss)
        if should_write_loss_figure(epoch, log_every_epoch):
            loss_curve(
                epoch_history, train_loss_history, val_loss_history, best_so_far_history,
                loss_curve_path, title="Stage 2 loss", event_epochs=loss_curve_events,
                reference_levels=loss_curve_levels,
            )
            write_loss_history(loss_curve_path, epoch_history, train_loss_history, val_loss_history, best_so_far_history)

        # loss_component_scatter: recon0 (always present, weight 1) plus
        # whichever of stats0/stats1/deriv are actually active this run
        # -- same active_terms filter the console breakdown already
        # uses, so the two never show a different set of components.
        current_train_components = {"recon0": train_recon / recon0_scale}
        current_val_components = {"recon0": val_recon / recon0_scale}
        for _, lbl, _ in active_terms:
            tw, vw, s, tv, vv = term_values[lbl]
            current_train_components[lbl] = tw * tv / s
            current_val_components[lbl] = vw * vv / s
        best_components = component_best_tracker.update(current_val_components, saved_this_epoch)
        for name in current_train_components:
            component_histories[name]["train"].append(current_train_components[name])
            component_histories[name]["val"].append(current_val_components[name])
            component_histories[name]["best_so_far"].append(best_components[name])
        if should_write_loss_figure(epoch, log_every_epoch) and not stage2a:
            # In stage 2a only the deriv stream trains -- recon0 and stats0 are
            # frozen and dead-flat, so a stacked-component scatter is two flat
            # bands plus deriv: visual noise. The single moving term is already
            # in the per-epoch console line and the plain loss_curve below.
            loss_component_scatter(
                epoch_history, component_histories, loss_components_path,
                title="Stage 2 loss components",
                ref_components=ref_components_for_scatter,
            )

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
            atomic_torch_save({
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
                               "mode": cfg.mode.value, "condition_on_theta": cfg.condition_on_theta,
                               "head_kind": cfg.head_kind, "head_hidden": cfg.head_hidden}
                        for name, cfg in stream_configs.items()
                    },
                    "recon_stream_name": recon_stream_name,
                    "decoder_for_stream": decoder_for_stream,
                },
                "stats_config": {"stat_names": stat_names, "stats_mean": mean.cpu(), "stats_std": std.cpu()},
                # Mirrors train_lds's own data_config, and exists for the
                # same reason: a diagnostic reading this checkpoint has
                # no other way to reproduce the window set it was
                # actually trained on, and its own CLI defaults
                # (min_step=None -> 0, min_stdev_phi=None) mean NO
                # filtering at all -- silently a different, much larger
                # population than training ever saw.
                #
                # min_std_deriv especially: it is applied ONLY here, in
                # stage 2, and on real 64x64 data discards tens of
                # thousands of windows (33683 train / 13090 val in one
                # real run). Until this was saved it appeared in no
                # checkpoint at all, so every stage-2 evaluation
                # SILENTLY ran against windows training had deliberately
                # excluded.
                #
                # Saving it made that difference REPORTABLE, not
                # reproducible -- an important distinction. The filter
                # is raw-pixel-only (it thresholds the spatial std of
                # the pixel-space derivative), and
                # MicrostructureEvolutionDataset rejects it outright in
                # cached-latent mode, where it has no defined meaning.
                # train_stage2 can apply it only because it trains E/D
                # and therefore runs in raw-pixel mode; every stage-2
                # DIAGNOSTIC uses a frozen encoder instead. So
                # check_deriv_temperature reads this value and prints an
                # explicit NOTE that its window population differs from
                # training's -- it does not, and cannot, match it.
                "data_config": {
                    "min_step": min_step, "min_stdev_phi": min_stdev_phi,
                    "min_passing_steps": min_passing_steps, "min_std_deriv": min_std_deriv,
                    "window_length": 3 if deriv_target_centered else 2,
                    "augment": augment,
                },
                "stage2_config": {"deriv_weight": deriv_weight,
                                   "interp_weight": interp_weight,
                                   "interp_scale": interp_scale,
                                   "trunk_from_deriv_weight": trunk_from_deriv_weight,
                                   "stage2a": stage2a,
                                   "deriv_weight_warmup_epochs": deriv_weight_warmup_epochs,
                                   "deriv_dt_weight_exponent": deriv_dt_weight_exponent,
                                   "deriv_target_centered": deriv_target_centered,
                                   "val_aug_averaging": val_aug_averaging,
                                   "deriv_switch_epoch": deriv_switch_epoch,
                                   "use_centered_at_save": use_centered_this_epoch,
                                   "stats0_weight": stats0_weight, "stats1_weight": stats1_weight,
                                   "n_frozen_stages": n_frozen_stages, "resumed_from": str(resume_from)},
            }, checkpoint_path)
            msg += "  -> saved"
            # Stamp the save time (HH:MM) so a "-> saved" epoch line can be
            # matched to the checkpoint file it produced -- the checkpoint
            # filename carries the timestamp (e.g. ...-20260819_11h20.pt), and
            # without this there was no way to tell WHICH epoch a given .pt is.
            msg += f" at {time.strftime('%H:%M')}"
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
        elif not was_in_grace_period:
            epochs_since_improvement += 1
        # During a grace window should_save is UNCONDITIONALLY False (see
        # CheckpointCriterionTracker.reset_with_grace) -- the criterion is
        # deliberately declining to answer while the EMA re-primes, not
        # reporting a plateau. Counting those epochs as non-improvement made
        # early stopping fire whenever grace_epochs >= early_stopping_patience,
        # which the deriv_target_centered switch reaches routinely: with
        # grace=5 and patience=4 the run stopped at epoch 18, one epoch before
        # the criterion became usable again, while its EMA was still falling
        # monotonically (50.36 -> 50.01 -> 47.86 -> 47.51). The grace window
        # could never complete.

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
    if not Path(checkpoint_path).exists() and resume_from is not None:
        # `resumed_from_stage2` is required, not just a reference: stage 2's
        # ancestor may be a SINGLE-STREAM stage-1 checkpoint (keys "encoder.*")
        # while its own output must be multi-stream ("encoders.shared.*").
        # Copying one verbatim produces a file that loads nowhere -- caught by
        # a test failing with KeyError on
        # 'encoders.shared.down_blocks.0.conv.block.0.weight'.
        #
        # When the ancestor IS a stage-2 checkpoint the formats match and
        # keeping it is exactly right.
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
        print(f"\nstage 2: no epoch improved on the ancestor "
              f"(reference val_loss {tracker.reference_val_loss if tracker.reference_val_loss is not None else float('nan'):.6f}), so the ancestor "
              f"remains the best model and has been kept as {checkpoint_path}. This is a "
              f"real outcome, not an error -- the run simply found nothing better.")
    elif not Path(checkpoint_path).exists():
        # Same failure train_lds already guards: the run finished without ever
        # saving, and the drift report below then dies on a missing file with a
        # bare FileNotFoundError that says nothing about why.
        #
        # It happens whenever no epoch ever beat the criterion -- most easily
        # when a mid-run grace period (deriv_target_centered's switch) is
        # followed by a val_loss that never falls below the EMA the grace
        # window left behind. On a small dataset that is a plausible outcome,
        # not a crash.
        # Built as an ordinary variable rather than a multi-line conditional
        # inside an f-string replacement field. That construct is legal from
        # 3.12 (PEP 701) but implicit concatenation across lines INSIDE {...}
        # is exactly the corner that tooling handles inconsistently -- it
        # raised "unterminated string literal" on a user's 3.13 run while
        # parsing fine here. Nothing is gained by inlining it.
        _epochs_zero_hint = (
            "An epochs=0 ablation cannot produce a checkpoint -- remove the stage from "
            "the params file rather than setting its epochs to 0, if anything downstream "
            "needs its output. " if epochs == 0 else ""
        )
        raise RuntimeError(
            f"stage 2 finished without ever saving a checkpoint to {checkpoint_path}. "
            f"{_epochs_zero_hint}"
            f"No epoch's val criterion beat the running best, so nothing was written. "
            f"If deriv_target_centered switched mid-run, the grace period leaves "
            f"best_val_loss at the EMA reached during it -- a val_loss that then only "
            f"rises never clears that bar. Check the loss curve at {loss_curve_path}."
        )
    final_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    final_state = final_checkpoint["model_state"]
    final_params = {k: final_state[k] for k in initial_params}
    final_buffers = {k: final_state[k] for k in initial_buffers}
    param_drift, buffer_drift = compute_weight_drift(
        initial_params, initial_buffers, final_params, final_buffers)

    # Unconditional final write: the in-loop calls are throttled (see
    # should_write_loss_figure), and a run can end on an epoch that was
    # skipped -- via early stopping, or simply because the last epoch
    # wasn't a multiple of the interval. Without this the figures left on
    # disk could be up to `every` epochs stale, which is exactly the
    # state a finished run gets judged from.
    loss_curve(
        epoch_history, train_loss_history, val_loss_history, best_so_far_history,
        loss_curve_path, title="Stage 2 loss", event_epochs=loss_curve_events,
        reference_levels=loss_curve_levels,
    )
    write_loss_history(loss_curve_path, epoch_history, train_loss_history, val_loss_history, best_so_far_history)
    if not stage2a:      # see the periodic call above: moot when only deriv moves
        loss_component_scatter(
            epoch_history, component_histories, loss_components_path,
            title="Stage 2 loss components",
            ref_components=ref_components_for_scatter,
        )

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


