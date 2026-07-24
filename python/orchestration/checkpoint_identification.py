"""
Identifies which pipeline stage actually produced a given checkpoint,
from its STRUCTURE rather than its filename or location. Extracted
from main.py during its split into orchestration/.
"""
import tempfile
from pathlib import Path

import torch


_STAGE_LABELS = {
    1: "stage 1 (autoencoder)",
    "1b": "stage 1b (deriv stream decoder)",
    2: "stage 2 (latent-space validation)",
    3: "stage 3 (latent dynamics surrogate)",
    4: "stage 4 (encoder refinement)",
    5: "stage 5 (end-to-end refinement)",
}
_STAGE_LABELS["3a"] = _STAGE_LABELS["3b"] = _STAGE_LABELS[3]


def identify_checkpoint_stage(checkpoint: dict) -> str:
    """
    Inspects a loaded checkpoint's STRUCTURE (not its filename) to
    determine which pipeline stage actually produced it -- so a
    mismatched checkpoint (e.g. a stage-1 file sitting where a stage-2
    file was expected, exactly what happened when console logs got lost
    and files got confused) is caught with a clear, specific error
    instead of failing deep inside training with a confusing
    shape-mismatch, or silently training on the wrong starting point.
    """
    if "ae_state" in checkpoint and "f_theta_state" in checkpoint:
        # The stage 4/5 joint format -- distinguished from stage 3 by
        # checking THIS first, since it also carries an "ae_checkpoint"
        # provenance field (recording its own ancestor) that would
        # otherwise satisfy the very next check below.
        freeze_decoder = checkpoint.get("stage45_config", {}).get("freeze_decoder")
        return _STAGE_LABELS[4] if freeze_decoder else _STAGE_LABELS[5]
    if "ae_checkpoint" in checkpoint:
        return _STAGE_LABELS[3]
    if "stage2_config" in checkpoint or "stage3_config" in checkpoint:
        # stage3_config: train_ae.py's OLD internal field name, from
        # before stages were renumbered -- still recognized here so a
        # checkpoint trained before that rename doesn't need retraining
        # just to be correctly identified.
        return _STAGE_LABELS[2]
    if "stage1b_config" in checkpoint:
        # MUST be checked before the plain stage-1 test just below --
        # a stage 1b checkpoint ALSO has model_state + a
        # config["latent_channels"] entry (same shape stage 1's own
        # save uses), so without this check first, every stage 1b
        # checkpoint would be silently misidentified as plain stage 1.
        return _STAGE_LABELS["1b"]
    if ("model_state" in checkpoint and isinstance(checkpoint.get("config"), dict)
            and "latent_channels" in checkpoint["config"]):
        return _STAGE_LABELS[1]
    return "unrecognized (doesn't match any known stage's checkpoint structure)"


def _validate_checkpoint_stage(path: Path, stage_num: int | str, device: str | None) -> None:
    """Raises a clear, specific error if `path` isn't actually a
    checkpoint from the expected stage. identify_checkpoint_stage()
    already tries every known stage's structure in turn regardless of
    which one was expected, so this always names what the file actually
    is, not just that it failed one specific check."""
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
    actual = identify_checkpoint_stage(checkpoint)
    expected = _STAGE_LABELS[stage_num]
    if actual != expected:
        raise ValueError(f"{path} is not a {expected} checkpoint: it is a {actual} checkpoint.")


def ensure_lds_checkpoint(
    checkpoint_path: Path, base_path: Path | None = None, size: int | None = None,
    device: str | None = None, **train_lds_kwargs,
) -> Path:
    """
    If checkpoint_path is ALREADY a stage-3 checkpoint, returns it
    unchanged -- a no-op, safe to call unconditionally from anything
    that wants to accept "either a stage-3 checkpoint, or something
    that can be turned into one" without its own branching. If it's a
    stage-1/1b/2 (AE-family) checkpoint instead, trains a FRESH,
    RANDOMLY-INITIALIZED f_theta against it for epochs=0 (see
    train_lds.py's own epoch-loop: epochs=0 still runs ONE validation-
    only pass and saves a checkpoint from it) -- an ephemeral, throwaway
    "stage 3a" wrapper that lets check_f_theta.py/
    check_parameter_dependence.py's own machinery run directly against
    an AE checkpoint, without a real stage 3a ever having been trained.

    IMPORTANT LIMITATION, not a bug: f_theta here is UNTRAINED (fresh
    random init, its own final layer zero-initialized -- see
    LatentDynamics' own docstring). Its own output is therefore ~0
    everywhere, meaning "full" (f_theta-corrected) is essentially
    IDENTICAL to "euler-only" throughout the resulting report -- any
    number that's actually ABOUT f_theta's own quality (f2_chained/
    f2_real, ratio_real, cos_sim_real, the 'D' coefficient in the
    Taylor-residual decomposition, any full-vs-euler-only comparison)
    is UNINFORMATIVE in this mode, not just imprecise. What DOES remain
    meaningful: everything that only depends on z0/z1 (state/deriv)
    themselves -- the euler-only error's own dt/temperature/noise
    dependence, z1's own bias/variance decomposition, the 'C' (~z0_ddot)
    coefficient -- exactly the properties an AE-family checkpoint
    (which has no f_theta at all) can actually be judged on.

    base_path/size: REQUIRED when checkpoint_path turns out to be
    AE-family (train_lds needs a real dataset to encode against) --
    not required, and unused, when checkpoint_path is already stage 3.
    Any other train_lds() keyword (min_step, min_stdev_phi,
    min_passing_steps, hidden_dim, n_hidden_layers, ae_stats_weight,
    condition_on_theta, ...) can be passed through via train_lds_kwargs;
    sensible defaults apply for anything not given, matching
    train_lds()'s own (hidden_dim=256, n_hidden_layers=2 here
    specifically, since those two have no meaningful "right" choice for
    an UNTRAINED network -- any values are equally arbitrary, so
    train_lds()'s own defaults are as good as any other pick).

    The returned checkpoint lives under a fresh tempfile.mkdtemp() --
    left for the OS's own normal temp-directory cleanup rather than
    deleted here, since the caller still needs to load it after this
    function returns.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=True)
    stage = identify_checkpoint_stage(checkpoint)
    if stage == _STAGE_LABELS[3]:
        return checkpoint_path
    if stage not in (_STAGE_LABELS[1], _STAGE_LABELS["1b"], _STAGE_LABELS[2]):
        raise ValueError(
            f"{checkpoint_path} is a {stage} checkpoint -- ensure_lds_checkpoint only knows how "
            f"to convert an AE-family checkpoint (stage 1/1b/2) into an ephemeral stage-3-shaped "
            f"one, or pass a real stage 3 checkpoint through directly."
        )
    # size, UNLIKE base_path, is already saved in every AE-family
    # checkpoint's own config (see train_autoencoder/train_stage1b/
    # train_stage2's own checkpoint-save calls) -- no reason to make
    # the caller repeat it by hand when it's sitting right there. Only
    # used as a FALLBACK (an explicitly-given size, even if it
    # happened to disagree, would be a strange thing for this function
    # to silently override) -- checked for a mismatch below instead.
    checkpoint_size = checkpoint.get("config", {}).get("size")
    if size is None:
        size = checkpoint_size
    elif checkpoint_size is not None and size != checkpoint_size:
        raise ValueError(
            f"size={size} was given, but {checkpoint_path}'s own config says size={checkpoint_size} "
            f"-- double check which checkpoint/size you meant (or omit size entirely to use the "
            f"checkpoint's own value)."
        )
    if base_path is None or size is None:
        missing = [n for n, v in [("base_path", base_path), ("size", size)] if v is None]
        raise ValueError(
            f"{checkpoint_path} is a {stage} checkpoint, not stage 3 -- converting it requires "
            f"{' and '.join(missing)} (train_lds() needs a real dataset to encode against), which "
            f"{'were' if len(missing) > 1 else 'was'} not given and could not be inferred from the "
            f"checkpoint itself. base_path is the SWEEP ROOT (e.g. '../datasets'), NOT including "
            f"the size-specific subdirectory -- train_lds() appends '{{size}}x{{size}}' itself; "
            f"including it in base_path too doubles it (e.g. '../datasets/64x64/64x64/metadata.txt')."
        )
    print(f"NOTE: {checkpoint_path} is a {stage} checkpoint, not stage 3 -- training a FRESH, "
          f"UNTRAINED f_theta against it (epochs=0) to produce an ephemeral stage-3-shaped "
          f"wrapper. Any number below that's actually about f_theta's own quality is "
          f"UNINFORMATIVE in this mode -- see ensure_lds_checkpoint's own docstring for exactly "
          f"which ones those are.")
    from training.train_lds import train_lds  # deferred: avoids a training/orchestration import
    # cycle at module load time (train_lds.py itself doesn't import
    # this module, but several orchestration-layer modules that DO
    # import this one also end up importing train_lds transitively).

    tmp_dir = Path(tempfile.mkdtemp(prefix="ensure_lds_checkpoint_"))
    tmp_checkpoint_path = tmp_dir / f"{checkpoint_path.stem}-ephemeral-stage3.pt"

    def _default_if_none(key, value):
        """train_lds_kwargs.setdefault(key, value) is the WRONG tool
        here and was the actual bug in an earlier version of this
        function: setdefault only fills in a key that's ABSENT, but
        every caller in this project (check_f_theta.py,
        check_parameter_dependence.py) always passes these kwargs
        explicitly, with value=None when the user didn't set them --
        so the key is already present (bound to None), and setdefault
        silently does nothing. This checks the VALUE, not just
        presence, matching what min_step already (correctly) did below
        even before this helper existed."""
        if train_lds_kwargs.get(key) is None:
            train_lds_kwargs[key] = value

    _default_if_none("hidden_dim", 256)
    _default_if_none("n_hidden_layers", 2)
    # ae_stats_weight: train_lds() requires SOME value (its own
    # unconditional validation, regardless of whether it would ever
    # actually be read), but its only two real uses are BOTH gated on
    # ae_checkpoint_path/checkpoint_path being None (reconstructing a
    # path by naming convention) -- both of which are ALWAYS given
    # explicitly here, so this value is provably inert in this specific
    # code path (verified directly, not assumed). Defaulted silently --
    # unlike min_step below, there's no real choice being hidden here.
    _default_if_none("ae_stats_weight", 1.0)
    if train_lds_kwargs.get("min_step") is None:
        # UNLIKE ae_stats_weight, min_step has a real, functional effect
        # (which snapshots are even in the pool this ephemeral f_theta
        # gets encoded against) -- defaulted to 0 (no filtering) only as
        # a last resort, WITH a visible warning, not silently like
        # ae_stats_weight above, since 0 may not match whatever the
        # checkpoint's own real training actually used.
        train_lds_kwargs["min_step"] = 0
        print("WARNING: min_step not given -- defaulting to 0 (no early-transient filtering). "
              "If the real training run this checkpoint came from used a nonzero min_step, pass "
              "the SAME value here for a consistent comparison, rather than relying on this "
              "default.")
    # loss_curve_path: train_lds() writes this UNCONDITIONALLY, once per
    # epoch, regardless of epochs=0 or log_every_epoch=False (see its
    # own epoch-loop call to loss_curve(), which isn't gated on either)
    # -- defaulting to its own real output/stage3/ path is exactly
    # right for a genuine training run, but here it would leave a real
    # PNG behind in the user's actual output tree for an ephemeral,
    # throwaway conversion never meant to persist anywhere. Keeping it
    # inside the SAME tmp_dir as tmp_checkpoint_path means it gets
    # cleaned up the same way (OS's own temp-directory cleanup),
    # instead of needing its own separate cleanup path.
    _default_if_none("loss_curve_path", tmp_dir / "loss_curve.png")
    train_lds(
        size=size, base_path=base_path, ae_checkpoint_path=checkpoint_path,
        epochs=0, checkpoint_path=tmp_checkpoint_path, device=device,
        log_every_epoch=False,
        **train_lds_kwargs,
    )
    return tmp_checkpoint_path
