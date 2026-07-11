"""
Identifies which pipeline stage actually produced a given checkpoint,
from its STRUCTURE rather than its filename or location. Extracted
from main.py during its split into orchestration/.
"""
from pathlib import Path

import torch


_STAGE_LABELS = {
    1: "stage 1 (autoencoder)",
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
    if ("model_state" in checkpoint and isinstance(checkpoint.get("config"), dict)
            and "latent_channels" in checkpoint["config"]):
        return _STAGE_LABELS[1]
    return "unrecognized (doesn't match any known stage's checkpoint structure)"


def _validate_checkpoint_stage(path: Path, stage_num: int, device: str | None) -> None:
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
