"""
Tests for orchestration/checkpoint_identification.py's
identify_checkpoint_stage and _validate_checkpoint_stage.

The whole point of identify_checkpoint_stage is to name what a
checkpoint ACTUALLY is from its structure, so a mislabeled file (a
stage-1 .pt sitting where a stage-2 one was expected -- the real
incident that motivated it) is caught with a clear error instead of a
confusing shape-mismatch deep inside training. Its correctness is
therefore mostly about the ORDER of its checks: several stages'
structures are strict supersets of earlier ones (stage 4/5 also carries
an ae_checkpoint provenance field; stage 1b also has model_state +
config["latent_channels"]), so the checks must be tried most-specific
first. These tests pin exactly those ordering hazards, plus the
unrecognized fallback and _validate_checkpoint_stage's clear error.

ensure_lds_checkpoint is NOT covered here: it trains a real (epochs=0)
f_theta and needs a dataset/AE fixture, out of scope for a pure
structure-identification unit test.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_checkpoint_identification.py -v
"""
import pytest
import torch

from orchestration.checkpoint_identification import (
    identify_checkpoint_stage,
    _validate_checkpoint_stage,
    _STAGE_LABELS,
)


# --- identify_checkpoint_stage: the ordering hazards ------------------

def test_stage4_is_ae_state_plus_f_theta_state_with_freeze_decoder():
    """The joint format with freeze_decoder=True is stage 4 (encoder
    refinement, decoder frozen)."""
    checkpoint = {
        "ae_state": {}, "f_theta_state": {},
        "stage45_config": {"freeze_decoder": True},
    }
    assert identify_checkpoint_stage(checkpoint) == _STAGE_LABELS[4]


def test_stage5_is_the_same_joint_format_without_freeze_decoder():
    """The identical joint shape but freeze_decoder falsey is stage 5
    (end-to-end). Missing stage45_config entirely also falls here (the
    .get default is None, which is falsey)."""
    frozen_false = {
        "ae_state": {}, "f_theta_state": {},
        "stage45_config": {"freeze_decoder": False},
    }
    no_config = {"ae_state": {}, "f_theta_state": {}}
    assert identify_checkpoint_stage(frozen_false) == _STAGE_LABELS[5]
    assert identify_checkpoint_stage(no_config) == _STAGE_LABELS[5]


def test_joint_format_is_detected_before_stage3_even_with_ae_checkpoint_present():
    """The critical ordering hazard the code's own comment calls out: a
    stage 4/5 joint checkpoint ALSO carries an "ae_checkpoint"
    provenance field, which would satisfy the stage-3 check just below.
    The joint (ae_state + f_theta_state) check must win."""
    checkpoint = {
        "ae_state": {}, "f_theta_state": {},
        "ae_checkpoint": "checkpoints/stage2/ancestor.pt",   # would look like stage 3
        "stage45_config": {"freeze_decoder": True},
    }
    assert identify_checkpoint_stage(checkpoint) == _STAGE_LABELS[4]


def test_stage3_is_a_bare_ae_checkpoint_field():
    """A stage-3 LDS checkpoint records its AE ancestor in
    ae_checkpoint and has neither ae_state nor f_theta_state."""
    checkpoint = {"ae_checkpoint": "checkpoints/stage2/ae.pt", "model_state": {}}
    assert identify_checkpoint_stage(checkpoint) == _STAGE_LABELS[3]


def test_stage2_is_detected_from_either_config_field_name():
    """stage2_config is the current field; stage3_config is the OLD
    internal name from before the renumbering, still recognized so an
    old checkpoint doesn't need retraining to be identified."""
    assert identify_checkpoint_stage({"stage2_config": {}}) == _STAGE_LABELS[2]
    assert identify_checkpoint_stage({"stage3_config": {}}) == _STAGE_LABELS[2]


def test_stage1b_is_detected_before_plain_stage1():
    """The other ordering hazard: a stage 1b checkpoint ALSO has
    model_state + config["latent_channels"] (stage 1's own shape), so
    without the stage1b_config check FIRST it would be misidentified as
    plain stage 1. Include the stage-1-looking fields to prove the
    stage1b_config check genuinely wins over them."""
    checkpoint = {
        "stage1b_config": {},
        "model_state": {}, "config": {"latent_channels": 8},   # also looks like stage 1
    }
    assert identify_checkpoint_stage(checkpoint) == _STAGE_LABELS["1b"]


def test_stage1_is_model_state_plus_config_with_latent_channels():
    checkpoint = {"model_state": {}, "config": {"latent_channels": 4, "size": 64}}
    assert identify_checkpoint_stage(checkpoint) == _STAGE_LABELS[1]


def test_stage1_requires_latent_channels_in_a_dict_config():
    """model_state alone is not enough: config must be a dict AND carry
    latent_channels. A config that is present but not a dict, or a dict
    lacking latent_channels, is NOT stage 1 -- it falls through to
    unrecognized."""
    assert identify_checkpoint_stage(
        {"model_state": {}, "config": "not-a-dict"}).startswith("unrecognized")
    assert identify_checkpoint_stage(
        {"model_state": {}, "config": {"size": 64}}).startswith("unrecognized")


def test_completely_unknown_structure_is_reported_as_unrecognized():
    assert identify_checkpoint_stage({"something_else": 1}).startswith("unrecognized")
    assert identify_checkpoint_stage({}).startswith("unrecognized")


# --- _validate_checkpoint_stage: clear error on mismatch --------------

def _save(tmp_path, name, checkpoint):
    path = tmp_path / name
    torch.save(checkpoint, path)
    return path


def test_validate_passes_silently_when_the_stage_matches(tmp_path):
    """A correctly-placed checkpoint validates without raising."""
    path = _save(tmp_path, "stage1.pt",
                 {"model_state": {}, "config": {"latent_channels": 8}})
    # Returns None, raises nothing.
    assert _validate_checkpoint_stage(path, 1, device=None) is None


def test_validate_raises_naming_actual_and_expected_on_mismatch(tmp_path):
    """A stage-1 file where stage 2 was expected: the error must name
    BOTH what was expected and what the file actually is -- that
    specificity is the whole reason identify_checkpoint_stage tries
    every stage, not just the expected one's own check."""
    path = _save(tmp_path, "mislabeled.pt",
                 {"model_state": {}, "config": {"latent_channels": 8}})  # actually stage 1
    with pytest.raises(ValueError) as excinfo:
        _validate_checkpoint_stage(path, 2, device=None)              # expected stage 2
    message = str(excinfo.value)
    assert _STAGE_LABELS[2] in message, "error should name the EXPECTED stage"
    assert _STAGE_LABELS[1] in message, "error should name the ACTUAL stage"


def test_validate_accepts_string_stage_labels_like_1b(tmp_path):
    """stage_num may be a string key ('1b', '3a', ...): validation of a
    genuine stage-1b checkpoint against '1b' passes."""
    path = _save(tmp_path, "stage1b.pt",
                 {"stage1b_config": {}, "model_state": {}, "config": {"latent_channels": 8}})
    assert _validate_checkpoint_stage(path, "1b", device=None) is None


def test_stage_labels_alias_3a_and_3b_to_the_stage3_label():
    """'3a' and '3b' are aliases for the single stage-3 label, so a
    stage-3 checkpoint validates against any of 3, '3a', or '3b'."""
    assert _STAGE_LABELS["3a"] == _STAGE_LABELS[3]
    assert _STAGE_LABELS["3b"] == _STAGE_LABELS[3]
