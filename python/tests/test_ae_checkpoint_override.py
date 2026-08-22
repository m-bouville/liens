"""
Tests for the Stage-3 ae_checkpoint_path ancestor override (lets a stage-3
section name a specific stage-2 encoder to build on, e.g. a hand-picked knee
checkpoint, without overwriting the canonical stage-2 file) and its exclusion
from global-section leakage.

Mirrors the resume_from override the same resolver already implements; these
pin the new key so the two can't silently diverge.
"""
from pathlib import Path

from orchestration.stage_params import (_resolve_stage_specific_ancestor,
                          _NEVER_GLOBAL_DEFAULT_KEYS)


def test_explicit_ae_checkpoint_path_overrides_the_default():
    default = Path("/canonical/128x128-stage2.pt")
    kwargs = {"ae_checkpoint_path": "/picked/13h05.pt", "lr": 5e-4}
    new_kwargs, resolved, overridden = _resolve_stage_specific_ancestor(
        kwargs, default, "Stage 3a", key="ae_checkpoint_path")
    assert overridden is True
    assert resolved == Path("/picked/13h05.pt")
    # the key is popped so it is NOT passed again via **kwargs to train_lds
    assert "ae_checkpoint_path" not in new_kwargs
    assert new_kwargs == {"lr": 5e-4}


def test_absent_ae_checkpoint_path_keeps_the_pipeline_default():
    default = Path("/canonical/128x128-stage2.pt")
    kwargs = {"lr": 5e-4}
    new_kwargs, resolved, overridden = _resolve_stage_specific_ancestor(
        kwargs, default, "Stage 3a", key="ae_checkpoint_path")
    assert overridden is False
    assert resolved is default
    assert new_kwargs == {"lr": 5e-4}


def test_ae_checkpoint_path_cannot_leak_from_the_global_section():
    # membership here is what _prepare_stage_kwargs consults to refuse using a
    # GLOBAL value as a per-stage default -- so an ancestor is only ever an
    # explicit, per-stage choice, never an accidental global one.
    assert "ae_checkpoint_path" in _NEVER_GLOBAL_DEFAULT_KEYS


def test_resume_from_override_still_works_unchanged():
    # the same resolver, its original key -- guards against the ae addition
    # perturbing resume_from.
    default = Path("/pipeline/default.pt")
    kwargs = {"resume_from": "/prior/3a.pt"}
    new_kwargs, resolved, overridden = _resolve_stage_specific_ancestor(
        kwargs, default, "Stage 3b")
    assert overridden is True and resolved == Path("/prior/3a.pt")
    assert "resume_from" not in new_kwargs
