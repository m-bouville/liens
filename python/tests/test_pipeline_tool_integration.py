"""
Structural guards that cross-module integrations stay wired in the heavy pipeline
tools -- modules that import torch and the full training stack, so they can't be
imported or run in a light test env. Each test reads the module source and asserts
a specific integration point is present, so it can't silently regress:

  - compare_f_theta: the eval-log integration (upsert calls, rollout_* metric
    names, n_steps encoded in the eval_variant, model-independent baselines keyed
    with epoch=None, corr stored as a fraction).
  - check_rollout: encode_both_streams passed when the deriv source is
    previous_quotient (which reads both latent streams).
  - pipeline: a non-fatal check_rollout None-return is not unpacked as a tuple;
    stage-4 input checkpoints are archived at consume time; paths come from
    utils.paths.
  - train_refinement: the load note pins each source checkpoint by epoch/val_loss
    (generic filenames are overwritten, so identity must be in the log).

These guard the WIRING, not runtime behavior (which needs a GPU + dataset). They
are loose on whitespace/spelling but tight on the tokens that encode each fix.
Per-module placement (each test beside its module's other tests) would be the
ideal home; they are collected here because they share the same structural-guard
technique and none could be import-tested.
"""
import re
from pathlib import Path

import pytest

# Resolve the module files relative to this test's location: tests/ next to the
# package dirs, or a flat dir. Fall back to a search so it works in both layouts.
_HERE = Path(__file__).resolve().parent


def _find(name: str) -> str:
    for cand in (_HERE / name, _HERE.parent / "evaluation" / name,
                 _HERE.parent / "orchestration" / name,
                 _HERE.parent / "training" / name, _HERE.parent / name):
        if cand.exists():
            return cand.read_text()
    # last resort: recursive search under the repo root
    for cand in _HERE.parent.rglob(name):
        return cand.read_text()
    raise FileNotFoundError(name)


# --------------------------------------------------------------------------- #
# compare_f_theta: eval-log wiring
# --------------------------------------------------------------------------- #
def test_compare_f_theta_imports_eval_log_helpers():
    src = _find("compare_f_theta.py")
    assert "from utils.eval_log import" in src
    assert "upsert_eval_row" in src and "params_from_checkpoint" in src


def test_compare_f_theta_uses_rollout_prefixed_metric_names():
    src = _find("compare_f_theta.py")
    for name in ("rollout_median_loss", "rollout_median_corr_dx",
                 "rollout_n_steps", "rollout_n_samples"):
        assert name in src, f"{name} missing from compare_f_theta eval-log wiring"
    # the OLD bare names must be gone as upsert metric keys
    assert '"median_loss"' not in src and '"median_corr_dx"' not in src
    assert '"n_steps"' not in src and '"n_samples"' not in src


def test_compare_f_theta_encodes_n_steps_in_variant():
    src = _find("compare_f_theta.py")
    # every eval_variant that a horizon-dependent row uses carries n_steps
    assert re.search(r'eval_variant=f"rollout\{n_steps\}"', src)
    assert re.search(r'eval_variant=f"previous_derivative_rollout\{n_steps\}"', src)
    assert re.search(r'eval_variant=f"stage2_z0z1dt_rollout\{n_steps\}"', src)


def test_compare_f_theta_baselines_keyed_with_none_epoch():
    """The two baselines are model-independent, so they upsert with epoch=None
    (the same-ckpt-different-results bug). Their upsert calls pass None as epoch."""
    src = _find("compare_f_theta.py")
    # previous_derivative baseline: keyed on a baseline:... path with None epoch
    assert re.search(r'f"baseline:previous_derivative:\{[^}]+\}",\s*None', src)
    # stage2 baseline: keyed on the stage-2 path with None epoch
    assert re.search(r'eval_csv_for_checkpoint\(_s2_path\),\s*_s2_path,\s*None', src)


def test_compare_f_theta_stores_corr_as_fraction():
    src = _find("compare_f_theta.py")
    # corr medians divided by 100 before storing (percent -> fraction)
    assert "_med_corr / 100.0" in src
    assert "_med_cc / 100.0" in src and "_med_sc / 100.0" in src


# --------------------------------------------------------------------------- #
# check_rollout: encode_both_streams for previous_quotient
# --------------------------------------------------------------------------- #
def test_check_rollout_passes_encode_both_streams_for_previous_quotient():
    """encode_both_streams must be GATED on derivative_source == "previous_quotient"
    (both streams needed then), not set unconditionally or to False. The assertion
    pins the actual gating expression -- an earlier version had an OR fallback that
    passed on any file merely mentioning previous_quotient, so it would have passed
    even if the flag were wired wrong."""
    src = _find("check_rollout.py")
    # the exact gating: encode_both_streams=(getattr(f_theta,"derivative_source",...) == "previous_quotient")
    assert re.search(
        r'encode_both_streams\s*=\s*\(?\s*getattr\(\s*f_theta\s*,\s*"derivative_source".*?'
        r'==\s*"previous_quotient"',
        src, re.S), "encode_both_streams is not gated on derivative_source == previous_quotient"
    # and it must NOT be hard-wired False
    assert "encode_both_streams=False" not in src


# --------------------------------------------------------------------------- #
# pipeline: None-guard, stage-4 input archiving, utils.paths
# --------------------------------------------------------------------------- #
def test_pipeline_does_not_unpack_check_rollout_result_blindly():
    src = _find("pipeline.py")
    # the fatal pattern was: `_, shared_windows = check_rollout(...)`
    assert not re.search(r"_,\s*shared_windows\s*=\s*check_rollout\(", src), \
        "pipeline unpacks check_rollout() directly -- crashes when it returns None"
    # the guard: result checked for None before indexing
    assert "_rollout_result" in src and "is not None" in src


def test_pipeline_pins_stage4_ancestors_to_timestamped_names():
    """Stage 4/5 must pin BOTH its ancestors -- the stage-2 encoder and the
    stage-3b f_theta -- to timestamped names via _archive_ancestor (which archives
    the file AND returns the timestamped path to record), so --with-ancestors
    resolves 3b to a timestamped identity instead of the overwritten generic name.
    This supersedes the earlier _backup_before_overwrite calls (which archived the
    file but left the recorded pointer generic)."""
    src = _find("pipeline.py")
    assert "_ae_ancestor = _archive_ancestor(stage2_checkpoint)" in src
    assert "_f_theta_ancestor = _archive_ancestor(stage3_checkpoint)" in src
    # the recorded signature and the trainer call must use the PINNED paths
    assert 'str(_ae_ancestor)' in src and 'str(_f_theta_ancestor)' in src
    assert "lds_checkpoint_path=_f_theta_ancestor" in src


def test_pipeline_does_not_archive_stage3_ancestor_in_lds_stage():
    """Regression for the misplaced-edit NameError: the _archive_ancestor calls for
    stage2/stage3 belong in run_refinement_stage, NOT run_lds_stage (where
    stage3_checkpoint does not yet exist). Guard that run_lds_stage never references
    the refinement-only ancestor locals."""
    src = _find("pipeline.py")
    # crude but effective: the archive-of-stage3_checkpoint must appear exactly once
    # (in run_refinement_stage), never duplicated into run_lds_stage.
    assert src.count("_archive_ancestor(stage3_checkpoint)") == 1


def test_pipeline_imports_paths_from_utils_not_orchestration():
    src = _find("pipeline.py")
    assert "from utils.paths import" in src
    assert "from orchestration.paths import" not in src


# --------------------------------------------------------------------------- #
# train_refinement: provenance note pins epoch/val_loss
# --------------------------------------------------------------------------- #
def test_train_refinement_pins_sources_by_epoch_val_loss():
    src = _find("train_refinement.py")
    # the _pin helper builds "(epoch N, val_loss=...)" from provenance
    assert "def _pin(" in src
    assert "val_loss=" in src and 'prov.get("epoch")' in src
    assert "_pin(ae_checkpoint_path" in src and "_pin(lds_checkpoint_path" in src


# --------------------------------------------------------------------------- #
# trainers save RAW components (before weight/scale) for the eval ledger
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", ["train_lds.py", "train_stage1.py",
                                    "train_stage2.py", "train_refinement.py"])
def test_trainer_saves_raw_val_components(module):
    """Every trainer records val_components_raw (un-weighted, un-scaled) in the
    checkpoint, alongside the weighted/scaled contributions -- so the ledger's
    component columns are comparable across runs with different weight/scale."""
    src = _find(module)
    assert "val_components_raw" in src, f"{module} does not save val_components_raw"


def test_save_checkpoint_accepts_and_stores_raw_components():
    src = _find("_checkpoint_criterion.py")
    assert "val_components_raw=None" in src
    assert '"val_components_raw"' in src


def test_check_latent_channels_prefers_raw_components():
    src = _find("check_latent_channels.py")
    assert 'checkpoint.get("val_components_raw")' in src
    assert "val_components_kind" in src


# --------------------------------------------------------------------------- #
# grace/warmup epochs must not count against early-stopping patience
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", ["train_refinement.py", "train_lds.py",
                                    "train_stage2.py", "train_stage1.py"])
def test_trainer_excludes_grace_epochs_from_patience(module):
    """Every trainer whose CheckpointCriterionTracker enters a grace period --
    mid-run via reset_with_grace() (train_stage2/train_refinement/train_lds), or
    up front via ema_warmup_epochs at construction (train_stage1) -- must (a)
    capture in_grace_period BEFORE tracker.update() (the flag flips inside
    update() on the last grace epoch) and (b) skip the patience increment for
    grace epochs. A bare `else: epochs_since_improvement += 1` counts forced
    non-saves as stagnation and early-stops unconditionally when grace >=
    patience-1.

    train_stage1 originally gated only the STOP CHECK (`epoch > _grace`) while
    still incrementing the counter unconditionally through the whole warmup --
    which is not equivalent: the counter reaches patience by the time warmup
    ends regardless, so the first post-warmup epoch stops immediately no matter
    its own result. Observed on a real 256x256 run with
    ema_warmup_epochs=patience=10: early stopping fired at epoch 11, the very
    first epoch checked, while the EMA had been falling monotonically through
    the entire warmup. Fixed to the same counter-level exclusion as the other
    three trainers.
    """
    src = _find(module)
    assert "was_in_grace_period = tracker.in_grace_period" in src, \
        f"{module}: grace flag not captured before update()"
    assert "elif not was_in_grace_period:" in src, \
        f"{module}: patience counter does not exclude grace epochs"
    assert not re.search(r"else:\s*\n\s*epochs_since_improvement \+= 1", src), \
        f"{module}: still has a bare else-increment of the patience counter"


def test_compare_f_theta_drops_orphaned_min_passing_steps():
    """An old checkpoint can record min_passing_steps but predate the stdev
    fields, leaving min_passing_steps set with no threshold to count against --
    which build_good_steps rejects. compare_f_theta's reconcile must detect that
    invalid combination and drop the (no-op) min_passing_steps rather than fail
    with an opaque dataset error."""
    src = _find("compare_f_theta.py")
    assert 'dc.get("min_passing_steps")' in src
    assert 'dc["min_passing_steps"] = None' in src
    # gated on there being no stdev threshold
    assert 'min_stdev_phi") is not None' in src and 'min_normalized_stdev_phi") is not None' in src


def test_check_latent_channels_resolves_ae_state_for_all_checkpoint_kinds():
    """check_latent_channels must read the AE state from 'model_state' (AE
    checkpoint), 'ae_state', OR 'model_states[ae_state]' (refinement checkpoint),
    and raise a CLEAR error (not a bare KeyError) when there is no AE state --
    so pointing it at a stage-4 checkpoint analyses the refined encoder instead
    of crashing on KeyError('model_state')."""
    src = _find("check_latent_channels.py")
    assert 'checkpoint.get("model_state")' in src
    assert 'checkpoint["model_states"].get("ae_state")' in src
    assert "has no autoencoder state to analyse" in src   # the clear error
    # no bare checkpoint["model_state"] indexing left (would KeyError on stage 4)
    assert 'checkpoint["model_state"]' not in src
