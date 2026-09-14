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
    src = _find("check_rollout.py")
    assert "encode_both_streams=" in src, "check_rollout never sets encode_both_streams"
    # it must be gated on the previous_quotient deriv source (both streams needed)
    assert re.search(r'encode_both_streams=\(?\s*getattr\(f_theta,\s*"derivative_source"',
                     src) or ('encode_both_streams=' in src and 'previous_quotient' in src)


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


def test_pipeline_archives_stage4_inputs_before_consuming():
    src = _find("pipeline.py")
    # both inputs (stage-2 encoder + stage-3 f_theta) archived at consume time
    assert "_backup_before_overwrite(stage2_checkpoint)" in src
    assert "_backup_before_overwrite(stage3_checkpoint)" in src


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
# grace epochs must not count against early-stopping patience
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", ["train_refinement.py", "train_lds.py", "train_stage2.py"])
def test_trainer_excludes_grace_epochs_from_patience(module):
    """Every trainer that uses reset_with_grace must (a) capture in_grace_period
    BEFORE tracker.update() (the flag flips inside update() on the last grace
    epoch) and (b) skip the patience increment for grace epochs. A bare
    `else: epochs_since_improvement += 1` counts forced non-saves as stagnation
    and early-stops unconditionally when grace >= patience-1."""
    src = _find(module)
    assert "was_in_grace_period = tracker.in_grace_period" in src, \
        f"{module}: grace flag not captured before update()"
    assert "elif not was_in_grace_period:" in src, \
        f"{module}: patience counter does not exclude grace epochs"
    assert not re.search(r"else:\s*\n\s*epochs_since_improvement \+= 1", src), \
        f"{module}: still has a bare else-increment of the patience counter"
