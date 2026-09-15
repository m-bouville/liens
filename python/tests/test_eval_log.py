"""
Tests for utils.eval_log -- the eval-stageN.csv schema and upsert.

Targets the bug classes that recurred while this was built:
  - upsert keying on (checkpoint_path, epoch, eval_variant): two tools merge into
    one row; a re-run updates in place (no duplicate); different variants/epochs
    get separate rows.
  - params_from_checkpoint: dumps the FULL config as p_*, is sparse (missing keys
    omitted), skips structured sub-configs, and surfaces source_stage2/resume_from.
  - schema: EVAL_COLUMNS composed from the groups; reconcile adds missing canonical
    columns and surfaces (keeps, not drops) unexpected ones.
"""
import csv
import io
import contextlib
from pathlib import Path

import pytest

from utils.eval_log import (
    upsert_eval_row, upsert_eval_metrics, params_from_checkpoint, reconcile_fieldnames, canonical_checkpoint_key,
    is_column_in_scope, columns_in_scope, _EVAL_METRIC_COLUMNS, OUTPUT_COLUMNS,
    strip_to_checkpoints,
    EVAL_COLUMNS, PARAM_COLUMNS, LOSS_COLUMNS, OUTPUT_COLUMNS, SOURCE_COLUMNS,
)


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# schema composition
# --------------------------------------------------------------------------- #
def test_eval_columns_is_composed_from_groups_no_duplicates():
    # EVAL_COLUMNS = key + source + params + losses + outputs, each contributing
    assert EVAL_COLUMNS[0] == "checkpoint_path"
    for group in (SOURCE_COLUMNS, PARAM_COLUMNS, LOSS_COLUMNS, OUTPUT_COLUMNS):
        for c in group:
            assert c in EVAL_COLUMNS, f"{c} missing from composed EVAL_COLUMNS"
    assert len(EVAL_COLUMNS) == len(set(EVAL_COLUMNS)), "duplicate columns in EVAL_COLUMNS"


def test_val_loss_is_a_loss_not_a_param():
    assert "val_loss" in LOSS_COLUMNS
    assert "val_loss" not in PARAM_COLUMNS


# --------------------------------------------------------------------------- #
# upsert keying
# --------------------------------------------------------------------------- #
def test_two_tools_same_key_merge_into_one_row(tmp_path):
    f = tmp_path / "eval-stage3a.csv"
    ck = "checkpoints/stage3a/x.pt"
    upsert_eval_row(f, ck, 1046, {"rollout_median_loss": 0.048})
    upsert_eval_row(f, ck, 1046, {"ch0_imp": 0.6})
    rows = _read(f)
    assert len(rows) == 1
    assert rows[0]["rollout_median_loss"] == "0.048" and rows[0]["ch0_imp"] == "0.6"


def test_rerun_updates_in_place_not_duplicate(tmp_path):
    f = tmp_path / "eval-stage3a.csv"
    ck = "checkpoints/stage3a/x.pt"
    upsert_eval_row(f, ck, 1046, {"rollout_median_corr_dx": 0.51})
    upsert_eval_row(f, ck, 1046, {"rollout_median_corr_dx": 0.53})   # re-run
    rows = _read(f)
    assert len(rows) == 1 and rows[0]["rollout_median_corr_dx"] == "0.53"


def test_distinct_variants_are_distinct_rows(tmp_path):
    f = tmp_path / "eval-stage3a.csv"
    ck = "checkpoints/stage3a/x.pt"
    upsert_eval_row(f, ck, 1046, {"rollout_median_corr_dx": 0.51}, eval_variant="")
    upsert_eval_row(f, ck, 1046, {"rollout_median_corr_dx": 0.63}, eval_variant="previous")
    rows = _read(f)
    assert len(rows) == 2


def test_distinct_epochs_are_distinct_rows(tmp_path):
    f = tmp_path / "eval-stage3a.csv"
    ck = "checkpoints/stage3a/x.pt"
    upsert_eval_row(f, ck, 476, {"rollout_median_corr_dx": 0.51})
    upsert_eval_row(f, ck, 1118, {"rollout_median_corr_dx": 0.52})
    assert len(_read(f)) == 2


# --------------------------------------------------------------------------- #
# params_from_checkpoint
# --------------------------------------------------------------------------- #
def test_params_dumps_full_config_as_p_prefixed():
    ck = {"config": {"lr": 0.002, "n_rollout_steps": 2, "dynamics_mode": "deriv_linear",
                     "stats0_predict_weight": 0.2}}
    p = params_from_checkpoint(ck)
    assert p["p_lr"] == 0.002 and p["p_n_rollout_steps"] == 2
    assert p["p_dynamics_mode"] == "deriv_linear"
    assert p["p_stats0_predict_weight"] == 0.2   # loss-WEIGHT is a param (input)


def test_params_is_sparse_missing_keys_omitted():
    p = params_from_checkpoint({"config": {"lr": 0.001}})
    assert p == {"p_lr": 0.001}                   # nothing fabricated for absent keys


def test_params_skips_structured_subconfigs():
    ck = {"config": {"latent_channels": 4, "stream_configs": {"state": {"channels": 4}}}}
    p = params_from_checkpoint(ck)
    assert p["p_latent_channels"] == 4
    assert not any("stream_configs" in k for k in p)


def test_params_surfaces_source_columns_when_present():
    ck = {"config": {"stage2_checkpoint": "a/s2.pt", "resumed_from": "a/prev.pt"}}
    p = params_from_checkpoint(ck)
    assert p["source_stage2"] == "a/s2.pt" and p["resume_from"] == "a/prev.pt"


# --------------------------------------------------------------------------- #
# reconcile_fieldnames
# --------------------------------------------------------------------------- #
def test_reconcile_adds_missing_canonical_columns():
    """stage4 is used because it is in-scope for every non-dynamics-stage-scoped
    canonical column EXCEPT source_stage2/source_stage3-for-non-refinement-stages
    trade-offs -- stage3a is used for stage3-scoped ones and stage4 covers the
    rest, so between the two tests every EVAL_COLUMNS entry is checked somewhere."""
    out = reconcile_fieldnames(["checkpoint_path", "epoch"], "eval-stage4.csv")
    for c in EVAL_COLUMNS:
        if c in ("p_dynamics_mode", "p_derivative_source", "p_derivative_time"):
            continue    # dynamics-only, correctly absent from stage4
        assert c in out, f"{c} missing from stage4 reconcile"


def test_reconcile_adds_dynamics_only_columns_for_dynamics_stages():
    out = reconcile_fieldnames(["checkpoint_path", "epoch"], "eval-stage3a.csv")
    for c in ("p_dynamics_mode", "p_derivative_source", "p_derivative_time"):
        assert c in out


def test_reconcile_keeps_unexpected_columns_and_reports_them():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = reconcile_fieldnames(["checkpoint_path", "epoch", "deriv_variant"],
                                   "eval-stage3a.csv")
    assert "deriv_variant" in out                 # kept, not dropped
    assert "deriv_variant" in buf.getvalue()      # surfaced


def test_reconcile_accepts_variable_columns_silently():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reconcile_fieldnames(["checkpoint_path", "p_lr", "ch3_imp", "val_rollout"],
                             "eval-stage3a.csv")
    out = buf.getvalue()
    for c in ("p_lr", "ch3_imp", "val_rollout"):
        assert c not in out                       # pattern-known -> not flagged


# --------------------------------------------------------------------------- #
# stage-specific schema + column ordering
# --------------------------------------------------------------------------- #
def test_source_stage2_not_added_to_stage2():
    """source_stage2 is a stage-3+ concept. reconcile must not ADD it to a stage-2
    file that lacks it (append-only never introduces an out-of-scope column). An
    EXISTING out-of-scope column is kept, not removed -- that is what
    --drop-out-of-scope-columns is for."""
    added = reconcile_fieldnames(["checkpoint_path", "epoch"], "eval-stage2.csv")
    assert "source_stage2" not in added
    kept = reconcile_fieldnames(["checkpoint_path", "epoch", "source_stage2"], "eval-stage2.csv")
    assert "source_stage2" in kept          # existing column kept (append-only, no removal)
    assert "resume_from" in added           # resume_from applies to any stage -> added


def test_source_stage2_present_for_stage3():
    out = reconcile_fieldnames(["checkpoint_path", "epoch"], "eval-stage3a.csv")
    assert "source_stage2" in out


def test_reconcile_is_append_only_existing_order_preserved():
    """reconcile must NOT reorder an existing file -- it keeps existing columns
    exactly where they are and appends only the in-scope canonical columns the
    file is missing."""
    cols = ["checkpoint_path", "ch0_imp", "val_loss", "p_lr", "epoch"]   # deliberately odd order
    out = reconcile_fieldnames(cols, "eval-stage2.csv")
    assert out[:len(cols)] == cols          # existing columns unmoved, in place
    for extra in out[len(cols):]:           # anything appended is a missing canonical column
        assert extra not in cols


def test_reconcile_new_file_gets_canonical_order():
    out = reconcile_fieldnames([], "eval-stage3a.csv")
    assert out[:3] == ["checkpoint_path", "epoch", "eval_variant"]


def test_stage1_does_not_gain_source_stage2():
    out = reconcile_fieldnames(["checkpoint_path", "epoch"], "eval-stage1.csv")
    assert "source_stage2" not in out       # not added; scoping prevents introducing it


# --------------------------------------------------------------------------- #
# audit findings: key normalisation + horizon-distinct rows
# --------------------------------------------------------------------------- #
def test_checkpoint_key_normalises_path_spellings(tmp_path):
    """The same checkpoint spelled with '/', '\\', or as an absolute path must be ONE
    row -- --all seeds forward-slash, a Windows CLI passes backslash/absolute; keying
    on the raw string gave one checkpoint several rows (the duplicate-row bug)."""
    f = tmp_path / "eval-stage3a.csv"
    for spelling in ("checkpoints/stage3a/X.pt", r"checkpoints\stage3a\X.pt",
                     r"D:\work\NN\phase_field\python\checkpoints\stage3a\X.pt"):
        upsert_eval_row(f, spelling, 5, {"val_loss": 1.0})
    rows = _read(f)
    assert len(rows) == 1
    assert rows[0]["checkpoint_path"] == "checkpoints/stage3a/X.pt"


def test_baseline_keys_are_left_untouched():
    assert canonical_checkpoint_key("baseline:previous_derivative:ae") == "baseline:previous_derivative:ae"


def test_different_rollout_horizons_do_not_overwrite(tmp_path):
    """compare_f_theta encodes n_steps in eval_variant ('rollout2' vs 'rollout6'):
    a 2-step and a 6-step evaluation of the same checkpoint are different
    measurements and must be separate rows, not the second clobbering the first."""
    f = tmp_path / "eval-stage3a.csv"
    ck = "checkpoints/stage3a/X.pt"
    upsert_eval_row(f, ck, 1118, {"rollout_median_corr_dx": 0.60, "rollout_n_steps": 2},
                    eval_variant="rollout2")
    upsert_eval_row(f, ck, 1118, {"rollout_median_corr_dx": 0.51, "rollout_n_steps": 6},
                    eval_variant="rollout6")
    rows = _read(f)
    assert len(rows) == 2
    by = {r["eval_variant"]: r["rollout_median_corr_dx"] for r in rows}
    assert by["rollout2"] == "0.6" and by["rollout6"] == "0.51"


# --------------------------------------------------------------------------- #
# upsert_eval_metrics: fill-or-branch (compare_f_theta's handful of cells)
# --------------------------------------------------------------------------- #
def test_eval_metrics_fill_empty_cells_in_place_no_twin(tmp_path):
    """A params/components row exists with empty eval-metric cells; writing metrics
    fills them in place (sets eval_variant), no redundant twin row."""
    f = tmp_path / "eval-stage4.csv"
    ck = "checkpoints/stage4/x.pt"
    upsert_eval_row(f, ck, 35, {"p_lr": "2e-7", "val_rollout": "0.25"})   # params row, "" variant
    upsert_eval_metrics(f, ck, 35, {"rollout_median_corr_dx": 0.699, "rollout_n_steps": 6},
                        eval_variant="rollout6")
    rows = _read(f)
    assert len(rows) == 1
    assert rows[0]["eval_variant"] == "rollout6"
    assert rows[0]["rollout_median_corr_dx"] == "0.699"
    assert rows[0]["p_lr"] == "2e-7" and rows[0]["val_rollout"] == "0.25"   # preserved


def test_eval_metrics_same_variant_is_noop(tmp_path):
    f = tmp_path / "eval-stage4.csv"; ck = "checkpoints/stage4/x.pt"
    upsert_eval_metrics(f, ck, 35, {"rollout_median_corr_dx": 0.699, "rollout_n_steps": 6},
                        eval_variant="rollout6")
    upsert_eval_metrics(f, ck, 35, {"rollout_median_corr_dx": 0.699, "rollout_n_steps": 6},
                        eval_variant="rollout6")   # re-run
    assert len(_read(f)) == 1


def test_eval_metrics_different_variant_adds_row(tmp_path):
    f = tmp_path / "eval-stage4.csv"; ck = "checkpoints/stage4/x.pt"
    upsert_eval_metrics(f, ck, 35, {"rollout_median_corr_dx": 0.699, "rollout_n_steps": 6},
                        eval_variant="rollout6")
    upsert_eval_metrics(f, ck, 35, {"rollout_median_corr_dx": 0.60, "rollout_n_steps": 2},
                        eval_variant="rollout2")   # different horizon
    rows = _read(f)
    assert len(rows) == 2
    assert {r["eval_variant"] for r in rows} == {"rollout6", "rollout2"}


def test_eval_metrics_creates_row_when_none_exists(tmp_path):
    f = tmp_path / "eval-stage4.csv"; ck = "checkpoints/stage4/x.pt"
    upsert_eval_metrics(f, ck, 35, {"rollout_median_corr_dx": 0.7}, eval_variant="rollout6")
    rows = _read(f)
    assert len(rows) == 1 and rows[0]["eval_variant"] == "rollout6"


# --------------------------------------------------------------------------- #
# single scope predicate + derived metric list (anti-drift)
# --------------------------------------------------------------------------- #
def test_is_column_in_scope_matches_all_three_scoped_categories():
    assert is_column_in_scope("source_stage2", "stage4") and not is_column_in_scope("source_stage2", "stage2")
    assert is_column_in_scope("source_stage3", "stage4") and not is_column_in_scope("source_stage3", "stage3a")
    assert is_column_in_scope("p_dynamics_mode", "stage3a") and not is_column_in_scope("p_dynamics_mode", "stage4")
    assert is_column_in_scope("p_lr", "stage4")           # unscoped column is always in scope


def test_eval_metric_columns_derived_from_output_columns_no_drift():
    """_EVAL_METRIC_COLUMNS must stay OUTPUT_COLUMNS minus non-metric bookkeeping,
    so a new metric added to OUTPUT_COLUMNS is auto-recognized by fill-or-branch."""
    assert set(_EVAL_METRIC_COLUMNS) == set(OUTPUT_COLUMNS) - {"log"}


def test_eval_metrics_fills_on_caller_owned_cells_not_shared_params(tmp_path):
    """A tool checks only ITS cells: check_latent_channels (owned=ch*_imp) must
    FILL its importances into an existing rollout row (whose shared params are
    already present) rather than branch to a twin -- the shared params are not the
    discriminator."""
    f = tmp_path / "eval-stage4.csv"; ck = "checkpoints/stage4/x.pt"
    upsert_eval_metrics(f, ck, 8, {"rollout_median_corr_dx": 0.692, "rollout_n_steps": 6,
                                   "p_lr": "2e-7", "p_latent_channels": "4"})
    imp = {"ch0_imp": 0.6, "ch1_imp": 0.06, "p_latent_channels": "4"}   # shares p_latent_channels
    upsert_eval_metrics(f, ck, 8, imp, owned_cols=["ch0_imp", "ch1_imp"])
    rows = _read(f)
    assert len(rows) == 1                       # filled, not twinned
    assert rows[0]["ch0_imp"] == "0.6" and rows[0]["rollout_median_corr_dx"] == "0.692"


def test_fill_does_not_downgrade_existing_eval_variant(tmp_path):
    """Filling a row that already has a variant (rollout6) with a blank-variant
    write must NOT overwrite the tag -- the metrics on the row came from rollout6."""
    f = tmp_path / "eval-stage4.csv"; ck = "checkpoints/stage4/x.pt"
    upsert_eval_metrics(f, ck, 8, {"rollout_median_corr_dx": 0.7}, eval_variant="rollout6")
    upsert_eval_metrics(f, ck, 8, {"ch0_imp": 0.6}, owned_cols=["ch0_imp"])   # variant ""
    assert _read(f)[0]["eval_variant"] == "rollout6"


# --------------------------------------------------------------------------- #
# strip_to_checkpoints: real-space path shortening (separator-preserving)
# --------------------------------------------------------------------------- #
def test_strip_to_checkpoints_shortens_absolute_backslash_path():
    got = strip_to_checkpoints(r"D:\\work\\NN\\phase_field\\python\\checkpoints\\stage2\\128x128-stage2-20260827_11h24.pt")
    assert got == r"checkpoints\\stage2\\128x128-stage2-20260827_11h24.pt"


def test_strip_to_checkpoints_preserves_forward_slash_separator():
    got = strip_to_checkpoints("/abs/checkpoints/stage2/x.pt")
    assert got == "checkpoints/stage2/x.pt"          # forward slash kept, not converted


def test_strip_to_checkpoints_keeps_the_epoch_pin():
    got = strip_to_checkpoints(r"D:\\w\\checkpoints\\stage3b\\b.pt (epoch 5671, val_loss=1.44)")
    assert got == r"checkpoints\\stage3b\\b.pt (epoch 5671, val_loss=1.44)"


def test_strip_to_checkpoints_noop_when_already_relative():
    assert strip_to_checkpoints("checkpoints/stage2/x.pt") == "checkpoints/stage2/x.pt"
    assert strip_to_checkpoints(r"checkpoints\\stage2\\x.pt") == r"checkpoints\\stage2\\x.pt"


def test_strip_to_checkpoints_returns_input_when_no_checkpoints_segment():
    assert strip_to_checkpoints("D:/random/path/file.pt") == "D:/random/path/file.pt"
