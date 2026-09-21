"""Source-level guards for two families of change from the past few days that had
no coverage: the scales-plot wiring across the trainers, and the lr_warmup_epochs
epoch-units semantics. These read the trainer sources (no torch needed) and pin
the specific lines that make the features work, so a future edit that silently
drops one is caught."""
import pathlib
import re

from _ast_helpers import parse_module, calls_to, multiplies

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(rel):
    return " ".join((_ROOT / rel).read_text(encoding="utf-8", errors="ignore").split())


# ---------- scales-plot wiring (L_XX / XX_scale, val) ----------

def test_refinement_builds_and_emits_the_scale_ratio_history():
    s = _src("training/train_refinement.py")
    assert "scale_ratio_history" in s
    assert "_val_raw[name] / _scales[name]" in s          # ratio = raw / scale (val)
    assert "scale_ratios=scale_ratio_history" in s        # passed to the figure


def test_stage2_builds_the_scale_ratio_history_from_term_values():
    s = _src("training/train_stage2.py")
    assert "scale_ratio_history" in s
    assert "val_recon / recon0_scale" in s                # recon0 ratio
    assert "_rvv / _rs" in s                               # each term's vv/scale
    assert "scale_ratios=scale_ratio_history" in s


def test_stage1_builds_and_emits_the_scale_ratio_history():
    s = _src("training/train_stage1.py")
    assert "scale_ratio_history" in s
    assert "val_recon0 / recon0_scale" in s
    assert "val_stats0 / stats0_scale" in s
    # stage 1 now emits its figures through the shared writer (the direct
    # loss_scale_curve call was migrated), passing the ratios as scale_ratios.
    assert "scale_ratios=scale_ratio_history" in s
    assert "write_epoch_figures(" in s


def test_train_lds_builds_and_emits_the_scale_ratio_history():
    """train_lds builds a scale_ratio_history too (one entry per active
    component, appended from _val_means[c + "_ratio"] every epoch) and passes
    it to write_epoch_figures alongside its own stage-3-specific rollout/1step
    scatter (extra=...) -- the only one of the four trainers wired for
    scale_ratio_history that had no coverage here. Missing this wiring would
    leave the "Stage 3 loss/scale ratios" figure silently blank or stale,
    exactly the class of regression this whole file exists to catch for the
    other three trainers."""
    s = _src("training/train_lds.py")
    assert "scale_ratio_history" in s
    assert 'scale_ratio_history[c].append(_val_means[c + "_ratio"])' in s
    assert "scale_ratios=scale_ratio_history" in s
    assert "scales_path=loss_scales_path" in s


def test_write_epoch_figures_forwards_scale_ratios_to_the_curve():
    s = _src("training/_training_loop.py")
    assert "scale_ratios" in s and "loss_scale_curve(" in s


# ---------- lr_warmup_epochs: rename + epoch-units + default 0 ----------

def test_lr_warmup_is_in_EPOCHS_not_batches_in_both_trainers():
    # The epoch->optimiser-step conversion lives in ONE place, make_lr_warmup
    # (training/_training_loop.py). Asserted STRUCTURALLY (AST), not as a source
    # substring: multiplies() finds the `lr_warmup_epochs * len(train_loader)`
    # BinOp however it is formatted, and returns False for a bare
    # `total_iters=lr_warmup_epochs` (the batch-units bug) -- so the test survives
    # a reformat but still catches the mistake.
    helper_tree = parse_module(_ROOT / "training/_training_loop.py")
    assert multiplies(helper_tree, "lr_warmup_epochs", "train_loader"), (
        "make_lr_warmup must convert epochs to optimiser steps: "
        "total_iters = lr_warmup_epochs * len(train_loader)")
    # both trainers must ROUTE THROUGH make_lr_warmup (a real call), not inline LinearLR
    for rel in ("training/train_lds.py", "training/train_refinement.py"):
        raw = _src(rel)
        assert "lr_warmup_steps" not in raw, f"{rel} still uses the old name"
        assert "lr_warmup_epochs" in raw
        tree = parse_module(_ROOT / rel)
        assert calls_to(tree, "make_lr_warmup"), (
            f"{rel}: must build the lr warmup via the shared make_lr_warmup")
        assert not calls_to(tree, "LinearLR"), (
            f"{rel}: no inline LinearLR -- the conversion belongs in make_lr_warmup")


def test_lr_warmup_epochs_defaults_to_zero_in_both_trainers():
    for rel in ("training/train_lds.py", "training/train_refinement.py"):
        s = _src(rel)
        assert "lr_warmup_epochs: int = 0" in s, f"{rel}: default must be 0"


def test_lr_scheduler_steps_only_on_a_taken_step():
    # advancing the schedule on a skipped batch consumes the warmup without
    # training (and torch warns); both trainers guard this.
    for rel in ("training/train_lds.py", "training/train_refinement.py"):
        s = _src(rel)
        assert "lr_scheduler" in s and "if lr_scheduler is not None" in s
