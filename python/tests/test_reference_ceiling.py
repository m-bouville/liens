"""
A resumed run must never save a checkpoint WORSE than the one it started from.

best_val_loss started at inf, so epoch 1 always saved. Observed on a real
stage-2 resume:

    ref| ... | 3.9054 ...  (before this run)
      1| ... | 4.0149 ...  -> saved
      2| ... | 4.3544
      3| ... | 4.8032
      4| ... | 3.3848

Epoch 1 overwrote a better checkpoint with a worse one, and epochs 2-3 were
worse still before epoch 4 finally beat the ancestor. Only the backup made it
recoverable.

The reference row is already a FULL validation pass of the ancestor under this
run's own objective -- its value was computed, printed, and discarded. Feeding
it to the tracker costs nothing.
"""
import inspect

import pytest

from conftest import source_without_comments
from training.checkpoint_criterion import CheckpointCriterionTracker


def test_without_a_reference_epoch_one_always_saves():
    """The behaviour being fixed, pinned so the fix is visibly a change."""
    t = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7)
    _, saved = t.update(1, 999.0)
    assert saved, "with no reference, anything beats inf"


def test_a_worse_first_epoch_does_not_save_when_a_reference_is_known():
    t = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7,
                                    reference_val_loss=3.9054)
    _, saved = t.update(1, 4.0149)          # the real numbers
    assert not saved


def test_a_better_epoch_still_saves():
    t = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7,
                                    reference_val_loss=3.9054)
    assert not t.update(1, 4.0149)[1]
    assert not t.update(2, 4.3544)[1]
    # by epoch 4 the EMA has come down below the ancestor
    t.update(3, 4.8032)
    for e, v in ((4, 3.3848), (5, 3.7504), (6, 3.2), (7, 3.0)):
        _, saved = t.update(e, v)
        if saved:
            break
    else:
        pytest.fail("a genuinely better model never saved")


def test_the_reference_is_a_CEILING_during_a_grace_period_too():
    """
    GUARDS the grace branch overwriting best_val_loss with its own EMA. Its
    job is to stop a lucky early epoch planting a flag -- not to license
    saving something worse than the run started from.
    """
    t = CheckpointCriterionTracker(ema_warmup_epochs=3, val_ema_decay=0.7,
                                    reference_val_loss=1.0)
    for e in range(1, 4):
        t.update(e, 5.0)                    # grace epochs, all far worse
    assert t.best_val_loss <= 1.0, (
        f"grace raised the bar to {t.best_val_loss} -- above the ancestor's 1.0"
    )
    assert not t.update(4, 4.0)[1], "a worse-than-ancestor epoch saved after grace"


def test_reset_with_grace_DISCARDS_the_old_reference():
    """
    A reset happens precisely because the objective changed, so a bar measured
    under the superseded one is exactly what must not be reused. The caller
    may supply a freshly-measured replacement.
    """
    t = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7,
                                    reference_val_loss=1.0)
    t.reset_with_grace(0)
    assert t.reference_val_loss is None
    assert t.best_val_loss == float("inf")

    t.reset_with_grace(0, reference_val_loss=2.5)
    assert t.reference_val_loss == 2.5
    assert t.best_val_loss == 2.5


def test_no_reference_reproduces_the_historical_behaviour_exactly():
    """Every existing caller passes nothing and must be unaffected."""
    assert (inspect.signature(CheckpointCriterionTracker).parameters
            ["reference_val_loss"].default is None)
    t = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7)
    assert t.update(1, 12345.0)[1]


def test_stage2_applies_the_ceiling_only_against_a_stage2_ancestor():
    """
    The ceiling protects against OVERWRITING a better checkpoint. When stage
    2's ancestor is a single-stream stage-1a checkpoint there is nothing to
    overwrite -- the output is a new file in a different key layout
    ("encoders.shared.*" vs "encoder.*") -- so no earlier stage-2 result is at
    risk and the ceiling protects nothing.

    Applied there anyway it blocked saving with no fallback available, and a
    short run that did not improve produced NO checkpoint at all.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    src = source_without_comments(root / "training/train_stage2.py")
    assert "if resumed_from_stage2:" in src
    guard = src[src.index("if resumed_from_stage2:"):]
    assert "tracker.reference_val_loss = ref_total" in guard[:120], (
        "the seeding must be inside the resumed_from_stage2 guard"
    )


@pytest.mark.parametrize("module", ["training/train_stage1.py", "training/train_stage2.py"])
def test_the_stages_that_print_a_reference_row_also_USE_it(module):
    """
    GUARDS computing the reference, printing it, and throwing it away -- which
    is what both stages did. The pass is already paid for.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    src = source_without_comments(root / module)
    assert "tracker.reference_val_loss = " in src, (
        f"{module} prints a reference row but does not seed the tracker with it"
    )


# --------------------------------------------------------------------
# "nothing improved" is an outcome, not a failure
# --------------------------------------------------------------------

@pytest.mark.parametrize("module,label", [
    ("training/train_stage1.py", "stage 1"),
    ("training/train_stage2.py", "stage 2"),
])
def test_nothing_beating_the_ancestor_keeps_the_ancestor(module, label):
    """
    REGRESSION from the ceiling itself.

    With a reference in place a short resumed run that does not improve saves
    NOTHING -- which is correct -- but the no-save guard then raised, so the
    run failed outright. Whether it did was data-dependent: two tests passed
    on one machine and failed on another for exactly this.

    The honest outcome is that the ANCESTOR is the best model available, so it
    is kept as the output. That preserves the contract every caller relies on
    -- a returned path that exists -- without pretending a worse model was an
    improvement.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    src = source_without_comments(root / module)
    assert "tracker.reference_val_loss is not None" in src, (
        f"{module} does not distinguish 'nothing improved' from 'nothing saved'"
    )
    assert "no epoch improved on the ancestor" in src
    assert "_shutil.copy2(resume_from, checkpoint_path)" in src, (
        "the ancestor must be kept as the output, not merely reported"
    )


@pytest.mark.parametrize("module", ["training/train_stage1.py", "training/train_stage2.py"])
def test_a_run_with_NO_ancestor_still_raises(module):
    """
    GUARDS softening the guard into silence. Without a reference there is no
    ancestor to fall back on, so nothing-saved really is a failure and must
    still raise -- the epochs=0 and never-improved cases both live there.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    src = source_without_comments(root / module)
    assert "elif not Path(checkpoint_path).exists():" in src
    assert "without ever saving" in src


def test_the_copy_is_skipped_when_the_paths_are_the_same():
    """A self-resume writes to the file it read from; copying it onto itself
    is at best pointless and at worst truncates it."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for module in ("training/train_stage1.py", "training/train_stage2.py"):
        src = source_without_comments(root / module)
        assert "Path(resume_from).resolve() != Path(checkpoint_path).resolve()" in src, module


# --------------------------------------------------------------------
# which stages have this, and why
# --------------------------------------------------------------------

def test_only_the_stages_with_a_reference_row_seed_the_tracker():
    """
    Pins the asymmetry, so it reads as a decision rather than an omission.

    Stages 1 and 2 print a reference row -- a full validation pass of the
    ANCESTOR under this run's own objective -- so the value is free and is
    handed to the tracker as a ceiling.

    Stage 3 has no such row: it resumes f_theta but its ancestor's val_loss was
    measured at a different n_rollout_steps, so there is nothing comparable to
    seed with. Stage 4/5 evaluates and SAVES at epoch 0, so its own first row
    already plays the reference's role.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    have = {}
    for name in ("train_stage1", "train_stage2", "train_lds", "train_refinement"):
        src = source_without_comments(root / f"training/{name}.py")
        have[name] = "tracker.reference_val_loss = " in src
    assert have["train_stage1"] and have["train_stage2"]
    assert not have["train_lds"] and not have["train_refinement"], (
        "a stage gained a reference seed without a reference row to measure it from"
    )


@pytest.mark.parametrize("module", ["training/train_stage1.py", "training/train_stage2.py"])
def test_no_improvement_keeps_the_ancestor_instead_of_raising(module):
    """
    GUARDS raising when a resumed run simply fails to improve.

    The ceiling makes that a COMMON outcome -- a resumed run must now beat what
    it started from rather than merely beat infinity -- and it is not a
    failure: the honest result is that the ancestor is still the best model.
    Raising made short resumed runs fail depending on the data split; two tests
    passed on one machine and failed on another for exactly this.

    Only the stages WITH a ceiling need the fallback, which is why stage 3 and
    4/5 keep the plain raise: there, nothing saving still means nothing ever
    beat infinity.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    src = source_without_comments(root / module)
    assert "no epoch improved on the ancestor" in src
    assert "shutil" in src, "the ancestor must be copied to the output path"
    # and the plain raise survives for the genuinely-empty case
    assert "without ever saving" in src


def test_the_fallback_survives_a_mid_run_criterion_reset():
    """
    GUARDS keying the ancestor-kept fallback on tracker.reference_val_loss
    still being set.

    reset_with_grace CLEARS the reference when the objective changes mid-run
    (stage 2's deriv_target_centered switch), so a run that switched would fall
    through to the raise -- and that is exactly the run most likely not to
    improve, since its criterion restarts from the post-grace EMA rather than
    from the ancestor.

    Observed as a stage-2 RuntimeError in a test whose ancestor was already
    past the switch point.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for module in ("training/train_stage1.py", "training/train_stage2.py"):
        src = source_without_comments(root / module)
        assert "exists() and resume_from is not None" in src, (
            f"{module} keys the fallback on something that a mid-run reset can clear"
        )
        assert "tracker.reference_val_loss is not None:" not in src


def test_the_epochs_zero_sentence_is_conditional():
    """
    GUARDS appending it unconditionally. Observed in the reported error:
    "An epochs=0 ablation cannot produce a checkpoint" on a run with
    epochs=20 -- which sends the reader to a setting that is not the problem.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for module in ("training/train_stage2.py", "training/train_lds.py"):
        src = source_without_comments(root / module)
        idx = src.find("An epochs=0 ablation cannot produce")
        if idx < 0:
            continue
        # The condition trails the string -- f"{'...' if epochs == 0 else ''}"
        # -- so the window looks FORWARD. An earlier version looked backward
        # and failed on correct code.
        window = src[idx:idx + 400]
        assert "if epochs == 0 else" in window, (
            f"{module} states the epochs=0 explanation unconditionally"
        )
