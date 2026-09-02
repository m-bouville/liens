"""
Regression test for the interaction between the checkpoint criterion's GRACE
PERIOD and early stopping.

During a grace window `should_save` is unconditionally False by design (see
CheckpointCriterionTracker.reset_with_grace) -- the criterion is deliberately
declining to answer while the EMA re-primes after a mid-run change of target,
not reporting a plateau. Counting those epochs as "no improvement" makes early
stopping fire whenever grace_epochs >= early_stopping_patience.

That is not a corner case: train_stage2 derives
`grace_epochs = max(2, round(1 / (1 - val_ema_decay)))` at the
deriv_target_centered switch, so val_ema_decay=0.8 gives 5 -- already above the
common early_stopping_patience=4. Observed on a real 32x32 run: the switch
happened at epoch 15, nothing could be saved during epochs 15-18, the counter
reached 4 and the run stopped at epoch 18, one epoch before the criterion would
have become usable again -- while its EMA was still falling monotonically
(50.3637 -> 50.0094 -> 47.8582 -> 47.5119).

These tests exercise the counter logic against the real tracker rather than
against train_stage2's whole training loop, which needs a dataset, a model and
several epochs to reach the same code path. The loop's own logic is three lines
and is reproduced here exactly.
"""
import pytest

from training._checkpoint_criterion import CheckpointCriterionTracker

# early_stopping_patience=4 is what the 32x32 and 64x64 params files use;
# grace=5 is what val_ema_decay=0.8 produces at the deriv_target_centered
# switch. The pair is the observed failing combination, not an invented one.
PATIENCE = 4
GRACE = 5


def _run_epochs(tracker, val_losses, patience, count_grace_epochs: bool):
    """train_stage2's own early-stopping loop, reduced to the three lines under
    test. `count_grace_epochs` selects the OLD behaviour (True: every
    non-saving epoch counts) or the new one (False: grace epochs are exempt).

    Returns (stopped_at_epoch or None, epochs_run).
    """
    epochs_since_improvement = 0
    for epoch, val_loss in enumerate(val_losses, start=1):
        was_in_grace_period = tracker.in_grace_period
        _, saved = tracker.update(epoch, val_loss)
        if saved:
            epochs_since_improvement = 0
        elif count_grace_epochs or not was_in_grace_period:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            return epoch, epoch
    return None, len(val_losses)


def _improving(n: int, start: float = 50.0, step: float = 1.0):
    """Monotonically falling val_loss -- a run that is genuinely getting
    better, so any early stop is unambiguously wrong."""
    return [start - i * step for i in range(n)]


def test_grace_period_alone_no_longer_triggers_early_stopping():
    """
    THE regression. With grace >= patience and a strictly improving val_loss,
    the old rule stopped inside the grace window every time.
    """
    tracker = CheckpointCriterionTracker(val_ema_decay=0.8)
    tracker.reset_with_grace(GRACE)
    stopped, _ = _run_epochs(tracker, _improving(12), PATIENCE, count_grace_epochs=False)
    assert stopped is None, f"stopped at epoch {stopped} despite improving throughout"


def test_the_old_rule_really_did_stop_inside_the_grace_window():
    """
    The negative twin, and the reason the test above is not vacuous: the same
    data under the old counting stops at exactly `patience` epochs -- inside a
    `grace`-epoch window it could never have escaped.
    """
    tracker = CheckpointCriterionTracker(val_ema_decay=0.8)
    tracker.reset_with_grace(GRACE)
    stopped, _ = _run_epochs(tracker, _improving(12), PATIENCE, count_grace_epochs=True)
    assert stopped == PATIENCE
    assert stopped < GRACE, "the failure needs grace > patience to be structural"


def test_a_real_plateau_after_grace_still_stops():
    """
    Exempting grace epochs must not disable early stopping. Once the window
    closes, a flat val_loss has to stop the run as before -- otherwise the fix
    has bought its correctness by removing the feature.
    """
    tracker = CheckpointCriterionTracker(val_ema_decay=0.8)
    tracker.reset_with_grace(GRACE)
    # improve through grace, then go flat at a value nothing can beat
    val_losses = _improving(GRACE) + [99.0] * 10
    stopped, _ = _run_epochs(tracker, val_losses, PATIENCE, count_grace_epochs=False)
    assert stopped is not None
    assert stopped > GRACE


def test_grace_epochs_are_exempt_but_still_advance_the_grace_counter():
    """
    The exemption must not extend the grace window itself: the tracker still
    counts down one epoch per update(), so the window is `grace` epochs long
    whether or not early stopping is looking.
    """
    tracker = CheckpointCriterionTracker(val_ema_decay=0.8)
    tracker.reset_with_grace(GRACE)
    for epoch in range(1, GRACE + 1):
        assert tracker.in_grace_period, f"epoch {epoch} should still be in grace"
        tracker.update(epoch, 50.0 - epoch)
    assert not tracker.in_grace_period


def test_the_final_grace_epoch_is_exempt_too():
    """
    tracker.in_grace_period flips to False on the LAST grace epoch -- during
    update(), after that epoch's own should_save has already been decided as
    False. So it must be read BEFORE update(), or the final grace epoch is
    counted as a real non-improvement even though nothing could have been saved
    in it.

    The miscount is a single epoch and an improving run usually saves on the
    next one, hiding it -- which is exactly why it is asserted directly here
    rather than through an end-to-end early-stopping scenario: the bug is real
    but only bites when the counter was already near patience before the switch.
    """
    tracker = CheckpointCriterionTracker(val_ema_decay=0.8)
    tracker.reset_with_grace(GRACE)

    exempted = []
    for epoch in range(1, GRACE + 1):
        before = tracker.in_grace_period
        _, saved = tracker.update(epoch, 50.0 - epoch)
        assert not saved, f"epoch {epoch} is inside the grace window and must not save"
        exempted.append(before)
        after = tracker.in_grace_period
        if epoch == GRACE:
            # the flip happens on this epoch, so before and after disagree
            assert before and not after
    assert all(exempted), "every grace epoch, including the last, must be exempt"


@pytest.mark.parametrize("grace", [0, 1, 2, 6])
def test_no_early_stop_during_grace_for_any_window_length(grace):
    tracker = CheckpointCriterionTracker(val_ema_decay=0.8)
    if grace:
        tracker.reset_with_grace(grace)
    stopped, _ = _run_epochs(tracker, _improving(grace + PATIENCE + 2), PATIENCE,
                              count_grace_epochs=False)
    assert stopped is None
