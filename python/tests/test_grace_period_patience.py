"""
The grace period must not count against early-stopping patience.

Grace epochs are FORCED non-saves (the tracker returns should_save=False for
every epoch of a grace window so no single epoch plants a flag the EMA hasn't
absorbed). They are therefore not evidence of stagnation. A trainer that
increments its patience counter on every non-save -- including grace -- will
early-stop unconditionally whenever grace_epochs >= patience - 1, killing a run
that is genuinely improving.

Observed (stage 4, patience=6): epoch 9 was a real no-improvement (counter 1),
then reset_with_grace(5) at epoch 10 forced no-saves through epoch 14 (counter
2..6) -> early stop at epoch 14 with val 5.49 vs the last save's 7.47.

The fix (already in train_stage2, now in train_refinement and train_lds) is to
capture tracker.in_grace_period BEFORE update() and skip the increment for grace
epochs. It must be captured BEFORE because update() flips the flag on the LAST
grace epoch, so the post-update value would still count that final epoch.
"""
import pytest

# The tracker is pure python (dataclass + math) -- import it directly.
try:
    from training._checkpoint_criterion import CheckpointCriterionTracker
except ImportError:                       # flat layout
    from _checkpoint_criterion import CheckpointCriterionTracker


def _simulate(patience: int, vals: list[float], grace_at: int | None, grace_epochs: int,
              count_grace: bool) -> int | None:
    """Drive the real tracker through `vals`; return the epoch early-stop would
    fire, or None. count_grace=True reproduces the BUGGY counter (increments on
    every non-save); False is the fixed one (skips grace epochs)."""
    t = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.8)
    since = 0
    for epoch, v in enumerate(vals, start=1):
        if grace_at is not None and epoch == grace_at:
            t.reset_with_grace(grace_epochs)
        was_grace = t.in_grace_period          # captured BEFORE update
        _, saved = t.update(epoch, v)
        if saved:
            since = 0
        elif count_grace or not was_grace:
            since += 1
        if since >= patience:
            return epoch
    return None


# val trajectory shaped like the stage-4 run: improving, one blip before grace
_VALS = [14.3, 12.4, 11.2, 9.1, 8.2, 9.4, 7.8, 7.5,   # epochs 1-8 (saves happen along here)
         8.3,                                         # epoch 9: real no-improvement
         8.8, 6.0, 6.5, 5.5, 5.8,                     # epochs 10-14: GRACE (improving!)
         5.2, 5.0, 4.9]                               # epochs 15-17: post-grace


def test_buggy_counter_early_stops_inside_grace_while_improving():
    """Reproduces the incident: counting grace epochs exhausts patience=6 at the
    end of a 5-epoch grace that began with the counter at 1."""
    stop = _simulate(patience=6, vals=_VALS, grace_at=10, grace_epochs=5, count_grace=True)
    assert stop == 14, f"buggy counter should stop at epoch 14 (got {stop})"


def test_fixed_counter_survives_the_grace_period():
    """With grace epochs excluded, the same run does NOT early-stop at 14; it
    continues into the post-grace epochs where the improvements can save."""
    stop = _simulate(patience=6, vals=_VALS, grace_at=10, grace_epochs=5, count_grace=False)
    assert stop is None or stop > 14, f"fixed counter must not stop at/within grace (got {stop})"


def test_last_grace_epoch_flag_flips_inside_update():
    """Why the flag must be captured BEFORE update(): after the final grace
    epoch's update(), in_grace_period is already False -- but that epoch was a
    forced non-save and must still be excluded."""
    t = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.8)
    t.reset_with_grace(2)
    assert t.in_grace_period is True
    t.update(1, 5.0)                          # grace epoch 1
    assert t.in_grace_period is True
    before = t.in_grace_period
    _, saved = t.update(2, 4.0)               # grace epoch 2 (the LAST one)
    assert saved is False                     # still a forced non-save
    assert before is True and t.in_grace_period is False   # flipped INSIDE update()


def test_fixed_counter_still_stops_on_genuine_stagnation():
    """The fix must not disable early stopping: a genuinely flat run with no
    grace still stops at patience."""
    flat = [10.0] + [11.0] * 10               # one save, then nothing improves
    stop = _simulate(patience=6, vals=flat, grace_at=None, grace_epochs=0, count_grace=False)
    assert stop == 7                          # save at 1, then 6 flat epochs -> stop at 7
