"""
Tests for training/checkpoint_criterion.py -- pure arithmetic, no
torch/model/dataset dependency, so these run instantly and exercise the
exact state machine train_autoencoder()/train_stage2()/train_lds() all
now share.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_checkpoint_criterion.py -v
"""
import pytest

from training.checkpoint_criterion import CheckpointCriterionTracker


def test_no_warmup_first_epoch_always_saves():
    """ema_warmup_epochs=0 (train_autoencoder/train_stage2's mode):
    criterion is the ema from epoch 1, initialized to that epoch's own
    val_loss -- the first epoch should always save (nothing to beat yet)."""
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7)
    criterion, should_save = tracker.update(1, val_loss=5.0)
    assert should_save is True
    assert criterion == 5.0
    assert tracker.val_ema == 5.0


def test_no_warmup_ema_formula_correct():
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7)
    tracker.update(1, val_loss=10.0)
    criterion, _ = tracker.update(2, val_loss=4.0)
    expected = 0.7 * 10.0 + 0.3 * 4.0
    assert criterion == pytest.approx(expected)
    assert tracker.val_ema == pytest.approx(expected)


def test_no_warmup_only_saves_on_genuine_improvement():
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7)
    tracker.update(1, val_loss=1.0)  # ema=1.0, saves
    _, should_save_2 = tracker.update(2, val_loss=5.0)  # ema rises -> should not save
    assert should_save_2 is False
    _, should_save_3 = tracker.update(3, val_loss=0.1)  # ema drops enough -> should save
    # ema after epoch 2: 0.7*1.0 + 0.3*5.0 = 2.2; after epoch 3: 0.7*2.2 + 0.3*0.1 = 1.57
    # 1.57 < 1.0 (best from epoch 1)? No -- so epoch 3 should NOT save either.
    assert should_save_3 is False


def test_warmup_uses_raw_val_loss_not_ema():
    """During warmup, val_ema should stay None -- the ema doesn't even
    exist yet, since the criterion is raw val_loss."""
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=3, val_ema_decay=0.7)
    for epoch, val_loss in [(1, 5.0), (2, 3.0), (3, 4.0)]:
        criterion, _ = tracker.update(epoch, val_loss)
        assert criterion == val_loss
        assert tracker.val_ema is None


def test_ema_starts_exactly_at_first_post_warmup_epoch():
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=3, val_ema_decay=0.7)
    tracker.update(1, 5.0)
    tracker.update(2, 3.0)
    tracker.update(3, 4.0)
    assert tracker.val_ema is None
    criterion, _ = tracker.update(4, 2.0)  # first post-warmup epoch
    assert tracker.val_ema == 2.0  # initialized directly to this epoch's val_loss
    assert criterion == 2.0


def test_regression_warmup_era_value_does_not_permanently_block_saves():
    """
    THE EXACT BUG, reproduced with the real log data that exposed it: a
    stage-3b run where a lucky, noisy warmup-era val_loss (0.295 at
    epoch 1) blocked every single later save for 95 more epochs -- even
    ones that reached a genuinely low, well-converged ema (0.886 at
    epoch 88) -- purely because 0.886 was never numerically < 0.295,
    despite the two numbers not being comparable quantities at all (one
    raw, one smoothed). With the fix, the tracker resets its bar exactly
    when the criterion itself changes, so the run saves 25 times instead
    of once, correctly tracking genuine progress.
    """
    val_losses = [0.295, 0.574, 35.084, 73.054, 76460.397, 337.302, 1122.387, 120.618,
                  20.321, 20.687, 18.502, 28.709, 8.017, 23.494, 13.010, 13.063, 7.306,
                  11.131, 49.719, 4.932, 10.910, 5.891, 6071.779, 39.458, 11.685, 7.634,
                  113.373, 20.447, 2.607, 364.250, 1.845, 10.557, 2.643, 3.458, 68.360,
                  5.286, 7.635, 3.278, 2.003, 6.932, 4.406, 0.914, 1.500, 3.230, 1.889,
                  1.082, 0.743, 1.877, 2.478, 0.944, 0.554]

    tracker = CheckpointCriterionTracker(ema_warmup_epochs=5, val_ema_decay=0.7)
    saves = []
    for epoch, val_loss in enumerate(val_losses, start=1):
        _, should_save = tracker.update(epoch, val_loss)
        if should_save:
            saves.append(epoch)

    assert len(saves) > 1, (
        "reproduces the bug: without the fix, only epoch 1 ever saves, "
        "regardless of how well training actually converges afterward"
    )
    assert saves[0] == 1
    assert 51 in saves  # the genuine best point in this real run
    assert len(saves) == 25  # exact count, matching the fix's verified behavior


def test_epochs_since_improvement_pattern_is_reconstructible():
    """
    The tracker itself doesn't track epochs_since_improvement (that
    stays the caller's responsibility, e.g. for early stopping) -- but
    should_save's sequence must be enough to reconstruct it correctly:
    a run of consecutive False values between True values.
    """
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.5)
    results = [tracker.update(e, v)[1] for e, v in enumerate([5.0, 6.0, 7.0, 1.0, 8.0], start=1)]
    # epoch 1: ema=5.0, saves (True)
    # epoch 2: ema=0.5*5+0.5*6=5.5, worse, no save (False)
    # epoch 3: ema=0.5*5.5+0.5*7=6.25, worse, no save (False)
    # epoch 4: ema=0.5*6.25+0.5*1=3.625, better than 5.0, saves (True)
    # epoch 5: ema=0.5*3.625+0.5*8=5.8125, worse than 3.625, no save (False)
    assert results == [True, False, False, True, False]
