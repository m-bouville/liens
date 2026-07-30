"""
Tests for training/checkpoint_criterion.py -- pure arithmetic, no
torch/model/dataset dependency, so these run instantly and exercise the
exact state machine train_autoencoder()/train_stage2()/train_lds() all
now share.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_checkpoint_criterion.py -v
"""
import pytest

from training.checkpoint_criterion import CheckpointCriterionTracker, clamp_grace_epochs


def test_no_warmup_first_epoch_always_saves():
    """ema_warmup_epochs=0 (train_autoencoder/train_stage2's mode):
    criterion is the ema from epoch 1, initialized to that epoch's own
    val_loss -- the first epoch should always save (nothing to beat yet).
    Unaffected by the grace-period mechanism: 0 grace epochs means
    __post_init__ leaves _grace_remaining at 0, so this call skips the
    grace branch entirely -- byte-identical to the pre-grace behavior."""
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7)
    assert tracker.in_grace_period is False
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


def test_warmup_accumulates_ema_but_never_saves():
    """
    REVISED from the pre-grace design: during warmup (now internally a
    grace period, see reset_with_grace's own docstring), val_ema DOES
    accumulate -- honestly reflecting what the EMA actually is at each
    point, and staying usable for anything watching it (e.g. a
    loss-curve plot), rather than being withheld as None until warmup
    ends. What must hold regardless: should_save is unconditionally
    False throughout, and in_grace_period reports the transition
    correctly.
    """
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=3, val_ema_decay=0.7)
    expected_ema = None
    for epoch, val_loss in [(1, 5.0), (2, 3.0), (3, 4.0)]:
        assert tracker.in_grace_period is True
        criterion, should_save = tracker.update(epoch, val_loss)
        expected_ema = val_loss if expected_ema is None else 0.7 * expected_ema + 0.3 * val_loss
        assert criterion == pytest.approx(expected_ema)
        assert tracker.val_ema == pytest.approx(expected_ema)
        assert should_save is False
    assert tracker.in_grace_period is False  # grace ended exactly after the 3rd call


def test_ema_continues_smoothly_across_the_grace_boundary():
    """
    REVISED from the pre-grace design: the first post-grace epoch
    CONTINUES the same EMA grace already built up (decay-blending into
    it), rather than resetting to a fresh, unsmoothed raw value -- that
    reset-to-raw is exactly the mechanism reset_with_grace's own
    docstring identifies as the actual problem (a single epoch's raw
    value becoming an unfair bar for smoothed values to beat).
    """
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=3, val_ema_decay=0.7)
    tracker.update(1, 5.0)
    tracker.update(2, 3.0)
    ema_at_grace_end, _ = tracker.update(3, 4.0)
    assert tracker.in_grace_period is False
    criterion, should_save = tracker.update(4, 2.0)  # first post-grace epoch
    expected = 0.7 * ema_at_grace_end + 0.3 * 2.0  # continues the SAME ema, not a fresh 2.0
    assert criterion == pytest.approx(expected)
    assert tracker.val_ema == pytest.approx(expected)
    # best_val_loss was seeded from ema_at_grace_end when grace ended --
    # whether THIS epoch saves depends on whether the continued ema
    # improved on that, not on being compared to inf.
    assert should_save == (criterion < ema_at_grace_end)


def test_no_save_possible_during_grace_even_for_an_excellent_val_loss():
    """The core guarantee: grace_epochs genuinely blocks EVERY save
    during the window, regardless of how good val_loss looks -- an
    exceptionally low value during grace must not be allowed to save,
    since it hasn't been smoothed into a trustworthy criterion yet."""
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=3, val_ema_decay=0.7)
    _, should_save_1 = tracker.update(1, val_loss=0.0001)  # about as good as it gets
    _, should_save_2 = tracker.update(2, val_loss=0.0001)
    _, should_save_3 = tracker.update(3, val_loss=0.0001)
    assert (should_save_1, should_save_2, should_save_3) == (False, False, False)


def test_regression_grace_period_produces_more_genuine_saves_than_raw_warmup():
    """
    Real log data that originally exposed a related bug (see this
    module's own docstring) -- re-verified here against the grace-period
    mechanism specifically. The OLD warmup design (raw val_loss compared
    every warmup epoch, best_val_loss reset once warmup ends) already
    avoided the WORST case for this exact dataset (only because epoch 6,
    the first post-warmup epoch, happened to land on an unlucky-HIGH raw
    value, 337.302, that later smoothed values could easily beat) -- 25
    saves. That "avoided the worst case" was luck, not a guarantee: had
    epoch 6 instead been unlucky-LOW, it would have reproduced the exact
    permanently-blocked-saves failure this module's own docstring
    describes. The grace period removes that dependency on luck entirely
    -- no epoch during grace can ever save, so there's no single raw
    value, lucky or not, for later epochs to be unfairly measured
    against. Verified here to produce MORE genuine saves on this same
    data (32, vs the old design's 25), not fewer -- the grace period
    isn't just "safer", it recognizes real improvement more often too.
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

    assert all(epoch > 5 for epoch in saves), (
        "no epoch during the 5-epoch grace window should ever save"
    )
    assert len(saves) == 32
    assert saves[0] == 6  # first epoch eligible to save at all


def test_reset_with_grace_blocks_saves_for_exactly_n_epochs_then_resumes():
    """The general mechanism, exercised directly (not via
    ema_warmup_epochs/__post_init__) -- a MID-RUN reset, matching
    train_stage2()'s own deriv_target_centered use case."""
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7)
    tracker.update(1, val_loss=1.0)  # saves, best_val_loss=1.0
    tracker.update(2, val_loss=0.5)  # saves, best_val_loss=0.5

    tracker.reset_with_grace(3)
    assert tracker.in_grace_period is True
    assert tracker.best_val_loss == float("inf")
    assert tracker.val_ema is None

    # A LUCKY, very low value right after the reset, followed by two
    # more grace epochs that dilute its own influence on the ema --
    # exactly why grace_epochs must be >1 for the fix to matter at all
    # (see test_reset_with_grace_single_epoch_grace_does_not_fix_anything
    # for the degenerate case where it doesn't).
    _, should_save_1 = tracker.update(3, val_loss=0.001)
    ema_at_grace_end, should_save_2 = tracker.update(4, val_loss=50.0)
    ema_at_grace_end, should_save_3 = tracker.update(5, val_loss=50.0)
    assert (should_save_1, should_save_2, should_save_3) == (False, False, False)
    assert tracker.in_grace_period is False  # grace ended after exactly 3 calls
    # best_val_loss is the FULLY-BLENDED ema at the moment grace ended
    # (0.7^2*0.001 + 0.7*0.3*50 + 0.3*50 ~= 25.5), NOT the raw 0.001 --
    # continuous tracking through grace (see the tracker's own update())
    # means the lucky value's own influence is ALREADY diluted by the
    # time grace closes, not frozen in place as an unbeatable bar.
    assert tracker.best_val_loss == pytest.approx(ema_at_grace_end)
    assert tracker.best_val_loss > 1.0  # nowhere near the raw, lucky 0.001

    # Post-grace: the continued ema is compared fairly against that
    # blended value -- a val_loss low enough to pull the ema back down
    # can still save; the deciding factor is the properly-smoothed
    # comparison, not whether it beats an artificially low raw value.
    _, should_save_4 = tracker.update(6, val_loss=5.0)
    assert should_save_4 is True


def test_reset_with_grace_single_epoch_grace_does_not_fix_anything():
    """
    Important, easy-to-miss nuance: grace_epochs=1 is MATHEMATICALLY
    IDENTICAL to the original bug -- with only one grace epoch, there is
    no second value for the ema to blend with, so best_val_loss at the
    moment grace ends is EXACTLY that one epoch's own raw value, lucky
    or not. The fix's own value comes entirely from grace_epochs being
    large enough for real smoothing to happen before comparisons resume
    -- callers must choose grace_epochs >= 2 (train_stage2's own choice,
    derived from val_ema_decay, lands well above this).
    """
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7)
    tracker.reset_with_grace(1)
    tracker.update(1, val_loss=0.001)  # the ONLY grace epoch -- lucky and low
    assert tracker.best_val_loss == 0.001  # unblended -- exactly the raw value

    _, should_save = tracker.update(2, val_loss=40.0)  # a perfectly reasonable value
    assert should_save is False, (
        "grace_epochs=1 reproduces the original bug exactly -- this is expected, "
        "documented behavior, not a regression"
    )


def test_reset_with_grace_zero_epochs_behaves_like_no_grace_at_all():
    """grace_epochs=0 should be a no-op transition -- the very next
    update() call is treated the same as any tracker's first-ever call."""
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.7)
    tracker.update(1, val_loss=1.0)
    tracker.reset_with_grace(0)
    assert tracker.in_grace_period is False
    criterion, should_save = tracker.update(2, val_loss=99.0)
    assert should_save is True  # nothing to beat, since best_val_loss was reset to inf
    assert criterion == 99.0
    assert tracker.val_ema == 99.0


def test_best_val_loss_stays_finite_and_continuous_through_grace():
    """best_val_loss must never be left at a raw float("inf") while
    grace is active -- code watching it (e.g. loss_curve's own y-axis
    scaling, which computes percentiles across every plotted value)
    would break on that. It should track the accumulating ema
    continuously instead, even though nothing is ever compared against
    it until grace ends."""
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=4, val_ema_decay=0.7)
    for epoch, val_loss in enumerate([3.0, 2.0, 5.0, 1.0], start=1):
        tracker.update(epoch, val_loss)
        assert tracker.best_val_loss != float("inf")
        assert tracker.best_val_loss == tracker.val_ema


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


def test_clamp_grace_epochs_always_leaves_one_saveable_epoch():
    """
    REGRESSION: a grace period covering EVERY remaining epoch means no
    checkpoint is written at all -- a missing FILE, not merely a worse
    checkpoint, which then fails far downstream with a confusing
    FileNotFoundError. Real crash: train_lds()'s epochs=0 ablation makes
    exactly ONE update() call and depends on it saving (it produces the
    ephemeral stage-3 wrapper ensure_lds_checkpoint needs), but
    ema_warmup_epochs defaults to 5, so an unclamped grace swallowed it.
    """
    assert clamp_grace_epochs(5, 1) == 0    # train_lds epochs=0: one call, must be able to save
    assert clamp_grace_epochs(5, 2) == 1
    assert clamp_grace_epochs(5, 3) == 2
    assert clamp_grace_epochs(5, 6) == 5    # enough room -> full requested grace
    assert clamp_grace_epochs(5, 100) == 5
    assert clamp_grace_epochs(0, 100) == 0  # no grace requested -> none applied
    # Degenerate/hostile inputs must never produce a negative grace
    assert clamp_grace_epochs(5, 0) == 0
    assert clamp_grace_epochs(5, -3) == 0


def test_single_update_call_can_still_save_after_clamping():
    """The epochs=0 ablation shape, end to end through the tracker: with
    the clamp applied, the one and only update() call must save."""
    tracker = CheckpointCriterionTracker(
        ema_warmup_epochs=clamp_grace_epochs(5, 1), val_ema_decay=0.7,
    )
    assert tracker.in_grace_period is False
    _, should_save = tracker.update(0, val_loss=0.0)
    assert should_save is True


def test_clamped_midrun_reset_still_permits_a_save():
    """The mid-run (reset_with_grace) shape on a run too short for the
    full grace period -- at least one save must remain possible."""
    tracker = CheckpointCriterionTracker(ema_warmup_epochs=0, val_ema_decay=0.8)
    tracker.update(1, val_loss=1.0)
    total_epochs, switch_epoch = 3, 2
    tracker.reset_with_grace(clamp_grace_epochs(5, total_epochs - switch_epoch + 1))
    saves = [tracker.update(e, v)[1] for e, v in [(2, 0.9), (3, 0.8)]]
    assert any(saves), "a short run must still be able to save at least once after a switch"
