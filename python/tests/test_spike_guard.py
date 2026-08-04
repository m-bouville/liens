"""
A single catastrophic batch must not end a 3000-epoch run.

Both 128x128 stage-3 runs were ENDED by one spike:

    3a: last save 3547, stopped 4047   gap = 500 = patience
    3b: last save 2294, stopped 2794   gap = 500 = patience

In each case the entire patience window sits after the spike and never
recovers to the pre-spike best. Neither run plateaued -- what looked like
convergence was a crash, and the reported val_loss is an upper bound on what
those configurations can reach.

val_loss spiking alongside train is what identifies it as WEIGHT damage
rather than one bad forward pass: val runs under no_grad with no z0 noise.
"""
import math
from pathlib import Path

import torch

import pytest

from conftest import source_without_comments
from training.train_lds import _SpikeGuard

import pathlib
_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _warm(guard, value=1.0, n=60):
    for _ in range(n):
        assert not guard.should_skip(value)
    return guard


def _warm_skewed(guard, n=60):
    """History whose MEAN is far above its MEDIAN.

    Warming with identical values makes the two coincide, so a test built on
    it cannot tell them apart -- an earlier version did exactly that and
    passed with the median swapped for the mean.
    """
    for _ in range(n - 6):
        assert not guard.should_skip(1.0)
    for _ in range(6):
        assert not guard.should_skip(20.0)      # large, but under 50x
    return guard


def test_a_nonfinite_loss_is_skipped_immediately():
    """
    No history needed and none waited for. grad_clip does not merely fail to
    help here -- it converts inf to nan: clip_coef = max_norm/(inf+eps) = 0
    and 0*inf = nan, poisoning every parameter in one step, permanently.
    """
    guard = _SpikeGuard(factor=50.0)
    for bad in (float("inf"), float("-inf"), float("nan")):
        assert guard.should_skip(bad), f"{bad} reached the optimizer"
    assert guard.n_nonfinite == 3


def test_clip_grad_norm_really_does_turn_inf_into_nan():
    """The claim above, verified against torch rather than asserted."""
    import torch

    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p.grad = torch.tensor([float("inf"), 1.0])
    torch.nn.utils.clip_grad_norm_([p], 1.0)
    assert torch.isnan(p.grad).any(), "the trapdoor this guard exists for is gone"

    q = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    q.grad = torch.tensor([1e20, 1.0])
    torch.nn.utils.clip_grad_norm_([q], 1.0)
    assert torch.isfinite(q.grad).all(), "a merely LARGE gradient must clip normally"


def test_a_huge_finite_loss_is_skipped_once_there_is_history():
    guard = _warm(_SpikeGuard(factor=50.0))
    assert guard.should_skip(1000.0)
    assert guard.n_skipped == 1
    assert guard.n_nonfinite == 0


def test_ordinary_noise_is_NOT_skipped():
    """
    GUARDS an over-eager threshold. Silently dropping hard batches would bias
    what the model learns -- worse than the crash, because the run would look
    healthy. The guard is for catastrophes, not for smoothing.
    """
    guard = _warm(_SpikeGuard(factor=50.0))
    for v in (2.0, 5.0, 0.2, 10.0, 30.0):
        assert not guard.should_skip(v), f"{v} (x{v:.0f} the median) was skipped"


def test_the_guard_waits_for_enough_history():
    """
    GUARDS judging against a median of one or two samples, where the first
    genuinely-large-but-normal batch would define the scale and everything
    after it look like an outlier.
    """
    guard = _SpikeGuard(factor=50.0, min_history=50)
    assert not guard.should_skip(1.0)
    assert not guard.should_skip(1000.0), "judged with a 1-sample history"


def test_the_threshold_uses_the_MEDIAN_not_the_mean():
    """
    The mean is what an outlier corrupts; the median is not. Built on a
    SKEWED history so the two give different verdicts -- with mean ~2.9 and
    median 1.0, a loss of 100 is 34x the mean (kept) but 100x the median
    (skipped). Mutation-verified: swapping median for mean fails here.
    """
    import statistics as _st
    guard = _warm_skewed(_SpikeGuard(factor=50.0))
    hist = list(guard._recent)
    mean, med = sum(hist) / len(hist), _st.median(hist)
    assert mean > 2 * med, f"history not skewed enough to discriminate: {mean=} {med=}"
    assert guard.should_skip(100.0), (
        f"100 is {100/med:.0f}x the median but only {100/mean:.0f}x the mean -- "
        f"the threshold is using the mean"
    )


def test_a_run_of_spikes_does_not_drag_the_threshold_up():
    """
    The feedback loop the guard exists to avoid: if skipped losses entered the
    history, a sustained burst would raise the bar until spikes read as
    normal and the guard silently stopped guarding.

    Needs MANY spikes -- one cannot move a 60-sample median, which is why an
    earlier single-spike version of this test missed the mutation.
    """
    guard = _warm(_SpikeGuard(factor=50.0))
    before = guard.median()
    for _ in range(200):
        assert guard.should_skip(1e6)
    assert math.isclose(guard.median(), before), (
        f"median moved {before} -> {guard.median()} -- skipped losses are being recorded"
    )





def test_factor_zero_disables_the_guard_entirely():
    guard = _SpikeGuard(factor=0.0)
    for v in (float("nan"), float("inf"), 1e12):
        assert not guard.should_skip(v)
    assert guard.n_skipped == 0


def test_skips_are_reported_not_silent():
    """
    GUARDS a guard that drops data quietly: the run would look healthy while
    training on a filtered distribution.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "skipped" in src and "catastrophic outlier" in src
    assert "spike_guard.n_skipped != _spikes_reported" in src


def test_the_report_names_what_was_IN_the_batch():
    """
    "Cursed with sudden peaks" is not actionable; "the spikes are all dt near
    max_dt at low noise" is. dt_max and theta are what turn one into the other.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "dt_max=" in src and "theta[0]" in src
    assert "def _record_spike" in src


def test_the_step_is_skipped_not_just_the_backward():
    """
    GUARDS calling zero_grad() and then stepping anyway on stale gradients,
    which would apply the PREVIOUS batch's update twice.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("if spike_guard.should_skip"):]
    block = block[:block.index("if lr_scheduler")]
    assert "optimizer.step()" not in block.split("else:")[0], (
        "the skip path still steps the optimizer"
    )


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_the_guard_reads_the_loss_BEFORE_backward(bad):
    """
    Order matters: a non-finite loss must never reach clip_grad_norm_. Checked
    structurally, since the ordering is what makes the inf->nan trapdoor
    unreachable.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("if spike_guard.should_skip"):]
    block = block[:block.index("if lr_scheduler")]
    assert block.index("should_skip") < block.index("backward()")


# --------------------------------------------------------------------
# val excursions must break through the log throttle
# --------------------------------------------------------------------

def test_an_excursion_prints_even_when_nothing_saved():
    """
    With log_every_epoch=False the epoch row is gated on saved_this_epoch, so
    a run that spikes and never recovers goes SILENT at exactly the moment it
    matters.

    Measured on 128x128 stage 3a: the log's last line is epoch 3547 and the
    run stopped at 4047. The 500 epochs holding the spike that ENDED it
    produced no output -- the log reads as a clean descent followed by "Early
    stopping", and only the loss curve showed otherwise. That log is what a
    reader would normally use, and it was actively misleading.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "if log_every_epoch or saved_this_epoch or _excursion:" in src, (
        "the excursion does not break through the throttle"
    )
    assert "val EXCURSION" in src


def test_the_reference_ema_is_taken_BEFORE_the_update():
    """
    GUARDS comparing against an EMA this epoch's own excursion has already
    pulled up, which shrinks the ratio and can hide the very spike being
    tested for.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("_ema_before = tracker.val_ema"):]
    block = block[:block.index("_excursion = (")]
    assert "tracker.update(" in block, (
        "_ema_before must be captured before update(), and update() must follow it"
    )
    assert block.index("_ema_before") < block.index("tracker.update(")


def test_a_nonfinite_val_loss_counts_as_an_excursion():
    """inf/nan is not > factor*ema (comparisons with nan are False), so it
    needs its own clause or the worst possible epoch prints nothing."""
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("_excursion = ("):]
    block = block[:block.index("\n\n")]
    assert "not math.isfinite(val_loss)" in block


def test_the_excursion_check_is_disabled_by_zero():
    from training.train_lds import train_lds
    import inspect
    assert inspect.signature(train_lds).parameters["val_excursion_factor"].default == 3.0
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "val_excursion_factor > 0" in src


def test_no_excursion_is_reported_during_the_ema_warmup():
    """
    GUARDS dividing by a None EMA, and reporting against a reference that does
    not exist yet -- the first epochs legitimately move by large factors.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("_excursion = ("):]
    block = block[:block.index("\n\n")]
    assert "_ema_before is not None" in block


# --------------------------------------------------------------------
# the guard must not deadlock the run
# --------------------------------------------------------------------

def test_end_epoch_detects_every_batch_being_skipped():
    """
    THE FAILURE THE GUARD ITSELF CAUSED. Once the weights are broken every
    batch is an outlier, so every batch is skipped, so no gradient step is
    taken and the weights can never recover. The median does not adapt either,
    because skipped losses are deliberately not recorded -- right for a
    transient spike, fatal for a permanent one.

    Observed on 128x128 stage 3b: from epoch 2340, 7 of 7 batches skipped
    every epoch with val_loss frozen at exactly 231105.000, for hundreds of
    epochs. Stage 3a, WITHOUT the guard, had survived its own spike -- so the
    guard turned a recoverable crash into a permanent freeze.
    """
    guard = _warm(_SpikeGuard(factor=10.0))
    for _ in range(7):
        assert guard.should_skip(1e7)
    assert guard.end_epoch(7) is True
    assert guard.consecutive_total_skip_epochs == 1


def test_a_partial_skip_is_not_a_deadlock():
    """GUARDS rolling back on an ordinary transient, where some batches still
    step and the model can recover on its own."""
    guard = _warm(_SpikeGuard(factor=10.0))
    guard.should_skip(1e7)
    assert not guard.should_skip(1.0)
    assert guard.end_epoch(7) is False
    assert guard.consecutive_total_skip_epochs == 0


def test_the_consecutive_counter_resets_on_any_progress():
    guard = _warm(_SpikeGuard(factor=10.0))
    for _ in range(2):
        for _ in range(7):
            guard.should_skip(1e7)
        guard.end_epoch(7)
    assert guard.consecutive_total_skip_epochs == 2
    assert not guard.should_skip(1.0)
    guard.end_epoch(7)
    assert guard.consecutive_total_skip_epochs == 0


def test_forget_history_clears_the_median_after_a_rollback():
    """
    After restoring older weights the old median describes a model that no
    longer exists; keeping it would re-trip the guard on the first step back
    and deadlock again immediately.
    """
    guard = _warm(_SpikeGuard(factor=10.0), value=5.0)
    assert not math.isnan(guard.median())
    guard.forget_history()
    assert math.isnan(guard.median())
    assert guard.consecutive_total_skip_epochs == 0


def test_the_trainer_rolls_back_rather_than_freezing():
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "spike_guard.end_epoch(" in src and "_n_train_batches" in src
    assert "consecutive_total_skip_epochs >= spike_deadlock_epochs" in src
    assert 'f_theta.load_state_dict(_restored["model_state"])' in src


def test_the_optimizer_is_rebuilt_on_rollback():
    """
    GUARDS carrying Adam's moment estimates across a rollback. They were built
    from the very gradients that caused the divergence, so the first step back
    would re-apply the damage.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("_n_rollbacks += 1"):]
    block = block[:block.index("print(")]
    assert "optimizer = torch.optim.Adam" in block
    assert "spike_guard.forget_history()" in block


def test_rollbacks_are_bounded_and_then_it_STOPS():
    """
    GUARDS an infinite rollback loop: restore, diverge, restore... burning the
    whole epoch budget while looking busy.

    But it must STOP, not raise, when a checkpoint exists. Rolling back three
    times and diverging again is a real training OUTCOME, not a program error,
    and the best checkpoint on disk is exactly what an ordinary early stop
    would have returned. Raising propagated out of run_lds_stage and killed
    the whole params file -- stages 4 and 5 never ran, with a perfectly good
    val_loss=1.074618 checkpoint sitting on disk.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "_n_rollbacks >= max_spike_rollbacks" in src
    block = src[src.index("_have_checkpoint = Path(checkpoint_path).exists()"):]
    block = block[:block.index("_n_rollbacks += 1")]
    assert "if _have_checkpoint:" in block and "break" in block, (
        "an exhausted rollback budget must STOP, keeping the best checkpoint"
    )
    assert "raise RuntimeError(" in block, (
        "with NO checkpoint there is nothing to keep, so raising is right"
    )


def test_the_advice_is_not_backwards():
    """
    The original message said "Lower lr or spike_skip_factor". LOWERING
    spike_skip_factor makes the guard skip MORE batches -- exactly the
    deadlock being reported -- so following the advice would make it worse.

    lr is the lever; the guard reports the symptom, it does not cause it.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "LOWER lr" in src
    assert "RAISING spike_skip_factor" in src and "lets the damaging batches through" in src
    assert "or spike_skip_factor " not in src, "the backwards advice is back"


def test_the_default_factor_is_tightened_to_ten():
    """
    50 was calibrated for a many-batch epoch. Stage 3 has ~7 batches, and the
    batches that destroyed the model at epoch 2339 were large but under 50x
    the median -- so they were taken. 10 would have skipped all seven and left
    the weights intact.
    """
    import inspect

    from training.train_lds import train_lds
    assert inspect.signature(train_lds).parameters["spike_skip_factor"].default == 10.0


# --------------------------------------------------------------------
# the GRADIENT guard: what the loss guard structurally cannot see
# --------------------------------------------------------------------

def test_the_gradient_norm_is_guarded_separately():
    """
    On 128x128 stage 3a the batch that destroyed the model had an ORDINARY
    loss. With a median of 0.797 and spike_skip_factor already at 10, nothing
    above 8.0 was ever seen -- and the guard correctly did not fire -- yet the
    weights were wrecked inside that epoch (train 0.797 -> 4.098, val
    0.768 -> 4.308, both in one step).

    The loss is not the quantity that moves the weights. The gradient is.

    Why an ordinary batch can do it: f_theta enters as f*dt^2/2, and at dt=125
    that is a factor of 7800, so a 1e-3 change in f_theta's output is a change
    of ~8 in the prediction. grad_clip bounds the NORM, then Adam re-normalises
    per-parameter, so each step still moves each weight by ~lr regardless.
    """
    import inspect

    from training.train_lds import train_lds
    assert "grad_spike_factor" in inspect.signature(train_lds).parameters
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "grad_guard.should_skip(float(_gnorm))" in src


def test_the_guard_uses_the_PRE_clip_norm():
    """
    clip_grad_norm_ returns the norm BEFORE scaling, so the guard is free --
    no second pass over the parameters. Post-clip it would be useless: every
    norm would read exactly grad_clip and no outlier could ever appear.
    """
    import torch

    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p.grad = torch.tensor([3.0, 4.0])
    returned = float(torch.nn.utils.clip_grad_norm_([p], 1.0))
    assert returned == pytest.approx(5.0), "torch no longer returns the pre-clip norm"
    assert float(p.grad.norm()) == pytest.approx(1.0), "and it does still clip"

    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("_gnorm = torch.nn.utils.clip_grad_norm_"):]
    block = block[:block.index("optimizer.step()")]
    assert "grad_guard.should_skip" in block, "the guard must read the returned norm"


def test_the_norm_is_still_measured_when_clipping_is_off():
    """GUARDS skipping the norm computation when grad_clip=0: the guard would
    silently stop working for anyone who disables clipping."""
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert 'grad_clip if grad_clip > 0 else float("inf")' in src


def test_the_two_guards_keep_separate_histories():
    """
    Gradient norms and losses live on different scales; one shared median
    would be meaningless for both.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "grad_guard = _SpikeGuard(grad_spike_factor)" in src
    assert "spike_guard = _SpikeGuard(spike_skip_factor)" in src


def test_deadlock_counts_skips_from_BOTH_guards():
    """
    A batch is skipped by at most one guard -- a loss-skipped batch never
    reaches backward, so its gradient is never inspected -- so the counts add.
    The deadlock is "no step was taken at all", whichever guard prevented it;
    counting only one would miss a freeze caused by the other.
    """
    guard = _warm(_SpikeGuard(factor=10.0))
    for _ in range(3):
        guard.should_skip(1e7)
    assert guard.end_epoch(7, extra_skipped=4) is True, (
        "3 loss-skips + 4 grad-skips = 7 of 7, but the deadlock was not detected"
    )

    # And the TRAINER must actually pass the other guard's count. Testing the
    # method alone left this unchecked: dropping extra_skipped from the call
    # site kept every behavioural test green -- verified.
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "extra_skipped=grad_guard.n_skipped_this_epoch" in src
    assert "extra_skipped=spike_guard.n_skipped_this_epoch" in src


def test_extra_skipped_defaults_to_zero():
    """Existing single-guard callers must be unaffected."""
    guard = _warm(_SpikeGuard(factor=10.0))
    for _ in range(7):
        guard.should_skip(1e7)
    assert guard.end_epoch(7) is True


def test_grad_skips_are_reported_distinctly():
    """
    The two guards mean different things: a loss outlier is a bad WINDOW, a
    gradient outlier with an ordinary loss is ill-conditioning. Reporting them
    identically would erase the distinction that took a CSV to establish.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "GRADIENT NORM was a" in src
    assert "despite an ordinary loss" in src


def test_a_rollback_also_resets_the_CRITERION():
    """
    Restoring the weights is only half of a rollback.

    The diverged epochs push val_ema to ~1e5. From there a perfectly healthy
    restored model cannot clear best_val_loss for tens of epochs (46, at
    val_ema_decay=0.7, from 1e5 down to 1.2) while epochs_since_improvement
    keeps counting the whole time. That is exactly the poisoned-criterion
    failure that ended stage 3a -- and the rollback reintroduces it unless the
    tracker is reset too.

    The restored checkpoint's own val_loss is the honest new bar: it is this
    model's measured performance, so it seeds the reference ceiling the same
    way stage 1/2's reference row does.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("_n_rollbacks += 1"):]
    block = block[:block.index("_ema_before")]
    assert "tracker.reset_with_grace(" in block, "the criterion survives the rollback"
    assert 'reference_val_loss=float(_restored["val_loss"])' in block, (
        "the restored checkpoint's own val_loss must seed the new bar"
    )
    assert "epochs_since_improvement = 0" in block, (
        "patience keeps counting across a rollback"
    )


def test_the_rollback_grace_is_clamped_like_every_other():
    """A grace covering all remaining epochs would mean nothing saves after the
    rollback -- a missing checkpoint rather than a worse one."""
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("_n_rollbacks += 1"):]
    block = block[:block.index("_ema_before")]
    assert "clamp_grace_epochs(_grace, epochs - epoch + 1)" in block


def test_the_attribution_is_per_epoch_not_all_time():
    """
    REGRESSION. guard.worst kept a running maximum, so every report after the
    first repeated the same stale numbers. On a real 3b run
    "loss 4.323e+07 ... dt_max=125, theta=-0.2763" was printed identically for
    dozens of epochs -- including AFTER a rollback had restored different
    weights, so it described a batch that no longer existed.
    """
    guard = _warm(_SpikeGuard(factor=10.0))
    guard.should_skip(1e7)
    guard.worst = (1e7, 1.0, 125.0, -0.28)
    guard.end_epoch(7)
    assert guard.worst is None, "the attribution survived into the next epoch"

    # ...but it must be HANDED to the report, not discarded. end_epoch() runs
    # BEFORE the per-epoch report, so clearing `worst` outright left every skip
    # line with no dt_max/theta at all -- verified against a real log, where
    # the attribution vanished completely after this fix landed. Fixing the
    # STALE attribution must not delete it.
    assert guard.last_worst == (1e7, 1.0, 125.0, -0.28)
    guard.end_epoch(7)
    assert guard.last_worst is None, "last_worst outlived its own epoch"


def test_the_report_reads_last_worst_not_worst():
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "_wg = grad_guard.last_worst" in src
    assert "_w = spike_guard.last_worst" in src
    assert "= spike_guard.worst" not in src, (
        "the report reads the field end_epoch has already cleared"
    )


def test_the_excursion_message_does_not_assert_weight_damage():
    """
    The first version said "the weights moved somewhere bad". On a real 3b run
    that was wrong for most reports: train stayed at ~1.40 and both 1-step
    columns at ~0.62-0.82 -- entirely normal -- while the 2-step val hit 337.
    Over 5335 windows that is ONE window at ~1.8e6. The weights were fine; a
    few val windows diverge under the 2-step autonomous rollout.

    A diagnostic that names the wrong cause is worse than one that names none,
    because it sends the reader to the wrong fix.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "The weights moved somewhere bad" not in src
    assert "with train={train_loss:.4f}" in src, (
        "the message must show the train loss, which is what distinguishes the two causes"
    )
    assert "a few VAL" in src and "the weights moved" in src, (
        "both readings must be offered, keyed on the train column"
    )


# --------------------------------------------------------------------
# the rollback path, EXECUTED
# --------------------------------------------------------------------

def test_END_TO_END_a_deadlock_really_rolls_back_and_keeps_training(tmp_path, capsys,
                                                                     isolated_project_root):
    """
    The rollback path had never actually run -- every other test of it matches
    source text, which cannot catch a wrong attribute name, a load that fails,
    a `break` in the wrong loop, or a closure that keeps the discarded
    optimizer.

    Forces the deadlock deterministically: let a few epochs train and save
    normally, then make every batch look like an outlier. That is exactly the
    128x128 stage-3b sequence -- 7 of 7 skipped for 5 consecutive epochs --
    which without the rollback froze the run permanently with val_loss pinned
    at one value.
    """
    import sys

    sys.path.insert(0, str(_ROOT / "tests"))
    from test_train_lds import _cached_stage2_ancestor
    from training.train_lds import train_lds, _SpikeGuard

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path, stats0_weight=0.01)

    real = _SpikeGuard.should_skip
    state = {"epoch_calls": 0}

    def always_skip_after_a_while(self, value):
        state["epoch_calls"] += 1
        if state["epoch_calls"] > 12:       # a few real epochs first, so a checkpoint exists
            self.n_skipped += 1
            self.n_skipped_this_epoch += 1
            return True
        return real(self, value)

    _SpikeGuard.should_skip = always_skip_after_a_while
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = train_lds(
                size=32, base_path=base_path, ae_checkpoint_path=stage2_path,
                ae_stats_weight=0.01, epochs=30, batch_size=4, hidden_dim=8,
                n_hidden_layers=1, val_fraction=0.34, test_fraction=0.17,
                num_workers=0, n_rollout_steps=1, min_step=0, min_stdev_phi=None,
                encode_batch_size=4, ema_warmup_epochs=0,
                checkpoint_path=tmp_path / "s3.pt", device="cpu", seed=0,
                log_every_epoch=False, loss_curve_path=tmp_path / "c3.png",
                spike_deadlock_epochs=2, max_spike_rollbacks=2,
            )
        finally:
            _SpikeGuard.should_skip = real

    # A skipped batch takes no optimizer step, so it must not consume a step of
    # the lr schedule either -- during warmup that spends lr_warmup_steps
    # without training. torch says so itself, and this is how it surfaced: the
    # first end-to-end run of the rollback path emitted this warning on every
    # skipped batch, invisible to every source-matching test.
    sched_warnings = [w for w in caught
                       if "lr_scheduler.step()" in str(w.message)
                       and "before" in str(w.message)]
    assert not sched_warnings, (
        f"{len(sched_warnings)} batch(es) advanced the lr schedule without taking a "
        f"step: {sched_warnings[0].message}"
    )

    text = capsys.readouterr().out
    assert "ROLLED BACK to the best checkpoint" in text, (
        f"the deadlock never triggered a rollback:\n{text[-1500:]}"
    )
    # It must SURVIVE the rollback and return a loadable checkpoint, not raise
    # and not leave a path with no file at it.
    assert Path(out).exists()
    loaded = torch.load(out, map_location="cpu", weights_only=True)
    assert "model_state" in loaded and math.isfinite(float(loaded["val_loss"]))


def test_END_TO_END_an_exhausted_rollback_budget_stops_without_raising(tmp_path, capsys,
                                                                        isolated_project_root):
    """
    Exhausting the budget must STOP and keep the best checkpoint. Raising
    propagated out of run_lds_stage and killed the whole params file -- stages
    4 and 5 never ran, with a good checkpoint sitting on disk.
    """
    import sys

    sys.path.insert(0, str(_ROOT / "tests"))
    from test_train_lds import _cached_stage2_ancestor
    from training.train_lds import train_lds, _SpikeGuard

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path, stats0_weight=0.01)
    real = _SpikeGuard.should_skip
    state = {"n": 0}

    def skip_everything_after_a_while(self, value):
        state["n"] += 1
        if state["n"] > 12:
            self.n_skipped += 1
            self.n_skipped_this_epoch += 1
            return True
        return real(self, value)

    _SpikeGuard.should_skip = skip_everything_after_a_while
    try:
        out = train_lds(
            size=32, base_path=base_path, ae_checkpoint_path=stage2_path,
            ae_stats_weight=0.01, epochs=60, batch_size=4, hidden_dim=8,
            n_hidden_layers=1, val_fraction=0.34, test_fraction=0.17,
            num_workers=0, n_rollout_steps=1, min_step=0, min_stdev_phi=None,
            encode_batch_size=4, ema_warmup_epochs=0,
            checkpoint_path=tmp_path / "s3.pt", device="cpu", seed=0,
            log_every_epoch=False, loss_curve_path=tmp_path / "c3.png",
            spike_deadlock_epochs=2, max_spike_rollbacks=1,
        )
    finally:
        _SpikeGuard.should_skip = real

    text = capsys.readouterr().out
    assert "STOPPING at epoch" in text, f"did not stop cleanly:\n{text[-1500:]}"
    assert Path(out).exists(), "stopped without leaving a checkpoint"


def test_the_scheduler_is_rebuilt_WITH_the_optimizer():
    """
    LinearLR holds a reference to the optimizer it was constructed on, so
    rebuilding the optimizer alone orphans it: the old scheduler keeps driving
    the DISCARDED optimizer while the live one runs at a frozen lr. Verified
    against torch below.

    Source-matched deliberately, and this is the honest case for it: with
    lr_warmup_steps=20 the warmup is finished thousands of steps before any
    rollback, so both paths read the same lr and NO behavioural test can tell
    them apart in the configuration that actually runs. The bug is latent --
    it bites a decaying schedule, or a rollback during warmup.
    """
    import torch

    p = torch.nn.Parameter(torch.zeros(2))
    opt1 = torch.optim.Adam([p], lr=1e-3)
    sched = torch.optim.lr_scheduler.LinearLR(opt1, start_factor=0.1, total_iters=20)
    opt2 = torch.optim.Adam([p], lr=1e-3)
    # optimizer.step() BEFORE scheduler.step(), the order torch asks for.
    # Without it this test emitted the very "lr_scheduler.step() before
    # optimizer.step()" UserWarning it exists to reason about -- noise from a
    # test, in a suite where that exact warning is a real signal elsewhere
    # (see the end-to-end deadlock test, which asserts on its ABSENCE).
    p.grad = torch.zeros(2)
    for _ in range(5):
        opt1.step()
        sched.step()
    assert opt1.param_groups[0]["lr"] != opt2.param_groups[0]["lr"], (
        "torch no longer binds a scheduler to its optimizer -- this test's premise is gone"
    )

    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("_n_rollbacks += 1"):]
    block = block[:block.index("print(")]
    assert "torch.optim.Adam" in block, "the optimizer is not rebuilt"
    assert "LinearLR" in block, (
        "the optimizer is rebuilt but the scheduler is not -- it is now orphaned"
    )


def test_the_lr_schedule_does_not_advance_on_a_skipped_batch():
    """
    A skipped batch takes no optimizer step, so it must not consume a step of
    the lr schedule either -- during warmup that spends lr_warmup_steps without
    training.

    Surfaced by the first END-TO-END run of the rollback path: torch emitted
    "Detected call of lr_scheduler.step() before optimizer.step()" on every
    skipped batch. None of the source-matching tests could have seen it.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("if spike_guard.should_skip"):]
    block = block[:block.index("l_1step_scaled") if "l_1step_scaled" in block else len(block)]
    # the scheduler must sit INSIDE the branch that stepped
    stepped = block.index("optimizer.step()")
    assert "lr_scheduler.step()" in block[stepped:], (
        "the scheduler no longer follows optimizer.step()"
    )
    skip_branch = block[:block.index("else:")]
    assert "lr_scheduler.step()" not in skip_branch, (
        "the schedule advances on a skipped batch"
    )
