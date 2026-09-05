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
from training._spike_guard import end_epoch_pair

import pathlib
from models.constants import N_THETA
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
    hist = list(guard.history())
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
    training on a filtered distribution. The message is produced by
    skip_report; here we call it and assert on the RETURNED text (the
    complementary empty-when-clean case is test_nothing_is_printed_when_nothing_was_skipped),
    then keep the one structural fact the output can't show: that train_lds
    actually drives the reporter.
    """
    from training._spike_guard import skip_report
    worst = (1.1e5, 0.63, 2.5e4, -0.09)
    report = skip_report(7, 3, worst, 0, None, 13, verbose=True)
    assert report != "", "a batch was skipped but the report was empty (silent drop)"
    assert "skipped" in report and "catastrophic" in report, (
        "the report must say a batch was skipped and why")
    # train_lds drives the reporter, which decides whether to call skip_report --
    # a wiring fact the report text cannot show, so it stays a source check.
    assert "_skip_reporter.report_epoch(" in source_without_comments(
        _ROOT / "training/train_lds.py")


def test_the_report_names_what_was_IN_the_batch():
    """
    "Cursed with sudden peaks" is not actionable; "the spikes are all dt near
    max_dt at low noise" is. dt_max and theta are what turn one into the other.
    Asserted on skip_report's OUTPUT, including that the dt label defaults to
    "dt_max" when not passed (u-mode passes du_max explicitly).
    """
    from training._spike_guard import skip_report
    # worst tuple: (worst_value, median, dt, theta[0])
    report = skip_report(31, 4, (4.4e8, 1.2e7, 1250.0, -0.15),
                         0, None, 13, verbose=True)   # no dt_label -> default
    assert "dt_max=1250" in report, "the report must name the dt of the batch (default label)"
    assert "theta[0]=-0.15" in report or "theta[0]=-0.1" in report, (
        "the report must name the batch's theta")
    # a u-mode caller passes its own label, which must replace the default
    u_report = skip_report(31, 4, (4.4e8, 1.2e7, 1250.0, -0.15),
                           0, None, 13, verbose=True, dt_label="du_max")
    assert "du_max=1250" in u_report and "dt_max" not in u_report
    # _record_spike (which builds the worst-tuple) lives in _spike_guard so both
    # trainers share it -- a structural fact, kept as a source check.
    assert "def _record_spike" in source_without_comments(_ROOT / "training/_spike_guard.py")


def test_the_step_is_skipped_not_just_the_backward():
    """
    GUARDS calling zero_grad() and then stepping anyway on stale gradients,
    which would apply the PREVIOUS batch's update twice.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("if _spike_guard.should_skip"):]
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
    block = src[src.index("if _spike_guard.should_skip"):]
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
    # The gate has since grown a STALL clause too, so matching the whole line
    # pinned the exact set of conditions and broke on an unrelated addition.
    # The property this test owns is that an excursion is in the gate at all.
    assert "or _excursion" in src, (
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
    assert "end_epoch_pair(_spike_guard, grad_guard, _n_train_batches)" in src
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
    assert "_spike_guard.forget_history()" in block


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
    block = src[src.index("_have_checkpoint = _saved_this_run and"):]
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
    # Matching the call PREFIX, not the whole line: the signature grew a
    # loss_was_ordinary argument and an exact match broke a test about a
    # different property.
    assert "grad_guard.should_skip(float(_gnorm), band=_band" in src


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
    assert "_spike_guard = _SpikeGuard(spike_skip_factor)" in src


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

    # And the TRAINERS must actually pass the other guard's count -- through
    # the shared handshake, which captures both counts BEFORE either reset
    # (see end_epoch_pair's own docstring for the inline ordering bug it
    # replaced). Site-enumerated: a third trainer gaining guards joins here.
    for trainer in ("training/train_lds.py", "training/train_refinement.py"):
        src = source_without_comments(_ROOT / trainer)
        assert "end_epoch_pair(_spike_guard, grad_guard, _n_train_batches)" in src, trainer
        assert "extra_skipped" not in src, (
            f"{trainer} calls end_epoch inline with extra_skipped -- the exact "
            f"capture-after-reset form end_epoch_pair replaced")


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
    Asserted on skip_report's OUTPUT: a grad-only skip and a loss-only skip
    produce visibly different text.
    """
    from training._spike_guard import skip_report
    gw = (6.0e6, 29.4, 2.5e4, -0.15)
    lw = (1.1e5, 0.63, 2.5e4, -0.09)
    grad_only = skip_report(3, 0, None, 5, gw, 48, verbose=True)
    loss_only = skip_report(3, 5, lw, 0, None, 48, verbose=True)
    assert "GRADIENT NORM was a" in grad_only, "a grad-only skip must name the gradient norm"
    assert "despite an ordinary loss" in grad_only, (
        "a grad outlier with an ordinary loss is the ill-conditioning case")
    assert "catastrophic" in loss_only and "despite an ordinary loss" not in loss_only, (
        "a loss outlier is a different event and must not borrow the grad wording")


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
    """
    `worst` is cleared by end_epoch; `last_worst` is the copy that survives it.
    Reading the cleared field would print stale or empty attributions -- so
    both guards' values must come from last_worst wherever the report is
    assembled.
    """
    # The report is assembled in SkipReporter.report_epoch now (both trainers
    # delegate to it), so the last_worst-not-worst property lives there.
    import inspect
    from training._spike_guard import SkipReporter
    re_src = inspect.getsource(SkipReporter.report_epoch)
    assert "loss_guard.last_worst" in re_src and "grad_guard.last_worst" in re_src
    assert ".worst" not in re_src.replace("last_worst", ""), (
        "report_epoch reads the field end_epoch has already cleared"
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

    def always_skip_after_a_while(self, value, band=None, loss_was_ordinary=False):
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
    # the lr schedule either -- during warmup that spends lr_warmup_epochs
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

    def skip_everything_after_a_while(self, value, band=None, loss_was_ordinary=False):
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
    lr_warmup_epochs=20 the warmup is finished thousands of steps before any
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
    the lr schedule either -- during warmup that spends lr_warmup_epochs without
    training.

    Surfaced by the first END-TO-END run of the rollback path: torch emitted
    "Detected call of lr_scheduler.step() before optimizer.step()" on every
    skipped batch. None of the source-matching tests could have seen it.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    block = src[src.index("if _spike_guard.should_skip"):]
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


# --------------------------------------------------------------------
# stages 4 and 5 get the same protection
# --------------------------------------------------------------------

def test_stage45_has_both_guards():
    """
    Stages 4/5 spike exactly like stage 3 and had NO protection:

        stage 4: epoch 37 train 0.3647 -> epoch 38 train 302,890.5   (830,520x)
        stage 5: epoch 20 train 1.4757 -> epoch 21 train 2,838.0     (1,923x)

    val barely moved in both, so the weights survived and a few train batches
    had exploded. Those runs got lucky -- the same event in stage 3 destroyed
    the model.
    """
    import inspect

    from training.train_refinement import train_refinement
    params = inspect.signature(train_refinement).parameters
    assert params["spike_skip_factor"].default == 10.0
    assert params["grad_spike_factor"].default == 10.0

    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "_spike_guard.should_skip(float(loss.detach()), band=_band)" in src
    # Matching the call PREFIX, not the whole line: the signature grew a
    # loss_was_ordinary argument and an exact match broke a test about a
    # different property.
    assert "grad_guard.should_skip(float(_gnorm), band=_band" in src


def test_stage45_reads_the_loss_BEFORE_backward():
    """A non-finite loss must never reach clip_grad_norm_, which turns inf
    into nan and poisons every parameter permanently."""
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    block = src[src.index("_spike_guard.should_skip(float(loss.detach()), band=_band)"):]
    block = block[:block.index("optimizer.step()")]
    assert block.index("should_skip") < block.index("backward()")


def test_stage45_uses_the_PRE_clip_norm_and_keeps_measuring_it_when_clipping_is_off():
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert 'grad_clip if grad_clip > 0 else float("inf")' in src


def test_stage45_reports_skips_and_stops_on_deadlock():
    """
    Silent skipping would be worse than the crash: the run would look healthy
    while training on a filtered distribution. And an all-skipped run must not
    freeze -- the stage-3b lesson, where the guard turned a survivable crash
    into a permanent one.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    # Skips are reported through the SAME shared digesting reporter stage 3
    # uses -- so stage 4/5 gets the merged single-block message when BOTH guards
    # fire (was two separate near-identical blocks) and the compact digest for
    # routine skips. The message strings themselves live in skip_report, tested
    # separately; here we assert stage 4/5 actually routes through it.
    assert "SkipReporter()" in src, "stage 4/5 does not instantiate the shared reporter"
    # routes through the reporter's report_epoch, which owns the running-total-
    # to-delta bookkeeping (tested in test_report_epoch_*); the trainer no longer
    # threads the four counters by hand.
    assert "_skip_reporter.report_epoch(" in src, "skips are computed but never reported"
    assert "_spike_guard.n_skipped - _spikes_reported" not in src, (
        "the hand-threaded delta bookkeeping should be gone -- report_epoch owns it"
    )
    assert "end_epoch_pair(_spike_guard" in src
    assert "consecutive_total_skip_epochs >= 5" in src
    assert "STOPPING at epoch" in src
    assert "LOWER lr" in src, "the advice must not be backwards"


def test_stage45_deadlock_counts_BOTH_guards():
    """
    BOTH directions, which the inline form got wrong: the second end_epoch
    call always received extra_skipped=0, because the first zeroes the
    per-epoch counter as it returns. The deadlock check read only
    _spike_guard's counter (correct, combined, computed first), so nothing
    visible broke -- grad_guard.consecutive_total_skip_epochs silently
    undercounted instead. Behavioral, since the source-matching version of
    this test passed against that bug -- verified.
    """
    spike, grad = _warm(_SpikeGuard(factor=10.0)), _warm(_SpikeGuard(factor=10.0))
    for _ in range(3):
        spike.should_skip(1e7)
    for _ in range(4):
        grad.should_skip(1e7)
    assert end_epoch_pair(spike, grad, 7) is True
    assert spike.consecutive_total_skip_epochs == 1
    assert grad.consecutive_total_skip_epochs == 1, (
        "grad_guard did not see _spike_guard's skips -- the counts were read "
        "after the reset instead of before it"
    )


def test_the_guard_lives_in_ONE_place():
    """
    Extracted to training/_spike_guard.py when stages 4/5 became a second
    caller -- the justification for extraction, since line count alone is not
    one. Two copies would drift, and the copy that drifts is the one nobody
    is looking at when the next spike lands.
    """
    shared = source_without_comments(_ROOT / "training/_spike_guard.py")
    assert "class _SpikeGuard" in shared
    for mod in ("training/train_lds.py", "training/train_refinement.py"):
        src = source_without_comments(_ROOT / mod)
        assert "from training._spike_guard import" in src, f"{mod} does not use the shared module"
        assert "class _SpikeGuard" not in src, f"{mod} has its own copy"


@pytest.mark.slow
def test_END_TO_END_stage45_guard_skips_and_reports(tmp_path, capsys,
                                                     isolated_project_root):
    """
    Every other stage-4/5 guard test matches SOURCE. That is what let an
    unwired helper, a wrong tuple index and a missing latent_cache_dir through
    earlier today: each part correct, the assembly untested.

    Forces the skip deterministically by monkeypatching should_skip, then
    checks the run SURVIVES it and says so -- the guard must protect the run,
    not end it.
    """
    import sys

    sys.path.insert(0, str(_ROOT / "tests"))
    from test_train_refinement import (
        _build_ae_checkpoint, _build_lds_checkpoint, _build_sweep,
    )
    from training._spike_guard import _SpikeGuard
    from training.train_refinement import train_refinement

    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_path, lds_path = tmp_path / "ae.pt", tmp_path / "lds.pt"
    _build_ae_checkpoint(ae_path, include_stats_head=True)
    _build_lds_checkpoint(lds_path)

    real = _SpikeGuard.should_skip
    state = {"n": 0}

    def skip_a_few(self, value, band=None, loss_was_ordinary=False):
        state["n"] += 1
        if 3 <= state["n"] <= 5:          # a handful, not all: no deadlock
            self.n_skipped += 1
            self.n_skipped_this_epoch += 1
            return True
        return real(self, value)

    _SpikeGuard.should_skip = skip_a_few
    try:
        out_path = train_refinement(
            base_path=base_path, ae_checkpoint_path=ae_path,
            lds_checkpoint_path=lds_path, freeze_decoder=True,
            rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.1,
            epochs=3, batch_size=4, n_rollout_steps=1,
            min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
            checkpoint_path=tmp_path / "s4.pt", device="cpu", log_every_epoch=False,
        )
    finally:
        _SpikeGuard.should_skip = real

    text = capsys.readouterr().out
    assert "catastrophic outlier" in text, (
        f"batches were skipped but never reported:\n{text[-1200:]}"
    )
    assert "The optimizer step was NOT taken" in text
    # and the run must SURVIVE: a guard that ends the run is worse than none
    assert Path(out_path).exists()
    loaded = torch.load(out_path, map_location="cpu", weights_only=True)
    assert math.isfinite(float(loaded["val_loss"]))


@pytest.mark.slow
def test_END_TO_END_stage45_deadlock_stops_without_raising(tmp_path, capsys,
                                                            isolated_project_root):
    """
    An all-skipped run must STOP cleanly, keeping the best checkpoint -- the
    stage-3b lesson, where raising killed the whole params file and stages 4/5
    never ran with a good checkpoint on disk.
    """
    import sys

    sys.path.insert(0, str(_ROOT / "tests"))
    from test_train_refinement import (
        _build_ae_checkpoint, _build_lds_checkpoint, _build_sweep,
    )
    from training._spike_guard import _SpikeGuard
    from training.train_refinement import train_refinement

    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_path, lds_path = tmp_path / "ae.pt", tmp_path / "lds.pt"
    _build_ae_checkpoint(ae_path, include_stats_head=True)
    _build_lds_checkpoint(lds_path)

    real = _SpikeGuard.should_skip
    state = {"n": 0}

    def skip_everything_after_a_while(self, value, band=None, loss_was_ordinary=False):
        state["n"] += 1
        if state["n"] > 6:                 # a couple of real epochs first
            self.n_skipped += 1
            self.n_skipped_this_epoch += 1
            return True
        return real(self, value)

    _SpikeGuard.should_skip = skip_everything_after_a_while
    try:
        out_path = train_refinement(
            base_path=base_path, ae_checkpoint_path=ae_path,
            lds_checkpoint_path=lds_path, freeze_decoder=True,
            rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.1,
            epochs=30, batch_size=4, n_rollout_steps=1,
            min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
            checkpoint_path=tmp_path / "s4.pt", device="cpu", log_every_epoch=False,
        )
    finally:
        _SpikeGuard.should_skip = real

    text = capsys.readouterr().out
    assert "STOPPING at epoch" in text, f"did not stop cleanly:\n{text[-1200:]}"
    assert "LOWER lr" in text, "the advice must not be backwards"
    assert Path(out_path).exists(), "stopped without leaving a checkpoint"


def test_the_rollback_requires_THIS_RUNS_own_save():
    """
    REPORTED BUG. A 3b run at max_dt=2000 diverged from epoch 1, saved
    nothing, and at epoch 49 announced

        ROLLED BACK to the best checkpoint (epoch 4124, val_loss=1.074618)

    -- the PREVIOUS session's best, trained at max_dt=200 with a different
    f_theta scale entirely. Every epoch after that restore was inf.

    `Path.exists()` is not the question being asked: with force=True the old
    file sits at checkpoint_path until the first save overwrites it, so "a
    checkpoint exists" was true from epoch 1 of a run that had produced none.

    Restoring a stranger's weights is worse than stopping -- it looks like
    recovery, produces a checkpoint that loads, and silently mixes two
    configurations.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "_have_checkpoint = _saved_this_run and Path(checkpoint_path).exists()" in src, (
        "the rollback still accepts any file at the path"
    )
    # and the flag must be set where the save ACTUALLY happens -- lds now
    # delegates the write to _checkpoint_criterion.save_checkpoint.
    block = src[src.index("if saved_this_epoch:"):]
    block = block[:block.index("save_checkpoint(")]
    assert "_saved_this_run = True" in block, (
        "the flag is not set in the branch that writes the checkpoint"
    )
    assert "_saved_this_run = False" in src, "the flag is never initialised"


def test_the_no_checkpoint_message_distinguishes_the_two_cases():
    """
    "no checkpoint exists" and "this run saved nothing, but someone else's
    file is sitting there" need different actions from the reader, and the
    second is the one that just cost a run.
    """
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "SAVED NOTHING" in src
    assert "belongs" in src and "previous run" in src


def test_the_deadlock_advice_names_the_knob_that_actually_sets_the_step():
    """
    BEHAVIORAL, by reading the advice rather than the source that produces it.

    The previous version asserted "n_substeps scales with" against
    train_lds.py, and broke the moment the text was legitimately branched (the
    branched wording spells it "n_substeps (currently 7) scales with max_dt").
    It failed on a correct change while never checking the thing that matters:
    whether the advice points at the parameter that actually sets delta_t.

    Under alpha it must NOT send the reader to n_substeps/max_dt -- neither
    does anything when alpha is in use, and a deadlock message is read at
    exactly the moment someone is deciding what to change.
    """
    from training.train_lds import deadlock_step_hint

    fixed = deadlock_step_hint(alpha=None, n_substeps=7, max_substeps=256, clamped=0)
    assert "n_substeps" in fixed and "max_dt" in fixed
    assert "alpha" in fixed, "the fixed-count advice should still offer alpha as the way out"

    adaptive = deadlock_step_hint(alpha=0.1, n_substeps=1, max_substeps=256, clamped=0)
    assert "LOWER alpha" in adaptive
    assert "0.1" in adaptive, "the advice must name the CURRENT value, not just the knob"
    assert "check_alpha" in adaptive, "calibration is not guesswork; point at the tool"
    assert "scales with max_dt" not in adaptive, (
        "the adaptive advice sends the reader to max_dt/n_substeps, which do not "
        "set delta_t when alpha is in use"
    )


def test_a_binding_clamp_changes_the_deadlock_diagnosis():
    """
    A clamped run is NOT an "alpha too loose" run: max_substeps overrode the
    criterion, so those transitions ran coarser than alpha asked for. Lowering
    alpha there tightens a criterion that was never what limited the step.
    """
    from training.train_lds import deadlock_step_hint

    clean = deadlock_step_hint(alpha=0.1, n_substeps=1, max_substeps=256, clamped=0)
    clamped = deadlock_step_hint(alpha=0.1, n_substeps=1, max_substeps=256, clamped=42)
    assert "max_substeps" not in clean, "no clamp bound; do not mention it"
    assert "max_substeps=256" in clamped and "42" in clamped
    assert "raise max_substeps before lowering alpha" in clamped


def test_END_TO_END_a_stale_checkpoint_is_NOT_restored(tmp_path, capsys,
                                                        isolated_project_root):
    """
    The behavioural half: plant a foreign checkpoint at the output path, make
    every batch skip from the start so the run saves nothing, and assert it
    STOPS rather than restoring the stranger.
    """
    import sys

    sys.path.insert(0, str(_ROOT / "tests"))
    from test_train_lds import _cached_stage2_ancestor
    from training.train_lds import train_lds, _SpikeGuard

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path, stats0_weight=0.01)
    common = dict(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path,
        ae_stats_weight=0.01, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, n_rollout_steps=1,
        min_step=0, min_stdev_phi=None, encode_batch_size=4, ema_warmup_epochs=0,
        device="cpu", seed=0, log_every_epoch=False,
    )
    # a REAL checkpoint from a different run, left at the path
    foreign = train_lds(epochs=1, checkpoint_path=tmp_path / "out.pt",
                         loss_curve_path=tmp_path / "a.png", **common)
    assert Path(foreign).exists()
    foreign_bytes = Path(foreign).read_bytes()

    # Skip every batch AND suppress every save. Both are needed: a run that
    # never steps can still "improve" on epoch 1 against an infinite bar and
    # save, which would make the rollback legitimate. The real 3b run saved
    # nothing, and that is the case under test.
    from training._checkpoint_criterion import CheckpointCriterionTracker

    real = _SpikeGuard.should_skip
    real_update = CheckpointCriterionTracker.update
    _SpikeGuard.should_skip = lambda self, value, band=None: (
        setattr(self, "n_skipped", self.n_skipped + 1) or
        setattr(self, "n_skipped_this_epoch", self.n_skipped_this_epoch + 1) or True
    )
    CheckpointCriterionTracker.update = lambda self, epoch, val: (
        real_update(self, epoch, val)[0], False)
    try:
        with pytest.raises(RuntimeError) as exc:
            train_lds(epochs=30, checkpoint_path=tmp_path / "out.pt",
                       loss_curve_path=tmp_path / "b.png",
                       spike_deadlock_epochs=2, max_spike_rollbacks=2, **common)
    finally:
        _SpikeGuard.should_skip = real
        CheckpointCriterionTracker.update = real_update

    assert "SAVED NOTHING" in str(exc.value), (
        f"it did not identify the stale-file case:\n{exc.value}"
    )
    assert "ROLLED BACK" not in capsys.readouterr().out, "it restored the foreign checkpoint"
    assert Path(foreign).read_bytes() == foreign_bytes, "the foreign checkpoint was overwritten"


# --------------------------------------------------------------------
# A skipped batch must leave the model UNCHANGED -- buffers included
# --------------------------------------------------------------------

def _bn_module():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
    )


def test_a_forward_pass_in_train_mode_moves_batchnorm_buffers():
    """
    THE PREMISE, stated first so the fix below is not tested against an
    assumption. If this ever stops being true the restore is dead weight --
    but while it is true, a skipped batch changes the model even though every
    parameter is frozen.
    """
    m = _bn_module()
    m.train()
    before = m[1].running_mean.detach().clone()
    m(torch.randn(8, 1, 6, 6))
    assert not torch.equal(m[1].running_mean, before), (
        "BatchNorm did not update its running stats in train mode -- the rest "
        "of this section is testing nothing"
    )


def test_restoring_undoes_exactly_what_a_skipped_forward_did():
    """
    Observed on stage 4: epochs 8 and 9 skipped 487 of 487 batches, so every
    parameter was frozen, and val_loss still moved 575.70 -> 582.34. Only the
    buffers could carry that, which means a deadlock was not the frozen,
    recoverable state the guard reported -- the encoder kept drifting in the
    one direction nothing was checking.
    """
    from training._spike_guard import restore_running_stats, snapshot_running_stats

    m = _bn_module()
    m.train()
    m(torch.randn(8, 1, 6, 6))          # some history, so the restore is not trivial
    mean_before = m[1].running_mean.detach().clone()
    var_before = m[1].running_var.detach().clone()
    count_before = m[1].num_batches_tracked.detach().clone()

    snapshot = snapshot_running_stats(m)
    m(torch.randn(8, 1, 6, 6) * 10 + 5)  # a "spike" batch, then skipped
    restore_running_stats(snapshot)

    assert torch.equal(m[1].running_mean, mean_before)
    assert torch.equal(m[1].running_var, var_before)
    assert torch.equal(m[1].num_batches_tracked, count_before), (
        "num_batches_tracked was not restored -- with momentum=None it drives "
        "the cumulative average, so every later update stays biased"
    )


def test_the_snapshot_covers_nested_and_multiple_modules():
    """ae and f_theta are passed together, and the encoder's BatchNorms are
    nested several levels down."""
    from training._spike_guard import restore_running_stats, snapshot_running_stats

    # NESTED, not chained: the encoder's BatchNorms sit several levels down
    # inside blocks, which is what .modules() has to walk. (Chaining two
    # 1-channel blocks would just be a shape error -- my first version of this
    # test did exactly that.)
    class _Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.ModuleList([_bn_module(), _bn_module()])

        def forward(self, x):
            return sum(m(x) for m in self.inner)

    a = _Nested()
    b = _bn_module()
    a.train()
    b.train()
    snapshot = snapshot_running_stats(a, b, None)   # None tolerated: f_theta may be absent
    assert len(snapshot) == 3 * 3, f"expected 3 buffers per BatchNorm x 3, got {len(snapshot)}"
    a(torch.randn(4, 1, 6, 6))
    b(torch.randn(4, 1, 6, 6))
    restore_running_stats(snapshot)
    for mod in (a.inner[0][1], a.inner[1][1], b[1]):
        assert float(mod.num_batches_tracked) == 0.0, (
            "a nested BatchNorm was missed by the snapshot walk"
        )


def test_a_module_without_batchnorm_snapshots_nothing():
    """LatentDynamics is Linear + LeakyReLU, so stage 3 pays nothing for this."""
    from training._spike_guard import snapshot_running_stats
    plain = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LeakyReLU())
    assert snapshot_running_stats(plain) == []


def test_both_skip_paths_restore_the_buffers():
    """
    SITE ENUMERATION over the two skip branches. The gradient path is the
    easier to forget -- its loss looked ordinary, so nothing about the batch
    suggests the model was touched -- and it ran the same forward.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert src.count("restore_running_stats(_bn_snapshot)") == 2, (
        "both the loss-skip and gradient-skip branches must restore; found "
        f"{src.count('restore_running_stats(_bn_snapshot)')}"
    )
    # and the snapshot must be taken BEFORE the forward that moves the buffers
    snap = src.index("_bn_snapshot = snapshot_running_stats")
    forward = src.index("loss, components = compute_stage45_loss")
    assert snap < forward, (
        "the snapshot is taken after the forward pass, so it records the "
        "already-drifted buffers and the restore is a no-op"
    )


# --------------------------------------------------------------------
# Non-finite skips must COUNT -- the 200-epoch max_dt=2000 incident
# --------------------------------------------------------------------

def test_nonfinite_skips_count_toward_the_epochs_total():
    """
    THE MISSING INCREMENT. A non-finite loss/gradient took the early-return
    branch, which skipped correctly but never touched n_skipped_this_epoch --
    so the most catastrophic kind of skip was the only kind that counted for
    nothing. On the max_dt=2000 run, ~19 of 23 batches per epoch were
    non-finite gradient skips: the combined total the deadlock check saw was
    4-8, never 23, for 200 straight epochs.
    """
    from training._spike_guard import _SpikeGuard
    g = _SpikeGuard(10.0)
    for v in (float("inf"), float("nan"), float("-inf")):
        assert g.should_skip(v)
    assert g.n_skipped_this_epoch == 3, (
        f"non-finite skips did not count toward the epoch total "
        f"(got {g.n_skipped_this_epoch}); the all-skipped deadlock can then "
        f"never fire on a run whose batches are non-finite"
    )
    assert g.n_nonfinite == 3


def test_the_max_dt_2000_deadlock_now_fires():
    """
    THE INCIDENT, replayed: 23 batches; each epoch ~19 non-finite gradient
    skips + the rest loss-outlier skips, split across the two guards exactly
    as end_epoch_pair sees them. Five such epochs must trip the deadlock
    counter on which the trainers' spike_deadlock_epochs check reads.
    The real run did this for TWO HUNDRED epochs without firing.
    """
    from training._spike_guard import _SpikeGuard, end_epoch_pair
    loss_g = _SpikeGuard(10.0, min_history=4)
    grad_g = _SpikeGuard(10.0)
    for v in (1.0, 1.1, 0.9, 1.05):     # a little finite history for the loss guard
        loss_g.should_skip(v)
    for epoch in range(5):
        for _ in range(4):               # loss outliers (finite, huge)
            assert loss_g.should_skip(1e6)
        for _ in range(19):              # infinite gradient norms
            assert grad_g.should_skip(float("inf"))
        deadlocked = end_epoch_pair(loss_g, grad_g, n_batches=23)
        assert deadlocked, f"epoch {epoch}: 23/23 skipped but not flagged as total"
    assert loss_g.consecutive_total_skip_epochs == 5


def test_a_partially_skipped_epoch_is_still_not_a_deadlock():
    """The counting fix must not make the guard trigger-happy: one surviving
    batch means a step was taken and the run can move."""
    from training._spike_guard import _SpikeGuard, end_epoch_pair
    loss_g = _SpikeGuard(10.0)
    grad_g = _SpikeGuard(10.0)
    for _ in range(22):
        grad_g.should_skip(float("inf"))
    assert not end_epoch_pair(loss_g, grad_g, n_batches=23)
    assert loss_g.consecutive_total_skip_epochs == 0


def test_nonfinite_values_are_never_recorded_into_the_history():
    """
    The complement of counting them: they must be COUNTED but not RECORDED,
    or one inf makes the median nan and the ratio threshold dead forever.
    """
    from training._spike_guard import _SpikeGuard
    g = _SpikeGuard(10.0, min_history=2)
    g.should_skip(1.0)
    g.should_skip(float("inf"))
    g.should_skip(1.2)
    # Check the HISTORY, not the median: my first version asserted on the
    # median of [1.0, inf, 1.2] -- whose median is the MIDDLE element, 1.2,
    # perfectly finite -- so the mutation that records the inf passed. An
    # odd-length window hides exactly one poison value; the even-length case
    # (mean of the two middle values) surfaces it. Inspecting the deque
    # asserts the actual invariant instead of one parity's symptom.
    assert all(math.isfinite(v) for v in g.history()), (
        f"a non-finite value entered the history: {list(g.history())}"
    )
    assert list(g.history()) == [1.0, 1.2]


# --------------------------------------------------------------------
# Honest reporting: empty history and never-saved exits
# --------------------------------------------------------------------

def test_an_empty_history_renders_as_words_not_nan():
    """
    "vs median nan" on the max_dt=2000 run read as a broken statistic and
    steered the diagnosis toward value-poisoning; the truth -- the guard had
    never once seen a finite value in 200 epochs -- was a louder finding than
    any threshold, and the report should say it in words.
    """
    from training._spike_guard import _SpikeGuard
    rendered = _SpikeGuard.median_display(float("nan"))
    assert "nan" not in rendered.lower().replace("no fin", "")
    assert "NO finite value" in rendered
    assert _SpikeGuard.median_display(3.25) == "3.25"


def test_the_reports_use_the_display_not_raw_formatting():
    """The skip-report formatting lives in ONE place now -- _spike_guard.skip_report,
    which BOTH trainers route through -- and it must render the median through
    median_display, or the nan reappears. The trainers must not raw-format a
    median themselves (they used to, inline; now they delegate)."""
    shared = source_without_comments(_ROOT / "training/_spike_guard.py")
    assert ("median {_SpikeGuard.median_display" in shared
            or "median_display(grad_worst" in shared
            or "median_display(loss_worst" in shared), (
        "_spike_guard.skip_report does not route the median through median_display"
    )
    for fname in ("training/_spike_guard.py", "training/train_refinement.py",
                  "training/train_lds.py"):
        src = source_without_comments(_ROOT / fname)
        assert "vs median {_w[1]:.4g}" not in src, f"{fname} still raw-formats a median"
        assert "vs median {_wg[1]:.4g}" not in src, f"{fname} still raw-formats a median"


def test_early_stop_distinguishes_never_saved_from_plateaued():
    """
    "No improvement for 200 epochs" describes convergence; the max_dt=2000 run
    used it to describe total paralysis. The two exits need different words,
    and the never-saved one must direct the reader at the skip counts and warn
    that any file on disk belongs to a previous run.
    """
    from training._spike_guard import early_stop_message
    plateaued = early_stop_message(2618, 500, saved_this_run=True)
    assert "no improvement for 500 epochs" in plateaued
    assert "NOTHING" not in plateaued

    paralyzed = early_stop_message(200, 200, saved_this_run=False)
    assert "NOTHING was ever saved" in paralyzed
    assert "previous run" in paralyzed
    assert "skip counts" in paralyzed
    assert "no improvement for 200 epochs" not in paralyzed


def test_both_trainers_route_skips_through_the_shared_reporter():
    """The merged skip message -- one block when BOTH guards fire, a compact
    digest for routine skips -- must reach stage 4/5 too, not just stage 3.
    Stage 4/5 used to hand-roll two separate blocks here, so a both-guards
    epoch printed two near-identical paragraphs (the bug this locks). Both
    trainers must instantiate SkipReporter and print its per-epoch line."""
    for fname in ("training/train_lds.py", "training/train_refinement.py"):
        src = source_without_comments(_ROOT / fname)
        assert "SkipReporter()" in src, (
            f"{fname} does not use the shared digesting skip reporter"
        )
        # via .report_epoch (delta bookkeeping owned by the reporter) or the
        # lower-level .epoch (train_lds, until it too is extracted) -- either
        # routes the merged single-block message through the shared reporter.
        assert "_skip_reporter.report_epoch(" in src or "_skip_reporter.epoch(" in src, (
            f"{fname} never calls the reporter -- skips go unreported or "
            f"un-merged"
        )
    # and stage 4/5 must NOT still carry the old hand-rolled two-block loop
    src45 = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "if _g.n_skipped != _seen:" not in src45, (
        "the old per-guard two-block printing is still present -- both guards "
        "firing will print two separate blocks instead of one merged one"
    )


def test_both_trainers_route_the_early_stop_through_the_function():
    """Wiring: the message can only stay honest if both sites call it with
    their live save flag."""
    for fname in ("training/train_lds.py", "training/train_refinement.py"):
        src = source_without_comments(_ROOT / fname)
        assert "early_stop_message(epoch, early_stopping_patience, _saved_this_run" in src, (
            f"{fname} does not pass its save flag to early_stop_message"
        )
        assert "longest_gap=longest_gap" in src, (
            f"{fname} does not pass the recovered-gap length to early_stop_message"
        )


# --------------------------------------------------------------------
# Per-band medians: a bucketed batch is judged against its own kind
# --------------------------------------------------------------------

def test_difficulty_band_groups_dt_by_factors_of_two():
    from training._spike_guard import difficulty_band
    # Boundaries are at POWERS OF TWO, not at round decimal values: 500 and
    # 999 are in different bands (512 separates them), which my first version
    # of this test asserted the other way round without checking.
    assert difficulty_band(600) == difficulty_band(1000)      # both inside 512-1024
    assert difficulty_band(511) != difficulty_band(512)
    assert difficulty_band(1024) == difficulty_band(512) + 1
    # degenerate inputs must not raise: dt can be 0 in a malformed window
    assert difficulty_band(0) == 0
    assert difficulty_band(float("nan")) == 0
    assert difficulty_band(float("inf")) == 0


def test_a_systematically_harder_band_is_not_skipped_as_a_block():
    """
    THE INCIDENT. Under cost-bucketed batching the hard windows share batches,
    so their loss is systematically high -- and against a POOLED median that
    reads as "catastrophic outlier" every time. Measured on the first
    max_dt=1000 run that trained: 14-19 of ~37 batches skipped every epoch for
    48 consecutive epochs, all attributed to dt_max=1000. The run looked
    stable because the guard had restored the max_dt=500 population by
    stealth, one batch at a time.

    With per-band medians the hard band is compared against itself, so an
    ordinary hard batch passes and only a real anomaly is skipped.
    """
    from training._spike_guard import _SpikeGuard, difficulty_band
    easy_band, hard_band = difficulty_band(100), difficulty_band(1000)

    pooled = _SpikeGuard(factor=10.0, min_history=20)
    banded = _SpikeGuard(factor=10.0, min_history=20)
    # Interleaved, as bucketing yields them: many easy batches at loss ~1,
    # some hard ones at ~40 -- 40x the easy median but utterly ordinary for
    # their own band.
    pooled_skips = banded_skips = 0
    for i in range(200):
        hard = (i % 3 == 0)
        value = 40.0 if hard else 1.0
        band = hard_band if hard else easy_band
        pooled_skips += pooled.should_skip(value)           # no band: one history
        banded_skips += banded.should_skip(value, band=band)
    assert pooled_skips > 30, (
        f"the pooled guard skipped only {pooled_skips} -- the fixture does not "
        f"reproduce the block-skipping this test is about"
    )
    assert banded_skips == 0, (
        f"the banded guard still skipped {banded_skips} ordinary hard batches"
    )


def test_a_real_outlier_within_a_hard_band_is_still_skipped():
    """
    The complement, and the thing that must NOT be lost: banding is meant to
    stop hard-but-normal batches being dropped, not to stop catastrophes in
    the hard band being caught.
    """
    from training._spike_guard import _SpikeGuard, difficulty_band
    band = difficulty_band(1000)
    g = _SpikeGuard(factor=10.0, min_history=20)
    for _ in range(50):
        assert not g.should_skip(40.0, band=band)
    assert g.should_skip(40_000.0, band=band), (
        "a 1000x outlier within its own band passed -- banding has disabled "
        "the guard rather than focusing it"
    )


def test_bands_do_not_contaminate_each_other():
    """A quiet band must not inherit a loud band's threshold, in either
    direction."""
    from training._spike_guard import _SpikeGuard
    g = _SpikeGuard(factor=10.0, min_history=10)
    for _ in range(20):
        g.should_skip(1000.0, band=9)      # loud band
    for _ in range(20):
        g.should_skip(1.0, band=3)         # quiet band
    assert g.median(9) == 1000.0 and g.median(3) == 1.0
    # 50 is ordinary for the loud band, a 50x outlier for the quiet one
    assert not g.should_skip(50.0, band=9)
    assert g.should_skip(50.0, band=3)


def test_the_reported_median_is_the_bands_own():
    """
    The report must name the number the decision was made against. Printing
    the pooled median would show a threshold that was never applied -- worse
    than useless when the bands sit at different levels by design.
    """
    from training._spike_guard import _SpikeGuard, _record_spike, difficulty_band
    g = _SpikeGuard(factor=10.0, min_history=5)
    for _ in range(10):
        g.should_skip(1.0, band=difficulty_band(100))
    for _ in range(10):
        g.should_skip(40.0, band=difficulty_band(1000))

    dt_window = torch.full((4, 2), 1000.0)
    theta = torch.zeros(4, N_THETA)
    _record_spike(g, torch.tensor(500.0), dt_window, theta)
    assert g.worst is not None
    assert g.worst[1] == 40.0, (
        f"reported median {g.worst[1]} is not the dt=1000 band's own (40.0)"
    )


def test_banding_is_a_no_op_without_bucketing():
    """
    SAFE TO SWITCH ON. An unbucketed batch of ~1000 windows drawn from the
    whole population contains a near-maximum-dt window with near-certainty,
    so every batch reports the same dt_max, lands in one band, and behaves
    exactly as it did before bands existed.
    """
    from training._spike_guard import _SpikeGuard, difficulty_band
    band = difficulty_band(1000)
    old_style = _SpikeGuard(factor=10.0, min_history=10)
    banded = _SpikeGuard(factor=10.0, min_history=10)
    values = [1.0, 1.2, 0.9, 5.0, 1.1, 300.0, 1.0, 0.8, 250.0, 1.3] * 8
    for v in values:
        a = old_style.should_skip(v)
        b = banded.should_skip(v, band=band)
        assert a == b, f"banding changed the decision for {v}: {a} vs {b}"
    assert old_style.n_skipped == banded.n_skipped


def test_forget_history_clears_every_band():
    """After a rollback every band's median describes weights that no longer
    exist -- clearing only the current one would leave the others re-tripping
    the guard immediately."""
    from training._spike_guard import _SpikeGuard
    g = _SpikeGuard(factor=10.0, min_history=2)
    for band in (3, 7, 9):
        for _ in range(5):
            g.should_skip(1.0, band=band)
    g.forget_history()
    for band in (3, 7, 9):
        assert math.isnan(g.median(band)), f"band {band} kept its history"


def test_both_trainers_pass_the_band():
    from conftest import source_without_comments
    for fname in ("training/train_lds.py", "training/train_refinement.py"):
        src = source_without_comments(_ROOT / fname)
        assert "_band = difficulty_band(float(dt_window.detach().max()))" in src, fname
        assert src.count("band=_band") == 2, (
            f"{fname}: both the loss guard and the gradient guard must be "
            f"banded, found {src.count('band=_band')}"
        )


# --------------------------------------------------------------------
# Compact skip reporting
# --------------------------------------------------------------------

def test_the_first_skip_explains_itself_and_later_ones_do_not():
    """
    The rationale is worth reading ONCE. Repeating it every epoch -- three
    lines per guard, six lines total, between consecutive epoch lines -- made
    the loss curve unreadable in exactly the regime that needed watching, once
    skipping became routine (1-4 batches of 13, every epoch, the guard working
    as intended).
    """
    from training._spike_guard import skip_report
    worst = (4.6e8, 4.8e6, 1250.0, -0.10)
    first = skip_report(9, 1, worst, 3, worst, 13, verbose=True)
    later = skip_report(10, 1, worst, 3, worst, 13, verbose=False)
    # First time spells out the full boilerplate (once, even when both guards
    # fire and the two reports are merged); later epochs collapse to one compact
    # line without it. That contrast is the point -- six boilerplate lines every
    # epoch is what made the loss curve unreadable.
    assert "The optimizer step was NOT taken" in first
    assert "The optimizer step was NOT taken" not in later
    assert len(later.splitlines()) == 1, later
    assert len(first.splitlines()) > len(later.splitlines())


def test_both_guards_firing_merge_into_one_verbose_block():
    """When BOTH the loss and gradient guards fire in an epoch, the verbose
    report is a SINGLE block -- both worst-clauses, the shared 'optimizer step
    NOT taken / BatchNorm restored' tail exactly once -- not two near-identical
    five-line messages. Grad-only and loss-only epochs keep their own wording."""
    from training._spike_guard import skip_report
    gw = (6.0e6, 29.4, 2.5e4, -0.15)
    lw = (1.1e5, 0.63, 2.5e4, -0.09)
    both = skip_report(2, 4, lw, 5, gw, 48, verbose=True)
    body = both.splitlines()[0]                      # the merged line (before the note)
    assert body.count("optimizer step was NOT taken") == 1, "boilerplate must appear once"
    assert "gradient norm" in body and "loss" in body     # both kinds named
    assert "6e+06" in body and "1.1e+05" in body          # both worst values kept
    # single-kind epochs unchanged
    assert "despite an ordinary loss" in skip_report(3, 0, None, 5, gw, 48, verbose=True)
    assert "whose loss was a catastrophic" in skip_report(3, 5, lw, 0, None, 48, verbose=True)


def test_the_compact_line_carries_what_varies():
    """
    Counts, the worst/median RATIO, and the dt band. The ratio rather than the
    two raw numbers because it is what the threshold tests -- a bare count
    cannot distinguish "10x, marginal" from "1e8x, catastrophic", and that
    distinction is what separated the block-skipping incident from genuine
    outliers.
    """
    from training._spike_guard import skip_report
    line = skip_report(31, 4, (4.4e8, 1.2e7, 1250.0, -0.15),
                        2, (4.6e8, 4.8e6, 1250.0, -0.10), 13, verbose=False)
    assert "2 grad" in line and "4 loss" in line
    assert "of 13" in line, "the batch count is what makes a skip count meaningful"
    assert "dt_max=1250" in line
    import re
    # A ratio, not a specific value -- asserting "96" pinned my own arithmetic
    # rather than the property, and broke on a fixture tweak.
    assert re.search(r"\d+(\.\d+)?x\)", line), f"no worst/median ratio in: {line}"


def test_nothing_is_printed_when_nothing_was_skipped():
    """A clean epoch must produce no line at all -- the caller prints the
    return value unconditionally."""
    from training._spike_guard import skip_report
    assert skip_report(12, 0, None, 0, None, 13, verbose=False) == ""
    assert skip_report(12, 0, None, 0, None, 13, verbose=True) == ""


def test_a_missing_or_unusable_median_does_not_break_the_ratio():
    """An empty history gives a nan median (see median_display); the compact
    line must degrade rather than print 'nanx'."""
    from training._spike_guard import skip_report
    line = skip_report(3, 0, None, 2, (1e6, float("nan"), 500.0, -0.2), 13,
                        verbose=False)
    assert "n/a" in line and "nan" not in line.lower().replace("n/a", "")


def test_train_lds_reports_once_then_compactly():
    """
    The once-then-compact behaviour moved INTO SkipReporter when notability
    gating was added -- the explanation now attaches to the first NOTABLE
    skip, not the first skip of any kind, so it cannot be checked from
    train_lds's source any more. Verified through the reporter instead.
    """
    from conftest import source_without_comments
    from training._spike_guard import SkipReporter
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "_skip_reporter.report_epoch(" in src

    r = SkipReporter()
    severe = (4.0e4, 1e2, 1250.0, -0.1)
    first = r.epoch(6, 0, None, 1, severe, 13)
    second = r.epoch(7, 0, None, 1, severe, 13)
    assert "loss guard cannot see" in first
    assert "loss guard cannot see" not in second
    assert len(second.splitlines()) == 1


# --------------------------------------------------------------------
# Notability: only report skips worth reading
# --------------------------------------------------------------------

def _marginal():
    """1 batch at 11.9x a 10x threshold -- the guard trimming a tail."""
    return (1.19e3, 1e2, 1250.0, -0.1)


def _severe():
    return (4.0e4, 1e2, 1250.0, -0.1)          # 400x


def test_a_marginal_single_skip_is_not_reported():
    """
    THE COMPLAINT, and it is right: 1 batch of 13 at 11.9x against a 10x
    threshold is the guard doing its job. A line for it every epoch trains the
    reader to skim past the line that matters -- and on this project that line
    is the difference between "a marginal tail is trimmed" and "half the epoch
    is excluded", or between 11.9x and the 1e7-1e8x of a real catastrophe.
    """
    from training._spike_guard import SkipReporter
    r = SkipReporter()
    assert r.epoch(7, 0, None, 1, _marginal(), 13) == ""
    assert r.epoch(8, 0, None, 1, _marginal(), 13) == ""


def test_a_large_ratio_is_reported_however_few_the_batches():
    """One batch two orders past the threshold is not a tail."""
    from training._spike_guard import SkipReporter
    r = SkipReporter()
    out = r.epoch(11, 0, None, 1, _severe(), 13)
    assert out and "epoch 11" in out


def test_a_large_share_of_the_epoch_is_reported_however_small_the_ratio():
    """Four marginal batches of 13 is a third of the gradient signal gone --
    the count matters even when each one is barely over."""
    from training._spike_guard import SkipReporter
    r = SkipReporter()
    assert r.epoch(9, 0, None, 4, _marginal(), 13) != ""


def test_non_finite_is_always_reported():
    """
    An inf gradient turns the whole parameter vector to nan in ONE step if it
    gets through -- grad_clip makes it worse, not better (clip_coef =
    max_norm/(inf+eps) = 0, and 0*inf = nan). It can never be routine.
    """
    from training._spike_guard import SkipReporter
    r = SkipReporter()
    out = r.epoch(4, 0, None, 1, _marginal(), 13, n_nonfinite_new=1)
    assert out != "", "a non-finite skip was suppressed as marginal"


def test_suppressed_skips_are_digested_not_discarded():
    """
    Silence must not mean lost. The digest states how many batches over how
    many epochs and the worst ratio seen, so a slow drift upward is still
    visible without a line per epoch.
    """
    from training._spike_guard import SkipReporter
    r = SkipReporter(digest_every=10)
    for ep in range(1, 10):
        assert r.epoch(ep, 0, None, 1, _marginal(), 13) == ""
    out = r.epoch(10, 0, None, 1, _marginal(), 13)
    assert "further batch(es) skipped" in out, out
    assert "10/" in out, f"digest lost some of the count: {out}"
    assert "11.9x" in out
    # and the counter resets, so the next digest is not cumulative
    for ep in range(11, 20):
        r.epoch(ep, 0, None, 1, _marginal(), 13)
    out2 = r.epoch(20, 0, None, 1, _marginal(), 13)
    assert "10/" in out2, out2


def test_the_explanation_appears_on_the_first_NOTABLE_skip():
    """
    Not on the first skip of any kind: if a marginal one came first, the
    explanation would be attached to the least interesting event and the
    genuinely notable one would get the compact form.
    """
    from training._spike_guard import SkipReporter
    r = SkipReporter()
    r.epoch(5, 0, None, 1, _marginal(), 13)          # suppressed
    out = r.epoch(6, 0, None, 1, _severe(), 13)
    assert "loss guard cannot see" in out, out
    later = r.epoch(7, 0, None, 1, _severe(), 13)
    assert "loss guard cannot see" not in later
    assert len(later.splitlines()) == 1


def test_train_lds_uses_the_reporter_and_feeds_it_nonfinite_counts():
    from conftest import source_without_comments
    src = source_without_comments(_ROOT / "training/train_lds.py")
    assert "_skip_reporter = SkipReporter()" in src
    assert "_skip_reporter.report_epoch(" in src, "train_lds must route skips through report_epoch"
    # report_epoch feeds the non-finite counts through to skip_report (so an inf
    # gradient is always notable, never suppressed as a marginal ratio); the
    # counting now lives in the shared method.
    import inspect
    from training._spike_guard import SkipReporter
    re_src = inspect.getsource(SkipReporter.report_epoch)
    assert "n_nonfinite" in re_src and "n_nonfinite_new=" in re_src
    assert "loss_guard.n_nonfinite" in re_src and "grad_guard.n_nonfinite" in re_src


def test_a_band_whose_batches_all_skip_does_not_freeze_out():
    """
    THE PERMANENT-EXCLUSION BUG. history.append sat only on the accept path,
    so a band whose every batch exceeded the threshold never updated its own
    median: the threshold froze, and the band was excluded from training for
    the rest of the run.

    Measured on stage 3b -- the same ~3 of 8 batches at the top dt skipped
    every epoch for hundreds of epochs, while the loss on those windows never
    improved because it never received a gradient. That population is not
    noise: the gradient carries a dt^2 factor, so the longest-dt batches are
    structurally the largest, and a fixed multiple of a stale median rejects
    them by construction.
    """
    from training._spike_guard import _SpikeGuard
    g = _SpikeGuard(factor=10.0, window=200, min_history=50)
    for _ in range(60):
        g.should_skip(1.0, band=3, loss_was_ordinary=True)

    # ORDINARY LOSS with a large gradient: the structural case. The real 3b
    # skips ran 33x to 625x their band median, so no magnitude cutoff would
    # have covered them -- the loss verdict is what distinguishes.
    skips = sum(g.should_skip(100.0, band=3, loss_was_ordinary=True)
                for _ in range(200))
    assert skips < 120, (
        f"skipped {skips}/200 batches of a band that has genuinely shifted -- "
        f"the median is not following, so the band is frozen out"
    )
    hist = sorted(g.history(3))
    assert hist[len(hist) // 2] > 10.0, (
        f"band median {hist[len(hist) // 2]} has not tracked the shift"
    )


def test_a_one_off_catastrophe_is_still_skipped_and_does_not_poison_the_median():
    """The case the guard exists for. Recording the THRESHOLD rather than the
    value is what keeps both properties: the band can follow a real shift,
    while a 1e6 spike contributes a bounded amount instead of a value that
    would raise the threshold forever."""
    from training._spike_guard import _SpikeGuard
    g = _SpikeGuard(factor=10.0, window=200, min_history=50)
    for _ in range(60):
        g.should_skip(1.0, band=3)

    assert g.should_skip(1e6, band=3) is True
    hist = sorted(g.history(3))
    assert hist[len(hist) // 2] == 1.0, (
        "the median moved on a single outlier, so one catastrophic batch "
        "raises the bar for every batch after it"
    )
    assert g.should_skip(1.0, band=3) is False


def test_the_recorded_value_is_bounded_by_the_threshold():
    """Not the raw value: appending it would put the band's median beyond
    anything real within `window` batches."""
    from training._spike_guard import _SpikeGuard
    g = _SpikeGuard(factor=10.0, window=200, min_history=50)
    for _ in range(60):
        g.should_skip(2.0, band=1, loss_was_ordinary=True)
    g.should_skip(1e9, band=1, loss_was_ordinary=True)
    assert max(g.history(1)) <= 10.0 * 2.0 + 1e-9, (
        f"recorded {max(g.history(1))} for a skipped batch; it must be capped "
        f"at factor x median"
    )


def test_a_DIVERGING_band_is_not_adapted_to():
    """
    The distinction the adaptation rests on. A band that has genuinely shifted
    sits a small multiple past a stale threshold; a diverging run sits orders
    beyond it. Measured: 2.7x for the real 3b skips, 20000x for the divergence
    the guard was built for. Adapting to the second would raise the bar until
    spikes read as normal and the guard silently stopped guarding.
    """
    from training._spike_guard import _SpikeGuard
    g = _SpikeGuard(factor=10.0, window=200, min_history=50)
    for _ in range(60):
        g.should_skip(1.0, band=3, loss_was_ordinary=True)
    # loss_was_ordinary=False (the default): the loss guard flagged this batch
    # too, which is what divergence looks like -- both quantities elevated.
    skips = sum(g.should_skip(1e6, band=3) for _ in range(200))
    assert skips == 200, f"only {skips}/200 diverging batches were skipped"
    hist = sorted(g.history(3))
    assert hist[len(hist) // 2] == 1.0, (
        f"median moved to {hist[len(hist) // 2]} on a diverging band -- the "
        f"guard is adapting itself out of existence"
    )


def test_both_stages_tell_the_gradient_guard_the_loss_was_ordinary():
    """
    THE WHOLE MECHANISM HANGS ON THIS ARGUMENT. Without it the default is
    False, nothing ever adapts, and the top-dt band is frozen out again --
    silently, since every test of the guard in isolation still passes. A
    mutation removing it from the call site did exactly that.

    The gradient guard is only REACHED when the loss guard declined to skip,
    so passing True there is a statement of fact about the control flow, not
    an assumption.
    """
    import pathlib

    from conftest import source_without_comments
    root = pathlib.Path(__file__).resolve().parent.parent
    for stage in ("training/train_lds.py", "training/train_refinement.py"):
        src = source_without_comments(root / stage)
        assert "grad_guard.should_skip" in src, f"{stage} has no gradient guard"
        call = src[src.index("grad_guard.should_skip"):]
        assert "loss_was_ordinary=True" in call[:200], (
            f"{stage} does not tell the gradient guard the loss was ordinary, "
            f"so its band can never adapt and long-dt batches are excluded "
            f"permanently"
        )


def test_the_loss_guard_itself_never_claims_an_ordinary_loss():
    """It has no second opinion to appeal to: for the loss guard, the value
    under test IS the loss. Passing True there would let a diverging run
    adapt its own threshold away."""
    import pathlib

    from conftest import source_without_comments
    root = pathlib.Path(__file__).resolve().parent.parent
    for stage in ("training/train_lds.py", "training/train_refinement.py"):
        src = source_without_comments(root / stage)
        call = src[src.index("_spike_guard.should_skip"):]
        assert "loss_was_ordinary" not in call[:200], (
            f"{stage}'s LOSS guard passes loss_was_ordinary; only the gradient "
            f"guard has an independent verdict to report"
        )


def test_report_label_is_du_max_in_u_mode_dt_max_otherwise():
    """In log10_t (u) mode the batch step is Delta-u, not physical dt, so the
    spike report must say du_max -- calling it dt_max mislabels every u-run's
    guard message (the reader can't tell Delta-u=0.079 from a physical dt).
    Default stays dt_max for t-mode."""
    from training._spike_guard import skip_report
    worst = (4.3e7, 1.0, 0.07918, -0.276)   # loss, median, step-max, theta0
    u_line = skip_report(120, 1, worst, 0, None, 13, verbose=False, dt_label="du_max")
    assert "du_max=0.07918" in u_line and "dt_max=" not in u_line, u_line
    t_line = skip_report(120, 1, worst, 0, None, 13, verbose=False)  # default
    assert "dt_max=0.07918" in t_line, t_line


def test_early_stop_message_reports_longest_recovered_gap():
    """The exit line answers 'one-off draught or chronic near-stops?': it names
    the longest no-save stretch the run RECOVERED from and the epochs it
    spanned -- number and range only, no editorialising."""
    from training._spike_guard import early_stop_message
    msg = early_stop_message(500, 500, saved_this_run=True,
                             longest_gap=120, longest_gap_range=(250, 370))
    assert "no improvement for 500 epochs" in msg
    assert "longest period without saves before that: 120 epochs (250 to 370)" in msg
    assert "->" not in msg                       # interpretation line removed by request
    assert "%" not in msg                        # no percentage by request


def test_early_stop_message_without_gap_info_is_the_plain_line():
    from training._spike_guard import early_stop_message
    assert early_stop_message(2618, 500, saved_this_run=True) == \
        "Early stopping at epoch 2618: no improvement for 500 epochs"


def test_early_stop_message_never_saved_case_unchanged_by_gap_args():
    from training._spike_guard import early_stop_message
    msg = early_stop_message(200, 200, saved_this_run=False,
                             longest_gap=50, longest_gap_range=(10, 60))
    assert "NOTHING was ever saved" in msg
    assert "longest period" not in msg           # gap clause is a saved-run detail


def test_report_epoch_diffs_running_totals_into_per_epoch_deltas():
    """report_epoch owns the running-total-to-delta bookkeeping the trainers used
    to thread by hand. Feeding cumulative guard counts across epochs must yield
    the per-epoch NEW counts, not the totals -- a double-count here would inflate
    every skip line."""
    from training._spike_guard import SkipReporter, _SpikeGuard

    class _G:  # minimal guard stand-in with the fields report_epoch reads
        def __init__(self): self.n_skipped = 0; self.n_nonfinite = 0; self.last_worst = None

    r = SkipReporter(notable_ratio=1.0, notable_fraction=0.0)  # report any skip
    loss_g, grad_g = _G(), _G()
    # epoch 1: 3 loss skips cumulative
    loss_g.n_skipped = 3; loss_g.last_worst = (1e5, 50.0, 1.0, -0.1)
    line1 = r.report_epoch(1, loss_g, grad_g, 10)
    assert "3" in line1
    # epoch 2: cumulative 5 -> delta must be 2, not 5
    loss_g.n_skipped = 5
    line2 = r.report_epoch(2, loss_g, grad_g, 10)
    assert "2" in line2 and " 5 " not in line2, f"double-counted: {line2!r}"


def test_report_epoch_handles_a_single_guard_stage():
    """Stage 2 has only the loss guard; report_epoch(grad_guard=None) must not
    crash and must report the loss skips alone."""
    from training._spike_guard import SkipReporter

    class _G:
        def __init__(self): self.n_skipped = 4; self.n_nonfinite = 0
        last_worst = (1e5, 80.0, 1.0, -0.1)
    r = SkipReporter(notable_ratio=1.0, notable_fraction=0.0)
    line = r.report_epoch(1, _G(), None, 10)
    assert "4" in line


def test_report_epoch_feeds_nonfinite_counts_so_they_are_always_notable():
    """A non-finite (inf/nan) gradient must ALWAYS get a line -- it turns the
    whole parameter vector to nan in one step if it slips through -- so
    report_epoch must feed the guards' n_nonfinite deltas to skip_report rather
    than let a single inf be judged as a marginal ratio. Coverage that moved out
    of train_lds when the bookkeeping became shared."""
    from training._spike_guard import SkipReporter

    class _G:
        def __init__(self): self.n_skipped = 1; self.n_nonfinite = 1
        last_worst = (float("inf"), float("inf"), 1250.0, -0.1)

    # notable_ratio huge, notable_fraction huge -> a finite skip would be
    # digested silently; the non-finite path must still force a line.
    r = SkipReporter(notable_ratio=1e9, notable_fraction=1.0)
    line = r.report_epoch(1, _G(), _G(), 1000)
    assert line != "", "a non-finite skip was suppressed as marginal"
    assert "non-finite" in line.lower() or "nonfinite" in line.lower() or "inf" in line.lower()
