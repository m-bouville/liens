"""Shared spike-fighting machinery for the training stages.

Extracted from train_lds when stages 4 and 5 became a SECOND CALLER -- the
only justification for extraction here, since line count alone is not one.

Stages 4/5 spike exactly like stage 3 and had no protection at all:

    stage 4: epoch 37 train 0.3647 -> epoch 38 train 302,890.5   (830,520x)
    stage 5: epoch 20 train 1.4757 -> epoch 21 train 2,838.0     (1,923x)

with val barely moving in both cases -- so the weights survived and a few
train batches had exploded. Those runs got lucky.
"""
import math
import statistics
from collections import deque

import torch


def _record_spike(guard: "_SpikeGuard", total, dt_window, theta) -> float | None:
    """Attribution for a skipped batch: what was IN it.

    "Cursed with sudden peaks" is not actionable; "the spikes are all
    dt near max_dt at low noise" is. Records the worst one seen so the epoch
    summary can name it, rather than printing per-batch and drowning the log.
    """
    try:
        loss_v = float(total.detach())
        dt_max = float(dt_window.detach().max())
        theta0 = float(theta.detach()[:, 0].mean()) if theta.numel() else float("nan")
    except Exception:
        return
    # Per-epoch, not all-time. Keeping the running maximum meant every report
    # after the first repeated the same stale numbers -- on a real run,
    # "loss 4.323e+07 ... dt_max=125, theta=-0.2763" was printed identically
    # for dozens of epochs, including AFTER a rollback had restored different
    # weights, so it described a batch that no longer existed.
    if guard.worst is None or loss_v > guard.worst[0]:
        # The band's OWN median, matching the threshold this batch was judged
        # against -- see _SpikeGuard.median.
        guard.worst = (loss_v, guard.median(difficulty_band(dt_max)), dt_max, theta0)


def difficulty_band(dt_max: float) -> int:
    """Which comparison population a batch belongs to, from its largest dt.

    The guard skips a batch whose loss exceeds a multiple of a running median.
    That works when batches are interchangeable samples of the population --
    a high loss then really does mean "something went wrong". Under
    cost-bucketed batching they are NOT interchangeable: the hard windows are
    deliberately grouped, so their batches have a systematically high loss and
    were skipped as a block, epoch after epoch. Measured on the first
    max_dt=1000 run that trained: 14-19 of ~37 batches skipped every epoch for
    48 epochs, every one attributed to dt_max=1000. The run was stable because
    the guard had quietly restored the max_dt=500 population, one batch at a
    time.

    Comparing a batch against batches of SIMILAR difficulty restores the
    original meaning: an outlier is an anomaly within its own band, not
    membership of the hard tail.

    log2 bands (factor-2 in dt) are coarse enough to keep each band's history
    populated and fine enough that loss level within a band is comparable --
    loss grows roughly as dt^0.7 here, so a factor-2 band spans ~1.6x, well
    inside the factor-10 threshold.

    NO-OP WITHOUT BUCKETING, which is what makes this safe to switch on: an
    unbucketed batch of ~1000 windows drawn from the whole population contains
    a near-maximum-dt window with near-certainty, so every batch reports the
    same dt_max and lands in one band -- exactly today's behaviour.
    """
    if not math.isfinite(dt_max) or dt_max <= 0:
        return 0
    return int(math.floor(math.log2(dt_max)))


class _SpikeGuard:
    """Skips an optimizer step whose loss is a catastrophic outlier.

    Both 128x128 stage-3 runs were ENDED by a single such spike: 3a's last
    save was epoch 3547 and it stopped at 4047, 3b's were 2294 and 2794 --
    in each case exactly one patience window, entirely spent after the spike,
    never recovering to the pre-spike best. Neither run had plateaued; the
    reported "convergence" was a crash.

    val_loss spiking alongside train confirms the WEIGHTS were damaged, not
    just one forward pass: val runs under no_grad with no z0 noise.

    Two distinct failure modes, both covered:

    * NON-FINITE loss. grad_clip does not save you here -- it makes it worse.
      clip_grad_norm_ computes clip_coef = max_norm / (total_norm + eps),
      which for an infinite norm is 0, and 0 * inf = nan. One infinite
      gradient turns the whole parameter vector to NaN in a single step,
      permanently. A merely large gradient clips correctly.

    * FINITE but enormous. Clipping bounds the step's norm but not its
      direction, so a wildly wrong batch still moves the weights somewhere
      the model needs hundreds of epochs to climb out of.

    Deliberately conservative: the threshold is a multiple of a running
    MEDIAN (not mean -- the mean is what the outliers corrupt), and the guard
    stays inactive until it has enough history to have a meaningful median.
    It is meant to catch catastrophes, not to smooth ordinary noise, because
    silently dropping hard batches would bias what the model learns.
    """

    def __init__(self, factor: float, window: int = 200, min_history: int = 50):
        self.factor = factor
        self.min_history = min_history
        self.window = window
        # One history PER DIFFICULTY BAND -- see difficulty_band. A single
        # shared deque made "is this batch an outlier?" mean "is this batch
        # harder than average?", which under bucketing is answered by the
        # sampler rather than by anything going wrong.
        self._recent: dict[int, deque[float]] = {}
        self.n_skipped = 0
        self.n_nonfinite = 0
        self.n_skipped_this_epoch = 0
        self.consecutive_total_skip_epochs = 0
        self.worst: tuple[float, float, float, float] | None = None   # loss, median, dt_max, theta0
        self.last_worst: tuple[float, float, float, float] | None = None

    def _band(self, band: int | None) -> deque:
        if band not in self._recent:
            self._recent[band] = deque(maxlen=self.window)
        return self._recent[band]

    def should_skip(self, loss_value: float, band: int | None = None,
                     loss_was_ordinary: bool = False) -> bool:
        """`band` defaults to None -- ONE shared history, i.e. the old
        behaviour -- so a caller that does not bucket is unaffected.

        `loss_was_ordinary` says the LOSS guard looked at this same batch and
        declined to skip it. It defaults to False, which is the conservative
        reading: never adapt unless the caller positively asserts the loss was
        fine. The loss guard itself must therefore leave it False -- it has no
        second opinion to appeal to.
        """
        if self.factor <= 0:
            return False
        if not math.isfinite(loss_value):
            self.n_skipped += 1
            self.n_nonfinite += 1
            # COUNTED toward this epoch's total, exactly like a finite outlier.
            # This line was missing, and the omission cost a 200-epoch run: at
            # max_dt=2000, ~19 of 23 batches per epoch had INFINITE gradient
            # norms, all skipped through this branch -- none counted -- so the
            # combined total end_epoch_pair saw was the loss guard's 4-8, never
            # 23, and the all-skipped deadlock could not fire. The run burned
            # 200 epochs x 23 full forward+backward passes taking not one
            # optimizer step, then exited claiming "no improvement", when
            # spike_deadlock_epochs=5 should have stopped it at epoch 5.
            #
            # The failure was invisible precisely because non-finite is the
            # MOST skipped-looking skip there is: the explicit isfinite check
            # made the batch obviously handled, and nothing suggested it was
            # handled off the books.
            self.n_skipped_this_epoch += 1
            return True
        history = self._band(band)
        if len(history) >= self.min_history:
            med = statistics.median(history)
            if med > 0 and loss_value > self.factor * med:
                self.n_skipped += 1
                self.n_skipped_this_epoch += 1
                # TWO FAILURE MODES, and they must not be conflated.
                #
                # (a) DIVERGENCE: the weights are moving somewhere bad. The
                #     LOSS is elevated too. Recording anything here would
                #     raise the bar until spikes read as normal and the guard
                #     silently stopped guarding -- which is why nothing was
                #     recorded originally.
                #
                # (b) A STRUCTURALLY LARGE BAND: an ORDINARY loss with a huge
                #     gradient. f_theta enters the update as f*dt^2/2, so at
                #     long dt the gradient carries a factor of thousands while
                #     the loss does not. These batches are not anomalies; they
                #     are what the top dt band looks like. Recording NOTHING
                #     freezes such a band out permanently, because its median
                #     can never move. Measured on stage 3b: ~3 of 10 batches
                #     skipped every epoch for hundreds of epochs, and a
                #     controlled comparison showed f_theta got WORSE at long
                #     dt over the run -- 0.1492 -> 0.1631 in decade 3 -- while
                #     improving nowhere.
                #
                # `loss_was_ordinary` is the discriminator, and it is exactly
                # what the caller already knows: the gradient guard only runs
                # at all when the loss guard declined to skip. An earlier
                # version used a magnitude cutoff instead, calibrated from one
                # example at 2.7x the threshold -- but the real skips run 33x
                # to 625x, so it almost never engaged. The distinction was
                # never about size.
                if loss_was_ordinary:
                    history.append(self.factor * med)
                return True
        history.append(loss_value)
        return False

    def history(self, band: int | None = None) -> deque:
        """The recorded values for `band` -- read-only view for diagnostics.

        Exists because `_recent` changed shape when the bands were introduced
        (a deque became a dict of deques), and several tests were reaching
        into it directly. An accessor keeps that internal.
        """
        return self._recent.get(band, deque())

    def median(self, band: int | None = None) -> float:
        """The median of `band`'s own history -- the value the threshold used.

        Reporting the pooled median instead would print a number the decision
        was never made against, which is worse than useless when the whole
        point of the bands is that they sit at different levels.
        """
        history = self._recent.get(band)
        return statistics.median(history) if history else float("nan")

    @staticmethod
    def median_display(value: float) -> str:
        """How a median renders in a skip report.

        nan here means EMPTY HISTORY -- non-finite values are never recorded,
        so the deque cannot contain one; it can only have nothing at all. On
        the max_dt=2000 run every batch reaching the gradient guard was
        non-finite for 200 straight epochs, so its history stayed empty and
        every report printed "vs median nan" -- which reads as a broken
        statistic and sent the diagnosis toward value-poisoning, when the
        honest statement was "this guard has never once seen a finite value".
        Worth saying THAT, since a guard with no finite history is itself a
        louder finding than any threshold it might have applied.
        """
        return (f"{value:.4g}" if math.isfinite(value)
                else "n/a -- NO finite value ever recorded")

    def end_epoch(self, n_batches: int, extra_skipped: int = 0) -> bool:
        """Returns True when EVERY batch this epoch was skipped.

        THE DEADLOCK THIS EXISTS FOR. Once the weights are broken, every batch
        is an outlier, so every batch is skipped, so no gradient step is ever
        taken and the weights can never recover. The median does not adapt
        either -- skipped losses are deliberately not recorded, which is right
        for a transient spike and fatal for a permanent one.

        Observed on 128x128 stage 3b: from epoch 2340 onward, 7 of 7 batches
        skipped every single epoch, val_loss frozen at exactly 231105.000, for
        hundreds of epochs. The guard turned a crash the run had previously
        SURVIVED (3a recovered from its own spike) into a permanent freeze.
        """
        # extra_skipped: batches the OTHER guard skipped. A batch is skipped
        # by at most one of them (a loss-skipped batch never reaches backward,
        # so its gradient is never inspected), so the counts simply add. The
        # deadlock is "no step was taken at all", regardless of which guard
        # prevented it.
        all_skipped = n_batches > 0 and (self.n_skipped_this_epoch + extra_skipped) >= n_batches
        self.consecutive_total_skip_epochs = (
            self.consecutive_total_skip_epochs + 1 if all_skipped else 0)
        self.n_skipped_this_epoch = 0
        # HANDED OVER, not discarded. end_epoch() runs BEFORE the per-epoch
        # report, so clearing `worst` outright left the report with nothing --
        # every skip line lost its dt_max/theta attribution, which was the
        # whole point of recording it. Fixing the STALE attribution must not
        # remove it: last_worst is this epoch's, and only this epoch's.
        self.last_worst = self.worst
        self.worst = None
        return all_skipped

    def forget_history(self) -> None:
        """After a rollback the old median describes weights that no longer
        exist; keeping it would re-trip the guard immediately."""
        self._recent.clear()
        self.consecutive_total_skip_epochs = 0


def skip_report(epoch: int, loss_new: int, loss_worst, grad_new: int, grad_worst,
                 n_batches: int, verbose: bool, dt_label: str = "dt_max") -> str:
    """One line per epoch for both guards, verbose only the FIRST time.

    The original printed a three-line paragraph PER GUARD PER EPOCH, each
    re-explaining what the guard is for. Once skipping became routine -- 1 to 4
    batches of 13, every epoch, which is the guard working as intended -- that
    was six lines of unchanging exposition between consecutive epoch lines, and
    the loss curve became unreadable in the one regime where it most needed
    watching.

    So: the explanation is printed ONCE (it is genuinely worth reading once),
    and every later epoch gets a single compact line carrying only what varies
    -- the counts, the worst-vs-median ratio, and the dt band the outliers came
    from. The ratio replaces printing both numbers because it is what the
    threshold actually tests, and a bare count with no ratio cannot distinguish
    "10x, marginal" from "1e8x, catastrophic".

    Returns "" when nothing was skipped, so the caller prints nothing at all.
    """
    if not (loss_new or grad_new):
        return ""

    def _ratio(worst) -> str:
        if worst is None or not math.isfinite(worst[1]) or worst[1] <= 0:
            return "n/a"
        return f"{worst[0] / worst[1]:.3g}x"

    def _count(n: int) -> str:
        # "210/1951 (10.8%)" -- the raw count alone hides whether 210 is a
        # crisis or routine; the denominator and percent make it legible.
        return f"{n}/{n_batches} ({100 * n / n_batches:.1f}%)" if n_batches else str(n)

    if verbose:
        # The two skip kinds share a long boilerplate tail ("optimizer step NOT
        # taken, BatchNorm restored, model untouched"). When BOTH fire in an
        # epoch, printing two near-identical five-line blocks is mostly repeated
        # boilerplate -- so merge into one: both worst-clauses, the shared tail
        # once. Single-kind epochs are unchanged.
        def _worst_clause(kind: str, worst) -> str:
            if worst is None:
                return kind
            return (f"{kind} (worst: {worst[0]:.4g} vs median "
                    f"{_SpikeGuard.median_display(worst[1])}, {dt_label}={worst[2]:.4g}, "
                    f"mean theta[0]={worst[3]:.4g})")
        _tail = ("The optimizer step was NOT taken, and the BatchNorm running "
                 "statistics its forward pass moved were restored, so the model "
                 "is untouched.")
        parts = []
        if grad_new and loss_new:
            parts.append(
                f"  [epoch {epoch}: skipped {_count(grad_new)} batch(es) whose "
                f"{_worst_clause('gradient norm', grad_worst)} and {_count(loss_new)} "
                f"batch(es) whose {_worst_clause('loss', loss_worst)} were catastrophic "
                f"outliers. {_tail}]")
        elif grad_new:
            parts.append(
                f"  [epoch {epoch}: skipped {_count(grad_new)} batch(es) whose GRADIENT NORM was a "
                f"catastrophic outlier, despite an ordinary loss. This is the case the loss "
                f"guard cannot see."
                + ("" if grad_worst is None else
                   f" worst: grad_norm {grad_worst[0]:.4g} vs median "
                   f"{_SpikeGuard.median_display(grad_worst[1])}, {dt_label}={grad_worst[2]:.4g}, "
                   f"mean theta[0]={grad_worst[3]:.4g}")
                + "]")
        elif loss_new:
            parts.append(
                f"  [epoch {epoch}: skipped {_count(loss_new)} batch(es) whose loss was a catastrophic "
                f"outlier. {_tail}"
                + ("" if loss_worst is None else
                   f" worst: loss {loss_worst[0]:.4g} vs median "
                   f"{_SpikeGuard.median_display(loss_worst[1])}, {dt_label}={loss_worst[2]:.4g}, "
                   f"mean theta[0]={loss_worst[3]:.4g}")
                + "]")
        parts.append(
            "  (further skips are reported on one compact line per epoch; the counts and "
            "the worst/median ratio are what vary.)")
        return "\n".join(parts)

    bits = []
    if grad_new:
        bits.append(f"{grad_new} grad ({_ratio(grad_worst)})")
    if loss_new:
        bits.append(f"{loss_new} loss ({_ratio(loss_worst)})")
    worst = grad_worst if grad_new else loss_worst
    band = "" if worst is None else f" @ {dt_label}={worst[2]:.4g}"
    return f"  [epoch {epoch}: skipped {' + '.join(bits)} of {n_batches}{band}]"


class SkipReporter:
    """Decides which skips are worth a line, and digests the rest.

    A guard that skips 1 batch of 13 at 11.9x its band median is WORKING: the
    threshold is 10x, so that batch was barely over, and trimming it is the
    whole point. Printing a line for it every epoch trains the reader to skim
    past the line that matters -- which on this project is the difference
    between "a marginal tail is being trimmed" and "half the epoch is being
    excluded", and between 11.9x and the 1e7-1e8x ratios that marked genuine
    catastrophes.

    So a skip is REPORTED when it is one of:

      * BIG RATIO (>= notable_ratio x the band median). 10x is the threshold
        itself; 100x is an order of magnitude past it, i.e. not marginal.
      * BIG SHARE (>= notable_fraction of the epoch's batches). One batch of
        13 is noise; four is a third of the gradient signal gone.
      * NON-FINITE. Always: an inf/nan gradient is the case that turns the
        whole parameter vector to nan in one step if it gets through.

    Everything else accumulates silently and comes out as one digest line
    every `digest_every` epochs, so the information is kept without the noise.
    """

    def __init__(self, notable_ratio: float = 100.0, notable_fraction: float = 0.25,
                 digest_every: int = 25):
        self.notable_ratio = notable_ratio
        self.notable_fraction = notable_fraction
        self.digest_every = digest_every
        self._explained = False
        self._quiet_epochs = 0
        self._quiet_batches = 0
        self._quiet_total = 0
        self._quiet_worst = 0.0
        self._last_digest_epoch = 0

    @staticmethod
    def _ratio(worst) -> float:
        if worst is None or not math.isfinite(worst[1]) or worst[1] <= 0:
            return float("inf") if worst is not None else 0.0
        return worst[0] / worst[1]

    def epoch(self, epoch: int, loss_new: int, loss_worst, grad_new: int, grad_worst,
               n_batches: int, n_nonfinite_new: int = 0, dt_label: str = "dt_max") -> str:
        """The line(s) to print for this epoch -- possibly empty."""
        lines = []
        total = loss_new + grad_new
        if total:
            ratio = max(self._ratio(loss_worst), self._ratio(grad_worst))
            share = total / max(n_batches, 1)
            notable = (n_nonfinite_new > 0
                       or ratio >= self.notable_ratio
                       or share >= self.notable_fraction)
            if notable:
                lines.append(skip_report(epoch, loss_new, loss_worst, grad_new,
                                          grad_worst, n_batches,
                                          verbose=not self._explained,
                                          dt_label=dt_label))
                self._explained = True
            else:
                self._quiet_epochs += 1
                self._quiet_batches += total
                self._quiet_total += n_batches
                self._quiet_worst = max(self._quiet_worst, ratio)

        if (self.digest_every and self._quiet_epochs
                and epoch - self._last_digest_epoch >= self.digest_every):
            _pct = 100.0 * self._quiet_batches / self._quiet_total if self._quiet_total else 0.0
            lines.append(
                f"  [epochs {self._last_digest_epoch + 1}-{epoch}: {self._quiet_batches}"
                f"/{self._quiet_total} ({_pct:.2g}%) "
                f"further batch(es) skipped across {self._quiet_epochs} epoch(s), worst "
                f"{self._quiet_worst:.3g}x its band median -- below the reporting bar "
                f"({self.notable_ratio:g}x, or {100 * self.notable_fraction:g}% of an "
                f"epoch's batches), i.e. the guard trimming a marginal tail]")
            self._last_digest_epoch = epoch
            self._quiet_epochs = self._quiet_batches = self._quiet_total = 0
            self._quiet_worst = 0.0
        return "\n".join(x for x in lines if x)


def early_stop_message(epoch: int, patience: int, saved_this_run: bool,
                       longest_gap: int | None = None,
                       longest_gap_range: tuple[int, int] | None = None) -> str:
    """The line printed when patience runs out, honest about the two cases.

    "No improvement for N epochs" describes a run that climbed, plateaued and
    stopped. A run that NEVER saved is a different event wearing the same
    exit: the max_dt=2000 run skipped every batch of every epoch, took not
    one optimizer step, and left with "no improvement for 200 epochs" -- a
    sentence that sent the reader toward convergence when the story was
    total paralysis. The distinction costs one boolean the trainers already
    track for the no-save guard.

    longest_gap / longest_gap_range: the longest stretch of epochs WITHOUT a
    save that the run recovered from before this final one, and the (first,
    last) epochs it spanned -- the raw figure to judge whether stopping now was
    expected (earlier gaps also near patience) or a one-off long draught (worth
    a larger patience next time). Left uninterpreted: the number and range say
    it.

    A pure function for the same reason deadlock_step_hint is one: inlined
    text can only be tested by matching source, and such tests break on
    correct changes while never checking what the message SAYS.
    """
    if saved_this_run:
        tail = ""
        if longest_gap is not None:
            _rng = (f" ({longest_gap_range[0]} to {longest_gap_range[1]})"
                    if longest_gap_range else "")
            tail = (f"\n  longest period without saves before that: "
                    f"{longest_gap} epochs{_rng}")
        return (f"Early stopping at epoch {epoch}: no improvement for "
                f"{patience} epochs{tail}")
    return (f"Early stopping at epoch {epoch}: NOTHING was ever saved -- this is "
            f"not a converged run, it is one that never improved on its starting "
            f"point (or never took a step at all; check the skip counts above). "
            f"The checkpoint on disk, if any, belongs to a previous run.")


def end_epoch_pair(spike_guard: _SpikeGuard, grad_guard: _SpikeGuard,
                    n_batches: int) -> bool:
    """Close BOTH guards' epochs, each seeing the other's skips.

    Extracted from the two trainers for the same reason the rest of this
    module was (a second caller), but also because the inline version had an
    ordering bug both sites shared: calling
    ``grad_guard.end_epoch(n, extra_skipped=spike_guard.n_skipped_this_epoch)``
    AFTER ``spike_guard.end_epoch(...)`` always passed 0, since end_epoch
    zeroes the per-epoch counter as it returns. The deadlock check happened to
    read only spike_guard's counter (which got the combined total first), so
    nothing visible broke -- but grad_guard.consecutive_total_skip_epochs
    silently undercounted, a trap for anyone who later reads it and trusts it.
    Capturing both counts BEFORE either reset is the entire fix.

    Returns spike_guard's own all-skipped verdict -- the one both trainers'
    deadlock checks actually use. The two verdicts are equal by construction
    (each guard judges the same combined total against the same n_batches),
    so returning one of them loses nothing.
    """
    spike_skips = spike_guard.n_skipped_this_epoch
    grad_skips = grad_guard.n_skipped_this_epoch
    deadlocked = spike_guard.end_epoch(n_batches, extra_skipped=grad_skips)
    grad_guard.end_epoch(n_batches, extra_skipped=spike_skips)
    return deadlocked


def snapshot_running_stats(*modules) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Clone every normalisation running buffer in `modules`.

    THE GUARD'S OWN CLAIM, made true. A skipped batch prints "the optimizer
    step was NOT taken, so the weights are untouched" -- true of PARAMETERS
    and false of BUFFERS: the forward pass has already run by the time the
    loss is known, and in train mode every BatchNorm has updated its
    running_mean/running_var/num_batches_tracked on the way through. There is
    no way to decide the skip earlier, because the loss IS the signal.

    Observed on stage 4: epochs 8 and 9 skipped 487 of 487 batches -- every
    parameter frozen -- and val_loss still moved, 575.70 -> 582.34. That is
    only possible through the buffers, and it means a deadlock was not the
    frozen, recoverable state the message described: the encoder kept drifting
    in the one direction nothing was checking, while the guard reported it as
    inert.

    Stage 3 is immune (its encoder is frozen and LatentDynamics is Linear +
    LeakyReLU, no normalisation at all), which is why this surfaced only once
    stages 4/5 gained the same guards.

    num_batches_tracked is included: with momentum=None it drives the
    cumulative average, so leaving it advanced by a skipped batch biases every
    later update.
    """
    saved = []
    for module in modules:
        if module is None:
            continue
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                for buf in (m.running_mean, m.running_var, m.num_batches_tracked):
                    if buf is not None:
                        saved.append((buf, buf.detach().clone()))
    return saved


def restore_running_stats(snapshot: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Undo the buffer updates a skipped batch's forward pass performed.

    copy_ rather than reassignment: the buffers are registered on their
    modules, so rebinding the name would leave the module holding the drifted
    tensor and silently do nothing.
    """
    for buf, saved in snapshot:
        buf.copy_(saved)
