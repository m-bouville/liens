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
        guard.worst = (loss_v, guard.median(), dt_max, theta0)


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
        self._recent: deque[float] = deque(maxlen=window)
        self.n_skipped = 0
        self.n_nonfinite = 0
        self.n_skipped_this_epoch = 0
        self.consecutive_total_skip_epochs = 0
        self.worst: tuple[float, float, float, float] | None = None   # loss, median, dt_max, theta0
        self.last_worst: tuple[float, float, float, float] | None = None

    def should_skip(self, loss_value: float) -> bool:
        if self.factor <= 0:
            return False
        if not math.isfinite(loss_value):
            self.n_skipped += 1
            self.n_nonfinite += 1
            return True
        if len(self._recent) >= self.min_history:
            med = statistics.median(self._recent)
            if med > 0 and loss_value > self.factor * med:
                self.n_skipped += 1
                self.n_skipped_this_epoch += 1
                return True
        self._recent.append(loss_value)
        return False

    def median(self) -> float:
        return statistics.median(self._recent) if self._recent else float("nan")

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


