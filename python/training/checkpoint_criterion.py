"""
Encapsulates the "when should a checkpoint be saved" state machine
shared by train_autoencoder(), train_stage2(), and train_lds()'s epoch
loops -- extracted into its own class specifically so it's unit-testable
in isolation, rather than duplicated inline three times. That
duplication is exactly how train_lds()'s extra warmup feature drifted
out of sync with the other two and developed a real bug (a noisy
warmup-era val_loss could permanently block every future save) that a
fast, automated test would have caught immediately, but was only found
via a confusing symptom after a real, hours-long training run -- fixed
here via reset_with_grace, the same shared mechanism now also used for
train_stage2()'s own deriv_target_centered mid-run switch, which turned
out to be exactly the same underlying problem reached a second way.
"""
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch


def clamp_grace_epochs(requested: int, epochs_remaining: int) -> int:
    """
    The largest usable grace period given how many update() calls will
    actually still happen: always leaves AT LEAST ONE epoch able to
    save, since a grace period covering every remaining epoch means NO
    checkpoint is ever written at all -- not "a slightly worse
    checkpoint", but a missing file, and every downstream consumer
    fails with a confusing FileNotFoundError far from the real cause.

    Found the hard way, twice, from one root cause:
      - train_lds()'s own epochs=0 ablation makes exactly ONE update()
        call (at epoch 0) and relies on it saving -- that call is the
        entire deliverable (an ephemeral stage-3-shaped wrapper, see
        ensure_lds_checkpoint). With ema_warmup_epochs defaulting to 5,
        an unclamped grace period swallowed it and produced no file.
      - train_stage2()'s own deriv_target_centered switch on a SHORT
        run (fewer remaining epochs than grace_epochs) has exactly the
        same shape. This was initially misread as a test-fixture
        problem and "fixed" by lengthening the test; it was really
        this bug, showing up early.

    epochs_remaining counts update() calls still to come INCLUDING the
    current one, so `epochs_remaining - 1` is how many can be spent on
    grace while still leaving one that can save.
    """
    return max(0, min(requested, epochs_remaining - 1))


@dataclass
class CheckpointCriterionTracker:
    """
    ema_warmup_epochs: the tracker starts in a GRACE period of this
    many epochs (see reset_with_grace's own docstring for the general
    mechanism -- this is just that same mechanism, applied once, up
    front, via __post_init__). 0 (train_autoencoder()/train_stage2()'s
    behavior) means no grace period at all -- the very first update()
    call is treated as ending an (empty) grace period immediately, so
    the EMA is seeded and comparisons begin from that same call.

    val_ema_decay: standard EMA, ema = decay*ema + (1-decay)*val_loss.
    """
    ema_warmup_epochs: int = 0
    val_ema_decay: float = 0.7
    best_val_loss: float = field(default=float("inf"), init=False)
    val_ema: float | None = field(default=None, init=False)
    in_grace_period: bool = field(default=False, init=False)
    _grace_remaining: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._grace_remaining = self.ema_warmup_epochs
        self.in_grace_period = self._grace_remaining > 0

    def reset_with_grace(self, grace_epochs: int) -> None:
        """
        Resets the tracker as if freshly constructed with
        ema_warmup_epochs=grace_epochs -- for a MID-RUN reset where
        the save criterion itself has fundamentally changed (e.g.
        train_stage2()'s own deriv_target_centered switching from a
        one-sided to a centered L_deriv target partway through a run:
        val_loss computed under the OLD target isn't a fair bar for
        the NEW target's own val_loss to clear, so best_val_loss must
        reset -- but NOT via the single-epoch cliff edge described
        below).

        The general problem this solves: naively doing
        `self.val_ema = None; self.best_val_loss = float("inf")` and
        letting update() run its own normal logic makes the VERY NEXT
        call treat that epoch's raw, unsmoothed val_loss as both the
        new EMA seed and the new best_val_loss -- if that one epoch
        happens to be lucky, every later, properly-smoothed EMA value
        struggles to beat it (smoothing pulls values toward an
        average, which rarely beats an extreme single-epoch low).
        This is the EXACT mechanism this class's own module docstring
        already documents happening for ema_warmup_epochs specifically
        (a lucky warmup-era val_loss blocking 95 further epochs) --
        that bug and the deriv_target_centered risk are the SAME
        problem, just reached via two different paths (one at
        construction, one mid-run), which is why this is a single,
        shared method rather than two separate, one-off fixes.

        The fix: for `grace_epochs` calls to update(), the EMA
        accumulates real history (so it's not wasted -- it's already
        meaningfully smoothed by the time grace ends, and val_ema/
        best_val_loss stay visible and honest throughout for anything
        watching them, e.g. a loss-curve plot -- not frozen at a
        placeholder like float("inf") that would need special-casing
        downstream), but should_save is unconditionally False --
        no single epoch in the grace window can plant a flag that
        later epochs must beat. Once grace ends, comparisons resume
        normally, already primed with `grace_epochs` epochs of real
        smoothing behind them.

        The trade-off, worth being explicit about rather than hiding:
        NO checkpoint can be saved during the grace window, even if
        training is now genuinely, obviously better than before the
        reset -- the previous checkpoint (from right before the reset)
        is the correct fallback for that window, since anything saved
        during it would be judged on evidence the EMA hasn't actually
        finished absorbing yet.
        """
        self.val_ema = None
        self.best_val_loss = float("inf")
        self._grace_remaining = grace_epochs
        self.in_grace_period = grace_epochs > 0

    def update(self, epoch: int, val_loss: float) -> tuple[float, bool]:
        """
        Call once per epoch, IN ORDER, with that epoch's val_loss.
        Returns (criterion, should_save) and updates internal state
        (self.val_ema, self.best_val_loss, self.in_grace_period)
        accordingly.
        """
        if self._grace_remaining > 0:
            self.val_ema = val_loss if self.val_ema is None else \
                self.val_ema_decay * self.val_ema + (1 - self.val_ema_decay) * val_loss
            # Tracked continuously (not left at a stale float("inf"))
            # specifically so a loss-curve plot watching best_val_loss
            # shows a real, continuous line through the grace window --
            # but this is NEVER what should_save is decided against
            # below; grace's whole point is that nothing is compared
            # against it while it's still this fresh.
            self.best_val_loss = self.val_ema
            self._grace_remaining -= 1
            self.in_grace_period = self._grace_remaining > 0
            return self.val_ema, False

        just_left_warmup = self.val_ema is None
        self.val_ema = val_loss if just_left_warmup else \
            self.val_ema_decay * self.val_ema + (1 - self.val_ema_decay) * val_loss
        criterion = self.val_ema
        if just_left_warmup:
            # ema_warmup_epochs=0 case ONLY (the tracker was constructed
            # with no grace period at all, so this is genuinely this
            # tracker's very first update() call ever) -- a real grace
            # period (above) already seeded best_val_loss from a
            # smoothed value when it ended, so just_left_warmup is
            # never True there.
            self.best_val_loss = float("inf")

        should_save = criterion < self.best_val_loss
        if should_save:
            self.best_val_loss = criterion
        return criterion, should_save


def atomic_torch_save(obj, path: Path) -> None:
    """
    torch.save(obj, path), but ATOMIC: any process reading `path`
    concurrently (e.g. a check_*.py diagnostic run against a
    checkpoint while training is still writing new "best" versions of
    it, hours into a long run) either sees the complete, PRIOR version
    of the file, or the complete, NEW version -- never a partially-
    written one.

    Real failure mode this fixes, reproduced directly (not
    hypothetical): torch.save's own default behavior writes its zip-
    format archive progressively, IN PLACE, at the destination path --
    a concurrent torch.load() landing mid-write sees a truncated
    archive and raises "PytorchStreamReader failed reading file
    data/N: file read failed", intermittently, depending on exactly
    when the read happens to land. Confirmed by directly racing a slow
    save against a concurrent read in a loop: this reproduces the
    identical error class within a handful of attempts.

    Achieved by writing to a temp file FIRST, in the SAME DIRECTORY as
    the real destination (critical: a rename across filesystems is a
    real copy, not atomic; same-directory guarantees same filesystem),
    then os.replace()-ing it into place -- os.replace (not os.rename)
    specifically, since only os.replace is documented as atomic AND
    guaranteed to overwrite an existing destination on BOTH POSIX and
    Windows (os.rename's own overwrite behavior on Windows is
    unspecified/can raise).

    On any failure before the replace completes, the temp file is
    cleaned up rather than left behind as accumulating garbage in the
    checkpoints directory -- the ORIGINAL file at `path`, if any, is
    never touched at all until the very last, atomic step.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)  # torch.save opens its own handle on the path; the mkstemp one was only to claim a unique name
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
