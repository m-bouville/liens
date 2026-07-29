"""
Encapsulates the "when should a checkpoint be saved" state machine
shared by train_autoencoder(), train_stage2(), and train_lds()'s epoch
loops -- extracted into its own class specifically so it's unit-testable
in isolation, rather than duplicated inline three times. That
duplication is exactly how train_lds()'s extra warmup feature drifted
out of sync with the other two and developed a real bug (a noisy
warmup-era val_loss could permanently block every future save) that a
fast, automated test would have caught immediately, but was only found
via a confusing symptom after a real, hours-long training run.
"""
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class CheckpointCriterionTracker:
    """
    ema_warmup_epochs: for the first N epochs, the raw val_loss is used
    directly as the save criterion (matching train_lds()'s original
    design intent -- LDS training can be wildly noisy in its first few
    epochs, before an EMA has enough history to be a stable estimate).
    0 (train_autoencoder()/train_stage2()'s behavior) means there's no
    separate warmup phase at all -- the EMA starts being the criterion
    from epoch 1, initialized to that epoch's own val_loss.

    val_ema_decay: standard EMA, ema = decay*ema + (1-decay)*val_loss.
    """
    ema_warmup_epochs: int = 0
    val_ema_decay: float = 0.7
    best_val_loss: float = field(default=float("inf"), init=False)
    val_ema: float | None = field(default=None, init=False)

    def update(self, epoch: int, val_loss: float) -> tuple[float, bool]:
        """
        Call once per epoch, IN ORDER, with that epoch's val_loss.
        Returns (criterion, should_save) and updates internal state
        (self.val_ema, self.best_val_loss) accordingly.
        """
        if epoch <= self.ema_warmup_epochs:
            criterion = val_loss
        else:
            just_left_warmup = self.val_ema is None
            self.val_ema = val_loss if just_left_warmup else \
                self.val_ema_decay * self.val_ema + (1 - self.val_ema_decay) * val_loss
            criterion = self.val_ema
            if just_left_warmup:
                # The criterion just changed from raw val_loss to a
                # smoothed running average -- fundamentally different
                # quantities, so best_val_loss (set, if at all, during
                # warmup) isn't a fair bar for the ema to clear. Without
                # this reset, a single noisy-but-lucky warmup epoch can
                # permanently outscore every later ema value for the
                # rest of the run -- observed directly in a real run: a
                # warmup epoch's val_loss=0.295 blocked every save for
                # 95 further epochs, even one reaching ema=0.886, simply
                # because 0.886 was never < 0.295.
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
