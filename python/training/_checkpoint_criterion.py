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


def grace_epochs_for_ema(val_ema_decay: float) -> int:
    """The EMA's own effective averaging window, 1/(1-decay), as a grace
    period -- the right length for any reset where the criterion's MEANING
    changed and the EMA must re-fill with values measured under the new one.

    Not a free parameter, which is why it is derived rather than exposed: it
    follows from val_ema_decay, and a separately-settable grace could be set
    inconsistently with the decay it is supposed to track.

    max(2, ...), NOT max(1, ...): a single-epoch grace is mathematically
    IDENTICAL to no grace at all (see reset_with_grace's own docstring) --
    with one epoch there is no second value for the EMA to blend with, so
    best_val_loss at the moment grace ends is exactly that one epoch's raw
    value, lucky or not. val_ema_decay >= 1 has no finite window and falls
    back to the same floor.

    Extracted once there were THREE callers (stage 2's deriv-target switch,
    stage 4/5's rollout ramp completing, stage 3's non-comparable resume) --
    the same criterion the spike guard was extracted under. Callers still
    clamp with clamp_grace_epochs afterwards: that answers a different
    question (will any epoch be left able to save) from this one (how long
    until the EMA means anything).
    """
    return max(2, round(1 / (1 - val_ema_decay))) if val_ema_decay < 1 else 2


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
    # The ANCESTOR's val_loss under THIS run's objective, if known: a CEILING
    # on best_val_loss, so a resumed run can never save something worse than
    # what it started from.
    #
    # Without it best_val_loss starts at inf, so epoch 1 always saves. Observed
    # on a real stage-2 resume: the reference row read 3.9054 and epoch 1
    # saved at 4.0149 -- a worse checkpoint written over a better one, and only
    # recoverable because the backup had fired. Epochs 2 and 3 were worse still
    # (4.35, 4.80) before epoch 4 finally beat the ancestor.
    #
    # A ceiling rather than an initial value, because both the grace branch and
    # the just_left_warmup branch RESET best_val_loss and would discard a plain
    # assignment.
    #
    # Costs nothing to obtain where a reference row is already printed -- that
    # row IS a full validation pass, and its value was previously computed and
    # thrown away.
    reference_val_loss: float | None = None
    best_val_loss: float = field(default=float("inf"), init=False)
    # best RAW val_loss (min_valid) -- the second minimum the save gate needs.
    # A save requires BOTH the smoothed EMA and the raw val to be at new lows:
    # EMA-only kept saving as the EMA drifted down even after raw val bottomed
    # and started rising (overtraining), silently selecting an overtrained
    # checkpoint. Requiring raw val < min_valid stops saves the moment raw val
    # stops making new lows, regardless of where the EMA drifts.
    best_raw_val_loss: float = field(default=float("inf"), init=False)
    val_ema: float | None = field(default=None, init=False)
    in_grace_period: bool = field(default=False, init=False)
    _grace_remaining: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._grace_remaining = self.ema_warmup_epochs
        self.in_grace_period = self._grace_remaining > 0

    def reset_with_grace(self, grace_epochs: int,
                          reference_val_loss: float | None = None) -> None:
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
        # The OLD reference is discarded, not carried over: a reset happens
        # precisely because the objective changed, so a bar measured under the
        # superseded one is exactly what must not be reused. A caller that can
        # re-measure under the NEW objective passes the fresh value; one that
        # cannot passes nothing and gets the historical inf.
        self.reference_val_loss = reference_val_loss
        self.best_val_loss = (float("inf") if reference_val_loss is None
                               else reference_val_loss)
        self.best_raw_val_loss = (float("inf") if reference_val_loss is None
                                   else reference_val_loss)
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
            # min: a grace period must not RAISE the bar above the ancestor.
            # Its job is to stop a lucky early epoch planting a flag, not to
            # license saving something worse than the run started from.
            self.best_val_loss = (self.val_ema if self.reference_val_loss is None
                                   else min(self.val_ema, self.reference_val_loss))
            # best_raw_val_loss is NOT lowered by grace epochs -- it changes
            # only on a real save (below). Grace may only CAP it at the ancestor
            # (reference), never let a lucky grace-window raw val plant a bar.
            if self.reference_val_loss is not None:
                self.best_raw_val_loss = min(self.best_raw_val_loss, self.reference_val_loss)
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
            self.best_val_loss = (float("inf") if self.reference_val_loss is None
                                   else self.reference_val_loss)
            self.best_raw_val_loss = (float("inf") if self.reference_val_loss is None
                                       else self.reference_val_loss)

        # THE GATE: save iff BOTH the raw val and the EMA reach new lows this
        # epoch. The two running minima update every epoch (independent of the
        # save decision), so a save requires them to dip together -- an EMA dip
        # while raw val rises (overtraining onset) no longer saves.
        should_save = (criterion < self.best_val_loss) and (val_loss < self.best_raw_val_loss)
        if should_save:
            # Both bars advance together, only on a save -- so best_raw_val_loss
            # is the raw val of the LAST SAVED checkpoint, not a running min that
            # a low-val/high-EMA epoch could quietly lower.
            self.best_val_loss = criterion
            self.best_raw_val_loss = val_loss
        return criterion, should_save


class ComponentBestTracker:
    """
    The per-COMPONENT analogue of CheckpointCriterionTracker.best_val_loss
    -- that tracker holds the running-minimum of the scalar val criterion,
    frozen between saves; this holds a whole DICT of val-side component
    values (e.g. {"recon0": ..., "stats0": ..., "deriv": ...}), frozen at
    whatever they were on the epoch a checkpoint was last actually saved.

    Not each component's OWN running minimum -- that wouldn't correspond
    to any single real checkpoint (component A's best epoch and component
    B's best epoch are generally different epochs). This instead answers
    "what were ALL the components at, on the epoch we'd actually reload
    if training stopped now" -- the co-occurring values a genuine restore
    would give, which is what loss_component_scatter's own "best so far"
    curve is meant to show alongside train/valid.

    Used by train_stage1()/train_stage2()/train_refinement() to build the
    per-component history lists that feed loss_component_scatter (see
    utils/plots.py); NOT used by train_lds(), whose own loss isn't a sum
    of separately-weighted components in this sense.
    """

    def __init__(self) -> None:
        self._best: dict[str, float] | None = None

    def update(self, current_val_components: dict[str, float], saved_this_epoch: bool) -> dict[str, float]:
        """Call once per epoch, with THIS epoch's own val-side component
        values and whether a checkpoint was actually saved this epoch.
        Returns the (possibly just-updated) frozen snapshot to append to
        that epoch's history -- unconditionally, so every epoch's history
        list stays the same length as epoch_history itself, including
        epoch 1 before any save has ever happened (falls back to this
        epoch's own values, matching CheckpointCriterionTracker's own
        first-epoch behavior of taking whatever the first real value is)."""
        if saved_this_epoch or self._best is None:
            self._best = dict(current_val_components)
        return dict(self._best)


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


def save_checkpoint(path, *, model_states, provenance, epoch, val_loss,
                    val_loss_ema, test_dirs, on_saved=None) -> str:
    """Atomically write a stage checkpoint and return the "  -> saved at HH:MM"
    suffix for the epoch line.

    `model_states` and `provenance` are the stage-specific halves: the
    .state_dict()s keyed as the stage's reload expects, plus the config
    sub-dicts. The fields EVERY stage needs -- epoch, val_loss, val_loss_ema and
    the resolved test_dirs -- are added here so a stage cannot silently omit one
    (the class of the stage45_config gap, where a saved checkpoint was missing
    the very config that defined its objective). The on_saved hook (registry
    upsert, etc.) is run here too, wrapped so a failing hook announces and
    continues rather than losing an hours-long run.
    """
    import time
    atomic_torch_save({
        **model_states,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_loss_ema": val_loss_ema,
        "test_dirs": [str(Path(d).resolve()) for d in test_dirs],
        **provenance,
    }, path)
    suffix = f"  -> saved at {time.strftime('%H:%M')}"
    if on_saved is not None:
        try:
            on_saved(path, epoch)
        except Exception as e:
            # Bookkeeping must never kill training: a failed registry upsert
            # must announce and continue, never lose the run.
            print(f"  WARNING: on_checkpoint_saved failed "
                  f"({type(e).__name__}: {e}) -- continuing training")
    return suffix


def ramp_completion_grace(epoch, epochs, warmup_epochs, tracker, val_ema_decay):
    """When the LAST active weight-ramp completes at `epoch`, reset the save
    criterion with a grace period and return (completed_names, grace); else None.

    `warmup_epochs` maps weight name -> its warmup length; entries of 0 (no ramp)
    are ignored. The grace fires ONCE, at max(active warmups): until the last
    ramp finishes, some term is still warming in and val_loss describes a model
    trained on a different objective -- exactly what reset_with_grace exists for.
    The grace is clamped to leave at least one saveable epoch.

    The caller owns the presentation -- the grace message wording and the
    loss-curve event label differ per stage -- so this returns which ramp(s)
    completed and how large the grace is, and performs the reset itself. Shared
    because all three trainers ramp something and reset on completion; the
    max-of-active timing (fire on the LAST ramp, not each) is the subtle part
    that must agree across them.
    """
    active = {name: w for name, w in warmup_epochs.items() if w > 0}
    if not active or epoch != max(active.values()):
        return None
    grace = clamp_grace_epochs(grace_epochs_for_ema(val_ema_decay), epochs - epoch + 1)
    completed = [name for name, w in active.items() if w == epoch]
    tracker.reset_with_grace(grace)
    return completed, grace


def scale_balance_report(
    contributions: dict[str, float],
    raw: dict[str, float],
    weights: dict[str, float],
    scales: dict[str, float],
) -> str | None:
    """Diagnose a mis-scaled multi-term objective, or return None if balanced.

    A weighted objective is `sum_k weight_k * raw_k / scale_k`. When a *_scale
    is not the raw magnitude of its own component, the weights beside it stop
    meaning what they say and the stage silently fails to balance what it
    exists to balance. This inspects one epoch's VALIDATION contributions and
    returns the warning string (already formatted, ready to print) or None.

    TWO-SIDED, because both failures are the same defect:
      - DOMINANT: one term is >99% of the loss -- its scale is too SMALL, so it
        drowns the others (stage 3 freezes the encoder and rollout loss is
        ~1e-6; stage 4 unfreezes it and the same quantity is ~0.7, 6e5x larger,
        so a global rollout_scale=1e-6 makes stage 4 99.997% rollout).
      - STARVED: a term with a NONZERO weight is <1% of the loss -- its scale is
        too LARGE, so the term the stage exists to add has effectively left the
        objective (rollout_scale=100 against a converged raw of 0.04 dropped
        L_rollout to 0.46% while the warning, in its dominant-only original
        form, stayed silent).
    Dominant takes priority when both could fire. STARVED is keyed on a nonzero
    weight so a term the user deliberately switched off is never reported.

    Extracted here (rather than inline in each epoch loop) so it is unit-tested
    against its OUTPUT, and so train_refinement and train_stage2 share ONE copy
    -- the two had byte-identical inline blocks, exactly the duplication this
    module exists to prevent.
    """
    total = sum(abs(v) for v in contributions.values())
    if total <= 0:
        return None
    shares = {k: abs(v) / total for k, v in contributions.items()}
    raw_str = ", ".join(f"{k}={raw[k]:.3e}" for k in shares)
    suggest = ", ".join(f"{k}_scale~{raw[k]:.3g}" for k in shares if raw[k] > 0)

    dominant = [k for k, sh in shares.items() if sh > 0.99]
    starved = [k for k, sh in shares.items()
               if sh < 0.01 and weights.get(k, 0.0) != 0.0]

    if dominant:
        k = dominant[0]
        others = ", ".join(f"{n} {100 * shares[n]:.4f}%" for n in shares if n != k)
        return (f"\n  WARNING: '{k}' is {100 * shares[k]:.4f}% of the validation "
                f"loss ({others}). The *_scale values are calibrated for a "
                f"different stage -- each scale should be the RAW magnitude of its "
                f"own component here, so the weights beside them mean what they say. "
                f"Raw values this epoch: {raw_str}. Suggested: {suggest}\n")
    if starved:
        detail = ", ".join(
            f"'{k}' {100 * shares[k]:.4f}% (weight {weights[k]:g}, "
            f"scale {scales[k]:g})" for k in starved)
        return (f"\n  WARNING: {detail} of the validation loss, despite a nonzero "
                f"weight -- that term is effectively OUT of the objective, so this "
                f"stage is not balancing what it exists to balance. Its scale is far "
                f"above its own raw magnitude. Raw values this epoch: {raw_str}. "
                f"Suggested: {suggest}\n")
    return None
