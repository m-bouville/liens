"""
Tees stdout/stderr to a per-stage log file in addition to the console.
Extracted from main.py during its split into orchestration/.
"""
import inspect
import sys
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path


class _Tee:
    """Writes to multiple streams at once (e.g. the real console AND a log file).

    CONVENTION: the FIRST stream is the live console; every other stream is a
    capture file. Chunks that begin with '\\r' are in-place progress updates
    (EpochProgress, the dataset encode/index bars, the interpolation counter):
    transient by definition -- on a terminal each overwrites the last, but in a
    log file every update would pile up as a permanent wall of repeated text.
    They are therefore written to the console ONLY. Everything informative is
    already emitted as a normal line as well (epoch summaries, the cache-hit
    count line, the check's own results), so the log loses nothing readable.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        streams = self.streams[:1] if data.startswith("\r") else self.streams
        for s in streams:
            s.write(data)
            # Flush EVERY write, not just at close. Without this the log
            # file sits in Python's ~8 KiB buffer and a killed run (Ctrl-C,
            # an IDE stop button, an OOM kill) discards it entirely -- which
            # is exactly when the log is most wanted, since a run that ends
            # normally could have been rerun anyway. A training log emits a
            # line every few minutes at most, so per-write flushing costs
            # nothing measurable; the console stream is line-buffered and
            # already effectively does this.
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def _log_to_file(log_path: Path):
    """
    Tees everything printed to stdout/stderr into log_path for the
    duration of this block, IN ADDITION TO the normal console output --
    so a stage's full progress log survives even if the console itself
    is later closed/lost (e.g. an IDE crash after a long training run).
    Uses try/finally so the log is properly flushed and stdout/stderr
    restored even if the wrapped code raises.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)
        sys.stderr = _Tee(original_stderr, log_file)
        try:
            yield
        finally:
            # Restore FIRST, then flush what the tee already wrote: a
            # KeyboardInterrupt unwinds through here, and leaving the tee
            # installed while flushing risks re-entering it.
            sys.stdout, sys.stderr = original_stdout, original_stderr
            log_file.flush()


# Infrastructure, not science: paths, callables, worker/device plumbing and
# logging switches. Excluded from the run-parameter block because none of them
# changes what the run MEANS -- rerunning with a different num_workers or
# loss_curve_path gives the same model. Everything else is printed by default,
# which is the whole point of the inversion below.
_PLUMBING = frozenset({
    "base_path", "device", "checkpoint_path", "loss_curve_path", "on_checkpoint_saved",
    "log_every_epoch", "num_workers", "resume_from", "ae_checkpoint_path",
    "lds_checkpoint_path", "encode_batch_size", "cache_in_memory", "latent_cache_dir",
    "vram_log_every",
})


def _format_parameter_value(value) -> str:
    """Paths by name only (the full path is already printed on its own line),
    everything else by its plain repr so a str stays quoted and None stays None
    -- the log has to be unambiguous about which it was."""
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        return repr(value)
    return str(value)


def print_run_parameters(func, values: dict, already_printed=()) -> list[str]:
    """Print every parameter of `func` that the caller's own preamble did NOT
    already print. Returns the lines, for testing.

    THE DEFAULT IS INVERTED, and that is the entire point. Every trainer's
    preamble was a hand-maintained list of print() calls, so a parameter added
    later was invisible unless someone remembered to add a line -- and across
    four trainers, 24 of train_lds()'s 46 parameters, plus lr/min_step in
    stage 2 and lr/latent_channels in stage 1, were never printed at all. The
    ones that hurt most were z1_resync, dt_cap and n_substeps: all three change
    what f_theta MEANS (see _resume_f_theta_from_checkpoint, which warns when
    they change), all three are saved into the checkpoint config, and none
    appeared in a log -- so a 3b log could not say which objective produced it.
    seed was missing from three of the four, making those runs unreproducible
    from their own logs.

    Here a new parameter is printed unless it is deliberately excluded, so the
    failure mode is a slightly noisy log rather than a silently unrecorded
    experiment.

    `already_printed` is the caller's own preamble coverage, passed explicitly
    rather than detected: the preamble prints values with their own context
    (max_dt with its exclusion rule, the loss weights next to their scales),
    which is more informative than a bare name=value, and duplicating them here
    would just make the block noisy enough to stop being read. It is checked
    against the signature, so renaming a parameter without updating the list
    raises instead of silently un-printing it.
    """
    params = list(inspect.signature(func).parameters)
    unknown = sorted(set(already_printed) - set(params))
    if unknown:
        raise ValueError(
            f"print_run_parameters({func.__name__}): already_printed names "
            f"{unknown} are not parameters of {func.__name__} -- they were probably "
            f"renamed or removed, which would silently drop them from the log."
        )
    remaining = [p for p in params
                 if p not in _PLUMBING and p not in set(already_printed) and p in values]
    if not remaining:
        return []
    items = [f"{name}={_format_parameter_value(values[name])}" for name in remaining]
    lines = textwrap.wrap("  ".join(items), width=100,
                          initial_indent="  ", subsequent_indent="  ")
    print("other parameters:")
    for line in lines:
        print(line)
    return lines


def format_progress_count(current: int, total: int) -> str:
    """Format a 'current/total' progress fraction, switching to units of
    thousands once the total is large enough that raw digits are hard to
    scan. Below the threshold the raw integers read fine and are exact, so
    they are kept.

        format_progress_count(2801, 27088)  -> '2.8/27.1 thousand'
        format_progress_count(151, 436)     -> '151/436'

    The threshold is on the TOTAL (not current), so the unit does not flip
    partway through a run.
    """
    if total >= 10_000:
        return f"{current / 1000:.1f}/{total / 1000:.1f} thousand"
    return f"{current}/{total}"


def _format_duration(seconds: float) -> str:
    """Compact human duration for an ETA: '45s', '2m30s', '1h05m'. Rounds to
    whole seconds; drops the smaller unit once we're into hours."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h}h{m:02d}m"


class EpochProgress:
    """In-place within-epoch batch counter that stays SILENT until the epoch
    is actually slow.

    Every-epoch summary lines already tell you the run is alive between epochs.
    What they don't cover is a SINGLE epoch that itself takes minutes -- at
    128x128 an epoch over a couple thousand batches can look frozen while it
    runs. But most stages (stage 3, small resolutions) have fast epochs where a
    per-batch counter would just be noise. So this prints nothing until the
    current epoch has already run longer than `delay_s`; only then does a slow
    epoch start showing "batch k/N", refreshed in place. A fast epoch finishes
    before the delay and prints nothing at all -- no per-stage tuning, no flag.

    Usage, once per epoch:
        prog = EpochProgress(len(train_loader))
        for batch in train_loader:
            prog.tick()
            ...
        prog.close()

    The per-batch cost is one time comparison (and, once past the delay, a
    formatted write every `every` batches); the common fast-epoch path does no
    I/O at all.

    `label`/`unit` name what is being counted, so the counter is meaningful for
    non-epoch loops too, e.g. EpochProgress(n, label="interpolation check",
    unit="triples") -> "interpolation check 2.8/27.1 thousand triples (~1m03s
    left)". The defaults reproduce the epoch-batch wording byte-for-byte, so
    existing callers are unchanged.
    """

    def __init__(self, n_batches, delay_s: float = 20.0, every: int = 50,
                 stream=None, label: str = "epoch progress: batch", unit: str = "",
                 tail_label: str | None = None, tail_seconds: float | None = None):
        self._n = n_batches
        self._delay = delay_s
        self._every = max(1, every)
        self._label = label            # what is being counted (default: the
        self._unit = unit              # epoch-batch wording, unchanged)
        # Optional post-loop phase (e.g. validation) whose time this bar's ETA
        # should also account for -- otherwise "~6m left" reads as "left in the
        # epoch" when it's only "left in TRAINING". tail_label names the phase;
        # tail_seconds is its estimated duration (None = not yet known, e.g.
        # the first epoch, in which case the ETA is marked "+ <tail_label>").
        self._tail_label = tail_label
        self._tail_seconds = tail_seconds
        self._stream = stream if stream is not None else sys.stdout
        self._start = time.monotonic()
        self._i = 0
        self._active = False        # becomes True once the delay is crossed
        self._max_width = 0         # widest line written, for erase-on-close
        self._activated_at = None   # time (and batch) the counter went live --
        self._activated_i = 0       # ETA rate is measured from here, not from
        #                             __init__, so the delay window isn't counted

    def tick(self):
        """Call once per batch, before (or after) the batch's work."""
        self._i += 1
        now = time.monotonic()
        if not self._active:
            if now - self._start < self._delay:
                return              # fast path: one comparison, no I/O
            self._active = True     # this epoch is slow -- start showing progress
            self._activated_at = now
            self._activated_i = self._i
        if self._i % self._every == 0 or self._i == self._n:
            # ETA from the rate SINCE activation (the delay window is excluded,
            # so it reflects steady batch speed). Needs at least one batch of
            # elapsed time past activation to have a rate at all.
            done_since = self._i - self._activated_i
            elapsed_since = now - self._activated_at
            if done_since > 0 and elapsed_since > 0:
                rate = done_since / elapsed_since          # batches/sec
                remaining = (self._n - self._i) / rate     # seconds of THIS loop
                if self._tail_label is None:
                    eta = f"~{_format_duration(remaining)} left"
                elif self._tail_seconds is None:
                    # tail expected but its duration isn't known yet (first
                    # epoch): don't hide it, flag the estimate as incomplete.
                    eta = (f"~{_format_duration(remaining)} left "
                           f"+ {self._tail_label}")
                else:
                    total = remaining + self._tail_seconds
                    eta = (f"~{_format_duration(total)} left = "
                           f"{_format_duration(remaining)} + "
                           f"{_format_duration(self._tail_seconds)}")
            else:
                eta = "estimating..."
            _u = f" {self._unit}" if self._unit else ""
            line = (f"\r  {self._label} "
                    f"{format_progress_count(self._i, self._n)}{_u}  ({eta})   ")
            self._max_width = max(self._max_width, len(line))
            self._stream.write(line)
            self._stream.flush()

    def close(self):
        """Call once after the epoch's loop. ERASES the progress line (rather
        than leaving it above the epoch summary): overwrite it with blanks and
        return the cursor to column 0, so the epoch's own summary line prints
        over clean ground and no per-batch clutter survives. No-op for a fast
        epoch that never activated."""
        if self._active:
            self._stream.write("\r" + " " * self._max_width + "\r")
            self._stream.flush()
