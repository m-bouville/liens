"""
Tees stdout/stderr to a per-stage log file in addition to the console.
Extracted from main.py during its split into orchestration/.
"""
import inspect
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path


class _Tee:
    """Writes to multiple streams at once (e.g. the real console AND a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
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
