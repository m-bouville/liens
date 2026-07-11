"""
Tees stdout/stderr to a per-stage log file in addition to the console.
Extracted from main.py during its split into orchestration/.
"""
import sys
from contextlib import contextmanager
from pathlib import Path


class _Tee:
    """Writes to multiple streams at once (e.g. the real console AND a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

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
            sys.stdout, sys.stderr = original_stdout, original_stderr
