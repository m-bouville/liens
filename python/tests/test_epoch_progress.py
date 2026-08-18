"""EpochProgress: the within-epoch batch counter must stay SILENT for fast
epochs (the common case -- stage 3, small resolutions) and only show progress
once an epoch has actually run past its delay. Gated on elapsed time so no
per-stage tuning is needed."""
import io

from utils.logging_utils import EpochProgress


def test_fast_epoch_prints_nothing():
    """delay not reached -> no output at all, no matter how many batches."""
    buf = io.StringIO()
    prog = EpochProgress(n_batches=1000, delay_s=999.0, every=10, stream=buf)
    for _ in range(1000):
        prog.tick()
    prog.close()
    assert buf.getvalue() == ""


def _render_terminal(s: str) -> str:
    """Collapse carriage-return overwrites the way a real terminal shows them:
    each '\\r' moves the cursor to column 0 and subsequent chars overwrite.
    Returns what would be visible on the final line."""
    line = ""
    col = 0
    for ch in s:
        if ch == "\r":
            col = 0
        elif ch == "\n":
            line = ""
            col = 0
        else:
            line = line[:col] + ch + line[col + 1:]
            col += 1
    return line


def test_slow_epoch_shows_batch_counter():
    """delay already elapsed (delay_s=0) -> counter appears and reaches n."""
    buf = io.StringIO()
    prog = EpochProgress(n_batches=100, delay_s=0.0, every=10, stream=buf)
    for _ in range(100):
        prog.tick()
    # before close, the terminal shows the counter having reached the final batch
    assert "batch 100/100" in _render_terminal(buf.getvalue())


def test_close_erases_the_line_leaving_clean_ground():
    """After close(), the terminal-visible line is BLANK -- the per-batch
    counter is erased, not left as clutter above the epoch summary."""
    buf = io.StringIO()
    prog = EpochProgress(n_batches=100, delay_s=0.0, every=10, stream=buf)
    for _ in range(100):
        prog.tick()
    prog.close()
    visible = _render_terminal(buf.getvalue())
    assert visible.strip() == "", repr(visible)   # nothing left on the line
    assert "\n" not in buf.getvalue()             # erase, not a newline terminate


def test_counter_refreshes_in_place_not_appending_lines():
    """Refreshes in place via carriage returns -- never emits a newline mid-epoch
    (which would scroll a new line per update instead of overwriting)."""
    buf = io.StringIO()
    prog = EpochProgress(n_batches=50, delay_s=0.0, every=10, stream=buf)
    for _ in range(50):
        prog.tick()
    # every=10 over 50 -> 5 in-place updates, all on ONE line (no newlines)
    assert buf.getvalue().count("\r") == 5, buf.getvalue()
    assert "\n" not in buf.getvalue()


def test_close_without_activation_is_silent():
    """A fast epoch's close() must not emit a stray newline."""
    buf = io.StringIO()
    prog = EpochProgress(n_batches=10, delay_s=999.0, stream=buf)
    for _ in range(10):
        prog.tick()
    prog.close()
    assert buf.getvalue() == ""


def test_eta_counts_down_with_a_controlled_clock():
    """With a fixed per-batch time, the ETA should reflect the remaining
    batches at the measured rate and shrink toward zero."""
    import io
    from unittest import mock
    from utils.logging_utils import EpochProgress

    buf = io.StringIO()
    t = [1000.0]
    with mock.patch("utils.logging_utils.time.monotonic", lambda: t[0]):
        prog = EpochProgress(n_batches=100, delay_s=0.0, every=10, stream=buf)
        for _ in range(100):
            t[0] += 0.1                 # 0.1s/batch -> 10 batches/s
            prog.tick()
    segs = [s for s in buf.getvalue().split("\r") if "batch" in s]
    # batch 10: 90 remain at 10/s -> ~9s
    assert "(~9s left)" in segs[0], segs[0]
    # last: essentially done
    assert "(~0s left)" in segs[-1], segs[-1]


def test_eta_estimating_on_activation_batch():
    """The batch that activates the counter has no elapsed rate yet, so it must
    say 'estimating...' rather than divide by zero."""
    import io
    from unittest import mock
    from utils.logging_utils import EpochProgress

    buf = io.StringIO()
    t = [1000.0]
    with mock.patch("utils.logging_utils.time.monotonic", lambda: t[0]):
        prog = EpochProgress(n_batches=50, delay_s=0.0, every=1, stream=buf)
        prog.tick()                     # activation batch, no rate yet
    segs = [s for s in buf.getvalue().split("\r") if "batch" in s]
    assert "estimating..." in segs[0], segs[0]


def test_eta_excludes_the_delay_window():
    """The rate is measured from activation, not from __init__, so a long delay
    before the epoch is judged slow does not distort the ETA."""
    import io
    from unittest import mock
    from utils.logging_utils import EpochProgress

    buf = io.StringIO()
    t = [1000.0]
    with mock.patch("utils.logging_utils.time.monotonic", lambda: t[0]):
        # delay 5s: the first ticks happen during the delay window and print
        # nothing; only after 5s elapsed does the counter activate.
        prog = EpochProgress(n_batches=100, delay_s=5.0, every=10, stream=buf)
        # 10 batches during the delay (0.5s each = 5s), silent
        for _ in range(10):
            t[0] += 0.5
            prog.tick()
        # now past delay; 10 more batches at 0.1s each
        for _ in range(10):
            t[0] += 0.1
            prog.tick()
    segs = [s for s in buf.getvalue().split("\r") if "batch" in s]
    # rate should reflect the 0.1s/batch steady state (~10/s), NOT the 0.5s
    # delay-window batches. At batch 20, 80 remain at ~10/s -> ~8s, not ~40s.
    assert "(~8s left)" in segs[-1], segs[-1]


def test_format_duration_units():
    from utils.logging_utils import _format_duration
    assert _format_duration(5) == "5s"
    assert _format_duration(90) == "1m30s"
    assert _format_duration(3661) == "1h01m"


def test_tee_keeps_progress_out_of_the_log_file():
    """In-place '\\r' progress chunks go to the console (first stream) ONLY --
    a log file capturing the raw stream would otherwise accumulate every
    update as a permanent wall of repeated text."""
    import io
    from utils.logging_utils import _Tee

    console, logfile = io.StringIO(), io.StringIO()
    tee = _Tee(console, logfile)
    tee.write("normal line\n")
    tee.write("\r  epoch progress: batch 50/9517   ")
    tee.write("\r  epoch progress: batch 100/9517   ")
    tee.write("second normal line\n")
    assert "epoch progress" in console.getvalue()        # console sees the bar
    assert "epoch progress" not in logfile.getvalue()     # log does NOT
    assert logfile.getvalue() == "normal line\nsecond normal line\n"


def test_epoch_progress_through_a_tee_leaves_log_clean():
    """End-to-end: a slow epoch's bar via a teed stdout leaves zero progress
    text in the log stream, while the epoch summary (a normal print) lands."""
    import io
    from utils.logging_utils import _Tee, EpochProgress

    console, logfile = io.StringIO(), io.StringIO()
    tee = _Tee(console, logfile)
    prog = EpochProgress(n_batches=100, delay_s=0.0, every=10, stream=tee)
    for _ in range(100):
        prog.tick()
    prog.close()
    tee.write("   1| 1.1498 = ... epoch summary\n")
    assert "epoch progress" in console.getvalue()
    assert "epoch progress" not in logfile.getvalue()
    assert "epoch summary" in logfile.getvalue()


def test_epoch_bar_uses_thousands_for_large_batch_counts():
    """A 128x128 epoch can exceed 10k batches (e.g. 12690 at batch size 48);
    the bar must follow the same thousands convention as the other counters,
    not print raw 5-digit fractions."""
    import io
    from unittest import mock
    from utils.logging_utils import EpochProgress

    buf = io.StringIO()
    t = [1000.0]
    with mock.patch("utils.logging_utils.time.monotonic", lambda: t[0]):
        prog = EpochProgress(n_batches=12690, delay_s=0.0, every=12690, stream=buf)
        for _ in range(12690):
            t[0] += 0.001
            prog.tick()
    out = buf.getvalue()
    assert "12.7/12.7 thousand" in out, out
    assert "12690" not in out
