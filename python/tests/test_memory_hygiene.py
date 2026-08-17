"""
Leaks that only matter in a PERSISTENT kernel (Spyder / Jupyter), where the
process outlives the run.

Reported symptom: main RAM and VRAM both dropped on a kernel restart. Two
mechanisms account for that, and neither is visible in a one-shot
`python main.py`:

  * sys.last_traceback pins every frame of the last failed call, and those
    frames' locals include the cached dataset, the DataLoader (with its
    persistent worker processes) and any live CUDA tensors. gc.collect()
    cannot free them -- they are genuinely still referenced. Measured: a 50 MB
    object allocated inside a raising function survives gc.collect() and is
    released only once sys.last_* is cleared.
  * a matplotlib figure that is never closed stays in pyplot's global figure
    registry for the life of the process.
"""
import gc
import pathlib
import sys

import pytest

from conftest import source_without_comments

_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------
# the traceback mechanism
# --------------------------------------------------------------------

def _free_vram_code() -> str:
    """_free_vram's body with comments and docstring stripped.

    Every assertion below is about what the code DOES; matching raw source
    matches the prose explaining it too, and the prose necessarily names the
    very things being checked.
    """
    src = (_ROOT / "main.py").read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(src) if l.startswith("def _free_vram"))
    body, in_doc = [], False
    for line in src[start + 1:]:
        if line.startswith("def ") or line.startswith("@"):
            break
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.endswith('"""'):
            in_doc = not in_doc
            continue
        if in_doc or stripped.startswith("#"):
            continue
        body.append(line)
    return "\n".join(body)


class _Tracked:
    released = False

    def __init__(self):
        _Tracked.released = False
        self.payload = bytearray(1024)

    def __del__(self):
        _Tracked.released = True


def test_a_traceback_really_does_pin_a_frames_locals():
    """
    Establishes the mechanism the fix targets, so the fix below is not
    protecting against an imagined problem.
    """
    def raiser():
        local = _Tracked()
        assert local is not None  # keeps the binding LIVE in the frame the
        #                           traceback will capture -- deleting it here
        #                           defeats the very mechanism under test
        raise ValueError("boom")

    try:
        raiser()
    except ValueError:
        sys.last_type, sys.last_value, sys.last_traceback = sys.exc_info()
    gc.collect()
    try:
        assert not _Tracked.released, "expected the traceback to keep the local alive"
        sys.last_traceback = sys.last_value = sys.last_type = None
        gc.collect()
        assert _Tracked.released, "clearing sys.last_* should release it"
    finally:
        sys.last_traceback = sys.last_value = sys.last_type = None


def test_free_vram_clears_the_pinned_traceback():
    """
    GUARDS a _free_vram that only calls gc.collect() and empty_cache(). Those
    release UNREFERENCED memory; a traceback's frames are still referenced, so
    the largest single hold in a long kernel session is exactly what they miss.
    """
    body = _free_vram_code()
    # CODE, not comments. An earlier version of this test matched the substring
    # "last_traceback" anywhere in the source -- which the explanatory comment
    # also contains -- so emptying the loop that actually does the clearing
    # left the test green.
    assert '"last_traceback"' in body, "_free_vram must clear sys.last_traceback"
    assert "setattr(sys, attr, None)" in body, (
        "the attribute names must actually be cleared, not merely mentioned"
    )
    assert body.index('"last_traceback"') < body.index("gc.collect()"), (
        "sys.last_* must be cleared BEFORE gc.collect(), or the collection runs "
        "while the frames are still pinned and frees nothing"
    )


def test_free_vram_still_collects_without_cuda():
    """
    GUARDS an early `return` on the no-CUDA path that skips the collection
    entirely. The traceback pin is a RAM problem first; it costs VRAM only
    because CUDA tensors happen to live in those frames.
    """
    body = _free_vram_code()
    no_cuda = body[body.index("if not torch.cuda.is_available():"):]
    assert "gc.collect()" in no_cuda[:no_cuda.index("return") + 20], (
        "the no-CUDA path must still collect -- the RAM half of the leak is real there too"
    )


# --------------------------------------------------------------------
# matplotlib figures
# --------------------------------------------------------------------

def _files_that_save_figures():
    """Only files that actually call savefig, decided at COLLECTION time.

    Parametrising over every evaluation module and skipping the ones with no
    figures produced six skips that meant nothing except "this file is not in
    scope" -- which is the parametrize list's job, not a skip's. Skips should
    report a genuine runtime condition (no CUDA, a missing fixture), so that
    a rising skip count is worth investigating.

    compare_integrators.py is excluded for a real reason and named here rather
    than skipped: it calls f_theta.forward_ab2(), which does not exist on
    LatentDynamics, so it cannot run at all. test_lds_reconstruction_fidelity
    asserts it stays broken, which is what forces it back into scope if it is
    ever revived.
    """
    candidates = sorted((_ROOT / "evaluation").glob("*.py")) + [_ROOT / "utils/plots.py"]
    out = []
    for path in candidates:
        try:
            src = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if ".savefig(" in src and "forward_ab2" not in src:
            out.append(path)
    return out


_PLOTTING = _files_that_save_figures()


@pytest.mark.parametrize("path", _PLOTTING, ids=lambda p: p.name)
def test_every_saved_figure_is_closed(path):
    """
    A figure written with savefig and never closed stays in pyplot's global
    registry for the life of the process. Counted rather than matched
    per-call: exact pairing needs real dataflow analysis, but a file that
    saves more figures than it closes is always wrong.

    show_snapshot in utils/plots.py is the documented exception -- it is an
    interactive helper that RETURNS its axes for the caller to keep, so
    closing would defeat it. It never calls savefig, hence the savefig-based
    count below does not see it.
    """
    # source_without_comments, NOT raw read_text: a comment that MENTIONS
    # .savefig( (utils/plots.py line ~72 documents savefig cost) counted as a
    # save and made the balance depend on unrelated code happening to contain
    # a spare plt.close() -- deleting dead make_video() (1 close, 0 saves)
    # exposed it. The same trap, and the same fix, as the source-matching
    # tests in source_without_comments' own docstring.
    src = source_without_comments(path)
    saves = src.count(".savefig(")
    closes = src.count("plt.close(")
    assert closes >= saves, (
        f"{path.name}: {saves} savefig call(s) but only {closes} plt.close() -- "
        f"an unclosed figure leaks for the life of a persistent kernel"
    )


# --------------------------------------------------------------------
# the leak detector must not mistake a MOVE for a write
# --------------------------------------------------------------------

# The PRODUCTION functions, imported -- not re-implementations. An earlier
# version of this file reimplemented both, so dropping the move check in the
# fixture left every test here green: they were checking their own arithmetic.
# That is why the rule is a module-level function rather than logic inline in
# the fixture.
from conftest import files_written_between, snapshot_files  # noqa: E402


def _snapshot(bases):
    return snapshot_files(bases)


def _leaked(before, after):
    return [p.name for p in files_written_between(before, after)]


def test_moving_a_file_within_the_tree_is_not_a_leak(tmp_path):
    """
    REGRESSION. The detector compared PATHS, so tidying old backups into
    checkpoints/stage2/_archives/ during a session showed up as four
    brand-new files and failed the suite -- when nothing had been written at
    all.

    Identity is (size, mtime_ns): shutil.move and copy2 both preserve mtime,
    so a new path whose identity matches something that DISAPPEARED is a move.
    """
    import os
    import shutil

    stage = tmp_path / "checkpoints" / "stage2"
    (stage / "_archives").mkdir(parents=True)
    f = stage / "128x128-stage2-20260802_09h40.pt"
    f.write_bytes(b"x" * 100)
    os.utime(f, (1_700_000_000, 1_700_000_000))

    before = _snapshot([stage])
    shutil.move(str(f), str(stage / "_archives" / f.name))
    assert _leaked(before, _snapshot([stage])) == []


def test_a_real_write_is_still_caught(tmp_path):
    """GUARDS making the detector so permissive it stops detecting."""
    stage = tmp_path / "checkpoints" / "stage2"
    stage.mkdir(parents=True)
    (stage / "old.pt").write_bytes(b"x" * 100)
    before = _snapshot([stage])
    (stage / "written-by-a-test.png").write_bytes(b"y" * 50)
    assert _leaked(before, _snapshot([stage])) == ["written-by-a-test.png"]


def test_a_COPY_is_still_caught_even_though_mtime_is_preserved(tmp_path):
    """
    copy2 preserves mtime too, but the source does NOT disappear -- so its
    identity is not in `vanished` and the new file is correctly reported.
    That distinction is the whole reason the check is on vanished entries
    rather than on mtime alone.
    """
    import shutil

    stage = tmp_path / "checkpoints" / "stage2"
    stage.mkdir(parents=True)
    src = stage / "orig.pt"
    src.write_bytes(b"x" * 100)
    before = _snapshot([stage])
    shutil.copy2(src, stage / "copy.pt")
    assert _leaked(before, _snapshot([stage])) == ["copy.pt"]


def test_the_detector_uses_identity_not_bare_paths():
    """The conftest fixture must use the same scheme these tests describe."""
    src = source_without_comments(_ROOT / "tests/conftest.py")
    assert "st.st_mtime_ns" in src, "the snapshot must record identity, not just paths"
    assert "vanished" in src, "moves must be recognised by matching a disappeared file"


def test_the_leak_report_does_not_assert_a_single_cause():
    """
    The first time this detector fired for real it named the wrong culprit.

    The four files were output/stage3b/128x128-stage3b-loss_curve.{png,csv},
    checkpoints/stage3b/128x128-stage3b.pt and registry-stage3b.csv -- size
    128, which NO test uses (they use 4, 8, 32 and 64) and a pipeline registry
    no test writes. They were the person's own training run, writing while the
    suite happened to be running. The message nevertheless stated flatly that
    "A test is using a default path anchored to _PYTHON_ROOT", sending the
    reader hunting through the suite for a bug that was not there.

    Same failure as the val-excursion message asserting weight damage: a
    diagnostic that names the wrong cause is worse than one that names none.
    """
    src = source_without_comments(_ROOT / "tests/conftest.py")
    assert "TWO possible causes" in src
    assert "concurrent TRAINING RUN" in src
    assert "size 4/8/32/64" in src, (
        "the message must give the reader a way to tell the two apart"
    )
    assert "A test is using a default path anchored to _PYTHON_ROOT.\\n" not in src


def test_the_leak_report_shows_timestamps():
    """A modification time is what distinguishes 'written just now by the
    suite' from 'written by the run that finished ten minutes ago'."""
    src = source_without_comments(_ROOT / "tests/conftest.py")
    assert "modified {when}" in src
    assert "st_mtime" in src


def test_a_file_still_changing_at_teardown_is_not_a_leak(tmp_path):
    """
    A file still being rewritten AFTER the tests have finished was not written
    by them. Two snapshots a second apart at teardown separate a live external
    writer -- a training run left running -- from a real leak, whose writes are
    complete by the time the session ends.

    Measured rather than assumed: the detector fired on
    output/stage3b/128x128-stage3b-loss_curve.png, and four xdist workers
    reported mtimes of 22:09:37, :47, :47 and :52 for it -- still advancing,
    long after any test had touched anything. The checkpoint beside it sat
    fixed at 22:09:33. Asking the person to tell those apart by eye is a worse
    answer than measuring it.
    """
    import threading
    import time as _t

    before = snapshot_files([tmp_path])
    (tmp_path / "leak.png").write_bytes(b"x" * 50)          # written once, then quiet

    stop = threading.Event()

    def _writer():                                           # a concurrent run
        n = 0
        while not stop.is_set():
            (tmp_path / "training.png").write_bytes(b"y" * (100 + n))
            n += 1
            _t.sleep(0.1)

    threading.Thread(target=_writer, daemon=True).start()
    _t.sleep(0.3)
    try:
        after = snapshot_files([tmp_path])
        _t.sleep(1.0)
        still = {k for k, v in snapshot_files([tmp_path]).items()
                 if k in after and after[k] != v}
        reported = [p for p in files_written_between(before, after) if p not in still]
    finally:
        stop.set()

    assert {p.name for p in still} == {"training.png"}
    assert {p.name for p in reported} == {"leak.png"}, (
        "the one-shot write must still be reported, or the detector stops detecting"
    )


def test_the_fixture_applies_the_still_changing_filter():
    src = source_without_comments(_ROOT / "tests/conftest.py")
    assert "still_changing" in src
    assert "time.sleep(1.0)" in src
    assert "p not in still_changing" in src, "the filter is computed but not applied"
    assert "ignoring" in src, "an ignored file must still be reported, not hidden"


# --------------------------------------------------------------------
# attributing a file to the tests or to the person
# --------------------------------------------------------------------

def test_a_grid_size_no_test_uses_is_attributed_to_the_person():
    """
    Timing cannot settle this. The still-changing check separates a training
    run that is ACTIVELY writing at teardown from a test leak -- but a real
    stage-4 run that FINISHED at 22:35:12, mid-session, is changing nothing by
    the time the session ends, and was reported as a leak anyway.

    The filename can settle it: no test ever builds a 128x128 sweep.
    """
    from pathlib import Path

    from conftest import TEST_GRID_SIZES, _looks_like_a_test_artifact

    assert 128 not in TEST_GRID_SIZES and 256 not in TEST_GRID_SIZES
    assert not _looks_like_a_test_artifact(Path("128x128-stage4-loss_curve.png"))
    assert _looks_like_a_test_artifact(Path("64x64-stage3b-f_theta_diagnostic.png")), (
        "the historical real leak must still be reported"
    )
    assert _looks_like_a_test_artifact(Path("32x32-stage2.pt"))


def test_an_unrecognised_name_is_treated_as_a_LEAK():
    """
    GUARDS an exoneration that swallows real leaks. Only a clear, non-test grid
    size lets a file off; anything unparseable is reported, because a genuine
    leak with an unexpected name must not vanish.
    """
    from pathlib import Path

    from conftest import _looks_like_a_test_artifact
    for name in ("something-odd.pt", "diagnostic.png", "registry-stage4.csv"):
        assert _looks_like_a_test_artifact(Path(name)), name


def test_no_test_builds_a_sweep_outside_TEST_GRID_SIZES():
    """
    The declared set is only trustworthy if it stays true. A test that starts
    building a 128x128 sweep would make the detector blind to its own leaks --
    silently, since the exoneration is by name.
    """
    import pathlib
    import re

    from conftest import TEST_GRID_SIZES

    root = pathlib.Path(__file__).resolve().parent
    offenders = []
    for p in sorted(root.glob("test_*.py")):
        src = source_without_comments(p)
        for m in re.finditer(r"_build_sweep\([^)]*size\s*=\s*(\d+)", src):
            if int(m.group(1)) not in TEST_GRID_SIZES:
                offenders.append(f"{p.name}: size={m.group(1)}")
        for m in re.finditer(r"_build_run_dir_with_stats\([^)]*size\s*=\s*(\d+)", src):
            if int(m.group(1)) not in TEST_GRID_SIZES:
                offenders.append(f"{p.name}: size={m.group(1)}")
    assert not offenders, (
        "a test builds a sweep at a size the detector does not know about, so a "
        "leak from it would be silently exonerated: " + ", ".join(offenders)
    )


def test_nameless_siblings_inherit_the_attribution():
    """
    registry-stage4.csv carries no <N>x<N> prefix, but the same run_lds_stage
    call wrote 128x128-stage4.pt beside it. Attributing the checkpoint and
    reporting the registry would be incoherent.
    """
    src = source_without_comments(_ROOT / "tests/conftest.py")
    assert "_their_dirs" in src
    assert "p.parent in _their_dirs" in src
    # The FIXTURE must actually call the helper. Defining it and never wiring
    # it in is precisely what happened: an indentation mismatch made two
    # str-replace edits no-ops, leaving _looks_like_a_test_artifact defined,
    # unused, and every attribution test passing on the helper alone.
    assert "_theirs = [p for p in added if not _looks_like_a_test_artifact(p)]" in src, (
        "the helper is defined but the detector never consults it"
    )
