"""
On first discovery of a sweep, report how far its runs were actually evolved.

A sweep is not necessarily homogeneous: runs regenerated later to reach past
tau_down carry many more saved steps than the originals, and nothing else in
the pipeline says so. Every window count downstream is then a mixture of two
populations, and a change in the mixture looks exactly like a change in the
physics.

It matters most where it is least visible. tau_down grows as ~L^2.5 (measured:
6e5 at 64x64, 2.5e6 at 128, ~1.4e7 at 256), so at large sizes only the
deliberately-extended runs reach coarsening completion -- and those are the
only source of the frozen absorbing states that dominate the large-dt error.
"""
import pathlib
import re

import pytest

import training.datasets as datasets
from training.datasets import complete_run_dirs


def _sweep(tmp_path, extend=0, extra_steps=(50000, 200000)):
    """A real sweep from the shared fixture, with `extend` runs lengthened.

    Built via test_train_lds._build_sweep rather than by hand: an earlier
    version wrote its own metadata.txt and discovery silently found no runs at
    all, so every assertion compared against empty output. Reusing the fixture
    the rest of the suite uses means the format cannot drift apart from it.
    """
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_train_lds import _build_sweep

    base = _build_sweep(tmp_path, n_runs=5, size=32)
    runs = sorted(d for d in (base / "32x32").iterdir() if d.is_dir())
    for run in runs[:extend]:
        meta = run / "metadata.txt"
        text = meta.read_text()
        # `save_steps = 0 1000 2000 ...` -- space separated, see the fixture
        text = re.sub(r"^(save_steps\s*=.*)$",
                       lambda m: m.group(1) + " " + " ".join(str(s) for s in extra_steps),
                       text, count=1, flags=re.M)
        meta.write_text(text)
    return base


@pytest.fixture(autouse=True)
def _forget_reported_sweeps():
    datasets._REPORTED_SWEEPS.clear()
    yield
    datasets._REPORTED_SWEEPS.clear()


def test_a_homogeneous_sweep_reports_one_line(tmp_path, capsys):
    base = _sweep(tmp_path)
    complete_run_dirs(base, 32, 32)
    out = capsys.readouterr().out
    assert "all 5 runs have" in out and "saved steps" in out
    assert "MIXED" not in out


def test_a_MIXED_sweep_is_flagged_and_broken_down(tmp_path, capsys):
    """
    THE case this exists for: some runs regenerated longer to pass tau_down.
    Both populations must be visible, and the word MIXED must appear -- a
    reader scanning the log needs to notice without parsing the table.
    """
    base = _sweep(tmp_path, extend=2)
    complete_run_dirs(base, 32, 32)
    out = capsys.readouterr().out
    assert "MIXED" in out
    assert "2 distinct lengths across 5 runs" in out
    # Percentages, so a reader sees at a glance whether the extended runs are
    # a rounding error or the bulk of the sweep -- 259/2788 and 1928/2788 read
    # very differently, and the raw counts alone do not say which is which.
    assert "(60.0%)" in out and "(40.0%)" in out, out
    assert "200,000" in out, "the longest run's final step must be visible"


def test_it_reports_only_once_per_sweep(tmp_path, capsys):
    """
    GUARDS re-reading every run's metadata at every stage. complete_run_dirs is
    called by stage 1, 2, 3a, 3b, 4 and 5, and by several diagnostics; the
    answer cannot change within a process.
    """
    base = _sweep(tmp_path)
    complete_run_dirs(base, 32, 32)
    assert "save schedule" in capsys.readouterr().out
    complete_run_dirs(base, 32, 32)
    assert "save schedule" not in capsys.readouterr().out


def test_the_report_is_tracked_per_SIZE_not_just_per_sweep(tmp_path):
    """
    GUARDS keying the once-only report on the base path alone. A 64x64 sweep
    discovered after a 32x32 one under the same root is a different population
    -- and at 256 the whole point is that its schedule differs -- so silencing
    it because "this root was already reported" would hide exactly the case
    this exists for.

    Asserted on the key rather than by building a second sweep: the property
    is that the size is part of the identity, and an incomplete hand-built
    64x64 fixture would test the fixture instead.
    """
    base = _sweep(tmp_path)
    complete_run_dirs(base, 32, 32)
    # The key is the directory the runs live in (.../datasets/32x32), which
    # encodes the size by construction. Asserted on that property rather than
    # on the key's literal shape, which has already changed once: it was a
    # (base, nx, ny) tuple while the dedup lived in complete_run_dirs, and
    # became a path when the dedup moved into the reporter so that the
    # diagnostics bypassing complete_run_dirs would get it too.
    assert any(k.endswith("32x32") for k in datasets._REPORTED_SWEEPS)
    assert not any(k.endswith("64x64") for k in datasets._REPORTED_SWEEPS), (
        "reporting 32x32 must not mark 64x64 as already done"
    )


def test_metadata_is_read_from_the_FILE_not_the_directory(tmp_path, capsys):
    """
    GUARDS load.read_metadata(run_dir), which raises IsADirectoryError -- an
    OSError, which the helper's own except clause swallows. The report then
    prints NOTHING at all, looking exactly like a sweep with no metadata.
    Caught only by checking real output; the first version had this bug.
    """
    base = _sweep(tmp_path)
    complete_run_dirs(base, 32, 32)
    assert "save schedule" in capsys.readouterr().out, (
        "the report produced no output -- metadata is probably being read from "
        "the directory instead of the metadata.txt inside it"
    )


def test_an_unreadable_run_does_not_break_discovery(tmp_path, capsys):
    base = _sweep(tmp_path)
    first = sorted(d for d in (base / "32x32").iterdir() if d.is_dir())[0]
    (first / "metadata.txt").write_text("garbage {{{")
    dirs = complete_run_dirs(base, 32, 32)
    assert len(dirs) >= 1, "a malformed run must not abort discovery"


# --------------------------------------------------------------------
# every discovery path must reach the report
# --------------------------------------------------------------------

def _modules_that_discover_runs():
    """Files calling enumerate_run_dirs_from_metadata directly.

    The report cannot live inside enumerate_run_dirs_from_metadata itself: its
    docstring promises it is "pure string construction -- does not touch
    individual run directories, so it works even before any run in the list
    actually exists", and reading each run's metadata.txt would break that for
    callers that legitimately enumerate a sweep before it is generated.

    So the report is a separate call, and this test is what stops the next
    direct caller from silently skipping it -- the same shape as
    test_lds_reconstruction_fidelity, and for the same reason: the codebase has
    several independent entry points to one thing.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    found = []
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if parts[0] not in ("evaluation", "training", "orchestration", "utils"):
            continue
        if path.name.startswith("test_") or path.name.endswith("_UPSTREAM.py"):
            continue
        src = path.read_text(encoding="utf-8")
        if "enumerate_run_dirs_from_metadata(" in src and "def enumerate_run_dirs" not in src:
            found.append((path, src))
    return found


def test_at_least_one_direct_caller_exists():
    """Sanity: if this returns nothing the test below is vacuous."""
    assert _modules_that_discover_runs()


@pytest.mark.parametrize("path_and_src", _modules_that_discover_runs(),
                          ids=lambda p: p[0].name)
def test_direct_discovery_still_reports_the_save_schedule(path_and_src):
    """
    GUARDS a module that enumerates runs itself and never asks for the report.
    check_stdev_phi_time did exactly that: it calls
    enumerate_run_dirs_from_metadata directly, so the report added to
    complete_run_dirs never fired for it, and a MIXED sweep looked homogeneous
    from the command line.

    datasets.py is exempt: it IS complete_run_dirs, which calls the reporter.
    """
    path, src = path_and_src
    if path.name == "datasets.py":
        assert "report_save_step_distribution(run_dirs)" in src
        return
    assert "report_save_step_distribution(" in src, (
        f"{path.name} discovers runs directly but never calls "
        f"report_save_step_distribution -- a mixed sweep will look homogeneous there"
    )


def test_the_reporter_deduplicates_itself_not_its_callers(tmp_path, capsys):
    """
    The dedup must live in the reporter. A key owned by complete_run_dirs would
    do nothing for the diagnostics that bypass it -- neither silencing repeats
    nor allowing the first report.
    """
    from training.datasets import report_save_step_distribution
    base = _sweep(tmp_path)
    runs = sorted(d for d in (base / "32x32").iterdir() if d.is_dir())
    report_save_step_distribution(runs)
    assert "save schedule" in capsys.readouterr().out
    report_save_step_distribution(runs)
    assert "save schedule" not in capsys.readouterr().out


def test_an_empty_run_list_is_a_no_op(tmp_path, capsys):
    from training.datasets import report_save_step_distribution
    report_save_step_distribution([])
    assert capsys.readouterr().out == ""


def test_percentages_are_of_READABLE_runs_and_sum_to_100(tmp_path, capsys):
    """
    GUARDS dividing by len(run_dirs). A run whose metadata cannot be read
    contributes to no group, so with that denominator the column silently stops
    reaching 100% -- and there is nothing in the output to say a run was
    dropped, so the shortfall reads as a rounding artefact.

    Distinguishable only with an unreadable run present: with a clean sweep the
    two denominators are equal and the bug is invisible.
    """
    base = _sweep(tmp_path, extend=2)
    runs = sorted(d for d in (base / "32x32").iterdir() if d.is_dir())
    (runs[-1] / "metadata.txt").write_text("not parseable {{{")

    datasets._REPORTED_SWEEPS.clear()
    datasets.report_save_step_distribution(runs)
    out = capsys.readouterr().out

    percentages = [float(m) for m in re.findall(r"\((\s*[\d.]+)%\)", out)]
    assert percentages, out
    assert abs(sum(percentages) - 100.0) < 0.25, (
        f"percentages sum to {sum(percentages)}, not 100 -- the denominator is "
        f"probably len(run_dirs) rather than the runs actually counted:\n{out}"
    )
