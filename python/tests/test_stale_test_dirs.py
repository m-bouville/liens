"""
A checkpoint's stored test_dirs must be validated, with a message that says
what is wrong.

Reported from a real stage-5 run: all FIVE diagnostics failed with

    FileNotFoundError: .../datasets/128x128/T775_n045_s191/metadata.txt
      in _parse_kv_file <- read_metadata <- build_good_steps <- Dataset.__init__

five times over, nothing naming the actual problem. Training in the SAME
process had succeeded, because it re-enumerates the sweep rather than
replaying a list stored at training time.

The pipeline survived only because the diagnostics are non-fatal.
"""
import pathlib

import pytest

import utils.load_datasets as load
from conftest import source_without_comments

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _dirs(tmp_path, n_total, n_missing):
    """n_total run dirs, the first n_missing of them without a metadata.txt."""
    out = []
    for i in range(n_total):
        d = tmp_path / f"T800_n010_s{i}"
        d.mkdir()
        if i >= n_missing:
            (d / "metadata.txt").write_text("x = 1\n")
        out.append(d)
    return out


def test_a_SMALL_fraction_is_skipped_and_reported(tmp_path, capsys):
    """
    THE REAL CASE: T775_n045_s191, one run of 427, absent for a few minutes
    while being regenerated with more time steps. Losing five diagnostics over
    1/427 of a statistical population is a bad trade.
    """
    dirs = _dirs(tmp_path, 40, 1)
    kept = load.validate_run_dirs(dirs, source="the stage-5 checkpoint")
    assert len(kept) == 39
    assert all((pathlib.Path(d) / "metadata.txt").is_file() for d in kept)

    out = capsys.readouterr().out
    assert "skipping 1 of 40" in out, "a silent skip is the failure mode being avoided"
    assert "T800_n010_s0" in out, "the skipped run is not named"
    assert "the stage-5 checkpoint" in out, "the source of the list is not named"


def test_a_LARGE_fraction_raises_as_a_stale_split(tmp_path):
    """
    The missing FRACTION is the only signal available that separates a race
    from a stale split: a handful is a regeneration, a hundred is a checkpoint
    trained on a different sweep. Evaluating on what remains would report
    numbers for a population nobody chose.
    """
    with pytest.raises(FileNotFoundError) as exc:
        load.validate_run_dirs(_dirs(tmp_path, 40, 5), source="a stage-3b checkpoint")
    msg = str(exc.value)
    assert "12.5%" in msg, "the fraction is what justifies the decision; state it"
    assert "STALE SPLIT" in msg
    assert "a stage-3b checkpoint" in msg
    assert "T800_n010_s0" in msg


def test_the_tolerance_boundary_is_configurable(tmp_path):
    dirs = _dirs(tmp_path, 40, 4)          # exactly 10%
    assert len(load.validate_run_dirs(dirs, max_missing_fraction=0.10)) == 36
    with pytest.raises(FileNotFoundError):
        load.validate_run_dirs(dirs, max_missing_fraction=0.05)


def test_ALL_missing_raises_even_though_it_is_the_whole_population(tmp_path):
    """
    GUARDS a fraction test that lets 100% through when the tolerance is 1.0,
    and more importantly guards returning an EMPTY list -- which would fail
    later, somewhere less informative.
    """
    with pytest.raises(FileNotFoundError):
        load.validate_run_dirs(_dirs(tmp_path, 3, 3), max_missing_fraction=1.0)


def test_all_present_returns_unchanged(tmp_path):
    """It wraps a lookup inline, so it must be transparent on the happy path."""
    d = tmp_path / "T800_n010_s0"
    d.mkdir()
    (d / "metadata.txt").write_text("x = 1\n")
    (d / "statistics.csv").write_text("step,stdev_phi\n0,0.1\n")
    assert load.validate_run_dirs([d]) == [d]


def test_the_listing_is_capped(tmp_path):
    """GUARDS printing 427 paths when a whole sweep has moved."""
    dirs = []
    for i in range(15):
        d = tmp_path / f"T800_n010_s{i}"
        d.mkdir()
        dirs.append(d)
    with pytest.raises(FileNotFoundError) as exc:
        load.validate_run_dirs(dirs)
    assert "and 5 more" in str(exc.value)


@pytest.mark.parametrize("module", [
    "check_reconstruction", "check_latent_channels", "check_interpolation",
    "check_perturbation", "check_rollout", "_latent_eval",
])
def test_every_diagnostic_validates_its_stored_test_dirs(module):
    """
    Per-module, not a file-scope substring: ALL FIVE failed on the real run,
    so one unvalidated site is one diagnostic that still reports a parser
    traceback instead of the cause.
    """
    src = source_without_comments(_ROOT / f"evaluation/{module}.py")
    assert "validate_run_dirs(" in src, f"{module} replays test_dirs unchecked"


def test_a_HALF_WRITTEN_run_is_caught_when_stdev_filtering_is_ON(tmp_path, capsys):
    """
    REGRESSION. Checking only metadata.txt let a run through that had
    metadata.txt but no statistics.csv yet -- build_good_steps reads BOTH --
    and the failure surfaced as FileNotFoundError inside pandas.read_csv, four
    frames deep, which is the opaque failure this function exists to replace.

    A run being regenerated gains its files in some order, so requiring all of
    them is the only check that does not depend on which.
    """
    dirs = []
    for i in range(40):
        d = tmp_path / f"T800_n010_s{i}"
        d.mkdir()
        (d / "metadata.txt").write_text("x = 1\n")
        if i != 7:                       # r7: metadata present, statistics.csv absent
            (d / "statistics.csv").write_text("step,stdev_phi\n0,0.1\n")
        dirs.append(d)

    # min_stdev_phi set -> build_good_steps WILL read statistics.csv, so it is
    # required. Without the filter it never opens the file and demanding it
    # would drop runs over a check nothing depends on (which broke nine
    # metadata-only fixtures -- verified).
    kept = load.validate_run_dirs(dirs, source="the stage-3b checkpoint", min_stdev_phi=0.01)
    assert len(kept) == 39, "a run with metadata.txt but no statistics.csv slipped through"
    assert "statistics.csv" in capsys.readouterr().out, (
        "the message must name WHICH file is missing"
    )


def test_the_requirement_FOLLOWS_the_filter(tmp_path):
    """
    statistics.csv is required exactly when it will be READ. With no
    min_stdev_phi, build_good_steps never opens it, and demanding it would
    drop runs over a check nothing depends on.
    """
    d = tmp_path / "T800_n010_s0"
    d.mkdir()
    (d / "metadata.txt").write_text("x = 1\n")
    assert load.validate_run_dirs([d]) == [d]                       # no filter -> fine
    with pytest.raises(FileNotFoundError):
        load.validate_run_dirs([d], min_stdev_phi=0.01)              # filter -> required
    assert load.validate_run_dirs([d], required=("metadata.txt",)) == [d]   # explicit override
