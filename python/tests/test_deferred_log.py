"""_log_to_file must NOT touch the log on disk until the run's first epoch, so a
crash during the (minutes-long) setup phase, or a mistaken re-launch, cannot
overwrite a valuable existing log with a near-empty one."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils.logging_utils import _log_to_file, _DeferredLogFile


def test_setup_crash_before_first_epoch_leaves_the_old_log_untouched(tmp_path):
    lp = tmp_path / "stage5.log"
    lp.write_text("VALUABLE PREVIOUS RUN LOG\n")
    try:
        with _log_to_file(lp):
            print("STAGE 5: end-to-end refinement")
            print("     4114 runs (94.3%):  71 saved steps")   # setup line, leading digits
            print("Starting 60 epochs...")
            raise RuntimeError("crash during dataset load")
    except RuntimeError:
        pass
    assert lp.read_text() == "VALUABLE PREVIOUS RUN LOG\n"


def test_it_commits_and_goes_live_at_the_first_epoch(tmp_path):
    lp = tmp_path / "s.log"
    lp.write_text("OLD")
    with _log_to_file(lp):
        print("Starting...")
        print("   1| 0.5415 = 0.17 | 2.15  -> saved")
        assert lp.read_text().startswith("Starting"), "did not commit at epoch 1"
        print("   2| 0.73 = ...")
    body = lp.read_text()
    assert "OLD" not in body and "0.5415" in body and "0.73" in body


def test_lds_epoch_line_format_also_triggers(tmp_path):
    lp = tmp_path / "s3.log"
    lp.write_text("OLD")
    with _log_to_file(lp):
        print("Starting...")
        print("   1   0.532,  0.610 |   0.610")     # stage-3 format: num, spaces, number
    assert "0.532" in lp.read_text() and "OLD" not in lp.read_text()


def test_setup_lines_with_leading_digits_do_not_trigger_early(tmp_path):
    """A run that crashes after printing setup lines that START with numbers
    ('4364 complete runs', '     4114 runs (...)') must NOT have committed on
    them -- only a real epoch line commits."""
    lp = tmp_path / "s.log"
    lp.write_text("KEEP ME")
    df = _DeferredLogFile(lp)
    for ln in ["4364 complete runs -> 2837 train\n",
               "     4114 runs (94.3%):  71 saved steps\n",
               "1023/103725 (1.0%) windows\n",
               "Starting 60 epochs...\n"]:
        df.write(ln)
    assert not df.committed, "committed on a setup line"
    assert lp.read_text() == "KEEP ME"
    df.write("   1| 0.54 = ...\n")     # now a real epoch line
    assert df.committed
    df.close()


def test_epochs_zero_run_without_an_epoch_line_never_overwrites(tmp_path):
    lp = tmp_path / "s.log"
    lp.write_text("PRECIOUS")
    with _log_to_file(lp):
        print("Starting 0 epochs (ablation)...")
        print("no epoch executed")
    assert lp.read_text() == "PRECIOUS"
