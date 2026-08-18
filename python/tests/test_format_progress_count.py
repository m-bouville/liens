"""format_progress_count: large totals display in thousands (raw digits are
hard to scan), small totals stay raw and exact. Threshold is on the total so
the unit never flips partway through a run."""
from utils.logging_utils import format_progress_count


def test_large_total_uses_thousands():
    assert format_progress_count(2801, 27088) == "2.8/27.1 thousand"
    assert format_progress_count(27088, 27088) == "27.1/27.1 thousand"


def test_small_total_stays_raw():
    assert format_progress_count(151, 436) == "151/436"
    assert format_progress_count(1, 436) == "1/436"


def test_threshold_is_on_the_total_not_the_current():
    # just below 10k -> raw; at 10k -> thousands. The unit is chosen from the
    # total, so a run never switches representation midway.
    assert format_progress_count(9999, 9999) == "9999/9999"
    assert format_progress_count(1, 10000) == "0.0/10.0 thousand"
    # current small, total large -> still thousands (no mid-run flip)
    assert "thousand" in format_progress_count(5, 27088)
