"""
Tests for the shared plotting helpers in utils.plot_helpers.

moving_window and pretty_label were lifted here this session to be the SINGLE
source for two consumers (compare_f_theta's temperature/statistics panels and
check_latent_channels' per-channel importance plots). They were previously local
to compare_f_theta and covered only indirectly; tested here at their real home so
a future edit/move is guarded where the code actually lives, not via an alias.

Run from python/ (imports rely on that root being on sys.path).
"""
import numpy as np

from utils.plot_helpers import moving_window, pretty_label


# --------------------------------------------------------------------------- #
# moving_window: SMA over DISTINCT x values (not fixed bins)
# --------------------------------------------------------------------------- #
def test_moving_window_returns_five_arrays_and_empty_on_no_data():
    """The contract is five arrays (x, median, q25, q75, N) on EVERY path --
    a caller unpacks five, so the empty case must return five, not four."""
    out = moving_window(np.array([]), np.array([]))
    assert len(out) == 5
    assert all(a.size == 0 for a in out)


def test_moving_window_centres_on_distinct_values_with_full_windows():
    """With half_width=1 and 5 distinct x, only the INTERIOR 3 values get a full
    window (the first/last are trimmed -- they could only draw on a truncated
    set). Centres land on real distinct x values, not arbitrary bin edges."""
    x = np.array([0.6, 0.6, 0.65, 0.7, 0.75, 0.8])   # 5 distinct: .6 .65 .7 .75 .8
    v = np.arange(6.0)
    cx, med, lo, hi, n = moving_window(x, v, half_width=1)
    assert list(cx) == [0.65, 0.70, 0.75]            # interior only (trim=1 each end)
    assert cx.size == med.size == lo.size == hi.size == n.size


def test_moving_window_counts_reflect_window_population():
    """N is how many samples each point rests on -- so a point drawing on two
    values can be told from one drawing on fifty. The .6 value has 2 samples;
    with half_width=1 the .65-centred window spans {.6,.65,.7} = 2+1+1 = 4."""
    x = np.array([0.6, 0.6, 0.65, 0.7, 0.75, 0.8])
    v = np.arange(6.0)
    cx, _med, _lo, _hi, n = moving_window(x, v, half_width=1)
    assert n[0] == 4                                  # window {.6,.65,.7}: 2+1+1


def test_moving_window_uses_median_and_quartiles():
    """MEDIAN (not mean) with a q25..q75 band -- one outlier must not drag the
    plotted point outside its own band."""
    # one x value, many samples with an outlier: median robust, mean would not be
    x = np.zeros(11)
    v = np.array([1.0] * 10 + [1000.0])
    cx, med, lo, hi, n = moving_window(x, v, half_width=0)
    assert med[0] == 1.0                              # median ignores the outlier
    assert lo[0] <= med[0] <= hi[0]                   # median bracketed by the band


def test_moving_window_falls_back_when_too_few_distinct_values():
    """Fewer than 2*half_width+1 distinct values (e.g. a single-temperature
    sweep) -> no full window exists; rather than an empty panel, every value is
    kept with whatever partial window it has."""
    x = np.array([0.7, 0.7, 0.7])                     # one distinct value
    v = np.array([1.0, 2.0, 3.0])
    cx, med, _lo, _hi, n = moving_window(x, v, half_width=2)
    assert cx.size == 1 and cx[0] == 0.7
    assert n[0] == 3


def test_moving_window_drops_nonfinite():
    """nan/inf in x or values are filtered before windowing."""
    x = np.array([0.6, np.nan, 0.65, 0.7, 0.75, 0.8, 0.85])
    v = np.array([1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0])
    cx, med, _lo, _hi, _n = moving_window(x, v, half_width=1)
    assert np.all(np.isfinite(cx)) and np.all(np.isfinite(med))


# --------------------------------------------------------------------------- #
# pretty_label: checkpoint stem -> human timestamp
# --------------------------------------------------------------------------- #
def test_pretty_label_parses_timestamp():
    assert pretty_label("stage 2-20260812_20h08") == "stage 2 (12/08 at 20:08)"


def test_pretty_label_year_toggle():
    assert pretty_label("stage 2-20251201_09h00", include_year=True) == \
        "stage 2 (01/12/2025 at 09:00)"
    assert pretty_label("stage 2-20251201_09h00", include_year=False) == \
        "stage 2 (01/12 at 09:00)"                                    # no year


def test_pretty_label_passthrough_without_timestamp():
    """A label with no -YYYYMMDD_HHhMM stamp is returned unchanged."""
    assert pretty_label("stage 3a") == "stage 3a"


def test_pretty_label_finds_timestamp_amid_extra_text():
    """The check_latent_channels stem carries a size prefix and a trailing
    '-latent_channels'; pretty_label locates the timestamp regardless and keeps
    everything before it as the stage part."""
    out = pretty_label("128x128-stage1-20260910_06h19-latent_channels")
    assert out == "128x128-stage1 (10/09 at 06:19)"
