"""
A degenerate panel must not kill the whole figure.

matplotlib raises "Data has no positive values, and therefore cannot be
log-scaled" from inside tight_layout(), NOT from set_yscale() -- so an
all-zero or all-negative panel takes down the entire figure at save time,
with a traceback pointing at the layout engine rather than at the panel that
caused it. Reported from a Windows run of
test_check_parameter_dependence_non_default_spatial_size; the same code only
WARNS on this container's matplotlib, which is why it passed here.

Degenerate populations reach this legitimately: a tiny run set, a diagnostic
on an untrained f_theta whose residuals are identically zero, a single-dt
slice. Losing one panel's log scaling beats losing the figure.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from conftest import source_without_comments
from utils.plot_helpers import log_scale_if_positive as _log_scale_if_positive

import pathlib
_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("ydata,expected", [
    ([0.0, 0.0, 0.0], False),
    ([-1.0, -2.0, -3.0], False),
    ([-1.0, 0.0, 0.5], True),
    ([1.0, 2.0, 3.0], True),
    ([float("nan"), 0.0], False),
    ([float("inf"), 0.0], False),
])
def test_log_scale_applied_only_when_positive_data_exists(ydata, expected):
    fig, ax = plt.subplots()
    ax.plot(range(len(ydata)), ydata)
    assert _log_scale_if_positive(ax, axis="y") is expected
    assert (ax.get_yscale() == "log") is expected


def test_the_figure_still_builds_with_an_all_zero_panel():
    """The actual contract: tight_layout() must not raise."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [0.0, 0.0, 0.0])
    _log_scale_if_positive(ax, axis="y")
    fig.tight_layout()          # this is what raised


def test_scatter_only_panels_are_inspected_too():
    """
    GUARDS reading get_lines() alone. scatter() puts its data in
    ax.collections, so a panel drawn ENTIRELY with scatter would look empty
    and silently skip log scaling it deserves -- the oracle/causal overlays in
    the |error| panel are scatters.
    """
    fig, ax = plt.subplots()
    ax.scatter([1.0, 2.0], [3.0, 4.0])
    assert _log_scale_if_positive(ax, axis="y") is True


def test_x_axis_variant_reads_x_data():
    fig, ax = plt.subplots()
    ax.plot([0.0, 0.0], [1.0, 2.0])     # x all zero, y positive
    assert _log_scale_if_positive(ax, axis="x") is False
    assert _log_scale_if_positive(ax, axis="y") is True


def test_no_raw_log_scale_calls_remain_in_the_figure_builders():
    """
    GUARDS a new panel bypassing the guard. Every log axis in this module must
    go through the helper, or the next degenerate population takes the figure
    down again.

    The helper now lives in utils/plot_helpers.py (shared home after the
    diverged-copies reconciliation), so check_parameter_dependence.py is
    scanned WHOLE -- it must contain no raw set_*scale("log") at all -- and
    _plot_helpers.py is scanned excluding the helper's own body (the one
    place a raw call belongs).
    """
    src = source_without_comments(_ROOT / "evaluation/check_parameter_dependence.py")
    for bad in ('set_yscale("log")', 'set_xscale("log")'):
        assert bad not in src, (
            f"a raw {bad} remains in check_parameter_dependence -- route it "
            f"through log_scale_if_positive"
        )
    helpers = source_without_comments(_ROOT / "utils/plot_helpers.py")
    body = helpers[helpers.index("def log_scale_if_positive"):]
    after = body[body.index("\ndef ", 1):] if "\ndef " in body[1:] else ""
    for bad in ('set_yscale("log")', 'set_xscale("log")'):
        assert bad not in after, (
            f"a raw {bad} in _plot_helpers outside the helper itself"
        )


def test_an_empty_axis_is_not_log_scaled():
    fig, ax = plt.subplots()
    assert _log_scale_if_positive(ax, axis="y") is False
