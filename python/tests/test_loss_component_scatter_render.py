"""
Rendering tests for loss_component_scatter.

These cover presentation defects that no correctness test would catch: the plot
was numerically right while showing two curves against three legend entries,
and placing its legend in the only occupied corner. Both were reported from a
real 128x128 run.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest  # noqa: E402

from utils.plots import _iso_total_levels, loss_component_scatter  # noqa: E402


def _histories(n_epochs=4, every_epoch_saves=True):
    """Two components over n_epochs. When every epoch saves, best_so_far is
    IDENTICAL to val -- which is what made one series invisible."""
    val_a = [1.5 - 0.1 * i for i in range(n_epochs)]
    val_b = [2.5 - 0.2 * i for i in range(n_epochs)]
    best_a = list(val_a) if every_epoch_saves else [val_a[0]] * n_epochs
    best_b = list(val_b) if every_epoch_saves else [val_b[0]] * n_epochs
    return {
        "recon0": {"train": [v + 0.3 for v in val_a], "val": val_a, "best_so_far": best_a},
        "stats0": {"train": [v + 0.4 for v in val_b], "val": val_b, "best_so_far": best_b},
    }


def test_best_so_far_is_dashed_so_a_coincident_val_stays_visible(tmp_path):
    """
    GUARDS drawing best_so_far solid. It equals val exactly on every saved
    epoch, and being drawn last at width 2.0 it hid the val line completely --
    two visible curves, three legend entries.

    Asserted against the PRODUCTION source's own series table. An earlier
    version of this test built its own figure and checked that, which passed
    happily while the real function drew a solid line: the test was verifying
    its own fixture, not the code.
    """
    import inspect
    from utils import plots
    src = inspect.getsource(plots.loss_component_scatter)
    row = [l for l in src.splitlines() if '"best_so_far"' in l and "tab:green" in l]
    assert row, "could not find the best_so_far series row"
    assert '"--"' in row[0], f"best_so_far must be dashed, got: {row[0].strip()}"
    val_row = [l for l in src.splitlines() if '"val"' in l and "tab:orange" in l]
    assert val_row and '"-"' in val_row[0]

    # and it still renders
    out = loss_component_scatter([1, 2, 3, 4], _histories(), tmp_path / "s.png", title="t")
    assert out is not None and Path(out).exists()


def test_all_three_series_are_drawn_even_when_two_coincide(tmp_path):
    """The legend promises three; all three must be plotted, not merely
    labelled."""
    import inspect
    from utils import plots
    src = inspect.getsource(plots.loss_component_scatter)
    for key in ("train", "val", "best_so_far"):
        assert f'"{key}"' in src


def test_legend_is_not_pinned_to_a_fixed_corner():
    """
    GUARDS loc="upper right". These trajectories head toward the origin, so
    late in a run the upper right is empty -- but EARLY, when every point is
    still up and to the right, a fixed corner lands the legend on the data
    while three quarters of the axes are empty. That is exactly what a short
    run produces.
    """
    import inspect
    from utils import plots
    src = inspect.getsource(plots.loss_component_scatter)
    assert 'loc="best"' in src
    assert 'loc="upper right"' not in src


@pytest.mark.parametrize("n_levels", [2, 4, 7])
def test_iso_levels_span_the_actual_data_and_are_countable(n_levels):
    """
    The dashed lines are x + y = c for c evenly spaced over the observed range
    of x+y -- data-driven, not a fixed grid, so they stay on screen whatever a
    run's own loss scale happens to be. The COUNT is a fixed default (4).

    n_levels=1 is excluded: np.linspace(lo, hi, 1) returns [lo] by definition,
    so a single level cannot span anything. That is correct behaviour, not a
    bug, but it makes the span assertion meaningless.
    """
    xs, ys = [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]
    levels = _iso_total_levels(xs, ys, n_levels=n_levels)
    assert len(levels) == n_levels
    assert min(levels) == pytest.approx(min(x + y for x, y in zip(xs, ys)))
    assert max(levels) == pytest.approx(max(x + y for x, y in zip(xs, ys)))


def test_default_iso_level_count_is_four():
    """The count is a fixed default, not derived from the data -- worth
    pinning because the SPACING is data-driven and the two are easily
    confused when reading the figure."""
    import inspect
    assert inspect.signature(_iso_total_levels).parameters["n_levels"].default == 4
    assert len(_iso_total_levels([1.0, 3.0], [1.0, 1.0])) == 4


def test_iso_levels_are_empty_rather_than_raising_on_no_data():
    assert _iso_total_levels([], []) == []
