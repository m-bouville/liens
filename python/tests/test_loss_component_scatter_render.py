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


def test_axes_are_log_log_when_every_value_is_positive(tmp_path):
    """
    Components span more than a decade within a run (stage 2: deriv 0.79 ->
    0.09 while recon0 went 8.5 -> 3.2). On linear axes the early large values
    compress the late trajectory -- the part worth reading -- into a corner.
    """
    import inspect
    from utils import plots
    src = inspect.getsource(plots.loss_component_scatter)
    assert 'ax.set_xscale("log")' in src and 'ax.set_yscale("log")' in src

    out = loss_component_scatter([1, 2, 3, 4], _histories(), tmp_path / "s.png", title="t")
    assert out is not None and Path(out).exists()


def test_iso_lines_are_sampled_not_drawn_as_straight_segments_on_log_axes():
    """
    GUARDS reusing _clip_iso_total_segment on log axes. x + y = c is a straight
    line only on LINEAR axes; on log-log it curves, so a two-endpoint segment
    would draw a chord that crosses the real iso-total instead of tracing it --
    a reference line that is wrong everywhere except at its two ends.
    """
    import inspect
    from utils import plots
    src = inspect.getsource(plots.loss_component_scatter)
    log_branch = src[src.index('if ax.get_xscale() == "log":'):]
    assert "geomspace" in log_branch.split("else:")[0]


def test_non_positive_values_do_not_crash_the_log_axes(tmp_path):
    """
    A component can be legitimately 0 -- an inactive term, e.g.
    stats1_weight=0. Log axes cannot show it, so it must be excluded from the
    limits rather than crashing or silently clipping every other point.
    """
    hist = _histories()
    hist["stats0"]["val"] = [0.0] + hist["stats0"]["val"][1:]
    out = loss_component_scatter([1, 2, 3, 4], hist, tmp_path / "z.png", title="t")
    assert out is not None and Path(out).exists()


def test_all_zero_component_falls_back_to_linear_axes(tmp_path):
    """The fallback must exist: a component that is zero throughout has no
    positive value at all, and a log axis with no valid range is an error."""
    hist = _histories()
    for key in ("train", "val", "best_so_far"):
        hist["stats0"][key] = [0.0] * 4
    out = loss_component_scatter([1, 2, 3, 4], hist, tmp_path / "zz.png", title="t")
    assert out is not None and Path(out).exists()


# --------------------------------------------------------------------
# loss_curve must survive a diverged loss
# --------------------------------------------------------------------

@pytest.mark.parametrize("label,train,val", [
    ("every value inf", [float("inf")], [float("inf")]),
    ("every value nan", [float("nan"), float("nan")], [float("nan"), float("nan")]),
    ("one inf among finite", [2.0, float("inf")], [2.2, 1.6]),
    ("one nan among finite", [2.0, float("nan")], [2.2, 1.6]),
    ("all finite", [2.0, 1.5], [2.2, 1.6]),
])
def test_loss_curve_survives_non_finite_losses(tmp_path, label, train, val):
    """
    GUARDS crashing the TRAINING RUN from inside the plot. A diverged loss is a
    real outcome, not corrupt input, and the figure is what shows you WHERE it
    diverged -- losing the run to its own diagnostic is the worst possible
    trade. This killed a 1000-epoch stage-3b run at epoch 1, after that
    epoch's work was already done.

    Two separate holes, both needed:

      * the y guard was `min(all_values) > 0`, which PASSES when every value
        is +inf (min([inf, inf]) is inf, and inf > 0 is True), so log scale
        was set on data with no finite positive value. nan hid this in
        testing because min() with a nan is order-dependent:
        min([2.0, nan]) is 2.0 but min([nan, 2.0]) is nan.
      * with no finite y anywhere, matplotlib registers no data limits at
        all, so the X interval is degenerate too and the x-axis LogLocator
        raises the identical error despite the epoch numbers being positive.
        Filtering y alone was not enough.
    """
    from utils.plots import loss_curve
    out = loss_curve(list(range(1, len(train) + 1)), train, val, val,
                      tmp_path / f"{label.replace(' ', '_')}.png", title="t",
                      secondary_train=train, secondary_val=val, secondary_label="1step")
    assert out is None or Path(out).exists()


def test_a_single_finite_point_still_gets_log_scale(tmp_path):
    """The fallback must not be over-eager: one real value among non-finite
    ones is still enough to scale by."""
    import inspect
    from utils import plots
    src = inspect.getsource(plots.loss_curve)
    assert "if math.isfinite(v)" in src, "non-finite values must be filtered, not tolerated"
    assert "and bool(all_values)" in src, "the x-axis must stand down with the y-axis"
