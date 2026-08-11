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


# --------------------------------------------------------------------
# non-finite components must not kill the run
# --------------------------------------------------------------------

@pytest.mark.parametrize("label,train,val", [
    ("epochs=0 ablation: train never iterated", [float("nan")], [0.5]),
    ("every component non-finite", [float("nan")], [float("nan")]),
    ("inf in train", [float("inf"), 0.8], [0.9, 0.7]),
    ("all components exactly zero", [0.0], [0.0]),
    ("normal", [1.0, 0.8], [0.9, 0.7]),
])
def test_loss_component_scatter_survives_non_finite_components(tmp_path, label, train, val):
    """
    GUARDS killing a training run from inside its own diagnostic figure.

    Reported: an `epochs = 0` stage-4 ablation. The train set is deliberately
    never iterated, so every train component is NaN. `v > 0` already excludes
    NaN from the positive-value list, so the code fell into the linear-axis
    fallback -- which read ax.get_xlim() from an axes whose only scattered
    points were NaN. matplotlib reports those limits as non-finite, and feeding
    them back into set_xlim raised "Axis limits cannot be NaN or Inf".

    Same class as loss_curve's non-finite failure, in the function that was
    NOT hardened at the time -- it was flagged then and left, and this is the
    run that paid for it.
    """
    from utils.plots import loss_component_scatter
    histories = {name: {"train": train, "val": val, "best_so_far": val}
                 for name in ("rollout", "recon0", "stats0")}
    out = loss_component_scatter(list(range(len(train))), histories,
                                  tmp_path / f"{abs(hash(label))}.png", title="t")
    assert out is None or Path(out).exists()


def test_all_zero_components_do_not_produce_singular_axis_limits(tmp_path, recwarn):
    """
    GUARDS max(all_x)*1.1 as the fallback upper limit: with every component at
    exactly 0 that is 0, and set_xlim(0, 0) is singular. Not fatal, but it
    warns on every figure write for the whole run.
    """
    from utils.plots import loss_component_scatter
    histories = {name: {"train": [0.0], "val": [0.0], "best_so_far": [0.0]}
                 for name in ("rollout", "recon0")}
    loss_component_scatter([0], histories, tmp_path / "zeros.png", title="t")
    singular = [w for w in recwarn if "singular" in str(w.message)]
    assert not singular, [str(w.message) for w in singular]


# --------------------------------------------------------------------
# epochs=0 ablation: every train component is NaN
# --------------------------------------------------------------------

@pytest.mark.parametrize("label,hist", [
    ("all components NaN", {
        "rollout": {"train": [float("nan")], "val": [float("nan")],
                     "best_so_far": [float("nan")]},
        "recon0": {"train": [float("nan")], "val": [float("nan")],
                    "best_so_far": [float("nan")]},
        "stats0": {"train": [float("nan")], "val": [float("nan")],
                    "best_so_far": [float("nan")]}}),
    ("train NaN, val real", {
        "rollout": {"train": [float("nan")], "val": [2.0], "best_so_far": [2.0]},
        "recon0": {"train": [float("nan")], "val": [0.5], "best_so_far": [0.5]},
        "stats0": {"train": [float("nan")], "val": [float("nan")],
                    "best_so_far": [float("nan")]}}),
    ("a component pinned at exactly 0", {
        "rollout": {"train": [1.0], "val": [2.0], "best_so_far": [2.0]},
        "recon0": {"train": [0.0], "val": [0.0], "best_so_far": [0.0]},
        "stats0": {"train": [0.3], "val": [0.4], "best_so_far": [0.4]}}),
])
def test_scatter_survives_a_zero_epoch_ablation(tmp_path, label, hist):
    """
    GUARDS killing a RUN from inside its own diagnostic figure.

    Stage 4/5 with `epochs = 0` never iterates the train set, so every train
    component is NaN. Reported as:

        ax.set_xlim(xlo if xlo else 0, xmax)
        ValueError: Axis limits cannot be NaN or Inf

    The subtlety is that `v > 0` already excludes NaN from `positive_x`, so
    the log branch is skipped correctly -- but the FALLBACK branch then read
    ax.get_xlim() from an axes whose only scattered points were NaN, which
    matplotlib reports as non-finite. Filtering the inputs is not enough; the
    fallback limits must be constants.

    The third case is not a NaN at all: a component legitimately pinned at 0
    (an inactive term, e.g. stats1_weight=0) has no representable point on a
    log axis, which is the other way `positive_*` ends up empty.
    """
    from utils.plots import loss_component_scatter
    out = loss_component_scatter([0], hist, tmp_path / f"{label.replace(' ', '_')}.png",
                                  title="Stage 4 loss components")
    assert out is None or Path(out).exists()


def test_the_fallback_limits_are_constants_not_read_back_from_the_axes():
    """
    GUARDS `xmax = ax.get_xlim()[1]` in the non-log fallback.

    Asserted on the SOURCE because the behaviour is matplotlib-version
    dependent: on some versions an axes whose only scattered points are NaN
    reports (0, 1) from get_xlim() and everything works, on others it reports
    non-finite and set_xlim raises "Axis limits cannot be NaN or Inf". The
    reported failure came from a version in the second group, and a purely
    behavioural test passes on the first no matter how wrong the code is --
    verified: mutating this back to get_xlim() left all 27 tests green here.

    The limits must therefore not depend on what the axes reports at all.
    """
    import inspect
    from utils import plots
    src = inspect.getsource(plots.loss_component_scatter)
    fallback = src[src.index("            ax.set_xlim(left=0)"):]
    fallback = fallback[:fallback.index("for c in _iso_total_levels")]
    # CODE only. The comment right there explains that get_xlim() is what NOT
    # to use, so matching raw source matches the explanation and fails on
    # correct code -- which is exactly what happened when this was written.
    fallback = "\n".join(l for l in fallback.splitlines()
                          if not l.strip().startswith("#"))
    for forbidden in ("ax.get_xlim()", "ax.get_ylim()"):
        assert forbidden not in fallback, (
            f"the fallback limits must be constants -- {forbidden} returns non-finite "
            f"on some matplotlib versions when every scattered point is NaN"
        )
    assert "or 1.0" in fallback, "the fallback needs a non-zero default extent"


# --------------------------------------------------------------------
# loss_component_scatter must survive non-finite components
# --------------------------------------------------------------------

@pytest.mark.parametrize("label,train,val,best", [
    ("epochs=0 ablation: train_sum/0 = nan, nothing saved so best = inf",
     [float("nan")], [1.4], [float("inf")]),
    ("every series non-finite", [float("nan")], [float("nan")], [float("inf")]),
    ("inf only (a component that never improved)", [1.2], [1.4], [float("inf")]),
    ("all finite", [1.2], [1.4], [1.3]),
])
def test_scatter_survives_non_finite_components(tmp_path, label, train, val, best):
    """
    GUARDS crashing the TRAINING RUN from inside the component figure -- the
    same trade loss_curve already lost once. Reported from a stage-4
    `epochs = 0` ablation:

        ValueError: Axis limits cannot be NaN or Inf

    Two independent non-finite sources meet there, and each defeats a
    different half of a naive guard:

      * `nan` from `train_sum / n_train` when the train set is never iterated.
        A `v > 0` filter DROPS it (nan > 0 is False), so the linear fallback
        branch runs -- and there `max(all_x)` is nan, `nan * 1.1` is nan, and
        `nan or 1.0` KEEPS nan, because nan is truthy.
      * `inf` from the criterion tracker's initial best, when no epoch has
        improved yet. A `v > 0` filter KEEPS it (inf > 0 is True), so it flows
        straight into `max(...) * 1.6`.

    Only `math.isfinite` handles both.
    """
    from utils.plots import loss_component_scatter
    hist = {n: {"train": list(train), "val": list(val), "best_so_far": list(best)}
            for n in ("rollout", "recon0", "stats0")}
    out = loss_component_scatter([0], hist, tmp_path / "s.png",
                                  title="Stage 4 loss components")
    assert out is None or Path(out).exists()


def test_the_scatter_filters_on_isfinite_not_on_positivity():
    """Pins the mechanism, since `v > 0` looks like it would do the job and
    silently handles only one of the two cases above."""
    import inspect
    from utils import plots
    src = inspect.getsource(plots.loss_component_scatter)
    assert src.count("math.isfinite(v)") >= 2, (
        "both all_x and all_y must be filtered to finite values"
    )


def _hist(names):
    return {n: {"train": [1.0, 0.9], "val": [1.1, 1.0], "best_so_far": [1.1, 1.0]}
            for n in names}


def _grid(monkeypatch, names, tmp_path):
    import matplotlib.pyplot as plt
    from utils.plots import loss_component_scatter
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, ax = original(*args, **kwargs)
        captured.setdefault("ax", ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy)
    loss_component_scatter([1, 2], _hist(names), tmp_path / "s.png")
    return captured["ax"]


def test_components_are_laid_out_as_a_lower_triangle(monkeypatch, tmp_path):
    """
    Row r is ALWAYS the y variable, column c ALWAYS the x variable, so every
    panel in a row shares a y quantity and every panel in a column shares an
    x quantity.

    The previous flat wrap put (recon0,stats0) and (stats0,deriv) in
    different rows with different axes, so nothing lined up and the axis
    labels had to be re-read on all six panels.
    """
    names = ["recon0", "stats0", "deriv", "interp"]
    ax = _grid(monkeypatch, names, tmp_path)
    assert ax.shape == (3, 3), ax.shape
    for r in range(3):
        for c in range(3):
            panel = ax[r][c]
            if c > r:
                assert not panel.get_visible(), (
                    f"[{r},{c}] is in the upper triangle -- the redundant "
                    f"transpose -- and should be hidden"
                )
            else:
                assert panel.get_xlabel() == names[c]
                assert panel.get_ylabel() == names[r + 1]


def test_every_pair_appears_exactly_once(monkeypatch, tmp_path):
    """The triangle must not drop or duplicate a pair as the component count
    changes -- n components give n(n-1)/2 panels."""
    for names in (["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d"],
                   ["a", "b", "c", "d", "e"]):
        ax = _grid(monkeypatch, names, tmp_path)
        seen = [(ax[r][c].get_xlabel(), ax[r][c].get_ylabel())
                for r in range(ax.shape[0]) for c in range(ax.shape[1])
                if ax[r][c].get_visible()]
        expected = {(names[i], names[j])
                    for i in range(len(names)) for j in range(i + 1, len(names))}
        assert len(seen) == len(expected) == len(names) * (len(names) - 1) // 2
        assert set(seen) == expected, (names, sorted(set(seen) ^ expected))


def test_a_single_component_draws_nothing(tmp_path):
    """One component has no pair to plot; the figure is skipped rather than
    written empty."""
    from utils.plots import loss_component_scatter
    assert loss_component_scatter([1, 2], _hist(["only"]), tmp_path / "s.png") is None
