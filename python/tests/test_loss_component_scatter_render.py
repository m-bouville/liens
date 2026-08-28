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

from utils.plots import _iso_total_levels, loss_component_scatter, rollout_vs_1step_scatter  # noqa: E402


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
                assert panel.get_visible()
                # labels live on the margins only (see the dedicated test);
                # here just assert the margins carry the RIGHT names
                if r == 2:
                    assert panel.get_xlabel() == names[c]
                if c == 0:
                    assert panel.get_ylabel() == names[r + 1]


def test_every_pair_appears_exactly_once(monkeypatch, tmp_path):
    """The triangle must not drop or duplicate a pair as the component count
    changes -- n components give n(n-1)/2 panels."""
    for names in (["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d"],
                   ["a", "b", "c", "d", "e"]):
        ax = _grid(monkeypatch, names, tmp_path)
        # Read the pairing from the GRID POSITION, not the axis labels:
        # interior panels deliberately carry no label text (corner-plot
        # convention), so labels identify only the margins. Position is the
        # structural fact -- row r is y=names[r+1], column c is x=names[c].
        seen = [(names[c], names[r + 1])
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


def test_all_axes_share_one_square_range_for_45deg_iso_lines(monkeypatch, tmp_path):
    """
    Every axis of every panel uses ONE shared [lo, hi], computed as the
    proportional limits over ALL scaled components at once (min over all
    vars, max over all vars). This is what makes the iso-total line x+y=c
    render at 45 deg and panels directly comparable: an iso-line is only a
    45-deg VISUAL line when both axes cover the same interval per unit
    length. (Superseded the former per-axis proportional fit, which tilted
    the iso-lines flat whenever two components differed in magnitude -- the
    reported near-horizontal-iso-line bug. The trade-off, deliberately
    accepted: a narrow-spread component sharing the plot with a wide-spread
    one no longer fills its own axis; its small motion is a small motion,
    read against the common scale rather than magnified.)
    """
    import matplotlib.pyplot as plt
    from utils.plots import loss_component_scatter

    captured = {"axes": []}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, ax = original(*args, **kwargs)
        captured["axes"].append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy)
    # three components with very different magnitudes -- recon0 ~1, stats0
    # ~0.44 (both narrow), deriv spanning 1..70 (wide), like a real 2a run.
    hist = {
        "recon0": {"train": [1.06, 1.03], "val": [1.06, 1.03], "best_so_far": [1.06, 1.03]},
        "stats0": {"train": [0.44, 0.44], "val": [0.44, 0.44], "best_so_far": [0.44, 0.44]},
        "deriv":  {"train": [70.0, 1.06], "val": [70.0, 1.06], "best_so_far": [70.0, 1.06]},
    }
    loss_component_scatter([1, 2], hist, tmp_path / "s.png")
    grid = captured["axes"][0]
    n = grid.shape[0]

    # collect every visible panel's x and y limits
    seen = []
    for r in range(n):
        for c in range(n):
            ax = grid[r][c]
            if not ax.get_visible():
                continue
            seen.append((ax.get_xlim(), ax.get_ylim()))
    assert seen, "no visible panels"

    ref_x, ref_y = seen[0]
    for xlim, ylim in seen:
        # x range == y range on every panel (square)
        assert xlim == pytest.approx(ylim, rel=1e-9), (
            f"axis not square: xlim={xlim} ylim={ylim}")
        # and identical across every panel (shared)
        assert xlim == pytest.approx(ref_x, rel=1e-9), (
            f"x range differs across panels: {xlim} vs {ref_x}")
        assert ylim == pytest.approx(ref_y, rel=1e-9), (
            f"y range differs across panels: {ylim} vs {ref_y}")

    # the shared range must actually span the global data (min ~0.44 .. max ~70),
    # not be fitted to any single component
    lo, hi = ref_x
    assert lo < 0.44 and hi > 70.0, (
        f"shared range [{lo:.3f}, {hi:.3f}] does not enclose all components "
        f"(expected to span the global min 0.44 and max 70.0)")


def test_only_the_margins_carry_axis_labels(monkeypatch, tmp_path):
    """Corner-plot convention: a column shares its x quantity and a row its
    y, so interior labels are pure clutter. In a lower triangle the margins
    are exactly the last row and the first column."""
    names = ["recon0", "stats0", "deriv", "interp"]
    ax = _grid(monkeypatch, names, tmp_path)
    n = ax.shape[0]
    for r in range(n):
        for c in range(r + 1):
            panel = ax[r][c]
            assert bool(panel.get_xlabel()) == (r == n - 1), (r, c, "xlabel")
            assert bool(panel.get_ylabel()) == (c == 0), (r, c, "ylabel")


def test_proportional_limits_handles_a_single_value():
    """One point has zero spread; the floor must keep the limits from
    collapsing to a singular range."""
    from utils.plots import _proportional_limits
    lo, hi = _proportional_limits([2.0])
    assert lo < 2.0 < hi
    lo, hi = _proportional_limits([2.0, 2.0])
    assert lo < 2.0 < hi


def test_ref_components_drawn_as_purple_circle(monkeypatch, tmp_path):
    """
    ref_components (the pre-run baseline -- the log's 'ref|' line) is drawn as
    a purple open circle on every panel, so the run's trajectory reads
    RELATIVE to where it resumed from, not just relative to its own epoch 1.
    Also: the ref point joins the shared-range computation, so it is always
    in-frame (a baseline drawn off the axes would be worse than none).
    """
    import matplotlib.pyplot as plt
    from utils.plots import loss_component_scatter

    captured = {}
    original = plt.subplots

    def spy(*a, **k):
        fig, ax = original(*a, **k)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy)
    hist = {
        "recon0": {"train": [0.9, 0.8], "val": [0.73, 0.68], "best_so_far": [0.73, 0.68]},
        "deriv":  {"train": [0.43, 0.87], "val": [0.44, 0.73], "best_so_far": [0.44, 0.44]},
    }
    # ref recon0 is LARGER than any plotted recon0 -> it also exercises the
    # "ref must widen the shared range to stay in-frame" path.
    ref = {"recon0": 1.03, "deriv": 0.47}
    loss_component_scatter([1, 2], hist, tmp_path / "c.png", ref_components=ref)

    grid = captured["ax"]
    purple_pts = []
    for row in grid:
        for ax in row:
            if not ax.get_visible():
                continue
            for coll in ax.collections:
                ec = coll.get_edgecolors()
                if len(ec) and abs(ec[0][0] - 0.5804) < 0.03 and abs(ec[0][2] - 0.7412) < 0.03:
                    off = coll.get_offsets()
                    if len(off):
                        purple_pts.append(tuple(off[0]))
    assert purple_pts, "no purple ref circle drawn on any panel"
    # the ref point sits at (ref[name_x], ref[name_y]); for the recon0-vs-deriv
    # panel that is (1.03, 0.47) in some axis order
    got = purple_pts[0]
    assert (abs(got[0] - 1.03) < 1e-6 and abs(got[1] - 0.47) < 1e-6) or \
           (abs(got[0] - 0.47) < 1e-6 and abs(got[1] - 1.03) < 1e-6), \
        f"purple circle not at the ref coordinates: {got}"

    # and it is IN-FRAME on that panel (range widened to include it)
    for row in grid:
        for ax in row:
            if ax.get_visible() and ax.collections:
                lo, hi = ax.get_xlim()
                assert lo <= 1.03 <= hi, f"ref x=1.03 out of frame [{lo},{hi}]"
                break


def test_ref_components_none_is_a_no_op(monkeypatch, tmp_path):
    """No ref_components (the default) draws no purple circle and does not
    error -- the feature is purely additive."""
    import matplotlib.pyplot as plt
    from utils.plots import loss_component_scatter

    captured = {}
    original = plt.subplots
    monkeypatch.setattr(plt, "subplots",
                        lambda *a, **k: (lambda fg, ax: (captured.__setitem__("ax", ax), (fg, ax))[1])(*original(*a, **k)))
    hist = {
        "recon0": {"train": [0.9, 0.8], "val": [0.73, 0.68], "best_so_far": [0.73, 0.68]},
        "deriv":  {"train": [0.43, 0.87], "val": [0.44, 0.73], "best_so_far": [0.44, 0.44]},
    }
    loss_component_scatter([1, 2], hist, tmp_path / "c.png")  # no ref
    for row in captured["ax"]:
        for ax in row:
            for coll in ax.collections:
                ec = coll.get_edgecolors()
                if len(ec):
                    assert not (abs(ec[0][0] - 0.5804) < 0.03 and abs(ec[0][2] - 0.7412) < 0.03), \
                        "purple circle drawn when ref_components was None"


def test_sub_decade_range_still_shows_axis_numbers(monkeypatch, tmp_path):
    """
    REGRESSION: on a converged run every component sits in a narrow sub-decade
    window (e.g. 0.4..1.0), so the log axes cross NO power of ten and
    LogLocator places no major (decade) ticks. The minor-tick labels must NOT
    be blanked in that case, or the axes show no numbers at all (the reported
    'not a single number on axes' bug). Blank minors only when a major tick is
    actually in view.
    """
    import matplotlib.pyplot as plt
    from utils.plots import loss_component_scatter

    captured = {}
    original = plt.subplots

    def spy(*a, **k):
        fig, ax = original(*a, **k)
        captured["ax"] = ax
        return fig, ax
    monkeypatch.setattr(plt, "subplots", spy)

    # all components strictly between 0.4 and 0.7 -- the shared range then
    # excludes 1.0, so NO axis crosses a power of ten (the real failure
    # condition; a range that happens to include 1.0 has a major tick and
    # would mask the bug).
    hist = {
        "recon0": {"train": [0.62, 0.60], "val": [0.61, 0.59], "best_so_far": [0.61, 0.59]},
        "deriv":  {"train": [0.43, 0.42], "val": [0.44, 0.45], "best_so_far": [0.44, 0.44]},
    }
    loss_component_scatter([1, 2], hist, tmp_path / "c.png",
                           ref_components={"recon0": 0.65, "deriv": 0.47})

    grid = captured["ax"]
    n = grid.shape[0]
    bl = grid[n - 1][0]           # bottom-left panel: has both axis labels
    grid[0][0].figure.canvas.draw()

    def any_numbers(ax):
        # count only labels whose TICK falls inside the axis limits -- an
        # out-of-view decade label object exists but does not render, so it
        # must not count as 'the axis has numbers'.
        def in_view(axis, get_labels, lo, hi, minor):
            locs = axis.get_minorticklocs() if minor else axis.get_majorticklocs()
            labels = get_labels(minor=minor)
            return [lb.get_text() for loc, lb in zip(locs, labels)
                    if lo <= loc <= hi and lb.get_text().strip()]
        xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
        x = (in_view(ax.xaxis, ax.get_xticklabels, xlo, xhi, False)
             + in_view(ax.xaxis, ax.get_xticklabels, xlo, xhi, True))
        y = (in_view(ax.yaxis, ax.get_yticklabels, ylo, yhi, False)
             + in_view(ax.yaxis, ax.get_yticklabels, ylo, yhi, True))
        return bool(x), bool(y)

    has_x, has_y = any_numbers(bl)
    assert has_x, "x-axis has no numbers on a sub-decade range (minor labels wrongly blanked)"
    assert has_y, "y-axis has no numbers on a sub-decade range (minor labels wrongly blanked)"


# --------------------------------------------------------------------
# rollout_vs_1step_scatter (stage-3b L_rollout vs L_1step, saved epochs)
# --------------------------------------------------------------------

def test_rollout_vs_1step_renders_train_and_valid(tmp_path):
    """Both series draw on shared log-log square axes; returns the path."""
    n = 6
    epochs = [25 * (i + 1) for i in range(n)]
    l_rollout_v = [15.0 / (i + 1) for i in range(n)]
    l_1step_v = [0.5 + 0.05 * i for i in range(n)]
    l_rollout_t = [14.0 / (i + 1) for i in range(n)]
    l_1step_t = [0.48 + 0.05 * i for i in range(n)]
    out = rollout_vs_1step_scatter(
        l_1step_v, l_rollout_v, tmp_path / "rv1.png",
        title="s", saved_epochs=epochs,
        l_1step_train=l_1step_t, l_rollout_train=l_rollout_t)
    assert out is not None and Path(out).exists()

    import inspect
    from utils import plots
    src = inspect.getsource(plots.rollout_vs_1step_scatter)
    # log-log SQUARE axes are the whole point of the plot
    assert 'set_xscale("log")' in src and 'set_yscale("log")' in src
    assert 'set_aspect("equal")' in src
    # both series are labelled (train + valid), matching loss_curve's convention
    assert '"train"' in src and '"valid"' in src


def test_rollout_vs_1step_valid_only_still_renders(tmp_path):
    """train series is optional (None) -- val alone must still draw."""
    out = rollout_vs_1step_scatter(
        [0.5, 0.55, 0.6], [10.0, 5.0, 2.0], tmp_path / "vo.png")
    assert out is not None and Path(out).exists()


def test_rollout_vs_1step_needs_two_points(tmp_path):
    """Fewer than 2 finite saved points -> nothing to trade off -> None,
    writes nothing (a 1-step 3a, or a run that never saved twice)."""
    assert rollout_vs_1step_scatter([1.0], [2.0], tmp_path / "one.png") is None
    # non-finite / non-positive entries are filtered before the count
    import math
    assert rollout_vs_1step_scatter(
        [1.0, math.inf, -1.0], [2.0, 3.0, 4.0], tmp_path / "nf.png") is None
