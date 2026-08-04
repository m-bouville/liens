"""
Tests for utils/plots.py's loss_curve(). matplotlib IS available in
this environment (unlike torch), so these actually run and produce
real files, checked directly -- not just syntax-checked.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_plots.py -v
"""
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed for tests

import matplotlib.pyplot as plt
import numpy as np

from utils import plots  # NOT "import utils.plots" -- that binds 'utils'
                           # in this namespace, not 'plots', so
                           # plots.loss_curve(...) below would raise
                           # NameError: name 'plots' is not defined.


def test_produces_a_real_file(tmp_path):
    epochs = [1, 2, 3]
    train_loss = [1.0, 0.8, 0.6]
    val_loss = [1.1, 0.9, 0.7]
    best_so_far = [1.1, 0.9, 0.7]
    out = tmp_path / "loss_curve.png"

    result = plots.loss_curve(epochs, train_loss, val_loss, best_so_far, out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_y_axis_capped_at_99th_percentile_of_all_curves(tmp_path, monkeypatch):
    """The core behavior this module exists for: a huge, rare spike must
    not stretch the y-axis so far that the other 99% of the run's data
    gets squashed into an unreadable flat line -- checked against the
    ACTUAL rendered axis limits (captured via the real plt.subplots call
    the function makes), not just independently recomputed and trusted.

    Uses a realistically-sized run (100 epochs, 300 concatenated values
    across train/val/best_so_far) with 2 rare spike epochs -- a tiny
    handful of points (as in the earlier 5-epoch version of this test)
    isn't enough for a 99th-percentile cut to mean anything (with n=15,
    the 99th percentile sits almost AT the max, barely excluding
    anything); this needs enough points that the top ~1% is a genuinely
    small, separable slice, matching how this actually gets used (a
    real training run with hundreds of epochs)."""
    n = 100
    epochs = list(range(1, n + 1))
    train_loss = [10.0 - 0.05 * i for i in range(n)]
    val_loss = [11.0 - 0.05 * i for i in range(n)]
    best_so_far = [11.0 - 0.05 * i for i in range(n)]
    train_loss[49] = 5000.0  # one rare, huge spike epoch
    val_loss[49] = 6000.0
    out = tmp_path / "loss_curve.png"

    captured_axes = {}
    real_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured_axes["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)

    plots.loss_curve(epochs, train_loss, val_loss, best_so_far, out)

    ax = captured_axes["ax"]
    ymin, ymax = ax.get_ylim()
    all_values = train_loss + val_loss + best_so_far
    expected_cap = 1.5 * np.percentile(all_values, 99)
    assert ymax == expected_cap, f"expected y-axis top exactly {expected_cap}, got {ymax}"
    # NOT ymin == 0: all values here are positive, so loss_curve() now
    # uses a log-scale y-axis by default (see its own docstring) --
    # log(0) is undefined, so matplotlib floors to the smallest
    # positive value actually present instead of exactly 0.
    assert ymin > 0
    assert ymax < max(val_loss), "the cap should be far below the actual spike -- confirms it's a real cap"


def test_y_axis_not_capped_when_nothing_exceeds_the_percentile(tmp_path, monkeypatch):
    """The companion behavior this test guards: a well-behaved run where
    the top 1% of values already IS the observed max (no real outlier)
    should NOT have its axis artificially stretched to a cap that
    equals the max anyway -- it should auto-scale normally instead, same
    rendered-axis verification approach as the capped case above."""
    epochs = list(range(1, 6))
    train_loss = [10.0, 8.0, 6.0, 5.0, 4.0]  # smooth decline, no spike
    val_loss = [11.0, 9.0, 7.0, 6.0, 5.0]
    best_so_far = [11.0, 9.0, 7.0, 6.0, 5.0]
    out = tmp_path / "loss_curve.png"

    captured_axes = {}
    real_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured_axes["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)

    plots.loss_curve(epochs, train_loss, val_loss, best_so_far, out)

    ax = captured_axes["ax"]
    ymin, ymax = ax.get_ylim()
    all_values = train_loss + val_loss + best_so_far
    # With this data, np.percentile(all_values, 99) == max(all_values)
    # (no real outlier to trim), so the cap is never engaged -- matplotlib
    # auto-scales instead, which pads slightly ABOVE the true max rather
    # than clamping exactly to it.
    assert ymax >= max(all_values), (
        f"axis should auto-scale to at least the real max ({max(all_values)}), got ymax={ymax}"
    )
    # NOT ymin == 0 -- see the identical comment in the capped test above.
    assert ymin > 0


def test_secondary_lines_do_not_crash_and_produce_a_file(tmp_path):
    epochs = list(range(1, 6))
    train_loss = [4.0, 3.5, 3.0, 2.8, 2.6]
    val_loss = [4.5, 4.0, 3.5, 3.2, 3.0]
    best_so_far = [4.5, 4.0, 3.5, 3.2, 3.0]
    secondary_train = [1.0, 0.9, 0.85, 0.8, 0.78]
    secondary_val = [1.1, 1.0, 0.95, 0.9, 0.88]
    out = tmp_path / "loss_curve_1step.png"

    result = plots.loss_curve(
        epochs, train_loss, val_loss, best_so_far, out,
        secondary_train=secondary_train, secondary_val=secondary_val, secondary_label="1step",
    )
    assert result.exists()
    assert result.stat().st_size > 0


def test_creates_parent_directories(tmp_path):
    """output_path's parent may not exist yet (matches every other
    file-saving convention in this project) -- should be created, not
    raise FileNotFoundError."""
    out = tmp_path / "nested" / "stage4" / "loss_curve.png"
    assert not out.parent.exists()

    plots.loss_curve([1], [1.0], [1.1], [1.1], out)

    assert out.exists()



def test_log_scale_used_when_all_values_positive(tmp_path, monkeypatch):
    """The requested change: log-scale y-axis by default."""
    out = tmp_path / "loss_curve.png"
    captured_axes = {}
    real_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured_axes["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)
    epochs = list(range(1, 6))
    train_loss = [124.097, 7.071, 5.202, 4.147, 3.856]
    val_loss = [7.202, 4.830, 4.321, 3.827, 3.465]
    best_so_far = [7.202, 6.490, 5.840, 5.236, 4.705]
    plots.loss_curve(epochs, train_loss, val_loss, best_so_far, out)

    ax = captured_axes["ax"]
    assert ax.get_yscale() == "log"


def test_falls_back_to_linear_when_a_zero_value_present(tmp_path, monkeypatch):
    """log(0) is undefined -- a zero (or negative) value anywhere in
    the data must fall back to linear scale, not crash."""
    out = tmp_path / "loss_curve.png"
    captured_axes = {}
    real_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured_axes["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)
    epochs = [1, 2, 3]
    train_loss = [1.0, 0.0, 0.5]
    val_loss = [1.0, 0.5, 0.3]
    best_so_far = [1.0, 0.5, 0.3]
    plots.loss_curve(epochs, train_loss, val_loss, best_so_far, out)  # must not raise

    ax = captured_axes["ax"]
    assert ax.get_yscale() == "linear"
    assert ax.get_ylim()[0] == 0


def test_x_axis_also_log_scale(tmp_path, monkeypatch):
    """Both axes should be log-log now, not just the y-axis."""
    out = tmp_path / "loss_curve.png"
    captured_axes = {}
    real_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured_axes["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy_subplots)
    epochs = list(range(1, 6))
    train_loss = [124.097, 7.071, 5.202, 4.147, 3.856]
    val_loss = [7.202, 4.830, 4.321, 3.827, 3.465]
    best_so_far = [7.202, 6.490, 5.840, 5.236, 4.705]
    plots.loss_curve(epochs, train_loss, val_loss, best_so_far, out)

    ax = captured_axes["ax"]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


# --------------------------------------------------------------------
# event markers on the loss curve
# --------------------------------------------------------------------

def test_event_lines_are_drawn_and_labelled(tmp_path):
    """
    Stage 2's centered-target switch drops val_loss sharply in ONE epoch
    because the measured QUANTITY changed, not the model. On the log-log axes
    that cliff is the most prominent feature of the figure, and unmarked it
    reads as a learning event. Reported from a real curve.
    """
    import matplotlib
    matplotlib.use("Agg")

    from utils.plots import loss_curve

    out = loss_curve(
        list(range(1, 10)),
        [3.5 - 0.02 * e for e in range(1, 10)],
        [4.0 if e < 8 else 3.0 for e in range(1, 10)],
        [4.0] * 7 + [3.03, 3.03],
        tmp_path / "c.png", title="t",
        event_epochs=[(7.5, "centered L_deriv target")],
    )
    assert out is None or (tmp_path / "c.png").exists()
    # The line's PRESENCE is asserted structurally: re-render into a figure we
    # hold, and count vertical lines at x=7.5.
    import matplotlib.pyplot as _plt
    from unittest.mock import patch
    lines = []
    orig = _plt.Axes.axvline
    def spy(self, x, *a, **k):
        lines.append((x, k.get("label")))
        return orig(self, x, *a, **k)
    with patch.object(_plt.Axes, "axvline", spy):
        loss_curve([1, 2, 3], [1.0, 0.9, 0.8], [1.1, 1.0, 0.9], [1.1, 1.0, 0.9],
                    tmp_path / "c2.png", title="t",
                    event_epochs=[(7.5, "centered L_deriv target")])
    assert (7.5, "centered L_deriv target") in lines, lines
    _plt.close("all")


def test_event_epochs_defaults_to_none_and_changes_nothing(tmp_path):
    """GUARDS making the parameter required or drawing spurious lines."""
    import inspect

    from utils.plots import loss_curve
    assert inspect.signature(loss_curve).parameters["event_epochs"].default is None
    loss_curve([1, 2], [1.0, 0.9], [1.1, 1.0], [1.1, 1.0], tmp_path / "p.png", title="t")
    assert (tmp_path / "p.png").exists()


def test_the_switch_records_an_event_between_the_two_epochs():
    import pathlib as _pl
    _ROOT = _pl.Path(__file__).resolve().parent.parent
    """
    x = epoch - 0.5, so the line sits BETWEEN the last old-target point and
    the first new one -- on either epoch it would overplot a real data point
    and imply the discontinuity belongs to that epoch's model.
    """
    from conftest import source_without_comments
    src = source_without_comments(_ROOT / "training/train_stage2.py")
    assert 'loss_curve_events.append((epoch - 0.5, "centered L_deriv target"))' in src
    assert "event_epochs=loss_curve_events" in src

    src45 = source_without_comments(_ROOT / "training/train_refinement.py")
    assert 'loss_curve_events.append((epoch - 0.5, "rollout ramp complete"))' in src45
    assert "event_epochs=loss_curve_events" in src45
