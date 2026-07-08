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


def test_y_axis_capped_at_2x_first_train_loss(tmp_path, monkeypatch):
    """The core behavior this module exists for: a huge early spike
    must not stretch the y-axis so far that later, more informative
    convergence gets squashed -- checked against the ACTUAL rendered
    axis limits (captured via the real plt.subplots call the function
    makes), not just independently recomputed and trusted."""
    epochs = list(range(1, 6))
    train_loss = [10.0, 8.0, 5000.0, 6.0, 5.0]  # huge spike at epoch 3
    val_loss = [11.0, 9.0, 6000.0, 7.0, 6.0]
    best_so_far = [11.0, 9.0, 9.0, 7.0, 6.0]
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
    expected_cap = 2 * train_loss[0]  # 20.0 -- NOT anywhere near the 5000/6000 spike
    assert ymax == expected_cap, f"expected y-axis top exactly {expected_cap}, got {ymax}"
    assert ymin == 0
    assert ymax < max(val_loss), "the cap should be far below the actual spike -- confirms it's a real cap"


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
