"""loss_scale_curve: the L_XX/XX_scale (val) diagnostic with the dual power-law /
constant fit. These guard the branching logic (which model is chosen, and how it
is labelled), the empty-data guard, and the event marker -- none of which had any
coverage."""
import io
import contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.plots import loss_scale_curve


def _labels_for(scale_ratios, epochs=None):
    """Render and capture the per-series legend labels (they encode the chosen
    model: 'name  amp·x^exp (R²=..)' for the power law, 'name  c=..' constant)."""
    epochs = epochs or list(range(1, len(next(iter(scale_ratios.values()))) + 1))
    from unittest.mock import patch
    labels = {}
    orig = plt.Axes.plot
    def spy(self, *a, **k):
        lbl = k.get("label")
        if lbl:
            labels[lbl.split()[0]] = lbl     # key by component name
        return orig(self, *a, **k)
    import tempfile, pathlib
    with patch.object(plt.Axes, "plot", spy):
        loss_scale_curve(epochs, scale_ratios,
                         pathlib.Path(tempfile.mkdtemp()) / "s.png")
    plt.close("all")
    return labels


def test_power_law_is_chosen_for_a_decaying_term_and_shows_the_AMPLITUDE():
    x = np.arange(1, 13, dtype=float)
    labels = _labels_for({"recon_predict": list(13.0 * x ** -0.6)})
    lbl = labels["recon_predict"]
    assert "x^" in lbl and "R" in lbl, lbl               # power law chosen
    assert "\u00b7x^" in lbl                              # amplitude·x^exp, not bare x^exp
    assert lbl.split("\u00b7")[0].split()[-1] not in ("x^", "")  # an amplitude precedes ·x^


def test_constant_is_chosen_when_the_exponent_is_POSITIVE():
    # a GROWING ratio: power law has b>0 and must be EXCLUDED -> constant.
    x = np.arange(1, 13, dtype=float)
    labels = _labels_for({"stats0": list(0.8 + 0.02 * x)})
    assert labels["stats0"].split()[-1].startswith("c=") or "c=" in labels["stats0"]
    assert "x^" not in labels["stats0"]


def test_constant_is_chosen_when_the_power_law_does_not_beat_the_mean():
    # a FLAT ratio: power-law R^2 <= 0 -> constant wins.
    labels = _labels_for({"rollout": [2.0] * 12})
    assert "c=" in labels["rollout"] and "x^" not in labels["rollout"]


def test_empty_and_epoch_zero_inputs_are_skipped_without_error(tmp_path):
    # epochs=0 ablation (single epoch-0 render) and empty ratios must not raise
    # or emit matplotlib log-scale/tight-layout warnings.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss_scale_curve([], {}, tmp_path / "a.png")               # nothing
        loss_scale_curve([0], {"recon0": [1.5]}, tmp_path / "b.png")  # x=0, can't log-scale
    assert not (tmp_path / "a.png").exists() and not (tmp_path / "b.png").exists()


def test_event_epochs_draws_a_marker(tmp_path):
    from unittest.mock import patch
    xs = []
    orig = plt.Axes.axvline
    with patch.object(plt.Axes, "axvline",
                      lambda self, x, *a, **k: (xs.append(x), orig(self, x, *a, **k))[1]):
        loss_scale_curve([1, 2, 3], {"recon0": [3.0, 2.0, 1.5]}, tmp_path / "c.png",
                         event_epochs=[(1.5, "ramp complete")])
    plt.close("all")
    assert 1.5 in xs
