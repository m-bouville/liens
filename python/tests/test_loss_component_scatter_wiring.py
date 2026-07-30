"""
Integration tests for loss_component_scatter's own wiring into
train_stage1(), train_stage2(), and train_refinement() -- the pairwise
component-vs-component trajectory diagnostic alongside loss_curve().

train_lds() is deliberately NOT covered here: its own loss isn't a sum
of separately-weighted components in this sense (ae_stats_weight is
used only for checkpoint naming; step() returns a single rollout loss
plus a same-shape-different-granularity 1step metric, not summed terms).

Stage 2's own regression tests for this diagnostic (the path-collision
bug, and value-reconstruction) live in test_train_stage2_l_deriv.py
instead, alongside that stage's other tests.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_loss_component_scatter_wiring.py -v
"""
import struct
from pathlib import Path

from test_train_stage2_l_deriv import _build_sweep
from test_train_refinement import (
    _build_ae_checkpoint, _build_lds_checkpoint, _build_sweep as _build_sweep_refinement,
)
from training.checkpoint_criterion import ComponentBestTracker
from training.train_stage1 import train_autoencoder
from training.train_refinement import train_refinement


def _png_size(path: Path) -> tuple[int, int]:
    return struct.unpack(">II", path.read_bytes()[16:24])


# ---------------------------------------------------------------------
# ComponentBestTracker -- isolated unit tests (see checkpoint_criterion.py
# for the class's own docstring on why "each component's own independent
# running minimum" is the WRONG semantics, and this instead freezes the
# co-occurring values at the epoch a checkpoint was actually saved).
# ---------------------------------------------------------------------

def test_component_best_tracker_freezes_at_last_save_not_each_components_own_minimum():
    t = ComponentBestTracker()
    assert t.update({"a": 5.0, "b": 5.0}, saved_this_epoch=False) == {"a": 5.0, "b": 5.0}, (
        "before any save, must fall back to the current epoch's own values"
    )
    assert t.update({"a": 9.0, "b": 9.0}, saved_this_epoch=False) == {"a": 5.0, "b": 5.0}, (
        "no save this epoch -- must stay frozen at the LAST save's values, not update"
    )
    assert t.update({"a": 1.0, "b": 8.0}, saved_this_epoch=True) == {"a": 1.0, "b": 8.0}, (
        "a real save -- both components must update together, even though 'b' alone "
        "got WORSE (8.0 > 5.0): the whole point is co-occurring values, not each "
        "component's own independent best"
    )
    assert t.update({"a": 0.1, "b": 0.1}, saved_this_epoch=False) == {"a": 1.0, "b": 8.0}, (
        "frozen again at the second save's values"
    )


def test_component_best_tracker_returns_independent_copies():
    t = ComponentBestTracker()
    d1 = t.update({"a": 1.0}, saved_this_epoch=True)
    d1["a"] = 999.0
    d2 = t.update({"a": 2.0}, saved_this_epoch=False)
    assert d2["a"] == 1.0, "mutating a returned snapshot must not affect the tracker's own state"


# ---------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------

def test_stage1_writes_components_figure_only_when_include_stats(tmp_path, isolated_project_root):
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)

    curve_with_stats = tmp_path / "s1_stats_curve.png"
    train_autoencoder(
        size=32, base_path=base_path, epochs=2, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "s1_stats.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=curve_with_stats,
    )
    components_path = tmp_path / "s1_stats_curve-components.png"
    assert components_path.exists(), "include_stats=True (stats0_weight>0) must produce a components figure"
    assert _png_size(curve_with_stats) == (800, 500)
    assert _png_size(components_path) != (800, 500)

    curve_no_stats = tmp_path / "s1_nostats_curve.png"
    train_autoencoder(
        size=32, base_path=base_path, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.0, stat_names=None,
        checkpoint_path=tmp_path / "s1_nostats.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=curve_no_stats,
    )
    assert curve_no_stats.exists()
    assert not (tmp_path / "s1_nostats_curve-components.png").exists(), (
        "include_stats=False (stats0_weight==0) has only ONE component (recon0) -- "
        "nothing to pair, so no components figure should be written at all"
    )


# ---------------------------------------------------------------------
# Refinement (stage 4/5)
# ---------------------------------------------------------------------

def test_refinement_writes_components_figure_and_survives_epochs_zero(tmp_path, isolated_project_root):
    base_path = _build_sweep_refinement(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)
    _build_lds_checkpoint(lds_checkpoint_path)

    curve_path = tmp_path / "s4curve.png"
    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
        rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.1,
        epochs=2, batch_size=4, n_rollout_steps=1,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=tmp_path / "stage4_out.pt", device="cpu", log_every_epoch=True,
        loss_curve_path=curve_path,
    )
    components_path = tmp_path / "s4curve-components.png"
    assert components_path.exists()
    assert _png_size(curve_path) == (800, 500)
    assert _png_size(components_path) != (800, 500)

    # REGRESSION-shaped check: the epochs=0 ablation sets every train_*
    # component to NaN (see train_refinement's own "epoch 0" branch) --
    # must not crash loss_component_scatter (matplotlib silently skips
    # NaN points, same as loss_curve() already relies on for its own
    # scalar train_loss).
    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
        rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.1,
        epochs=0, batch_size=4, n_rollout_steps=1,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=tmp_path / "stage4_out0.pt", device="cpu", log_every_epoch=True,
        loss_curve_path=tmp_path / "s4curve0.png",
    )
    assert (tmp_path / "s4curve0-components.png").exists()


def test_refinement_component_values_reconstruct_train_total(tmp_path, isolated_project_root, monkeypatch):
    import io, contextlib
    import training.train_refinement as tr

    captured = {}
    real_fn = tr.loss_component_scatter

    def spy(epoch_history, component_histories, output_path, **kw):
        captured["epoch_history"] = list(epoch_history)
        captured["component_histories"] = {
            k: {kk: list(vv) for kk, vv in v.items()} for k, v in component_histories.items()
        }
        return real_fn(epoch_history, component_histories, output_path, **kw)

    monkeypatch.setattr(tr, "loss_component_scatter", spy)

    base_path = _build_sweep_refinement(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)
    _build_lds_checkpoint(lds_checkpoint_path)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tr.train_refinement(
            base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
            lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
            rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.1,
            epochs=2, batch_size=4, n_rollout_steps=1,
            min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
            checkpoint_path=tmp_path / "stage4_recon.pt", device="cpu", log_every_epoch=True,
            loss_curve_path=tmp_path / "s4recon.png",
        )
    printed = buf.getvalue()

    assert captured["component_histories"]
    for i, epoch in enumerate(captured["epoch_history"]):
        reconstructed = sum(captured["component_histories"][c]["train"][i]
                            for c in captured["component_histories"])
        assert f"{epoch:4d}|{reconstructed:7.4f}" in printed, (
            f"epoch {epoch}: reconstructed train total {reconstructed:.4f} doesn't match "
            f"what the console actually printed"
        )
