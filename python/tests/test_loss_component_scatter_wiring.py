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
import re
import struct
from pathlib import Path

import pytest

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
    import utils.plots as plots

    captured = {}
    # write_epoch_figures (in training.training_loop) now owns the scatter call
    # and imports loss_component_scatter LAZILY from utils.plots, so patch the
    # canonical source -- it is resolved at call time wherever the figure write
    # ends up living, rather than a name bound into one trainer module.
    real_fn = plots.loss_component_scatter

    def spy(epoch_history, component_histories, output_path, **kw):
        captured["epoch_history"] = list(epoch_history)
        captured["component_histories"] = {
            k: {kk: list(vv) for kk, vv in v.items()} for k, v in component_histories.items()
        }
        return real_fn(epoch_history, component_histories, output_path, **kw)

    monkeypatch.setattr(plots, "loss_component_scatter", spy)

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
        m = re.search(rf"^\s*{epoch}\|\s*([\d.]+)\s*=", printed, re.MULTILINE)
        assert m, f"epoch {epoch}: no console line found to compare against"
        printed_total = float(m.group(1))
        assert reconstructed == pytest.approx(printed_total, abs=5e-4), (
            f"epoch {epoch}: reconstructed train total {reconstructed:.4f} doesn't match the "
            f"console's own {printed_total:.4f} (tolerance 5e-4, matching the console's own "
            f"4-decimal rounding -- these are two independently computed, mathematically equal "
            f"sums, not required to be bit-identical; a real Windows run failed here by exactly "
            f"1e-4, right at the .4f rounding boundary, from BLAS/platform summation-order "
            f"differences)"
        )


def test_figures_are_throttled_but_ALWAYS_flushed_at_the_end(tmp_path, isolated_project_root,
                                                              monkeypatch):
    """
    The per-epoch figure writes are throttled (see
    utils.plots.should_write_loss_figure -- they cost ~0.7s/epoch and
    only the last one is ever looked at). Two things must hold:

      1. Throttling actually happens (far fewer writes than epochs).
      2. The FINAL state is never stale. This is the subtle one: a run
         can end on an epoch the throttle skipped -- via early stopping,
         or simply a final epoch that isn't a multiple of the interval
         -- so each stage must write once unconditionally AFTER its
         loop. Without that, a finished run would be judged from
         figures up to `every` epochs out of date.

    Uses 12 epochs against an interval of 10, so the last in-loop write
    is epoch 10 and epochs 11-12 are skipped -- the final flush is the
    only thing that can make the figure reflect all 12.
    """
    import training.train_stage1 as t1
    from utils.plots import LOSS_FIGURE_EVERY

    epochs = 12
    assert epochs % LOSS_FIGURE_EVERY != 0, (
        "fixture assumption: the final epoch must NOT be a write epoch, or this test "
        "cannot distinguish a real final flush from an in-loop write"
    )

    seen_epoch_counts = []
    real = t1.loss_curve

    def spy(epoch_history, *args, **kwargs):
        seen_epoch_counts.append(len(epoch_history))
        return real(epoch_history, *args, **kwargs)

    monkeypatch.setattr(t1, "loss_curve", spy)

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    t1.train_autoencoder(
        size=32, base_path=base_path, epochs=epochs, batch_size=4, base_channels=4,
        latent_channels=4, val_fraction=0.34, test_fraction=0.17, num_workers=0,
        augment=False, min_step=0, min_stdev_phi=None, stats0_weight=0.01,
        stat_names=["avg_phi"], checkpoint_path=tmp_path / "s1_thr.pt", device="cpu",
        seed=0, log_every_epoch=False, loss_curve_path=tmp_path / "c_thr.png",
    )

    assert len(seen_epoch_counts) < epochs, (
        f"no throttling: loss_curve ran {len(seen_epoch_counts)}x for {epochs} epochs"
    )
    assert seen_epoch_counts[-1] == epochs, (
        f"the LAST write saw only {seen_epoch_counts[-1]} epochs, not all {epochs} -- "
        f"the post-loop final flush is missing, so a finished run's figures are stale"
    )


def test_log_every_epoch_true_writes_every_epoch_not_throttled(tmp_path, isolated_project_root,
                                                                monkeypatch):
    """
    REGRESSION-shaped, for the OTHER half of should_write_loss_figure:
    log_every_epoch=True means a closely-watched run (its whole point is
    a console line every epoch), typically also a SLOW one -- a real
    stage-1 epoch can take ~20 minutes, against which the ~0.7s these
    figures cost is ~0.06%, negligible. Throttling that case would only
    make the plot someone is actively watching go stale, for savings
    that don't matter at that timescale -- so it must NOT be throttled,
    unlike the log_every_epoch=False (quiet/automated) case covered by
    the sibling test above.
    """
    import training.train_stage1 as t1

    epochs = 12
    seen_epoch_counts = []
    real = t1.loss_curve

    def spy(epoch_history, *args, **kwargs):
        seen_epoch_counts.append(len(epoch_history))
        return real(epoch_history, *args, **kwargs)

    monkeypatch.setattr(t1, "loss_curve", spy)

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    t1.train_autoencoder(
        size=32, base_path=base_path, epochs=epochs, batch_size=4, base_channels=4,
        latent_channels=4, val_fraction=0.34, test_fraction=0.17, num_workers=0,
        augment=False, min_step=0, min_stdev_phi=None, stats0_weight=0.01,
        stat_names=["avg_phi"], checkpoint_path=tmp_path / "s1_watched.pt", device="cpu",
        seed=0, log_every_epoch=True, loss_curve_path=tmp_path / "c_watched.png",
    )

    # >= epochs, not == : the unconditional post-loop flush fires as a
    # (harmless) extra call even when every epoch already wrote -- see
    # should_write_loss_figure's own docstring.
    assert len(seen_epoch_counts) >= epochs, (
        f"log_every_epoch=True was throttled ({len(seen_epoch_counts)} writes for {epochs} "
        f"epochs) -- a closely-watched run must see every epoch's own update"
    )
    assert seen_epoch_counts[-1] == epochs
