"""
Tests for training/checkpoint_components.py. Builds synthetic
checkpoints matching the EXACT dict shape train_ae.py/train_lds.py
actually save (confirmed against their source, not assumed) -- so these
tests exercise the adapter's real parsing logic without needing an
actual trained model or a full training run.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_checkpoint_components.py -v
"""
import torch
import pytest
import warnings

from training.checkpoint_components import (
    ComponentCheckpoint, load_ae_components, load_lds_component,
    validate_component_compatibility, assemble_joint_checkpoint,
    split_joint_checkpoint_for_evaluation,
)
from models.constants import LATENT_SPATIAL_SIZE


def _make_ae_checkpoint(latent_channels=8, include_stats_head=True, latent_spatial_size=None):
    """Synthetic stage-1/2-shaped checkpoint (same shape either way).
    latent_spatial_size=None (default) omits the key entirely, matching
    a pre-latent_spatial_size checkpoint -- load_ae_components should
    then fall back to models.constants.LATENT_SPATIAL_SIZE. Pass a real
    value to test that an EXPLICIT, non-default size actually
    propagates instead of always silently defaulting."""
    state = {
        "encoder.down_blocks.0.conv1.weight": torch.randn(4, 1, 3, 3),
        "encoder.bottleneck.weight": torch.randn(latent_channels, 4, 1, 1),
        "decoder.unbottleneck.weight": torch.randn(4, latent_channels, 1, 1),
        "decoder.output_conv.weight": torch.randn(1, 4, 3, 3),
    }
    config = {"size": 64, "base_channels": 4, "latent_channels": latent_channels,
              "stats_weight": 0.01}
    if latent_spatial_size is not None:
        config["latent_spatial_size"] = latent_spatial_size
    checkpoint = {
        "model_state": state,
        "epoch": 12,
        "val_loss": 0.0021,
        "val_loss_ema": 0.0025,
        "test_dirs": ["/fake/run1", "/fake/run2"],
        "config": config,
    }
    if include_stats_head:
        checkpoint["stats_head_state"] = {"fc.weight": torch.randn(12, latent_channels * 64)}
        checkpoint["stats_config"] = {
            "stat_names": ["angle", "avg_phi"], "stats_mean": torch.zeros(12),
            "stats_std": torch.ones(12),
        }
    return checkpoint


def _make_lds_checkpoint(latent_channels=8):
    return {
        "model_state": {"mlp.0.weight": torch.randn(256, latent_channels * 64 + 1)},
        "epoch": 214,
        "val_loss": 0.038593,
        "val_loss_ema": 0.041,
        "ae_checkpoint": "/fake/stage2.pt",
        "test_dirs": ["/fake/run1"],
        "config": {"latent_channels": latent_channels, "n_theta": 1,
                   "hidden_dim": 256, "n_hidden_layers": 2},
        "data_config": {"min_step": 4000, "min_stdev_phi": 0.01,
                         "window_length": 2, "n_rollout_steps": 1},
    }


def test_load_ae_components_splits_encoder_and_decoder(tmp_path):
    checkpoint = _make_ae_checkpoint()
    path = tmp_path / "fake-stage2.pt"
    torch.save(checkpoint, path)

    components = load_ae_components(path)

    assert set(components.keys()) == {"encoder", "decoder", "stats_head"}
    # Prefix correctly stripped -- these are now bare keys, loadable
    # directly into a standalone Encoder()/Decoder(), not the combined
    # Autoencoder they were saved from.
    assert set(components["encoder"].state_dict.keys()) == {
        "down_blocks.0.conv1.weight", "bottleneck.weight"}
    assert set(components["decoder"].state_dict.keys()) == {
        "unbottleneck.weight", "output_conv.weight"}
    # No cross-contamination: encoder's dict has nothing decoder-prefixed and vice versa.
    assert all(not k.startswith("decoder") for k in components["encoder"].state_dict)
    assert all(not k.startswith("encoder") for k in components["decoder"].state_dict)


def test_load_ae_components_config_and_provenance(tmp_path):
    checkpoint = _make_ae_checkpoint(latent_channels=8)
    path = tmp_path / "fake-stage2.pt"
    torch.save(checkpoint, path)

    components = load_ae_components(path)

    assert components["encoder"].config == {
        "size": 64, "base_channels": 4, "latent_channels": 8,
        "latent_spatial_size": LATENT_SPATIAL_SIZE,
        "stream_configs": {"state": {"channels": 8, "spatial_size": LATENT_SPATIAL_SIZE,
                                      "mode": "autoencoder"}},
        "recon_stream_name": "state",
    }
    assert components["encoder"].provenance["epoch"] == 12
    assert components["encoder"].provenance["val_loss"] == pytest.approx(0.0021)
    assert str(path.resolve()) == components["encoder"].provenance["source_checkpoint"]


def test_load_ae_components_propagates_explicit_latent_spatial_size(tmp_path):
    """THE case backward-compat fallback alone doesn't cover: a
    checkpoint that DOES specify a non-default latent_spatial_size must
    have that real value come through, not silently default."""
    checkpoint = _make_ae_checkpoint(latent_spatial_size=4)
    path = tmp_path / "fake-stage2-smaller-bottleneck.pt"
    torch.save(checkpoint, path)

    components = load_ae_components(path)

    assert components["encoder"].config["latent_spatial_size"] == 4
    assert components["decoder"].config["latent_spatial_size"] == 4


def test_load_ae_components_self_heals_stale_config_with_real_multi_stream_weights(tmp_path):
    """THE actual bug hit during this project's own multi-stream
    rollout: a checkpoint saved by an intermediate version of the code
    that had the CONSTRUCTION fix (so training genuinely produced
    2-stream weights) but not yet the separate checkpoint-SAVE fix (so
    "config" never recorded that) -- config claims only "state", but
    model_state's own keys prove "deriv" is really there too.

    Uses the CURRENT (post-redesign) key naming directly
    (encoder.bottlenecks.<name>.*), unlike _make_ae_checkpoint's own
    fixture (which deliberately uses the OLD single-bottleneck naming
    for its own, different purposes, and wouldn't exercise this cross-
    check at all -- the regex it depends on requires the "bottlenecks"
    -- plural -- form)."""
    state = {
        "encoder.down_blocks.0.conv1.weight": torch.randn(4, 1, 3, 3),
        "encoder.bottlenecks.state.weight": torch.randn(8, 4, 1, 1),
        "encoder.bottlenecks.state.bias": torch.randn(8),
        "encoder.bottlenecks.deriv.weight": torch.randn(8, 4, 1, 1),
        "encoder.bottlenecks.deriv.bias": torch.randn(8),
        "decoder.unbottleneck.weight": torch.randn(4, 8, 1, 1),
        "decoder.output_conv.weight": torch.randn(1, 4, 3, 3),
    }
    checkpoint = {
        "model_state": state,
        "epoch": 12,
        "val_loss": 0.0021,
        "val_loss_ema": 0.0025,
        "test_dirs": ["/fake/run1"],
        # Deliberately STALE: no "stream_configs"/"recon_stream_name"
        # at all, and "latent_channels" only describes "state" -- the
        # exact shape a pre-checkpoint-save-fix version of this
        # codebase would have produced.
        "config": {"size": 64, "base_channels": 4, "latent_channels": 8, "stats_weight": 0.01},
    }
    path = tmp_path / "fake-stage2-stale-metadata.pt"
    torch.save(checkpoint, path)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        components = load_ae_components(path)
    assert any(issubclass(x.category, UserWarning) and "stale" in str(x.message) for x in w), \
        f"expected a 'stale' UserWarning, got: {[str(x.message) for x in w]}"

    # The self-healed config now correctly describes BOTH streams --
    # this is what actually matters: a caller resolving stream_configs
    # from components["encoder"].config afterward (e.g. model_assembly.py)
    # must see "deriv" too, not just what the stale metadata claimed.
    healed_streams = components["encoder"].config["stream_configs"]
    assert set(healed_streams.keys()) == {"state", "deriv"}
    assert healed_streams["deriv"]["channels"] == 8
    assert healed_streams["deriv"]["mode"] == "decoder"
    assert components["encoder"].config["recon_stream_name"] == "state"
    # Decoder's componentized config is the SAME shared dict (see
    # load_ae_components' own code) -- healed identically, not just
    # the encoder's copy.
    assert components["decoder"].config["stream_configs"] == healed_streams


def test_load_ae_components_omits_stats_head_when_absent(tmp_path):
    """stats_weight <= 0 in stage 1 means no stats_head at all -- the
    adapter should reflect that (no key), not synthesize an empty one."""
    checkpoint = _make_ae_checkpoint(include_stats_head=False)
    path = tmp_path / "fake-stage1-nostats.pt"
    torch.save(checkpoint, path)

    components = load_ae_components(path)
    assert set(components.keys()) == {"encoder", "decoder"}


def test_load_lds_component(tmp_path):
    checkpoint = _make_lds_checkpoint(latent_channels=8)
    path = tmp_path / "fake-stage3.pt"
    torch.save(checkpoint, path)

    component = load_lds_component(path)
    assert isinstance(component, ComponentCheckpoint)
    assert component.config == {"latent_channels": 8, "n_theta": 1,
                                 "hidden_dim": 256, "n_hidden_layers": 2}
    assert component.provenance["epoch"] == 214
    assert component.provenance["ae_checkpoint"] == "/fake/stage2.pt"
    assert "mlp.0.weight" in component.state_dict


def test_validate_component_compatibility_passes_when_consistent():
    components = {
        "encoder": ComponentCheckpoint({}, {"latent_channels": 8}, {}),
        "lds": ComponentCheckpoint({}, {"latent_channels": 8}, {}),
    }
    validate_component_compatibility(components)  # should not raise


def test_validate_component_compatibility_catches_mismatch():
    components = {
        "encoder": ComponentCheckpoint({}, {"latent_channels": 8}, {}),
        "lds": ComponentCheckpoint({}, {"latent_channels": 16}, {}),
    }
    with pytest.raises(ValueError, match="disagree on latent_channels"):
        validate_component_compatibility(components)


def test_assemble_joint_checkpoint_end_to_end(tmp_path):
    """The actual stage-4 entry point: two real checkpoint files on
    disk, both with matching latent_channels, merged into one structure."""
    ae_path = tmp_path / "fake-stage2.pt"
    lds_path = tmp_path / "fake-stage3.pt"
    torch.save(_make_ae_checkpoint(latent_channels=8), ae_path)
    torch.save(_make_lds_checkpoint(latent_channels=8), lds_path)

    components = assemble_joint_checkpoint(ae_path, lds_path)

    assert set(components.keys()) == {"encoder", "decoder", "stats_head", "lds"}
    assert components["lds"].config["latent_channels"] == 8


def test_assemble_joint_checkpoint_rejects_mismatched_ancestors(tmp_path):
    """The realistic failure case this whole module exists to catch
    early: an AE checkpoint and an LDS checkpoint that don't actually
    belong together."""
    ae_path = tmp_path / "fake-stage2.pt"
    lds_path = tmp_path / "fake-stage3.pt"
    torch.save(_make_ae_checkpoint(latent_channels=8), ae_path)
    torch.save(_make_lds_checkpoint(latent_channels=16), lds_path)  # mismatched on purpose

    with pytest.raises(ValueError, match="disagree on latent_channels"):
        assemble_joint_checkpoint(ae_path, lds_path)


def _make_joint_checkpoint(latent_channels=8, include_stats_head=True):
    """Synthetic checkpoint matching the EXACT shape train_refinement.py
    actually saves (ae_state/f_theta_state together, not two separate
    files)."""
    ae_state = {
        "encoder.down_blocks.0.conv1.weight": torch.randn(4, 1, 3, 3),
        "encoder.bottleneck.weight": torch.randn(latent_channels, 4, 1, 1),
        "decoder.unbottleneck.weight": torch.randn(4, latent_channels, 1, 1),
        "decoder.output_conv.weight": torch.randn(1, 4, 3, 3),
    }
    checkpoint = {
        "ae_state": ae_state,
        "f_theta_state": {"mlp.0.weight": torch.randn(256, latent_channels * 64 + 1)},
        "stats_head_state": None, "epoch": 7, "val_loss": 0.0123, "val_loss_ema": 0.0125,
        "ae_checkpoint": "/fake/stage2.pt", "lds_checkpoint": "/fake/stage3.pt",
        "test_dirs": ["/fake/run1", "/fake/run2"],
        "config": {"size": 64, "base_channels": 4, "latent_channels": latent_channels},
        "lds_config": {"latent_channels": latent_channels, "n_theta": 1,
                       "hidden_dim": 256, "n_hidden_layers": 2},
        "data_config": {"min_step": 3000, "min_stdev_phi": 0.01, "window_length": 2,
                        "n_rollout_steps": 1},
        "stats_config": None,
        "stage45_config": {"freeze_decoder": True, "rollout_weight": 1.0,
                           "recon_weight": 0.1, "stats_weight": 0.0, "n_rollout_steps": 1},
    }
    if include_stats_head:
        checkpoint["stats_head_state"] = {"net.0.weight": torch.randn(16, latent_channels * 64)}
        checkpoint["stats_config"] = {
            "stat_names": ["angle", "avg_phi"], "stats_mean": torch.zeros(2),
            "stats_std": torch.ones(2),
        }
    return checkpoint


def test_split_joint_checkpoint_produces_both_files(tmp_path):
    checkpoint = _make_joint_checkpoint()
    joint_path = tmp_path / "64x64-stage4.pt"
    torch.save(checkpoint, joint_path)

    ae_view_path, lds_view_path = split_joint_checkpoint_for_evaluation(joint_path, tmp_path / "views")

    assert ae_view_path.exists()
    assert lds_view_path.exists()


def test_split_joint_checkpoint_ae_view_has_standalone_shape(tmp_path):
    checkpoint = _make_joint_checkpoint(include_stats_head=True)
    joint_path = tmp_path / "64x64-stage4.pt"
    torch.save(checkpoint, joint_path)

    ae_view_path, _ = split_joint_checkpoint_for_evaluation(joint_path, tmp_path / "views")
    ae_view = torch.load(ae_view_path, map_location="cpu", weights_only=True)

    for key in ["model_state", "epoch", "val_loss", "test_dirs", "config",
                "stats_head_state", "stats_config"]:
        assert key in ae_view, f"missing '{key}' -- check_reconstruction.py needs this"
    assert ae_view["config"] == checkpoint["config"]

    # Cross-check: the derived file should ALSO parse correctly through
    # the EXISTING, already-tested load_ae_components adapter -- confirms
    # it's a genuine, valid standalone AE checkpoint, not just superficially shaped.
    components = load_ae_components(ae_view_path)
    assert set(components.keys()) == {"encoder", "decoder", "stats_head"}


def test_split_joint_checkpoint_omits_stats_when_absent(tmp_path):
    checkpoint = _make_joint_checkpoint(include_stats_head=False)
    joint_path = tmp_path / "64x64-stage4-nostats.pt"
    torch.save(checkpoint, joint_path)

    ae_view_path, _ = split_joint_checkpoint_for_evaluation(joint_path, tmp_path / "views")
    ae_view = torch.load(ae_view_path, map_location="cpu", weights_only=True)

    assert "stats_head_state" not in ae_view
    assert "stats_config" not in ae_view


def test_split_joint_checkpoint_lds_view_points_at_refined_ae(tmp_path):
    """The key correctness property: lds_view's ae_checkpoint field must
    point at the DERIVED ae_view (stage 4/5's refined E/D), not the
    original stage-2 ancestor -- otherwise check_rollout would silently
    evaluate against the wrong, pre-refinement encoder/decoder."""
    checkpoint = _make_joint_checkpoint()
    joint_path = tmp_path / "64x64-stage4.pt"
    torch.save(checkpoint, joint_path)

    ae_view_path, lds_view_path = split_joint_checkpoint_for_evaluation(joint_path, tmp_path / "views")
    lds_view = torch.load(lds_view_path, map_location="cpu", weights_only=True)

    assert lds_view["ae_checkpoint"] == str(ae_view_path.resolve())
    assert lds_view["ae_checkpoint"] != checkpoint["ae_checkpoint"]  # NOT the original stage-2 ancestor


def test_split_joint_checkpoint_lds_view_carries_data_config(tmp_path):
    checkpoint = _make_joint_checkpoint()
    joint_path = tmp_path / "64x64-stage4.pt"
    torch.save(checkpoint, joint_path)

    _, lds_view_path = split_joint_checkpoint_for_evaluation(joint_path, tmp_path / "views")
    lds_view = torch.load(lds_view_path, map_location="cpu", weights_only=True)

    assert lds_view["data_config"] == checkpoint["data_config"]
    assert lds_view["test_dirs"] == checkpoint["test_dirs"]
    assert lds_view["config"] == checkpoint["lds_config"]
