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

from training.checkpoint_components import (
    ComponentCheckpoint, load_ae_components, load_lds_component,
    validate_component_compatibility, assemble_joint_checkpoint,
    split_joint_checkpoint_for_evaluation,
)


def _make_ae_checkpoint(latent_channels=8, include_stats_head=True):
    """Synthetic stage-1/2-shaped checkpoint (same shape either way)."""
    state = {
        "encoder.down_blocks.0.conv1.weight": torch.randn(4, 1, 3, 3),
        "encoder.bottleneck.weight": torch.randn(latent_channels, 4, 1, 1),
        "decoder.unbottleneck.weight": torch.randn(4, latent_channels, 1, 1),
        "decoder.output_conv.weight": torch.randn(1, 4, 3, 3),
    }
    checkpoint = {
        "model_state": state,
        "epoch": 12,
        "val_loss": 0.0021,
        "val_loss_ema": 0.0025,
        "test_dirs": ["/fake/run1", "/fake/run2"],
        "config": {"size": 64, "base_channels": 4, "latent_channels": latent_channels,
                   "stats_weight": 0.01},
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

    assert components["encoder"].config == {"size": 64, "base_channels": 4, "latent_channels": 8}
    assert components["encoder"].provenance["epoch"] == 12
    assert components["encoder"].provenance["val_loss"] == pytest.approx(0.0021)
    assert str(path.resolve()) == components["encoder"].provenance["source_checkpoint"]


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
