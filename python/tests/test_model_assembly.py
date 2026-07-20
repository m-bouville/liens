"""
Tests for training/model_assembly.py. Builds small, REAL Autoencoder/
StatsHead/LatentDynamics instances, saves them through the same dict
shapes train_ae.py/train_lds.py actually produce, runs them through the
checkpoint_components adapter, and checks build_models_from_components
reconstructs models that are genuinely identical to the originals --
not just the right shape.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_model_assembly.py -v
"""
import torch
import pytest

from models.autoencoder import Autoencoder
from models.latent_dynamics import LatentDynamics
from training.stats_head import StatsHead
from training.checkpoint_components import load_ae_components, load_lds_component
from training.model_assembly import build_models_from_components


LATENT_CHANNELS = 4
STAT_NAMES = ["angle", "avg_phi", "stdev_phi"]


def _save_ae_checkpoint(path, latent_channels=LATENT_CHANNELS, base_channels=4,
                         stats_hidden_dim=16, include_stats_head=True):
    """Builds and saves a REAL small autoencoder (+ optionally stats_head)
    through the exact checkpoint shape train_ae.py's save calls use."""
    ae = Autoencoder(size=64, channels=1, base_channels=base_channels,
                      latent_channels=latent_channels)
    checkpoint = {
        "model_state": ae.state_dict(),
        "epoch": 5,
        "val_loss": 0.01,
        "val_loss_ema": 0.012,
        "test_dirs": ["/fake/run"],
        "config": {"size": 64, "base_channels": base_channels,
                   "latent_channels": latent_channels, "stats_weight": 0.01},
    }
    if include_stats_head:
        stats_head = StatsHead(latent_channels=latent_channels, stat_names=STAT_NAMES,
                                hidden_dim=stats_hidden_dim)
        checkpoint["stats_head_state"] = stats_head.state_dict()
        checkpoint["stats_config"] = {
            "stat_names": STAT_NAMES, "stats_mean": torch.zeros(len(STAT_NAMES)),
            "stats_std": torch.ones(len(STAT_NAMES)),
        }
    torch.save(checkpoint, path)
    return ae, (stats_head if include_stats_head else None)


def _save_lds_checkpoint(path, latent_channels=LATENT_CHANNELS, hidden_dim=32, n_hidden_layers=1):
    f_theta = LatentDynamics(latent_channels=latent_channels, n_theta=1,
                              hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    checkpoint = {
        "model_state": f_theta.state_dict(),
        "epoch": 40,
        "val_loss": 0.05,
        "val_loss_ema": 0.06,
        "ae_checkpoint": "/fake/stage2.pt",
        "test_dirs": ["/fake/run"],
        "config": {"latent_channels": latent_channels, "n_theta": 1,
                   "hidden_dim": hidden_dim, "n_hidden_layers": n_hidden_layers},
        "data_config": {"min_step": 4000, "min_stdev_phi": 0.01,
                         "window_length": 2, "n_rollout_steps": 1},
    }
    torch.save(checkpoint, path)
    return f_theta


def _build_components(tmp_path, **ae_kwargs):
    ae_path = tmp_path / "fake-stage2.pt"
    lds_path = tmp_path / "fake-stage3.pt"
    original_ae, original_stats_head = _save_ae_checkpoint(ae_path, **ae_kwargs)
    original_lds = _save_lds_checkpoint(lds_path, latent_channels=ae_kwargs.get(
        "latent_channels", LATENT_CHANNELS))

    components = {**load_ae_components(ae_path), "lds": load_lds_component(lds_path)}
    return components, original_ae, original_stats_head, original_lds


def _state_dicts_equal(a, b):
    if set(a.keys()) != set(b.keys()):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def test_reassembled_autoencoder_matches_original(tmp_path):
    components, original_ae, _, _ = _build_components(tmp_path)
    ae, _, _, _, _, _ = build_models_from_components(components, device="cpu")
    assert _state_dicts_equal(ae.state_dict(), original_ae.state_dict())


def test_reassembled_lds_matches_original(tmp_path):
    components, _, _, original_lds = _build_components(tmp_path)
    _, _, f_theta, _, _, _ = build_models_from_components(components, device="cpu")
    assert _state_dicts_equal(f_theta.state_dict(), original_lds.state_dict())


def test_stats_head_hidden_dim_inferred_correctly(tmp_path):
    """The whole point of inferring hidden_dim from the weight shape:
    a NON-DEFAULT hidden_dim (16, not StatsHead's default of 128) must
    still load correctly, not silently fall back to the default and
    fail (or worse, coincidentally succeed with wrong values)."""
    components, _, original_stats_head, _ = _build_components(tmp_path, stats_hidden_dim=16)
    _, stats_head, _, _, _, _ = build_models_from_components(components, device="cpu")
    assert stats_head is not None
    assert stats_head.net[0].out_features == 16
    assert _state_dicts_equal(stats_head.state_dict(), original_stats_head.state_dict())


def test_missing_stats_head_yields_none(tmp_path):
    components, _, _, _ = _build_components(tmp_path, include_stats_head=False)
    assert "stats_head" not in components
    _, stats_head, _, frozen_modules, _, _ = build_models_from_components(components, device="cpu")
    assert stats_head is None
    assert frozen_modules == []  # nothing to freeze -- no decoder freeze, no stats_head


def test_stats_head_always_frozen(tmp_path):
    components, _, _, _ = _build_components(tmp_path)
    _, stats_head, _, frozen_modules, _, _ = build_models_from_components(
        components, device="cpu", freeze_decoder=False,
    )
    assert stats_head is not None
    assert all(not p.requires_grad for p in stats_head.parameters())
    assert stats_head in frozen_modules


def test_freeze_decoder_true_freezes_decoder_not_encoder(tmp_path):
    components, _, _, _ = _build_components(tmp_path)
    ae, _, _, frozen_modules, _, _ = build_models_from_components(
        components, device="cpu", freeze_decoder=True,
    )
    assert all(not p.requires_grad for p in ae.decoder.parameters())
    assert all(p.requires_grad for p in ae.encoder.parameters())
    assert ae.decoder in frozen_modules


def test_freeze_decoder_false_leaves_decoder_trainable(tmp_path):
    components, _, _, _ = _build_components(tmp_path)
    ae, _, _, frozen_modules, _, _ = build_models_from_components(
        components, device="cpu", freeze_decoder=False,
    )
    assert all(p.requires_grad for p in ae.decoder.parameters())
    assert ae.decoder not in frozen_modules


def test_version_mismatch_shape_error_raises_clear_error(tmp_path):
    """
    A component built for a DIFFERENT latent_channels than what's passed
    to Autoencoder's own constructor should fail with a clear message,
    not a raw PyTorch RuntimeError. NOTE: this specifically exercises the
    SHAPE-mismatch path (see model_assembly.py) -- PyTorch's own
    load_state_dict raises RuntimeError directly for a shape mismatch
    even with strict=False, before ever reaching the missing/unexpected
    keys check (strict=False only relaxes THAT check, not shapes).
    """
    components, _, _, _ = _build_components(tmp_path, latent_channels=4)
    # Corrupt the encoder's reported config to claim a different
    # latent_channels than the actual saved weights have -- simulates
    # a version mismatch between the checkpoint and this codebase.
    # Both the legacy flat key AND stream_configs (which
    # resolve_stream_configs_from_checkpoint_config now reads FIRST,
    # since load_ae_components always populates it -- see that
    # function's own self-healing logic) need corrupting, or the
    # resolution would just use the still-correct stream_configs value
    # and this corruption would silently do nothing.
    components["encoder"].config["latent_channels"] = 999
    components["decoder"].config["latent_channels"] = 999
    recon_name = components["encoder"].config["recon_stream_name"]
    components["encoder"].config["stream_configs"][recon_name]["channels"] = 999
    components["decoder"].config["stream_configs"][recon_name]["channels"] = 999
    with pytest.raises(ValueError, match="doesn't match the current model definition"):
        build_models_from_components(components, device="cpu")


def test_version_mismatch_missing_key_raises_clear_error(tmp_path):
    """
    The OTHER failure mode: a genuine key-name mismatch (e.g. a future
    model definition renaming/adding a layer), where the checkpoint's
    keys simply don't overlap with the current model's -- no shape
    comparison ever happens, so this exercises the missing_keys branch
    specifically, distinct from the shape-mismatch test above.
    """
    components, _, _, _ = _build_components(tmp_path)
    del components["encoder"].state_dict["bottlenecks.state.weight"]
    with pytest.raises(ValueError, match="missing keys"):
        build_models_from_components(components, device="cpu")
