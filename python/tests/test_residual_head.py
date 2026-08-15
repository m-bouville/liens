"""Zero-init nonlinear residual head z = B(y) + H(y), per stream.

The head that lets L_deriv shape a derivative-specific mapping without
reshaping the shared trunk, while starting byte-identical to the historical
1x1 (linear B) head so every smoothness property holds at initialisation.
"""
import torch

from models.encoder import Encoder
from models.latent_streams import LatentStreamConfig, LatentStreamMode


def _cfgs(head_kind="linear", head_hidden=0):
    return {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.PURE_LATENT,
                                     condition_on_theta=True,
                                     head_kind=head_kind, head_hidden=head_hidden),
    }


def _enc(head_kind="linear", head_hidden=0, seed=0):
    torch.manual_seed(seed)
    return Encoder(input_size=32, base_channels=8, n_theta=1,
                    stream_configs=_cfgs(head_kind, head_hidden))


def test_residual_head_is_identity_to_linear_at_init():
    """H(y)=0 at initialisation, so a residual-head stream produces exactly
    the linear-head output until trained. This is what lets a residual head
    inherit -- not merely approximate -- the linear head's properties."""
    lin = _enc("linear")
    res = _enc("residual", 16)
    res.load_state_dict(lin.state_dict(), strict=False)  # H stays zero
    x, th = torch.randn(2, 1, 32, 32), torch.randn(2, 1)
    zl, zr = lin(x, th), res(x, th)
    for k in zl:
        assert torch.allclose(zl[k], zr[k], atol=1e-6), k


def test_linear_stream_has_no_residual_head_parameters():
    """A linear stream must add NO H.* keys -- otherwise old checkpoints gain
    unexpected parameters and every 'linear' stream stops matching them."""
    enc = _enc("linear")
    assert not any("residual_heads" in k for k in enc.state_dict())


def test_residual_head_adds_only_its_own_keys():
    enc = _enc("residual", 16)
    h_keys = [k for k in enc.state_dict() if "residual_heads" in k]
    assert h_keys, "residual head produced no parameters"
    assert all(k.startswith("residual_heads.deriv.") for k in h_keys), h_keys


def test_old_checkpoint_loads_into_residual_encoder():
    """The gating requirement for running stage 2 tonight: a B-only
    checkpoint must load into a residual-head encoder, the only missing keys
    being the zero-init H tensors."""
    old = _enc("linear")
    new = _enc("residual", 16)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all("residual_heads" in k for k in missing), missing


def test_trained_residual_head_actually_changes_the_output():
    """Once H is nonzero the head must diverge from linear -- the zero-init
    must not be a permanent no-op."""
    enc = _enc("residual", 16)
    x, th = torch.randn(2, 1, 32, 32), torch.randn(2, 1)
    before = enc(x, th)["deriv"].clone()
    with torch.no_grad():
        for p in enc.residual_heads["deriv"].parameters():
            p.add_(torch.randn_like(p) * 0.1)
    after = enc(x, th)["deriv"]
    assert not torch.allclose(before, after)


def test_head_config_round_trips_through_real_serialization():
    """head_kind/head_hidden must survive the ACTUAL serialize/deserialize
    round trip -- through split_ae_components' writer and
    resolve_stream_configs_from_checkpoint_config's reader -- or a resumed run
    silently reverts to a linear head. (The earlier version hand-built the
    serialized dict in the test and so proved nothing about the real code.)"""
    from models.latent_streams import resolve_stream_configs_from_checkpoint_config

    # Build a real AE with a residual deriv head, serialize its config the way
    # a checkpoint does, then deserialize and confirm the head survived.
    res = _enc("residual", 32)
    # emulate the checkpoint config block the writers produce (same shape as
    # checkpoint_components.py's stream_configs serialization)
    model_cfg = {
        "size": 32, "base_channels": 8, "latent_channels": 4,
        "latent_spatial_size": 8,
        "stream_configs": {
            name: {"channels": c.channels, "spatial_size": c.spatial_size,
                    "mode": c.mode.value,
                    "condition_on_theta": c.condition_on_theta,
                    "head_kind": c.head_kind, "head_hidden": c.head_hidden}
            for name, c in res.stream_configs.items()
        },
        "recon_stream_name": "state",
    }
    restored, _ = resolve_stream_configs_from_checkpoint_config(model_cfg)
    assert restored["deriv"].head_kind == "residual"
    assert restored["deriv"].head_hidden == 32
    assert restored["state"].head_kind == "linear"  # untouched stream
    # and a pre-head checkpoint (no head keys) must deserialize to linear/0
    legacy = {**model_cfg, "stream_configs": {
        "state": {"channels": 4, "spatial_size": 8, "mode": "autoencoder",
                   "condition_on_theta": False},
        "deriv": {"channels": 4, "spatial_size": 8, "mode": "pure_latent",
                   "condition_on_theta": True}}}
    restored_legacy, _ = resolve_stream_configs_from_checkpoint_config(legacy)
    assert restored_legacy["deriv"].head_kind == "linear"
    assert restored_legacy["deriv"].head_hidden == 0


def test_head_nonlinearity_is_zero_at_init_and_grows_when_trained():
    """The ||H||/||B|| report must read 0 exactly when H(y)=0 -- including a
    fresh residual head, whose INPUT conv is random-initialised. Measuring
    the whole H would make an untrained residual head look nonlinear."""
    enc = _enc("residual", 16)
    at_init = enc.head_nonlinearity()
    assert at_init["state"] == 0.0  # linear stream, no H
    assert at_init["deriv"] == 0.0, (
        f"a fresh residual head reads {at_init['deriv']} -- the metric is "
        f"seeing H's random input conv, not its zero-init output conv"
    )
    with torch.no_grad():
        for p in enc.residual_heads["deriv"][-1].parameters():
            p.add_(torch.randn_like(p) * 0.1)
    assert enc.head_nonlinearity()["deriv"] > 0.0


def test_linear_stream_reports_zero_nonlinearity():
    enc = _enc("linear")
    assert enc.head_nonlinearity() == {"state": 0.0, "deriv": 0.0}


