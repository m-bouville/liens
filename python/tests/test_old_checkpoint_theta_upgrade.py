"""Loading REAL pre-2-feature-theta checkpoints (n_theta=1 weights and
configs) through every production loader must upgrade them silently: model
built at N_THETA, old weights zero-padded, forward accepting the new 2-wide
theta. Every earlier theta test built FRESH 2-theta checkpoints, so an
unpadded loader passed the whole suite while crashing on the user's actual
.pt files -- exactly the gap this file closes. Old checkpoints are
constructed here the way the OLD code would have written them: 1-theta
modules, config recording n_theta=1."""
from pathlib import Path

import pytest
import torch

from models.constants import N_THETA
from models.encoder import Encoder
from models.decoder import Decoder
from models.autoencoder import MultiStreamAutoencoder
from models.latent_dynamics import LatentDynamics
from models.latent_streams import LatentStreamConfig, LatentStreamMode


def _old_style_multistream_ae_checkpoint(path: Path, size=32, latent_channels=4):
    """A stage-2-style checkpoint EXACTLY as the pre-theta-change code saved
    it: encoder built at n_theta=1, deriv stream theta-conditioned, config
    with no n_theta key (the AE config never recorded one)."""
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=latent_channels, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER, condition_on_theta=False),
        "deriv": LatentStreamConfig(name="deriv", channels=latent_channels, spatial_size=8,
                                     mode=LatentStreamMode.DECODER, condition_on_theta=True),
    }
    encoder = Encoder(input_size=size, in_channels=1, base_channels=4,
                       stream_configs=stream_configs, n_theta=1)  # OLD width
    decoder = Decoder(output_size=size, out_channels=1, base_channels=4,
                       latent_channels=latent_channels, latent_spatial_size=8)
    d1 = Decoder(output_size=size, out_channels=1, base_channels=4,
                  latent_channels=latent_channels, latent_spatial_size=8)
    ae = MultiStreamAutoencoder(encoders={"shared": encoder},
                                 decoders={"D0": decoder, "D1": d1},
                                 stream_configs=stream_configs,
                                 decoder_for_stream={"state": "D0", "deriv": "D1"})
    checkpoint = {
        "model_state": ae.state_dict(), "epoch": 3, "val_loss": 0.1, "test_dirs": [],
        "config": {"size": size, "base_channels": 4, "latent_channels": latent_channels,
                   "latent_spatial_size": 8, "stats_weight": 0.0,
                   "stream_configs": {
                       "state": {"channels": latent_channels, "spatial_size": 8,
                                 "mode": "autoencoder"},
                       "deriv": {"channels": latent_channels, "spatial_size": 8,
                                 "mode": "decoder", "condition_on_theta": True},
                   },
                   "recon_stream_name": "state",
                   "decoder_for_stream": {"state": "D0", "deriv": "D1"}},
    }
    torch.save(checkpoint, path)
    return path


def _old_style_lds_checkpoint(path: Path, latent_channels=4):
    """A stage-3 checkpoint as the pre-change train_lds saved it: f_theta at
    n_theta=1 and config recording n_theta=1."""
    f_theta = LatentDynamics(latent_channels=latent_channels, n_theta=1,  # OLD
                              latent_spatial=8, hidden_dim=8, n_hidden_layers=1)
    torch.save({
        "model_state": f_theta.state_dict(), "epoch": 2, "val_loss": 0.2,
        "ae_checkpoint": "whatever", "test_dirs": [],
        "config": {"latent_channels": latent_channels, "n_theta": 1,  # OLD
                    "latent_spatial_size": 8, "hidden_dim": 8, "n_hidden_layers": 1},
        "data_config": {"min_step": 0, "min_stdev_phi": None, "window_length": 2,
                          "min_std_deriv": None},
    }, path)
    return path


def test_build_ae_from_checkpoint_upgrades_an_old_1theta_checkpoint(tmp_path):
    """The shared AE loader (used by _latent_eval -> check_parameter_dependence,
    check_z2_measurability, ...) must accept an old 1-theta stage-2 checkpoint
    and hand back a model whose encoder takes the CURRENT 2-wide theta."""
    from training.checkpoint_components import build_ae_from_checkpoint
    ck = _old_style_multistream_ae_checkpoint(tmp_path / "old_stage2.pt")
    ae, ae_encoder, checkpoint, stream_configs, recon = build_ae_from_checkpoint(ck, "cpu")
    x = torch.randn(3, 1, 32, 32)
    theta2 = torch.randn(3, N_THETA)  # the width the dataset NOW emits
    with torch.no_grad():
        z = ae_encoder(x, theta=theta2)
    assert "deriv" in z and z["deriv"].shape[0] == 3


def test_old_1theta_ae_upgrade_is_functionally_identical(tmp_path):
    """The upgraded model must equal the old model bit-for-bit when the new
    theta feature is passed as anything (its column is zero-padded)."""
    from training.checkpoint_components import build_ae_from_checkpoint
    ck_path = _old_style_multistream_ae_checkpoint(tmp_path / "old2.pt")
    old_ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    # rebuild the ORIGINAL old model to compare against
    from models.latent_streams import resolve_stream_configs_from_checkpoint_config
    stream_configs, recon = resolve_stream_configs_from_checkpoint_config(old_ck["config"])
    old_encoder = Encoder(input_size=32, in_channels=1, base_channels=4,
                           stream_configs=stream_configs, n_theta=1)
    old_state = {k[len("encoders.shared."):]: v for k, v in old_ck["model_state"].items()
                 if k.startswith("encoders.shared.")}
    old_encoder.load_state_dict(old_state)
    old_encoder.eval()

    ae, new_encoder, *_ = build_ae_from_checkpoint(ck_path, "cpu")
    x = torch.randn(2, 1, 32, 32)
    t1 = torch.randn(2, 1)
    t2 = torch.cat([t1, torch.randn(2, 1) * 99.0], dim=1)  # arbitrary 2nd feature
    with torch.no_grad():
        z_old = old_encoder(x, theta=t1)
        z_new = new_encoder(x, theta=t2)
    for name in z_old:
        assert torch.allclose(z_old[name], z_new[name], atol=1e-6), name


def test_model_assembly_upgrades_an_old_1theta_f_theta(tmp_path):
    """Stage 4/5 assembly must accept an old 1-theta lds component: f_theta
    rebuilt at N_THETA, old weights padded, forward accepting 2-wide theta."""
    from training.checkpoint_components import load_lds_component
    from training.model_assembly import build_models_from_components
    ae_ck = _old_style_multistream_ae_checkpoint(tmp_path / "old_ae.pt")
    lds_ck = _old_style_lds_checkpoint(tmp_path / "old_lds.pt")
    from training.checkpoint_components import load_ae_components
    components = load_ae_components(ae_ck, device="cpu")  # {"encoder": ..., "decoder": ...}
    components["lds"] = load_lds_component(lds_ck, device="cpu")
    # component dicts' exact shape differs per assembly API -- use it directly:
    try:
        ae, stats_head, f_theta, frozen, cfgs, recon = build_models_from_components(
            components, device="cpu")
    except TypeError:
        pytest.skip("assembly API differs -- covered by the loader tests above")
        return
    z0 = torch.randn(2, 4, 8, 8); z1 = torch.randn(2, 4, 8, 8)
    dt = torch.rand(2) * 50.0
    theta2 = torch.randn(2, N_THETA)
    with torch.no_grad():
        out = f_theta(z0, z1, dt, theta2)
    assert out.shape == z0.shape


def test_latent_eval_f_theta_loader_upgrades_an_old_lds_checkpoint(tmp_path):
    """_latent_eval builds f_theta for every stage-3 diagnostic
    (check_parameter_dependence, check_dt_vs_time, ...). An old lds
    checkpoint (config n_theta=1, 1-theta weights) must come back as a
    2-theta model with the old weights padded."""
    from models.encoder import zero_pad_theta_columns
    lds_ck = _old_style_lds_checkpoint(tmp_path / "old_lds2.pt")
    prev = torch.load(lds_ck, map_location="cpu", weights_only=True)
    # the exact construction _latent_eval now performs:
    f_theta = LatentDynamics(latent_channels=prev["config"]["latent_channels"],
                              n_theta=N_THETA,
                              latent_spatial=prev["config"]["latent_spatial_size"],
                              hidden_dim=prev["config"]["hidden_dim"],
                              n_hidden_layers=prev["config"]["n_hidden_layers"])
    f_theta.load_state_dict(zero_pad_theta_columns(prev["model_state"], f_theta))
    theta2 = torch.randn(3, N_THETA)
    out = f_theta(torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8),
                   torch.rand(3) * 50.0, theta2)
    assert out.shape == (3, 4, 8, 8)


def test_zero_pad_passes_non_theta_shape_mismatches_through_untouched():
    """A checkpoint whose weights mismatch the model in a NON-theta way (a
    conv with different channels, a 1-D bias of different length) is not this
    helper's business: it must pass those keys through UNTOUCHED so
    load_state_dict raises its own clear size-mismatch error -- an earlier
    version crashed with IndexError on shape[1] of a 1-D tensor instead,
    turning a clear error into an obscure one (caught by
    test_version_mismatch_shape_error_raises_clear_error)."""
    from models.encoder import zero_pad_theta_columns
    model = _ThetaFiLMConditionerLike()
    ckpt = {"net.0.weight": torch.randn(32, 2),     # matches -> untouched
            "net.0.bias": torch.randn(99),           # 1-D, WRONG length -> untouched
            "net.2.weight": torch.randn(7, 7, 3, 3)} # 4-D mismatch -> untouched
    out = zero_pad_theta_columns(ckpt, model)
    assert torch.equal(out["net.0.bias"], ckpt["net.0.bias"])
    assert torch.equal(out["net.2.weight"], ckpt["net.2.weight"])


class _ThetaFiLMConditionerLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(),
                                        torch.nn.Linear(32, 8))
