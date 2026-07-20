"""
Tests for training/refinement_loss.py. Uses small, REAL model instances
(same pattern as test_model_assembly.py) so gradient-flow tests are
genuine, not simulated.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_refinement_loss.py -v
"""
import torch
import pytest

from models.autoencoder import MultiStreamAutoencoder
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_streams import DEFAULT_STREAM_NAME, LatentStreamConfig, LatentStreamMode
from models.latent_dynamics import LatentDynamics
from training.stats_head import StatsHead
from training.losses import StatsLoss
from training.refinement_loss import compute_stage45_loss


LATENT_CHANNELS = 4
STAT_NAMES = ["angle", "avg_phi", "stdev_phi"]


def _make_models(base_channels=4, hidden_dim=16, n_hidden_layers=1, include_stats_head=True):
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=LATENT_CHANNELS, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=LATENT_CHANNELS, spatial_size=8,
                                     mode=LatentStreamMode.DECODER),
    }
    encoder = Encoder(input_size=64, in_channels=1, base_channels=base_channels, stream_configs=stream_configs)
    decoder = Decoder(output_size=64, out_channels=1, base_channels=base_channels,
                       latent_channels=LATENT_CHANNELS, latent_spatial_size=8)
    ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                 stream_configs=stream_configs)
    f_theta = LatentDynamics(latent_channels=LATENT_CHANNELS, n_theta=1,
                              hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    stats_head = StatsHead(latent_channels=LATENT_CHANNELS, stat_names=STAT_NAMES,
                            hidden_dim=hidden_dim) if include_stats_head else None
    return ae, f_theta, stats_head


def _make_batch(batch_size=2, n_rollout_steps=2, size=64):
    torch.manual_seed(0)
    x_window = torch.randn(batch_size, n_rollout_steps + 1, 1, size, size)
    dt_window = torch.rand(batch_size, n_rollout_steps) + 0.1
    theta = torch.rand(batch_size, 1)
    return x_window, dt_window, theta


def test_forward_backward_succeeds():
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch()
    total = compute_stage45_loss(ae, f_theta, stats_head, x_window, dt_window, theta,
                                  rollout_weight=1.0, recon0_weight=0.1)
    assert total.dim() == 0
    total.backward()  # should not raise


def test_collapse_prevention_detach_is_structurally_present():
    """
    THE core mechanism this module exists for: z_true (L_rollout's
    target, built from the real continuation frames) must have
    requires_grad=False, while z0 (built from the real starting frame,
    feeding both the prediction chain and L_recon/L_stats) must have
    requires_grad=True. Checked directly and mechanistically, not
    inferred indirectly from gradient magnitudes.
    """
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch()
    _, components = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=1.0, return_components=True,
    )
    assert components["z0"].requires_grad is True
    assert components["z_true"].requires_grad is False


def test_rollout_alone_produces_encoder_gradient():
    """The prediction path (z0 -> f_theta -> z_hat, compared against the
    detached z_true) must still produce real gradient into E, even with
    recon/stats weights at zero -- L_rollout is the one term that MUST
    train E on its own."""
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch()
    total = compute_stage45_loss(ae, f_theta, stats_head, x_window, dt_window, theta,
                                  rollout_weight=1.0, recon0_weight=0.0, stats0_weight=0.0)
    total.backward()
    encoder_grads = [p.grad for p in ae.encoders["shared"].parameters() if p.grad is not None]
    assert len(encoder_grads) > 0
    assert any(torch.any(g != 0) for g in encoder_grads)


def test_decoder_gets_no_gradient_when_recon0_weight_zero():
    """L_rollout never touches D at all -- with recon0_weight=0 (and no
    stats term touching D either), D's parameters should receive
    exactly zero gradient contribution, confirming L_rollout is fully
    decoder-independent."""
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch()
    total = compute_stage45_loss(ae, f_theta, stats_head, x_window, dt_window, theta,
                                  rollout_weight=1.0, recon0_weight=0.0, stats0_weight=0.0)
    total.backward()
    for p in ae.pathways[DEFAULT_STREAM_NAME].decoder.parameters():
        assert p.grad is None or torch.all(p.grad == 0)


def test_decoder_gets_gradient_when_recon0_weight_nonzero():
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch()
    total = compute_stage45_loss(ae, f_theta, stats_head, x_window, dt_window, theta,
                                  rollout_weight=1.0, recon0_weight=1.0)
    total.backward()
    decoder_grads = [p.grad for p in ae.pathways[DEFAULT_STREAM_NAME].decoder.parameters() if p.grad is not None]
    assert len(decoder_grads) > 0
    assert any(torch.any(g != 0) for g in decoder_grads)


def test_missing_stats_head_skips_stats_term_gracefully():
    ae, f_theta, _ = _make_models(include_stats_head=False)
    x_window, dt_window, theta = _make_batch()
    total, components = compute_stage45_loss(
        ae, f_theta, None, x_window, dt_window, theta,
        rollout_weight=1.0, stats0_weight=1.0, return_components=True,
    )
    assert components["stats0"].item() == 0.0
    total.backward()  # still works fine with no stats term at all


def test_missing_stats_loss_fn_or_true_stats_also_skips_gracefully():
    ae, f_theta, stats_head = _make_models(include_stats_head=True)
    x_window, dt_window, theta = _make_batch()
    # stats_head given, but no stats_loss_fn/true_stats -- should still skip cleanly
    _, components = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=1.0, stats0_weight=1.0, return_components=True,
    )
    assert components["stats0"].item() == 0.0


def test_stats_term_matches_independent_computation():
    ae, f_theta, stats_head = _make_models(include_stats_head=True)
    x_window, dt_window, theta = _make_batch()
    mean = torch.zeros(len(STAT_NAMES))
    std = torch.ones(len(STAT_NAMES))
    stats_loss_fn = StatsLoss(mean, std, stat_names=STAT_NAMES)
    true_stats = torch.randn(x_window.shape[0], len(STAT_NAMES))

    _, components = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=1.0, stats0_weight=1.0, stats_loss_fn=stats_loss_fn,
        true_stats=true_stats, return_components=True,
    )

    with torch.no_grad():
        z0_independent = ae.encoders["shared"](x_window[:, 0])[DEFAULT_STREAM_NAME]
        pred_stats_independent = stats_head(z0_independent)
        expected_l_stats = stats_loss_fn(pred_stats_independent, true_stats)

    assert torch.isclose(components["stats0"], expected_l_stats, atol=1e-5)


def test_total_matches_manual_weighted_sum():
    ae, f_theta, stats_head = _make_models(include_stats_head=True)
    x_window, dt_window, theta = _make_batch()
    mean = torch.zeros(len(STAT_NAMES))
    std = torch.ones(len(STAT_NAMES))
    stats_loss_fn = StatsLoss(mean, std, stat_names=STAT_NAMES)
    true_stats = torch.randn(x_window.shape[0], len(STAT_NAMES))

    rollout_weight, recon0_weight, stats0_weight = 1.0, 0.3, 0.7
    total, components = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=rollout_weight, recon0_weight=recon0_weight, stats0_weight=stats0_weight,
        stats_loss_fn=stats_loss_fn, true_stats=true_stats, return_components=True,
    )
    expected_total = (rollout_weight * components["rollout"] + recon0_weight * components["recon0"]
                       + stats0_weight * components["stats0"])
    assert torch.isclose(total, expected_total, atol=1e-6)


def test_rollout_component_matches_independent_rollout_loss():
    """Cross-check against RolloutLoss directly, on the same
    (z_hat, z_true) this function itself computed internally --
    confirms compute_stage45_loss isn't silently doing something
    different from the already-tested RolloutLoss machinery."""
    from training.losses import RolloutLoss

    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch()
    _, components = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=1.0, return_components=True,
    )

    with torch.no_grad():
        batch_size, n_rollout_steps = x_window.shape[0], x_window.shape[1] - 1
        x0 = x_window[:, 0]
        x_future = x_window[:, 1:]
        x0_encoded = ae.encoders["shared"](x0)
        z0_independent = x0_encoded[DEFAULT_STREAM_NAME]
        z1_0 = x0_encoded["deriv"]
        x_future_flat = x_future.reshape(batch_size * n_rollout_steps, *x_future.shape[2:])
        x_future_encoded = ae.encoders["shared"](x_future_flat)
        z1_future = x_future_encoded["deriv"].reshape(batch_size, n_rollout_steps, *z1_0.shape[1:])
        z1_seq = torch.cat([z1_0.unsqueeze(1), z1_future], dim=1)
        z_hat_full = f_theta.rollout(z0_independent, z1_seq, dt_window, theta)
        z_hat_independent = z_hat_full[:, 1:]
        expected_rollout = RolloutLoss()(z_hat_independent, components["z_true"])

    assert torch.isclose(components["rollout"], expected_rollout, atol=1e-5)


def test_default_return_is_just_total_tensor():
    """return_components=False (the default) should return a plain
    tensor, not a tuple -- backward-compatible, minimal-surprise default."""
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch()
    result = compute_stage45_loss(ae, f_theta, stats_head, x_window, dt_window, theta)
    assert isinstance(result, torch.Tensor)
