"""
Tests for training/refinement_loss.py. Uses small, REAL model instances
(same pattern as test_model_assembly.py) so gradient-flow tests are
genuine, not simulated.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_refinement_loss.py -v
"""
import torch
import pytest

from models.autoencoder import Autoencoder
from models.latent_dynamics import LatentDynamics
from training.stats_head import StatsHead
from training.losses import StatsLoss
from training.refinement_loss import compute_stage45_loss


LATENT_CHANNELS = 4
STAT_NAMES = ["angle", "avg_phi", "stdev_phi"]


def _make_models(base_channels=4, hidden_dim=16, n_hidden_layers=1, include_stats_head=True):
    ae = Autoencoder(size=64, channels=1, base_channels=base_channels,
                      latent_channels=LATENT_CHANNELS)
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
                                  rollout_weight=1.0, recon_weight=0.1)
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
                                  rollout_weight=1.0, recon_weight=0.0, stats_weight=0.0)
    total.backward()
    encoder_grads = [p.grad for p in ae.encoder.parameters() if p.grad is not None]
    assert len(encoder_grads) > 0
    assert any(torch.any(g != 0) for g in encoder_grads)


def test_decoder_gets_no_gradient_when_recon_weight_zero():
    """L_rollout never touches D at all -- with recon_weight=0 (and no
    stats term touching D either), D's parameters should receive
    exactly zero gradient contribution, confirming L_rollout is fully
    decoder-independent."""
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch()
    total = compute_stage45_loss(ae, f_theta, stats_head, x_window, dt_window, theta,
                                  rollout_weight=1.0, recon_weight=0.0, stats_weight=0.0)
    total.backward()
    for p in ae.decoder.parameters():
        assert p.grad is None or torch.all(p.grad == 0)


def test_decoder_gets_gradient_when_recon_weight_nonzero():
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch()
    total = compute_stage45_loss(ae, f_theta, stats_head, x_window, dt_window, theta,
                                  rollout_weight=1.0, recon_weight=1.0)
    total.backward()
    decoder_grads = [p.grad for p in ae.decoder.parameters() if p.grad is not None]
    assert len(decoder_grads) > 0
    assert any(torch.any(g != 0) for g in decoder_grads)


def test_missing_stats_head_skips_stats_term_gracefully():
    ae, f_theta, _ = _make_models(include_stats_head=False)
    x_window, dt_window, theta = _make_batch()
    total, components = compute_stage45_loss(
        ae, f_theta, None, x_window, dt_window, theta,
        rollout_weight=1.0, stats_weight=1.0, return_components=True,
    )
    assert components["stats"].item() == 0.0
    total.backward()  # still works fine with no stats term at all


def test_missing_stats_loss_fn_or_true_stats_also_skips_gracefully():
    ae, f_theta, stats_head = _make_models(include_stats_head=True)
    x_window, dt_window, theta = _make_batch()
    # stats_head given, but no stats_loss_fn/true_stats -- should still skip cleanly
    _, components = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=1.0, stats_weight=1.0, return_components=True,
    )
    assert components["stats"].item() == 0.0


def test_stats_term_matches_independent_computation():
    ae, f_theta, stats_head = _make_models(include_stats_head=True)
    x_window, dt_window, theta = _make_batch()
    mean = torch.zeros(len(STAT_NAMES))
    std = torch.ones(len(STAT_NAMES))
    stats_loss_fn = StatsLoss(mean, std, stat_names=STAT_NAMES)
    true_stats = torch.randn(x_window.shape[0], len(STAT_NAMES))

    _, components = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=1.0, stats_weight=1.0, stats_loss_fn=stats_loss_fn,
        true_stats=true_stats, return_components=True,
    )

    with torch.no_grad():
        z0_independent = ae.encoder(x_window[:, 0])
        pred_stats_independent = stats_head(z0_independent)
        expected_l_stats = stats_loss_fn(pred_stats_independent, true_stats)

    assert torch.isclose(components["stats"], expected_l_stats, atol=1e-5)


def test_total_matches_manual_weighted_sum():
    ae, f_theta, stats_head = _make_models(include_stats_head=True)
    x_window, dt_window, theta = _make_batch()
    mean = torch.zeros(len(STAT_NAMES))
    std = torch.ones(len(STAT_NAMES))
    stats_loss_fn = StatsLoss(mean, std, stat_names=STAT_NAMES)
    true_stats = torch.randn(x_window.shape[0], len(STAT_NAMES))

    rollout_weight, recon_weight, stats_weight = 1.0, 0.3, 0.7
    total, components = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=rollout_weight, recon_weight=recon_weight, stats_weight=stats_weight,
        stats_loss_fn=stats_loss_fn, true_stats=true_stats, return_components=True,
    )
    expected_total = (rollout_weight * components["rollout"] + recon_weight * components["recon"]
                       + stats_weight * components["stats"])
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
        z0_independent = ae.encoder(x_window[:, 0])
        z_hat_full = f_theta.rollout(z0_independent, dt_window, theta)
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
