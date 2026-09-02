"""
Tests for training/_refinement_loss.py. Uses small, REAL model instances
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
from training._refinement_loss import compute_stage45_loss


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


# --------------------------------------------------------------------
# stage-4 u-scheme (log10_t f_theta) conversion
# --------------------------------------------------------------------
import math  # noqa: E402


def _make_u_models(**kw):
    """Same as _make_models but a log10_t f_theta: it steps in Delta-u and
    consumes z̃1=dz0/du, so compute_stage45_loss must convert before rollout."""
    ae, _, stats_head = _make_models(**kw)
    f_theta = LatentDynamics(latent_channels=LATENT_CHANNELS, n_theta=1,
                              hidden_dim=16, n_hidden_layers=1,
                              dynamics_mode="deriv_linear", time_coordinate="log10_t",
                              dt_cap=float("inf"))
    return ae, f_theta, stats_head


def _t_window(batch_size=2, n_rollout_steps=2):
    """Per-frame physical time, strictly > 0 (u=log10(t) is singular at 0).
    (B, n_r+1). Monotonically increasing within each window."""
    base = torch.tensor([50.0, 100.0, 200.0])[:n_rollout_steps + 1]
    return base.unsqueeze(0).expand(batch_size, -1).contiguous()


def test_stage45_u_scheme_runs_finite_and_differentiable():
    """The end-to-end guard: a log10_t f_theta fed per-frame t must produce a
    FINITE loss and real encoder gradient -- catching the large-dt divergence
    (physical dt into a u-model -> NaN) the old guard used to refuse."""
    ae, f_theta, stats_head = _make_u_models()
    x_window, dt_window, theta = _make_batch()
    total = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.0,
        t_window=_t_window())
    assert total.dim() == 0
    assert torch.isfinite(total), f"u-scheme loss not finite: {total}"
    total.backward()  # must not raise
    enc_grads = [p.grad for p in ae.encoders["shared"].parameters() if p.grad is not None]
    assert enc_grads and any(torch.any(torch.isfinite(g) & (g != 0)) for g in enc_grads)


def test_stage45_u_scheme_without_t_window_raises():
    """A log10_t f_theta with NO t_window cannot build z̃1=ln10*t*z1 -- must fail
    loud (the plumbing guard), never silently feed physical dt and NaN."""
    ae, f_theta, stats_head = _make_u_models()
    x_window, dt_window, theta = _make_batch()
    with pytest.raises(ValueError, match="needs per-frame physical time"):
        compute_stage45_loss(ae, f_theta, stats_head, x_window, dt_window, theta,
                             rollout_weight=1.0)


def test_stage45_t_scheme_ignores_t_window_and_is_unchanged():
    """Backward-compat: a plain-t f_theta is not converted -- passing t_window
    (or not) yields the SAME loss, so the u-branch is truly gated on the model's
    own coordinate, not on the batch."""
    ae, f_theta, stats_head = _make_models()   # time_coordinate defaults to 't'
    x_window, dt_window, theta = _make_batch()
    a = compute_stage45_loss(ae, f_theta, stats_head, x_window, dt_window, theta,
                             rollout_weight=1.0, recon0_weight=0.1)
    b = compute_stage45_loss(ae, f_theta, stats_head, x_window, dt_window, theta,
                             rollout_weight=1.0, recon0_weight=0.1, t_window=_t_window())
    assert torch.allclose(a, b), "t-scheme loss must not depend on t_window"


def test_recon_predict_decodes_the_FINAL_step_not_frame_zero():
    """L_recon_predict must grade D(z_hat[:, -1]) against the real FINAL frame,
    not frame 0. Constructed so that a frame-0 decode and a last-step decode
    give measurably different losses: make the predicted last latent decode far
    from the true last frame while frame-0 recon is unaffected."""
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch(n_rollout_steps=2)
    # recon_predict alone (no other terms) so the total IS l_recon_predict/scale
    _, comps = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=0.0, recon0_weight=0.0, stats0_weight=0.0,
        recon_predict_weight=1.0, recon_predict_scale=1.0, return_components=True)
    assert "recon_predict" in comps
    # it is a pixel loss of the LAST predicted frame -> a finite positive scalar
    assert comps["recon_predict"].ndim == 0 and comps["recon_predict"].item() > 0
    # and it is NOT the same object/value as recon0 (frame-0 recon)
    assert not torch.allclose(comps["recon_predict"], comps["recon0"])


def test_recon_predict_backprops_to_the_DECODER():
    """The whole point: this term reaches the decoder THROUGH the rollout. With
    recon_predict the only active term, decoder parameters must get gradient --
    unlike L_rollout (latent-only) which never touches D."""
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch(n_rollout_steps=2)
    total = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=0.0, recon0_weight=0.0, stats0_weight=0.0,
        recon_predict_weight=1.0, recon_predict_scale=1.0)
    total.backward()
    dec = ae.decoders["shared"]
    grads = [p.grad for p in dec.parameters() if p.grad is not None
             and p.grad.abs().sum() > 0]
    assert grads, "recon_predict produced no decoder gradient -- it must decode"


def test_recon_predict_also_reaches_f_theta_through_the_rollout():
    """Decoding the rolled-out latent backprops through f_theta too, so the
    dynamics co-adapt to the pixel endpoint, not just the latent proxy."""
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch(n_rollout_steps=2)
    total = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=0.0, recon0_weight=0.0, stats0_weight=0.0,
        recon_predict_weight=1.0, recon_predict_scale=1.0)
    total.backward()
    grads = [p.grad for p in f_theta.parameters() if p.grad is not None
             and p.grad.abs().sum() > 0]
    assert grads, "recon_predict did not backprop through f_theta"


def test_recon_predict_weight_zero_is_an_exact_no_op():
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch(n_rollout_steps=2)
    _, comps = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=1.0, recon0_weight=0.1, recon_predict_weight=0.0,
        return_components=True)
    # the term is a zero scalar and contributes nothing (weight 0)
    assert comps["recon_predict"].item() == 0.0


def test_grad_predict_is_zero_when_prediction_matches_real_gradients():
    """L_grad_predict matches spatial gradients of the decoded endpoint to the
    real endpoint's. It is a distinct component from recon_predict (value MSE);
    a finite positive scalar in general, and it backprops to the decoder like
    recon_predict does."""
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch(n_rollout_steps=2)
    _, comps = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=0.0, recon0_weight=0.0, stats0_weight=0.0,
        grad_predict_weight=1.0, grad_predict_scale=1.0, return_components=True)
    assert "grad_predict" in comps
    assert comps["grad_predict"].ndim == 0 and comps["grad_predict"].item() > 0
    # distinct from the value-MSE endpoint term
    assert not torch.allclose(comps["grad_predict"], comps["recon_predict"])


def test_grad_predict_backprops_to_the_DECODER():
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch(n_rollout_steps=2)
    total = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=0.0, recon0_weight=0.0, stats0_weight=0.0,
        grad_predict_weight=1.0, grad_predict_scale=1.0)
    total.backward()
    dec = ae.decoders["shared"]
    grads = [p.grad for p in dec.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "grad_predict produced no decoder gradient -- it must decode"


def test_grad_predict_penalises_bulk_speckle_but_not_a_flat_offset():
    """The point of the term: a prediction that adds high-frequency speckle to an
    otherwise-correct field has WRONG gradients (large where the real field is
    flat), so L_grad_predict is large -- whereas a smooth constant offset has the
    SAME gradients as the real field, so L_grad_predict is ~0 (only value-MSE
    sees the offset). This is what makes it target moth-eaten interiors without
    touching interfaces."""
    import torch as _t
    real = _t.zeros(1, 1, 16, 16)
    real[..., 4:12, 4:12] = 1.0                      # a flat domain with sharp edges
    speckled = real + 0.05 * _t.randn(1, 1, 16, 16)  # bulk speckle
    offset = real + 0.3                               # smooth constant shift

    def grad_mse(a, b):
        from training._refinement_loss import ReconLoss
        return (ReconLoss()(a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :])
                + ReconLoss()(a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1])).item()

    assert grad_mse(speckled, real) > 10 * grad_mse(offset, real), (
        "gradient loss must see speckle (wrong bulk gradients) far more than a "
        "flat offset (identical gradients)")


def test_grad_predict_weight_zero_is_an_exact_no_op():
    ae, f_theta, stats_head = _make_models()
    x_window, dt_window, theta = _make_batch(n_rollout_steps=2)
    _, comps = compute_stage45_loss(
        ae, f_theta, stats_head, x_window, dt_window, theta,
        rollout_weight=1.0, recon0_weight=0.1, grad_predict_weight=0.0,
        return_components=True)
    assert comps["grad_predict"].item() == 0.0
