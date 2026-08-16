"""Two-feature theta conditioning [T-T0, log(T0-T)] and its zero-pad upgrade."""
import math
import torch
import pytest

from models.constants import N_THETA, theta_coordinates
from models.encoder import _ThetaFiLMConditioner, zero_pad_theta_columns
from models.latent_dynamics import LatentDynamics


def test_theta_coordinates_are_the_two_physical_features():
    # T=0.9, T0=1.0 -> [T-T0, log(T0-T)] = [-0.1, log(0.1)]
    c = theta_coordinates(0.9, 1.0)
    assert len(c) == N_THETA == 2
    assert c[0] == pytest.approx(-0.1)
    assert c[1] == pytest.approx(math.log(0.1))


def test_theta_coordinates_reject_supercritical():
    # T >= T0: log(T0-T) undefined; a supercritical run has no phases to evolve
    with pytest.raises(ValueError, match="T < T0"):
        theta_coordinates(1.0, 1.0)
    with pytest.raises(ValueError, match="T < T0"):
        theta_coordinates(1.2, 1.0)


def test_log_feature_separates_near_critical_temperatures():
    # the whole point: 0.95 and 0.99 are crowded in T-T0 but separated in log
    lin = abs(theta_coordinates(0.95, 1.0)[0] - theta_coordinates(0.99, 1.0)[0])
    log = abs(theta_coordinates(0.95, 1.0)[1] - theta_coordinates(0.99, 1.0)[1])
    assert lin == pytest.approx(0.04)          # crowded
    assert log > 1.5                            # well separated (log0.05 vs log0.01)


def test_film_1theta_checkpoint_upgrades_to_2theta_bit_identically():
    torch.manual_seed(0)
    old = _ThetaFiLMConditioner(n_theta=1, n_channels=8)
    # CRITICAL: the conditioner zero-inits its OUTPUT layer, so at init theta
    # has no effect and the test would pass even for a WRONG (non-zero) pad of
    # the input column. De-zero the output layer first so feature-0 genuinely
    # drives the output; only then does zero-vs-nonzero padding of the new
    # input column actually show up. (This is the "toy regime can't exhibit
    # the behavior" trap -- an untrained conditioner is inert in theta.)
    with torch.no_grad():
        old.net[-1].weight.normal_()
        old.net[-1].bias.normal_()
    new = _ThetaFiLMConditioner(n_theta=2, n_channels=8)
    new.load_state_dict(zero_pad_theta_columns(old.state_dict(), new))
    x = torch.randn(4, 8, 8, 8)
    t = torch.randn(4, 1)
    # feature-0 identical, feature-1 arbitrary; a ZERO pad must ignore it, and
    # (crucially) feature-0 must still drive the output so a non-zero pad breaks
    t2 = torch.cat([t, torch.randn(4, 1) * 99.0], dim=1)
    with torch.no_grad():
        o_old, o_new = old(x, t), new(x, t2)
        assert torch.allclose(o_old, o_new, atol=1e-6)
        # guard against the test passing trivially: theta must actually matter
        assert not torch.allclose(o_old, old(x, torch.zeros_like(t)), atol=1e-6)


def test_f_theta_1theta_checkpoint_upgrades_to_2theta_bit_identically():
    torch.manual_seed(1)
    d_old = LatentDynamics(latent_channels=8, n_theta=1, latent_spatial=8)
    # f_theta also zero-inits its output layer; de-zero it so theta drives the
    # output and a wrong (non-zero) pad of the new theta column would show.
    with torch.no_grad():
        lin_layers = [m for m in d_old.modules() if isinstance(m, torch.nn.Linear)]
        lin_layers[-1].weight.normal_(std=0.01)
        lin_layers[-1].bias.normal_(std=0.01)
    d_new = LatentDynamics(latent_channels=8, n_theta=2, latent_spatial=8)
    d_new.load_state_dict(zero_pad_theta_columns(d_old.state_dict(), d_new))
    z0, z1 = torch.randn(4, 8, 8, 8), torch.randn(4, 8, 8, 8)
    dt = torch.rand(4) * 100.0
    t1 = torch.randn(4, 1)
    t2 = torch.cat([t1, torch.randn(4, 1) * 99.0], dim=1)
    with torch.no_grad():
        o_old, o_new = d_old(z0, z1, dt, t1), d_new(z0, z1, dt, t2)
        assert torch.allclose(o_old, o_new, atol=1e-6)
        # theta must actually matter, or the identity is vacuous
        assert not torch.allclose(o_old, d_old(z0, z1, dt, torch.zeros_like(t1)),
                                   atol=1e-6)


def test_zero_pad_is_noop_at_matching_width():
    torch.manual_seed(2)
    m = _ThetaFiLMConditioner(n_theta=2, n_channels=8)
    padded = zero_pad_theta_columns(m.state_dict(), m)
    for k in m.state_dict():
        assert torch.equal(padded[k], m.state_dict()[k])


def test_zero_pad_refuses_to_shrink():
    # checkpoint WIDER than model is a real mismatch, not an upgrade
    wide = _ThetaFiLMConditioner(n_theta=3, n_channels=8)
    narrow = _ThetaFiLMConditioner(n_theta=2, n_channels=8)
    with pytest.raises(ValueError, match="WIDER"):
        zero_pad_theta_columns(wide.state_dict(), narrow)
