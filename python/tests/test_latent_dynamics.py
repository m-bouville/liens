"""
Tests for models/latent_dynamics.py: f_theta predicts z0's own
second-order Taylor correction, with z1 TEACHER-FORCED at its real
(ground-truth) value at every step:
    z0(t+dt) = z0(t) + z1(t)*dt + f(z0,z1,theta)*(dt^2/2)
z1's own evolution is NOT predicted by this class at all (g_theta's
future job) -- see the class docstring on why.
"""

import torch
import pytest

from models.latent_dynamics import LatentDynamics


def _make(latent_channels=4, latent_spatial=4, hidden_dim=8, n_hidden_layers=1, n_theta=1):
    torch.manual_seed(0)
    return LatentDynamics(latent_channels=latent_channels, n_theta=n_theta,
                           latent_spatial=latent_spatial, hidden_dim=hidden_dim,
                           n_hidden_layers=n_hidden_layers)


def test_f_is_exactly_zero_at_initialization():
    """The whole zero-init rationale (see class docstring) depends on
    this holding for ANY z0/z1/theta, not just by luck on one sample."""
    f_theta = _make()
    torch.manual_seed(1)
    z0 = torch.randn(5, 4, 4, 4)
    z1 = torch.randn(5, 4, 4, 4)
    theta = torch.randn(5, 1)
    assert torch.equal(f_theta.f(z0, z1, theta), torch.zeros_like(z0))


def test_forward_reduces_to_pure_euler_at_initialization():
    """At init, z0_next must be EXACTLY z0 + z1*dt (f contributes
    nothing) -- the physically sensible "trust z1's own first-order
    estimate completely" starting point this architecture is designed
    around. forward() returns ONLY z0_next now (a single tensor, not a
    tuple) -- z1 is teacher-forced, never predicted by this class."""
    f_theta = _make()
    torch.manual_seed(2)
    z0 = torch.randn(3, 4, 4, 4)
    z1 = torch.randn(3, 4, 4, 4)
    dt = torch.tensor([10.0, 20.0, 0.5])
    theta = torch.randn(3, 1)

    z0_next = f_theta(z0, z1, dt, theta)

    expected_z0_next = z0 + z1 * dt.view(-1, 1, 1, 1)
    assert torch.allclose(z0_next, expected_z0_next)


def test_forward_with_nonzero_f_matches_the_explicit_taylor_formula():
    """Once f is nonzero (post-training, simulated here by a manual
    weight tweak), z0_next must match the exact formula from the class
    docstring -- computed independently here, not just trusted from
    reading forward()'s own source."""
    f_theta = _make()
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.1)  # break the zero-init, deterministically

    torch.manual_seed(3)
    z0 = torch.randn(2, 4, 4, 4)
    z1 = torch.randn(2, 4, 4, 4)
    dt = torch.tensor([4.0, 6.0])
    theta = torch.randn(2, 1)

    f_val = f_theta.f(z0, z1, theta)
    dt_r = dt.view(-1, 1, 1, 1)
    expected_z0_next = z0 + z1 * dt_r + f_val * (dt_r ** 2 / 2)

    z0_next = f_theta(z0, z1, dt, theta)
    assert torch.allclose(z0_next, expected_z0_next, atol=1e-6)


def test_rollout_first_step_matches_forward_exactly():
    """rollout() with n_steps=1 must be identical to a single forward()
    call -- not an approximation, not off-by-one. z1_sequence here has
    2 entries (z1 at the start and at the one predicted step), matching
    the "real z1 at every step, including the one being predicted
    into" contract."""
    f_theta = _make()
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.05)
    torch.manual_seed(4)
    z0 = torch.randn(2, 4, 4, 4)
    z1_sequence = torch.randn(2, 2, 4, 4, 4)  # (B, n_steps+1=2, C, H, W)
    dt = torch.tensor([7.0, 3.0])
    theta = torch.randn(2, 1)

    z0_next_fwd = f_theta(z0, z1_sequence[:, 0], dt, theta)
    dts = dt.unsqueeze(1)  # (B, 1) -- a single step
    z0_hats = f_theta.rollout(z0, z1_sequence, dts, theta)

    assert z0_hats.shape == (2, 2, 4, 4, 4)  # (B, n_steps+1, C, H, W)
    assert torch.equal(z0_hats[:, 0], z0)  # starting point returned exactly, not re-predicted
    assert torch.allclose(z0_hats[:, 1], z0_next_fwd)


def test_rollout_uses_the_real_z1_at_each_step_not_a_predicted_one():
    """THE defining property of this design: z1_sequence[:, i] (the
    REAL, ground-truth z1 at step i) is what gets used for step i's own
    prediction -- verified here by using a z1_sequence where each step
    is a DELIBERATELY different, known value, and confirming rollout()
    picks up each one correctly rather than reusing z1_sequence[:, 0]
    throughout or ignoring the sequence structure."""
    f_theta = _make()
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.0)  # keep f=0, isolates JUST the z1-usage question
    torch.manual_seed(5)
    z0 = torch.randn(1, 4, 4, 4)
    z1_step0 = torch.randn(1, 4, 4, 4)
    z1_step1 = torch.randn(1, 4, 4, 4)  # DIFFERENT from z1_step0
    z1_sequence = torch.stack([z1_step0, z1_step1], dim=1)  # (1, 2, 4, 4, 4)
    dts = torch.tensor([[2.0, 3.0]])
    theta = torch.randn(1, 1)

    z0_hats = f_theta.rollout(z0, z1_sequence, dts, theta)

    # With f=0: z0_hats[1] = z0 + z1_step0*dt[0], z0_hats[2] = z0_hats[1] + z1_step1*dt[1]
    expected_step1 = z0 + z1_step0 * 2.0
    expected_step2 = expected_step1 + z1_step1 * 3.0
    assert torch.allclose(z0_hats[:, 1], expected_step1)
    assert torch.allclose(z0_hats[:, 2], expected_step2, atol=1e-6)


def test_rollout_chains_on_its_own_z0_prediction_not_ground_truth_z0():
    """z0's own chaining still compounds normally (only z1 is teacher-
    forced) -- confirmed by manually chaining forward() and comparing
    against rollout() step by step, not just at the final step."""
    f_theta = _make()
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.02)
    torch.manual_seed(6)
    z0 = torch.randn(2, 4, 4, 4)
    z1_sequence = torch.randn(2, 4, 4, 4, 4)  # (B, n_steps+1=4, C, H, W)
    dts = torch.tensor([[5.0, 10.0, 3.0], [8.0, 2.0, 6.0]])
    theta = torch.randn(2, 1)

    z0_hats = f_theta.rollout(z0, z1_sequence, dts, theta)

    z0_manual = z0
    for i in range(3):
        z0_manual = f_theta(z0_manual, z1_sequence[:, i], dts[:, i], theta)
        assert torch.allclose(z0_hats[:, i + 1], z0_manual, atol=1e-6)


def test_channel_count_mismatch_between_z0_and_z1_fails_loudly():
    """z0(t) + z1(t)*dt requires the same channel count for both --
    not a well-defined operation otherwise (see class docstring). A
    mismatch should fail with a clear broadcasting/shape error, not
    silently produce a wrong-shaped or garbage result."""
    f_theta = _make(latent_channels=4)
    z0 = torch.randn(2, 4, 4, 4)
    z1_wrong_channels = torch.randn(2, 6, 4, 4)  # 6 != 4
    dt = torch.tensor([1.0, 1.0])
    theta = torch.randn(2, 1)

    with pytest.raises(RuntimeError):
        f_theta(z0, z1_wrong_channels, dt, theta)


def test_final_layer_is_the_only_zero_initialized_layer_not_the_whole_net():
    """Only the FINAL layer needs zero-init for f(z0,z1,theta)=0 to
    hold overall -- earlier layers are free to have their usual
    (nonzero) random initialization, since ReLU(...) @ 0 = 0 regardless
    of what feeds into that final layer. Confirms the zero-init wasn't
    accidentally applied to (or skipped on) the wrong layer."""
    f_theta = _make()
    first_layer_weight = f_theta.net[0].weight
    assert not torch.equal(first_layer_weight, torch.zeros_like(first_layer_weight)), (
        "the net's own first layer should NOT be zero-initialized -- only the final one should"
    )
    assert torch.equal(f_theta.net[-1].weight, torch.zeros_like(f_theta.net[-1].weight))
    assert torch.equal(f_theta.net[-1].bias, torch.zeros_like(f_theta.net[-1].bias))
