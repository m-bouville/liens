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


# ---- dt_cap -------------------------------------------------------------
#
# dt_cap caps dt ITSELF inside the second-order term only
# (f(z0,z1,theta)*(min(dt,dt_cap)^2/2)), never the first-order z1*dt
# term. Default float("inf") is an exact no-op (min(dt,inf)=dt always).
# Rationale (see __init__'s own docstring): a SATURATED f_val alone
# cannot prevent the second-order term from eventually dominating the
# first-order one, since |f_val|*(dt^2/2) still grows as dt^2 for any
# fixed, nonzero f_val -- capping dt itself is what actually guarantees
# a reversal back to euler-dominated behavior at large dt, not just a
# delay of when the crossover happens.

def _make_with_dt_cap(dt_cap, latent_channels=4, latent_spatial=4, hidden_dim=8,
                       n_hidden_layers=1, n_theta=1, bias=0.1):
    """Like _make(), but with a nonzero f (via a manual bias tweak, same
    technique the existing tests above use) and an explicit dt_cap --
    dt_cap=inf (the default) would make every dt_cap-specific test below
    vacuous, since capping at infinity never actually engages."""
    torch.manual_seed(0)
    f_theta = LatentDynamics(latent_channels=latent_channels, n_theta=n_theta,
                              latent_spatial=latent_spatial, hidden_dim=hidden_dim,
                              n_hidden_layers=n_hidden_layers, dt_cap=dt_cap)
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(bias)
    return f_theta


def test_dt_cap_default_is_exact_no_op():
    """The single most important property for backward compatibility:
    every existing caller of LatentDynamics (this whole file's own
    tests above, plus every checkpoint trained before dt_cap existed)
    must get IDENTICAL behavior with the default, not just
    "approximately the same" -- verified here across dt values spanning
    several orders of magnitude, not just one convenient case."""
    torch.manual_seed(10)
    f_theta_default = _make()  # no dt_cap given -- uses the float("inf") default
    with torch.no_grad():
        f_theta_default.net[-1].bias.fill_(0.1)
    z0 = torch.randn(5, 4, 4, 4)
    z1 = torch.randn(5, 4, 4, 4)
    theta = torch.randn(5, 1)

    for dt_val in [0.1, 1.0, 100.0, 1e4, 1e8, 1e12]:
        dt = torch.full((5,), dt_val)
        f_val = f_theta_default.f(z0, z1, theta)
        dt_r = dt.view(-1, 1, 1, 1)
        expected = z0 + z1 * dt_r + f_val * (dt_r ** 2 / 2)  # the ORIGINAL, uncapped formula
        actual = f_theta_default(z0, z1, dt, theta)
        assert torch.allclose(actual, expected, atol=1e-4), f"diverged from uncapped formula at dt={dt_val}"


def test_dt_cap_below_threshold_matches_uncapped_formula_exactly():
    """For dt < dt_cap, capping must have literally zero effect --
    min(dt, dt_cap) == dt in this regime, so a capped and an identically-
    weighted uncapped model must agree exactly, not just approximately."""
    dt_cap = 100.0
    f_capped = _make_with_dt_cap(dt_cap)
    f_uncapped = _make_with_dt_cap(float("inf"))
    f_uncapped.load_state_dict(f_capped.state_dict())  # identical weights, only dt_cap differs

    torch.manual_seed(11)
    z0 = torch.randn(3, 4, 4, 4)
    z1 = torch.randn(3, 4, 4, 4)
    theta = torch.randn(3, 1)

    for dt_val in [1.0, 50.0, 99.9]:  # all strictly below dt_cap
        dt = torch.full((3,), dt_val)
        assert torch.allclose(f_capped(z0, z1, dt, theta), f_uncapped(z0, z1, dt, theta))


def test_dt_cap_at_exact_threshold_matches_uncapped_formula():
    """Boundary case: dt == dt_cap should still match the uncapped
    formula exactly (min(x, x) == x), not be treated as "already
    capped" -- confirms no off-by-one in the clamp's own boundary."""
    dt_cap = 100.0
    f_capped = _make_with_dt_cap(dt_cap)
    torch.manual_seed(12)
    z0 = torch.randn(2, 4, 4, 4)
    z1 = torch.randn(2, 4, 4, 4)
    theta = torch.randn(2, 1)
    dt = torch.full((2,), dt_cap)

    f_val = f_capped.f(z0, z1, theta)
    dt_r = dt.view(-1, 1, 1, 1)
    expected = z0 + z1 * dt_r + f_val * (dt_r ** 2 / 2)
    assert torch.allclose(f_capped(z0, z1, dt, theta), expected, atol=1e-5)


def test_dt_cap_above_threshold_second_order_term_saturates():
    """The core, structural behavior this parameter exists for: for
    dt > dt_cap, the second-order term's own magnitude must freeze at
    its dt_cap value, NOT continue growing with the real dt -- verified
    against an independently-computed expected value using
    min(dt, dt_cap) explicitly, not just trusting forward()'s own
    internals."""
    dt_cap = 50.0
    f_theta = _make_with_dt_cap(dt_cap)
    torch.manual_seed(13)
    z0 = torch.randn(3, 4, 4, 4)
    z1 = torch.randn(3, 4, 4, 4)
    theta = torch.randn(3, 1)
    f_val = f_theta.f(z0, z1, theta)

    for dt_val in [51.0, 500.0, 1e6]:  # all strictly above dt_cap
        dt = torch.full((3,), dt_val)
        dt_r = dt.view(-1, 1, 1, 1)
        wrong_uncapped = z0 + z1 * dt_r + f_val * (dt_r ** 2 / 2)  # what it would be WITHOUT capping
        expected_capped = z0 + z1 * dt_r + f_val * (dt_cap ** 2 / 2)  # second-order term frozen at dt_cap
        actual = f_theta(z0, z1, dt, theta)
        assert torch.allclose(actual, expected_capped, atol=1e-4), f"failed to saturate at dt={dt_val}"
        assert not torch.allclose(actual, wrong_uncapped, atol=1e-2), (
            f"matches the UNCAPPED formula at dt={dt_val} -- dt_cap is having no effect at all"
        )


def test_dt_cap_never_caps_the_first_order_term():
    """z1*dt must keep growing linearly without bound past dt_cap --
    only the SECOND-order term saturates. Isolated here by using f=0
    (via bias=0.0), so the entire prediction reduces to pure z0+z1*dt,
    and confirming THAT keeps scaling correctly even at dt >> dt_cap."""
    dt_cap = 10.0
    f_theta = _make_with_dt_cap(dt_cap, bias=0.0)  # f(z0,z1,theta) = 0 exactly -- isolates the z1*dt term alone
    torch.manual_seed(14)
    z0 = torch.randn(2, 4, 4, 4)
    z1 = torch.randn(2, 4, 4, 4)
    theta = torch.randn(2, 1)

    for dt_val in [5.0, 100.0, 1e5]:  # spans below, at, and far above dt_cap
        dt = torch.full((2,), dt_val)
        expected = z0 + z1 * dt.view(-1, 1, 1, 1)  # pure euler, dt UNCAPPED
        actual = f_theta(z0, z1, dt, theta)
        assert torch.allclose(actual, expected, atol=1e-5), f"first-order term was affected by dt_cap at dt={dt_val}"


def test_dt_cap_handles_a_mixed_batch_independently_per_window():
    """A single batch with SOME windows below dt_cap and OTHERS above
    must cap each independently -- not apply a single, batch-wide
    decision based on (e.g.) the max or mean dt in the batch."""
    dt_cap = 20.0
    f_theta = _make_with_dt_cap(dt_cap)
    torch.manual_seed(15)
    z0 = torch.randn(4, 4, 4, 4)
    z1 = torch.randn(4, 4, 4, 4)
    theta = torch.randn(4, 1)
    dt = torch.tensor([5.0, 20.0, 50.0, 1000.0])  # below, at, and (twice) above dt_cap

    f_val = f_theta.f(z0, z1, theta)
    dt_r = dt.view(-1, 1, 1, 1)
    dt_capped_expected = torch.tensor([5.0, 20.0, 20.0, 20.0]).view(-1, 1, 1, 1)
    expected = z0 + z1 * dt_r + f_val * (dt_capped_expected ** 2 / 2)

    actual = f_theta(z0, z1, dt, theta)
    assert torch.allclose(actual, expected, atol=1e-4)


def test_dt_cap_applies_per_step_in_rollout_not_only_the_first_step():
    """rollout() must cap EACH step's own dt independently against
    dt_cap, not just the first one or some aggregate -- verified by
    chaining forward() manually (already-trusted, per this file's own
    pre-existing chaining test above) with a per-step dt sequence that
    straddles dt_cap in both directions across the rollout."""
    dt_cap = 15.0
    f_theta = _make_with_dt_cap(dt_cap)
    torch.manual_seed(16)
    z0 = torch.randn(2, 4, 4, 4)
    z1_sequence = torch.randn(2, 4, 4, 4, 4)  # (B, n_steps+1=4, C, H, W)
    dts = torch.tensor([[5.0, 15.0, 100.0], [1000.0, 10.0, 14.9]])  # straddles dt_cap every step
    theta = torch.randn(2, 1)

    z0_hats = f_theta.rollout(z0, z1_sequence, dts, theta)

    z0_manual = z0
    for i in range(3):
        z0_manual = f_theta(z0_manual, z1_sequence[:, i], dts[:, i], theta)
        assert torch.allclose(z0_hats[:, i + 1], z0_manual, atol=1e-4), f"step {i} diverged from manual chaining"


def test_dt_cap_gradient_flows_correctly_on_both_sides_of_the_threshold():
    """torch.clamp's own gradient is zero w.r.t. the CLAMPED tensor once
    past the boundary -- but what actually matters for training is that
    gradients w.r.t. the NETWORK'S OWN PARAMETERS stay finite and
    nonzero on both sides, since dt itself is data, not a learnable
    parameter. A broken (all-zero or NaN) gradient path here would
    silently stop training from working at all for large-dt windows.

    Perturbs the final layer's WEIGHT here, not just its bias (unlike
    _make_with_dt_cap's own default) -- with the final layer's weight
    still at its zero-init value, NO gradient reaches earlier layers at
    all regardless of dt_cap (net[-1].weight=0 blocks the chain rule at
    exactly that point), which would make this test fail for a reason
    having nothing to do with dt_cap itself."""
    dt_cap = 30.0
    f_theta = _make_with_dt_cap(dt_cap)
    with torch.no_grad():
        f_theta.net[-1].weight.fill_(0.05)  # break the zero-init on the WEIGHT too
    torch.manual_seed(17)
    z0 = torch.randn(3, 4, 4, 4)
    z1 = torch.randn(3, 4, 4, 4)
    theta = torch.randn(3, 1)
    dt = torch.tensor([5.0, 30.0, 500.0])  # below, at, above dt_cap in the SAME batch

    z0_next = f_theta(z0, z1, dt, theta)
    loss = z0_next.pow(2).mean()
    loss.backward()

    for name, param in f_theta.named_parameters():
        assert param.grad is not None, f"{name} got no gradient at all"
        assert torch.isfinite(param.grad).all(), f"{name}'s gradient has NaN/inf"
        assert param.grad.abs().sum() > 0, f"{name}'s gradient is exactly zero everywhere"


def test_f_is_independent_of_dt_cap_itself():
    """f(z0,z1,theta) never takes dt as an input at all (see class
    docstring) -- dt_cap should therefore have ZERO effect on what f()
    itself computes, only on how its output gets combined with dt
    afterward in forward(). Two models, identical weights, different
    dt_cap: f() must agree exactly."""
    f_cap_small = _make_with_dt_cap(10.0)
    f_cap_large = _make_with_dt_cap(1e6)
    f_cap_large.load_state_dict(f_cap_small.state_dict())
    torch.manual_seed(18)
    z0 = torch.randn(3, 4, 4, 4)
    z1 = torch.randn(3, 4, 4, 4)
    theta = torch.randn(3, 1)
    assert torch.equal(f_cap_small.f(z0, z1, theta), f_cap_large.f(z0, z1, theta))


def test_dt_cap_zero_fully_suppresses_the_second_order_term():
    """Extreme edge case: dt_cap=0 should reduce the whole model to pure
    Euler (min(dt, 0) = 0 for any dt >= 0, so the correction term's own
    dt-dependence vanishes entirely, regardless of how large f_val
    itself is) -- confirms this edge doesn't produce NaN/inf or some
    other degenerate failure, just a clean, predictable reduction."""
    f_theta = _make_with_dt_cap(0.0, bias=5.0)  # deliberately large f_val, to make sure IT gets suppressed too
    torch.manual_seed(19)
    z0 = torch.randn(2, 4, 4, 4)
    z1 = torch.randn(2, 4, 4, 4)
    theta = torch.randn(2, 1)
    dt = torch.tensor([10.0, 1000.0])

    actual = f_theta(z0, z1, dt, theta)
    expected = z0 + z1 * dt.view(-1, 1, 1, 1)  # pure euler -- the whole f_val*(...) term should vanish
    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.isfinite(actual).all()
