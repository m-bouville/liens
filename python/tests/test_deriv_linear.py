"""
Tests for dynamics_mode='deriv_linear' (Step A: f takes dt as an input and the
second-order term uses a LINEAR prefactor f*dt, not f*dt^2/2) and for the
rollout guard that governs when it may run.

Written after the guard boundary was gotten WRONG twice: first placed at
rollout() entry (rejected any deriv_linear+z1_resync=False call, taking down
1-step diagnostics that never propagate z1), then narrowed to fire only on the
actual undefined OPERATION -- propagating z1 across MORE THAN ONE step. The
truth table in test_rollout_guard_fires_only_on_autonomous_multistep is exactly
that boundary; it is the regression guard for it.

Mirrors test_latent_dynamics.py's style: local _make helpers, manual seeds,
nonzero f simulated by setting the (otherwise zero-init) final-layer bias.
Can be merged into test_latent_dynamics.py or kept standalone.
"""
import pytest
import torch

from models.latent_dynamics import LatentDynamics


def _make_taylor(latent_channels=4, latent_spatial=4, hidden_dim=8,
                  n_hidden_layers=1, n_theta=1, **kw):
    torch.manual_seed(0)
    return LatentDynamics(latent_channels=latent_channels, n_theta=n_theta,
                           latent_spatial=latent_spatial, hidden_dim=hidden_dim,
                           n_hidden_layers=n_hidden_layers, **kw)


def _make_dl(**kw):
    # deriv_linear REQUIRES dt_cap=inf (default) and full-step (n_substeps=1,
    # alpha None, both defaults), so the plain constructor is legal.
    return _make_taylor(dynamics_mode="deriv_linear", **kw)


# --------------------------------------------------------------------------
# The default (z1_taylor) is untouched by adding the mode flag.
# --------------------------------------------------------------------------

def test_z1_taylor_is_the_default():
    assert _make_taylor().dynamics_mode == "z1_taylor"


def test_z1_taylor_forward_still_uses_the_dt_squared_taylor_form():
    """Adding dynamics_mode must not perturb the historical path: with
    nonzero f, z1_taylor forward stays z0 + z1*dt + f*dt^2/2."""
    f_theta = _make_taylor()
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.05)
    torch.manual_seed(2)
    z0 = torch.randn(3, 4, 4, 4)
    z1 = torch.randn(3, 4, 4, 4)
    dt = torch.tensor([7.0, 3.0, 0.5])
    theta = torch.randn(3, 1)
    dt_r = dt.view(-1, 1, 1, 1)

    f_val = f_theta.f(z0, z1, theta)          # z1_taylor f takes NO dt
    expected = z0 + z1 * dt_r + f_val * (dt_r ** 2 / 2)
    assert torch.allclose(f_theta(z0, z1, dt, theta), expected)


# --------------------------------------------------------------------------
# deriv_linear: the linear prefactor and the dt input.
# --------------------------------------------------------------------------

def test_deriv_linear_adds_exactly_one_input_channel_for_dt():
    """The log(dt) input widens net[0] by exactly one column vs z1_taylor --
    the shape that a cross-mode resume would (correctly) reject."""
    assert (_make_dl().net[0].in_features
            == _make_taylor().net[0].in_features + 1)


def test_deriv_linear_forward_is_linear_in_f_not_quadratic():
    """z0_next = z0 + z1*dt + f*dt (LINEAR), not f*dt^2/2. Checked against the
    linear formula AND shown to DIFFER from the Taylor formula (dt != 2 so the
    two prefactors, dt and dt^2/2, are unequal)."""
    f_theta = _make_dl()
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.05)   # constant nonzero f (final weight still 0)
    torch.manual_seed(3)
    z0 = torch.randn(3, 4, 4, 4)
    z1 = torch.randn(3, 4, 4, 4)
    dt = torch.tensor([7.0, 3.0, 0.5])     # none equal 2 -> dt != dt^2/2
    theta = torch.randn(3, 1)
    dt_r = dt.view(-1, 1, 1, 1)

    f_val = f_theta.f(z0, z1, theta, dt=dt_r)
    linear = z0 + z1 * dt_r + f_val * dt_r
    taylor = z0 + z1 * dt_r + f_val * (dt_r ** 2 / 2)

    got = f_theta(z0, z1, dt, theta)
    assert torch.allclose(got, linear)
    assert not torch.allclose(got, taylor)


def test_deriv_linear_still_reduces_to_pure_euler_at_init():
    """Zero-init final layer => f=0 => z0 + z1*dt, same sane start as z1_taylor."""
    f_theta = _make_dl()
    torch.manual_seed(2)
    z0 = torch.randn(3, 4, 4, 4)
    z1 = torch.randn(3, 4, 4, 4)
    dt = torch.tensor([10.0, 20.0, 0.5])
    theta = torch.randn(3, 1)
    assert torch.allclose(f_theta(z0, z1, dt, theta),
                          z0 + z1 * dt.view(-1, 1, 1, 1))


def test_deriv_linear_f_requires_dt_argument():
    """f() is a network input in this mode: a direct caller (e.g. check_f_theta)
    that omits dt must fail loudly, not silently pass the wrong-width tensor."""
    f_theta = _make_dl()
    torch.manual_seed(1)
    z0 = torch.randn(2, 4, 4, 4)
    z1 = torch.randn(2, 4, 4, 4)
    theta = torch.randn(2, 1)
    with pytest.raises(ValueError, match="requires dt"):
        f_theta.f(z0, z1, theta)            # no dt
    # ...and succeeds when dt is supplied.
    dt = torch.tensor([2.0, 5.0]).view(-1, 1, 1, 1)
    assert f_theta.f(z0, z1, theta, dt=dt).shape == z0.shape


# --------------------------------------------------------------------------
# Constructor guards: full-step object, no dt_cap.
# --------------------------------------------------------------------------

def test_deriv_linear_forbids_finite_dt_cap():
    with pytest.raises(ValueError, match="forbids a finite dt_cap"):
        _make_taylor(dynamics_mode="deriv_linear", dt_cap=125.0)


def test_deriv_linear_forbids_substepping():
    with pytest.raises(ValueError, match="requires n_substeps=1"):
        _make_taylor(dynamics_mode="deriv_linear", n_substeps=4)
    with pytest.raises(ValueError, match="requires n_substeps=1"):
        _make_taylor(dynamics_mode="deriv_linear", alpha=0.5)


def test_z1_taylor_still_allows_dt_cap_and_substepping():
    """The guards are deriv_linear-only; the historical mode keeps both."""
    _make_taylor(dt_cap=125.0)              # no raise
    _make_taylor(n_substeps=4)              # no raise


# --------------------------------------------------------------------------
# Capability property.
# --------------------------------------------------------------------------

def test_supports_autonomous_rollout_property():
    assert _make_taylor().supports_autonomous_rollout is True
    assert _make_dl().supports_autonomous_rollout is False


# --------------------------------------------------------------------------
# THE guard truth table -- the boundary gotten wrong twice.
# --------------------------------------------------------------------------

def _rollout_inputs(n_steps, batch=2, c=4, s=4, seed=7):
    torch.manual_seed(seed)
    z0 = torch.randn(batch, c, s, s)
    z1_sequence = torch.randn(batch, n_steps + 1, c, s, s)
    dts = torch.rand(batch, n_steps) * 5 + 1.0     # positive dt (log(dt) is taken)
    theta = torch.randn(batch, 1)
    return z0, z1_sequence, dts, theta


def test_deriv_linear_one_step_no_resync_does_not_raise_and_equals_forward():
    """The case that used to crash: a SINGLE step never propagates z1, so
    z1_resync is moot -- it must run (via forward()), not raise."""
    f_theta = _make_dl()
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.03)
    z0, z1_sequence, dts, theta = _rollout_inputs(n_steps=1)

    z0_hats = f_theta.rollout(z0, z1_sequence, dts, theta, z1_resync=False)
    z0_fwd = f_theta(z0, z1_sequence[:, 0], dts[:, 0], theta)

    assert z0_hats.shape == (2, 2, 4, 4, 4)
    assert torch.equal(z0_hats[:, 0], z0)
    assert torch.allclose(z0_hats[:, 1], z0_fwd)


def test_deriv_linear_teacher_forced_multistep_does_not_raise():
    """z1_resync=True never propagates z1 (it is reset from the sequence each
    step), so any horizon is fine and equals iterated forward()."""
    f_theta = _make_dl()
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.02)
    z0, z1_sequence, dts, theta = _rollout_inputs(n_steps=3)

    z0_hats = f_theta.rollout(z0, z1_sequence, dts, theta, z1_resync=True)

    z0_manual = z0
    for i in range(3):
        z0_manual = f_theta(z0_manual, z1_sequence[:, i], dts[:, i], theta)
        assert torch.allclose(z0_hats[:, i + 1], z0_manual, atol=1e-6)


def test_deriv_linear_autonomous_multistep_raises():
    """>1 step with z1_resync=False is the ONE undefined case (no z1-update
    equation to propagate z1 across steps) -- and the only one that raises."""
    f_theta = _make_dl()
    z0, z1_sequence, dts, theta = _rollout_inputs(n_steps=2)
    with pytest.raises(ValueError, match="autonomously for >1 step"):
        f_theta.rollout(z0, z1_sequence, dts, theta, z1_resync=False)


def test_z1_taylor_autonomous_multistep_is_unchanged():
    """The narrowed guard is deriv_linear-only: z1_taylor still rolls out
    autonomously across many steps (through _integrate) without raising."""
    f_theta = _make_taylor()
    z0, z1_sequence, dts, theta = _rollout_inputs(n_steps=3)
    z0_hats = f_theta.rollout(z0, z1_sequence, dts, theta, z1_resync=False)
    assert z0_hats.shape == (2, 4, 4, 4, 4)


# --------------------------------------------------------------------------
# q-scheme (Step B): derivative_source='previous_quotient'.
# --------------------------------------------------------------------------

def _make_q(**kw):
    return _make_taylor(dynamics_mode="deriv_linear",
                        derivative_source="previous_quotient", **kw)


def test_previous_quotient_requires_deriv_linear():
    with pytest.raises(ValueError, match="only defined for dynamics_mode"):
        _make_taylor(dynamics_mode="z1_taylor", derivative_source="previous_quotient")


def test_previous_quotient_supports_autonomous_rollout():
    assert _make_q().supports_autonomous_rollout is True
    assert _make_dl().supports_autonomous_rollout is False   # derivative_source='z1'


def test_q_scheme_autonomous_multistep_does_not_raise():
    f_theta = _make_q()
    z0, z1_sequence, dts, theta = _rollout_inputs(n_steps=3)
    z0_hats = f_theta.rollout(z0, z1_sequence, dts, theta, z1_resync=False)
    assert z0_hats.shape == (2, 4, 4, 4, 4)
    assert torch.equal(z0_hats[:, 0], z0)


def test_q_scheme_propagates_the_backward_quotient_of_its_own_trajectory():
    """Step 0 uses the seed z1; each later step feeds q_i=(z0_i-z0_{i-1})/dt_{i-1}.
    Reconstruct the rollout by hand from forward() and the quotient rule."""
    f_theta = _make_q()
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.02)
    z0, z1_sequence, dts, theta = _rollout_inputs(n_steps=3)

    z0_hats = f_theta.rollout(z0, z1_sequence, dts, theta, z1_resync=False)

    # hand rollout
    z0_cur = z0
    z0_prev = z0
    deriv = z1_sequence[:, 0]
    manual = [z0]
    for i in range(3):
        if i > 0:
            deriv = (z0_cur - z0_prev) / dts[:, i - 1].view(-1, 1, 1, 1)
        z0_prev = z0_cur
        z0_cur = f_theta(z0_cur, deriv, dts[:, i], theta)
        manual.append(z0_cur)
    manual = torch.stack(manual, dim=1)
    assert torch.allclose(z0_hats, manual, atol=1e-6)

    # step 1's derivative is literally the step-0 quotient, not the seed z1
    q1 = (z0_hats[:, 1] - z0_hats[:, 0]) / dts[:, 0].view(-1, 1, 1, 1)
    expected_step2 = f_theta(z0_hats[:, 1], q1, dts[:, 1], theta)
    assert torch.allclose(z0_hats[:, 2], expected_step2, atol=1e-6)


def test_q_scheme_one_step_equals_forward_with_seed():
    """At one step the q-scheme is just forward() on the seed derivative."""
    f_theta = _make_q()
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.02)
    z0, z1_sequence, dts, theta = _rollout_inputs(n_steps=1)
    z0_hats = f_theta.rollout(z0, z1_sequence, dts, theta, z1_resync=False)
    assert torch.allclose(z0_hats[:, 1],
                          f_theta(z0, z1_sequence[:, 0], dts[:, 0], theta))
