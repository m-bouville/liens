"""
Tests for training/losses.py, focused on RolloutLoss's return_per_step
option -- previously only checked ad hoc with a numpy stand-in during
development, never saved as a permanent regression test.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_losses.py -v
"""
import torch

from training.losses import RolloutLoss


def test_return_per_step_default_matches_scalar_loss():
    """return_per_step=False (the default) must keep returning a plain
    scalar tensor, unchanged from before this option existed."""
    torch.manual_seed(0)
    z_hat = torch.randn(4, 3, 4, 8, 8)
    z_true = torch.randn(4, 3, 4, 8, 8)

    loss_fn = RolloutLoss()
    loss = loss_fn(z_hat, z_true)
    assert loss.dim() == 0, "default call should return a scalar"


def test_per_step_zero_equals_independent_single_step_computation():
    """
    The exact property this option exists for: per_step[0] must be
    mathematically identical to computing the loss on ONLY the first
    predicted step's data, independently -- not merely close, exactly
    equal, since it's the same computation either way (see
    RolloutLoss.forward's docstring / train_lds.py's L_1step usage).
    """
    torch.manual_seed(0)
    z_hat = torch.randn(4, 3, 4, 8, 8)
    z_true = torch.randn(4, 3, 4, 8, 8)

    loss_fn = RolloutLoss()
    loss, per_step = loss_fn(z_hat, z_true, return_per_step=True)

    assert per_step.shape == (3,)
    assert torch.isclose(loss, per_step.mean())

    # Independent computation restricted to step 0 only, with no
    # knowledge of RolloutLoss's internals beyond "it's an L2 mean".
    independent_l_1step = (z_hat[:, 0] - z_true[:, 0]).pow(2).mean()
    assert torch.isclose(per_step[0], independent_l_1step, atol=1e-6), (
        f"per_step[0] ({per_step[0].item()}) should exactly match an independent "
        f"n_rollout_steps=1 computation ({independent_l_1step.item()})"
    )


def test_step_weights_still_work_with_return_per_step():
    """Weighted reduction should still be respected in the scalar
    return even when return_per_step is also requested."""
    torch.manual_seed(0)
    z_hat = torch.randn(2, 3, 4, 8, 8)
    z_true = torch.randn(2, 3, 4, 8, 8)
    weights = torch.tensor([1.0, 2.0, 3.0])

    loss_fn = RolloutLoss(step_weights=weights)
    loss, per_step = loss_fn(z_hat, z_true, return_per_step=True)

    expected = (per_step * weights).sum() / weights.sum()
    assert torch.isclose(loss, expected, atol=1e-6)
    # per_step itself is the UNWEIGHTED per-step breakdown -- weighting
    # is only applied when collapsing to the scalar loss.
    assert not torch.isclose(loss, per_step.mean(), atol=1e-6), (
        "with non-uniform step_weights, the weighted scalar loss should "
        "differ from a plain (unweighted) mean of per_step"
    )


def test_exponent_deriv_default_matches_pre_existing_dt_oblivious_behavior():
    """exponent_deriv defaults to 1.0, which must reproduce the EXACT
    pre-existing loss (diff = z_hat - z_true directly, no dt
    dependency at all) -- both when dt is omitted entirely (old call
    signature) and when dt IS given (since z_hat-z_true already equals
    dt*err for a single transition -- Euler integration, see
    LatentDynamics' own docstring -- q=1.0 is exact backward
    compatibility, not "no reweighting" in the sense of q=0."""
    torch.manual_seed(0)
    z_hat = torch.randn(4, 2, 3, 4, 4)
    z_true = torch.randn(4, 2, 3, 4, 4)
    dt = torch.tensor([[10.0, 20.0]] * 4)

    old_style = RolloutLoss()(z_hat, z_true)  # no dt, no exponent_deriv at all
    explicit_q1_no_dt = RolloutLoss(exponent_deriv=1.0)(z_hat, z_true)
    explicit_q1_with_dt = RolloutLoss(exponent_deriv=1.0)(z_hat, z_true, dt=dt)

    assert torch.equal(old_style, explicit_q1_no_dt)
    assert torch.equal(old_style, explicit_q1_with_dt)


def test_exponent_deriv_zero_equals_pure_rate_space_error():
    """q=0.0 must equal ((z_hat-z_true)/dt)^2 exactly -- the fully
    dt-independent rate-space error, computed independently here with
    no knowledge of RolloutLoss's own internals beyond the definition
    itself."""
    torch.manual_seed(1)
    z_hat = torch.randn(4, 2, 3, 4, 4)
    z_true = torch.randn(4, 2, 3, 4, 4)
    dt = torch.tensor([[10.0, 20.0]] * 4)

    _, per_step = RolloutLoss(exponent_deriv=0.0)(z_hat, z_true, dt=dt, return_per_step=True)

    dt_b = dt.view(4, 2, 1, 1, 1)
    independent_err_sq = ((z_hat - z_true) / dt_b).pow(2).mean(dim=(0, 2, 3, 4))
    assert torch.allclose(per_step, independent_err_sq, atol=1e-6)


def test_exponent_deriv_half_matches_sqrt_dt_weighting():
    """q=0.5 must equal dt^-1 * diff^2 exactly (the algebraic
    simplification of ||dt^0.5 * err||^2), the specific value with a
    physical motivation (Brownian-increment std scaling as sqrt(dt))."""
    torch.manual_seed(2)
    z_hat = torch.randn(3, 2, 4, 4, 4)
    z_true = torch.randn(3, 2, 4, 4, 4)
    dt = torch.tensor([[5.0, 15.0]] * 3)

    _, per_step = RolloutLoss(exponent_deriv=0.5)(z_hat, z_true, dt=dt, return_per_step=True)

    dt_b = dt.view(3, 2, 1, 1, 1)
    diff = z_hat - z_true
    independent = (diff.pow(2) * dt_b.pow(-1.0)).mean(dim=(0, 2, 3, 4))
    assert torch.allclose(per_step, independent, atol=1e-6)


def test_exponent_deriv_nonzero_requires_dt():
    """A clear, immediate error -- not a silent wrong answer -- if dt
    is missing when reweighting is actually requested."""
    torch.manual_seed(3)
    z_hat = torch.randn(2, 2, 3, 4, 4)
    z_true = torch.randn(2, 2, 3, 4, 4)

    import pytest
    with pytest.raises(ValueError, match="dt"):
        RolloutLoss(exponent_deriv=0.5)(z_hat, z_true)


def test_exponent_deriv_works_with_l1_kind_too():
    """The l1 variant (|dt^q * err| = dt^(q-1) * |diff|) gets the same
    reweighting treatment, not just the default l2 kind."""
    torch.manual_seed(4)
    z_hat = torch.randn(3, 2, 3, 4, 4)
    z_true = torch.randn(3, 2, 3, 4, 4)
    dt = torch.tensor([[8.0, 16.0]] * 3)

    _, per_step = RolloutLoss(kind="l1", exponent_deriv=0.5)(
        z_hat, z_true, dt=dt, return_per_step=True)

    dt_b = dt.view(3, 2, 1, 1, 1)
    diff = z_hat - z_true
    independent = (diff.abs() * dt_b.pow(-0.5)).mean(dim=(0, 2, 3, 4))
    assert torch.allclose(per_step, independent, atol=1e-6)
