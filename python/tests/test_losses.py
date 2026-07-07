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
