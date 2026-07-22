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


def test_uniform_weights_match_unweighted_loss():
    """Every window weighted equally (any constant) must reproduce
    today's plain, unweighted mean exactly -- a genuinely uniform
    reweighting is a no-op."""
    torch.manual_seed(5)
    z_hat = torch.randn(6, 2, 3, 4, 4)
    z_true = torch.randn(6, 2, 3, 4, 4)

    unweighted = RolloutLoss()(z_hat, z_true)
    uniform_weighted = RolloutLoss()(z_hat, z_true, weights=torch.full((6, 2), 3.7))
    assert torch.allclose(unweighted, uniform_weighted, atol=1e-6)


def test_zero_weight_window_is_equivalent_to_excluding_it_entirely():
    """The actual, independently-verifiable correctness claim: giving
    one window a weight of exactly 0 must produce EXACTLY the same
    result as computing the loss on a batch with that window physically
    removed -- not just "a smaller contribution", genuinely zero
    influence, verified against a completely independent computation
    rather than trusting the weighted formula's own arithmetic."""
    torch.manual_seed(6)
    z_hat = torch.randn(5, 2, 3, 4, 4)
    z_true = torch.randn(5, 2, 3, 4, 4)
    # Window 2 given an enormous, otherwise loss-dominating error --
    # if zero-weighting genuinely excludes it, this shouldn't matter at all.
    z_hat_with_outlier = z_hat.clone()
    z_hat_with_outlier[2] += 1000.0

    weights = torch.ones(5, 2)
    weights[2] = 0.0
    weighted_loss = RolloutLoss()(z_hat_with_outlier, z_true, weights=weights)

    # Independent computation: physically drop window 2 (indices 0,1,3,4).
    keep = [0, 1, 3, 4]
    reference_loss = RolloutLoss()(z_hat_with_outlier[keep], z_true[keep])

    assert torch.allclose(weighted_loss, reference_loss, atol=1e-5)


def test_weights_compose_correctly_with_step_weights():
    """weights (per-window) and step_weights (per-step) are independent
    knobs -- confirms applying both together matches an explicit,
    from-scratch manual computation, not just "doesn't crash together"."""
    torch.manual_seed(7)
    z_hat = torch.randn(4, 2, 3, 4, 4)
    z_true = torch.randn(4, 2, 3, 4, 4)
    weights = torch.tensor([[1.0, 2.0], [3.0, 0.5], [2.0, 1.0], [0.5, 3.0]])
    step_weights = torch.tensor([1.0, 4.0])

    loss = RolloutLoss(step_weights=step_weights)(z_hat, z_true, weights=weights)

    diff = z_hat - z_true
    per_window_step = (diff ** 2).mean(dim=(2, 3, 4))  # (4, 2)
    per_step = (per_window_step * weights).sum(dim=0) / weights.sum(dim=0)  # (2,)
    expected = (per_step * step_weights).sum() / step_weights.sum()

    assert torch.allclose(loss, expected, atol=1e-6)


def test_dt_decade_weights_matches_known_formula():
    """A constructed distribution with KNOWN, exact per-decade counts
    AND known, exact per-decade mean losses -- verifies the CORRECTED
    weight formula directly: weight_d = K / (n_d * mean_loss_d), with
    K = total_n / sum_d(1/mean_loss_d).

    decade 1 (dt in [10,100)): 8 windows, mean_loss=2.0
    decade 2 (dt in [100,1000)): 2 windows, mean_loss=5.0
    total_n=10 -> K = 10 / (1/2.0 + 1/5.0) = 10 / 0.7 = 100/7
    weight_1 = K / (8 * 2.0) = (100/7) / 16 = 25/28
    weight_2 = K / (2 * 5.0) = (100/7) / 10 = 10/7
    """
    import numpy as np
    import pytest
    from training.losses import compute_dt_decade_weights

    all_dts = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,  # decade 1: 8 windows
                         100.0, 200.0])                                    # decade 2: 2 windows
    all_losses = np.array([2.0] * 8 + [5.0] * 2)  # exact, known mean per decade
    weights_fn = compute_dt_decade_weights(all_dts, all_losses)

    K = 10.0 / (1.0 / 2.0 + 1.0 / 5.0)
    assert weights_fn.decade_weight[1] == pytest.approx(K / (8 * 2.0))
    assert weights_fn.decade_weight[2] == pytest.approx(K / (2 * 5.0))


def test_dt_decade_weights_inverts_loss_mass_not_count():
    """The exact bug this class fixes, reproduced directly: a decade
    with FEWER windows but a MUCH LARGER per-window loss (mirroring the
    real failure mode: decade 4 had the fewest windows, 220 of 3058,
    but ~2500x the per-window error of decade 1) must end up with the
    SMALLER weight, not the larger one a count-only scheme would give
    it. Count and magnitude are deliberately made to point in OPPOSITE
    directions here so a formula that (bug-style) responds to count
    alone would fail this test, while one that correctly targets loss
    mass passes it."""
    import numpy as np
    import pytest
    from training.losses import compute_dt_decade_weights

    # Decade 1: many windows (900), small per-window loss (50).
    # Decade 4: few windows (100), huge per-window loss (125,000) --
    # a ~2500x per-window magnitude gap, few windows carrying most of
    # the raw loss mass, exactly the real measurement's own shape.
    all_dts = np.concatenate([
        np.random.uniform(10, 99, size=900),        # decade 1
        np.random.uniform(10000, 99999, size=100),  # decade 4
    ])
    all_losses = np.concatenate([
        np.full(900, 50.0),
        np.full(100, 125_000.0),
    ])
    weights_fn = compute_dt_decade_weights(all_dts, all_losses)

    # The extreme-dt, huge-loss decade must get the SMALLER weight --
    # the old (buggy) count-only scheme would have given IT the larger
    # weight (fewer windows), compounding an already-huge error.
    assert weights_fn.decade_weight[4] < weights_fn.decade_weight[1], (
        "decade 4 (fewer windows, much larger per-window loss) must get "
        "the SMALLER weight -- got a larger one, meaning this is still "
        "inverting window count instead of loss mass."
    )

    # The actual point of the scheme: total post-weight loss MASS
    # (count * weight * mean_loss) should be equal across decades.
    mass_1 = 900 * weights_fn.decade_weight[1] * 50.0
    mass_4 = 100 * weights_fn.decade_weight[4] * 125_000.0
    assert mass_1 == pytest.approx(mass_4, rel=1e-6)


def test_dt_decade_weights_gives_each_decade_equal_total_loss_mass():
    """The actual point of the corrected scheme, verified directly with
    a genuinely skewed input (90 windows in one decade at one loss
    scale, 10 in another at a very different loss scale): each
    decade's own TOTAL weighted loss mass (count * weight * mean_loss)
    should be equal, not merely each decade's raw weight sum."""
    import numpy as np
    import pytest
    from training.losses import compute_dt_decade_weights

    np.random.seed(8)
    decade1_dts = np.random.uniform(10, 99, size=90)
    decade2_dts = np.random.uniform(1000, 9999, size=10)
    all_dts = np.concatenate([decade1_dts, decade2_dts])
    decade1_losses = np.random.uniform(40, 60, size=90)     # mean ~50
    decade2_losses = np.random.uniform(9000, 11000, size=10)  # mean ~10000
    all_losses = np.concatenate([decade1_losses, decade2_losses])
    weights_fn = compute_dt_decade_weights(all_dts, all_losses)

    mean_loss_1 = decade1_losses.mean()
    mean_loss_2 = decade2_losses.mean()
    mass_1 = 90 * weights_fn.decade_weight[1] * mean_loss_1
    mass_2 = 10 * weights_fn.decade_weight[3] * mean_loss_2
    assert mass_1 == pytest.approx(mass_2, rel=1e-6)


def test_dt_decade_weights_call_looks_up_correctly_on_new_tensor():
    """__call__ applied to a genuinely NEW dt tensor (not the original
    fitting data) -- confirms each element gets its own decade's own
    weight, looked up correctly on a realistic (B, n_r)-shaped tensor."""
    import numpy as np
    import pytest
    from training.losses import compute_dt_decade_weights

    all_dts = np.array([10.0] * 5 + [1000.0] * 5)  # decades 1 and 3, equal counts
    all_losses = np.array([3.0] * 5 + [7.0] * 5)   # arbitrary, unequal per-decade means
    weights_fn = compute_dt_decade_weights(all_dts, all_losses)

    new_dt = torch.tensor([[15.0, 1500.0], [50.0, 2000.0]])  # (2, 2) -- decade 1, decade 3 mixed
    weights = weights_fn(new_dt)
    assert weights.shape == new_dt.shape
    assert weights[0, 0] == pytest.approx(weights_fn.decade_weight[1])
    assert weights[0, 1] == pytest.approx(weights_fn.decade_weight[3])
    assert weights[1, 0] == pytest.approx(weights_fn.decade_weight[1])
    assert weights[1, 1] == pytest.approx(weights_fn.decade_weight[3])


def test_dt_decade_weights_clamps_out_of_range_dt_to_nearest_known_decade():
    """A dt value in a decade never seen during the original fit (e.g.
    a val-set dt slightly outside train's own range) should be clamped
    to the nearest KNOWN decade's own weight, not raise or silently
    default to 1 (which would inconsistently favor out-of-range
    windows relative to their in-range neighbors)."""
    import numpy as np
    import pytest
    from training.losses import compute_dt_decade_weights

    all_dts = np.array([10.0, 20.0, 100.0, 200.0])  # decades 1 and 2 only
    all_losses = np.array([4.0, 6.0, 8.0, 12.0])
    weights_fn = compute_dt_decade_weights(all_dts, all_losses)

    # 100000.0 is decade 5 -- never seen; should clamp to decade 2's own weight (the max known).
    out_of_range_high = weights_fn(torch.tensor([100000.0]))
    assert out_of_range_high[0] == pytest.approx(weights_fn.decade_weight[2])

    # 1.0 is decade 0 -- never seen; should clamp to decade 1's own weight (the min known).
    out_of_range_low = weights_fn(torch.tensor([1.0]))
    assert out_of_range_low[0] == pytest.approx(weights_fn.decade_weight[1])
