"""
Tests for training/losses.py, focused on RolloutLoss's return_per_step
option -- previously only checked ad hoc with a numpy stand-in during
development, never saved as a permanent regression test.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_losses.py -v
"""
import torch

from training.losses import StatsLoss, stats0_predict_loss, z0_growth_loss

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


# ---------------------------------------------------------------------
# stats0_predict_loss -- self-consistent latent-space stats loss.
# REGRESSION: it once routed through StatsLoss._wrapped_diff, which re-normalized
# its (already-normalized) second argument, adding a constant -mean/std offset that
# dominated the difference and FROZE the term model-independent. These pin that the
# term is identity-zero and actually descends as the prediction approaches truth.
# ---------------------------------------------------------------------

class _IdentityStatsHead(torch.nn.Module):
    """A stub 'stats head': flattens the latent and takes the first `n_stats` means as
    the (already-normalized) statistics. Deterministic, differentiable -- enough to
    exercise the comparison logic without a trained head."""
    def __init__(self, n_stats=4):
        super().__init__()
        self.n_stats = n_stats

    def forward(self, z):                                   # z: (N, C, H, W)
        flat = z.flatten(1)
        # n_stats simple linear reductions of the latent -> "normalized" stats
        return torch.stack([flat[:, i::self.n_stats].mean(dim=1) for i in range(self.n_stats)], dim=1)


def _stats_loss(n_stats=4):
    # mean/std chosen LARGE and off-zero: the old double-normalization bug was worst
    # exactly when mean/std are big, so this makes a regressed version fail loudly.
    mean = torch.tensor([5.0, -3.0, 10.0, 2.0][:n_stats])
    std = torch.tensor([2.0, 0.5, 4.0, 1.5][:n_stats])
    return StatsLoss(mean, std, stat_names=["avg_phi", "energy", "stdev_phi", "gradient_sqr"][:n_stats])


def test_stats0_predict_is_exactly_zero_when_prediction_equals_truth():
    """The sharp anti-freeze test: identical z_hat and z_true must give EXACTLY 0.
    The double-normalization bug (_wrapped_diff on two head outputs) returns a
    constant (-mean/std)^2 offset here instead of 0."""
    head, sl = _IdentityStatsHead(), _stats_loss()
    z = torch.randn(2, 3, 4, 4)
    z_hat = z.unsqueeze(1).repeat(1, 3, 1, 1, 1)            # (B, n, C, H, W)
    loss = stats0_predict_loss(head, z_hat, z_hat, sl)
    assert loss.item() == 0.0, "identical prediction and truth must give exactly 0 (not a constant offset)"


def test_stats0_predict_descends_as_prediction_approaches_truth():
    """It must DECREASE as z_hat -> z_true. The frozen (double-normalized) version was
    model-independent, so this monotone descent would not hold."""
    head, sl = _IdentityStatsHead(), _stats_loss()
    z_true = torch.randn(2, 3, 4, 4).unsqueeze(1).repeat(1, 3, 1, 1, 1)
    noise = torch.randn_like(z_true)
    losses = [stats0_predict_loss(head, z_true + f * noise, z_true, sl).item()
              for f in (1.0, 0.5, 0.1, 0.0)]
    assert losses == sorted(losses, reverse=True), f"must descend toward 0, got {losses}"
    assert losses[-1] == 0.0


# ---------------------------------------------------------------------
# z0_growth_loss -- symmetric latent-norm growth penalty.
# ---------------------------------------------------------------------

def test_z0_growth_is_zero_for_a_constant_norm():
    z0 = torch.ones(2, 3, 4, 4)                             # norm 1
    z0_hat = torch.ones(2, 3, 3, 4, 4)                      # (B, n, C, H, W), same norm
    assert z0_growth_loss(z0, z0_hat).item() < 1e-12, "constant norm -> ~0 growth penalty"


def test_z0_growth_is_geometrically_symmetric_explode_equals_collapse():
    """The key property: x10 per step and /10 per step must cost the SAME (the log
    makes it symmetric). A linear |ratio-1| penalty would score explosion far higher."""
    C = torch.ones(1, 2, 4, 4)                              # unit-norm predecessor
    steps = torch.arange(1, 4).float()                     # 1,2,3
    explode = torch.stack([(10.0 ** k) * torch.ones(1, 2, 4, 4) for k in steps], dim=1)  # x10/step
    collapse = torch.stack([(10.0 ** -k) * torch.ones(1, 2, 4, 4) for k in steps], dim=1)  # /10/step
    l_up = z0_growth_loss(C, explode).item()
    l_dn = z0_growth_loss(C, collapse).item()
    assert abs(l_up - l_dn) < 1e-6, f"explosion and collapse must cost equally, got {l_up} vs {l_dn}"
    assert l_up > 0


# =====================================================================
# Tier 3 coverage additions: the remaining public losses that had no
# direct tests -- ReconLoss, InterpLoss, OneStepLoss, centered_deriv_target,
# dt_weighted_deriv_loss, StatsLoss.forward/per_stat_mse, z0_scale_loss.
# Each independent computation below is written from the docstring
# definition, not from re-reading the implementation.
# =====================================================================

import pytest

from training.losses import (
    ReconLoss,
    InterpLoss,
    OneStepLoss,
    centered_deriv_target,
    dt_weighted_deriv_loss,
    z0_scale_loss,
)


# --- ReconLoss -------------------------------------------------------

def test_recon_loss_l2_is_mean_squared_error():
    """Default kind='l2' is a MEAN (not summed) squared error -- the
    deviation from the written summed-norm formula that keeps the loss
    scale resolution-independent (see ReconLoss's own docstring)."""
    torch.manual_seed(0)
    x_recon = torch.randn(3, 1, 8, 8)
    x = torch.randn(3, 1, 8, 8)
    loss = ReconLoss()(x_recon, x)
    assert torch.allclose(loss, (x_recon - x).pow(2).mean(), atol=1e-6)


def test_recon_loss_l1_is_mean_absolute_error():
    torch.manual_seed(1)
    x_recon = torch.randn(3, 1, 8, 8)
    x = torch.randn(3, 1, 8, 8)
    loss = ReconLoss(kind="l1")(x_recon, x)
    assert torch.allclose(loss, (x_recon - x).abs().mean(), atol=1e-6)


def test_recon_loss_mean_reduction_is_resolution_independent():
    """A uniform per-pixel error gives the SAME loss at 8x8 and 16x16 --
    the whole reason mean reduction is used instead of the summed norm
    the docs write (a sum would scale with image size)."""
    small = ReconLoss()(torch.full((2, 1, 8, 8), 0.5), torch.zeros(2, 1, 8, 8))
    large = ReconLoss()(torch.full((2, 1, 16, 16), 0.5), torch.zeros(2, 1, 16, 16))
    assert torch.allclose(small, large, atol=1e-6)


def test_recon_loss_rejects_unknown_kind():
    with pytest.raises(ValueError, match="l1.*l2|kind"):
        ReconLoss(kind="huber")


# --- OneStepLoss (same shape as ReconLoss, but its own class) --------

def test_one_step_loss_l2_matches_mean_squared_error():
    torch.manual_seed(2)
    z_next_pred = torch.randn(4, 3, 8, 8)
    z_next_true = torch.randn(4, 3, 8, 8)
    loss = OneStepLoss()(z_next_pred, z_next_true)
    assert torch.allclose(loss, (z_next_pred - z_next_true).pow(2).mean(), atol=1e-6)


def test_one_step_loss_l1_matches_mean_absolute_error():
    torch.manual_seed(3)
    z_next_pred = torch.randn(4, 3, 8, 8)
    z_next_true = torch.randn(4, 3, 8, 8)
    loss = OneStepLoss(kind="l1")(z_next_pred, z_next_true)
    assert torch.allclose(loss, (z_next_pred - z_next_true).abs().mean(), atol=1e-6)


def test_one_step_loss_rejects_unknown_kind():
    with pytest.raises(ValueError, match="l1.*l2|kind"):
        OneStepLoss(kind="nope")


# --- InterpLoss ------------------------------------------------------

def test_interp_loss_is_zero_on_an_exactly_affine_trajectory():
    """The degenerate minimum the docstring warns about: for ANY z0
    that is affine in t, z2 == (1-alpha)*z1 + alpha*z3 exactly, so the
    loss is exactly 0. Deliberately unguarded -- this pins that no
    self-protection was silently added."""
    torch.manual_seed(4)
    z1 = torch.randn(3, 2, 4, 4)
    z3 = torch.randn(3, 2, 4, 4)
    alpha = torch.tensor([0.2, 0.5, 0.8])
    a = alpha.view(3, 1, 1, 1)
    z2 = (1.0 - a) * z1 + a * z3            # exactly on the straight line
    loss = InterpLoss()(z1, z2, z3, alpha)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_interp_loss_constant_z0_is_a_degenerate_minimum():
    """A CONSTANT z0 (z1==z2==z3) is affine in t too and must score 0
    for any alpha -- the specific collapse only L_recon0 prevents."""
    z = torch.randn(2, 2, 4, 4)
    z_seq = z  # same tensor for all three frames
    loss = InterpLoss()(z_seq, z_seq, z_seq, alpha=torch.tensor([0.37, 0.91]))
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_interp_loss_alpha_is_reshaped_per_sample_not_broadcast_against_width():
    """alpha is (B,) and must be applied PER SAMPLE. With B != W the
    naive trailing-dim broadcast would either error or silently align
    alpha against the width axis; this pins the per-sample reshape by
    comparing against an explicit per-sample loop. B=2, W=3 so the two
    axes cannot be confused."""
    torch.manual_seed(5)
    z1 = torch.randn(2, 1, 2, 3)          # B=2, W=3 -- deliberately different
    z2 = torch.randn(2, 1, 2, 3)
    z3 = torch.randn(2, 1, 2, 3)
    alpha = torch.tensor([0.25, 0.75])

    loss = InterpLoss()(z1, z2, z3, alpha)

    # Independent per-sample computation.
    blended = torch.stack([
        (1.0 - alpha[b]) * z1[b] + alpha[b] * z3[b] for b in range(2)
    ])
    expected = (blended - z2).pow(2).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_interp_loss_l1_kind_uses_absolute_error():
    torch.manual_seed(6)
    z1 = torch.randn(3, 2, 4, 4)
    z2 = torch.randn(3, 2, 4, 4)
    z3 = torch.randn(3, 2, 4, 4)
    alpha = torch.tensor([0.1, 0.5, 0.9])
    a = alpha.view(3, 1, 1, 1)

    loss = InterpLoss(kind="l1")(z1, z2, z3, alpha)
    expected = ((1.0 - a) * z1 + a * z3 - z2).abs().mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_interp_loss_rejects_unknown_kind():
    with pytest.raises(ValueError, match="l1.*l2|kind"):
        InterpLoss(kind="l3")


# --- centered_deriv_target -------------------------------------------

def test_centered_deriv_reduces_to_symmetric_difference_at_equal_spacing():
    """With dt_minus == dt_plus == h the non-uniform formula must reduce
    exactly to the familiar (z_after - z_before)/(2h)."""
    torch.manual_seed(7)
    z_before = torch.randn(2, 3, 4, 4)
    z_t = torch.randn(2, 3, 4, 4)
    z_after = torch.randn(2, 3, 4, 4)
    h = torch.tensor(0.5)

    got = centered_deriv_target(z_before, z_t, z_after, h, h)
    expected = (z_after - z_before) / (2.0 * h)
    assert torch.allclose(got, expected, atol=1e-6)


def test_centered_deriv_is_exact_for_a_quadratic_with_unequal_spacing():
    """A 3-point central difference is exact (2nd-order) for any
    quadratic, EVEN with unequal spacing -- the property that removes
    the O(dt) truncation bias of the one-sided target. Sample
    f(t)=a+b*t+c*t^2 at t-dm, t, t+dp with dm != dp and require the
    exact analytic derivative b+2*c*t back."""
    a, b, c = 1.3, -0.7, 2.1
    t = 2.0
    dm = torch.tensor(1.0)
    dp = torch.tensor(3.0)                 # deliberately unequal

    def f(tt):
        return a + b * tt + c * tt ** 2

    z_before = torch.tensor(f(t - dm.item()))
    z_t = torch.tensor(f(t))
    z_after = torch.tensor(f(t + dp.item()))

    got = centered_deriv_target(z_before, z_t, z_after, dm, dp)
    analytic = b + 2.0 * c * t
    assert got.item() == pytest.approx(analytic, abs=1e-5)


# --- dt_weighted_deriv_loss ------------------------------------------

def test_dt_weighted_deriv_exponent_zero_is_exactly_plain_recon_loss():
    """exponent=0.0 must return the ReconLoss instance's own output
    EXACTLY (torch.equal) -- the historical uniform-weight L_deriv,
    routed straight through the trusted ReconLoss, not a numerically
    close reconstruction."""
    torch.manual_seed(8)
    recon = ReconLoss()
    z1_pred = torch.randn(4, 3, 4, 4)
    target = torch.randn(4, 3, 4, 4)
    dt = torch.rand(4, 1, 1, 1) + 0.1

    got = dt_weighted_deriv_loss(recon, z1_pred, target, dt, exponent=0.0)
    assert torch.equal(got, recon(z1_pred, target))


def test_dt_weighted_deriv_exponent_one_is_mean1_renormalized_inverse_dt():
    """exponent=1.0: weight_i = (1/dt_i) / mean(1/dt), then a weighted
    mean of the squared per-element error -- verified against an
    independent computation."""
    torch.manual_seed(9)
    recon = ReconLoss()
    z1_pred = torch.randn(3, 2, 4, 4)
    target = torch.randn(3, 2, 4, 4)
    dt = (torch.rand(3, 1, 1, 1) + 0.1)

    got = dt_weighted_deriv_loss(recon, z1_pred, target, dt, exponent=1.0)

    weight = dt.pow(-1.0)
    weight = weight / weight.mean()
    per_element = (z1_pred - target) ** 2
    expected = (weight * per_element).mean()
    assert torch.allclose(got, expected, atol=1e-6)


def test_dt_weighted_deriv_l1_kind_uses_absolute_per_element_error():
    torch.manual_seed(10)
    recon = ReconLoss(kind="l1")
    z1_pred = torch.randn(3, 2, 4, 4)
    target = torch.randn(3, 2, 4, 4)
    dt = (torch.rand(3, 1, 1, 1) + 0.1)

    got = dt_weighted_deriv_loss(recon, z1_pred, target, dt, exponent=1.0)

    weight = dt.pow(-1.0)
    weight = weight / weight.mean()
    per_element = (z1_pred - target).abs()
    expected = (weight * per_element).mean()
    assert torch.allclose(got, expected, atol=1e-6)


# --- StatsLoss.forward / per_stat_mse --------------------------------

def test_stats_loss_forward_normalizes_the_target_per_stat():
    """forward() normalizes the (raw) target by (target-mean)/std before
    comparing to the (already-normalized) prediction, then means the
    squared difference. Verified independently, no angle column."""
    mean = torch.tensor([5.0, -3.0, 10.0])
    std = torch.tensor([2.0, 0.5, 4.0])
    sl = StatsLoss(mean, std, stat_names=["avg_phi", "energy", "stdev_phi"])  # no "angle"

    torch.manual_seed(11)
    pred = torch.randn(6, 3)               # already-normalized predictions
    target = torch.randn(6, 3) * 5.0 + 2.0  # raw-scale targets

    got = sl(pred, target)
    target_norm = (target - mean) / std
    expected = (pred - target_norm).pow(2).mean()
    assert torch.allclose(got, expected, atol=1e-6)


def test_stats_loss_wraps_only_the_angle_column_by_its_normalized_period():
    """angle is defined mod pi (physical orientation has no front): a
    prediction off from the normalized target by exactly one period
    (pi/std in normalized units) on the angle column must contribute
    ZERO, while a non-angle column offset by the same amount does not."""
    mean = torch.tensor([5.0, 0.0])
    std = torch.tensor([2.0, 0.5])
    sl = StatsLoss(mean, std, stat_names=["avg_phi", "angle"])
    angle_idx = 1
    period = torch.pi / std[angle_idx]

    target = torch.tensor([[7.0, 0.3]])          # raw
    target_norm = (target - mean) / std

    # Prediction exactly on target_norm except the angle column shifted
    # by a full normalized period -> wrapped angle diff is 0.
    pred = target_norm.clone()
    pred[0, angle_idx] = target_norm[0, angle_idx] + period
    assert sl(pred, target).item() == pytest.approx(0.0, abs=1e-5)

    # The same-sized shift on the NON-angle column is a real error.
    pred_nonangle = target_norm.clone()
    pred_nonangle[0, 0] = target_norm[0, 0] + period
    assert sl(pred_nonangle, target).item() > 1e-3


def test_stats_loss_with_no_angle_name_wraps_nothing():
    """When stat_names has no 'angle' (or is None), angle_idx is None
    and every column is treated as a plain normalized difference."""
    mean = torch.tensor([1.0, 2.0])
    std = torch.tensor([1.0, 1.0])
    sl = StatsLoss(mean, std, stat_names=None)
    assert sl.angle_idx is None

    pred = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[1.0, 2.0]])          # target_norm = 0 -> diff = pred
    assert sl(pred, target).item() == pytest.approx(0.0, abs=1e-6)


def test_stats_loss_per_stat_mse_returns_one_value_per_stat():
    """per_stat_mse means over the batch dim only, returning (n_stats,)
    -- and its mean over stats must equal the scalar forward()."""
    mean = torch.tensor([5.0, -3.0, 10.0])
    std = torch.tensor([2.0, 0.5, 4.0])
    sl = StatsLoss(mean, std, stat_names=["avg_phi", "energy", "stdev_phi"])

    torch.manual_seed(12)
    pred = torch.randn(8, 3)
    target = torch.randn(8, 3) * 3.0

    per_stat = sl.per_stat_mse(pred, target)
    assert per_stat.shape == (3,)
    assert torch.allclose(per_stat.mean(), sl(pred, target), atol=1e-6)

    target_norm = (target - mean) / std
    expected_per_stat = (pred - target_norm).pow(2).mean(dim=0)
    assert torch.allclose(per_stat, expected_per_stat, atol=1e-6)


# --- z0_scale_loss ---------------------------------------------------

def test_z0_scale_loss_is_mean_squared_latent_element():
    """mean over batch of ||z0||^2/(C*H*W) == the mean squared latent
    element over the whole tensor."""
    torch.manual_seed(13)
    z0 = torch.randn(4, 3, 4, 4)
    assert torch.allclose(z0_scale_loss(z0), z0.pow(2).mean(), atol=1e-6)


def test_z0_scale_loss_scale_is_independent_of_latent_channels():
    """A constant-amplitude latent gives the SAME anchor value at 3 and
    at 12 channels -- the mean-per-element normalization the docstring
    promises (so the term isn't silently rescaled by latent_channels)."""
    few = z0_scale_loss(torch.full((2, 3, 4, 4), 0.5))
    many = z0_scale_loss(torch.full((2, 12, 4, 4), 0.5))
    assert torch.allclose(few, many, atol=1e-6)
    assert few.item() == pytest.approx(0.25, abs=1e-6)
