"""
Tests for evaluation/check_f_theta.py.
"""

import numpy as np
import pytest
import torch

from evaluation.check_f_theta import _log_corr, compute_f_diagnostics
from models.latent_dynamics import LatentDynamics


def test_log_corr_returns_none_for_float32_round_tripped_constant_values():
    """Regression test for a real, reported bug: genuinely constant
    values (e.g. uniform dt spacing in a toy/small dataset) do NOT come
    back bit-exactly identical after round-tripping through float32 via
    torch -- std() lands near machine epsilon (~1e-7), not exactly 0.0.
    An exact std()==0 guard misses this entirely, and np.corrcoef then
    silently returns nan (with a RuntimeWarning) instead of a clean,
    honest "not enough variance to correlate" result. Reproduces the
    EXACT round-trip that broke this originally, not just a hand-picked
    zero array."""
    constant_dt = torch.tensor([50.0, 50.0, 50.0], dtype=torch.float32).numpy()
    varying_norms = np.array([1.2, 3.4, 5.6])

    assert _log_corr(constant_dt, varying_norms) is None


def test_log_corr_returns_none_when_y_is_the_constant_one():
    """Same guard, checking the OTHER argument -- both x and y are
    checked independently, not just the first one."""
    varying_dt = np.array([10.0, 100.0, 1000.0])
    constant_norms = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).numpy()

    assert _log_corr(varying_dt, constant_norms) is None


def test_log_corr_computes_a_real_correlation_when_both_vary():
    """Sanity check that the guard doesn't over-trigger -- genuine
    variance in both arrays should still produce a real, non-None
    correlation, computed correctly (verified here against an
    independent, direct np.corrcoef call, not just "isn't None")."""
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    y = np.array([2.0, 15.0, 90.0, 1200.0])

    result = _log_corr(x, y)
    assert result is not None
    expected = np.corrcoef(np.log10(x), np.log10(y))[0, 1]
    assert result == pytest.approx(expected)


def test_compute_f_diagnostics_rejects_window_length_below_3():
    """The diagnostic needs step 1 AND step 2's own real transition --
    window_length=2 (only one transition) can't support the
    real-vs-chained comparison at step 2 at all, and should fail loudly
    rather than silently computing something meaningless."""
    class _FakeDataset:
        window_length = 2

    f_theta = LatentDynamics(latent_channels=4, n_theta=1, latent_spatial=4,
                              hidden_dim=8, n_hidden_layers=1)
    with pytest.raises(ValueError, match="window_length"):
        compute_f_diagnostics(_FakeDataset(), f_theta, torch.device("cpu"))


def test_compute_f_diagnostics_chaining_matches_manual_computation():
    """The core computation, verified against an independent, manual
    walk of the same math -- not just trusted from reading the
    function's own source. Specifically checks: z0_hat_1 is genuinely
    f_theta's own step-1 PREDICTION (not accidentally the real z0(t1)),
    f2_chained uses THAT prediction while f2_real uses the real z0(t1)
    at the identical dt2/theta, and z0_step1_error is the actual gap
    between them."""
    torch.manual_seed(0)
    f_theta = LatentDynamics(latent_channels=4, n_theta=1, latent_spatial=4,
                              hidden_dim=8, n_hidden_layers=1)
    with torch.no_grad():
        torch.manual_seed(2)
        f_theta.net[-1].weight.normal_(std=0.05)  # break zero-init with a real, input-dependent map
        f_theta.net[-1].bias.fill_(0.03)

    torch.manual_seed(1)
    B = 3
    window0 = torch.randn(B, 3, 4, 4, 4)  # (B, window_length=3, C, 8->4 spatial for a small test, 4, 4)
    window1 = torch.randn(B, 3, 4, 4, 4)
    dt_window = torch.tensor([[5.0, 8.0], [3.0, 12.0], [7.0, 2.0]])
    theta = torch.randn(B, 1)

    class _FakeDataset(torch.utils.data.Dataset):
        window_length = 3

        def __len__(self):
            return B

        def __getitem__(self, idx):
            return window0[idx], window1[idx], dt_window[idx], theta[idx]

    d = compute_f_diagnostics(_FakeDataset(), f_theta, torch.device("cpu"))

    # Manual, independent walk of the same math:
    z0_0, z0_1 = window0[:, 0], window0[:, 1]
    z1_0, z1_1 = window1[:, 0], window1[:, 1]
    dt1, dt2 = dt_window[:, 0], dt_window[:, 1]

    with torch.no_grad():
        f1_real_expected = f_theta.f(z0_0, z1_0, theta)
        z0_hat_1_expected = f_theta(z0_0, z1_0, dt1, theta)
        f2_chained_expected = f_theta.f(z0_hat_1_expected, z1_1, theta)
        f2_real_expected = f_theta.f(z0_1, z1_1, theta)

    assert np.allclose(d["f1_real_norm"], f1_real_expected.flatten(start_dim=1).norm(dim=1).numpy(), atol=1e-5)
    assert np.allclose(d["f2_chained_norm"], f2_chained_expected.flatten(start_dim=1).norm(dim=1).numpy(), atol=1e-5)
    assert np.allclose(d["f2_real_norm"], f2_real_expected.flatten(start_dim=1).norm(dim=1).numpy(), atol=1e-5)

    expected_step1_error = (z0_hat_1_expected - z0_1).flatten(start_dim=1).norm(dim=1).numpy()
    assert np.allclose(d["z0_step1_error"], expected_step1_error, atol=1e-5)

    # f2_chained and f2_real must genuinely DIFFER here (nonzero f_head
    # bias means z0_hat_1 != z0_1) -- if they matched, the diagnostic
    # would be silently comparing the same thing to itself.
    assert not np.allclose(d["f2_chained_norm"], d["f2_real_norm"])

    assert np.allclose(d["dt1"], dt1.numpy())
    assert np.allclose(d["dt2"], dt2.numpy())


def test_ideal_target_matches_manual_computation():
    """The over/under-shooting diagnostic's own core formula, verified
    against an independent, manual walk -- not just trusted from
    reading the function's own source."""
    torch.manual_seed(3)
    f_theta = LatentDynamics(latent_channels=4, n_theta=1, latent_spatial=4,
                              hidden_dim=8, n_hidden_layers=1)
    with torch.no_grad():
        torch.manual_seed(4)
        f_theta.net[-1].weight.normal_(std=0.05)
        f_theta.net[-1].bias.fill_(0.02)

    torch.manual_seed(5)
    B = 2
    window0 = torch.randn(B, 3, 4, 4, 4)
    window1 = torch.randn(B, 3, 4, 4, 4)
    dt_window = torch.tensor([[4.0, 6.0], [9.0, 3.0]])
    theta = torch.randn(B, 1)

    class _FakeDataset(torch.utils.data.Dataset):
        window_length = 3

        def __len__(self):
            return B

        def __getitem__(self, idx):
            return window0[idx], window1[idx], dt_window[idx], theta[idx]

    d = compute_f_diagnostics(_FakeDataset(), f_theta, torch.device("cpu"))

    z0_0, z0_1, z0_2 = window0[:, 0], window0[:, 1], window0[:, 2]
    z1_0, z1_1 = window1[:, 0], window1[:, 1]
    dt1, dt2 = dt_window[:, 0], dt_window[:, 1]
    dt2_r = dt2.view(-1, 1, 1, 1)

    with torch.no_grad():
        z0_hat_1 = f_theta(z0_0, z1_0, dt1, theta)
        f2_chained = f_theta.f(z0_hat_1, z1_1, theta)
        f2_real = f_theta.f(z0_1, z1_1, theta)
        f2_chained_ideal_expected = (z0_2 - z0_hat_1 - z1_1 * dt2_r) / (dt2_r ** 2 / 2)
        f2_real_ideal_expected = (z0_2 - z0_1 - z1_1 * dt2_r) / (dt2_r ** 2 / 2)

        expected_ratio_chained = (f2_chained.flatten(start_dim=1).norm(dim=1)
                                   / f2_chained_ideal_expected.flatten(start_dim=1).norm(dim=1))
        expected_ratio_real = (f2_real.flatten(start_dim=1).norm(dim=1)
                                / f2_real_ideal_expected.flatten(start_dim=1).norm(dim=1))

    assert np.allclose(d["ratio_chained"], expected_ratio_chained.numpy(), atol=1e-4)
    assert np.allclose(d["ratio_real"], expected_ratio_real.numpy(), atol=1e-4)


def test_ideal_target_formula_genuinely_inverts_forward():
    """Constructive check that the ideal-target formula is a correct
    inverse of forward()'s own update rule -- not just internally
    self-consistent with a copy of the same formula (which the test
    above alone wouldn't catch, since it derives its own expected
    values via the identical expression). Builds a scenario where
    z0(t2) is chosen to be EXACTLY forward()'s own prediction for a
    KNOWN, chosen f value -- so the true ideal target is that exact
    known f value by construction, verified against ratio=1.0 and
    cos_sim=1.0 exactly (not approximately close to f_theta's own,
    generally-different output)."""
    f_theta = LatentDynamics(latent_channels=4, n_theta=1, latent_spatial=4,
                              hidden_dim=8, n_hidden_layers=1)
    known_f_value = 0.7
    with torch.no_grad():
        # Zero weight (input-independent) + constant bias -- f_theta.f()
        # now genuinely, always outputs known_f_value, matching what
        # z0(t2) gets constructed from below. Without this, f_theta
        # stays at its own default zero-init and actually outputs 0,
        # not known_f_value -- the ideal target would then correctly
        # NOT match f_theta's real output, defeating this test's own
        # purpose (checking the FORMULA's correctness in isolation,
        # not whether it happens to agree with an untouched model).
        f_theta.net[-1].weight.zero_()
        f_theta.net[-1].bias.fill_(known_f_value)
    torch.manual_seed(6)
    B = 2
    z0_0 = torch.randn(B, 4, 4, 4)
    z1_0 = torch.randn(B, 4, 4, 4)
    z1_1 = torch.randn(B, 4, 4, 4)
    dt1 = torch.tensor([5.0, 5.0])
    dt2 = torch.tensor([3.0, 3.0])
    theta = torch.randn(B, 1)

    with torch.no_grad():
        z0_hat_1 = f_theta(z0_0, z1_0, dt1, theta)
        # Construct z0(t2) to be EXACTLY z0_hat_1 + z1_1*dt2 + KNOWN_F*(dt2^2/2)
        # -- forward()'s own formula, using f_theta's own (now constant,
        # known) output value.
        known_f = torch.full((B, 4, 4, 4), known_f_value)
        dt2_r = dt2.view(-1, 1, 1, 1)
        z0_2 = z0_hat_1 + z1_1 * dt2_r + known_f * (dt2_r ** 2 / 2)

    window0 = torch.stack([z0_0, torch.zeros_like(z0_0), z0_2], dim=1)  # index 1 (real z0_1) unused by f2_chained_ideal
    window1 = torch.stack([z1_0, z1_1, torch.zeros_like(z1_1)], dim=1)
    dt_window = torch.stack([dt1, dt2], dim=1)

    class _FakeDataset(torch.utils.data.Dataset):
        window_length = 3

        def __len__(self):
            return B

        def __getitem__(self, idx):
            return window0[idx], window1[idx], dt_window[idx], theta[idx]

    d = compute_f_diagnostics(_FakeDataset(), f_theta, torch.device("cpu"))

    assert np.allclose(d["ratio_chained"], 1.0, atol=1e-4), (
        "ideal target should exactly recover the known f value used to construct z0(t2) -- "
        "ratio should be exactly 1.0, not approximately close to f_theta's own output"
    )
    assert np.allclose(d["cos_sim_chained"], 1.0, atol=1e-4)


def _make_dataset_for_dead_relu_check(n_samples=32):
    torch.manual_seed(7)
    window0 = torch.randn(n_samples, 3, 4, 4, 4)
    window1 = torch.randn(n_samples, 3, 4, 4, 4)
    dt_window = torch.rand(n_samples, 2) * 10 + 1
    theta = torch.randn(n_samples, 1)

    class _FakeDataset(torch.utils.data.Dataset):
        window_length = 3

        def __len__(self):
            return n_samples

        def __getitem__(self, idx):
            return window0[idx], window1[idx], dt_window[idx], theta[idx]

    return _FakeDataset()


def test_check_dead_relus_reports_near_zero_on_a_healthy_freshly_initialized_network():
    """A freshly-initialized network (default PyTorch init on the trunk,
    only the final layer zero-init per LatentDynamics' own design) has
    no reason for its ReLU units to be systematically dead -- confirms
    the diagnostic doesn't over-report on ordinary, healthy networks."""
    from evaluation.check_f_theta import check_dead_relus

    torch.manual_seed(8)
    f_theta = LatentDynamics(latent_channels=4, n_theta=1, latent_spatial=4,
                              hidden_dim=32, n_hidden_layers=2)
    dataset = _make_dataset_for_dead_relu_check()

    result = check_dead_relus(f_theta, dataset, torch.device("cpu"))

    assert "trunk_output" in result
    assert result["trunk_output"] < 0.5, (
        f"a freshly-initialized network should not have a majority-dead trunk output, "
        f"got {result['trunk_output']:.1%}"
    )


def test_check_dead_relus_detects_a_genuinely_collapsed_trunk():
    """Regression test for the exact failure mode this diagnostic exists
    to catch: reproduces a real collapse directly (not just trusting
    the diagnostic's own logic) by manually driving every HIDDEN
    layer's weight/bias so its pre-activation is permanently negative
    for any realistic input -- then confirms check_dead_relus correctly
    reports trunk_output near 1.0, AND confirms this actually produces
    the exact symptom that motivated building this diagnostic:
    f_theta.f(...) returning an (almost) constant output regardless of
    z0/z1/theta. Only the HIDDEN layers are collapsed here, deliberately
    leaving the FINAL (output-producing) layer at its normal, random
    init -- zeroing that layer too would trivially force a constant
    output by itself, regardless of whether the trunk is actually
    stuck, which wouldn't genuinely test what this test claims to."""
    from evaluation.check_f_theta import check_dead_relus, compute_f_diagnostics

    f_theta = LatentDynamics(latent_channels=4, n_theta=1, latent_spatial=4,
                              hidden_dim=16, n_hidden_layers=2)
    with torch.no_grad():
        hidden_linears = [layer for layer in f_theta.net if isinstance(layer, torch.nn.Linear)][:-1]
        for layer in hidden_linears:
            # Weight=0 (input can't push pre-activation positive regardless of
            # magnitude) + a large negative bias -- every unit's pre-activation
            # is guaranteed negative for ANY input, exactly what a real
            # catastrophic gradient spike could in principle drive weights
            # toward (this test doesn't claim training actually does this --
            # only that IF it happened, the diagnostic correctly detects it).
            layer.weight.zero_()
            layer.bias.fill_(-100.0)

    dataset = _make_dataset_for_dead_relu_check()
    result = check_dead_relus(f_theta, dataset, torch.device("cpu"))

    assert result["trunk_output"] > 0.99, (
        f"expected a (near-)fully stuck trunk output, got {result['trunk_output']:.1%}"
    )

    # Confirm this ACTUALLY produces the symptom that motivated building
    # this diagnostic in the first place -- f() output collapsing to an
    # (almost) constant value regardless of input. "Almost", not exactly
    # constant, since this trunk uses LeakyReLU (not plain ReLU): a
    # unit stuck on the negative side still outputs
    # negative_slope*pre_activation, a small but genuinely nonzero,
    # input-dependent value -- not identically zero the way a truly
    # dead ReLU unit would be. The variance should still collapse
    # dramatically relative to a healthy network, just not to bit-exact
    # zero.
    d = compute_f_diagnostics(dataset, f_theta, torch.device("cpu"))
    f1_real_norm = d["f1_real_norm"]
    assert f1_real_norm.std() < 0.05 * max(f1_real_norm.mean(), 1e-8), (
        "a fully stuck trunk should produce a near-constant (low-variance-relative-to-mean) "
        "f() output across genuinely different z0/z1/theta inputs"
    )


def test_print_binned_by_dt2_uses_correct_decade_boundaries_and_medians(capsys):
    """Verifies both the bin assignment (values land in the decade
    their own dt2 actually falls into, not an off-by-one neighbor) and
    that MEDIAN (not mean) is what's reported per bin -- deliberately
    robust to a heavy-tailed outlier within a bin, which is the whole
    point of this breakdown (see its own docstring)."""
    from evaluation.check_f_theta import _print_binned_by_dt2

    # decade 1 (10-100): 3 values, median should be 20 (not pulled by the range)
    # decade 3 (1000-10000): 3 values, one deliberately huge outlier --
    # median should stay at the middle value (8.0), NOT get dragged
    # toward the outlier the way a MEAN of the same 3 values would be
    dt2 = np.array([15.0, 20.0, 90.0, 1500.0, 2000.0, 3000.0])
    values = np.array([10.0, 20.0, 30.0, 5.0, 8.0, 1_000_000.0])

    _print_binned_by_dt2("test_metric", dt2, values)
    output = capsys.readouterr().out

    assert "1e1-1e2" in output
    assert "n=    3" in output
    assert "median=    2.0000e+01" in output  # median of [10, 20, 30] == 20

    assert "1e3-1e4" in output
    assert "n=    3" in output
    assert "median=    8.0000e+00" in output  # median of [5, 8, 1_000_000] == 8, not dragged toward the outlier
