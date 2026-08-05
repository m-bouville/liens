"""
Tests for evaluation/check_f_theta.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import cached_sweep

from evaluation.check_f_theta import _log_corr, check_f_theta, compute_f_diagnostics
from models.autoencoder import MultiStreamAutoencoder
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_dynamics import LatentDynamics
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from utils import load_datasets as load


SIZE = 32
LATENT_CHANNELS = 4
STEPS = [0, 1000, 2000, 3000, 4000]


def _build_run_dir_uncached(tmp_path, name="T800_n010_s0", temperature=0.8, seed=0):
    run_dir = tmp_path / name
    run_dir.mkdir()
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {SIZE}", f"Ny = {SIZE}", "dt = 0.05", "steps = 4000",
        f"save_steps = {' '.join(str(s) for s in STEPS)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", f"temperature = {temperature}",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        f"seed = {seed}", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    for step in STEPS:
        arr = np.full((SIZE, SIZE), step / 10000.0, dtype="<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))
    pd.DataFrame([{"step": s, "avg_phi": s / 1000.0} for s in STEPS]).to_csv(
        run_dir / "statistics.csv", index=False)
    (run_dir / "COMPLETE").touch()
    return run_dir


def _build_run_dir(tmp_path, *args, **kwargs):
    return cached_sweep(
        (__name__, args, tuple(sorted(kwargs.items()))),
        lambda d: _build_run_dir_uncached(d, *args, **kwargs),
    )


def _build_ae_checkpoint(path: Path):
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=LATENT_CHANNELS, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=LATENT_CHANNELS, spatial_size=8,
                                     mode=LatentStreamMode.DECODER, condition_on_theta=True),
    }
    encoder = Encoder(input_size=SIZE, in_channels=1, base_channels=4, stream_configs=stream_configs)
    decoder = Decoder(output_size=SIZE, out_channels=1, base_channels=4,
                       latent_channels=LATENT_CHANNELS, latent_spatial_size=8)
    ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                 stream_configs=stream_configs)
    checkpoint = {
        "model_state": ae.state_dict(), "epoch": 1, "val_loss": 0.01,
        "val_loss_ema": 0.01, "test_dirs": [],
        "config": {"size": SIZE, "base_channels": 4, "latent_channels": LATENT_CHANNELS,
                   "latent_spatial_size": 8, "stats_weight": 0.0,
                   "stream_configs": {
                       "state": {"channels": LATENT_CHANNELS, "spatial_size": 8, "mode": "autoencoder"},
                       "deriv": {"channels": LATENT_CHANNELS, "spatial_size": 8, "mode": "decoder",
                                 "condition_on_theta": True},
                   },
                   "recon_stream_name": "state"},
    }
    torch.save(checkpoint, path)


def _build_lds_checkpoint(path: Path, ae_checkpoint_path: Path, run_dirs, dt_cap: float = float("inf")):
    f_theta = LatentDynamics(latent_channels=LATENT_CHANNELS, n_theta=1, hidden_dim=8, n_hidden_layers=1,
                              dt_cap=dt_cap)
    checkpoint = {
        "model_state": f_theta.state_dict(), "epoch": 1, "val_loss": 0.05,
        "val_loss_ema": 0.05, "ae_checkpoint": str(ae_checkpoint_path),
        "test_dirs": [str(d) for d in run_dirs],
        "config": {"latent_channels": LATENT_CHANNELS, "n_theta": 1, "hidden_dim": 8,
                   "n_hidden_layers": 1, "dt_cap": dt_cap},
        "data_config": {"min_step": 0, "min_stdev_phi": None, "window_length": 2},
    }
    torch.save(checkpoint, path)


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


def _fake_run_dir(tmp_path):
    """Minimal run dir: validate_run_dirs only requires metadata.txt to exist."""
    d = tmp_path / "fake_run_dir"
    d.mkdir(exist_ok=True)
    (d / "metadata.txt").write_text("Nx = 32\nNy = 32\n")
    return d


def test_check_f_theta_threads_dt_cap_from_the_saved_checkpoint(monkeypatch, tmp_path):
    """
    REGRESSION: check_f_theta()'s own LatentDynamics reconstruction is
    SEPARATE from model_assembly.py's own build_models_from_components
    and from evaluation._latent_eval.py's own copy, both of which
    already had this fix -- fixing dt_cap in either of THOSE did not
    fix it here. A checkpoint saved with a real, finite dt_cap used to
    silently evaluate as if dt_cap were still inf.

    Intercepts LatentDynamics.__init__ to capture the dt_cap kwarg
    actually passed, then lets the rest of check_f_theta() fail past
    that point -- no real dataset/AE fixture needed, the capture
    already happened before the failure. ensure_lds_checkpoint is
    monkeypatched at ITS OWN source module (orchestration.
    checkpoint_identification), not on check_f_theta's own namespace --
    it's imported LOCALLY inside check_f_theta(), so patching it there
    wouldn't be picked up.
    """
    import orchestration.checkpoint_identification as ci

    captured = {}
    real_init = LatentDynamics.__init__

    def capturing_init(self, *args, **kwargs):
        captured["dt_cap"] = kwargs.get("dt_cap")
        real_init(self, *args, **kwargs)
        raise RuntimeError("stop here -- dt_cap already captured")

    monkeypatch.setattr(LatentDynamics, "__init__", capturing_init)
    monkeypatch.setattr(ci, "ensure_lds_checkpoint", lambda path, **kw: path)

    import evaluation._latent_eval as latent_eval
    from types import SimpleNamespace
    fake_ae_checkpoint = {"config": {"size": 32}}
    fake_ae = SimpleNamespace(decoder=object())  # ae_decoder is read BEFORE LatentDynamics
    monkeypatch.setattr(
        latent_eval, "build_ae_from_checkpoint",
        lambda path, device: (fake_ae, None, fake_ae_checkpoint, {}, "state"),
    )

    checkpoint_path = tmp_path / "fake-stage3.pt"
    torch.save({
        # A run dir that EXISTS with a metadata.txt: check_f_theta now
        # validates the checkpoint's stored test_dirs before building
        # anything, so a purely fictional path is rejected before
        # LatentDynamics is ever constructed. The stub relied on that
        # validation not existing.
        "epoch": 1, "val_loss": 0.05, "test_dirs": [str(_fake_run_dir(tmp_path))],
        "ae_checkpoint": "does-not-need-to-exist.pt",
        "config": {"latent_channels": 4, "n_theta": 1, "hidden_dim": 8,
                   "n_hidden_layers": 1, "dt_cap": 125.0},
        "data_config": {"min_step": 0, "min_stdev_phi": None, "window_length": 3},
    }, checkpoint_path)

    from evaluation.check_f_theta import check_f_theta
    try:
        check_f_theta(checkpoint_path, device="cpu", output_path=tmp_path / "out.png")
    except Exception:
        pass  # expected -- fails past the point this test cares about

    assert "dt_cap" in captured, "LatentDynamics was never constructed at all"
    assert captured["dt_cap"] == 125.0


def test_check_f_theta_runs_end_to_end(tmp_path, isolated_project_root):
    """
    Real, on-disk fixture -- an actual encoder/decoder/f_theta and a
    genuine run directory -- exercising check_f_theta() through its
    ACTUAL call path (not a mocked build_ae_from_checkpoint/
    ensure_lds_checkpoint), specifically because this function's own
    setup phase was just rewritten to share
    evaluation._latent_eval._load_ae_f_theta_and_dataset rather than
    duplicating it. The dt_cap-focused test above verifies the
    parameter threads through correctly via mocks; this one verifies
    the real, unmocked integration still produces a saved figure.
    """
    run_dir = _build_run_dir(tmp_path)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    _build_ae_checkpoint(ae_checkpoint_path)
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_lds_checkpoint(lds_checkpoint_path, ae_checkpoint_path, [run_dir])

    output_path = tmp_path / "f_theta_diag.png"
    result_path = check_f_theta(
        lds_checkpoint_path=lds_checkpoint_path, min_step=0, device="cpu",
        output_path=output_path,
    )
    assert result_path == output_path
    assert output_path.exists()


def test_check_f_theta_default_output_path_uses_the_real_stage_folder(
    tmp_path, isolated_project_root,
):
    """
    REGRESSION: check_f_theta()'s own default output_path used to be
    hardcoded to output/stage3/ regardless of whether the checkpoint
    was actually stage3a or stage3b -- the exact same bug already fixed
    once in evaluation._latent_eval.py's own
    _stage_folder_from_checkpoint_stem, now shared here too rather than
    recurring a third time. A stage-3a and a stage-3b checkpoint's own
    figures must land in DIFFERENT folders, not silently overwrite each
    other in a shared output/stage3/.
    """
    run_dir = _build_run_dir(tmp_path)
    ae_checkpoint_path = tmp_path / "fake-stage2-b.pt"
    _build_ae_checkpoint(ae_checkpoint_path)
    lds_checkpoint_path = tmp_path / "64x64-stage3b.pt"
    _build_lds_checkpoint(lds_checkpoint_path, ae_checkpoint_path, [run_dir])

    result_path = check_f_theta(
        lds_checkpoint_path=lds_checkpoint_path, min_step=0, device="cpu",
    )
    assert result_path.parent.name == "stage3b", (
        f"expected output under a 'stage3b' folder, got '{result_path.parent.name}' -- "
        f"the stage-folder-from-stem logic isn't being used"
    )


def test_untrained_f_theta_does_not_produce_broken_log_scaled_panels(recwarn, tmp_path,
                                                                     isolated_project_root):
    """
    REGRESSION: LatentDynamics zero-initializes its own final layer, so
    f_theta is EXACTLY zero everywhere until trained -- which is
    precisely what ensure_lds_checkpoint produces when this script is
    pointed at an AE-family (stage-1/1b/2) checkpoint, a mode
    check_f_theta()'s own docstring documents as supported. Every
    ||f||-derived quantity is then identically 0, and three panels used
    to be set to a log scale regardless, rendering unusable with only a
    matplotlib UserWarning to indicate it.

    Asserts NO "cannot be log-scaled" warning is emitted -- i.e. the
    panels genuinely fall back to linear rather than silently breaking.
    """
    run_dir = _build_run_dir(tmp_path)
    ae_checkpoint_path = tmp_path / "fake-stage2-untrained.pt"
    _build_ae_checkpoint(ae_checkpoint_path)
    lds_checkpoint_path = tmp_path / "fake-stage3-untrained.pt"
    _build_lds_checkpoint(lds_checkpoint_path, ae_checkpoint_path, [run_dir])

    # Confirm the premise: this checkpoint's own f_theta really is
    # all-zero, so the all-zero code path is genuinely being exercised.
    f_theta = LatentDynamics(latent_channels=LATENT_CHANNELS, n_theta=1,
                              hidden_dim=8, n_hidden_layers=1)
    assert f_theta.f(torch.randn(2, LATENT_CHANNELS, 8, 8),
                     torch.randn(2, LATENT_CHANNELS, 8, 8),
                     torch.randn(2, 1)).abs().max().item() == 0.0

    check_f_theta(
        lds_checkpoint_path=lds_checkpoint_path, min_step=0, device="cpu",
        output_path=tmp_path / "untrained.png",
    )

    log_warnings = [w for w in recwarn if "cannot be log-scaled" in str(w.message)]
    assert not log_warnings, (
        f"{len(log_warnings)} panel(s) were log-scaled against all-zero data and render "
        f"broken -- _log_scale_if_positive should have fallen back to linear"
    )
