"""
Tests for evaluation/check_rollout.py.

_padded_bounds and parse_fixed_window are pure Python/numpy -- no torch
needed -- so these actually run here and are checked directly, same as
every other test in this suite. compute_sample needs real models and is
included for completeness (the chaining logic is exactly what was fixed
a few turns back, replacing a bug where only steps[0]->steps[1] was ever
tested regardless of n_rollout_steps), but is NOT executed in this
sandbox (no torch available) -- traced carefully by hand instead, same
honest limitation as every torch-dependent test in this project.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_check_rollout.py -v
"""
import numpy as np
import pytest

from evaluation.check_rollout import _padded_bounds, parse_fixed_window


def test_padded_bounds_asymmetric_case():
    """Deliberately asymmetric, not +-max(abs(...)) -- a symmetric
    scale would waste half the color range on a side the data barely
    uses."""
    vals = np.array([-0.05, 0.1, 0.3, -0.02])
    vmin, vmax = _padded_bounds(vals, factor=1.2)
    assert vmin == pytest.approx(-0.06)
    assert vmax == pytest.approx(0.36)


def test_padded_bounds_all_positive_case():
    """vmin should be a tiny negative epsilon (not a scaled positive
    number), so zero-centered diverging colormaps stay meaningful even
    for one-sided data."""
    vals = np.array([0.1, 0.2, 0.5])
    vmin, vmax = _padded_bounds(vals, factor=1.2)
    assert vmin < 0
    assert vmax == pytest.approx(0.6)


def test_padded_bounds_all_negative_case():
    vals = np.array([-0.1, -0.2, -0.5])
    vmin, vmax = _padded_bounds(vals, factor=1.2)
    assert vmax > 0
    assert vmin == pytest.approx(-0.6)


def test_padded_bounds_smaller_factor_for_error_scale():
    """The error panel uses factor=0.25 against the SAME real_delta
    values as the main delta panels -- both derived from one fixed
    reference, not each auto-scaled independently."""
    vals = np.array([-0.05, 0.1, 0.3, -0.02])
    vmin, vmax = _padded_bounds(vals, factor=0.25)
    assert vmin == pytest.approx(-0.0125)
    assert vmax == pytest.approx(0.075)


def test_padded_bounds_never_degenerate():
    """Even exactly-zero or single-sign-at-the-boundary data must not
    produce a zero-width or invalid range (this guards TwoSlopeNorm's
    vmin < vcenter < vmax requirement downstream)."""
    vals = np.array([0.0, 0.0, 0.0])
    vmin, vmax = _padded_bounds(vals, factor=1.2)
    assert vmin < 0 < vmax


def test_parse_fixed_window_two_steps():
    from pathlib import Path
    run_dir, steps = parse_fixed_window("../../datasets/64x64/T800_n050_s79:100000:120000")
    assert run_dir == Path("../../datasets/64x64/T800_n050_s79")
    assert steps == [100000, 120000]


def test_parse_fixed_window_full_chain():
    """The format this was extended to support -- a full multi-step
    window (e.g. 4 steps for a checkpoint trained at n_rollout_steps=3),
    not just two steps."""
    from pathlib import Path
    run_dir, steps = parse_fixed_window(
        "../../datasets/64x64/T800_n050_s79:100000:110000:120000:130000"
    )
    assert run_dir == Path("../../datasets/64x64/T800_n050_s79")
    assert steps == [100000, 110000, 120000, 130000]


def test_parse_fixed_window_rejects_too_few_parts():
    with pytest.raises(ValueError, match="run_dir:step0:step1"):
        parse_fixed_window("only_a_run_dir")


def test_parse_fixed_window_rejects_single_step():
    """A run_dir plus exactly one step isn't a valid window -- need at
    least a start and an end."""
    with pytest.raises(ValueError):
        parse_fixed_window("some/run_dir:100000")


def test_compute_sample_chains_through_full_window(tmp_run_dir):
    """
    THE core property the earlier bug fix exists for: compute_sample
    must chain f_theta.rollout() across the FULL window (steps[0] ->
    steps[-1]), not just compare steps[0]->steps[1] regardless of how
    many steps are given -- that was the exact bug (always testing
    1-step quality even for a 3-step window).

    Verified two ways: (1) the returned x_t_raw/x_next_raw/dt_val match
    the fixture's own known, deterministic values exactly; (2) the
    predicted result is cross-checked against an INDEPENDENT manual
    call to f_theta.rollout() with the same per-transition dts, using
    the real models -- not just trusted from reading the source.

    NOT EXECUTED in this sandbox (no torch available) -- traced by hand
    against the actual fixture values and the real Autoencoder/
    LatentDynamics/rollout() implementations instead. Should be run for
    real via `pytest tests/test_check_rollout.py -v` to confirm.
    """
    import torch
    from models.autoencoder import Autoencoder
    from models.latent_dynamics import LatentDynamics
    from evaluation.check_rollout import compute_sample

    run_dir, steps = tmp_run_dir  # steps = [0, 1000, 2000, 3000, 4000], dt=0.05, size=64

    ae = Autoencoder(size=64, channels=1, base_channels=4, latent_channels=4)
    ae.eval()
    f_theta = LatentDynamics(latent_channels=4, n_theta=1, hidden_dim=16, n_hidden_layers=1)
    f_theta.eval()
    ae_config = {"size": 64}
    device = torch.device("cpu")

    window = steps[:4]  # [0, 1000, 2000, 3000] -- a 3-step window, matching n_rollout_steps=3

    x_t_raw, x_next_raw, x_next_pred, x_next_ae_baseline, dt_val = compute_sample(
        run_dir, window, ae, f_theta, ae_config, device,
    )

    # (1) known, deterministic fixture values: constant field = step/10000
    assert np.allclose(x_t_raw, 0 / 10000.0, atol=1e-3)       # step 0
    assert np.allclose(x_next_raw, 3000 / 10000.0, atol=1e-3)  # step 3000 (the FINAL step, not step 1000)
    assert dt_val == pytest.approx((3000 - 0) * 0.05)  # metadata.dt=0.05 in the fixture

    # (2) independent cross-check: manually chain the SAME real models
    # with the same per-transition dts, and confirm compute_sample's
    # prediction matches exactly -- proves it's genuinely chaining
    # through rollout(), not silently doing a single big-dt call (the
    # old, buggy behavior) or stopping after one step.
    with torch.no_grad():
        x_t = torch.from_numpy(x_t_raw).unsqueeze(0).unsqueeze(0)
        z_t = ae.encoder(x_t)
        dts = torch.tensor([[1000 * 0.05, 1000 * 0.05, 1000 * 0.05]], dtype=torch.float32)
        theta = torch.tensor([[0.8 - 1.0]], dtype=torch.float32)  # temperature - T0, from the fixture
        z_hat_full = f_theta.rollout(z_t, dts, theta)
        expected_pred = ae.decoder(z_hat_full[:, -1])[0, 0].numpy()

    assert np.allclose(x_next_pred, expected_pred, atol=1e-5), (
        "compute_sample's prediction doesn't match an independently chained "
        "rollout() call with the same models and dts -- the chaining fix may "
        "have regressed."
    )
