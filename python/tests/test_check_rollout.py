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

from utils import load_datasets as load

from evaluation.check_rollout import _padded_bounds, _error_bounds, parse_fixed_window, _correlation_pct, _format_small
from models.latent_streams import DEFAULT_STREAM_NAME


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


def test_padded_bounds_symmetric_mixed_sign():
    """symmetric=True uses +-max(|lo|, |hi|) * factor, unlike the
    default asymmetric mode -- same vals as the asymmetric test above,
    but here both bounds are set by the larger-magnitude side (0.3)."""
    vals = np.array([-0.05, 0.1, 0.3, -0.02])
    vmin, vmax = _padded_bounds(vals, factor=1.2, symmetric=True)
    assert vmin == pytest.approx(-0.36)
    assert vmax == pytest.approx(0.36)


def test_padded_bounds_symmetric_all_positive_case():
    """THE case this was built for: real Delta x that's entirely
    one-signed (e.g. a purely shrinking domain). The default asymmetric
    mode would collapse vmin to ~0, so any negative PREDICTED value
    saturates to the same extreme color regardless of magnitude.
    symmetric=True instead mirrors the positive side, keeping negative
    predictions resolvable."""
    vals = np.array([0.1, 0.2, 0.5])
    vmin, vmax = _padded_bounds(vals, factor=1.2, symmetric=True)
    assert vmin == pytest.approx(-0.6)
    assert vmax == pytest.approx(0.6)


def test_padded_bounds_symmetric_all_negative_case():
    vals = np.array([-0.1, -0.2, -0.5])
    vmin, vmax = _padded_bounds(vals, factor=1.2, symmetric=True)
    assert vmin == pytest.approx(-0.6)
    assert vmax == pytest.approx(0.6)


def test_padded_bounds_symmetric_never_degenerate():
    vals = np.array([0.0, 0.0, 0.0])
    vmin, vmax = _padded_bounds(vals, factor=1.2, symmetric=True)
    assert vmin < 0 < vmax


def test_format_small_matches_example_from_the_conversation():
    """The exact case that motivated this: '.4f' rendered AE=0.0003 as
    hard-to-scan '0.0003' -- fixed-exponent notation makes it '0.3e-3'
    instead."""
    assert _format_small(0.0003) == "0.3e-3"


def test_format_small_zero():
    assert _format_small(0.0) == "0.0e-3"


def test_format_small_rounds_to_one_decimal():
    assert _format_small(0.00124) == "1.2e-3"


def test_correlation_pct_perfect_positive():
    """Same shape, different (positive) scale -- Pearson correlation is
    scale-invariant, which is exactly why it's a useful COMPLEMENT to
    the (scale-sensitive) loss: a 'right direction but weak' prediction
    scores high here even though its loss is middling."""
    predicted = np.array([[2.0, 4.0], [6.0, 8.0]])
    real = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert _correlation_pct(predicted, real) == pytest.approx(100.0)


def test_correlation_pct_perfect_negative():
    predicted = np.array([[4.0, 3.0], [2.0, 1.0]])
    real = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert _correlation_pct(predicted, real) == pytest.approx(-100.0)


def test_correlation_pct_none_for_constant_array():
    """Correlation is undefined (zero std -> division by zero) when
    either array is numerically constant -- must return None, not nan
    or a misleading 0, so the caller can render 'n/a' instead of a
    fake-looking number."""
    real = np.array([[1.0, 2.0], [3.0, 4.0]])
    constant = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert _correlation_pct(constant, real) is None
    assert _correlation_pct(real, constant) is None


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


def test_parse_fixed_window_windows_drive_letter():
    """Regression test: a Windows path's OWN colon (after the drive
    letter, e.g. 'D:\\work\\...') must not be mistaken for the
    run_dir/step delimiter. Before the fix, split(':') sliced this into
    ('D', '\\work\\...\\T950_n020_s79', '400000', ...), and int() on the
    path segment crashed -- this is the exact error report that motivated
    parse_fixed_window's rewrite to scan from the right for the trailing
    integer step numbers instead of naively splitting on every ':'."""
    from pathlib import Path
    run_dir, steps = parse_fixed_window(
        r"D:\work\NN\phase_field\datasets\64x64\T950_n020_s79:400000:600000:800000"
    )
    assert run_dir == Path(r"D:\work\NN\phase_field\datasets\64x64\T950_n020_s79")
    assert steps == [400000, 600000, 800000]


def test_parse_fixed_window_windows_drive_letter_two_steps():
    """Same Windows-colon regression, but the minimal 2-step case (the
    shape main.py's own truncation-for-3a reuse produces from a 3-step
    3b window)."""
    from pathlib import Path
    run_dir, steps = parse_fixed_window(r"D:\work\datasets\64x64\T625_n005_s191:100000:120000")
    assert run_dir == Path(r"D:\work\datasets\64x64\T625_n005_s191")
    assert steps == [100000, 120000]


def test_parse_fixed_window_rejects_all_numeric_parts():
    """Edge case inherent to the right-to-left scan heuristic: if EVERY
    colon-separated part parses as an int, there's no run_dir left at
    all. Documents the (rare, not expected in this project's T<temp>_
    n<noise>_s<seed>-shaped directory names) failure mode rather than
    silently returning a nonsense empty path."""
    with pytest.raises(ValueError, match="run_dir:step0:step1"):
        parse_fixed_window("100000:200000:300000")


def test_compute_sample_chains_through_full_window(tmp_run_dir):
    """
    THE core property the earlier bug fix exists for: compute_sample
    must chain f_theta.rollout() across the FULL window (steps[0] ->
    steps[-1]), not just compare steps[0]->steps[1] regardless of how
    many steps are given -- that was the exact bug (always testing
    1-step quality even for a 3-step window).

    Verified two ways: (1) the returned x_t_raw/x_next_raw/dt_total match
    the fixture's own known, deterministic values exactly; (2) the
    predicted result is cross-checked against an INDEPENDENT manual
    call to f_theta.rollout() with the same per-transition dts, using
    the real models -- not just trusted from reading the source.

    Originally written without being run (no torch in that sandbox) --
    traced by hand against the fixture values and the real
    Autoencoder/LatentDynamics/rollout() implementations. Confirmed
    against a real run since: it caught exactly the kind of gap hand-
    tracing can't -- ae_config here was minimal (just {"size": 64}),
    adequate for compute_sample() at the time this was written, but a
    LATER change (compute_sample resolving stream_configs from
    ae_config internally) started requiring latent_channels too, and
    nothing caught the mismatch until this test actually executed.
    """
    import torch
    from models.autoencoder import MultiStreamAutoencoder
    from models.encoder import Encoder
    from models.decoder import Decoder
    from models.latent_streams import LatentStreamConfig, LatentStreamMode
    from models.latent_dynamics import LatentDynamics
    from evaluation.check_rollout import compute_sample

    run_dir, steps = tmp_run_dir  # steps = [0, 1000, 2000, 3000, 4000], dt=0.05, size=64

    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.DECODER),
    }
    encoder = Encoder(input_size=64, in_channels=1, base_channels=4, stream_configs=stream_configs)
    decoder = Decoder(output_size=64, out_channels=1, base_channels=4,
                       latent_channels=4, latent_spatial_size=8)
    ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                 stream_configs=stream_configs)
    ae.eval()
    f_theta = LatentDynamics(latent_channels=4, n_theta=1, hidden_dim=16, n_hidden_layers=1)
    f_theta.eval()
    ae_config = {"size": 64, "latent_channels": 4, "latent_spatial_size": 8,
                 "stream_configs": {"state": {"channels": 4, "spatial_size": 8, "mode": "autoencoder"},
                                     "deriv": {"channels": 4, "spatial_size": 8, "mode": "decoder"}},
                 "recon_stream_name": "state"}
    device = torch.device("cpu")

    window = steps[:4]  # [0, 1000, 2000, 3000] -- a 3-step window, matching n_rollout_steps=3

    x_t_raw, x_next_raw, x_next_pred, x_next_ae_baseline, dt_total, dt_per_step = compute_sample(
        run_dir, window, ae, f_theta, ae_config, device,
    )

    # (1) known, deterministic fixture values: constant field = step/10000
    assert np.allclose(x_t_raw, 0 / 10000.0, atol=1e-3)       # step 0
    assert np.allclose(x_next_raw, 3000 / 10000.0, atol=1e-3)  # step 3000 (the FINAL step, not step 1000)
    assert dt_total == pytest.approx((3000 - 0) * 0.05)  # metadata.dt=0.05 in the fixture
    assert dt_per_step == pytest.approx([1000 * 0.05, 1000 * 0.05, 1000 * 0.05])

    # (2) independent cross-check: manually chain the SAME real models
    # with the same per-transition dts, and confirm compute_sample's
    # prediction matches exactly -- proves it's genuinely chaining
    # through rollout(), not silently doing a single big-dt call (the
    # old, buggy behavior) or stopping after one step.
    with torch.no_grad():
        x_all = torch.stack([
            torch.from_numpy(load.read_phi_half(run_dir / load.snapshot_filename(step), 64, 64))
            for step in window
        ]).unsqueeze(1)  # (len(window), 1, 64, 64) -- EVERY step, real z1 needed at each one
        x_all_encoded = ae.encoders["shared"](x_all)
        z0_t = x_all_encoded[DEFAULT_STREAM_NAME][0:1]
        z1_sequence = x_all_encoded["deriv"].unsqueeze(0)  # (1, len(window), C, 8, 8)
        dts = torch.tensor([[1000 * 0.05, 1000 * 0.05, 1000 * 0.05]], dtype=torch.float32)
        theta = torch.tensor([[0.8 - 1.0]], dtype=torch.float32)  # temperature - T0, from the fixture
        z0_hat_full = f_theta.rollout(z0_t, z1_sequence, dts, theta)
        expected_pred = ae.pathways["state"].decoder(z0_hat_full[:, -1])[0, 0].numpy()

    assert np.allclose(x_next_pred, expected_pred, atol=1e-5), (
        "compute_sample's prediction doesn't match an independently chained "
        "rollout() call with the same models and dts -- the chaining fix may "
        "have regressed."
    )


def test_error_bounds_expands_beyond_floor_for_a_genuinely_bad_prediction():
    """Regression test for a real, reported bug: the error panel's own
    scale used to be derived ONLY from real_delta (a fixed floor,
    factor=0.25), never checking whether the actual error data itself
    needed more room. A genuinely bad, noise-like prediction has error
    on the scale of the STATE itself, not real_delta -- far exceeding
    that floor -- and got clipped to the same narrow range regardless,
    saturating the panel and hiding exactly how wrong the prediction
    was. Reproduces the reported case directly: real_delta on a small,
    well-behaved scale (~0.016), but error on a much larger,
    noise-like scale (~0.9) -- exactly what a bad, large-dt
    extrapolation produces."""
    real_delta = np.array([-0.016, 0.016, 0.001, -0.001, 0.005, -0.008])
    error = np.array([-0.9, 0.85, 0.3, -0.6, 0.95, -0.4])

    floor_vmin, floor_vmax = _padded_bounds(real_delta, factor=0.25)
    vmin, vmax = _error_bounds(real_delta, error)

    # The OLD behavior (floor only) would have saturated -- confirm
    # the floor alone genuinely does NOT cover the actual error range,
    # so this test is actually exercising the fix, not a no-op.
    assert error.max() > floor_vmax and error.min() < floor_vmin

    # The FIX: actual error range must be fully covered, not clipped.
    assert vmin <= error.min()
    assert vmax >= error.max()


def test_error_bounds_keeps_the_floor_for_a_genuinely_small_error():
    """The floor's own benefit (fixed, comparable scale across images
    for the common case) must NOT be lost -- a small, accurate
    prediction's error panel should still use the real_delta-derived
    floor, not blow up to match noise in a tiny error array."""
    real_delta = np.array([-0.5, 0.5, 0.1, -0.1])
    error = np.array([-0.01, 0.008, 0.005, -0.006])  # comfortably within the floor

    floor_vmin, floor_vmax = _padded_bounds(real_delta, factor=0.25)
    vmin, vmax = _error_bounds(real_delta, error)

    assert vmin == pytest.approx(floor_vmin)
    assert vmax == pytest.approx(floor_vmax)
