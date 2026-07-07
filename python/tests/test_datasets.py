"""
Tests for MicrostructureEvolutionDataset, in particular the
encoder=None (raw-pixel, stage 4/5) mode added alongside the existing
encoder-given (cached-latent, stage 3) mode.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_datasets.py -v
"""
import torch

from training.datasets import MicrostructureEvolutionDataset


def test_cached_mode_window_shape(tmp_run_dir, fake_encoder):
    """Stage 3's existing mode: encoder given -> latent windows."""
    run_dir, steps = tmp_run_dir
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=fake_encoder, window_length=3, min_step=0, min_stdev_phi=None,
    )
    assert len(ds) == len(steps) - 3 + 1  # 5 steps, window_length=3 -> 3 windows
    window, dt_window, theta = ds[0]
    assert window.shape == (3, 4, 8, 8)  # (window_length, latent_channels, 8, 8)
    assert dt_window.shape == (2,)
    assert theta.shape == (1,)


def test_raw_mode_window_shape(tmp_run_dir):
    """New mode: encoder=None -> raw pixel windows, no encoding done here."""
    run_dir, steps = tmp_run_dir
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
    )
    assert len(ds) == len(steps) - 3 + 1
    window, dt_window, theta = ds[0]
    assert window.shape == (3, 1, 64, 64)  # (window_length, 1, ny, nx) -- raw, unencoded
    assert dt_window.shape == (2,)
    assert theta.shape == (1,)


def test_raw_mode_values_match_real_files(tmp_run_dir):
    """
    The raw window's actual pixel values should be exactly what's on
    disk for those steps -- not just the right shape. Each snapshot in
    the fixture is a constant field equal to step/10000, so this is a
    direct, checkable round-trip.
    """
    run_dir, steps = tmp_run_dir
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
    )
    window, _, _ = ds[0]  # should be steps[0:3] = [0, 1000, 2000]
    for i, step in enumerate(steps[:3]):
        expected_value = step / 10000.0
        assert torch.allclose(window[i], torch.full_like(window[i], expected_value), atol=1e-3), \
            f"frame {i} (step {step}): expected constant {expected_value}, got {window[i].unique()}"


def test_window_info_matches_dataset_content(tmp_run_dir):
    """
    window_info's reported (run_dir, steps) should always correspond to
    what __getitem__ actually returned for that index -- true in either
    mode, since window_info doesn't depend on which mode built the
    dataset (see its docstring).
    """
    run_dir, steps = tmp_run_dir
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
    )
    for idx in range(len(ds)):
        window, _, _ = ds[idx]
        info_run_dir, info_steps = ds.window_info(idx)
        assert info_run_dir == run_dir
        for i, step in enumerate(info_steps):
            expected_value = step / 10000.0
            assert torch.allclose(window[i], torch.full_like(window[i], expected_value), atol=1e-3)


def test_cross_mode_consistency(tmp_run_dir, fake_encoder):
    """
    The strongest check: cached-mode latents and raw-mode frames should
    be genuinely THE SAME WINDOW, just deferred vs precomputed -- not
    two subtly different things that happen to look similar. Manually
    encoding the raw-mode window with the same encoder should exactly
    reproduce the cached-mode window.
    """
    run_dir, steps = tmp_run_dir
    cached_ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=fake_encoder, window_length=3, min_step=0, min_stdev_phi=None,
    )
    raw_ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
    )
    assert len(cached_ds) == len(raw_ds)

    fake_encoder.eval()
    for idx in range(len(cached_ds)):
        cached_window, cached_dt, cached_theta = cached_ds[idx]
        raw_window, raw_dt, raw_theta = raw_ds[idx]

        with torch.no_grad():
            manually_encoded = fake_encoder(raw_window)  # (window_length, latent_channels, 8, 8)

        assert torch.allclose(cached_window, manually_encoded, atol=1e-5), \
            f"idx {idx}: cached-mode latents don't match manually-encoding the raw-mode window"
        assert torch.allclose(cached_dt, raw_dt)
        assert torch.allclose(cached_theta, raw_theta)


def test_raw_mode_allows_gradient_flow(tmp_run_dir, fake_encoder):
    """
    The whole POINT of the raw-pixel mode: encoding must happen with
    gradient tracking enabled when the caller wants it (stage 4/5 needs
    gradient to flow from a rollout loss back into E). Confirms the
    dataset itself doesn't do anything (e.g. an accidental .detach() or
    no_grad()) that would silently break this.
    """
    run_dir, steps = tmp_run_dir
    raw_ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
    )
    window, _, _ = raw_ds[0]
    encoded = fake_encoder(window)
    loss = encoded.sum()
    loss.backward()

    grad = fake_encoder.conv.weight.grad
    assert grad is not None, "gradient did not flow back into the encoder's parameters"
    assert torch.any(grad != 0), "gradient flowed but was all zero -- suspicious"
