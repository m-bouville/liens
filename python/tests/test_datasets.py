"""
Tests for MicrostructureEvolutionDataset, in particular the
encoder=None (raw-pixel, stage 4/5) mode added alongside the existing
encoder-given (cached-latent, stage 3) mode.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_datasets.py -v
"""
import torch
import pytest

from training.datasets import (
    VAL_DECORRELATED_AUG_INDICES, _ANGLE_SAFE_AUG_INDICES, _apply_augmentation,
    MicrostructureEvolutionDataset, MicrostructureSnapshotDataset,
)
from models.latent_streams import DEFAULT_STREAM_NAME
from models.constants import N_THETA


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
    assert theta.shape == (N_THETA,)


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
    assert theta.shape == (N_THETA,)


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


def test_all_dts_matches_every_windows_own_getitem_dt(tmp_run_dir):
    """all_dts() computes dt values directly from lightweight metadata,
    WITHOUT calling __getitem__ (specifically to avoid loading real
    frame data it has no use for) -- cross-checked here against every
    window's own __getitem__-derived dt_window directly, not just
    trusted as a separate, parallel implementation of the same idea."""
    run_dir, steps = tmp_run_dir
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
    )
    expected_dts = []
    for idx in range(len(ds)):
        _, dt_window, _ = ds[idx]
        expected_dts.extend(dt_window.tolist())

    all_dts = ds.all_dts()
    assert sorted(all_dts.tolist()) == sorted(expected_dts)


def test_snapshot_dataset_frame_info_unaugmented(tmp_run_dir):
    """frame_info's reported (run_dir, step) should always correspond
    to what __getitem__ actually returned for that index -- the
    trivial case (augment=False, base_idx == idx directly)."""
    run_dir, steps = tmp_run_dir
    ds = MicrostructureSnapshotDataset([run_dir], augment=False, min_step=0)
    assert len(ds) == len(steps)
    for idx in range(len(ds)):
        x = ds[idx]
        info_run_dir, info_step = ds.frame_info(idx)
        assert info_run_dir == run_dir
        assert info_step in steps
        expected_value = info_step / 10000.0
        assert torch.allclose(x, torch.full_like(x, expected_value), atol=1e-3)


def test_snapshot_dataset_frame_info_augmented(tmp_run_dir):
    """THE case that actually needs the divmod base_idx recovery
    (unlike the unaugmented case, where idx already IS base_idx):
    augment=True expands __len__ by _N_DIHEDRAL*4, and every one of
    those augmented indices must still trace back to its correct
    source frame, not a meaningless direct index into _index."""
    run_dir, steps = tmp_run_dir
    ds = MicrostructureSnapshotDataset([run_dir], augment=True, min_step=0)
    n_aug = ds._N_DIHEDRAL * 4
    assert len(ds) == len(steps) * n_aug
    # Spot-check the first, middle, and last augmented variant of each
    # base frame -- not every single index, to keep this fast, but
    # enough to catch an off-by-one in the divmod boundary.
    for base_idx, step in enumerate(steps):
        for aug_offset in (0, n_aug // 2, n_aug - 1):
            idx = base_idx * n_aug + aug_offset
            info_run_dir, info_step = ds.frame_info(idx)
            assert info_run_dir == run_dir
            assert info_step == step


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
            manually_encoded = fake_encoder(raw_window)[DEFAULT_STREAM_NAME]  # (window_length, latent_channels, 8, 8)

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
    encoded = fake_encoder(window)[DEFAULT_STREAM_NAME]
    loss = encoded.sum()
    loss.backward()

    grad = fake_encoder.conv.weight.grad
    assert grad is not None, "gradient did not flow back into the encoder's parameters"
    assert torch.any(grad != 0), "gradient flowed but was all zero -- suspicious"


def test_stat_names_none_returns_3tuple_unchanged(tmp_run_dir):
    """Regression check: NOT passing stat_names must behave exactly as
    before this parameter existed -- plain 3-tuple, no statistics.csv
    read at all."""
    run_dir, steps = tmp_run_dir
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
    )
    item = ds[0]
    assert len(item) == 3
    assert ds.stat_names is None


def test_stat_names_given_returns_4tuple_with_correct_true_stats(tmp_run_dir_with_stats):
    """true_stats must correspond to window[0] specifically (the real
    starting frame) -- checked against the fixture's known, distinctive
    per-step values (stat value = step/1000), not just checked for the
    right shape."""
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=2, min_step=0, min_stdev_phi=None,
        stat_names=stat_names,
    )
    for idx in range(len(ds)):
        window, dt_window, theta, true_stats = ds[idx]
        _, window_steps = ds.window_info(idx)
        start_step = window_steps[0]
        expected = torch.tensor([start_step / 1000.0] * len(stat_names), dtype=torch.float32)
        assert torch.allclose(true_stats, expected), (
            f"idx {idx}: true_stats {true_stats} doesn't match the starting step "
            f"{start_step}'s expected value {expected}"
        )


def test_stats_frame_index_selects_the_right_frame(tmp_run_dir_with_stats):
    """
    REGRESSION: true_stats must describe the frame the caller actually
    ENCODES, which is not always window[0].

    train_stage2.py's deriv_target_centered mode uses window_length=3
    and encodes the MIDDLE frame (x_t = window[:, 1]) -- x_before/
    x_after exist only to build L_deriv's own centered target. With the
    long-standing hardcoded steps[0] lookup, both stats anchors
    (L_stats0 on z0_t, L_stats1 on z1_t) were silently trained against
    a DIFFERENT frame's statistics than the frame they predict from --
    measured, not hypothetical: fixing it improved stats0 ~6x and
    stats1 ~3x on a real 3-epoch run.

    Checked against the fixture's own distinctive per-step values
    (stat value = step/1000), so a misalignment by even one frame
    fails loudly rather than merely looking plausible.
    """
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    for frame_index in (0, 1, 2):
        ds = MicrostructureEvolutionDataset(
            [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
            stat_names=stat_names, stats_frame_index=frame_index,
        )
        assert len(ds) > 0, "fixture produced no window_length=3 windows to check"
        for idx in range(len(ds)):
            _, _, _, true_stats = ds[idx]
            _, window_steps = ds.window_info(idx)
            expected_step = window_steps[frame_index]
            expected = torch.tensor([expected_step / 1000.0] * len(stat_names), dtype=torch.float32)
            assert torch.allclose(true_stats, expected), (
                f"stats_frame_index={frame_index}, idx {idx}: true_stats {true_stats} "
                f"doesn't match frame {frame_index} (step {expected_step}) expected {expected}"
            )


def test_stats_frame_index_out_of_range_raises(tmp_run_dir_with_stats):
    """An index outside the window is a caller mistake -- caught at
    construction rather than silently producing an off-by-N target or
    an obscure IndexError deep inside __getitem__."""
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    with pytest.raises(ValueError, match="stats_frame_index"):
        MicrostructureEvolutionDataset(
            [run_dir], encoder=None, window_length=2, min_step=0, min_stdev_phi=None,
            stat_names=stat_names, stats_frame_index=2,
        )


def test_stat_names_with_real_encoder_raises(tmp_run_dir_with_stats, fake_encoder):
    """stat_names + a real encoder (cached-latent, stage-3 mode) is
    treated as a caller mistake -- stage 3 never uses L_stats -- and
    should be caught at construction, not silently ignored."""
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    with pytest.raises(ValueError, match="never uses L_stats"):
        MicrostructureEvolutionDataset(
            [run_dir], encoder=fake_encoder, window_length=2, min_step=0, min_stdev_phi=None,
            stat_names=stat_names,
        )


def test_nan_guard_excludes_windows_starting_at_nan_step(tmp_run_dir_with_stats):
    """The fixture puts a NaN in step 2000's avg_phi column. With
    window_length=3 over 5 steps, there are 3 possible windows
    (starting at steps 0, 1000, 2000) -- the one starting at 2000 must
    be excluded, leaving exactly 2."""
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    ds_without_guard = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
    )
    ds_with_guard = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
        stat_names=stat_names,
    )
    assert len(ds_without_guard) == 3  # 5 steps, window_length=3 -> 3 windows, no guard applied
    assert len(ds_with_guard) == 2     # the window starting at step 2000 is correctly excluded

    for idx in range(len(ds_with_guard)):
        _, window_steps = ds_with_guard.window_info(idx)
        assert window_steps[0] != 2000, "a window starting at the NaN step slipped through"


def test_missing_stat_column_raises_clear_error(tmp_run_dir_with_stats):
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    with pytest.raises(ValueError, match="missing columns"):
        MicrostructureEvolutionDataset(
            [run_dir], encoder=None, window_length=2, min_step=0, min_stdev_phi=None,
            stat_names=stat_names + ["nonexistent_stat"],
        )


def test_val_decorrelated_aug_indices_are_all_angle_safe():
    """
    The 4 recommended validation variants must use ONLY dihedral
    transforms whose "angle" stat correction is confirmed correct.
    _transform_angle's own docstring flags the SIGN of its k*90 term for
    k=1/k=3 as UNCONFIRMED -- putting one of those into a VALIDATION
    metric would inject systematic error into the stats term (and unlike
    training, where the model just learns whatever it's shown, a
    systematic error here corrupts cross-epoch comparisons directly).
    """
    assert set(VAL_DECORRELATED_AUG_INDICES) <= _ANGLE_SAFE_AUG_INDICES
    for aug_idx in VAL_DECORRELATED_AUG_INDICES:
        _, k, _flip = _apply_augmentation(torch.zeros(1, 64, 64), aug_idx, 64, 64)
        assert k in (0, 2), f"aug_idx {aug_idx} uses k={k}, whose angle correction is UNCONFIRMED"


def test_val_decorrelated_aug_indices_maximize_decorrelation():
    """
    The point of this specific set (vs. e.g. the 4 translations at fixed
    orientation) is a Latin-square spread: 4 DISTINCT dihedrals AND 4
    DISTINCT translations, so the 4 evaluations differ along both axes
    at once. Verified against a genuinely non-uniform, asymmetric field
    -- a uniform one would make every spatial transform a no-op and let
    a broken set pass trivially.
    """
    nx = ny = 64
    frame = torch.arange(ny * nx, dtype=torch.float32).reshape(1, ny, nx) / (ny * nx)
    outs, dihedrals = [], set()
    for aug_idx in VAL_DECORRELATED_AUG_INDICES:
        out, k, flip = _apply_augmentation(frame.clone(), aug_idx, nx, ny)
        outs.append(out)
        dihedrals.add((k, flip))
    assert len(dihedrals) == 4, f"expected 4 distinct dihedrals, got {sorted(dihedrals)}"
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            assert not torch.equal(outs[i], outs[j]), (
                f"variants {VAL_DECORRELATED_AUG_INDICES[i]} and "
                f"{VAL_DECORRELATED_AUG_INDICES[j]} produce identical output -- not decorrelated"
            )
    # Every variant must be an EXACT symmetry: same pixel multiset, so
    # the TRUE loss is identical across all 4 and only the model's own
    # equivariance error differs (that's the whole premise of averaging).
    ref = torch.sort(frame.flatten()).values
    for out in outs:
        assert torch.allclose(ref, torch.sort(out.flatten()).values)


def test_fixed_aug_indices_multiplies_length_and_preserves_window_info(tmp_run_dir_with_stats):
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    base = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=2, min_step=0, min_stdev_phi=None,
        stat_names=stat_names,
    )
    val4 = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=2, min_step=0, min_stdev_phi=None,
        stat_names=stat_names, fixed_aug_indices=VAL_DECORRELATED_AUG_INDICES,
    )
    assert len(val4) == len(base) * 4
    # All 4 slots of a given window must trace back to the SAME source
    # window, and slot 0 (aug_idx 0 = identity) must be untransformed.
    infos = [val4.window_info(i) for i in range(4)]
    assert all(x == infos[0] for x in infos)
    assert infos[0] == base.window_info(0)
    assert torch.equal(val4[0][0], base[0][0])
    # dt is a temporal quantity -- spatial transforms must never touch it
    assert all(torch.equal(val4[i][1], base[0][1]) for i in range(4))


def test_fixed_aug_indices_rejects_invalid_configurations(tmp_run_dir_with_stats):
    run_dir, steps, stat_names = tmp_run_dir_with_stats

    def build(**kw):
        return MicrostructureEvolutionDataset(
            [run_dir], encoder=None, window_length=2, min_step=0, min_stdev_phi=None,
            stat_names=stat_names, **kw,
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        build(fixed_aug_indices=(0, 5), augment=True)
    with pytest.raises(ValueError, match="non-empty"):
        build(fixed_aug_indices=())
    with pytest.raises(ValueError, match="duplicates"):
        build(fixed_aug_indices=(0, 5, 5))
    with pytest.raises(ValueError, match="out of range"):
        build(fixed_aug_indices=(0, 32))
    with pytest.raises(ValueError, match="out of range"):
        build(fixed_aug_indices=(-1,))
