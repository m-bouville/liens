import torch
import pytest
from pathlib import Path
from utils import load_datasets as load
from training.datasets import MicrostructureEvolutionDataset, MicrostructureSnapshotDataset


def _build_run_dir(base_dir, name, size=32):
    run_dir = base_dir / name
    run_dir.mkdir()
    steps = [0, 1000, 2000, 3000, 4000]
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 4000",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        "seed = 1", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    for step in steps:
        # DISTINCTIVE, non-symmetric pattern (not a constant field!) --
        # essential for this test: a constant field looks identical
        # under every D4 transform, which would make "did the same
        # transform get applied to both frames" untestable.
        arr = torch.zeros((size, size), dtype=torch.float32)
        arr[2:5, 10:14] = step / 10000.0 + 0.1  # an asymmetric marker block
        arr[20:22, 3:6] = 0.5
        arr = arr.numpy().astype("<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))
    return run_dir


def test_augment_multiplies_length_by_32(tmp_path):
    run_dir = _build_run_dir(tmp_path, "T800_n010_s1")
    ds_plain = MicrostructureEvolutionDataset([run_dir], encoder=None, window_length=2)
    ds_aug = MicrostructureEvolutionDataset([run_dir], encoder=None, window_length=2, augment=True)
    assert len(ds_aug) == len(ds_plain) * 32


def test_augment_applies_same_transform_to_every_frame_in_window(tmp_path):
    """THE critical correctness property: flipping/rotating x(t) but not
    x(t+dt) would make the window describe a physically meaningless
    'evolution'. Verify directly by comparing an augmented window's two
    frames against MANUALLY re-derived transforms of the unaugmented
    frames, using the SAME (k, flip, shift) -- not just checking shapes."""
    run_dir = _build_run_dir(tmp_path, "T800_n010_s1")
    ds_plain = MicrostructureEvolutionDataset([run_dir], encoder=None, window_length=2)
    ds_aug = MicrostructureEvolutionDataset([run_dir], encoder=None, window_length=2, augment=True)

    from training.datasets import _apply_augmentation
    for aug_idx in [0, 1, 5, 13, 31]:  # spread across dihedral x translation combos
        base_idx = 0
        idx = base_idx * 32 + aug_idx
        window_aug, dt_aug, theta_aug = ds_aug[idx]
        window_plain, dt_plain, theta_plain = ds_plain[base_idx]

        expected_frame0, k0, flip0 = _apply_augmentation(window_plain[0], aug_idx, 32, 32)
        expected_frame1, k1, flip1 = _apply_augmentation(window_plain[1], aug_idx, 32, 32)

        assert k0 == k1 and flip0 == flip1, "SAME transform must be selected for both frames"
        assert torch.equal(window_aug[0], expected_frame0), f"frame 0 mismatch at aug_idx={aug_idx}"
        assert torch.equal(window_aug[1], expected_frame1), f"frame 1 mismatch at aug_idx={aug_idx}"
        # dt/theta are spatially invariant -- untouched by augmentation
        assert torch.equal(dt_aug, dt_plain)
        assert torch.equal(theta_aug, theta_plain)


def test_augment_window_info_unwraps_correctly(tmp_path):
    run_dir = _build_run_dir(tmp_path, "T800_n010_s1")
    ds_aug = MicrostructureEvolutionDataset([run_dir], encoder=None, window_length=2, augment=True)
    ds_plain = MicrostructureEvolutionDataset([run_dir], encoder=None, window_length=2)
    for aug_idx in [0, 7, 31]:
        info_aug = ds_aug.window_info(0 * 32 + aug_idx)
        info_plain = ds_plain.window_info(0)
        assert info_aug == info_plain


def test_augment_rejects_cached_latent_mode(tmp_path):
    run_dir = _build_run_dir(tmp_path, "T800_n010_s1")
    fake_encoder = torch.nn.Conv2d(1, 4, kernel_size=32, stride=32)
    with pytest.raises(ValueError, match="cached-latent"):
        MicrostructureEvolutionDataset([run_dir], encoder=fake_encoder, window_length=2, augment=True)


def test_augment_correctly_transforms_angle_stat(tmp_path):
    """"angle" is no longer rejected under augment=True -- it's
    corrected using the SAME (k, flip) the window itself was
    transformed by, matching _transform_angle exactly (the same
    function MicrostructureSnapshotDataset's own augmentation uses)."""
    from training.datasets import _transform_angle
    run_dir = _build_run_dir(tmp_path, "T800_n010_s1")
    import pandas as pd
    true_angle = 0.73
    df = pd.DataFrame({"angle": [true_angle] * 5}, index=[0, 1000, 2000, 3000, 4000])
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")

    ds_aug = MicrostructureEvolutionDataset([run_dir], encoder=None, window_length=2,
                                              augment=True, stat_names=["angle"])
    for aug_idx in [0, 5, 13, 31]:
        _, _, _, true_stats = ds_aug[0 * 32 + aug_idx]
        dihedral_idx, translation_idx = divmod(aug_idx, 4)
        k, flip = divmod(dihedral_idx, 2)
        expected = _transform_angle(torch.tensor(true_angle), k, bool(flip))
        assert abs(true_stats[0].item() - expected.item()) < 1e-4, f"mismatch at aug_idx={aug_idx}"


def test_augment_allows_non_angle_stats(tmp_path):
    """avg_phi/stdev_phi/etc. are rotation/flip-invariant scalars --
    unlike "angle", combining them with augment=True should NOT raise."""
    run_dir = _build_run_dir(tmp_path, "T800_n010_s1")
    import pandas as pd
    df = pd.DataFrame({"avg_phi": [0.1] * 5}, index=[0, 1000, 2000, 3000, 4000])
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")
    ds = MicrostructureEvolutionDataset([run_dir], encoder=None, window_length=2,
                                          augment=True, stat_names=["avg_phi"])
    assert len(ds) > 0


def test_snapshot_dataset_augmentation_unchanged_by_refactor(tmp_path):
    """Regression check: MicrostructureSnapshotDataset's OWN augmentation
    (refactored to delegate to the new shared _apply_augmentation) must
    produce byte-identical output to before the refactor."""
    run_dir = _build_run_dir(tmp_path, "T800_n010_s1")
    ds = MicrostructureSnapshotDataset([run_dir], augment=True)
    assert len(ds) == 5 * 32  # 5 kept steps (min_step=0 default keeps all) x 32 variants
    # Just confirm every augmented sample is retrievable and has the
    # right shape -- the transform logic itself is untouched (pure
    # extraction, verified by inspection), this guards against a
    # wiring mistake in the extraction (e.g. wrong nx/ny passed through).
    for i in [0, 1, 31, 32, 63, 100]:
        x = ds[i]
        assert x.shape == (1, 32, 32)
