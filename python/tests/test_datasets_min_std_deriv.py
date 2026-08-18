import torch
import pytest
from utils import load_datasets as load
from training.datasets import MicrostructureEvolutionDataset


def _build_straight_strip_run(base_dir, name, size=32):
    """Matches the user's own example exactly: a straight-strip
    interface, spatially complex (high std(OP)) but with ZERO
    curvature -- physically stationary, zero velocity, regardless of
    how sharp the interface is. The pattern is IDENTICAL across every
    saved step (a genuinely degenerate derivative), so min_stdev_phi
    alone (spatial-only) would keep every step, while min_std_deriv
    should correctly exclude every window built from them."""
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

    # A sharp, straight vertical strip: high spatial std, but IDENTICAL
    # at every single step -- the derivative between any two of them
    # is exactly zero everywhere.
    field = torch.full((size, size), -1.0)
    field[:, size // 3: 2 * size // 3] = 1.0
    for step in steps:
        arr = field.numpy().astype("<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))

    import pandas as pd
    stdev_phi = field.std().item()
    df = pd.DataFrame({"stdev_phi": [stdev_phi] * len(steps)}, index=steps)
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")
    return run_dir


def test_min_std_deriv_excludes_straight_strip_windows(tmp_path):
    run_dir = _build_straight_strip_run(tmp_path, "T800_n010_s1")

    # Sanity check: min_stdev_phi ALONE (spatial-only) does NOT exclude
    # this -- the strip is genuinely spatially complex.
    ds_spatial_only = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=2, min_stdev_phi=0.1,
    )
    assert len(ds_spatial_only) > 0, (
        "sanity check failed: min_stdev_phi alone should NOT exclude a straight strip "
        "(it's spatially complex, just not evolving) -- if this fails, the test's own "
        "synthetic data isn't set up the way this test assumes"
    )

    # min_std_deriv SHOULD exclude every window here -- the derivative
    # is exactly zero everywhere, every single window is degenerate.
    ds_with_deriv_filter = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=2, min_stdev_phi=0.1, min_std_deriv=1e-8,
    )
    assert len(ds_with_deriv_filter) == 0, (
        f"expected ALL windows excluded (zero derivative everywhere), got "
        f"{len(ds_with_deriv_filter)} remaining"
    )


def test_min_std_deriv_keeps_genuinely_evolving_windows(tmp_path):
    """Complementary check: a run that DOES evolve should NOT have its
    windows excluded -- confirms this isn't just rejecting everything."""
    run_dir = tmp_path / "T800_n010_s2"
    run_dir.mkdir()
    size = 32
    steps = [0, 1000, 2000, 3000, 4000]
    metadata_text = "\n".join([
        "directory = T800_n010_s2", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 4000",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        "seed = 1", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    # A strip whose position genuinely MOVES over time -- real, nonzero
    # derivative in the region the boundary sweeps through.
    import pandas as pd
    stdev_phis = []
    for i, step in enumerate(steps):
        field = torch.full((size, size), -1.0)
        boundary = size // 4 + i * 2  # moves each step
        field[:, boundary:boundary + 8] = 1.0
        arr = field.numpy().astype("<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))
        stdev_phis.append(field.std().item())
    df = pd.DataFrame({"stdev_phi": stdev_phis}, index=steps)
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")

    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=2, min_stdev_phi=0.1, min_std_deriv=1e-8,
    )
    assert len(ds) > 0, "a genuinely evolving run should NOT have all its windows excluded"


def test_min_std_deriv_rejects_cached_latent_mode(tmp_path):
    run_dir = _build_straight_strip_run(tmp_path, "T800_n010_s1")
    fake_encoder = torch.nn.Conv2d(1, 4, kernel_size=32, stride=32)
    with pytest.raises(ValueError, match="cached-latent"):
        MicrostructureEvolutionDataset([run_dir], encoder=fake_encoder, window_length=2,
                                         min_std_deriv=0.01)


def _build_varying_deriv_run(base_dir, name, size=16):
    """A run whose successive transitions have DIFFERENT derivative std --
    some frames barely change, some change a lot -- so a mid-range
    min_std_deriv keeps SOME windows and drops others. This is what
    exercises the vectorized filter's per-transition alignment (window
    `start` must use transition `start`, divided by ITS OWN dt), not just
    the all-in / all-out cases the strip test covers."""
    import numpy as np
    import pandas as pd
    run_dir = base_dir / name
    run_dir.mkdir()
    steps = [0, 1000, 3000, 6000, 10000, 15000]      # UNEVEN spacing on purpose
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 15000",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        "seed = 1", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)

    rng = np.random.default_rng(0)
    base = rng.standard_normal((size, size)).astype("float32")
    frames = [base]
    # each successive frame adds a perturbation of GROWING magnitude, so the
    # per-transition std increases along the run
    for k in range(1, len(steps)):
        frames.append(frames[-1] + (0.3 * k) * rng.standard_normal((size, size)).astype("float32"))
    for step, fr in zip(steps, frames):
        fr.astype("<f2").tofile(run_dir / load.snapshot_filename(step))

    stdev_phi = float(np.stack(frames).std())
    df = pd.DataFrame({"stdev_phi": [stdev_phi] * len(steps)}, index=steps)
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")
    return run_dir, steps


def test_vectorized_min_std_deriv_matches_the_scalar_formula(tmp_path):
    """The vectorized per-transition std filter must keep EXACTLY the windows
    the old per-window scalar computation would. Independently recompute the
    old formula ((run_data[s+1]-run_data[s])/(dstep*dt)).std() per window and
    compare the surviving window count against the dataset's own."""
    import numpy as np
    run_dir, steps = _build_varying_deriv_run(tmp_path, "T800_n010_s1")

    # Reconstruct the raw frames exactly as the loader sees them (read_phi_half)
    frames = np.stack([
        load.read_phi_half(run_dir / load.snapshot_filename(s), 16, 16) for s in steps
    ])  # (n, ny, nx) float32
    dt = 0.05
    window_length = 2

    # pick a threshold strictly between the smallest and largest transition std
    old_stds = []
    for start in range(len(steps) - 1):
        dstep = (steps[start + 1] - steps[start]) * dt
        d = (frames[start + 1] - frames[start]) / dstep
        old_stds.append(d.std(ddof=1))          # unbiased, matching torch default
    old_stds = np.array(old_stds)
    threshold = float(np.median(old_stds))       # keeps some, drops some
    expected_kept = int(np.sum(old_stds[:len(steps) - window_length + 1] >= threshold))

    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=window_length,
        min_stdev_phi=None, min_std_deriv=threshold,
    )
    assert len(ds) == expected_kept, (
        f"vectorized filter kept {len(ds)} windows, scalar formula expects "
        f"{expected_kept} (stds={old_stds}, thr={threshold})"
    )
    # and the mix is non-trivial -- guards against 'kept all' / 'kept none'
    # accidentally matching
    assert 0 < expected_kept < len(steps) - window_length + 1
