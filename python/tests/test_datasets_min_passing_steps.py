"""
Tests for build_good_steps' min_passing_steps parameter: excluding an
ENTIRE run when too few of its steps clear min_stdev_phi, rather than
keeping whatever small, borderline-passing handful individual per-step
filtering alone would let through (see build_good_steps' own docstring
for the full rationale -- a real example from this project's own
sweep, T990_n007_s599, has only 2-4 of 65 steps ever clear a 1%
stdev_phi threshold).

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_datasets_min_passing_steps.py -v
"""
import numpy as np
import pandas as pd
import pytest

from training.datasets import MicrostructureEvolutionDataset, build_good_steps
from utils import load_datasets as load


def _write_run(base_path, name, n_steps, n_passing, threshold=0.01, size=64):
    """
    A fake run directory in the format load.read_metadata/
    read_phi_half/read_statistics_csv expect, with statistics.csv's
    stdev_phi constructed so EXACTLY n_passing of its n_steps clear
    threshold -- deterministic (not random-and-hope), so tests assert
    an exact expected count rather than "probably enough".
    """
    run_dir = base_path / name
    run_dir.mkdir()
    steps = list(range(0, n_steps * 1000, 1000))
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", f"steps = {steps[-1]}",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.99", "kappa = 0.2",
        "mobility = 0.05", "phi0 = 0.0", "noise = 0.007", "seed = 1",
        "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)

    for step in steps:
        arr = np.full((size, size), 0.1, dtype="<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))

    assert 0 <= n_passing <= n_steps
    stdevs = [threshold + 0.01] * n_passing + [threshold - 0.005] * (n_steps - n_passing)
    pd.DataFrame({"step": steps, "stdev_phi": stdevs}).to_csv(run_dir / "statistics.csv", index=False)
    return run_dir


def test_min_passing_steps_drops_entire_sparse_run(tmp_path):
    """The T990_n007_s599 scenario directly: a run where only a handful
    of steps EVER clear min_stdev_phi must be excluded ENTIRELY (zero
    kept steps), not reduced to just those few survivors."""
    sparse_run = _write_run(tmp_path, "sparse", n_steps=65, n_passing=4)
    normal_run = _write_run(tmp_path, "normal", n_steps=65, n_passing=52)

    good_steps = build_good_steps([sparse_run, normal_run], min_step=0,
                                   min_stdev_phi=0.01, min_passing_steps=8)

    assert good_steps[sparse_run] == [], "a run with only 4 passing steps (< 8) must be dropped entirely"
    assert len(good_steps[normal_run]) == 52, "a run with 52 passing steps (>= 8) must be untouched"


def test_min_passing_steps_none_is_default_behavior(tmp_path):
    """min_passing_steps=None (the default) must reproduce the exact
    prior behavior -- per-step filtering only, no whole-run exclusion,
    regardless of how few steps in a run happen to pass."""
    sparse_run = _write_run(tmp_path, "sparse", n_steps=65, n_passing=4)

    good_steps = build_good_steps([sparse_run], min_step=0, min_stdev_phi=0.01)  # min_passing_steps omitted
    assert len(good_steps[sparse_run]) == 4, "without min_passing_steps, the 4 individually-passing steps should remain"


def test_min_passing_steps_exactly_at_threshold_is_kept(tmp_path):
    """A run with EXACTLY min_passing_steps passing steps must be kept
    (>= comparison, not strictly greater-than)."""
    run_dir = _write_run(tmp_path, "exact", n_steps=20, n_passing=8)
    good_steps = build_good_steps([run_dir], min_step=0, min_stdev_phi=0.01, min_passing_steps=8)
    assert len(good_steps[run_dir]) == 8, "a run with exactly min_passing_steps passing steps must be KEPT, not dropped"


def test_min_passing_steps_requires_min_stdev_phi(tmp_path):
    """min_passing_steps without min_stdev_phi is a meaningless
    configuration (nothing to 'pass') and must fail loudly, not
    silently do nothing or silently drop every run."""
    with pytest.raises(ValueError, match="min_stdev_phi"):
        build_good_steps([tmp_path / "doesnt_matter"], min_stdev_phi=None, min_passing_steps=8)


def test_min_passing_steps_excludes_run_from_actual_dataset(tmp_path, fake_encoder):
    """End-to-end, not just build_good_steps in isolation: a run
    dropped by min_passing_steps must contribute ZERO windows to a real
    MicrostructureEvolutionDataset -- not just have an empty entry in
    the intermediate good_steps mapping that something downstream
    forgets to respect."""
    sparse_run = _write_run(tmp_path, "sparse", n_steps=65, n_passing=4)
    normal_run = _write_run(tmp_path, "normal", n_steps=65, n_passing=52)

    ds_with_filter = MicrostructureEvolutionDataset(
        [sparse_run, normal_run], encoder=fake_encoder, window_length=2,
        min_step=0, min_stdev_phi=0.01, min_passing_steps=8,
    )
    assert sparse_run not in ds_with_filter._run_dirs, "the dropped run must not appear in the dataset at all"
    assert normal_run in ds_with_filter._run_dirs

    ds_without_filter = MicrostructureEvolutionDataset(
        [sparse_run, normal_run], encoder=fake_encoder, window_length=2,
        min_step=0, min_stdev_phi=0.01,  # min_passing_steps omitted
    )
    assert sparse_run in ds_without_filter._run_dirs, "without min_passing_steps, the sparse run's 4 steps still count"
