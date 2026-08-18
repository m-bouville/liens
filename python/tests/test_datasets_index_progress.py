"""The snapshot dataset's per-run stats/index build parses a statistics.csv
per run -- silent for a large sweep, right after build_good_steps' summary.
It now prints an in-place 'indexing runs:' counter, gated to a non-trivial
run set (small/tiny sets stay silent)."""
import io
from contextlib import redirect_stdout

import numpy as np

from utils import load_datasets as load
from training.datasets import MicrostructureSnapshotDataset


def _make_run(base, name, size=16):
    import pandas as pd
    run_dir = base / name
    run_dir.mkdir()
    steps = [0, 1000, 2000, 3000]
    (run_dir / "metadata.txt").write_text("\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 3000",
        f"save_steps = {' '.join(map(str, steps))}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        "seed = 1", "equation = allen_cahn", "solver = explicit", "",
    ]))
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    for step in steps:
        rng.standard_normal((size, size)).astype("<f2").tofile(
            run_dir / load.snapshot_filename(step))
    df = pd.DataFrame({"stdev_phi": [0.5] * len(steps), "avg_phi": [0.0] * len(steps)}, index=steps)
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")
    return run_dir


def test_indexing_progress_shown_for_a_large_run_set(tmp_path):
    run_dirs = [_make_run(tmp_path, f"T800_n010_s{i}") for i in range(25)]
    buf = io.StringIO()
    with redirect_stdout(buf):
        MicrostructureSnapshotDataset(
            run_dirs, min_step=0, min_stdev_phi=None, include_stats=False)
    assert "indexing runs:" in buf.getvalue()
    assert "25/25" in buf.getvalue()


def test_indexing_progress_silent_for_tiny_run_set(tmp_path):
    run_dirs = [_make_run(tmp_path, f"T800_n010_s{i}") for i in range(3)]
    buf = io.StringIO()
    with redirect_stdout(buf):
        MicrostructureSnapshotDataset(
            run_dirs, min_step=0, min_stdev_phi=None, include_stats=False)
    assert "indexing runs:" not in buf.getvalue()
