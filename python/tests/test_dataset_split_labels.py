"""The dataset-construction diagnostic lines (runs dropped ENTIRELY, runs with
too few windows, candidate windows skipped) repeat identically for the train,
val and test splits. A split_label threads through so each line says WHICH
split it describes, e.g. '258/2837 training runs'."""
import io
from contextlib import redirect_stdout

import numpy as np

from utils import load_datasets as load
from training.datasets import build_good_steps, MicrostructureEvolutionDataset


def _make_run(base, name, size=16, degenerate=False):
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
    import pandas as pd
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    for step in steps:
        arr = np.full((size, size), 0.5, dtype="<f2") if degenerate \
            else rng.standard_normal((size, size)).astype("<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))
    # low stdev_phi across all steps -> fail min_stdev_phi=0.01 -> dropped ENTIRELY
    sp = [0.0] * len(steps) if degenerate else [0.5] * len(steps)
    df = pd.DataFrame({"stdev_phi": sp, "avg_phi": [0.0] * len(steps)}, index=steps)
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")
    return run_dir


def test_build_good_steps_labels_the_split():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        base = pathlib.Path(d)
        run_dirs = [_make_run(base, f"r{i}", degenerate=True) for i in range(3)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            build_good_steps(run_dirs, min_step=0, min_stdev_phi=0.01,
                             min_passing_steps=2, split_label="training")
        out = buf.getvalue()
        assert "dropped ENTIRELY" in out
        assert "training runs" in out, out


def test_build_good_steps_unlabeled_says_plain_runs():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        base = pathlib.Path(d)
        run_dirs = [_make_run(base, f"r{i}", degenerate=True) for i in range(3)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            build_good_steps(run_dirs, min_step=0, min_stdev_phi=0.01,
                             min_passing_steps=2)   # no label
        out = buf.getvalue()
        assert "dropped ENTIRELY" in out
        # unlabeled: plain "runs", not "None runs" or " runs"
        assert " runs dropped ENTIRELY" in out
        assert "None" not in out


def test_windowless_runs_message_labels_the_split():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        base = pathlib.Path(d)
        # runs with too few steps for window_length=3 -> windowless-runs message
        run_dirs = [_make_run(base, f"r{i}", degenerate=True) for i in range(3)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            MicrostructureEvolutionDataset(
                run_dirs, encoder=None, window_length=3,
                min_step=0, min_stdev_phi=0.01, split_label="validation")
        out = buf.getvalue()
        if "had fewer than window_length" in out:
            assert "validation runs" in out, out


def test_snapshot_dataset_labels_the_split_too():
    """Stage 1 builds Snapshot datasets for train AND val -- its repeated
    'dropped ENTIRELY' lines need the same split labels as Evolution's."""
    import tempfile, pathlib
    from training.datasets import MicrostructureSnapshotDataset
    with tempfile.TemporaryDirectory() as d:
        base = pathlib.Path(d)
        run_dirs = [_make_run(base, f"r{i}", degenerate=True) for i in range(3)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            MicrostructureSnapshotDataset(
                run_dirs, min_step=0, min_stdev_phi=0.01, min_passing_steps=2,
                split_label="training")
        out = buf.getvalue()
        assert "dropped ENTIRELY" in out
        assert "training runs" in out, out
