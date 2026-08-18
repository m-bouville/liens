"""The read/encode pass of MicrostructureEvolutionDataset is the one long
silent stretch of construction -- at 128x128 on CPU it runs for minutes with
no output, which looks hung. It now prints an in-place per-run counter. These
tests lock that it appears when actually encoding a non-trivial run set, and
stays SILENT for the raw-pixel (encoder=None) path and for tiny run sets."""
import io
from contextlib import redirect_stdout

import numpy as np

from utils import load_datasets as load
from training.datasets import MicrostructureEvolutionDataset
from tests.conftest import FakeEncoder


def _make_run(base, name, size=16):
    run_dir = base / name
    run_dir.mkdir()
    steps = [0, 1000, 2000, 3000, 4000]
    meta = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 4000",
        f"save_steps = {' '.join(map(str, steps))}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        "seed = 1", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(meta)
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    for step in steps:
        # non-degenerate frames so nothing is filtered before encoding
        (rng.standard_normal((size, size)).astype("<f2")).tofile(
            run_dir / load.snapshot_filename(step))
    return run_dir


def test_encode_pass_prints_progress_for_a_nontrivial_run_set(tmp_path):
    run_dirs = [_make_run(tmp_path, f"T800_n010_s{i}") for i in range(25)]
    enc = FakeEncoder(size=16, latent_channels=4)
    buf = io.StringIO()
    with redirect_stdout(buf):
        MicrostructureEvolutionDataset(
            run_dirs, encoder=enc, window_length=3,
            min_step=0, min_stdev_phi=None,
        )
    out = buf.getvalue()
    assert "encoding runs:" in out, out
    assert f"/{len(run_dirs)}" in out          # shows the total
    # the counter reaches the last run and the line is terminated with a newline
    assert f"{len(run_dirs)}/{len(run_dirs)}" in out
    assert out.rstrip(" ").endswith("\n") or "\n" in out.split("encoding runs:")[-1]


def test_raw_pixel_mode_stays_silent(tmp_path):
    """encoder=None encodes nothing here -- no progress line should appear."""
    run_dirs = [_make_run(tmp_path, f"T800_n010_s{i}") for i in range(25)]
    buf = io.StringIO()
    with redirect_stdout(buf):
        MicrostructureEvolutionDataset(
            run_dirs, encoder=None, window_length=3,
            min_step=0, min_stdev_phi=None,
        )
    assert "encoding runs:" not in buf.getvalue()


def test_tiny_run_set_stays_silent(tmp_path):
    """Below the threshold the counter would be noise, not help."""
    run_dirs = [_make_run(tmp_path, f"T800_n010_s{i}") for i in range(3)]
    enc = FakeEncoder(size=16, latent_channels=4)
    buf = io.StringIO()
    with redirect_stdout(buf):
        MicrostructureEvolutionDataset(
            run_dirs, encoder=enc, window_length=3,
            min_step=0, min_stdev_phi=None,
        )
    assert "encoding runs:" not in buf.getvalue()
