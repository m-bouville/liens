"""MicrostructureEvolutionDataset.require_consecutive: a window must be
window_length CONSECUTIVE SAVED steps. When a step-level filter drops a frame
inside a run, the two kept frames on either side become adjacent in kept_steps
while a real saved frame sits between them -- a window spanning that seam jumps
over the gap and is NOT window_length adjacent frames of the trajectory (it
silently carries a large-dt transition). require_consecutive (default True)
excludes such windows at the definition of a window."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.encoder import Encoder
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from utils import load_datasets as load
from training.datasets import MicrostructureEvolutionDataset

SIZE = 32


@pytest.fixture
def two_stream_encoder():
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.DECODER),
    }
    encoder = Encoder(input_size=SIZE, in_channels=1, base_channels=4,
                       stream_configs=stream_configs)
    encoder.eval()
    return encoder


def _build_run(base_dir, name, steps, stdevs, dt=0.05):
    """A run with explicit save_steps and a per-step stdev_phi column, so a
    MIDDLE step can be dropped by min_stdev_phi to create a kept-steps gap."""
    run_dir = base_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.txt").write_text("\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {SIZE}", f"Ny = {SIZE}", f"dt = {dt}", f"steps = {steps[-1]}",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        "seed = 0", "equation = allen_cahn", "solver = explicit", "",
    ]))
    for step in steps:
        np.full((SIZE, SIZE), step / 10000.0, dtype="<f2").tofile(
            run_dir / load.snapshot_filename(step))
    pd.DataFrame([{"step": s, "stdev_phi": sd, "avg_phi": s / 1000.0}
                 for s, sd in zip(steps, stdevs)]).to_csv(
        run_dir / "statistics.csv", index=False)
    (run_dir / "COMPLETE").touch()
    return run_dir


def _dataset(run_dirs, encoder, window_length, require_consecutive,
             min_stdev_phi=None):
    return MicrostructureEvolutionDataset(
        run_dirs, encoder=encoder, device=torch.device("cpu"),
        window_length=window_length, min_step=0, encode_both_streams=True,
        min_stdev_phi=min_stdev_phi, require_consecutive=require_consecutive)


# save_steps = 5 frames; the MIDDLE one (2000) is quiet and gets filtered.
STEPS = [0, 1000, 2000, 3000, 4000]
STDEVS = [0.5, 0.5, 0.001, 0.5, 0.5]   # only step 2000 is below a 0.1 threshold


def test_no_filter_consecutive_is_a_no_op(tmp_path, two_stream_encoder):
    """With no step dropped, every kept pair is already save-adjacent, so
    require_consecutive changes nothing."""
    run = _build_run(tmp_path / "32x32", "T800_n010_s0", STEPS, STDEVS)
    assert len(_dataset([run], two_stream_encoder, 2, True)) == 4   # 5 steps -> 4 pairs
    assert len(_dataset([run], two_stream_encoder, 2, False)) == 4


def test_consecutive_drops_the_window_that_jumps_the_filtered_step(tmp_path, two_stream_encoder):
    """min_stdev_phi=0.1 drops step 2000 -> kept [0,1000,3000,4000].
    Pairs: (0,1000) ok, (1000,3000) JUMPS over 2000, (3000,4000) ok."""
    run = _build_run(tmp_path / "32x32", "T800_n010_s0", STEPS, STDEVS)
    keep = _dataset([run], two_stream_encoder, 2, True, min_stdev_phi=0.1)
    drop_off = _dataset([run], two_stream_encoder, 2, False, min_stdev_phi=0.1)
    assert len(drop_off) == 3, "old behaviour keeps all 3 consecutive-kept pairs"
    assert len(keep) == 2, "require_consecutive drops the (1000,3000) jump window"


def test_consecutive_length3_drops_every_window_spanning_the_gap(tmp_path, two_stream_encoder):
    """At window_length=3, kept [0,1000,3000,4000] has NO run of 3 consecutive
    saved steps (the gap at 2000 breaks both), so require_consecutive keeps 0."""
    run = _build_run(tmp_path / "32x32", "T800_n010_s0", STEPS, STDEVS)
    assert len(_dataset([run], two_stream_encoder, 3, True, min_stdev_phi=0.1)) == 0
    assert len(_dataset([run], two_stream_encoder, 3, False, min_stdev_phi=0.1)) == 2
