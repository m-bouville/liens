"""
Tests for MicrostructureEvolutionDataset's own max_dt filter.

Motivation, from check_parameter_dependence.py's own oracle-z1
attribution on real 64x64 data: z1 BEATS a causal backward difference
below dt ~= 150 (x1.47) and loses badly above it (x0.08 by dt~3e3,
x0.001 by dt~3e4), with ~48% of test windows past that horizon. There
the first-order (z0 + z1*dt) term is already off by ~x60, so f_theta --
which only ever ADDS a dt^2/2 correction on top of it -- cannot recover
the prediction. max_dt excludes those windows so stage 3 trains only
where f_theta can actually help.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import pytest

from models.encoder import Encoder
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from training.datasets import MicrostructureEvolutionDataset
from utils import load_datasets as load


SIZE = 32


@pytest.fixture
def two_stream_encoder():
    """conftest's own `fake_encoder` yields a SINGLE stream, but these
    datasets are built with encode_both_streams=True (matching stage 3's
    own usage), which needs a real "deriv" stream too."""
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


def _build_run(base_dir: Path, name: str, steps: list[int], dt: float = 0.05) -> Path:
    """One run whose save_steps are given explicitly, so per-transition
    dt can be made deliberately UNEVEN -- which is what the
    any-transition rule below actually needs to be tested against."""
    run_dir = base_dir / name
    run_dir.mkdir(parents=True)
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {SIZE}", f"Ny = {SIZE}", f"dt = {dt}", f"steps = {steps[-1]}",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        "seed = 0", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    for step in steps:
        np.full((SIZE, SIZE), step / 10000.0, dtype="<f2").tofile(
            run_dir / load.snapshot_filename(step))
    pd.DataFrame([{"step": s, "avg_phi": s / 1000.0} for s in steps]).to_csv(
        run_dir / "statistics.csv", index=False)
    (run_dir / "COMPLETE").touch()
    return run_dir


def _dataset(run_dirs, encoder, window_length, max_dt):
    return MicrostructureEvolutionDataset(
        run_dirs, encoder=encoder, device=torch.device("cpu"),
        window_length=window_length, min_step=0, encode_both_streams=True,
        max_dt=max_dt,
    )


def test_max_dt_none_is_an_exact_no_op(tmp_path, two_stream_encoder):
    """The default must change nothing at all -- same window count as
    before this parameter existed."""
    run = _build_run(tmp_path / "32x32", "T800_n010_s0", [0, 1000, 2000, 3000])
    assert len(_dataset([run], two_stream_encoder, 2, None)) == 3


def test_max_dt_keeps_windows_at_or_below_the_cap_and_drops_those_above(tmp_path, two_stream_encoder):
    """Every transition here is dt = 1000 * 0.05 = 50."""
    run = _build_run(tmp_path / "32x32", "T800_n010_s0", [0, 1000, 2000, 3000])
    assert len(_dataset([run], two_stream_encoder, 2, 60.0)) == 3, "50 <= 60, all kept"
    assert len(_dataset([run], two_stream_encoder, 2, 50.0)) == 3, "50 <= 50, boundary is inclusive"
    assert len(_dataset([run], two_stream_encoder, 2, 40.0)) == 0, "50 > 40, all dropped"


def test_max_dt_drops_a_window_for_ANY_bad_transition_not_just_the_first(tmp_path, two_stream_encoder):
    """
    THE rule that matters for rollout windows: steps [0, 1000, 2000,
    10000] give per-transition dt of [50, 50, 400]. At window_length=3
    the two windows span transitions (50, 50) and (50, 400).

    max_dt=100 must keep only the first. The second's FIRST transition
    is a perfectly fine 50 -- checking only that one (the way
    min_std_deriv does) would wrongly keep it. A rollout window is only
    as usable as its worst single step: one transition past the horizon
    puts the whole chained prediction off-distribution from there on.
    """
    run = _build_run(tmp_path / "32x32", "T800_n010_s0", [0, 1000, 2000, 10000])
    assert len(_dataset([run], two_stream_encoder, 3, None)) == 2
    assert len(_dataset([run], two_stream_encoder, 3, 500.0)) == 2, "400 <= 500, both kept"
    assert len(_dataset([run], two_stream_encoder, 3, 100.0)) == 1, (
        "the (50, 400) window must be dropped despite its first transition being fine"
    )


def test_max_dt_round_trips_through_train_lds_data_config(tmp_path):
    """max_dt must be SAVED, so evaluation reproduces the same filtering
    the checkpoint was actually trained under -- otherwise a diagnostic
    silently evaluates on windows training never saw."""
    import inspect
    from training import train_lds as train_lds_module

    src = inspect.getsource(train_lds_module.train_lds)
    assert '"max_dt": max_dt' in src, (
        "max_dt is not written into the saved data_config -- evaluation would fall back "
        "to unfiltered windows without any indication of the mismatch"
    )
