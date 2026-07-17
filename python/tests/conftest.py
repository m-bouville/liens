"""
Shared pytest fixtures. Deliberately avoid needing a real trained
checkpoint anywhere -- these are small, deterministic stand-ins whose
only job is to have the right SHAPE and be cheap to run, so tests stay
fast and don't depend on any particular training run having happened.
"""
import sys
from pathlib import Path

# Every module in this project (training/, models/, utils/) is meant to
# be imported with python/ itself on sys.path -- true when running e.g.
# `python -m training.train_ae` from python/, but NOT automatic for
# pytest, which by default only adds tests/'s own directory. Without
# this, `from training.losses import RolloutLoss` fails with
# ModuleNotFoundError regardless of which directory pytest is invoked
# from. conftest.py is always imported before any test file, so this
# runs early enough regardless of pytest version.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import torch
import torch.nn as nn
import pytest

from models.constants import LATENT_SPATIAL_SIZE
from models.latent_streams import DEFAULT_STREAM_NAME


class FakeEncoder(nn.Module):
    """
    Maps (B, 1, size, size) -> {DEFAULT_STREAM_NAME: (B, latent_channels,
    LATENT_SPATIAL_SIZE, LATENT_SPATIAL_SIZE)} with a single strided
    conv -- not meant to encode anything meaningful, just to be a real
    nn.Module with the right input/output shape and (crucially) real,
    trainable parameters, so gradient-flow tests are genuine. Uses the
    SAME shared constant the real Encoder derives its bottleneck size
    from (models.constants.LATENT_SPATIAL_SIZE), not an independently
    hardcoded 8 -- so this fixture can't silently drift from the real
    architecture's actual bottleneck size the way it previously could.

    Returns a dict (matching the real, current Encoder's multi-stream
    return shape), NOT a bare tensor -- an earlier version of this
    fixture returned a bare tensor, from before the multi-stream (C0/
    C1) redesign, and every test using it kept passing even after
    Encoder itself started returning a dict, because the fixture never
    stopped reproducing the OLD interface. That's exactly how
    training/datasets.py's own encoder(batch).cpu() call broke in real
    use (a bare dict has no .cpu()) without any test catching it first
    -- the fixture wasn't faithful to what it was standing in for.
    """
    def __init__(self, size: int = 64, latent_channels: int = 4):
        super().__init__()
        stride = size // LATENT_SPATIAL_SIZE
        assert stride * LATENT_SPATIAL_SIZE == size, \
            "FakeEncoder assumes size is a multiple of LATENT_SPATIAL_SIZE"
        self.conv = nn.Conv2d(1, latent_channels, kernel_size=stride, stride=stride)

    def forward(self, x):
        return {DEFAULT_STREAM_NAME: self.conv(x)}


@pytest.fixture
def fake_encoder():
    return FakeEncoder(size=64, latent_channels=4)


@pytest.fixture
def tmp_run_dir(tmp_path):
    """
    A single fake run directory with a metadata.txt and real
    binary snapshot files, in the ACTUAL format load.read_metadata/
    read_phi_half/snapshot_filename expect (verified against their
    source, not assumed) -- so dataset tests exercise the real
    file-reading code path, not a mocked stand-in of it.
    """
    from utils import load_datasets as load

    run_dir = tmp_path / "T800_n010_s1"
    run_dir.mkdir()

    steps = [0, 1000, 2000, 3000, 4000]
    size = 64
    metadata_text = "\n".join([
        "directory = T800_n010_s1",
        "code version = test",
        "status = complete",
        f"Nx = {size}",
        f"Ny = {size}",
        "dt = 0.05",
        "steps = 4000",
        f"save_steps = {' '.join(str(s) for s in steps)}",  # whitespace-separated, not comma
        "a0 = 1.0",
        "b = 1.0",
        "T0 = 1.0",
        "temperature = 0.8",
        "kappa = 0.2",
        "mobility = 0.05",
        "phi0 = 0.0",
        "noise = 0.01",
        "seed = 1",
        "equation = allen_cahn",
        "solver = explicit",
        "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)

    import numpy as np
    for step in steps:
        # Distinctive, checkable per-step value (constant field = step/10000),
        # written as raw little-endian float16 -- the actual on-disk format
        # (see read_phi_half's docstring), NOT a generic numpy .npy/.tofile
        # dump in some other dtype.
        arr = np.full((size, size), step / 10000.0, dtype="<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))

    return run_dir, steps


STAT_NAMES = ["angle", "avg_phi", "stdev_phi"]


@pytest.fixture
def tmp_run_dir_with_stats(tmp_run_dir):
    """
    Layers a real statistics.csv onto tmp_run_dir's run directory --
    separate fixture (not folded into tmp_run_dir itself) since most
    dataset tests don't need statistics.csv at all, and building it
    needlessly would just be dead weight for those.

    Values are deterministic and distinctive per step (stat value =
    step/1000 for every column) so tests can check WHICH row ended up
    associated with a given sample, the same way tmp_run_dir's own
    snapshot values are. One step (2000) is deliberately given a NaN in
    one column, specifically to test the NaN-guard that's supposed to
    exclude any window starting there.
    """
    import pandas as pd

    run_dir, steps = tmp_run_dir
    rows = []
    for step in steps:
        row = {"step": step}
        for name in STAT_NAMES:
            row[name] = step / 1000.0
        rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[df["step"] == 2000, "avg_phi"] = float("nan")  # deliberate NaN, for the guard test
    df.to_csv(run_dir / "statistics.csv", index=False)

    return run_dir, steps, STAT_NAMES


@pytest.fixture
def isolated_project_root(tmp_path, monkeypatch):
    """
    For any test that calls run_from_params_file() or another
    pipeline-level entry point WITHOUT explicitly overriding every
    checkpoint_path/loss_curve_path/output_path itself: every module
    in this project independently anchors its own default output
    locations to _PYTHON_ROOT = Path(__file__).resolve().parent.parent
    (see test_path_policy.py's own docstring on why this anchoring
    exists at all -- it replaced an earlier, CWD-relative-path bug).
    That's the CORRECT behavior for real use, but it means any test
    that doesn't override these defaults writes directly into this
    project's own real checkpoints/ and output/ directories -- not a
    tmp_path, an actually-active working directory the person using
    this project sees and has to clean up by hand.

    Redirects every such module's own _PYTHON_ROOT (and any OTHER
    module-level constant derived from it at import time, which won't
    automatically follow a patched _PYTHON_ROOT on its own) to a fresh
    tmp_path instead, for the duration of one test. Covers every
    module actually involved in a full stage 1->1b->2->3a->3b run
    (train_ae.py/train_lds.py/train_refinement.py/orchestration.paths/
    orchestration.pipeline) -- NOT the individual evaluation/*.py
    scripts, which each have their own, separate _PYTHON_ROOT too, but
    only ever use it for CLI argument defaults (argparse), never when
    called programmatically with an explicit checkpoint_path/
    output_path the way the pipeline itself always does internally.

    orchestration.pipeline specifically imports _PYTHON_ROOT/
    _STAGE_DIRS via `from orchestration.paths import ...` -- a
    same-name BINDING in its own namespace, not a live reference back
    to orchestration.paths's own copy, so patching paths.py's copy
    alone would NOT affect what pipeline.py itself actually reads.
    Both need patching separately, or this fixture would silently miss
    exactly the module that matters most (the one run_from_params_file
    itself lives in).
    """
    root = tmp_path / "isolated_project_root"
    (root / "checkpoints").mkdir(parents=True)
    (root.parent / "output").mkdir(parents=True, exist_ok=True)

    stage_dirs = {1: root / "checkpoints" / "stage1",
                  "1b": root / "checkpoints" / "stage1b",
                  2: root / "checkpoints" / "stage2",
                  3: root / "checkpoints" / "stage3",
                  "3a": root / "checkpoints" / "stage3a",
                  "3b": root / "checkpoints" / "stage3b",
                  4: root / "checkpoints" / "stage4",
                  5: root / "checkpoints" / "stage5"}

    import training.train_ae as train_ae
    import training.train_lds as train_lds
    import training.train_refinement as train_refinement
    import orchestration.paths as orch_paths
    import orchestration.pipeline as orch_pipeline

    for module in (train_ae, train_lds, train_refinement, orch_paths, orch_pipeline):
        monkeypatch.setattr(module, "_PYTHON_ROOT", root, raising=True)
    for module in (orch_paths, orch_pipeline):
        monkeypatch.setattr(module, "_STAGE_DIRS", stage_dirs, raising=True)
    monkeypatch.setattr(orch_paths, "_CHECKPOINTS_ROOT", root / "checkpoints", raising=True)

    return root
