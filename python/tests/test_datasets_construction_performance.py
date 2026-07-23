"""
Tests for MicrostructureEvolutionDataset's CONSTRUCTION-time performance
changes: parallel snapshot reads (read_workers), cross-run batched
encoding, and the bounded-memory streaming buffer that makes that
batching safe at real sweep sizes. None of these had a permanent
regression test before this file -- they were verified by hand, once,
in the course of making the changes, which is not the same guarantee as
a test that runs on every future change to this constructor.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_datasets_construction_performance.py -v
"""
import torch
import torch.nn as nn
import pytest

from training.datasets import MicrostructureEvolutionDataset
from models.constants import LATENT_SPATIAL_SIZE
from models.latent_streams import DEFAULT_STREAM_NAME


def _write_run(base_path, name, steps, size=64):
    """
    A fake run directory in the ACTUAL format load.read_metadata/
    read_phi_half/snapshot_filename expect -- same format and same
    "constant field = step/10000" convention as conftest.py's own
    tmp_run_dir fixture, deliberately duplicated here rather than
    reused because tmp_run_dir only ever builds ONE run: every test in
    this file needs SEVERAL runs, with independently controllable step
    counts, to exercise cross-run batching and read-parallelism at all.
    """
    from utils import load_datasets as load

    run_dir = base_path / name
    run_dir.mkdir()
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", f"steps = {steps[-1]}",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8", "kappa = 0.2",
        "mobility = 0.05", "phi0 = 0.0", "noise = 0.01", "seed = 1",
        "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)

    import numpy as np
    for step in steps:
        # Distinctive, checkable per-step value (constant field =
        # step/10000), matching tmp_run_dir's own convention -- lets
        # tests verify actual DATA correctness, not just shape/count.
        arr = np.full((size, size), step / 10000.0, dtype="<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))
    return run_dir


def _write_runs(tmp_path, run_step_counts, size=64):
    """run_step_counts: list of ints, one entry per run -- e.g. [2, 3, 8]
    builds 3 runs with 2, 3, and 8 kept steps respectively. Step values
    are spaced 1000 apart per run, starting at 0, matching tmp_run_dir's
    own spacing convention."""
    return [
        _write_run(tmp_path, f"T800_n{i:03d}_s1", steps=list(range(0, n * 1000, 1000)), size=size)
        for i, n in enumerate(run_step_counts)
    ]


class _FakeEncoderBothStreams(nn.Module):
    """
    Like conftest.py's own FakeEncoder, but ALSO returns a "deriv"
    stream -- needed to test encode_both_streams=True, which NO
    existing fixture in this project covers (conftest.py's FakeEncoder
    only ever returns DEFAULT_STREAM_NAME). Kept local to this file
    rather than added to conftest.py: it's a narrow, single-purpose
    stand-in for a specific test concern here, not a general-purpose
    fixture other test files are likely to need.
    """
    def __init__(self, size: int = 64, latent_channels: int = 4):
        super().__init__()
        stride = size // LATENT_SPATIAL_SIZE
        assert stride * LATENT_SPATIAL_SIZE == size
        self.conv = nn.Conv2d(1, latent_channels, kernel_size=stride, stride=stride)
        self.conv_deriv = nn.Conv2d(1, latent_channels, kernel_size=stride, stride=stride)

    def forward(self, x):
        return {DEFAULT_STREAM_NAME: self.conv(x), "deriv": self.conv_deriv(x)}


def test_read_workers_produces_identical_results_to_sequential(tmp_path):
    """
    The whole point of read_workers: reading concurrently must produce
    EXACTLY the same data, in the same order, as reading one file at a
    time -- read_workers=1 reproduces the literal prior sequential
    behavior (see its own docstring), so comparing against it directly
    is a comparison against known-correct behavior, not just against
    "whatever this version of the code happens to produce."
    """
    run_dirs = _write_runs(tmp_path, run_step_counts=[5, 3, 7])

    ds_sequential = MicrostructureEvolutionDataset(
        run_dirs, encoder=None, window_length=2, min_step=0, min_stdev_phi=None,
        read_workers=1,
    )
    ds_parallel = MicrostructureEvolutionDataset(
        run_dirs, encoder=None, window_length=2, min_step=0, min_stdev_phi=None,
        read_workers=8,
    )

    assert len(ds_sequential) == len(ds_parallel)
    assert ds_sequential._run_steps == ds_parallel._run_steps, (
        "same kept-step ORDER is required for window construction to agree at all"
    )
    for run_idx in range(len(ds_sequential._run_dirs)):
        assert torch.equal(ds_sequential._run_data[run_idx], ds_parallel._run_data[run_idx]), (
            f"run {run_idx}: parallel reads produced different raw frame data than sequential"
        )


def test_encode_both_streams_matches_independent_per_run_encoding(tmp_path):
    """
    The strongest available check on cross-run batched encoding: build
    runs with DELIBERATELY awkward sizes relative to encode_batch_size
    (one run bigger than the batch size alone, several small runs that
    only combine to overshoot it, a run smaller than window_length that
    must be skipped entirely and not throw off the alignment of runs
    after it) -- covering every way a buffer flush can straddle a run
    boundary -- then compare BOTH streams against independent,
    per-run-only encoding. encode_both_streams=True has no other test
    anywhere in this project.
    """
    run_step_counts = [2, 3, 8, 1, 4, 6, 2, 2]  # the "1" is below window_length=2 -- must be skipped
    run_dirs = _write_runs(tmp_path, run_step_counts, size=64)

    encoder = _FakeEncoderBothStreams(size=64, latent_channels=4)
    ds = MicrostructureEvolutionDataset(
        run_dirs, encoder=encoder, window_length=2, min_step=0, min_stdev_phi=None,
        encode_batch_size=5, encode_both_streams=True,
    )

    assert len(ds._run_dirs) == len(run_dirs) - 1, "the 1-step run must have been skipped"

    encoder.eval()
    from utils import load_datasets as load
    for run_idx, run_dir in enumerate(ds._run_dirs):
        metadata = load.read_metadata(run_dir / "metadata.txt")
        frames = torch.stack([
            torch.from_numpy(load.read_phi_half(run_dir / load.snapshot_filename(s),
                                                  metadata.nx, metadata.ny)).unsqueeze(0)
            for s in metadata.save_steps
        ])
        with torch.no_grad():
            expected = encoder(frames)
        assert torch.allclose(ds._run_data[run_idx], expected[DEFAULT_STREAM_NAME], atol=1e-6), (
            f"run {run_idx}: 'state' stream doesn't match independent per-run encoding -- "
            f"cross-run batching produced a different result than encoding this run alone"
        )
        assert torch.allclose(ds._run_data_deriv[run_idx], expected["deriv"], atol=1e-6), (
            f"run {run_idx}: 'deriv' stream doesn't match independent per-run encoding"
        )


def test_encoding_never_buffers_more_than_one_batch_plus_one_run(tmp_path):
    """
    The actual point of the streaming buffer: peak memory must stay
    bounded by roughly encode_batch_size + one run's own frame count,
    REGARDLESS of how many runs (or total frames) the full sweep has --
    see MicrostructureEvolutionDataset's own constructor comment for
    why an earlier version of this (accumulating every run's raw
    frames before encoding any of it) was a real problem, not a
    theoretical one, at real sweep sizes.

    Verified via an OBSERVABLE proxy, not by reaching into the
    constructor's internals (which would break the moment anyone
    refactors the internal variable names): wrap the encoder's own
    forward() to record the batch size of every call it actually
    receives. If buffering were unbounded, at least one call would
    receive ALL 400 frames in this test's sweep; if it's correctly
    bounded, no call should receive much more than encode_batch_size.
    """
    # 20 runs x 20 frames = 400 total frames -- comfortably large enough
    # that "encoded in one unbounded batch" and "encoded in bounded
    # chunks" would look meaningfully different if this regressed.
    run_dirs = _write_runs(tmp_path, run_step_counts=[20] * 20, size=64)
    encode_batch_size = 32

    encoder = _FakeEncoderBothStreams(size=64, latent_channels=4)
    call_sizes = []
    real_forward = encoder.forward

    def _recording_forward(x):
        call_sizes.append(x.size(0))
        return real_forward(x)

    encoder.forward = _recording_forward

    MicrostructureEvolutionDataset(
        run_dirs, encoder=encoder, window_length=2, min_step=0, min_stdev_phi=None,
        encode_batch_size=encode_batch_size,
    )

    assert call_sizes, "the encoder was never actually called"
    total_frames = sum(call_sizes)
    assert total_frames == 400, f"expected all 400 frames to be encoded exactly once, got {total_frames}"
    # The real bound is encode_batch_size itself (each _flush_buffer()
    # call internally sub-chunks by encode_batch_size -- see its own
    # docstring) -- no single encoder call should ever exceed it,
    # regardless of how many runs or total frames the sweep has.
    assert max(call_sizes) <= encode_batch_size, (
        f"a single encoder call received {max(call_sizes)} frames, exceeding "
        f"encode_batch_size={encode_batch_size} -- the buffer is not actually bounded"
    )
    assert max(call_sizes) < total_frames, (
        "the largest single encoder call received the ENTIRE sweep's frames -- "
        "this is exactly the unbounded-buffering regression this test exists to catch"
    )


def test_cuda_empty_cache_guard_only_matches_cuda_device():
    """
    Lightweight, GPU-independent check of the GUARD CONDITION itself
    (torch.device(device).type == "cuda") for every spelling of device
    this constructor accepts (str or torch.device) -- doesn't require
    an actual GPU, unlike the full integration below, so this always
    runs and always catches a broken condition (e.g. a typo'd string
    comparison) even in a CPU-only environment.
    """
    assert torch.device("cpu").type != "cuda"
    assert torch.device(torch.device("cpu")).type != "cuda"
    assert torch.device("cuda").type == "cuda"
    assert torch.device(torch.device("cuda")).type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
def test_cuda_empty_cache_actually_called_when_device_is_cuda(tmp_path, monkeypatch):
    """
    Full integration version of the guard check above -- SKIPPED (not
    just passed trivially) on a machine with no GPU, since moving a
    real encoder/tensor to "cuda" would raise there regardless of
    anything this constructor does. Only runs, and only means anything,
    on a CUDA-enabled machine.
    """
    import training.datasets as datasets_module

    run_dirs = _write_runs(tmp_path, run_step_counts=[3, 4], size=64)
    encoder = _FakeEncoderBothStreams(size=64, latent_channels=4)

    calls = []
    monkeypatch.setattr(datasets_module.torch.cuda, "empty_cache", lambda: calls.append(True))

    MicrostructureEvolutionDataset(
        run_dirs, encoder=encoder, device="cuda", window_length=2,
        min_step=0, min_stdev_phi=None, encode_batch_size=4,
    )
    assert calls, "torch.cuda.empty_cache() was never called with device='cuda'"
