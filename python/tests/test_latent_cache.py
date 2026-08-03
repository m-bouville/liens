"""
Latent cache for the frozen stage-3 encoder, and the max_dt truncation that
shrinks what has to be encoded at all.

Stages 3a and 3b load the SAME stage-2 checkpoint as their frozen encoder and
both encode the same runs; each stage-3 diagnostic then encodes the test set
again. The latents are a pure function of (encoder weights, run_dir, steps),
and stage 3 freezes the encoder by definition.

The hazard is a STALE HIT: training would proceed on latents from a different
encoder, with no shape error and no warning. Hence a weight hash rather than a
checkpoint path -- paths get reused (force=True overwrites in place) and mtimes
survive a copy.
"""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_train_lds import _build_sweep  # noqa: E402

from training.datasets import MicrostructureEvolutionDataset, _truncate_to_max_dt  # noqa: E402
from training.latent_cache import encoder_fingerprint  # noqa: E402


class _Meta:
    def __init__(self, dt):
        self.dt = dt


class _TinyEncoder(torch.nn.Module):
    def __init__(self, seed=0, channels=4, spatial=8):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = torch.nn.Conv2d(1, channels, 3, padding=1)
        self.spatial = spatial

    def forward(self, x, theta=None):
        # A DICT keyed by stream name, matching the real multi-stream Encoder.
        # An earlier version of this stub returned a bare tensor and every
        # end-to-end test here failed with "too many indices" from inside the
        # dataset -- the stub was wrong, not the cache.
        from models.latent_streams import DEFAULT_STREAM_NAME
        z = torch.nn.functional.adaptive_avg_pool2d(self.conv(x), self.spatial)
        return {DEFAULT_STREAM_NAME: z, "deriv": z * 0.5}


# --------------------------------------------------------------------
# max_dt truncation
# --------------------------------------------------------------------

def test_truncation_stops_at_the_first_oversized_transition():
    steps = [0, 100, 200, 400, 900, 2000]        # gaps 100,100,200,500,1100
    kept = _truncate_to_max_dt(steps, _Meta(dt=1.0), max_dt=200, window_length=2)
    assert kept == [0, 100, 200, 400], kept


def test_truncation_is_inclusive_at_exactly_max_dt():
    """
    Matches the window filter, which skips on `dt > max_dt`. An off-by-one here
    would silently drop every window at exactly the boundary -- and 125 is a
    real, common gap in this project's schedule.
    """
    steps = [0, 100, 200]                        # both gaps exactly 100
    assert _truncate_to_max_dt(steps, _Meta(dt=1.0), max_dt=100, window_length=2) == steps


def test_truncation_is_a_no_op_without_max_dt():
    steps = [0, 100, 5000]
    assert _truncate_to_max_dt(steps, _Meta(dt=1.0), max_dt=None, window_length=2) == steps


def test_truncation_survives_non_monotonic_gaps_conservatively():
    """
    kept_steps is the schedule AFTER min_stdev_phi filtering, and dropping an
    interior step merges two gaps into a larger one -- so monotonicity is not
    guaranteed. Stopping at the first oversized transition is correct either
    way (a later window would have to cross it) and merely conservative if a
    short gap follows: it never keeps an invalid window.
    """
    steps = [0, 100, 5000, 5100]                 # gaps 100, 4900, 100
    kept = _truncate_to_max_dt(steps, _Meta(dt=1.0), max_dt=200, window_length=2)
    assert kept == [0, 100], kept


# --------------------------------------------------------------------
# the fingerprint
# --------------------------------------------------------------------

def test_identical_weights_give_the_same_fingerprint():
    assert encoder_fingerprint(_TinyEncoder(seed=1)) == encoder_fingerprint(_TinyEncoder(seed=1))


def test_different_weights_give_different_fingerprints():
    assert encoder_fingerprint(_TinyEncoder(seed=1)) != encoder_fingerprint(_TinyEncoder(seed=2))


def test_a_changed_BUFFER_changes_the_fingerprint():
    """
    GUARDS hashing parameters only. BatchNorm running_mean/running_var change
    what the encoder OUTPUTS while leaving every parameter untouched -- which
    is exactly the state a re-estimated port is in, and exactly the case where
    reusing cached latents would be wrong.
    """
    encoder = torch.nn.BatchNorm2d(3)
    before = encoder_fingerprint(encoder)
    encoder.running_mean += 1.0
    assert encoder_fingerprint(encoder) != before


# --------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------

def _dataset(run_dirs, encoder, cache_dir, **kw):
    return MicrostructureEvolutionDataset(
        run_dirs, encoder=encoder, device="cpu", window_length=2,
        min_step=0, min_stdev_phi=None, latent_cache_dir=cache_dir, **kw)


@pytest.fixture
def sweep(tmp_path):
    base = _build_sweep(tmp_path, n_runs=4, size=32)
    from training.datasets import complete_run_dirs
    return complete_run_dirs(base, 32, 32)


def test_a_cache_hit_reproduces_the_latents_exactly(sweep, tmp_path, capsys):
    """
    THE test. Not "the cache is used" but "using it changes nothing": every
    window's latents must be bit-identical to the uncached build.
    """
    cache = tmp_path / "latents"
    encoder = _TinyEncoder(seed=3)
    cold = _dataset(sweep, encoder, cache)
    capsys.readouterr()
    warm = _dataset(sweep, encoder, cache)
    assert "read from the latent cache" in capsys.readouterr().out

    assert len(cold) == len(warm) > 0
    for i in range(len(cold)):
        a, b = cold[i], warm[i]
        for t_cold, t_warm in zip(a, b):
            if isinstance(t_cold, torch.Tensor):
                assert torch.equal(t_cold, t_warm), f"window {i} differs"


def test_a_different_encoder_does_not_hit(sweep, tmp_path, capsys):
    """
    GUARDS keying on anything but the weights. A hit here would train stage 3
    on latents from a DIFFERENT encoder -- same shapes, no error, every
    downstream number wrong.
    """
    cache = tmp_path / "latents"
    _dataset(sweep, _TinyEncoder(seed=3), cache)
    capsys.readouterr()
    _dataset(sweep, _TinyEncoder(seed=99), cache)
    assert "read from the latent cache" not in capsys.readouterr().out


def test_a_different_step_list_does_not_hit(sweep, tmp_path, capsys):
    """
    The encoded prefix depends on max_dt, so a SHORTER cached prefix must not
    satisfy a request for a longer one.
    """
    cache = tmp_path / "latents"
    encoder = _TinyEncoder(seed=3)
    _dataset(sweep, encoder, cache, max_dt=30)
    capsys.readouterr()
    _dataset(sweep, encoder, cache)          # no max_dt: needs more steps
    assert "read from the latent cache" not in capsys.readouterr().out


def test_no_cache_dir_means_no_caching(sweep, tmp_path, capsys):
    encoder = _TinyEncoder(seed=3)
    MicrostructureEvolutionDataset(sweep, encoder=encoder, device="cpu", window_length=2,
                                    min_step=0, min_stdev_phi=None)
    capsys.readouterr()
    MicrostructureEvolutionDataset(sweep, encoder=encoder, device="cpu", window_length=2,
                                    min_step=0, min_stdev_phi=None)
    assert "latent cache" not in capsys.readouterr().out


def test_a_corrupt_cache_entry_falls_back_to_recomputing(sweep, tmp_path):
    """
    A cache that can BREAK the run it is meant to speed up is worse than no
    cache. A half-written or truncated file must degrade to a recompute.
    """
    cache = tmp_path / "latents"
    encoder = _TinyEncoder(seed=3)
    reference = _dataset(sweep, encoder, cache)
    for entry in cache.rglob("*.pt"):
        entry.write_bytes(b"not a torch file")
    recovered = _dataset(sweep, encoder, cache)
    assert len(recovered) == len(reference)
    assert torch.equal(recovered[0][0], reference[0][0])


def test_each_run_is_paired_with_its_OWN_latents(sweep, tmp_path):
    """
    GUARDS the index-keyed assembly. A cache hit skips the encode buffer, so
    if the per-run results were still collected in FLUSH order a hit would
    land wherever the next flush happened to reach -- pairing one run's
    latents with another run's steps. Same shapes, no error.

    Checked against ground truth computed OUTSIDE the dataset: comparing a
    cached build to an uncached one does not catch this, because both go
    through the same assembly and a misordering breaks them identically.
    Verified: reversing the assembly order left all other tests here green.
    """
    from models.latent_streams import DEFAULT_STREAM_NAME
    import utils.load_datasets as load

    cache = tmp_path / "latents"
    encoder = _TinyEncoder(seed=7)
    dataset = _dataset(sweep, encoder, cache)          # cold: writes the cache
    warm = _dataset(sweep, encoder, cache)             # hot: reads it back

    for ds, label in ((dataset, "cold"), (warm, "warm")):
        for run_idx, run_dir in enumerate(ds._run_dirs):
            kept_steps = ds._run_steps[run_idx]
            metadata = load.read_metadata(run_dir / "metadata.txt")
            frames = torch.stack([
                torch.from_numpy(load.read_phi_half(
                    run_dir / load.snapshot_filename(step), metadata.nx, metadata.ny)).unsqueeze(0)
                for step in kept_steps
            ])
            with torch.no_grad():
                expected = encoder(frames)[DEFAULT_STREAM_NAME]
            assert torch.allclose(ds._run_data[run_idx], expected, atol=1e-6), (
                f"{label} build: run {run_idx} ({run_dir.name}) is paired with "
                f"another run's latents"
            )


def test_cached_latents_are_on_the_SAME_DEVICE_as_freshly_encoded_ones(sweep, tmp_path):
    """
    REGRESSION. load_cached originally took a `device` argument and honoured
    it, while the encode path it substitutes for ends in `.cpu()`. A dataset
    whose runs were a MIXTURE then failed much later, inside a DataLoader
    worker's collate:

        RuntimeError: Expected all tensors to be on the same device, but
        found at least two devices, cuda:0 and cpu!

    pointing at torch.stack rather than at the cache. Every other test here
    passed throughout -- none of them looked at `.device`, and on a CPU-only
    machine the two agree by accident.

    Compares hits against misses IN THE SAME dataset, which is the situation
    that actually breaks: a partially-populated cache.
    """
    cache = tmp_path / "latents"
    encoder = _TinyEncoder(seed=11)
    full = _dataset(sweep, encoder, cache)             # populate for all runs
    entries = sorted(cache.rglob("*.pt"))
    assert len(entries) >= 2, "need several runs to make a partial cache"
    entries[0].unlink()                                # force ONE run to re-encode

    mixed = _dataset(sweep, encoder, cache)
    devices = {t.device.type for t in mixed._run_data}
    assert len(devices) == 1, (
        f"cache hits and fresh encodes disagree on device: {devices} -- collate "
        f"will fail in a DataLoader worker, far from here"
    )
    assert devices == {"cpu"}, devices
    assert {t.device.type for t in full._run_data} == {"cpu"}
