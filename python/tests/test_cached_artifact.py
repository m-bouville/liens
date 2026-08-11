"""
Tests for conftest's cached_artifact / copy_cached_files.

These exist because the cache trades correctness risk for speed, and the risk
is SILENT: a key too coarse hands a test an artifact built with different
settings, and the test keeps passing while no longer testing what it claims.
A shared mutable artifact corrupts later tests in a way that depends on
execution order and disappears under `-k`. Both are worse than the slowness
the cache removes, so both are asserted directly.
"""

import pytest
import torch

from conftest import cached_artifact, copy_cached_files


def test_the_builder_runs_once_per_distinct_key():
    calls = []

    def builder(d):
        calls.append(d)
        path = d / "artifact.txt"
        path.write_text("built")
        return path

    key_a = ("t_once", "config-a")
    first = cached_artifact(key_a, builder)
    second = cached_artifact(key_a, builder)
    assert first == second
    assert len(calls) == 1, "same key must not rebuild"

    cached_artifact(("t_once", "config-b"), builder)
    assert len(calls) == 2, "a different key MUST rebuild"


def test_different_keys_get_different_directories():
    """GUARDS reusing one cache directory, which would make the second build
    overwrite the first and hand both keys the same artifact."""
    def builder(d):
        path = d / "a.txt"
        path.write_text(str(d))
        return path

    a = cached_artifact(("t_dirs", 1), builder)
    b = cached_artifact(("t_dirs", 2), builder)
    assert a.parent != b.parent
    assert a.read_text() != b.read_text()


def test_each_caller_gets_its_own_writable_copy(tmp_path):
    """
    THE correctness guard. cached_sweep is safe because sweeps are read-only;
    checkpoints are not. A test that resumes from an ancestor and saves in
    place must not corrupt it for tests scheduled later -- a failure that
    would depend on execution order and vanish under `-k`.
    """
    def builder(d):
        path = d / "ckpt.pt"
        path.write_text("original")
        return path

    cached = cached_artifact(("t_copy", "cfg"), builder)
    mine = copy_cached_files(cached, tmp_path / "a")
    yours = copy_cached_files(cached, tmp_path / "b")

    assert mine != cached and yours != cached
    assert mine != yours
    mine.write_text("i overwrote my copy")
    assert cached.read_text() == "original", "the cached artifact must be untouched"
    assert yours.read_text() == "original", "another test's copy must be untouched"


def test_directories_pass_through_uncopied(tmp_path):
    """
    A cached SWEEP is read-only and shared deliberately; copying it per test
    would throw away exactly the saving the sweep cache exists for. Only files
    are duplicated.
    """
    def builder(d):
        sweep = d / "sweep"
        sweep.mkdir()
        (sweep / "run").mkdir()
        ckpt = d / "ckpt.pt"
        ckpt.write_text("x")
        return sweep, ckpt

    cached_sweep_dir, cached_ckpt = cached_artifact(("t_passthrough", "cfg"), builder)
    got_dir, got_ckpt = copy_cached_files((cached_sweep_dir, cached_ckpt), tmp_path / "dest")
    assert got_dir == cached_sweep_dir, "the sweep directory must be shared, not copied"
    assert got_ckpt != cached_ckpt, "the checkpoint must be copied"
    assert got_ckpt.read_text() == "x"


def test_copy_preserves_the_container_shape():
    """A builder returning a tuple must get a tuple back -- callers unpack."""
    def builder(d):
        p = d / "one.pt"
        p.write_text("1")
        return (p,)

    cached = cached_artifact(("t_shape", "cfg"), builder)
    assert isinstance(cached, tuple)


@pytest.mark.parametrize("value", ["plain", 42, None])
def test_non_path_values_pass_through(tmp_path, value):
    def builder(d):
        return value

    assert copy_cached_files(cached_artifact(("t_scalar", value), builder),
                              tmp_path / "d") == value


def test_cached_paths_live_outside_any_single_test_tmp_path(tmp_path):
    """
    The whole point: the artifact must OUTLIVE the tmp_path of whichever test
    happened to build it first, or the second caller gets a dangling path.
    """
    def builder(d):
        p = d / "x.pt"
        p.write_text("x")
        return p

    cached = cached_artifact(("t_lifetime", "cfg"), builder)
    assert tmp_path not in cached.parents
    assert cached.exists()


def test_building_a_cached_artifact_does_not_disturb_the_callers_rng():
    """
    THE regression. A cache must be INVISIBLE: a caller reaches its own next
    step in the same RNG state whether it built the artifact or got a hit.
    Building trains a model, which advances torch's global RNG, so without a
    save/restore only the FIRST caller pays that advance.

    This broke test_deriv_target_centered_resume_skips_completed_warmup --
    two stage-2 trainings on 12 windows, sensitive to the RNG. It failed only
    under xdist, only on one worker, and only after the caching refactor,
    surfacing as FileNotFoundError on a checkpoint that was never written
    because nothing improved so nothing saved.
    """
    def builder(d):
        torch.manual_seed(0)
        for _ in range(20):
            torch.randn(8)
        return d / "x"

    torch.manual_seed(4242)
    hit_key = ("t_rng", "prebuilt")
    cached_artifact(hit_key, builder)          # build it once, elsewhere
    torch.manual_seed(4242)
    after_hit = torch.randn(3)                 # caller gets a CACHE HIT

    torch.manual_seed(4242)
    cached_artifact(("t_rng", "fresh"), builder)   # caller BUILDS it
    after_miss = torch.randn(3)

    assert torch.equal(after_hit, after_miss), (
        "building the artifact advanced the caller's RNG -- a cache hit and a "
        "cache miss must leave the caller in the same state"
    )


def test_the_rng_is_restored_even_if_the_builder_raises():
    """GUARDS a save/restore that is not in a finally: a builder that fails
    would leave the RNG advanced AND the cache empty, so the next attempt
    starts somewhere different again."""
    def bad(d):
        torch.manual_seed(0)
        torch.randn(16)
        raise RuntimeError("builder failed")

    torch.manual_seed(99)
    before = torch.get_rng_state()
    with pytest.raises(RuntimeError):
        cached_artifact(("t_rng_raise", "x"), bad)
    assert torch.equal(torch.get_rng_state(), before)


def test_cache_directory_carries_the_size_label():
    """
    A bare fingerprint told a directory listing nothing about which
    resolution a cache belonged to -- a 32x32 test cache and a 128x128
    training cache were indistinguishable without opening one.

    The size is a LABEL, not part of the key: the fingerprint already
    separates encoders, and two encoders at different resolutions cannot
    collide anyway because their weight shapes differ and shapes are hashed.
    """
    from pathlib import Path
    from training.latent_cache import cache_path_for_run

    p = cache_path_for_run(Path("/c"), "abc123", Path("T625_n020_s3"),
                            [0, 100, 200], False, size=128)
    assert p.parent.name == "128x128-abc123"
    # the filename itself is unchanged
    assert p.name.startswith("T625_n020_s3-")
    assert p.name.endswith("-state.pt")


def test_cache_path_without_a_size_keeps_the_bare_fingerprint():
    """Back-compat: caches written before the rename stay readable."""
    from pathlib import Path
    from training.latent_cache import cache_path_for_run

    p = cache_path_for_run(Path("/c"), "abc123", Path("T625_n020_s3"),
                            [0, 100, 200], False)
    assert p.parent.name == "abc123"


def test_size_does_not_change_the_cache_key():
    """Same encoder, same steps, same stream -> same FILE name regardless of
    the directory label, so the label cannot silently split a cache."""
    from pathlib import Path
    from training.latent_cache import cache_path_for_run

    a = cache_path_for_run(Path("/c"), "abc", Path("r"), [0, 1], True, size=128)
    b = cache_path_for_run(Path("/c"), "abc", Path("r"), [0, 1], True)
    assert a.name == b.name
