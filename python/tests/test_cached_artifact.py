"""
Tests for conftest's cached_artifact / copy_cached_files.

These exist because the cache trades correctness risk for speed, and the risk
is SILENT: a key too coarse hands a test an artifact built with different
settings, and the test keeps passing while no longer testing what it claims.
A shared mutable artifact corrupts later tests in a way that depends on
execution order and disappears under `-k`. Both are worse than the slowness
the cache removes, so both are asserted directly.
"""
from pathlib import Path

import pytest

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
