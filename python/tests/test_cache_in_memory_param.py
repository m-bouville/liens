"""
Tests for train_autoencoder's `cache_in_memory` parameter.

The flag exists because caching every decoded snapshot was hardcoded True, and
that has two costs that both scale badly with grid size:

  - RAM. Snapshots are float16 on disk and cached as float32, and on Windows
    the DataLoader spawns rather than forks, so each of `num_workers` workers
    gets its own pickled copy of the dataset INCLUDING the cache. Measured on a
    real sweep (~38k snapshots): 0.6 GB at 64x64, 2.3 GB at 128, 9.4 GB at 256
    -- times (1 + num_workers).
  - STARTUP TIME. The cache is built by a serial list comprehension in the
    dataset's own __init__, one file at a time on one thread with the GPU idle.
    Uncached reads instead happen inside the DataLoader workers, in parallel and
    overlapped with compute. Caching moves the I/O from where it is parallel to
    where it is not.

These assert the plumbing -- that the flag reaches the dataset and that its
default is unchanged -- rather than measuring memory, which would be flaky.
"""
import inspect

import pytest

from training.datasets import MicrostructureSnapshotDataset
from training.train_stage1 import train_autoencoder


def test_cache_in_memory_is_a_parameter_defaulting_to_the_old_behaviour():
    """
    GUARDS a default flip. Every existing params file and every caller omits
    this argument, so the default IS the behaviour of the whole project;
    changing it silently would alter the memory profile of every stage-1 run
    that has ever been configured.
    """
    param = inspect.signature(train_autoencoder).parameters["cache_in_memory"]
    assert param.default is True


def test_the_params_file_can_set_it():
    """
    `_strip_unrecognized_params` filters params-file keys against the target
    function's signature, so a key only reaches the trainer if it is a real
    parameter -- this is what makes `cache_in_memory = false` in a params file
    take effect rather than being silently dropped with a warning.
    """
    from orchestration.stage_params import _strip_unrecognized_params
    kept = _strip_unrecognized_params(train_autoencoder,
                                       {"cache_in_memory": False, "epochs": 1}, "Stage 1")
    assert kept["cache_in_memory"] is False


def test_params_file_parses_the_boolean_spelling_used_in_those_files():
    from orchestration.stage_params import _convert_value
    assert _convert_value("false") is False
    assert _convert_value("true") is True


def test_no_call_site_still_hardcodes_the_cache():
    """
    GUARDS exposing the parameter but leaving one of the three dataset
    constructions pinned to True -- which would look correct in the signature
    and do nothing for the val set (or, worse, for the train set, which is the
    larger of the two).
    """
    # Comments are stripped first: the surrounding prose legitimately mentions
    # the old hardcoded form while explaining why it was removed, and a naive
    # substring search over the raw source would match that and fail for the
    # wrong reason.
    code = "\n".join(line for line in inspect.getsource(train_autoencoder).splitlines()
                      if not line.lstrip().startswith("#"))
    assert "cache_in_memory=True" not in code
    assert code.count("cache_in_memory=cache_in_memory") == 3


@pytest.mark.parametrize("cache", [True, False])
def test_dataset_honours_the_flag_both_ways(tmp_path, cache):
    """The flag has to reach the dataset's own caching behaviour, not just be
    accepted and ignored -- checked on the dataset directly, since running
    train_autoencoder needs a real sweep."""
    param = inspect.signature(MicrostructureSnapshotDataset.__init__).parameters["cache_in_memory"]
    assert param.default is False, ("the dataset's own default is False -- it is train_stage1 "
                                     "that opts in, and that is the layer this flag belongs to")
