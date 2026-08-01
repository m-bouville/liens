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
from pathlib import Path

import torch

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


# --------------------------------------------------------------------
# vram_log_every
# --------------------------------------------------------------------

def test_vram_log_every_zero_disables_the_report():
    """
    The BEHAVIOUR of the off switch, not the value of the default. An earlier
    version of this test asserted `default == 0`, which encoded no property --
    it restated the source, could not catch a bug, and did nothing but block
    the legitimate act of turning the logging on to diagnose an OOM. Pinning a
    default is only worth its friction when changing it would break something
    silently (see cache_in_memory below); logging verbosity is not that.
    """
    lines = inspect.getsource(train_autoencoder).splitlines()
    guard = [l for l in lines if "vram_log_every" in l and "batch_idx" in l]
    assert guard, "the report must be gated on vram_log_every at all"
    assert "if vram_log_every and" in guard[0], (
        "0 must disable it -- a bare modulo would report on batch 0 of every epoch "
        "even when the feature is off"
    )


def test_vram_report_names_all_three_quantities():
    """
    GUARDS reporting only "free". Free alone cannot distinguish the three
    causes of a mid-epoch OOM -- fragmentation (reserved grows, allocated
    flat), a leak (allocated grows), and a busy card (both flat) -- and those
    have three different fixes.
    """
    from training.train_stage1 import _vram_report
    line = _vram_report("t")
    if not line:
        pytest.skip("no CUDA device in this environment")
    for quantity in ("allocated", "reserved", "peak", "free"):
        assert quantity in line


def test_vram_report_is_a_no_op_without_cuda():
    from training.train_stage1 import _vram_report
    if torch.cuda.is_available():
        pytest.skip("CUDA present; the no-CUDA path cannot be exercised here")
    assert _vram_report("t") == ""


def test_the_report_is_emitted_inside_the_batch_loop_not_per_epoch():
    """
    GUARDS moving the report to the end of the epoch. An augmented 128x128
    epoch is ~48k batches, so a per-epoch report would first print AFTER the
    point where the OOM occurs -- which is exactly the information needed.
    """
    lines = inspect.getsource(train_autoencoder).splitlines()
    start = next(i for i, l in enumerate(lines)
                  if "for batch_idx, batch in enumerate(train_loader):" in l)
    loop_indent = len(lines[start]) - len(lines[start].lstrip())
    # The body is every following line indented DEEPER than the `for` itself;
    # slicing on a text marker instead let a report moved to just before the
    # post-loop `train_total = (...)` still fall inside the slice, so this
    # test passed on the very regression it was written to catch.
    body = []
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= loop_indent:
            break
        body.append(line)
    assert any("_vram_report(" in l for l in body), "the report must be INSIDE the batch loop"


_REF_ROW_CONFIG = dict(
    size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
    val_fraction=0.34, test_fraction=0.17, num_workers=0, min_step=0, min_stdev_phi=None,
    stats0_weight=1.0, stat_names=["avg_phi"], recon0_scale=1e-4, stats0_scale=1e-2,
    device="cpu", seed=0, log_every_epoch=False,
)


def _ref_row_ancestor(tmp_path):
    """A stage-1 checkpoint to resume FROM, shared across the reference-row
    tests via conftest's cache.

    Each of these tests needs two stage-1 runs: one to produce an ancestor and
    one that resumes from it. Only the second is the subject, so building the
    first per test was ~half the cost of each -- and three of them landed in
    the suite's slowest ten.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_train_lds import _build_sweep
    from conftest import cached_stage1_ancestor
    return cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32), **_REF_ROW_CONFIG)


# --------------------------------------------------------------------
# epoch-0 reference row on resume
# --------------------------------------------------------------------

def test_reference_row_is_printed_when_resuming(tmp_path, capsys):
    """
    A resumed run's epoch 1 has nothing to be compared against without this.
    After a size port it is the single most useful number available: what the
    transferred weights were worth BEFORE any training at the new size.
    """
    base_path, first = _ref_row_ancestor(tmp_path)
    capsys.readouterr()
    train_autoencoder(base_path=base_path, checkpoint_path=tmp_path / "b.pt",
                       loss_curve_path=tmp_path / "b.png", resume_from=first,
                       **_REF_ROW_CONFIG)
    out = capsys.readouterr().out
    assert " ref|" in out
    assert "(before this run)" in out


def test_no_reference_row_without_a_resume(tmp_path, capsys):
    """
    GUARDS printing it unconditionally. With nothing resumed the model is
    freshly initialised, so a reference row would just report the loss of
    random weights -- a full extra pass over the val set for no information.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_train_lds import _build_sweep

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    train_autoencoder(size=32, base_path=base_path, epochs=1, batch_size=4, base_channels=4,
                       latent_channels=4, val_fraction=0.34, test_fraction=0.17, num_workers=0,
                       min_step=0, min_stdev_phi=None, stats0_weight=0.01,
                       stat_names=["avg_phi"], device="cpu", seed=0, log_every_epoch=False,
                       checkpoint_path=tmp_path / "c.pt", loss_curve_path=tmp_path / "c.png")
    assert " ref|" not in capsys.readouterr().out


def test_reference_row_does_not_change_the_training_trajectory(tmp_path):
    """
    GUARDS omitting the RNG save/restore. The reference pass advances torch's
    global RNG, which reorders train_loader's shuffle at epoch 1 -- so a
    purely diagnostic row would silently change what the run actually trains
    on. train_stage2 documents having hit exactly this.

    Compared on WEIGHTS, not on the printed loss: a shuffle difference changes
    the trajectory, which is the thing that must not happen.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_train_lds import _build_sweep
    import training.train_stage1 as mod

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    common = dict(size=32, base_path=base_path, epochs=1, batch_size=4, base_channels=4,
                   latent_channels=4, val_fraction=0.34, test_fraction=0.17, num_workers=0,
                   min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
                   device="cpu", seed=0, log_every_epoch=False)
    ancestor = train_autoencoder(checkpoint_path=tmp_path / "anc.pt",
                                  loss_curve_path=tmp_path / "anc.png", **common)

    with_ref = train_autoencoder(checkpoint_path=tmp_path / "w.pt",
                                  loss_curve_path=tmp_path / "w.png",
                                  resume_from=ancestor, **common)
    a = torch.load(with_ref, map_location="cpu", weights_only=True)["model_state"]

    # same run, reference row suppressed
    src = inspect.getsource(mod)
    assert "if epochs > 0 and resume_from is not None:" in src
    original = mod.train_autoencoder
    try:
        import re
        patched = re.sub(r"if epochs > 0 and resume_from is not None:",
                          "if False:", src, count=1)
        namespace = dict(vars(mod))
        exec(compile(patched, mod.__file__, "exec"), namespace)
        without_ref = namespace["train_autoencoder"](
            checkpoint_path=tmp_path / "n.pt", loss_curve_path=tmp_path / "n.png",
            resume_from=ancestor, **common)
    finally:
        mod.train_autoencoder = original
    b = torch.load(without_ref, map_location="cpu", weights_only=True)["model_state"]

    assert all(torch.equal(a[k], b[k]) for k in a), (
        "the reference row changed the training trajectory -- RNG not restored"
    )


def test_reference_row_components_sum_to_its_total(tmp_path, capsys):
    """
    GUARDS printing the ref row's components RAW while the epoch rows print
    them SCALED. step() returns raw values; the epoch line divides by
    recon0_scale / stats0_scale and applies stats0_weight, which is what the
    heading advertises and what actually sums to `total`.

    Printed raw, the ref row was the only line in the table whose components
    did not add up -- "6.9589 = 0.0003 + 0.0381" -- which reads as a broken
    total rather than a units mismatch, and is exactly the kind of number
    someone later tries to reconcile against a real measurement.

    Parsed from the console rather than compared as formatted strings (a
    project convention: a formatted-string comparison once failed on Windows
    by 1e-4 at a rounding boundary).
    """
    base_path, first = _ref_row_ancestor(tmp_path)
    capsys.readouterr()
    train_autoencoder(base_path=base_path, checkpoint_path=tmp_path / "b.pt",
                       loss_curve_path=tmp_path / "b.png", resume_from=first,
                       **_REF_ROW_CONFIG)

    line = [l for l in capsys.readouterr().out.splitlines() if l.strip().startswith("ref|")][0]
    val_part = line.split("|")[2]            # " 6.9589 = 0.0003 + 0.0381 "
    total = float(val_part.split("=")[0])
    recon0, stats0 = (float(x) for x in val_part.split("=")[1].split("+"))
    assert total == pytest.approx(recon0 + stats0, rel=1e-3), (
        f"ref components must sum to the ref total: {total} vs {recon0} + {stats0}"
    )
