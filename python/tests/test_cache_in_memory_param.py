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

@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_reference_row_does_not_change_the_training_trajectory(tmp_path, capsys):
    """
    GUARDS omitting the RNG save/restore. The reference pass advances torch's
    global RNG, which reorders train_loader's shuffle at epoch 1 -- so a
    purely diagnostic row would silently change what the run actually trains
    on. train_stage2 documents having hit exactly this.

    Compared on the epoch-1 TRAIN LOSS, which is what a shuffle difference
    would change, rather than on the saved checkpoint's weights.

    NOT on weights, because the reference row acquired a SECOND, intended
    effect: it supplies tracker.reference_val_loss, the ceiling that stops a
    resumed run saving something worse than its ancestor. Suppressing the row
    therefore suppresses the ceiling too, so the two arms differ in SAVE POLICY
    rather than trajectory:

        with ref:    "no epoch improved on the ancestor ... kept as w.pt"
                      -> w.pt holds the ANCESTOR
        without ref: "1| ... -> saved"
                      -> n.pt holds this run's epoch-1 weights

    Different by construction, whatever the RNG did. Comparing weights reported
    "the reference row changed the training trajectory -- RNG not restored",
    which was the wrong diagnosis of a real difference.

    It used to compare weights, and that stopped working once the reference
    row began seeding the tracker's ceiling: the ref-row arm then declines to
    save an epoch worse than its ancestor while the no-ref arm always saves
    epoch 1, so the two arms legitimately hold DIFFERENT checkpoints. Whether
    they did was data-dependent -- this test passed on one machine and failed
    on another.

    The underlying property is unchanged and is still what is asserted: the
    reference pass must not perturb the RNG, so the training trajectory must
    be identical. Only the observable had to move off the checkpoint, which is
    now a function of the criterion as well as of the trajectory.
    """
    import training.train_stage1 as mod

    # SHARED ancestor: this test's own config was arbitrary -- it exercises RNG
    # restoration, which is config-independent -- so building a private one
    # cost a full extra stage-1 run for nothing. _ref_row_ancestor is cached
    # across every test in this file that resumes.
    #
    # log_every_epoch is overridden only on the RESUMED arms: the ancestor
    # never needs its rows printed, and keeping it out of the cache key is
    # what lets the ancestor be shared at all.
    base_path, ancestor = _ref_row_ancestor(tmp_path)
    common = dict(_REF_ROW_CONFIG, base_path=base_path, log_every_epoch=True)

    # CLEARED before the measured run: the ancestor above prints its own
    # epoch-1 row, and parsing the FIRST such row in the capture picked that
    # up instead -- comparing the ancestor against the no-ref arm and
    # reporting a trajectory difference that did not exist.
    capsys.readouterr()
    with_ref = train_autoencoder(checkpoint_path=tmp_path / "w.pt",
                                  loss_curve_path=tmp_path / "w.png",
                                  resume_from=ancestor, **common)
    del with_ref  # the checkpoint is no longer the observable -- see the docstring
    out_with_ref = capsys.readouterr().out

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
    del without_ref
    out_without_ref = capsys.readouterr().out

    # Epoch-1 TRAIN loss from each arm. A reordered shuffle changes which
    # samples land in which batch, so the epoch average moves -- that is the
    # observable, and unlike the saved checkpoint it does not depend on
    # whether the criterion chose to save.
    def _epoch1_train_loss(text):
        for line in text.splitlines():
            head = line.split("|")[0].strip()
            if head == "1":
                return float(line.split("|")[1].split("=")[0].strip())
        raise AssertionError(f"no epoch-1 row found in:\n{text[-600:]}")

    with_ref_loss = _epoch1_train_loss(out_with_ref)
    without_ref_loss = _epoch1_train_loss(out_without_ref)
    assert with_ref_loss == without_ref_loss, (
        f"the reference row changed the training trajectory -- RNG not restored "
        f"({with_ref_loss} with the row, {without_ref_loss} without)"
    )


@pytest.mark.slow
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


# --------------------------------------------------------------------
# ema_warmup_epochs in stage 1
# --------------------------------------------------------------------

def test_stage1_has_an_ema_warmup_by_default():
    """
    GUARDS ema_warmup_epochs=0, which made stage 1 the ONLY stage without the
    protection every other stage has. With no warmup, epoch 1's raw val_loss
    seeds BOTH the EMA and best_val_loss, so a lucky first epoch sets a bar
    later smoothed values struggle to clear.

    Observed on a real 128x128 run: epoch 1 was the minimum of all 11 epochs,
    nothing saved again, early stop at epoch 11 with TRAIN loss still falling
    19.6% -- and stage 2 then trained from that epoch-1 checkpoint.
    """
    assert inspect.signature(train_autoencoder).parameters["ema_warmup_epochs"].default == 5


def test_stage1_warmup_epochs_do_not_count_toward_early_stopping():
    """
    GUARDS counting warmup epochs as non-improvement. During the window
    should_save is unconditionally False, so every warmup epoch increments the
    counter -- any ema_warmup_epochs >= early_stopping_patience would stop the
    run before the criterion had begun answering. Exactly the interaction that
    made stage 2's deriv_target_centered switch stop one epoch short of its
    own grace window.
    """
    source = inspect.getsource(train_autoencoder)
    guard = [l for l in source.splitlines() if "epochs_since_improvement >= early_stopping_patience" in l]
    assert guard, "could not find the early-stopping check"
    window = source[source.index("if (early_stopping_patience is not None"):]
    assert "epoch > _grace" in window.split(":")[0]


def test_stage1_warmup_is_clamped_so_a_short_run_can_still_save():
    """
    GUARDS passing ema_warmup_epochs straight through. A grace period covering
    every remaining epoch means NO checkpoint is written at all -- a missing
    file rather than a worse one, and every downstream consumer fails far from
    the cause. clamp_grace_epochs exists for this and stage 1 must use it.
    """
    from training.checkpoint_criterion import clamp_grace_epochs
    assert "clamp_grace_epochs(ema_warmup_epochs, epochs)" in inspect.getsource(train_autoencoder)
    assert clamp_grace_epochs(5, 3) == 2      # always leaves one epoch able to save
    assert clamp_grace_epochs(5, 1) == 0


@pytest.mark.slow
def test_no_epoch_inside_the_warmup_window_can_save(tmp_path, capsys):
    """
    The behaviour, end to end: no epoch within the warmup window may save, so
    none of them can plant a flag later epochs must beat. That -- not "a save
    happens afterwards" -- is the actual guarantee.

    An earlier version of this test asserted a save occurred AFTER the window.
    It passed alone and failed in-file, because whether any epoch improves at
    all depends on the loss trajectory, which depends on which sweep the shared
    ancestor cache happened to build first. Asserting on the mechanism instead
    of on a data-dependent outcome makes it deterministic.
    """
    base_path, first = _ref_row_ancestor(tmp_path)
    config = dict(_REF_ROW_CONFIG)
    warmup = 2
    config.update(epochs=3, ema_warmup_epochs=warmup, early_stopping_patience=None,
                   log_every_epoch=True)
    checkpoint_path = tmp_path / "w.pt"
    capsys.readouterr()

    # Nothing may save during grace, so with epochs=3 and warmup=2 the ONLY
    # epoch that can save is 3 -- and whether it does depends on the loss
    # trajectory, because the tracker leaves best_val_loss at the EMA reached
    # during grace and epoch 3 has to beat it.
    #
    # Both outcomes are consistent with the property under test, and the test
    # accepts both rather than pretending the trajectory is deterministic. An
    # earlier version asserted a save occurred, which was flaky; the version
    # after that read the checkpoint unconditionally, which turned the same
    # flake into a RuntimeError from the no-save guard.
    #
    # It still catches the bug: with grace=0, epoch 1 always saves (nothing
    # beats inf), so a checkpoint would exist carrying epoch 1 <= warmup.
    try:
        train_autoencoder(base_path=base_path, checkpoint_path=checkpoint_path,
                           loss_curve_path=tmp_path / "w.png", resume_from=first, **config)
    except RuntimeError as exc:
        assert "without ever saving" in str(exc), exc
        assert not checkpoint_path.exists()
        return          # nothing saved at all -- vacuously inside no window

    # The output file may be the ANCESTOR, copied forward because nothing beat
    # it -- a third legitimate outcome, added when the reference ceiling made
    # "no improvement" common. Its epoch belongs to the PREVIOUS run's
    # numbering and says nothing about this run's grace period, so compare the
    # bytes rather than trusting the epoch field.
    if checkpoint_path.read_bytes() == Path(first).read_bytes():
        return          # nothing this run produced was better; nothing to check

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["epoch"] > warmup, (
        f"the saved checkpoint is from epoch {saved['epoch']}, inside the "
        f"{warmup}-epoch warmup window -- the grace period is not suppressing saves"
    )
    saved_epochs = [int(l.split("|")[0].strip()) for l in capsys.readouterr().out.splitlines()
                    if "-> saved" in l and l.split("|")[0].strip().isdigit()]
    assert all(e > warmup for e in saved_epochs), (
        f"epoch(s) {[e for e in saved_epochs if e <= warmup]} saved inside the "
        f"{warmup}-epoch warmup window"
    )
