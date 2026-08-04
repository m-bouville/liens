"""
Shared pytest fixtures. Deliberately avoid needing a real trained
checkpoint anywhere -- these are small, deterministic stand-ins whose
only job is to have the right SHAPE and be cheap to run, so tests stay
fast and don't depend on any particular training run having happened.
"""
import os
import sys
from pathlib import Path

# Every module in this project (training/, models/, utils/) is meant to
# be imported with python/ itself on sys.path -- true when running e.g.
# `python -m training.train_stage1` from python/, but NOT automatic for
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
import itertools
import shutil
import tempfile

import inspect

import time

import re

import pytest

from models.constants import LATENT_SPATIAL_SIZE
from training.train_stage1 import train_autoencoder
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


# ---------------------------------------------------------------------
# Shared, memoized synthetic sweeps.
#
# Seven test modules each define their own _build_sweep(), and between
# them they call it ~45 times. Each call writes a full set of synthetic
# run directories (snapshots + metadata + statistics.csv) and costs
# ~0.5s -- about 21s of a ~150s suite spent rebuilding data that is
# then only ever READ.
#
# Verified read-only before adding this: every write into a sweep
# happens inside the _build_* helpers themselves; no test mutates a
# sweep after receiving it (checkpoints and outputs go to the test's own
# tmp_path, which is untouched by this). So one build per distinct
# argument set can be shared across every test that asks for it.
#
# The modules' builders are NOT interchangeable -- they disagree on
# directory layout, on _build_run_dir's signature, and on what they
# return -- so this caches per (module, args) rather than trying to
# unify them. Unifying is a separate refactor; this is just the cost fix.
#
# Deliberately a module-level dict rather than a session-scoped fixture:
# the builders are called as plain functions from inside test bodies
# (_build_sweep(tmp_path, ...)), so a fixture would mean touching all
# ~45 call sites and their signatures.
_SWEEP_CACHE: dict = {}
_SWEEP_ROOT: Path | None = None
# MONOTONIC, deliberately -- NOT len(_SWEEP_CACHE). Using the dict's own
# length as the directory name collides the moment any entry is removed
# (two different keys then compute the same name, and mkdir fails with
# FileExistsError). Real runs never remove entries, so this only showed
# up once a test popped its own probe key -- but a counter that can go
# backwards is fragile regardless of who currently triggers it.
_SWEEP_SEQ = itertools.count()
_ARTIFACT_CACHE: dict = {}
_ARTIFACT_SEQ = itertools.count()


@pytest.fixture(scope="session", autouse=True)
def _sweep_cache_root(tmp_path_factory):
    """
    Anchors every cached sweep under pytest's OWN session temp root, so
    pytest's normal retention policy (keep the last few sessions, prune
    older ones) applies to them.

    A first version of this used a bare tempfile.mkdtemp() instead --
    which nothing ever cleans up. That leaked one directory per distinct
    sweep per test run, permanently: 154 orphaned sweep_* directories had
    accumulated in /tmp before this was caught. Using tmp_path_factory
    keeps the whole point of the cache (a sweep must OUTLIVE any single
    test's own tmp_path, so it can be shared across tests) while putting
    cleanup back under pytest's control.

    autouse + session scope so it is guaranteed to have run before any
    test body calls cached_sweep().
    """
    global _SWEEP_ROOT
    _SWEEP_ROOT = tmp_path_factory.mktemp("cached_sweeps")
    yield
    _SWEEP_CACHE.clear()
    _ARTIFACT_CACHE.clear()


def cached_sweep(key, builder):
    """Builds once per distinct `key`, into a directory that OUTLIVES
    any single test's own tmp_path (see _sweep_cache_root for where it
    actually lives, and why that matters for cleanup), and returns
    whatever `builder` returned. Safe only because sweeps are read-only
    -- verified before this was introduced; see the block comment above."""
    if key not in _SWEEP_CACHE:
        if _SWEEP_ROOT is None:
            # Only reachable if cached_sweep is called outside a pytest
            # session entirely (e.g. an ad-hoc import). Fall back rather
            # than crash, but this path does NOT get pytest's cleanup.
            base = Path(tempfile.mkdtemp(prefix="sweep_uncleaned_"))
        else:
            base = _SWEEP_ROOT / f"sweep_{next(_SWEEP_SEQ):03d}"
            base.mkdir(parents=True, exist_ok=False)
        _SWEEP_CACHE[key] = builder(base)
    return _SWEEP_CACHE[key]


def cached_artifact(key, builder):
    """
    Like cached_sweep, but for artifacts a test may WRITE to -- checkpoints,
    above all.

    Motivation: across tests/, `train_autoencoder` is called 31 times and
    `train_stage2` 38 times, and almost none of those are testing stage 1 or
    stage 2. They are manufacturing an ANCESTOR so that stage 3, or the
    pipeline, or checkpoint extraction can be tested at all. In
    test_train_lds.py alone, 11 tests build 11 stage-1 checkpoints from a
    single distinct config and 11 stage-2 checkpoints from two -- roughly
    three quarters of each test's runtime spent reconstructing something
    another test already built.

    THE DIFFERENCE FROM cached_sweep, and why this is not just an alias:
    cached_sweep's own docstring notes it is "safe only because sweeps are
    read-only". Checkpoints are not. A test that passes an ancestor as
    `resume_from` to a stage which then saves in place would corrupt the
    shared copy for every test scheduled after it, producing failures that
    depend on execution order and vanish under `-k`. So `builder` writes into
    a cached directory ONCE, and every caller receives a fresh COPY (a
    ~1 MB file copy, microseconds) rather than the cached path itself.

    KEYING. Derive the key MECHANICALLY from the full set of arguments that
    reach the artifact -- not from a hand-written label. A key that is too
    coarse hands a test an ancestor built with different settings, and the
    failure mode is silent: the test still passes, it just stops testing what
    it claims to. That is strictly worse than the slowness this removes.
    Output paths must be excluded (they differ per test by construction and
    cannot affect content); everything else must be included, even arguments
    that look inert.

    builder(dir) -> a Path, or a tuple/list whose Path entries are copied.
    Non-Path entries (e.g. a sweep directory, which IS read-only) pass
    through unchanged.
    """
    if key not in _ARTIFACT_CACHE:
        if _SWEEP_ROOT is None:
            base = Path(tempfile.mkdtemp(prefix="artifact_uncleaned_"))
        else:
            base = _SWEEP_ROOT / f"artifact_{next(_ARTIFACT_SEQ):03d}"
            base.mkdir(parents=True, exist_ok=False)
        # RNG SAVED AND RESTORED AROUND THE BUILD. A cache must be INVISIBLE:
        # a caller has to reach its own next step in the same state whether it
        # built the artifact or got a hit. Building trains a model, which
        # advances torch's global RNG -- so without this, only the FIRST caller
        # pays that advance and every later one starts from a different state
        # than it did before the cache existed.
        #
        # Not hypothetical: it broke
        # test_deriv_target_centered_resume_skips_completed_warmup, which runs
        # two stage-2 trainings on 12 windows and is sensitive to the RNG. It
        # failed only under xdist, only on one worker, and only after this
        # refactor -- surfacing as FileNotFoundError on a checkpoint that was
        # never written, because nothing improved so nothing saved.
        #
        # Exactly the hazard the stage-1 reference row documents: an action
        # taken for INFRASTRUCTURE must not perturb the sequence the real work
        # depends on.
        rng_state = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            _ARTIFACT_CACHE[key] = builder(base)
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
    return _ARTIFACT_CACHE[key]


def copy_cached_files(cached, dest_dir: Path):
    """Copy every Path in `cached` into dest_dir, returning the same shape.

    Files, not directories: a cached sweep directory is read-only and shared
    deliberately (see cached_sweep), so copying it would throw away the very
    saving this exists for. Only the writable artifacts are duplicated.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _one(item):
        if isinstance(item, Path) and item.is_file():
            target = dest_dir / item.name
            shutil.copy2(item, target)
            return target
        return item

    if isinstance(cached, (tuple, list)):
        return type(cached)(_one(i) for i in cached)
    return _one(cached)


def cached_stage1_ancestor(tmp_path, build_sweep, **stage1_kwargs):
    """A (base_path, stage1_checkpoint) pair, built once per distinct config.

    Most tests that call train_autoencoder are not testing stage 1 -- they need
    an ANCESTOR before stage 2 or stage 3 can be exercised at all. Across
    tests/ that is 31 calls producing a handful of distinct checkpoints.

    build_sweep(dir) must return the sweep's base_path; pass the caller's own
    _build_sweep so each file keeps its own sweep shape. The returned
    checkpoint is a fresh copy in tmp_path (see cached_artifact on why a copy
    and not the cached path); base_path is the shared, read-only sweep.

    The key is the FULL stage1_kwargs, so a test differing in any argument --
    even one that looks inert, such as augment=False vs its default -- gets its
    own ancestor rather than silently inheriting another test's.
    """
    def _build(cache_dir):
        base_path = build_sweep(cache_dir)
        checkpoint = train_autoencoder(
            base_path=base_path,
            checkpoint_path=cache_dir / "stage1_ancestor.pt",
            loss_curve_path=cache_dir / "stage1_ancestor_curve.png",
            **stage1_kwargs,
        )
        return base_path, Path(checkpoint)

    key = ("stage1_ancestor", getattr(build_sweep, "__module__", "?"),
           tuple(sorted((k, repr(v)) for k, v in stage1_kwargs.items())))
    base_path, cached = cached_artifact(key, _build)
    return base_path, copy_cached_files(cached, tmp_path)


@pytest.fixture(scope="session", autouse=True)
def _limit_torch_threads_under_xdist():
    """One torch intra-op thread per pytest-xdist worker.

    torch sizes its thread pool to the machine's core count, per PROCESS. Under
    `-n 4` that is four pools of N threads competing for N cores, and the
    oversubscription is severe rather than marginal: measured on this suite,
    a test that takes ~4 s serially took 20.65 s under `-n 4` -- 5.2x slower
    each -- so four-way parallelism returned only 1.7x overall (204 s -> 121 s).

    Nothing here is large enough to want intra-op threading anyway. These
    fixtures train base_channels=4 models on 32x32 inputs at batch 4; at that
    size thread dispatch costs more than it saves, which is why capping to 1
    is close to free per test and lets the four workers actually run in
    parallel.

    Applied ONLY under xdist (PYTEST_XDIST_WORKER is set per worker), so a
    plain serial `pytest` keeps whatever threading it had -- this fixes a
    contention problem that does not exist there, and changing serial
    behaviour would be an unmeasured side effect.
    """
    if os.environ.get("PYTEST_XDIST_WORKER"):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(previous)
    else:
        yield


def source_without_comments(target) -> str:
    """Source of `target` with COMMENTS AND DOCSTRINGS removed.

    For tests that assert a piece of production code exists by matching its
    text. Matching raw source also matches the prose ABOUT that code -- and
    the prose necessarily names the very thing being checked, because that is
    what makes it a useful comment.

    Not hypothetical: it happened three times in one session (a forwarding
    test hit a docstring mention of `f_theta.rollout(`; a _free_vram test hit
    the comment explaining `last_traceback`; a scatter test hit the comment
    saying NOT to use `ax.get_xlim()`, so it failed on correct code). An audit
    afterwards found three MORE tests where commenting out the implementation,
    leaving the comment behind, kept them green.

    tokenize rather than a line-based startswith("#"): it strips trailing
    comments too, and will not mistake a '#' inside a string for one.

    `target` may be a module, a function/class, or a Path.
    """
    import io
    import tokenize as _tok

    if isinstance(target, (str, Path)):
        src = Path(target).read_text(encoding="utf-8")
    else:
        src = inspect.getsource(target)
    try:
        tokens = list(_tok.generate_tokens(io.StringIO(src).readline))
    except (_tok.TokenError, IndentationError):
        return src          # unparseable fragment: raw is better than nothing
    kept = []
    for tok in tokens:
        if tok.type == _tok.COMMENT:
            continue
        # A STRING whose own line STARTS with a triple quote is a docstring
        # (or a standalone block string). Comparing against the PREVIOUS
        # token's line does not work: INDENT sits on the same line as the
        # docstring it precedes, so the docstring never looks like the first
        # token on its line -- which is why the first version of this left
        # every docstring in place.
        if tok.type == _tok.STRING and tok.line.lstrip().startswith(('"""', "'''",
                                                                      'r"""', "r'''")):
            continue
        kept.append(tok)
    try:
        return _tok.untokenize(kept)
    except (ValueError, IndexError):
        return src


def assert_figure_was_really_written(output_path, min_kb: float = 5.0):
    """The figure exists and is not empty or truncated.

    This is a MODEST check and the docstring says so on purpose, because two
    stronger versions were tried and neither works:

    - a fixed byte floor cannot separate blank from real across panel counts.
      Measured at dpi=120: a CLEARED six-panel figure is 16.4 kB while a REAL
      single-panel one is 13.3 kB. Any threshold that passes the second
      accepts the first.
    - normalising by a blank figure of the same pixel dimensions does not fix
      it either: cleared-1-panel scores 4.66x its blank equivalent, real
      24-panel only 2.21x.
    - pixel metrics are worse still. "Fraction of non-white pixels" reads
      0.0184 blank against 0.0230 for a real curve; "distinct colours" reads
      198 against 439 but only 213 for a legitimate GRAYSCALE image plot --
      and the reconstruction figures are exactly that.

    So what this catches is a file that is MISSING, EMPTY or TRUNCATED -- a
    real class of failure (an exception between savefig and close, a full
    disk) that bare `.exists()` misses, and nothing more.

    For "the figure was written but is WRONG or was drawn from NO DATA", assert
    on what the script REPORTS: several return their window/sample counts
    directly, and the rest print them. That is the check that actually
    discriminates, and it is what the callers of this helper also do.
    """
    output_path = Path(output_path)
    assert output_path.exists(), f"{output_path} was not written at all"
    size_kb = output_path.stat().st_size / 1024
    assert size_kb >= min_kb, (
        f"{output_path.name} is only {size_kb:.1f} kB -- empty or truncated, not a figure"
    )
    return output_path


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
    module actually involved in a full stage 1->2->3a->3b run
    (train_stage1.py/train_stage2.py/train_lds.py/train_refinement.py/
    orchestration.paths/orchestration.pipeline) -- NOT the individual
    evaluation/*.py scripts, which each have their own, separate
    _PYTHON_ROOT too, but only ever use it for CLI argument defaults
    (argparse), never when called programmatically with an explicit
    checkpoint_path/output_path the way the pipeline itself always
    does internally.

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
                  2: root / "checkpoints" / "stage2",
                  3: root / "checkpoints" / "stage3",
                  "3a": root / "checkpoints" / "stage3a",
                  "3b": root / "checkpoints" / "stage3b",
                  4: root / "checkpoints" / "stage4",
                  5: root / "checkpoints" / "stage5"}

    import training.train_stage1 as train_stage1
    import training.train_stage2 as train_stage2
    import training.train_lds as train_lds
    import training.train_refinement as train_refinement
    import orchestration.paths as orch_paths
    import orchestration.pipeline as orch_pipeline
    # checkpoint_registry does `from orchestration.paths import
    # _CHECKPOINTS_ROOT, _STAGE_DIRS`, binding its OWN module-level names
    # at import time -- so patching orch_paths alone never reaches them,
    # and its _resolve_checkpoint_field would resolve relative registry
    # paths against the REAL checkpoints tree during tests. Read-only, so
    # nothing is written there, but a cache hit/miss test could then pass
    # or fail against real-tree paths instead of the isolated ones.
    # Exactly the per-module-copy problem this fixture's own docstring
    # describes, and the same shape as train_stage1/train_stage2's own
    # _PYTHON_ROOT copies already handled below.
    import orchestration.checkpoint_registry as orch_registry

    # EVERY evaluation script too, not just the training/orchestration ones.
    # This fixture's docstring used to claim the evaluation modules touch
    # _PYTHON_ROOT "only ever for CLI argument defaults ... never when called
    # programmatically". That is false for twelve of them: check_f_theta,
    # check_rollout, check_reconstruction and others compute a default
    # output_path from _PYTHON_ROOT INSIDE the function, which is exactly the
    # path a test of the default-folder logic has to take -- and a stray
    # 64x64-stage3b-f_theta_diagnostic.png duly appeared in the real
    # output/stage3b/ during an unrelated 128x128 run.
    #
    # Discovered by import rather than listed: a hand-maintained list is what
    # let this drift in the first place, and a new script with a default path
    # would not be added to it.
    import importlib
    import pkgutil
    evaluation_modules = []
    for info in pkgutil.iter_modules([str(Path(__file__).resolve().parent.parent
                                           / "evaluation")]):
        try:
            mod = importlib.import_module(f"evaluation.{info.name}")
        except Exception:  # noqa: BLE001 - a broken script must not break the fixture
            continue
        if hasattr(mod, "_PYTHON_ROOT"):
            evaluation_modules.append(mod)

    for module in (train_stage1, train_stage2, train_lds, train_refinement, orch_paths,
                    orch_pipeline, *evaluation_modules):
        monkeypatch.setattr(module, "_PYTHON_ROOT", root, raising=True)
    for module in (orch_paths, orch_pipeline, orch_registry):
        monkeypatch.setattr(module, "_STAGE_DIRS", stage_dirs, raising=True)
    for module in (orch_paths, orch_registry):
        monkeypatch.setattr(module, "_CHECKPOINTS_ROOT", root / "checkpoints", raising=True)

    return root


# Grid sizes the test suite ever builds a sweep at. Declared, not inferred, and
# enforced by test_no_test_builds_a_sweep_outside_TEST_GRID_SIZES.
#
# The leak detector uses it to attribute a file: `128x128-stage4-...` cannot
# have come from a test, because no test ever creates a 128x128 sweep. That is
# a stronger discriminator than timing, which cannot separate "a training run
# that FINISHED during the session" from "a test wrote it" -- and a real stage-4
# run finishing mid-session is exactly the case that defeated the timing check.
TEST_GRID_SIZES = frozenset({8, 16, 32, 64})


def _looks_like_a_test_artifact(path) -> bool:
    """True when the filename's <N>x<N> prefix is a size the tests use.

    Unrecognised names are treated as OURS (conservative): a genuine leak with
    an unexpected name must still be reported. Only a clear, non-test grid size
    exonerates a file.
    """
    import re as _re
    m = _re.match(r"(\d+)x(\d+)[-_.]", path.name)
    if not m:
        return True                      # cannot tell -> report it
    return int(m.group(1)) in TEST_GRID_SIZES


def snapshot_files(bases) -> dict:
    """path -> (size, mtime_ns) for every file under `bases`.

    Identity, not just path, so a MOVE can be told apart from a write.
    """
    seen = {}
    for base in bases:
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            seen[f] = (st.st_size, st.st_mtime_ns)
    return seen


def files_written_between(before: dict, after: dict) -> list:
    """Paths a test WROTE: newly appeared, or modified in place.

    A module-level function rather than logic inline in the fixture, so a test
    can exercise the ACTUAL rule. An inline version forced the tests to
    re-implement it, and dropping the move check in the fixture then left them
    all green -- they were checking their own arithmetic.

    A new path whose (size, mtime_ns) matches something that DISAPPEARED is a
    move: shutil.move and copy2 both preserve mtime. A COPY is still reported,
    because its source does not disappear, so its identity is not in
    `vanished` -- which is why the check is on vanished entries rather than on
    mtime alone.
    """
    vanished = {v for k, v in before.items() if k not in after}
    appeared = [p for p in set(after) - set(before) if after[p] not in vanished]
    # MODIFIED IN PLACE, not just appeared. Reporting only new paths meant a
    # leak was caught the FIRST time and invisible ever after, because the file
    # already existed. Tests had been overwriting real output/*.png for a long
    # time unseen; adding a .csv beside those PNGs created a genuinely new path
    # and finally surfaced it. Same write, same test -- only the visibility
    # changed.
    modified = [p for p in set(after) & set(before) if after[p] != before[p]]
    return sorted(appeared + modified)


@pytest.fixture(scope="session", autouse=True)
def _fail_on_writes_to_the_real_project_tree():
    """Fail the session if any test writes into the REAL output/ or checkpoints/.

    Every module anchors its default paths to
    `_PYTHON_ROOT = Path(__file__).resolve().parent.parent`, so a test that
    does not override them writes into the working project the person is
    actually using -- files they then have to recognise as junk and delete by
    hand. Reported after a stray `64x64-stage3b-f_theta_diagnostic.png`
    appeared in output/stage3b/ during a 128x128 run.

    `isolated_project_root` exists for this, but its docstring asserted that
    the evaluation/*.py scripts use _PYTHON_ROOT "only ever for CLI argument
    defaults ... never when called programmatically". That is false for TWELVE
    of them: check_f_theta computes its default output_path from _PYTHON_ROOT
    inside the function, which is exactly the path a test exercising the
    default-folder logic must take.

    A detector rather than a longer patch list: the list would drift the next
    time a script gains a default, and this catches whatever leaks by whatever
    route -- including one the fixture was never told about.
    """
    root = Path(__file__).resolve().parent.parent
    watched = [root.parent / "output", root / "checkpoints"]

    before = snapshot_files(watched)
    yield

    # A FILE STILL CHANGING AFTER THE TESTS HAVE FINISHED WAS NOT WRITTEN BY
    # THEM. Two snapshots a second apart at teardown separate a live external
    # writer -- a training run the person left going -- from a test leak,
    # whose writes are complete by the time the session ends.
    #
    # This detector fired four times on
    # output/stage3b/128x128-stage3b-loss_curve.png, and the four xdist workers
    # reported mtimes of 22:09:37, :47, :47 and :52 for it: still advancing,
    # long after any test had touched anything. Meanwhile the checkpoint and
    # registry beside it sat fixed at 22:09:33, a single save. Asking the
    # person to reason about that is a worse answer than measuring it.
    after = snapshot_files(watched)
    time.sleep(1.0)
    still_changing = {k for k, v in snapshot_files(watched).items()
                      if k in after and after[k] != v}
    added = files_written_between(before, after)
    if still_changing:
        _live = sorted(str(p.name) for p in still_changing)[:3]
        print(f"\n[leak detector: ignoring {len(still_changing)} file(s) still being "
              f"written after the session ended -- a concurrent training run, not a "
              f"test leak: {', '.join(_live)}]")
        added = [p for p in added if p not in still_changing]
    # The latent cache is a deliberate, self-invalidating artifact keyed on
    # encoder weights, so a test that populates it is not leaking state a
    # person has to reason about.
    added = [p for p in added if "latent_cache" not in p.parts]
    # ATTRIBUTION BY GRID SIZE. See TEST_GRID_SIZES: no test builds a 128x128
    # sweep, so 128x128-stage4-* is the person's own run whether or not it
    # happened to still be changing at teardown -- which is the case timing
    # cannot decide, since a real stage-4 run that FINISHED mid-session is
    # changing nothing by the time the session ends.
    #
    # Nameless siblings inherit: registry-stage4.csv carries no <N>x<N> prefix,
    # but the same run_lds_stage call wrote 128x128-stage4.pt beside it, so
    # attributing one and reporting the other would be incoherent. Only
    # NAMELESS files inherit; anything carrying a test grid size is still
    # reported on its own name.
    _theirs = [p for p in added if not _looks_like_a_test_artifact(p)]
    _their_dirs = {q.parent for q in _theirs}
    _theirs += [p for p in added
                 if p not in _theirs and p.parent in _their_dirs
                 and not re.match(r"\d+x\d+[-_.]", p.name)]
    if _theirs:
        print(f"\n[leak detector: ignoring {len(_theirs)} file(s) whose grid size is not one "
              f"the tests use ({sorted(TEST_GRID_SIZES)}) -- your own training run: "
              f"{', '.join(sorted(q.name for q in _theirs)[:3])}]")
        added = [p for p in added if p not in _theirs]
    if added:
        # mtimes included, and BOTH causes offered. The message used to assert
        # "A test is using a default path anchored to _PYTHON_ROOT", which was
        # simply false the first time it fired for real: the four files were
        # output/stage3b/128x128-stage3b-loss_curve.{png,csv},
        # checkpoints/stage3b/128x128-stage3b.pt and registry-stage3b.csv --
        # size 128, which NO test uses (they use 32, 64, and 8). They were the
        # person's own training run writing while the suite happened to be
        # running. A detector that names the wrong culprit sends the reader
        # hunting through tests for a bug that is not there.
        import datetime as _dt

        def _describe(path):
            try:
                when = _dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
            except OSError:
                when = "??:??:??"
            return f"{path.relative_to(root.parent)}  (modified {when})"

        listing = "\n  ".join(_describe(p) for p in added[:20])
        more = f"\n  ... and {len(added) - 20} more" if len(added) > 20 else ""
        pytest.fail(
            f"{len(added)} file(s) appeared or changed in the REAL project tree during "
            f"this test session:\n  {listing}{more}\n"
            f"TWO possible causes -- check the timestamps and the SIZE in the "
            f"filenames first:\n"
            f"  (a) a concurrent TRAINING RUN of your own. The tests only ever use "
            f"size 4/8/32/64, so anything named 128x128 or larger, or any "
            f"registry-stage*.csv, is almost certainly yours and not a leak.\n"
            f"  (b) a genuine leak: a test using a default path anchored to "
            f"_PYTHON_ROOT. Pass an explicit output_path/checkpoint_path, or add "
            f"the module to isolated_project_root -- and note that fixture does "
            f"NOT cover the evaluation/*.py scripts."
        )
