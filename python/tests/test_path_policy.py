"""
Regression guard for the CWD-relative-path bug that recurred multiple
times on this project (train_ae.py/train_lds.py, check_latent_channels.py's
CLI default, compare_integrators.py's docstring not even matching its own
default, and 3 more instances caught by this very test's first draft --
see conftest.py's own _PYTHON_ROOT for the established fix pattern).

Static/textual, not behavioral: doesn't run any of the scripts (most need
torch, which this environment doesn't have -- see other test files' own
"can't be imported directly" notes), just scans their SOURCE for the
banned construct. That's enough to catch the actual bug class (a bare
relative string literal standing in for a proper anchor), and doesn't
need a real checkpoint or a real dataset directory to run.

Deliberately excludes tests/ itself: test fixtures/inputs legitimately
contain relative-looking path STRINGS as test data (e.g. parse_fixed_window
test cases), which is a different thing entirely from a script's own
default output location.
"""
import re
from pathlib import Path

# tests/test_path_policy.py -> python/
_PYTHON_ROOT = Path(__file__).resolve().parent.parent

# Every directory that has historically had this bug, or could
# plausibly grow a new instance of it: the orchestrator (now split into
# orchestration/, see main.py's own refactor), training scripts, and
# evaluation/diagnostic scripts. NOT models/ or utils/, which don't
# build checkpoint/output paths themselves (checked: no hits there
# either, but they're a different KIND of module -- pure architecture/
# IO helpers -- with no obvious reason to ever need this pattern,
# unlike training/evaluation/orchestration scripts which all save
# something).
_SCAN_DIRS = ["orchestration", "training", "evaluation"]
_SCAN_FILES = ["main.py"]

# Matches Path("..  or  Path(f"..  -- a bare relative-string literal
# passed straight into Path(), starting with '..'. This is deliberately
# narrow: it does NOT flag `some_anchor / "output" / f"..."` (no `Path(`
# call at all there, just tuple/string args to `/`), which is exactly
# the fixed pattern -- so this test can't be satisfied by nervously
# avoiding the literal text `Path(` while still building an
# unanchored path some other way. It's a starting point, not a
# guarantee: a sufficiently creative new bad pattern (e.g. os.path.join
# with "..") wouldn't be caught. Good enough for the bug class actually
# seen on this project.
_BANNED_PATTERN = re.compile(r'Path\(\s*f?"\.\.')


def _project_python_files():
    files = [_PYTHON_ROOT / name for name in _SCAN_FILES]
    for d in _SCAN_DIRS:
        files.extend(sorted((_PYTHON_ROOT / d).glob("*.py")))
    return files


def test_scan_target_files_actually_exist():
    """Guards the guard: if the directory layout ever changes and
    _SCAN_DIRS/_SCAN_FILES silently stop matching anything, the tests
    below would trivially and misleadingly pass with zero files
    checked. Fails loudly instead."""
    files = _project_python_files()
    assert len(files) >= 5, (
        f"expected several .py files under {_SCAN_DIRS + _SCAN_FILES}, found "
        f"{len(files)} -- _PYTHON_ROOT or _SCAN_DIRS may be wrong: {_PYTHON_ROOT}"
    )


def test_no_bare_relative_checkpoint_or_output_paths():
    """THE regression test: no script builds a checkpoint/output/dataset
    path from a bare relative string literal like Path("../output/...").
    Every one of these must instead be built from a _PYTHON_ROOT-style
    anchor (see training/train_refinement.py, the first file to use this
    pattern, and this file's own module docstring)."""
    offenders = []
    for path in _project_python_files():
        text = path.read_text()
        for match in _BANNED_PATTERN.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(_PYTHON_ROOT)}:{line_no}")
    assert not offenders, (
        "found bare relative-string Path(...) construction(s) -- should be "
        "anchored via _PYTHON_ROOT instead:\n  " + "\n  ".join(offenders)
    )


def test_scripts_that_define_output_paths_have_the_anchor():
    """Every scanned file that actually constructs a checkpoint/output
    path (i.e. references "checkpoints" or "output" as a path component
    anywhere) must have access to a real _PYTHON_ROOT-style anchor --
    EITHER by defining its own (the original per-file pattern, still
    used by utils/paths.py itself and every training/evaluation
    script), OR by importing it from utils.paths (the pattern
    main.py and the rest of the orchestration/ package use instead,
    specifically so there's ONE shared anchor rather than N
    independently-computed copies that could drift apart -- see
    utils/paths.py's own docstring). Catches a file being given
    a fixed relative-string replacement without ever gaining a working
    anchor either way (a plain NameError/ImportError in practice, but
    worth failing here with a clearer message than that)."""
    anchor_pattern = re.compile(r"_PYTHON_ROOT\s*=\s*Path\(__file__\)\.resolve\(\)")
    shared_anchor_import = re.compile(r"from orchestration\.paths import\b")
    missing = []
    for path in _project_python_files():
        text = path.read_text()
        builds_a_path = ('"checkpoints"' in text) or ('"output"' in text) \
            or ("/ \"checkpoints\"" in text) or ("/ \"output\"" in text)
        has_anchor = bool(anchor_pattern.search(text)) or bool(shared_anchor_import.search(text))
        if builds_a_path and not has_anchor:
            missing.append(str(path.relative_to(_PYTHON_ROOT)))
    assert not missing, (
        "these files reference checkpoints/output paths but never define or "
        "import _PYTHON_ROOT:\n  " + "\n  ".join(missing)
    )


def test_cached_sweeps_live_under_pytest_basetemp_not_an_uncleaned_tempdir(tmp_path):
    """
    REGRESSION: conftest.cached_sweep originally built each shared sweep
    into a bare tempfile.mkdtemp(), which NOTHING ever cleans up -- one
    leaked directory per distinct sweep per test run, permanently (154
    orphaned sweep_* directories had accumulated before this was
    caught). Anchoring them under pytest's own tmp_path_factory root
    instead keeps the cache's whole purpose (a sweep must outlive any
    single test's own tmp_path so it can be shared) while restoring
    pytest's normal retention/pruning.

    Asserts the sweep root is inside pytest's own basetemp, which is
    what makes it subject to that pruning.
    """
    import conftest

    assert conftest._SWEEP_ROOT is not None, (
        "the session-scoped _sweep_cache_root fixture did not run -- cached_sweep would "
        "fall back to an UNCLEANED tempfile.mkdtemp()"
    )

    # Exercise cached_sweep FOR REAL and check where it actually put the
    # directory. Asserting on _SWEEP_ROOT alone would be vacuous: that
    # stays correctly set even if cached_sweep ignores it and calls
    # tempfile.mkdtemp() anyway -- which is exactly the bug, and exactly
    # what an earlier version of this test failed to catch.
    key = ("test_path_policy_leak_probe", (), ())
    try:
        base = conftest.cached_sweep(key, lambda d: d)
    finally:
        conftest._SWEEP_CACHE.pop(key, None)

    # tmp_path and the sweep root are both under pytest's own per-session
    # basetemp (tmp_path is <basetemp>/<test name>N, the sweep root is
    # <basetemp>/cached_sweepsN), so a correctly-placed sweep sits
    # somewhere beneath that same basetemp.
    basetemp = tmp_path.parent
    assert base.is_relative_to(basetemp), (
        f"cached_sweep built into {base}, outside pytest's basetemp ({basetemp}) -- "
        f"nothing will ever prune it"
    )


def test_cached_sweep_reuses_one_build_per_key():
    """The cache's actual purpose -- one build per distinct key, not one
    per call. Guards against a future change that silently rebuilds every
    time (correct, but re-introduces the ~21s the cache exists to save)."""
    import conftest

    builds = []

    def builder(base):
        builds.append(base)
        return base

    key = ("test_path_policy_probe", (), ())
    try:
        first = conftest.cached_sweep(key, builder)
        second = conftest.cached_sweep(key, builder)
        assert first == second
        assert len(builds) == 1, f"builder ran {len(builds)}x for one key -- cache not working"
    finally:
        conftest._SWEEP_CACHE.pop(key, None)
