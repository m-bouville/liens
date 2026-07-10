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
# plausibly grow a new instance of it: the orchestrator, training
# scripts, and evaluation/diagnostic scripts. NOT models/ or utils/,
# which don't build checkpoint/output paths themselves (checked: no
# hits there either, but they're a different KIND of module -- pure
# architecture/IO helpers -- with no obvious reason to ever need this
# pattern, unlike training/evaluation scripts which all save something).
_SCAN_DIRS = ["training", "evaluation"]
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
    anywhere) must define its own _PYTHON_ROOT -- catches a file being
    given a fixed relative-string replacement without ever gaining the
    anchor those replacements depend on (a plain NameError at import
    time in practice, but worth failing here with a clearer message
    than that)."""
    anchor_pattern = re.compile(r"_PYTHON_ROOT\s*=\s*Path\(__file__\)\.resolve\(\)")
    missing = []
    for path in _project_python_files():
        text = path.read_text()
        builds_a_path = ('"checkpoints"' in text) or ('"output"' in text) \
            or ("/ \"checkpoints\"" in text) or ("/ \"output\"" in text)
        if builds_a_path and not anchor_pattern.search(text):
            missing.append(str(path.relative_to(_PYTHON_ROOT)))
    assert not missing, (
        "these files reference checkpoints/output paths but never define "
        "_PYTHON_ROOT:\n  " + "\n  ".join(missing)
    )
