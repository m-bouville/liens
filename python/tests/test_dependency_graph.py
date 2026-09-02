"""Guards for tests/data/dependency_graph.py -- the checked-in snapshot of
the project's internal import graph.

Two properties, both cheap and both load-bearing for using the snapshot to
reason about locality and refactor blast radius:

  1. STALENESS: the stored `imports` matches what the source actually imports.
     Without this the snapshot silently rots and every conclusion drawn from it
     (this module is local, that rename is safe) is drawn from fiction.

  2. INVERSE (f o f = Id): `imported_by` is exactly the transpose of `imports`.
     The two maps are stored separately for readability, so they can disagree;
     this pins that they don't.
"""
import importlib.util
import pathlib

from _import_graph import compute_imports, invert

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_snapshot():
    spec = importlib.util.spec_from_file_location(
        "dependency_graph", _ROOT / "tests" / "data" / "dependency_graph.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.dict_key_imports_values, mod.dict_key_imported_by_values


def test_the_snapshot_matches_the_actual_source_imports():
    """The stored `imports` is what the source really imports right now. When
    this fails, a module was added/removed or its imports changed -- update the
    named keys in tests/data/dependency_graph.py (both maps)."""
    stored_imports, _ = _load_snapshot()
    actual = compute_imports(_ROOT)
    assert stored_imports == actual, (
        "dependency_graph.py is stale -- regenerate on THIS tree with "
        "`python tests/_import_graph.py`, or edit the named keys by hand.\n"
        "Differences (module: stored -> actual):\n"
        + "\n".join(
            f"  {m}: {sorted(stored_imports.get(m, set()))} -> {sorted(actual.get(m, set()))}"
            for m in sorted(set(stored_imports) | set(actual))
            if stored_imports.get(m, set()) != actual.get(m, set()))
    )


def test_imported_by_is_the_exact_inverse_of_imports():
    """f o f = Id: imported_by is the transpose of imports, no more, no less."""
    stored_imports, stored_imported_by = _load_snapshot()
    assert invert(stored_imports) == stored_imported_by


def test_inverting_twice_returns_the_original():
    """The property stated directly: invert(invert(imports)) == imports (over
    the same key set), so the two maps carry identical information."""
    stored_imports, _ = _load_snapshot()
    assert invert(invert(stored_imports)) == {k: v for k, v in stored_imports.items()}


def test_training_loop_is_local_to_training():
    """The concrete decision this data was built to settle: _training_loop is
    imported only from within training/, so its leading-underscore name is
    warranted (all its importers share its directory)."""
    _, imported_by = _load_snapshot()
    key = "training/_training_loop.py"
    importers = imported_by[key]
    assert importers, "nothing imports _training_loop -- dead code?"
    assert all(m.split("/")[0] == "training" for m in importers), (
        f"_training_loop is imported from outside training/: "
        f"{sorted(m for m in importers if not m.startswith('training/'))}"
    )


def test_incremental_update_matches_a_full_recompute():
    """update_imports re-parses only the named modules but must land on the same
    graph a full recompute would -- otherwise 'update just the changed keys' could
    silently diverge from reality. Feed it a deliberately-wrong starting point for
    two modules and confirm re-parsing just those two repairs it exactly."""
    from _import_graph import compute_imports, update_imports
    full = compute_imports(_ROOT)
    changed = sorted(full)[:2]
    corrupted = {k: (set() if k in changed else set(v)) for k, v in full.items()}
    repaired = update_imports(_ROOT, corrupted, set(changed))
    assert repaired == full, "re-parsing the changed modules did not reproduce the true graph"


# --------------------------------------------------------------------------
# the two analysis tools + parser-completeness guards
# --------------------------------------------------------------------------
from _import_graph import (imported_by_nobody, local_modules, compute_imports,
                           ENTRY_SCRIPTS)


def test_local_modules_are_imported_only_within_their_own_dir():
    """The tool's contract: every module it returns has all its importers in its
    own directory (so a leading-underscore name is warranted)."""
    imports, _ = _load_snapshot()
    ib = {}
    for imp, tgts in imports.items():
        for t in tgts:
            ib.setdefault(t, set()).add(imp)
    for m in local_modules(imports):
        d = m.rsplit("/", 1)[0]
        assert all(x.rsplit("/", 1)[0] == d for x in ib[m]), m
    assert "training/_training_loop.py" in local_modules(imports)


def test_imported_by_nobody_excludes_entry_scripts():
    """main.py imports the pipeline but nothing imports main.py -- it must NOT be
    reported as dead. The tool excludes ENTRY_SCRIPTS for exactly this reason."""
    imports, _ = _load_snapshot()
    nobody = imported_by_nobody(imports)
    assert not (nobody & set(ENTRY_SCRIPTS))


def test_relative_imports_are_captured():
    """models use `from .blocks import ...`; the graph must record that edge, or
    blocks.py falsely reads as imported-by-nobody (dead). Regression guard for
    the relative-import parser."""
    imports = compute_imports(_ROOT)
    assert "models/blocks.py" in imports["models/encoder.py"], (
        "relative import `from .blocks` in encoder.py was not captured"
    )


def test_from_package_import_module_is_captured():
    """`from utils import load_datasets` (module imported from its package) must
    be recorded -- it was the widest missed edge (load_datasets is imported by
    ~20 modules). Regression guard."""
    imports = compute_imports(_ROOT)
    users = [m for m, t in imports.items() if "utils/load_datasets.py" in t]
    assert len(users) > 5, f"expected many importers of load_datasets, got {users}"


def test_entry_script_edges_are_captured():
    """main.py -> orchestration/pipeline.py must be in the graph, or pipeline
    (imported only by main) falsely reads as imported-by-nobody. Regression guard
    for the entry-script scan."""
    imports = compute_imports(_ROOT)
    assert "orchestration/pipeline.py" in imports.get("main.py", set())
