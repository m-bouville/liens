"""Compute the internal import graph over the package tree.

Shared by tests/data/dependency_graph.py (the checked-in snapshot) and
tests/test_dependency_graph.py (which regenerates and compares). Kept out of
the package tree itself -- it is test/tooling infrastructure, not shipped code.

Only imports BETWEEN the project's own package modules are recorded; stdlib and
third-party imports are ignored. Test files are excluded: a module imported only
to be tested is still architecturally local, so counting tests/ as an importer
would make every leaf look cross-dir.

CLI:
    python tests/_import_graph.py
        Regenerate the whole snapshot from the current tree.
    python tests/_import_graph.py training/foo.py training/bar.py
        Re-parse ONLY those modules and rewrite the snapshot -- for the common
        case of adding a module or changing one module's imports, without
        re-parsing the whole tree. Pass a removed module's path too: it is
        dropped from both maps and purged from every other module's sets.
"""
from __future__ import annotations

import ast
import pathlib
import sys

PACKAGES = ("training", "models", "evaluation", "utils", "orchestration")

# Top-level entry scripts: they live outside every package dir, so they are not
# import TARGETS (nothing imports them), but they DO import package modules and
# those edges matter -- without main.py, orchestration/pipeline.py (imported only
# by main) falsely reads as imported-by-nobody. Scanned as importer-only nodes.
# An explicit list, not a top-level glob: a glob would sweep in scratch files,
# editor copies and session aliases that are not part of the project.
ENTRY_SCRIPTS = ("main.py",)


def _module_map(root: pathlib.Path):
    """(modpaths, dotted): 'training/x.py' -> Path, and 'training.x' -> 'training/x.py'."""
    modpaths = {}
    for pkg in PACKAGES:
        d = root / pkg
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.py")):
            if p.stem != "__init__":
                modpaths[f"{pkg}/{p.name}"] = p
    dotted = {k[:-3].replace("/", "."): k for k in modpaths}
    return modpaths, dotted


def _imports_of(key: str, path: pathlib.Path, dotted: dict[str, str]) -> set[str]:
    """The project modules that the module at `path` (graph key `key`) imports.

    Handles absolute (`import training.x`, `from training.x import y`,
    `from training import x`) AND relative (`from .blocks import y`,
    `from . import blocks`) forms. Relative imports resolve against the module's
    own package -- e.g. `from .blocks` in models/encoder.py means models.blocks.
    The packages here are one level deep, so only level-1 relatives target a
    project module; deeper levels resolve above the package root and are ignored.
    """
    own_pkg = key.rsplit("/", 1)[0] if "/" in key else ""
    found: set[str] = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8", errors="ignore"))):
        if isinstance(n, ast.ImportFrom):
            if n.level and n.level == 1 and own_pkg:
                if n.module:                             # from .blocks import y
                    cand = f"{own_pkg}.{n.module}"
                    if cand in dotted:
                        found.add(dotted[cand])
                else:                                     # from . import blocks
                    for a in n.names:
                        cand = f"{own_pkg}.{a.name}"
                        if cand in dotted:
                            found.add(dotted[cand])
            elif n.module in dotted:                      # from training.x import y
                found.add(dotted[n.module])
            elif n.module and not n.level:                # from training import x
                for a in n.names:
                    cand = f"{n.module}.{a.name}"
                    if cand in dotted:
                        found.add(dotted[cand])
        elif isinstance(n, ast.Import):                   # import training.x
            for a in n.names:
                if a.name in dotted:
                    found.add(dotted[a.name])
    return found


def compute_imports(root: pathlib.Path) -> dict[str, set[str]]:
    """`{module_path: {module_paths it imports}}` over the project's own modules,
    plus each ENTRY_SCRIPTS file as an importer-only node."""
    modpaths, dotted = _module_map(root)
    imports = {key: _imports_of(key, p, dotted) for key, p in modpaths.items()}
    for script in ENTRY_SCRIPTS:
        p = root / script
        if p.exists():
            imports[script] = _imports_of(script, p, dotted)
    return imports


def invert(imports: dict[str, set[str]]) -> dict[str, set[str]]:
    """`{module_path: {module_paths that import it}}` -- the transpose. A module
    whose importers are all in its own directory is local to that directory."""
    imported_by: dict[str, set[str]] = {k: set() for k in imports}
    for importer, targets in imports.items():
        for t in targets:
            imported_by.setdefault(t, set()).add(importer)
    return imported_by


def update_imports(root: pathlib.Path, imports: dict[str, set[str]],
                   changed: set[str]) -> dict[str, set[str]]:
    """Return `imports` with only the `changed` modules re-parsed.

    Handles all three cases without touching unchanged modules: a module whose
    imports changed is re-parsed; a NEW module is added; a REMOVED module (its
    file gone) is dropped and purged from every other module's set, so no dangling
    edge survives. imported_by is derived (cheaply) by inverting the result -- it
    is never updated in place, so it cannot drift from imports.
    """
    _, dotted = _module_map(root)
    imports = {k: set(v) for k, v in imports.items()}   # copy, don't mutate caller's
    for path in changed:
        full = root / path
        if full.exists():
            imports[path] = _imports_of(path, full, dotted)
        else:
            imports.pop(path, None)
    existing = set(imports)
    return {k: {v for v in vs if v in existing} for k, vs in imports.items()}


# --------------------------------------------------------------------------
# analysis tools
# --------------------------------------------------------------------------
def _dir_of(module_path: str) -> str:
    """Directory of a module path; '' for a top-level entry script."""
    return module_path.rsplit("/", 1)[0] if "/" in module_path else ""


def imported_by_nobody(imports: dict[str, set[str]]) -> set[str]:
    """Modules that nothing imports. Either DEAD code, or a CLI entry point run
    directly (`python -m evaluation.check_x`, or main.py). A package module under
    training/models/utils/orchestration that nobody imports is the suspicious
    kind -- those are libraries, meant to be imported; an evaluation/ tool or an
    ENTRY_SCRIPTS file is expected here."""
    imported_by = invert(imports)
    return {m for m, importers in imported_by.items()
            if not importers and m not in ENTRY_SCRIPTS}


def local_modules(imports: dict[str, set[str]]) -> set[str]:
    """Modules imported ONLY by files in their own directory -- local
    subroutines, candidates for a leading-underscore name. A module imported by
    nobody is NOT local (it has no importers to be same-dir with)."""
    imported_by = invert(imports)
    return {m for m, importers in imported_by.items()
            if importers and all(_dir_of(x) == _dir_of(m) for x in importers)}


def _report(imports: dict[str, set[str]]) -> str:
    """Human-readable summary of the two tools, for the CLI."""
    nobody = imported_by_nobody(imports)
    suspicious = sorted(m for m in nobody if _dir_of(m) not in ("evaluation", ""))
    cli_like = sorted(m for m in nobody if _dir_of(m) == "evaluation")
    lines = ["", "IMPORTED BY NOBODY:",
             f"  library modules (check: dead, or CLI?): {suspicious or 'none'}",
             f"  evaluation/ (expected -- run via python -m): {len(cli_like)} tool(s)",
             "LOCAL SUBROUTINES (imported only within their own dir):"]
    for m in sorted(local_modules(imports)):
        under = "" if m.rsplit("/", 1)[-1].startswith("_") else "   <- not _-prefixed"
        lines.append(f"  {m}{under}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# snapshot file I/O
# --------------------------------------------------------------------------
_IMPORTS_VAR = "dict_key_imports_values"
_IMPORTED_BY_VAR = "dict_key_imported_by_values"

_HEADER = '''"""Checked-in snapshot of the project's INTERNAL import graph (its own package
modules only; stdlib/third-party and test imports excluded). This file is DATA
-- it defines two dicts and imports nothing; the module-path strings are keys and
values, not import statements.

  dict_key_imports_values[m]      = {{modules m imports}}
  dict_key_imported_by_values[m]  = {{modules that import m}}    (the transpose)

USES
  - dict_key_imported_by_values[m] with all values in m's own directory  => m is
    LOCAL to that directory (a candidate for a leading-underscore name).
  - a diff of dict_key_imports_values before/after a change is the refactor's
    blast radius.

MAINTENANCE (see tests/_import_graph.py)
  Add a module or change its imports, then rewrite this file with ONLY the
  affected keys re-parsed:
      python tests/_import_graph.py training/changed_one.py training/changed_two.py
  or regenerate the whole thing:
      python tests/_import_graph.py
  Generate it on the tree it will guard -- a snapshot made against a different
  file set is stale by construction. test_dependency_graph.py fails if this
  drifts from the real source, or if the two maps stop being inverses.
"""
'''


def _format(d: dict[str, set[str]]) -> str:
    out = ["{"]
    for k in sorted(d):
        vs = sorted(d[k])
        if not vs:
            out.append(f"    {k!r}: set(),")
        else:
            out.append(f"    {k!r}: {{")
            out.extend(f"        {v!r}," for v in vs)
            out.append("    },")
    out.append("}")
    return "\n".join(out)


def _write(out_path: pathlib.Path, imports: dict[str, set[str]]) -> None:
    out_path.write_text(
        _HEADER + f"\n{_IMPORTS_VAR} = " + _format(imports)
        + f"\n\n{_IMPORTED_BY_VAR} = " + _format(invert(imports)) + "\n",
        encoding="utf-8")


def _load_stored(out_path: pathlib.Path) -> dict[str, set[str]]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_snapshot", out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, _IMPORTS_VAR)


if __name__ == "__main__":
    _root = pathlib.Path(__file__).resolve().parent.parent
    _out = _root / "tests" / "data" / "dependency_graph.py"
    _changed = set(sys.argv[1:])
    if _changed and _out.exists():
        _imports = update_imports(_root, _load_stored(_out), _changed)
        print(f"updated {len(_changed)} module(s): {', '.join(sorted(_changed))}")
    else:
        _imports = compute_imports(_root)
        print(f"regenerated whole snapshot ({len(_imports)} modules)")
    _write(_out, _imports)
    print(f"wrote {_out.relative_to(_root)}")
    print(_report(_imports))
