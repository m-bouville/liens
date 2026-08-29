"""Walk a checkpoint's stored ancestry (stage 4 -> 3b -> 2, and 3b -> 3a via
the registry) so a single checkpoint can be expanded into its whole lineage.

Standalone (torch + stdlib only) so it imports without the heavy comparison
stack -- reusable by compare_f_theta, the pipeline, or any lineage query.
"""
import re
import warnings
from pathlib import Path, PurePosixPath

import torch

_UNSET = object()


_STAGE_FROM_STEM = re.compile(r"stage\s*(\d+[ab]?)", re.I)
_STAGE_ORDER = {"stage 1": 0, "stage 1a": 1, "stage 2": 2, "stage 3a": 3,
                "stage 3b": 4, "stage 4": 5, "stage 5": 6}


def _stage_label(path) -> str | None:
    """'128x128-stage3b-20260828_07h53' -> 'stage 3b'. None if unrecognised."""
    m = _STAGE_FROM_STEM.search(Path(path).stem)
    return f"stage {m.group(1).lower()}" if m else None


def _ancestor_pointers(path: Path, device) -> dict:
    """The ancestor paths a checkpoint records ABOUT ITSELF (self-contained,
    travels with the file): ae_checkpoint -> stage-2 encoder, lds_checkpoint ->
    stage-3b f_theta (stage 4/5 joint only), resumed_from -> a prior 4/5."""
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: ck.get(k) for k in ("ae_checkpoint", "lds_checkpoint", "resumed_from")}


_TIMESTAMP_SUFFIX = re.compile(r"-\d{8}_\d{2}h\d{2}(?:m\d{2})?$")


def _base_stem(name: str) -> str:
    """Strip a trailing -YYYYMMDD_HHhMM timestamp so a TIMESTAMPED checkpoint
    (128x128-stage3b-20260828_07h53) matches the CANONICAL registry key
    (128x128-stage3b), which is how the registry records each stage."""
    return _TIMESTAMP_SUFFIX.sub("", PurePosixPath(str(name).replace("\\", "/")).stem)


def _default_registry_for(checkpoint_path: Path) -> Path | None:
    """The registry that sits NEXT TO a checkpoint: for
    .../stage3b/x.pt -> .../stage3b/registry-stage3b.csv. This is where a
    stage's resume link (3b -> 3a) is recorded, so --with-ancestors can find 3a
    with no explicit --registry."""
    stage = _stage_label(checkpoint_path)
    if stage is None:
        return None
    tag = stage.replace("stage ", "stage")            # 'stage 3b' -> 'stage3b'
    cand = checkpoint_path.parent / f"registry-{tag}.csv"
    return cand if cand.exists() else None


def _registry_resume_of(checkpoint_path: Path, registry_path=None) -> str | None:
    """The resume ancestor (3b -> 3a) recorded in the registry CSV -- the ONE
    lineage link the checkpoint file itself doesn't store. registry_path=None
    falls back to the registry sitting next to the checkpoint. Returns None (and
    prints WHY) if there is no registry, no matching row, or no resume field."""
    checkpoint_path = Path(checkpoint_path)
    if registry_path is None:
        registry_path = _default_registry_for(checkpoint_path)
    if registry_path is None or not Path(registry_path).exists():
        print(f"  [lineage] no registry for {checkpoint_path.name} "
              f"(looked for registry-<stage>.csv next to it) -- 3a not resolved.")
        return None
    from orchestration.checkpoint_registry import _read_registry
    _, rows = _read_registry(Path(registry_path))
    def _name(pth):                                   # split on \\ or / regardless of OS
        return PurePosixPath(str(pth).replace("\\", "/")).name
    target = str(checkpoint_path.resolve())
    tname = _name(checkpoint_path)
    row = None
    for r in rows:                                    # 1) exact resolved-path match
        rp = r.get("checkpoint_path")
        if rp and str(Path(str(rp).replace("\\", "/")).resolve()) == target:
            row = r
            break
    if row is None:                                   # 2) fall back to filename match
        for r in rows:
            rp = r.get("checkpoint_path")
            if rp and _name(rp) == tname:
                row = r
                break
    if row is None:                                   # 3) timestamp-stripped base
        tbase = _base_stem(tname)
        for r in rows:
            rp = r.get("checkpoint_path")
            if rp and _base_stem(_name(rp)) == tbase:
                row = r
                print(f"  [lineage] matched {tname} to registry key "
                      f"'{_name(rp)}' by base name (timestamp stripped).")
                break
    if row is None:
        print(f"  [lineage] {tname} not found in {Path(registry_path).name} "
              f"({len(rows)} rows) -- 3a not resolved. (path form mismatch?)")
        return None
    resumed = row.get("resumed_from")
    if not resumed:
        print(f"  [lineage] {tname} has no 'resumed_from' in the registry "
              f"(trained without resuming a 3a?) -- 3a not resolved.")
        return None
    return resumed


_STAGE_DIR_RE = re.compile(r"stage\d+[ab]?$", re.I)


def _rebase_to_root(stored: str, checkpoints_root) -> Path | None:
    """A stored pointer is an ABSOLUTE path from wherever the checkpoint was
    trained (often another machine, e.g. 'D:\\...\\checkpoints\\stage2\\x.pt').
    If it does not exist as-is, re-base it to <checkpoints_root>/stage<N>/<file>
    -- the project's own checkpoints/stageN/ layout, same convention the
    registry resolves by. Handles both '\\' and '/' separators so a Windows
    path resolves on POSIX and vice versa. Returns the found path, else None."""
    if checkpoints_root is None:
        return None
    norm = PurePosixPath(str(stored).replace("\\", "/"))
    filename = norm.name
    stage_dir = next((part for part in norm.parts
                      if _STAGE_DIR_RE.fullmatch(part)), None)
    root = Path(checkpoints_root)
    for cand in ([root / stage_dir / filename] if stage_dir else []) + [root / filename]:
        if cand.exists():
            return cand
    return None


def _infer_checkpoints_root(checkpoint_path: Path) -> Path | None:
    """<root>/stage<N>/file.pt -> <root>. None if the input isn't under a
    stageN directory (e.g. a checkpoint sitting in tests/data)."""
    parent = checkpoint_path.parent
    return parent.parent if _STAGE_DIR_RE.fullmatch(parent.name) else None


def resolve_lineage(checkpoint_path, select=None, registry_resume=None,
                     checkpoints_root=_UNSET, device="cpu"):
    """Walk a checkpoint's ancestry to its stage-2 encoder root.

    Reads each checkpoint's OWN pointers (ae_checkpoint -> 2, lds_checkpoint ->
    3b, resumed_from -> prior 4/5). The 3b -> 3a link is NOT in the checkpoint;
    pass registry_resume (a path -> path|None callable) to supply it from the
    registry, else 3a is simply absent.

    Consistency: every checkpoint names a stage-2 ancestor, and they must all be
    the SAME encoder (different encoders = not one lineage) -- raises ValueError
    otherwise. Missing ancestor files (a canonical path a later run overwrote)
    are skipped with a warning, never fatal.

    select: keep only these stage labels (e.g. ["2","3a","3b"], bare numbers ok);
    the input checkpoint itself is always kept. None keeps all.

    Returns [(stage_label, Path), ...] oldest -> newest, INCLUDING the input.
    """
    checkpoint_path = Path(checkpoint_path)
    # An EXPLICIT checkpoints_root means "resolve against THIS tree" -- it takes
    # priority over each pointer's stored absolute path (which may point at
    # another machine, or at real-but-unrelated checkpoints on THIS one). With
    # no root given, fall back to inferring one and trying the stored path first.
    explicit_root = checkpoints_root is not _UNSET and checkpoints_root is not None
    if checkpoints_root is _UNSET or checkpoints_root is None:
        checkpoints_root = _infer_checkpoints_root(checkpoint_path)
    lineage: dict = {}
    stage2_by_source: dict = {}

    def _exists(path, named_by):
        if explicit_root:
            rebased = _rebase_to_root(path, checkpoints_root)
            if rebased is not None:
                return rebased
        pth = Path(str(path).replace("\\", "/")) if "\\" in str(path) else Path(path)
        if pth.exists():
            return pth
        if not explicit_root:
            rebased = _rebase_to_root(path, checkpoints_root)
            if rebased is not None:
                return rebased
        warnings.warn(
            f"{named_by} names ancestor '{PurePosixPath(str(path).replace(chr(92), '/')).name}' "
            f"but it is missing on disk (a path from another machine, or a "
            f"canonical name a later run overwrote?) -- skipping it. Pass "
            f"checkpoints_root to re-base stored paths to your own tree.")
        return None

    root_label = _stage_label(checkpoint_path) or "input"
    lineage[root_label] = checkpoint_path
    queue = [(checkpoint_path, root_label)]
    visited: set = set()
    while queue:
        path, label = queue.pop(0)
        rp = str(path.resolve())
        if rp in visited:
            continue
        visited.add(rp)
        ptrs = _ancestor_pointers(path, device)
        if ptrs["ae_checkpoint"]:
            p2 = _exists(ptrs["ae_checkpoint"], label)
            if p2 is not None:
                stage2_by_source[label] = p2.resolve()
                lineage.setdefault(_stage_label(p2) or "stage 2", p2)
        for key in ("lds_checkpoint", "resumed_from"):
            if ptrs[key]:
                pa = _exists(ptrs[key], label)
                if pa is not None:
                    lbl = _stage_label(pa) or key
                    lineage.setdefault(lbl, pa)
                    queue.append((pa, lbl))
        if registry_resume and label == "stage 3b":
            p3a = registry_resume(path)
            if p3a:
                pa = _exists(p3a, "stage 3b (registry)")
                if pa is not None:
                    lbl = _stage_label(pa) or "stage 3a"
                    lineage.setdefault(lbl, pa)
                    queue.append((pa, lbl))

    distinct = {p.resolve() for p in stage2_by_source.values()}
    if len(distinct) > 1:
        detail = "; ".join(f"{k} -> {v.name}" for k, v in stage2_by_source.items())
        warnings.warn(
            f"FORKED lineage: the checkpoints reference DIFFERENT stage-2 "
            f"encoders ({detail}). This is a real fork -- a stage refined or "
            f"paired a different encoder than an ancestor used -- not a clean "
            f"single-encoder lineage. The rollout comparison is still valid "
            f"(each model uses its OWN encoder+f_theta); the stage-2 baseline "
            f"is taken from the FIRST stage-2 seen. If unintended, check which "
            f"encoder each stage was launched with.")

    if select is not None:
        want = {s if str(s).startswith("stage") else f"stage {str(s).lower()}"
                for s in select}
        lineage = {k: v for k, v in lineage.items()
                   if k in want or v == checkpoint_path}

    return sorted(lineage.items(), key=lambda kv: _STAGE_ORDER.get(kv[0], 99))


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m evaluation.lineage",
        description="Print a checkpoint's ancestry lineage (stage 4/5 -> 3b -> "
                    "2, and 3b -> 3a via the registry). Reads each checkpoint's "
                    "own stored ancestor pointers; --root re-bases stale/off-"
                    "machine paths.")
    ap.add_argument("checkpoint", type=Path,
                    help="the checkpoint to trace (e.g. a stage-4/5 .pt)")
    ap.add_argument("--root", type=Path, default=None,
                    help="checkpoints root to re-base stored ancestor paths "
                         "against (use when the checkpoint was trained elsewhere "
                         "or the tree moved); inferred when omitted")
    ap.add_argument("--registry", type=Path, default=None,
                    help="registry CSV, to also resolve the 3b -> 3a link the "
                         "checkpoint file itself does not store")
    ap.add_argument("--select", nargs="*", default=None, metavar="STAGE",
                    help="keep only these stages, e.g. --select 2 3b (default: all)")
    args = ap.parse_args(argv)

    kw = {}
    if args.root is not None:
        kw["checkpoints_root"] = args.root
    if args.registry is not None:
        kw["registry_resume"] = lambda pth: _registry_resume_of(pth, args.registry)

    lineage = resolve_lineage(args.checkpoint, select=args.select, **kw)
    print(f"lineage of {Path(args.checkpoint).name} (oldest -> newest):")
    for label, path in lineage:
        print(f"  {label:9s} {path}")
    return lineage


if __name__ == "__main__":
    _main()
