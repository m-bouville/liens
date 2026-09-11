"""
Backfill missing per-component val losses in eval-stageN.csv from the run .logs.

The eval log (eval-stageN.csv, one row per checkpoint+epoch) may have EMPTY
component columns for a checkpoint whose training predates val_components being
saved in the checkpoint dict. But the run's .log carries the per-epoch component
breakdown (it is written incrementally from epoch 1). This tool scans a stage's
directory for .log files, parses their per-epoch val-component breakdown, and
fills the blank component cells in eval-stageN.csv -- recording where each value
came from in a `components_source` column so a filled value is never mistaken for
an original and a still-blank one is explained.

Usage:
    python -m evaluation.backfill_eval_components checkpoints/stage3a
    python -m evaluation.backfill_eval_components checkpoints/stage3a --eval-csv output/eval-stage3a.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# header:  "/N train = [w*]name/scale +[w*]name/scale ... | valid ..."
_HEADER_RE = re.compile(r"train\s*=\s*(.+?)\s*\|")
_COMPONENT_RE = re.compile(r"([A-Za-z0-9_]+)\s*/\s*[0-9eE.+\-]+")
# epoch line: leading int, then "... | <valid section> | <ema>"
_EPOCH_RE = re.compile(r"^\s*(\d+)[|\s]")     # epoch is "N|" (format A) or "N " (format B)
# a "total = a + b + c" breakdown
_BREAKDOWN_RE = re.compile(r"=\s*([0-9eE.+\-]+(?:\s*\+\s*[0-9eE.+\-]+)*)")


def _valid_section(line: str) -> str | None:
    """Return the VALIDATION part of an epoch line, across both log formats.

    Format A ("N| train=... | valid=... | ema"): >=3 pipe fields, the valid
    section is the one just before the ema (parts[-2]).
    Format B ("N  train(1s), val(1s) | ema"): exactly 1 pipe; train and val are
    comma-separated before it, so the valid section is the text after the last
    comma of parts[0]. Returns None if neither shape matches.
    """
    parts = line.split("|")
    if len(parts) >= 3:                      # A: ... | valid | ema
        return parts[-2]
    if len(parts) == 2:                      # B: "N train(1s), val(1s)" | ema
        head = parts[0]
        return head.rsplit(",", 1)[1] if "," in head else None
    return None


def parse_log(path: Path):
    """Return (component_names, {epoch: {name: val}}) parsed from one .log.

    component_names come from the header's `train = ...` term order. Each epoch
    line's VALID section (the part between the first '|' and the ema) is split on
    '=' into its component values and mapped to those names in order. A line with
    no '=' breakdown in the valid section (single-component run) maps its lone
    valid total to the first (only) component name.
    """
    names: list[str] = []
    per_epoch: dict[int, dict] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.rstrip("\r")
        if not names:
            h = _HEADER_RE.search(line)
            if h and ("/" in h.group(1)):
                names = _COMPONENT_RE.findall(h.group(1))
                continue
        if line.lstrip().startswith("ref"):            # pre-run reference line
            continue
        m = _EPOCH_RE.match(line)
        if not m or "|" not in line:
            continue
        epoch = int(m.group(1))
        valid_section = _valid_section(line)
        if valid_section is None:
            continue
        bd = _BREAKDOWN_RE.search(valid_section)
        if bd:
            vals = [float(x) for x in re.split(r"\s*\+\s*", bd.group(1))]
        else:                                          # single component: lone total
            nums = re.findall(r"[0-9eE.+\-]+", valid_section.split("(")[0])
            vals = [float(nums[0])] if nums else []
        if not vals:
            continue
        use_names = names if len(names) == len(vals) else [f"comp{i}" for i in range(len(vals))]
        per_epoch[epoch] = {n: v for n, v in zip(use_names, vals)}
    return names, per_epoch


_STAMP_RE = re.compile(r"(.*)-(\d{8})_(\d{2})h(\d{2})$")   # <prefix>-YYYYMMDD_HHhMM


def _stamp(stem: str):
    """(prefix, YYYYMMDD, HH, MM) from a stage stem, or None if it has no stamp."""
    m = _STAMP_RE.match(stem)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)), int(m.group(4))


def _stamp_key(stem: str):
    """Sortable (YYYYMMDD, HH, MM) for a stamped stem, or None. Prefix-agnostic
    key used to order runs in time."""
    st = _stamp(stem)
    if st is None:
        return None
    _pre, date, hh, mm = st
    return (date, hh, mm)


def confident_log_for(ck_stem: str, logs, epoch: int, parsed):
    """The log that belongs to THIS checkpoint, using the true timing relation.

    A checkpoint's stamp is its BACKUP mtime; a log's stamp is its run's START.
    A run's log is therefore always stamped AT OR BEFORE the checkpoint that run
    produced (the backup happens when a LATER run overwrites, i.e. after this run
    finished). So the owning log is the SAME-prefix log with the greatest start
    time that is still <= the checkpoint's mtime -- the most recent run to have
    started by backup time. A log stamped AFTER the checkpoint is a different,
    later run and cannot own it.

    Consistency check (the caller's point that one log spans all its epochs): the
    owning run backed up at `epoch`, so its log must REACH `epoch`. If the best
    candidate does not contain `epoch`, refuse (return None) rather than fill from
    a run that never got there.

    Exact stem match wins outright. Returns the log Path or None (refuse).
    """
    exact = [lg for lg in logs if lg.stem == ck_stem]
    if len(exact) == 1:
        return exact[0]
    cs = _stamp(ck_stem)
    if cs is None:
        return None
    c_pre = cs[0]
    c_key = _stamp_key(ck_stem)
    # same-prefix logs whose START is at or before this checkpoint's mtime
    candidates = []
    for lg in logs:
        ls = _stamp(lg.stem)
        if ls is None or ls[0] != c_pre:
            continue
        lk = _stamp_key(lg.stem)
        if lk is not None and lk <= c_key:
            candidates.append((lk, lg))
    if not candidates:
        return None
    candidates.sort()                       # latest start <= mtime = best owner
    best = candidates[-1][1]
    # the owning run must have reached this epoch
    return best if epoch in parsed(best) else None


def _row_missing_components(row: dict) -> bool:
    """A row needs backfill if it has NO val_<component> column filled."""
    comp_cols = [k for k in row if k.startswith("val_") and k != "val_loss"]
    return not comp_cols or all(not (row.get(c) or "").strip() for c in comp_cols)


def backfill(stage_dir: Path, eval_csv: Path, dry_run: bool = False):
    if not eval_csv.exists():
        print(f"No eval CSV at {eval_csv} -- nothing to backfill "
              f"(expected beside registry-{stage_dir.name}.csv).")
        return
    with open(eval_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]

    # rows that actually need filling -- drive off the CSV, not the logs
    todo = [r for r in rows if _row_missing_components(r) and (r.get("epoch") or "").strip()]
    skipped_no_epoch = sum(1 for r in rows if _row_missing_components(r)
                           and not (r.get("epoch") or "").strip())
    if not todo:
        print("No rows need component backfill"
              + (f" ({skipped_no_epoch} missing-component row(s) have no epoch to match on)"
                 if skipped_no_epoch else "") + ".")
        return

    # parse logs LAZILY and QUIETLY: cache by stem, and a by-epoch index built only
    # as needed. No blanket dump of every log.
    logs = sorted(stage_dir.glob("*.log"))
    _cache: dict[str, dict] = {}
    def _parsed(stem_path: Path):
        if stem_path.stem not in _cache:
            _cache[stem_path.stem] = parse_log(stem_path)[1]
        return _cache[stem_path.stem]

    filled = 0
    for row in todo:
        epoch = int(row["epoch"])
        ck_stem = Path(row.get("checkpoint_path", "")).stem
        comps, src = None, None
        # CONFIDENT match only: the log that provably belongs to THIS checkpoint
        # (exact stem, or same date + time within tolerance, uniquely). No
        # by-epoch guessing across unrelated runs -- that fabricated 09/09
        # components from an 11/09 log. Better an honest blank than a wrong fill.
        _tag = f"{ck_stem} @ epoch {epoch}"
        match = confident_log_for(ck_stem, logs, epoch, _parsed)
        if match is None:
            _with_epoch = [lg.name for lg in logs if epoch in _parsed(lg)]
            _hint = (f" ({len(_with_epoch)} log(s) contain epoch {epoch} but none is "
                     f"the run that produced this checkpoint: {_with_epoch})"
                     if _with_epoch else " (no same-prefix log started at/before this "
                     f"checkpoint's mtime that reaches epoch {epoch})")
            print(f"  {_tag}: NOT filled -- no confident log match{_hint}")
            continue
        comps, src = _parsed(match)[epoch], match.name
        for name, val in comps.items():
            col = f"val_{name}"
            if col not in fieldnames:
                fieldnames.append(col)
            row[col] = str(val)
        if "components_source" not in fieldnames:
            fieldnames.append("components_source")
        row["components_source"] = src
        filled += 1
        print(f"  {_tag}: found in {src} -> {comps}")

    print(f"{filled}/{len(todo)} row(s) filled"
          + (f", {skipped_no_epoch} skipped (no epoch)" if skipped_no_epoch else "")
          + (" (dry run -- not written)" if dry_run else ""))
    if dry_run or not filled:
        return
    tmp = eval_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fieldnames})
    tmp.replace(eval_csv)
    print(f"Wrote {eval_csv}")


def main():
    ap = argparse.ArgumentParser(description="Backfill eval-stageN.csv component losses from .logs.")
    ap.add_argument("stage_dir", type=Path, help="e.g. checkpoints/stage3a")
    ap.add_argument("--eval-csv", type=Path, default=None,
                    help="defaults to output/eval-<stage>.csv from the stage dir name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stage = args.stage_dir.name
    eval_csv = args.eval_csv
    if eval_csv is None:
        # search common locations rather than assume one: beside the stage dir,
        # in the stage dir itself, and under output/.
        candidates = [
            args.stage_dir / f"eval-{stage}.csv",          # beside registry-<stage>.csv (default)
            args.stage_dir.parent.parent / "output" / f"eval-{stage}.csv",
            Path.cwd() / f"eval-{stage}.csv",
        ]
        eval_csv = next((c for c in candidates if c.exists()), candidates[0])
    print(f"Stage dir: {args.stage_dir}\nEval CSV:  {eval_csv}"
          + ("" if eval_csv.exists() else "  (NOT FOUND -- pass --eval-csv <path>; searched: "
             + ", ".join(str(c) for c in (candidates if args.eval_csv is None else [])) + ")"))
    backfill(args.stage_dir, eval_csv, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
