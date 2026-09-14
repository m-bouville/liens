"""
Backfill missing per-component val losses in runs-stageN.csv from the run .logs.

The runs ledger (runs-stageN.csv, one row per checkpoint+epoch+variant) may have EMPTY
component columns for a checkpoint whose training predates val_components being
saved in the checkpoint dict. But the run's .log carries the per-epoch component
breakdown (it is written incrementally from epoch 1). This tool scans a stage's
directory for .log files, parses their per-epoch val-component breakdown, and
fills the blank component cells in runs-stageN.csv -- recording where each value
came from in a `components_source` column so a filled value is never mistaken for
an original and a still-blank one is explained.

Usage:
    python -m evaluation.backfill_eval_components checkpoints/stage3a
    python -m evaluation.backfill_eval_components checkpoints/stage3a --eval-csv checkpoints/stage3a/runs-stage3a.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

from utils.eval_log import (params_from_checkpoint, reconcile_fieldnames,
                            PARAM_COLUMNS, LOSS_COLUMNS, SOURCE_COLUMNS,
                            canonical_checkpoint_key, _DYNAMICS_ONLY_PARAMS,
                            _DYNAMICS_PARAM_STAGES, _stage_of, columns_in_scope)

# Path anchor (policy: default checkpoint/output paths resolve from the repo
# root, never the process CWD). backfill's inputs are CLI args, but the
# output/ fallback is anchored here rather than derived CWD-relatively.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent   # python/evaluation/X.py -> python/

# header:  "/N train = [w*]name/scale +[w*]name/scale ... | valid ..."
_HEADER_RE = re.compile(r"train\s*=\s*(.+?)\s*\|")
_COMPONENT_RE = re.compile(r"([A-Za-z0-9_]+)\s*/\s*[0-9eE.+\-]+")
# full term: optional "weight*", then name/scale. Captures weight (None -> 1.0) and scale,
# so a contribution (weight*raw/scale, what the breakdown prints) can be UN-weighted to
# the raw loss: raw = contribution * scale / weight.
_TERM_RE = re.compile(r"(?:([0-9eE.+\-]+)\s*\*\s*)?([A-Za-z0-9_]+)\s*/\s*([0-9eE.+\-]+)")
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
    ws: dict[str, tuple[float, float]] = {}   # name -> (weight, scale) from the header
    per_epoch: dict[int, dict] = {}
    saved_at: dict[int, str] = {}    # epoch -> "HH:MM" from the "-> saved at HH:MM" annotation
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.rstrip("\r")
        if not names:
            h = _HEADER_RE.search(line)
            if h and ("/" in h.group(1)):
                names = _COMPONENT_RE.findall(h.group(1))
                for w, n, sc in _TERM_RE.findall(h.group(1)):
                    ws[n] = (float(w) if w else 1.0, float(sc))
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
        # The breakdown prints weight*raw/scale (contributions summing to the total).
        # Store the RAW loss (contribution * scale / weight) so the ledger's components
        # are comparable across runs with different weight/scale hyperparameters. A
        # component with no header term (single-component old logs) is left as-is.
        comps = {}
        for n, v in zip(use_names, vals):
            if n in ws:
                w, sc = ws[n]
                # An INACTIVE term (weight 0, or a printed contribution of exactly 0)
                # gives no information about its raw loss: the breakdown prints 0
                # regardless of what the raw value was. Storing 0 would read as a
                # perfect (zero) loss -- false. Leave it unknown (None -> blank cell).
                if w == 0.0 or v == 0.0:
                    comps[n] = None
                else:
                    comps[n] = v * sc / w
            else:
                comps[n] = v if v != 0.0 else None   # single/unnamed: same zero rule
        per_epoch[epoch] = comps
        _sv = re.search(r"saved at (\d{2}):(\d{2})", line)
        if _sv:
            saved_at[epoch] = f"{_sv.group(1)}h{_sv.group(2)}"   # normalise to HHhMM
    return names, per_epoch, saved_at


_STAMP_RE = re.compile(r"(.*)-(\d{8})_(\d{2})h(\d{2})$")   # <prefix>-YYYYMMDD_HHhMM


def _stamp(stem: str):
    """(prefix, YYYYMMDD, HH, MM) from a stage stem, or None if it has no stamp."""
    m = _STAMP_RE.match(stem)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)), int(m.group(4))


def parse_log_params(path: Path) -> dict:
    """Training params from a log's HEADER block (the `key=value` lines before the
    epoch table). The checkpoint's config only holds ARCHITECTURE params; the
    TRAINING hyperparameters (lr, n_rollout_steps, dynamics_mode, stats0_predict_weight,
    ...) are printed here, so this is the source for "the rest" backfill fills.
    Collects every key=value token, dropping '-- comment' tails and quotes, up to
    the first epoch/header line."""
    out = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.rstrip("\r")
        if re.match(r"^\s*/\d|^\s*\d+[\s|]", line):   # epoch header (/N ...) or epoch line -> stop
            break
        # prose provenance lines (not key=value): the frozen encoder (stage-2 source,
        # with its latent_channels) and the resume checkpoint. These carry the source
        # .pt paths, which are NOT in the checkpoint config nor the param block.
        # stage-3: "Loaded frozen encoder from <path> (epoch N, val_loss=X, latent_channels=C)"
        # stage-4/5: "Stage 4: loaded E/D/stats_head from <path> (...), f_theta from <path2> (...)"
        # Both name the stage-2 source; only the wording differs.
        _enc = (re.search(r"Loaded frozen encoder from (\S+)", line)
               or re.search(r"loaded E/D(?:/stats_head)? from (\S+)", line))
        if _enc:
            out["source_stage2"] = _enc.group(1).rstrip(":,;)")
            _lc = re.search(r"latent_channels=(\d+)", line)
            if _lc:
                out["latent_channels"] = _lc.group(1)   # -> p_latent_channels (authoritative for the run)
        _f_theta = re.search(r"f_theta from (\S+)", line)   # stage-4/5's second source
        if _f_theta:
            out["source_stage3"] = _f_theta.group(1).rstrip(":,;)")
        _res = re.search(r"Resuming from (\S+)", line)
        if _res:
            out["resume_from"] = _res.group(1).rstrip(":,;)")
        head = line.split("--")[0]                     # drop '-- comment' tails
        # Skip PROSE lines that merely contain '=' (lists, embedded parens, or
        # "key=value: sentence"): they corrupt the scan (e.g. stat_names=['angle',
        # head_hidden=64), key=value: explanation). A genuine param line is only
        # key=value tokens; prose has quotes/brackets or a ': ' explanation tail.
        # Skip PROSE: a genuine param-block line is only "key=value" tokens joined by
        # whitespace. Prose that happens to contain "key=value" also has brackets,
        # parens, or sentence punctuation (';', ', ', ': ', '.') around it -- e.g.
        # "...residual head (head_hidden=64); H is zero-init." Any of those markers on
        # the line means it is prose, not the param block.
        # Skip PROSE, but do not blanket-reject every '(' or ',' -- a legitimate
        # parenthetical parameter group like "(a0=1.0, b=1.0, kappa=0.2)" must still
        # be scanned (it was being dropped wholesale, losing allen_cahn_a0 etc.).
        # Prose is: a list ('['), or sentence punctuation (';'/':'/'.') followed by a
        # space, UNLESS every '(...)' span on the line is itself a clean
        # comma-separated key=value list -- in which case commas/parens inside it
        # don't count as prose markers.
        _stripped = re.sub(r"\(([^()]*)\)", lambda m: "" if
                           re.fullmatch(r"\s*\w+\s*=\s*\S+?\s*(,\s*\w+\s*=\s*\S+?\s*)*",
                                        m.group(1)) else m.group(0), head)
        if "[" in _stripped or re.search(r"[;:.]\s", _stripped) or "(" in _stripped:
            continue
        # clean scalar OR quoted scalar: dynamics_mode='deriv_linear', lr=0.002.
        # (Quotes are allowed for the VALUE; a stray "'" mid-word in prose is excluded
        # because such lines are the ': sentence' / '[' cases already skipped above.)
        for k, v in re.findall(r"(\w+)=('[^']*'|\"[^\"]*\"|[\w.+\-]+)", head):
            if k == "latent_channels" and "latent_channels" in out:
                continue                               # keep the prose-captured clean value
            out[k] = v.strip("'\"")

    # De-alias: a bare name from a parenthetical summary group (e.g. "(a0=1.0,
    # b=1.0, kappa=0.2, M=0.05, phi_max=1.5)") duplicates a fully-qualified name
    # printed elsewhere in the SAME block (e.g. "allen_cahn_a0=1.0" in "other
    # parameters:") -- same physical parameter, two names, which produced BOTH
    # p_a0 and p_allen_cahn_a0 in the ledger. Prefer the qualified name (it is
    # self-describing) and drop the bare alias when its qualified counterpart is
    # also present in this block.
    for bare, prefixed in (("a0", "allen_cahn_a0"), ("b", "allen_cahn_b"),
                           ("kappa", "allen_cahn_kappa"), ("M", "allen_cahn_mobility"),
                           ("phi_max", "allen_cahn_phi_max")):
        if bare in out and prefixed in out:
            del out[bare]
    return out


def _stamp_key(stem: str):
    """Sortable (YYYYMMDD, HH, MM) for a stamped stem, or None. Prefix-agnostic
    key used to order runs in time."""
    st = _stamp(stem)
    if st is None:
        return None
    _pre, date, hh, mm = st
    return (date, hh, mm)


def confident_log_for(ck_stem: str, logs, epoch: int, parsed):
    """The log that produced THIS checkpoint, using the log's own record.

    A checkpoint's stem stamp is when IT was created (the save's wall-clock);
    a log's stamp is when the run STOPPED. Same run: checkpoint save <= run end,
    so the checkpoint stem's HHhMM equals the "-> saved at HHhMM" the log printed
    for that very epoch. That annotation is the authoritative identity -- match on
    it, not on name-timestamp proximity.

    Order of confidence:
      1) exact stem match (rare -- stamps differ), else
      2) a log whose `epoch N` line is annotated `saved at <ckpt HHhMM>` AND same
         prefix -- the log literally says it made this save. Unique -> use it.
      3) refuse (None). No proximity guessing.
    parsed(log) -> (per_epoch, saved_at).
    """
    exact = [lg for lg in logs if lg.stem == ck_stem]
    if len(exact) == 1:
        # Exact stem match is a strong identity signal, but NOT a guarantee this
        # log reached the requested epoch (e.g. a live/unbackuped file queried at
        # an epoch later than what got logged) -- confirm before returning, or the
        # caller's per_epoch[epoch] lookup raises KeyError on a log that matched by
        # name but doesn't have that epoch.
        if epoch in parsed(exact[0])[0]:
            return exact[0]
        return None
    cs = _stamp(ck_stem)
    if cs is None:
        return None
    c_pre, _c_date, c_hh, c_mm = cs
    ck_hhmm = f"{c_hh:02d}h{c_mm:02d}"
    matches = []
    for lg in logs:
        ls = _stamp(lg.stem)
        if ls is None or ls[0] != c_pre:          # must be same prefix (size/stage)
            continue
        _pe, saved_at = parsed(lg)
        if saved_at.get(epoch) == ck_hhmm:        # the log says: this epoch saved at ck's time
            matches.append(lg)
    return matches[0] if len(matches) == 1 else None


def _row_missing_params(row: dict, stage: str = "") -> bool:
    """A row needs param backfill if any IN-SCOPE param or source column is empty.
    Scoped to the CSV's stage via columns_in_scope: a column a stage cannot have
    (dynamics params on stage 4/5, source_stage3 on stage 3) is NOT demanded --
    else the row is reported unfilled every run, chasing a column that can never
    be filled. Source columns come from the log too, so a row with p_* filled but
    a source blank still needs the log pass."""
    wanted = columns_in_scope(list(PARAM_COLUMNS) + list(SOURCE_COLUMNS), stage)
    return any(not (row.get(c) or "").strip() for c in wanted)


def _params_from_row_checkpoint(row: dict, stage_dir: Path):
    """Load the row's checkpoint (by stem, under stage_dir) and return its input
    params via the SAME extractor the diagnostics use -- so backfilled params match
    live-logged ones exactly. None if the .pt is absent (a deleted backup) -- then
    the row's params stay blank, honestly, rather than fabricated."""
    ck_file = stage_dir / (Path(row.get("checkpoint_path", "")).stem + ".pt")
    if not ck_file.exists():
        return None
    try:
        ckpt = torch.load(ck_file, map_location="cpu", weights_only=True)
    except Exception:
        return None
    return params_from_checkpoint(ckpt)


def _row_missing_components(row: dict) -> bool:
    """A row needs component backfill if it has NO per-component loss column filled.
    Per-component columns are the stage-variable val_<name> ones -- i.e. val_*
    columns that are NOT the fixed total(s) in LOSS_COLUMNS (val_loss)."""
    comp_cols = [k for k in row if k.startswith("val_") and k not in LOSS_COLUMNS]
    return not comp_cols or all(not (row.get(c) or "").strip() for c in comp_cols)


def collapse_redundant_empty_variant_rows(eval_csv: Path, dry_run: bool = False) -> int:
    """Legacy cleanup for the pre-fill-rule split: an empty-variant row whose data
    is entirely a subset of a tagged (eval-metric) row for the SAME checkpoint+epoch
    is redundant -- compare_f_theta used to spawn a variant twin instead of filling
    the existing row. Drop such empty-variant rows. An empty-variant row with NO
    tagged sibling (its own unique params/components) is KEPT. Returns rows removed."""
    if not eval_csv.exists():
        return 0
    with open(eval_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]

    def _key(r):
        # canonical key, not a partial .replace -- so absolute vs relative path
        # spellings of the same checkpoint group together (matches upsert_eval_row).
        return (canonical_checkpoint_key(r.get("checkpoint_path", "")),
                (r.get("epoch") or "").strip())

    def _data(r):
        return {k: v for k, v in r.items()
                if (v or "").strip() and k not in ("checkpoint_path", "epoch", "eval_variant")}

    drop = set()
    from collections import defaultdict
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[_key(r)].append(i)
    for idxs in groups.values():
        empt = [i for i in idxs if not (rows[i].get("eval_variant") or "").strip()]
        tagged = [i for i in idxs if (rows[i].get("eval_variant") or "").strip()]
        def _cell_matches(a, b):
            # numeric cells written by different code paths can differ only in
            # float formatting (7.357999802 vs 7.357999801635742) -- compare with
            # tolerance; fall back to exact string match for non-numeric cells.
            a, b = (a or "").strip(), (b or "").strip()
            if a == b:
                return True
            try:
                fa, fb = float(a), float(b)
                return abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb))
            except ValueError:
                return False

        for e in empt:
            edata = _data(rows[e])
            if tagged and any(all(_cell_matches(rows[t].get(k, ""), v)
                                  for k, v in edata.items()) for t in tagged):
                drop.add(e)

    if not drop:
        print("collapse-empty-variant: no redundant empty-variant rows found.")
        return 0
    print(f"collapse-empty-variant: removing {len(drop)} redundant empty-variant row(s) "
          f"(subsumed by a tagged sibling)" + (" (dry run -- not written)" if dry_run else ""))
    if not dry_run:
        kept = [r for i, r in enumerate(rows) if i not in drop]
        tmp = eval_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in kept:
                w.writerow({c: r.get(c, "") for c in fieldnames})
        tmp.replace(eval_csv)
        print(f"Wrote {eval_csv}")
    return len(drop)


_ALIAS_PAIRS = (("p_a0", "p_allen_cahn_a0"), ("p_b", "p_allen_cahn_b"),
               ("p_kappa", "p_allen_cahn_kappa"), ("p_M", "p_allen_cahn_mobility"),
               ("p_phi_max", "p_allen_cahn_phi_max"))


def drop_out_of_scope_columns(eval_csv: Path, dry_run: bool = False) -> int:
    """Legacy cleanup: drop stage-scoped columns (currently the dynamics-only
    p_dynamics_mode/p_derivative_source/p_derivative_time) that reconcile_fieldnames
    used to add to every stage's header before it became stage-aware. Only drops a
    column that is EMPTY on every row -- a column with real data is never removed
    even if it is out of scope for the stage (that would be silent data loss)."""
    if not eval_csv.exists():
        return 0
    with open(eval_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    stage = _stage_of(eval_csv.name)
    if stage in _DYNAMICS_PARAM_STAGES:
        print(f"drop-out-of-scope-columns: {eval_csv.name} is a dynamics stage -- nothing to drop.")
        return 0
    candidates = [c for c in _DYNAMICS_ONLY_PARAMS if c in fieldnames]
    dropped = [c for c in candidates if all(not (r.get(c) or "").strip() for r in rows)]
    kept_with_data = [c for c in candidates if c not in dropped]
    if kept_with_data:
        print(f"drop-out-of-scope-columns: {kept_with_data} are out of scope for "
              f"{stage} but hold data -- NOT dropped (would lose it).")
    if not dropped:
        print("drop-out-of-scope-columns: nothing to drop.")
        return 0
    print(f"drop-out-of-scope-columns: removing empty out-of-scope column(s) {dropped}"
          + (" (dry run -- not written)" if dry_run else ""))
    if not dry_run:
        new_fields = [c for c in fieldnames if c not in dropped]
        tmp = eval_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in new_fields})
        tmp.replace(eval_csv)
        print(f"Wrote {eval_csv}")
    return len(dropped)


def drop_aliased_param_columns(eval_csv: Path, dry_run: bool = False) -> int:
    """Legacy cleanup: a bare-name column (p_a0, p_b, p_kappa, ...) duplicates a
    fully-qualified column (p_allen_cahn_a0, ...) written before the parser's
    de-alias fix -- same physical parameter parsed twice from a parenthetical
    summary group and the flat param block. Drop the bare-name COLUMN entirely
    (not per-row) once every row's value in it either matches the qualified
    column or is blank. Returns the number of columns dropped."""
    if not eval_csv.exists():
        return 0
    with open(eval_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]

    def _matches(a, b):
        a, b = (a or "").strip(), (b or "").strip()
        if not a:
            return True                # blank bare cell is always fine to drop
        if a == b:
            return True
        try:
            return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
        except ValueError:
            return False

    dropped = []
    for bare, qualified in _ALIAS_PAIRS:
        if bare not in fieldnames or qualified not in fieldnames:
            continue
        if all(_matches(r.get(bare, ""), r.get(qualified, "")) for r in rows):
            dropped.append(bare)

    if not dropped:
        print("drop-aliased-params: no aliased columns found.")
        return 0
    print(f"drop-aliased-params: removing column(s) {dropped} (duplicate of their "
          f"p_allen_cahn_* counterpart)" + (" (dry run -- not written)" if dry_run else ""))
    if not dry_run:
        new_fields = [c for c in fieldnames if c not in dropped]
        tmp = eval_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in new_fields})
        tmp.replace(eval_csv)
        print(f"Wrote {eval_csv}")
    return len(dropped)


def blank_zero_components(eval_csv: Path, dry_run: bool = False) -> int:
    """Legacy cleanup: a component cell stored as exactly 0 came from an INACTIVE
    term (weight 0 / zero printed contribution) under the old un-weighting, which
    wrongly recorded 0 instead of unknown. No real loss is exactly 0. Blank those
    cells so they read as unknown. Returns the number of cells blanked."""
    if not eval_csv.exists():
        return 0
    with open(eval_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    n = 0
    for r in rows:
        for c in list(r):
            if c.startswith("val_") and c not in LOSS_COLUMNS:
                v = (r.get(c) or "").strip()
                try:
                    if v and float(v) == 0.0:
                        r[c] = ""; n += 1
                except ValueError:
                    pass
    print(f"blank-zero-components: {n} exact-zero component cell(s) blanked"
          + (" (dry run -- not written)" if dry_run else ""))
    if n and not dry_run:
        tmp = eval_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in fieldnames})
        tmp.replace(eval_csv)
        print(f"Wrote {eval_csv}")
    return n


def prune_stale_baseline_rows(eval_csv: Path, dry_run: bool = False) -> int:
    """Remove stale baseline rows: a 'baseline:*' row (or a stage2_z0z1dt variant)
    that carries a model epoch. Baselines are independent of the dynamics model, so
    the current code keys them with a BLANK epoch; any baseline row WITH an epoch was
    written by an older version that wrongly keyed on the model's epoch, so the same
    baseline appears once per evaluated model instead of once per encoder. Dropping
    them lets the next compare_f_theta run rewrite the single correct (blank-epoch)
    row. Returns the number removed."""
    if not eval_csv.exists():
        return 0
    with open(eval_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]

    def _is_stale_baseline(r):
        is_baseline = (r.get("checkpoint_path", "").startswith("baseline:")
                       or "stage2_z0z1dt" in (r.get("eval_variant", "") or ""))
        has_epoch = bool((r.get("epoch", "") or "").strip())
        return is_baseline and has_epoch

    kept = [r for r in rows if not _is_stale_baseline(r)]
    removed = len(rows) - len(kept)
    if removed:
        print(f"prune-baselines: removing {removed} stale baseline row(s) "
              f"(baseline rows that carry a model epoch)"
              + (" (dry run -- not written)" if dry_run else ""))
    else:
        print("prune-baselines: no stale baseline rows found.")
    if removed and not dry_run:
        tmp = eval_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in kept:
                w.writerow({c: r.get(c, "") for c in fieldnames})
        tmp.replace(eval_csv)
        print(f"Wrote {eval_csv}")
    return removed


def backfill(stage_dir: Path, eval_csv: Path, dry_run: bool = False, debug: bool = False,
             all_pts: bool = False):
    if not eval_csv.exists():
        if not all_pts:
            print(f"No eval CSV at {eval_csv} -- nothing to backfill "
                  f"(expected beside registry-{stage_dir.name}.csv). Use --all to "
                  f"seed it from the .pt files in {stage_dir}.")
            return
        fieldnames, rows = ["checkpoint_path", "epoch"], []
    else:
        with open(eval_csv, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]

    # --all: SEED a row for every .pt in the stage dir not already in the CSV, so an
    # empty/partial ledger is populated from what is on disk. Epoch is read from each
    # checkpoint (its own 'epoch' field), which is also the upsert key -- so re-running
    # --all never duplicates. Backups excluded? No: every .pt is a real checkpoint
    # worth a row; the fill passes then add its params/components.
    if all_pts:
        # dedup on the CANONICAL key so a row stored under another spelling
        # (backslashes / absolute path, from an older writer) is still recognised.
        existing = {canonical_checkpoint_key(r.get("checkpoint_path", "")) for r in rows}
        seeded = 0
        for pt in sorted(stage_dir.glob("*.pt")):
            cp = canonical_checkpoint_key(f"checkpoints/{stage_dir.name}/{pt.name}")
            if cp in existing:
                continue
            try:
                _ck = torch.load(pt, map_location="cpu", weights_only=True)
                _ep = _ck.get("epoch")
            except Exception as _e:
                print(f"  --all: skip {pt.name} (could not read epoch: {_e})")
                continue
            _new = {"checkpoint_path": cp,
                    "epoch": "" if _ep is None else str(int(_ep))}
            _new["_seeded"] = True          # transient marker, dropped before write
            rows.append(_new)
            existing.add(cp)
            seeded += 1
        print(f"--all: seeded {seeded} new row(s) from {stage_dir}/*.pt "
              f"({len(list(stage_dir.glob('*.pt')))} .pt total)")
    else:
        seeded = 0

    # A "baseline:*" row is a methodology reference, not a checkpoint -- it has no
    # .pt and no owning log epoch, and is complete as written. Exclude it from both
    # passes so it is not reported as a spurious fill failure.
    def _is_checkpoint_row(r):
        return not str(r.get("checkpoint_path", "")).startswith("baseline:")
    # rows that need filling -- driven off the CSV. Components need an epoch (to
    # match a log line); params only need the checkpoint file (no epoch required).
    todo = [r for r in rows if _is_checkpoint_row(r) and _row_missing_components(r)
            and (r.get("epoch") or "").strip()]
    _stage = _stage_of(eval_csv.name)   # scope the missing-check to this CSV's stage
    param_todo = [r for r in rows if _is_checkpoint_row(r) and _row_missing_params(r, _stage)]
    skipped_no_epoch = sum(1 for r in rows if _row_missing_components(r)
                           and not (r.get("epoch") or "").strip())
    if not todo and not param_todo:
        print("No rows need backfill (components or params)"
              + (f"; {skipped_no_epoch} missing-component row(s) have no epoch to match on"
                 if skipped_no_epoch else "") + ".")
        return

    # parse logs LAZILY and QUIETLY: cache by stem, and a by-epoch index built only
    # as needed. No blanket dump of every log.
    logs = sorted(stage_dir.glob("*.log"))
    _cache: dict[str, tuple] = {}
    def _parsed(stem_path: Path):
        """(per_epoch, saved_at) for a log, cached."""
        if stem_path.stem not in _cache:
            _n, _pe, _sv = parse_log(stem_path)
            _cache[stem_path.stem] = (_pe, _sv)
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
            _with_epoch = [lg.name for lg in logs if epoch in _parsed(lg)[0]]
            _hint = (f" ({len(_with_epoch)} log(s) contain epoch {epoch} but none is "
                     f"the run that produced this checkpoint: {_with_epoch})"
                     if _with_epoch else " (no same-prefix log started at/before this "
                     f"checkpoint's mtime that reaches epoch {epoch})")
            print(f"  {_tag}: NOT filled -- no confident log match{_hint}")
            continue
        _pe_match, _ = _parsed(match)
        if epoch not in _pe_match:
            # Defense in depth: confident_log_for is supposed to guarantee this, but
            # never trust a single-sited guarantee -- report and skip rather than
            # KeyError if it's ever violated (e.g. by a future edit to the matcher).
            print(f"  {_tag}: matched {match.name} but it lacks epoch {epoch} -- "
                  f"left blank (this indicates a matcher bug; please report)")
            continue
        comps, src = _pe_match[epoch], match.name
        for name, val in comps.items():
            col = f"val_{name}"
            if col not in fieldnames:
                fieldnames.append(col)
            row[col] = "" if val is None else str(val)   # inactive term -> blank, not "None"
        if "log" not in fieldnames:
            fieldnames.append("log")
        row["log"] = src
        # components recovered from the log are un-weighted to RAW (see parse_log)
        if "val_components_kind" not in fieldnames:
            fieldnames.append("val_components_kind")
        if not (row.get("val_components_kind") or "").strip():
            row["val_components_kind"] = "raw"
        filled += 1
        print(f"  {_tag}: found in {src} -> {comps}")

    # --- params pass: architecture params from the checkpoint config + TRAINING
    #     params from the matched log's header block (the checkpoint config has only
    #     architecture; training hyperparameters are printed in the log). ---
    p_filled = 0
    for row in param_todo:
        _tag = f"{Path(row.get('checkpoint_path','')).stem}"
        pr = dict(_params_from_row_checkpoint(row, stage_dir) or {})   # architecture (p_*)
        n_cfg = len(pr)
        epoch_s = (row.get("epoch") or "").strip()
        n_log = 0
        _match = None
        if epoch_s:                                    # training params from the log block
            _match = confident_log_for(_tag, logs, int(epoch_s), _parsed)
            if _match is not None:
                _lp = parse_log_params(_match)
                for k, v in _lp.items():
                    if k in ("source_stage2", "source_stage3", "resume_from"):
                        pr.setdefault(k, v)            # provenance columns (not p_)
                    else:
                        pr.setdefault(f"p_{k}", v)     # config wins on overlap
                n_log = len(pr) - n_cfg
                if debug:
                    print(f"    [debug] {_tag} @ {epoch_s}: matched {_match.name}; "
                          f"log source_stage2={_lp.get('source_stage2')!r}")
            elif debug:
                _cands = [lg.name for lg in logs if int(epoch_s) in _parsed(lg)[0]]
                print(f"    [debug] {_tag} @ {epoch_s}: NO confident log match "
                      f"(epoch present in: {_cands})")
        if not pr:
            print(f"  {_tag}: params NOT filled -- no checkpoint .pt and no matched log")
            continue
        wrote = 0                                     # cells this call actually FILLS (blank -> value)
        for k, v in pr.items():
            if k not in fieldnames:
                fieldnames.append(k)
            if not (row.get(k) or "").strip():        # only fill blanks; don't clobber
                row[k] = str(v)
                wrote += 1
        if wrote:
            p_filled += 1
            _src = f"{n_cfg} config" + (f" + {n_log} log" if n_log else "")
            if row.pop("_seeded", False):     # brand-new row from --all
                print(f"  {_tag}: created with {wrote} param field(s) ({_src})")
            else:
                print(f"  {_tag}: {wrote} param cell(s) filled  "
                      f"[{len(pr)} available ({_src}), rest already present]")
        # silent when nothing was added (wrote == 0): don't clutter with no-op rows

    print(f"{filled}/{len(todo)} component-row(s) filled; {p_filled}/{len(param_todo)} "
          f"param-row(s) filled"
          + (f"; {skipped_no_epoch} skipped (no epoch)" if skipped_no_epoch else "")
          + (" (dry run -- not written)" if dry_run else ""))
    if dry_run or (not filled and not p_filled and not seeded):
        return
    for _r in rows:
        _r.pop("_seeded", None)          # drop the transient seed marker
    fieldnames = reconcile_fieldnames(fieldnames, eval_csv.name)   # add missing canonical headers
    tmp = eval_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fieldnames})
    tmp.replace(eval_csv)
    print(f"Wrote {eval_csv}")


def main():
    ap = argparse.ArgumentParser(description="Backfill runs-stageN.csv component losses from .logs.")
    ap.add_argument("stage_dir", type=Path, help="e.g. checkpoints/stage3a")
    ap.add_argument("--eval-csv", type=Path, default=None,
                    help="defaults to output/eval-<stage>.csv from the stage dir name")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--drop-out-of-scope-columns", action="store_true",
                    help="drop empty stage-out-of-scope columns (e.g. dynamics params on stage 4/5), then exit")
    ap.add_argument("--drop-aliased-params", action="store_true",
                    help="drop bare-name param columns duplicating a p_allen_cahn_* counterpart, then exit")
    ap.add_argument("--collapse-empty-variant", action="store_true",
                    help="drop empty-variant rows subsumed by a tagged eval row, then exit")
    ap.add_argument("--blank-zero-components", action="store_true",
                    help="legacy cleanup: blank component cells stored as exactly 0 (inactive terms), then exit")
    ap.add_argument("--prune-baselines", action="store_true",
                    help="remove stale baseline rows that carry a model epoch, then exit")
    ap.add_argument("--all", dest="all_pts", action="store_true",
                    help="seed a row for every .pt in the stage dir (create the "
                         "CSV if absent), then fill -- for initial population.")
    ap.add_argument("--debug", action="store_true",
                    help="per-row trace of which log matched each checkpoint")
    args = ap.parse_args()
    stage = args.stage_dir.name
    eval_csv = args.eval_csv
    if eval_csv is None:
        # search common locations rather than assume one: beside the stage dir,
        # in the stage dir itself, and under output/.
        candidates = [
            args.stage_dir / f"eval-{stage}.csv",           # beside registry-<stage>.csv (default)
            _PYTHON_ROOT.parent / "output" / f"eval-{stage}.csv",
            Path.cwd() / f"eval-{stage}.csv",
        ]
        eval_csv = next((c for c in candidates if c.exists()), candidates[0])
    print(f"Stage dir: {args.stage_dir}\nEval CSV:  {eval_csv}"
          + ("" if eval_csv.exists() else "  (NOT FOUND -- pass --eval-csv <path>; searched: "
             + ", ".join(str(c) for c in (candidates if args.eval_csv is None else [])) + ")"))
    if args.prune_baselines:
        prune_stale_baseline_rows(eval_csv, dry_run=args.dry_run)
        return
    if args.drop_out_of_scope_columns:
        drop_out_of_scope_columns(eval_csv, dry_run=args.dry_run)
        return
    if args.drop_aliased_params:
        drop_aliased_param_columns(eval_csv, dry_run=args.dry_run)
        return
    if args.collapse_empty_variant:
        collapse_redundant_empty_variant_rows(eval_csv, dry_run=args.dry_run)
        return
    if args.blank_zero_components:
        blank_zero_components(eval_csv, dry_run=args.dry_run)
        return
    backfill(args.stage_dir, eval_csv, dry_run=args.dry_run, debug=args.debug,
             all_pts=args.all_pts)


if __name__ == "__main__":
    main()
