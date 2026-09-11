"""
Per-epoch evaluation log: eval-stageN.csv, one row per (checkpoint_path, epoch).

Complements registry-stageN.csv (inputs, one row per RUN) with OUTPUTS at
epoch granularity -- so a run evaluated at several saved epochs gets several
rows, keyed on (checkpoint_path, epoch). Each diagnostic UPSERTS its own
columns: running compare_f_theta then check_latent_channels on the same epoch
fills ONE row with both tools' metrics (a second run of the same tool updates
its columns in place, it does not add a duplicate row). New metric columns are
added to the header on demand, so tools can contribute different fields without
a fixed schema.

The epoch and the training losses come from the CHECKPOINT dict (epoch,
val_loss, val_components), which the eval tools already load -- no log parsing,
no dependency on the .log being flushed.
"""
from __future__ import annotations

import csv
from pathlib import Path

_KEY = ("checkpoint_path", "epoch", "eval_variant")


def upsert_eval_row(csv_path: Path, checkpoint_path, epoch, metrics: dict,
                    eval_variant: str = "") -> None:
    """Insert-or-update the row for (checkpoint_path, epoch, eval_variant).

    metrics: column -> value for THIS diagnostic call. Columns not in metrics
    are left untouched on an existing row (so different TOOLS merge into one
    row when they share a key). eval_variant distinguishes different
    evaluation METHODS on the same checkpoint+epoch (e.g. compare_f_theta's
    "previous_quotient" vs "previous" vs "stage2_z0z1dt" derivative variants)
    -- these are genuinely different measurements, not columns of one row, so
    they get separate rows. Leave eval_variant="" for tools with only one
    measurement per checkpoint+epoch (e.g. check_latent_channels): then the
    key collapses to (checkpoint_path, epoch) as before, and a second tool's
    call with eval_variant="" merges into that same row.
    epoch None is allowed (stored as "") for checkpoints that carry no epoch,
    but then re-runs can't dedupe by epoch -- pass the checkpoint's real epoch
    whenever it has one.
    """
    csv_path = Path(csv_path)
    key = {"checkpoint_path": str(checkpoint_path),
           "epoch": "" if epoch is None else str(int(epoch)),
           "eval_variant": eval_variant or ""}

    rows: list[dict] = []
    fieldnames: list[str] = list(_KEY)
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or _KEY)
            rows = [dict(r) for r in reader]

    # union the columns: existing + key + this diagnostic's metric columns
    for col in list(_KEY) + list(metrics):
        if col not in fieldnames:
            fieldnames.append(col)

    # find the row for this key
    target = None
    for r in rows:
        if (r.get("checkpoint_path") == key["checkpoint_path"]
                and r.get("epoch") == key["epoch"]
                and (r.get("eval_variant") or "") == key["eval_variant"]):
            target = r
            break
    if target is None:
        target = dict(key)
        rows.append(target)
    # update this diagnostic's columns (stringify; None -> "")
    for col, val in metrics.items():
        target[col] = "" if val is None else str(val)

    # rewrite atomically (small files; simplest correct approach)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in fieldnames})
    tmp.replace(csv_path)


def eval_csv_for_checkpoint(checkpoint_path, output_root: Path | None = None) -> Path:
    """eval-<stage>.csv BESIDE the checkpoint's own stage directory (next to
    registry-<stage>.csv) -- e.g. checkpoints/stage3a/128x128-....pt ->
    checkpoints/stage3a/eval-stage3a.csv. This is the file backfill_eval_components
    reads by default and where the registry already lives; output_root is IGNORED
    (kept only so existing call sites that still pass it don't need editing) --
    the eval CSV is never written under output/.
    """
    p = Path(checkpoint_path)
    stage_dir = p.parent
    stage = stage_dir.name if stage_dir.name and stage_dir.name != "checkpoints" else "unknown"
    return stage_dir / f"eval-{stage}.csv"
