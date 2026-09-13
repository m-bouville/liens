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
import re
from pathlib import Path

_KEY = ("checkpoint_path", "epoch", "eval_variant")

# The eval CSV's columns fall into meaningful GROUPS, each owned by a source:
#
#   PARAM_COLUMNS  -- run INPUTS, filled from the checkpoint's config
#                     (params_from_checkpoint / backfill's param pass).
#   LOSS_COLUMNS   -- TRAINING losses: the total (val_loss, from the checkpoint)
#                     plus per-component losses (variable per stage, val_<name>,
#                     filled from the .log by backfill's loss pass).
#   OUTPUT_COLUMNS -- EVAL results, computed by the diagnostics themselves
#                     (compare_f_theta / check_latent_channels).
#   _KEY           -- checkpoint_path, epoch, eval_variant.
#
# backfill uses PARAM_COLUMNS and LOSS_COLUMNS to know exactly what it is
# responsible for filling (and therefore what "missing" means), rather than
# guessing by column-name prefix. EVAL_COLUMNS is their COMPOSITION -- the full
# canonical header -- not a hand-maintained flat list, so it can't drift out of
# sync with the groups.

# p_* columns are the checkpoint's full config, so they are VARIABLE per stage
# (accepted by the p_ pattern below), not a fixed list. This is a PREFERRED-ORDER
# hint for the common ones so the header reads sensibly; any other p_* config key
# is appended after. Loss-WEIGHT params (p_stats0_predict_weight, p_rollout_scale,
# ...) live here too -- they are inputs, unlike measured losses in LOSS_COLUMNS.
PARAM_COLUMNS = [
    "p_size", "p_latent_channels", "p_normalize_phi", "p_lr", "p_n_rollout_steps",
    "p_dynamics_mode", "p_derivative_source", "p_derivative_time",
]

# Fixed loss columns. Per-COMPONENT losses (val_rollout, val_recon0, ...) are
# stage-dependent and therefore variable -- matched by pattern below, not listed.
# val_components_kind: "raw" (before weight/scale -- comparable across runs) or
# "weighted_scaled" (a legacy checkpoint's contributions, which depend on the run's
# weight/scale hyperparameters). Recorded so a mixed column is never misread.
LOSS_COLUMNS = ["val_loss", "val_components_kind"]

# compare_f_theta rollout-eval outputs (prefixed so they're unambiguous standalone)
# + n_samples (windows the medians were computed over).
OUTPUT_COLUMNS = ["rollout_median_loss", "rollout_median_corr_dx", "rollout_n_steps",
                  "rollout_n_samples", "comparable", "log"]

# source-checkpoint provenance columns (the .pt this run built on)
SOURCE_COLUMNS = ["source_stage2", "resume_from"]

# Full canonical order = key + params + losses + outputs (composed, not flat).
EVAL_COLUMNS = list(_KEY) + SOURCE_COLUMNS + PARAM_COLUMNS + LOSS_COLUMNS + OUTPUT_COLUMNS

# Provenance columns. source_stage2 is a STAGE-3+ concept (those stages freeze a
# stage-2 encoder); it must NOT appear in stage-1/stage-2 CSVs. resume_from applies
# to any stage. _stage_of() derives the stage from the CSV name so the schema is
# stage-specific -- a file never carries a header that can't exist for its stage.
_SOURCE_STAGE2_STAGES = ("stage3a", "stage3b", "stage3", "stage4", "stage5")


def _stage_of(csv_name: str) -> str:
    import re as _re
    m = _re.search(r"eval-([0-9a-z]+)\.csv", csv_name)
    return m.group(1) if m else ""


_KNOWN_PATTERNS = (re.compile(r"^p_\w+$"), re.compile(r"^ch\d+_imp$"),
                   re.compile(r"^val_\w+$"), re.compile(r"^notes$"))


def _is_known_column(col: str) -> bool:
    return col in EVAL_COLUMNS or any(p.match(col) for p in _KNOWN_PATTERNS)


def _ordered_columns(cols, csv_name: str) -> list[str]:
    """Bucket `cols` into the canonical order for this stage:
    identity -> provenance -> input params (p_*) -> losses (val_* + rollout_*)
    -> channels (ch*_imp) -> anything else. source_stage2 is dropped for stages
    that never have it. Variable columns (extra p_*, val_<component>, ch*_imp) are
    placed in their group, so the file reads params, then losses, then channels."""
    import re as _re
    stage = _stage_of(csv_name)
    allow_src2 = stage in _SOURCE_STAGE2_STAGES
    present = [c for c in cols if not (c == "source_stage2" and not allow_src2)]
    seen = set()

    def take(pred, preferred=()):
        out = [c for c in preferred if c in present and c not in seen]
        out += sorted(c for c in present if c not in seen and c not in preferred and pred(c))
        seen.update(out)
        return out

    identity   = take(lambda c: c in ("checkpoint_path", "epoch", "eval_variant"),
                      ("checkpoint_path", "epoch", "eval_variant"))
    provenance = take(lambda c: c in ("source_stage2", "resume_from", "log"),
                      ("source_stage2", "resume_from", "log"))
    params     = take(lambda c: c.startswith("p_"), PARAM_COLUMNS)
    losses     = take(lambda c: c.startswith("val_") or c.startswith("rollout_")
                      or c in ("comparable",),
                      ["val_loss"] + [c for c in LOSS_COLUMNS if c != "val_loss"] + OUTPUT_COLUMNS)
    channels   = take(lambda c: _re.match(r"^ch\d+_imp$", c))
    rest       = [c for c in present if c not in seen]        # unknown -> last
    return identity + provenance + params + losses + channels + rest


_WARNED_FILES: set = set()   # files already reported for unexpected columns (once per process)


def reconcile_fieldnames(fieldnames: list[str], csv_name: str) -> list[str]:
    """Return fieldnames with every EVAL_COLUMNS entry present (missing ones added
    in canonical order), followed by existing extra columns in their original
    order. Any column that is neither canonical nor an expected variable column is
    reported on the console (once) -- surfaced, not silently kept or dropped."""
    present = set(fieldnames)
    unknown = [c for c in fieldnames if not _is_known_column(c)]
    if unknown and csv_name not in _WARNED_FILES:
        _WARNED_FILES.add(csv_name)
        print(f"  eval-log NOTE: {csv_name} has unexpected column(s) {unknown} "
              f"-- kept, but not written by any current tool (renamed/legacy?). "
              f"Reconcile or add to EVAL_COLUMNS.")
    # canonical first (added if missing), then any extra columns (known-variable or
    # unknown) preserved in original order
    # canonical columns for THIS stage (source_stage2 excluded where it can't exist)
    stage = _stage_of(csv_name)
    canon = [c for c in EVAL_COLUMNS
             if not (c == "source_stage2" and stage not in _SOURCE_STAGE2_STAGES)]
    # union canonical + existing, then order by group (params -> losses -> channels)
    union = list(dict.fromkeys(canon + list(fieldnames)))
    return _ordered_columns(union, csv_name)


def canonical_checkpoint_key(checkpoint_path) -> str:
    """The ONE spelling every writer keys on: 'checkpoints/<stage>/<name>.pt'.

    Different tools receive the same file spelled differently -- --all seeds
    'checkpoints/stage3a/X.pt', a Windows CLI passes 'checkpoints\\stage3a\\X.pt' or an
    absolute 'D:\\...\\X.pt'. Keying on the raw string would give one checkpoint
    several rows. Normalise to forward slashes and to the part from 'checkpoints/'
    onward; a synthetic 'baseline:...' key is returned untouched.
    """
    s = str(checkpoint_path)
    if s.startswith("baseline:"):
        return s
    s = s.replace("\\", "/")
    i = s.rfind("checkpoints/")
    return s[i:] if i >= 0 else s


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
    key = {"checkpoint_path": canonical_checkpoint_key(checkpoint_path),
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
    # ensure the canonical schema is present (add missing standard columns) and
    # surface any unexpected column -- one authoritative place for the header shape.
    fieldnames = reconcile_fieldnames(fieldnames, csv_path.name)

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


def params_from_checkpoint(checkpoint: dict) -> dict:
    """ALL input params for an eval row, read from the checkpoint's config -- every
    config key becomes a p_<key> column, so the FULL stage-specific parameter set
    is captured (stage 3a has ~50: n_rollout_steps, lr_warmup_epochs, dynamics_mode,
    stats0_predict_weight, rollout_scale, ...), not a hand-curated handful. These
    include loss-WEIGHT/scale hyperparameters -- those are INPUTS (params), distinct
    from the MEASURED losses in LOSS_COLUMNS (val_loss / val_<component>). Sparse by
    construction: a checkpoint only yields the keys its config actually has, so
    different stages produce different p_* columns and older checkpoints omit newer
    keys (blank, backward-compatible). Nested/list config values are stringified.
    """
    cfg = checkpoint.get("config") or {}
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (dict, list, tuple)):
            continue                     # skip structured sub-configs (e.g. stream_configs)
        out[f"p_{k}"] = v
    for k in ("epoch", "val_loss"):      # top-level provenance
        if checkpoint.get(k) is not None:
            out[k] = checkpoint[k]
    # explicit source-checkpoint columns (also present as p_stage2_checkpoint /
    # p_resumed_from via the generic dump, but named clearly here for the ledger):
    if cfg.get("stage2_checkpoint"):
        out["source_stage2"] = cfg["stage2_checkpoint"]
    if cfg.get("resumed_from"):
        out["resume_from"] = cfg["resumed_from"]
    return out


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
