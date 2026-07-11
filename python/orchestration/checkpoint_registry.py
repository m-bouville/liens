"""
Checkpoint registry I/O: matching an existing checkpoint's recorded
parameters against a requested configuration, and recording new
entries as training proceeds. Extracted from main.py during its split
into orchestration/.
"""
import csv
import re
from pathlib import Path

import torch

from orchestration.paths import _CHECKPOINTS_ROOT, _STAGE_DIRS


_NON_SIGNATURE_KEYS = {"batch_size", "log_every_epoch"}


def _signature_kwargs(kwargs: dict) -> dict:
    """Excludes purely-operational parameters (see _NON_SIGNATURE_KEYS)
    from a cache-matching signature -- they still get passed to the
    actual training call via the original kwargs dict, just not
    considered when deciding whether a cached checkpoint counts as a
    match for this configuration."""
    return {k: v for k, v in kwargs.items() if k not in _NON_SIGNATURE_KEYS}

def _report_checkpoint_epoch(path: Path, target_epochs: int | None, device: str | None) -> None:
    """Prints the epoch a reused checkpoint was actually saved at,
    against the target upper bound -- e.g. 40/50 is normal early
    stopping, 4/50 suggests the process was killed shortly after it
    started and this checkpoint is likely worth deleting by hand."""
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
    saved_epoch = checkpoint.get("epoch")
    if saved_epoch is None:
        return
    if target_epochs is not None:
        print(f"  (saved at epoch {saved_epoch}/{target_epochs} -- if this looks very "
              f"low relative to the target, the run may have been killed shortly after "
              f"starting; consider deleting this checkpoint and retraining)")
    else:
        print(f"  (saved at epoch {saved_epoch})")


def _in_progress_signature(final_signature: dict, epoch: int) -> dict:
    """
    Same as the eventual final signature, but with 'epochs' replaced by
    a marker reflecting ACTUAL progress so far, not the planned target --
    an interrupted run's registry entry should be honest about only
    having reached epoch N, and this marker can never accidentally
    match a future query for a genuinely completed run at the same
    target epochs (a string never equals the int a real target would be).
    """
    sig = dict(final_signature)
    sig["epochs"] = f"in_progress_epoch_{epoch}"
    return sig


def _make_checkpoint_callback(registry_path: Path, final_signature: dict):
    """Returns an on_checkpoint_saved callback that upserts the registry
    with an in-progress signature every time a checkpoint is actually
    saved -- so a crash or interrupt still leaves a registry entry
    connected to whatever .pt/.log made it to disk, rather than nothing
    at all until the training function fully returns."""
    def callback(checkpoint_path: Path, epoch: int) -> None:
        _upsert_registry(registry_path, checkpoint_path, _in_progress_signature(final_signature, epoch))
    return callback

_CHECKPOINT_ANCESTRY_KEY = re.compile(r"^stage(\d+)_checkpoint$")


def _resolve_path_in_checkpoints(value: str, stage_num: int) -> str:
    """
    Resolves a path string expected to point at a file under
    checkpoints/ to a concrete, canonical path -- RESOLUTION, not fuzzy
    filename-only comparison (which would wrongly treat files in
    different directories that happen to share a name as the same
    file):
      - absolute path -> used as-is
      - relative WITH a directory component (e.g. 'stage2/foo.pt' or
        'stage2\\foo.pt') -> trusted as given, resolved relative to
        checkpoints/
      - bare filename (no directory component at all) -> autocompleted
        to checkpoints/stage<stage_num>/<filename> -- the only case
        that gets "corrected", since a bare filename doesn't name a
        location on its own
      - blank -> left blank (never resolved into a directory path,
        which would break blank-means-non-match elsewhere)
    Never resolved relative to the process's CWD, which -- as with
    _STAGE_DIRS -- can vary between invocations.
    """
    if value == "":
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    has_dir = ("/" in value) or ("\\" in value)
    if has_dir:
        return str((_CHECKPOINTS_ROOT / value.replace("\\", "/")).resolve())
    return str((_STAGE_DIRS[stage_num] / value).resolve())


def _resolve_checkpoint_field(key: str, value: str) -> str:
    """For a 'stageN_checkpoint' field, derives N from the key name
    itself (e.g. 'stage2_checkpoint' -> 2) and resolves via
    _resolve_path_in_checkpoints. Every other field is returned
    unchanged."""
    match = _CHECKPOINT_ANCESTRY_KEY.match(key)
    if not match:
        return value
    return _resolve_path_in_checkpoints(value, int(match.group(1)))


def _read_registry(registry_path: Path) -> tuple[list[str], list[dict]]:
    if not registry_path.exists():
        return [], []
    with open(registry_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _upsert_registry(registry_path: Path, checkpoint_path: Path, params: dict,
                      backfill_defaults: dict | None = None) -> None:
    """
    Writes an entry for checkpoint_path, GROWING the CSV's columns if
    `params` introduces a key not seen before -- explicit, visible schema
    evolution: a human (or this code) can see directly which old entries
    predate which parameter, rather than a flexible-but-opaque JSON blob
    silently hiding the same fact.

    UPSERT, not append: if a row for this EXACT checkpoint_path already
    exists, it's updated in place rather than duplicated -- needed
    because the same checkpoint_path now gets written repeatedly as
    training progresses (see on_checkpoint_saved in train_ae.py/
    train_lds.py), not just once at the very end. Without this, a stage
    that saves several improving checkpoints before finishing would
    accumulate one stale row per save instead of ending with one
    accurate row.

    backfill_defaults: for a NEW column that's really a former hardcoded
    constant becoming configurable (the common case -- adding a
    parameter usually means something that USED to be fixed now can
    vary), old rows implicitly used that known former value, so it can
    be backfilled instead of left blank. Only affects columns that are
    actually new in this call; a column with no default given here is
    left blank (safe default when there's no real prior equivalent,
    e.g. a genuinely new capability).
    """
    params = {k: _resolve_checkpoint_field(k, str(v)) for k, v in params.items()}
    backfill_defaults = backfill_defaults or {}
    fieldnames, rows = _read_registry(registry_path)
    if not fieldnames:
        fieldnames = ["checkpoint_path"]
    new_keys = [k for k in params.keys() if k not in fieldnames]
    fieldnames = fieldnames + new_keys

    for row in rows:
        for key in new_keys:
            if key in backfill_defaults:
                row[key] = str(backfill_defaults[key])
            # else: leave blank via restval="" below

    target_str = str(checkpoint_path)
    new_row = {"checkpoint_path": target_str}
    new_row.update({k: str(v) for k, v in params.items()})

    existing = next((row for row in rows if row.get("checkpoint_path") == target_str), None)
    if existing is not None:
        existing.clear()
        existing.update(new_row)
    else:
        rows.append(new_row)

    with open(registry_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _find_matching_checkpoint(registry_path: Path, params: dict, stage_num: int) -> Path | None:
    """
    Any EXISTING checkpoint -- under any name -- whose recorded
    parameters match exactly. A row BLANK for a key we're checking
    (e.g. an older entry recorded before that parameter existed) is a
    NON-match, never a wildcard -- we can't assume it used the same
    value for a parameter it doesn't even record. A stale registry
    entry (recorded but the file no longer exists) is also correctly
    ignored.

    'stageN_checkpoint' fields are RESOLVED to a concrete path before
    comparing (see _resolve_checkpoint_field), applied to both sides --
    a manually-edited CSV entry can be a bare filename, a relative path
    with a directory prefix, or an absolute path, and all resolve to
    the same canonical location rather than being compared as raw
    strings. A blank value is left blank, never resolved into a
    directory path.

    checkpoint_path ITSELF is also resolved (via _resolve_path_in_checkpoints,
    using `stage_num` since that field's key name doesn't encode a stage
    number the way 'stageN_checkpoint' does) before checking existence --
    otherwise a hand-filled relative entry like 'stage1\\foo.pt' is
    checked against the process's CWD, which practically never matches,
    silently causing a spurious retrain instead of the intended reuse.
    """
    _, rows = _read_registry(registry_path)
    target = {k: _resolve_checkpoint_field(k, str(v)) for k, v in params.items()}
    for row in rows:
        if all(_resolve_checkpoint_field(k, row.get(k, "")) == v for k, v in target.items()):
            path = Path(_resolve_path_in_checkpoints(row["checkpoint_path"], stage_num))
            if path.exists():
                return path
    return None
