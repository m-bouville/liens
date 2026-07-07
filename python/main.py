"""
Orchestrates the LIENS pipeline from stage-PARAMETERS FILES, one per run
configuration, rather than a long list of CLI flags.

Stage numbering (see docs/neural_nets.md):
    0. Generate phase-field simulations (C++, not this pipeline)
    1. Train autoencoder (E, D) on individual microstructures -- real
       space, L_recon + lambda1*L_stats
    2. Latent-space validation: interpolation-consistency fine-tuning
       (E, D trainable; stats_head FROZEN, used only as a fixed
       measuring instrument for L_interp) -- L_recon + lambda1*L_interp,
       optionally + stats_weight*L_stats as an anchor (gradient flows
       into E only, since stats_head is frozen) against the encoder
       drifting away from anything stats_head can still interpret --
       see train_stage2()'s docstring for why this is needed at all.
    3. Latent Dynamics Surrogate (f) on a FROZEN encoder -- latent space,
       L_rollout (or L_1step, identical at n_rollout_steps=1). Can run
       as ONE phase ('# Stage 3') or as a two-phase CURRICULUM
       ('# Stage 3a' + '# Stage 3b'): 3a trains at n_rollout_steps=1
       (stable, fast), 3b resumes from 3a's checkpoint at the target
       (larger) n_rollout_steps -- avoids the instability of jumping
       straight to multi-step rollout training from scratch. Both
       conventions run the exact same train_lds() function underneath
       (see run_lds_stage() below) -- 3a/3b is a params-file-level
       naming choice, not a separate code path.
    4/5. Encoder refinement / end-to-end -- NOT YET IMPLEMENTED

config.txt is NOT read by this module AT ALL -- not even for grid size.
Nx/Ny come directly from the stage-parameters file, and dataset directory
enumeration reads each size's OWN datasets/<nx>x<ny>/metadata.txt (see
utils/load_datasets.py's read_sweep_metadata), co-located with the
actual dataset rather than a separate, possibly-stale or
describing-a-different-sweep config.txt. This is what lets one
invocation process several resolutions (e.g. 64x64 and 128x128) while a
completely unrelated sweep (e.g. 256x256) is being generated in C++ at
the same time.

STAGE-PARAMETERS FILE FORMAT, e.g. 64x64_no_stage2.txt:

    Nx = 64                          # required -- config.txt is never read
    Ny = 64
    base = ../datasets               # optional; falls back to --base if omitted
    # Stage 1
    min_step      = 4000            # inline '#' comments are stripped
    min_stdev_phi = 0.01
    stats_weight  = 0.01
    latent_channels = 8
    force = True                    # always retrain, even if a match already exists
    # Stage 2
    epochs = 0                      # 0 = SKIP this stage entirely
    min_stdev_phi = same            # inherit from the nearest preceding stage/global
    n_frozen_stages = 2             # optional: freeze E/D's outer layers (see train_stage2)
    stats_weight = 0.01             # optional: L_stats anchor weight (0 = no anchor)
    # Stage 3
    epochs = 50
    patience = 10                   # renamed to early_stopping_patience internally

Stage 3's two-phase curriculum uses '# Stage 3a' + '# Stage 3b' INSTEAD
of '# Stage 3' (giving both, or neither -- 3a alone is just a warmup,
not a usable final result; mixing bare 3 with 3a/3b is an error too):

    # Stage 3a
    n_rollout_steps = 1
    epochs = 100
    # Stage 3b
    n_rollout_steps = 6             # the actual target
    epochs = 300
    min_step = same                 # 'same' looks at 3a first, then Stage 2, then Stage 1

Any key not specially handled (Nx, Ny, base, force, epochs=0 skip,
'same' inheritance) is passed straight through as a keyword argument to
that stage's training function (train_autoencoder/train_stage2/
train_lds), after best-effort str->int/float/bool conversion.

CACHING: two independent checks before training any stage, in order:
  1. Does THIS params file's own expected output
     (python/checkpoints/stage<N>/<stem>-stage<N>.pt) already exist?
  2. If not, does the PARAMETER REGISTRY (registry-stage<N>.csv, in the
     same directory) record any OTHER checkpoint -- under any name --
     whose recorded parameters for this stage match EXACTLY? This is
     what lets, e.g., 64x64_no_stage2.txt's stage 1 reuse 64x64.txt's
     stage-1 output without the two files needing matching names, as
     long as their stage-1 parameters are identical.
Either hit skips training. A stage's force=True skips BOTH checks and
always retrains (overwriting its own expected filename, with a warning
if that file already existed). Stage 3a/3b use this identical machinery
under their own names (checkpoints/stage3a/, checkpoints/stage3b/) --
not because reusing a 3a checkpoint saves much time (stage 3 is fast
enough that retraining it isn't a real cost), just because it's the
same generic per-stage mechanism every other stage already uses, so
there's no reason to special-case it away.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m main params/64x64_no_stage2.txt
    python -m main params/64x64.txt params/128x128.txt
"""

import argparse
import csv
import inspect
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from evaluation.check_interpolation import check_interpolation
from evaluation.check_perturbation import check_perturbation
from evaluation.check_reconstruction import check_reconstruction
from evaluation.check_rollout import check_rollout
from training.train_ae import train_autoencoder, train_stage2
from training.train_lds import train_lds
from utils import load_datasets as load

_STAGE_LABELS = {
    1: "stage 1 (autoencoder)",
    2: "stage 2 (latent-space validation)",
    3: "stage 3 (latent dynamics surrogate)",
}
_STAGE_LABELS["3a"] = _STAGE_LABELS["3b"] = _STAGE_LABELS[3]

_STAGE_HEADER = re.compile(r"^\s*#\s*Stage\s+(\d+[a-zA-Z]?)\s*$", re.IGNORECASE)
_KEY_RENAMES = {"patience": "early_stopping_patience", "batches": "batch_size"}
# Purely operational parameters: they affect HOW training runs (speed,
# gradient noise) but not what it's trying to learn, so they're excluded
# from the cache-matching signature -- a run with a different batch_size
# but otherwise identical parameters still counts as "the same" checkpoint.
_NON_SIGNATURE_KEYS = {"batch_size", "log_every_epoch"}
_MAIN_DIR = Path(__file__).resolve().parent
_STAGE_DIRS = {1: _MAIN_DIR / "checkpoints" / "stage1",
               2: _MAIN_DIR / "checkpoints" / "stage2",
               3: _MAIN_DIR / "checkpoints" / "stage3",
               "3a": _MAIN_DIR / "checkpoints" / "stage3a",
               "3b": _MAIN_DIR / "checkpoints" / "stage3b"}


class _Tee:
    """Writes to multiple streams at once (e.g. the real console AND a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def _log_to_file(log_path: Path):
    """
    Tees everything printed to stdout/stderr into log_path for the
    duration of this block, IN ADDITION TO the normal console output --
    so a stage's full progress log survives even if the console itself
    is later closed/lost (e.g. an IDE crash after a long training run).
    Uses try/finally so the log is properly flushed and stdout/stderr
    restored even if the wrapped code raises.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)
        sys.stderr = _Tee(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def identify_checkpoint_stage(checkpoint: dict) -> str:
    """
    Inspects a loaded checkpoint's STRUCTURE (not its filename) to
    determine which pipeline stage actually produced it -- so a
    mismatched checkpoint (e.g. a stage-1 file sitting where a stage-2
    file was expected, exactly what happened when console logs got lost
    and files got confused) is caught with a clear, specific error
    instead of failing deep inside training with a confusing
    shape-mismatch, or silently training on the wrong starting point.
    """
    if "ae_checkpoint" in checkpoint:
        return _STAGE_LABELS[3]
    if "stage2_config" in checkpoint or "stage3_config" in checkpoint:
        # stage3_config: train_ae.py's OLD internal field name, from
        # before stages were renumbered -- still recognized here so a
        # checkpoint trained before that rename doesn't need retraining
        # just to be correctly identified.
        return _STAGE_LABELS[2]
    if ("model_state" in checkpoint and isinstance(checkpoint.get("config"), dict)
            and "latent_channels" in checkpoint["config"]):
        return _STAGE_LABELS[1]
    return "unrecognized (doesn't match any known stage's checkpoint structure)"


def _validate_checkpoint_stage(path: Path, stage_num: int, device: str | None) -> None:
    """Raises a clear, specific error if `path` isn't actually a
    checkpoint from the expected stage. identify_checkpoint_stage()
    already tries every known stage's structure in turn regardless of
    which one was expected, so this always names what the file actually
    is, not just that it failed one specific check."""
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
    actual = identify_checkpoint_stage(checkpoint)
    expected = _STAGE_LABELS[stage_num]
    if actual != expected:
        raise ValueError(f"{path} is not a {expected} checkpoint: it is a {actual} checkpoint.")


def _signature_kwargs(kwargs: dict) -> dict:
    """Excludes purely-operational parameters (see _NON_SIGNATURE_KEYS)
    from a cache-matching signature -- they still get passed to the
    actual training call via the original kwargs dict, just not
    considered when deciding whether a cached checkpoint counts as a
    match for this configuration."""
    return {k: v for k, v in kwargs.items() if k not in _NON_SIGNATURE_KEYS}


def _strip_unrecognized_params(func, kwargs: dict, label: str) -> dict:
    """
    Returns a copy of kwargs with any key that isn't an actual parameter
    of `func` REMOVED (not just warned about -- a warning that leaves
    the bad key in place doesn't prevent the TypeError it's warning
    about). Catches a typo'd, misplaced, or renamed-but-not-mapped
    parameter (e.g. Nx/Ny left over in a stage section after the
    config.txt-validation logic that used to consume them was removed)
    before it reaches the actual training call.
    """
    accepted = set(inspect.signature(func).parameters)
    unrecognized = set(kwargs) - accepted
    if unrecognized:
        print(f"WARNING: {label} has parameter(s) not recognized by its training "
              f"function -- IGNORED, not used: {sorted(unrecognized)}")
    return {k: v for k, v in kwargs.items() if k in accepted}


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


def check_sweep_status(base_path: Path) -> None:
    """
    Scan every <nx>x<ny> subdirectory under base_path -- each one has its
    own metadata.txt (see load.read_sweep_metadata), so this no longer
    depends on config.txt describing any particular sweep. Reports
    COMPLETE/INCOMPLETE/missing run directories per size found.
    """
    if not base_path.exists():
        print(f"{base_path} does not exist")
        return
    size_dirs = sorted(d for d in base_path.iterdir() if d.is_dir() and (d / "metadata.txt").exists())
    if not size_dirs:
        print(f"No <nx>x<ny> subdirectories with a metadata.txt found under {base_path}")
        return

    for size_dir in size_dirs:
        metadata = load.read_sweep_metadata(size_dir / "metadata.txt")
        dirs = [size_dir / subdir for subdir in metadata.subdirs]

        print(f"\n=== {size_dir.name} ===")
        n_complete = n_incomplete = n_missing = 0
        for d in dirs:
            if not d.exists():
                n_missing += 1
                continue
            if load.is_complete(d):
                n_complete += 1
                print(f"COMPLETE    {d}")
                run_metadata = load.read_metadata(d / "metadata.txt")
                check = load.check_snapshots_saved(d, run_metadata)
                if check["missing"] or check["bad_size"]:
                    print(f"            ! {len(check['missing'])} missing, "
                          f"{len(check['bad_size'])} bad size")
            else:
                n_incomplete += 1
                print(f"INCOMPLETE  {d}")

        print(f"{len(dirs)} runs listed in metadata.txt -> "
              f"{n_complete} complete, {n_incomplete} incomplete, {n_missing} missing (ignored)")


def parse_stage_params(path: Path) -> tuple[dict[str, str], dict[int | str, dict[str, str]]]:
    """
    Parses a stage-parameters file (see module docstring for format).
    Returns (global_params, {stage_key: {key: value}}), all values
    still raw strings -- see _prepare_stage_kwargs for type conversion.
    stage_key is an int (1, 2, 3) for ordinary stages, or a string
    ('3a', '3b') for stage 3's optional two-phase curriculum -- see
    module docstring.
    """
    global_params: dict[str, str] = {}
    stages: dict[int | str, dict[str, str]] = {}
    current_stage: int | str | None = None
    current_dict = global_params

    for raw_line in path.read_text().splitlines():
        header_match = _STAGE_HEADER.match(raw_line)
        if header_match:
            raw_stage = header_match.group(1)
            try:
                current_stage = int(raw_stage)
            except ValueError:
                current_stage = raw_stage.lower()  # e.g. "3a"
            current_dict = stages.setdefault(current_stage, {})
            continue

        line = raw_line.split("#", 1)[0].strip()  # strip inline comments
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))

        if value.lower() == "same":
            value = _resolve_same(key, current_stage, global_params, stages)

        current_dict[key] = value

    return global_params, stages


def _preceding_stages(stage: int | str | None) -> list[int | str]:
    """Stages that come before `stage` in the pipeline, NEAREST first --
    used for 'same' value inheritance. Stage 3 has two mutually
    exclusive conventions (bare 3 for single-phase, 3a/3b for the
    curriculum -- see module docstring), both handled here."""
    order: dict[int | str, list[int | str]] = {
        1: [], 2: [1], 3: [2, 1], "3a": [2, 1], "3b": ["3a", 2, 1],
    }
    return order.get(stage, [])


def _resolve_same(key: str, stage: int | str | None, global_params: dict[str, str],
                   stages: dict[int | str, dict[str, str]]) -> str:
    """Walk backward through preceding stages, then the global section,
    for the nearest defined value of `key`."""
    for s in _preceding_stages(stage):
        if s in stages and key in stages[s]:
            return stages[s][key]
    if key in global_params:
        return global_params[key]
    raise ValueError(f"'{key} = same' but no preceding stage or global section defines '{key}'")


def _convert_value(value: str):
    """Best-effort str -> bool/int/float conversion; left as a string
    (e.g. a path) if none apply."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _prepare_stage_kwargs(raw_params: dict[str, str]) -> dict:
    """Converts a parsed stage's raw string params into typed kwargs,
    renaming a few keys (e.g. patience -> early_stopping_patience) to
    match the underlying function's actual parameter names."""
    kwargs = {}
    for key, value in raw_params.items():
        kwargs[_KEY_RENAMES.get(key, key)] = _convert_value(value)
    return kwargs


_CHECKPOINT_ANCESTRY_KEY = re.compile(r"^stage(\d+)_checkpoint$")
_CHECKPOINTS_ROOT = _MAIN_DIR / "checkpoints"


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


def run_from_params_file(params_path: Path, default_base: Path,
                          device: str | None = None) -> Path:
    """
    Runs stages 1->2->3 as specified by a stage-parameters file. See the
    module docstring for the file format, naming convention, and caching
    behavior (own expected filename, then the parameter registry).
    """
    global_params, stages = parse_stage_params(params_path)
    if not stages:
        print(f"WARNING: {params_path} has no recognized '# Stage N' section headers -- "
              f"every key ended up in the global section and NONE will be used for "
              f"training (stage 1 requires at least --latent-channels, for instance). "
              f"'# Stage N' must be on its own line, exactly '# Stage' followed by a "
              f"number, nothing else on that line.")
    base_path = Path(global_params.get("base", default_base))

    # Nx/Ny: the ONLY source of grid size now (config.txt is no longer
    # read at all -- see module docstring). Checked in both the global
    # section (their intended home) and Stage 1's section (an easy,
    # reasonable place to put them by mistake, since grid size is most
    # associated with the autoencoder) so either placement works.
    stage1_raw = stages.setdefault(1, {})
    nx = global_params.pop("Nx", None) or stage1_raw.pop("Nx", None)
    ny = global_params.pop("Ny", None) or stage1_raw.pop("Ny", None)
    if nx is None or ny is None:
        raise ValueError(f"{params_path}: Nx and Ny are required (config.txt is no longer "
                          f"read for grid size at all) -- give them in the global section "
                          f"or Stage 1's section.")
    nx, ny = int(nx), int(ny)
    if nx != ny:
        raise ValueError(f"{params_path}: only square grids are supported, got Nx={nx}, Ny={ny}")
    size = nx
    extra_signature = {"Nx": nx, "Ny": ny}

    stem = params_path.stem
    for stage_dir in _STAGE_DIRS.values():
        stage_dir.mkdir(parents=True, exist_ok=True)

    def stage_output_path(stage_num: int) -> Path:
        return _STAGE_DIRS[stage_num] / f"{stem}-stage{stage_num}.pt"

    def resolve_checkpoint(stage_num: int, force: bool, signature: dict,
                           target_epochs: int | None) -> Path | None:
        """Two-tier cache check, in order: own expected filename, then
        this stage's own parameter registry. Returns an existing path
        to reuse, or None if this stage must actually be (re)trained.
        Either hit is structurally validated before being trusted --
        catches a mismatched/mislabeled file before wasting time
        training on top of it -- and reports the epoch it was actually
        saved at against the target, to flag likely-killed-early runs."""
        own_path = stage_output_path(stage_num)
        if own_path.exists():
            if force:
                print(f"WARNING: {own_path} already exists and will be OVERWRITTEN (force=True)")
            else:
                _validate_checkpoint_stage(own_path, stage_num, device)
                print(f"Stage {stage_num}: found existing {own_path}, reusing.")
                _report_checkpoint_epoch(own_path, target_epochs, device)
                return own_path
        if not force:
            registry_path = _STAGE_DIRS[stage_num] / f"registry-stage{stage_num}.csv"
            match = _find_matching_checkpoint(registry_path, signature, stage_num)
            if match is not None:
                _validate_checkpoint_stage(match, stage_num, device)
                print(f"Stage {stage_num}: found matching checkpoint in registry "
                      f"({match}), reusing -- parameters are identical to an "
                      f"already-trained run under a different name.")
                _report_checkpoint_epoch(match, target_epochs, device)
                return match
        return None

    # ---- Stage 1: autoencoder ----
    stage1_kwargs = _prepare_stage_kwargs(stages.get(1, {}))
    force1 = stage1_kwargs.pop("force", False)
    stage1_kwargs = _strip_unrecognized_params(train_autoencoder, stage1_kwargs, "Stage 1")
    signature1 = {"base_path": str(base_path),
                  **extra_signature, **_signature_kwargs(stage1_kwargs)}
    stage1_checkpoint = resolve_checkpoint(1, force1, signature1, stage1_kwargs.get("epochs"))
    if stage1_checkpoint is None:
        with _log_to_file(stage_output_path(1).with_suffix(".log")):
            print("=" * 70)
            print("STAGE 1: training autoencoder")
            print("=" * 70)
            registry1_path = _STAGE_DIRS[1] / "registry-stage1.csv"
            stage1_checkpoint = train_autoencoder(
                size=size, base_path=base_path,
                checkpoint_path=stage_output_path(1), device=device,
                on_checkpoint_saved=_make_checkpoint_callback(registry1_path, signature1),
                **stage1_kwargs,
            )
            print(f"\nStage 1 complete: {stage1_checkpoint}\n")
            _upsert_registry(registry1_path, stage1_checkpoint, signature1)

    # ---- Stage 2: latent-space validation ----
    stage2_kwargs = _prepare_stage_kwargs(stages.get(2, {}))
    force2 = stage2_kwargs.pop("force", False)
    stage2_kwargs = _strip_unrecognized_params(train_stage2, stage2_kwargs, "Stage 2")
    if stage2_kwargs.get("epochs") == 0:
        print("Stage 2: epochs=0 -> skipping, using stage 1's output directly\n")
        stage2_checkpoint = stage1_checkpoint
    else:
        # Naming note: registries use a consistent "stageN_checkpoint" ancestry
        # convention across ALL stages, independent of whatever the underlying
        # function calls its own parameter (train_stage2 calls it resume_from,
        # train_lds calls it ae_checkpoint_path) -- so "stage1_checkpoint" means
        # the same thing everywhere you look, in any registry.
        signature2 = {"base_path": str(base_path),
                       "stage1_checkpoint": str(stage1_checkpoint),
                       **extra_signature, **_signature_kwargs(stage2_kwargs)}
        stage2_checkpoint = resolve_checkpoint(2, force2, signature2, stage2_kwargs.get("epochs"))
        if stage2_checkpoint is None:
            with _log_to_file(stage_output_path(2).with_suffix(".log")):
                print("=" * 70)
                print("STAGE 2: latent-space validation (interpolation-consistency fine-tuning)")
                print("=" * 70)
                registry2_path = _STAGE_DIRS[2] / "registry-stage2.csv"
                stage2_checkpoint = train_stage2(
                    base_path=base_path, resume_from=stage1_checkpoint,
                    checkpoint_path=stage_output_path(2), device=device,
                    on_checkpoint_saved=_make_checkpoint_callback(registry2_path, signature2),
                    **stage2_kwargs,
                )
                print(f"\nStage 2 complete: {stage2_checkpoint}\n")
                _upsert_registry(registry2_path, stage2_checkpoint, signature2)

                print("=" * 70)
                print("Sanity check: reconstruction quality (stage 2 checkpoint)")
                print("=" * 70)
                check_reconstruction(
                    checkpoint_path=stage2_checkpoint, device=device,
                    output_path=Path(f"../output/reconstruction_check_png/{stage2_checkpoint.stem}.png"),
                )
                print()

                print("=" * 70)
                print("Sanity check: interpolation consistency (stage 2 checkpoint)")
                print("=" * 70)
                check_interpolation(
                    checkpoint_path=stage2_checkpoint, device=device,
                    output_path=Path(f"../output/interpolation_check_png/{stage2_checkpoint.stem}.png"),
                )
                print()

                print("=" * 70)
                print("Sanity check: perturbation response (stage 2 checkpoint)")
                print("=" * 70)
                check_perturbation(
                    checkpoint_path=stage2_checkpoint, device=device,
                    output_path=Path(f"../output/perturbation_check_png/{stage2_checkpoint.stem}.png"),
                )
                print()

    # ---- Stage 3: LDS -- either single-phase ('# Stage 3') or a
    # two-phase curriculum ('# Stage 3a' + '# Stage 3b': train
    # n_rollout_steps=1 first, stable and fast, then resume with the
    # target n_rollout_steps -- see train_lds()'s resume_from docstring
    # for why). Same train_lds() function and same caching/registry
    # machinery either way; only the params-file section names and
    # whether a resume_from is threaded through differ. ----
    has_3a, has_3b, has_bare_3 = "3a" in stages, "3b" in stages, 3 in stages
    if has_3a != has_3b:
        raise ValueError(f"{params_path}: Stage 3a and Stage 3b must both be given together "
                          f"(3a alone is just a warmup, not a usable final result) -- found "
                          f"only {'3a' if has_3a else '3b'}.")
    if has_bare_3 and has_3a:
        raise ValueError(f"{params_path}: give EITHER '# Stage 3' (single-phase) OR "
                          f"'# Stage 3a' + '# Stage 3b' (curriculum), not both.")

    def run_lds_stage(stage_key: int | str, resume_from: Path | None = None) -> Path:
        """Runs one train_lds() phase -- stage_key is 3 (single-phase),
        or '3a'/'3b' (curriculum). Caching/registry/logging are
        identical regardless of which phase this is; resume_from (only
        given for 3b) is recorded in the signature so a 3b result is
        correctly tied to the specific 3a checkpoint it built on, not
        just its own hyperparameters."""
        kwargs = _prepare_stage_kwargs(stages.get(stage_key, {}))
        force = kwargs.pop("force", False)
        kwargs = _strip_unrecognized_params(train_lds, kwargs, f"Stage {stage_key}")
        # BOTH ancestors recorded explicitly -- since two stage-2 checkpoints
        # can share identical stage-2 parameters while differing in stage 1
        # (e.g. a quick vs a fully-trained stage 1), stage2_checkpoint alone
        # already disambiguates for MATCHING purposes (it's produced by
        # exactly one stage-1 checkpoint), but recording stage1_checkpoint
        # too means you can see the full ancestry from this ONE registry,
        # without having to open stage 2's own registry and follow ITS
        # stage1_checkpoint field.
        signature = {"base_path": str(base_path),
                      "stage1_checkpoint": str(stage1_checkpoint),
                      "stage2_checkpoint": str(stage2_checkpoint),
                      **({"resumed_from": str(resume_from)} if resume_from is not None else {}),
                      **extra_signature, **_signature_kwargs(kwargs)}
        checkpoint = resolve_checkpoint(stage_key, force, signature, kwargs.get("epochs"))
        if checkpoint is None:
            with _log_to_file(stage_output_path(stage_key).with_suffix(".log")):
                print("=" * 70)
                resume_note = f" (resuming from {resume_from})" if resume_from is not None else ""
                print(f"STAGE {stage_key}: latent dynamics surrogate (frozen encoder){resume_note}")
                print("=" * 70)
                registry_path = _STAGE_DIRS[stage_key] / f"registry-stage{stage_key}.csv"
                checkpoint = train_lds(
                    size=size, base_path=base_path, ae_checkpoint_path=stage2_checkpoint,
                    checkpoint_path=stage_output_path(stage_key), device=device,
                    resume_from=resume_from,
                    on_checkpoint_saved=_make_checkpoint_callback(registry_path, signature),
                    **kwargs,
                )
                print(f"\nStage {stage_key} complete: {checkpoint}\n")
                _upsert_registry(registry_path, checkpoint, signature)
        return checkpoint

    if has_3a:
        stage3a_checkpoint = run_lds_stage("3a")
        stage3_checkpoint = run_lds_stage("3b", resume_from=stage3a_checkpoint)
    else:
        stage3_checkpoint = run_lds_stage(3)

    print("=" * 70)
    print("Sanity check: rollout quality (stage 3 checkpoint)")
    print("=" * 70)
    check_rollout(
        lds_checkpoint_path=stage3_checkpoint, device=device,
        output_path=Path(f"../output/rollout_check_png/{stage3_checkpoint.stem}.png"),
    )
    print()

    return stage3_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("params_files", type=Path, nargs="*",
                         help="one or more stage-parameters file paths -- the pipeline runs "
                              "once per file, in order given")
    parser.add_argument("--base", type=Path, default=Path("../datasets"),
                         help="fallback dataset base path, only used if a params file "
                              "doesn't specify its own 'base = ...' in its global section")
    parser.add_argument("--scan-only", action="store_true",
                         help="just report sweep status (scanning every size directory's "
                              "own metadata.txt under --base) and exit, don't train anything")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.scan_only:
        check_sweep_status(args.base)
        return

    # if not args.params_files:
    #     raise ValueError("Provide at least one stage-parameters file (or --scan-only)")

    for params_path in [Path("params/64x64.txt")]:
        print("#" * 70)
        print(f"# {params_path}")
        print("#" * 70)
        run_from_params_file(params_path, default_base=args.base, device=args.device)
        print()


if __name__ == "__main__":
    main()
