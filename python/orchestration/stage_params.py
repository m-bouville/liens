"""
Parses a stage-parameters file into per-stage keyword arguments (see
main.py's own module docstring for the file format). Extracted from
main.py during its split into orchestration/.
"""
import inspect
import re
import shutil
from datetime import datetime
from pathlib import Path


_STAGE_HEADER = re.compile(r"^\s*#\s*Stage\s+(\d+[a-zA-Z]?)\s*$", re.IGNORECASE)
_KEY_RENAMES = {"patience": "early_stopping_patience", "batches": "batch_size",
                 "latent_size": "latent_spatial_size"}


def parse_stage_params(path: Path) -> tuple[dict[str, str], dict[int | str, dict[str, str]]]:
    """
    Parses a stage-parameters file (see module docstring for format).
    Returns (global_params, {stage_key: {key: value}}), all values
    still raw strings -- see _prepare_stage_kwargs for type conversion.
    stage_key is an int (1, 2, 3) for ordinary stages, or a string
    ('1b', '3a', '3b') for stage 1's deriv-stream decoder or stage 3's
    optional two-phase curriculum -- see module docstring. '1a' is
    accepted as an ALIAS for plain stage 1 (normalized to the int key
    1, not kept as a separate '1a' string) -- unlike '3a'/'3b' (two
    genuinely distinct curriculum phases of stage 3), '1a' isn't a
    different phase of anything, it's just another, more consistent-
    looking name for stage 1 itself (matching the project's own
    '1a'/'1b' naming convention from when stage 1 and its former deriv-
    stream-decoder extension were separate stages). '1b' is NOT touched
    by this -- it's still a distinct, recognized section key (though
    no longer used for anything -- see orchestration/pipeline.py's own
    warning when one is present), not an alias.
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
                current_stage = raw_stage.lower()  # e.g. "3a", "1b"
                if current_stage == "1a":
                    current_stage = 1
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
    curriculum -- see module docstring), both handled here. Stage 4 and
    5's chains list BOTH stage-3 conventions ("3b"/"3a" AND bare 3):
    only one will actually exist in stages{} for any given params file,
    and _resolve_same simply skips entries not present there, so
    listing both here is harmless and correct regardless of which
    convention that file actually used."""
    order: dict[int | str, list[int | str]] = {
        1: [], 2: [1], 3: [2, 1], "3a": [2, 1], "3b": ["3a", 2, 1],
        4: ["3b", "3a", 3, 2, 1], 5: [4, "3b", "3a", 3, 2, 1],
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


# Keys that need a LIST of strings (comma-separated in the params
# file), not the normal scalar bool/int/float conversion above. A
# fixed set of key NAMES, not a generic "does the value contain a
# comma" heuristic, specifically so this doesn't misfire on some
# future unrelated param that happens to also use commas for a
# different reason.
_LIST_VALUED_KEYS = {"stat_names"}

# Keys that must NEVER be picked up as a global default, even though
# _prepare_stage_kwargs's own merge is otherwise deliberately
# unscoped (see its own docstring for that general rationale).
# resume_from's own MEANING is always stage-specific -- train_stage1's
# own resume_from means "restart this exact stage from a crash";
# train_stage2's own means "which ancestor to extend"; train_lds's own
# means "which prior stage-3 curriculum phase to continue" -- so a
# single global value is never actually "the same ancestry" for a
# different stage, unlike e.g. num_workers/device/seed, which
# genuinely mean the same thing everywhere. A STAGE'S OWN section
# setting resume_from directly is unaffected by this and still works
# exactly as documented -- this only blocks the GLOBAL-default path.
_NEVER_GLOBAL_DEFAULT_KEYS = {"resume_from"}


def _prepare_stage_kwargs(raw_params: dict[str, str], global_params: dict[str, str] | None = None) -> dict:
    """
    Converts a parsed stage's raw string params into typed kwargs,
    renaming a few keys (e.g. patience -> early_stopping_patience) to
    match the underlying function's actual parameter names.

    global_params, if given, supplies a DEFAULT for any key this stage
    doesn't specify at all -- NOT the same as '= same' (which requires
    the key to be present in the stage's own section, just pointing at
    an earlier value): omitting the key here means "use the global
    default if there is one", while '= same' means "use the specific
    value from the nearest preceding stage or the global section". The
    stage's own value always wins over the global default when both
    are given -- this is a fallback, not an override. Deliberately a
    plain dict merge, not scoped to any particular set of keys: any
    global key applies to any stage that happens to accept a parameter
    by that name (see _strip_unrecognized_params for what happens to
    ones that don't) -- EXCEPT _NEVER_GLOBAL_DEFAULT_KEYS (see its own
    comment above), which are dropped from global_params before the
    merge, regardless of whether this specific stage would otherwise
    have accepted them.
    """
    global_params = {k: v for k, v in (global_params or {}).items()
                      if k not in _NEVER_GLOBAL_DEFAULT_KEYS}
    merged = {**global_params, **raw_params}
    kwargs = {}
    for key, value in merged.items():
        renamed_key = _KEY_RENAMES.get(key, key)
        if renamed_key in _LIST_VALUED_KEYS:
            kwargs[renamed_key] = [part.strip() for part in value.split(",")]
        else:
            kwargs[renamed_key] = _convert_value(value)
    return kwargs

# Params the PIPELINE itself consumes rather than forwarding to any
# training function (see run_from_params_file's own preamble, which pops
# them). Legitimate in a params file, so they must not be reported as
# unrecognized -- they're simply not training-function parameters.
_PIPELINE_CONSUMED_KEYS = frozenset({"base", "Nx", "Ny", "force"})


def renamed_keys(raw_params: dict) -> set[str]:
    """
    The parameter names raw_params will actually CARRY once
    _prepare_stage_kwargs has applied _KEY_RENAMES -- so a caller can
    ask "which of these keys came from this stage's OWN section?"
    against the post-rename names the kwargs dict is keyed by.
    """
    return {_KEY_RENAMES.get(k, k) for k in raw_params}


def report_unrecognized_global_params(global_params: dict, funcs) -> list[str]:
    """
    Warns ONCE, in the preamble, about global-section params that no
    stage's training function accepts at all -- i.e. genuine typos or
    leftovers, dead everywhere rather than merely inapplicable here.

    Split out from the per-stage check deliberately. A global param is
    by design offered to EVERY stage as a default (see
    _prepare_stage_kwargs), so most globals are legitimately unusable by
    most stages -- e.g. a global `latent_channels` is meaningful to
    stage 1 and meaningless to stage 3. Reporting those per-stage
    produced a spurious warning in every stage that couldn't use them,
    which is noise that trains the reader to ignore the warning
    entirely; the one case that IS a real mistake (a key no stage
    accepts) then hides among them. So: globals are checked once here,
    against the UNION of every training function's parameters, and the
    per-stage check below only reports keys from a stage's OWN section.
    """
    accepted = set(_PIPELINE_CONSUMED_KEYS)
    for f in funcs:
        accepted |= set(inspect.signature(f).parameters)
    unknown = sorted(renamed_keys(global_params) - accepted)
    if unknown:
        print(f"WARNING: global/preamble parameter(s) accepted by NO stage's training "
              f"function -- IGNORED everywhere, likely a typo or a leftover: {unknown}")
    return unknown


def _strip_unrecognized_params(func, kwargs: dict, label: str,
                                own_keys: set[str] | None = None) -> dict:
    """
    Returns a copy of kwargs with any key that isn't an actual parameter
    of `func` REMOVED (not just warned about -- a warning that leaves
    the bad key in place doesn't prevent the TypeError it's warning
    about). Catches a typo'd, misplaced, or renamed-but-not-mapped
    parameter (e.g. Nx/Ny left over in a stage section after the
    config.txt-validation logic that used to consume them was removed)
    before it reaches the actual training call.

    own_keys: if given, only keys in this set are WARNED about -- the
    rest are still stripped, silently. Pass renamed_keys(<this stage's
    own raw section>) so that inherited global defaults that simply
    don't apply to this stage are dropped without comment (they're
    reported once, globally, by report_unrecognized_global_params
    instead -- see its own docstring for why the split matters). Omit it
    to warn about everything, which is the right behavior for a caller
    that has no globals to inherit from.
    """
    accepted = set(inspect.signature(func).parameters)
    unrecognized = set(kwargs) - accepted
    to_warn = sorted(unrecognized if own_keys is None else (unrecognized & own_keys))
    if to_warn:
        print(f"WARNING: {label}'s OWN section has parameter(s) not recognized by its "
              f"training function -- IGNORED, not used: {to_warn}")
    return {k: v for k, v in kwargs.items() if k in accepted}


def _resolve_stage_specific_ancestor(kwargs: dict, default, label: str,
                                      key: str = "resume_from"):
    """
    Returns (kwargs_without_key, resolved_value, was_overridden):
    `default` (the pipeline's own, normally-correct ancestor for this
    stage) unless `key` is present in kwargs, in which case THAT value
    overrides it, with a clear print explaining the override.
    was_overridden distinguishes the two cases explicitly (rather than
    a caller comparing resolved_value != default itself, which would
    misfire in the unlikely case an override happens to equal the
    default) -- used by callers that need to back up an about-to-be-
    overwritten file ONLY in the explicit-override case (see
    _backup_before_overwrite's own docstring), never for the normal,
    automatic stage-chaining case, where the output path and the
    ancestor are two different files by construction.

    A value here did NOT come from a typo (that's
    _strip_unrecognized_params's own job) or a global-section leak
    (that's already excluded before this point -- see
    _NEVER_GLOBAL_DEFAULT_KEYS and _prepare_stage_kwargs's own
    docstring): reaching here means it was set explicitly, under THIS
    stage's own section. That's a legitimate, intentional choice to
    honor, not discard -- e.g. train_stage2's own deriv_target_centered
    curriculum is BUILT around resuming from an already-trained stage-2
    checkpoint (see train_stage2's own docstring), which requires
    exactly this: a stage-2-section resume_from pointing at a prior
    stage-2 checkpoint instead of the pipeline's own default (stage 1's
    checkpoint).

    Exists as a named function (not inlined at each call site) because
    train_stage2/train_lds/train_refinement all happen to share this
    SAME parameter name for DIFFERENT roles (train_stage2's own
    resume_from means "which ancestor to extend"; train_lds's own means
    "which PRIOR stage-3 curriculum phase to continue") -- each call
    site already passes its own correct default explicitly; this only
    ever substitutes an explicit, same-stage override for that default,
    never invents a value the underlying function wouldn't otherwise
    have been given.
    """
    if key not in kwargs:
        return kwargs, default, False
    override = Path(kwargs[key])
    print(f"NOTE: {label} has its own '{key}' set explicitly -- using it "
          f"({override}) INSTEAD of the pipeline's own default ancestor for this stage.")
    return {k: v for k, v in kwargs.items() if k != key}, override, True


def _backup_before_overwrite(path: Path) -> None:
    """
    If `path` already exists, copies it to `<stem>-<timestamp><suffix>`
    in the same directory, before whatever runs next overwrites it in
    place. Timestamp format matches this project's own snapshot/log
    naming convention elsewhere: YYYYMMDD_HHhMM -- taken from `path`'s
    own last-modified time, NOT the current time: this name is meant
    to answer "when was THIS checkpoint/log actually produced", not
    "when did we happen to archive it" (those can easily differ by
    hours if the pipeline sits queued, or differ across the .pt and
    its own .log if one gets written slightly after the other).

    Using the source's own mtime (rather than "now") also means the
    resulting backup path is DETERMINISTIC for an unchanged source
    file -- so if `path` hasn't actually changed since the last backup
    (e.g. the pipeline is invoked again before this stage would even
    rerun, or resolve_checkpoint finds a cache hit on a LATER stage
    but this one's own backup already ran earlier in the same
    invocation), the backup already exists at that exact name, and
    this returns without copying again rather than leaving redundant,
    byte-identical archives at different timestamps.

    Only meant to be called when resume_from is an EXPLICIT, stage-
    specific override (see _resolve_stage_specific_ancestor's own
    was_overridden return value) -- NOT for the normal, automatic
    stage-chaining resume_from (stage 1 -> stage 2, 3a -> 3b, etc.),
    where the output path and the ancestor are two DIFFERENT files by
    construction, so there is no overwrite risk to protect against in
    the first place; backing up on every ordinary run would just leave
    a growing pile of redundant copies nobody asked for.

    The real risk this protects against: resuming a checkpoint at the
    SAME path this run's own output (or its .log) will be written to
    -- e.g. continuing an already-trained stage-2 checkpoint's own
    deriv_target_centered curriculum in place, which force=True would
    otherwise silently overwrite mid-run, with NOTHING left of the
    original if anything goes wrong partway through -- the log file
    especially, since _log_to_file's own open(log_path, "w") truncates
    it immediately, before training even starts, regardless of whether
    the run succeeds.
    """
    if not path.exists():
        return
    timestamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%Hh%M")
    backup_path = path.with_name(f"{path.stem}-{timestamp}{path.suffix}")
    if backup_path.exists():
        print(f"NOTE: {path} is already archived at {backup_path} (unchanged since -- "
              f"same mtime) -- not re-copying.")
        return
    shutil.copy2(path, backup_path)
    print(f"NOTE: backed up {path} -> {backup_path} before this run's own output "
          f"(force=True) overwrites it in place.")
