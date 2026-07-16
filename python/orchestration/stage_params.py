"""
Parses a stage-parameters file into per-stage keyword arguments (see
main.py's own module docstring for the file format). Extracted from
main.py during its split into orchestration/.
"""
import inspect
import re
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
    '1a'/'1b' pairing convention: stage 1a = single-stream autoencoder,
    stage 1b = deriv stream decoder). '1b' is NOT touched by this --
    it's a genuinely distinct stage (train_stage1b), not an alias.
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
    ones that don't).
    """
    merged = {**(global_params or {}), **raw_params}
    kwargs = {}
    for key, value in merged.items():
        renamed_key = _KEY_RENAMES.get(key, key)
        if renamed_key in _LIST_VALUED_KEYS:
            kwargs[renamed_key] = [part.strip() for part in value.split(",")]
        else:
            kwargs[renamed_key] = _convert_value(value)
    return kwargs

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
