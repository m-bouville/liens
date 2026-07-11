"""
Parses a stage-parameters file into per-stage keyword arguments (see
main.py's own module docstring for the file format). Extracted from
main.py during its split into orchestration/.
"""
import inspect
import re
from pathlib import Path


_STAGE_HEADER = re.compile(r"^\s*#\s*Stage\s+(\d+[a-zA-Z]?)\s*$", re.IGNORECASE)
_KEY_RENAMES = {"patience": "early_stopping_patience", "batches": "batch_size"}


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


def _prepare_stage_kwargs(raw_params: dict[str, str]) -> dict:
    """Converts a parsed stage's raw string params into typed kwargs,
    renaming a few keys (e.g. patience -> early_stopping_patience) to
    match the underlying function's actual parameter names."""
    kwargs = {}
    for key, value in raw_params.items():
        kwargs[_KEY_RENAMES.get(key, key)] = _convert_value(value)
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
