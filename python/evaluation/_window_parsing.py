"""
Shared parsing for the '--fixed-windows run_dir:step0:step1:...:stepN'
CLI argument shape, used by check_rollout.py and
compare_rollout_training.py. Extracted from check_rollout.py, which had
the correct, Windows-path-safe implementation -- compare_rollout_training.py
had an independent, naive split(':') duplicate that broke on any Windows
path (confirmed directly: 'C:\\Users\\...\\T800_n010_s0:0:1000:2000:3000'
raised ValueError trying to int() the path's own drive-letter-adjacent
fragment, since a bare split(':') has no way to tell a Windows path's own
colon apart from the ones separating the step numbers).
"""
from pathlib import Path


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def parse_fixed_window(s: str) -> tuple[Path, list[int]]:
    """
    'run_dir:step0:step1:...:stepN' -> (Path(run_dir), [step0, ..., stepN]).

    NOT a naive split(':') -- run_dir itself can contain a colon (e.g. a
    Windows path like 'D:\\work\\...\\T950_n020_s79'), which would
    otherwise be sliced apart from the rest of the path and misread as
    if it were a step number. Instead, split on ':' and then scan from
    the RIGHT, greedily taking trailing parts that parse as int (the
    step numbers); everything before that point is the run_dir, rejoined
    with ':'. This relies on run directory names in this project never
    being purely numeric themselves (they're always 'T<temp>_n<noise>_
    s<seed>'-shaped) -- a directory literally named e.g. '12345' would
    be misparsed as an extra step instead, but that never occurs here.
    """
    parts = s.split(":")
    split_idx = len(parts)
    for i in range(len(parts) - 1, -1, -1):
        if _is_int(parts[i]):
            split_idx = i
        else:
            break
    path_parts, step_strs = parts[:split_idx], parts[split_idx:]
    if len(step_strs) < 2 or not path_parts:
        raise ValueError(f"--fixed-windows entry must be 'run_dir:step0:step1:...:stepN' "
                          f"(a run_dir followed by at least 2 step numbers), got '{s}'")
    return Path(":".join(path_parts)), [int(x) for x in step_strs]
