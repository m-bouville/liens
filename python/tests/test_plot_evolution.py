"""Tests for evaluation/plot_evolution.py's pure helpers.

Only the IO-free core is tested here (run-spec parsing, saved-step slicing):
the roll-and-decode is compare_f_theta.compute_trajectory (its own tests) and
the layout needs a real model + dataset. These two helpers carry the tool's
own logic -- turning a RUNDIR:STARTSTEP + --steps into the exact saved-step
list compute_trajectory consumes -- and their failure modes (a mistyped step,
a start too late in an irregular save schedule) must be clear errors, not
silent short rows.

Imports the helpers WITHOUT importing the module top-level (which pulls in
torch/matplotlib via compare_f_theta): the pure functions are extracted by
AST, so the suite runs even where those aren't installed. If plot_evolution's
imports become lighter, this can become a plain `from evaluation.plot_evolution
import ...`.
"""
import ast
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parent.parent / "evaluation" / "plot_evolution.py")


def _load_pure_helpers():
    """_parse_run_spec and _pick_steps, exec'd in isolation (no torch import)."""
    tree = ast.parse(_SRC.read_text())
    ns = {"Path": Path}
    for name in ("_parse_run_spec", "_pick_steps"):
        node = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module([node], []), "<plot_evolution>", "exec"), ns)
    return ns["_parse_run_spec"], ns["_pick_steps"]


_parse_run_spec, _pick_steps = _load_pure_helpers()


# --------------------------------------------------------------------------- #
# _parse_run_spec
# --------------------------------------------------------------------------- #
def test_parse_run_spec_relative_path():
    assert _parse_run_spec("../datasets/128x128/T725_n003_s123:15000") == (
        Path("../datasets/128x128/T725_n003_s123"), 15000)


def test_parse_run_spec_splits_on_last_colon_so_windows_drive_survives():
    # A Windows absolute path has a drive ':'; the start step is still the
    # LAST token, so rsplit(':', 1) keeps the path intact.
    assert _parse_run_spec(r"D:\work\NN\datasets\128x128\T725_n003_s123:15000") == (
        Path(r"D:\work\NN\datasets\128x128\T725_n003_s123"), 15000)


def test_parse_run_spec_rejects_missing_step():
    with pytest.raises(ValueError):
        _parse_run_spec("../datasets/128x128/T725_n003_s123")


def test_parse_run_spec_rejects_noninteger_step():
    with pytest.raises(ValueError):
        _parse_run_spec("../datasets/128x128/T725_n003_s123:early")


# --------------------------------------------------------------------------- #
# _pick_steps  (irregular / geometric save schedule)
# --------------------------------------------------------------------------- #
_SAVES = [2000, 5000, 10000, 15000, 25000, 40000, 60000, 100000]


def test_pick_steps_returns_consecutive_saved_steps():
    assert _pick_steps(_SAVES, 15000, 4) == [15000, 25000, 40000, 60000]


def test_pick_steps_single_frame():
    assert _pick_steps(_SAVES, 2000, 1) == [2000]


def test_pick_steps_rejects_unsaved_start_and_names_nearest():
    with pytest.raises(ValueError) as e:
        _pick_steps(_SAVES, 16000, 3)
    assert "15000" in str(e.value)   # nearest actual saved step, for the fix


def test_pick_steps_rejects_too_few_remaining_actionably():
    with pytest.raises(ValueError) as e:
        _pick_steps(_SAVES, 60000, 4)      # only 60000, 100000 remain
    assert "reduce --steps" in str(e.value)
