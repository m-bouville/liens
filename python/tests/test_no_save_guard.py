"""
No trainer may return a checkpoint path that does not exist.

Every stage's caller consumes the returned path immediately -- the pipeline
feeds stage 1's to check_reconstruction, stage 2's and 3's to the next stage,
stage 4's to stage 5. A run that saved nothing therefore surfaces as a bare
FileNotFoundError somewhere downstream, naming a file rather than a reason.

Reported twice before this was made uniform: stage 3b early-stopped without
saving (nothing improved) and the sanity check died on the missing file; stage
4 with `epochs = 0` never enters the epoch loop at all.

`epochs = 0` is the sharpest case because it is a DELIBERATE ablation that
cannot possibly produce a checkpoint, so the error says so and points at the
fix (remove the stage, do not zero its epochs).
"""
import inspect
import pathlib

import pytest

from conftest import source_without_comments

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TRAINERS = {
    "training/train_stage1.py": "train_autoencoder",
    "training/train_stage2.py": "train_stage2",
    "training/train_lds.py": "train_lds",
    "training/train_refinement.py": "train_refinement",
}


@pytest.mark.parametrize("module_path,func", list(_TRAINERS.items()))
def test_every_trainer_refuses_to_return_a_path_it_never_wrote(module_path, func):
    src = source_without_comments(_ROOT / module_path)
    assert "without ever saving" in src, (
        f"{module_path} can return a checkpoint_path that was never written -- the "
        f"caller then fails with FileNotFoundError far from the cause"
    )


@pytest.mark.parametrize("module_path", list(_TRAINERS))
def test_the_guard_precedes_the_return(module_path):
    """
    GUARDS a check placed after `return checkpoint_path`, which is
    unreachable, or one attached to the wrong return in a module with several.
    """
    src = source_without_comments(_ROOT / module_path)
    assert src.index("without ever saving") < src.rindex("return checkpoint_path")


# train_refinement is exempt: it DOES evaluate and save at epoch 0 even with
# epochs=0 (a real run wrote "0| ... -> saved"), so reaching its no-save guard
# means the epoch-0 save itself failed -- which epochs=0 does not explain, and
# saying so would send the reader to the wrong place.
_EPOCHS_ZERO_CANNOT_SAVE = [m for m in _TRAINERS if "refinement" not in m]


@pytest.mark.parametrize("module_path", _EPOCHS_ZERO_CANNOT_SAVE)
def test_the_epochs_zero_case_is_named(module_path):
    """
    An epochs=0 ablation is not a failure, it is a configuration that cannot
    produce a checkpoint. Saying only "nothing improved" would send the reader
    looking at the loss curve for a run that never trained.
    """
    src = source_without_comments(_ROOT / module_path)
    # Window opens BEFORE the raise, not at it: the epochs=0 sentence is now
    # built as a variable just above (it was a multi-line conditional inside
    # an f-string field, which some parsers reject), so a window starting at
    # the message text no longer contains it.
    _start = src.index("without ever saving")
    guard = src[max(0, _start - 900):]
    guard = guard[:guard.index("return checkpoint_path")]
    assert "epochs" in guard and "0" in guard, module_path


@pytest.mark.parametrize("module_path,func", list(_TRAINERS.items()))
def test_the_guard_can_actually_see_epochs(module_path, func):
    """
    GUARDS referring to a variable that is not in scope -- the message would
    then raise NameError while trying to report the real problem.
    """
    import importlib
    mod = importlib.import_module(module_path.replace("/", ".").removesuffix(".py"))
    assert "epochs" in inspect.signature(getattr(mod, func)).parameters, module_path


# --------------------------------------------------------------------
# the message must not use a construct that some toolchains reject
# --------------------------------------------------------------------

def test_no_multiline_implicit_concatenation_inside_an_fstring_field():
    """
    REGRESSION: the epochs=0 hint was written as a multi-line conditional with
    implicit string concatenation INSIDE an f-string replacement field:

        f"{'first part '
           'second part ' if epochs == 0 else ''}"

    Legal from Python 3.12 (PEP 701) and it parses here -- but it raised
    "SyntaxError: unterminated string literal" on a user's 3.13 run. That
    corner is handled inconsistently across parsers and editors, and nothing
    is gained by inlining it: an ordinary variable above the raise reads
    better and cannot trip anything.

    Scanned across the whole production tree, not just the two files that had
    it, so the construct does not come back somewhere new.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for folder in ("training", "models", "evaluation", "orchestration", "utils"):
        for path in (root / folder).glob("*.py"):
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                stripped = line.rstrip()
                opens_field = stripped.count("{") > stripped.count("}")
                ends_in_quote = stripped.endswith(("'", '"'))
                nxt = lines[i + 1].lstrip() if i + 1 < len(lines) else ""
                if opens_field and ends_in_quote and nxt.startswith(("'", '"')):
                    offenders.append(f"{path.name}:{i + 1}")
    assert not offenders, (
        "multi-line implicit concatenation inside an f-string field at: "
        + ", ".join(offenders)
    )


def test_the_epochs_zero_hint_is_still_conditional():
    """The rewrite must not lose the condition -- the sentence is wrong when
    epochs != 0, which is how it was reported in the first place."""
    from conftest import source_without_comments
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("train_stage2", "train_lds"):
        src = source_without_comments(root / f"training/{name}.py")
        assert "_epochs_zero_hint = (" in src, name
        block = src[src.index("_epochs_zero_hint = ("):]
        block = block[:block.index(")")]
        assert 'if epochs == 0 else ""' in block, name


def test_no_multiline_fstring_expressions_in_production():
    """
    REGRESSION: a conditional built with implicit string concatenation ACROSS
    LINES inside an f-string replacement field:

        f"{'first part '
           'second part ' if epochs == 0 else ''}"

    That is legal from 3.12 (PEP 701), and it parsed here and passed the whole
    suite -- but raised "SyntaxError: unterminated string literal" on a user's
    3.13 run, in two files at once, killing the import before anything ran.
    Tooling handles this corner inconsistently and nothing is gained by
    inlining it; build the string in an ordinary variable first.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for d in ("training", "models", "evaluation", "utils", "orchestration"):
        for p in (root / d).glob("*.py"):
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r'f"[^"]*\{[^}]*\'[^\']*\'\s*$', line):
                    offenders.append(f"{p.name}:{i}")
    assert not offenders, (
        "multi-line f-string replacement field(s) at: " + ", ".join(offenders)
    )


def test_every_production_module_actually_parses():
    """
    The suite imports most modules, but a module reached only through a
    late-bound path could ship a SyntaxError undetected. Cheap insurance:
    parse every file outright.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for d in ("training", "models", "evaluation", "utils", "orchestration"):
        for p in (root / d).glob("*.py"):
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                raise AssertionError(f"{p}: {exc}") from exc
