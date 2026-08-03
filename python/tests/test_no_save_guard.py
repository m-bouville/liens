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
    guard = src[src.index("without ever saving"):]
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
