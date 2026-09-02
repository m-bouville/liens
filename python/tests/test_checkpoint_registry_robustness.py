"""Guards for the two fixes from the 03h22 crash: a malformed registry row
must never kill a save, and a failing bookkeeping callback must never kill
training. Both regressed-by-simplification risks: each guard looks removable
until you know it cost a run."""
import pathlib

import pytest

from orchestration.checkpoint_registry import _upsert_registry

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_malformed_registry_row_is_tolerated_and_announced(tmp_path, capsys):
    """A row with MORE fields than the header (older writer or hand edit) makes
    DictReader stuff the extras under key None; DictWriter then refused the
    whole rewrite, crashing the save callback mid-training. The upsert must
    survive it, drop the unattributable extras LOUDLY, and normalize the file."""
    reg = tmp_path / "registry-stage2.csv"
    reg.write_text("checkpoint_path,lr\nckpt_old.pt,0.001,ORPHAN\n")
    _upsert_registry(reg, tmp_path / "ckpt_new.pt", {"lr": "0.0001"})
    out = capsys.readouterr().out
    assert "registry NOTE" in out and "ORPHAN" in out, "drop must be announced"
    text = reg.read_text()
    assert "ORPHAN" not in text and "ckpt_new.pt" in text, "file normalized + row added"


@pytest.mark.parametrize("trainer", [
    "training/train_stage1.py", "training/train_stage2.py",
    "training/train_lds.py", "training/train_refinement.py"])
def test_checkpoint_callback_is_wrapped_nonfatally(trainer):
    """The callback fires BETWEEN the checkpoint save and the epoch line; an
    unguarded exception there loses the run and the log line (checkpoint ends
    one epoch newer than the log). Every trainer must wrap the call."""
    from conftest import source_without_comments
    src = source_without_comments(_ROOT / trainer)
    # Two ways to satisfy the invariant: delegate the save to
    # _checkpoint_criterion.save_checkpoint (which runs the hook inside its own
    # try -- covered by test_save_checkpoint_hook_failure_does_not_kill_training)
    # by passing on_saved=on_checkpoint_saved; or, for a not-yet-extracted
    # trainer, call the hook inline wrapped in a try.
    if "save_checkpoint(" in src and "on_saved=on_checkpoint_saved" in src:
        return
    assert "on_checkpoint_saved(checkpoint_path, epoch)" in src
    call = src.index("on_checkpoint_saved(checkpoint_path, epoch)")
    assert "try:" in src[max(0, call - 200):call], (
        f"{trainer}: on_checkpoint_saved is not inside a try -- a bookkeeping "
        f"failure would kill the run again")
