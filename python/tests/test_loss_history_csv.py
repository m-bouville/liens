"""
The per-epoch history must survive as NUMBERS, not only as pixels.

Before this it lived in memory during a run, was drawn into the loss-curve
PNG, and discarded -- the checkpoint stores 23 keys and none of them is the
history, and nothing wrote CSV or JSON. The console log is no substitute: it
is throttled to saved epochs, so a run that spikes and stops improving prints
nothing for its final hundreds of epochs.

That combination cost a real diagnosis. 128x128 stage 3a spiked at epoch 3568
and the only evidence was a vertical line on a plot; answering "how big, how
long, did it recover" meant reading pixels.
"""
import pathlib

import pytest

from conftest import source_without_comments
from utils.plots import write_loss_history

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_writes_a_csv_beside_the_figure(tmp_path):
    out = write_loss_history(tmp_path / "c.png", [1, 2, 3],
                              [1.0, 0.9, 0.8], [1.1, 1.0, 5.0], [1.1, 1.0, 1.0])
    assert out == tmp_path / "c.csv"
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "epoch,train_loss,val_loss,best_ema_so_far"
    assert len(lines) == 4
    assert lines[3].startswith("3,")


def test_full_precision_is_preserved(tmp_path):
    """
    GUARDS formatting with something like :.4f. The whole point is analysis
    after the fact -- a spike's exact magnitude and the EMA it is measured
    against both matter, and rounding throws that away silently.
    """
    v = 0.5965740123456789
    out = write_loss_history(tmp_path / "c.png", [1], [v], [v], [v])
    body = out.read_text().splitlines()[1]
    assert repr(v) in body, f"value was rounded: {body}"


def test_the_secondary_columns_are_included_when_present(tmp_path):
    out = write_loss_history(tmp_path / "c.png", [1], [1.0], [1.0], [1.0],
                              secondary_train=[0.5], secondary_val=[0.6],
                              secondary_label="1step")
    header = out.read_text().splitlines()[0]
    assert header.endswith("train_1step,val_1step")


def test_omitted_secondaries_do_not_add_empty_columns(tmp_path):
    out = write_loss_history(tmp_path / "c.png", [1], [1.0], [1.0], [1.0])
    assert out.read_text().splitlines()[0].count(",") == 3


def test_the_write_is_atomic(tmp_path):
    """
    GUARDS leaving a half-written file when a run is killed mid-write -- which
    is exactly when the history is most wanted. Rewritten every time the figure
    is, so it stays current.
    """
    src = source_without_comments(_ROOT / "utils/plots.py")
    block = src[src.index("def write_loss_history"):]
    block = block[:block.index("\ndef ")]
    assert ".tmp" in block and "replace(" in block
    out = write_loss_history(tmp_path / "c.png", [1], [1.0], [1.0], [1.0])
    assert not (tmp_path / "c.csv.tmp").exists(), "the temp file was left behind"
    assert out.exists()


@pytest.mark.parametrize("module", ["train_stage1", "train_stage2",
                                     "train_lds", "train_refinement"])
def test_every_trainer_writes_its_history(module):
    """
    Per-module, not a file-scope substring: the history is only useful if EVERY
    stage produces it, and a single missing call is exactly the shape of
    omission this suite has caught repeatedly.
    """
    src = source_without_comments(_ROOT / f"training/{module}.py")
    # A trainer either writes the history inline (paired 1:1 with loss_curve) OR
    # delegates both to the shared write_epoch_figures (which writes the curve,
    # its CSV and the scatter together -- guaranteed paired, covered by
    # test_training_loop). Either satisfies "every stage produces its history".
    if "write_epoch_figures(" in src:
        assert src.count("write_epoch_figures(") >= 1
    else:
        assert src.count("write_loss_history(") >= 1, f"{module} never writes its history"
        assert src.count("write_loss_history(") == src.count("loss_curve("), (
            f"{module} draws the curve more often than it writes the numbers"
        )


def test_full_weight_overlay_is_saved_in_the_csv(tmp_path):
    """The dashed full-weight overlay's DATA must survive as numbers in the CSV
    sidecar, not only as pixels in the PNG -- the whole reason this CSV exists.
    Present as a 'train_full_weight' column when given, absent otherwise (no empty
    trailing column for the stages/runs without a weight ramp)."""
    from utils.plots import write_loss_history
    ep, tr, va = [1, 2, 3], [0.5, 0.7, 0.9], [2.1, 2.2, 2.0]
    p = write_loss_history(tmp_path / "a.png", ep, tr, va, va,
                           train_full_weight=[2.0, 1.9, 1.8])
    header, first = p.read_text().splitlines()[0], p.read_text().splitlines()[1]
    assert header.split(",")[-1] == "train_full_weight"
    assert first.split(",")[-1] == "2.0"
    # absent when not provided
    p2 = write_loss_history(tmp_path / "b.png", ep, tr, va, va)
    assert "train_full_weight" not in p2.read_text().splitlines()[0]
