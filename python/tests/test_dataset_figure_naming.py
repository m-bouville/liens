"""
Dataset-level figure names must never contain a stringified None.

Reported: `NonexNone-dz0dt.png` in output/datasets/, from

    python -m evaluation.check_parameter_dependence --lds-checkpoint ... \\
           --base-path ../datasets

`--size` is optional, and the default path was built as f"{size}x{size}-...".
Nothing raised; the file is simply named after a variable that was never set,
and it sorts alongside real ones as though it described a grid.

The size is not recoverable from the LDS checkpoint's own config (that holds
latent_channels/latent_spatial_size/hidden_dim, not the grid) and the AE
checkpoint is not loaded where the path is built -- but the checkpoint STEM
carries it, which is the same source _stage_folder_from_checkpoint_stem
already reads.
"""
import pathlib
import re

import pytest

from conftest import source_without_comments
from evaluation._latent_eval import _grid_size_for_dataset_filename as grid_size

_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("size,stem,expected", [
    (128, "128x128-stage3a", "128x128"),
    (64, "anything-at-all", "64x64"),          # explicit size always wins
    (None, "128x128-stage3a", "128x128"),      # THE regression
    (None, "64x64-stage3b", "64x64"),
    (None, "256x256-stage2", "256x256"),
])
def test_the_grid_size_is_recovered_when_the_cli_omits_it(size, stem, expected):
    assert grid_size(size, pathlib.Path(f"{stem}.pt")) == expected


def test_a_stem_without_a_size_falls_back_to_the_stem_not_to_None():
    """
    A figure named after the checkpoint that produced it is worse-grouped but
    not WRONG. `NonexNone` is a filename a reader will mistake for a bug in
    the physics.
    """
    out = grid_size(None, pathlib.Path("my-experiment.pt"))
    assert "None" not in out
    assert out == "my-experiment"


@pytest.mark.parametrize("size,stem", [
    (None, "128x128-stage3a"), (None, "no-size-here"), (256, "whatever"),
])
def test_no_stringified_None_ever_reaches_a_filename(size, stem):
    assert "None" not in grid_size(size, pathlib.Path(f"{stem}.pt"))


def test_the_default_path_actually_uses_the_helper():
    """
    GUARDS the f-string being rebuilt inline from `size` again. Checked on
    CODE, not raw source: the comment beside it necessarily names the broken
    pattern it replaced.
    """
    src = source_without_comments(_ROOT / "evaluation/_latent_eval.py")
    # The helper's OWN body legitimately contains the f-string -- it runs only
    # on the branch where size is not None. Excluded by name rather than by a
    # looser regex, so a copy reappearing anywhere else still fails.
    helper_start = src.index("def _grid_size_for_dataset_filename")
    helper_end = src.index("def ", helper_start + 10)
    outside = src[:helper_start] + src[helper_end:]
    assert not re.search(r'f"\{size\}x\{size\}', outside), (
        "a raw {size}x{size} f-string is back outside the helper -- it renders "
        "NonexNone when --size is omitted"
    )
    assert "_grid_size_for_dataset_filename(" in outside, (
        "the default path no longer calls the helper"
    )
