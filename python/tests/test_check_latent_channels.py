"""
Tests for evaluation/check_latent_channels.py's torch-free logic
(parse_fixed_frame). Actually run in this environment -- no torch
needed.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_check_latent_channels.py -v
"""
from pathlib import Path

import pytest

from evaluation.check_latent_channels import parse_fixed_frame


def test_parse_fixed_frame_basic():
    run_dir, step = parse_fixed_frame("../../datasets/64x64/T800_n050_s79:100000")
    assert run_dir == Path("../../datasets/64x64/T800_n050_s79")
    assert step == 100000


def test_parse_fixed_frame_windows_drive_letter():
    """Regression test, same class of bug fixed in
    check_rollout.parse_fixed_window: a Windows path's OWN colon (after
    the drive letter) must not be mistaken for the run_dir/step
    delimiter."""
    run_dir, step = parse_fixed_frame(r"D:\work\NN\phase_field\datasets\64x64\T950_n020_s79:400000")
    assert run_dir == Path(r"D:\work\NN\phase_field\datasets\64x64\T950_n020_s79")
    assert step == 400000


def test_parse_fixed_frame_rejects_missing_step():
    with pytest.raises(ValueError, match="run_dir:step"):
        parse_fixed_frame("only_a_run_dir")


def test_parse_fixed_frame_rejects_non_integer_step():
    with pytest.raises(ValueError, match="integer step"):
        parse_fixed_frame("some/run_dir:not_a_number")
