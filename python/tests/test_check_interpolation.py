"""
Tests for evaluation/check_interpolation.py. parse_fixed_triple and
find_all_triples are pure Python (find_all_triples uses only
utils.load_datasets, which has no torch dependency) -- both actually
run here and are checked directly.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_check_interpolation.py -v
"""
from pathlib import Path

import pytest

from evaluation.check_interpolation import parse_fixed_triple, find_all_triples


def test_parse_fixed_triple():
    run_dir, t1, t2, t3 = parse_fixed_triple("../../datasets/64x64/T800_n050_s79:100000:120000:140000")
    assert run_dir == Path("../../datasets/64x64/T800_n050_s79")
    assert (t1, t2, t3) == (100000, 120000, 140000)


def test_parse_fixed_triple_rejects_wrong_part_count():
    with pytest.raises(ValueError, match="run_dir:t1:t2:t3"):
        parse_fixed_triple("only:two:parts")


def test_find_all_triples_over_real_fixture(tmp_run_dir):
    """tmp_run_dir has 5 kept steps ([0, 1000, 2000, 3000, 4000], all
    valid/complete) -- 5 steps should yield exactly 3 consecutive
    triples: (0,1000,2000), (1000,2000,3000), (2000,3000,4000)."""
    run_dir, steps = tmp_run_dir
    triples = find_all_triples([run_dir], min_step=0)

    assert len(triples) == 3
    expected = [
        (run_dir, 0, 1000, 2000),
        (run_dir, 1000, 2000, 3000),
        (run_dir, 2000, 3000, 4000),
    ]
    assert triples == expected


def test_find_all_triples_respects_min_step(tmp_run_dir):
    """min_step=1500 should exclude step 0 and step 1000 entirely,
    leaving only steps [2000, 3000, 4000] -- exactly one triple."""
    run_dir, steps = tmp_run_dir
    triples = find_all_triples([run_dir], min_step=1500)

    assert len(triples) == 1
    assert triples == [(run_dir, 2000, 3000, 4000)]


def test_find_all_triples_empty_for_too_few_steps(tmp_run_dir):
    """min_step excluding all but the last 2 steps shouldn't produce
    any triples at all (need at least 3 consecutive kept steps)."""
    run_dir, steps = tmp_run_dir
    triples = find_all_triples([run_dir], min_step=3500)

    assert triples == []
