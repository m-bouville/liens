"""
Tests for evaluation/_ae_stats_eval.py -- the shared AE+StatsHead
checkpoint-loading setup extracted from check_interpolation.py and
check_perturbation.py, which had independently reimplemented the same
~25 lines (device resolution, output_path defaulting, AE loading,
StatsHead construction/loading, two validation checks).
"""
from types import SimpleNamespace

import pytest

import evaluation._ae_stats_eval as ae_stats_eval
import evaluation.check_interpolation as ci
import evaluation.check_perturbation as cp
from evaluation._ae_stats_eval import load_ae_and_stats_head


def test_both_diagnostics_share_the_loader_rather_than_redefining_it():
    """
    REGRESSION: both modules must import load_ae_and_stats_head from
    evaluation._ae_stats_eval, not carry their own copy again. `is`, not
    numeric/behavioral equality -- the point is ONE implementation that
    can't silently drift, the same way the two original copies could
    have (they'd already drifted slightly in their "no stats_head"
    error message -- see no_stats_head_context below, which preserves
    that difference deliberately rather than erasing it).
    """
    assert ci.load_ae_and_stats_head is load_ae_and_stats_head
    assert cp.load_ae_and_stats_head is load_ae_and_stats_head


def _patch_build_ae(monkeypatch, has_stats_head: bool):
    """build_ae_from_checkpoint's own return shape, minus anything
    load_ae_and_stats_head doesn't touch before raising -- lets the
    "no stats_head" path be exercised directly, without a real,
    on-disk AE checkpoint."""
    checkpoint = {"epoch": 1, "config": {"size": 32}}
    if has_stats_head:
        checkpoint["stats_config"] = {"stat_names": ["avg_phi"]}
        checkpoint["stats_head_state"] = {}
    stream_configs = {"state": SimpleNamespace(channels=4, spatial_size=8)}
    monkeypatch.setattr(
        ae_stats_eval, "build_ae_from_checkpoint",
        lambda path, device: (None, None, checkpoint, stream_configs, "state"),
    )


def test_no_stats_head_message_matches_check_interpolation_exactly(monkeypatch, tmp_path):
    _patch_build_ae(monkeypatch, has_stats_head=False)
    ckpt = tmp_path / "no_stats.pt"
    with pytest.raises(ValueError) as exc:
        load_ae_and_stats_head(ckpt, "x", tmp_path / "out.png", "cpu")
    assert str(exc.value) == f"{ckpt} has no stats_head (trained with --stats-weight 0)"


def test_no_stats_head_message_matches_check_perturbation_exactly(monkeypatch, tmp_path):
    """check_perturbation.py's own message has always had one extra
    sentence check_interpolation.py's own doesn't -- must survive the
    merge exactly, not get silently unified."""
    _patch_build_ae(monkeypatch, has_stats_head=False)
    ckpt = tmp_path / "no_stats.pt"
    with pytest.raises(ValueError) as exc:
        load_ae_and_stats_head(
            ckpt, "x", tmp_path / "out.png", "cpu",
            no_stats_head_context="-- this check is built entirely around stats_head.",
        )
    assert str(exc.value) == (
        f"{ckpt} has no stats_head (trained with --stats-weight 0) "
        f"-- this check is built entirely around stats_head."
    )


def test_stats_head_loaded_correctly_when_present(monkeypatch, tmp_path):
    """
    output_path=None specifically, to genuinely exercise
    load_ae_and_stats_head's own output_subdir default-building logic
    (not just skip past it) -- _PYTHON_ROOT is monkeypatched locally
    to tmp_path first, since evaluation/*.py's own _PYTHON_ROOT is
    deliberately NOT covered by isolated_project_root (see that
    fixture's own docstring: these modules only use it for CLI
    defaults, so a caller providing output_path explicitly is the
    normal case -- but THIS test's whole point is the None/default
    case, so it needs its own, local isolation instead).
    """
    import torch
    fake_python_root = tmp_path / "fake_python_root"
    monkeypatch.setattr(ae_stats_eval, "_PYTHON_ROOT", fake_python_root)
    _patch_build_ae(monkeypatch, has_stats_head=True)
    monkeypatch.setattr(
        ae_stats_eval.StatsHead, "load_state_dict", lambda self, sd: None,
    )
    ctx = load_ae_and_stats_head(tmp_path / "ok.pt", "some_subdir", None, "cpu")
    assert ctx.stats_config["stat_names"] == ["avg_phi"]
    assert ctx.output_path.parent.name == "some_subdir"
    assert ctx.output_path.is_relative_to(tmp_path), (
        "output_path escaped tmp_path -- _PYTHON_ROOT patch didn't take effect, "
        "this would otherwise write into the real project output/ tree"
    )
    assert ctx.device == torch.device("cpu")
