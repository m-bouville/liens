import warnings
from pathlib import Path
import torch
import pytest
from evaluation.lineage import resolve_lineage, _rebase_to_root, _ancestor_pointers

_DATA = Path(__file__).parent / "data"
_FIXTURE = _DATA / "32x32-stage5.pt"     # real stage-5 joint checkpoint


# ---------------------------------------------------------------------
# synthetic-chain tests (fast, no fixture) -- the core walk logic
# ---------------------------------------------------------------------
def _chain(tmp_path):
    p2 = tmp_path / "128x128-stage2-20260827_11h24.pt"
    p3a = tmp_path / "128x128-stage3a-20260828_02h12.pt"
    p3b = tmp_path / "128x128-stage3b-20260828_07h53.pt"
    p4 = tmp_path / "128x128-stage4-20260828_19h33.pt"
    torch.save({"ae_checkpoint": None}, p2)
    torch.save({"ae_checkpoint": str(p2)}, p3a)
    torch.save({"ae_checkpoint": str(p2)}, p3b)
    torch.save({"ae_state": {}, "f_theta_state": {}, "ae_checkpoint": str(p2),
                "lds_checkpoint": str(p3b)}, p4)
    return p2, p3a, p3b, p4


def test_full_lineage_from_stage4_with_registry(tmp_path):
    p2, p3a, p3b, p4 = _chain(tmp_path)
    reg = lambda p: str(p3a) if "stage3b" in Path(p).stem else None
    got = [lbl for lbl, _ in resolve_lineage(p4, registry_resume=reg)]
    assert got == ["stage 2", "stage 3a", "stage 3b", "stage 4"]


def test_without_registry_3a_is_absent(tmp_path):
    p2, p3a, p3b, p4 = _chain(tmp_path)
    assert [l for l, _ in resolve_lineage(p4)] == ["stage 2", "stage 3b", "stage 4"]


def test_3b_with_resumed_from_resolves_3a_without_registry(tmp_path):
    """After train_lds records resumed_from, a 3b carries its 3a pointer in the
    checkpoint itself -- so the walk reaches 3a with NO registry needed (the
    asymmetry with stage 4/5 is gone)."""
    p2 = tmp_path / "128x128-stage2-20260827_11h24.pt"
    p3a = tmp_path / "128x128-stage3a-20260828_02h12.pt"
    p3b = tmp_path / "128x128-stage3b-20260828_07h53.pt"
    torch.save({"ae_checkpoint": None}, p2)
    torch.save({"ae_checkpoint": str(p2)}, p3a)
    torch.save({"ae_checkpoint": str(p2), "resumed_from": str(p3a)}, p3b)   # <- new field
    assert [l for l, _ in resolve_lineage(p3b)] == ["stage 2", "stage 3a", "stage 3b"]


def test_select_keeps_chosen_plus_input(tmp_path):
    p2, p3a, p3b, p4 = _chain(tmp_path)
    reg = lambda p: str(p3a) if "stage3b" in Path(p).stem else None
    got = [l for l, _ in resolve_lineage(p4, select=["2", "3b"], registry_resume=reg)]
    assert got == ["stage 2", "stage 3b", "stage 4"]


def test_walk_from_3b(tmp_path):
    p2, p3a, p3b, p4 = _chain(tmp_path)
    reg = lambda p: str(p3a) if "stage3b" in Path(p).stem else None
    assert [l for l, _ in resolve_lineage(p3b, registry_resume=reg)] == \
        ["stage 2", "stage 3a", "stage 3b"]


def test_forked_stage2_warns_but_still_resolves(tmp_path):
    """A stage that refined/paired a DIFFERENT stage-2 than an ancestor used is
    a real fork -- warn (so the user knows), but still return the lineage: the
    rollout comparison is valid since each model uses its own encoder+f_theta."""
    p2, p3a, p3b, p4 = _chain(tmp_path)
    p2b = tmp_path / "128x128-stage2-20260826_03h22.pt"
    torch.save({"ae_checkpoint": None}, p2b)
    p3b_bad = tmp_path / "128x128-stage3b-20260828_23h08.pt"
    torch.save({"ae_checkpoint": str(p2b)}, p3b_bad)   # 3b on a DIFFERENT stage 2
    p4_bad = tmp_path / "128x128-stage4-bad.pt"
    torch.save({"ae_state": {}, "f_theta_state": {}, "ae_checkpoint": str(p2),
                "lds_checkpoint": str(p3b_bad)}, p4_bad)
    with pytest.warns(UserWarning, match="FORKED lineage"):
        got = [l for l, _ in resolve_lineage(p4_bad)]
    assert "stage 4" in got and "stage 3b" in got   # still resolves, not fatal


def test_missing_ancestor_warns_and_skips(tmp_path):
    p2, p3a, p3b, p4 = _chain(tmp_path)
    p4m = tmp_path / "128x128-stage4-missing.pt"
    torch.save({"ae_state": {}, "f_theta_state": {}, "ae_checkpoint": str(p2),
                "lds_checkpoint": str(tmp_path / "128x128-stage3b-GONE.pt")}, p4m)
    with pytest.warns(UserWarning, match="missing on disk"):
        got = [l for l, _ in resolve_lineage(p4m)]
    assert got == ["stage 2", "stage 4"]


# ---------------------------------------------------------------------
# path re-basing (the robustness fix)
# ---------------------------------------------------------------------
def test_rebase_maps_windows_absolute_path_to_checkpoints_root(tmp_path):
    """A Windows absolute pointer from another machine re-bases to
    <root>/stage<N>/<filename> and resolves on POSIX."""
    (tmp_path / "stage2").mkdir()
    target = tmp_path / "stage2" / "32x32-tiny-stage2.pt"
    target.write_bytes(b"x")
    stored = r"D:\work\NN\phase_field\python\checkpoints\stage2\32x32-tiny-stage2.pt"
    assert _rebase_to_root(stored, tmp_path) == target
    # unknown filename under root -> None (not fabricated)
    assert _rebase_to_root(r"D:\...\stage2\other.pt", tmp_path) is None


# ---------------------------------------------------------------------
# the REAL fixture (tests/data/32x32-stage5.pt)
# ---------------------------------------------------------------------
@pytest.mark.skipif(not _FIXTURE.exists(), reason="stage-5 fixture not present")
def test_fixture_is_a_joint_stage5_with_all_ancestor_pointers():
    ptrs = _ancestor_pointers(_FIXTURE, "cpu")
    # a real stage-5 joint checkpoint records all three ancestor pointers
    assert ptrs["ae_checkpoint"] and "stage2" in ptrs["ae_checkpoint"]
    assert ptrs["lds_checkpoint"] and "stage3b" in ptrs["lds_checkpoint"]
    assert ptrs["resumed_from"] and "stage4" in ptrs["resumed_from"]


@pytest.mark.skipif(not _FIXTURE.exists(), reason="stage-5 fixture not present")
def test_fixture_walk_resolves_when_ancestors_present_under_root(tmp_path):
    """Re-create the fixture's named ancestors under a checkpoints_root; the
    absolute pointers then re-base and the full lineage resolves."""
    from pathlib import PurePosixPath
    ptrs = _ancestor_pointers(_FIXTURE, "cpu")
    ae_ptr = ptrs["ae_checkpoint"]              # the fixture's OWN stage-2 pointer

    def _stage_and_name(stored):
        pp = PurePosixPath(str(stored).replace("\\", "/"))
        stage = next(p for p in pp.parts if p.startswith("stage"))
        return stage, pp.name

    for key, extra in (("ae_checkpoint", {"ae_checkpoint": None}),
                       ("lds_checkpoint", {"ae_checkpoint": ae_ptr}),
                       ("resumed_from", {"ae_state": {}, "f_theta_state": {},
                                         "ae_checkpoint": ae_ptr})):
        stage, name = _stage_and_name(ptrs[key])
        (tmp_path / stage).mkdir(parents=True, exist_ok=True)
        torch.save(extra, tmp_path / stage / name)   # every ancestor -> the SAME stage 2

    got = [lbl for lbl, _ in resolve_lineage(_FIXTURE, checkpoints_root=tmp_path)]
    assert got == ["stage 2", "stage 3b", "stage 4", "stage 5"]
