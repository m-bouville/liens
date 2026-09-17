"""Guard: stage 2 must timestamp-pin its stage-1 ancestor, like stages 3/4/5.

Stage 1 rotates a single canonical name (128x128-stage1.pt) that the next
stage-1 run overwrites in place. Without archiving, stage 2's recorded ancestor
("resumed from 128x128-stage1.pt") no longer identifies WHICH stage 1 seeded it
-- which is how a killed, undertrained (ep26/50) stage 1 silently reseeded a
whole 3b->4->5 lineage without anyone noticing. Stages 3/4/5 already call
_archive_ancestor on their ancestors (see test_ancestor_timestamping.py's
test_refinement_records_timestamped_ancestors_in_signature); these guards assert
stage 2 now does the same for its stage-1 ancestor.

Source-inspection guards, matching test_ancestor_timestamping.py's style. If the
stage-2 section is refactored, update the asserted substrings to match.
"""
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _find(name: str) -> str:
    for cand in (_HERE / name, _HERE.parent / "orchestration" / name,
                 _HERE.parent / "training" / name, _HERE.parent / name):
        if cand.exists():
            return cand.read_text()
    for cand in _HERE.parent.rglob(name):
        return cand.read_text()
    raise FileNotFoundError(name)


def test_stage2_archives_its_stage1_ancestor():
    """The stage-2 section pins its auto-chained stage-1 ancestor via
    _archive_ancestor -- so the recorded ancestor is the timestamped copy, not
    the rotating canonical 128x128-stage1.pt the next stage-1 run overwrites."""
    src = _find("pipeline.py")
    assert "stage2_resume_from = _archive_ancestor(Path(stage2_resume_from))" in src


def test_stage2_records_timestamped_stage1_in_signature():
    """signature2's stage1_checkpoint is the archived (timestamped) path on the
    auto-chain, so a RETRAINED stage 1 (new mtime -> new name) invalidates stage
    2's cache instead of hiding behind the unchanged canonical name."""
    src = _find("pipeline.py")
    assert "str(stage1_checkpoint if stage2_overridden else stage2_resume_from)" in src


def test_stage2_passes_timestamped_ancestor_to_trainer():
    """train_stage2 receives the pinned path, so its 'Resuming from ...' log line
    names the timestamped stage-1 ancestor, not the generic canonical name."""
    src = _find("pipeline.py")
    assert "resume_from=stage2_resume_from" in src
