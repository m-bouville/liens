"""
_find_existing_backup recognizes a stage2 <-> stage2a stem-swapped archive.

A 2a run and a joint stage-2 run share the checkpoint lineage, and an existing
backup often got renamed across that pair. The backup dedup must therefore treat
"-stage2-<ts>.pt" and "-stage2a-<ts>.pt" as the same archive (same timestamp +
size), so it does not create a redundant copy. Same identity test the function
already uses for the exact-name case (name + size), extended to the one stem swap.
The SIZE guard must still reject a same-named-but-different-size file -- skipping
a backup of genuinely different content is the one failure this must never make.
"""
from pathlib import Path

from orchestration.stage_params import _find_existing_backup


def _write(p: Path, nbytes: int):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * nbytes)


def test_finds_stage2a_archive_when_backing_up_stage2(tmp_path):
    """Backing up 128x128-stage2.pt (whose backup_name is stage2-<ts>): an existing
    stage2a-<ts> of matching size counts as the archive -> no duplicate."""
    source = tmp_path / "128x128-stage2.pt"; _write(source, 1000)
    existing = tmp_path / "128x128-stage2a-20260910_10h35.pt"; _write(existing, 1000)
    found = _find_existing_backup(source, "128x128-stage2-20260910_10h35.pt")
    assert found == existing


def test_finds_stage2_archive_when_backing_up_stage2a(tmp_path):
    """Swap direction: backing up a stage2a source finds an existing stage2 archive."""
    source = tmp_path / "128x128-stage2a.pt"; _write(source, 1000)
    existing = tmp_path / "128x128-stage2-20260910_10h35.pt"; _write(existing, 1000)
    found = _find_existing_backup(source, "128x128-stage2a-20260910_10h35.pt")
    assert found == existing


def test_size_mismatch_is_not_accepted_as_the_archive(tmp_path):
    """A stem-swapped file of DIFFERENT size is genuinely different content --
    must NOT be treated as the archive (else a real backup gets skipped)."""
    source = tmp_path / "128x128-stage2.pt"; _write(source, 1000)
    _write(tmp_path / "128x128-stage2a-20260910_10h35.pt", 2000)   # different size
    found = _find_existing_backup(source, "128x128-stage2-20260910_10h35.pt")
    assert found is None


def test_exact_name_still_wins(tmp_path):
    """The exact-name archive is still matched (unchanged behavior)."""
    source = tmp_path / "128x128-stage2.pt"; _write(source, 1000)
    exact = tmp_path / "128x128-stage2-20260910_10h35.pt"; _write(exact, 1000)
    found = _find_existing_backup(source, "128x128-stage2-20260910_10h35.pt")
    assert found == exact


def test_non_stage2_stem_unaffected(tmp_path):
    """A stage1/stage3 name has no swap sibling -> only the exact name is sought."""
    source = tmp_path / "128x128-stage1.pt"; _write(source, 1000)
    _write(tmp_path / "128x128-stage1a-20260910_10h35.pt", 1000)   # not a real pairing
    found = _find_existing_backup(source, "128x128-stage1-20260910_10h35.pt")
    assert found is None
