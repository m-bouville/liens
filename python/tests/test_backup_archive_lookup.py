"""
_backup_before_overwrite must find a backup that has been MOVED into an
archive subdirectory, not just one sitting beside the original.

Reported: with old copies tidied into checkpoints/stage2/_archives/, the
helper reported "not re-copying" only when the backup was still a sibling --
otherwise it made another copy of a file already archived, defeating the point
of having filed it away.

Identity is name AND size. The name already encodes the stem and the source's
own mtime, but a bare name match would accept an unrelated collision, and the
cost of a false positive is SKIPPING a backup -- losing the very file this
protects.
"""
import shutil

import pytest

from orchestration.stage_params import (
    _ARCHIVE_DIR_NAMES, _backup_before_overwrite, _find_existing_backup,
)


def _checkpoint(tmp_path, name="128x128-stage2.pt", content=b"weights", mtime=1_700_000_000):
    path = tmp_path / name
    path.write_bytes(content)
    import os
    os.utime(path, (mtime, mtime))
    return path


def _expected_backup_name(path):
    from datetime import datetime
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d_%Hh%M")
    return f"{path.stem}-{stamp}{path.suffix}"


def test_a_sibling_backup_is_found(tmp_path):
    src = _checkpoint(tmp_path)
    name = _expected_backup_name(src)
    shutil.copy2(src, tmp_path / name)
    assert _find_existing_backup(src, name) == tmp_path / name


@pytest.mark.parametrize("archive_dir", _ARCHIVE_DIR_NAMES)
def test_a_backup_filed_into_an_archive_dir_is_found(tmp_path, archive_dir):
    """THE regression: tidying a backup away must not cause it to be remade."""
    src = _checkpoint(tmp_path)
    name = _expected_backup_name(src)
    (tmp_path / archive_dir).mkdir()
    shutil.copy2(src, tmp_path / archive_dir / name)
    assert _find_existing_backup(src, name) == tmp_path / archive_dir / name


def test_no_second_copy_is_made_when_one_is_already_archived(tmp_path, capsys):
    src = _checkpoint(tmp_path)
    name = _expected_backup_name(src)
    (tmp_path / "_archives").mkdir()
    shutil.copy2(src, tmp_path / "_archives" / name)

    _backup_before_overwrite(src)

    assert not (tmp_path / name).exists(), "a redundant sibling copy was made"
    out = capsys.readouterr().out
    assert "already archived" in out
    assert "_archives" in out, "the message should say WHERE it found it"


def test_a_name_collision_with_a_different_size_is_not_trusted(tmp_path, capsys):
    """
    GUARDS matching on name alone. A false positive here SKIPS the backup, so
    the failure mode is losing the file -- worse than one redundant copy.
    """
    src = _checkpoint(tmp_path, content=b"the real weights, longer")
    name = _expected_backup_name(src)
    (tmp_path / "_archives").mkdir()
    (tmp_path / "_archives" / name).write_bytes(b"something else")

    assert _find_existing_backup(src, name) is None
    _backup_before_overwrite(src)
    assert (tmp_path / name).exists(), "the backup must still be made"
    assert "different size" in capsys.readouterr().out


def test_a_backup_is_still_made_when_none_exists(tmp_path):
    src = _checkpoint(tmp_path)
    _backup_before_overwrite(src)
    assert (tmp_path / _expected_backup_name(src)).exists()


def test_a_changed_source_gets_its_own_backup(tmp_path):
    """
    The name is keyed on the source's mtime, so a genuinely NEW version is a
    different name and must be archived separately -- the archive lookup must
    not swallow it.
    """
    src = _checkpoint(tmp_path, mtime=1_700_000_000)
    _backup_before_overwrite(src)
    first = _expected_backup_name(src)

    import os
    src.write_bytes(b"retrained weights")
    later = 1_700_000_000 + 7200
    os.utime(src, (later, later))
    _backup_before_overwrite(src)
    second = _expected_backup_name(src)

    assert first != second
    assert (tmp_path / first).exists() and (tmp_path / second).exists()


def test_a_missing_source_is_a_no_op(tmp_path):
    _backup_before_overwrite(tmp_path / "never-written.pt")
    assert not list(tmp_path.iterdir())


def test_the_archive_dir_list_covers_the_reported_convention():
    assert "_archives" in _ARCHIVE_DIR_NAMES
