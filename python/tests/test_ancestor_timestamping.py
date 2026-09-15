"""
Stage N+1 must SAVE and LOG its stage-N ancestor under a TIMESTAMPED name, not the
bare generic `128x128-stageN.pt`.

Why this matters: stages overwrite a generic checkpoint name (`128x128-stage2.pt`,
`128x128-stage3b.pt`). If stage N+1 records/loads its ancestor by that generic
name, then once a later run overwrites it, the ancestor is ambiguous -- and
`compare_f_theta --with-ancestors` (and reruns) silently resolve to whatever now
sits at the path, which is exactly how two "same-config" runs ended up on
different f_theta. `_archive_ancestor` fixes this: it copies the generic file to
its mtime-timestamped name (`128x128-stage2-YYYYMMDD_HHhMM.pt`) and returns that
name, which the pipeline records in the checkpoint signature and the trainer prints
in its load line.

Two levels:
  - unit: _archive_ancestor actually produces + returns the timestamped path
    (runs here -- pure file op, no GPU);
  - structural: run_refinement_stage records the timestamped ancestor and passes it
    to train_refinement (which logs it), for BOTH the stage-2 encoder and the
    stage-3b f_theta -- the edge that regressed.
"""
import re
import time
from pathlib import Path

import pytest

# _archive_ancestor is a pure stdlib file op -- import and run it directly.
try:
    from orchestration.stage_params import _archive_ancestor, _timestamped_name
except ImportError:                       # flat layout
    from stage_params import _archive_ancestor, _timestamped_name


_HERE = Path(__file__).resolve().parent
# 128x128-stage2-20260915_11h24.pt  -> the -YYYYMMDD_HHhMM timestamp before .pt
_TIMESTAMP_RE = re.compile(r"-\d{8}_\d{2}h\d{2}\.pt$")


def _find(name: str) -> str:
    for cand in (_HERE / name, _HERE.parent / "orchestration" / name,
                 _HERE.parent / "training" / name, _HERE.parent / name):
        if cand.exists():
            return cand.read_text()
    for cand in _HERE.parent.rglob(name):
        return cand.read_text()
    raise FileNotFoundError(name)


# --------------------------------------------------------------------------- #
# unit: _archive_ancestor produces + returns a timestamped copy
# --------------------------------------------------------------------------- #
def test_archive_ancestor_returns_timestamped_name(tmp_path):
    generic = tmp_path / "128x128-stage2.pt"
    generic.write_bytes(b"encoder-weights")
    returned = _archive_ancestor(generic)
    # the returned path is the timestamped name, NOT the bare generic one
    assert returned != generic
    assert _TIMESTAMP_RE.search(returned.name), \
        f"{returned.name} is not timestamped (-YYYYMMDD_HHhMM.pt)"
    assert returned.name.startswith("128x128-stage2-")


def test_archive_ancestor_actually_saves_the_timestamped_file(tmp_path):
    generic = tmp_path / "128x128-stage3b.pt"
    generic.write_bytes(b"f-theta-weights")
    returned = _archive_ancestor(generic)
    # the timestamped copy exists on disk and is a faithful copy
    assert returned.exists()
    assert returned.read_bytes() == b"f-theta-weights"
    # the generic original is untouched (copy, not move)
    assert generic.exists()


def test_archive_ancestor_timestamp_follows_mtime_not_now(tmp_path):
    """The name uses the file's OWN mtime, so a checkpoint keeps a stable
    timestamped identity regardless of when it is later consumed."""
    generic = tmp_path / "128x128-stage2.pt"
    generic.write_bytes(b"w")
    expected = _timestamped_name(generic).name    # derived from mtime
    returned = _archive_ancestor(generic)
    assert returned.name == expected


def test_archive_ancestor_idempotent_same_mtime(tmp_path):
    """Consuming the same unchanged ancestor twice must not make a second copy
    (same mtime -> same name -> returns the existing one)."""
    generic = tmp_path / "128x128-stage2.pt"
    generic.write_bytes(b"w")
    first = _archive_ancestor(generic)
    second = _archive_ancestor(generic)
    assert first == second


def test_retrained_ancestor_gets_a_new_timestamp(tmp_path):
    """A RETRAINED ancestor (new mtime) yields a DIFFERENT timestamped name, so the
    recorded pointer distinguishes the two -- the core reason reruns diverged."""
    generic = tmp_path / "128x128-stage3b.pt"
    generic.write_bytes(b"v1")
    name1 = _archive_ancestor(generic).name
    # simulate a retrain: rewrite with a later mtime
    later = generic.stat().st_mtime + 120
    generic.write_bytes(b"v2")
    import os
    os.utime(generic, (later, later))
    name2 = _archive_ancestor(generic).name
    assert name1 != name2, "retrained ancestor must not reuse the old timestamp"


# --------------------------------------------------------------------------- #
# structural: N+1 records + logs the timestamped ancestor (both edges)
# --------------------------------------------------------------------------- #
def test_refinement_records_timestamped_ancestors_in_signature():
    """run_refinement_stage must archive BOTH ancestors (stage-2 encoder AND
    stage-3b f_theta) via _archive_ancestor and record the RETURNED timestamped
    paths in the checkpoint signature -- so N+1's saved ancestor pointer is
    timestamped, not the overwritten generic name."""
    src = _find("pipeline.py")
    assert "_ae_ancestor = _archive_ancestor(stage2_checkpoint)" in src
    assert "_f_theta_ancestor = _archive_ancestor(stage3_checkpoint)" in src
    assert 'str(_ae_ancestor)' in src and 'str(_f_theta_ancestor)' in src


def test_refinement_passes_timestamped_ancestors_to_trainer_for_logging():
    """The trainer receives the timestamped paths (not the generic names), so its
    'loaded ... / f_theta from ...' log line names the timestamped ancestor."""
    src = _find("pipeline.py")
    assert "ae_checkpoint_path=_ae_ancestor" in src
    assert "lds_checkpoint_path=_f_theta_ancestor" in src


def test_trainer_logs_the_ancestor_path_it_was_given():
    """train_refinement builds its load-line ancestor_note from the paths it is
    handed -- so passing timestamped paths (above) makes the LOG timestamped."""
    src = _find("train_refinement.py")
    assert "f_theta from" in src and "ancestor_note" in src
    assert "_pin(ae_checkpoint_path" in src and "_pin(lds_checkpoint_path" in src
