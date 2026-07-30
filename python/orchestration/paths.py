"""
Path anchors shared across the orchestration package (main.py and its
extracted helper modules). GENERAL POLICY (matches training/
train_refinement.py's own _PYTHON_ROOT): every checkpoint/output path
is built from _PYTHON_ROOT, never from a bare relative string like
"../output/...". Relative strings resolve against the process's CWD at
invocation time, which silently differs across bare CLI, `python -m`,
and being imported and called from another module -- exactly the
recurring "output ended up in the wrong place" bug hit repeatedly on
this project. Path(__file__) is anchored to THIS FILE's own on-disk
location instead, which is invariant regardless of how/from-where the
process was launched.

Extracted from main.py during its split into orchestration/ -- kept as
one shared module (not redefined per-file) specifically so every
orchestration module's notion of "where checkpoints/ is" stays
identical by construction, rather than N independently-computed copies
that could drift apart the way the same class of constant did across
train_ae.py/train_lds.py before that was fixed (see the project's own
path-policy history).
"""
from pathlib import Path

_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/orchestration/paths.py -> python/
# No "1b" entry: stage 1b was removed entirely (train_stage2 now builds
# the deriv stream itself, see training/extend_encoder.py), so a
# checkpoints/stage1b directory has nothing that could ever write to it.
# It lingered here only because the old wholesale "create every stage
# dir up front" loop in pipeline.py kept re-creating it on every run.
_STAGE_DIRS = {1: _PYTHON_ROOT / "checkpoints" / "stage1",
               2: _PYTHON_ROOT / "checkpoints" / "stage2",
               3: _PYTHON_ROOT / "checkpoints" / "stage3",
               "3a": _PYTHON_ROOT / "checkpoints" / "stage3a",
               "3b": _PYTHON_ROOT / "checkpoints" / "stage3b",
               4: _PYTHON_ROOT / "checkpoints" / "stage4",
               5: _PYTHON_ROOT / "checkpoints" / "stage5"}
_CHECKPOINTS_ROOT = _PYTHON_ROOT / "checkpoints"
