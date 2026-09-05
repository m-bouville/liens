"""
Stage 1 must archive its own output before overwriting it on a self-resume.

Stage 2 has guarded this for a while; stage 1 did not. The asymmetry is easy to
miss because stage 1 has NO ancestor in the ordinary flow -- its output is never
at risk automatically. But an explicit `resume_from` commonly points at stage
1's own prior output, which is exactly how a size port is continued, and
force=True then overwrites it in place. The console prints

    WARNING: ...128x128-stage1.pt already exists and will be OVERWRITTEN

which is a warning, not a backup: the previous checkpoint is simply gone. On a
run whose epochs take hours, that is the expensive kind of gone.
"""
from pathlib import Path

from conftest import cached_stage1_ancestor
from orchestration.pipeline import run_from_params_file


def _params(tmp_path, base_path, resume_from=None, stage=1):
    # "# Stage 1", not "[stage 1]": _STAGE_HEADER is
    # ^\s*#\s*Stage\s+(\d+[a-zA-Z]?)\s*$. With the wrong header every key
    # lands in the GLOBAL section, where resume_from is deliberately dropped
    # (_NEVER_GLOBAL_DEFAULT_KEYS) -- so the run still worked, just without the
    # resume, and the test failed for a reason that had nothing to do with the
    # backup it was checking.
    lines = [
        "Nx = 32", "Ny = 32",
        "# Stage 1", "epochs = 1", "batch_size = 4", "base_channels = 4",
        "latent_channels = 4", "val_fraction = 0.34", "test_fraction = 0.17",
        "num_workers = 0", "min_step = 0", "stats0_weight = 0.01", "force = true",
    ]
    if resume_from is not None:
        lines.append(f"resume_from = {resume_from}")
    path = tmp_path / "params.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_stage1_archives_its_own_output_before_a_self_resume(tmp_path, isolated_project_root,
                                                              monkeypatch):
    """
    GUARDS overwriting stage 1's checkpoint in place when resume_from names it.
    Asserted on the backup FILE, not on console text: the warning was already
    being printed while the data was being destroyed, so the message proves
    nothing.
    """
    # stage_output_path is nested inside run_from_params_file, so the test
    # reproduces its one-line rule: _STAGE_DIRS[n] / f"{params stem}-stage{n}.pt"
    from utils.paths import _STAGE_DIRS
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_train_lds import _build_sweep

    base_path, ancestor = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, min_step=0,
        min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False)

    out = _STAGE_DIRS[1] / "params-stage1.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(ancestor.read_bytes())
    original = out.read_bytes()

    params = _params(tmp_path, base_path, resume_from=out)
    run_from_params_file(params, default_base=base_path, device="cpu")

    archives = [p for p in out.parent.glob(f"{out.stem}-*.pt")]
    assert archives, f"no archive written; {out.name} was overwritten in place"
    assert any(p.read_bytes() == original for p in archives), (
        "an archive exists but none holds the pre-overwrite bytes"
    )


def test_no_archive_when_stage1_is_not_resuming_from_itself(tmp_path, isolated_project_root):
    """
    GUARDS backing up unconditionally. In the ordinary flow stage 1 writes a
    file nothing else was going to keep, and an archive per run would
    accumulate copies of checkpoints no one asked to preserve.
    """
    from utils.paths import _STAGE_DIRS
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_train_lds import _build_sweep

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    out = _STAGE_DIRS[1] / "params-stage1.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The output must ALREADY EXIST, or _backup_before_overwrite is a no-op
    # and an unconditional backup would look identical to a conditional one.
    # This is the ordinary "stage 1 ran before, force=true, no resume_from"
    # case, where archiving would accumulate copies nobody asked to keep.
    out.write_bytes(b"a previous run's output")

    params = _params(tmp_path, base_path, resume_from=None)
    run_from_params_file(params, default_base=base_path, device="cpu")
    assert not list(out.parent.glob(f"{out.stem}-*.pt")), (
        "stage 1 archived its output without being asked to resume from it"
    )
