import torch
from pathlib import Path

from conftest import cached_sweep
from utils import load_datasets as load
from orchestration.pipeline import run_from_params_file
from orchestration.checkpoint_identification import identify_checkpoint_stage


def _build_run_dir_with_stats(base_dir, name, size=32):
    run_dir = base_dir / name
    run_dir.mkdir()
    steps = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 7000",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        "seed = 1", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    for step in steps:
        value = step / 10000.0 + hash(name) % 100 / 10000.0
        arr = torch.full((size, size), value, dtype=torch.float16).numpy()
        arr.tofile(run_dir / load.snapshot_filename(step))
    import pandas as pd
    df = pd.DataFrame({"avg_phi": [s / 10000.0 for s in steps]}, index=steps)
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")
    return run_dir


def _build_sweep_uncached(tmp_path, n_runs=6, size=32):
    base_dir = tmp_path / "datasets" / f"{size}x{size}"
    base_dir.mkdir(parents=True)
    run_names = [f"T800_n010_s{i}" for i in range(n_runs)]
    for name in run_names:
        _build_run_dir_with_stats(base_dir, name, size=size)
    sweep_meta = "\n".join([
        f"Nx = {size}", f"Ny = {size}", "temperatures = 0.8", "noises = 0.01",
        f"seeds = {','.join(str(i) for i in range(n_runs))}", "subdirs =", *run_names,
    ])
    (base_dir / "metadata.txt").write_text(sweep_meta)
    for name in run_names:
        (base_dir / name / "COMPLETE").touch()
    return tmp_path / "datasets"


def test_pipeline_warns_but_ignores_a_stale_stage1b_section(tmp_path, isolated_project_root, capsys):
    """Stage 1b no longer exists as a separate pass -- train_stage2()
    now builds the deriv stream itself, directly from stage 1's own
    checkpoint (see training/extend_encoder.py's own module docstring
    for the full rationale). An existing params file with a leftover
    '# Stage 1b' section must not error or silently do nothing
    unexplained -- it should warn clearly and otherwise proceed exactly
    as if that section weren't there at all."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    params_text = f"""
Nx = 32
Ny = 32
base = {base_path}
min_step = 0
num_workers = 0
val_fraction = 0.34
test_fraction = 0.17
augment = false

# Stage 1
epochs = 1
batch_size = 4
base_channels = 4
latent_channels = 4
stats0_weight = 0.01
stat_names = avg_phi

# Stage 1b
epochs = 1
batch_size = 4
stats1_weight = 0.01

# Stage 2
epochs = 0
"""
    params_path = tmp_path / "test_pipeline_stale_1b.txt"
    params_path.write_text(params_text)

    try:
        run_from_params_file(params_path, default_base=base_path, device="cpu",
                                             run_sanity_checks=False)
    except ValueError as e:
        if "train_lds()" not in str(e):
            raise

    captured = capsys.readouterr()
    assert "Stage 1b" in captured.out and "no longer exists" in captured.out, (
        "expected an explicit warning about the stale '# Stage 1b' section, got none"
    )

    from orchestration.paths import _CHECKPOINTS_ROOT, _STAGE_DIRS
    # Strictly stronger than the old "no .pt files in stage1b/" check:
    # _STAGE_DIRS has no "1b" key at all anymore, and nothing should
    # create that directory either. The old wholesale "mkdir every stage
    # dir up front" loop in pipeline.py used to create checkpoints/stage1b
    # (and checkpoints/stage3, on a 3a/3b run) on EVERY run, long after
    # stage 1b stopped existing.
    assert "1b" not in _STAGE_DIRS
    assert not (_CHECKPOINTS_ROOT / "stage1b").exists(), (
        "checkpoints/stage1b was created despite stage 1b no longer being a real pass"
    )
    stage2_files = list(_STAGE_DIRS[2].glob("test_pipeline_stale_1b-stage2.pt"))
    assert len(stage2_files) == 1, "stage 2 should still have run normally, ignoring the stale section"


def test_pipeline_runs_1_then_2_directly(tmp_path, isolated_project_root):
    """THE actual end-to-end claim now in scope: stage 2 resumes DIRECTLY
    from stage 1's own checkpoint, with no stage 1b pass at all --
    confirms the full pipeline runs this way via run_from_params_file(),
    not just that train_stage2() works when called directly
    (test_train_stage2_l_deriv.py already covers that in more detail).
    recon1_weight/L_recon1 no longer exist as a concept at all (D1 is
    confirmed permanently unnecessary and no longer built at all) --
    nothing here to set or avoid."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    params_text = f"""
Nx = 32
Ny = 32
base = {base_path}
min_step = 0
num_workers = 0
val_fraction = 0.34
test_fraction = 0.17
augment = false
force = true

# Stage 1
epochs = 1
batch_size = 4
base_channels = 4
latent_channels = 4
stats0_weight = 0.01
stat_names = avg_phi

# Stage 2
epochs = 1
batch_size = 4
deriv_weight = 1.0
stats0_weight = 0.01
"""
    params_path = tmp_path / "test_pipeline_stage2.txt"
    params_path.write_text(params_text)

    # run_from_params_file itself always attempts stage 3 onward too
    # (stage 2/3 aren't gated on section presence the way stage 4/5
    # are), so a params file with no "# Stage 3" section will hit a
    # stage-3 config error afterward. Irrelevant to what THIS test
    # verifies: that stage 1 -> 2 themselves ran and produced a correct
    # checkpoint, checked directly from disk below regardless of what
    # happens after that.
    try:
        run_from_params_file(params_path, default_base=base_path, device="cpu",
                                             run_sanity_checks=False)
    except ValueError as e:
        if "train_lds()" not in str(e):
            raise

    from orchestration.paths import _STAGE_DIRS
    stage2_files = list(_STAGE_DIRS[2].glob("test_pipeline_stage2-stage2.pt"))
    assert len(stage2_files) == 1, "stage 2's own checkpoint file was not created"
    checkpoint = torch.load(stage2_files[0], map_location="cpu", weights_only=True)
    identity = identify_checkpoint_stage(checkpoint)
    print(f"Stage 2 checkpoint identified as: {identity}")
    assert identity == "stage 2 (latent-space validation)"
    assert checkpoint["stats_head1_state"] is not None
    assert checkpoint["config"]["decoder_for_stream"] == {"state": "D0"}
    assert checkpoint["config"]["stream_configs"]["deriv"]["mode"] == "pure_latent"
    assert not any(k.startswith("decoders.D1") for k in checkpoint["model_state"])
    print("Full pipeline (1 -> 2, no stage 1b at all) ran end to end via run_from_params_file")


def test_pipeline_stops_after_stage1_without_stage2_section(tmp_path, isolated_project_root):
    """If '# Stage 2' isn't in the params file at all, the pipeline
    should stop at stage 1's own output. Gating moved here from stage
    1b's own former section (see pipeline.py's own comment) -- '#
    Stage 1b' is deliberately absent from this params file entirely,
    confirming it's genuinely no longer needed for anything, not just
    optional."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    params_text = f"""
Nx = 32
Ny = 32
base = {base_path}
min_step = 0
num_workers = 0
val_fraction = 0.34
test_fraction = 0.17
augment = false

# Stage 1
epochs = 1
batch_size = 4
base_channels = 4
latent_channels = 4
stats0_weight = 0.0
"""
    params_path = tmp_path / "test_pipeline_no_stage2.txt"
    params_path.write_text(params_text)

    final_checkpoint = run_from_params_file(params_path, default_base=base_path, device="cpu",
                                             run_sanity_checks=False)
    checkpoint = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    identity = identify_checkpoint_stage(checkpoint)
    print(f"Final checkpoint (no '# Stage 2' given): {identity}")
    assert identity == "stage 1 (autoencoder)", (
        f"pipeline should have stopped at stage 1 without a '# Stage 2' section, got {identity}"
    )
    print("Correctly stopped after stage 1 -- stage 2 was NOT attempted")


def test_pipeline_ignores_global_resume_from_for_pipeline_managed_stages(tmp_path, isolated_project_root, capsys):
    """
    REGRESSION: a global 'resume_from' (e.g. left over from testing a
    stage-2 deriv_target_centered curriculum, or intended for some
    OTHER stage entirely) must not leak into stage 1 or stage 2's own
    kwargs at all. Real crash reproduced directly against the ORIGINAL
    code for stage 2 specifically: 'TypeError: training.train_stage2.
    train_stage2() got multiple values for keyword argument
    'resume_from''; stage 1 (which doesn't hardcode its own resume_from
    at all) would instead have SILENTLY tried to resume from the bogus
    global path and failed with a confusing FileNotFoundError, or
    worse, silently succeeded against the WRONG checkpoint if a path
    happened to exist.

    resume_from's own MEANING is always stage-specific (see
    _NEVER_GLOBAL_DEFAULT_KEYS's own comment in stage_params.py), so
    it's excluded from the global-default merge entirely -- no warning
    needed here, since this is now expected, documented behavior, not
    an error condition being caught after the fact.
    """
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    params_text = f"""
Nx = 32
Ny = 32
base = {base_path}
min_step = 0
num_workers = 0
val_fraction = 0.34
test_fraction = 0.17
augment = false
resume_from = {tmp_path / "not_a_real_checkpoint.pt"}

# Stage 1
epochs = 1
batch_size = 4
base_channels = 4
latent_channels = 4
stats0_weight = 0.01
stat_names = avg_phi

# Stage 2
epochs = 0
"""
    params_path = tmp_path / "test_pipeline_global_resume_from.txt"
    params_path.write_text(params_text)

    # run_from_params_file itself always attempts stage 3 onward too
    # (see test_pipeline_runs_1_then_2_directly's own identical
    # comment) -- irrelevant to what THIS test verifies.
    try:
        run_from_params_file(params_path, default_base=base_path, device="cpu",
                                             run_sanity_checks=False)
    except ValueError as e:
        if "train_lds()" not in str(e):
            raise

    from orchestration.paths import _STAGE_DIRS
    stage2_files = list(_STAGE_DIRS[2].glob("test_pipeline_global_resume_from-stage2.pt"))
    assert len(stage2_files) == 1, "stage 2's own checkpoint file was not created"
    checkpoint = torch.load(stage2_files[0], map_location="cpu", weights_only=True)
    identity = identify_checkpoint_stage(checkpoint)
    assert identity == "stage 2 (latent-space validation)", (
        f"pipeline should have completed stage 2 normally (built from stage 1, not the bogus "
        f"global resume_from path), got {identity}"
    )


def test_pipeline_honors_stage2_specific_resume_from_override(tmp_path, isolated_project_root, capsys):
    """
    A DIFFERENT case from the global one above: resume_from set
    EXPLICITLY under '# Stage 2' itself, pointing at a REAL, prior
    stage-2 checkpoint -- not a leftover global value, and not a
    mistake either: this is exactly what train_stage2's own
    deriv_target_centered curriculum is BUILT around (see its own
    docstring) -- continuing an already-trained stage-2 checkpoint
    with a different set of Stage-2 params, rather than restarting
    from stage 1. This must be HONORED (used INSTEAD of the pipeline's
    own default, stage 1's checkpoint), not stripped -- an earlier,
    overly-broad version of this fix stripped it unconditionally,
    which would have silently defeated this exact curriculum feature.
    """
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    params_text = f"""
Nx = 32
Ny = 32
base = {base_path}
min_step = 0
num_workers = 0
val_fraction = 0.34
test_fraction = 0.17
augment = false

# Stage 1
epochs = 1
batch_size = 4
base_channels = 4
latent_channels = 4
stats0_weight = 0.01
stat_names = avg_phi

# Stage 2
epochs = 1
batch_size = 4
deriv_weight = 1.0
stats0_weight = 0.01
force = true
"""
    params_path = tmp_path / "test_pipeline_stage2_ancestor.txt"
    params_path.write_text(params_text)

    # First pass: builds a real stage-1 -> stage-2 checkpoint pair, the
    # PRIOR checkpoint this test will explicitly resume from below --
    # exactly the "12 hours already spent" scenario this feature exists
    # for, just with a trivially cheap fixture instead of a real run.
    try:
        run_from_params_file(params_path, default_base=base_path, device="cpu",
                                             run_sanity_checks=False)
    except ValueError as e:
        if "train_lds()" not in str(e):
            raise

    from orchestration.paths import _STAGE_DIRS
    prior_stage2_files = list(_STAGE_DIRS[2].glob("test_pipeline_stage2_ancestor-stage2.pt"))
    assert len(prior_stage2_files) == 1
    prior_stage2_checkpoint = prior_stage2_files[0]

    # Second pass: a DIFFERENT params file, whose own '# Stage 2'
    # section explicitly resumes from the FIRST pass's own stage-2
    # checkpoint -- not stage 1's.
    params_text2 = f"""
Nx = 32
Ny = 32
base = {base_path}
min_step = 0
num_workers = 0
val_fraction = 0.34
test_fraction = 0.17
augment = false

# Stage 1
epochs = 1
batch_size = 4
base_channels = 4
latent_channels = 4
stats0_weight = 0.01
stat_names = avg_phi

# Stage 2
epochs = 1
batch_size = 4
deriv_weight = 1.0
stats0_weight = 0.01
resume_from = {prior_stage2_checkpoint}
force = true
"""
    params_path2 = tmp_path / "test_pipeline_stage2_resumed.txt"
    params_path2.write_text(params_text2)

    try:
        run_from_params_file(params_path2, default_base=base_path, device="cpu",
                                             run_sanity_checks=False)
    except ValueError as e:
        if "train_lds()" not in str(e):
            raise

    captured = capsys.readouterr()
    assert "resume_from" in captured.out and str(prior_stage2_checkpoint) in captured.out, (
        "expected a clear NOTE that Stage 2's own explicit resume_from override was used, got none"
    )

    stage2_files = list(_STAGE_DIRS[2].glob("test_pipeline_stage2_resumed-stage2.pt"))
    assert len(stage2_files) == 1, "stage 2's own checkpoint file was not created"
    checkpoint = torch.load(stage2_files[0], map_location="cpu", weights_only=True)
    assert checkpoint["stage2_config"]["resumed_from"] == str(prior_stage2_checkpoint), (
        "the resulting stage-2 checkpoint doesn't record the EXPLICIT override as its own ancestor "
        "-- it should have resumed from the prior stage-2 checkpoint, not stage 1's"
    )


def test_pipeline_backs_up_before_self_resume_overwrite(tmp_path, isolated_project_root, capsys):
    """
    REGRESSION: the actual scenario resume_from's own backup exists
    for -- the SAME params file (hence the SAME stage_output_path)
    used both to build the original stage-2 checkpoint AND, on a
    second run, with an explicit 'resume_from' pointing at THAT SAME
    checkpoint (continuing its own deriv_target_centered curriculum in
    place, per train_stage2's own docstring). force=True (needed for
    resolve_checkpoint to even attempt a rerun against an existing
    checkpoint) would otherwise overwrite BOTH the .pt AND, per
    _log_to_file's own open(log_path, "w"), the .log -- silently, with
    nothing left of either if the second run failed partway through.

    Deliberately does NOT check the normal stage1->stage2 flow here:
    that's covered implicitly by every OTHER test in this file never
    producing a spurious backup file, since none of them set an
    explicit resume_from at all.
    """
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    params_text = f"""
Nx = 32
Ny = 32
base = {base_path}
min_step = 0
num_workers = 0
val_fraction = 0.34
test_fraction = 0.17
augment = false
force = true

# Stage 1
epochs = 1
batch_size = 4
base_channels = 4
latent_channels = 4
stats0_weight = 0.01
stat_names = avg_phi

# Stage 2
epochs = 1
batch_size = 4
deriv_weight = 1.0
stats0_weight = 0.01
"""
    params_path = tmp_path / "test_pipeline_self_resume.txt"
    params_path.write_text(params_text)

    # First pass: produces the original stage-2 checkpoint AND its log,
    # both at the SAME path a second run of this SAME params file would
    # also write to.
    try:
        run_from_params_file(params_path, default_base=base_path, device="cpu",
                                             run_sanity_checks=False)
    except ValueError as e:
        if "train_lds()" not in str(e):
            raise

    from orchestration.paths import _STAGE_DIRS
    stage2_pt = _STAGE_DIRS[2] / "test_pipeline_self_resume-stage2.pt"
    stage2_log = _STAGE_DIRS[2] / "test_pipeline_self_resume-stage2.log"
    assert stage2_pt.exists() and stage2_log.exists(), "fixture assumption: first pass must produce both files"
    original_pt_bytes = stage2_pt.read_bytes()
    original_log_bytes = stage2_log.read_bytes()

    def _backup_glob(suffix):
        return list(_STAGE_DIRS[2].glob(f"test_pipeline_self_resume-stage2-*{suffix}"))

    assert not _backup_glob(".pt") and not _backup_glob(".log"), (
        "the normal stage1->stage2 flow must never produce a backup file -- "
        "found one after the FIRST pass, before any resume_from override was even used"
    )

    # Second pass: the SAME params file, now with an explicit
    # resume_from pointing at the SAME checkpoint the first pass just
    # produced -- the actual self-resume scenario.
    params_text_resumed = params_text.replace(
        "# Stage 2\nepochs = 1",
        f"# Stage 2\nresume_from = {stage2_pt}\nepochs = 1",
    )
    params_path.write_text(params_text_resumed)

    try:
        run_from_params_file(params_path, default_base=base_path, device="cpu",
                                             run_sanity_checks=False)
    except ValueError as e:
        if "train_lds()" not in str(e):
            raise

    backup_pts = _backup_glob(".pt")
    backup_logs = _backup_glob(".log")
    assert len(backup_pts) == 1, f"expected exactly one .pt backup, got {backup_pts}"
    assert len(backup_logs) == 1, f"expected exactly one .log backup, got {backup_logs}"
    assert backup_pts[0].read_bytes() == original_pt_bytes, (
        "the backed-up .pt doesn't match the ORIGINAL checkpoint's own bytes -- "
        "backup must happen BEFORE the overwrite, not after"
    )
    assert backup_logs[0].read_bytes() == original_log_bytes, (
        "the backed-up .log doesn't match the ORIGINAL log's own bytes -- "
        "backup must happen BEFORE _log_to_file's own truncating open(..., 'w')"
    )

    captured = capsys.readouterr()
    assert "backed up" in captured.out, "expected a clear NOTE that a backup was made, got none"


def _build_sweep(tmp_path, *args, **kwargs):
    """
    Memoized wrapper around this module's own _build_sweep_uncached --
    see conftest.cached_sweep for the full rationale and the read-only
    justification. tmp_path is accepted for call-site compatibility and
    deliberately IGNORED: the sweep lives in a shared, longer-lived
    directory so repeated calls with the same arguments reuse one build
    instead of rewriting the same synthetic snapshots per test. Anything
    a test WRITES (checkpoints, figures, logs) still goes to its own
    tmp_path, which this never touches.
    """
    return cached_sweep((__name__, args, tuple(sorted(kwargs.items()))),
                        lambda d: _build_sweep_uncached(d, *args, **kwargs))


def test_pipeline_sanity_checks_actually_run_at_the_DEFAULT(tmp_path, isolated_project_root, capsys):
    """
    REGRESSION: every other pipeline test in this file passes
    run_sanity_checks=False (they exercise parameter plumbing and stage
    sequencing, and the four diagnostics plus their matplotlib figures
    dominate the wall time of such a short run). That left the
    diagnostics block itself -- check_reconstruction,
    check_latent_channels, check_interpolation, check_perturbation, as
    wired into run_from_params_file -- with NO test coverage at all, so
    broken wiring there would go unnoticed.

    This test deliberately does NOT pass run_sanity_checks, exercising
    the production DEFAULT (True), and asserts all four actually ran.
    """
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    params_text = f"""
Nx = 32
Ny = 32
base = {base_path}
min_step = 0
num_workers = 0
val_fraction = 0.34
test_fraction = 0.17
augment = false
force = true

# Stage 1
epochs = 1
batch_size = 4
base_channels = 4
latent_channels = 4
stats0_weight = 0.01
stat_names = avg_phi

# Stage 2
epochs = 1
batch_size = 4
deriv_weight = 1.0
stats0_weight = 0.01
"""
    params_path = tmp_path / "test_pipeline_sanity_default.txt"
    params_path.write_text(params_text)

    # NOTE: run_sanity_checks deliberately NOT passed -- the whole point
    # is to exercise its production default.
    #
    # The trailing stage-3 config error is expected and irrelevant here,
    # for the same reason documented in
    # test_pipeline_runs_1_then_2_directly: run_from_params_file always
    # attempts stage 3 onward, so a params file with no "# Stage 3"
    # section fails there. The stage-2 diagnostics this test is about
    # run BEFORE that point, so they're already captured either way.
    try:
        run_from_params_file(params_path, default_base=base_path, device="cpu")
    except ValueError as e:
        if "train_lds()" not in str(e):
            raise

    printed = capsys.readouterr().out
    for expected in ("Sanity check: reconstruction quality",
                      "Sanity check: latent channel activations",
                      "Sanity check: interpolation consistency",
                      "Sanity check: perturbation response"):
        assert expected in printed, (
            f"'{expected}' never ran -- the pipeline's own diagnostics block is not wired up, "
            f"and every OTHER pipeline test disables it, so nothing else would catch this"
        )
