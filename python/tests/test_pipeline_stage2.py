import torch
from pathlib import Path
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


def _build_sweep(tmp_path, n_runs=6, size=32):
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
        run_from_params_file(params_path, default_base=base_path, device="cpu")
    except ValueError as e:
        if "train_lds()" not in str(e):
            raise

    captured = capsys.readouterr()
    assert "Stage 1b" in captured.out and "no longer exists" in captured.out, (
        "expected an explicit warning about the stale '# Stage 1b' section, got none"
    )

    from orchestration.paths import _STAGE_DIRS
    stage1b_files = list(_STAGE_DIRS["1b"].glob("*.pt"))
    assert len(stage1b_files) == 0, (
        "a stage 1b checkpoint file was created despite stage 1b no longer being a real pass"
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
        run_from_params_file(params_path, default_base=base_path, device="cpu")
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

    final_checkpoint = run_from_params_file(params_path, default_base=base_path, device="cpu")
    checkpoint = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    identity = identify_checkpoint_stage(checkpoint)
    print(f"Final checkpoint (no '# Stage 2' given): {identity}")
    assert identity == "stage 1 (autoencoder)", (
        f"pipeline should have stopped at stage 1 without a '# Stage 2' section, got {identity}"
    )
    print("Correctly stopped after stage 1 -- stage 2 was NOT attempted")
