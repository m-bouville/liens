import torch
from pathlib import Path
from utils import load_datasets as load
from orchestration.pipeline import run_from_params_file
from orchestration.checkpoint_identification import identify_checkpoint_stage


def _build_run_dir_with_stats(base_dir, name, size=32):
    run_dir = base_dir / name
    run_dir.mkdir()
    steps = [0, 1000, 2000, 3000, 4000]
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 4000",
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


def test_pipeline_runs_1_then_1b(tmp_path):
    """THE actual end-to-end claim currently in scope: a params file
    with Stage 1 and 1b sections runs that chain correctly. Stage 2 is
    DELIBERATELY not included here -- it still assumes a single shared
    decoder and cannot yet load stage 1b's separate-D0/D1 output (a
    real, known, not-yet-fixed limitation, not something this test
    should paper over -- see train_stage2's own pending redesign)."""
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
stats_weight = 0.01
stat_names = avg_phi

# Stage 1b
epochs = 1
batch_size = 4
stats_weight = 0.01

# Stage 2
epochs = 0
"""
    params_path = tmp_path / "test_pipeline.txt"
    params_path.write_text(params_text)

    # run_from_params_file itself always attempts stage 3 onward too
    # (unlike stage 4/5, stage 2/3 aren't gated on section presence --
    # a separate, PRE-EXISTING pipeline behavior, unrelated to stage 1b
    # -- so a minimal params file like this one will hit a stage-3
    # config error afterward). Irrelevant to what THIS test verifies:
    # that stage 1 -> 1b themselves ran and produced a correct
    # checkpoint, which is checked directly from disk below regardless
    # of what happens to stages after that.
    try:
        run_from_params_file(params_path, default_base=base_path, device="cpu")
    except ValueError as e:
        if "train_lds()" not in str(e):
            raise

    from orchestration.paths import _STAGE_DIRS
    stage1b_files = list(_STAGE_DIRS["1b"].glob("test_pipeline-stage1b.pt"))
    assert len(stage1b_files) == 1, "stage 1b's own checkpoint file was not created"
    stage1b_checkpoint = torch.load(stage1b_files[0], map_location="cpu", weights_only=True)
    identity = identify_checkpoint_stage(stage1b_checkpoint)
    print(f"Stage 1b checkpoint identified as: {identity}")
    assert identity == "stage 1b (deriv stream decoder)"
    print("ALL CHECKS PASSED")


def test_pipeline_stage2_still_incompatible_with_stage1b_output(tmp_path):
    """Documents the CURRENT, known, not-yet-fixed boundary directly --
    if a '# Stage 2' section IS given after '# Stage 1b', the pipeline
    correctly reaches stage 2 (unlike before stage 1b existed, where it
    failed with '0 deriv streams'), but stage 2 itself still cannot
    load a separate-D0/D1 checkpoint. This SHOULD start failing once
    stage 2 is redesigned to match -- at which point this test should
    be replaced with a real end-to-end 1->1b->2 test, not deleted
    silently."""
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
stats_weight = 0.01
stat_names = avg_phi

# Stage 1b
epochs = 1
batch_size = 4
stats_weight = 0.01

# Stage 2
epochs = 1
batch_size = 4
deriv_weight = 1.0
"""
    params_path = tmp_path / "test_pipeline_stage2.txt"
    params_path.write_text(params_text)

    import pytest
    with pytest.raises(RuntimeError, match="decoders.shared"):
        run_from_params_file(params_path, default_base=base_path, device="cpu")
    print("Confirmed: stage 2 reaches the KNOWN decoder-structure mismatch, "
          "not the old (now-fixed) '0 deriv streams' error")


def test_pipeline_stops_after_stage1_without_stage1b_section(tmp_path):
    """If '# Stage 1b' isn't in the params file at all, the pipeline
    should stop at stage 1's own output -- not attempt stage 2 (which
    would fail anyway, requiring a multi-stream ancestor it doesn't have)."""
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
stats_weight = 0.0

# Stage 2
epochs = 1
batch_size = 4
"""
    params_path = tmp_path / "test_pipeline_no_1b.txt"
    params_path.write_text(params_text)

    final_checkpoint = run_from_params_file(params_path, default_base=base_path, device="cpu")
    checkpoint = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    identity = identify_checkpoint_stage(checkpoint)
    print(f"Final checkpoint (no '# Stage 1b' given): {identity}")
    assert identity == "stage 1 (autoencoder)", (
        f"pipeline should have stopped at stage 1 without a '# Stage 1b' section, got {identity}"
    )
    print("Correctly stopped after stage 1 -- stage 2 was NOT attempted despite being configured")
