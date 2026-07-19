import torch
from pathlib import Path
from utils import load_datasets as load
from training.train_ae import train_autoencoder, train_stage2
import pytest


def _build_run_dir(base_dir, name, temperature=0.8, noise=0.01, seed_val=1, size=32):
    run_dir = base_dir / name
    run_dir.mkdir()
    steps = [0, 1000, 2000, 3000, 4000]
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 4000",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", f"temperature = {temperature}",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", f"noise = {noise}",
        f"seed = {seed_val}", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    for step in steps:
        value = step / 10000.0 + hash(name) % 100 / 10000.0
        arr = torch.full((size, size), value, dtype=torch.float16).numpy()
        arr.tofile(run_dir / load.snapshot_filename(step))
    # statistics.csv, needed for stage 1's stats_head AND stage 2
    import pandas as pd
    df = pd.DataFrame({"avg_phi": [v / 10000.0 for v in steps]}, index=steps)
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")
    return run_dir


def _build_sweep(tmp_path, n_runs=6, size=32):
    base_dir = tmp_path / "datasets" / f"{size}x{size}"
    base_dir.mkdir(parents=True)
    run_names = [f"T800_n010_s{i}" for i in range(n_runs)]
    for name in run_names:
        _build_run_dir(base_dir, name, size=size)
    sweep_metadata_text = "\n".join([
        f"Nx = {size}", f"Ny = {size}", "temperatures = 0.8", "noises = 0.01",
        f"seeds = {','.join(str(i) for i in range(n_runs))}",
        "subdirs =", *run_names,
    ])
    (base_dir / "metadata.txt").write_text(sweep_metadata_text)
    for name in run_names:
        (base_dir / name / "COMPLETE").touch()
    return tmp_path / "datasets"


def test_stage2_rejects_single_stream_ancestor(tmp_path):
    """L_deriv has no meaning without a deriv stream -- must raise
    clearly, not silently do something else. The two tests that used
    to sit here (deriv-training-in-stage-2, freeze_outer_layers-with-
    multi-stream) were superseded by test_train_stage2_c0c1.py's own,
    more complete end-to-end test once stage 2 was redesigned for the
    real stage 1a/1b split (separate D0/D1, both stats heads, all
    five loss terms) -- removed here rather than kept skipped, since
    resurrecting their old latent_names/latent_modes-based fixtures
    would just duplicate what that file already covers more
    thoroughly."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)

    stage1_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1_single.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1.png",
    )

    with pytest.raises(ValueError, match="exactly one"):
        train_stage2(
            base_path=base_path, resume_from=stage1_path,
            deriv_weight=1.0, stats0_weight=0.0,
            epochs=1, batch_size=4, num_workers=0,
            min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / "stage2_should_fail.pt", device="cpu",
        )
