import torch
from pathlib import Path
from utils import load_datasets as load
from training.train_ae import train_autoencoder, train_stage1b
from training.train_lds import train_lds


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


def test_train_lds_loads_stage1b_checkpoint_correctly(tmp_path):
    """Regression test for a real bug: train_lds (stage 3) had the exact
    same construction gap check_reconstruction.py did -- assumed a
    single shared decoder, couldn't load stage 1b's separate-D0/D1
    checkpoint at all. Only the encoder is ever actually USED here
    (decoder weights are inert), but load_state_dict still needs the
    right key structure to succeed in the first place."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats_weight=0.01,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b.png",
    )

    # THE actual test: this must not raise the old "decoders.shared.*
    # missing" RuntimeError.
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage1b_path, stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        checkpoint_path=tmp_path / "stage3.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3.png",
    )
    assert lds_path.exists()
    print("train_lds successfully loaded a stage 1b checkpoint and completed training")
