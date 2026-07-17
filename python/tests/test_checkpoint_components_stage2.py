import torch
from pathlib import Path
from utils import load_datasets as load
from training.train_ae import train_autoencoder, train_stage1b, train_stage2
from training.checkpoint_components import load_ae_components, ComponentCheckpoint
from training.model_assembly import build_models_from_components
from models.latent_dynamics import LatentDynamics


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


def test_checkpoint_components_decoder_extraction_from_stage2(tmp_path, isolated_project_root):
    """Regression test for a real, reported crash: build_models_from_
    components() failed with 'Reassembled Autoencoder state_dict
    doesn't match the current model definition -- missing keys:
    decoder.*' on a real stage 4 run resuming from a stage 2
    checkpoint. Root cause: checkpoint_components.py's own
    _strip_prefix() only tried "decoder."/"decoders.shared." -- a
    stage 1b/2-derived checkpoint has separate per-stream decoders
    ("decoders.D0."/"decoders.D1."), so the decoder component was
    silently extracted as an EMPTY dict, making every real decoder key
    look "missing" downstream. Confirms the decoder component is
    non-empty AND that build_models_from_components can actually
    reconstruct a working model from it."""
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
    stage2_path = train_stage2(
        base_path=base_path, resume_from=stage1b_path, stats_weight=0.01,
        recon1_weight=0.5, stats1_weight=0.02, deriv_weight=1.0, deriv_weight_warmup_epochs=0,
        epochs=1, batch_size=4, num_workers=0,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve2.png",
    )

    components = load_ae_components(stage2_path, device="cpu")
    assert len(components["decoder"].state_dict) > 0, (
        "decoder component is empty -- _strip_prefix failed to match stage 2's own "
        "separate-decoder keys"
    )

    latent_channels = components["encoder"].config["latent_channels"]
    latent_spatial = components["encoder"].config["latent_spatial_size"]
    f_theta = LatentDynamics(latent_channels=latent_channels, n_theta=1, latent_spatial=latent_spatial,
                              hidden_dim=8, n_hidden_layers=1)
    components["lds"] = ComponentCheckpoint(
        state_dict=f_theta.state_dict(),
        config={"latent_channels": latent_channels, "n_theta": 1, "latent_spatial_size": latent_spatial,
                "hidden_dim": 8, "n_hidden_layers": 1},
        provenance={},
    )

    # THE actual regression check: must not raise the reported ValueError.
    ae, stats_head, f_theta_out, frozen = build_models_from_components(components, device="cpu")
    assert ae is not None
    print("build_models_from_components correctly reconstructed the model from a stage 2 checkpoint")
