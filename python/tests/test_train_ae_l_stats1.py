import torch
from pathlib import Path
from utils import load_datasets as load
from training.train_ae import train_autoencoder


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


def test_l_stats1_trains_stats_head1_and_saves_checkpoint(tmp_path):
    base_dir = tmp_path / "datasets" / "32x32"
    base_dir.mkdir(parents=True)
    run_names = [f"T800_n010_s{i}" for i in range(6)]
    for name in run_names:
        _build_run_dir_with_stats(base_dir, name, size=32)
    sweep_meta = "\n".join([
        "Nx = 32", "Ny = 32", "temperatures = 0.8", "noises = 0.01",
        f"seeds = {",".join(str(i) for i in range(6))}", "subdirs =", *run_names,
    ])
    (base_dir / "metadata.txt").write_text(sweep_meta)
    for name in run_names:
        (base_dir / name / "COMPLETE").touch()

    checkpoint_path = tmp_path / "ckpt.pt"
    result_path = train_autoencoder(
        size=32, base_path=tmp_path / "datasets",
        epochs=2, batch_size=4, base_channels=4,
        latent_names=["state", "deriv"], latent_modes=["autoencoder", "decoder"],
        latent_channels_decoder=4, latent_spatial_decoder=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=checkpoint_path, device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve.png",
    )
    checkpoint = torch.load(result_path, map_location="cpu", weights_only=True)

    assert checkpoint["stats_head1_state"] is not None, "stats_head1 was not saved"
    assert any("net.0.weight" in k for k in checkpoint["stats_head1_state"])

    # Confirm stats_head1's weights actually moved from random init --
    # same proof pattern as the deriv bottleneck's own test.
    from training.stats_head import StatsHead
    fresh_head1 = StatsHead(latent_channels=4, stat_names=["avg_phi"], latent_spatial=4)
    fresh_weight = fresh_head1.net[0].weight.detach()
    trained_weight = checkpoint["stats_head1_state"]["net.0.weight"]
    assert not torch.allclose(fresh_weight, trained_weight), (
        "stats_head1's weights are UNCHANGED from random init -- L_stats1 is not "
        "actually reaching this module's parameters"
    )
    print(f"stats_head1 weight changed: max abs diff = "
          f"{(trained_weight - fresh_weight).abs().max().item():.6f}")
