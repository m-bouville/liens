import torch
from pathlib import Path
from utils import load_datasets as load
from training.train_ae import train_autoencoder


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
        # deterministic but distinctive per (run, step), like conftest.py's own tmp_run_dir
        value = step / 10000.0 + hash(name) % 100 / 10000.0
        arr = torch.full((size, size), value, dtype=torch.float16).numpy()
        arr.tofile(run_dir / load.snapshot_filename(step))
    return run_dir


def test_train_autoencoder_c1_alternation_actually_trains_deriv_stream(tmp_path):
    """THE real regression test for this session's biggest change: does
    the deriv bottleneck's weights actually CHANGE from random init
    after training, proving gradient genuinely reaches it now (unlike
    before this change, where it was structurally guaranteed to stay
    at its random initialization forever)."""
    base_dir = tmp_path / "datasets" / "32x32"
    base_dir.mkdir(parents=True)
    # 6 runs: with val_fraction=0.34, test_fraction=0.17 this should
    # give a non-empty split across all three groups from a small count.
    run_names = [f"T800_n010_s{i}" for i in range(6)]
    for name in run_names:
        _build_run_dir(base_dir, name, size=32)

    # Sweep-level metadata.txt (DIFFERENT format/purpose from each run's
    # own metadata.txt above -- see load_datasets.read_sweep_metadata's
    # own docstring): complete_run_dirs reads THIS to enumerate which
    # run directories exist at all, before ever touching an individual
    # run's own metadata.
    sweep_metadata_text = "\n".join([
        "Nx = 32", "Ny = 32", "temperatures = 0.8", "noises = 0.01",
        f"seeds = {','.join(str(i) for i in range(6))}",
        "subdirs =",
        *run_names,
    ])
    (base_dir / "metadata.txt").write_text(sweep_metadata_text)
    for name in run_names:
        (base_dir / name / "COMPLETE").touch()

    checkpoint_path = tmp_path / "ckpt.pt"

    result_path = train_autoencoder(
        size=32, base_path=tmp_path / "datasets",
        epochs=2, batch_size=4, base_channels=4,
        latent_names=["state", "deriv"], latent_modes=["autoencoder", "decoder"],
        latent_channels_decoder=4, latent_spatial_decoder=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats_weight=0.0,
        checkpoint_path=checkpoint_path, device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve.png",
    )
    assert result_path.exists()

    checkpoint = torch.load(result_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]
    assert "encoders.shared.bottlenecks.deriv.weight" in state
    deriv_weight_after = state["encoders.shared.bottlenecks.deriv.weight"]

    # Rebuild the SAME architecture fresh (same seed=0 used above) to
    # get the ACTUAL pre-training initial weight, rather than assuming
    # what it "should" be -- direct comparison, not inference.
    from models.encoder import Encoder
    from models.latent_streams import LatentStreamConfig, LatentStreamMode
    torch.manual_seed(0)
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=4, mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=4, mode=LatentStreamMode.DECODER),
    }
    fresh_encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    deriv_weight_initial = fresh_encoder.bottlenecks["deriv"].weight.detach()

    assert not torch.allclose(deriv_weight_after, deriv_weight_initial), (
        "deriv bottleneck weights are UNCHANGED from random init -- C1 alternation "
        "is not actually reaching this stream's parameters"
    )
    print(f"deriv bottleneck weight changed: max abs diff = "
          f"{(deriv_weight_after - deriv_weight_initial).abs().max().item():.6f}")

    assert checkpoint["config"]["stream_configs"].keys() == {"state", "deriv"}
    assert checkpoint.get("val_loss_c1") is not None
    print(f"val_loss_c1 recorded: {checkpoint['val_loss_c1']}")
