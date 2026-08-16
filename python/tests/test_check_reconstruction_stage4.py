import pytest
import torch
from pathlib import Path
from utils import load_datasets as load
from models.autoencoder import EncoderDecoderPair
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from evaluation.check_reconstruction import check_reconstruction
from models.constants import N_THETA


def _build_run_dir(base_dir, name, size=32):
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
    return run_dir


def test_check_reconstruction_loads_flat_stage4_checkpoint(tmp_path, capsys):
    """Regression test for a real reported bug: check_reconstruction
    decided single-pathway (Autoencoder) vs multi-stream
    (MultiStreamAutoencoder) construction based on len(stream_configs)
    -- but a stage 4/5 checkpoint (model_assembly.py's
    build_models_from_components) ALWAYS saves a flat, single-pathway
    EncoderDecoderPair, regardless of how many streams its own
    (inherited) stream_configs still lists -- the ancestor's full
    multi-stream config is carried forward in the checkpoint's config
    even though only ONE decoder actually exists in the saved weights.
    len(stream_configs) was never the right signal; the actual key
    structure ("encoder." flat vs "encoders." nested) is."""
    base_dir = tmp_path / "32x32"
    base_dir.mkdir(parents=True)
    run_names = [f"T800_n010_s{i}" for i in range(6)]
    for name in run_names:
        _build_run_dir(base_dir, name, size=32)
    sweep_meta = "\n".join([
        "Nx = 32", "Ny = 32", "temperatures = 0.8", "noises = 0.01",
        f"seeds = {','.join(str(i) for i in range(6))}", "subdirs =", *run_names,
    ])
    (base_dir / "metadata.txt").write_text(sweep_meta)
    for name in run_names:
        (base_dir / name / "COMPLETE").touch()

    torch.manual_seed(0)
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=4, mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=4, mode=LatentStreamMode.DECODER),
    }
    encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    decoder = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=4)
    ae = EncoderDecoderPair(encoder, decoder, stream_name="state", mode=LatentStreamMode.AUTOENCODER)

    checkpoint = {
        "model_state": ae.state_dict(),
        "epoch": 1, "val_loss": 0.08842,
        "test_dirs": [str(base_dir / n) for n in run_names[-2:]],
        "config": {
            "size": 32, "base_channels": 4, "latent_channels": 4, "latent_spatial_size": 4,
            "stream_configs": {n: {"channels": c.channels, "spatial_size": c.spatial_size, "mode": c.mode.value}
                                for n, c in stream_configs.items()},
            "recon_stream_name": "state",
            # NOTE: deliberately no decoder_for_stream key, matching a real stage 4/5 checkpoint
        },
    }
    ckpt_path = tmp_path / "stage4.pt"
    torch.save(checkpoint, ckpt_path)

    # THE actual test: must not raise the old "encoders.shared.* missing" RuntimeError
    check_reconstruction(checkpoint_path=ckpt_path, device="cpu", min_step=0,
                          output_path=tmp_path / "recon.png")
    output = capsys.readouterr().out
    assert "no derivative panel is possible" in output
    assert (tmp_path / "recon.png").exists()


@pytest.mark.slow
def test_check_reconstruction_loads_a_real_train_refinement_checkpoint(
    tmp_path, isolated_project_root,
):
    """Regression test for a real, reported bug: train_refinement.py's
    own checkpoint saves the AE's state under "ae_state" (not
    "model_state" -- that key is reserved for Stage 1/1b/2's own,
    single-model checkpoints; Stage 4/5's own checkpoint bundles
    ae_state/f_theta_state/stats_head_state together, so "model_state"
    alone would be ambiguous). check_reconstruction.py was only ever
    written to read "model_state" -- raised a confusing KeyError on any
    REAL Stage 4/5 checkpoint. The test above this one never caught
    this: it hand-constructs its own checkpoint dict using
    "model_state", so it never actually exercised train_refinement.py's
    own real save format at all. This test calls the REAL
    train_refinement() end to end instead, reproducing the user's own
    exact scenario directly rather than a hand-approximated one."""
    from models.autoencoder import MultiStreamAutoencoder
    from models.latent_dynamics import LatentDynamics
    from training.stats_head import StatsHead
    from training.train_refinement import train_refinement

    base_dir = tmp_path / "32x32"
    base_dir.mkdir(parents=True)
    run_names = [f"T800_n010_s{i}" for i in range(6)]
    for name in run_names:
        _build_run_dir(base_dir, name, size=32)
    sweep_meta = "\n".join([
        "Nx = 32", "Ny = 32", "temperatures = 0.8", "noises = 0.01",
        f"seeds = {','.join(str(i) for i in range(6))}", "subdirs =", *run_names,
    ])
    (base_dir / "metadata.txt").write_text(sweep_meta)
    for name in run_names:
        (base_dir / name / "COMPLETE").touch()
        import pandas as pd
        steps = [0, 1000, 2000, 3000, 4000]
        pd.DataFrame({"avg_phi": [s / 10000.0 for s in steps]}, index=steps).rename_axis(
            "step").to_csv(base_dir / name / "statistics.csv")

    latent_channels = 4
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=latent_channels, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=latent_channels, spatial_size=8,
                                     mode=LatentStreamMode.DECODER),
    }
    encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    decoder = Decoder(output_size=32, out_channels=1, base_channels=4,
                       latent_channels=latent_channels, latent_spatial_size=8)
    ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                 stream_configs=stream_configs)
    ae_path = tmp_path / "fake-stage2.pt"
    torch.save({
        "model_state": ae.state_dict(), "epoch": 1, "val_loss": 0.01, "test_dirs": [],
        "config": {"size": 32, "base_channels": 4, "latent_channels": latent_channels,
                   "latent_spatial_size": 8, "stats_weight": 0.01,
                   "stream_configs": {n: {"channels": c.channels, "spatial_size": c.spatial_size,
                                           "mode": c.mode.value} for n, c in stream_configs.items()},
                   "recon_stream_name": "state"},
    }, ae_path)

    f_theta = LatentDynamics(latent_channels=latent_channels, n_theta=N_THETA, hidden_dim=8, n_hidden_layers=1)
    lds_path = tmp_path / "fake-stage3.pt"
    torch.save({
        "model_state": f_theta.state_dict(), "epoch": 1, "val_loss": 0.05, "ae_checkpoint": "fake",
        "test_dirs": [],
        "config": {"latent_channels": latent_channels, "n_theta": N_THETA, "hidden_dim": 8, "n_hidden_layers": 1},
        "data_config": {"min_step": 0, "min_stdev_phi": None, "window_length": 2, "n_rollout_steps": 1},
    }, lds_path)

    stage4_path = tmp_path / "stage4.pt"
    train_refinement(
        base_path=tmp_path, ae_checkpoint_path=ae_path, lds_checkpoint_path=lds_path,
        freeze_decoder=True, rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.0,
        epochs=1, batch_size=4, n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        val_fraction=0.3, test_fraction=0.2, num_workers=0, checkpoint_path=stage4_path,
        device="cpu", seed=0, log_every_epoch=False,
        # Explicit, and isolated_project_root above as a second line of
        # defence. Without either, train_refinement's default loss_curve_path
        # is anchored to _PYTHON_ROOT and the figure lands in the REAL
        # output/stage4/ -- junk in the working tree of whoever runs the suite.
        loss_curve_path=tmp_path / "stage4-loss_curve.png",
    )
    assert stage4_path.exists()

    # THE actual test: must not raise the "model_state" KeyError.
    check_reconstruction(checkpoint_path=stage4_path, device="cpu", min_step=0,
                          output_path=tmp_path / "recon_real.png")
    assert (tmp_path / "recon_real.png").exists()
