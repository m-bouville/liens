"""
Integration test for compare_rollout_training.py -- this script has
never had any test coverage, and was found to be completely
non-functional (three separate API-drift bugs: encoder() missing
theta=, which crashes on any theta-conditioned stream;
LatentDynamics.rollout() called without its own required z1_sequence
argument, which the class was redesigned around after this script was
written) before this test existed. Exercises the real, on-disk-fixture
path end to end so a future signature change breaks a fast test here,
not a confusing runtime crash the next time someone actually runs this
diagnostic.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_compare_rollout_training.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conftest import cached_sweep

from models.autoencoder import MultiStreamAutoencoder
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_dynamics import LatentDynamics
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from utils import load_datasets as load

from evaluation.compare_rollout_training import compare_rollout_training
from utils.window_parsing import parse_fixed_window
from models.constants import N_THETA


SIZE = 32
LATENT_CHANNELS = 4
STEPS = [0, 1000, 2000, 3000, 4000]


def _build_run_dir_uncached(tmp_path, name="T800_n010_s0", temperature=0.8, seed=0):
    run_dir = tmp_path / name
    run_dir.mkdir()
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {SIZE}", f"Ny = {SIZE}", "dt = 0.05", "steps = 4000",
        f"save_steps = {' '.join(str(s) for s in STEPS)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", f"temperature = {temperature}",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        f"seed = {seed}", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    for step in STEPS:
        arr = np.full((SIZE, SIZE), step / 10000.0, dtype="<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))
    pd.DataFrame([{"step": s, "avg_phi": s / 1000.0} for s in STEPS]).to_csv(
        run_dir / "statistics.csv", index=False)
    (run_dir / "COMPLETE").touch()
    return run_dir


def _build_run_dir(tmp_path, *args, **kwargs):
    return cached_sweep(
        (__name__, args, tuple(sorted(kwargs.items()))),
        lambda d: _build_run_dir_uncached(d, *args, **kwargs),
    )


def _build_ae_checkpoint(path: Path):
    """A real, theta-conditioned "deriv" stream -- deliberately, since
    that's exactly the configuration whose absence let the missing
    theta= bug through unnoticed for however long this script existed."""
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=LATENT_CHANNELS, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=LATENT_CHANNELS, spatial_size=8,
                                     mode=LatentStreamMode.DECODER, condition_on_theta=True),
    }
    encoder = Encoder(input_size=SIZE, in_channels=1, base_channels=4, stream_configs=stream_configs)
    decoder = Decoder(output_size=SIZE, out_channels=1, base_channels=4,
                       latent_channels=LATENT_CHANNELS, latent_spatial_size=8)
    ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                 stream_configs=stream_configs)
    checkpoint = {
        "model_state": ae.state_dict(), "epoch": 1, "val_loss": 0.01,
        "val_loss_ema": 0.01, "test_dirs": [],
        "config": {"size": SIZE, "base_channels": 4, "latent_channels": LATENT_CHANNELS,
                   "latent_spatial_size": 8, "stats_weight": 0.0,
                   "stream_configs": {
                       "state": {"channels": LATENT_CHANNELS, "spatial_size": 8, "mode": "autoencoder"},
                       "deriv": {"channels": LATENT_CHANNELS, "spatial_size": 8, "mode": "decoder",
                                 "condition_on_theta": True},
                   },
                   "recon_stream_name": "state"},
    }
    torch.save(checkpoint, path)


def _build_lds_checkpoint(path: Path, ae_checkpoint_path: Path, dt_cap: float = float("inf")):
    f_theta = LatentDynamics(latent_channels=LATENT_CHANNELS, n_theta=N_THETA, hidden_dim=8, n_hidden_layers=1,
                              dt_cap=dt_cap)
    checkpoint = {
        "model_state": f_theta.state_dict(), "epoch": 1, "val_loss": 0.05,
        "val_loss_ema": 0.05, "ae_checkpoint": str(ae_checkpoint_path), "test_dirs": [],
        "config": {"latent_channels": LATENT_CHANNELS, "n_theta": N_THETA, "hidden_dim": 8,
                   "n_hidden_layers": 1, "dt_cap": dt_cap},
        "data_config": {"min_step": 0, "min_stdev_phi": None, "window_length": 2},
    }
    torch.save(checkpoint, path)


def test_compare_rollout_training_runs_end_to_end(tmp_path, isolated_project_root):
    run_dir = _build_run_dir(tmp_path)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    _build_ae_checkpoint(ae_checkpoint_path)
    lds_a_path = tmp_path / "fake-stage3-a.pt"
    lds_b_path = tmp_path / "fake-stage3-b.pt"
    _build_lds_checkpoint(lds_a_path, ae_checkpoint_path)
    _build_lds_checkpoint(lds_b_path, ae_checkpoint_path)

    output_path = tmp_path / "compare.png"
    result_path = compare_rollout_training(
        checkpoint_a=lds_a_path, checkpoint_b=lds_b_path,
        fixed_windows=[f"{run_dir}:0:1000:2000:3000"],
        label_a="A", label_b="B", output_path=output_path, device="cpu",
    )
    assert result_path == output_path
    assert output_path.exists()


def test_rollout_errors_theta_conditioned_stream_does_not_crash(tmp_path, isolated_project_root):
    """
    REGRESSION: encoder() called without theta= crashes specifically
    when a theta-conditioned stream exists (Encoder.forward's own
    ValueError check) -- exercised directly here via a real,
    theta-conditioned "deriv" stream, the exact configuration that let
    this bug through unnoticed.
    """
    from evaluation.compare_rollout_training import rollout_errors
    from training.checkpoint_components import build_ae_from_checkpoint

    run_dir = _build_run_dir(tmp_path)
    ae_checkpoint_path = tmp_path / "fake-stage2-solo.pt"
    _build_ae_checkpoint(ae_checkpoint_path)
    lds_path = tmp_path / "fake-stage3-solo.pt"
    _build_lds_checkpoint(lds_path, ae_checkpoint_path)

    device = torch.device("cpu")
    ae, ae_encoder, ae_checkpoint, stream_configs, recon_stream_name = build_ae_from_checkpoint(
        ae_checkpoint_path, device)
    lds_checkpoint = torch.load(lds_path, map_location=device, weights_only=True)
    f_theta = LatentDynamics(**lds_checkpoint["config"]).to(device)
    f_theta.load_state_dict(lds_checkpoint["model_state"])
    f_theta.eval()

    errors = rollout_errors(
        run_dir, [0, 1000, 2000, 3000], ae_encoder, f_theta,
        ae_checkpoint["config"], device, recon_stream_name=recon_stream_name,
    )
    assert errors.shape == (3,)
    assert np.all(np.isfinite(errors))


def test_parse_fixed_window_handles_a_real_windows_path():
    """
    REGRESSION: compare_rollout_training.py used to carry its own,
    independent, naive split(':') copy of this parser (check_rollout.py
    already had the correct, right-to-left-scanning one) -- broke on
    any real Windows path, since 'C:\\Users\\...' has its own colon a
    naive split(':') can't tell apart from the ones separating step
    numbers. Exact string from a real failure.
    """
    s = r"C:\Users\matie\AppData\Local\Temp\sweep_sqx55qel\T800_n010_s0:0:1000:2000:3000"
    run_dir, steps = parse_fixed_window(s)
    assert str(run_dir) == r"C:\Users\matie\AppData\Local\Temp\sweep_sqx55qel\T800_n010_s0"
    assert steps == [0, 1000, 2000, 3000]


def test_compare_rollout_training_shares_parse_fixed_window_not_its_own_copy():
    """`is`, not just correct output -- the point is ONE implementation
    shared with check_rollout.py, not two that happen to agree today
    and can silently drift apart again."""
    import evaluation.check_rollout as cr
    import evaluation.compare_rollout_training as crt
    assert crt.parse_fixed_window is parse_fixed_window
    assert cr.parse_fixed_window is parse_fixed_window


def test_load_lds_threads_dt_cap_from_the_saved_checkpoint(tmp_path, isolated_project_root):
    """
    REGRESSION: this reconstruction is SEPARATE from
    model_assembly.py's own build_models_from_components and from
    evaluation._latent_eval.py's own copy, both of which already had
    this fix -- fixing dt_cap in either of THOSE did not fix it here.
    A checkpoint saved with a real, finite dt_cap used to silently
    evaluate as if dt_cap were still inf, with no error anywhere to
    indicate the mismatch.
    """
    from evaluation.compare_rollout_training import load_lds

    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    _build_ae_checkpoint(ae_checkpoint_path)
    lds_path = tmp_path / "fake-stage3-dtcap.pt"
    _build_lds_checkpoint(lds_path, ae_checkpoint_path, dt_cap=125.0)

    f_theta, _ = load_lds(lds_path, torch.device("cpu"))
    assert f_theta.dt_cap == 125.0


def test_load_lds_falls_back_to_inf_for_an_old_checkpoint_with_no_dt_cap_key(
    tmp_path, isolated_project_root,
):
    """An old checkpoint saved before dt_cap existed at all has no such
    key in its own saved config -- inf (the class's own exact-no-op
    default) is the correct fallback, not a KeyError."""
    from evaluation.compare_rollout_training import load_lds

    ae_checkpoint_path = tmp_path / "fake-stage2-old.pt"
    _build_ae_checkpoint(ae_checkpoint_path)
    lds_path = tmp_path / "fake-stage3-old.pt"
    torch.save({
        "model_state": LatentDynamics(latent_channels=LATENT_CHANNELS, n_theta=N_THETA,
                                       hidden_dim=8, n_hidden_layers=1).state_dict(),
        "epoch": 1, "val_loss": 0.05, "val_loss_ema": 0.05,
        "ae_checkpoint": str(ae_checkpoint_path), "test_dirs": [],
        "config": {"latent_channels": LATENT_CHANNELS, "n_theta": N_THETA, "hidden_dim": 8,
                   "n_hidden_layers": 1},  # deliberately: no "dt_cap" key at all
        "data_config": {"min_step": 0, "min_stdev_phi": None, "window_length": 2},
    }, lds_path)

    f_theta, _ = load_lds(lds_path, torch.device("cpu"))
    assert f_theta.dt_cap == float("inf")
