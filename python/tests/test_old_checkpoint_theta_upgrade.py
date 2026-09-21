"""Loading REAL pre-2-feature-theta checkpoints (n_theta=1 weights and
configs) through every production loader must upgrade them silently: model
built at N_THETA, old weights zero-padded, forward accepting the new 2-wide
theta. Every earlier theta test built FRESH 2-theta checkpoints, so an
unpadded loader passed the whole suite while crashing on the user's actual
.pt files -- exactly the gap this file closes. Old checkpoints are
constructed here the way the OLD code would have written them: 1-theta
modules, config recording n_theta=1."""
from pathlib import Path

import numpy as np
import pytest
import torch

from models.constants import N_THETA
from models.encoder import Encoder
from models.decoder import Decoder
from models.autoencoder import MultiStreamAutoencoder
from models.latent_dynamics import LatentDynamics
from models.latent_streams import LatentStreamConfig, LatentStreamMode


def _old_style_multistream_ae_checkpoint(path: Path, size=32, latent_channels=4):
    """A stage-2-style checkpoint EXACTLY as the pre-theta-change code saved
    it: encoder built at n_theta=1, deriv stream theta-conditioned, config
    with no n_theta key (the AE config never recorded one)."""
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=latent_channels, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER, condition_on_theta=False),
        "deriv": LatentStreamConfig(name="deriv", channels=latent_channels, spatial_size=8,
                                     mode=LatentStreamMode.DECODER, condition_on_theta=True),
    }
    encoder = Encoder(input_size=size, in_channels=1, base_channels=4,
                       stream_configs=stream_configs, n_theta=1)  # OLD width
    decoder = Decoder(output_size=size, out_channels=1, base_channels=4,
                       latent_channels=latent_channels, latent_spatial_size=8)
    d1 = Decoder(output_size=size, out_channels=1, base_channels=4,
                  latent_channels=latent_channels, latent_spatial_size=8)
    ae = MultiStreamAutoencoder(encoders={"shared": encoder},
                                 decoders={"D0": decoder, "D1": d1},
                                 stream_configs=stream_configs,
                                 decoder_for_stream={"state": "D0", "deriv": "D1"})
    checkpoint = {
        "model_state": ae.state_dict(), "epoch": 3, "val_loss": 0.1, "test_dirs": [],
        "config": {"size": size, "base_channels": 4, "latent_channels": latent_channels,
                   "latent_spatial_size": 8, "stats_weight": 0.0,
                   "stream_configs": {
                       "state": {"channels": latent_channels, "spatial_size": 8,
                                 "mode": "autoencoder"},
                       "deriv": {"channels": latent_channels, "spatial_size": 8,
                                 "mode": "decoder", "condition_on_theta": True},
                   },
                   "recon_stream_name": "state",
                   "decoder_for_stream": {"state": "D0", "deriv": "D1"}},
    }
    torch.save(checkpoint, path)
    return path


def _old_style_lds_checkpoint(path: Path, latent_channels=4, ae_checkpoint="whatever",
                                test_dirs=None, window_length=2):
    """A stage-3 checkpoint as the pre-change train_lds saved it: f_theta at
    n_theta=1 and config recording n_theta=1.

    ae_checkpoint/test_dirs default to the original placeholder values used by
    every test that only inspects this checkpoint's OWN model_state/config
    (test_model_assembly_upgrades_an_old_1theta_f_theta, via
    load_lds_component) -- those never open ae_checkpoint or read test_dirs.
    A test that instead runs this through the real evaluation._latent_eval
    loader must pass both explicitly (a real AE checkpoint path, and at least
    one real run dir), since that loader DOES open and validate them."""
    f_theta = LatentDynamics(latent_channels=latent_channels, n_theta=1,  # OLD
                              latent_spatial=8, hidden_dim=8, n_hidden_layers=1)
    torch.save({
        "model_state": f_theta.state_dict(), "epoch": 2, "val_loss": 0.2,
        "ae_checkpoint": str(ae_checkpoint), "test_dirs": test_dirs or [],
        "config": {"latent_channels": latent_channels, "n_theta": 1,  # OLD
                    "latent_spatial_size": 8, "hidden_dim": 8, "n_hidden_layers": 1},
        "data_config": {"min_step": 0, "min_stdev_phi": None, "window_length": window_length,
                          "min_std_deriv": None},
    }, path)
    return path


def test_build_ae_from_checkpoint_upgrades_an_old_1theta_checkpoint(tmp_path):
    """The shared AE loader (used by _latent_eval -> check_parameter_dependence,
    check_z2_measurability, ...) must accept an old 1-theta stage-2 checkpoint
    and hand back a model whose encoder takes the CURRENT 2-wide theta."""
    from training.checkpoint_components import build_ae_from_checkpoint
    ck = _old_style_multistream_ae_checkpoint(tmp_path / "old_stage2.pt")
    ae, ae_encoder, checkpoint, stream_configs, recon = build_ae_from_checkpoint(ck, "cpu")
    x = torch.randn(3, 1, 32, 32)
    theta2 = torch.randn(3, N_THETA)  # the width the dataset NOW emits
    with torch.no_grad():
        z = ae_encoder(x, theta=theta2)
    assert "deriv" in z and z["deriv"].shape[0] == 3


def test_old_1theta_ae_upgrade_is_functionally_identical(tmp_path):
    """The upgraded model must equal the old model bit-for-bit when the new
    theta feature is passed as anything (its column is zero-padded)."""
    from training.checkpoint_components import build_ae_from_checkpoint
    ck_path = _old_style_multistream_ae_checkpoint(tmp_path / "old2.pt")
    old_ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    # rebuild the ORIGINAL old model to compare against
    from models.latent_streams import resolve_stream_configs_from_checkpoint_config
    stream_configs, recon = resolve_stream_configs_from_checkpoint_config(old_ck["config"])
    old_encoder = Encoder(input_size=32, in_channels=1, base_channels=4,
                           stream_configs=stream_configs, n_theta=1)
    old_state = {k[len("encoders.shared."):]: v for k, v in old_ck["model_state"].items()
                 if k.startswith("encoders.shared.")}
    old_encoder.load_state_dict(old_state)
    old_encoder.eval()

    ae, new_encoder, *_ = build_ae_from_checkpoint(ck_path, "cpu")
    x = torch.randn(2, 1, 32, 32)
    t1 = torch.randn(2, 1)
    t2 = torch.cat([t1, torch.randn(2, 1) * 99.0], dim=1)  # arbitrary 2nd feature
    with torch.no_grad():
        z_old = old_encoder(x, theta=t1)
        z_new = new_encoder(x, theta=t2)
    for name in z_old:
        assert torch.allclose(z_old[name], z_new[name], atol=1e-6), name


def test_model_assembly_upgrades_an_old_1theta_f_theta(tmp_path):
    """Stage 4/5 assembly must accept an old 1-theta lds component: f_theta
    rebuilt at N_THETA, old weights padded, forward accepting 2-wide theta."""
    from training.checkpoint_components import load_lds_component
    from training.model_assembly import build_models_from_components
    ae_ck = _old_style_multistream_ae_checkpoint(tmp_path / "old_ae.pt")
    lds_ck = _old_style_lds_checkpoint(tmp_path / "old_lds.pt")
    from training.checkpoint_components import load_ae_components
    components = load_ae_components(ae_ck, device="cpu")  # {"encoder": ..., "decoder": ...}
    components["lds"] = load_lds_component(lds_ck, device="cpu")
    # build_models_from_components returns EXACTLY six values; a TypeError here
    # is a real assembly regression, so it must FAIL rather than be swallowed as
    # a skip (the earlier try/except TypeError: pytest.skip could never go red).
    ae, stats_head, f_theta, frozen, cfgs, recon = build_models_from_components(
        components, device="cpu")
    z0 = torch.randn(2, 4, 8, 8); z1 = torch.randn(2, 4, 8, 8)
    dt = torch.rand(2) * 50.0
    theta2 = torch.randn(2, N_THETA)
    with torch.no_grad():
        out = f_theta(z0, z1, dt, theta2)
    assert out.shape == z0.shape


def _write_tiny_run(base_path: Path, name: str, size: int = 32, temperature: float = 0.8,
                     steps=(0, 1000, 2000)) -> Path:
    """A minimal real run directory -- metadata.txt plus real snapshot files
    in the actual on-disk format (see utils.load_datasets.snapshot_filename/
    read_phi_half) -- big enough for MicrostructureEvolutionDataset to build
    at least one window_length=2 window from it. No statistics.csv: with
    min_stdev_phi=None (this file's data_config), build_good_steps never
    opens it, and validate_run_dirs only requires it when min_stdev_phi is
    set (see load_datasets.validate_run_dirs's own required-files logic)."""
    from utils import load_datasets as load
    run_dir = base_path / name
    run_dir.mkdir()
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", f"steps = {steps[-1]}",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", f"temperature = {temperature}",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01", "seed = 1",
        "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    rng = np.random.default_rng(0)
    for step in steps:
        arr = rng.standard_normal((size, size)).astype("<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))
    return run_dir


def test_latent_eval_f_theta_loader_upgrades_an_old_lds_checkpoint(tmp_path):
    """_latent_eval builds f_theta for every stage-3 diagnostic
    (check_parameter_dependence, check_dt_vs_time, ...). An old lds
    checkpoint (config n_theta=1, 1-theta weights) must come back as a
    2-theta model with the old weights padded.

    This calls the REAL production loader (evaluation._latent_eval's
    _load_ae_f_theta_and_dataset), not an inline re-implementation -- an
    earlier version of this test built its own LatentDynamics +
    zero_pad_theta_columns by hand, which could only ever prove the upgrade
    pattern WORKS, never that this specific call site actually uses it. It
    didn't: the real function built f_theta at the checkpoint's own
    (n_theta=1) width with a plain load_state_dict, which would crash the
    first time it was handed a batch from MicrostructureEvolutionDataset
    (which always yields theta at the CURRENT N_THETA=2 width, regardless of
    what the checkpoint itself was trained with -- see datasets.py's own
    docstring). Fixed at the call site to mirror check_rollout.py/
    check_stats_head_rollout.py's own pattern: build at n_theta=N_THETA, then
    zero_pad_theta_columns before load_state_dict.
    """
    from evaluation._latent_eval import _load_ae_f_theta_and_dataset
    from models.encoder import zero_pad_theta_columns

    ae_ck = _old_style_multistream_ae_checkpoint(tmp_path / "old_ae2.pt")
    run_dir = _write_tiny_run(tmp_path, "T800_n001_s0")
    lds_ck = _old_style_lds_checkpoint(
        tmp_path / "old_lds2.pt", ae_checkpoint=ae_ck, test_dirs=[str(run_dir)],
        window_length=2,
    )
    prev = torch.load(lds_ck, map_location="cpu", weights_only=True)

    (device, euler_only, lds_checkpoint_path, ae_config, dataset,
     ae_decoder, f_theta) = _load_ae_f_theta_and_dataset(
        lds_checkpoint_path=lds_ck, min_step=None, min_stdev_phi=None,
        min_passing_steps=None, base_path=None, size=None,
        ae_stats_weight=None, hidden_dim=8, n_hidden_layers=1,
        condition_on_theta=None, euler_only=None, device="cpu",
    )

    assert len(dataset) > 0, "the tiny run must yield at least one window_length=2 window"

    # f_theta must actually accept the CURRENT theta width, which is what a
    # real DataLoader batch over `dataset` hands it -- this is the exact call
    # that crashed before the fix (shape mismatch between a 1-theta first
    # Linear and a 2-column theta tensor).
    theta2 = torch.randn(3, N_THETA)
    z0 = torch.randn(3, 4, 8, 8)
    z1 = torch.randn(3, 4, 8, 8)
    dt = torch.rand(3) * 50.0
    with torch.no_grad():
        out = f_theta(z0, z1, dt, theta2)
    assert out.shape == z0.shape

    # And the upgrade must be the SAME zero-pad upgrade used everywhere else
    # in the project, not just "some" 2-theta model that happens to run:
    # rebuild independently via zero_pad_theta_columns and compare outputs.
    expected_f_theta = LatentDynamics(
        latent_channels=prev["config"]["latent_channels"], n_theta=N_THETA,
        latent_spatial=prev["config"]["latent_spatial_size"],
        hidden_dim=prev["config"]["hidden_dim"],
        n_hidden_layers=prev["config"]["n_hidden_layers"],
    )
    expected_f_theta.load_state_dict(zero_pad_theta_columns(prev["model_state"], expected_f_theta))
    expected_f_theta.eval()
    with torch.no_grad():
        expected_out = expected_f_theta(z0, z1, dt, theta2)
    assert torch.allclose(out, expected_out, atol=1e-6), (
        "the real loader's f_theta does not match the standard zero-pad upgrade of the "
        "same old checkpoint -- it upgraded, but not the same way every other reconstruction "
        "site in the project does"
    )


def test_zero_pad_passes_non_theta_shape_mismatches_through_untouched():
    """A checkpoint whose weights mismatch the model in a NON-theta way (a
    conv with different channels, a 1-D bias of different length) is not this
    helper's business: it must pass those keys through UNTOUCHED so
    load_state_dict raises its own clear size-mismatch error -- an earlier
    version crashed with IndexError on shape[1] of a 1-D tensor instead,
    turning a clear error into an obscure one (caught by
    test_version_mismatch_shape_error_raises_clear_error)."""
    from models.encoder import zero_pad_theta_columns
    model = _ThetaFiLMConditionerLike()
    ckpt = {"net.0.weight": torch.randn(32, 2),     # matches -> untouched
            "net.0.bias": torch.randn(99),           # 1-D, WRONG length -> untouched
            "net.2.weight": torch.randn(7, 7, 3, 3)} # 4-D mismatch -> untouched
    out = zero_pad_theta_columns(ckpt, model)
    assert torch.equal(out["net.0.bias"], ckpt["net.0.bias"])
    assert torch.equal(out["net.2.weight"], ckpt["net.2.weight"])


class _ThetaFiLMConditionerLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(),
                                        torch.nn.Linear(32, 8))
