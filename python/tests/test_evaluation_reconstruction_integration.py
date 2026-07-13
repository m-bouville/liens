"""
Integration tests for the evaluation/comparison scripts' model-
reconstruction logic -- the part that broke, repeatedly, across a long
session of real pipeline runs. Every one of these bugs was invisible
to unit tests of the underlying pure functions (resolve_stream_configs_
from_checkpoint_config, cross_check_stream_configs_against_state_dict,
etc. -- see test_latent_streams.py) because the bug was in HOW each
script WIRED those functions together, not in the functions
themselves. The only thing that actually catches a wiring bug is
calling the real function end-to-end against a real checkpoint --
which is what these tests do.

Two realistic checkpoint shapes get exercised throughout, matching the
two actual failure modes hit in production:

1. NON-DEFAULT latent_spatial_size (4, not the default 8) -- catches
   any reconstruction site that forgot to pass latent_spatial/
   latent_spatial_size at all (silently defaulting to 8 instead of
   reading the checkpoint's real value; the exact bug that broke
   LatentDynamics reconstruction in four separate files this session).

2. STALE multi-stream metadata (config claims one stream, encoder
   weights genuinely contain two) -- catches any reconstruction site
   that resolves stream_configs without also cross-checking against
   the real weights (the exact bug that broke model_assembly.py and
   several evaluation scripts this session).

A real Autoencoder/LatentDynamics is used throughout (not mocked) --
these tests call torch.save/torch.load and the real construction code,
so a wiring mistake shows up as an actual RuntimeError/NameError/
AttributeError, not something a mock could paper over.
"""
from pathlib import Path

import torch
import pytest

from models.autoencoder import Autoencoder, EncoderDecoderPair
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_dynamics import LatentDynamics
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from training.stats_head import StatsHead

from evaluation.check_latent_channels import check_latent_channels
from evaluation.check_reconstruction import check_reconstruction
from evaluation.check_interpolation import check_interpolation
from evaluation.check_perturbation import check_perturbation
from evaluation.check_rollout import check_rollout
from evaluation.check_parameter_dependence import check_parameter_dependence


_NON_DEFAULT_SPATIAL = 4  # anything != models.constants.LATENT_SPATIAL_SIZE (8)


def _save_ae_checkpoint(path, run_dirs, size=64, base_channels=4, latent_channels=4,
                         latent_spatial_size=_NON_DEFAULT_SPATIAL, multi_stream=False,
                         stale_metadata=False, include_stats_head=True, stat_names=None):
    """
    Builds and saves a REAL Autoencoder (or EncoderDecoderPair, for
    multi_stream) checkpoint -- real weights, real state_dict, exactly
    what train_ae.py itself produces. multi_stream + stale_metadata
    together reproduces the exact "self_heals" scenario: config claims
    only one stream, but the encoder's actual weights have two.
    """
    if multi_stream:
        stream_configs = {
            "state": LatentStreamConfig(name="state", channels=latent_channels,
                                         spatial_size=latent_spatial_size,
                                         mode=LatentStreamMode.AUTOENCODER),
            "deriv": LatentStreamConfig(name="deriv", channels=latent_channels,
                                         spatial_size=latent_spatial_size,
                                         mode=LatentStreamMode.DECODER),
        }
        encoder = Encoder(input_size=size, in_channels=1, base_channels=base_channels,
                           stream_configs=stream_configs)
        decoder = Decoder(output_size=size, out_channels=1, base_channels=base_channels,
                           latent_channels=latent_channels, latent_spatial_size=latent_spatial_size)
        ae = EncoderDecoderPair(encoder, decoder)
    else:
        ae = Autoencoder(size=size, channels=1, base_channels=base_channels,
                          latent_channels=latent_channels, latent_spatial_size=latent_spatial_size)

    config = {"size": size, "base_channels": base_channels, "latent_channels": latent_channels,
              "latent_spatial_size": latent_spatial_size, "stats_weight": 0.01}
    if not stale_metadata:
        streams_for_config = (
            {"state": {"channels": latent_channels, "spatial_size": latent_spatial_size,
                       "mode": "autoencoder"},
             "deriv": {"channels": latent_channels, "spatial_size": latent_spatial_size,
                       "mode": "decoder"}}
            if multi_stream else
            {"state": {"channels": latent_channels, "spatial_size": latent_spatial_size,
                       "mode": "autoencoder"}}
        )
        config["stream_configs"] = streams_for_config
        config["recon_stream_name"] = "state"
    # stale_metadata=True: deliberately omit stream_configs/recon_stream_name
    # entirely, even though multi_stream=True means the ACTUAL weights
    # (saved below) genuinely have both streams -- exactly the
    # intermediate-version-of-the-code checkpoint shape this session
    # hit for real.

    checkpoint = {
        "model_state": ae.state_dict(),
        "epoch": 5,
        "val_loss": 0.05,
        "val_loss_ema": 0.05,
        "test_dirs": [str(d) for d in run_dirs],
        "config": config,
    }
    if include_stats_head:
        names = stat_names or ["angle", "avg_phi"]
        stats_head = StatsHead(latent_channels=latent_channels, stat_names=names,
                                latent_spatial=latent_spatial_size)
        checkpoint["stats_head_state"] = stats_head.state_dict()
        checkpoint["stats_config"] = {
            "stat_names": names, "stats_mean": torch.zeros(len(names)),
            "stats_std": torch.ones(len(names)),
        }
    torch.save(checkpoint, path)
    return checkpoint


def _save_lds_checkpoint(path, ae_checkpoint_path, latent_channels=4,
                          latent_spatial_size=_NON_DEFAULT_SPATIAL, run_dirs=None):
    """Real LatentDynamics checkpoint, pointing at a real AE ancestor."""
    f_theta = LatentDynamics(latent_channels=latent_channels, n_theta=1,
                              latent_spatial=latent_spatial_size, hidden_dim=16, n_hidden_layers=1)
    checkpoint = {
        "model_state": f_theta.state_dict(),
        "epoch": 5,
        "val_loss": 0.05,
        "test_dirs": [str(d) for d in (run_dirs or [])],
        "ae_checkpoint": str(Path(ae_checkpoint_path).resolve()),
        "config": {"latent_channels": latent_channels, "n_theta": 1,
                   "latent_spatial_size": latent_spatial_size, "hidden_dim": 16, "n_hidden_layers": 1},
    }
    torch.save(checkpoint, path)
    return checkpoint


# ---- check_latent_channels -------------------------------------------------
# THE function that actually had a real NameError (rank_channel_importance
# referencing recon_stream_name without receiving it) -- this is the most
# direct regression test for that specific bug: if the scoping bug came
# back, this would fail with NameError, not a normal assertion failure.

def test_check_latent_channels_non_default_spatial_size(tmp_path, tmp_run_dir):
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL)

    output_path = check_latent_channels(
        ae_checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert output_path.exists()


def test_check_latent_channels_stale_multi_stream_metadata(tmp_path, tmp_run_dir):
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2-stale.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         multi_stream=True, stale_metadata=True)

    # Must not raise -- neither the old "unexpected keys" RuntimeError
    # (stale metadata undercounting streams) nor a NameError (the
    # scoping bug) -- both real failure modes this session hit.
    output_path = check_latent_channels(
        ae_checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert output_path.exists()


# ---- check_reconstruction ---------------------------------------------------

def test_check_reconstruction_non_default_spatial_size(tmp_path, tmp_run_dir):
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL)

    output_path = check_reconstruction(
        checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert output_path.exists()


def test_check_reconstruction_stale_multi_stream_metadata(tmp_path, tmp_run_dir):
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2-stale.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         multi_stream=True, stale_metadata=True)

    output_path = check_reconstruction(
        checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert output_path.exists()


# ---- check_interpolation / check_perturbation -------------------------------
# Both need real statistics.csv data (stat_names must match real columns)
# -- tmp_run_dir_with_stats provides that.

def test_check_interpolation_non_default_spatial_size(tmp_path, tmp_run_dir_with_stats):
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    ae_path = tmp_path / "fake-stage2.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         stat_names=stat_names)

    output_path = check_interpolation(
        checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert output_path.exists()


def test_check_perturbation_non_default_spatial_size(tmp_path, tmp_run_dir_with_stats):
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    ae_path = tmp_path / "fake-stage2.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         stat_names=stat_names)

    output_path = check_perturbation(
        checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
        n_samples=2, n_repeats=2,
    )
    assert output_path.exists()


# ---- check_rollout / check_parameter_dependence -----------------------------
# Both need a real (AE, LDS) pair, LDS pointing back at the AE via
# "ae_checkpoint" -- the exact shape train_lds.py itself produces.

def test_check_rollout_non_default_spatial_size(tmp_path, tmp_run_dir):
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2.pt"
    lds_path = tmp_path / "fake-stage3b.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         include_stats_head=False)
    _save_lds_checkpoint(lds_path, ae_path, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                          run_dirs=[run_dir])

    output_path, _windows = check_rollout(
        lds_checkpoint_path=lds_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
        n_samples=1,
    )
    assert output_path.exists()


def test_check_parameter_dependence_non_default_spatial_size(tmp_path, tmp_run_dir_with_stats):
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    # check_parameter_dependence specifically needs "autocorr_length"
    # for its length-scale panel -- the shared fixture's stat set
    # doesn't include it (it's a smaller, generic set other tests
    # rely on staying fixed), so it's added here, local to this test,
    # rather than changing the shared fixture for everyone.
    import pandas as pd
    stats_path = run_dir / "statistics.csv"
    df = pd.read_csv(stats_path)
    df["autocorr_length"] = 5.0
    df.to_csv(stats_path, index=False)

    ae_path = tmp_path / "fake-stage2.pt"
    lds_path = tmp_path / "fake-stage3b.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         include_stats_head=False)
    _save_lds_checkpoint(lds_path, ae_path, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                          run_dirs=[run_dir])

    output_path = check_parameter_dependence(
        lds_checkpoint_path=lds_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert output_path.exists()


def test_check_rollout_stale_multi_stream_metadata(tmp_path, tmp_run_dir):
    """THE combination that broke model_assembly.py originally, exercised
    here for check_rollout.py's own (separate) reconstruction path."""
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2-stale.pt"
    lds_path = tmp_path / "fake-stage3b.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         multi_stream=True, stale_metadata=True, include_stats_head=False)
    _save_lds_checkpoint(lds_path, ae_path, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                          run_dirs=[run_dir])

    output_path, _windows = check_rollout(
        lds_checkpoint_path=lds_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
        n_samples=1,
    )
    assert output_path.exists()
