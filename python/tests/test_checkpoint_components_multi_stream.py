import torch
import tempfile
from pathlib import Path

from models.autoencoder import MultiStreamAutoencoder
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from models.latent_dynamics import LatentDynamics
from training.checkpoint_components import load_ae_components, ComponentCheckpoint
from training.model_assembly import build_models_from_components


def test_load_ae_components_handles_multi_stream_checkpoint(tmp_path):
    """
    Regression test: _strip_prefix used to look ONLY for the
    "encoder."/"decoder." prefix (Autoencoder's own flat structure).
    MultiStreamAutoencoder saves under "encoders.shared."/
    "decoders.shared." instead (see autoencoder.py's own docstring) --
    a checkpoint saved by train_autoencoder() with a real deriv stream
    would silently produce an EMPTY encoder/decoder component (no keys
    start with "encoder." when they actually start with
    "encoders.shared."), which then surfaced downstream as "every
    single key is missing" when trying to load it back into a fresh
    Encoder/Decoder -- a confusing failure far from its actual cause.

    No existing test caught this: every other checkpoint_components/
    model_assembly test only ever built single-stream (Autoencoder)
    checkpoints, which happened to still use the old flat structure and
    therefore never exercised this path.
    """
    torch.manual_seed(0)
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8, mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=8, mode=LatentStreamMode.DECODER),
    }
    encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    decoder = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    model = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                    stream_configs=stream_configs)

    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": 1, "val_loss": 0.5,
        "config": {
            "size": 32, "base_channels": 4, "latent_channels": 4, "latent_spatial_size": 8,
            "stream_configs": {n: {"channels": c.channels, "spatial_size": c.spatial_size, "mode": c.mode.value}
                                for n, c in stream_configs.items()},
            "recon_stream_name": "state",
        },
        "stats_head_state": None, "stats_config": None,
    }
    ckpt_path = tmp_path / "stage1.pt"
    torch.save(checkpoint, ckpt_path)

    components = load_ae_components(ckpt_path)
    assert len(components["encoder"].state_dict) > 0, (
        "encoder component is EMPTY -- _strip_prefix didn't recognize "
        "the encoders.shared. prefix"
    )
    assert len(components["decoder"].state_dict) > 0, (
        "decoder component is EMPTY -- _strip_prefix didn't recognize "
        "the decoders.shared. prefix"
    )
    # Every key should have had its prefix stripped down to the bare
    # Encoder/Decoder-relative name -- none should still start with
    # "encoders." or "decoders." (would mean stripping silently failed
    # and these are just the RAW, unstripped keys passing through).
    assert not any(k.startswith("encoders.") for k in components["encoder"].state_dict)
    assert not any(k.startswith("decoders.") for k in components["decoder"].state_dict)

    lds = LatentDynamics(latent_channels=4, n_theta=1, latent_spatial=8, hidden_dim=8, n_hidden_layers=1)
    components["lds"] = ComponentCheckpoint(
        state_dict=lds.state_dict(),
        config={"latent_channels": 4, "n_theta": 1, "latent_spatial_size": 8,
                "hidden_dim": 8, "n_hidden_layers": 1},
        provenance={},
    )

    # THE actual end-to-end check: this must not raise -- neither the
    # old "everything missing" RuntimeError, nor any other failure mode.
    ae, stats_head, f_theta, frozen_modules, _, _ = build_models_from_components(components, device="cpu")
    assert ae is not None
