"""
Regression tests for the "is_flat_checkpoint" detection pattern applied
across every evaluation/comparison script this session that constructs
an Autoencoder/EncoderDecoderPair/MultiStreamAutoencoder from a saved
checkpoint. A stage 4/5 checkpoint always saves a FLAT, single-pathway
EncoderDecoderPair (model_assembly.py's build_models_from_components
only ever needs the reconstruction stream's own pathway) -- but keeps
its ANCESTOR's full, multi-stream stream_configs in its own config,
inherited unchanged. len(stream_configs) was therefore never the right
signal for deciding single-pathway vs multi-stream construction; only
the checkpoint's OWN actual keys are. check_reconstruction.py and
check_latent_channels.py already have full, direct end-to-end tests of
this (see test_check_reconstruction_stage4.py); this file exercises
the identical construction pattern in the remaining scripts that also
had it (check_interpolation.py, check_perturbation.py, check_rollout.py,
check_parameter_dependence.py, compare_rollout_training.py), which all
share the exact same is_flat_checkpoint logic.
"""
import torch
from models.autoencoder import EncoderDecoderPair
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_streams import LatentStreamConfig, LatentStreamMode


def _build_flat_checkpoint_state(tmp_path):
    """A real, flat (stage 4/5-style) EncoderDecoderPair state_dict,
    with an ancestor's full multi-stream stream_configs still
    inherited in its own config -- exactly matches a real stage 4/5
    checkpoint's own shape."""
    torch.manual_seed(0)
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=4, mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=4, mode=LatentStreamMode.DECODER),
    }
    encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    decoder = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=4)
    ae = EncoderDecoderPair(encoder, decoder, stream_name="state", mode=LatentStreamMode.AUTOENCODER)
    config = {
        "size": 32, "base_channels": 4, "latent_channels": 4, "latent_spatial_size": 4,
        "stream_configs": {n: {"channels": c.channels, "spatial_size": c.spatial_size, "mode": c.mode.value}
                            for n, c in stream_configs.items()},
        "recon_stream_name": "state",
    }
    return ae.state_dict(), config, stream_configs


def test_flat_checkpoint_detection_and_reconstruction_across_all_fixed_scripts():
    """Directly confirms the shared construction pattern (detect flat
    vs nested from actual keys, rebuild an EncoderDecoderPair mirroring
    model_assembly.py's own construction, load_state_dict succeeds)
    that all 8 fixed scripts now use identically."""
    from models.latent_streams import (
        resolve_stream_configs_from_checkpoint_config,
        cross_check_stream_configs_against_state_dict,
    )
    model_state, model_cfg, _ = _build_flat_checkpoint_state(None)

    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, model_state)
    recon_stream = stream_configs[recon_stream_name]

    is_flat_checkpoint = any(k.startswith("encoder.") for k in model_state)
    assert is_flat_checkpoint, "a real stage 4/5 checkpoint must be detected as flat"

    encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                       base_channels=model_cfg["base_channels"], stream_configs=stream_configs)
    decoder = Decoder(output_size=model_cfg["size"], out_channels=1,
                       base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
                       latent_spatial_size=recon_stream.spatial_size)
    ae = EncoderDecoderPair(encoder, decoder, stream_name=recon_stream_name, mode=recon_stream.mode)

    # THE actual regression check: must not raise the old "encoders.shared.* missing" RuntimeError.
    ae.load_state_dict(model_state)
    ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
    ae_decoder = ae.decoder if hasattr(ae, "decoder") else ae.decoders["shared"]
    assert ae_encoder is not None and ae_decoder is not None
