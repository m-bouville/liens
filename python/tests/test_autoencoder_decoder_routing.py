import torch
import pytest

from models.autoencoder import MultiStreamAutoencoder
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_streams import LatentStreamConfig, LatentStreamMode


def _build_stream_configs():
    return {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8, mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=8, mode=LatentStreamMode.DECODER),
    }


def test_default_decoder_for_stream_preserves_existing_single_decoder_behavior():
    """decoder_for_stream=None (the default) must behave EXACTLY as
    before this parameter existed -- every existing caller
    (train_stage1.py/train_stage2.py, model_assembly.py, every
    evaluation script) constructs with a single decoder and no
    decoder_for_stream at all."""
    torch.manual_seed(0)
    stream_configs = _build_stream_configs()
    encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    decoder = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    model = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                    stream_configs=stream_configs)
    assert model.pathways["state"].decoder is decoder
    assert model.pathways["deriv"].decoder is decoder


def test_explicit_decoder_for_stream_routes_correctly():
    torch.manual_seed(0)
    stream_configs = _build_stream_configs()
    encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    D0 = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    D1 = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    model = MultiStreamAutoencoder(
        encoders={"shared": encoder}, decoders={"D0": D0, "D1": D1},
        stream_configs=stream_configs, decoder_for_stream={"state": "D0", "deriv": "D1"},
    )
    assert model.pathways["state"].decoder is D0
    assert model.pathways["deriv"].decoder is D1


def test_parameters_deduplicated_with_separate_decoders():
    """Shared trunk + two genuinely separate decoders -- parameters()
    must count the trunk once, each decoder once, no double-counting
    despite the trunk being reachable through both pathways."""
    torch.manual_seed(0)
    stream_configs = _build_stream_configs()
    encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    D0 = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    D1 = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    model = MultiStreamAutoencoder(
        encoders={"shared": encoder}, decoders={"D0": D0, "D1": D1},
        stream_configs=stream_configs, decoder_for_stream={"state": "D0", "deriv": "D1"},
    )
    n_expected = (sum(p.numel() for p in encoder.parameters())
                  + sum(p.numel() for p in D0.parameters())
                  + sum(p.numel() for p in D1.parameters())
                  + 1)  # deriv stream's own learnable log_output_scale
    n_actual = sum(p.numel() for p in model.parameters())
    assert n_actual == n_expected


def test_d0_isolated_from_d1_gradient_but_trunk_still_shared():
    """THE actual point of separate decoders: training only through the
    deriv/D1 pathway must leave D0 completely untouched, while the
    shared trunk still trains (it remains genuinely shared by design,
    regardless of how many decoders exist)."""
    torch.manual_seed(0)
    stream_configs = _build_stream_configs()
    encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    D0 = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    D1 = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    model = MultiStreamAutoencoder(
        encoders={"shared": encoder}, decoders={"D0": D0, "D1": D1},
        stream_configs=stream_configs, decoder_for_stream={"state": "D0", "deriv": "D1"},
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    D0_before = list(D0.up_blocks[0].parameters())[0].clone()
    D1_before = list(D1.up_blocks[0].parameters())[0].clone()
    trunk_before = encoder.down_blocks[0].conv.block[0].weight.clone()

    x = torch.randn(4, 1, 32, 32)
    x_recon_deriv, _ = model.pathways["deriv"](x)
    loss = x_recon_deriv.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    D0_after = list(D0.up_blocks[0].parameters())[0]
    D1_after = list(D1.up_blocks[0].parameters())[0]
    trunk_after = encoder.down_blocks[0].conv.block[0].weight

    assert torch.equal(D0_before, D0_after), "D0 moved despite training only through D1 -- gradient isolation broken"
    assert not torch.equal(D1_before, D1_after), "D1 did not train"
    assert not torch.equal(trunk_before, trunk_after), "shared trunk did not train (it should still be shared)"


def test_ambiguous_decoder_routing_raises():
    torch.manual_seed(0)
    stream_configs = _build_stream_configs()
    encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    D0 = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    D1 = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    with pytest.raises(ValueError, match="ambiguous"):
        MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"D0": D0, "D1": D1},
                                stream_configs=stream_configs)


def test_multiple_encoders_rejected():
    torch.manual_seed(0)
    stream_configs = _build_stream_configs()
    encoder1 = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    encoder2 = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    D0 = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    with pytest.raises(ValueError, match="exactly one shared encoder"):
        MultiStreamAutoencoder(encoders={"e1": encoder1, "e2": encoder2}, decoders={"D0": D0},
                                stream_configs=stream_configs)
