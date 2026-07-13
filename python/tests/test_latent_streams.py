"""
Tests for models/latent_streams.py -- the type system and enforcement
functions for the multi-stream (C0/C1) latent redesign. Genuinely new
logic (introduced during that redesign) that had zero test coverage
before -- unlike the Encoder/Decoder/Autoencoder architecture itself,
which the golden-master regression test (test_architecture_stability.py)
covers, decode_stream/autoencode_stream/build_stream_configs have their
own real branching logic that nothing else exercises.

decode_stream/autoencode_stream are tested with plain callables/tensors
standing in for a real Decoder/Autoencoder -- both functions only ever
CALL whatever they're given after their own checks pass, so a fake
decoder (identity function) and a bare tensor of the right shape
exercise the actual logic under test without needing a real, trained
architecture.
"""
import torch
import pytest

from models.latent_streams import (
    LatentStreamConfig, LatentStreamMode, autoencode_stream,
    build_stream_configs, decode_stream,
)


def _stream(mode, channels=4, spatial=8, name="test"):
    return LatentStreamConfig(name=name, channels=channels, spatial_size=spatial, mode=mode)


# ---- decode_stream ----------------------------------------------------

def test_decode_stream_autoencoder_mode_calls_decoder():
    stream = _stream(LatentStreamMode.AUTOENCODER)
    z = torch.zeros(1, 4, 8, 8)
    calls = []
    decoder = lambda z_in: calls.append(z_in) or z_in * 2  # noqa: E731
    result = decode_stream(decoder, z, stream)
    assert len(calls) == 1 and calls[0] is z
    assert torch.equal(result, z * 2)


def test_decode_stream_decoder_mode_also_allowed():
    """DECODER-mode streams CAN be decoded (that's the whole point --
    the result just means something other than a reconstruction, see
    LatentStreamMode's own docstring) -- only PURE_LATENT is refused."""
    stream = _stream(LatentStreamMode.DECODER)
    z = torch.zeros(1, 4, 8, 8)
    decoder = lambda z_in: z_in  # noqa: E731
    result = decode_stream(decoder, z, stream)
    assert torch.equal(result, z)


def test_decode_stream_refuses_pure_latent():
    stream = _stream(LatentStreamMode.PURE_LATENT)
    z = torch.zeros(1, 4, 8, 8)
    with pytest.raises(TypeError, match="pure_latent"):
        decode_stream(lambda z_in: z_in, z, stream)  # noqa: E731


def test_decode_stream_rejects_wrong_channel_count():
    stream = _stream(LatentStreamMode.AUTOENCODER, channels=4, spatial=8)
    z = torch.zeros(1, 6, 8, 8)  # 6 channels, stream declares 4
    with pytest.raises(ValueError, match="doesn't match"):
        decode_stream(lambda z_in: z_in, z, stream)  # noqa: E731


def test_decode_stream_rejects_wrong_spatial_size():
    stream = _stream(LatentStreamMode.AUTOENCODER, channels=4, spatial=8)
    z = torch.zeros(1, 4, 16, 16)  # 16x16, stream declares 8x8
    with pytest.raises(ValueError, match="doesn't match"):
        decode_stream(lambda z_in: z_in, z, stream)  # noqa: E731


def test_decode_stream_never_calls_decoder_when_shape_is_wrong():
    """The shape check must happen BEFORE calling decoder, not after --
    catches a caller passing the wrong stream's z before that mistake
    can propagate into (and be masked by) whatever decoder itself does
    with a wrong-shaped input."""
    stream = _stream(LatentStreamMode.AUTOENCODER, channels=4, spatial=8)
    z = torch.zeros(1, 6, 8, 8)
    calls = []
    decoder = lambda z_in: calls.append(z_in)  # noqa: E731
    with pytest.raises(ValueError):
        decode_stream(decoder, z, stream)
    assert calls == []


# ---- autoencode_stream --------------------------------------------------

def test_autoencode_stream_autoencoder_mode_calls_ae():
    stream = _stream(LatentStreamMode.AUTOENCODER)
    x = torch.zeros(1, 1, 8, 8)
    calls = []
    ae = lambda x_in: calls.append(x_in) or (x_in, x_in)  # noqa: E731
    result = autoencode_stream(ae, x, stream)
    assert len(calls) == 1 and calls[0] is x
    assert result == (x, x)


def test_autoencode_stream_refuses_decoder_mode():
    """THE specific danger this function exists to prevent (see its
    own docstring and the project's design discussion): a DECODER-mode
    stream's decode is compared against a DIFFERENT target than its
    own input (e.g. a time derivative, never the input itself) --
    routing it through the reconstruction wrapper would silently
    compute a meaningless loss against the wrong target."""
    stream = _stream(LatentStreamMode.DECODER)
    x = torch.zeros(1, 1, 8, 8)
    with pytest.raises(TypeError, match="not\\s+autoencoder"):
        autoencode_stream(lambda x_in: (x_in, x_in), x, stream)  # noqa: E731


def test_autoencode_stream_refuses_pure_latent():
    stream = _stream(LatentStreamMode.PURE_LATENT)
    x = torch.zeros(1, 1, 8, 8)
    with pytest.raises(TypeError, match="not\\s+autoencoder"):
        autoencode_stream(lambda x_in: (x_in, x_in), x, stream)  # noqa: E731


def test_autoencode_stream_never_calls_ae_when_mode_is_wrong():
    stream = _stream(LatentStreamMode.DECODER)
    x = torch.zeros(1, 1, 8, 8)
    calls = []
    ae = lambda x_in: calls.append(x_in)  # noqa: E731
    with pytest.raises(TypeError):
        autoencode_stream(ae, x, stream)
    assert calls == []


# ---- build_stream_configs -----------------------------------------------

def test_build_stream_configs_replicates_pre_redesign_single_stream():
    """The specific case that must keep working exactly: a single
    'state' stream, mode=autoencoder -- what a stage-parameters file
    with no explicit latent_names/latent_modes at all should resolve
    to, matching every checkpoint saved before this redesign."""
    configs = build_stream_configs(
        names=["state"], modes=["autoencoder"],
        channels_decoder=8, spatial_decoder=8,
    )
    assert list(configs.keys()) == ["state"]
    assert configs["state"].channels == 8
    assert configs["state"].spatial_size == 8
    assert configs["state"].mode == LatentStreamMode.AUTOENCODER


def test_build_stream_configs_two_stream_c0_c1():
    configs = build_stream_configs(
        names=["state", "deriv"], modes=["autoencoder", "decoder"],
        channels_decoder=8, spatial_decoder=8,
    )
    assert configs["state"].mode == LatentStreamMode.AUTOENCODER
    assert configs["deriv"].mode == LatentStreamMode.DECODER
    # Equal-channel-count-for-decodability is structural here, not a
    # separate check: both streams share channels_decoder/spatial_decoder,
    # so there's no way to even EXPRESS a mismatch via this function.
    assert configs["state"].channels == configs["deriv"].channels
    assert configs["state"].spatial_size == configs["deriv"].spatial_size


def test_build_stream_configs_rejects_mismatched_list_lengths():
    with pytest.raises(ValueError, match="same length"):
        build_stream_configs(
            names=["a", "b"], modes=["autoencoder"],
            channels_decoder=8, spatial_decoder=8,
        )


def test_build_stream_configs_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        build_stream_configs(
            names=["a", "a"], modes=["autoencoder", "decoder"],
            channels_decoder=8, spatial_decoder=8,
        )


def test_build_stream_configs_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown latent stream mode"):
        build_stream_configs(
            names=["a"], modes=["not_a_real_mode"],
            channels_decoder=8, spatial_decoder=8,
        )


def test_build_stream_configs_pure_latent_not_implemented():
    """pure_latent streams need their own per-stream channels/
    spatial-size syntax, which doesn't exist yet -- must raise clearly
    rather than silently building a stream with the wrong (shared
    decoder) size."""
    with pytest.raises(NotImplementedError, match="pure_latent"):
        build_stream_configs(
            names=["a"], modes=["pure_latent"],
            channels_decoder=8, spatial_decoder=8,
        )


def test_build_stream_configs_warns_on_multiple_autoencoder_streams():
    """At most one AUTOENCODER-mode stream is meaningful (Autoencoder's
    reconstruction wrapper needs a single unambiguous stream) -- a
    config mistake worth warning about, not silently accepting, but
    not necessarily fatal either (raw Encoder/Decoder use doesn't care)."""
    with pytest.warns(UserWarning, match="multiple streams"):
        build_stream_configs(
            names=["a", "b"], modes=["autoencoder", "autoencoder"],
            channels_decoder=8, spatial_decoder=8,
        )
