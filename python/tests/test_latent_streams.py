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
import warnings

from models.latent_streams import (
    LatentStreamConfig, LatentStreamMode, autoencode_stream,
    build_stream_configs, cross_check_stream_configs_against_state_dict, decode_stream,
    resolve_stream_configs_from_checkpoint_config,
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_stream_configs(
            names=["a", "b"], modes=["autoencoder", "autoencoder"],
            channels_decoder=8, spatial_decoder=8,
        )
    assert any(issubclass(x.category, UserWarning) and "multiple streams" in str(x.message)
               for x in w), f"expected a 'multiple streams' UserWarning, got: {[str(x.message) for x in w]}"


# ---- resolve_stream_configs_from_checkpoint_config -----------------------
# Never had dedicated tests before -- exercised only indirectly through
# every production file that calls it. Given how many rounds of bugs
# this whole area caused, worth pinning down directly.

def test_resolve_stream_configs_falls_back_for_pre_redesign_checkpoint():
    """A checkpoint saved before the multi-stream redesign has neither
    'stream_configs' nor 'recon_stream_name' -- must fall back to
    exactly the single implicit 'autoencoder'-mode stream it always
    had, built from the flat latent_channels/latent_spatial_size keys
    that DO exist on every checkpoint this project has ever saved."""
    model_cfg = {"size": 64, "base_channels": 32, "latent_channels": 16,
                 "latent_spatial_size": 8, "stats_weight": 0.01}
    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    assert recon_stream_name == "state"
    assert list(stream_configs.keys()) == ["state"]
    assert stream_configs["state"].channels == 16
    assert stream_configs["state"].spatial_size == 8
    assert stream_configs["state"].mode == LatentStreamMode.AUTOENCODER


def test_resolve_stream_configs_missing_latent_spatial_size_uses_default():
    """latent_spatial_size itself can ALSO be missing (an even older
    checkpoint, or one saved before that specific field existed) --
    falls back to the shared LATENT_SPATIAL_SIZE default, not a crash."""
    model_cfg = {"size": 64, "base_channels": 32, "latent_channels": 16}
    stream_configs, _ = resolve_stream_configs_from_checkpoint_config(model_cfg)
    assert stream_configs["state"].spatial_size == 8  # models.constants.LATENT_SPATIAL_SIZE


def test_resolve_stream_configs_reads_real_multi_stream_metadata():
    """A checkpoint saved by the current code, with real stream_configs
    -- read back exactly, not re-derived or approximated."""
    model_cfg = {
        "size": 64, "base_channels": 32,
        "stream_configs": {
            "state": {"channels": 8, "spatial_size": 8, "mode": "autoencoder"},
            "deriv": {"channels": 8, "spatial_size": 8, "mode": "decoder"},
        },
        "recon_stream_name": "state",
    }
    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    assert recon_stream_name == "state"
    assert set(stream_configs.keys()) == {"state", "deriv"}
    assert stream_configs["deriv"].mode == LatentStreamMode.DECODER


# ---- cross_check_stream_configs_against_state_dict ------------------------
# THE bug from this session's own troubleshooting: a checkpoint whose
# encoder weights genuinely contain a second stream (training succeeded
# with the CONSTRUCTION fix in place), but whose saved "config" doesn't
# say so (saved by a version of the code before the separate
# checkpoint-SAVE fix landed) -- no amount of correctly reading stale
# metadata can fix this; only reading the real weights can.

class _FakeTensor:
    """Minimal stand-in for a torch.Tensor -- only .shape is needed by
    the function under test, so this avoids requiring torch here."""
    def __init__(self, shape):
        self.shape = shape


def test_cross_check_is_a_noop_when_metadata_already_matches():
    """The common, correct case: state_dict's own stream names already
    match what stream_configs claims -- returned unchanged, not
    rebuilt, and no warning (nothing to warn about)."""
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
    }
    state_dict = {
        "encoder.bottlenecks.state.weight": _FakeTensor((4, 64, 1, 1)),
        "encoder.bottlenecks.state.bias": _FakeTensor((4,)),
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        corrected, recon_name = cross_check_stream_configs_against_state_dict(
            stream_configs, "state", state_dict,
        )
        assert len(w) == 0
    assert corrected is stream_configs
    assert recon_name == "state"


def test_cross_check_corrects_stale_metadata_missing_a_real_stream():
    """THE actual bug: metadata claims only 'state', but the encoder's
    OWN weights also contain 'deriv' -- must be detected and added,
    with a warning (this is a real inconsistency worth surfacing, even
    though it's handled gracefully rather than raised as an error)."""
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
    }
    state_dict = {
        "encoder.down_blocks.0.conv1.weight": _FakeTensor((4, 1, 3, 3)),
        "encoder.bottlenecks.state.weight": _FakeTensor((4, 64, 1, 1)),
        "encoder.bottlenecks.state.bias": _FakeTensor((4,)),
        "encoder.bottlenecks.deriv.weight": _FakeTensor((4, 64, 1, 1)),
        "encoder.bottlenecks.deriv.bias": _FakeTensor((4,)),
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        corrected, recon_name = cross_check_stream_configs_against_state_dict(
            stream_configs, "state", state_dict,
        )
    assert any(issubclass(x.category, UserWarning) and "stale" in str(x.message) for x in w), \
        f"expected a 'stale' UserWarning, got: {[str(x.message) for x in w]}"
    assert set(corrected.keys()) == {"state", "deriv"}
    assert recon_name == "state"  # unchanged -- the recon stream was never in question
    # Channel count read from the ACTUAL weight tensor, not assumed:
    assert corrected["deriv"].channels == 4
    # Spatial size can't be read from a 1x1 conv's weight shape --
    # assumed to match the recon stream's own, per this project's
    # decodable-streams-share-one-size design constraint.
    assert corrected["deriv"].spatial_size == 8
    # Mode assumed DECODER (not AUTOENCODER) for a stream the
    # metadata never claimed as the reconstruction target -- the
    # safer default, since at most one AUTOENCODER-mode stream is
    # meaningful and recon_stream_name already identifies it.
    assert corrected["deriv"].mode == LatentStreamMode.DECODER
    # The original stream's own config is untouched:
    assert corrected["state"].channels == 4
    assert corrected["state"].mode == LatentStreamMode.AUTOENCODER


def test_cross_check_infers_correct_channel_count_from_weight_shape():
    """The inferred stream's channel count must come from its ACTUAL
    weight tensor shape, not copy the recon stream's -- different
    streams could plausibly have different channel counts (nothing
    about this function assumes they must match)."""
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
    }
    state_dict = {
        "encoder.bottlenecks.state.weight": _FakeTensor((4, 64, 1, 1)),
        "encoder.bottlenecks.deriv.weight": _FakeTensor((6, 64, 1, 1)),  # DIFFERENT channel count
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        corrected, _ = cross_check_stream_configs_against_state_dict(
            stream_configs, "state", state_dict,
        )
    assert any(issubclass(x.category, UserWarning) for x in w)
    assert corrected["deriv"].channels == 6  # read from ITS OWN weight, not copied from state's 4


def test_cross_check_does_not_remove_a_stream_metadata_claims_but_state_dict_lacks():
    """A DIFFERENT, stranger kind of inconsistency (metadata claims a
    stream the weights don't have) is deliberately NOT auto-corrected
    -- this function only ADDS what the weights prove exists, never
    REMOVES what the metadata claims; that's a different failure this
    function doesn't try to silently paper over."""
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.DECODER),
    }
    state_dict = {
        "encoder.bottlenecks.state.weight": _FakeTensor((4, 64, 1, 1)),
        # no "deriv" keys at all in the state_dict
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        corrected, recon_name = cross_check_stream_configs_against_state_dict(
            stream_configs, "state", state_dict,
        )
        assert len(w) == 0  # not this function's job to flag this case
    assert set(corrected.keys()) == {"state", "deriv"}  # unchanged, still claims both


def test_cross_check_no_bottleneck_keys_at_all_is_a_noop():
    """A state_dict with no 'bottlenecks.' keys at all (e.g. a decoder-
    only state_dict passed by mistake, or some other unrelated shape)
    -- returns the original stream_configs unchanged rather than
    misinterpreting the absence of any match as anything meaningful."""
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
    }
    state_dict = {"decoder.output_conv.weight": _FakeTensor((1, 4, 3, 3))}
    corrected, recon_name = cross_check_stream_configs_against_state_dict(
        stream_configs, "state", state_dict,
    )
    assert corrected is stream_configs
    assert recon_name == "state"


def test_cross_check_reconciles_head_kind_from_residual_head_weights():
    """A stage-4/5 joint checkpoint saved before head_kind was serialized has a
    deriv stream whose config says head_kind='linear' but whose weights contain
    a residual head. cross_check must read head_kind/head_hidden back from the
    weights so the rebuilt Encoder creates the matching residual_heads.<name>."""
    import torch
    cfgs = {
        "state": LatentStreamConfig(name="state", channels=8, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=8, spatial_size=8,
                                     mode=LatentStreamMode.DECODER),  # head_kind defaults linear
    }
    state = {
        "encoders.shared.bottlenecks.state.weight": torch.zeros(8, 16, 1, 1),
        "encoders.shared.bottlenecks.deriv.weight": torch.zeros(8, 16, 1, 1),
        "encoders.shared.residual_heads.deriv.0.weight": torch.zeros(32, 16, 3, 3),  # h=32
        "encoders.shared.residual_heads.deriv.2.weight": torch.zeros(8, 32, 3, 3),
    }
    with pytest.warns(UserWarning, match="head_kind"):
        corrected, _ = cross_check_stream_configs_against_state_dict(cfgs, "state", state)
    assert corrected["deriv"].head_kind == "residual"
    assert corrected["deriv"].head_hidden == 32
    assert corrected["state"].head_kind == "linear"   # untouched


def test_cross_check_head_kind_is_a_noop_when_config_already_residual():
    """A correctly-saved residual config must be returned UNCHANGED (identity),
    not rebuilt -- the reconciliation only fires on the stale-config failure."""
    import torch
    cfgs = {
        "state": LatentStreamConfig(name="state", channels=8, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=8, spatial_size=8,
                                     mode=LatentStreamMode.DECODER,
                                     head_kind="residual", head_hidden=32),
    }
    state = {
        "encoders.shared.bottlenecks.state.weight": torch.zeros(8, 16, 1, 1),
        "encoders.shared.bottlenecks.deriv.weight": torch.zeros(8, 16, 1, 1),
        "encoders.shared.residual_heads.deriv.0.weight": torch.zeros(32, 16, 3, 3),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        corrected, _ = cross_check_stream_configs_against_state_dict(cfgs, "state", state)
    assert corrected is cfgs
