"""
Type system for the multi-stream latent redesign (C0/C1/... -- see the
project's own design doc for the full architecture/strategy). Kept
separate from constants.py, which is for plain shared VALUES
(LATENT_SPATIAL_SIZE); this is for the actual TYPES describing what a
latent stream is and what it's allowed to be used for.
"""
from dataclasses import dataclass
from enum import Enum
import warnings

# The single AUTOENCODER-mode stream's conventional name, when there's
# only one -- e.g. Autoencoder's own internal single-stream config, and
# any code (like refinement_loss.py, pre-C0/C1-redesign) that still
# only knows about one stream and needs to unwrap Encoder's dict return
# to get at it. A NAMED, SHARED constant specifically so this doesn't
# become yet another independently-hardcoded magic string duplicated
# across files -- the exact class of problem LATENT_SPATIAL_SIZE (see
# constants.py) already exists to avoid for the bottleneck size.
DEFAULT_STREAM_NAME = "state"


class LatentStreamMode(Enum):
    """
    What a given latent stream is allowed to be used for -- enforced
    at the point of use (decode_stream/autoencode_stream), not just
    documented, specifically because two streams sharing one Decoder
    (required whenever both are decodable, since Decoder itself is
    fully stream-agnostic) can have IDENTICAL shape while meaning
    completely different things -- e.g. C0 (a microstructure) and C1
    (its time derivative) -- so a shape check alone cannot catch one
    being passed in place of the other. The mode is what's left to
    catch that, at the one remaining seam (the caller has to say which
    stream this actually is).

    AUTOENCODER: can go through Autoencoder's own wrapper -- decoded
    and compared against THIS STREAM's own encoder input (a genuine
    reconstruction). At most one stream may have this mode per model
    (enforced with a warning, not a hard error, at config-build time --
    see build_stream_configs).

    DECODER: can be decoded (same Decoder, since Decoder is
    stream-agnostic), but the result means something ELSE -- compared
    against a DIFFERENT target than this stream's own encoder input
    (e.g. C1 decoded and compared against a finite-difference time
    derivative, never against x itself). Calling Autoencoder's
    reconstruction wrapper on a DECODER-mode stream is a mistake this
    project has specifically flagged as dangerous (silently plausible,
    numerically nonsensical) -- decode_stream() allows it,
    autoencode_stream() refuses it.

    PURE_LATENT: never decoded at all. Calling decode_stream() on it is
    a mistake, refused the same way. Not used by anything yet -- its
    own per-stream channel/spatial-size syntax in the stage-parameters
    file doesn't exist yet either (see build_stream_configs's own
    docstring) -- included now so the type system doesn't need
    revisiting when it is.
    """
    AUTOENCODER = "autoencoder"
    DECODER = "decoder"
    PURE_LATENT = "pure_latent"


@dataclass
class LatentStreamConfig:
    """
    One latent stream's full description: what it's called, its shape,
    and what it's allowed to be used for. A dict[str, LatentStreamConfig]
    keyed by name (not a list) is the standard way these are passed
    around -- so a caller always looks a stream up BY NAME (e.g.
    z["deriv"]) rather than by a positional index that a reorder or
    rename could silently invalidate.
    """
    name: str
    channels: int
    spatial_size: int
    mode: LatentStreamMode
    description: str = ""


def decode_stream(decoder, z, stream: LatentStreamConfig):
    """
    decoder(z), guarded by two checks decoder itself can't perform (it
    stays fully stream-agnostic by design -- see decoder.py):

    1. stream.mode must not be PURE_LATENT -- decoding a stream that's
       declared never-decodable is refused outright, not silently run.
    2. z's actual shape must match stream's declared shape.

    NEITHER check can catch a same-shaped stream passed under the
    WRONG config (e.g. z0 mislabeled as the "deriv" stream) -- that's
    a real, deliberately-accepted limitation (see LatentStreamMode's
    own docstring): correctness there has to come from wherever z was
    actually built (Encoder.forward, which constructs each entry
    directly from its own stream_configs), not from a check performed
    here on tensors and configs that are, by the time they reach this
    function, independent of each other.
    """
    if stream.mode == LatentStreamMode.PURE_LATENT:
        raise TypeError(f"stream '{stream.name}' is pure_latent, cannot be decoded")
    expected_shape = (stream.channels, stream.spatial_size, stream.spatial_size)
    if tuple(z.shape[-3:]) != expected_shape:
        raise ValueError(f"z has shape {tuple(z.shape)}, doesn't match stream "
                          f"'{stream.name}' (expected (..., {expected_shape[0]}, "
                          f"{expected_shape[1]}, {expected_shape[2]}))")
    return decoder(z)


def autoencode_stream(ae, x, stream: LatentStreamConfig):
    """
    ae(x) (full encode-decode-return-both round trip), guarded by
    mode: only a genuinely AUTOENCODER-mode stream may go through this
    -- a DECODER-mode stream's decode is compared against a DIFFERENT
    target than its own input (e.g. C1 vs a finite-difference
    derivative, never vs x), so routing it through the reconstruction
    wrapper would silently compute a meaningless "reconstruction loss"
    against the wrong target. Use decode_stream() directly for those.
    """
    if stream.mode != LatentStreamMode.AUTOENCODER:
        raise TypeError(f"stream '{stream.name}' has mode={stream.mode.value}, not "
                         f"autoencoder -- use decode_stream() instead, against "
                         f"whatever target this stream's mode actually implies")
    return ae(x)


def build_stream_configs(
    names: list[str], modes: list[str], channels_decoder: int, spatial_decoder: int,
) -> dict[str, LatentStreamConfig]:
    """
    Builds stream_configs from a stage-parameters file's 4 keys:
        latent_names            = state, deriv
        latent_modes            = autoencoder, decoder
        latent_channels_decoder = 8
        latent_spatial_decoder  = 8

    channels_decoder/spatial_decoder are SHARED across every
    AUTOENCODER/DECODER-mode stream -- this is what the
    equal-channel-count-for-decodability constraint (see
    LatentStreamMode's own docstring, and decoder.py) looks like in the
    params file: one shared value, not a per-stream one, so there's no
    way to even EXPRESS mismatched decodable-stream sizes by accident.

    PURE_LATENT streams would need their OWN per-stream channels/
    spatial-size syntax -- doesn't exist yet (nothing uses PURE_LATENT
    currently) -- raises NotImplementedError if requested rather than
    silently building something wrong.

    To replicate pre-redesign (single-stream) behavior exactly:
        latent_names = state
        latent_modes = autoencoder
    """
    if len(names) != len(modes):
        raise ValueError(f"latent_names ({len(names)}) and latent_modes ({len(modes)}) "
                          f"must have the same length, got {names} and {modes}")
    if len(set(names)) != len(names):
        raise ValueError(f"latent_names must be unique, got {names}")

    configs: dict[str, LatentStreamConfig] = {}
    autoencoder_streams = []
    for name, mode_str in zip(names, modes):
        try:
            mode = LatentStreamMode(mode_str)
        except ValueError:
            raise ValueError(
                f"unknown latent stream mode '{mode_str}' for stream '{name}' -- "
                f"must be one of {[m.value for m in LatentStreamMode]}"
            )
        if mode == LatentStreamMode.PURE_LATENT:
            raise NotImplementedError(
                f"stream '{name}': pure_latent streams need their own per-stream "
                f"channels/spatial-size syntax in the params file, which doesn't "
                f"exist yet (see build_stream_configs' own docstring)"
            )
        configs[name] = LatentStreamConfig(
            name=name, channels=channels_decoder, spatial_size=spatial_decoder, mode=mode,
        )
        if mode == LatentStreamMode.AUTOENCODER:
            autoencoder_streams.append(name)

    if len(autoencoder_streams) > 1:
        warnings.warn(
            f"multiple streams declared mode=autoencoder: {autoencoder_streams} -- "
            f"at most one is meaningful (Autoencoder's own reconstruction wrapper "
            f"needs a single unambiguous stream); only relevant if something later "
            f"tries to build an Autoencoder from this config, not to raw "
            f"Encoder/Decoder use, which doesn't care."
        )

    return configs
