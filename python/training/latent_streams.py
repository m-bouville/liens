"""
Type system for the multi-stream latent redesign (C0/C1/... -- see the
project's own design doc for the full architecture/strategy). Kept
separate from constants.py, which is for plain shared VALUES
(LATENT_SPATIAL_SIZE); this is for the actual TYPES describing what a
latent stream is and what it's allowed to be used for.
"""
from dataclasses import dataclass
from enum import Enum
import re
import warnings

from .constants import LATENT_SPATIAL_SIZE

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

    condition_on_theta: whether Encoder should FiLM-condition THIS
    stream's own bottleneck on theta (temperature, centered at T0 -- see
    LatentDynamics' own docstring for the same convention downstream).
    Per-stream, not global: e.g. "deriv" needs it (the driving force
    a(T)=a0*(T-T0) genuinely vanishes near T0 -- critical slowing down --
    so a state-only encoder can get the DIRECTION of change right but
    has no way to know the physically-correct MAGNITUDE without T), while
    "state" (a well-posed function of the image alone, no reason to
    depend on which run produced it) should not be. Deliberately NOT
    made to also depend on dt -- see Encoder's own docstring for why
    that's a materially different (and NOT done here) proposal.
    """
    name: str
    channels: int
    spatial_size: int
    mode: LatentStreamMode
    description: str = ""
    condition_on_theta: bool = False


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


def remap_pre_multistream_state_dict_key(key: str) -> str:
    """
    Maps an Encoder/Autoencoder state_dict key from BEFORE the
    multi-stream redesign to what current code calls the same
    parameter -- the ONE place this rename lives, shared by both
    tests/test_architecture_stability.py's golden-master comparison
    and migrate_checkpoint_to_multistream.py's real checkpoint
    migration, rather than each maintaining its own independent copy
    of the same mapping (the exact class of duplication this whole
    redesign has been eliminating elsewhere -- see e.g.
    constants.LATENT_SPATIAL_SIZE, DEFAULT_STREAM_NAME above).

    encoder.bottleneck.* (a single nn.Conv2d) became
    encoder.bottlenecks.state.* (one entry -- named "state", matching
    Autoencoder's own single-stream default -- of an nn.ModuleDict,
    needed so each stream's projection can be frozen/unfrozen
    independently). Every other key (decoder.*, encoder.down_blocks.*)
    is unchanged, since neither Decoder nor the shared trunk changed.
    """
    if key.startswith("encoder.bottleneck."):
        return key.replace("encoder.bottleneck.", f"encoder.bottlenecks.{DEFAULT_STREAM_NAME}.", 1)
    return key


def resolve_stream_configs_from_checkpoint_config(model_cfg: dict) -> tuple[dict[str, "LatentStreamConfig"], str]:
    """
    (stream_configs, recon_stream_name) from a checkpoint's saved
    "config" dict -- the ONE shared place this resolution happens, used
    by every consumer that needs to rebuild a checkpoint's actual
    architecture (train_stage2.py's train_stage2, when resuming from a
    stage 1 ancestor; evaluation/check_reconstruction.py, when
    visualizing any checkpoint), rather than each maintaining its own
    copy of the same fallback logic.

    "stream_configs"/"recon_stream_name" are NEW checkpoint fields (see
    train_stage1.py's train_autoencoder) -- checkpoints saved before they
    existed have neither, and fall back here to exactly the single
    "autoencoder"-mode stream they've always implicitly had, built from
    the flat latent_channels/latent_spatial_size keys those older
    checkpoints DO have (matching this project's usual
    .get(..., default) backward-compat convention rather than requiring
    every old checkpoint to be migrated just to be read).
    """
    recon_stream_name = model_cfg.get("recon_stream_name", DEFAULT_STREAM_NAME)
    stream_configs_raw = model_cfg.get("stream_configs") or {
        recon_stream_name: {
            "channels": model_cfg["latent_channels"],
            "spatial_size": model_cfg.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
            "mode": LatentStreamMode.AUTOENCODER.value,
        }
    }
    stream_configs = {
        name: LatentStreamConfig(name=name, channels=cfg["channels"],
                                  spatial_size=cfg["spatial_size"], mode=LatentStreamMode(cfg["mode"]),
                                  # .get(..., False): checkpoints saved before theta-conditioning
                                  # existed have no such key -- False (no conditioning) is the
                                  # correct fallback, matching this project's usual backward-compat
                                  # convention rather than requiring every old checkpoint migrated.
                                  condition_on_theta=cfg.get("condition_on_theta", False))
        for name, cfg in stream_configs_raw.items()
    }
    return stream_configs, recon_stream_name


def cross_check_stream_configs_against_state_dict(
    stream_configs: dict[str, LatentStreamConfig], recon_stream_name: str,
    encoder_state_dict: dict,
) -> tuple[dict[str, LatentStreamConfig], str]:
    """
    Cross-checks (and corrects, if needed) a checkpoint's already-
    resolved stream_configs against what its OWN encoder state_dict
    actually contains -- specifically for a checkpoint whose saved
    config metadata is stale or incomplete relative to its real
    weights. This is a genuine failure mode this project hit: an
    intermediate version of the code could train a genuinely
    multi-stream model correctly (the CONSTRUCTION fix was in place)
    without yet recording that fact in the saved checkpoint config
    (the SEPARATE checkpoint-SAVE fix landed later) -- producing a
    checkpoint whose weights are completely valid but whose own
    self-description undercounts its streams, which no amount of
    correctly reading that description can fix. Only reading the
    weights themselves can.

    encoder_state_dict: keys as saved directly under "model_state"/
    "ae_state", NOT yet prefix-stripped -- either "encoder."-prefixed
    (Autoencoder, and pre-MultiStreamAutoencoder checkpoints, e.g.
    "encoder.bottlenecks.state.weight") or "encoders.shared."-prefixed
    (MultiStreamAutoencoder, e.g.
    "encoders.shared.bottlenecks.deriv.weight" -- see autoencoder.py's
    own docstring on why the container holds a NAMED DICT of encoders,
    not a bare .encoder attribute). Both recognized, since this
    function's whole purpose is reading whatever a checkpoint's ACTUAL
    weights say regardless of which version of the code produced them.

    Returns stream_configs/recon_stream_name UNCHANGED if the
    state_dict's own bottleneck stream names already match AND every
    stream's condition_on_theta claim agrees with whether the
    state_dict actually has a theta_conditioners submodule for it (the
    common, correct case -- this is a cheap no-op then). Adds streams
    the metadata was missing entirely, and separately corrects
    condition_on_theta for any stream that WAS present by name but
    whose saved claim disagreed with its actual weights (a distinct,
    narrower case than a fully-missing stream -- a real one this
    project hit more than once: several checkpoint-saving call sites
    across the codebase independently forgot to include
    condition_on_theta when serializing stream_configs, producing a
    checkpoint whose "deriv" stream is correctly NAMED in its saved
    config but silently claims condition_on_theta=False regardless of
    whether its actual encoder was ever conditioned). Never removes a
    stream the metadata claims but the state_dict doesn't have (a
    stranger, different kind of corruption this function doesn't try
    to fix).
    """
    _BOTTLENECK_KEY_RE = re.compile(r"(encoders?\.(?:shared\.)?bottlenecks\.)([^.]+)\.")
    _THETA_CONDITIONER_RE = re.compile(r"encoders?\.(?:shared\.)?theta_conditioners\.([^.]+)\.")
    found_names = set()
    prefix = None
    for key in encoder_state_dict:
        m = _BOTTLENECK_KEY_RE.match(key)
        if m:
            prefix = m.group(1)  # "encoder.bottlenecks." or "encoders.shared.bottlenecks."
            found_names.add(m.group(2))
    theta_conditioned_names = {m.group(1) for key in encoder_state_dict
                                for m in [_THETA_CONDITIONER_RE.match(key)] if m}

    corrected = dict(stream_configs)
    changed = False

    missing = found_names - set(stream_configs) if found_names else set()
    if missing:
        # Channel count is read directly from the actual weight tensor
        # (out_channels of a 1x1 conv, shape[0]) -- a real value, not
        # assumed. Spatial size can't be recovered the same way (1x1
        # convs don't encode it), so an inferred stream is assumed to
        # share the recon stream's own spatial_size, matching this
        # project's own decodable-streams-share-one-size design
        # constraint (see build_stream_configs). Mode is assumed
        # DECODER, not AUTOENCODER -- the safer default for a stream
        # the metadata never claimed as the reconstruction target; at
        # most one AUTOENCODER-mode stream is meaningful, and
        # recon_stream_name already correctly identifies whichever one
        # that is.
        recon_stream = stream_configs[recon_stream_name]
        for name in sorted(missing):
            weight_key = f"{prefix}{name}.weight"
            channels = encoder_state_dict[weight_key].shape[0]
            corrected[name] = LatentStreamConfig(
                name=name, channels=channels, spatial_size=recon_stream.spatial_size,
                mode=LatentStreamMode.DECODER, condition_on_theta=name in theta_conditioned_names,
            )
        warnings.warn(
            f"checkpoint's saved config only described streams {sorted(stream_configs)}, but its "
            f"encoder weights also contain {sorted(missing)} -- likely a checkpoint saved by an "
            f"intermediate version of this codebase (weights correct, config metadata stale). "
            f"Corrected automatically; consider retraining this checkpoint fresh when convenient "
            f"so its saved config matches its weights without needing this correction."
        )
        changed = True

    theta_mismatches = []
    for name, cfg in corrected.items():
        actual = name in theta_conditioned_names
        if cfg.condition_on_theta != actual:
            corrected[name] = LatentStreamConfig(
                name=cfg.name, channels=cfg.channels, spatial_size=cfg.spatial_size,
                mode=cfg.mode, description=cfg.description, condition_on_theta=actual,
            )
            theta_mismatches.append((name, cfg.condition_on_theta, actual))
    if theta_mismatches:
        warnings.warn(
            f"checkpoint's saved config disagreed with its actual weights on condition_on_theta "
            f"for stream(s) {[n for n, _, _ in theta_mismatches]} (claimed vs actual: "
            f"{[(n, claimed, actual) for n, claimed, actual in theta_mismatches]}) -- likely a "
            f"checkpoint-saving call site that forgot to include condition_on_theta when writing "
            f"stream_configs. Corrected automatically from the weights themselves (authoritative); "
            f"consider retraining this checkpoint fresh when convenient so its saved config matches "
            f"its weights without needing this correction."
        )
        changed = True

    if not changed:
        return stream_configs, recon_stream_name
    return corrected, recon_stream_name
