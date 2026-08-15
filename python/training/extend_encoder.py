"""
Replaces stage 1b's own checkpoint-saving setup phase -- see this
module's own investigation notes (session discussion, not yet a file)
for the full rationale: stage 1b's actual TRAINING LOOP has been
inert since it started running at epochs=0 (no gradient step ever
executes), and its D1 decoder is confirmed permanently unnecessary
(deriv lives purely in latent space from stage 2 onward, via L_deriv;
D1's own pixel-space L_recon1 was never wired into stage 2's default
config, and stage 2's own decoder_for_stream-routed access to it would
in fact now KeyError once deriv becomes PURE_LATENT -- see below).
What ACTUALLY needs preserving is the one-time, non-random part of
stage 1b's setup: extending stage 1a's single-stream encoder with a
fresh deriv bottleneck (+ theta-conditioner), transferring stage 1a's
own trained weights into it unchanged, and carrying stats_head0/1
forward -- all done here directly, in memory, with NO intermediate
checkpoint file, unlike stage 1b's own save-then-immediately-reload
round trip.

deriv's own mode is PURE_LATENT here, not DECODER (stage 1b's own
choice) -- accurately reflecting that D1 is never built at all, not
just unused. MultiStreamAutoencoder's own pathway construction
already excludes PURE_LATENT streams entirely (see its own __init__),
so this is not new functionality being added, just a mode this
project's own type system already anticipated but hadn't yet used
for anything real (see models/latent_streams.py's own
LatentStreamMode.PURE_LATENT docstring).
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from models.autoencoder import MultiStreamAutoencoder
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_streams import (
    LatentStreamConfig, LatentStreamMode,
    cross_check_stream_configs_against_state_dict, resolve_stream_configs_from_checkpoint_config,
)
from training.checkpoint_components import _strip_prefix
from training.stats_head import StatsHead


@dataclass
class ExtendedStateCheckpoint:
    """Everything a caller (train_stage2, eventually) needs to continue
    training from a freshly state+deriv-extended model -- returned
    directly, never written to or reloaded from disk. stats_head1 is
    ALWAYS built (unlike stage 1b's own include_stats-conditional
    construction) -- kept available even if never called, since
    nothing about its own construction cost is high enough to justify
    conditioning it on whether some caller's own stats1_weight happens
    to be nonzero right now."""
    ae: MultiStreamAutoencoder
    stats_head0: StatsHead
    stats_head1: StatsHead
    stream_configs: dict[str, LatentStreamConfig]
    state_name: str
    stat_names: list[str]
    mean: torch.Tensor
    std: torch.Tensor
    size: int
    base_channels: int


def extend_state_checkpoint_with_deriv_stream(
    resume_from: Path, latent_channels: int | None = None,
    condition_on_theta: bool = True, device: str | torch.device | None = None,
    deriv_head_hidden: int = 0,
) -> ExtendedStateCheckpoint:
    """
    resume_from: a stage 1a (single-stream, state-only) checkpoint --
    same requirement stage 1b's own resume_from had, and the same
    error if given anything else (checked below).

    latent_channels: the deriv stream's OWN channel count -- None
    (default) matches state's own, identical to stage 1b's own
    parameter of the same name and meaning.

    condition_on_theta: True (default) FiLM-conditions the new deriv
    bottleneck on theta -- identical rationale to stage 1b's own
    parameter (see its docstring): the driving force a(T)=a0*(T-T0)
    vanishes near T0, so a state-only encoder can get direction right
    from the image alone but not magnitude without T. THIS remains the
    only place this structural decision gets made, same as before --
    once built, later stages only ever inherit it from what's saved.

    D1 (stage 1b's own deriv decoder) is NOT built here -- confirmed
    permanently unnecessary (see this module's own docstring). deriv's
    own LatentStreamConfig is PURE_LATENT, not DECODER.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    prev = torch.load(resume_from, map_location=device, weights_only=True)
    model_cfg = prev["config"]
    prev_stream_configs, prev_recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    prev_stream_configs, prev_recon_stream_name = cross_check_stream_configs_against_state_dict(
        prev_stream_configs, prev_recon_stream_name, prev["model_state"],
    )
    if len(prev_stream_configs) != 1:
        raise ValueError(
            f"extend_state_checkpoint_with_deriv_stream() requires resume_from to be a "
            f"SINGLE-stream (stage 1a) checkpoint -- got {len(prev_stream_configs)} streams: "
            f"{list(prev_stream_configs)}. This function's whole point is extending a "
            f"state-only checkpoint with a NEW deriv stream; a checkpoint that already has "
            f"more than one stream isn't what this resumes from."
        )
    state_name = prev_recon_stream_name
    state_cfg = prev_stream_configs[state_name]
    size = model_cfg["size"]
    base_channels = model_cfg["base_channels"]

    prev_stats_config = prev.get("stats_config")
    if prev_stats_config is None:
        raise ValueError(f"{resume_from} has no stats_head (it was trained with stats_weight <= 0 "
                          f"in stage 1a) -- L_stats1 needs the SAME stat_names/normalization "
                          f"stats_head0 uses, which isn't available without it.")
    stat_names = prev_stats_config["stat_names"]
    mean = prev_stats_config["stats_mean"].to(device)
    std = prev_stats_config["stats_std"].to(device)

    deriv_channels = latent_channels if latent_channels is not None else state_cfg.channels
    deriv_spatial = state_cfg.spatial_size

    stream_configs = {
        state_name: state_cfg,
        "deriv": LatentStreamConfig(name="deriv", channels=deriv_channels,
                                     spatial_size=deriv_spatial, mode=LatentStreamMode.PURE_LATENT,
                                     condition_on_theta=condition_on_theta,
                                     head_kind=("residual" if deriv_head_hidden > 0 else "linear"),
                                     head_hidden=deriv_head_hidden),
    }

    # Encoder EXTENDED with the new deriv bottleneck -- built fresh
    # (random init for every parameter, including the trunk+state
    # bottleneck), then the trunk+state parts are overwritten with
    # stage 1a's own trained weights; the deriv bottleneck (and its own
    # theta-FiLM conditioner) are left at their own fresh random init,
    # since stage 1a never had either. Identical mechanism to stage
    # 1b's own equivalent step.
    encoder = Encoder(input_size=size, in_channels=1, base_channels=base_channels,
                       stream_configs=stream_configs, n_theta=1)
    old_encoder_state = _strip_prefix(prev["model_state"], "encoder")
    load_result = encoder.load_state_dict(old_encoder_state, strict=False)
    unexpected_missing = [k for k in load_result.missing_keys
                           if not (k.startswith("bottlenecks.deriv.")
                                   or k.startswith("theta_conditioners.deriv.")
                                   or k.startswith("residual_heads.deriv."))]
    if unexpected_missing or load_result.unexpected_keys:
        raise ValueError(
            f"Loading stage 1a's encoder weights into the extended (state+deriv) encoder didn't "
            f"go as expected -- missing (besides the new deriv bottleneck, which SHOULD be "
            f"missing): {unexpected_missing}, unexpected: {load_result.unexpected_keys}. Likely a "
            f"version mismatch between this codebase and whatever produced the checkpoint."
        )

    D0 = Decoder(output_size=size, out_channels=1, base_channels=base_channels,
                 latent_channels=state_cfg.channels, latent_spatial_size=state_cfg.spatial_size)
    D0.load_state_dict(_strip_prefix(prev["model_state"], "decoder"))

    # decoder_for_stream has NO "deriv" entry -- not just unused, ABSENT.
    # MultiStreamAutoencoder's own pathway construction filters out
    # PURE_LATENT streams before ever indexing decoder_for_stream for
    # them (see its own __init__), so this is never looked up for
    # "deriv" at all -- an absent key is the accurate representation of
    # "no decoder exists for this stream", not an oversight.
    ae = MultiStreamAutoencoder(
        encoders={"shared": encoder}, decoders={"D0": D0},
        stream_configs=stream_configs, decoder_for_stream={state_name: "D0"},
    ).to(device)

    stats_head0 = StatsHead(latent_channels=state_cfg.channels, stat_names=stat_names,
                             latent_spatial=state_cfg.spatial_size).to(device)
    stats_head0.load_state_dict(prev["stats_head_state"])

    # Always built (see this function's own docstring on why, unlike
    # stage 1b's own include_stats-conditional construction) -- fresh
    # random init, since there are no prior stats_head1 weights to
    # transfer (stage 1a never had a deriv stream at all).
    stats_head1 = StatsHead(latent_channels=deriv_channels, stat_names=stat_names,
                             latent_spatial=deriv_spatial).to(device)

    return ExtendedStateCheckpoint(
        ae=ae, stats_head0=stats_head0, stats_head1=stats_head1, stream_configs=stream_configs,
        state_name=state_name, stat_names=stat_names, mean=mean, std=std,
        size=size, base_channels=base_channels,
    )
