"""
Turns componentized checkpoints (see checkpoint_components.py) into
live nn.Modules ready for the stage 4/5 training loop -- the direct
next step after loading, since nothing can train on a ComponentCheckpoint
directly, only on real Autoencoder/StatsHead/LatentDynamics instances.
"""
import torch
import torch.nn as nn

from models.autoencoder import Autoencoder, MultiStreamAutoencoder
from models.constants import LATENT_SPATIAL_SIZE
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_dynamics import LatentDynamics
from models.latent_streams import LatentStreamConfig, LatentStreamMode, resolve_stream_configs_from_checkpoint_config
from training.checkpoint_components import ComponentCheckpoint
from training.stats_head import StatsHead


def build_models_from_components(
    components: dict[str, ComponentCheckpoint], device: str | None = None,
    freeze_decoder: bool = False, in_channels: int = 1,
) -> tuple[Autoencoder | MultiStreamAutoencoder, StatsHead | None, LatentDynamics, list[nn.Module],
           dict[str, LatentStreamConfig], str]:
    """
    freeze_decoder: sets requires_grad_(False) on every decoder
    parameter and puts it in .eval() mode -- stage 4's mode (D stays
    fixed, used only as a tether for L_recon). False for stage 5 (D
    trains too). NOTE this does NOT reduce forward/backward compute
    through D -- gradient still has to flow through it to reach E via
    L_recon = ||D(E(x)) - x||. It only removes D's parameters from the
    optimizer's own state.

    stats_head, if present, is ALWAYS frozen regardless of
    freeze_decoder -- a deliberate choice, not an oversight: it stays
    the same fixed measuring instrument throughout stage 4/5 that it
    was in stage 2, rather than being retrained at any point in this
    pipeline.

    in_channels: NOT recorded anywhere in the existing checkpoint
    format (every checkpoint this project has ever produced used
    grayscale, in_channels=1) -- exposed rather than silently
    hardcoded, but defaults to 1 to match all of them.

    Returns (ae, stats_head, f_theta, frozen_modules, stream_configs,
    recon_stream_name). frozen_modules is for the training loop:
    ae.train() is recursive and would otherwise flip any frozen
    BatchNorm layers back to train mode every epoch (the exact bug
    fixed in stage 2's freeze_outer_layers) -- the caller must re-apply
    .eval() to exactly this list right after every ae.train() call.

    stream_configs/recon_stream_name are the ACTUAL ones `ae` was built
    with -- NOT necessarily identical to what resolve_stream_configs_
    from_checkpoint_config(components["encoder"].config) alone would
    give, since the multi-stream branch below force-overrides every
    non-recon stream to PURE_LATENT (see that branch's own comment).
    A caller that goes on to SAVE a checkpoint (train_refinement.py)
    needs THESE, not the ancestor's own stale config -- saving the
    ancestor's original config against a state_dict that was actually
    built PURE_LATENT (no deriv decoder pathway at all) produces a
    checkpoint that looks internally consistent but silently isn't:
    every OTHER tool that later loads it (e.g. check_reconstruction.py)
    trusts the saved config, tries to build a decoder pathway the
    state_dict was never given, and fails with a confusing missing-key
    RuntimeError far from where the actual mismatch was introduced.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    frozen_modules: list[nn.Module] = []

    encoder_cfg = components["encoder"].config
    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(encoder_cfg)
    recon_stream = stream_configs[recon_stream_name]

    if len(stream_configs) == 1:
        final_stream_configs = stream_configs
        ae = Autoencoder(size=encoder_cfg["size"], channels=in_channels,
                          base_channels=encoder_cfg["base_channels"],
                          latent_channels=recon_stream.channels,
                          latent_spatial_size=recon_stream.spatial_size).to(device)

        # Reassemble the combined Autoencoder's state_dict by re-adding the
        # "encoder."/"decoder." prefixes ComponentCheckpoint's _strip_prefix
        # removed -- the exact inverse operation.
        combined_state = {}
        combined_state.update({f"encoder.{k}": v for k, v in components["encoder"].state_dict.items()})
        combined_state.update({f"decoder.{k}": v for k, v in components["decoder"].state_dict.items()})
        try:
            result = ae.load_state_dict(combined_state, strict=False)
        except RuntimeError as e:
            # strict=False only relaxes missing/unexpected KEYS -- a SHAPE
            # mismatch for a key present in both (the most likely symptom of
            # a real version mismatch, e.g. latent_channels disagreement)
            # still raises RuntimeError directly, before even reaching the
            # missing/unexpected check below. Re-raised with the same clear
            # message rather than left as a raw PyTorch error.
            raise ValueError(
                f"Reassembled Autoencoder state_dict doesn't match the current model "
                f"definition (shape mismatch): {e}. Likely a version mismatch between "
                f"this codebase and whatever produced the checkpoint."
            ) from e
        # log_output_scale is a genuinely NEW, model-level attribute (see
        # autoencoder.py's own EncoderDecoderPair) -- never part of either
        # component's own state_dict, so it's always "missing" when
        # reassembling from componentized encoder/decoder checkpoints,
        # regardless of when they were saved. Its absence here means
        # "this checkpoint predates the scale-correction feature," not a
        # real version mismatch -- defaults to log_output_scale=0 (scale=1,
        # no correction), the same as if the feature never existed, exactly
        # matching what these older checkpoints actually did. Filtered out
        # before the strict check below so a REAL missing/unexpected key
        # (an actual version mismatch) still raises.
        missing_keys = [k for k in result.missing_keys if not k.endswith("log_output_scale")]
        if missing_keys or result.unexpected_keys:
            raise ValueError(
                f"Reassembled Autoencoder state_dict doesn't match the current model "
                f"definition -- missing keys: {missing_keys}, unexpected keys: "
                f"{result.unexpected_keys}. Likely a version mismatch between this codebase "
                f"and whatever produced the checkpoint."
            )
        decoder_module = ae.decoder
    else:
        # Every non-recon stream forced to PURE_LATENT -- see this
        # function's own docstring on why (z1 is teacher-forced, never
        # decoded, and only the recon stream's own decoder was ever
        # preserved by checkpoint_components.py in the first place).
        pure_latent_stream_configs = {
            name: (cfg if name == recon_stream_name
                   else LatentStreamConfig(name=cfg.name, channels=cfg.channels,
                                            spatial_size=cfg.spatial_size,
                                            mode=LatentStreamMode.PURE_LATENT))
            for name, cfg in stream_configs.items()
        }
        final_stream_configs = pure_latent_stream_configs
        encoder = Encoder(input_size=encoder_cfg["size"], in_channels=in_channels,
                           base_channels=encoder_cfg["base_channels"],
                           stream_configs=pure_latent_stream_configs)
        decoder = Decoder(output_size=encoder_cfg["size"], out_channels=in_channels,
                           base_channels=encoder_cfg["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size)
        # Component state dicts are ALREADY bare/unprefixed (see
        # _strip_prefix's own docstring: built specifically so the
        # result "loads directly into a standalone Encoder()/Decoder()
        # instance") -- no prefix reconstruction needed here, unlike
        # the single-stream Autoencoder path above.
        try:
            encoder.load_state_dict(components["encoder"].state_dict)
            decoder.load_state_dict(components["decoder"].state_dict)
        except RuntimeError as e:
            raise ValueError(
                f"Reassembled multi-stream Encoder/Decoder state_dict doesn't match the "
                f"current model definition (shape mismatch): {e}. Likely a version "
                f"mismatch between this codebase and whatever produced the checkpoint."
            ) from e
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                     stream_configs=pure_latent_stream_configs).to(device)
        decoder_module = ae.pathways[recon_stream_name].decoder

    if freeze_decoder:
        for p in decoder_module.parameters():
            p.requires_grad_(False)
        decoder_module.eval()
        frozen_modules.append(decoder_module)

    stats_head = None
    if "stats_head" in components:
        sh = components["stats_head"]
        # hidden_dim isn't stored explicitly in the checkpoint (stage 1's
        # stats_config only records stat_names/stats_mean/stats_std) --
        # inferred from the saved weights' own shape instead, since a
        # checkpoint could have used a non-default hidden_dim (e.g. the
        # hidden_dim=16 experiment from a few turns back) and silently
        # defaulting to 128 here would fail to load with a shape
        # mismatch, or worse, load incorrectly if the shapes coincided.
        hidden_dim = sh.state_dict["net.0.weight"].shape[0]
        stats_head = StatsHead(latent_channels=sh.config["latent_channels"],
                                stat_names=sh.config["stat_names"],
                                latent_spatial=sh.config.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
                                hidden_dim=hidden_dim).to(device)
        stats_head.load_state_dict(sh.state_dict)
        stats_head.eval()
        for p in stats_head.parameters():
            p.requires_grad_(False)
        frozen_modules.append(stats_head)

    lds_cfg = components["lds"].config
    f_theta = LatentDynamics(latent_channels=lds_cfg["latent_channels"], n_theta=lds_cfg["n_theta"],
                              latent_spatial=lds_cfg.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
                              hidden_dim=lds_cfg["hidden_dim"],
                              n_hidden_layers=lds_cfg["n_hidden_layers"]).to(device)
    f_theta.load_state_dict(components["lds"].state_dict)

    return ae, stats_head, f_theta, frozen_modules, final_stream_configs, recon_stream_name
