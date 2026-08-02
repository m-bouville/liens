"""
Port a trained stage-1 checkpoint to a LARGER grid size (64 -> 128 -> 256 ...),
keeping `latent_spatial_size` fixed so each doubling of the input adds exactly
one down/up-block pair.

Why a stage-1 checkpoint specifically, and not stage 2 or 3
-----------------------------------------------------------
`orchestration/pipeline.py` already has stage 2 resuming DIRECTLY from stage 1's
own checkpoint (via training/extend_encoder.py), which requires a single-stream,
state-only input. So porting at stage 1 fits the existing pipeline shape with no
new sequencing, and nothing of value is lost by not porting the later stages:

- the deriv bottleneck is `Conv2d(channels[-1], C, 1)`, and `channels[-1]`
  doubles with every added stage (128 -> 256 at 64 -> 128), so it is
  reinitialised whichever checkpoint you start from;
- `f_theta`'s weights are shape-valid at the new size but SEMANTICALLY EMPTY.
  It maps latent -> latent, and the latent basis is *defined* by the state
  bottleneck -- which is reinitialised here. Once stage 1 retrains, z0 is a
  different function of the image in a new, arbitrary coordinate system, and
  f_theta's weights encode a mapping in the old one. Nothing makes stage 1
  converge back to that basis.

The second point is worth stating explicitly because "f_theta and stats_head
transfer wholesale" has been used as an argument for holding
latent_spatial_size fixed. The TENSORS transfer; the MEANING does not.

What transfers and what does not
---------------------------------
Under same-`dx` scaling a 128x128 image is not a 64x64 image at finer
resolution -- it is four times the area with features of the same pixel size.
So index-aligned down_blocks operate at the SAME physical scale in both models,
and shape-matched transfer is also physically meaningful transfer. Concretely,
for 64 -> 128:

    transfers   encoder.down_blocks.{0,1,2}         (~328k params)
                decoder.up_blocks.{1,2,3}           (the shallower ones)
                decoder.output_conv
                stats_head                          (fixed 8x8 latent)
    fresh       encoder.down_blocks.3               (new deepest, ~1.48M)
                decoder.up_blocks.0                 (its mirror, ~427k)
                encoder.bottlenecks.*               built on channels[-1]
                encoder.theta_conditioners.*        built on channels[-1]
                decoder.unbottleneck                built on channels[-1]

Note the decoder's up_blocks are indexed DEEPEST-FIRST (up_blocks[0] is the
one nearest the latent), so the transferring decoder blocks are shifted by the
number of added stages -- old up_blocks[i] becomes new up_blocks[i + n_added].
That offset is the single most error-prone part of this module and is what
`_shift_up_block_indices` exists for.

By parameter count the port is ~25% transfer / ~75% fresh at every rung, which
sounds worse than it is: the fresh parameters sit on the SMALLEST spatial maps
(16x16 at 128), so they do the least work per parameter and sit closest to the
loss in the gradient path. The real cost of the port is the 4x activation
memory in the shallow blocks that DID transfer.

What this deliberately does NOT do
-----------------------------------
- It STRIPS both `stats_head_state` and `stats_config`'s stats_mean/stats_std,
  keeping `stat_names`.

  stats_head_state goes for the SAME reason f_theta does: StatsHead maps
  latent -> statistics, and the latent basis is defined by the state
  bottleneck, which this function reinitialises. Its weights encode a mapping
  in a coordinate system that no longer exists. Keeping them was also an
  outright crash whenever the new run requests a different `stat_names` than
  the source used -- StatsHead's output layer is sized by len(stat_names), so
  a 12-stat source resumed into a 1-stat run fails with a bare
  "size mismatch for net.2.weight", nowhere near anything mentioning a port.
  Dropping it lets train_stage1 build a correctly-sized head instead; it
  already guards on `prev.get("stats_head_state") is not None`.

  stats_mean/stats_std go because they are properties of the DATA, not the
  model, and the distributions genuinely differ at the new size
  (autocorr_length's search cap is min(Nx,Ny)*2/3, so 42 -> 84) -- carrying
  the old values forward would leave every stats target miscentred and
  mis-scaled with no error anywhere.

  Stripping rather than refusing, because `train_stage1` recomputes the
  normalisation from its own dataset on every run (via
  `MicrostructureEvolutionDataset.stats_normalization()`) and never reads the
  resumed checkpoint's copy -- so the staleness self-corrects the moment stage 1
  runs, and refusing would force every legitimate caller to pass an override,
  which is how a warning becomes a reflex. Setting the values to None makes the
  wrong numbers unavailable rather than merely discouraged.

  CONSEQUENCE: the returned checkpoint must go through stage 1 before stage 2.
  `extend_state_checkpoint_with_deriv_stream` requires a complete stats_config
  and will refuse one with None values -- correctly, since it has no dataset of
  its own to recompute from, though its error message will talk about
  stats_weight rather than about porting.
- It does not re-estimate BatchNorm running statistics. They transfer
  numerically but were estimated over a different spatial population; run a few
  hundred forward passes at the new size in train() mode before measuring
  anything.
- It does not shrink. Downscaling would have to DISCARD a trained block and
  choose which, and there is no use case; `to_size < size` is refused.
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from models.autoencoder import Autoencoder
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_streams import (
    cross_check_stream_configs_against_state_dict,
    resolve_stream_configs_from_checkpoint_config,
)
from training.checkpoint_components import (
    _strip_decoder_prefix_for_stream, _strip_encoder_or_decoder_prefix,
)
from training.stats_head import StatsHead


@dataclass
class RescaledCheckpoint:
    """A stage-1 checkpoint rebuilt at a new size, in memory -- never written
    to or reloaded from disk, matching extend_encoder.py's own convention.

    `checkpoint` is a full stage-1-shaped dict ready for
    train_autoencoder(resume_from=...) once written, so the port can be fed
    into the existing pipeline without a new code path. `transferred` and
    `fresh` are the state_dict key prefixes in each category -- returned rather
    than merely logged so a test can assert on them."""
    checkpoint: dict
    encoder: Encoder
    decoder: Decoder
    stats_head: StatsHead | None
    from_size: int
    to_size: int
    n_stages_added: int
    transferred: list[str]
    fresh: list[str]


def _shift_up_block_indices(decoder_state: dict, n_added: int, n_stages_new: int) -> dict:
    """Re-index decoder up_blocks by `n_added` places.

    The decoder's up_blocks run DEEPEST-FIRST: up_blocks[0] takes the latent and
    doubles it once, up_blocks[-1] produces full resolution. Adding a stage
    therefore inserts a NEW up_blocks[0] and pushes every existing block one
    index later, unlike the encoder where the new block is appended at the end
    and existing indices are untouched.

    Getting this wrong is silent, not loud: the shapes of a mis-indexed block
    can still match (up_blocks[2] is 64->32 at 64x64 and up_blocks[3] is 32->32,
    so a one-off lands on a shape mismatch here) but for other channel
    configurations they need not, and a load that succeeds with the wrong
    physical scale per block would be undetectable downstream.
    """
    shifted = {}
    for key, value in decoder_state.items():
        if not key.startswith("up_blocks."):
            shifted[key] = value
            continue
        _, index, rest = key.split(".", 2)
        new_index = int(index) + n_added
        if new_index >= n_stages_new:
            raise ValueError(
                f"up_blocks.{index} shifts to {new_index}, but the new decoder only has "
                f"{n_stages_new} stages -- the checkpoint has more decoder blocks than the "
                f"target size can hold, which means from_size/to_size were computed wrongly."
            )
        shifted[f"up_blocks.{new_index}.{rest}"] = value
    return shifted


def rescale_checkpoint_to_size(
    resume_from: Path | dict, to_size: int, device: str | torch.device | None = None,
    keep_trunk_from_multi_stream: bool = False,
) -> RescaledCheckpoint:
    """
    resume_from: a stage-1 (single-stream, state-only) checkpoint, or an
    already-loaded checkpoint dict. Anything with more than one stream is
    refused -- see this module's docstring for why porting a later stage buys
    nothing.

    to_size: the new grid size. Must be `from_size * 2^k` for integer k >= 1;
    equal or smaller sizes are refused rather than silently no-oping, since a
    caller asking to rescale to the size it already has is almost certainly
    passing the wrong checkpoint.

    keep_trunk_from_multi_stream: accept a stage-2+ checkpoint and port its
    SHARED TRUNK, discarding every non-recon stream. Off by default so that
    passing the wrong checkpoint is an error rather than a silent partial port,
    but worth turning on deliberately: stage 2 trains the shared trunk as well
    as the deriv head, so a stage-2 checkpoint's down_blocks have had training
    the stage-1 checkpoint never saw. The output is single-stream either way,
    which is what stage 2 requires as ITS input.

    What is NOT recoverable from a stage-2 checkpoint, at any setting: the deriv
    bottleneck and its FiLM conditioner. Not merely because their shapes change
    (channels[-1] doubles) but because the FEATURE SPACE they read is produced
    by the newly-added, randomly-initialised deepest down_block -- so new
    channel i has no relationship to old channel i, and no reshaping recovers
    the mapping. The same argument applies to f_theta.

    Any stats_mean/stats_std in the source are stripped (see this module's
    docstring); stat_names is kept.
    """
    device = torch.device(device or "cpu")

    if isinstance(resume_from, dict):
        prev = resume_from
        source = "<in-memory checkpoint>"
    else:
        prev = torch.load(resume_from, map_location=device, weights_only=True)
        source = str(resume_from)

    model_cfg = prev["config"]
    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, prev["model_state"],
    )
    dropped_streams = [n for n in stream_configs if n != recon_stream_name]
    if dropped_streams and not keep_trunk_from_multi_stream:
        raise ValueError(
            f"rescale_checkpoint_to_size() got a {len(stream_configs)}-stream checkpoint "
            f"({list(stream_configs)}) from {source}, and every non-recon stream's bottleneck "
            f"would have to be discarded (they are built on channels[-1], which doubles). "
            f"Pass keep_trunk_from_multi_stream=True to port the SHARED TRUNK from it anyway -- "
            f"that trunk has had stage-2 training the stage-1 checkpoint never saw, and it is "
            f"the only part of a later checkpoint that survives a rescale."
        )
    # Only the recon stream's config is carried; the others' bottlenecks cannot
    # survive the channels[-1] change and the pipeline rebuilds them anyway
    # (stage 2 calls extend_state_checkpoint_with_deriv_stream on a
    # single-stream input).
    stream_configs = {recon_stream_name: stream_configs[recon_stream_name]}
    state_cfg = stream_configs[recon_stream_name]
    from_size = int(model_cfg["size"])
    base_channels = int(model_cfg["base_channels"])
    latent_spatial = state_cfg.spatial_size

    if to_size <= from_size:
        raise ValueError(
            f"to_size={to_size} must be LARGER than the checkpoint's own size={from_size}. "
            f"Downscaling would have to discard a trained block and choose which; it is not "
            f"supported."
        )
    ratio = to_size / from_size
    n_added = int(round(ratio)).bit_length() - 1
    if 2 ** n_added != int(round(ratio)) or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"to_size={to_size} must be from_size * 2^k for integer k >= 1 "
            f"(from_size={from_size}); each doubling adds exactly one down/up-block pair."
        )
    if to_size % latent_spatial != 0 or (to_size // latent_spatial) & (to_size // latent_spatial - 1):
        raise ValueError(
            f"to_size={to_size} is not latent_spatial_size={latent_spatial} * 2^k -- Encoder "
            f"would refuse it too, but failing here says why."
        )

    n_stages_old = (from_size // latent_spatial).bit_length() - 1
    n_stages_new = (to_size // latent_spatial).bit_length() - 1

    if dropped_streams:
        print(f"porting the shared trunk from a {len(dropped_streams) + 1}-stream checkpoint; "
              f"dropping stream(s) {dropped_streams} -- their bottlenecks read features produced "
              f"by the new, randomly-initialised deepest down_block, so their weights have no "
              f"meaning at the new size regardless of shape")

    # Fresh model at the NEW size: every parameter randomly initialised, then
    # the transferable subset is overwritten. Same mechanism as
    # extend_encoder.py's own -- build new, overwrite old, never mutate in
    # place -- so a shape that fails to match surfaces as a load error rather
    # than as a silently half-ported model.
    encoder = Encoder(input_size=to_size, in_channels=1, base_channels=base_channels,
                       stream_configs=stream_configs, n_theta=model_cfg.get("n_theta", 1))
    decoder = Decoder(output_size=to_size, out_channels=1, base_channels=base_channels,
                       latent_channels=state_cfg.channels, latent_spatial_size=latent_spatial)

    # NOT _strip_prefix directly: this codebase uses two different key
    # layouts. Stage 1's plain Autoencoder gives "encoder."/"decoder."; stage
    # 2's MultiStreamAutoencoder gives "encoders.shared." and per-stream
    # "decoders.D0.". _strip_prefix returns an EMPTY DICT on a mismatch rather
    # than raising, so guessing wrong shows up as "every key is missing" at the
    # load call, far from the cause -- which is exactly what happened here on
    # the first real --from-stage2 run, and what these two resolvers already
    # exist to prevent (see their own docstrings).
    old_encoder_state = _strip_encoder_or_decoder_prefix(prev["model_state"], "encoder")
    if not old_encoder_state:
        raise ValueError(
            f"No encoder weights found in {source} under any known key layout "
            f"('encoders.shared.' or 'encoder.'). Keys begin: "
            f"{sorted(prev['model_state'])[:4]}"
        )
    # The bottleneck and theta-conditioner are built on channels[-1], which
    # doubles with every added stage, so their old weights are the WRONG SHAPE
    # and must be dropped rather than offered to load_state_dict -- an
    # unexpected-key error here would be correct but far less clear than
    # naming them up front.
    encoder_transfer = {k: v for k, v in old_encoder_state.items()
                         if not k.startswith(("bottlenecks.", "theta_conditioners."))}
    encoder_result = encoder.load_state_dict(encoder_transfer, strict=False)

    expected_missing_encoder = tuple(f"down_blocks.{i}." for i in range(n_stages_old, n_stages_new))
    unexpected_missing = [
        k for k in encoder_result.missing_keys
        if not k.startswith(expected_missing_encoder + ("bottlenecks.", "theta_conditioners."))
    ]
    if unexpected_missing or encoder_result.unexpected_keys:
        raise ValueError(
            f"Transferring the encoder from {from_size}x{from_size} to {to_size}x{to_size} did "
            f"not go as expected -- missing (besides the {n_stages_new - n_stages_old} new deepest "
            f"down_block(s) and the reinitialised bottleneck/conditioner, which SHOULD be "
            f"missing): {unexpected_missing}; unexpected: {encoder_result.unexpected_keys}."
        )

    old_decoder_state = _strip_decoder_prefix_for_stream(
        prev["model_state"], model_cfg, recon_stream_name)
    if not old_decoder_state:
        raise ValueError(
            f"No decoder weights found in {source} for stream {recon_stream_name!r} under any "
            f"known key layout ('decoders.<name>.', 'decoders.shared.' or 'decoder.'). Keys "
            f"begin: {sorted(prev['model_state'])[:4]}"
        )
    decoder_transfer = {k: v for k, v in old_decoder_state.items()
                         if not k.startswith("unbottleneck.")}
    decoder_transfer = _shift_up_block_indices(decoder_transfer, n_added, n_stages_new)
    decoder_result = decoder.load_state_dict(decoder_transfer, strict=False)

    expected_missing_decoder = tuple(f"up_blocks.{i}." for i in range(n_added))
    unexpected_missing = [
        k for k in decoder_result.missing_keys
        if not k.startswith(expected_missing_decoder + ("unbottleneck.",))
    ]
    if unexpected_missing or decoder_result.unexpected_keys:
        raise ValueError(
            f"Transferring the decoder from {from_size}x{from_size} to {to_size}x{to_size} did "
            f"not go as expected -- missing (besides the {n_added} new deepest up_block(s) and "
            f"the reinitialised unbottleneck, which SHOULD be missing): {unexpected_missing}; "
            f"unexpected: {decoder_result.unexpected_keys}."
        )

    # stats_head is on the fixed latent grid, so its weights are shape-valid.
    # Its NORMALISATION is not (see strict_stats above) -- carried over here
    # only so the returned checkpoint has the same shape as a real stage-1 one.
    # Deliberately NOT carried forward -- see this module's docstring. Built
    # here only so callers that want to inspect the source's head still can;
    # it is not written into the returned checkpoint.
    stats_head = None
    if prev.get("stats_head_state") is not None and prev.get("stats_config") is not None:
        stats_head = StatsHead(latent_channels=state_cfg.channels,
                                stat_names=prev["stats_config"]["stat_names"],
                                latent_spatial=latent_spatial).to(device)
        stats_head.load_state_dict(prev["stats_head_state"])

    encoder, decoder = encoder.to(device), decoder.to(device)

    transferred = sorted({f"encoder.{k.split('.')[0]}.{k.split('.')[1]}"
                           if k.split(".")[0] in ("down_blocks",) else f"encoder.{k.split('.')[0]}"
                           for k in encoder_transfer})
    transferred += sorted({f"decoder.{k.split('.')[0]}.{k.split('.')[1]}"
                            if k.split(".")[0] in ("up_blocks",) else f"decoder.{k.split('.')[0]}"
                            for k in decoder_transfer})
    # Only what the model ACTUALLY contains. theta_conditioners exist solely
    # for streams with condition_on_theta=True (see Encoder.__init__), so a
    # checkpoint whose recon stream is unconditioned has none at all -- listing
    # it as "fresh" would report a reinitialised module that does not exist,
    # which is exactly the sort of small inaccuracy that gets trusted later.
    fresh = [f"encoder.down_blocks.{i}" for i in range(n_stages_old, n_stages_new)]
    fresh.append("encoder.bottlenecks")
    if len(encoder.theta_conditioners):
        fresh.append("encoder.theta_conditioners")
    fresh += [f"decoder.up_blocks.{i}" for i in range(n_added)]
    fresh.append("decoder.unbottleneck")

    # Built from a REAL Autoencoder rather than by concatenating encoder.* and
    # decoder.* by hand. train_stage1 does
    # `ae.load_state_dict(prev["model_state"])` with strict=True, so the key set
    # must be exactly what Autoencoder produces -- and Autoencoder owns state
    # belonging to NEITHER submodule: `log_output_scale`, an
    # EncoderDecoderPair-level buffer (a constant-zero buffer in AUTOENCODER
    # mode, so its VALUE is never in doubt -- but it must be present).
    # Hand-enumerating the two submodules dropped it, and the strict load failed
    # with a bare "Missing key(s): log_output_scale", a long way from anything
    # suggesting a rescale.
    #
    # Taking the whole state_dict from the assembled object means any future
    # top-level parameter is included automatically instead of having to be
    # remembered here.
    autoencoder = Autoencoder(size=to_size, channels=1, base_channels=base_channels,
                               latent_channels=state_cfg.channels,
                               latent_spatial_size=latent_spatial)
    autoencoder.encoder = encoder
    autoencoder.decoder = decoder
    model_state = {k: v.detach().clone() for k, v in autoencoder.state_dict().items()}

    # stats_mean/stats_std stripped, stat_names kept -- see this module's
    # docstring. train_stage1 recomputes them from its own dataset, so the
    # correct values arrive on the first save; what must not happen is the OLD
    # values being read by anything in the meantime.
    stats_config = prev.get("stats_config")
    if stats_config is not None:
        stats_config = dict(stats_config)
        stats_config["stats_mean"] = None
        stats_config["stats_std"] = None

    new_config = dict(model_cfg)
    new_config["size"] = to_size
    # The saved stream_configs must describe the weights that are ACTUALLY
    # present, or cross_check_stream_configs_against_state_dict fails later on
    # a missing bottleneck -- an obscure shape error a long way from its cause.
    new_config["stream_configs"] = {
        name: {"channels": cfg.channels, "spatial_size": cfg.spatial_size,
                "mode": cfg.mode.value, "condition_on_theta": cfg.condition_on_theta}
        for name, cfg in stream_configs.items()
    }
    new_config["recon_stream_name"] = recon_stream_name
    # ported_from_size is provenance, and it is the field that makes a
    # transferred checkpoint distinguishable from a natively-trained one --
    # without it, a checkpoint whose 75%-fresh parameters simply hadn't
    # converged would be indistinguishable from a badly-trained native model.
    new_config["ported_from_size"] = from_size

    checkpoint = {
        "model_state": model_state,
        "stats_head_state": None,
        "epoch": 0,
        "val_loss": float("inf"),
        "val_loss_ema": None,
        "normalized": prev.get("normalized", False),
        # test_dirs point at from_size run directories and would silently leak
        # the wrong sweep into every downstream evaluation.
        "test_dirs": [],
        "config": new_config,
        "stats_config": stats_config,
    }

    return RescaledCheckpoint(
        checkpoint=checkpoint, encoder=encoder, decoder=decoder, stats_head=stats_head,
        from_size=from_size, to_size=to_size, n_stages_added=n_added,
        transferred=transferred, fresh=fresh,
    )


def describe_rescale(rescaled: RescaledCheckpoint) -> str:
    """One-line-per-fact summary for the console.

    Exists because the dangerous outcome of this operation is a SILENT SUCCESS:
    most of the load works, and nothing downstream would otherwise say that a
    quarter of the model is carrying trained weights and three quarters is not.
    """
    lines = [
        f"rescaled checkpoint: trained at {rescaled.from_size}x{rescaled.from_size} "
        f"-> built at {rescaled.to_size}x{rescaled.to_size} "
        f"(+{rescaled.n_stages_added} down/up-block pair(s))",
        f"  transferred: {', '.join(rescaled.transferred)}",
        f"  fresh:       {', '.join(rescaled.fresh)}",
    ]
    n_transferred = sum(p.numel() for p in rescaled.encoder.parameters())
    lines.append(f"  encoder now has {n_transferred:,d} parameters in total")
    lines.append("  NOTE: BatchNorm running statistics were estimated at the OLD size -- "
                  "re-estimate with a few hundred forward passes in train() mode before "
                  "measuring anything (see reestimate_batchnorm_statistics).")
    if rescaled.checkpoint.get("stats_config") is not None:
        lines.append("  NOTE: stats_mean/stats_std were STRIPPED (they are properties of the "
                      f"{rescaled.from_size}x{rescaled.from_size} data). Stage 1 recomputes them; "
                      "run stage 1 before stage 2.")
    return "\n".join(lines)


@torch.no_grad()
def reestimate_batchnorm_statistics(
    model: torch.nn.Module, batches, device: str | torch.device | None = None,
    theta_for: "callable | None" = None, max_batches: int | None = None,
) -> int:
    """
    Recompute every BatchNorm's running mean/var at the NEW input size, by
    running forward passes in train() mode with no optimiser and no gradients.

    Why this is not optional after a rescale. `norm="batch"` means every
    ConvBlock carries running statistics estimated over 64x64 batches. They
    transfer numerically -- the tensors are per-channel and the channel counts
    of the transferred blocks are unchanged -- and they are approximately right,
    since per-channel field statistics do not depend much on how much field
    there is. But they were estimated over a different spatial population, and
    the model runs in eval() mode during every validation and every diagnostic,
    where those running values are used directly rather than the batch's own.
    So a stale estimate shows up as a systematically wrong val_loss from epoch
    0, which is indistinguishable from the transfer having failed.

    Method: reset each BatchNorm's running estimate and momentum to None, which
    switches PyTorch to a CUMULATIVE moving average over the batches seen --
    the exact mean/var over everything passed in, rather than an exponential
    window whose result depends on batch order. Momentum is restored afterwards
    so subsequent training behaves normally.

    `batches` is any iterable of input tensors (or of tuples whose first element
    is the input); `theta_for` optionally maps a batch to its theta argument for
    models that require one. Returns the number of batches consumed, so a caller
    can assert it actually saw data -- an empty iterable would otherwise leave
    the statistics at their reset (mean 0, var 1) state, which is WORSE than the
    stale values it replaced and completely silent.
    """
    device = torch.device(device or next(model.parameters()).device)
    bn_modules = [m for m in model.modules()
                   if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_modules:
        return 0

    saved_momentum = [m.momentum for m in bn_modules]
    for m in bn_modules:
        m.reset_running_stats()
        m.momentum = None  # cumulative average -- order-independent

    was_training = model.training
    model.train()
    seen = 0
    try:
        for batch in batches:
            if max_batches is not None and seen >= max_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            if theta_for is not None:
                model(x, theta_for(batch).to(device))
            else:
                model(x)
            seen += 1
    finally:
        model.train(was_training)
        for m, momentum in zip(bn_modules, saved_momentum):
            m.momentum = momentum

    if seen == 0:
        raise ValueError(
            "reestimate_batchnorm_statistics() consumed zero batches, so every BatchNorm is now "
            "at its reset state (mean 0, var 1) -- worse than the stale values it was meant to "
            "replace, and silent. Check that `batches` is not an exhausted iterator."
        )
    return seen


def extract_stage1_checkpoint(
    source: Path | dict, device: str | torch.device | None = None,
) -> dict:
    """
    A stage-1-shaped checkpoint from a stage-2 one, AT THE SAME SIZE.

    Use when stage 1 stopped early and stage 2 has since kept training the
    autoencoder: with z0_from_deriv_weight=0, stage 2's L_deriv cannot shape
    z0, so every improvement in its recon0/stats0 is ordinary stage-1 progress
    bought at stage-2 prices. This hands that progress back so stage 1 can
    resume from it instead of from its own (possibly much earlier) checkpoint.

    NOT a rescale, and deliberately a separate function rather than
    rescale_checkpoint_to_size(to_size == from_size). The two differ in the
    thing that matters most:

        rescale   channels[-1] doubles, so the bottleneck, the FiLM
                  conditioners and the decoder's unbottleneck CANNOT carry
                  over -- they are reinitialised, and f_theta/stats_head are
                  discarded because the latent basis is rebuilt.
        extract   nothing changes shape, so EVERYTHING the recon pathway
                  owns carries over unchanged, stats_head included. The only
                  losses are the deriv stream and its head, which stage 2
                  rebuilds from scratch anyway.

    Running a rescale at equal size would therefore throw away exactly the
    trained weights this function exists to preserve.

    stats_config is kept INTACT here, unlike in a port: stats_mean/stats_std
    are properties of the DATA, and at the same size on the same sweep they
    are still correct. Stripping them would force a needless recomputation and
    would look like the port's behaviour for a reason that does not apply.

    Returns the checkpoint dict; the caller writes it (see port_checkpoint.py
    for the atomic-save + backup convention).
    """
    device = torch.device(device or "cpu")
    if isinstance(source, dict):
        prev, origin = source, "<in-memory checkpoint>"
    else:
        prev = torch.load(source, map_location=device, weights_only=True)
        origin = str(source)

    model_cfg = prev["config"]
    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, prev["model_state"],
    )
    dropped = [n for n in stream_configs if n != recon_stream_name]

    encoder_state = _strip_encoder_or_decoder_prefix(prev["model_state"], "encoder")
    decoder_state = _strip_decoder_prefix_for_stream(
        prev["model_state"], model_cfg, recon_stream_name)
    if not encoder_state or not decoder_state:
        raise ValueError(
            f"Could not find encoder and/or decoder weights in {origin} under any known key "
            f"layout. Keys begin: {sorted(prev['model_state'])[:4]}"
        )

    # Only the recon stream's own bottleneck/conditioner: the ModuleDicts are
    # keyed by stream name, so the dropped streams' entries must go with them
    # or the single-stream Encoder will refuse them as unexpected keys.
    keep = {}
    for key, value in encoder_state.items():
        parts = key.split(".")
        if parts[0] in ("bottlenecks", "theta_conditioners"):
            if len(parts) > 1 and parts[1] != recon_stream_name:
                continue
        keep[key] = value
    encoder_state = keep

    state_cfg = stream_configs[recon_stream_name]
    size = int(model_cfg["size"])
    autoencoder = Autoencoder(size=size, channels=1,
                               base_channels=int(model_cfg["base_channels"]),
                               latent_channels=state_cfg.channels,
                               latent_spatial_size=state_cfg.spatial_size)
    # The single-stream Autoencoder names its bottleneck without a stream key
    # (see models/autoencoder.py's own _STREAM_NAME), so the recon stream's
    # entries are renamed rather than dropped.
    target = set(autoencoder.encoder.state_dict())
    renamed = {}
    for key, value in encoder_state.items():
        if key not in target:
            stripped = key.replace(f".{recon_stream_name}.", ".", 1)
            if stripped in target:
                key = stripped
        renamed[key] = value
    autoencoder.encoder.load_state_dict(renamed)
    autoencoder.decoder.load_state_dict(decoder_state)
    autoencoder = autoencoder.to(device)

    new_config = dict(model_cfg)
    new_config["stream_configs"] = {
        recon_stream_name: {"channels": state_cfg.channels,
                             "spatial_size": state_cfg.spatial_size,
                             "mode": state_cfg.mode.value,
                             "condition_on_theta": state_cfg.condition_on_theta}
    }
    new_config["recon_stream_name"] = recon_stream_name
    new_config.pop("decoder_for_stream", None)
    new_config["extracted_from_stage2"] = True

    if dropped:
        print(f"extracted a stage-1 checkpoint from {origin}: kept the recon pathway "
              f"({recon_stream_name} + its decoder + stats_head), dropped stream(s) {dropped}. "
              f"Nothing was reinitialised -- the size is unchanged, so every trained weight the "
              f"recon pathway owns carries over.")

    return {
        "model_state": {k: v.detach().clone() for k, v in autoencoder.state_dict().items()},
        # stats_head0 is on the recon latent, whose basis is UNCHANGED here, so
        # unlike a port its weights are still meaningful.
        "stats_head_state": prev.get("stats_head_state"),
        "epoch": 0,
        "val_loss": float("inf"),
        "val_loss_ema": None,
        "normalized": prev.get("normalized", False),
        "test_dirs": prev.get("test_dirs", []),
        "config": new_config,
        "stats_config": prev.get("stats_config"),
    }
