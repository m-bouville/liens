"""
Shared helpers between training/train_stage1.py and
training/train_stage2.py -- extracted from what used to be one
combined train_ae.py, split once stage 1b's own removal shrank stage 1
(train_autoencoder) and stage 2 (train_stage2) down to roughly the two
functions this module now sits between. Neither of these is meaningful
on its own outside that context: freeze_outer_layers/compute_weight_drift
are specifically about a curriculum-style, partially-frozen resume from
a prior stage's own checkpoint (stage 2's own n_frozen_stages), not a
general-purpose utility this project needs anywhere else.
"""

import torch

from models.autoencoder import Autoencoder, EncoderDecoderPair, MultiStreamAutoencoder


def freeze_outer_layers(ae: Autoencoder | EncoderDecoderPair | MultiStreamAutoencoder,
                         n_frozen_stages: int) -> list[torch.nn.Module]:
    """
    Freezes the OUTERMOST n_frozen_stages layers on each side (closest to
    real space, farthest from the latent bottleneck): the encoder's
    FIRST n_frozen_stages DownBlocks, and the decoder's LAST
    n_frozen_stages UpBlocks plus its final output_conv. Layers closest
    to the latent bottleneck (encoder's bottleneck 1x1 conv, decoder's
    unbottleneck 1x1 conv, and any remaining inner DownBlocks/UpBlocks)
    stay trainable, since that's where stage 2 needs room to reshape the
    latent geometry.

    NOTE, corrected after checking actual parameter counts: outer layers
    are NOT the largest in this network, despite operating at the
    largest spatial resolution (most FLOPs-expensive to run forward).
    Conv parameter count depends only on channel counts (in_channels *
    out_channels * kernel_size^2), not spatial size -- and channels
    DOUBLE going inward (e.g. 32 -> 64 -> 128), so the outermost stage
    has the FEWEST channels, hence the fewest parameters, while the
    deepest/innermost stage alone can hold the majority of the network's
    weights. Freezing outer stages therefore does NOT meaningfully speed
    up training via reduced optimizer work -- for a shallow network
    (few stages total), it barely reduces trainable parameter count at
    all unless nearly every stage is frozen, which would leave stage 2
    almost nothing to actually work with. Treat n_frozen_stages as a
    regularization/degrees-of-freedom knob, not a speed optimization.

    IMPORTANT CAVEAT, not just a footnote: this reduces the degrees of
    freedom available for the encoder/decoder to drift together, and
    makes such drift much less likely to be found by gradient descent
    (fewer trainable parameters, starting from a good stage-1
    initialization) -- but it is NOT a structural guarantee against the
    scale-collapse failure mode observed empirically. bottleneck and
    unbottleneck (both plain 1x1 convs, i.e. arbitrary linear maps) sit
    immediately adjacent to z on each side and stay trainable here --
    on their own, they are SUFFICIENT to implement an arbitrary
    rescaling of z (one scales up, the other scales back down) that
    L_recon cannot detect, regardless of how many other layers are
    frozen. Freezing narrows the search space; it doesn't close this
    loophole structurally. See compute_weight_drift() for a diagnostic
    that catches it if it happens anyway.

    Returns the frozen submodules. requires_grad_(False) alone stops
    gradient-based updates, but NOT BatchNorm's running_mean/running_var
    -- those are buffers, updated via a forward-pass EMA every time the
    module runs in train() mode, entirely independent of requires_grad.
    Since the training loop calls ae.train() every epoch (recursively
    setting EVERY submodule, frozen or not, back to train mode), callers
    must re-apply .eval() to exactly this returned list right after each
    ae.train() call, or "frozen" blocks with BatchNorm will keep
    drifting via their running stats even though their learnable weights
    are correctly held fixed.

    Freezes EVERY decoder found (ae.decoders, if the container has more
    than one -- e.g. a stage-1b-derived checkpoint's own D0/D1), not
    just one -- symmetric treatment for every decode pathway, not just
    whichever happens to be named "shared".
    """
    frozen_modules: list[torch.nn.Module] = []
    if n_frozen_stages <= 0:
        return frozen_modules
    # MultiStreamAutoencoder doesn't expose .encoder/.decoder directly
    # (only .encoders["shared"]/.decoders[...], see its own docstring
    # on why) -- Autoencoder/EncoderDecoderPair still do.
    encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
    decoders = [ae.decoder] if hasattr(ae, "decoder") else list(ae.decoders.values())
    for block in encoder.down_blocks[:n_frozen_stages]:
        for p in block.parameters():
            p.requires_grad_(False)
        frozen_modules.append(block)
    for decoder in decoders:
        for block in decoder.up_blocks[-n_frozen_stages:]:
            for p in block.parameters():
                p.requires_grad_(False)
            frozen_modules.append(block)
        for p in decoder.output_conv.parameters():
            p.requires_grad_(False)
        frozen_modules.append(decoder.output_conv)
    return frozen_modules


def _param_group(key: str) -> str:
    """Groups a state_dict/named_parameters/named_buffers key by its
    containing block. Keys come from MultiStreamAutoencoder specifically
    (the only kind of model train_stage2 ever builds), e.g.
    'encoders.shared.down_blocks.0.conv1.weight' ->
    'encoders.shared.down_blocks.0', 'decoders.shared.output_conv.weight'
    -> 'decoders.shared.output_conv', 'pathways.deriv.log_output_scale'
    -> 'pathways.deriv.log_output_scale' (its own group -- a single
    scalar, not part of a larger block to group further)."""
    parts = key.split(".")
    if len(parts) >= 4 and parts[2] in ("down_blocks", "up_blocks"):
        return ".".join(parts[:4])
    return ".".join(parts[:3])


def _drift_by_block(initial: dict, final: dict) -> dict[str, float]:
    totals: dict[str, float] = {}
    for key in initial:
        group = _param_group(key)
        diff_sq = (final[key].float() - initial[key].float()).pow(2).sum().item()
        totals[group] = totals.get(group, 0.0) + diff_sq
    return {group: total**0.5 for group, total in totals.items()}


def compute_weight_drift(
    initial_params: dict, initial_buffers: dict, final_params: dict, final_buffers: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Per-block L2 norm of the change in learnable PARAMETERS and,
    SEPARATELY, in BUFFERS (e.g. BatchNorm's running_mean/running_var)
    between initial and final state.

    Kept separate deliberately: parameter drift in a frozen block should
    be EXACTLY 0 (a red flag if it isn't -- something is wrong with the
    freeze itself). Buffer drift in a frozen block should ALSO be ~0 once
    freeze_outer_layers()'s returned modules are kept in .eval() mode
    every epoch (see that function's docstring) -- but treating the two
    as one number (as an earlier version of this function did) made it
    impossible to tell "the freeze isn't working" apart from "BatchNorm
    bookkeeping moved a little", which are very different problems.
    """
    return (_drift_by_block(initial_params, final_params),
            _drift_by_block(initial_buffers, final_buffers))


