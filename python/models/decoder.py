"""
Convolutional decoder: maps a latent representation back to real-space
PIXELS -- what those pixels actually MEAN depends entirely on which
latent stream produced the input, not on anything Decoder itself knows
or checks (it's fully stream-agnostic by design, see the class
docstring below). Fed a "state" stream, the output is a reconstructed
microstructure. Fed a "deriv" stream (see the project's own C0/C1
design doc), the output is NOT a microstructure at all -- it's
whatever that stream represents (e.g. a time derivative), decoded
through the exact same weights. Nothing about the return value, its
shape, or this module's own code distinguishes the two cases -- that
distinction has to come from the caller, via a LatentStreamConfig (see
latent_streams.py's decode_stream, which is the safer way to call this
for exactly this reason).
Mirrors encoder.py's depth/channel schedule exactly.
"""

import math

import torch
import torch.nn as nn

from .blocks import UpBlock
from .constants import LATENT_SPATIAL_SIZE


class Decoder(nn.Module):
    """
    A 1x1 conv expands latent_channels back to the encoder's final
    channel count (spatial stays at latent_spatial_size x
    latent_spatial_size, 8x8 by default), then repeated UpBlocks double
    spatial resolution back up to output_size, mirroring Encoder's
    DownBlocks in reverse.

    IMPORTANT: output_size, base_channels, norm, and the specific
    latent_channels/latent_spatial_size of WHICHEVER stream is being
    decoded must match that stream's own values as declared in the
    paired Encoder's stream_configs, or shapes won't line up. Since
    Encoder can produce multiple streams (see latent_streams.py) while
    Decoder only ever consumes one at a time, this is a per-stream
    match, not "N values that must agree" the way it was before the
    multi-stream redesign -- Decoder itself has no notion of "streams"
    at all, and doesn't need one; whichever tensor it's handed just
    needs to be shaped like something Encoder actually produces. This
    isn't enforced here (Decoder is built standalone) -- autoencoder.py
    should be the single place that constructs both from one shared
    config.

    Skip connections: accepted as a `skips` argument for forward-API
    compatibility with the future U-Net path, but NOT wired up yet.
    Encoder's skip features and this decoder's injection points differ
    by one channel-schedule index (skips[j] has channels[j+1], but the
    matching UpBlock stage needs channels[j] -- see docs/neural_nets.md's
    U-Net note), so a real connection needs an adapter conv, which
    belongs in skip_connections.py, not here. For now skips is always
    ignored and every UpBlock runs skip=None, i.e. "zero" as requested.
    """

    def __init__(
        self,
        output_size: int,
        out_channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 16,
        latent_spatial_size: int = LATENT_SPATIAL_SIZE,
        norm: str = "batch",
        use_skips: bool = False,
    ):
        super().__init__()

        n_stages = math.log2(output_size / latent_spatial_size)
        if not n_stages.is_integer() or n_stages < 1:
            raise ValueError(
                f"output_size must be latent_spatial_size * 2^k for integer k >= 1 "
                f"(latent_spatial_size={latent_spatial_size}), got output_size={output_size}"
            )
        n_stages = int(n_stages)

        self.output_size = output_size
        self.latent_spatial_size = latent_spatial_size
        self.n_stages = n_stages
        self.use_skips = use_skips

        # Same formula as Encoder's channel list (index 0 = channel count
        # at full resolution, i.e. out_channels here vs in_channels there).
        channels = [out_channels] + [base_channels * 2**i for i in range(n_stages)]
        self.channels = channels
        hidden_channels = channels[1:]  # [c_1, ..., c_n], excludes out_channels

        self.unbottleneck = nn.Conv2d(latent_channels, hidden_channels[-1], kernel_size=1)

        # n_stages UpBlocks give n_stages spatial doublings (8 -> output_size),
        # same as before. But the LAST one now stays at hidden_channels[0]
        # (channel-preserving upsample) instead of dropping straight to
        # out_channels -- because UpBlock's internal ConvBlock always ends
        # in ReLU, and ending the whole decoder on a ReLU would clamp every
        # negative pixel value to exactly 0. Order parameters here are
        # signed (~-a to +a), so the true output layer needs to be linear.
        up_channels = list(reversed(hidden_channels)) + [hidden_channels[0]]
        self.up_blocks = nn.ModuleList([
            UpBlock(up_channels[i], up_channels[i + 1], norm=norm)
            for i in range(n_stages)
        ])

        # Final output layer: plain linear conv, no norm, no activation.
        # This is the only thing standing between the network and the
        # pixel values it returns, so it must be able to output negatives.
        self.output_conv = nn.Conv2d(hidden_channels[0], out_channels, kernel_size=3, padding=1,
                                      padding_mode="circular")

    def forward(self, z: torch.Tensor, skips: list[torch.Tensor] | None = None) -> torch.Tensor:
        expected_shape = (self.latent_spatial_size, self.latent_spatial_size)
        if z.shape[-2:] != expected_shape:
            raise ValueError(f"Decoder expects a {expected_shape[0]}x{expected_shape[1]} "
                              f"latent map, got {tuple(z.shape[-2:])}")

        x = self.unbottleneck(z)

        # skips intentionally unused for now -- see class docstring.
        for up in self.up_blocks:
            x = up(x, skip=None)

        return self.output_conv(x)
