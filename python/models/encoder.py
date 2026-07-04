"""
Convolutional encoder: maps a real-space microstructure to a compact
latent representation.
"""

import math

import torch
import torch.nn as nn

from .blocks import DownBlock


class Encoder(nn.Module):
    """
    Repeated DownBlocks halving spatial resolution down to an 8x8
    bottleneck, then a 1x1 conv reducing to latent_channels. Depth is
    derived from input_size so 8x8 is reached exactly: 3 stages for
    64x64, 5 for 256x256, matching docs/neural_nets.md.

    Skip connections (the pre-downsampling features at each level) are
    always computed by DownBlock at negligible extra cost, but are only
    *returned* when use_skips=True. Defaults to False: the plumbing is
    here for when skip_connections.py is trained later (encoder frozen,
    decoder given skips) -- for now the encoder runs skip-free.
    """

    def __init__(
        self,
        input_size: int,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 16,
        norm: str = "batch",
        use_skips: bool = False,
    ):
        super().__init__()

        n_stages = math.log2(input_size / 8)
        if not n_stages.is_integer() or n_stages < 1:
            raise ValueError(
                f"input_size must be 8 * 2^k for integer k >= 1, got {input_size}"
            )
        n_stages = int(n_stages)

        self.input_size = input_size
        self.n_stages = n_stages
        self.use_skips = use_skips

        # channels[0] = input channels, channels[i] = output channels of stage i.
        # Doubling per stage is a starting choice, not dictated by the docs --
        # exposed via base_channels so it's easy to sweep.
        channels = [in_channels] + [base_channels * 2**i for i in range(n_stages)]
        self.channels = channels

        self.down_blocks = nn.ModuleList([
            DownBlock(channels[i], channels[i + 1], norm=norm)
            for i in range(n_stages)
        ])

        self.bottleneck = nn.Conv2d(channels[-1], latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            raise ValueError(
                f"Encoder built for input_size={self.input_size}, "
                f"got input of shape {tuple(x.shape[-2:])}"
            )

        skips = []
        for down in self.down_blocks:
            x, features = down(x)
            skips.append(features)

        z = self.bottleneck(x)

        if self.use_skips:
            return z, skips
        return z
