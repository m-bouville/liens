"""
Convolutional decoder: maps a latent representation back to a real-space
microstructure. Mirrors encoder.py's depth/channel schedule exactly.
"""

import math

import torch
import torch.nn as nn

from   models.blocks import UpBlock


class Decoder(nn.Module):
    """
    A 1x1 conv expands latent_channels back to the encoder's final
    channel count (spatial stays 8x8), then repeated UpBlocks double
    spatial resolution back up to output_size, mirroring Encoder's
    DownBlocks in reverse.

    IMPORTANT: output_size, base_channels, latent_channels, and norm
    must match the paired Encoder's construction arguments exactly, or
    shapes won't line up. This isn't enforced here (Decoder is built
    standalone) -- autoencoder.py should be the single place that
    constructs both from one shared config.

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
        norm: str = "batch",
        use_skips: bool = False,
    ):
        super().__init__()

        n_stages = math.log2(output_size / 8)
        if not n_stages.is_integer() or n_stages < 1:
            raise ValueError(
                f"output_size must be 8 * 2^k for integer k >= 1, got {output_size}"
            )
        n_stages = int(n_stages)

        self.output_size = output_size
        self.n_stages = n_stages
        self.use_skips = use_skips

        # Same formula as Encoder's channel list (index 0 = channel count
        # at full resolution, i.e. out_channels here vs in_channels there).
        channels = [out_channels] + [base_channels * 2**i for i in range(n_stages)]
        self.channels = channels
        reverse_channels = list(reversed(channels))  # coarse -> fine

        self.unbottleneck = nn.Conv2d(latent_channels, reverse_channels[0], kernel_size=1)

        self.up_blocks = nn.ModuleList([
            UpBlock(reverse_channels[i], reverse_channels[i + 1], norm=norm)
            for i in range(n_stages)
        ])

    def forward(self, z: torch.Tensor, skips: list[torch.Tensor] | None = None) -> torch.Tensor:
        if z.shape[-2:] != (8, 8):
            raise ValueError(f"Decoder expects an 8x8 latent map, got {tuple(z.shape[-2:])}")

        x = self.unbottleneck(z)

        # skips intentionally unused for now -- see class docstring.
        for up in self.up_blocks:
            x = up(x, skip=None)

        return x