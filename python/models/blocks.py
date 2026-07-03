"""
Shared convolutional building blocks for the autoencoder's encoder and
decoder. Every resolution level, on either side, is built from these.
"""

import torch
import torch.nn as nn


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "layer":
        # GroupNorm(1, C) normalizes over (C, H, W) per sample -- the
        # standard stand-in for LayerNorm in conv nets, since nn.LayerNorm
        # itself expects a trailing channel dim rather than NCHW.
        return nn.GroupNorm(1, channels)
    raise ValueError(f"norm must be 'batch' or 'layer', got '{norm}'")


class ConvBlock(nn.Module):
    """
    Two padded 3x3 convolutions, each followed by normalization and ReLU.
    The basic building block at every resolution level: used inside
    DownBlock/UpBlock, and standalone for the decoder's optional
    post-skip refinement once skip connections are added.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    One encoder resolution level: ConvBlock at the current resolution,
    then a stride-2 conv that halves spatial size (64 -> 32 -> 16 -> 8, etc.)
    and hands channels to the next level.

    Returns (downsampled, features): features are the pre-downsampling
    activations at this resolution, kept for skip connections
    (skip_connections.py, added later and trained only after the encoder
    is frozen -- see docs/neural_nets.md). Ignore `features` until then.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch"):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, norm=norm)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        downsampled = self.down(features)
        return downsampled, features


class UpBlock(nn.Module):
    """
    One decoder resolution level, mirroring DownBlock: learned upsampling
    (transposed conv) that doubles spatial size, then a ConvBlock.

    Accepts an optional skip connection to be merged with the upsampled
    features before the ConvBlock. The merge is additive for now (simplest
    option that doesn't change channel counts); revisit once
    skip_connections.py defines how skip features are adapted to match.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = x + skip
        return self.conv(x)
