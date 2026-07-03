# python/models/autoencoder.py
"""
Wraps a matched Encoder/Decoder pair built from one shared config, so
their shapes are guaranteed to line up.
"""

import torch
import torch.nn as nn

from   models.decoder import Decoder
from   models.encoder import Encoder


class Autoencoder(nn.Module):
    """
    Constructs Encoder and Decoder together from a single set of
    hyperparameters -- unlike building them separately, where nothing
    enforces that input_size/output_size, base_channels, latent_channels,
    and norm actually agree between the two.

    forward() returns (x_recon, z): both are needed downstream (z feeds
    the LDS training in stage 4; x_recon feeds the reconstruction loss),
    so returning just one and forcing a second call would waste compute.
    """

    def __init__(
        self,
        size: int,
        channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 16,
        norm: str = "batch",
        use_skips: bool = False,
    ):
        super().__init__()

        self.size = size
        self.channels = channels
        self.latent_channels = latent_channels
        self.use_skips = use_skips

        self.encoder = Encoder(
            input_size=size,
            in_channels=channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            norm=norm,
            use_skips=use_skips,
        )
        self.decoder = Decoder(
            output_size=size,
            out_channels=channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            norm=norm,
            use_skips=use_skips,
        )

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor, skips: list[torch.Tensor] | None = None) -> torch.Tensor:
        return self.decoder(z, skips=skips)

    def forward(self, x: torch.Tensor):
        if self.use_skips:
            z, skips = self.encode(x)
            x_recon = self.decode(z, skips=skips)
        else:
            z = self.encode(x)
            x_recon = self.decode(z)
        return x_recon, z