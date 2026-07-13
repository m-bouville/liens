"""
Wraps a matched Encoder/Decoder pair built from one shared config, so
their shapes are guaranteed to line up.
"""

import torch
import torch.nn as nn

from .constants import LATENT_SPATIAL_SIZE
from .decoder import Decoder
from .encoder import Encoder
from .latent_streams import DEFAULT_STREAM_NAME, LatentStreamConfig, LatentStreamMode

_STREAM_NAME = DEFAULT_STREAM_NAME


class Autoencoder(nn.Module):
    """
    Constructs Encoder and Decoder together from a single set of
    hyperparameters -- unlike building them separately, where nothing
    enforces that input_size/output_size, base_channels, latent_channels,
    latent_spatial_size, and norm actually agree between the two.

    Deliberately the SIMPLE, single-stream case: same constructor
    signature as before the multi-stream (C0/C1) redesign -- a plain
    latent_channels: int, not a stream_configs dict. Internally this
    builds a single-entry stream_configs (one stream, named "state",
    mode=AUTOENCODER) and delegates to the same general Encoder every
    multi-stream model uses -- Autoencoder is the C0-only path (see the
    project's own design doc: C0 genuinely IS an autoencoder in the
    full sense; other streams, e.g. a time-derivative stream, are NOT,
    so they're never routed through this wrapper -- see
    latent_streams.decode_stream, used directly instead).

    forward() returns (x_recon, z): both are needed downstream (z feeds
    LDS training -- stage 3 onward, refined jointly with the encoder in
    stage 4/5; x_recon feeds the reconstruction loss), so returning
    just one and forcing a second call would waste compute. Both are
    the single "state" stream's own tensors -- Autoencoder never
    exposes the underlying Encoder's dict-of-streams return shape;
    call self.encoder directly for that.
    """

    def __init__(
        self,
        size: int,
        channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 16,
        latent_spatial_size: int = LATENT_SPATIAL_SIZE,
        norm: str = "batch",
        use_skips: bool = False,
    ):
        super().__init__()

        self.size = size
        self.channels = channels
        self.latent_channels = latent_channels
        self.latent_spatial_size = latent_spatial_size
        self.use_skips = use_skips

        stream_configs = {
            _STREAM_NAME: LatentStreamConfig(
                name=_STREAM_NAME, channels=latent_channels,
                spatial_size=latent_spatial_size, mode=LatentStreamMode.AUTOENCODER,
            )
        }

        self.encoder = Encoder(
            input_size=size,
            in_channels=channels,
            base_channels=base_channels,
            stream_configs=stream_configs,
            norm=norm,
            use_skips=use_skips,
        )
        self.decoder = Decoder(
            output_size=size,
            out_channels=channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            latent_spatial_size=latent_spatial_size,
            norm=norm,
            use_skips=use_skips,
        )

    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.use_skips:
            z, skips = z
            return z[_STREAM_NAME], skips
        return z[_STREAM_NAME]

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
