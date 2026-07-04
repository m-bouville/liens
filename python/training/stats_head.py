"""
Small dense network predicting real-space microstructure statistics
from the latent representation. Per docs/neural_nets.md: statistics are
auxiliary prediction targets, not recomputed live from x' or z-hat --
the loss just compares this head's predictions to precomputed targets
loaded from statistics.csv.
"""

import torch
import torch.nn as nn


class StatsHead(nn.Module):
    """
    latent (flattened) -> Linear(-> hidden) -> ReLU -> Linear(-> Ns).

    docs/neural_nets.md writes this as "Linear(1024 -> 128) -> ReLU ->
    Linear(128 -> Ns)", but 1024 there assumes latent_channels=16
    (16*8*8=1024). Input width is computed from latent_channels instead
    of hardcoded, since latent_channels=4 (confirmed to reconstruct well
    at 64x64) gives 4*8*8=256 -- a hardcoded 1024 would silently be wrong
    for any latent size other than the doc's original example.
    """

    def __init__(self, latent_channels: int, stat_names: list[str],
                 latent_spatial: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.stat_names = list(stat_names)
        in_dim = latent_channels * latent_spatial * latent_spatial

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, len(self.stat_names)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.flatten(start_dim=1)
        return self.net(z_flat)
