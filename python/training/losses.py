"""
Loss functions for autoencoder and latent-dynamics training.
"""

import torch
import torch.nn as nn


class ReconLoss(nn.Module):
    """
    Reconstruction loss: compares x' = D(E(x)) to x in real space.
    L2 (MSE) by default; L1 available if sharper interfaces are wanted
    (docs/neural_nets.md).

    NOTE: docs write L_recon = ||x' - x||_2^2, which literally means a
    summed squared-error norm. This uses PyTorch's default *mean*
    reduction instead (mean over batch x channels x H x W) -- sum would
    scale with image size (e.g. 16x larger for 256x256 vs 64x64), making
    loss weights (lambda_1, lambda_2 in later stages) resolution-dependent.
    Flagging this since it's a real deviation from the written formula,
    not just an implementation detail.
    """

    def __init__(self, kind: str = "l2"):
        super().__init__()
        if kind not in ("l1", "l2"):
            raise ValueError(f"kind must be 'l1' or 'l2', got '{kind}'")
        self.kind = kind
        self.loss_fn = nn.L1Loss() if kind == "l1" else nn.MSELoss()

    def forward(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(x_recon, x)


class StatsLoss(nn.Module):
    """
    Auxiliary loss: MSE between StatsHead's predictions (from the latent
    z) and precomputed real-space statistics loaded from statistics.csv.
    Per docs/neural_nets.md's "no live calculations" note, targets are
    never recomputed from x' or z-hat -- they're fixed ground truth.

    Per-stat normalization is required, not optional: statistics span
    wildly different raw scales (e.g. avg_phi has std~0.016 vs energy
    std~12.8 in a real run -- ~800x apart). Unweighted MSE would be
    dominated almost entirely by whichever stat has the largest raw
    magnitude, leaving near-zero gradient signal for the rest.

    mean/std should be computed once from the TRAIN split's dataset via
    train_set.stats_normalization() (train/val/test are now independent
    MicrostructureSnapshotDataset instances built from disjoint run
    directories -- see training/datasets.py's split_run_dirs -- so this
    is automatically correct with no risk of leaking val/test statistics
    into the normalization). Do not recompute per-batch, which would
    make the effective target drift as batch composition changes across
    training.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_norm = (target - self.mean) / self.std
        return nn.functional.mse_loss(pred, target_norm)
