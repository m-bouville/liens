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

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, stat_names: list[str] | None = None):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        # angle (docs: local orientation, arctan(v1y/v1x)) is defined mod
        # pi -- an interface has no distinguishable "front" direction, so
        # e.g. 1.55 and -1.55 rad are nearly the SAME physical orientation
        # (both near the +-pi/2 wrap boundary), not ~pi apart. A naive
        # difference in the MSE would report a large, spurious error (and
        # gradient) for two predictions that are actually almost correct --
        # this directly corrupts encoder training via L_stats, not just a
        # diagnostic-script cosmetic issue. Wrapping the difference into
        # the equivalent normalized half-period fixes this; every other
        # stat is untouched.
        self.angle_idx = stat_names.index("angle") if stat_names and "angle" in stat_names else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_norm = (target - self.mean) / self.std
        diff = pred - target_norm
        if self.angle_idx is not None:
            # Period in NORMALIZED units is pi/std (normalization is a
            # linear rescale, which rescales the period too) -- wrap the
            # angle column's difference into (-period/2, period/2].
            period = torch.pi / self.std[self.angle_idx]
            angle_diff = diff[..., self.angle_idx]
            wrapped = ((angle_diff + period / 2) % period) - period / 2
            diff = diff.clone()
            diff[..., self.angle_idx] = wrapped
        return (diff ** 2).mean()


class OneStepLoss(nn.Module):
    """
    L_1step = || [z(t) + f_theta(z(t), dt)] - z(t+dt) ||_2^2 (docs/neural_nets.md),
    computed entirely in latent space -- this avoids relying on the
    decoder, cleanly separating decoder loss (ReconLoss) from prediction
    loss, per the docs' stated rationale.

    Deliberately agnostic to how z_next_pred was computed: this class
    does NOT call LatentDynamics itself, it just compares two tensors,
    mirroring ReconLoss/StatsLoss's (pred, target) signature. The
    training loop is responsible for computing
    z_next_pred = z_t + f_theta(z_t, dt) and passing both tensors in.

    Same mean-vs-sum reduction rationale as ReconLoss: the docs write a
    summed norm, but mean-reduction keeps the loss scale comparable
    across different latent_channels choices (e.g. 4 vs 16), which a
    sum would not (a larger latent would trivially inflate the summed
    loss with no change in per-element prediction quality).
    """

    def __init__(self, kind: str = "l2"):
        super().__init__()
        if kind not in ("l1", "l2"):
            raise ValueError(f"kind must be 'l1' or 'l2', got '{kind}'")
        self.kind = kind
        self.loss_fn = nn.L1Loss() if kind == "l1" else nn.MSELoss()

    def forward(self, z_next_pred: torch.Tensor, z_next_true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(z_next_pred, z_next_true)


class RolloutLoss(nn.Module):
    """
    L_rollout = sum_{i=1}^{Nr} || z_hat(t_{k+i}) - z(t_{k+i}) ||_2^2 (docs/neural_nets.md),
    where z_hat is generated by REPEATED application of f_theta
    (LatentDynamics.rollout), NOT by one-step ground-truth-conditioned
    predictions.

    This is the critical difference from OneStepLoss: OneStepLoss always
    resets to the true z(t) before predicting z(t+dt), so it never sees
    compounding drift -- a model can look good under OneStepLoss while
    still drifting badly over a real multi-step trajectory, since each
    one-step evaluation starts from a clean, correct state. RolloutLoss
    feeds the model's OWN (possibly imperfect) prediction back in as the
    next input, exactly as happens at real inference time, so error
    accumulation is part of what's being minimized, not hidden from it.

    step_weights: optional (n_r,) tensor to weight later steps more than
    earlier ones (docs: "perhaps weigh later predictions slightly more?
    long-term stability is what matters"). None (default) weights all
    steps equally.

    Same mean-reduction rationale as ReconLoss/OneStepLoss: keeps the
    loss scale independent of latent_channels and of n_r (rollout
    length) itself, so switching from a 1-step to a 6-step rollout
    doesn't rescale the loss purely from summing more terms.
    """

    def __init__(self, kind: str = "l2", step_weights: torch.Tensor | None = None):
        super().__init__()
        if kind not in ("l1", "l2"):
            raise ValueError(f"kind must be 'l1' or 'l2', got '{kind}'")
        self.kind = kind
        if step_weights is not None:
            self.register_buffer("step_weights", step_weights)
        else:
            self.step_weights = None

    def forward(self, z_hat: torch.Tensor, z_true: torch.Tensor,
                return_per_step: bool = False):
        """
        z_hat, z_true: (B, n_r, C, H, W) -- predicted (chained) and true
        continuations, NOT including the shared starting point z0 (i.e.
        LatentDynamics.rollout()'s output with index 0 already dropped).

        return_per_step: if True, ALSO returns the (n_r,) per-step
        tensor (before averaging across steps) -- e.g. so a caller can
        read off per_step[0] as L_1step for direct comparison against a
        model trained with n_rollout_steps=1, without a second forward
        pass or recomputing this diff.
        """
        diff = z_hat - z_true
        per_step = (diff ** 2).mean(dim=(0, 2, 3, 4)) if self.kind == "l2" \
            else diff.abs().mean(dim=(0, 2, 3, 4))  # (n_r,) -- mean over batch+spatial per step

        if self.step_weights is not None:
            w = self.step_weights
            loss = (per_step * w).sum() / w.sum()
        else:
            loss = per_step.mean()

        return (loss, per_step) if return_per_step else loss
