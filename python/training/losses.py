"""
Loss functions for autoencoder and latent-dynamics training.
"""

import numpy as np
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

    def _wrapped_diff(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        return diff

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = self._wrapped_diff(pred, target)
        return (diff ** 2).mean()

    def per_stat_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Same computation as forward(), but mean over the BATCH dim only,
        returning (n_stats,) instead of collapsing everything into one
        scalar -- lets a caller see which specific stats a given latent
        actually predicts well versus poorly, rather than only their
        blended average. Exists specifically because bulk-derived stats
        (e.g. avg_phi) and interface-derived ones can have very
        different answerability from a given stream (e.g. the deriv
        stream, which only ever sees interface motion) -- collapsing
        them into one number would hide exactly that distinction.
        """
        diff = self._wrapped_diff(pred, target)
        return (diff ** 2).mean(dim=tuple(range(diff.ndim - 1)))


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


class DtDecadeWeights:
    """
    Precomputed per-decade loss reweighting -- built ONCE from the full
    training set's own dt distribution AND its own raw per-transition
    loss (see compute_dt_decade_weights), then queried per-batch during
    training to produce a weights tensor matching that batch's own dt
    shape, for RolloutLoss's own `weights` parameter.

    Directly counteracts a training set where a small fraction of
    windows (extreme-dt ones) carry a large majority of the raw loss
    MASS -- confirmed empirically (~7% of windows carrying ~68% of the
    total loss, in one real measurement) -- by giving each dt DECADE
    equal total loss-mass contribution, rather than each WINDOW equal
    weight (today's implicit default, which lets whichever decade
    happens to have the largest raw errors dominate the gradient
    regardless of how few windows produce it).

    BUG THIS FIXES (previous version of this class): the first attempt
    at this weighting equalized each decade's WINDOW COUNT contribution
    -- weight_d = K / n_d, using only how many windows fall in decade
    d, never their actual loss magnitude. That is a DIFFERENT problem
    than the one being solved here, and conflating them made things
    worse, not better: in a real measurement, decade 4 (extreme dt) had
    both the FEWEST windows (220 of 3058, ~7% -- matching the original
    imbalance finding) AND the LARGEST per-window raw error (~123,277,
    vs ~49 in decade 1 -- a ~2500x difference). Weighting by 1/n_d alone
    gave that same decade the LARGEST weight of all four (its small
    count), which then multiplied an already-enormous per-window error
    by the biggest weight available -- actively amplifying the exact
    imbalance this class exists to correct, not fixing it (confirmed by
    a real run getting WORSE after the fix: Stage 3a's full/euler ratio
    rose to 321.4 from an unweighted 281.7, and Stage 3b's loss spikes
    were unchanged). Equalizing window-count contribution and
    equalizing loss-mass contribution are different problems; only the
    latter is what actually needs fixing here.

    CORRECTED formula: weight_d = K / (n_d * mean_loss_d), where
    mean_loss_d is the decade's own empirically-measured mean raw loss
    (see compute_dt_decade_weights for how it's measured) and K is
    chosen so the AVERAGE weight across all windows is 1 (keeps the
    overall loss scale roughly where it was before). This makes each
    decade's OWN total contribution to the weighted loss
    (n_d * weight_d * mean_loss_d = K) equal across every decade,
    regardless of how many windows populate it AND regardless of how
    large its raw per-window error is -- both count and magnitude are
    accounted for, not just one of them.

    Computed GLOBALLY (once, from the full training set), not
    per-batch: a per-batch version would give windows in a sparsely-
    represented decade an enormous individual weight whenever a given
    batch happens to draw few or none of them, reintroducing
    instability from a different angle -- exactly the problem this
    exists to fix, just relocated to a new source.
    """

    def __init__(self, all_dts: np.ndarray, all_losses: np.ndarray):
        if len(all_dts) == 0:
            raise ValueError("all_dts is empty -- cannot compute per-decade weights from no data.")
        if len(all_dts) != len(all_losses):
            raise ValueError(f"all_dts and all_losses must be the same length -- one raw loss "
                              f"value is required per dt (got {len(all_dts)} dts and "
                              f"{len(all_losses)} losses).")
        log_dt = np.log10(np.maximum(all_dts, 1e-12))
        decades = np.floor(log_dt).astype(np.int64)
        unique_decades, counts = np.unique(decades, return_counts=True)
        total_n = len(all_dts)

        # Empirical mean raw loss PER DECADE -- this, not window count
        # alone, is what the corrected weighting inverts. See this
        # class's own docstring for why count alone was the bug.
        mean_loss_per_decade: dict[int, float] = {}
        for d in unique_decades:
            decade_losses = all_losses[decades == d]
            mean_loss = float(decade_losses.mean())
            if not (mean_loss > 0):
                raise ValueError(f"decade {int(d)} has non-positive mean raw loss "
                                  f"({mean_loss}) -- cannot invert a zero/negative loss "
                                  f"mass into a finite weight.")
            mean_loss_per_decade[int(d)] = mean_loss
        counts_by_decade = {int(d): int(c) for d, c in zip(unique_decades, counts)}

        # K normalizes so the AVERAGE weight (over all windows, i.e.
        # weighted by count) is 1: sum_d(count_d * weight_d) = total_n,
        # with weight_d = K / (count_d * mean_loss_d) substituted in
        # gives sum_d(K / mean_loss_d) = total_n, i.e. K = total_n /
        # sum_d(1 / mean_loss_d). With this K, each decade's own total
        # weighted-loss contribution (count_d * weight_d * mean_loss_d)
        # collapses to exactly K for every decade -- equal loss mass
        # per decade, the actual goal.
        inv_mean_loss_sum = sum(1.0 / m for m in mean_loss_per_decade.values())
        K = total_n / inv_mean_loss_sum
        self.decade_weight: dict[int, float] = {
            d: K / (counts_by_decade[d] * mean_loss_per_decade[d]) for d in mean_loss_per_decade
        }
        self.min_decade = int(unique_decades.min())
        self.max_decade = int(unique_decades.max())

    def __call__(self, dt: torch.Tensor) -> torch.Tensor:
        """
        dt: any shape. Returns a weights tensor of the SAME shape,
        looking up each element's own decade against the global fit.

        Decades never seen during the original fit (e.g. a val/test dt
        that happens to fall in a decade absent from train) are CLAMPED
        to the nearest known decade's own weight, rather than raising
        or silently defaulting to weight=1 (which would inconsistently
        favor unseen-decade windows relative to their in-range
        neighbors).
        """
        log_dt = torch.log10(dt.clamp(min=1e-12))
        decades = torch.floor(log_dt).long()
        decades_clamped = decades.clamp(min=self.min_decade, max=self.max_decade)
        weights = torch.zeros_like(dt)
        for d, w in self.decade_weight.items():
            weights = torch.where(decades_clamped == d, torch.full_like(dt, w), weights)
        return weights


def compute_dt_decade_weights(all_dts: np.ndarray, all_losses: np.ndarray) -> DtDecadeWeights:
    """See DtDecadeWeights' own docstring, especially the "BUG THIS
    FIXES" section -- all_losses is not optional decoration, it's the
    quantity the whole scheme is actually built to invert. Typical
    usage (see training/train_lds.py's compute_euler_only_losses for
    how all_losses gets measured in practice -- a single pass with a
    freshly-initialized f_theta, BEFORE any real training):
        all_dts, all_losses = compute_euler_only_losses(f_theta, train_set, device)
        weights_fn = compute_dt_decade_weights(all_dts, all_losses)
        ...
        weights = weights_fn(dt_window)  # per batch, inside the training loop
        loss = rollout_loss(z_hat, z_true, weights=weights)
    """
    return DtDecadeWeights(all_dts, all_losses)


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

    exponent_deriv (q): reweights toward the RATE-space residual err =
    (z_hat-z_true)/dt, not the raw state-space diff compared here by
    default. diff = z_hat - z_true ALREADY equals dt*err for a single
    transition (LatentDynamics does explicit Euler integration --
    z(t+dt) = z(t) + dt*g_theta(z(t),theta) -- see its own docstring),
    so ||diff||^2 = dt^2 * err^2 ALREADY -- i.e. the DEFAULT
    (exponent_deriv=1.0) is exact backward compatibility with the
    original, dt-oblivious loss, not "no reweighting" in the sense of
    q=0. Reweighting expresses L_q = E[dt^(2q) * err^2] directly:
    q=1.0 (default): dt^0 * diff^2 = diff^2 -- today's unchanged loss.
    q=0.0: dt^-2 * diff^2 = err^2 -- fully dt-independent rate error.
    q=0.5: dt^-1 * diff^2 -- intermediate (variance of a Brownian
    increment scales as dt, so its std scales as sqrt(dt) -- matches
    this simulator's own explicit thermal noise term, see metadata.txt's
    own noise parameter, making q=0.5 a physically-motivated choice,
    not just an arbitrary compromise between 0 and 1).
    0/0.5/1 (not the algebraically-equivalent 0/1/2 on err^2 directly)
    specifically because the caller-facing quantity here is z (state),
    not the derivative itself -- q is more directly readable as "which
    power of dt does the RATE prediction get scaled by before squaring"
    this way, matching how the physical reasoning above is phrased too.

    weights (passed to forward(), not __init__ -- varies per BATCH, not
    fixed for the whole loss instance): per-window, per-step weights,
    same shape as dt -- (B, n_r). Independent of, and composable with,
    exponent_deriv above: exponent_deriv rescales each element's own
    loss MAGNITUDE based on its own dt (a physical rate-vs-state
    distinction); weights rescales each window's RELATIVE CONTRIBUTION
    to the batch average (a training-distribution correction -- see
    compute_dt_decade_weights, built specifically to counteract a small
    fraction of extreme-dt windows otherwise dominating the loss mass
    entirely, confirmed empirically: ~7% of windows carrying ~68% of
    the raw loss in one real measurement -- and see that function's own
    docstring for why the weighting must invert per-decade loss MASS,
    not window count alone). None (default) is a plain, unweighted mean
    -- today's unchanged behavior.

    Same mean-reduction rationale as ReconLoss/OneStepLoss: keeps the
    loss scale independent of latent_channels and of n_r (rollout
    length) itself, so switching from a 1-step to a 6-step rollout
    doesn't rescale the loss purely from summing more terms.
    """

    def __init__(self, kind: str = "l2", step_weights: torch.Tensor | None = None,
                 exponent_deriv: float = 1.0):
        super().__init__()
        if kind not in ("l1", "l2"):
            raise ValueError(f"kind must be 'l1' or 'l2', got '{kind}'")
        self.kind = kind
        self.exponent_deriv = exponent_deriv
        if step_weights is not None:
            self.register_buffer("step_weights", step_weights)
        else:
            self.step_weights = None

    def forward(self, z_hat: torch.Tensor, z_true: torch.Tensor, dt: torch.Tensor | None = None,
                weights: torch.Tensor | None = None, return_per_step: bool = False):
        """
        z_hat, z_true: (B, n_r, C, H, W) -- predicted (chained) and true
        continuations, NOT including the shared starting point z0 (i.e.
        LatentDynamics.rollout()'s output with index 0 already dropped).

        weights: (B, n_r) or None -- see this class's own docstring.

        return_per_step: if True, ALSO returns the (n_r,) per-step
        tensor (before averaging across steps) -- e.g. so a caller can
        read off per_step[0] as L_1step for direct comparison against a
        model trained with n_rollout_steps=1, without a second forward
        pass or recomputing this diff.
        """
        diff = z_hat - z_true
        if self.exponent_deriv != 1.0:
            if dt is None:
                raise ValueError("exponent_deriv != 1.0 requires dt to be given -- the "
                                  "per-transition dt used to convert the state-space diff "
                                  "back to a rate before reweighting.")
            # dt: (B, n_r) -> broadcast against diff's (B, n_r, C, H, W).
            dt_b = dt.view(*dt.shape, 1, 1, 1)
            if self.kind == "l2":
                per_element = diff.pow(2) * dt_b.pow(2 * self.exponent_deriv - 2)
            else:
                per_element = diff.abs() * dt_b.pow(self.exponent_deriv - 1)
        else:
            per_element = diff ** 2 if self.kind == "l2" else diff.abs()

        # Spatial mean only (dim 2,3,4) first -- (B, n_r) -- so `weights`
        # (also (B, n_r)) can be applied at exactly this point, before
        # collapsing the batch dimension. Splitting this reduction into
        # two stages (spatial, then batch) is what makes that possible;
        # previously both were done in a single .mean(dim=(0,2,3,4)) call.
        per_window_step = per_element.mean(dim=(2, 3, 4))  # (B, n_r)
        if weights is not None:
            per_step = (per_window_step * weights).sum(dim=0) / weights.sum(dim=0).clamp(min=1e-12)
        else:
            per_step = per_window_step.mean(dim=0)  # (n_r,) -- unchanged from before

        if self.step_weights is not None:
            w = self.step_weights
            loss = (per_step * w).sum() / w.sum()
        else:
            loss = per_step.mean()

        return (loss, per_step) if return_per_step else loss
