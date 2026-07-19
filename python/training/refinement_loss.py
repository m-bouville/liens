"""
Stage 4/5's combined training objective:
L_rollout/rollout_scale + recon0_weight*L_recon0/recon0_scale
+ stats0_weight*L_stats0/stats0_scale, computed from a RAW PIXEL window
(see MicrostructureEvolutionDataset's encoder=None mode) -- the encoder
is trainable here, so unlike stage 3, encoding happens fresh in this
function on every call, not once upfront.

*_scale terms are magnitude normalization, NOT importance weights (see
train_stage2's own docstring on this same distinction) -- default 1.0
(no-op), live in the params file's shared preamble.

One function covers both stages 4 and 5 (matching train_lds()'s 3a/3b
pattern of sharing one function across curriculum phases): stage 4 uses
rollout_weight=1, recon0_weight=small (an anchor, D frozen via
model_assembly's freeze_decoder); stage 5 uses recon0_weight=1,
rollout_weight=small (D trainable). Which term dominates is entirely a
caller-side weight choice, not a structural difference in this function.
"""
import torch

from models.latent_streams import DEFAULT_STREAM_NAME
from training.losses import ReconLoss, RolloutLoss, StatsLoss


def compute_stage45_loss(
    ae, f_theta, stats_head, x_window: torch.Tensor, dt_window: torch.Tensor,
    theta: torch.Tensor, rollout_weight: float = 1.0, recon0_weight: float = 0.0,
    stats0_weight: float = 0.0,
    rollout_scale: float = 1.0, recon0_scale: float = 1.0, stats0_scale: float = 1.0,
    exponent_deriv: float = 1.0,
    stats_loss_fn: StatsLoss | None = None,
    true_stats: torch.Tensor | None = None, return_components: bool = False,
    recon_stream_name: str = DEFAULT_STREAM_NAME,
):
    """
    x_window: (B, n_r+1, 1, ny, nx) raw pixel window -- x_window[:,0] is
    the real starting frame, x_window[:,1:] are the real continuation
    frames used as L_rollout's target.
    dt_window: (B, n_r). theta: (B, n_theta).

    true_stats: (B, n_stats) ground-truth statistics.csv values for the
    STARTING frame x_window[:,0] specifically -- L_stats, like stage
    1/2's, is always anchored to a real, observed frame, never a
    predicted one. stats_loss_fn: a StatsLoss instance (carries its own
    mean/std normalization buffers -- see that class). Both optional;
    if either is missing (or stats_head is None, e.g. the ancestor AE
    was trained with stats_weight<=0), L_stats is skipped entirely
    (reported as 0.0) rather than raising.

    freeze_decoder is NOT a parameter here -- it's a property of how the
    models were assembled (see model_assembly.build_models_from_components),
    not of this loss computation. Whatever gradient this function
    produces for D's parameters simply goes unused by the optimizer if
    D was frozen at assembly time; this function doesn't need to know.

    Returns just `total` by default, or (total, components_dict) if
    return_components=True -- components_dict includes the three raw
    (unweighted) loss values plus z0/z_true themselves, exposed
    specifically so the collapse-prevention detach below is directly,
    mechanistically testable (z_true.requires_grad should be False,
    z0.requires_grad should be True) rather than only checkable
    indirectly through gradient magnitudes.
    """
    batch_size = x_window.shape[0]
    n_rollout_steps = x_window.shape[1] - 1
    x0 = x_window[:, 0]
    x_future = x_window[:, 1:]

    # z0: WITH gradient -- feeds both the rollout PREDICTION chain
    # (z0 -> f_theta -> z_hat) and L_recon/L_stats, both anchored to
    # this real, observed starting frame.
    #
    # ae.encoder(x) returns dict[str, Tensor] (one entry per latent
    # stream -- see models/latent_streams.py); this function predates
    # the multi-stream (C0/C1) redesign and still only knows about the
    # single default stream, so it unwraps explicitly rather than
    # silently assuming a bare-tensor return. Will need revisiting once
    # this function itself is redesigned to use more than one stream.
    z0 = ae.encoder(x0)[recon_stream_name]

    # z_true: the L_rollout TARGET, built entirely under no_grad. This
    # is the critical collapse-prevention mechanism (discussed at
    # length before ever being implemented): without it, gradient could
    # flow into E via BOTH the prediction path and the target path
    # simultaneously, letting E trivially minimize L_rollout by
    # collapsing to a constant (with f_theta just learning to output
    # that same constant) rather than genuinely learning dynamics.
    # no_grad() (not a plain forward pass followed by .detach()) also
    # avoids building and immediately discarding a graph for these
    # frames, which matters given stage 4/5 already re-encodes every
    # frame fresh every epoch (see the dataset's own docstring on the
    # resulting cost-model change from stage 3).
    with torch.no_grad():
        x_future_flat = x_future.reshape(batch_size * n_rollout_steps, *x_future.shape[2:])
        z_true_flat = ae.encoder(x_future_flat)[recon_stream_name]
    z_true = z_true_flat.reshape(batch_size, n_rollout_steps, *z_true_flat.shape[1:])

    z_hat_full = f_theta.rollout(z0, dt_window, theta)
    z_hat = z_hat_full[:, 1:]

    rollout_loss_fn = RolloutLoss(exponent_deriv=exponent_deriv)
    l_rollout = rollout_loss_fn(z_hat, z_true, dt=dt_window)

    # L_recon: the real starting frame only, never a predicted frame --
    # matches stage 1/2's convention of anchoring recon/stats to
    # OBSERVED data.
    x0_recon = ae.decoder(z0)
    l_recon0 = ReconLoss()(x0_recon, x0)

    if stats_head is not None and stats_loss_fn is not None and true_stats is not None:
        pred_stats = stats_head(z0)
        l_stats0 = stats_loss_fn(pred_stats, true_stats)
    else:
        l_stats0 = torch.zeros((), device=x_window.device, dtype=x_window.dtype)

    total = (rollout_weight * l_rollout / rollout_scale + recon0_weight * l_recon0 / recon0_scale
             + stats0_weight * l_stats0 / stats0_scale)

    if return_components:
        components = {
            "rollout": l_rollout, "recon0": l_recon0, "stats0": l_stats0,
            "z0": z0, "z_true": z_true,
        }
        return total, components
    return total
