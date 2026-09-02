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
from models.latent_dynamics import convert_derivative_coordinate


def compute_stage45_loss(
    ae, f_theta, stats_head, x_window: torch.Tensor, dt_window: torch.Tensor,
    theta: torch.Tensor, rollout_weight: float = 1.0, recon0_weight: float = 0.0,
    stats0_weight: float = 0.0,
    recon_predict_weight: float = 0.0,
    grad_predict_weight: float = 0.0,
    rollout_scale: float = 1.0, recon0_scale: float = 1.0, stats0_scale: float = 1.0,
    recon_predict_scale: float = 1.0,
    grad_predict_scale: float = 1.0,
    stats_loss_fn: StatsLoss | None = None,
    true_stats: torch.Tensor | None = None, return_components: bool = False,
    recon_stream_name: str = DEFAULT_STREAM_NAME, deriv_stream_name: str = "deriv",
    z1_resync: bool = True, t_window: torch.Tensor | None = None,
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

    # z0, z1: WITH gradient -- z0 feeds both the rollout PREDICTION
    # chain (z0 -> f_theta -> z_hat) and L_recon/L_stats, z1 feeds ONLY
    # the prediction chain (z1 is teacher-forced, never decoded --
    # matches Stage 3's own split-latent design), both anchored to this
    # real, observed starting frame.
    #
    # ae.encoders["shared"](x) returns dict[str, Tensor] (one entry per
    # latent stream -- see models/latent_streams.py). ae here is always
    # a MultiStreamAutoencoder -- the split-latent architecture requires
    # z1 (the "deriv" stream), which only exists on a multi-stream
    # model; a single-stream Autoencoder has no such stream to give.
    #
    # theta passed here too (this function's own parameter, previously
    # accepted but never actually forwarded into either encode call) --
    # needed because a theta-conditioned "deriv" stream requires it,
    # regardless of z0/z1_0 being the only entries this line keeps.
    x0_encoded = ae.encoders["shared"](x0, theta=theta)
    z0 = x0_encoded[recon_stream_name]
    z1_0 = x0_encoded[deriv_stream_name]

    # z_true, z1_future: the L_rollout TARGET and the teacher-forced z1
    # values for every step BEYOND the start, both built entirely under
    # no_grad. This is the critical collapse-prevention mechanism
    # (discussed at length before ever being implemented): without it,
    # gradient could flow into E via BOTH the prediction path and the
    # target path simultaneously, letting E trivially minimize
    # L_rollout by collapsing to a constant (with f_theta just learning
    # to output that same constant) rather than genuinely learning
    # dynamics. z1_future needs the SAME treatment as z_true, not
    # gradient like z1_0 -- it comes from the same real FUTURE frames
    # z_true does, so the identical collapse risk applies if left
    # with-gradient. no_grad() (not a plain forward pass followed by
    # .detach()) also avoids building and immediately discarding a
    # graph for these frames, which matters given stage 4/5 already
    # re-encodes every frame fresh every epoch (see the dataset's own
    # docstring on the resulting cost-model change from stage 3).
    with torch.no_grad():
        x_future_flat = x_future.reshape(batch_size * n_rollout_steps, *x_future.shape[2:])
        # theta expanded to match x_future_flat's own flattening
        # (batch_size*n_rollout_steps rows, sample b's own theta
        # repeated across its n_rollout_steps consecutive rows) -- theta
        # is constant PER SAMPLE across all its own rollout steps (same
        # run, same temperature), but the reshape above flattens the
        # (B, n_r) structure away, so theta needs the identical
        # expand-then-reshape to stay aligned row-for-row with it.
        theta_future_flat = theta.unsqueeze(1).expand(-1, n_rollout_steps, -1).reshape(
            batch_size * n_rollout_steps, -1)
        x_future_encoded = ae.encoders["shared"](x_future_flat, theta=theta_future_flat)
        z_true_flat = x_future_encoded[recon_stream_name]
        z1_future_flat = x_future_encoded[deriv_stream_name]
    z_true = z_true_flat.reshape(batch_size, n_rollout_steps, *z_true_flat.shape[1:])
    z1_future = z1_future_flat.reshape(batch_size, n_rollout_steps, *z1_future_flat.shape[1:])

    # z1_sequence: z1 at EVERY step, start through the last predicted
    # one -- exactly what rollout() teacher-forces at each step (see
    # LatentDynamics' own class docstring).
    z1_sequence = torch.cat([z1_0.unsqueeze(1), z1_future], dim=1)

    # z1_resync must match the regime f_theta was TRAINED in. Stage 3b can
    # train with z1_resync=False -- z1 propagated throughout, no reset at real
    # frames, matching inference -- and applying such an f_theta teacher-forced
    # is the same "NOT equivalent" direction that n_substeps N -> 1 is. Both
    # are inherited from the LDS checkpoint; this one was missed because the
    # rollout call sits in _refinement_loss.py rather than beside the model
    # construction in model_assembly.py.
    # u-scheme: a log10_t f_theta steps in Delta-u and consumes z̃1=dz0/du,
    # NOT physical dt and z1=dz0/dt. Convert both here, sourced from the
    # per-frame physical time t_window (step*sim_dt) the batch now carries.
    # ln10*t and log10(t ratio) match the dataset's own z̃1/Delta-u construction
    # EXACTLY (sim_dt cancels in the ratio). convert_derivative_coordinate is
    # the canonical t->log10_t definition (deriv * ln10 * t).
    if getattr(f_theta, "time_coordinate", "t") == "log10_t":
        if t_window is None:
            raise ValueError(
                "compute_stage45_loss got a log10_t f_theta but no t_window: "
                "the u-conversion z̃1=ln10*t*z1 needs per-frame physical time. "
                "Construct the stage-4 dataset with return_frame_t=True.")
        _t = t_window[:, :, None, None, None]          # (B, n_r+1, 1,1,1)
        z1_sequence = convert_derivative_coordinate(z1_sequence, _t, "t", "log10_t")
        dt_window = torch.log10(t_window[:, 1:] / t_window[:, :-1])   # (B, n_r) = Delta-u

    z_hat_full = f_theta.rollout(z0, z1_sequence, dt_window, theta,
                                  z1_resync=z1_resync)
    z_hat = z_hat_full[:, 1:]

    # No exponent_deriv here -- see train_lds.py's own identical
    # change: that reweighting assumed a plain diff=dt*err relationship
    # that no longer holds once the update rule is a mix of dt and
    # dt^2 terms; the split-latent architecture handles dt-scaling
    # structurally via the explicit z1*dt + f*(dt^2/2) terms
    # themselves, not a loss-level reweighting knob.
    rollout_loss_fn = RolloutLoss()
    l_rollout = rollout_loss_fn(z_hat, z_true)

    # L_recon: the real starting frame only, never a predicted frame --
    # matches stage 1/2's convention of anchoring recon/stats to
    # OBSERVED data. EncoderDecoderPair.forward() starts from a raw
    # image (encodes then decodes) -- z0 is already computed above, so
    # decode it directly instead, applying the output-scale correction
    # manually (matches check_reconstruction.py's own established
    # pattern for this same situation).
    recon_pathway = ae.pathways[recon_stream_name]
    x0_recon = recon_pathway.decoder(z0) * torch.exp(recon_pathway.log_output_scale)
    l_recon0 = ReconLoss()(x0_recon, x0)

    if stats_head is not None and stats_loss_fn is not None and true_stats is not None:
        pred_stats = stats_head(z0)
        l_stats0 = stats_loss_fn(pred_stats, true_stats)
    else:
        l_stats0 = torch.zeros((), device=x_window.device, dtype=x_window.dtype)

    # L_recon_predict: decode the FINAL rolled-out latent and grade it against
    # the real final frame IN PIXELS. This is the only term that closes the loop
    # on what is actually rendered at inference -- D(f_theta^n(E(x0))) vs the
    # true image at step n. L_rollout checks f_theta^n(E(x0)) only in LATENT
    # space (a proxy), and L_recon0 trains the decoder only on frame-0 latents;
    # neither asks the decoder to render a PREDICTED latent. Solely the last
    # step (not every step): it is the endpoint that gets decoded, it is the
    # most-drifted latent -- the one least protected by L_recon0's frame-0
    # training -- and one decode is cheaper than n. The decoder backprops
    # THROUGH the rollout here (z_hat carries grad), so this is also the term
    # that co-adapts encoder, f_theta and decoder toward the pixel endpoint.
    # Both endpoint terms grade the SAME decoded prediction against the SAME
    # real final frame, so decode once if either is active.
    if recon_predict_weight != 0.0 or grad_predict_weight != 0.0:
        x_pred_n = recon_pathway.decoder(z_hat[:, -1]) * torch.exp(
            recon_pathway.log_output_scale)
        x_real_n = x_future[:, -1]
    else:
        x_pred_n = x_real_n = None

    if recon_predict_weight != 0.0:
        l_recon_predict = ReconLoss()(x_pred_n, x_real_n)
    else:
        l_recon_predict = torch.zeros((), device=x_window.device, dtype=x_window.dtype)

    # L_grad_predict: the SPATIAL-GRADIENT sibling of L_recon_predict -- same
    # decoded endpoint, same real frame, but matched in first difference instead
    # of value. L_recon_predict is MSE on the field, which is blind to WHERE the
    # error sits: a decoder can lower it by spraying low-amplitude speckle across
    # the flat domain interiors (most of the pixels, each error tiny) while
    # keeping interfaces sharp -- the "moth-eaten" bulk seen in stage-5 rollouts.
    # Matching gradients penalizes exactly that: in the bulk grad(real)=0 so any
    # predicted variation is error, while at an interface grad(real) is large so
    # the term REQUIRES a matching sharp transition rather than blurring it. No
    # bulk mask needed -- the real field's own gradient says where variation is
    # allowed, and (since interface width is T-dependent in Allen-Cahn) the term
    # also teaches the physical interface profile, not a fixed target. Plain
    # non-periodic finite differences: the same operator on pred and real makes
    # it a fair match regardless of the boundary convention.
    if grad_predict_weight != 0.0:
        l_grad_predict = (
            ReconLoss()(x_pred_n[..., 1:, :] - x_pred_n[..., :-1, :],
                        x_real_n[..., 1:, :] - x_real_n[..., :-1, :])
            + ReconLoss()(x_pred_n[..., :, 1:] - x_pred_n[..., :, :-1],
                          x_real_n[..., :, 1:] - x_real_n[..., :, :-1]))
    else:
        l_grad_predict = torch.zeros((), device=x_window.device, dtype=x_window.dtype)

    total = (rollout_weight * l_rollout / rollout_scale + recon0_weight * l_recon0 / recon0_scale
             + stats0_weight * l_stats0 / stats0_scale
             + recon_predict_weight * l_recon_predict / recon_predict_scale
             + grad_predict_weight * l_grad_predict / grad_predict_scale)

    if return_components:
        components = {
            "rollout": l_rollout, "recon0": l_recon0, "stats0": l_stats0,
            "recon_predict": l_recon_predict, "grad_predict": l_grad_predict,
            "z0": z0, "z_true": z_true,
        }
        return total, components
    return total
