"""
Latent Dynamics Surrogate: f_theta predicts how z0 (the state stream)
evolves over time, given z1 (the derivative stream) as a known input at
every step -- without solving the discretized phase-field PDE.
"""

import torch
import torch.nn as nn

from .constants import LATENT_SPATIAL_SIZE


class LatentDynamics(nn.Module):
    """
    Taylor expansion of z0 (the state stream):
        z0(t+dt) = z0(t) + z0_dot(t)*dt + z0_ddot(t)*(dt^2/2) + o(dt^2)

    z1 is TRAINED (stage 2, L_deriv) to approximate z0_dot -- so the
    first-order term above is z1(t)*dt, known WITHOUT this network at
    all. What's actually unknown is the curvature z0_ddot -- equivalently
    z1_dot, since z1 approximates z0_dot means d/dt of z1 IS z0_ddot,
    the same physical quantity under two names -- PLUS whatever gap
    exists between z1(t) and the true z0_dot(t) (z1 is only ever an
    approximation, trained with nonzero residual). f_theta's own target
    folds BOTH of those in automatically, rather than needing them
    disentangled into separate terms:
        f_theta(z0, z1, theta) trained against
            [z0(t+dt) - z0(t) - z1(t)*dt] / (dt^2/2)

    THIS CLASS ONLY PREDICTS z0. z1's own evolution (a similar Taylor
    expansion, z1(t+dt) = z1(t) + z1_dot(t)*dt + z1_ddot(t)*(dt^2/2) +
    o(dt^2), where z1_dot IS f_theta -- see above -- and a second
    network g_theta would approximate z1_ddot = f_theta_dot) is NOT YET
    IMPLEMENTED. Earlier versions of this class filled that gap with an
    ad hoc Euler step (z1_next = z1 + f_theta*dt) as a placeholder --
    that conflated testing f_theta's OWN accuracy with an unvalidated
    stand-in mechanism for g_theta, which was never a real design
    decision, just a way to have SOME z1 to chain forward with.

    Removed. Instead, z1 is TEACHER-FORCED: rollout() takes the REAL
    (encoder-provided, ground-truth) z1 value at EVERY step, not a
    predicted one -- a genuinely isolated test of f_theta alone, with
    zero dependence on g_theta or any placeholder for it. This is the
    correct way to validate/train f_theta on its own, per the "start
    with f, worry about training them together or one at a time once
    g_theta exists" plan -- g_theta, once built, is what will let z1's
    own rollout stop needing ground truth at every step and predict its
    own evolution instead.

    z0 and z1 MUST share the same channel count for z0(t) + z1(t)*dt
    to even be well-defined (they're being added directly) -- a real
    requirement of THIS architecture specifically, not just an
    incidental convenience. The codebase elsewhere (Stage 1b/2's own
    training) permits z0/z1 to differ in channel count; that
    flexibility does not carry over here.

    Architecture: z0 and z1 are each flattened ((B,C,8,8) -> (B,C*64))
    and concatenated with theta before a small MLP, then reshaped back
    -- matching the "small dense net" style already used in
    stats_head.py, appropriate since z's spatial structure at an 8x8
    resolution is small enough that a dense (non-convolutional) network
    can mix all of it directly. Single network now (not yet a shared
    trunk with two heads) -- that split becomes meaningful once
    g_theta actually exists to share the trunk with; premature with
    only one output to produce.
    """

    def __init__(self, latent_channels: int, n_theta: int = 1,
                 latent_spatial: int = LATENT_SPATIAL_SIZE, hidden_dim: int = 256,
                 n_hidden_layers: int = 2):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial

        flat_dim = latent_channels * latent_spatial * latent_spatial
        in_dim = 2 * flat_dim + n_theta  # z0 + z1 + theta -- dt is not a network input (see forward())

        # LeakyReLU, not ReLU: ReLU's own gradient is EXACTLY zero for
        # negative inputs, so a unit pushed sufficiently negative by any
        # single bad gradient event (e.g. one of the extreme, badly-
        # conditioned steps a chained-input rollout loss can produce --
        # see LatentDynamics' own docstring on why those inputs are
        # off-distribution) is dead FOREVER: zero gradient means no
        # future update can ever move it back, regardless of how much
        # more training runs. LeakyReLU's small negative_slope keeps a
        # nonzero gradient path on the negative side too, so a unit
        # driven deep negative can still recover in later training
        # rather than being gone for good. This isn't a precautionary
        # change -- check_dead_relus (evaluation/check_f_theta.py)
        # found a real, confirmed collapse in a trained checkpoint: its
        # SECOND hidden layer (the one feeding directly into the final,
        # output-producing Linear layer -- closest to the loss, least
        # diluted by backprop through other layers) was 100% dead,
        # while its first hidden layer stayed 0% dead. A fully dead
        # final hidden layer means f(z0,z1,theta) can only ever return
        # the final layer's own bias -- a fixed constant, regardless of
        # z0/z1/theta -- exactly what that checkpoint's own diagnostic
        # showed (zero-variance f() output across 3058 real test
        # windows). Swapping the activation fixes recoverability going
        # forward; it does not undo an already-dead checkpoint, and
        # does not by itself address whatever produced the extreme
        # gradient event in the first place.
        layers = [nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
        layers.append(nn.Linear(hidden_dim, flat_dim))
        self.net = nn.Sequential(*layers)

        # Zero-init the final layer: f(z0,z1,theta) = 0 at initialization,
        # for ANY z0/z1/theta -- so the untrained model reduces EXACTLY
        # to the pure Euler step z0(t+dt) = z0(t) + z1(t)*dt (trust
        # z1's own first-order estimate completely until training says
        # otherwise). A physically sensible starting point whose loss
        # reflects actual second-order state change, not random-init
        # noise amplified by dt^2.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def f(self, z0: torch.Tensor, z1: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        f_theta(z0, z1, theta): predicted curvature of z0 (equivalently,
        z1's own rate of change) -- independent of dt itself. Exposed
        as its own method (not just inlined in forward()) so it can be
        inspected directly -- e.g. visualizing the learned correction
        field as a physical sanity check, or comparing it against how
        z1 actually changes across a real trajectory to see whether the
        network is systematically correcting in one direction.
        """
        batch_size = z0.shape[0]
        x = torch.cat([z0.flatten(start_dim=1), z1.flatten(start_dim=1), theta], dim=1)
        f_flat = self.net(x)
        return f_flat.view(batch_size, self.latent_channels, self.latent_spatial, self.latent_spatial)

    def forward(self, z0: torch.Tensor, z1: torch.Tensor, dt: torch.Tensor,
                theta: torch.Tensor) -> torch.Tensor:
        """
        z0, z1: (B, C, 8, 8) -- z1 is the REAL value at this timestep
        (encoder-provided, not predicted -- see class docstring). MUST
        share the same C as z0.
        dt: (B,) or (B, 1)
        theta: (B, n_theta)

        Returns z0_next: (B, C, 8, 8) = z0 + z1*dt + f(z0,z1,theta)*(dt^2/2).
        ONLY z0 is predicted -- z1's own next value is the caller's
        responsibility to supply (real data during training/testing;
        g_theta's own job once it exists).
        """
        f_val = self.f(z0, z1, theta)
        dt_r = dt.view(-1, 1, 1, 1)  # broadcast against (B, C, 8, 8), works for (B,) or (B,1) input
        return z0 + z1 * dt_r + f_val * (dt_r ** 2 / 2)

    def rollout(self, z0: torch.Tensor, z1_sequence: torch.Tensor, dts: torch.Tensor,
                theta: torch.Tensor) -> torch.Tensor:
        """
        Repeated application of forward(), with z1 TEACHER-FORCED at
        its real (ground-truth) value at every step -- see class
        docstring for why. This is what makes f_theta directly
        testable/trainable without g_theta existing at all: the loss
        only ever measures f_theta's own accuracy at predicting z0's
        curvature, never contaminated by compounding error in a
        not-yet-implemented z1 prediction.

        z0: (B, C, 8, 8) starting latent, assumed exact (z0 = E(x(t_k))["state"])
        z1_sequence: (B, n_steps+1, C, 8, 8) -- REAL z1 at EVERY step,
        including the starting one (z1_sequence[:, 0] is z1(t_k), used
        for the first prediction; z1_sequence[:, i] for i>0 is the real
        z1 at the i-th subsequent step, used for that step's own
        prediction -- NOT a value this class ever predicts itself).
        dts: (B, n_steps) per-transition dt values
        theta: (B, n_theta), constant across the rollout (same run)

        Returns z0_hats: (B, n_steps+1, C, 8, 8), with z0_hats[:, 0] ==
        z0 exactly and every subsequent step predicted -- matching the
        docs' rollout notation where z_hat(t_k) = z(t_k) is the given
        starting point, not itself a prediction.
        """
        n_steps = dts.shape[1]
        z0_hats = [z0]
        z0_cur = z0
        for i in range(n_steps):
            z0_cur = self.forward(z0_cur, z1_sequence[:, i], dts[:, i], theta)
            z0_hats.append(z0_cur)
        return torch.stack(z0_hats, dim=1)
