"""
Latent Dynamics Surrogate: f_theta predicts how the latent representation
evolves over time, without solving the discretized phase-field PDE.
"""

import torch
import torch.nn as nn


class LatentDynamics(nn.Module):
    """
    f_theta predicts a RATE g_theta(z, theta) = dz/dt, NOT a function of
    dt directly. The residual is then an explicit multiplication:
        z(t + dt) = z(t) + dt * g_theta(z(t), theta)
    -- forward-Euler integration of an autonomous (theta-conditioned)
    ODE dz/dt = g_theta(z, theta), rather than a generic dt-conditioned map.

    RATIONALE (dt vs theta are different in kind, not just different
    features): dt says how far to integrate a rate; theta changes the
    rate itself (a genuinely different dynamics field per temperature).
    Concatenating dt as a raw MLP input (an earlier version of this
    class did this) asks the network to LEARN the multiplicative
    relationship between dt and the state change from data alone, with
    no guarantee it generalizes to unseen dt values, and no guarantee
    of the physically necessary limit dz -> 0 as dt -> 0 (zero elapsed
    time must mean zero change; a concatenated-feature MLP has no
    structural reason to satisfy this unless it happens to learn it).
    The multiplicative structure here gets that limit exactly right by
    construction. theta, by contrast, IS appropriately concatenated as
    an input to the network computing g, since it changes the rate
    field's form, not just the integration length.

    Architecture: z is flattened ((B,C,8,8) -> (B,C*64)) and concatenated
    with theta before a small MLP, then reshaped back -- matching the
    "small dense net" style already used in stats_head.py, appropriate
    since z's spatial structure at an 8x8 resolution is small enough
    that a dense (non-convolutional) network can mix all of it directly.
    """

    def __init__(self, latent_channels: int, n_theta: int = 1,
                 latent_spatial: int = 8, hidden_dim: int = 256, n_hidden_layers: int = 2):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial

        flat_dim = latent_channels * latent_spatial * latent_spatial
        in_dim = flat_dim + n_theta  # z + theta ONLY -- dt is not a network input

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_dim, flat_dim))
        self.net = nn.Sequential(*layers)

        # Zero-init the final layer: g_theta(z,theta) = 0 at initialization,
        # for ANY z/theta, so dz = g*dt = 0 regardless of dt's scale. Your
        # dt spans ~4-5 orders of magnitude across a sweep (save_steps are
        # roughly geometrically spaced) -- an ordinarily-initialized (O(1))
        # rate multiplied by a large raw dt would otherwise start training
        # from a huge, meaninglessly-scaled loss (observed: ~5000 on a
        # single sample). Zero-init instead makes the untrained model
        # exactly a persistence forecast (z(t+dt) = z(t)), a physically
        # sensible starting point whose loss reflects actual state change,
        # not random-init noise amplified by dt.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def rate(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        g_theta(z, theta): predicted dz/dt, independent of dt itself.
        Exposed as its own method (not just inlined in forward) so it
        can be inspected directly later -- e.g. visualizing the learned
        rate field as a physical sanity check, or plugging into a
        higher-order integrator (RK2/RK4) instead of forward-Euler,
        both of which need the rate itself, not a dt-scaled residual.
        """
        batch_size = z.shape[0]
        z_flat = z.flatten(start_dim=1)
        x = torch.cat([z_flat, theta], dim=1)
        g_flat = self.net(x)
        return g_flat.view(batch_size, self.latent_channels, self.latent_spatial, self.latent_spatial)

    def forward(self, z: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        z: (B, C, 8, 8)
        dt: (B,) or (B, 1)
        theta: (B, n_theta)
        Returns dz = g_theta(z, theta) * dt: (B, C, 8, 8) -- the
        predicted residual, NOT z(t+dt) itself (caller still adds it).
        """
        g = self.rate(z, theta)
        dt = dt.view(-1, 1, 1, 1)  # broadcast against (B, C, 8, 8), works for (B,) or (B,1) input
        return g * dt

    def rollout(self, z0: torch.Tensor, dts: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Repeated application of forward(), for RolloutLoss (added once
        OneStepLoss training is confirmed working). Kept as a method
        here rather than reimplemented separately in training and
        evaluation code, per the original architecture plan. Unaffected
        by the rate-vs-concatenation change above, since forward()'s
        external signature is unchanged.

        z0: (B, C, 8, 8) starting latent, assumed exact (z0 = E(x(t_k)))
        dts: (B, n_steps) per-transition dt values
        theta: (B, n_theta), constant across the rollout (same run)

        Returns z_hat: (B, n_steps+1, C, 8, 8), with z_hat[:, 0] == z0
        exactly and every subsequent step predicted -- matching the
        docs' rollout notation where z_hat(t_k) = z(t_k) is the given
        starting point, not itself a prediction.
        """
        n_steps = dts.shape[1]
        z_hats = [z0]
        z = z0
        for i in range(n_steps):
            dz = self.forward(z, dts[:, i], theta)
            z = z + dz
            z_hats.append(z)
        return torch.stack(z_hats, dim=1)

    def forward_ab2(self, z_prev: torch.Tensor, z_curr: torch.Tensor,
                     dt_prev: torch.Tensor, dt_curr: torch.Tensor,
                     theta: torch.Tensor) -> torch.Tensor:
        """
        Second-order (Adams-Bashforth 2-step) prediction of z(t+dt_curr),
        using the SAME learned rate() evaluated at both the current
        state z(t) and the previous state z(t-dt_prev) -- no
        architecture change or retraining needed. This is a different
        NUMERICAL INTEGRATION SCHEME applied to the same learned rate
        field, exactly analogous to a classical ODE solver switching to
        a higher-order multistep method without changing its
        right-hand-side function.

        forward()/rollout() (forward-Euler) assume the rate is constant
        over the WHOLE interval [t, t+dt] -- a first-order approximation
        that degrades for large dt, and the likely explanation for the
        systematic underestimation of |dz| observed at large dt in
        check_rollout.py/check_parameter_dependence.py. AB2 instead fits a line
        through the previous and current rate estimates and integrates
        THAT forward, capturing whether the rate is increasing or
        decreasing rather than assuming it's flat -- formally
        second-order accurate for smooth dynamics.

        Standard non-uniform-step Adams-Bashforth 2-step formula
        (derived by fitting a line through (t-dt_prev, f_prev) and
        (t, f_curr), then integrating that line from t to t+dt_curr):
            z(t+dt_curr) = z(t) + dt_curr * [f_curr + (dt_curr / (2*dt_prev)) * (f_curr - f_prev)]

        Reduces to forward-Euler exactly when f_curr == f_prev (no trend
        to extrapolate) -- this is a strict enhancement of the same
        learned rate function, not a replacement, so it can be tried
        directly against an ALREADY-TRAINED model with no retraining,
        as a pure inference-time change.

        z_prev, z_curr: (B, C, 8, 8) -- states at t-dt_prev and t
        dt_prev, dt_curr: (B,) or (B, 1)
        theta: (B, n_theta), assumed the same at both points (same run)
        Returns z_pred: (B, C, 8, 8), the AB2 prediction of z(t+dt_curr).
        """
        f_prev = self.rate(z_prev, theta)
        f_curr = self.rate(z_curr, theta)

        dt_curr_r = dt_curr.view(-1, 1, 1, 1)
        dt_prev_r = dt_prev.view(-1, 1, 1, 1)

        dz = dt_curr_r * (f_curr + (dt_curr_r / (2 * dt_prev_r)) * (f_curr - f_prev))
        return z_curr + dz
