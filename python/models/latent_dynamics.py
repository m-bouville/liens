"""
Latent Dynamics Surrogate: f_theta predicts how z0 (the state stream)
evolves over time, given z1 (the derivative stream) as a known input at
every step -- without solving the discretized phase-field PDE.
"""

from models.constants import N_THETA

import math

import torch
import torch.nn as nn

from .constants import LATENT_SPATIAL_SIZE


# EVERY field that changes what f_theta MEANS, in ONE place.
#
# These are the arguments that must survive a save/rebuild round trip: two
# checkpoints with identical weights and different values here are different
# models, not the same model configured differently. Architecture fields
# (latent_channels, hidden_dim, ...) are excluded because a mismatch there
# fails loudly at load_state_dict; these fail SILENTLY, by integrating with a
# step the weights were never fitted for.
#
# The list exists because it was got wrong the moment it grew. alpha was added
# to the checkpoint config and to train_lds's own resume, and NOT to
# model_assembly -- so stage 4 rebuilt an f_theta fitted at delta_t~36 and ran
# it ONE-SHOT at dt=500, skipping 325 of 325 batches by epoch 4. The comment
# beside n_substeps at that site already explained why exactly this must not
# happen; the field list was simply not enumerated anywhere, so a new field
# reached four of five sites.
_MEANING_FIELDS = {
    "dt_cap": float("inf"),   # inf is an exact no-op: pre-dt_cap checkpoints
    "n_substeps": 1,          # 1 is the historical default
    "alpha": None,            # None = fixed count, i.e. behaviour before alpha
    "max_substeps": 256,
    "dynamics_mode": "z1_taylor",  # historical form; "deriv_linear" changes the
                                   # update equation (f*dt, f sees dt) with the
                                   # SAME weights -- meaning-changing, so it must
                                   # round-trip like dt_cap/n_substeps.
    # truncate_bptt is deliberately ABSENT. Every other field here changes what
    # f_theta MEANS -- two checkpoints with the same weights and different
    # values integrate differently, so a rebuild must reproduce them. Truncation
    # changes only how the GRADIENT was computed during training; the forward
    # pass is bit-identical with it and without it, so a checkpoint rebuilt
    # without it evaluates exactly as it trained. Propagating it would also be
    # actively wrong for the diagnostics, which run under no_grad and would pay
    # detach calls for a graph they never build.
}


def integration_kwargs_from_config(config: dict) -> dict:
    """The integration settings a saved config implies, defaulted for age.

    Call this at EVERY site that rebuilds a LatentDynamics from a checkpoint
    (model_assembly, the check_* diagnostics, compare_*). Each default is the
    value that reproduces the behaviour of a checkpoint saved before that field
    existed, so an old checkpoint rebuilds exactly as it always did.

    Deliberately NOT a classmethod on LatentDynamics: the caller usually also
    needs the architecture fields from the same config, and mixing "read the
    config" into the constructor would make the model responsible for a file
    format it never writes.
    """
    return {name: config.get(name, default) for name, default in _MEANING_FIELDS.items()}


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

    def __init__(self, latent_channels: int, n_theta: int = N_THETA,
                 latent_spatial: int = LATENT_SPATIAL_SIZE, hidden_dim: int = 256,
                 n_hidden_layers: int = 2, dt_cap: float = float("inf"),
                 n_substeps: int = 1, alpha: float | None = None,
                 max_substeps: int = 256, truncate_bptt: int | None = None,
                 dynamics_mode: str = "z1_taylor"):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial
        # n_substeps: how many integration steps rollout() takes BETWEEN two
        # real frames. 1 is the historical behaviour exactly -- one step of
        # the full dt, which is what forward() does and what stage 3a uses.
        #
        # It lives on the MODEL (like dt_cap, and unlike a training-loop
        # policy) because it changes what f_theta MEANS. At n_substeps=1 the
        # exact target is [z0(t+dt) - z0 - z1*dt]*2/dt^2, the AVERAGE
        # curvature over the interval -- a quantity that depends on dt, which
        # f_theta cannot see. As n_substeps grows that target converges to the
        # pointwise z1_dot the design doc defines f_theta to be. Two f_theta
        # checkpoints with the same shapes can therefore mean different
        # things, which is why this is checkpointed and cross-checked on
        # resume exactly as dt_cap is.
        if n_substeps < 1:
            raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")
        self.n_substeps = int(n_substeps)
        # `alpha is not None` joins the condition: an ADAPTIVE model sub-steps
        # too, so the same clash applies -- and less predictably, because h is
        # derived per sample per transition. Any sample whose h exceeds the cap
        # gets its quadratic term truncated while its z1 trapezoid runs on the
        # UNCAPPED h; whether that happens depends on |f|/|z1| at that state,
        # so the inconsistency comes and goes within a batch. Found by review:
        # the original condition predates alpha and could not see it, so an
        # adaptive model with a finite cap warned about nothing.
        if (self.n_substeps > 1 or alpha is not None) and math.isfinite(dt_cap):
            # dt_cap and sub-stepping are two answers to ONE question: how to
            # stop f*(dt^2/2) dominating at large dt. The cap truncates the
            # term; sub-stepping removes the large dt. Using both makes an
            # n_substeps sweep measure their interaction rather than
            # integration accuracy -- and the interaction is violent, because
            # the cap bounds z0's second-order term while z1 still evolves on
            # the UNCAPPED h. Measured at dt=2500, dt_cap=125, f=1: the total
            # f contribution goes 7.8e3 (N=1) -> 1.6e6 (N=2) -> 3.1e6 (N=32,
            # where h < cap and the cap is finally inert). A 400x spread that
            # is entirely an artifact of the two mechanisms fighting.
            #
            # Not an error: a cap set BELOW every h is harmless, and refusing
            # would be over-strict. But it must be visible, because the
            # symptom is a plausible-looking sweep with a meaningless trend.
            print(f"WARNING: n_substeps={self.n_substeps} together with a finite dt_cap="
                  f"{dt_cap}. These compensate for the SAME thing, and any dt where "
                  f"dt/n_substeps > dt_cap will have its second-order term truncated while z1 "
                  f"keeps evolving on the uncapped step -- so an n_substeps sweep measures the "
                  f"cap, not the integration. Use dt_cap=inf when sweeping n_substeps.")
        # dt_cap: caps dt ITSELF inside the second-order term only
        # (f_val*(dt^2/2) becomes f_val*(min(dt,dt_cap)^2/2)), NOT the
        # first-order z1*dt term, which keeps growing unbounded. Default
        # inf is an exact no-op (min(dt,inf)=dt always) -- every existing
        # caller that doesn't know about this parameter gets identical
        # behavior to before it existed.
        #
        # Why cap dt here rather than cap f_val's own output: the two
        # terms are |z1|*dt (first order) and |f_val|*(dt^2/2) (second
        # order) -- even a SATURATED f_val (bounded to some f_max) still
        # has a term that grows as dt^2, which inevitably overtakes a
        # term growing only as dt for large enough dt (crossover at
        # dt*=2|z1|/f_max, always finite for any f_max>0 -- saturating
        # f_val only pushes that crossover further out, never eliminates
        # it). Capping dt here instead makes the second-order term's own
        # growth stop entirely past dt_cap, while z1*dt keeps growing --
        # guaranteeing the first-order term eventually dominates again
        # for dt > roughly 2|z1|*dt_cap/f_max, an actual structural
        # reversal back toward the correct euler-dominated large-dt
        # regime, not just a delay of the point where it breaks down.
        self.dt_cap = dt_cap
        # STEP A (correction order). "z1_taylor" is the historical form
        # exactly. "deriv_linear" lets f own its dt-scaling: f takes dt as an
        # input and the second-order term uses a LINEAR prefactor (f*dt), not
        # f*dt^2/2. dt_cap exists ONLY to contain the dt^2 explosion, so it is
        # FORBIDDEN in deriv_linear -- asserted, not silently coerced, so a
        # stale params file cannot quietly reintroduce the containment this
        # mode is measuring without. Remove the guard only once proven.
        if dynamics_mode not in ("z1_taylor", "deriv_linear"):
            raise ValueError(
                f"dynamics_mode must be 'z1_taylor' or 'deriv_linear', got {dynamics_mode!r}")
        self.dynamics_mode = dynamics_mode
        if dynamics_mode == "deriv_linear" and math.isfinite(dt_cap):
            raise ValueError(
                f"dynamics_mode='deriv_linear' forbids a finite dt_cap (got {dt_cap}): "
                "the linear-dt prefactor has no dt^2 term for the cap to contain, and "
                "keeping it would confound the order measurement. Set dt_cap=inf.")
        if dynamics_mode == "deriv_linear" and (alpha is not None or n_substeps != 1):
            # deriv_linear is a FULL-STEP object: forward() implements its
            # linear update (z0 + z1*dt + f(.,dt)*dt), but the sub-stepping
            # integrator (_integrate) implements the OLD Taylor form and would
            # NOT be the linear update even if reached. Sub-stepping is also the
            # other lever on large-dt, so allowing it here would confound the
            # order change this mode exists to measure. Forbid it -- rollout's
            # fast path (n_substeps==1, alpha None, z1_resync) routes through
            # forward(), which is the only place deriv_linear is defined.
            raise ValueError(
                f"dynamics_mode='deriv_linear' requires n_substeps=1 and alpha=None "
                f"(full-step; sub-stepping runs the Taylor integrator, not the linear "
                f"update), got n_substeps={n_substeps}, alpha={alpha}.")

        # alpha: the TAYLOR-VALIDITY RATIO that replaces n_substeps as the
        # thing held constant. Every sub-step contributes a linear term
        # z1*delta_t and a curvature correction f_theta*delta_t^2/2; alpha
        # bounds the second as a fraction of the first,
        #
        #     |f_theta|*delta_t / |z1|  <=  alpha
        #
        # so the step follows the local dynamics instead of the save schedule.
        # Solved for the count, that is
        #
        #     n = ceil(|f_theta|*dt / (alpha*|z1|))
        #
        # which is the SAME equation n_substeps expresses, read the other way:
        # n_substeps fixes the step and lets alpha fall where it may, alpha
        # fixes the validity and lets the step follow. Measured on the
        # 128x128 sweep (see evaluation/check_alpha.py), fixing n_substeps=14
        # gave a median alpha of 0.105 and a MAX of 1.07 -- the worst sub-steps
        # had a "correction" as large as the displacement it corrected, i.e.
        # outside Taylor validity entirely, while the typical window was
        # over-resolved tenfold to make those survivable.
        #
        # Scale-free by construction: the ratio form has no units, so it does
        # not need retuning when max_dt moves. It also handles both degenerate
        # ends correctly. |f_theta| -> 0 asks for an unbounded step, which is
        # right (no curvature means linear extrapolation is exact) and is the
        # state of every fresh stage 3a, since f's final layer is zero-init --
        # an ABSOLUTE tolerance on |f|*delta_t^2 would divide by zero there.
        # |z1| -> 0 asks for delta_t -> 0, guarded by max_substeps below.
        #
        # None (the default) keeps the FIXED n_substeps behaviour exactly, so
        # every existing checkpoint and caller is unaffected.
        if alpha is not None and not (alpha > 0):
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if alpha is not None and self.n_substeps != 1:
            # Both set means two answers to one question -- the same clash
            # dt_cap and n_substeps have, and it must be an ERROR rather than
            # a warning here because there is no reading under which it does
            # something sensible: the count would be adaptive AND multiplied.
            raise ValueError(
                f"alpha={alpha} and n_substeps={self.n_substeps} both set. alpha REPLACES "
                f"n_substeps (it derives the count per transition); leave n_substeps at its "
                f"default of 1 when using alpha."
            )
        self.alpha = alpha
        # max_substeps: a COST bound, not a stability one -- the criterion
        # itself is unbounded when |z1| -> 0 (a state with no velocity and
        # nonzero curvature admits no valid step at all), and one such window
        # must not stall a batch forever. Hit only in the degenerate corner;
        # if it binds routinely, that is a finding about the data, so
        # rollout() counts and reports it rather than silently clamping.
        # TRUNCATED BPTT: detach the sub-step graph every k steps. None (the
        # default) keeps the graph whole, so every existing run is unchanged.
        # See _integrate's loop for what it buys and what it costs.
        if truncate_bptt is not None and truncate_bptt < 2:
            # >= 2, NOT >= 1. At k=1 the output carries NO gradient at all:
            # each sub-step's z0 update consumes the f_n carried in from the
            # PREVIOUS step (that is the recycling that keeps the scheme at one
            # f evaluation per step), so if every step detaches, the final
            # z0's only parameter-dependent term is a detached f_n and
            # backward() raises "element 0 does not require grad". A segment
            # needs at least one f evaluation that both lives inside it and
            # feeds a later z0 update, which takes two steps.
            raise ValueError(
                f"truncate_bptt must be >= 2 or None, got {truncate_bptt}. "
                f"At 1 the recycled f_n is detached before it ever reaches a z0 "
                f"update, leaving the output with no gradient path; None keeps "
                f"the full chain."
            )
        self.truncate_bptt = truncate_bptt
        # max_substeps is the CLAMP on the alpha-derived count and is read ONLY
        # in the adaptive path (alpha set). In fixed mode (alpha is None,
        # n_substeps drives everything) it is never used, so None is legal
        # there and means "no adaptive clamp -- there is nothing to clamp".
        # Requiring a positive int unconditionally crashed a fixed-mode run
        # that passed max_substeps=None, before the substepping logic ran.
        if alpha is not None:
            if max_substeps is None or max_substeps < 1:
                raise ValueError(
                    f"max_substeps must be >= 1 when alpha is set (adaptive "
                    f"substepping clamps the derived count at it), got "
                    f"{max_substeps}")
            self.max_substeps = int(max_substeps)
        else:
            # Fixed mode: keep an int for any incidental use, defaulting when
            # None so nothing downstream sees NoneType, but it is not consulted
            # by the fixed-count path.
            self.max_substeps = int(max_substeps) if max_substeps is not None else 1
        self.n_substeps_clamped = 0
        # Running totals for the epoch report. The whole argument for alpha
        # over a fixed count is that the step ADAPTS as f_theta sharpens --
        # and that claim is unobservable unless the realised count is
        # reported. A fixed n_substeps appears in the log because it is a
        # parameter; a derived one appears nowhere unless it is measured.
        self._substep_total = 0
        self._substep_batches = 0
        self._substep_max = 0
        self._retained_peak = 0.0
        # Raw realised counts of recent transitions, for the per-batch memory
        # diagnostic. Cleared by retained_peak(reset=True).
        self._last_counts: list = []

        flat_dim = latent_channels * latent_spatial * latent_spatial
        # deriv_linear feeds log(dt) so f can pick its own order; z1_taylor
        # keeps f dt-blind (its dt-dependence is imposed by the dt^2/2 form).
        in_dim = 2 * flat_dim + n_theta + (1 if dynamics_mode == "deriv_linear" else 0)

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

    def f(self, z0: torch.Tensor, z1: torch.Tensor, theta: torch.Tensor,
          dt: torch.Tensor | None = None) -> torch.Tensor:
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
        parts = [z0.flatten(start_dim=1), z1.flatten(start_dim=1), theta]
        if self.dynamics_mode == "deriv_linear":
            # log(dt) as the order-selection input. REQUIRED here -- the net's
            # in_dim includes it, so a direct f() call (e.g. check_f_theta,
            # OUTSIDE forward()) that omits dt is a silent shape error, which
            # is exactly the trap flagged for this change: pass the same dt
            # forward() would.
            if dt is None:
                raise ValueError(
                    "f() requires dt in dynamics_mode='deriv_linear' (it is a "
                    "network input); a direct caller must pass the transition's dt.")
            dt_flat = dt.reshape(batch_size, 1)
            parts.append(torch.log(dt_flat))
        x = torch.cat(parts, dim=1)
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

        Returns z0_next: (B, C, 8, 8) = z0 + z1*dt + f(z0,z1,theta)*(min(dt,dt_cap)^2/2).
        The dt_cap clamp applies ONLY inside the second-order term, not
        to the z1*dt term -- see __init__'s own docstring for why capping
        dt here (rather than saturating f_val's own output) is what
        actually guarantees the first-order term eventually dominates
        again at large dt, rather than merely delaying when the
        second-order term overtakes it. self.dt_cap defaults to inf, an
        exact no-op recovering the original, uncapped formula.
        ONLY z0 is predicted -- z1's own next value is the caller's
        responsibility to supply (real data during training/testing;
        g_theta's own job once it exists).
        """
        dt_r = dt.view(-1, 1, 1, 1)  # broadcast against (B, C, 8, 8), works for (B,) or (B,1) input
        if self.dynamics_mode == "deriv_linear":
            # f owns its dt-scaling: LINEAR prefactor, f conditioned on dt.
            # No dt_cap (forbidden in __init__), so no dt_capped term here.
            f_val = self.f(z0, z1, theta, dt=dt_r)
            return z0 + z1 * dt_r + f_val * dt_r
        f_val = self.f(z0, z1, theta)
        dt_capped = torch.clamp(dt_r, max=self.dt_cap)
        return z0 + z1 * dt_r + f_val * (dt_capped ** 2 / 2)

    @property
    def supports_autonomous_rollout(self) -> bool:
        """Can rollout() propagate z1 from its own predictions (z1_resync=False)?

        False for deriv_linear: its f is a derivative CORRECTION, not z0's
        curvature, so there is no z1-update equation to advance z1 with -- only
        the teacher-forced path (z1_resync=True, via forward()) is defined.
        z1_taylor advances z1 by f*dt inside _integrate, so it can. A diagnostic
        should read this before asking for an autonomous rollout rather than
        discover the limitation via the guard in rollout()/_integrate.
        """
        return self.dynamics_mode != "deriv_linear"

    def _substeps_for(self, z0: torch.Tensor, z1: torch.Tensor, dt: torch.Tensor,
                       theta: torch.Tensor, f_n: torch.Tensor) -> torch.Tensor:
        """Per-sample sub-step count from the alpha criterion, as a (B,) long tensor.

        n = ceil(|f_theta|*dt / (alpha*|z1|)), floored at 1 and capped at
        max_substeps. Norms over the WHOLE latent tensor, not per channel:
        alpha describes ONE step taken for the whole state, and a per-channel
        ratio would be dominated by whichever channel happens to sit near zero
        -- a property of that channel, not of the step.

        DETACHED, and the guarantee is STRUCTURAL rather than a matter of
        discipline: the return is a long tensor, and integer tensors cannot
        carry gradients, so no path from the count back into f can exist
        however this body is later rewritten. That is what protects against
        the real hazard -- if the count depended differentiably on |f_theta|,
        gradient descent would discover that shrinking f earns longer steps
        and lowers the loss that way, optimising the integrator instead of the
        physics.

        The surrounding no_grad is therefore a MEMORY optimisation, not the
        correctness guarantee: without it the norms would build a graph that
        the .long() conversion immediately orphans. Worth keeping (the graph
        is per-transition and the rollout is a loop), worth not mistaking for
        the safety property. Removing it changes nothing observable, which is
        exactly why a test asserting "the count has no grad_fn" passes against
        its removal and proves nothing.

        Uses the f_n already computed at the transition's start, so the
        criterion costs no extra evaluation. It is therefore a PREDICTOR: it
        cannot see |f_theta| rising within the transition, which is the
        dangerous direction. rollout() re-evaluates it per transition rather
        than once per window for exactly that reason.
        """
        with torch.no_grad():
            b = z0.shape[0]
            f_norm = torch.linalg.vector_norm(f_n.reshape(b, -1), dim=1)
            z1_norm = torch.linalg.vector_norm(z1.reshape(b, -1), dim=1)
            dt_flat = dt.reshape(b).abs()
            # 0/0 (no velocity AND no curvature) is a DEAD state: nothing is
            # happening, so one step suffices. Distinguished from |z1|=0 with
            # |f|>0, which genuinely admits no valid step and is what
            # max_substeps then bounds.
            raw = torch.where(
                f_norm > 0,
                f_norm * dt_flat / (self.alpha * z1_norm.clamp_min(torch.finfo(z1_norm.dtype).tiny)),
                torch.zeros_like(f_norm),
            )
            n = torch.ceil(raw).clamp(min=1.0, max=float(self.max_substeps))
            n = torch.nan_to_num(n, nan=float(self.max_substeps),
                                  posinf=float(self.max_substeps))
            self.n_substeps_clamped += int((raw > self.max_substeps).sum())
            self._substep_total += int(n.sum())
            self._substep_batches += int(n.numel())
            self._substep_max = max(self._substep_max, int(n.max()))
            # REALISED retained cost of this transition, in sample-substeps --
            # the same quantity BudgetedBatchSampler budgets on, but computed
            # from the counts the integrator ACTUALLY takes rather than from
            # the pre-epoch estimate. The two diverge because the estimate uses
            # each window's first state and worst transition, while the counts
            # are re-derived per transition from the state actually reached;
            # with bucket_refresh_epochs=0 nothing ever corrects that drift.
            # Reporting max(realised)/budget is what says whether the budget is
            # bounding memory at all.
            return n.long()

    def retained_peak(self, reset: bool = False) -> float:
        """Largest REALISED retained cost (sample-substeps) since the reset.

        Separate from substep_stats because that returns None when no adaptive
        transition ran, while this is meaningful whenever truncation is on --
        and because the caller that needs it (the budget calibration) must not
        clear the sub-step statistics as a side effect of reading it.
        """
        value = self._retained_peak
        if reset:
            self._retained_peak = 0.0
            self._last_counts = []
        return value

    def last_counts(self):
        """Realised per-sample counts of the transitions since the last
        retained_peak(reset=True) -- one tensor per _integrate call. For the
        per-batch memory diagnostic; empty on the fixed-n path."""
        return list(self._last_counts)

    def substep_stats(self, reset: bool = True) -> dict | None:
        """Mean/max realised sub-steps since the last reset, or None if fixed.

        None rather than a dict of the constant when alpha is unset: a caller
        printing "mean 7.0 max 7" every epoch for a fixed n_substeps=7 is
        noise, and the distinction between a derived and a declared count is
        exactly what the report is for.

        ONE condition, not two. The counters advance only inside
        _substeps_for, which runs only when alpha is set -- so an empty
        counter IS the fixed-count case, and an explicit `alpha is None` test
        beside it is redundant. Verified by removing it: nothing changed,
        which is the definition of dead code rather than a missing test.
        """
        if self._substep_batches == 0:
            # The retained peak is meaningful even with a FIXED n_substeps
            # (truncation still applies), and reset must still clear it --
            # otherwise a fixed-n run accumulates a cumulative maximum for the
            # whole run and the per-epoch drift it exists to show is lost.
            if reset:
                self._retained_peak = 0.0
            return None
        stats = {"mean": self._substep_total / self._substep_batches,
                 "max": self._substep_max, "clamped": self.n_substeps_clamped,
                 "transitions": self._substep_batches,
                 "retained_peak": self._retained_peak}
        if reset:
            # Per-epoch, not cumulative -- a cumulative mean would flatten the
            # drift this exists to show. n_substeps_clamped is deliberately
            # NOT reset: a clamp that bound once in a run is worth carrying.
            self._substep_total = self._substep_batches = self._substep_max = 0
            self._retained_peak = 0.0
        return stats

    def _integrate(self, z0: torch.Tensor, z1: torch.Tensor, dt: torch.Tensor,
                    theta: torch.Tensor, f_carry: torch.Tensor | None = None):
        """
        Advance (z0, z1) across ONE real transition of `dt`, in sub-steps of
        delta_t = dt / n -- with n either the fixed self.n_substeps or, when
        self.alpha is set, derived per sample from the alpha criterion (see
        _substeps_for).

        Returns (z0_next, z1_next, f_next). f_next is the f_theta value at the
        arrival state, carried into the following transition so that each
        sub-step costs exactly ONE f_theta evaluation.

        SCHEME -- semi-implicit (predictor-corrector) velocity Verlet:

            z0_{n+1} = z0_n + z1_n*h + f_n*h^2/2
            z1~      = z1_n + f_n*h                        (predictor, Euler)
            f_{n+1}  = f_theta(z0_{n+1}, z1~, theta)
            z1_{n+1} = z1_n + (f_n + f_{n+1})*h/2          (corrector, trapezoid)

        Semi-implicit, not implicit: z1_{n+1} appears inside f_{n+1}, and that
        dependence is resolved by the Euler predictor rather than by a solve.
        Nothing is evaluated at a state that has not yet been computed, so the
        scheme is causal and runs unchanged at inference, where no future frame
        exists.

        WHY THE TRAPEZOIDAL z1 UPDATE, and not z1 += f*h. The one-sided Euler
        update is FIRST order, and it drags the whole scheme down with it even
        though z0's own update is second order: z1's O(h) error feeds back
        through the z1*h term. Measured orders of convergence:

            z1 += f*h                   1.02
            z1 += (f_n + f_{n+1})*h/2   2.00

        This is also what makes g_theta unnecessary HERE. g_theta approximates
        z1_ddot and would buy a third-order z1 update; a centred average of
        f_theta already recovers second order using f_theta alone. Note the
        parallel with derivative_estimators.md: there a centred stencil beat a
        one-sided one for ESTIMATING z0_dot; here a centred average beats a
        one-sided evaluation for PROPAGATING z1. Same asymmetry, same fix.

        RECYCLING f_{n+1} as the next sub-step's f_n (rather than
        re-evaluating at the corrected z1) keeps the order at 2.00 for half the
        evaluations, and at equal budget is slightly MORE accurate
        (4.9e-5 at N=64 recycled vs 5.7e-5 at N=32 re-evaluated).

        dt_cap applies to the SUB-step, so it becomes inert as n grows --
        which is the point: it exists to bound f*(dt^2/2) at large dt, and
        sub-stepping removes the large dt.

        VARIABLE delta_t, per sample. Samples in a batch need different counts,
        so the loop runs to the batch's MAXIMUM and masks samples that have
        already arrived: their h is zeroed, which makes each further sub-step
        an exact no-op (z0 += 0, z1 += 0) rather than a special case. Cost is
        therefore the batch max, which is why callers should batch windows of
        similar required count -- see rollout()'s own note. Every sample still
        lands EXACTLY on the transition endpoint, because h = dt/n divides dt
        exactly and n sub-steps of it are taken; the milestone is hit by
        construction, not by a final correction step.
        """
        dt_r = dt.view(-1, 1, 1, 1)
        f_n = self.f(z0, z1, theta) if f_carry is None else f_carry
        if self.alpha is None:
            n_steps = torch.full((z0.shape[0],), self.n_substeps,
                                  dtype=torch.long, device=z0.device)
        else:
            n_steps = self._substeps_for(z0, z1, dt, theta, f_n)
        h = dt_r / n_steps.view(-1, 1, 1, 1).to(dt_r.dtype)
        h_capped = torch.clamp(h, max=self.dt_cap)
        n_max = int(n_steps.max().item())
        # REALISED retained cost of this transition, in sample-substeps -- the
        # same quantity BudgetedBatchSampler budgets on, but from the counts
        # the integrator ACTUALLY uses rather than the pre-epoch estimate. The
        # two drift because the estimate uses each window's first state and
        # worst transition, while these are re-derived per transition from the
        # state actually reached; with bucket_refresh_epochs=0 nothing ever
        # corrects it. max(realised)/budget is what says whether the budget
        # bounds memory at all.
        #
        # Recorded HERE, not in _substeps_for, so it measures what the loop
        # runs on -- including the fixed-n path, and including any caller that
        # supplies counts by other means.
        if self.truncate_bptt is not None:
            _lo, _hi = float(n_steps.min()), float(n_max)
            _n = int(n_steps.numel())
            # The RAW counts of the most recent transition, for the per-batch
            # memory diagnostic. A list, not a tensor: rollout() calls
            # _integrate n_rollout_steps times per batch, and the diagnostic
            # wants all of them -- cleared by retained_peak(reset=True), which
            # the diagnostic's caller invokes per batch.
            self._last_counts.append(n_steps.detach().cpu())
            self._retained_peak = max(
                self._retained_peak,
                _n * min(_hi, _n * float(self.truncate_bptt),
                         (_hi - _lo) + float(self.truncate_bptt)))
        # Arrival accumulators, only under truncation (see the loop). Seeded
        # from the inputs so a sample arriving at step 0 is still covered, and
        # so the tensors exist with the right shape/dtype either way.
        if self.truncate_bptt is not None:
            z0_out, z1_out, f_out = z0, z1, f_n
        for step in range(n_max):
            # active: samples that still have sub-steps left. Zeroing h rather
            # than indexing keeps the batch shape static, which matters for
            # autograd and for not fragmenting the f_theta call.
            active = (n_steps > step).view(-1, 1, 1, 1).to(h.dtype)
            h_i, h_capped_i = h * active, h_capped * active
            z0 = z0 + z1 * h_i + f_n * (h_capped_i ** 2 / 2)
            z1_pred = z1 + f_n * h_i
            f_next = self.f(z0, z1_pred, theta)
            z1 = z1 + (f_n + f_next) * (h_i / 2)
            # An INACTIVE sample must keep the f it arrived with: f_next was
            # evaluated at its unchanged state, so the two agree in exact
            # arithmetic, but taking f_next unconditionally would let
            # floating-point drift accumulate over the remaining no-op steps
            # and, worse, would make the carried f depend on how many other
            # samples happened to share the batch.
            f_n = torch.where(active.bool(), f_next, f_n)
            # TRUNCATED BACKPROPAGATION THROUGH THE SUB-STEPS. Detaching every
            # `truncate_bptt` sub-steps cuts the backward graph into segments of
            # that depth. The forward integration is UNCHANGED -- same values,
            # same milestones, same order of convergence -- and the gradient
            # becomes the sum of within-segment gradients instead of the full
            # chain.
            #
            # WHY: the backward pass multiplies one Jacobian per sub-step, so
            # its magnitude is exponential in the count. Measured on this
            # project: depth 264 (max_dt=500) trains with gradient norms ~3e3;
            # depth ~1700 (max_dt=2000 at alpha=0.1 unclamped, max 867 sub-steps
            # x 2 rollout steps) gives INFINITE norms on every batch while the
            # losses stay finite. No learning rate, memory budget or clamp
            # addresses that -- it is float32 range, not optimisation. Raising
            # max_substeps makes it worse, since a larger cap permits a deeper
            # graph.
            #
            # WHAT IT COSTS: the gradient loses the dependence of late sub-steps
            # on early ones, so it is biased. The bias is small when a segment
            # spans the dynamics' own correlation length. Opt-in: None (the
            # default) leaves the graph whole and every existing run identical.
            #
            # z0 AND z1 AND f_n detach TOGETHER. Detaching a subset leaves a
            # path around the cut -- z1 carries gradient into the next segment's
            # z0 through z1*h, and f_n through both -- so any one left attached
            # reconstitutes the chain and the truncation silently does nothing.
            # `step + 2 < n_max`, NOT `step + 1 < n_max`: the final segment
            # must contain at least TWO sub-steps.
            #
            # THE BUG THIS FIXES, hit in production at k=64. Each sub-step's z0
            # update consumes the f_n RECYCLED from the previous step -- that
            # recycling is what keeps the scheme at one f evaluation per step.
            # A final segment of ONE step therefore updates z0 using an f_n
            # that was just detached, and the step's own f evaluation feeds
            # only z1 and the next f_n, neither of which reaches the output. So
            # z0 emerges with NO gradient path whenever
            #
            #     n_max % truncate_bptt == 1
            #
            # and backward() raises "element 0 of tensors does not require grad
            # and does not have a grad_fn". At k=64 the trap values are 65, 129,
            # ... 961 -- squarely inside the observed range of 876-1024 -- and
            # with n_rollout_steps=2 over cost-BUCKETED (hence homogeneous)
            # batches, both transitions can land on one together and take the
            # whole loss's graph with them.
            #
            # I had already found and documented this at k=1 and "fixed" it by
            # requiring k >= 2, which addressed the symptom I happened to test
            # rather than the condition: k=1 is simply the case where EVERY
            # segment is length 1. Requiring two steps at the end covers both.
            # PER-SAMPLE, not batch-wide. The counts in a batch are
            # heterogeneous (that is the adaptive criterion's whole point), and
            # a batch-wide detach at global boundaries ZEROES the gradient of
            # every sample that has already arrived: its final value's graph is
            # severed, and the only surviving paths run through the
            # zero-multiplied no-op updates -- so requires_grad stays True,
            # backward() runs, and the sample contributes exactly nothing.
            # Measured on counts (70, 100, 140, 200) with k=64: samples 0-2
            # had |grad| = 0.0 to the last digit; only the deepest trained. In
            # a real deep bucketed batch that silently reduces the gradient to
            # the few windows past the last boundary.
            #
            # The rule: cut a sample at a boundary only if >= 2 of ITS OWN
            # steps remain. remaining <= 0 (arrived): its value IS the output,
            # cutting it kills its loss. remaining == 1: its final z0 update
            # consumes the f_n recycled from THIS step, so cutting now leaves
            # that update with a detached f_n -- the one-step-final-segment
            # bug, per sample. remaining >= 2: safe, it rebuilds before
            # arrival.
            #
            # Memory stays bounded: the samples KEPT attached at a boundary
            # are those within 1 step of arrival, whose retained history is
            # their own final segment (<= k+1 steps of batch-wide ops).
            # torch.where preserves values exactly, so the forward remains
            # bit-identical -- only which graph each sample's output hangs on
            # changes.
            if self.truncate_bptt is not None:
                # ARRIVAL CAPTURE, then an unconditional cut.
                #
                # The previous per-sample form kept arrived samples attached
                # through torch.where -- but a graph belongs to a TENSOR, not
                # to an element, so keeping one sample attached retained that
                # segment's ops for the whole batch. Measured retained-node
                # counts at k=16, depth 256: homogeneous batch 17x saving,
                # [200..256] only 4x, [16,64,128,256] NONE (4902 vs 4857
                # untruncated). Truncation quietly stopped bounding memory in
                # proportion to how heterogeneous the batch was -- and
                # bucketing makes batches heterogeneous by construction.
                #
                # Instead: the step a sample ARRIVES, copy its value out into
                # an accumulator. That copy carries the sample's own gradient
                # path, back only as far as the previous boundary. The running
                # state then has no reason to stay attached and is detached
                # outright, freeing the segment for every sample at once.
                #
                # Retained memory is therefore (number of distinct segments in
                # which some sample arrives) x k, rather than the full depth.
                # Under bucketing arrivals cluster in one or two segments, so
                # this is ~k; in the worst (unbucketed) case it degrades to the
                # untruncated cost, which is what it was already doing.
                arrive = (n_steps == step + 1).view(-1, 1, 1, 1)
                if bool(arrive.any()):
                    z0_out = torch.where(arrive, z0, z0_out)
                    z1_out = torch.where(arrive, z1, z1_out)
                    f_out = torch.where(arrive, f_n, f_out)
                # DEFER a boundary when a sample arrives on the very next
                # step. That step's z0 update consumes the f_n recycled from
                # THIS one, so detaching now would hand the arriving sample a
                # gradient-free value -- the one-step-final-segment bug,
                # which making the cut unconditional reintroduced (caught by
                # the every-count sweep at n_max % k == 1). Deferring costs
                # one extra segment of retention only for the boundaries
                # immediately preceding an arrival.
                if ((step + 1) % self.truncate_bptt == 0
                        and not bool((n_steps == step + 2).any())):
                    z0, z1, f_n = z0.detach(), z1.detach(), f_n.detach()
        if self.truncate_bptt is not None:
            # Every sample arrives at some step <= n_max, so the accumulators
            # are complete; the running tensors are detached leftovers.
            return z0_out, z1_out, f_out
        return z0, z1, f_n

    def rollout(self, z0: torch.Tensor, z1_sequence: torch.Tensor, dts: torch.Tensor,
                theta: torch.Tensor, z1_resync: bool = True) -> torch.Tensor:
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

        if self.dynamics_mode == "deriv_linear" and not z1_resync and n_steps > 1:
            # deriv_linear's f is a derivative CORRECTION, not z0's curvature, so
            # there is no z1-update equation to PROPAGATE z1 across steps -- an
            # autonomous (z1_resync=False) rollout of more than one step would
            # have to, and that is undefined until the q-scheme (Step B).
            # A SINGLE step never propagates z1 (no next step to carry it into),
            # so z1_resync is moot at n_steps==1 and the step is well-defined via
            # forward() below -- only n_steps > 1 is forbidden. Fire on the
            # OPERATION (propagation), not the config, so 1-step and
            # teacher-forced calls are never wrongly rejected.
            raise ValueError(
                "dynamics_mode='deriv_linear' cannot roll out autonomously for >1 "
                "step: f is a derivative correction, not z0's curvature, so there is no "
                "z1-update equation to propagate z1 across steps (the q-scheme, Step B). "
                "Use z1_resync=True, or a single step.")

        if self.n_substeps == 1 and self.alpha is None and (
                z1_resync or self.dynamics_mode == "deriv_linear"):
            # The z1_resync path uses forward() per step, teacher-forcing z1 from
            # z1_sequence -- for z1_taylor this is the historical fast path, kept
            # EXACTLY as it was. deriv_linear ALSO routes here unconditionally:
            # it always has n_substeps==1/alpha None (ctor guard), and by the
            # guard just above it only reaches this line when z1_resync is True
            # or n_steps==1 -- in both cases teacher-forcing z1 is exactly right
            # (at one step there is no propagated z1 to differ from), and it is
            # the ONLY defined path for deriv_linear (the _integrate path below
            # is the Taylor form and has no dt-conditioned f).
            # expressed as a special case of the loop below. _integrate's
            # trapezoidal z1 update produces a different (better) z1, and with
            # z1 resynced at every frame that z1 is discarded anyway -- so the
            # two agree numerically, but only for a reason that has to be
            # re-derived every time someone reads it. A separate branch makes
            # "the default is untouched" checkable by inspection.
            #
            # `self.alpha is None` is part of the condition because alpha
            # leaves n_substeps at 1 by construction (setting both is an
            # error, see __init__) -- so without this clause an adaptive model
            # would silently take the ONE-SHOT path and never sub-step at all,
            # which is the exact configuration alpha exists to prevent.
            for i in range(n_steps):
                z0_cur = self.forward(z0_cur, z1_sequence[:, i], dts[:, i], theta)
                z0_hats.append(z0_cur)
            return torch.stack(z0_hats, dim=1)

        # z1 is now PROPAGATED across sub-steps by _integrate. z1_resync
        # decides only what happens at a REAL frame: True overwrites the
        # propagated value with the encoder's (today's teacher-forcing),
        # False keeps it (what inference is obliged to do, since no frame
        # exists there to resync to).
        z1_cur = z1_sequence[:, 0]
        f_carry = None
        for i in range(n_steps):
            if z1_resync:
                z1_cur = z1_sequence[:, i]
                f_carry = None  # the carried f belongs to the value just discarded
            z0_cur, z1_cur, f_carry = self._integrate(
                z0_cur, z1_cur, dts[:, i], theta, f_carry)
            z0_hats.append(z0_cur)
        return torch.stack(z0_hats, dim=1)
