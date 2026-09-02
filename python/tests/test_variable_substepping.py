"""
Variable delta_t: the step follows the local dynamics, not the save schedule.

alpha bounds the curvature correction as a fraction of the linear term,
|f_theta|*delta_t/|z1| <= alpha, and the sub-step COUNT is derived from it per
transition per sample. The properties that matter are not "it runs" but:

  * at a forced count it is EXACTLY the fixed-n_substeps integrator (so the
    2.00 convergence order and everything measured under it still holds);
  * a batch of mixed counts gives each sample exactly what it would get alone
    (the masking must be a no-op, not an approximation);
  * every sample lands ON the transition endpoint, since training compares z0
    only at real frames;
  * the count is DETACHED, or gradient descent optimises the integrator
    instead of the physics;
  * the degenerate ends (|f|=0, |z1|=0, both) resolve sensibly rather than to
    nan/inf step counts.
"""
import math

import pytest
import torch

from models.latent_dynamics import LatentDynamics
from models.constants import N_THETA


def _model(nonzero_field=True, **kwargs):
    """A model whose f is ACTUALLY NONZERO by default.

    LatentDynamics zero-initialises its final layer, so a freshly built model
    has f == 0 identically -- and an integrator with no curvature term is
    exact linear extrapolation at ANY sub-step count. Tests written on the
    default model therefore pass against almost any mutation: verified, three
    separate mutations (unconditional f_carry, floor-instead-of-ceil, and the
    criterion check itself) survived precisely because every sub-step was a
    no-op. The zero-init case still deserves its own tests -- it is the state
    of every fresh stage 3a -- but it must not be the default fixture.
    """
    torch.manual_seed(0)
    m = LatentDynamics(latent_channels=4, latent_spatial=8, hidden_dim=8,
                        n_hidden_layers=1, **kwargs)
    if nonzero_field:
        with torch.no_grad():
            torch.manual_seed(11)
            m.net[-1].weight.normal_(0.0, 0.02)
            m.net[-1].bias.normal_(0.0, 0.02)
    return m


def _twin(reference, **kwargs):
    """A model with identical weights but different integration settings."""
    other = _model(nonzero_field=False, **kwargs)
    other.load_state_dict(reference.state_dict())
    return other


def _inputs(b=2, seed=1):
    torch.manual_seed(seed)
    return (torch.randn(b, 4, 8, 8), torch.randn(b, 4, 8, 8),
            torch.full((b,), 100.0), torch.zeros(b, N_THETA))


def test_a_forced_count_reproduces_the_fixed_substep_integrator_exactly():
    """
    THE EQUIVALENCE THAT PROTECTS EVERYTHING ELSE. Every property measured on
    the fixed-n_substeps scheme -- the 2.00 order of convergence, the f
    recycling, the stability bracket -- carries over only if the adaptive path
    computes the identical thing at an identical count. Any discrepancy here
    means the two schemes are different integrators, and the alpha calibration
    measured against the old one would not transfer.
    """
    fixed = _model(n_substeps=5)
    adaptive = _twin(fixed, alpha=0.1)
    adaptive._substeps_for = lambda *a, **k: torch.tensor([5, 5])

    z0, z1, dt, theta = _inputs()
    a0, a1, af = fixed._integrate(z0.clone(), z1.clone(), dt, theta)
    b0, b1, bf = adaptive._integrate(z0.clone(), z1.clone(), dt, theta)
    assert torch.allclose(a0, b0, atol=1e-6)
    assert torch.allclose(a1, b1, atol=1e-6)
    assert torch.allclose(af, bf, atol=1e-6)


def test_a_mixed_batch_gives_each_sample_what_it_would_get_alone():
    """
    THE MASKING. Samples needing different counts share one loop that runs to
    the batch maximum, with arrived samples masked by zeroing h. If that is
    not an exact no-op, a window's result depends on which other windows
    happened to share its batch -- a silent, batch-order-dependent corruption
    that no loss curve would reveal.
    """
    counts = torch.tensor([2, 7, 3])
    adaptive = _model(alpha=0.1)
    adaptive._substeps_for = lambda *a, **k: counts

    z0, z1, dt, theta = _inputs(b=3)
    mixed0, mixed1, mixedf = adaptive._integrate(z0.clone(), z1.clone(), dt, theta)

    for i, c in enumerate(counts):
        solo = _twin(adaptive, n_substeps=int(c))
        s0, s1, sf = solo._integrate(z0[i:i + 1].clone(), z1[i:i + 1].clone(),
                                      dt[i:i + 1], theta[i:i + 1])
        # RELATIVE, and measured against the tensor's own scale. With a
        # nonzero field the state reaches O(1e6) over seven sub-steps, where
        # float32's own epsilon (1.2e-7) makes any ABSOLUTE tolerance
        # meaningless -- measured discrepancies are 1-3e-7 relative across
        # every sample, i.e. exactly rounding. Two earlier versions of this
        # test failed on tolerance alone while the masking was correct.
        for name, got, want in (("z0", mixed0, s0), ("z1", mixed1, s1), ("f", mixedf, sf)):
            scale = max(float(want.detach().abs().max()), 1e-30)
            rel = float((got[i:i + 1] - want).detach().abs().max()) / scale
            assert rel < 1e-5, (
                f"{name} differs for sample {i} by {rel:.2e} relative -- far above "
                f"float32 rounding, so the masking is not an exact no-op"
            )


def test_every_sample_lands_exactly_on_the_transition_endpoint():
    """
    MILESTONES. The loss compares z0 only at real frames, so an integrator
    that overshoots or undershoots the endpoint is computing a different loss
    than the one claimed. h = dt/n divides dt exactly and n sub-steps of it
    are taken, so this holds by construction -- pinned because a future
    'take steps of size h until we arrive' rewrite would break it for any dt
    not divisible by h.
    """
    adaptive = _model(alpha=0.05)  # f zeroed below ON PURPOSE, see comment
    counts = torch.tensor([3, 8])
    adaptive._substeps_for = lambda *a, **k: counts
    b = 2
    z0 = torch.zeros(b, 4, 8, 8)
    z1 = torch.ones(b, 4, 8, 8)
    dt = torch.tensor([120.0, 250.0])
    theta = torch.zeros(b, N_THETA)

    # With f == 0 the scheme is exact linear extrapolation, so the arrival
    # state is z0 + z1*dt for ANY count -- which is precisely the statement
    # that the sub-steps sum to dt.
    with torch.no_grad():
        for p in adaptive.net.parameters():
            p.zero_()
    out0, _, _ = adaptive._integrate(z0, z1, dt, theta)
    expected = z0 + z1 * dt.view(-1, 1, 1, 1)
    assert torch.allclose(out0, expected, atol=1e-4), (
        "sub-steps do not sum to the transition: the endpoint was missed"
    )


def test_the_substep_count_is_integral_which_is_what_detaches_it():
    """
    THE HAZARD: if n depended differentiably on |f_theta|, gradient descent
    could lower the loss by SHRINKING f to earn longer steps -- optimising the
    integrator rather than the physics.

    THE GUARANTEE is the integer dtype, not the no_grad block. Integer tensors
    cannot carry gradients, so no path from the count back into f can exist
    however _substeps_for is later rewritten. Asserting `n.grad_fn is None`
    directly would be vacuous -- verified: removing the no_grad entirely
    leaves every such assertion passing, because .long() orphans the graph
    regardless. So what is pinned here is the dtype, which is the thing that
    would actually have to change for the hazard to return.
    """
    adaptive = _model(alpha=0.1)
    z0, z1, dt, theta = _inputs()
    z0.requires_grad_(True)
    f_n = adaptive.f(z0, z1, theta)
    n = adaptive._substeps_for(z0, z1, dt, theta, f_n)
    assert not n.is_floating_point(), (
        "the count is a float tensor -- it could carry gradient back into f, "
        "letting training shrink f to earn longer steps"
    )
    assert n.dtype in (torch.long, torch.int32, torch.int64), n.dtype


def test_the_count_still_lets_gradients_flow_through_the_integration():
    """The complement: detaching the COUNT must not detach the STATE."""
    adaptive = _model(alpha=0.1)
    z0, z1, dt, theta = _inputs()
    z0.requires_grad_(True)
    out, _, _ = adaptive._integrate(z0, z1, dt, theta)
    out.sum().backward()
    assert z0.grad is not None and torch.any(z0.grad != 0), (
        "no gradient reached z0 -- the adaptive path is not differentiable"
    )


def test_a_tighter_alpha_never_takes_fewer_substeps():
    """Monotonicity, the property that makes alpha a usable dial."""
    z0, z1, dt, theta = _inputs(b=4, seed=3)
    counts = []
    for alpha in (0.5, 0.2, 0.05):
        m = _model(alpha=alpha)
        with torch.no_grad():
            for p in m.net.parameters():
                p.mul_(0.0).add_(0.01)
        f_n = m.f(z0, z1, theta)
        counts.append(m._substeps_for(z0, z1, dt, theta, f_n).float().mean().item())
    assert counts[0] <= counts[1] <= counts[2], counts


def test_a_zero_curvature_state_takes_one_substep():
    """
    f_theta's final layer is zero-initialised, so this is the state of EVERY
    fresh stage 3a, not a hypothetical. No curvature means linear
    extrapolation is exact, so one step suffices -- an absolute-tolerance
    criterion would divide by zero here instead.
    """
    m = _model(nonzero_field=False, alpha=0.1)
    z0, z1, dt, theta = _inputs()
    f_zero = torch.zeros_like(z0)
    n = m._substeps_for(z0, z1, dt, theta, f_zero)
    assert torch.all(n == 1), n


def test_a_zero_velocity_state_is_bounded_by_max_substeps_not_infinite():
    """
    |z1| = 0 with curvature present admits no valid step under a ratio
    criterion. That must clamp to max_substeps -- a COST bound -- rather than
    produce inf/nan and stall or crash the batch. The clamp is also COUNTED,
    so a run where it binds routinely is visible rather than silent.
    """
    m = _model(alpha=0.1, max_substeps=32)
    z0 = torch.randn(2, 4, 8, 8)
    z1 = torch.zeros(2, 4, 8, 8)
    dt = torch.full((2,), 100.0)
    theta = torch.zeros(2, N_THETA)
    f_big = torch.ones_like(z0)
    n = m._substeps_for(z0, z1, dt, theta, f_big)
    assert torch.all(n == 32), n
    assert m.n_substeps_clamped == 2, m.n_substeps_clamped


def test_a_dead_state_takes_one_substep():
    """Neither velocity nor curvature: nothing is happening, one step."""
    m = _model(alpha=0.1)
    z0 = torch.randn(2, 4, 8, 8)
    z1 = torch.zeros(2, 4, 8, 8)
    dt = torch.full((2,), 100.0)
    theta = torch.zeros(2, N_THETA)
    n = m._substeps_for(z0, z1, dt, theta, torch.zeros_like(z0))
    assert torch.all(n == 1), n
    assert m.n_substeps_clamped == 0, "a dead state is not a clamp"


def test_the_count_satisfies_the_criterion_it_was_derived_from():
    """
    The whole point, stated directly: after choosing n, the realised ratio
    |f|*(dt/n)/|z1| must actually be <= alpha (up to the ceiling from ceil()).
    """
    alpha = 0.1
    m = _model(alpha=alpha)
    torch.manual_seed(7)
    z0 = torch.randn(8, 4, 8, 8)
    z1 = torch.randn(8, 4, 8, 8)
    dt = torch.full((8,), 400.0)
    theta = torch.zeros(8, N_THETA)
    # No multiplier: at this field scale the criterion already demands ~120-160
    # sub-steps, which binds hard while staying under max_substeps=256. Scaling
    # f up pushed it into the clamp, where the guarantee legitimately does not
    # hold -- the next test covers that case on purpose.
    f_n = m.f(z0, z1, theta)
    assert float(torch.linalg.vector_norm(f_n.detach())) > 0, (
        "the field is identically zero -- this test would assert 0 <= alpha and "
        "pass against any implementation")
    n = m._substeps_for(z0, z1, dt, theta, f_n)
    assert m.n_substeps_clamped == 0, (
        "max_substeps bound here, so the criterion is EXPECTED to be violated -- "
        "that is what a cost bound means. Scale the field down so this test "
        "measures the criterion rather than the clamp.")
    assert torch.any(n > 1), "the criterion never bound; this would pass trivially"
    f_norm = torch.linalg.vector_norm(f_n.reshape(8, -1), dim=1)
    z1_norm = torch.linalg.vector_norm(z1.reshape(8, -1), dim=1)
    realised = f_norm * (dt / n.float()) / z1_norm
    assert torch.all(realised <= alpha + 1e-6), realised


def test_the_criterion_is_violated_exactly_where_max_substeps_binds():
    """
    The complement, and the reason max_substeps is COUNTED rather than silent:
    where the cost bound binds, the alpha guarantee is gone, and a run must be
    able to tell that from the outside. A clamp that binds routinely is a
    finding about the data, not a detail.
    """
    m = _model(alpha=0.001, max_substeps=4)
    torch.manual_seed(7)
    z0 = torch.randn(4, 4, 8, 8)
    z1 = torch.randn(4, 4, 8, 8) * 0.01
    dt = torch.full((4,), 500.0)
    theta = torch.zeros(4, N_THETA)
    f_n = m.f(z0, z1, theta) * 100.0
    n = m._substeps_for(z0, z1, dt, theta, f_n)
    assert torch.all(n == 4), n
    assert m.n_substeps_clamped == 4, m.n_substeps_clamped


def test_alpha_and_n_substeps_together_are_an_error():
    """
    Two answers to one question, with no sensible reading: the count would be
    adaptive AND multiplied. An error rather than a warning, unlike the
    dt_cap/n_substeps clash, where a cap below every h is genuinely harmless.
    """
    with pytest.raises(ValueError, match="alpha REPLACES n_substeps"):
        _model(alpha=0.1, n_substeps=4)


def test_alpha_must_be_positive():
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError, match="alpha must be"):
            _model(alpha=bad)


def test_default_construction_is_unchanged_by_the_feature():
    """alpha=None must leave the historical behaviour bit-for-bit."""
    m = _model()
    assert m.alpha is None
    assert m.n_substeps == 1
    z0, z1, dt, theta = _inputs()
    z1_seq = torch.randn(2, 3, 4, 8, 8)
    dts = torch.full((2, 2), 100.0)
    out = m.rollout(z0, z1_seq, dts, theta)
    # one-shot path: each step is exactly forward()
    expected = z0
    for i in range(2):
        expected = m.forward(expected, z1_seq[:, i], dts[:, i], theta)
    assert torch.allclose(out[:, -1], expected, atol=1e-6)


def test_an_adaptive_model_does_not_take_the_one_shot_fast_path():
    """
    THE ROUTING BUG THIS GUARDS. alpha leaves n_substeps at 1 by construction
    (setting both is an error), so a fast path keyed on `n_substeps == 1 and
    z1_resync` would route every adaptive model straight to the one-shot
    forward() -- never sub-stepping at all, which is the exact configuration
    alpha exists to prevent, and it would look like a working run.
    """
    m = _model(alpha=0.02)
    with torch.no_grad():
        for p in m.net.parameters():
            p.mul_(0.0).add_(0.05)
    z0 = torch.randn(2, 4, 8, 8)
    z1_seq = torch.randn(2, 3, 4, 8, 8)
    dts = torch.full((2, 2), 500.0)
    theta = torch.zeros(2, N_THETA)

    adaptive_out = m.rollout(z0, z1_seq, dts, theta, z1_resync=True)

    one_shot = _twin(m, n_substeps=1)
    one_shot.alpha = None  # force the historical path with identical weights
    one_shot_out = one_shot.rollout(z0, z1_seq, dts, theta, z1_resync=True)

    assert not torch.allclose(adaptive_out, one_shot_out, atol=1e-5), (
        "the adaptive model produced the one-shot result -- it took the fast "
        "path and never sub-stepped"
    )


# ---------------------------------------------------------------------------
# Wiring: alpha must survive the round trip through train_lds and the
# checkpoint, or the model-level work above is unreachable from a real run.
# ---------------------------------------------------------------------------

def test_alpha_is_saved_in_the_checkpoint_config():
    """
    alpha belongs in the SAME config block as dt_cap, and for the same reason:
    it defines what f_theta was fitted to MEAN. Two checkpoints with identical
    weights and different alpha are correctors calibrated to different step
    sizes, and nothing downstream could tell them apart without this.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert '"alpha": alpha,' in src, "alpha is not written into the checkpoint config"
    assert '"max_substeps": max_substeps,' in src


def test_alpha_participates_in_resume_comparability():
    """
    THE CEILING HAZARD. A resume whose ancestor used a different alpha measured
    a different quantity, so its val_loss is not a bar this run should clear --
    exactly the argument that put n_substeps in this key. Omitting alpha would
    let an alpha change silently inherit a ceiling from a differently
    integrated run, which is the quiet version of the regression the ceiling
    exists to prevent (a 3a resume once wrote a 0.82% worse checkpoint over a
    better one for the mirror-image reason).
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert 'prev_config.get("alpha") == alpha' in src, (
        "alpha is missing from the comparability key -- an alpha change would "
        "inherit its ancestor's val_loss ceiling"
    )


def test_train_lds_accepts_alpha_and_max_substeps():
    """The signature is the interface a params file reaches through."""
    import inspect
    from training.train_lds import train_lds
    params = inspect.signature(train_lds).parameters
    assert "alpha" in params and params["alpha"].default is None, (
        "alpha must default to None so every existing params file and "
        "checkpoint keeps the fixed-n_substeps behaviour exactly"
    )
    assert "max_substeps" in params and params["max_substeps"].default == 256


def test_alpha_reaches_the_model_through_a_real_run(tmp_path, isolated_project_root):
    """
    END TO END, because the source-matching version of this test was useless:
    it asserted "alpha=alpha" appears in train_lds.py, which stayed true after
    the argument was deleted from the LatentDynamics constructor -- the string
    still occurs in the resume call. Verified: that mutation passed.

    So this runs train_lds twice on a real (tiny) sweep, once adaptive and once
    not, and requires the two to DISAGREE. A parameter accepted by the
    signature and dropped before the constructor cannot survive that, however
    the source happens to be spelled.
    """
    from test_train_lds import _build_sweep, _cached_stage2_ancestor
    from training.train_lds import train_lds

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    common = dict(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path,
        ae_stats_weight=0.01, epochs=1, batch_size=4, hidden_dim=8,
        n_hidden_layers=1, val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=2, min_step=0, min_stdev_phi=None, encode_batch_size=4,
        ema_warmup_epochs=0, device="cpu", seed=0, log_every_epoch=False,
        z1_resync=False,
    )
    plain = train_lds(checkpoint_path=tmp_path / "plain.pt",
                       loss_curve_path=tmp_path / "plain.png", **common)
    adaptive = train_lds(checkpoint_path=tmp_path / "adaptive.pt",
                          loss_curve_path=tmp_path / "adaptive.png",
                          alpha=0.001, **common)

    a = torch.load(adaptive, map_location="cpu", weights_only=True)
    p = torch.load(plain, map_location="cpu", weights_only=True)
    assert a["config"]["alpha"] == 0.001, a["config"]
    assert p["config"]["alpha"] is None, p["config"]
    assert a["val_loss"] != p["val_loss"], (
        "an adaptive run produced the same val_loss as a non-adaptive one -- "
        "alpha never reached the integrator"
    )


def test_alpha_is_logged_without_being_added_to_any_list():
    """
    The run-parameter block's inverted default, exercised on a REAL new
    parameter: alpha and max_substeps must appear in a run's log without
    anyone having remembered to add a print. This is the property the whole
    logging rework exists for, and a new integration parameter that changes
    what f_theta means is precisely the case where silence would hurt.
    """
    import inspect

    from training.train_lds import train_lds, _LDS_PREAMBLE_PARAMS
    from utils.logging_utils import _PLUMBING, print_run_parameters
    values = {n: (p.default if p.default is not inspect.Parameter.empty else None)
              for n, p in inspect.signature(train_lds).parameters.items()}
    values["alpha"] = 0.25
    lines = print_run_parameters(train_lds, values, _LDS_PREAMBLE_PARAMS)
    printed = " ".join(lines)
    assert "alpha=0.25" in printed, printed
    assert "max_substeps=256" in printed, printed
    for name in ("alpha", "max_substeps"):
        assert name not in _PLUMBING, f"{name} changes what the run means; it is not plumbing"


# ---------------------------------------------------------------------------
# Observability: the realised sub-step count, and the messages that tell a
# reader which knob to turn.
# ---------------------------------------------------------------------------

def test_substep_stats_report_the_realised_count():
    """
    THE MECHANISM'S ONLY WITNESS. alpha's whole claim over a fixed count is
    that the step tightens by itself as f_theta sharpens. Nothing else in a
    run reports the derived count, so without this the claim can never be
    checked against a log -- and "alpha happened to pick a safe constant"
    would look identical to "alpha adapted".
    """
    m = _model(alpha=0.1)
    z0, z1, dt, theta = _inputs(b=4, seed=5)
    m._integrate(z0, z1, dt, theta)
    stats = m.substep_stats()
    assert stats is not None
    assert stats["transitions"] == 4
    assert stats["mean"] >= 1 and stats["max"] >= stats["mean"]


def test_substep_stats_are_per_epoch_not_cumulative():
    """
    A cumulative mean flattens the drift this exists to show: after 500 epochs
    a single high epoch moves the average by 0.2%. Reset on read.
    """
    m = _model(alpha=0.1)
    z0, z1, dt, theta = _inputs(b=4, seed=5)
    m._integrate(z0, z1, dt, theta)
    first = m.substep_stats()
    assert first["transitions"] == 4
    assert m.substep_stats() is None, "stats were not reset, so the next epoch pools with this one"


def test_a_clamp_is_carried_across_resets():
    """
    The mean and max are per-epoch, but a clamp that bound even once is worth
    keeping: it means the criterion was OVERRIDDEN on those transitions, so
    they ran coarser than alpha asked for -- a different diagnosis from
    "alpha is too loose", and one the deadlock message names.
    """
    m = _model(alpha=0.001, max_substeps=2)
    z0 = torch.randn(3, 4, 8, 8)
    z1 = torch.randn(3, 4, 8, 8) * 0.01
    dt = torch.full((3,), 500.0)
    theta = torch.zeros(3, N_THETA)
    m._integrate(z0, z1, dt, theta)
    first = m.substep_stats()
    assert first["clamped"] > 0
    m._integrate(z0, z1, dt, theta)
    assert m.substep_stats()["clamped"] > first["clamped"], (
        "the clamp counter was reset -- a binding clamp is a property of the run, "
        "not of one epoch"
    )


def test_a_fixed_count_run_reports_no_substep_stats():
    """
    None, not a dict of the constant: printing "mean 7.0 max 7" every epoch
    for a declared n_substeps=7 is noise, and the point of the report is
    precisely the distinction between a DERIVED and a DECLARED count.
    """
    m = _model(n_substeps=7)
    z0, z1, dt, theta = _inputs()
    m._integrate(z0, z1, dt, theta)
    assert m.substep_stats() is None


def test_the_misleading_n_substeps_note_is_suppressed_under_alpha():
    """
    alpha leaves n_substeps at 1 by construction, so resuming an n_substeps=7
    checkpoint under alpha printed "now running at n_substeps=1 ... going
    N -> 1 would use a pointwise f_theta as a one-shot corrector" -- a warning
    about the exact configuration alpha exists to PREVENT, fired at a run that
    was sub-stepping adaptively throughout. Observed on a real run and read as
    alarming; the alpha note beside it already says the true thing.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "if prev_n_substeps != n_substeps and alpha is None:" in src, (
        "the n_substeps resume note is not gated on alpha, so an adaptive run "
        "is warned about a one-shot configuration it is not in"
    )


def test_the_deadlock_advice_reaches_the_epoch_loop():
    """
    The ADVICE ITSELF is tested behaviorally in test_spike_guard.py, against
    deadlock_step_hint directly. What is checked here is only that the epoch
    loop calls it and hands it the live clamp count -- the wiring, not the
    wording.

    Split this way because the previous version asserted the message's
    substrings against train_lds.py and broke twice on legitimate changes:
    once when the text was branched, once when it was extracted into a
    function. Neither break said anything about whether the advice was right.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "deadlock_step_hint(" in src, "the epoch loop does not use the shared advice"
    assert 'getattr(f_theta, "n_substeps_clamped", 0)' in src, (
        "the live clamp count is not passed, so a run whose criterion was overridden "
        "gets told to tighten a criterion that was not what limited it"
    )


def test_the_count_rises_on_its_own_as_f_theta_sharpens():
    """
    THE PROPERTY ALPHA EXISTS FOR, stated as a test rather than an argument.

    A fixed n_substeps cannot do this, and that is why n=7 survived ~115
    epochs and then escalated to a deadlock: the step that was safe for the
    epoch-4 weights was not safe for the epoch-115 ones, and nothing adjusted.
    Under alpha the criterion re-reads |f_theta| every transition, so a
    sharper field buys itself a finer step automatically.

    Measured here by sharpening the field directly rather than by training,
    which isolates the mechanism from everything else a real run changes.
    """
    m = _model(alpha=0.1, max_substeps=4096)
    z0, z1, dt, theta = _inputs(b=6, seed=13)
    dt = torch.full((6,), 400.0)
    m._integrate(z0, z1, dt, theta)
    before = m.substep_stats()["mean"]

    with torch.no_grad():
        m.net[-1].weight.mul_(4.0)
        m.net[-1].bias.mul_(4.0)
    m._integrate(z0, z1, dt, theta)
    after = m.substep_stats()["mean"]

    assert after > 2 * before, (
        f"f_theta sharpened 4x but the sub-step count went {before:.1f} -> "
        f"{after:.1f}. The criterion is not re-reading |f_theta|, so alpha is "
        f"behaving as a fixed count under another name"
    )


def test_gradients_of_a_mixed_batch_match_the_solo_runs():
    """
    THE OTHER HALF of the masking guarantee. The forward equivalence is tested
    above; this pins the BACKWARD pass, because a masking scheme can reproduce
    outputs exactly while corrupting gradients (e.g. through the f_n
    torch.where, whose inactive branch must route gradient to the OLD f_n's
    graph, not the discarded f_next's). A gradient corruption would bias
    training silently -- no loss curve distinguishes "learning the physics"
    from "learning the physics as filtered through a wrong Jacobian".

    Measured: mixed-vs-solo relative gradient differences are 3-4e-7, i.e.
    float32 rounding, with a live field over mixed counts (2, 7, 3).
    """
    counts = torch.tensor([2, 7, 3])
    adaptive = _model(alpha=0.1)
    adaptive._substeps_for = lambda *a, **k: counts
    z0, z1, dt, theta = _inputs(b=3)

    out, _, _ = adaptive._integrate(z0.clone(), z1.clone(), dt, theta)
    out.square().sum().backward()
    mixed = {n: p.grad.clone() for n, p in adaptive.named_parameters()}

    solo = None
    for i, c in enumerate(counts):
        m = _twin(adaptive, n_substeps=int(c))
        o, _, _ = m._integrate(z0[i:i + 1].clone(), z1[i:i + 1].clone(),
                                dt[i:i + 1], theta[i:i + 1])
        o.square().sum().backward()
        grads = {n: p.grad for n, p in m.named_parameters()}
        solo = grads if solo is None else {n: solo[n] + grads[n] for n in solo}

    for name in mixed:
        scale = max(float(solo[name].abs().max()), 1e-30)
        rel = float((mixed[name] - solo[name]).abs().max()) / scale
        assert rel < 1e-5, (
            f"gradient of {name} differs by {rel:.2e} relative between the mixed "
            f"batch and the solo runs -- the masking corrupts the backward pass"
        )


def test_an_adaptive_model_with_a_finite_dt_cap_warns(capsys):
    """
    The dt_cap/sub-stepping clash warning predates alpha and keyed on
    `n_substeps > 1`, which alpha never sets -- so an adaptive model with a
    finite cap warned about NOTHING while having the same structural
    inconsistency, arriving less predictably: h is derived per sample, so
    whether the cap truncates the quadratic term (while the z1 trapezoid runs
    on the uncapped h) depends on |f|/|z1| at each state. Found by review.
    """
    _model(alpha=0.1, dt_cap=125.0)
    out = capsys.readouterr().out
    assert "WARNING" in out and "dt_cap" in out, (
        "an adaptive model with a finite dt_cap raised no warning -- the two "
        "mechanisms compensate for the same thing and fight per-sample"
    )
    # and inf stays silent, for both configurations
    _model(alpha=0.1)
    _model(n_substeps=7)
    assert "WARNING" not in capsys.readouterr().out


def test_the_substep_report_covers_training_transitions_only():
    """
    The report exists to show drift-with-TRAINING; averaging the fixed val
    population into every epoch dilutes exactly that signal. Found by review:
    the capture sat after the val loop. Pinned structurally -- the capture must
    happen between the two loops, and val's accumulation must be discarded so
    it cannot leak into the NEXT epoch's train numbers.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    train_end = src.index("_train_substeps = f_theta.substep_stats()")
    # val-start marker: the val pass now runs through accumulate_epoch(val_loader,
    # ...) rather than an inline val_loss_sum accumulator.
    val_start = src.index("val_loader, lambda b: step(b, train=False)")
    assert train_end < val_start, (
        "the train sub-step capture happens after the val loop begins -- the "
        "reported numbers mix the two populations"
    )
    assert "_substeps = _train_substeps" in src, (
        "the report does not use the train-only capture"
    )


# --------------------------------------------------------------------
# Truncated backpropagation through the sub-steps
# --------------------------------------------------------------------

def _tbptt_model(scale=0.005, **kw):
    torch.manual_seed(0)
    m = LatentDynamics(latent_channels=4, latent_spatial=8, hidden_dim=16,
                        n_hidden_layers=1, **kw)
    with torch.no_grad():
        torch.manual_seed(11)
        m.net[-1].weight.normal_(0, scale)
        m.net[-1].bias.normal_(0, scale)
    return m


def _tbptt_inputs(b=2):
    torch.manual_seed(3)
    return (torch.randn(b, 4, 8, 8) * 0.3, torch.randn(b, 4, 8, 8) * 0.3,
            torch.full((b,), 120.0), torch.zeros(b, N_THETA))


def test_truncation_leaves_the_forward_pass_bit_identical():
    """
    THE NON-NEGOTIABLE PROPERTY. Truncation is a gradient-only device: it must
    not perturb the trajectory, the milestones, or the order of convergence.
    If it changed the forward, a checkpoint trained with it would evaluate
    differently from one trained without, and truncate_bptt would have to join
    _MEANING_FIELDS and propagate to every rebuild site.
    """
    z0, z1, dt, theta = _tbptt_inputs()
    with torch.no_grad():
        ref, ref_z1, ref_f = _tbptt_model(n_substeps=256)._integrate(
            z0.clone(), z1.clone(), dt, theta)
    for k in (2, 8, 64):
        with torch.no_grad():
            got, got_z1, got_f = _tbptt_model(n_substeps=256, truncate_bptt=k)._integrate(
                z0.clone(), z1.clone(), dt, theta)
        assert torch.equal(ref, got), f"k={k} changed z0"
        assert torch.equal(ref_z1, got_z1), f"k={k} changed z1"
        assert torch.equal(ref_f, got_f), f"k={k} changed the carried f"


def _max_graph_depth(out) -> int:
    """Deepest chain reachable from `out`, over ALL branches.

    BFS, not a single-successor walk: torch.where keeps an edge to its
    unselected branch, so following one next_function per node can wander a
    path that hides a fully retained graph behind a masked edge -- the first
    version of this test did exactly that."""
    seen: dict = {}
    frontier = [(out.grad_fn, 0)]
    best = 0
    while frontier:
        fn, d = frontier.pop()
        if fn is None or seen.get(fn, -1) >= d:
            continue
        seen[fn] = d
        best = max(best, d)
        for nf, _ in fn.next_functions:
            frontier.append((nf, d + 1))
    return best


def test_truncation_shortens_the_backward_graph():
    """Structural: fewer chained Jacobians AND fewer retained ops -- the
    memory. The all-cut fast path is what frees them; torch.where alone
    retains its unselected attached branch, so without that path the depth
    does not shrink at all (measured 138 vs 131 untruncated)."""
    def depth(k):
        m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                            n_hidden_layers=1, n_substeps=64, truncate_bptt=k)
        torch.manual_seed(1)
        out, _, _ = m._integrate(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4),
                                  torch.full((1,), 10.0), torch.zeros(1, N_THETA))
        return _max_graph_depth(out)

    full = depth(None)
    assert depth(8) < full / 3, (depth(8), full)
    assert depth(4) < depth(8)


def test_every_sample_of_a_heterogeneous_batch_gets_gradient():
    """
    THE REVIEW FINDING, and the worst kind: silent. Counts in a batch are
    heterogeneous -- that is the adaptive criterion's point -- and a
    batch-wide detach at global boundaries zeroes the gradient of every
    sample that has already arrived: its output's graph is severed, the only
    surviving paths run through the zero-multiplied no-op updates, so
    requires_grad stays True, backward() runs, and the sample contributes
    EXACTLY nothing. Measured before the fix on counts (70, 100, 140, 200)
    with k=64: samples 0-2 had |grad| = 0.0; only the deepest trained. Every
    deep bucketed batch of the best-yet max_dt=2000 run trained that way.

    Count 129 is in the fixture deliberately: it has remaining == 1 at the
    128 boundary, so it also pins the per-sample form of the
    one-step-final-segment rule (cut only when >= 2 of the sample's OWN
    steps remain).
    """
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, alpha=0.1, max_substeps=2048,
                        truncate_bptt=64)
    torch.manual_seed(0)
    B = 5
    z0, z1 = torch.randn(B, 2, 4, 4), torch.randn(B, 2, 4, 4)
    counts = torch.tensor([70, 100, 129, 140, 200])
    m._substeps_for = lambda *a, **k: counts
    out, _, _ = m._integrate(z0, z1, torch.full((B,), 100.0), torch.zeros(B, N_THETA))
    per_sample = out.reshape(B, -1).square().sum(dim=1)
    for i in range(B):
        m.zero_grad()
        per_sample[i].backward(retain_graph=True)
        g = sum(float(p.grad.abs().sum()) for p in m.parameters()
                if p.grad is not None)
        assert g > 0.0, (
            f"sample {i} (n={int(counts[i])}) contributes zero gradient -- "
            f"truncation is cutting samples that have already arrived (or are "
            f"one step from arriving)"
        )


def test_deep_samples_are_still_truncated_in_a_mixed_batch():
    """
    The complement: keeping arrived samples attached must not stop the DEEP
    samples being cut. A mutation that detaches only when every sample can be
    cut would leave any batch containing one early-arriving window fully
    untruncated -- with counts (3, 2048) that is the difference between a
    bounded gradient and the exponential one truncation exists to prevent.
    """
    def deep_norm(k):
        m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                            n_hidden_layers=1, alpha=0.1, max_substeps=4096,
                            truncate_bptt=k)
        torch.manual_seed(0)
        z0, z1 = torch.randn(2, 2, 4, 4) * 0.3, torch.randn(2, 2, 4, 4) * 0.3
        m._substeps_for = lambda *a, **kw: torch.tensor([3, 2048])
        out, _, _ = m._integrate(z0, z1, torch.full((2,), 120.0),
                                  torch.zeros(2, N_THETA))
        out[1].square().sum().backward()
        return max(float(p.grad.abs().max()) for p in m.parameters()
                   if p.grad is not None)

    assert deep_norm(16) < deep_norm(None) / 50, (
        "the deep sample of a mixed batch is not being truncated"
    )


def test_truncation_bounds_the_gradient_magnitude():
    """
    THE FAILURE IT EXISTS FOR. Gradient magnitude is exponential in sub-step
    count: on this project depth 264 trained with norms ~3e3, while depth
    ~1700 gave INFINITE norms on every batch with finite losses. Truncation
    must break that growth -- the untruncated norm should be flat in depth
    (already saturated) while the truncated one stays orders below.
    """
    z0, z1, dt, theta = _tbptt_inputs()

    def norm(n, k):
        m = _tbptt_model(n_substeps=n, truncate_bptt=k)
        out, _, _ = m._integrate(z0.clone(), z1.clone(), dt, theta)
        out.square().sum().backward()
        return float(torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")))

    deep_full, deep_trunc = norm(2048, None), norm(2048, 16)
    assert deep_trunc < deep_full / 100, (
        f"truncated norm {deep_trunc:.3e} is not meaningfully below the full "
        f"{deep_full:.3e} -- the cut is not biting"
    )
    # and a SMALLER segment bounds it further
    assert norm(2048, 16) < norm(2048, 64)


@pytest.mark.parametrize("k", [2, 3, 8, 64])
def test_every_substep_count_produces_a_usable_gradient(k):
    """
    THE PRODUCTION CRASH. Each sub-step's z0 update consumes the f_n RECYCLED
    from the previous step, so a final segment of ONE step updates z0 from an
    f_n that was just detached -- and that step's own f evaluation feeds only
    z1 and the next f_n, neither of which reaches the output. z0 therefore has
    no gradient path whenever

        n_max % truncate_bptt == 1

    and backward() raises "element 0 of tensors does not require grad". At
    k=64 the trap values are 65, 129, ... 961, inside the observed range of
    876-1024; with n_rollout_steps=2 over cost-bucketed (homogeneous) batches
    both transitions can land on one together and the whole loss loses its
    graph. It killed a run at epoch 38.

    I had found this at k=1 and "fixed" it by requiring k >= 2 -- addressing
    the case I happened to test rather than the condition, since k=1 is just
    the case where every segment has length 1. This sweeps EVERY count, which
    is what the earlier test should have done.
    """
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, truncate_bptt=k)
    for n in range(2, 4 * k + 3):
        m.n_substeps = n
        torch.manual_seed(0)
        out, _, _ = m._integrate(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4),
                                  torch.full((1,), 5.0), torch.zeros(1, N_THETA))
        assert out.grad_fn is not None, (
            f"k={k}, n_max={n} (n%k={n % k}): output has no gradient path"
        )
        out.square().sum().backward()
        largest = max(float(p.grad.abs().max()) for p in m.parameters()
                      if p.grad is not None)
        assert largest > 0.0, f"k={k}, n_max={n}: gradient is identically zero"
        m.zero_grad()


def test_k_of_one_is_rejected_because_it_leaves_no_gradient():
    """
    FOUND BY MEASUREMENT, not by reading. Each sub-step's z0 update consumes
    the f_n RECYCLED from the previous step -- that recycling is what keeps the
    scheme at one f evaluation per step. So at k=1 every f_n is detached before
    it ever reaches a z0 update, the output has no grad_fn at all, and
    backward() raises. A segment needs two steps for one f evaluation to both
    live inside it and feed a later z0.
    """
    with pytest.raises(ValueError, match="truncate_bptt must be >= 2"):
        LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, truncate_bptt=1)
    with pytest.raises(ValueError, match="truncate_bptt must be >= 2"):
        LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, truncate_bptt=0)


def test_the_gradient_still_flows_at_the_smallest_allowed_segment():
    """k=2 is the boundary -- it must produce a usable gradient, or the
    validation is off by one."""
    z0, z1, dt, theta = _tbptt_inputs()
    m = _tbptt_model(n_substeps=64, truncate_bptt=2)
    out, _, _ = m._integrate(z0, z1, dt, theta)
    out.square().sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().max()) > 0 for g in grads)


def test_all_three_carried_tensors_detach_together():
    """
    z0, z1 and f_n all cross a segment boundary. Detaching a subset leaves a
    path around the cut -- z1 reaches the next z0 through z1*h, f_n through
    both -- so the chain reconstitutes and the truncation silently does
    nothing. Checked by requiring the truncated gradient to differ from the
    untruncated one; a partial detach would leave them equal.
    """
    z0, z1, dt, theta = _tbptt_inputs()

    def norm(n, k):
        m = _tbptt_model(n_substeps=n, truncate_bptt=k)
        out, _, _ = m._integrate(z0.clone(), z1.clone(), dt, theta)
        out.square().sum().backward()
        return float(torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")))

    # A partial detach is NOT caught by "the gradient changed" -- detaching
    # only z0 changes it too, and my first version of this test passed against
    # exactly that mutation. What separates them is the SUPPRESSION FACTOR at
    # depth: with all three cut the norm falls ~6400x below the untruncated
    # one at depth 2048; with only z0 cut the surviving z1/f_n chain keeps it
    # within ~170x. Measured on this fixture, so the 1000x bar sits well
    # between the two.
    suppression = norm(2048, None) / norm(2048, 16)
    assert suppression > 1000.0, (
        f"truncation suppresses the gradient only {suppression:.0f}x at depth "
        f"2048 -- a path is surviving the cut (z1 and f_n must detach too, not "
        f"just z0)"
    )


def test_truncation_is_off_by_default_and_absent_from_the_meaning_fields():
    """
    Off by default so every existing run is unchanged, and NOT a meaning field
    because the forward is identical with and without it -- a checkpoint
    rebuilt without it evaluates exactly as it trained. Putting it in
    _MEANING_FIELDS would also make the no_grad diagnostics pay detach calls
    for a graph they never build.
    """
    from models.latent_dynamics import _MEANING_FIELDS
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1)
    assert m.truncate_bptt is None
    assert "truncate_bptt" not in _MEANING_FIELDS


def test_train_lds_exposes_truncation_and_records_it():
    import inspect

    import pathlib

    from conftest import source_without_comments
    from training.train_lds import train_lds
    assert inspect.signature(train_lds).parameters["truncate_bptt"].default is None
    root = pathlib.Path(__file__).resolve().parent.parent
    src = source_without_comments(root / "training/train_lds.py")
    assert "truncate_bptt=truncate_bptt" in src, "the model is not built with it"
    assert '"truncate_bptt": truncate_bptt' in src, (
        "not recorded in the checkpoint config -- provenance for how the "
        "gradient was computed is lost"
    )


def _retained_nodes(out) -> int:
    """Autograd nodes reachable from `out` -- proportional to the activations
    the backward pass must keep alive, i.e. to VRAM."""
    seen, frontier, n = set(), [out.grad_fn], 0
    while frontier:
        fn = frontier.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        n += 1
        for nf, _ in fn.next_functions:
            frontier.append(nf)
    return n


def _integrate_counts(counts, k):
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, alpha=0.1, max_substeps=4096,
                        truncate_bptt=k)
    torch.manual_seed(0)
    b = len(counts)
    m._substeps_for = lambda *a, **kw: torch.tensor(counts)
    return m._integrate(torch.randn(b, 2, 4, 4) * 0.3, torch.randn(b, 2, 4, 4) * 0.3,
                         torch.full((b,), 60.0), torch.zeros(b, N_THETA))


@pytest.mark.parametrize("counts", [
    [256] * 4,                      # homogeneous
    [240, 246, 250, 256],           # narrow, as a well-bucketed batch
    [200, 218, 237, 256],           # moderate spread
    [16, 64, 128, 256],             # wide -- the worst case for arrival spread
])
def test_truncation_bounds_retained_memory_however_heterogeneous_the_batch(counts):
    """
    THE VRAM REGRESSION. A graph belongs to a TENSOR, not an element, so the
    per-sample torch.where form -- keeping arrived samples attached -- retained
    that segment's ops for the WHOLE batch. Measured at k=16, depth 256:
    homogeneous kept its 17x saving, [200..256] fell to 4x, and
    [16,64,128,256] to 0.99x, i.e. truncation bounded nothing at all. Bucketing
    makes batches heterogeneous by construction, so this was the common case.

    Capturing each sample's value AT ARRIVAL and then detaching the running
    state outright restores the bound: the copy carries that sample's gradient
    back only to the previous boundary, and nothing needs the running tensors
    attached.
    """
    full = _retained_nodes(_integrate_counts(counts, None)[0])
    trunc = _retained_nodes(_integrate_counts(counts, 16)[0])
    assert trunc < full / 3.5, (
        f"counts={counts}: truncation retains {trunc} nodes vs {full} "
        f"untruncated ({full / trunc:.1f}x) -- memory is not being bounded"
    )


def test_arrival_capture_keeps_the_forward_bit_identical():
    """The accumulator returns each sample's value from the step it arrived;
    the masked loop leaves that value unchanged afterwards, so the two must
    agree exactly -- including z1 and the carried f."""
    for counts in ([70, 100, 140, 200], [65, 66], [1, 2, 3, 4], [256] * 3):
        plain = _integrate_counts(counts, None)
        trunc = _integrate_counts(counts, 64)
        for name, a, b in zip(("z0", "z1", "f"), plain, trunc):
            assert torch.equal(a, b), f"counts={counts}: {name} differs"


def test_the_realised_retained_cost_is_recorded_per_transition():
    """
    THE GAP BETWEEN BUDGET AND REALITY, made measurable. The sampler budgets
    on a pre-epoch ESTIMATE (each window's first state, worst transition); the
    integrator re-derives counts per transition from the state it actually
    reaches. Nothing has ever compared the two, so "the budget is 30000" has
    never implied "memory is bounded by 30000".
    """
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, alpha=0.1, max_substeps=4096,
                        truncate_bptt=64)
    torch.manual_seed(0)
    B = 8
    counts = torch.tensor([50, 60, 70, 80, 300, 320, 340, 360])
    m._substeps_for = lambda *a, **k: counts
    m._integrate(torch.randn(B, 2, 4, 4), torch.randn(B, 2, 4, 4),
                  torch.full((B,), 100.0), torch.zeros(B, N_THETA))
    hi, lo = float(counts.max()), float(counts.min())
    expected = B * min(hi, B * 64.0, (hi - lo) + 64.0)
    assert m._retained_peak == pytest.approx(expected), (
        f"realised retained peak {m._retained_peak} != {expected}"
    )

    # AND a case where the n bound is the binding one: two samples, counts far
    # apart. Two samples can produce at most two arrival segments however wide
    # the gap, so span + k (18064) grossly over-charges what 2 * k (128) can
    # cost. Without this case the fixture above passes with the n bound
    # removed, since its span is narrow enough for span + k to bind anyway.
    m2 = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                         n_hidden_layers=1, alpha=0.1, max_substeps=32768,
                         truncate_bptt=64)
    m2._substeps_for = lambda *a, **k: torch.tensor([100, 18000])
    torch.manual_seed(0)
    m2._integrate(torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4),
                   torch.full((2,), 100.0), torch.zeros(2, N_THETA))
    assert m2._retained_peak == pytest.approx(2 * 128.0), (
        f"{m2._retained_peak} -- the n bound is missing, so a 2-sample batch "
        f"is charged for arrival segments it cannot have"
    )


def test_the_realised_peak_resets_per_epoch():
    """Cumulative would hide the drift it exists to show."""
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, alpha=0.1, max_substeps=4096,
                        truncate_bptt=64)
    torch.manual_seed(0)
    m._substeps_for = lambda *a, **k: torch.tensor([100, 200])
    m._integrate(torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4),
                  torch.full((2,), 100.0), torch.zeros(2, N_THETA))
    assert m._retained_peak > 0
    m.substep_stats(reset=True)
    assert m._retained_peak == 0.0, "cumulative would hide the drift"

    # The same through the ADAPTIVE path, where substep_stats returns a dict
    # rather than early-returning: the reset must clear the peak there too.
    # Without this the mutation that drops the reset from the dict branch
    # survives, because the only covered path was the early return.
    m3 = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                         n_hidden_layers=1, alpha=0.1, max_substeps=4096,
                         truncate_bptt=64)
    torch.manual_seed(0)
    m3._integrate(torch.randn(4, 2, 4, 4) * 0.3, torch.randn(4, 2, 4, 4) * 0.3,
                   torch.full((4,), 300.0), torch.zeros(4, N_THETA))
    assert m3.substep_stats(reset=False) is not None, "fixture took no adaptive path"
    assert m3._retained_peak > 0
    m3.substep_stats(reset=True)
    assert m3._retained_peak == 0.0


def test_it_is_not_recorded_without_truncation():
    """Without truncation the whole history is retained and the quantity has
    no meaning -- reporting a number there would invite sizing a budget from
    it."""
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, alpha=0.1, max_substeps=4096)
    torch.manual_seed(0)
    m._substeps_for = lambda *a, **k: torch.tensor([100, 200])
    m._integrate(torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4),
                  torch.full((2,), 100.0), torch.zeros(2, N_THETA))
    assert m._retained_peak == 0.0
