"""
Step 2: the semi-implicit integrator itself.

These pin the SCHEME, not a trained model -- they run in milliseconds and need
no GPU, no dataset and no checkpoint. Two of them carry most of the weight:

  * with f_theta CONSTANT, every n_substeps must give the IDENTICAL answer.
    That is an algebraic identity, not an approximation: summing
    (z1 + k*f*h)*h + f*h^2/2 over N sub-steps telescopes to exactly
    z1*dt + f*dt^2/2. It proves any N-dependence seen later comes from
    f_theta being re-evaluated, and not from a bookkeeping error in the loop.

  * with f non-constant, error must fall ~4x per doubling of N (order 2).
    That is what distinguishes the trapezoidal z1 update from the one-sided
    Euler one, which is only order 1 and drags the whole scheme down with it.
"""
import math

import pytest
import torch

from models.latent_dynamics import LatentDynamics
from models.constants import N_THETA

LC, LS, B = 4, 8, 2


def _model(n_substeps=1, **kw):
    return LatentDynamics(latent_channels=LC, latent_spatial=LS, hidden_dim=8,
                           n_hidden_layers=1, n_substeps=n_substeps, **kw).eval()


def _state(seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, LC, LS, LS), torch.randn(B, LC, LS, LS) * 0.01,
            torch.zeros(B, N_THETA))


# --------------------------------------------------------------------
# the telescoping identity
# --------------------------------------------------------------------

@pytest.mark.parametrize("n_substeps", [1, 2, 4, 8, 16])
def test_constant_f_gives_the_same_answer_for_every_n_substeps(n_substeps):
    """
    GUARDS a bookkeeping error in the sub-step loop -- an h that is not dt/N,
    a missed carry, an off-by-one in the range. With f constant the exact
    answer is z0 + z1*dt + f*dt^2/2 regardless of N, so any N-dependence here
    is a bug and nothing else.
    """
    model = _model(n_substeps)
    z0, z1, theta = _state()
    const = torch.full_like(z0, 3e-4)
    model.f = lambda a, b, c: const  # noqa: E731 - deliberate stub
    dt = torch.full((B,), 250.0)

    with torch.no_grad():
        got_z0, got_z1, _ = model._integrate(z0, z1, dt, theta)
    dt_r = dt.view(-1, 1, 1, 1)
    assert torch.allclose(got_z0, z0 + z1 * dt_r + const * dt_r ** 2 / 2, atol=1e-5)
    # z1 too: with f constant the trapezoidal average IS f, so z1 += f*dt exactly
    assert torch.allclose(got_z1, z1 + const * dt_r, atol=1e-6)


def test_constant_f_matches_forward_exactly_at_n_substeps_1():
    model = _model(1)
    z0, z1, theta = _state()
    const = torch.full_like(z0, 3e-4)
    model.f = lambda a, b, c: const  # noqa: E731
    dt = torch.full((B,), 250.0)
    with torch.no_grad():
        assert torch.allclose(model._integrate(z0, z1, dt, theta)[0],
                               model.forward(z0, z1, dt, theta), atol=1e-6)


# --------------------------------------------------------------------
# order of convergence
# --------------------------------------------------------------------

def test_second_order_convergence_with_a_non_constant_f():
    """
    GUARDS reverting the trapezoidal z1 update to the one-sided
    `z1 += f*h`, which is order 1. The distinction is the entire reason
    g_theta is not needed for this: a CENTRED average of f_theta recovers
    second order using f_theta alone.
    """
    # float64, and an excursion of O(0.1) against z0 ~ 1. Both matter: in
    # float32 the error floors at ~1e-7 and the measured order collapses at
    # large N, and with a large excursion (an earlier version used
    # f*dt^2/2 ~ 64) the scheme is nowhere near its asymptotic regime and the
    # ratios are meaningless. Neither is a property of the integrator, so
    # neither belongs in a test of its order.
    torch.set_default_dtype(torch.float64)
    try:
        model = LatentDynamics(latent_channels=LC, latent_spatial=LS, hidden_dim=8,
                                n_hidden_layers=1).eval().double()
        model.f = lambda z0, z1, th: -2e-5 * torch.sin(z0) - 1e-6 * z1  # noqa: E731
        torch.manual_seed(1)
        z0 = torch.randn(B, LC, LS, LS, dtype=torch.float64)
        z1 = torch.randn(B, LC, LS, LS, dtype=torch.float64) * 1e-3
        theta = torch.zeros(B, 1, dtype=torch.float64)
        dt = torch.full((B,), 100.0, dtype=torch.float64)

        def run(n):
            model.n_substeps = n
            with torch.no_grad():
                return model._integrate(z0, z1, dt, theta)[0]

        reference = run(8192)
        errors = [float((run(n) - reference).abs().max()) for n in (8, 16, 32, 64)]
        orders = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    finally:
        torch.set_default_dtype(torch.float32)
    assert all(o > 1.8 for o in orders), (
        f"orders {orders} -- expected ~2. Order ~1 means the z1 update reverted "
        f"to the one-sided `z1 += f*h`."
    )


def test_one_f_evaluation_per_substep():
    """
    GUARDS re-evaluating f at the corrected z1, which doubles the cost for no
    gain in order (and is slightly WORSE at equal budget). f_{n+1} must be
    carried into the next sub-step.
    """
    model = _model(8)
    calls = {"n": 0}
    real_f = model.f

    def counting(z0, z1, th):
        calls["n"] += 1
        return real_f(z0, z1, th)

    model.f = counting
    z0, z1, theta = _state(2)
    with torch.no_grad():
        model._integrate(z0, z1, torch.full((B,), 100.0), theta)
    # 1 priming evaluation + 1 per sub-step
    assert calls["n"] == 1 + 8, calls["n"]


def test_carried_f_avoids_the_priming_evaluation():
    model = _model(4)
    calls = {"n": 0}
    real_f = model.f

    def counting(z0, z1, th):
        calls["n"] += 1
        return real_f(z0, z1, th)

    model.f = counting
    z0, z1, theta = _state(3)
    carried = torch.zeros_like(z0)
    with torch.no_grad():
        model._integrate(z0, z1, torch.full((B,), 100.0), theta, f_carry=carried)
    assert calls["n"] == 4, calls["n"]


# --------------------------------------------------------------------
# dt_cap becomes inert
# --------------------------------------------------------------------

def test_dt_cap_stops_binding_as_n_substeps_grows():
    """
    dt_cap exists to bound f*(dt^2/2) at large dt. Sub-stepping removes the
    large dt, so the cap should stop mattering -- which is what makes the
    N-sweep a clean test of what dt_cap was compensating for.
    """
    z0, z1, theta = _state(4)
    dt = torch.full((B,), 1000.0)
    capped, uncapped = [], []
    for n in (1, 64):
        a = _model(n, dt_cap=10.0); b = _model(n, dt_cap=float("inf"))
        a.f = b.f = lambda z0, z1, th: torch.full_like(z0, 2e-4)  # noqa: E731
        with torch.no_grad():
            capped.append(a._integrate(z0, z1, dt, theta)[0])
            uncapped.append(b._integrate(z0, z1, dt, theta)[0])
    gap_1 = float((capped[0] - uncapped[0]).abs().max())
    gap_64 = float((capped[1] - uncapped[1]).abs().max())
    assert gap_1 > 0, "dt_cap should bind at n_substeps=1 with dt=1000 >> cap=10"
    assert gap_64 < gap_1 / 100, f"cap still binding at n=64: {gap_64} vs {gap_1}"


# --------------------------------------------------------------------
# rollout dispatch
# --------------------------------------------------------------------

def test_rollout_default_path_is_untouched():
    """The historical branch must be reached, and be exactly repeated
    forward() with z1 teacher-forced."""
    model = _model(1)
    torch.manual_seed(5)
    n = 3
    z0 = torch.randn(B, LC, LS, LS)
    z1_seq = torch.randn(B, n + 1, LC, LS, LS) * 0.01
    dts = torch.rand(B, n) * 100 + 1
    theta = torch.zeros(B, N_THETA)
    with torch.no_grad():
        got = model.rollout(z0, z1_seq, dts, theta)
        cur, expected = z0, [z0]
        for i in range(n):
            cur = model.forward(cur, z1_seq[:, i], dts[:, i], theta)
            expected.append(cur)
    assert torch.equal(got, torch.stack(expected, dim=1))


def test_z1_resync_false_ignores_the_supplied_z1_after_the_first_frame():
    """
    With resync off, only z1_sequence[:, 0] may be read -- everything after is
    the model's own propagated z1. Checked by corrupting the later entries: the
    result must not move.
    """
    model = _model(4)
    torch.manual_seed(6)
    n = 3
    z0 = torch.randn(B, LC, LS, LS)
    z1_seq = torch.randn(B, n + 1, LC, LS, LS) * 0.01
    dts = torch.rand(B, n) * 100 + 1
    theta = torch.zeros(B, N_THETA)
    corrupted = z1_seq.clone()
    corrupted[:, 1:] = 999.0
    with torch.no_grad():
        a = model.rollout(z0, z1_seq, dts, theta, z1_resync=False)
        b = model.rollout(z0, corrupted, dts, theta, z1_resync=False)
        c = model.rollout(z0, corrupted, dts, theta, z1_resync=True)
    assert torch.equal(a, b), "resync=False must not read z1_sequence past index 0"
    assert not torch.equal(a, c), "resync=True must read it"


# --------------------------------------------------------------------
# dt_cap and sub-stepping compensate for the same thing
# --------------------------------------------------------------------

def test_combining_a_finite_dt_cap_with_substepping_warns(capsys):
    """
    GUARDS silently allowing the combination. They are two answers to one
    question -- how to stop f*(dt^2/2) dominating at large dt -- and together
    an n_substeps sweep measures their interaction instead of integration
    accuracy. Measured at dt=2500, dt_cap=125, f=1, the total f contribution
    runs 7.8e3 (N=1) -> 1.6e6 (N=2) -> 3.1e6 (N=32): a 400x spread that is
    pure artifact.

    A warning rather than an error: a cap below every sub-step is harmless.
    """
    LatentDynamics(latent_channels=LC, latent_spatial=LS, hidden_dim=8,
                    n_hidden_layers=1, n_substeps=4, dt_cap=125.0)
    out = capsys.readouterr().out
    assert "dt_cap" in out and "n_substeps" in out
    assert "dt_cap=inf" in out


@pytest.mark.parametrize("n_substeps,dt_cap", [(1, 125.0), (4, float("inf")), (1, float("inf"))])
def test_no_warning_when_only_one_mechanism_is_active(capsys, n_substeps, dt_cap):
    LatentDynamics(latent_channels=LC, latent_spatial=LS, hidden_dim=8,
                    n_hidden_layers=1, n_substeps=n_substeps, dt_cap=dt_cap)
    assert "WARNING" not in capsys.readouterr().out


def test_the_capped_substep_divergence_is_real_not_a_test_artifact():
    """
    Pins the magnitude quoted in the warning, so the claim stays checkable if
    the integrator changes. z1 evolving on the UNCAPPED h while z0's
    second-order term is capped is what produces it.
    """
    z0 = torch.zeros(1, LC, LS, LS)
    z1 = torch.zeros(1, LC, LS, LS)
    theta = torch.zeros(1, N_THETA)
    dt = torch.full((1,), 2500.0)
    totals = []
    for n in (1, 32):
        m = LatentDynamics(latent_channels=LC, latent_spatial=LS, hidden_dim=8,
                            n_hidden_layers=1, n_substeps=n, dt_cap=125.0).eval()
        m.f = lambda a, b, c: torch.ones_like(a)  # noqa: E731
        with torch.no_grad():
            totals.append(float(m._integrate(z0, z1, dt, theta)[0].max()))
    assert totals[1] / totals[0] > 100, totals
