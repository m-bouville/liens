"""
alpha must be measured, not guessed -- so these tests are on SYNTHETIC
latents with a KNOWN |f_theta| and |z1|, where the correct alpha can be
written down in closed form. Real latents would only show that the code runs.

The property under test throughout is the single relation

    alpha = |f_theta| * delta_t / |z1|,     delta_t = Delta_t / n_substeps

read in both directions: alpha_at_substeps solves it for alpha, and
substeps_for_alpha solves the same equation for n_substeps. If those two ever
stop being inverses, a calibration measured with one and applied with the
other is silently wrong, which is the failure this file exists to prevent.
"""
import numpy as np
import pytest
import torch

from evaluation.check_alpha import (
    alpha_at_substeps, collect_alpha, substeps_for_alpha,
)


class _ConstantField(torch.nn.Module):
    """f_theta whose field has a KNOWN norm, so alpha is known in closed form.

    Only `.f` is exercised by collect_alpha (deliberately -- forward() would
    fold in the z1*dt term and dt_cap, neither of which alpha is about), so
    only `.f` needs to exist here.
    """

    def __init__(self, value: float, shape=(8, 8, 8)):
        super().__init__()
        self.value = value
        self.shape = shape
        self.n_calls = 0

    def f(self, z0, z1, theta):
        self.n_calls += 1
        return torch.full((z0.shape[0], *self.shape), self.value)

    def eval(self):
        return self


def _fake_dataset(n_runs=2, n_steps=5, dt_scale=1.0, z1_value=1.0, step_gap=100):
    """A dataset stub exposing only the attributes collect_alpha reads."""
    from types import SimpleNamespace
    shape = (8, 8, 8)
    ds = SimpleNamespace()
    ds._run_data = [torch.zeros(n_steps, *shape) for _ in range(n_runs)]
    ds._run_data_deriv = [torch.full((n_steps, *shape), z1_value) for _ in range(n_runs)]
    ds._run_steps = [[i * step_gap for i in range(n_steps)] for _ in range(n_runs)]
    ds._run_dt_scale = [dt_scale] * n_runs
    ds._run_theta = [torch.tensor([-0.22]) for _ in range(n_runs)]
    return ds


def _expected_norm(value, shape=(8, 8, 8)):
    return abs(value) * np.sqrt(np.prod(shape))


def test_alpha_matches_the_closed_form_on_known_fields():
    """
    THE DEFINITION. With |f| and |z1| known exactly, alpha at a given
    n_substeps is arithmetic -- so any discrepancy is a bug in the measurement,
    not an artefact of the data.
    """
    ds = _fake_dataset(n_runs=1, n_steps=4, z1_value=2.0, step_gap=100, dt_scale=1.0)
    f_theta = _ConstantField(0.5)
    data = collect_alpha(ds, f_theta, torch.device("cpu"))

    expected_f = _expected_norm(0.5)
    expected_z1 = _expected_norm(2.0)
    assert np.allclose(data["f_norm"], expected_f)
    assert np.allclose(data["z1_norm"], expected_z1)
    assert np.allclose(data["dt"], 100.0)

    alpha = alpha_at_substeps(data, 4)
    assert np.allclose(alpha, expected_f * (100.0 / 4) / expected_z1)


def test_alpha_and_substeps_are_inverses():
    """
    THE INVARIANT THAT MATTERS. alpha is measured with one function and spent
    with the other; if they drift apart, a calibration is applied to a
    different quantity than the one it was measured on. Round-tripping pins
    them together up to the ceiling.
    """
    ds = _fake_dataset(n_runs=3, n_steps=6, z1_value=1.5, step_gap=250)
    data = collect_alpha(ds, _ConstantField(0.3), torch.device("cpu"))

    for n_sub in (1, 3, 7, 14, 56):
        alpha = alpha_at_substeps(data, n_sub)
        recovered = substeps_for_alpha(data, float(np.median(alpha)))
        # ceil, so recovery is exact only where the division was; allow the
        # one-step rounding the ceiling introduces.
        assert np.all(np.abs(recovered - n_sub) <= 1), (
            f"n_substeps={n_sub} did not round-trip through alpha: got "
            f"{np.unique(recovered)[:5]}"
        )


def test_alpha_scales_inversely_with_n_substeps():
    """
    Doubling n_substeps must halve alpha -- the relation the whole bracket
    argument rests on, since the two known runs differ by exactly a factor 2.
    """
    ds = _fake_dataset(n_runs=2, n_steps=5)
    data = collect_alpha(ds, _ConstantField(0.4), torch.device("cpu"))
    a7 = alpha_at_substeps(data, 7)
    a14 = alpha_at_substeps(data, 14)
    assert np.allclose(a14, a7 / 2)


def test_a_zero_velocity_state_yields_infinite_alpha_not_nan():
    """
    |z1| = 0 with nonzero curvature has NO valid step under this criterion.
    That must surface as inf, which a tail quantile carries, rather than nan,
    which np.quantile silently drops -- turning "this state admits no step"
    into "this state is fine".
    """
    ds = _fake_dataset(n_runs=1, n_steps=3, z1_value=0.0)
    data = collect_alpha(ds, _ConstantField(0.5), torch.device("cpu"))
    alpha = alpha_at_substeps(data, 7)
    assert np.all(np.isinf(alpha)), f"expected inf, got {alpha}"
    assert not np.any(np.isnan(alpha)), "nan would be dropped from quantiles silently"


def test_a_dead_state_is_alpha_zero_not_nan():
    """
    |z1| = 0 AND |f| = 0: nothing is happening, so the step is unbounded --
    alpha = 0, not the nan that 0/0 produces. nan is the dangerous outcome
    because np.quantile drops it silently, so a population of dead states
    would shrink the sample without changing any reported number.

    This is the case the guard actually has to handle. An earlier version
    guarded |z1|=0 with |f|>0 instead, which numpy already yields inf for --
    dead code, and the test asserting it passed against its own removal.
    """
    ds = _fake_dataset(n_runs=1, n_steps=3, z1_value=0.0)
    data = collect_alpha(ds, _ConstantField(0.0), torch.device("cpu"))
    alpha = alpha_at_substeps(data, 7)
    assert not np.any(np.isnan(alpha)), "nan would be dropped from quantiles silently"
    assert np.allclose(alpha, 0.0), f"a dead state should admit any step, got {alpha}"


def test_a_zero_curvature_state_admits_an_unbounded_step():
    """
    The complementary degenerate case, and the one that made the ABSOLUTE-
    tolerance form of this criterion wrong: with f = 0 the linear
    extrapolation is exact, so alpha = 0 and the step is unbounded. f_theta's
    final layer is zero-initialised, so this is the state of every fresh
    stage-3a run, not a hypothetical.
    """
    ds = _fake_dataset(n_runs=1, n_steps=3, z1_value=1.0)
    data = collect_alpha(ds, _ConstantField(0.0), torch.device("cpu"))
    assert np.allclose(alpha_at_substeps(data, 7), 0.0)
    # ... and asking for any alpha costs exactly one sub-step, not zero or inf.
    assert np.all(substeps_for_alpha(data, 0.05) == 1.0)


def test_substeps_are_at_least_one():
    """A window needing no correction still needs its single step taken."""
    ds = _fake_dataset(n_runs=1, n_steps=4, z1_value=1000.0)
    data = collect_alpha(ds, _ConstantField(1e-9), torch.device("cpu"))
    assert np.all(substeps_for_alpha(data, 0.5) >= 1.0)


def test_a_tighter_alpha_never_costs_fewer_substeps():
    """Monotonicity -- the property that makes alpha a usable dial at all."""
    ds = _fake_dataset(n_runs=2, n_steps=6, z1_value=0.8)
    data = collect_alpha(ds, _ConstantField(0.6), torch.device("cpu"))
    counts = [substeps_for_alpha(data, a).mean() for a in (0.5, 0.2, 0.1, 0.05, 0.02)]
    assert all(b >= a for a, b in zip(counts, counts[1:])), counts


def test_the_last_frame_of_each_run_is_skipped():
    """
    It has no transition, so its alpha would be measured against a step that is
    never taken. n_steps frames give n_steps-1 transitions per run.
    """
    ds = _fake_dataset(n_runs=3, n_steps=5)
    data = collect_alpha(ds, _ConstantField(0.5), torch.device("cpu"))
    assert data["f_norm"].size == 3 * 4


def test_runs_too_short_to_have_a_transition_are_skipped_not_fatal():
    """A single-frame run contributes nothing and must not raise."""
    ds = _fake_dataset(n_runs=2, n_steps=4)
    ds._run_data.append(torch.zeros(1, 8, 8, 8))
    ds._run_data_deriv.append(torch.ones(1, 8, 8, 8))
    ds._run_steps.append([0])
    ds._run_dt_scale.append(1.0)
    ds._run_theta.append(torch.tensor([-0.2]))
    data = collect_alpha(ds, _ConstantField(0.5), torch.device("cpu"))
    assert data["f_norm"].size == 2 * 3


def test_an_empty_dataset_raises_rather_than_reporting_on_nothing():
    from types import SimpleNamespace
    ds = SimpleNamespace(_run_data=[], _run_data_deriv=[], _run_steps=[],
                          _run_dt_scale=[], _run_theta=[])
    with pytest.raises(ValueError, match="no frames collected"):
        collect_alpha(ds, _ConstantField(0.5), torch.device("cpu"))


def test_dt_scale_is_applied():
    """
    Delta_t is in PHYSICAL units, not step counts -- the scale is per-run
    metadata and dropping it would make alpha wrong by that factor, silently
    and consistently, which is the hardest kind of error to notice in a
    calibration.
    """
    ds_a = _fake_dataset(n_runs=1, n_steps=3, dt_scale=1.0, step_gap=100)
    ds_b = _fake_dataset(n_runs=1, n_steps=3, dt_scale=0.05, step_gap=100)
    f_theta = _ConstantField(0.5)
    a = collect_alpha(ds_a, f_theta, torch.device("cpu"))
    b = collect_alpha(ds_b, f_theta, torch.device("cpu"))
    assert np.allclose(a["dt"], 100.0)
    assert np.allclose(b["dt"], 5.0)
    assert np.allclose(alpha_at_substeps(b, 7), alpha_at_substeps(a, 7) * 0.05)


def test_norms_are_over_the_whole_tensor_not_per_channel():
    """
    alpha describes ONE step taken for the WHOLE state. A per-channel ratio
    would be dominated by whichever channel is nearest zero -- a property of
    that channel, not of the step -- so a channel-wise reduction would report
    a wildly tighter alpha than the integrator actually needs.
    """
    ds = _fake_dataset(n_runs=1, n_steps=2, z1_value=1.0)
    # one channel of z1 set to near-zero: a per-channel alpha would explode,
    # a whole-tensor alpha barely moves.
    ds._run_data_deriv[0][:, 0] = 1e-8
    data = collect_alpha(ds, _ConstantField(0.5), torch.device("cpu"))
    full = _expected_norm(1.0)
    assert data["z1_norm"][0] == pytest.approx(full * np.sqrt(7 / 8), rel=1e-3), (
        "z1 norm is not the whole-tensor norm -- one damped channel changed it "
        "by far more than 1/8 of the energy"
    )


def test_batching_does_not_change_the_measurement():
    """
    collect_alpha batches its f_theta calls; the batch size is a performance
    knob and must not touch the numbers. Also pins that batching actually
    happens, so a future rewrite that evaluates frame-by-frame is visible.
    """
    ds = _fake_dataset(n_runs=1, n_steps=9, z1_value=1.2)
    small = _ConstantField(0.5)
    large = _ConstantField(0.5)
    a = collect_alpha(ds, small, torch.device("cpu"), batch_size=2)
    b = collect_alpha(ds, large, torch.device("cpu"), batch_size=1024)
    assert np.allclose(a["f_norm"], b["f_norm"])
    assert small.n_calls > large.n_calls, "batch_size had no effect on call count"
    assert large.n_calls == 1, "a batch larger than the run should be one call"


def _synthetic_context(n_runs=12, n_steps=10):
    """A dataset + f_theta pair standing in for the loader, for report tests."""
    from types import SimpleNamespace
    ds = SimpleNamespace(_run_data=[], _run_data_deriv=[], _run_steps=[],
                          _run_dt_scale=[], _run_theta=[])
    for r in range(n_runs):
        z1 = torch.randn(n_steps, 8, 8, 8) * (0.5 + 0.1 * r)
        ds._run_data.append(torch.zeros(n_steps, 8, 8, 8))
        ds._run_data_deriv.append(z1)
        ds._run_steps.append([int(1000 * 1.5 ** i) for i in range(n_steps)])
        ds._run_dt_scale.append(0.05)
        ds._run_theta.append(torch.tensor([-0.22]))

    class _F(torch.nn.Module):
        def f(self, z0, z1, theta):
            return z1 * 0.002

        def eval(self):
            return self

    return ds, _F()


def test_the_report_renders_without_nan(monkeypatch, capsys):
    """
    THE REPORT IS THE PRODUCT, and no unit test above touches it -- they all
    call the numeric functions directly. That gap hid a real bug: _quantiles
    computed p50/p90/p99/p100 while the cost table printed p95, so that whole
    column was 'nan' on every row of a report that otherwise looked correct.

    A calibration read off a nan column is worse than no calibration, so this
    renders the real report and asserts no nan reaches the output.
    """
    import pathlib

    import evaluation.check_alpha as mod
    ds, f_theta = _synthetic_context()
    monkeypatch.setattr(mod, "_load_ae_f_theta_and_dataset",
                         lambda *a, **k: (torch.device("cpu"), False, None, None,
                                          ds, None, f_theta))
    mod.check_alpha(pathlib.Path("unused.pt"))
    out = capsys.readouterr().out
    assert "nan" not in out.lower(), (
        f"the report printed nan:\n{out}"
    )
    assert "THE BRACKET" in out
    assert "cost of each candidate alpha" in out


def test_the_report_anchors_calibration_on_the_STABLE_median(monkeypatch, capsys):
    """
    THE GUIDANCE IS PART OF THE TOOL, and a wrong sentence in it cost a real
    run: the first version told the reader to choose alpha below the UNSTABLE
    configuration's tail, which yielded alpha=0.3 and a deadlock within 9
    epochs. A fixed n_substeps spreads alpha over a distribution; a fixed alpha
    puts every window AT alpha, so the unstable tail is the wrong anchor by
    construction.

    Pinned because the numbers alone cannot convey it -- the report has to say
    which quantile to read, and that instruction has to stay correct.
    """
    import pathlib

    import evaluation.check_alpha as mod
    ds, f_theta = _synthetic_context()
    monkeypatch.setattr(mod, "_load_ae_f_theta_and_dataset",
                         lambda *a, **k: (torch.device("cpu"), False, None, None,
                                          ds, None, f_theta))
    result = mod.check_alpha(pathlib.Path("unused.pt"))
    out = capsys.readouterr().out
    assert "ANCHOR ON THE STABLE RUN'S MEDIAN" in out
    assert "below the unstable" not in out.lower(), (
        "the report still advises anchoring on the unstable configuration"
    )
    # the anchor VALUE quoted must be the stable run's own median, not a
    # tail or the other configuration's number
    stable_median = result["by_substeps"][max(result["by_substeps"])]["p50"]
    assert f"{stable_median:.3g}" in out, (
        f"the stable median {stable_median:.3g} is not the value the report quotes"
    )


def test_the_report_prices_every_candidate_alpha(monkeypatch, capsys):
    """
    Each candidate alpha must reach the table with a real cost, since the
    entire point of the inversion is to state alpha in the familiar units
    before committing to it.
    """
    import pathlib

    import evaluation.check_alpha as mod
    ds, f_theta = _synthetic_context()
    monkeypatch.setattr(mod, "_load_ae_f_theta_and_dataset",
                         lambda *a, **k: (torch.device("cpu"), False, None, None,
                                          ds, None, f_theta))
    result = mod.check_alpha(pathlib.Path("unused.pt"),
                              candidate_alphas=(0.3, 0.1, 0.03))
    assert set(result["by_alpha"]) == {0.3, 0.1, 0.03}
    for alpha, stats in result["by_alpha"].items():
        for key in ("mean", "p95", "p100"):
            assert key in stats and np.isfinite(stats[key]), (alpha, key, stats)
    # tighter alpha must cost more, in the report as well as in the function
    means = [result["by_alpha"][a]["mean"] for a in (0.3, 0.1, 0.03)]
    assert means[0] < means[1] < means[2], means
