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


def test_the_report_renders_without_nan(tmp_path, monkeypatch, capsys):
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
    mod.check_alpha(pathlib.Path("unused.pt"), latent_cache_dir=tmp_path / "latent_cache")
    out = capsys.readouterr().out
    assert "nan" not in out.lower(), (
        f"the report printed nan:\n{out}"
    )
    assert "THE BRACKET" in out
    assert "cost of each candidate alpha" in out


def test_the_report_anchors_calibration_on_the_STABLE_median(tmp_path, monkeypatch, capsys):
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
    result = mod.check_alpha(pathlib.Path("unused.pt"), latent_cache_dir=tmp_path / "latent_cache")
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


def test_the_report_prices_every_candidate_alpha(tmp_path, monkeypatch, capsys):
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
                              candidate_alphas=(0.3, 0.1, 0.03), latent_cache_dir=tmp_path / "latent_cache")
    assert set(result["by_alpha"]) == {0.3, 0.1, 0.03}
    for alpha, stats in result["by_alpha"].items():
        for key in ("mean", "p95", "p100"):
            assert key in stats and np.isfinite(stats[key]), (alpha, key, stats)
    # tighter alpha must cost more, in the report as well as in the function
    means = [result["by_alpha"][a]["mean"] for a in (0.3, 0.1, 0.03)]
    assert means[0] < means[1] < means[2], means


# --------------------------------------------------------------------
# The per-decade step-size table: delta_t, delta_t/t, n_substeps, depth
# --------------------------------------------------------------------

def _decade_dataset(n_runs=8, n_steps=14, growth=1.8, z1_decay=0.35):
    """Runs on a geometric save schedule, so Delta_t spans several decades,
    with |z1| decaying as coarsening slows -- the shape of the real sweep."""
    from types import SimpleNamespace
    ds = SimpleNamespace(_run_data=[], _run_data_deriv=[], _run_steps=[],
                          _run_dt_scale=[], _run_theta=[])
    for _ in range(n_runs):
        z1 = torch.stack([torch.randn(8, 8, 8) * (2.0 / (1 + z1_decay * i))
                           for i in range(n_steps)])
        ds._run_data.append(torch.zeros(n_steps, 8, 8, 8))
        ds._run_data_deriv.append(z1)
        ds._run_steps.append([int(400 * growth ** i) for i in range(n_steps)])
        ds._run_dt_scale.append(0.05)
        ds._run_theta.append(torch.tensor([-0.22]))
    return ds


class _ProportionalField(torch.nn.Module):
    """f proportional to z1, so alpha depends on Delta_t alone -- the cleanest
    case for reasoning about what the table SHOULD say."""

    alpha = 0.1
    max_substeps = 256

    def __init__(self, k=0.0015):
        super().__init__()
        self.k = k

    def f(self, z0, z1, theta):
        return z1 * self.k

    def eval(self):
        return self


def test_delta_t_is_the_step_actually_taken():
    """
    delta_t = Delta_t / n_substeps, and it is the number that matters for
    stability -- alpha fixes a RATIO, and two runs sharing an alpha can take
    wildly different steps. Checked against the closed form.
    """
    from evaluation.check_alpha import collect_alpha, step_size_table, substeps_for_alpha
    torch.manual_seed(0)
    data = collect_alpha(_decade_dataset(), _ProportionalField(), torch.device("cpu"))
    rows = step_size_table(data, alpha=0.1, max_substeps=10 ** 6)
    counts = substeps_for_alpha(data, 0.1)
    for r in rows:
        lo, hi = (10.0 ** int(r["decade"].split("e")[1].split(" ")[0]),
                   10.0 ** int(r["decade"].split("e")[-1]))
        m = (data["dt"] >= lo) & (data["dt"] < hi)
        expected = float(np.median(data["dt"][m] / counts[m]))
        assert r["delta_t_median"] == pytest.approx(expected, rel=1e-9), r["decade"]


def test_the_clamp_is_reported_and_shortens_nothing_below_it():
    """
    A binding max_substeps means those windows ran COARSER than alpha asked,
    so the guarantee lapsed -- the table must say so rather than showing a
    delta_t the run never took. Checked both ways: with the clamp far away
    (0%), and with it biting the top decade.
    """
    from evaluation.check_alpha import collect_alpha, step_size_table
    torch.manual_seed(0)
    data = collect_alpha(_decade_dataset(), _ProportionalField(), torch.device("cpu"))

    loose = step_size_table(data, alpha=0.1, max_substeps=10 ** 6)
    assert all(r["clamped_pct"] == 0.0 for r in loose)

    tight = step_size_table(data, alpha=0.1, max_substeps=16)
    assert any(r["clamped_pct"] > 0 for r in tight), "the clamp never bound; test is vacuous"
    by_decade = {r["decade"]: r for r in loose}
    n_strict = 0
    for r in tight:
        assert r["n_sub_max"] <= 16, "a reported count exceeds the clamp"
        unclamped = by_decade[r["decade"]]
        if r["clamped_pct"] > 50.0:
            # STRICTLY coarser, not merely "not finer". The >= version of this
            # assertion passed against a mutation that computed delta_t from
            # the UNCLAMPED count -- equality satisfies >=, so the test could
            # not tell the step actually taken from the step alpha asked for,
            # which is the single thing this column exists to report.
            assert r["delta_t_median"] > unclamped["delta_t_median"], (
                f"{r['decade']}: {r['clamped_pct']:.0f}% clamped, yet delta_t "
                f"matches the unclamped value -- the table is reporting a step "
                f"the run never took"
            )
            n_strict += 1
        else:
            assert r["delta_t_median"] >= unclamped["delta_t_median"]
    assert n_strict, "no decade was mostly clamped; the strict check never ran"


def test_depth_is_substeps_times_rollout_steps():
    """
    The quantity that overflows. Training pays it, inference does not, and it
    is why raising max_substeps made max_dt=2000 worse rather than better --
    a larger cap permits a deeper graph.
    """
    from evaluation.check_alpha import collect_alpha, step_size_table
    torch.manual_seed(0)
    data = collect_alpha(_decade_dataset(), _ProportionalField(), torch.device("cpu"))
    for n_roll in (1, 2, 4):
        rows = step_size_table(data, alpha=0.1, max_substeps=256, n_rollout_steps=n_roll)
        for r in rows:
            assert r["depth_max"] == pytest.approx(r["n_sub_max"] * n_roll)


def test_delta_over_t_falls_when_the_step_outpaces_the_slowdown():
    """
    THE SUBLINEARITY QUESTION. Coarsening slows as t grows, so the natural
    timescale grows with t. delta_t/t roughly constant across decades means
    the criterion follows that slowdown and reaching late t costs
    logarithmically many evaluations; falling means the step is refined faster
    than the physics slows and full-trajectory cost grows without bound.

    Here the save schedule is geometric (Delta_t ~ t) while |f|/|z1| is fixed,
    so n_substeps ~ Delta_t and delta_t saturates -- delta_t/t must therefore
    FALL. The test pins that the metric can detect the falling case at all;
    the real question is what it reports on real data.
    """
    from evaluation.check_alpha import collect_alpha, step_size_table
    torch.manual_seed(0)
    data = collect_alpha(_decade_dataset(), _ProportionalField(), torch.device("cpu"))
    rows = step_size_table(data, alpha=0.1, max_substeps=10 ** 6)
    ratios = [r["delta_over_t_median"] for r in rows]
    assert len(ratios) >= 3, "need several decades for the trend to mean anything"
    assert all(b < a for a, b in zip(ratios, ratios[1:])), (
        f"delta_t/t did not fall across decades: {ratios}"
    )


def test_the_table_defaults_to_the_checkpoints_own_alpha(tmp_path, monkeypatch, capsys):
    """
    The table must describe the run that PRODUCED the weights, not a
    hypothetical: reading alpha and max_substeps off the model is what makes
    it a diagnosis of the actual configuration rather than a what-if.
    """
    import pathlib

    import evaluation.check_alpha as mod
    field = _ProportionalField()
    field.alpha = 0.037
    field.max_substeps = 99
    monkeypatch.setattr(mod, "_load_ae_f_theta_and_dataset",
                         lambda *a, **k: (torch.device("cpu"), False, None, None,
                                          _decade_dataset(), None, field))
    result = mod.check_alpha(pathlib.Path("unused.pt"), latent_cache_dir=tmp_path / "latent_cache")
    assert result["report_alpha"] == 0.037
    assert result["max_substeps"] == 99
    out = capsys.readouterr().out
    assert "step size at alpha=0.037" in out
    assert "max_substeps=99" in out


def test_the_table_renders_every_column_without_nan(tmp_path, monkeypatch, capsys):
    """Same class of bug as the p95 column that was silently nan: the report
    is the product, and no unit test above renders it."""
    import pathlib

    import evaluation.check_alpha as mod
    monkeypatch.setattr(mod, "_load_ae_f_theta_and_dataset",
                         lambda *a, **k: (torch.device("cpu"), False, None, None,
                                          _decade_dataset(), None, _ProportionalField()))
    result = mod.check_alpha(pathlib.Path("unused.pt"), latent_cache_dir=tmp_path / "latent_cache")
    out = capsys.readouterr().out
    assert "step size at alpha=" in out
    assert "delta_t/t" in out and "depth" in out
    table_lines = [ln for ln in out.splitlines() if "1e2 - 1e3" in ln]
    assert table_lines, "the step-size table printed no decade rows"
    assert "nan" not in table_lines[0].lower(), table_lines[0]
    for r in result["step_size_rows"]:
        for key, v in r.items():
            if isinstance(v, float):
                assert np.isfinite(v), (key, r)


# --------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------

def test_the_figure_is_written_with_every_panel(monkeypatch, tmp_path):
    """Six panels, one file, and no exception on ordinary data."""
    import pathlib

    import matplotlib
    matplotlib.use("Agg")
    import evaluation.check_alpha as mod
    ds, f_theta = _synthetic_context()
    monkeypatch.setattr(mod, "_load_ae_f_theta_and_dataset",
                         lambda *a, **k: (torch.device("cpu"), False, None, None,
                                          ds, None, f_theta))
    out = tmp_path / "alpha.png"
    mod.check_alpha(pathlib.Path("unused.pt"), output_path=out, latent_cache_dir=tmp_path / "latent_cache")
    assert out.exists() and out.stat().st_size > 10_000


def test_no_figure_is_written_when_no_path_is_given(monkeypatch, tmp_path):
    """The tables must remain usable headless and cheap -- plotting is opt-in
    from the caller's side, so a pipeline that only wants numbers pays nothing."""
    import pathlib

    import evaluation.check_alpha as mod
    ds, f_theta = _synthetic_context()
    monkeypatch.setattr(mod, "_load_ae_f_theta_and_dataset",
                         lambda *a, **k: (torch.device("cpu"), False, None, None,
                                          ds, None, f_theta))
    called = []
    monkeypatch.setattr(mod, "_plot", lambda *a, **k: called.append(1))
    mod.check_alpha(pathlib.Path("unused.pt"), output_path=None, latent_cache_dir=tmp_path / "latent_cache")
    assert not called


def test_the_figure_survives_a_degenerate_field(monkeypatch, tmp_path):
    """
    f == 0 is the state of every fresh stage 3a (zero-init final layer), so
    every count is 1, every alpha is 0, and the log-scaled panels have nothing
    positive to draw. The figure must still be produced rather than raising
    partway through and leaving the caller with tables but no plot.
    """
    import pathlib

    import matplotlib
    matplotlib.use("Agg")
    import evaluation.check_alpha as mod

    class _Zero(torch.nn.Module):
        alpha = 0.1
        max_substeps = 256

        def f(self, z0, z1, theta):
            return torch.zeros_like(z0)

        def eval(self):
            return self

    ds, _ = _synthetic_context()
    monkeypatch.setattr(mod, "_load_ae_f_theta_and_dataset",
                         lambda *a, **k: (torch.device("cpu"), False, None, None,
                                          ds, None, _Zero()))
    out = tmp_path / "degenerate.png"
    mod.check_alpha(pathlib.Path("unused.pt"), output_path=out, latent_cache_dir=tmp_path / "latent_cache")
    assert out.exists()


def _draw_and_capture(monkeypatch, mod, *args):
    """Run _plot but keep the figure, so a test can read what was DRAWN.

    _plot closes its figure (correctly -- a diagnostic that leaks figures
    exhausts memory only after enough checkpoints). Intercepting close is the
    only way to inspect the result without weakening that.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kept = []
    monkeypatch.setattr(plt, "close", lambda f: kept.append(f))
    mod._plot(*args)
    assert kept, "_plot did not close (and therefore did not finish) its figure"
    return kept[0]


def test_the_plotted_step_respects_the_clamp(monkeypatch, tmp_path):
    """
    The figure must draw the step ACTUALLY taken: a clamped window ran coarser
    than alpha asked, and plotting the requested step would show the run doing
    something it never did.

    Reads the depth panel's SCATTER OFFSETS. Two earlier versions of this test
    recomputed the counts themselves and asserted on their own arithmetic --
    both passed against a mutation that removed the clamp from _plot entirely.
    A test of a plot has to read the plot.
    """
    import evaluation.check_alpha as mod
    from evaluation.check_alpha import collect_alpha, substeps_for_alpha
    torch.manual_seed(0)
    ds, f_theta = _synthetic_context()
    data = collect_alpha(ds, f_theta, torch.device("cpu"))
    max_substeps, n_roll = 16, 2
    assert (substeps_for_alpha(data, 0.001) > max_substeps).any(), (
        "the clamp would not bind; the test is vacuous"
    )

    fig = _draw_and_capture(monkeypatch, mod, data, 0.001, max_substeps, n_roll,
                             (0.1, 0.01), (7, 14), tmp_path / "clamped.png")
    # Row-major subplot order: [0,0] [0,1] [0,2] [1,0] [1,1] [1,2]; the depth
    # panel is the last of the six (a colorbar axis may follow it).
    depth_ax = fig.axes[5]
    drawn = depth_ax.collections[0].get_offsets()
    assert drawn.shape[0] > 0
    # EQUALITY, not <=: the clamp binds here, so the deepest window sits
    # exactly at the ceiling. The <= form passed against a mutation that
    # dropped n_rollout_steps from the depth (halving every value still
    # satisfies <=), which would have understated the training cost the panel
    # exists to show.
    assert float(drawn[:, 1].max()) == pytest.approx(max_substeps * n_roll), (
        f"the depth panel's maximum is {drawn[:, 1].max():.0f}, not the "
        f"{max_substeps * n_roll} a bound clamp implies -- either the clamp or "
        f"the x n_rollout_steps factor is missing"
    )
    assert float(drawn[:, 1].max()) <= max_substeps * n_roll, (
        f"the depth panel drew {drawn[:, 1].max():.0f}, above the clamp's "
        f"ceiling of {max_substeps * n_roll} -- the figure is showing a step "
        f"the run never took"
    )


def test_the_zero_field_panels_say_so_instead_of_drawing_nothing(monkeypatch, tmp_path):
    """
    With f == 0 (every fresh, zero-initialised stage 3a) the alpha histogram
    and the |f|/|z1| scatter have nothing to draw. Empty axes with a legend
    look like a rendering failure; the panels must state the cause, which is
    itself the most useful thing the figure can say about such a checkpoint.
    """
    import evaluation.check_alpha as mod
    from evaluation.check_alpha import collect_alpha

    class _Zero(torch.nn.Module):
        def f(self, z0, z1, theta):
            return torch.zeros_like(z0)

        def eval(self):
            return self

    ds, _ = _synthetic_context()
    data = collect_alpha(ds, _Zero(), torch.device("cpu"))
    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "zero.png")
    hist_texts = " ".join(t.get_text() for t in fig.axes[1].texts)
    assert "f_theta is identically zero" in hist_texts, (
        f"the alpha histogram drew empty axes instead of explaining why: {hist_texts!r}"
    )
    drive_texts = " ".join(t.get_text() for t in fig.axes[4].texts)
    assert "== 0 everywhere" in drive_texts, drive_texts


def test_plot_closes_its_figure(tmp_path):
    """
    A diagnostic that leaks figures makes a pipeline run out of memory only
    after enough checkpoints -- the worst kind of slow failure. Cheap to pin.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import evaluation.check_alpha as mod
    from evaluation.check_alpha import collect_alpha
    ds, f_theta = _synthetic_context()
    data = collect_alpha(ds, f_theta, torch.device("cpu"))
    before = set(plt.get_fignums())
    for i in range(3):
        mod._plot(data, 0.1, 256, 2, (0.2, 0.1, 0.05), (7, 14), tmp_path / f"f{i}.png")
    assert set(plt.get_fignums()) == before


def test_the_figure_defaults_into_the_stage_output_folder():
    """
    output/<stage>/<stem>-alpha.png, alongside -parameter_dependence.png and
    the rest -- NOT next to the .pt. An earlier default put it in
    checkpoints/, where nothing else in this project writes figures, breaking
    the one-folder-per-stage grouping the rest of output/ relies on.

    The stage comes from the checkpoint STEM, so a timestamped ancestor still
    lands in its own stage folder rather than a folder named after the
    timestamp.
    """
    import pathlib

    from evaluation.check_alpha import default_alpha_figure_path
    p = default_alpha_figure_path(
        pathlib.Path("checkpoints/stage3b/128x128-stage3b-20260806_18h10.pt"))
    assert p.parent.name == "stage3b", p
    assert p.parent.parent.name == "output", p
    assert p.name == "128x128-stage3b-20260806_18h10-alpha.png", p
    assert "checkpoints" not in p.parts, f"the figure would land in checkpoints/: {p}"

    q = default_alpha_figure_path(pathlib.Path("checkpoints/stage3a/64x64-stage3a.pt"))
    assert q.parent.name == "stage3a" and q.name == "64x64-stage3a-alpha.png", q


def test_the_output_directory_is_created(monkeypatch, tmp_path):
    """A stage folder that does not exist yet must not lose the figure --
    every other diagnostic mkdirs its parent, and a first run of a new stage
    is exactly when this bites."""
    import pathlib

    import matplotlib
    matplotlib.use("Agg")
    import evaluation.check_alpha as mod
    ds, f_theta = _synthetic_context()
    monkeypatch.setattr(mod, "_load_ae_f_theta_and_dataset",
                         lambda *a, **k: (torch.device("cpu"), False, None, None,
                                          ds, None, f_theta))
    out = tmp_path / "does" / "not" / "exist" / "alpha.png"
    mod.check_alpha(pathlib.Path("unused.pt"), output_path=out, latent_cache_dir=tmp_path / "latent_cache")
    assert out.exists()


# --------------------------------------------------------------------
# Presentation fixes: colour, axis units, tick density, break honesty
# --------------------------------------------------------------------

def test_the_scatter_and_the_reference_lines_do_not_share_a_colour(monkeypatch, tmp_path):
    """
    The eye reads shared colour as shared identity. With the scatter and the
    n=7 line both blue, the measured points looked like they BELONGED to that
    line -- in a panel whose entire claim is that they belong to neither.
    """
    import evaluation.check_alpha as mod
    from evaluation.check_alpha import collect_alpha
    ds, f_theta = _synthetic_context()
    data = collect_alpha(ds, f_theta, torch.device("cpu"))
    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "c.png")
    ax = fig.axes[0]
    scatter_colour = tuple(ax.collections[0].get_facecolor()[0][:3])
    line_colours = [tuple(ln.get_color() if isinstance(ln.get_color(), tuple)
                           else matplotlib_to_rgb(ln.get_color())) for ln in ax.lines]
    for lc in line_colours:
        assert not np.allclose(scatter_colour, lc, atol=0.02), (
            f"the scatter and a reference line share {lc}"
        )


def matplotlib_to_rgb(c):
    import matplotlib.colors as mcolors
    return mcolors.to_rgb(c)


def test_the_alpha_axis_shows_alpha_not_its_logarithm(monkeypatch, tmp_path):
    """
    A reader wants to see 0.1, not -1. log10(alpha) as the QUANTITY forces a
    mental exponentiation on every glance; a log SCALE shows the same spacing
    with readable numbers.
    """
    import evaluation.check_alpha as mod
    from evaluation.check_alpha import collect_alpha
    ds, f_theta = _synthetic_context()
    data = collect_alpha(ds, f_theta, torch.device("cpu"))
    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "a.png")
    ax = fig.axes[1]
    assert ax.get_xlabel() == "alpha", ax.get_xlabel()
    assert ax.get_xscale() == "log"
    # the marker line must sit at alpha itself, not at log10(alpha)
    verticals = [ln for ln in ax.lines if len(set(ln.get_xdata())) == 1]
    assert any(abs(float(ln.get_xdata()[0]) - 0.1) < 1e-9 for ln in verticals), (
        "the fixed-alpha marker is not at alpha=0.1 -- it is probably still at -1"
    )


def test_log_axes_carry_more_than_one_readable_tick(monkeypatch, tmp_path):
    """
    Delta_t runs 12-500: less than two decades, so matplotlib's decade-only
    locator labels a SINGLE tick and the reader cannot get a number off the
    axis at all. Shared with check_stdev_phi_time, which hit this first.
    """
    import evaluation.check_alpha as mod
    from evaluation.check_alpha import collect_alpha
    ds, f_theta = _synthetic_context()
    data = collect_alpha(ds, f_theta, torch.device("cpu"))
    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "t.png")
    for idx in (0, 5):        # delta_t-vs-Delta_t and the depth panel
        ax = fig.axes[idx]
        lo, hi = ax.get_xlim()
        labelled = [t for t in ax.get_xticks() if lo <= t <= hi]
        assert len(labelled) >= 2, (
            f"panel {idx} has {len(labelled)} labelled x tick(s): unreadable"
        )


def test_ticks_thin_out_over_a_wide_span():
    """
    {1,2,3,5} per decade is right below a decade and unreadable over three --
    the alpha histogram came out as "0.00050.0010.0020.003" run together. A
    labelled tick nobody can read is worse than the default it replaced.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.plots import log_axis_ticks
    fig, ax = plt.subplots()
    try:
        log_axis_ticks(ax.xaxis, 1e-4, 1e1)          # five decades
        wide = len(ax.get_xticks())
        # ABSOLUTE bound, not a ratio against the narrow case: {1,2,3,5} gives
        # ~4 per decade at BOTH spans, so a per-decade comparison is satisfied
        # by the un-thinned version and passed against its own mutation. What
        # matters is how many labels land on one axis: 20 across five decades
        # is the crowding that was actually observed.
        assert wide <= 8, f"{wide} labelled ticks across five decades will collide"
        log_axis_ticks(ax.xaxis, 0.05, 0.4)          # under one decade
        narrow = len(ax.get_xticks())
        assert narrow >= 3, (
            f"only {narrow} ticks under a decade -- thinning went too far and "
            f"the sub-decade case is unreadable again"
        )
    finally:
        plt.close(fig)


def test_a_break_is_only_reported_when_it_earns_its_parameters():
    """
    A broken power law has two extra parameters and can only fit better, so a
    small improvement means "one power law on noisy data". A knee, once drawn
    with a dotted line and a number, gets believed -- so the threshold matters
    more than the fit.
    """
    from evaluation._fits import fit_broken_power_law
    rng = np.random.default_rng(0)
    x = np.logspace(2, 4, 400)

    genuine = np.where(x < 1000, 1.0, (x / 1000) ** -0.5) * np.exp(rng.normal(0, 0.05, x.size))
    knee, p1, p2, sse_b, sse_s = fit_broken_power_law(x, genuine)
    assert 700 < knee < 1400, knee
    assert abs(p1) < 0.1 and abs(p2 + 0.5) < 0.1, (p1, p2)
    assert 1 - sse_b / sse_s > 0.5, "a real break should buy a lot"

    single = x ** -0.3 * np.exp(rng.normal(0, 0.05, x.size))
    _, _, _, sse_b2, sse_s2 = fit_broken_power_law(x, single)
    assert 1 - sse_b2 / sse_s2 < 0.25, (
        "a pure power law bought a large improvement from a spurious break"
    )


def _data_with_slope(slope: float, n: int = 1200, knee_at: float | None = None,
                      slope_after: float = -1.0):
    """A `data` dict whose delta_t/t follows a KNOWN law.

    Built directly rather than through _synthetic_context, because that
    fixture's own delta_t/t genuinely bends -- a geometric save schedule with
    decaying |z1| produces a real crossover -- so it cannot serve as the
    single-regime control. Testing a threshold needs data whose answer is
    known by construction, in both directions.
    """
    t = np.logspace(2, 4, n)
    dt = t / 8.0
    # Work in COUNTS, not in the target ratio: delta_t/t = (dt/t)/count, and
    # dt/t is constant here, so a power law in the count is a power law in the
    # ratio. Starting at 30 rather than 1 keeps ceil() a ~3% perturbation --
    # my first version let the count hit its floor of 1 at small t, which
    # manufactured a flat regime and then failed the single-regime test for a
    # reason that had nothing to do with the threshold under test.
    counts = 30.0 * (t / t[0]) ** (-slope)
    if knee_at is not None:
        counts = np.where(t < knee_at,
                           30.0 * (t / t[0]) ** (-slope),
                           30.0 * (knee_at / t[0]) ** (-slope)
                           * (t / knee_at) ** (-slope_after))
    counts = np.ceil(counts)
    assert counts.min() > 1, "the count hit its floor; the law is not what it claims"
    # invert the criterion so substeps_for_alpha reproduces those counts
    z1_norm = np.full(n, 1.0)
    f_norm = counts * 0.1 * z1_norm / dt
    return {"f_norm": f_norm, "z1_norm": z1_norm, "dt": dt, "t": t,
            "theta0": np.full(n, -0.22)}


def test_the_panel_does_not_draw_a_knee_it_cannot_justify(monkeypatch, tmp_path):
    """
    The threshold, tested where it is APPLIED. Testing fit_broken_power_law
    alone says nothing about whether the panel believes it -- and a knee, once
    drawn with a dotted line and a number in the title, is what gets believed.
    """
    import evaluation.check_alpha as mod
    data = _data_with_slope(-0.3)
    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "one.png")
    title = fig.axes[3].get_title()
    assert "one regime" in title, f"a knee was reported on single-regime data: {title!r}"
    assert "knee" not in title


def test_the_panel_does_report_a_knee_that_is_there(monkeypatch, tmp_path):
    """
    The other half: a threshold that never fires is just a disabled feature.
    Flat then falling -- the shape the real scatter suggested and the reason
    this fit exists.
    """
    import evaluation.check_alpha as mod
    data = _data_with_slope(0.0, knee_at=1000.0, slope_after=-0.8)
    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "two.png")
    ax = fig.axes[3]
    title = ax.get_title()
    assert "at t=" in title, f"a real break went unreported: {title!r}"
    assert "one regime" not in title
    knee_lines = [ln for ln in ax.lines
                   if len(set(ln.get_xdata())) == 1 and ln.get_linestyle() == ":"]
    assert knee_lines, "no knee marker was drawn"
    assert 400 < float(knee_lines[0].get_xdata()[0]) < 2500, (
        f"knee at {knee_lines[0].get_xdata()[0]:.3g}, nowhere near the true 1000"
    )


def test_a_regime_covering_a_sliver_of_the_range_is_refused():
    """
    THE FRACTIONAL GUARD. An absolute floor of 8 points is nothing at 1800:
    on the real fixture it let a "regime" be eight early windows and fitted
    them a slope of +3.76, with a 30% apparent improvement -- a knee a reader
    would have believed. A regime covering under a sixth of the range is a
    tail artefact whatever it does to the SSE.
    """
    from evaluation._fits import fit_broken_power_law
    rng = np.random.default_rng(3)
    x = np.logspace(2, 4, 1800)
    y = x ** -0.3 * np.exp(rng.normal(0, 0.05, x.size))
    y[:10] *= 40.0                      # a sliver of wild points at the start

    loose = fit_broken_power_law(x, y, min_side=8, min_side_fraction=0.0)
    tight = fit_broken_power_law(x, y, min_side=8, min_side_fraction=0.15)
    assert loose[0] < tight[0], (
        "the fractional guard did not push the knee away from the sliver "
        f"(unguarded {loose[0]:.3g}, guarded {tight[0]:.3g})"
    )
    # The guarded knee lands ON the boundary, so compare with a tolerance
    # rather than a bare >= -- the exact float is the boundary itself.
    assert tight[0] >= x[int(0.15 * x.size)] * (1 - 1e-9), (
        f"guarded knee {tight[0]:.4g} inside the first 15% "
        f"(boundary {x[int(0.15 * x.size)]:.4g})"
    )
    # And the point of the guard: the unguarded fit invents a violent slope to
    # explain the sliver, which is what makes a spurious knee look convincing.
    assert loose[1] < -10, f"the unguarded fit did not chase the outliers: p1={loose[1]:.3g}"
    assert abs(tight[1]) < 2, f"the guarded fit still has a wild slope: p1={tight[1]:.3g}"


def test_the_broken_fit_refuses_a_knee_at_the_very_edge():
    """min_side keeps at least a few points on each side, so a 'regime change'
    cannot be fitted to two outliers at one end."""
    from evaluation._fits import fit_broken_power_law
    x = np.logspace(2, 4, 200)
    y = x ** -0.3
    # A few wild points at the very end. Unguarded, the best two-segment fit
    # hinges right before them and "explains" the outliers with a regime
    # change; that is precisely what min_side exists to forbid. An exact
    # power law with no outliers leaves the search indifferent between knees,
    # so it cannot distinguish the guard from its absence -- my first version
    # of this test used one, and passed against the mutation.
    y[-3:] *= 30.0
    # min_side_fraction=0 isolates the ABSOLUTE floor, which is what this test
    # is about. Leaving the fractional guard active made the test pass with
    # the floor removed -- the fraction alone was doing the work, so the
    # assertion said nothing about min_side.
    knee, _, _, _, _ = fit_broken_power_law(x, y, min_side=20, min_side_fraction=0.0)
    assert x[20] <= knee <= x[-21], (
        f"knee {knee:.3g} outside the guarded interior [{x[20]:.3g}, {x[-21]:.3g}] "
        f"-- it was fitted to the trailing outliers"
    )


def test_the_drive_panel_reports_the_exponent_it_fits(monkeypatch, tmp_path):
    """
    The panel's whole content is the exponent p in |f|/|z1| ~ Delta_t^p,
    because n ~ (|f|/|z1|)*Delta_t means the COUNT grows as Delta_t^(1+p) --
    the number that decides whether widening max_dt is affordable. Reading it
    off by eye produced a wrong guess (-0.6 claimed against the fit); the
    label must carry the fitted value.

    Constructed with a KNOWN exponent, so the label is checked against truth
    rather than against itself.
    """
    import evaluation.check_alpha as mod
    from evaluation.check_alpha import collect_alpha

    class _PowerLaw(torch.nn.Module):
        """|f| chosen so that |f|/|z1| ~ Delta_t^-0.5 exactly."""

        def __init__(self, dt_of_frame):
            super().__init__()
            self.dt = dt_of_frame
            self.i = 0

        def f(self, z0, z1, theta):
            b = z0.shape[0]
            dt = torch.as_tensor(self.dt[self.i:self.i + b], dtype=torch.float32)
            self.i += b
            return z1 * (dt ** -0.5).view(-1, 1, 1, 1) * 0.01

        def eval(self):
            return self

    ds, _ = _synthetic_context(n_runs=10, n_steps=12)
    # dt each frame will see, in the order collect_alpha walks them
    dts = []
    for steps, scale in zip(ds._run_steps, ds._run_dt_scale):
        dts += [(steps[i + 1] - steps[i]) * scale for i in range(len(steps) - 1)]
    data = collect_alpha(ds, _PowerLaw(np.array(dts)), torch.device("cpu"))

    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "drive.png")
    labels = " ".join(t.get_text() for t in fig.axes[4].get_legend().get_texts())
    assert "Delta_t^-0.50" in labels, (
        f"the fitted exponent is not the -0.50 the data was built with: {labels!r}"
    )
    # and the derived count exponent, which is the actionable one
    assert "n ~ Delta_t^+0.50" in labels, labels

    # THE DRAWN LINE, not just its label. A label states the fit; only the
    # line's own slope shows the fit was plotted -- a mutation that drew a
    # flat line left the label intact and passed.
    line = fig.axes[4].get_lines()[0]
    lx, ly = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    slope = np.polyfit(np.log(lx), np.log(ly), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.02), (
        f"the drawn regression has slope {slope:+.2f}, not the -0.50 its label "
        f"claims -- the line and the fit have come apart"
    )


def test_the_alpha_histogram_does_not_claim_the_curves_differ_in_shape(monkeypatch, tmp_path):
    """
    alpha_at_n = D/n with D = |f|*Delta_t/|z1|, so the curves for different n
    are ONE distribution translated on a log axis. I described one as "clearly
    bimodal" and the other as "much less so" -- impossible by construction.
    Pinned as a property of the data, not just of the wording: the two
    histograms must have identical bin counts once shifted.
    """
    import evaluation.check_alpha as mod
    from evaluation.check_alpha import alpha_at_substeps, collect_alpha
    ds, f_theta = _synthetic_context()
    data = collect_alpha(ds, f_theta, torch.device("cpu"))
    a7 = alpha_at_substeps(data, 7)
    a14 = alpha_at_substeps(data, 14)
    ok = np.isfinite(a7) & (a7 > 0)
    assert np.allclose(a14[ok], a7[ok] / 2.0), (
        "alpha at fixed n is not a pure rescaling -- the panel's premise is wrong"
    )
    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "hist.png")
    title = fig.axes[1].get_title()
    assert "same shape, shifted" in title, title


def test_max_dt_is_marked_on_every_panel_with_a_dt_axis(monkeypatch, tmp_path):
    """
    max_dt is a DATA-SELECTION boundary, and without it drawn the eye reads
    the edge of the scatter as where the physics stops rather than where the
    filter cut. Four panels put Delta_t on an axis; all four must show it.
    """
    import evaluation.check_alpha as mod
    from evaluation.check_alpha import collect_alpha
    ds, f_theta = _synthetic_context()
    data = collect_alpha(ds, f_theta, torch.device("cpu"))
    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "m.png", 500.0)

    # [0,0] delta_t vs Delta_t, [1,1] drive, [1,2] depth -- axes 0, 4, 5.
    for idx in (0, 4, 5):
        verticals = [ln for ln in fig.axes[idx].get_lines()
                     if len(set(np.round(ln.get_xdata(), 6))) == 1
                     and float(np.asarray(ln.get_xdata())[0]) == pytest.approx(500.0)]
        assert verticals, f"axes[{idx}] has no max_dt line"
        assert verticals[0].get_linestyle() in (":", "dotted"), (
            f"axes[{idx}]'s max_dt line is not dotted: {verticals[0].get_linestyle()}"
        )


def test_no_max_dt_line_when_the_population_is_unbounded(monkeypatch, tmp_path):
    """
    A dataset built without max_dt (or a checkpoint that predates it) has no
    such boundary -- drawing one at an arbitrary place would invent a filter
    that never ran.
    """
    import evaluation.check_alpha as mod
    from evaluation.check_alpha import collect_alpha
    ds, f_theta = _synthetic_context()
    data = collect_alpha(ds, f_theta, torch.device("cpu"))
    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "n.png", None)
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "max_dt" not in texts, texts


def test_the_bracket_configurations_keep_one_colour_across_panels(monkeypatch, tmp_path):
    """
    [0,0] and [0,1] show the SAME two configurations -- as a step-size curve
    and as a distribution. Independent colour cycles made n=7 red in one and
    blue in the other, which invites reading the panels as being about
    different things.
    """
    import evaluation.check_alpha as mod
    from evaluation.check_alpha import _BRACKET_COLOURS, collect_alpha
    ds, f_theta = _synthetic_context()
    data = collect_alpha(ds, f_theta, torch.device("cpu"))
    fig = _draw_and_capture(monkeypatch, mod, data, 0.1, 256, 2,
                             (0.2, 0.1), (7, 14), tmp_path / "c.png", 500.0)

    step_lines = {ln.get_label(): ln.get_color() for ln in fig.axes[0].get_lines()
                  if ln.get_label().startswith("fixed n_substeps=")}
    hist_patches = {}
    handles, labels = fig.axes[1].get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label.startswith("fixed n_substeps="):
            hist_patches[label] = handle.get_edgecolor()

    assert set(step_lines) == set(hist_patches) != set(), (step_lines, hist_patches)
    import matplotlib.colors as mcolors
    for label, line_colour in step_lines.items():
        assert mcolors.to_rgba(line_colour) == pytest.approx(
            mcolors.to_rgba(hist_patches[label]), abs=1e-6), (
            f"{label} is {line_colour} in the step panel and "
            f"{hist_patches[label]} in the histogram"
        )
    assert set(step_lines.values()) <= set(_BRACKET_COLOURS)
