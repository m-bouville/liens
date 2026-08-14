import numpy as np
import pytest
import torch

import evaluation.check_z2_measurability as cz2


def _run_with_synthetic_field(tmp_path, monkeypatch, curvature_scale,
                               velocity_scale=1e-3, n_windows=200,
                               noise=1e-3, record_kwargs=None,
                               spread_decades=False):
    """Drive check_z2_measurability end to end over a KNOWN latent field.

    The tool's own verdict is the thing under test, so everything upstream
    of the measurement (checkpoint loading, encoder construction, run-dir
    validation, metadata) is stubbed and only the stencils, the statistics
    and the reporting run for real. Frames are laid out on a geometric
    schedule like the real save list, and each window gets its own
    curvature drawn at `curvature_scale` -- 0.0 gives a field that is pure
    encoding noise.
    """
    import types
    from pathlib import Path

    class _Dataset:
        def __init__(self, *args, **kwargs):
            if record_kwargs is not None:
                record_kwargs.update(kwargs)
            rng = np.random.default_rng(0)
            self._items = []
            for i in range(n_windows):
                # spread_decades: windows cycle through dt decades, so the
                # per-decade table has more than one row to compare. The
                # scaled columns are ONLY meaningful across decades.
                base = 10.0 * (10 ** (i % 3)) if spread_decades else 400.0
                dts = ([base] * 5 if spread_decades
                       else [400.0, 460.0, 530.0, 610.0, 700.0])
                t = np.cumsum([0.0] + list(dts))
                k = rng.normal(scale=curvature_scale)
                v = rng.normal(scale=velocity_scale)
                frames = [(0.5 * k * ti * ti + v * ti) * np.ones((2, 4, 4))
                          for ti in t]
                frames = [f + rng.normal(scale=noise, size=(2, 4, 4))
                          for f in frames]
                self._items.append((
                    torch.tensor(np.stack(frames), dtype=torch.float32),
                    torch.tensor(dts), torch.zeros(1)))

        def __len__(self):
            return len(self._items)

        def __getitem__(self, i):
            return self._items[i]

        def window_info(self, i):
            return (Path(f"T{600 + i % 300}_n020_s{i}"), list(range(6)))

    monkeypatch.setattr(cz2, "MicrostructureEvolutionDataset", _Dataset)
    monkeypatch.setattr(cz2, "build_ae_from_checkpoint",
                         lambda path, device: (None,) * 5)
    monkeypatch.setattr(cz2.load, "validate_run_dirs",
                         lambda dirs, source=None, min_stdev_phi=None: dirs)
    monkeypatch.setattr(cz2.load, "read_metadata", lambda path: types.SimpleNamespace(
        temperature=int(str(path).split("T")[1][:3]) / 1000))
    ckpt = tmp_path / "ck.pt"
    torch.save({"test_dirs": ["d"], "data_config": {}}, ckpt)
    return cz2.check_z2_measurability(ckpt, device="cpu", n_windows=n_windows)


def test_second_difference_is_exact_for_a_quadratic_at_uneven_spacing():
    """
    The save schedule is geometric, so dt_+ != dt_- for essentially every
    window. The nonuniform form is therefore load-bearing, not a nicety:
    the uniform-spacing formula carries an O(dt_+ - dt_-) first-derivative
    leak and would report z1's spacing asymmetry as curvature.
    """
    a, b, c = 0.7, -0.3, 2.1          # z'' = 2a exactly
    f = lambda x: a * x * x + b * x + c
    t = 1000.0
    for dt_minus, dt_plus in ((100.0, 100.0), (100.0, 135.0), (500.0, 150.0),
                               (37.0, 911.0)):
        got = cz2.second_difference(
            torch.tensor(f(t - dt_minus), dtype=torch.float64),
            torch.tensor(f(t), dtype=torch.float64),
            torch.tensor(f(t + dt_plus), dtype=torch.float64),
            dt_minus, dt_plus)
        assert abs(float(got) - 2 * a) < 1e-12, (
            f"dt-={dt_minus} dt+={dt_plus}: got {float(got)}, want {2 * a}"
        )


def test_the_uniform_formula_would_be_badly_wrong_here():
    """Documents WHY the general form is used: on a realistic geometric
    window the naive uniform stencil is off by hundreds of percent, i.e. it
    would manufacture curvature out of uneven spacing alone."""
    a, b, c = 0.7, -0.3, 2.1
    f = lambda x: a * x * x + b * x + c
    t, dt_minus, dt_plus = 1000.0, 100.0, 135.0
    h = 0.5 * (dt_minus + dt_plus)
    naive = (f(t + dt_plus) - 2 * f(t) + f(t - dt_minus)) / (h * h)
    assert abs(naive - 2 * a) / (2 * a) > 1.0, (
        "the uniform formula is not actually wrong on this window, so the "
        "nonuniform form would not be justified by it"
    )


def test_bias_fraction_separates_consistent_from_cancelling_error():
    """|E[x]|/E[|x|] near 1 for a consistent direction, near 0 for one that
    cancels -- the distinction E[|x|] alone cannot make."""
    rng = np.random.default_rng(0)
    consistent = np.full((200, 4, 4), 3.0) + rng.normal(scale=0.01, size=(200, 4, 4))
    _b, _t, frac = cz2._bias_fraction(consistent)
    assert frac > 0.95, frac

    cancelling = rng.normal(scale=3.0, size=(200, 4, 4))
    _b, _t, frac = cz2._bias_fraction(cancelling)
    assert frac < 0.2, frac


def test_agreement_is_high_for_real_curvature_and_zero_for_noise():
    """
    THE DISCRIMINATING TEST. Two stencils centred on the same frame see the
    same curvature through independent noise. If the underlying field is
    real they correlate; if both are just encoding noise they do not. This
    is what makes the diagnostic able to return "z2 is unlearnable" rather
    than only ever returning a number.
    """
    rng = np.random.default_rng(1)
    shape = (300, 8, 8)
    # real curvature, sign-varying across windows (so the bias fraction
    # would NOT reveal it -- only the agreement does)
    curvature = rng.normal(scale=1.0, size=(300, 1, 1)) * np.ones((1, 8, 8))
    narrow = curvature + rng.normal(scale=0.05, size=shape)
    wide = curvature + rng.normal(scale=0.05, size=shape)
    assert cz2._corr(narrow, wide) > 0.9

    # and the bias fraction is blind to it, which is the ambiguity the
    # agreement test exists to resolve
    _b, _t, frac = cz2._bias_fraction(narrow)
    assert frac < 0.2, (
        f"bias fraction {frac:.3f} -- the fixture was meant to be "
        f"sign-varying, so this test no longer demonstrates the ambiguity"
    )

    # pure noise: two independent estimates of nothing
    narrow_n = rng.normal(scale=1.0, size=shape)
    wide_n = rng.normal(scale=1.0, size=shape)
    assert abs(cz2._corr(narrow_n, wide_n)) < 0.15


def test_corr_handles_degenerate_input():
    """A constant field has zero variance -- correlation is undefined, and
    must not raise or silently return 0 as if it had been measured."""
    const = np.ones((10, 3, 3))
    assert np.isnan(cz2._corr(const, const))
    assert np.isnan(cz2._corr(np.array([1.0]), np.array([2.0])))


def test_pure_noise_scores_near_zero_agreement(tmp_path, monkeypatch):
    """
    THE FIX FOR A FALSE POSITIVE, tested through the ACTUAL OUTPUT rather
    than the source text. The obvious design -- two stencils centred on the
    same frame -- puts that frame's encoding noise into both estimates with
    a large coefficient, correlating them at 4/sqrt(6*6) = 0.667 with no
    curvature present at all. A diagnostic whose "no signal" answer is 0.67
    cannot report "unlearnable", which is the answer it exists to give.

    So: feed a field that is PURE NOISE and require the verdict to be ~0.
    This holds whatever the stencils are, as long as they share no frame --
    it constrains the property that matters, not the spelling.
    """
    out = _run_with_synthetic_field(tmp_path, monkeypatch, curvature_scale=0.0)
    assert abs(out["agreement"]) < 0.15, (
        f"a pure-noise field scored {out['agreement']:+.3f} agreement -- the "
        f"stencils are sharing a frame, so the tool cannot report "
        f"'unlearnable'"
    )
    assert abs(out["per_window"]["median"]) < 0.15, out["per_window"]


def test_real_curvature_is_detected(tmp_path, monkeypatch):
    """The counterpart: a field with real curvature must score high, or the
    disjoint design has traded a false positive for a false negative."""
    out = _run_with_synthetic_field(tmp_path, monkeypatch, curvature_scale=1e-6)
    assert out["per_window"]["median_cosine"] > 0.5, (
        f"real curvature scored only {out['per_window']['median_cosine']:+.3f} "
        f"-- the tool cannot detect the signal it exists to find"
    )


def test_shared_centre_stencils_would_correlate_on_pure_noise():
    """The quantitative reason for the disjoint design, as a live check
    rather than a claim in a comment: shared-centre stencils score ~2/3 on
    noise, disjoint ones ~0."""
    rng = np.random.default_rng(3)
    n, shape = 4000, (2, 2)
    # five frames of pure encoding noise, no curvature whatsoever
    z = rng.normal(size=(n, 5) + shape)
    h = 1.0
    shared_a = (z[:, 1] - 2 * z[:, 2] + z[:, 3]) / h ** 2
    shared_b = (z[:, 0] - 2 * z[:, 2] + z[:, 4]) / (4 * h ** 2)
    assert cz2._corr(shared_a, shared_b) > 0.5, (
        "shared-centre stencils did not show the noise floor, so this test "
        "no longer motivates the disjoint design"
    )
    z6 = rng.normal(size=(n, 6) + shape)
    disjoint_a = (z6[:, 0] - 2 * z6[:, 1] + z6[:, 2]) / h ** 2
    disjoint_b = (z6[:, 3] - 2 * z6[:, 4] + z6[:, 5]) / h ** 2
    assert abs(cz2._corr(disjoint_a, disjoint_b)) < 0.1


def test_the_dataset_is_asked_for_six_frame_windows(tmp_path, monkeypatch):
    """A 3-frame window gives the bias fraction but no agreement test; a
    5-frame window only fits stencils that share a centre. Asserted on the
    REQUEST the tool makes, not on its source text."""
    seen = {}
    _run_with_synthetic_field(tmp_path, monkeypatch, curvature_scale=0.0,
                               record_kwargs=seen)
    assert seen.get("window_length") == 6, (
        f"the dataset was built with window_length={seen.get('window_length')}, "
        f"which cannot hold two disjoint 3-point stencils"
    )


def test_first_difference_is_exact_for_a_quadratic_at_uneven_spacing():
    """The velocity control needs the same nonuniform treatment as the
    curvature stencil, for the same reason: geometric spacing."""
    a, b, c = 0.7, -0.3, 2.1
    f = lambda x: a * x * x + b * x + c
    fp = lambda x: 2 * a * x + b
    t = 1000.0
    for dt_minus, dt_plus in ((100.0, 100.0), (100.0, 135.0), (500.0, 150.0)):
        got = cz2.first_difference(
            torch.tensor(f(t - dt_minus), dtype=torch.float64),
            torch.tensor(f(t), dtype=torch.float64),
            torch.tensor(f(t + dt_plus), dtype=torch.float64),
            dt_minus, dt_plus)
        assert abs(float(got) - fp(t)) < 1e-9, (dt_minus, dt_plus, float(got))


def test_the_noise_invariant_is_magnitude_times_dt_squared():
    """
    Noise in a second difference is sqrt(6)*sigma/dt^2, so E|z2|*dt^2 is
    CONSTANT under the noise hypothesis while E|z2| itself falls as 1/dt^2.
    That is the direct test of what the estimate is made of, independent of
    any correlation.

    The exponent is load-bearing: |z2|^2*dt^2 -- the plausible-looking
    alternative -- still falls as 1/dt^2 and would read as a decaying
    signal where there is none.
    """
    rng = np.random.default_rng(0)
    scaled, wrong = [], []
    for h in (10.0, 100.0, 1000.0, 10000.0):
        z = rng.normal(scale=1.0, size=(20000, 3))
        z2 = (z[:, 0] - 2 * z[:, 1] + z[:, 2]) / h ** 2
        scaled.append(float(np.abs(z2).mean()) * h ** 2)
        wrong.append(float((z2 ** 2).mean()) * h ** 2)
    assert max(scaled) / min(scaled) < 1.1, (
        f"|z2|*dt^2 is not constant on pure noise: {scaled}"
    )
    assert max(wrong) / min(wrong) > 100, (
        f"|z2|^2*dt^2 was expected to still decay, so that the exponent "
        f"choice matters: {wrong}"
    )


def test_per_window_agreement_is_not_variance_weighted():
    """
    THE FIX FOR THE POOLED STATISTIC. A few high-|z2| windows must not
    decide the answer: on real data the pooled correlation landed on top of
    the smallest-dt decade's value while that decade held 8% of the sample.
    The median counts every window once.
    """
    rng = np.random.default_rng(2)
    # 90 quiet windows that agree, 10 loud ones that disagree
    quiet = rng.normal(scale=1.0, size=(90, 4, 4))
    quiet_b = quiet + rng.normal(scale=0.1, size=(90, 4, 4))
    loud = rng.normal(scale=100.0, size=(10, 4, 4))
    loud_b = -loud + rng.normal(scale=1.0, size=(10, 4, 4))
    a = np.concatenate([quiet, loud])
    b = np.concatenate([quiet_b, loud_b])

    pooled = cz2._corr(a, b)
    per_window = cz2._per_window_agreement(a, b)
    assert pooled < -0.5, (
        f"pooled correlation {pooled:.3f} was expected to be captured by the "
        f"10 loud windows, so this test no longer demonstrates the problem"
    )
    assert per_window["median"] > 0.8, (
        f"per-window median {per_window['median']:.3f} followed the loud "
        f"windows too -- it is still variance-weighted"
    )
    assert per_window["frac_negative"] == pytest.approx(0.10, abs=0.01)


def test_per_window_agreement_reports_centred_and_uncentred():
    """
    Centring removes a spatially UNIFORM curvature component, which for a
    coarsening field may be signal rather than nuisance. Observed on a
    synthetic field whose curvature was entirely uniform: the centred
    correlation read -0.009 (indistinguishable from noise) while the cosine
    read +0.54. Reporting only one would let that choice decide the verdict.
    """
    rng = np.random.default_rng(5)
    uniform = rng.normal(scale=1.0, size=(200, 1, 1)) * np.ones((1, 4, 4))
    a = uniform + rng.normal(scale=0.02, size=(200, 4, 4))
    b = uniform + rng.normal(scale=0.02, size=(200, 4, 4))
    out = cz2._per_window_agreement(a, b)
    assert abs(out["median"]) < 0.3, out["median"]
    assert out["median_cosine"] > 0.9, out["median_cosine"]


def test_per_window_agreement_survives_degenerate_windows():
    """A constant window has zero variance -- undefined correlation. It must
    be skipped, not counted as zero agreement."""
    a = np.concatenate([np.ones((3, 2, 2)), np.random.default_rng(0).normal(size=(5, 2, 2))])
    b = np.concatenate([np.ones((3, 2, 2)), np.random.default_rng(1).normal(size=(5, 2, 2))])
    out = cz2._per_window_agreement(a, b)
    assert out["n"] == 5, f"degenerate windows were not skipped: n={out['n']}"


def test_first_derivative_control_reports_the_cosine(tmp_path, monkeypatch,
                                                      capsys):
    """
    REGRESSION (audit finding): the control printed only the CENTRED
    per-window correlation. Velocity in a coarsening field has a large
    coherent component -- the causal baseline holds 93% correlation on
    exactly these spans -- and centring subtracts precisely that, so the
    control read +0.04 on data whose velocity was demonstrably persistent,
    and that was then read as "the trajectory is rough at first order too".

    Tested on OUTPUT: a field built with a strongly coherent velocity must
    have its control report a high cosine, whatever the centred number does.
    """
    out = _run_with_synthetic_field(tmp_path, monkeypatch, curvature_scale=0.0,
                                     velocity_scale=1.0)
    printed = capsys.readouterr().out
    assert "cosine" in printed.split("FIRST-derivative control")[1][:200], (
        "the first-derivative control does not print a cosine"
    )
    assert out["per_window_first_deriv"]["median_cosine"] > 0.5, (
        f"a strongly coherent velocity field reported cosine "
        f"{out['per_window_first_deriv']['median_cosine']:+.3f} -- the "
        f"control cannot see persistence that is actually there"
    )


def test_first_difference_noise_invariant_is_magnitude_times_dt():
    """
    Noise in a FIRST difference is sqrt(2)*sigma/dt, so E|d1|*dt is constant
    under the noise hypothesis while E|d1| falls as 1/dt -- one power of dt
    less than the second difference's invariant. Getting the exponent from
    the second-derivative case would make a noise-dominated velocity look
    like a decaying signal.
    """
    rng = np.random.default_rng(0)
    scaled, raw = [], []
    for h in (10.0, 100.0, 1000.0, 10000.0):
        z = rng.normal(scale=1.0, size=(20000, 2))
        d1 = (z[:, 1] - z[:, 0]) / h
        scaled.append(float(np.abs(d1).mean()) * h)
        raw.append(float(np.abs(d1).mean()))
    assert max(scaled) / min(scaled) < 1.1, f"E|d1|*dt not constant: {scaled}"
    assert raw[0] / raw[-1] > 100, f"E|d1| did not fall with dt: {raw}"


def test_velocity_columns_separate_a_real_field_from_noise(tmp_path, monkeypatch,
                                                            capsys):
    """
    The two scaled columns together distinguish two very different
    diagnoses. Curvature-only noise means a smooth trajectory encoded
    wiggly -- L_interp's target. Velocity noise TOO means z0's own
    frame-to-frame encoding noise swamps the dynamics, and the fix is
    upstream in stage 1 rather than in stage 2's objective.
    """
    _run_with_synthetic_field(tmp_path, monkeypatch, curvature_scale=0.0,
                               velocity_scale=0.0, noise=1e-3,
                               n_windows=300, spread_decades=True)
    noise_out = capsys.readouterr().out
    assert "E|d1|*dt" in noise_out, "the velocity columns are not printed"

    # a REAL velocity must make the scaled column grow, on the same fixture
    _run_with_synthetic_field(tmp_path, monkeypatch, curvature_scale=0.0,
                               velocity_scale=1e-4, noise=1e-3,
                               n_windows=300, spread_decades=True)
    signal_out = capsys.readouterr().out

    def scaled_column(text):
        rows = [ln for ln in text.split("\n") if ln.strip().startswith("1e")]
        return [float(ln.split()[-1]) for ln in rows]

    noise_col, signal_col = scaled_column(noise_out), scaled_column(signal_out)
    assert len(noise_col) >= 2 and len(signal_col) >= 2
    # flat under noise, growing under a real velocity
    assert max(noise_col) / min(noise_col) < 3.0, (
        f"E|d1|*dt was expected flat on pure noise, got {noise_col}"
    )
    assert max(signal_col) / min(signal_col) > 5.0, (
        f"E|d1|*dt was expected to grow with a real velocity, got {signal_col}"
    )
