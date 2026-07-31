"""
Tests for evaluation/check_stdev_phi_time.py.

Like check_stdev_phi_temperature.py, this diagnostic is CHECKPOINT-FREE
-- it reads statistics.csv and metadata.txt only -- so the fixture is
cheap and needs no model, GPU, or encoder. It does need precise control
over each run's temperature, Landau parameters, noise, seed, and its own
stdev_phi(t) curve, which no existing checkpoint-based fixture provides.

The central test builds a sweep whose temperature dependence is a KNOWN
PURE TIME RESCALING and asserts the collapse is detected, alongside its
negative twin (curves differing in SHAPE, not just timing) where the
same code must NOT report a collapse. Each is written so it genuinely
fails if the collapse logic is broken -- see the per-test comments.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_check_stdev_phi_time.py -v
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.check_stdev_phi_time import (
    _characteristic_time, _curve_spread, _delta_t_at, _observed_plateau,
    check_stdev_phi_time,
)

SIZE = 32
# Log-spaced, mimicking the real sweep's own schedule shape (which is
# what makes t and Delta_t collinear in the first place) rather than a
# uniform grid that would not exercise the log interpolation at all.
STEPS = [0, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
T0 = 1.0


def _build_run(base_dir: Path, name: str, temperature: float, stdev_by_step: dict[int, float],
                noise: float = 0.01, seed: int = 0, a0: float = 1.0, b: float = 1.0,
                complete: bool = True, with_stats: bool = True,
                steps: list[int] | None = None,
                autocorr_by_step: dict[int, float] | None = None) -> None:
    """One run directory. stdev_phi is given EXPLICITLY per step rather
    than generated from a field: every assertion here is about how those
    specific numbers are averaged, normalized and rescaled, and deriving
    them from a synthetic field would make expected results depend on the
    generator instead of on this diagnostic's own logic."""
    steps = STEPS if steps is None else steps
    run_dir = base_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.txt").write_text("\n".join([
        f"directory = {name}", "code version = test",
        f"status = {'complete' if complete else 'running'}",
        f"Nx = {SIZE}", f"Ny = {SIZE}", "dt = 0.05", f"steps = {steps[-1]}",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        f"a0 = {a0}", f"b = {b}", f"T0 = {T0}", f"temperature = {temperature}",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", f"noise = {noise}",
        f"seed = {seed}", "equation = allen_cahn", "solver = explicit", "",
    ]))
    if with_stats:
        rows = [{"step": s, "stdev_phi": stdev_by_step[s]} for s in steps]
        if autocorr_by_step is not None:
            # int, not float: the real statistics.csv records whole pixels,
            # so pandas infers int64 for this column. Writing floats here
            # made the fixture accept values the real data would not --
            # np.int64 does not subclass int, np.float64 does subclass
            # float -- and hid a bug that emptied the whole table.
            for row in rows:
                row["autocorr_length"] = int(autocorr_by_step[row["step"]])
        pd.DataFrame(rows).to_csv(run_dir / "statistics.csv", index=False)
    if complete:
        (run_dir / "COMPLETE").touch()


def _build_sweep(tmp_path: Path, runs: list[dict]) -> Path:
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    for spec in runs:
        _build_run(base_dir, **spec)
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *[spec["name"] for spec in runs],
    ]))
    return tmp_path / "datasets"


def _sigmoid_curve(tau: float, amplitude: float, width: float = 1.0) -> dict[int, float]:
    """
    stdev_phi(t) = amplitude * logistic(log10(t/tau)/width): a saturating
    growth curve whose only temperature-dependent knobs are its
    characteristic time tau and its plateau height. Changing tau alone is
    exactly a TIME RESCALING, which is the hypothesis under test; changing
    `width` alone changes the SHAPE, which is its negation.
    """
    out = {}
    for s in STEPS:
        if s == 0:
            out[s] = 0.0
        else:
            out[s] = amplitude / (1.0 + math.exp(-math.log10(s / tau) / width))
    return out


def _equilibrium_amplitude(temperature: float, a0: float = 1.0, b: float = 1.0) -> float:
    return math.sqrt(max(-a0 * (temperature - T0) / b, 0.0))


# --------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------

def test_characteristic_time_interpolates_in_log_t_not_linear_t():
    """
    GUARDS: a linear-in-t interpolation of the crossing. With log-spaced
    samples the two answers differ by ~40% here, far more than the ~1e-9
    tolerance -- and the real schedule's own 15% spacing makes that same
    error the same order as the tau differences the diagnostic exists to
    resolve, so this is not a cosmetic distinction.
    """
    steps = np.array([100.0, 1000.0])
    values = np.array([0.0, 1.0])
    # plateau = median of last quarter = 1.0; target = 0.5
    tau = _characteristic_time(steps, values, ref_fraction=0.5)
    assert tau == pytest.approx(math.sqrt(100.0 * 1000.0))  # log-midpoint
    assert tau != pytest.approx(550.0)  # the linear-midpoint answer


def test_characteristic_time_uses_an_absolute_threshold_not_each_curves_own_plateau():
    """
    GUARDS the bug this test found: tau defined as a fraction of each
    curve's OWN observed plateau. Both curves here are exact time
    rescalings of one another, but the slow one is truncated by the
    observation window and only reaches 0.6 while the fast one reaches
    1.0. Under a self-referential plateau the slow curve's target would
    drop to 0.3 and its tau would land on a different feature entirely;
    under the absolute threshold both cross 0.5 at the same point of
    their shared shape, so tau_slow/tau_fast recovers the true rescaling
    factor of 10.
    """
    steps = np.array([100.0, 1000.0, 10000.0, 100000.0])
    fast = np.array([0.0, 0.5, 1.0, 1.0])
    slow = np.array([0.0, 0.0, 0.0, 0.5])  # same shape, 100x later, truncated at 0.5
    tau_fast = _characteristic_time(steps, fast, ref_fraction=0.5)
    tau_slow = _characteristic_time(steps, slow, ref_fraction=0.5)
    assert tau_fast == pytest.approx(1000.0)
    assert tau_slow == pytest.approx(100000.0)


def test_observed_plateau_is_reported_from_the_tail_median():
    """
    The column that explains a NaN tau. Median of the last quarter, so a
    lone high step cannot flatter a curve that never really got there.
    """
    assert _observed_plateau(np.array([0.0, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6])) == pytest.approx(0.6)


def test_characteristic_time_is_nan_when_curve_never_reaches_target():
    steps = np.array([100.0, 1000.0, 10000.0])
    values = np.array([0.0, 0.0, 0.0])
    assert math.isnan(_characteristic_time(steps, values, ref_fraction=0.5))


def test_curve_spread_is_nan_not_zero_for_a_single_curve():
    """
    GUARDS: returning 0.0 for a lone curve, which would read as a PERFECT
    collapse in the ratio and silently invert the diagnostic's own
    conclusion on a single-temperature sweep.
    """
    one = [(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.5, 1.0]))]
    assert math.isnan(_curve_spread(one))


def test_curve_spread_compares_only_the_overlapping_x_range():
    """
    Two identical curves offset in x overlap on part of their domain;
    the spread over that overlap is what must be measured. If the
    implementation extrapolated instead (np.interp clamps at the edges),
    identical-but-shifted curves would report a spuriously large spread.
    """
    x = np.array([1.0, 10.0, 100.0])
    y = np.array([0.0, 0.5, 1.0])
    spread = _curve_spread([(x, y), (x * 2.0, y)])
    assert np.isfinite(spread)
    assert spread < 0.5


def test_delta_t_at_returns_the_bracketing_gap():
    assert _delta_t_at([0, 500, 1000, 2000], 750) == pytest.approx(500.0)
    assert _delta_t_at([0, 500, 1000, 2000], 1500) == pytest.approx(1000.0)
    assert math.isnan(_delta_t_at([0, 500], 9999))


# --------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------

def test_runs_end_to_end_and_writes_its_figure(tmp_path):
    base_path = _build_sweep(tmp_path, [
        {"name": "T600_n010_s0", "temperature": 0.6,
         "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(0.6))},
        {"name": "T800_n010_s0", "temperature": 0.8,
         "stdev_by_step": _sigmoid_curve(10000, _equilibrium_amplitude(0.8))},
    ])
    output_path = tmp_path / "stdev_phi_time.png"
    result = check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=output_path)
    assert result == output_path
    assert output_path.exists()


def test_pure_time_rescaling_is_detected_as_a_collapse(tmp_path, capsys):
    """
    THE central test. Every temperature gets the SAME curve shape and the
    SAME normalized height (amplitude scaled by its own Landau
    prediction, which the diagnostic divides back out), differing ONLY in
    tau. That is a pure time rescaling by construction, so the rescaled
    spread must be far smaller than the unrescaled one.
    """
    runs = []
    for temp, tau in [(0.6, 2000.0), (0.8, 10000.0), (0.95, 100000.0)]:
        runs.append({"name": f"T{round(temp*1000)}_n010_s0", "temperature": temp,
                      "stdev_by_step": _sigmoid_curve(tau, _equilibrium_amplitude(temp))})
    base_path = _build_sweep(tmp_path, runs)
    check_stdev_phi_time(base_path=base_path, size=SIZE,
                          output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    before = float(out.split("normalized stdev_phi vs t        = ")[1].split("\n")[0])
    after = float(out.split("normalized stdev_phi vs t/tau(T) = ")[1].split("\n")[0])
    assert after < 0.5 * before
    assert "COLLAPSES" in out


def test_shape_difference_is_NOT_reported_as_a_collapse(tmp_path, capsys):
    """
    The negative twin of the test above, and the reason that one is not
    vacuous: here the curves differ in WIDTH as well as tau, so no
    rescaling of the time axis can superpose them. A diagnostic that
    reported a collapse for any input at all would pass the previous test
    and fail this one.
    """
    runs = [
        {"name": "T600_n010_s0", "temperature": 0.6,
         "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(0.6), width=0.3)},
        {"name": "T800_n010_s0", "temperature": 0.8,
         "stdev_by_step": _sigmoid_curve(10000, _equilibrium_amplitude(0.8), width=3.0)},
    ]
    base_path = _build_sweep(tmp_path, runs)
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "COLLAPSES" not in out


def test_averages_over_noise_and_seed_at_fixed_temperature(tmp_path, capsys):
    """
    Two runs at ONE temperature differing in noise and seed, with curves
    straddling a known mean. The reported tau must match the tau of the
    MEAN curve, not of either run alone -- i.e. the averaging axis is
    (noise, seed) and it happens before tau is estimated.
    """
    amp = _equilibrium_amplitude(0.8)
    runs = [
        {"name": "T800_n005_s0", "temperature": 0.8, "noise": 0.005, "seed": 0,
         "stdev_by_step": _sigmoid_curve(5000, amp)},
        {"name": "T800_n020_s1", "temperature": 0.8, "noise": 0.020, "seed": 1,
         "stdev_by_step": _sigmoid_curve(20000, amp)},
        # second temperature only so the run does not hit the
        # single-temperature UNDEFINED branch
        {"name": "T600_n010_s0", "temperature": 0.6,
         "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(0.6))},
    ]
    base_path = _build_sweep(tmp_path, runs)
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.strip().startswith("0.800")][0]
    assert line.split()[1] == "2"  # n_runs
    assert line.split()[2] == "2"  # n_seeds: the two runs differ in seed, so both count
    assert line.split()[3] == "2"  # seeds/step: same population at every step
    tau = float(line.split()[5])
    assert 5000.0 < tau < 20000.0  # the mean curve's tau, not either individual run's


def test_single_temperature_sweep_reports_undefined_rather_than_a_number(tmp_path, capsys):
    """
    GUARDS: silently returning a spread of 0.0 (or NaN formatted as if it
    were a result) for a sweep that cannot pose the question at all.
    """
    base_path = _build_sweep(tmp_path, [
        {"name": "T800_n010_s0", "temperature": 0.8,
         "stdev_by_step": _sigmoid_curve(10000, _equilibrium_amplitude(0.8))},
    ])
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "UNDEFINED" in out


def test_temperature_at_or_above_T0_is_excluded_not_silently_dropped(tmp_path, capsys):
    """
    At T >= T0 the equilibrium amplitude is 0, so the normalized curve
    and tau are undefined. That must be REPORTED -- a run set silently
    vanishing from the collapse test is exactly the kind of invisible
    population change this project has been bitten by before.
    """
    runs = [
        {"name": "T600_n010_s0", "temperature": 0.6,
         "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(0.6))},
        {"name": "T800_n010_s0", "temperature": 0.8,
         "stdev_by_step": _sigmoid_curve(10000, _equilibrium_amplitude(0.8))},
        {"name": "T1000_n010_s0", "temperature": 1.0,
         "stdev_by_step": _sigmoid_curve(50000, 0.01)},
    ]
    base_path = _build_sweep(tmp_path, runs)
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "equilibrium amplitude is 0" in out
    assert "1.0" in out


def test_incomplete_and_statless_runs_are_skipped_and_counted(tmp_path, capsys):
    runs = [
        {"name": "T600_n010_s0", "temperature": 0.6,
         "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(0.6))},
        {"name": "T800_n010_s0", "temperature": 0.8,
         "stdev_by_step": _sigmoid_curve(10000, _equilibrium_amplitude(0.8))},
        {"name": "T700_n010_s0", "temperature": 0.7, "complete": False,
         "stdev_by_step": _sigmoid_curve(5000, _equilibrium_amplitude(0.7))},
        {"name": "T650_n010_s0", "temperature": 0.65, "with_stats": False,
         "stdev_by_step": _sigmoid_curve(5000, _equilibrium_amplitude(0.65))},
    ]
    base_path = _build_sweep(tmp_path, runs)
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "1 incomplete" in out
    assert "1 missing" in out


def test_reports_delta_t_in_force_at_tau(tmp_path, capsys):
    """
    The bridge back to the confound: tau alone does not show that the
    same physical state is sampled at different step spacings across
    temperature. The Delta_t column is what makes that explicit, so its
    presence and correctness is worth asserting rather than assuming.
    """
    runs = [
        {"name": "T600_n010_s0", "temperature": 0.6,
         "stdev_by_step": _sigmoid_curve(700, _equilibrium_amplitude(0.6))},
        {"name": "T800_n010_s0", "temperature": 0.8,
         "stdev_by_step": _sigmoid_curve(70000, _equilibrium_amplitude(0.8))},
    ]
    base_path = _build_sweep(tmp_path, runs)
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    cold = [ln for ln in out.splitlines() if ln.strip().startswith("0.600")][0]
    warm = [ln for ln in out.splitlines() if ln.strip().startswith("0.800")][0]
    # STEPS brackets 700 by (500, 1000) -> gap 500; 70000 by (50000, 100000) -> gap 50000
    assert float(cold.split()[7]) == pytest.approx(500.0)
    assert float(warm.split()[7]) == pytest.approx(50000.0)


# --------------------------------------------------------------------
# labelling / axis helpers
# --------------------------------------------------------------------

def test_format_temp_keeps_three_decimals_where_two_would_collide():
    """
    GUARDS: a fixed 2-decimal temperature label. The sweep steps by 0.005
    near T0, so 0.975/0.980/0.985 must stay distinguishable -- that is
    precisely the region every panel is about. Short form is still used
    where it is unambiguous.
    """
    from evaluation.check_stdev_phi_time import _format_temp
    assert _format_temp(0.55) == "0.55"
    assert _format_temp(0.975) == "0.975"
    assert len({_format_temp(t) for t in (0.975, 0.980, 0.985, 0.990, 0.995)}) == 5


def test_log_axis_ticks_labels_a_sub_decade_range_more_than_once():
    """
    GUARDS: leaving matplotlib's decade-only LogLocator on an axis that
    spans less than a decade, which leaves a single labelled tick and
    makes the panel unreadable (-a(T)/b runs 0.005..0.45).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evaluation.check_stdev_phi_time import _log_axis_ticks
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    _log_axis_ticks(ax.xaxis, 0.02, 0.45)
    assert len(ax.xaxis.get_ticklocs()) >= 3
    plt.close(fig)


def test_last_and_max_are_nan_safe_and_distinct():
    from evaluation.check_stdev_phi_time import _last_and_max
    last, mx = _last_and_max(np.array([0.1, 0.9, 0.4]))
    assert last == pytest.approx(0.4)
    assert mx == pytest.approx(0.9)
    assert all(math.isnan(v) for v in _last_and_max(np.array([np.nan, np.nan])))


# --------------------------------------------------------------------
# per-run tau, used by other diagnostics to de-dimensionalize their dt
# --------------------------------------------------------------------

def test_characteristic_time_for_run_matches_the_curve_it_was_built_from(tmp_path):
    from evaluation.check_stdev_phi_time import characteristic_time_for_run
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    _build_run(base_dir, "T800_n010_s0", 0.8,
                _sigmoid_curve(10000.0, _equilibrium_amplitude(0.8)))
    assert characteristic_time_for_run(base_dir / "T800_n010_s0") == pytest.approx(10000.0, rel=1e-6)


@pytest.mark.parametrize("kwargs,reason", [
    ({"complete": False}, "incomplete run"),
    ({"with_stats": False}, "no statistics.csv"),
])
def test_characteristic_time_for_run_is_nan_when_the_run_is_unusable(tmp_path, kwargs, reason):
    from evaluation.check_stdev_phi_time import characteristic_time_for_run
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    _build_run(base_dir, "T800_n010_s0", 0.8,
                _sigmoid_curve(10000.0, _equilibrium_amplitude(0.8)), **kwargs)
    assert math.isnan(characteristic_time_for_run(base_dir / "T800_n010_s0")), reason


def test_characteristic_time_for_run_is_nan_at_or_above_T0(tmp_path):
    """
    GUARDS: normalizing by a zero equilibrium amplitude. Without the
    guard this divides by zero and returns inf or a garbage crossing
    rather than declaring the run unusable, and the caller would
    de-dimensionalize its dt axis by nonsense.
    """
    from evaluation.check_stdev_phi_time import characteristic_time_for_run
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    _build_run(base_dir, "T1000_n010_s0", 1.0, _sigmoid_curve(10000.0, 0.01))
    assert math.isnan(characteristic_time_for_run(base_dir / "T1000_n010_s0"))


# --------------------------------------------------------------------
# saturation cap: runs of differing length
# --------------------------------------------------------------------

def test_saturation_step_returns_a_step_a_real_run_actually_reached():
    """
    GUARDS: np.quantile-style interpolation. With these extents the 10th
    percentile by linear interpolation is 1900, which NO run reaches --
    truncating there would keep a step that only some runs have, i.e.
    exactly the artifact being removed. The answer must be one of the
    observed maxima.
    """
    from evaluation.check_stdev_phi_time import _saturation_step
    extents = [1000.0, 2000.0, 2000.0, 2000.0, 2000.0,
               2000.0, 2000.0, 2000.0, 2000.0, 5000.0]
    cap = _saturation_step(extents, 0.9)
    assert cap in extents
    assert cap == 1000.0
    assert sum(e >= cap for e in extents) >= 0.9 * len(extents)


def test_saturation_step_disabled_by_zero_fraction():
    from evaluation.check_stdev_phi_time import _saturation_step
    assert _saturation_step([1.0, 2.0, 3.0], 0.0) == float("inf")


def test_late_steps_from_a_few_long_runs_are_dropped(tmp_path, capsys):
    """
    Nine runs stop at 100000; one continues to 1000000. Without the cap
    the mean curve past 100000 is a single run's curve, and it steps
    discontinuously at that point. The cap must remove those samples.
    """
    short_steps = [s for s in STEPS if s <= 100000]
    runs = []
    for i in range(9):
        temp = 0.6 if i < 5 else 0.8
        runs.append({"name": f"T{round(temp*1000)}_n010_s{i}", "temperature": temp, "seed": i,
                      "steps": short_steps,
                      "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(temp))})
    runs.append({"name": "T800_n010_s99", "temperature": 0.8, "seed": 99,
                  "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(0.8))})
    base_path = _build_sweep(tmp_path, runs)
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "saturation cap: t <= 100000" in out
    assert "beyond it dropped" in out
    assert " 0 (temperature, seed, noise, step) sample(s)" not in out


def test_saturation_cap_can_be_disabled(tmp_path, capsys):
    short_steps = [s for s in STEPS if s <= 100000]
    runs = [{"name": "T600_n010_s0", "temperature": 0.6, "steps": short_steps,
              "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(0.6))},
             {"name": "T800_n010_s0", "temperature": 0.8,
              "stdev_by_step": _sigmoid_curve(10000, _equilibrium_amplitude(0.8))}]
    base_path = _build_sweep(tmp_path, runs)
    check_stdev_phi_time(base_path=base_path, size=SIZE, min_run_fraction=0.0,
                          output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "saturation cap: DISABLED" in out


def test_normalized_ylim_is_a_floor_for_the_axis_not_a_ceiling_for_the_data():
    """
    GUARDS: a hard set_ylim(top=1.0). Curves are normalized by the
    PREDICTED equilibrium amplitude, which real data may exceed; clamping
    there would crop the plot silently. The axis top must be raised to
    1.0 when data falls short, and left alone when it does not.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evaluation.check_stdev_phi_time import _normalized_ylim

    fig, ax = plt.subplots()
    ax.plot([1, 2], [0.0, 30.0])
    _normalized_ylim(ax)
    assert ax.get_ylim()[1] == pytest.approx(100.0)

    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2], [0.0, 140.0])
    _normalized_ylim(ax2)
    assert ax2.get_ylim()[1] >= 140.0
    plt.close(fig)
    plt.close(fig2)


def test_varying_contributor_count_across_steps_is_reported(tmp_path, capsys):
    """
    GUARDS: averaging a heterogeneous population silently. Two runs at one
    temperature with DIFFERENT save schedules make the mean curve change
    population along its own length, so a feature at the changeover is a
    composition artifact. The saturation cap cannot catch this (both runs
    reach the same final step); only the per-step count can.
    """
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    curve = _sigmoid_curve(10000.0, _equilibrium_amplitude(0.8))
    sparse = [s for s in STEPS if s in (0, 500, 5000, 50000, 1000000)]
    _build_run(base_dir, "T800_n010_s0", 0.8, curve)
    _build_run(base_dir, "T800_n010_s1", 0.8, {s: curve[s] for s in sparse},
                seed=1, steps=sparse)
    _build_run(base_dir, "T600_n010_s0", 0.6,
                _sigmoid_curve(2000.0, _equilibrium_amplitude(0.6)))
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", "T800_n010_s0", "T800_n010_s1", "T600_n010_s0"]))
    check_stdev_phi_time(base_path=tmp_path / "datasets", size=SIZE,
                          output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "VARYING number of contributing seeds" in out
    assert "1-2" in out


# --------------------------------------------------------------------
# two-level aggregation: noise collapsed within a seed, stats across seeds
# --------------------------------------------------------------------

def test_spread_is_across_seeds_not_across_runs(tmp_path, capsys):
    """
    THE point of the two-level aggregation. One seed is anomalous and
    carries FOUR noise runs with it; the other two seeds carry four each
    too. Averaged over runs, the anomaly is 4 of 12 samples; averaged
    correctly, it is 1 of 3 seeds. The reported per-step population must
    therefore be 3 (seeds), not 12 (runs).
    """
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    amp = _equilibrium_amplitude(0.8)
    names = []
    for seed in (0, 1, 2):
        for i, noise in enumerate((0.005, 0.01, 0.02, 0.05)):
            tau = 200000.0 if seed == 2 else 10000.0  # seed 2 is the outlier
            name = f"T800_n{round(noise*1000):03d}_s{seed}"
            _build_run(base_dir, name, 0.8, _sigmoid_curve(tau, amp), noise=noise, seed=seed)
            names.append(name)
    _build_run(base_dir, "T600_n010_s0", 0.6,
                _sigmoid_curve(2000.0, _equilibrium_amplitude(0.6)))
    names.append("T600_n010_s0")
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *names]))
    check_stdev_phi_time(base_path=tmp_path / "datasets", size=SIZE,
                          output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.strip().startswith("0.800")][0]
    assert line.split()[1] == "12"  # runs
    assert line.split()[2] == "3"   # seeds
    assert line.split()[3] == "3"   # seeds contributing at every step -- NOT 12
    assert "across seeds" in out


def test_per_seed_table_makes_an_anomalous_seed_identifiable(tmp_path, capsys):
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    amp = _equilibrium_amplitude(0.8)
    names = []
    for seed in (0, 1, 2, 3, 4):
        tau = 500000.0 if seed == 4 else 10000.0
        for noise in (0.01, 0.02):
            name = f"T800_n{round(noise*1000):03d}_s{seed}"
            _build_run(base_dir, name, 0.8, _sigmoid_curve(tau, amp), noise=noise, seed=seed)
            names.append(name)
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *names]))
    check_stdev_phi_time(base_path=tmp_path / "datasets", size=SIZE,
                          output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "per-seed comparison, stratified" in out
    assert "--exclude-seeds" in out


def test_exclude_seeds_removes_a_seed_everywhere(tmp_path, capsys):
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    names = []
    for temp in (0.6, 0.8):
        for seed in (0, 1):
            name = f"T{round(temp*1000)}_n010_s{seed}"
            _build_run(base_dir, name, temp,
                        _sigmoid_curve(3100.0 / (1 - temp), _equilibrium_amplitude(temp)),
                        seed=seed)
            names.append(name)
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *names]))
    check_stdev_phi_time(base_path=tmp_path / "datasets", size=SIZE,
                          exclude_seeds={1}, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "2 excluded by --exclude-seeds" in out
    for line in out.splitlines():
        if line.strip().startswith(("0.600", "0.800")):
            assert line.split()[2] == "1"  # one seed left at every temperature


def test_outlier_seed_is_flagged_even_when_the_others_agree_exactly():
    """
    GUARDS the "> 3 * MAD" test alone. When a majority of seeds agree to
    the bit -- which is exactly what makes a lone outlier obvious -- the
    MAD is zero and a pure multiple-of-MAD rule flags nothing. Silently
    finding no outlier in the clearest possible case is worse than not
    looking.
    """
    import numpy as _np
    medians = _np.array([0.20, 0.20, 0.20, 0.20, 0.85])
    centre = float(_np.median(medians))
    spread = float(_np.median(_np.abs(medians - centre)))
    assert spread == 0.0  # the degenerate case this guards
    tolerance = 1e-9 + 0.01 * abs(centre)
    flagged = [i for i, m in enumerate(medians) if abs(m - centre) > tolerance]
    assert flagged == [4]


# --------------------------------------------------------------------
# composition bias: seeds run at different temperature subsets
# --------------------------------------------------------------------

def _sweep_with_specialised_seed(tmp_path, specialist_behaves_normally=True):
    """
    Four general-purpose seeds at every temperature plus one that runs
    ONLY at the high-T end -- the composition this sweep actually has.
    stdev_phi is genuinely smaller at high T, so the specialist's RAW
    median is low for a reason that has nothing to do with it being
    anomalous.
    """
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    names = []
    for temp in (0.60, 0.70, 0.80, 0.95, 0.97):
        seeds = (0, 1, 2, 3, 99) if temp >= 0.95 else (0, 1, 2, 3)
        for seed in seeds:
            tau = 3100.0 / (1 - temp)
            if seed == 99 and not specialist_behaves_normally:
                tau *= 8  # genuinely anomalous, not merely specialised
            name = f"T{round(temp * 1000)}_n010_s{seed}"
            _build_run(base_dir, name, temp,
                        _sigmoid_curve(tau, _equilibrium_amplitude(temp)), seed=seed)
            names.append(name)
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *names]))
    return tmp_path / "datasets"


def test_seed_covering_only_high_T_is_not_flagged_for_its_coverage(tmp_path, capsys):
    """
    GUARDS the raw per-seed median. Seed 99 behaves EXACTLY like the
    others in every cell it appears in; it is merely absent from the low
    temperatures. Ranking seeds by their own median over their own
    samples would put it far from the rest and flag it -- a pure
    composition artifact. Stratified rank must not.
    """
    base_path = _sweep_with_specialised_seed(tmp_path, specialist_behaves_normally=True)
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "stratified by (temperature, step)" in out
    assert "seeds cover different numbers of temperatures" in out
    flagged = [ln for ln in out.splitlines() if "more than 3 MADs" in ln]
    assert not flagged or "99" not in flagged[0], flagged


def test_genuinely_anomalous_seed_is_still_flagged(tmp_path, capsys):
    """
    The negative twin: same coverage imbalance, but now seed 99 really
    does evolve differently. A test immune to composition must still
    catch this, or it has bought its immunity by detecting nothing.
    """
    base_path = _sweep_with_specialised_seed(tmp_path, specialist_behaves_normally=False)
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    flagged = [ln for ln in out.splitlines() if "more than 3 MADs" in ln]
    assert flagged and "99" in flagged[0], out


def test_rank_is_reported_per_seed_with_its_temperature_coverage(tmp_path, capsys):
    base_path = _sweep_with_specialised_seed(tmp_path)
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.strip().startswith("99")][0]
    assert line.split()[1] == "2"                    # covers 2 temperatures, not 5
    assert "0.950- 0.970" in line.replace("  ", " ")  # and says which


def test_average_ranks_give_tied_values_the_same_rank():
    """
    GUARDS argsort-of-argsort. Four identical values must all land on the
    same rank; spreading them over 0..3 makes a set of seeds that agree
    perfectly look maximally dispersed, which inflates the MAD and hides
    a real outlier inside it.
    """
    from evaluation.check_stdev_phi_time import _average_ranks
    r = _average_ranks(np.array([0.9, 0.9, 0.9, 0.9, 0.2]))
    assert r[4] == pytest.approx(0.0)
    assert list(r[:4]) == [pytest.approx(2.5)] * 4
    r2 = _average_ranks(np.array([3.0, 1.0, 2.0]))
    assert list(r2) == [pytest.approx(2.0), pytest.approx(0.0), pytest.approx(1.0)]


# --------------------------------------------------------------------
# seed vs runs: what to do about a flagged seed
# --------------------------------------------------------------------

def _seed_breakdown_sweep(tmp_path, anomalous_noises):
    """
    Five seeds at two temperatures and four noise values. Seed 99 is
    anomalous only at the noise values listed -- so `anomalous_noises`
    covering all of them is the SEED case, and covering one is the RUNS
    case, with everything else held identical.
    """
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    names = []
    for temp in (0.80, 0.90):
        for seed in (0, 1, 2, 3, 99):
            for noise in (0.005, 0.01, 0.02, 0.05):
                tau = 3100.0 / (1 - temp)
                if seed == 99 and noise in anomalous_noises:
                    tau *= 8
                name = f"T{round(temp * 1000)}_n{round(noise * 1000):03d}_s{seed}"
                _build_run(base_dir, name, temp,
                            _sigmoid_curve(tau, _equilibrium_amplitude(temp)),
                            noise=noise, seed=seed)
                names.append(name)
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *names]))
    return tmp_path / "datasets"


def test_seed_wide_anomaly_is_attributed_to_the_seed(tmp_path, capsys):
    base_path = _seed_breakdown_sweep(tmp_path, {0.005, 0.01, 0.02, 0.05})
    check_stdev_phi_time(base_path=base_path, size=SIZE, inspect_seed=99,
                          output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "-> SEED." in out
    assert "--exclude-seeds 99" in out


def test_anomaly_in_a_few_runs_is_NOT_attributed_to_the_seed(tmp_path, capsys):
    """
    The negative twin, and the reason the test above is not vacuous: same
    seed, same everything, but only one of its four noise values is
    anomalous. A verdict that said SEED regardless would pass the other
    test and fail this one.
    """
    base_path = _seed_breakdown_sweep(tmp_path, {0.01})
    check_stdev_phi_time(base_path=base_path, size=SIZE, inspect_seed=99,
                          output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "-> RUNS." in out
    assert "-> SEED." not in out


def test_seed_breakdown_names_the_offending_runs(tmp_path, capsys):
    base_path = _seed_breakdown_sweep(tmp_path, {0.01})
    check_stdev_phi_time(base_path=base_path, size=SIZE, inspect_seed=99,
                          output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if "most extreme runs" in ln][0]
    assert "0.01" in line  # the anomalous noise value is named


# --------------------------------------------------------------------
# median vs mean
# --------------------------------------------------------------------

def _bimodal_sweep(tmp_path, n_trapped, n_seeds=5):
    """
    n_trapped of n_seeds never coarsen out (they hold ~the equilibrium
    amplitude forever, like a run trapped on flat interfaces); the rest
    collapse to zero. The two absorbing states this sweep really has.
    """
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    amp = _equilibrium_amplitude(0.8)
    names = []
    for seed in range(n_seeds):
        trapped = seed < n_trapped
        curve = {s: (amp if s > 3000 else 0.0) if trapped else
                     (amp if 3000 < s <= 20000 else 0.0) for s in STEPS}
        name = f"T800_n010_s{seed}"
        _build_run(base_dir, name, 0.8, curve, seed=seed)
        names.append(name)
    _build_run(base_dir, "T600_n010_s0", 0.6,
                _sigmoid_curve(2000.0, _equilibrium_amplitude(0.6)))
    names.append("T600_n010_s0")
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *names]))
    return tmp_path / "datasets"


def _plateau_at_T800(out):
    line = [ln for ln in out.splitlines() if ln.strip().startswith("0.800")][0]
    return float(line.split()[-1])


@pytest.mark.parametrize("n_trapped,expected", [(1, 0.0), (4, 1.0)])
def test_median_snaps_to_the_majority_absorbing_state(tmp_path, capsys, n_trapped, expected):
    """
    Documents the cost of the median on a BIMODAL tail: it reports which
    state most seeds are in, never the mixture. 1-of-5 trapped and
    4-of-5 trapped must give the two extremes, not 0.2 and 0.8.
    """
    check_stdev_phi_time(base_path=_bimodal_sweep(tmp_path, n_trapped), size=SIZE,
                          statistic="median", output_path=tmp_path / "out.png")
    assert _plateau_at_T800(capsys.readouterr().out) == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize("n_trapped", [1, 4])
def test_mean_recovers_the_trapping_fraction(tmp_path, capsys, n_trapped):
    """
    The complement, and the reason the mean is kept available: over the
    same data it returns p * (trapped amplitude), so the trapping
    probability can be read straight off it.
    """
    check_stdev_phi_time(base_path=_bimodal_sweep(tmp_path, n_trapped), size=SIZE,
                          statistic="mean", output_path=tmp_path / "out.png")
    assert _plateau_at_T800(capsys.readouterr().out) == pytest.approx(n_trapped / 5, abs=0.02)


def test_band_matches_the_statistic(tmp_path, capsys):
    base_path = _bimodal_sweep(tmp_path, 2)
    check_stdev_phi_time(base_path=base_path, size=SIZE, statistic="median",
                          output_path=tmp_path / "m.png")
    assert "MEDIAN at both levels" in capsys.readouterr().out
    check_stdev_phi_time(base_path=base_path, size=SIZE, statistic="mean",
                          output_path=tmp_path / "a.png")
    out = capsys.readouterr().out
    assert "MEAN at both levels" in out
    assert "bimodal" not in out  # the caveat belongs to the median only


def test_unknown_statistic_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="statistic must be one of"):
        check_stdev_phi_time(base_path=_bimodal_sweep(tmp_path, 2), size=SIZE,
                              statistic="average", output_path=tmp_path / "x.png")


def test_band_type_follows_the_statistic(tmp_path, capsys):
    """
    GUARDS a sd band drawn around a median, which describes neither the
    centre it is drawn around nor a quantile range. Asserted through the
    console line because the band itself lives in the figure.
    """
    base_path = _bimodal_sweep(tmp_path, 2)
    check_stdev_phi_time(base_path=base_path, size=SIZE, statistic="median",
                          output_path=tmp_path / "m.png")
    assert "band = p25-p75" in capsys.readouterr().out
    check_stdev_phi_time(base_path=base_path, size=SIZE, statistic="mean",
                          output_path=tmp_path / "a.png")
    assert "band = +-1 sd" in capsys.readouterr().out


def test_statistic_is_applied_to_noise_within_a_seed_too(tmp_path, capsys):
    """
    GUARDS applying the statistic across seeds only, leaving noise
    averaged. One seed, three noise runs, two of which coarsen out: the
    within-seed median is 0 while the within-seed mean is 1/3 of the
    amplitude. A one-level implementation returns the mean here whatever
    --statistic says, and nothing else in the suite notices because every
    other fixture has one noise value per seed.
    """
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    amp = _equilibrium_amplitude(0.8)
    trapped = {s: (amp if s > 3000 else 0.0) for s in STEPS}
    collapsed = {s: (amp if 3000 < s <= 20000 else 0.0) for s in STEPS}
    names = []
    for noise, curve in ((0.01, trapped), (0.02, collapsed), (0.05, collapsed)):
        name = f"T800_n{round(noise * 1000):03d}_s0"
        _build_run(base_dir, name, 0.8, curve, noise=noise, seed=0)
        names.append(name)
    _build_run(base_dir, "T600_n010_s0", 0.6,
                _sigmoid_curve(2000.0, _equilibrium_amplitude(0.6)))
    names.append("T600_n010_s0")
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *names]))

    check_stdev_phi_time(base_path=tmp_path / "datasets", size=SIZE, statistic="median",
                          output_path=tmp_path / "m.png")
    assert _plateau_at_T800(capsys.readouterr().out) == pytest.approx(0.0, abs=0.02)
    check_stdev_phi_time(base_path=tmp_path / "datasets", size=SIZE, statistic="mean",
                          output_path=tmp_path / "a.png")
    assert _plateau_at_T800(capsys.readouterr().out) == pytest.approx(1 / 3, abs=0.02)


# --------------------------------------------------------------------
# tau_down: coarsening completion, the size-DEPENDENT companion to tau
# --------------------------------------------------------------------

def test_characteristic_time_down_is_nan_for_a_curve_that_never_falls():
    """
    The expected answer on a large box: coarsening does not complete
    inside the simulated window, so there is no downward crossing. Must
    be NaN rather than, say, the last step.
    """
    from evaluation.check_stdev_phi_time import _characteristic_time_down
    steps = np.array([100.0, 1000.0, 10000.0, 100000.0])
    rising = np.array([0.0, 0.3, 0.9, 0.95])
    assert math.isnan(_characteristic_time_down(steps, rising, 0.5))


def test_characteristic_time_down_finds_the_descent_and_interpolates_in_log_t():
    from evaluation.check_stdev_phi_time import _characteristic_time_down
    steps = np.array([100.0, 1000.0, 10000.0, 100000.0])
    values = np.array([0.0, 0.9, 0.9, 0.1])
    tau_d = _characteristic_time_down(steps, values, 0.5)
    assert 10000.0 < tau_d < 100000.0
    assert tau_d == pytest.approx(10000.0 * (10 ** 0.5), rel=1e-6)  # log-interpolated


def test_characteristic_time_down_takes_the_LAST_crossing():
    """
    GUARDS returning the first downward crossing. A curve can dip below
    the threshold and recover while domains reorganise; what is wanted is
    when it finally leaves. Here the first crossing is at ~2e3 and the
    real one at ~2e5 -- two decades apart, so the two answers are not
    close.
    """
    from evaluation.check_stdev_phi_time import _characteristic_time_down
    steps = np.array([1e2, 1e3, 1e4, 1e5, 1e6])
    values = np.array([0.9, 0.9, 0.4, 0.9, 0.1])  # dips, recovers, then falls for good
    tau_d = _characteristic_time_down(steps, values, 0.5)
    assert tau_d > 1e5


def test_tau_down_column_and_collapse_are_reported(tmp_path, capsys):
    """
    Rise-then-fall curves whose descent is a pure time rescaling: both
    the column and the second collapse test must appear, and the
    descent-rescaled spread must beat the unrescaled one.
    """
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    names = []
    for temp, scale in ((0.6, 1.0), (0.8, 5.0)):
        amp = _equilibrium_amplitude(temp)
        rise, fall = 5000.0 * scale, 200000.0 * scale
        curve = {s: (0.0 if s < rise else (0.9 * amp if s < fall else 0.0)) for s in STEPS}
        name = f"T{round(temp * 1000)}_n010_s0"
        _build_run(base_dir, name, temp, curve)
        names.append(name)
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *names]))
    check_stdev_phi_time(base_path=tmp_path / "datasets", size=SIZE,
                          output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "tau_down" in out
    assert "collapse test on tau_down" in out
    assert "scales with the box" not in out  # that phrase belongs to the figure, not the console


def test_tau_down_absent_everywhere_is_reported_as_such(tmp_path, capsys):
    """
    The large-box case. Saying so explicitly matters: a blank column
    could be read as a bug, when it is the finding -- no absorbing state
    was reached, so nothing contaminates the tau collapse.
    """
    base_path = _build_sweep(tmp_path, [
        {"name": "T600_n010_s0", "temperature": 0.6,
         "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(0.6))},
        {"name": "T800_n010_s0", "temperature": 0.8,
         "stdev_by_step": _sigmoid_curve(10000, _equilibrium_amplitude(0.8))},
    ])
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "NOT REACHED at any temperature" in out


# --------------------------------------------------------------------
# output location and axis presentation
# --------------------------------------------------------------------

def test_default_output_is_size_prefixed_under_output_datasets(tmp_path, monkeypatch):
    """
    GUARDS an unprefixed default filename. Every conclusion this
    diagnostic reaches (tau_down, trapping, the near-T0 amplitude
    deficit) is size-dependent, so one sweep's figure silently
    overwriting another's is a correctness problem, not a tidiness one.
    """
    import evaluation.check_stdev_phi_time as mod
    monkeypatch.setattr(mod, "_PYTHON_ROOT", tmp_path / "python")
    base_path = _build_sweep(tmp_path, [
        {"name": "T600_n010_s0", "temperature": 0.6,
         "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(0.6))},
        {"name": "T800_n010_s0", "temperature": 0.8,
         "stdev_by_step": _sigmoid_curve(10000, _equilibrium_amplitude(0.8))},
    ])
    out = mod.check_stdev_phi_time(base_path=base_path, size=SIZE)
    assert out.parent == tmp_path / "output" / "datasets"
    assert out.name == f"{SIZE}x{SIZE}-stdev_phi_time.png"
    assert out.exists()


def _threshold_line_drawn(out_text):
    return "tau threshold" in out_text


def test_threshold_line_suppressed_when_every_curve_clears_it(tmp_path):
    """
    GUARDS an unconditional axhline at 50%. When all maxima sit at
    90-98% -- the large-box case -- a line at 50% carries no information
    and stretches the axis over the empty half, compressing the real
    variation into a sliver. Checked on the axis limits, since the line
    is a figure object.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evaluation.check_stdev_phi_time import _normalized_ylim
    # all curves clear the threshold: nothing should force the axis down to 50
    fig, ax = plt.subplots()
    ax.plot([0.6, 0.8], [92.0, 96.0])
    _normalized_ylim(ax)
    assert ax.get_ylim()[0] > 50.0, "no 50% line, so the axis need not reach it"
    plt.close(fig)


# --------------------------------------------------------------------
# autocorr_length: how far coarsening got, in units the C++ can express
# --------------------------------------------------------------------

CAP_32 = 21  # what the C++ search cap works out to for a 32x32 fixture


def _sweep_with_autocorr(tmp_path, cap, saturate_from_step):
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    names = []
    for temp in (0.6, 0.8):
        ac = {s: (float(cap) if s >= saturate_from_step else 5.0) for s in STEPS}
        for seed in (0, 1):
            name = f"T{round(temp * 1000)}_n010_s{seed}"
            _build_run(base_dir, name, temp,
                        _sigmoid_curve(3100.0 / (1 - temp), _equilibrium_amplitude(temp)),
                        seed=seed, autocorr_by_step=ac)
            names.append(name)
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *names]))
    return tmp_path / "datasets"


def test_autocorr_cap_is_taken_from_the_data_not_the_formula(tmp_path, capsys):
    """
    GUARDS computing the cap as min(nx,ny)*2/3 and testing >= against it.
    Integer arithmetic in the C++ can land a unit either side (the 128
    sweep clips at 84 where the formula gives 85), and a one-off would
    report 0% saturation for a sweep that is entirely saturated.
    """
    off_by_one = int(SIZE * 2 / 3) - 1
    check_stdev_phi_time(base_path=_sweep_with_autocorr(tmp_path, off_by_one, 20000),
                          size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert f"capped at {off_by_one:g} px" in out
    assert "DIFFERS" in out            # the mismatch with the nominal formula is flagged
    assert "0.0%" not in out.split("overall")[1][:40]  # saturation detected, not missed


def test_autocorr_reports_saturation_fraction_and_onset(tmp_path, capsys):
    check_stdev_phi_time(base_path=_sweep_with_autocorr(tmp_path, CAP_32, 20000),
                          size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "autocorr_length, search capped at 21 px" in out
    line = [ln for ln in out.splitlines() if ln.strip().startswith("0.600")][-1]
    assert float(line.split()[3].rstrip("%")) > 0     # sat%
    assert float(line.split()[4]) == pytest.approx(20000.0)  # t_sat = onset step


def test_autocorr_absent_is_reported_not_crashed(tmp_path, capsys):
    base_path = _build_sweep(tmp_path, [
        {"name": "T600_n010_s0", "temperature": 0.6,
         "stdev_by_step": _sigmoid_curve(2000, _equilibrium_amplitude(0.6))},
        {"name": "T800_n010_s0", "temperature": 0.8,
         "stdev_by_step": _sigmoid_curve(10000, _equilibrium_amplitude(0.8))},
    ])
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    assert "autocorr_length: not present" in capsys.readouterr().out


def test_finite_number_accepts_numpy_ints_as_well_as_floats():
    """
    GUARDS `isinstance(v, (int, float, np.floating))` on pandas output.
    np.float64 subclasses float but np.int64 does NOT subclass int, so
    that test silently rejects every row of an integer column --
    autocorr_length is whole pixels and pandas infers int64 for it. The
    symptom is an empty result, not an error.
    """
    from evaluation.check_stdev_phi_time import _finite_number
    assert _finite_number(np.int64(42)) == pytest.approx(42.0)
    assert _finite_number(np.float64(1.5)) == pytest.approx(1.5)
    assert _finite_number(7) == pytest.approx(7.0)
    assert _finite_number(np.float64("nan")) is None
    assert _finite_number("42") is None
    assert _finite_number(None) is None
    assert _finite_number(np.bool_(True)) is None  # bool subclasses int; not a measurement


def test_autocorr_survives_an_integer_typed_column(tmp_path, capsys):
    """
    End-to-end version of the above, against a fixture whose
    autocorr_length column pandas types as int64 exactly as the real
    statistics.csv does.
    """
    import pandas as _pd
    base_path = _sweep_with_autocorr(tmp_path, CAP_32, 20000)
    written = _pd.read_csv(base_path / f"{SIZE}x{SIZE}" / "T600_n010_s0" / "statistics.csv")
    assert written["autocorr_length"].dtype.kind == "i", "fixture must reproduce the real dtype"
    check_stdev_phi_time(base_path=base_path, size=SIZE, output_path=tmp_path / "out.png")
    out = capsys.readouterr().out
    assert "autocorr_length: not present" not in out
    assert "search capped at" in out
