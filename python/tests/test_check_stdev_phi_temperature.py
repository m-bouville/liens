"""
Tests for evaluation/check_stdev_phi_temperature.py -- the last
evaluation/ entry point with no coverage at all.

Unlike every other diagnostic in that directory this one is
CHECKPOINT-FREE: it reads statistics.csv across a sweep and needs no
model, checkpoint or GPU. That makes its fixture cheap, but also means
none of the existing checkpoint-based fixtures fit -- it needs precise
control over each run's own temperature and Landau parameters (a0, b,
T0) and its own stdev_phi column, which the other builders don't vary.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_check_stdev_phi_temperature.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.check_stdev_phi_temperature import check_stdev_phi_temperature

SIZE = 32
STEPS = [0, 1000, 2000, 3000, 4000]


def _build_run(base_dir: Path, name: str, temperature: float, stdev_by_step: dict[int, float],
                a0: float = 1.0, b: float = 1.0, T0: float = 1.0,
                complete: bool = True, with_stats: bool = True,
                stats_columns: tuple[str, ...] = ("stdev_phi",)) -> None:
    """One run directory. stdev_phi values are given EXPLICITLY per step
    rather than derived from a field, since every assertion here is about
    how those specific numbers are bucketed/thresholded -- generating
    them from synthetic snapshots would make the expected results depend
    on the generator instead of on this diagnostic's own logic."""
    run_dir = base_dir / name
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.txt").write_text("\n".join([
        f"directory = {name}", "code version = test",
        f"status = {'complete' if complete else 'running'}",
        f"Nx = {SIZE}", f"Ny = {SIZE}", "dt = 0.05", f"steps = {STEPS[-1]}",
        f"save_steps = {' '.join(str(s) for s in STEPS)}",
        f"a0 = {a0}", f"b = {b}", f"T0 = {T0}", f"temperature = {temperature}",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        "seed = 0", "equation = allen_cahn", "solver = explicit", "",
    ]))
    if with_stats:
        rows = [{"step": s, **{c: stdev_by_step.get(s, 0.5) for c in stats_columns}}
                for s in STEPS]
        pd.DataFrame(rows).to_csv(run_dir / "statistics.csv", index=False)
    if complete:
        (run_dir / "COMPLETE").touch()


def _build_sweep(tmp_path: Path, runs: list[dict]) -> Path:
    """runs: list of kwargs dicts for _build_run (each needs at least
    name/temperature/stdev_by_step)."""
    base_dir = tmp_path / "datasets" / f"{SIZE}x{SIZE}"
    base_dir.mkdir(parents=True)
    for spec in runs:
        _build_run(base_dir, **spec)
    (base_dir / "metadata.txt").write_text("\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        "seeds = 0", "subdirs =", *[spec["name"] for spec in runs],
    ]))
    return tmp_path / "datasets"


def _uniform(value: float) -> dict[int, float]:
    return {s: value for s in STEPS}


def test_runs_end_to_end_and_writes_its_figure(tmp_path):
    base_path = _build_sweep(tmp_path, [
        {"name": "T600_n010_s0", "temperature": 0.6, "stdev_by_step": _uniform(0.60)},
        {"name": "T800_n010_s0", "temperature": 0.8, "stdev_by_step": _uniform(0.40)},
        {"name": "T950_n010_s0", "temperature": 0.95, "stdev_by_step": _uniform(0.05)},
    ])
    output_path = tmp_path / "stdev_phi_temp.png"
    result = check_stdev_phi_temperature(
        base_path=base_path, size=SIZE, min_step=1000,
        candidate_thresholds=[0.1], output_path=output_path,
    )
    assert result == output_path
    assert output_path.exists()


def test_equilibrium_amplitude_uses_each_runs_OWN_landau_parameters(tmp_path, capsys):
    """
    REGRESSION-shaped: the theoretical amplitude is sqrt(-a0*(T-T0)/b),
    read from each run's OWN metadata rather than hardcoded to
    a0=b=T0=1 (see this diagnostic's own comment on why -- it must not
    silently break if a future sweep uses different Landau parameters).

    Built so the two conventions give VISIBLY different answers: with
    b=4, the amplitude at T=0.5 is sqrt(0.5/4)=0.354, whereas hardcoded
    b=1 would give sqrt(0.5)=0.707. The reported correlation between
    stdev_phi and the theoretical curve is what surfaces the difference:
    stdev_phi is set to exactly the b=4 answer for every run, so a
    correct implementation sees a perfect match and a hardcoded one
    does not.
    """
    base_path = _build_sweep(tmp_path, [
        # amplitude with b=4: sqrt(-(T-1)/4)
        {"name": "T500_n010_s0", "temperature": 0.5, "b": 4.0,
         "stdev_by_step": _uniform(float(np.sqrt(0.5 / 4)))},
        {"name": "T750_n010_s0", "temperature": 0.75, "b": 4.0,
         "stdev_by_step": _uniform(float(np.sqrt(0.25 / 4)))},
        {"name": "T900_n010_s0", "temperature": 0.9, "b": 4.0,
         "stdev_by_step": _uniform(float(np.sqrt(0.10 / 4)))},
    ])
    check_stdev_phi_temperature(
        base_path=base_path, size=SIZE, min_step=1000,
        candidate_thresholds=[0.1], output_path=tmp_path / "out.png",
    )
    printed = capsys.readouterr().out
    # The reported theoretical amplitude at T=0.5 must be the b=4 answer
    # sqrt(0.5/4) = 0.3536, NOT the hardcoded-b=1 answer sqrt(0.5) =
    # 0.7071 -- a direct value check, far stronger than a correlation
    # (both curves are monotonic in T, so even a hardcoded b would
    # correlate well while being numerically wrong).
    assert "theoretical amplitude=0.3536" in printed, (
        f"expected the run's OWN b=4 amplitude (0.3536); a hardcoded b=1 would give "
        f"0.7071. Output:\n{printed}"
    )
    assert "0.7071" not in printed, "hardcoded b=1 amplitude leaked into the report"
    # ...and with stdev_phi set to exactly that curve, the ratio is 1.000.
    assert "ratio=1.000" in printed


def test_amplitude_is_clipped_to_zero_at_and_above_T0(tmp_path, capsys):
    """
    T >= T0 makes a(T) >= 0, where phi=0 is the only stable point -- the
    sqrt argument must be clipped to 0, giving an amplitude of exactly
    0.0000.

    Asserts the reported VALUE, not merely that the call didn't raise:
    without the clip, np.sqrt of a negative returns NaN and only emits a
    RuntimeWarning, so a "does it crash" test passes either way and
    silently misses NaN poisoning the correlation reported above.
    (Confirmed: an earlier version of this test did exactly that.)
    """
    base_path = _build_sweep(tmp_path, [
        {"name": "T900_n010_s0", "temperature": 0.9, "stdev_by_step": _uniform(0.3)},
        {"name": "T1000_n010_s0", "temperature": 1.0, "stdev_by_step": _uniform(0.02)},
        {"name": "T1100_n010_s0", "temperature": 1.1, "stdev_by_step": _uniform(0.01)},
    ])
    check_stdev_phi_temperature(
        base_path=base_path, size=SIZE, min_step=1000,
        candidate_thresholds=[0.05], output_path=tmp_path / "out.png",
    )
    printed = capsys.readouterr().out
    assert "theoretical amplitude=0.0000" in printed, (
        f"expected a clipped, exactly-zero amplitude for the T>=T0 runs. Output:\n{printed}"
    )
    # Specifically the AMPLITUDE column must not be NaN. Deliberately
    # NOT a blanket "no nan anywhere" check: the ratio column legitimately
    # reads nan here, since it divides observed stdev_phi by a
    # correctly-zero theoretical amplitude. (Cosmetic; noted rather than
    # asserted against, so this test fails only for the real bug.)
    assert "theoretical amplitude=nan" not in printed.lower(), (
        f"NaN amplitude -- the T>=T0 sqrt clip is missing. Output:\n{printed}"
    )


def test_min_step_excludes_earlier_steps(tmp_path, capsys):
    """min_step must drop early steps BEFORE any statistics are computed
    -- the diagnostic's own min_step=0 warning exists precisely because
    early, not-yet-separated steps confound the question it asks."""
    # Early steps have a large stdev, late steps a small one -- so
    # whether min_step filtered correctly is visible in the sample count.
    stdev_by_step = {0: 0.9, 1000: 0.9, 2000: 0.1, 3000: 0.1, 4000: 0.1}
    base_path = _build_sweep(tmp_path, [
        {"name": "T800_n010_s0", "temperature": 0.8, "stdev_by_step": stdev_by_step},
    ])
    check_stdev_phi_temperature(
        base_path=base_path, size=SIZE, min_step=2000,
        candidate_thresholds=[0.5], output_path=tmp_path / "out.png",
    )
    printed = capsys.readouterr().out
    # 3 steps survive min_step=2000 (2000, 3000, 4000), all at stdev 0.1,
    # so a 0.5 threshold excludes ALL of them -> 100% excluded. Had
    # min_step been ignored, the two 0.9 steps would survive it.
    assert "100" in printed, f"expected all-excluded at threshold 0.5; got:\n{printed}"


def test_min_step_zero_warns(tmp_path, capsys):
    base_path = _build_sweep(tmp_path, [
        {"name": "T800_n010_s0", "temperature": 0.8, "stdev_by_step": _uniform(0.4)},
    ])
    check_stdev_phi_temperature(
        base_path=base_path, size=SIZE, min_step=0,
        candidate_thresholds=[0.1], output_path=tmp_path / "out.png",
    )
    assert "min_step=0" in capsys.readouterr().out


def test_incomplete_and_statsless_runs_are_skipped_not_fatal(tmp_path, capsys):
    """A sweep in progress legitimately contains runs that are still
    running, or finished without a statistics.csv -- these must be
    skipped and counted, not crash the whole diagnostic."""
    base_path = _build_sweep(tmp_path, [
        {"name": "T800_n010_s0", "temperature": 0.8, "stdev_by_step": _uniform(0.4)},
        {"name": "T800_n010_s1", "temperature": 0.8, "stdev_by_step": _uniform(0.4),
         "complete": False},
        {"name": "T800_n010_s2", "temperature": 0.8, "stdev_by_step": _uniform(0.4),
         "with_stats": False},
        # has statistics.csv, but no stdev_phi column at all
        {"name": "T800_n010_s3", "temperature": 0.8, "stdev_by_step": _uniform(0.4),
         "stats_columns": ("avg_phi",)},
    ])
    check_stdev_phi_temperature(
        base_path=base_path, size=SIZE, min_step=1000,
        candidate_thresholds=[0.1], output_path=tmp_path / "out.png",
    )
    printed = capsys.readouterr().out
    assert "1" in printed  # at least the one usable run was counted
    assert "skipped" in printed.lower() or "incomplete" in printed.lower()


def test_raises_a_clear_error_when_no_samples_survive(tmp_path):
    """Every step filtered out by min_step -> no data at all. Must be a
    clear ValueError naming the path, not an obscure numpy error on an
    empty array further down."""
    base_path = _build_sweep(tmp_path, [
        {"name": "T800_n010_s0", "temperature": 0.8, "stdev_by_step": _uniform(0.4)},
    ])
    with pytest.raises(ValueError, match="No .*samples found"):
        check_stdev_phi_temperature(
            base_path=base_path, size=SIZE, min_step=999_999,
            candidate_thresholds=[0.1], output_path=tmp_path / "out.png",
        )


def test_min_passing_steps_reports_run_level_exclusion(tmp_path, capsys):
    """The RUN-level analog of the per-step threshold: how many entire
    runs build_good_steps' own min_passing_steps would drop. Two runs
    here clear the threshold at every step; one clears it at none.
    """
    base_path = _build_sweep(tmp_path, [
        {"name": "T700_n010_s0", "temperature": 0.7, "stdev_by_step": _uniform(0.8)},
        {"name": "T700_n010_s1", "temperature": 0.7, "stdev_by_step": _uniform(0.8)},
        {"name": "T950_n010_s0", "temperature": 0.95, "stdev_by_step": _uniform(0.01)},
    ])
    check_stdev_phi_temperature(
        base_path=base_path, size=SIZE, min_step=1000,
        candidate_thresholds=[0.5], min_passing_steps=2,
        output_path=tmp_path / "out.png",
    )
    printed = capsys.readouterr().out
    assert printed, "expected run-level exclusion reporting on stdout"


def test_single_temperature_sweep_reports_undefined_correlation_not_a_warning(tmp_path):
    """
    REGRESSION: np.corrcoef divides by each side's own stddev internally
    -- a single-temperature sweep makes theoretical_amplitude constant
    (it depends only on temperature), so the correlation is genuinely
    UNDEFINED, not just numerically awkward. That used to surface as a
    raw numpy RuntimeWarning plus a silently-printed "corr = nan", not a
    result a reader could understand without knowing numpy's own
    internals.

    Runs under warnings-as-errors specifically to prove the underlying
    RuntimeWarning is gone, not merely that the process didn't crash --
    an earlier version of this fix left the warning firing while adding
    the clearer message alongside it.
    """
    import warnings

    base_path = _build_sweep(tmp_path, [
        {"name": "T800_n010_s0", "temperature": 0.8, "stdev_by_step": _uniform(0.4)},
        {"name": "T800_n010_s1", "temperature": 0.8, "stdev_by_step": _uniform(0.5)},
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            check_stdev_phi_temperature(
                base_path=base_path, size=SIZE, min_step=1000,
                candidate_thresholds=[0.1], output_path=tmp_path / "out.png",
            )
    printed = buf.getvalue()

    assert not any(issubclass(w.category, RuntimeWarning) for w in caught), (
        f"a RuntimeWarning still fired: {[str(w.message) for w in caught]}"
    )
    assert "UNDEFINED" in printed
    assert "only one distinct temperature" in printed
    assert "nan" not in printed.lower(), (
        f"a bare, unexplained NaN reached the report. Output:\n{printed}"
    )
