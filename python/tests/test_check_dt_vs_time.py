"""
Break the t / Delta_t collinearity, or the diagnostic is pointless.

    corr(log t, log Delta_t) = 0.997   on the real save schedule

so every "error vs dt decade" table in this project is equally a table of
error vs TIME. This diagnostic pairs NON-ADJACENT frames -- (i, i+k) for
k = 1, 2, 4, 8 -- to fill the (t, Delta_t) grid the schedule leaves empty.

The tests below are mostly on synthetic latents with a KNOWN dependence,
because the whole value of the tool is that it attributes error to the right
variable, and only a construction where the true answer is known can show
that it does.
"""
import math
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.check_dt_vs_time import (
    collect_pairs, joint_exponents, regime_split, variation_per_decade,
)


def _fake_dataset(n_runs=1, steps=None, dt_scale=0.05, err_model=None):
    """A dataset stub exposing only the attributes collect_pairs reads.

    Synthetic on purpose: err_model defines the TRUE dependence, so a test can
    assert the diagnostic recovers it. Real latents would only show that the
    code runs.
    """
    # Shaped like the REAL schedule -- repeating {1, 1.15, 1.3, 1.5, ...} x 10^k
    # per decade -- because that is what makes the same gap recur at different
    # t, which is precisely what fills the grid. A purely geometric list
    # (1000, 2000, 4000, 8000) has almost no repeated gaps and cannot exercise
    # the column direction at all: verified, an earlier version of this stub
    # failed the "same dt at several t" test for that reason alone.
    steps = steps or [0, 500, 1000, 1150, 1300, 1500, 1750, 2000, 2250, 2500,
                       3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000]
    run_data, run_deriv, run_steps, run_scale, run_theta = [], [], [], [], []
    for r in range(n_runs):
        n = len(steps)
        state = torch.zeros(n, 4)
        deriv = torch.zeros(n, 4)
        for i, s in enumerate(steps):
            t = s * dt_scale
            state[i, 0] = t                      # z0 = t  -> exact under euler
            deriv[i, 0] = 1.0                    # z1 = dz0/dt = 1
            if err_model is not None:
                state[i, 1] = err_model(t)       # a second channel carrying the error
        run_data.append(state)
        run_deriv.append(deriv)
        run_steps.append(list(steps))
        run_scale.append(dt_scale)
        run_theta.append(torch.tensor([-0.28]))
    return SimpleNamespace(_run_data=run_data, _run_data_deriv=run_deriv,
                            _run_steps=run_steps, _run_dt_scale=run_scale,
                            _run_theta=run_theta)


def test_pairs_are_NON_adjacent_and_cover_a_dt_range_at_fixed_t():
    """
    THE POINT OF THE TOOL. The dataset's own window index contains only
    adjacent pairs, which is the collinearity. Strides must produce several
    DIFFERENT dt values starting from the SAME t.
    """
    # Strides up to 16: on a schedule with ~15% spacing early on, k=4 spans
    # only 2.6x in dt, which is inside the schedule's OWN Delta_t/t scatter and
    # so cannot separate anything. The stride range is what buys the leverage,
    # and this asserts it is actually bought.
    rows = collect_pairs(_fake_dataset(), strides=[1, 2, 4, 8, 16])
    from_zero = sorted({r["dt"] for r in rows if r["t"] == 0.0})
    assert len(from_zero) >= 4, f"only {len(from_zero)} distinct dt at t=0: {from_zero}"
    assert max(from_zero) / min(from_zero) >= 10, (
        f"the dt sweep at fixed t spans only {max(from_zero)/min(from_zero):.1f}x -- "
        f"comparable to the schedule's own scatter, so it separates nothing"
    )


def test_the_same_dt_appears_at_several_different_t():
    """The other half: a dt column must have rows to compare down."""
    rows = collect_pairs(_fake_dataset(), strides=[1, 2, 4])
    by_dt = {}
    for r in rows:
        by_dt.setdefault(round(r["dt"], 6), set()).add(round(r["t"], 6))
    assert any(len(ts) >= 3 for ts in by_dt.values()), (
        "no dt value occurs at 3+ distinct times -- the grid is still a diagonal"
    )


def test_a_perfectly_linear_trajectory_has_zero_euler_error():
    """z0 = t, z1 = 1 is exact under z0 + z1*dt at ANY dt. A nonzero error
    here would mean the pairing arithmetic is wrong, not the model."""
    rows = collect_pairs(_fake_dataset(), strides=[1, 2, 4])
    assert all(r["err"] == pytest.approx(0.0, abs=1e-5) for r in rows)


def test_dt_is_computed_from_the_run_scale_not_the_step_index():
    """dt = (step_b - step_a) * metadata.dt. Using raw step numbers would be
    off by 20x here and silently rescale every conclusion."""
    rows = collect_pairs(_fake_dataset(steps=[0, 1000], dt_scale=0.05), strides=[1])
    assert rows[0]["dt"] == pytest.approx(50.0)


def test_a_PURE_dt_effect_is_attributed_to_dt():
    """
    Error that depends only on the STEP, not the time. The tool must report
    variation across a row and little down a column.
    """
    rows = []
    for t in (10.0, 100.0, 1000.0):
        for dt in (1.0, 10.0, 100.0):
            rows.append({"t": t, "dt": dt, "rel_err": dt, "err": dt, "k": 1,
                          "run_idx": 0, "theta0": -0.28})
    a, b, _ = joint_exponents(rows)
    assert abs(a) < 0.05, f"a pure dt effect gave t exponent {a}"
    assert b > 0.9, f"a pure dt effect gave dt exponent {b}"


def test_a_PURE_time_effect_is_attributed_to_TIME():
    """
    The negative twin, and the reason the previous test is not vacuous: the
    same code on error depending only on t must report the opposite. This is
    the outcome that would overturn max_dt as the right lever.
    """
    rows = []
    for t in (10.0, 100.0, 1000.0):
        for dt in (1.0, 10.0, 100.0):
            rows.append({"t": t, "dt": dt, "rel_err": t, "err": t, "k": 1,
                          "run_idx": 0, "theta0": -0.28})
    a, b, _ = joint_exponents(rows)
    assert a > 0.9 and abs(b) < 0.05, f"a pure time effect gave a={a}, b={b}"


def test_variation_ignores_lines_with_a_single_populated_cell():
    """
    GUARDS diluting the verdict with no-evidence lines. A one-cell line has no
    span at all, and the real grid IS sparse.
    """
    mean = np.array([[1.0, 10.0], [5.0, np.nan]])
    centers = np.array([1.0, 10.0])
    assert variation_per_decade(mean, centers, axis=1) == pytest.approx(10.0)


def test_variation_is_nan_when_nothing_can_be_compared():
    assert math.isnan(variation_per_decade(np.array([[1.0, np.nan]]),
                                            np.array([1.0, 10.0]), axis=1))


def test_per_decade_normalisation_removes_the_RANGE_bias():
    """
    THE REPORTED BUG. On a triangular grid each line spans a different range
    of the other variable, so raw max/min compares ranges as much as effects.

    Measured on the real off-distribution run: dt=6.24e3 reported 108x and
    dt=4.95e4 reported 24x, which read as a non-monotonic outlier. Per decade
    they are 8.74x and 9.10x -- the SAME effect over different spans.
    """
    # Two columns with the SAME per-decade effect (10x) over DIFFERENT spans:
    # col 0 covers 2 decades of t, col 1 only 1. My first attempt gave the two
    # columns genuinely different effects and so demonstrated nothing --
    # constructing this correctly is the whole point.
    mean = np.array([[1.0, np.nan],
                     [10.0, 1.0],
                     [100.0, 10.0]])
    centers = np.array([1.0, 10.0, 100.0])
    per_decade = [variation_per_decade(mean[:, j:j + 1], centers, axis=0) for j in range(2)]
    assert per_decade[0] == pytest.approx(10.0, rel=1e-6)
    assert per_decade[1] == pytest.approx(10.0, rel=1e-6), (
        f"same per-decade effect reported differently: {per_decade}"
    )
    # ...whereas the RAW ratios are 100x and 10x -- a factor of 10 apart, which
    # is exactly the artefact that made dt=6.24e3 look like an outlier.
    raw = [100.0 / 1.0, 10.0 / 1.0]
    assert raw[0] / raw[1] == pytest.approx(10.0)


def test_the_joint_fit_is_not_biased_by_the_TRIANGULAR_geometry():
    """
    The decisive summary. Every pair sits at its OWN (t, dt), so unequal line
    spans cannot tilt it -- which is why it contradicted the per-line medians
    on the real run (they said "dt dominates 2:1", the fit gave a=0.52,
    b=0.47, equal to within 10%).

    Built here on a deliberately triangular sample with a KNOWN answer.
    """
    rows = []
    for lt in range(5):
        for ld in range(5):
            if ld < lt:                     # lower-left triangle empty, as in reality
                continue
            t, dt = 10.0 ** lt, 10.0 ** ld
            rows.append({"t": t, "dt": dt, "rel_err": (t ** 0.5) * (dt ** 0.5)})
    a, b, r2 = joint_exponents(rows)
    assert a == pytest.approx(0.5, abs=0.02)
    assert b == pytest.approx(0.5, abs=0.02)
    assert r2 > 0.99


def test_regime_split_exposes_the_INTERACTION():
    """
    A single "dt or t" verdict cannot express what the data shows: below
    dt~1e3 time costs ~1.4x per decade, above it ~9x. dt does not merely ADD
    error -- it changes how much t matters.
    """
    mean = np.array([[1.0, 1.0],
                     [1.4, 9.0]])
    t_c, dt_c = np.array([1.0, 10.0]), np.array([100.0, 10000.0])
    split = regime_split(mean, t_c, dt_c)
    assert len(split) == 2
    assert split[0][1] == pytest.approx(1.4, rel=1e-6)
    assert split[1][1] == pytest.approx(9.0, rel=1e-6)


def test_the_verdict_says_COMPARABLE_when_the_exponents_are_close():
    """
    GUARDS the original failure: `across 9.87x vs down 4.73x` is a factor of
    2.08, and the tool printed a confident "dt dominates" on what the joint
    fit shows to be a tie. A near-tie must be reported as one.
    """
    from conftest import source_without_comments
    src = source_without_comments(_ROOT / "evaluation/check_dt_vs_time.py")
    assert "0.5 <= ratio <= 2.0" in src
    assert "COMPARABLY" in src
    assert "max_dt cannot address the t term" in src


def test_relative_error_is_normalized_by_the_ACTUAL_change():
    """
    An error of 0.1 across a pair where nothing moved is a different statement
    from the same error across a large excursion -- and the coarsening
    slowdown means late pairs move less, so an unnormalized error would read
    as a time effect that is really a magnitude effect.
    """
    ds = _fake_dataset(steps=[0, 1000], err_model=lambda t: 3.0 if t > 0 else 0.0)
    rows = collect_pairs(ds, strides=[1])
    r = rows[0]
    change = math.hypot(50.0, 3.0)          # z0 moved 50 in ch0 and 3 in ch1
    assert r["rel_err"] == pytest.approx(r["err"] / change, rel=1e-4)


def test_max_pairs_per_run_bounds_the_work():
    many = collect_pairs(_fake_dataset(steps=list(range(0, 20000, 1000))), strides=[1, 2, 4])
    few = collect_pairs(_fake_dataset(steps=list(range(0, 20000, 1000))), strides=[1, 2, 4],
                         max_pairs_per_run=5)
    assert len(few) == 5 < len(many)


def test_the_docstring_states_it_does_not_run_f_theta():
    """
    f_theta is trained at the dataset's own spacing, so scoring it at k>1
    would confound "this dt is hard" with "f_theta never saw this dt". The
    euler-only choice is the whole reason the result is interpretable.
    """
    import ast

    src = pathlib.Path(__file__).resolve().parent.parent / "evaluation/check_dt_vs_time.py"
    tree = ast.parse(src.read_text())
    # AST, not a substring: the module DOCSTRING explains why f_theta is
    # excluded, and a text search matches that prose. Only an actual call
    # matters -- an earlier substring version failed on its own rationale.
    called = {n.func.id for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "f_theta" not in called
    attrs = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "f_theta" not in attrs


def test_the_diagnostic_uses_the_SHARED_latent_cache():
    """
    With max_dt unset -- which this diagnostic requires, since the ceiling is
    what it tests -- there is no prefix truncation, so it encodes EVERY frame
    of every run. The first version omitted latent_cache_dir entirely, making
    it the most expensive diagnostic in the project for no reason.
    """
    import inspect
    import pathlib as _pl

    from conftest import source_without_comments
    from evaluation.check_dt_vs_time import check_dt_vs_time

    default = inspect.signature(check_dt_vs_time).parameters["latent_cache_dir"].default
    assert default is not None, (
        "None as the default would make the cache unturnoffable: None is a "
        "MEANINGFUL value (caching off), so it cannot double as 'not specified'"
    )
    src = source_without_comments(_pl.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_dt_vs_time.py")
    assert "default_latent_cache_dir(_PYTHON_ROOT)" in src
    assert "latent_cache_dir is _UNSET_CACHE" in src, (
        "None must stay meaningful (caching off), so it cannot double as 'not specified'"
    )
    assert '"--no-latent-cache"' in src


_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_per_decade_span_uses_bin_CENTRES_not_edges():
    """
    The normalisation divides by the decades actually covered, so it must use
    the centres of the populated bins. Using edges shifts every span by half a
    bin -- on a 5-bin log grid spanning 3 decades that is ~0.3 decades, a
    ~20% error in every reported factor.

    Source-matched deliberately, and this is the honest case for it: the tests
    above pass `centers` in directly, so the CALLER's choice is not exercised
    by any of them, and a behavioural test would have to re-run the whole
    binning pipeline to see a 20% shift. Verified: the edges mutation passes
    every other test in this file.
    """
    from conftest import source_without_comments
    src = source_without_comments(_ROOT / "evaluation/check_dt_vs_time.py")
    assert "t_centers = np.sqrt(t_edges[:-1] * t_edges[1:])" in src
    assert "dt_centers = np.sqrt(dt_edges[:-1] * dt_edges[1:])" in src


# Imported, NOT reimplemented. An earlier version of this file duplicated the
# splitting logic in a local helper, so mutating the production code changed
# nothing and two mutations passed -- the same "test the unit while the wiring
# is missing" trap this suite keeps catching, in its worst form.
from evaluation.check_dt_vs_time import split_regimes as _regime_message


def test_the_regime_split_does_not_produce_OVERLAPPING_ranges():
    """
    REPORTED BUG. Splitting at the MEDIAN of the per-decade factors put a
    1.40x column into the HIGH group -- factors [1.39, 1.40, 1.35, 8.78,
    9.10], median 1.40, and "f >= median" swept up the 1.40. The message read
    "below dt~2.22e3 ... above dt~279": two overlapping ranges, with a
    high-group median of 5.1x that had averaged a low column into the high
    ones.

    The largest GAP is the right cut, because a regime is a discontinuity.
    """
    split = [(35.2, 1.39), (279.0, 1.40), (2.22e3, 1.35), (1.76e4, 8.78), (1.39e5, 9.10)]
    lo_dt, lo_f, hi_dt, hi_f, _gap = _regime_message(split)
    assert lo_dt < hi_dt, f"ranges overlap: below {lo_dt}, above {hi_dt}"
    assert lo_f == pytest.approx(1.39, abs=0.01)
    assert hi_f == pytest.approx(8.94, abs=0.01), (
        "the high group median must not be diluted by a low-effect column"
    )


def test_no_regime_is_reported_when_the_factors_are_smooth():
    """
    GUARDS splitting anything at all. A gradual trend has no gap, and calling
    it a threshold would invent structure -- the opposite error to the one
    above and just as misleading.
    """
    assert _regime_message([(10.0, 1.0), (100.0, 1.5), (1e3, 2.2), (1e4, 3.3)]) is None


def test_the_split_matches_the_implementation():
    src_path = _ROOT / "evaluation/check_dt_vs_time.py"
    from conftest import source_without_comments
    src = source_without_comments(src_path)
    assert "gaps = sorted_f[1:] / sorted_f[:-1]" in src
    assert "gaps.max() <= min_gap" in src
    assert "np.median(factors[low])" in src
    assert "f >= median" not in src
    # The BOUNDARY belongs to the low group. The original bug was `f >=
    # median`, which swept a 1.40x column into the high group; `<=` / `>`
    # makes the median and the gap agree on this data and cannot recur.
    assert "factors <= cut" in src and "factors > cut" in src
    # And the trainer must actually CALL it -- testing split_regimes directly
    # leaves the call site unchecked, which is how the earlier duplicate-logic
    # version passed two mutations. Verified: stubbing `regimes = None` passes
    # every other test here.
    assert "regimes = split_regimes(split)" in src
    assert "TWO REGIMES, split at the largest gap" in src


# --------------------------------------------------------------------
# the binning, and the entry point that wires everything together
# --------------------------------------------------------------------

def test_grid_table_bins_logarithmically():
    """
    t and dt both span decades, so linear bins would put almost every pair in
    the first cell and leave the rest empty -- and the whole reading of the
    table is "down a column" vs "across a row".
    """
    from evaluation.check_dt_vs_time import grid_table
    rows = [{"t": 10.0 ** i, "dt": 10.0 ** j, "rel_err": 1.0}
            for i in range(3) for j in range(3)]
    t_edges, dt_edges, mean, count = grid_table(rows, n_t_bins=3, n_dt_bins=3)
    ratios = t_edges[1:] / t_edges[:-1]
    assert np.allclose(ratios, ratios[0]), f"bin edges are not geometric: {t_edges}"


def test_grid_table_counts_every_pair_exactly_once():
    """
    GUARDS a digitize off-by-one. The largest value sits ON the last edge and
    np.digitize would put it in bin n (out of range) without the clip -- the
    single most extreme pair, silently dropped, in a diagnostic whose whole
    subject is the extremes.
    """
    from evaluation.check_dt_vs_time import grid_table
    rows = [{"t": 10.0 ** i, "dt": 10.0 ** j, "rel_err": 1.0}
            for i in range(4) for j in range(4)]
    _, _, _, count = grid_table(rows, n_t_bins=3, n_dt_bins=3)
    assert count.sum() == len(rows), f"{count.sum()} of {len(rows)} pairs binned"


def test_grid_table_empty_cells_are_NaN_not_zero():
    """
    Zero would read as "no error here", the opposite of "no data here" -- and
    on a triangular grid most cells are empty by construction.
    """
    from evaluation.check_dt_vs_time import grid_table
    rows = [{"t": 1.0, "dt": 1.0, "rel_err": 5.0}, {"t": 100.0, "dt": 100.0, "rel_err": 5.0}]
    _, _, mean, count = grid_table(rows, n_t_bins=2, n_dt_bins=2)
    empty = count == 0
    assert empty.any(), "this fixture is supposed to leave cells empty"
    assert np.isnan(mean[empty]).all()


def test_grid_table_reports_the_MEAN_of_a_cell():
    from evaluation.check_dt_vs_time import grid_table
    rows = [{"t": 1.0, "dt": 1.0, "rel_err": v} for v in (1.0, 3.0, 5.0)]
    _, _, mean, count = grid_table(rows, n_t_bins=1, n_dt_bins=1)
    assert count[0, 0] == 3
    assert mean[0, 0] == pytest.approx(3.0)


def test_log_bins_ignores_nonpositive_values():
    """GUARDS geomspace on a zero or negative bound, which returns nan edges
    and silently empties the whole table."""
    from evaluation.check_dt_vs_time import _log_bins
    edges = _log_bins(np.array([0.0, -1.0, 10.0, 1000.0]), 2)
    assert edges.size == 3 and np.isfinite(edges).all()
    assert edges[0] == pytest.approx(10.0)


def test_END_TO_END_check_dt_vs_time_runs_and_reports(tmp_path, capsys,
                                                       isolated_project_root):
    """
    The entry point had NO test invoking it -- only its helpers. That is the
    exact gap that let an unwired helper, a wrong tuple index and a missing
    latent_cache_dir through earlier today: every part correct, the assembly
    untested.
    """
    import sys

    sys.path.insert(0, str(_ROOT / "tests"))
    from test_train_lds import _cached_stage2_ancestor
    from training.train_lds import train_lds

    from evaluation.check_dt_vs_time import check_dt_vs_time

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path, stats0_weight=0.01)
    ck = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path,
        ae_stats_weight=0.01, epochs=1, batch_size=4, hidden_dim=8,
        n_hidden_layers=1, val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None, encode_batch_size=4,
        ema_warmup_epochs=0, checkpoint_path=tmp_path / "s3.pt", device="cpu",
        seed=0, log_every_epoch=False, loss_curve_path=tmp_path / "c.png",
    )
    capsys.readouterr()

    result = check_dt_vs_time(ck, strides=(1, 2), base_path=base_path, size=32,
                               min_step=0, device="cpu", n_t_bins=3, n_dt_bins=3,
                               latent_cache_dir=None)
    out = capsys.readouterr().out

    assert result["rows"], "no pairs collected"
    assert "non-adjacent pairs" in out
    assert "median per-DECADE effect of t" in out
    assert "joint fit over all" in out
    # the fit must be REAL numbers, not nan formatted into the message
    assert math.isfinite(result["a"]) and math.isfinite(result["b"])
    assert result["mean"].shape == (3, 3)
    assert int(result["count"].sum()) == len(result["rows"])
