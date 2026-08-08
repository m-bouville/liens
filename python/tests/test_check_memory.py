"""Tests for evaluation/check_memory.py -- the memory diagnostic.

The diagnostic exists because six successive memory models were falsified by
measurement; its pure parts (selection, waste, predictors, fitting) are what
these tests pin. The CUDA measurement itself cannot run here and is asserted
only structurally (mirrors the training step's unpack).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from evaluation.check_memory import (  # noqa: E402
    _fit_and_residuals, masked_loop_waste, predictor_table, select_batches,
)


def _batches(counts, sizes):
    """Split `counts`' indices into consecutive batches of the given sizes."""
    out, at = [], 0
    for s in sizes:
        out.append(np.arange(at, at + s))
        at += s
    return out


def test_selection_spans_the_cost_range_and_always_includes_the_widest_span():
    """
    Measuring every batch costs an epoch of compute, so a SPREAD is chosen --
    cheapest to deepest by predicted cost. The widest-SPAN batch is forced in
    because span is where per-sample truncation's arrival segments multiply,
    i.e. where the candidate models disagree most; a quantile pick can miss
    it entirely when the widest batch is mid-priced, which is exactly the
    case measured on the real population (median span 5, one batch at 214).
    """
    counts = np.concatenate([
        np.full(60, 10.0),          # batch 0: cheap, narrow
        np.full(60, 50.0),          # batch 1: mid, narrow
        np.linspace(20, 400, 60),   # batch 2: mid PRICE but widest span
        np.full(60, 400.0),         # batch 3: deep, narrow
    ])
    batches = _batches(counts, [60, 60, 60, 60])
    picks = select_batches(batches, counts, n_probe=3)
    assert 2 in picks, "the widest-span batch was not selected"
    assert 0 in picks and 3 in picks, "the cost extremes were not selected"


def test_selection_handles_the_degenerate_cases():
    assert select_batches([], np.zeros(0)) == []
    counts = np.full(10, 5.0)
    assert select_batches([np.arange(10)], counts) == [0]


def test_masked_loop_waste_is_the_arrived_fraction():
    """
    waste = 1 - mean/max: the share of f evaluations the masked loop spends
    on samples whose h is already zeroed. Homogeneous batch -> 0; a batch
    where most windows finish early -> large.
    """
    assert masked_loop_waste(np.full(8, 64.0)) == pytest.approx(0.0)
    assert masked_loop_waste(np.array([10.0, 10.0, 10.0, 100.0])) == \
        pytest.approx(1.0 - 32.5 / 100.0)
    assert masked_loop_waste(np.zeros(0)) == 0.0
    assert masked_loop_waste(np.zeros(3)) == 0.0


def test_the_predictor_table_keeps_the_falsified_models():
    """
    Deliberate: the table's purpose is to show, side by side, which predictor
    tracks measured bytes and which do not. Deleting the losers would turn
    the diagnostic back into an assertion -- the very failure mode (model
    first, measurement later) it exists to end.
    """
    counts = np.array([100.0, 300.0, 600.0, 900.0])
    t = predictor_table(counts, truncate_bptt=64)
    assert t["raw"] == pytest.approx(4 * 900.0)
    assert t["retained"] == pytest.approx(4 * 64.0)
    assert t["span_aware"] == pytest.approx(4 * min(900, 4 * 64, 800 + 64))
    # 100..900 at k=64 straddles ceil(800/64)+1 = 14 segments, capped by n=4
    assert t["arrival_segments"] == 4
    assert t["segments_x_k"] == pytest.approx(4 * 4 * 64.0)
    assert predictor_table(np.zeros(0), 64) == {}


def test_predictors_without_truncation_collapse_to_the_raw_model():
    counts = np.array([100.0, 300.0, 900.0])
    t = predictor_table(counts, truncate_bptt=None)
    assert t["retained"] == t["raw"] == t["span_aware"] == pytest.approx(3 * 900.0)
    assert t["arrival_segments"] == 1


def test_the_fit_reports_the_worst_residual_not_the_average():
    """
    A model is only as good as its worst batch -- an average residual hides
    exactly the one batch that OOMs. Fixture: y = 2x on three points and one
    50% outlier; the fit must report ~50%, not ~12%.
    """
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = np.array([2.0, 4.0, 8.0, 8.0])          # last point is 2x off
    a, worst = _fit_and_residuals(x, y)
    assert worst > 0.35, f"worst residual {worst:.2f} looks averaged away"
    a2, worst2 = _fit_and_residuals(x, 3.0 * x)
    assert a2 == pytest.approx(3.0) and worst2 == pytest.approx(0.0, abs=1e-12)
    a3, w3 = _fit_and_residuals(np.array([1.0]), np.array([2.0]))
    assert np.isnan(a3) and np.isnan(w3)


def test_the_measurement_mirrors_the_training_steps_batch_layout():
    """
    The dataset returns (window, window_deriv, dt_window, theta) under
    encode_both_streams, and train_lds slices z0 = window[:, 0],
    targets = window[:, 1:], z1 = window_deriv[:, 0]. My first version
    invented a five-tuple layout that does not exist; measuring memory on a
    differently-shaped step measures a different graph.
    """
    from conftest import source_without_comments
    src = source_without_comments(_ROOT / "evaluation/check_memory.py")
    assert "window[:, 0], window[:, 1:]" in src
    assert "window_deriv[:, 0]" in src
    assert "dt_window[:, step]" in src
    assert "loss.backward()" in src, "no backward -- forward-only memory is half the graph"
    assert "reset_peak_memory_stats" in src


def test_the_power_law_beats_the_fixed_predictors_on_real_measurements():
    """
    MEASURED on a real 3b checkpoint, 6 batches spanning n 167-2048 and depth
    43-470. The per-sample-per-substep figure falls monotonically with depth
    (9.9 -> 6.0 x1e-3 MiB), which no p=1 predictor can express: raw n*depth
    lands at 41% worst residual, a fitted exponent at 12%.

    The numbers are pinned so a change to the fit is visible as a change to
    THESE conclusions rather than as a silently different table.
    """
    import numpy as np

    from evaluation.check_memory import _fit_and_residuals, fit_power_law
    n = np.array([2048, 1343, 821, 593, 427, 167], float)
    hi = np.array([43, 93, 152, 210, 291, 470], float)
    mib = np.array([869.9, 1108.5, 1125.4, 1047.7, 1010.1, 474.1])

    _, worst_raw = _fit_and_residuals(n * hi, mib)
    p, _, worst_pow = fit_power_law(n, hi, mib)

    assert 0.35 < worst_raw < 0.50, f"raw residual moved: {worst_raw:.1%}"
    assert 0.70 <= p <= 0.90, f"fitted exponent moved: {p:.2f}"
    assert worst_pow < worst_raw / 2, (
        f"the power law no longer beats raw: {worst_pow:.1%} vs {worst_raw:.1%}"
    )


def test_the_fit_normalises_residuals_per_point():
    """
    Each residual against ITS OWN measurement, not against the largest. I got
    this wrong reading the table by hand -- normalising by the maximum turned
    a 41% worst case into 17% and made a failing model look usable, which is
    the same error as measuring a quantity adjacent to the one that matters.
    """
    import numpy as np

    from evaluation.check_memory import _fit_and_residuals
    # one small point badly predicted, one large point predicted well
    x = np.array([1.0, 100.0])
    y = np.array([10.0, 100.0])          # a=~1 fits the large point
    a, worst = _fit_and_residuals(x, y)
    assert worst > 0.5, (
        f"worst residual {worst:.1%}: a badly-mispredicted SMALL batch is "
        f"being hidden by normalising against the large one"
    )


def test_truncate_bptt_can_be_overridden_for_older_checkpoints():
    """
    Without the override the tool reports None for checkpoints saved before
    the field existed, every predictor collapses to the same expression, and
    the whole comparison discriminates nothing -- which is exactly what two
    diagnostic runs produced.
    """
    import inspect

    from evaluation.check_memory import check_memory, predictor_table
    assert "truncate_bptt" in inspect.signature(check_memory).parameters

    import numpy as np
    counts = np.array([100.0, 150.0, 300.0])
    none_tab = predictor_table(counts, None)
    k_tab = predictor_table(counts, 64)
    keys = ("raw", "retained", "span_aware", "segments_x_k")
    assert len({none_tab[k] for k in keys}) == 1, (
        "the predictors already differ without truncation, so a None reading "
        "would not be the degenerate case it was"
    )
    assert len({k_tab[k] for k in keys}) > 1, (
        "the predictors do not diverge even WITH truncation, so the table can "
        "never discriminate between them"
    )

    # AND the override must reach the MODEL, not only the predictors --
    # otherwise the table compares truncation-aware predictions against bytes
    # measured WITHOUT truncation, which is worse than not measuring. Checked
    # at source level because the surrounding function needs a checkpoint and
    # a dataset; a mutation removing the assignment passed everything else.
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_memory.py")
    assert "f_theta.truncate_bptt = int(truncate_bptt)" in src, (
        "the override never reaches f_theta, so the measured bytes come from "
        "an untruncated model"
    )


def test_the_fixed_n_probe_holds_the_batch_size_constant():
    """
    THE IDENTIFIABILITY FIX. In cost-budgeted batches deep ones are small by
    construction, so log(n) and log(depth) correlate at -0.97 on real data and
    no fit can separate them: q=1.18/p=0.88 and q=1.00/p=0.76 fit the same
    five measurements at 13.6% and 13.7%. The exponent that looked like a
    finding was an artifact of the batching.
    """
    import numpy as np

    from evaluation.check_memory import fixed_n_probe_batches
    rng = np.random.default_rng(0)
    counts = np.sort(np.ceil(np.exp(rng.uniform(np.log(5), np.log(500), 4000))))
    probes = fixed_n_probe_batches(counts, 256, n_levels=5)
    assert len(probes) >= 3
    assert all(len(b) == 256 for b in probes), (
        f"sizes {[len(b) for b in probes]} -- n is not held fixed, so the "
        f"sweep reproduces the confound it exists to break"
    )
    depths = [counts[list(b)].max() for b in probes]
    assert max(depths) > 3 * min(depths), (
        f"depths {depths} span too little to identify an exponent"
    )
    # and each probe must be NARROW in span, or span confounds the sweep in turn
    spans = [counts[list(b)].max() - counts[list(b)].min() for b in probes]
    assert max(spans) < max(depths), f"spans {spans} rival the depths"


def test_the_probe_declines_when_the_population_is_too_small():
    """
    Below 2*n_fixed the quantile slices overlap, so the "independent" rows
    would share windows and the sweep would measure the same batch several
    times. Asserted on the RETURN, not just emptiness, because a mutation
    removing the guard still returns something -- overlapping slices -- and
    that is the failure worth catching.
    """
    import numpy as np

    from evaluation.check_memory import fixed_n_probe_batches
    tiny = np.arange(10.0)
    assert fixed_n_probe_batches(tiny, 256) == []

    # 400 with n_fixed=256 is the size at which the UNGUARDED path yields two
    # slices that overlap ([48:304] and [136:392]); 300 yields at most one and
    # so cannot detect the missing guard.
    small = np.sort(np.arange(400.0))
    probes = fixed_n_probe_batches(small, 256)
    if probes:
        seen = set()
        for b in probes:
            ids = set(int(i) for i in b)
            assert not (ids & seen), (
                "probe batches overlap, so the sweep re-measures the same "
                "windows and the depth levels are not independent"
            )
            seen |= ids


def test_the_sweep_is_reported_as_the_identifiable_one():
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_memory.py")
    assert "FIXED-n DEPTH SWEEP" in src
    # the exponent must be SEARCHED, not asserted: a mutation pinning the loop
    # to p=1 left the message intact and passed everything else.
    sweep = src[src.index("FIXED-n DEPTH SWEEP"):]
    assert "for p in np.arange(" in sweep, (
        "the sweep does not search over p, so it cannot find an exponent"
    )
    assert "THIS exponent is identifiable" in src, (
        "the sweep's exponent is reported without distinguishing it from the "
        "unidentifiable one in the batch table, which is the whole point"
    )


def test_check_memory_never_treats_the_device_parameter_as_a_torch_device():
    """
    `device` is a STRING parameter; `resolved_device` is the torch.device the
    context builder returns. Using the raw parameter crashed the fixed-n sweep
    with AttributeError on a real run -- and it crashed only there, because
    every other site had already been written against resolved_device, so
    nothing in the sandbox (which has no CUDA and never enters those branches)
    could catch it.

    An AST check rather than a source-string match: it covers any future site,
    and cannot be satisfied by a comment mentioning the right name.
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "evaluation/check_memory.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "check_memory")
    offenders = [n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Attribute)
                 and isinstance(n.value, ast.Name) and n.value.id == "device"]
    assert not offenders, (
        f"check_memory reads an attribute off the raw `device` string at "
        f"line(s) {offenders}; use resolved_device"
    )


def test_the_sweep_measures_with_the_resolved_device_too():
    """The guard and the measurement must agree: guarding on resolved_device
    while measuring with the string would fail one step later."""
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parent.parent
           / "evaluation/check_memory.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "check_memory")
    for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
        if getattr(call.func, "id", None) == "measure_batch_bytes":
            names = [a.id for a in call.args if isinstance(a, ast.Name)]
            assert "resolved_device" in names, (
                f"measure_batch_bytes called with {names} at line {call.lineno}"
            )


def test_a_fitted_constant_changes_which_models_look_viable():
    """
    FITTING THROUGH THE ORIGIN OVERSTATES EVERY MODEL. Parameters, optimizer
    state and the batch tensors cost the same at any depth, so a proportional
    fit bends its slope to absorb them. On the real fixed-n sweep, raw n*depth
    scored 78.1% through the origin and 29.6% with a 67 MiB constant -- the
    difference between "hopeless" and "poor", and I read the first number as
    the second for several turns.
    """
    import numpy as np

    from evaluation.check_memory import _fit_and_residuals, _fit_with_intercept
    d = np.array([8, 28, 61, 127, 279], float)
    mib = np.array([59.5, 93.7, 148.3, 272.6, 407.4]) * 2 ** 20

    _, worst_origin = _fit_and_residuals(d, mib)
    _, c, worst_int = _fit_with_intercept(d, mib)
    assert worst_origin > 0.6, f"origin fit moved: {worst_origin:.1%}"
    assert worst_int < worst_origin / 2, "the constant is not being fitted"
    assert 40 * 2 ** 20 < c < 100 * 2 ** 20, f"constant {c / 2**20:.0f} MiB"


def test_depth_is_sublinear_even_after_the_constant_is_removed():
    """
    THE MEASUREMENT THAT SETTLES THE DESIGN. With n held FIXED (so the
    exponent is identifiable) and a constant fitted separately (so fixed
    overhead cannot masquerade as curvature), memory still grows as about
    depth^0.7. No proportional cost model exists -- which is why every budget
    in sample-substeps mispriced some batch, and why measure-and-correct is
    the design rather than a stopgap.
    """
    import numpy as np

    from evaluation.check_memory import _fit_with_intercept
    d = np.array([8, 28, 61, 127, 279], float)
    mib = np.array([59.5, 93.7, 148.3, 272.6, 407.4]) * 2 ** 20

    _, _, worst_linear = _fit_with_intercept(d, mib)
    best_p, best_w = None, float("inf")
    for p in np.arange(0.0, 1.51, 0.01):
        _, _, w = _fit_with_intercept(d ** p, mib)
        if np.isfinite(w) and w < best_w:
            best_p, best_w = float(p), w
    assert 0.55 <= best_p <= 0.85, f"exponent moved: {best_p:.2f}"
    assert best_w < worst_linear / 2, (
        f"p={best_p:.2f} at {best_w:.1%} no longer beats p=1 at "
        f"{worst_linear:.1%} -- the sublinearity claim rests on this"
    )


def test_the_intercept_fit_needs_three_points():
    """Two points and two parameters is not a fit."""
    import numpy as np

    from evaluation.check_memory import _fit_with_intercept
    b, c, w = _fit_with_intercept(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(w)


def test_the_width_sweep_varies_n_at_a_fixed_depth():
    """
    The complement of the depth sweep, and needed for the same reason. With
    only a depth sweep the per-window cost and the per-depth cost stay
    entangled: fitting A*n + B*n*depth^p to the batch table returned
    A = -1088 KiB per window and predicted NEGATIVE memory at depth 8.
    """
    import numpy as np

    from evaluation.check_memory import fixed_depth_probe_batches
    counts = np.sort(np.ceil(np.exp(
        np.random.default_rng(0).uniform(np.log(5), np.log(500), 4000))))
    probes = fixed_depth_probe_batches(counts, sizes=(128, 256, 512, 1024))
    assert len(probes) >= 3
    sizes = sorted(len(b) for b in probes)
    assert sizes == [128, 256, 512, 1024], sizes
    # IDENTICAL depth range for every size, not merely similar. Contiguous
    # slices gave 28-88 for n=1024 against 45-53 for n=128 -- a 1.66x spread,
    # which leaves n and depth co-varying in the very sweep meant to separate
    # them.
    ranges = {(counts[list(b)].min(), counts[list(b)].max()) for b in probes}
    assert len(ranges) == 1, (
        f"depth ranges differ across sizes: {sorted(ranges)}"
    )


def test_the_width_sweep_drops_sizes_that_do_not_fit_the_population():
    import numpy as np

    from evaluation.check_memory import fixed_depth_probe_batches
    # Below the LARGEST size there is no pool to subsample, so the sweep
    # declines entirely rather than returning rows whose depth ranges differ.
    assert fixed_depth_probe_batches(np.arange(300.0),
                                      sizes=(128, 256, 512, 1024)) == []
    probes = fixed_depth_probe_batches(np.arange(2000.0),
                                        sizes=(128, 256, 512, 1024))
    assert probes, "a population of 2000 should support every size"
    # EXACTLY the requested sizes, or absent. A slice past the end truncates
    # silently, which would put a 300-window batch in the table labelled 1024
    # and corrupt the fit it feeds.
    assert {len(b) for b in probes} <= {128, 256, 512, 1024}, (
        f"sizes {sorted(len(b) for b in probes)} -- an oversized request was "
        f"truncated rather than dropped"
    )


def test_the_joint_fit_recovers_known_parameters():
    """
    The two sweeps TOGETHER span the design matrix; either alone does not.
    Verified against synthetic data with a known answer, because on real data
    there is nothing to check the fit against -- which is exactly how the
    unphysical constants (-116 MiB, -1088 KiB/window) went unnoticed.
    """
    import numpy as np

    A_true, B_true, p_true = 40_000.0, 9_000.0, 0.70
    rows = [(256, d) for d in (8, 28, 61, 127, 279)]
    rows += [(nn, 90) for nn in (128, 256, 512, 1024)]
    n = np.array([r[0] for r in rows], float)
    d = np.array([r[1] for r in rows], float)
    y = A_true * n + B_true * n * d ** p_true

    best = None
    for p in np.arange(0.2, 1.31, 0.01):
        X = np.vstack([n, n * d ** p]).T
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        w = float((np.abs(X @ coef - y) / y).max())
        if best is None or w < best[3]:
            best = (float(p), float(coef[0]), float(coef[1]), w)
    p, A, B, w = best
    assert abs(p - p_true) < 0.02, f"p={p}"
    assert abs(A - A_true) / A_true < 0.02, f"A={A}"
    assert abs(B - B_true) / B_true < 0.02, f"B={B}"


def test_a_negative_per_window_cost_is_called_out():
    """An unphysical fitted coefficient must be reported as such, not printed
    as though it were a measurement -- span_aware's -116 MiB constant read as
    its best score."""
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_memory.py")
    assert "A IS NEGATIVE" in src
    assert "fitting artifact" in src
    # the guard itself, not only its text: a mutation disabling the condition
    # left the message in the source and passed.
    assert "if A < 0:" in src, (
        "the negative-coefficient warning is unreachable, so an impossible "
        "fit would print as though it were a measurement"
    )


def test_the_joint_fit_prints_a_pasteable_params_snippet():
    """
    The three coefficients were added as parameters with no way to discover
    their values -- the fit that produces them is printed by this tool, so
    the recipe belongs here rather than in a message the user has to find
    again. It must name the parameters exactly as train_lds spells them, or
    the snippet cannot be pasted.
    """
    import inspect
    import pathlib

    from conftest import source_without_comments
    from training.train_lds import train_lds
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_memory.py")
    for name in ("memory_cost_a_bytes", "memory_cost_b_bytes", "memory_cost_p"):
        assert name in src, f"{name} is not in the emitted snippet"
        assert name in inspect.signature(train_lds).parameters, (
            f"the snippet names {name}, which train_lds does not accept"
        )
    # and it must warn that the fit is per stage: n_rollout_steps changes it
    assert "PER STAGE" in src
    # EVERY emission site must be guarded by a positivity check on A -- there
    # are two now (the joint fit and the forced-depth calibration), so
    # comparing first-occurrence indices stopped meaning anything.
    marker = "memory_cost_a_bytes = "
    sites = [i for i in range(len(src)) if src.startswith(marker, i)]
    assert sites, "the snippet is not emitted at all"
    for i in sites:
        window = src[max(0, i - 900):i]
        assert ("A IS NEGATIVE" in window or "A IS NOT POSITIVE" in window), (
            "a snippet is printed without a preceding positivity guard, so an "
            "impossible fit would be offered for pasting"
        )


def _fake_cost(A=40000.0, B=9000.0, p=0.70):
    def measure(f_theta, dataset, batch, device, n_rollout_steps):
        d, n = float(f_theta.n_substeps), float(len(batch))
        return A * n + B * n * d ** p
    return measure


def test_calibration_forces_the_depth_so_no_trained_model_is_needed():
    """
    THE CIRCULARITY. The probe sweeps get their depth spread by selecting
    windows whose sub-step counts differ, which needs a model whose alpha
    criterion already produces a spread -- i.e. a trained one. So "measure
    the coefficients, then train" required training first.

    The coefficients do not depend on the weights: they describe the
    architecture and the card. The weights only decide which depths get
    ASKED for, which is the model's input. Forcing n_substeps measures the
    same thing on any checkpoint.
    """
    import evaluation.check_memory as cm
    from models.latent_dynamics import LatentDynamics
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, alpha=0.15, max_substeps=4096,
                        truncate_bptt=64)
    seen = []
    real = cm.measure_batch_bytes

    def spy(f_theta, dataset, batch, device, n_rollout_steps):
        # alpha must be None AT THE MOMENT OF MEASUREMENT: while it is set,
        # the criterion overrides n_substeps and the depth is not forced at
        # all, however carefully the attribute was assigned.
        assert f_theta.alpha is None, (
            "alpha is still set during the measurement, so the criterion is "
            "choosing the depth and the grid is decorative"
        )
        seen.append((len(batch), int(f_theta.n_substeps)))
        return _fake_cost()(f_theta, dataset, batch, device, n_rollout_steps)

    cm.measure_batch_bytes = spy
    try:
        fit = cm.calibrate_cost_model(m, range(2000), "cpu", 2,
                                       sizes=(128, 256), depths=(8, 64, 256))
    finally:
        cm.measure_batch_bytes = real

    assert sorted({d for _, d in seen}) == [8, 64, 256], (
        f"depths {sorted({d for _, d in seen})} -- the grid is not being "
        f"forced, so this still depends on what the criterion asks for"
    )
    assert sorted({n for n, _ in seen}) == [128, 256]
    # FACTORIAL: every size at every depth, so n and depth are independent by
    # construction rather than by careful sampling
    assert len(seen) == 6 and len(set(seen)) == 6
    assert m.alpha == 0.15, "alpha leaked out of the calibration"
    assert 0.65 <= fit["p"] <= 0.75


def test_calibration_recovers_known_coefficients():
    import evaluation.check_memory as cm
    from models.latent_dynamics import LatentDynamics
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, n_substeps=1, truncate_bptt=64)
    real = cm.measure_batch_bytes
    cm.measure_batch_bytes = _fake_cost(A=40000.0, B=9000.0, p=0.70)
    try:
        fit = cm.calibrate_cost_model(m, range(2000), "cpu", 2)
    finally:
        cm.measure_batch_bytes = real
    assert abs(fit["A"] - 40000.0) / 40000.0 < 0.02
    assert abs(fit["B"] - 9000.0) / 9000.0 < 0.02
    assert abs(fit["p"] - 0.70) < 0.02
    assert fit["worst"] < 0.01


def test_calibration_declines_rather_than_fitting_a_tiny_grid():
    """Two points and three parameters is not a fit."""
    import evaluation.check_memory as cm
    from models.latent_dynamics import LatentDynamics
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, n_substeps=1, truncate_bptt=64)
    real = cm.measure_batch_bytes
    cm.measure_batch_bytes = _fake_cost()
    try:
        fit = cm.calibrate_cost_model(m, range(2000), "cpu", 2,
                                       sizes=(128,), depths=(8, 64, 256))
    finally:
        cm.measure_batch_bytes = real
    assert "p" not in fit and "reason" in fit


def test_the_calibrate_flag_is_exposed():
    import inspect

    from conftest import source_without_comments
    import pathlib
    from evaluation.check_memory import check_memory
    assert inspect.signature(check_memory).parameters["calibrate"].default is False
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_memory.py")
    assert "--calibrate" in src
    assert "calibrate=args.calibrate" in src


def test_two_depths_cannot_identify_the_exponent_and_are_refused():
    """
    A + B*d^p has three unknowns, so with two depth levels EVERY exponent
    admits an exact (A, B). The search then returns whichever it tried first,
    with a zero residual that reads as a perfect fit: on exact synthetic data
    a 2x2 grid returned p=0.36 against a true 0.70, at 0% residual.
    """
    import evaluation.check_memory as cm
    from models.latent_dynamics import LatentDynamics
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, n_substeps=1, truncate_bptt=64)
    real = cm.measure_batch_bytes
    cm.measure_batch_bytes = _fake_cost()
    try:
        fit = cm.calibrate_cost_model(m, range(2000), "cpu", 2,
                                       sizes=(128, 256), depths=(8, 64))
    finally:
        cm.measure_batch_bytes = real
    assert "p" not in fit, (
        f"fitted p={fit.get('p')} from two depths, where every p fits exactly"
    )
    assert "3 distinct depths" in fit["reason"]
