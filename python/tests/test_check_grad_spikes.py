"""
Tests for evaluation/check_grad_spikes.py -- what distinguishes the windows
that produce catastrophic gradient norms.

The guard names dt_max in every skip line, but only because dt_max is the one
feature it happens to carry. This tool exists to check whether dt is actually
what separates them, or merely what was printed.
"""
import numpy as np
import torch

from evaluation.check_grad_spikes import (
    _grad_norm, measure_window, separation,
)
from models.latent_dynamics import LatentDynamics


def _model(**kw):
    torch.manual_seed(0)
    kw.setdefault("alpha", 0.5)
    kw.setdefault("max_substeps", 96)
    return LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=16,
                           n_hidden_layers=1, **kw)


def _row(dt=(50.0, 80.0), scale=0.3, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(3, 2, 4, 4) * scale, torch.randn(3, 2, 4, 4) * scale,
            torch.tensor(list(dt)), torch.zeros(1))


def test_the_gradient_is_decomposed_across_the_rollout():
    """
    THE DECOMPOSITION THAT MATTERS. Every excursion in the training logs shows
    a normal one-step column beside a diverging two-step one, so a single
    gradient number cannot say where the trouble is. Measuring the first
    transition alone, the second alone, and the full rollout does.
    """
    m = _model()
    out = measure_window(m, _row(), torch.device("cpu"), 2)
    for key in ("grad_step0", "grad_step1", "grad_full"):
        assert key in out, f"{key} missing; the rollout is not decomposed"
    assert out["grad_step0"] != out["grad_step1"], (
        "the two steps report identical gradients, so the decomposition is "
        "not actually isolating them"
    )


def test_the_first_step_gradient_excludes_the_second_transition():
    """step0 must stop after one transition -- if it integrated both and only
    dropped the second loss term, its gradient would still carry the chained
    path and the decomposition would be meaningless."""
    m = _model()
    calls = {"n": 0}
    original = m._integrate

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    m._integrate = counting
    measure_window(m, _row(), torch.device("cpu"), 2)
    # step0 -> 1 integration, step1 -> 2, full -> 2
    assert calls["n"] == 5, (
        f"{calls['n']} integrations; expected 5 (1 + 2 + 2). step0 is "
        f"integrating the second transition it is supposed to exclude"
    )


def test_clamping_is_reported_per_window_not_cumulatively():
    """
    n_substeps_clamped is deliberately NEVER reset -- a clamp that bound once
    in a run is worth carrying. Reading it directly gives every window the run
    total, so the last window measured always looks like the worst offender.
    I misread exactly this counter as a per-epoch rate in a training log.
    """
    m = _model(max_substeps=2)
    dev = torch.device("cpu")
    # a big f is needed for the criterion to demand more than 2 sub-steps
    with torch.no_grad():
        m.net[-1].weight.normal_(0, 5.0)
        m.net[-1].bias.normal_(0, 5.0)
    first = measure_window(m, _row(dt=(900.0, 900.0), seed=1), dev, 2)
    second = measure_window(m, _row(dt=(900.0, 900.0), seed=2), dev, 2)
    assert "clamped" in first and "clamped" in second
    assert second["clamped"] < m.n_substeps_clamped, (
        "the per-window figure equals the run total, so it is the cumulative "
        "counter rather than this window's own clamping"
    )
    assert first["clamped"] >= 0 and second["clamped"] >= 0


def test_separation_reports_a_rank_metric_not_just_a_ratio():
    """
    A ratio of medians alone is not evidence. Two populations can differ 10x
    in median and overlap almost entirely, in which case filtering on that
    feature discards mostly-healthy windows.
    """
    rng = np.random.default_rng(0)
    n = 400
    mask = np.zeros(n, bool)
    mask[:40] = True
    rows = [{"clean": (100.0 if mask[i] else 10.0) * rng.uniform(0.9, 1.1),
             "null": rng.uniform(1.0, 2.0)}
            for i in range(n)]

    ratio, overlap, auc = separation(rows, "clean", mask)
    assert 8 < ratio < 13 and overlap < 0.05 and auc > 0.95, (
        f"clean separation reported ratio {ratio:.1f} overlap {overlap:.2f} "
        f"AUC {auc:.2f}"
    )
    ratio, overlap, auc = separation(rows, "null", mask)
    assert 0.7 < ratio < 1.4, f"null feature reports a ratio of {ratio:.2f}"
    assert abs(auc - 0.5) < 0.1, (
        f"a feature with no signal reports AUC {auc:.2f}; the metric cannot "
        f"distinguish 'no separation' from 'clean separation'"
    )


def test_a_constant_feature_cannot_win():
    """
    MEASURED ON REAL DATA. `clamped` was identically zero across all 256
    windows, so its ratio was 0/0 = 0.00 and no bulk value exceeded the (zero)
    outlier median -- a perfect overlap score of 0. The tool ranked it the
    best separator of all. A feature that never varies separates nothing, and
    the AUC says so: exactly 0.5.
    """
    mask = np.zeros(100, bool)
    mask[:20] = True
    rows = [{"flat": 0.0} for _ in range(100)]
    ratio, overlap, auc = separation(rows, "flat", mask)
    assert auc == 0.5, f"a constant feature scores AUC {auc}"


def test_ties_at_the_top_of_a_range_are_not_mistaken_for_separation():
    """
    dt cannot exceed max_dt, so once the outliers sit at the cap, "no bulk
    window exceeds the outlier median" is arithmetic rather than evidence --
    and it reads as a perfect 0.00 overlap even when many bulk windows are at
    the cap too. Ties are exactly what overlap cannot see; the AUC counts them
    at half.
    """
    rng = np.random.default_rng(0)
    n = 256
    mask = np.zeros(n, bool)
    mask[:52] = True
    # every outlier at the cap, and HALF the bulk at the cap as well
    rows = [{"dt": 500.0 if (mask[i] or i % 2 == 0)
             else float(rng.choice([60.0, 125.0, 250.0]))}
            for i in range(n)]
    ratio, overlap, auc = separation(rows, "dt", mask)
    assert overlap == 0.0, "fixture no longer exercises the tie case"
    assert auc < 0.85, (
        f"AUC {auc:.2f} with half the bulk tied at the cap -- ties are being "
        f"counted as separation"
    )


def test_separation_survives_a_missing_feature():
    """Features are absent when a model runs at fixed n (no substep_stats), so
    the scorer must skip rather than crash."""
    mask = np.array([True, False, True, False])
    rows = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}, {"a": 4.0}]
    ratio, overlap, auc = separation(rows, "missing", mask)
    assert np.isnan(ratio) and np.isnan(overlap) and np.isnan(auc)


def test_grad_norm_sums_over_all_parameters():
    m = _model()
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    expected = sum(p.numel() for p in m.parameters()) ** 0.5
    assert abs(_grad_norm(m) - expected) < 1e-3


def test_features_include_the_criterions_own_quantity():
    """
    |f|/|z1| is what the alpha criterion divides to pick a sub-step count, so
    it is the feature most likely to explain a window that demands more depth
    than the clamp allows. Recording dt alone would repeat the guard's own
    blind spot.
    """
    m = _model()
    out = measure_window(m, _row(), torch.device("cpu"), 2)
    assert "f_over_z1" in out and "f_norm" in out and "z1_norm" in out


def test_a_constant_feature_is_never_ranked_the_winner():
    """
    MEASURED. `clamped` was identically zero across all 256 windows and the
    tool announced it as the best separator, because ranking on overlap gives
    a constant a perfect score by vacuity. It must stay VISIBLE in the table
    -- knowing it never clamped is useful -- while being unable to win.
    """
    from evaluation.check_grad_spikes import rank_features
    rng = np.random.default_rng(0)
    n = 200
    mask = np.zeros(n, bool)
    mask[:40] = True
    rows = [{"flat": 0.0,
             "real": (10.0 if mask[i] else 1.0) * rng.uniform(0.9, 1.1)}
            for i in range(n)]
    table, ranked = rank_features(rows, mask, keys=("flat", "real"))

    assert {r["key"] for r in table} == {"flat", "real"}, "the table hides a feature"
    assert next(r for r in table if r["key"] == "flat")["constant"] is True
    assert ranked and ranked[0][1] == "real", (
        f"ranked winner is {ranked[0][1]!r}; a constant feature cannot separate"
    )
    assert all(k != "flat" for _, k, _, _ in ranked)


def test_ranking_is_by_auc_not_overlap():
    """
    The two disagree exactly where it matters. Built so the overlap-ranked
    winner is the feature whose outliers merely sit at the top of a capped
    range, while the AUC-ranked winner is the one that actually separates.
    """
    from evaluation.check_grad_spikes import rank_features
    rng = np.random.default_rng(1)
    n = 200
    mask = np.zeros(n, bool)
    mask[:40] = True
    rows = []
    for i in range(n):
        rows.append({
            # capped: every outlier AND most of the bulk sit at the cap, so
            # overlap reads 0.00 while the AUC sees the ties
            "capped": 500.0 if (mask[i] or rng.random() < 0.8) else 250.0,
            # genuine: clean rank separation, but some bulk above the median
            "genuine": (10.0 if mask[i] else 1.0) * rng.lognormal(0, 0.35),
        })
    table, ranked = rank_features(rows, mask, keys=("capped", "genuine"))
    by_overlap = min(table, key=lambda r: r["overlap"])["key"]
    assert by_overlap == "capped", "fixture no longer exercises the disagreement"
    assert ranked[0][1] == "genuine", (
        f"ranked {ranked[0][1]!r} first -- that is the overlap ordering, and "
        f"it prefers a feature whose separation is an artifact of its cap"
    )
