"""
Tests for evaluation/check_substep_convergence.py -- the measurement that
decides whether f_theta is a vector field or a one-shot corrector, and
therefore whether sub-stepping means anything at all.

The verdict logic is tested on CONSTRUCTED sweeps rather than on a real
model. A randomly-initialised LatentDynamics has |f| ~ 0, so its endpoint
barely moves with the step count and every case looks the same -- which is a
finding about the fixture, not about the code, and it is why the vacuous
guard exists.
"""
import numpy as np
import pytest
import torch

from evaluation.check_substep_convergence import (
    _integrate_at, decade_of, sweep, verdict,
)
from models.latent_dynamics import LatentDynamics
from models.constants import N_THETA


def _rows(truth, self_):
    out = []
    for i, t in enumerate(truth):
        r = {"n": 2 ** i, "truth_rms": np.full(8, float(t))}
        if i < len(self_):
            r["self_rms"] = np.full(8, float(self_[i]))
        out.append(r)
    return out


def test_a_vector_field_is_recognised():
    """Self-difference falls AND the truth error falls: the integration is
    approaching a limit and the limit is right. This is the case that makes
    tightening alpha worthwhile."""
    v = verdict(_rows([1.0, .5, .25, .13, .12], [.5, .25, .12, .06]))
    assert "VECTOR FIELD" in v


def test_a_corrector_is_recognised():
    """
    THE FAILURE THIS PROJECT ALREADY HIT. f_theta fitted at n_substeps=1
    absorbs a whole transition's curvature, so evaluating it at intermediate
    states applies that correction repeatedly and the FINEST step is the
    WORST -- monotonically. Stage 3b integrating a stage-3a corrector at
    40-100 sub-steps is exactly this.
    """
    v = verdict(_rows([0.1, 0.3, 0.7, 1.2, 2.0], [.2, .4, .8, 1.6]))
    assert "CORRECTOR" in v
    assert "refitted" in v, "the verdict does not say what to do about it"


def test_self_convergence_to_a_wrong_limit_is_distinguished():
    """
    A model can converge beautifully to the wrong answer. Reporting only
    self-convergence would call this a success and invite tightening alpha,
    which cannot help: the error is in the fit, not the integration.
    """
    v = verdict(_rows([1.0, 1.2, 1.4, 1.5, 1.5], [.5, .25, .12, .06]))
    assert "SELF-CONVERGES" in v and "WORSE" in v
    assert "not be tightened" in v


def test_a_sweep_that_cannot_discriminate_gives_no_verdict():
    """
    THE GUARD, and it was earned. Run against an untrained model the script
    confidently reported "CORRECTOR" on endpoint differences of 1e-6 -- |f|
    is near zero at initialisation, so the state hardly evolves and every
    step count agrees to rounding. A verdict there is noise dressed as a
    finding.
    """
    v = verdict(_rows([1.0, 1.0, 1.001, 1.0, 1.002], [1e-6, 2e-6, 3e-6, 4e-6]))
    assert "INDISTINGUISHABLE" in v
    assert "untrained" in v, "the verdict does not name the likely cause"


def test_the_step_count_is_forced_and_the_model_is_left_as_found():
    """
    _integrate_at must override alpha for the call -- otherwise the criterion
    picks N and the sweep measures nothing -- and must restore it, since the
    model is shared with the caller and a leaked alpha=None would silently
    change every later measurement.
    """
    # alpha and n_substeps cannot BOTH be given to the constructor -- alpha
    # replaces the count -- so the fixture sets alpha and records whatever
    # n_substeps the model carries, which is what must be restored.
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, alpha=0.15, max_substeps=2048)
    prev_n = m.n_substeps
    torch.manual_seed(0)
    z0, z1 = torch.randn(4, 2, 4, 4), torch.randn(4, 2, 4, 4)
    dt, th = torch.full((4,), 60.0), torch.zeros(4, N_THETA)

    # Counting f_theta EVALUATIONS, not _substeps_for calls: on the fixed-count
    # path the criterion is never consulted, so spying on it sees nothing and
    # proves nothing.
    calls = {"n": 0}
    original_f = m.f

    def counting_f(*a, **k):
        calls["n"] += 1
        return original_f(*a, **k)

    m.f = counting_f
    _integrate_at(m, z0, z1, dt, th, 4)
    four = calls["n"]
    calls["n"] = 0
    _integrate_at(m, z0, z1, dt, th, 16)
    sixteen = calls["n"]
    m.f = original_f

    assert m.alpha == 0.15, "alpha leaked; later measurements are corrupted"
    assert m.n_substeps == prev_n, "n_substeps leaked"
    assert sixteen > 3 * four, (
        f"{four} f evaluations at N=4 and {sixteen} at N=16 -- the requested "
        f"count is not reaching the integrator, so the sweep measures nothing"
    )


def test_the_model_is_restored_even_when_integration_raises():
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, alpha=0.15, max_substeps=2048)

    def boom(*a, **k):
        raise RuntimeError("nope")

    m._integrate = boom
    with pytest.raises(RuntimeError):
        _integrate_at(m, None, None, None, None, 8)
    assert m.alpha == 0.15


def test_the_sweep_reports_self_difference_between_consecutive_counts():
    """The last row has no successor, so it carries a truth error and no
    self-difference -- a sweep that reported one there would be comparing a
    count against itself."""
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, n_substeps=1)
    torch.manual_seed(0)
    z0, z1 = torch.randn(6, 2, 4, 4) * 0.3, torch.randn(6, 2, 4, 4) * 0.3
    dt, th = torch.full((6,), 40.0), torch.zeros(6, N_THETA)
    rows = sweep(m, z0, z1, dt, th, z0.clone(), [1, 2, 4])
    assert [r["n"] for r in rows] == [1, 2, 4]
    assert all("self_rms" in r for r in rows[:-1])
    assert "self_rms" not in rows[-1]
    assert all(len(r["truth_rms"]) == 6 for r in rows), "per-window, not pooled"


def test_decades_band_by_powers_of_ten():
    """Same banding as the training weights, so the two can be read together."""
    assert list(decade_of(np.array([5.0, 12.0, 150.0, 1500.0]))) == [1, 2, 3, 4]


def test_max_dt_can_be_overridden_for_a_like_for_like_comparison():
    """
    Each checkpoint's own config filters the runs, so a 3a check at max_dt=2000
    and a 3b check at max_dt=1000 put DIFFERENT window sets in "decade 3" --
    and the two floors (0.1430 vs 0.1919) cannot then be compared, which is
    exactly the comparison that matters for asking whether 3b degraded the
    long-dt accuracy it inherited.
    """
    import inspect

    from evaluation.check_substep_convergence import check_substep_convergence
    sig = inspect.signature(check_substep_convergence).parameters
    assert "max_dt" in sig and sig["max_dt"].default is None

    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_substep_convergence.py")
    assert "max_dt=max_dt" in src, "the override never reaches the dataset loader"
    assert "--max-dt" in src and "max_dt=args.max_dt" in src


def test_the_default_warns_that_floors_are_not_comparable():
    """Without the override the tool reads each checkpoint's own max_dt, which
    is right for describing one checkpoint and wrong for comparing two. It has
    to say so, or the comparison gets made anyway."""
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_substep_convergence.py")
    # The warning must be PRINTED, not merely present in the argparse help --
    # the phrase appears in both, so matching it anywhere let a mutation that
    # disabled the print pass.
    assert 'print("  NOTE: max_dt comes from this checkpoint' in src, (
        "the comparability warning is not printed at run time"
    )
    assert "not comparable" in src


def test_window_length_can_be_overridden_for_cross_stage_comparison():
    """
    THE SECOND POPULATION CONFOUND, after max_dt. A 3a checkpoint
    (n_rollout_steps=1) builds window_length=2 and a 3b one builds 3;
    requiring a second consecutive valid transition selects earlier,
    faster-evolving -- harder -- windows. So "3a scores 0.1492 where 3b
    scores 0.1631" compared two different populations, and the degradation
    claim built on it was unsupported. Passing --window-length 3 to BOTH
    makes the sets identical.
    """
    import inspect
    import pathlib

    from conftest import source_without_comments
    from evaluation.check_substep_convergence import check_substep_convergence
    sig = inspect.signature(check_substep_convergence).parameters
    assert "window_length" in sig and sig["window_length"].default is None

    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_substep_convergence.py")
    assert "window_length_override=window_length" in src, (
        "the override never reaches the dataset loader, so both stages still "
        "build their own window_length and the populations differ"
    )
    assert "--window-length" in src and "window_length=args.window_length" in src
