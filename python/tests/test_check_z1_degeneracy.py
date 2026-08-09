"""
Tests for evaluation/check_z1_degeneracy.py.

The tool separates two readings of one observation: the dz0dt figure's
first- and second-derivative rows fall with the same slope and a constant
ratio (~1.3e-3). Either z1 is a rescaled copy of z0 (so the second row is
the first row times a constant, carrying no curvature), or both increments
have saturated and the constant ratio is a coincidence of ensemble scales.
"""
import numpy as np
import torch

from evaluation.check_z1_degeneracy import _corr_and_scale


def test_an_exact_rescaling_is_detected():
    """z1 = c*z0 -- correlation 1, the scale recovered, residual ~0. This is
    reading (a): the second-derivative row would be the first times c."""
    torch.manual_seed(0)
    z0 = torch.randn(4, 2, 4, 4)
    corr, scale, resid = _corr_and_scale(0.0013 * z0, z0)
    assert torch.allclose(corr.abs(), torch.ones(4).double(), atol=1e-6)
    assert torch.allclose(scale, torch.full((4,), 0.0013).double(), rtol=1e-5)
    assert float(resid.max()) < 1e-6


def test_independent_fields_are_not_flagged():
    """Reading (b): no relation, so a constant ratio between the rows would
    be a coincidence of scales rather than an identity."""
    torch.manual_seed(0)
    corr, _, resid = _corr_and_scale(torch.randn(8, 2, 8, 8),
                                      torch.randn(8, 2, 8, 8))
    assert float(corr.abs().median()) < 0.3
    assert float(resid.median()) > 0.8


def test_the_residual_is_not_just_the_correlation_restated():
    """
    Correlation is invariant to scale; the claim under test is specifically
    u = c*v, so the residual after the best-fit rescaling is the quantity
    that decides it. A field correlated with v but with a large independent
    part must show a LARGE residual despite a respectable correlation.
    """
    torch.manual_seed(0)
    v = torch.randn(6, 2, 8, 8)
    u = 0.0013 * v + 0.0013 * v.std() * torch.randn(6, 2, 8, 8)
    corr, _, resid = _corr_and_scale(u, v)
    assert 0.4 < float(corr.abs().median()) < 0.9
    assert float(resid.median()) > 0.4, (
        "a half-independent field reports a small residual, so the metric is "
        "not measuring what is left after the rescaling"
    )


def test_a_constant_field_does_not_divide_by_zero():
    """A quiet window's increment can be identically zero -- the low-|z1|
    population the gradient profiler found. That must not produce nan."""
    v = torch.zeros(2, 2, 4, 4)
    u = torch.randn(2, 2, 4, 4)
    corr, scale, resid = _corr_and_scale(u, v)
    assert torch.isfinite(corr).all()
    assert torch.isfinite(scale).all()
    assert torch.isfinite(resid).all()


def test_the_verdict_names_the_three_outcomes():
    """Degenerate, not degenerate, and partial -- the middle case is the
    likely one and must not be reported as either extreme."""
    import inspect

    from evaluation.check_z1_degeneracy import check_z1_degeneracy
    src = inspect.getsource(check_z1_degeneracy)
    assert "DEGENERATE." in src and "NOT degenerate." in src
    assert "PARTIAL:" in src
    # the verdict rests on the INCREMENTS, which is what the figure plots
    assert 'np.abs(out["corr_incr"])' in src


def test_both_state_and_increments_are_measured():
    """z1 could track z0 without the increments doing so, or the reverse, and
    only the increments feed the figure's rows."""
    import inspect

    from evaluation.check_z1_degeneracy import check_z1_degeneracy
    src = inspect.getsource(check_z1_degeneracy)
    assert "_corr_and_scale(z1, z0)" in src
    assert "_corr_and_scale(z1_next - z1, z0_next - z0)" in src


def test_the_scale_is_least_squares_not_a_ratio_of_norms():
    """||u||/||v|| would report a 'scale' for two orthogonal fields; the
    least-squares projection reports ~0, which is the honest answer."""
    torch.manual_seed(1)
    v = torch.randn(1, 1, 8, 8)
    u = torch.randn(1, 1, 8, 8)
    u = u - (u * v).sum() / (v * v).sum() * v          # exactly orthogonal
    _, scale, resid = _corr_and_scale(u, v)
    assert abs(float(scale[0])) < 1e-5
    assert float(resid[0]) > 0.99
    assert float(u.norm() / v.norm()) > 0.1, "fixture is degenerate"


def test_no_f_theta_is_loaded():
    """
    This diagnostic is entirely about the ENCODER, so it must not go through
    _load_ae_f_theta_and_dataset -- which, given a stage-2 checkpoint,
    converts it into a stage-3 one and demands a base_path it has no use for.
    An AE-family checkpoint is the natural input: z1 IS the encoder's deriv
    stream.
    """
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_z1_degeneracy.py")
    assert "_load_ae_f_theta_and_dataset" not in src, (
        "the tool still routes through the stage-3 loader"
    )
    assert "build_ae_from_checkpoint" in src
    assert "MicrostructureEvolutionDataset(" in src


def test_an_ae_checkpoint_without_ae_checkpoint_key_uses_itself():
    """A stage-1/2 checkpoint IS the AE; it has no 'ae_checkpoint' field
    pointing elsewhere."""
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_z1_degeneracy.py")
    assert 'checkpoint.get("ae_checkpoint") or lds_checkpoint_path' in src


def test_missing_test_dirs_is_refused_with_a_reason():
    """No saved split means no population to measure -- better to say so than
    to silently evaluate on whatever happens to be on disk."""
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_z1_degeneracy.py")
    assert "has no saved test_dirs" in src


def test_the_deriv_stream_is_actually_requested():
    """
    Without encode_both_streams the dataset yields (window, dt, theta) and
    never encodes the deriv stream at all -- z1, the entire subject of this
    diagnostic, would be absent, and the unpack fails with "expected 4,
    got 3". The 4-tuple is the signal that z1 is present.
    """
    import inspect
    import pathlib

    from conftest import source_without_comments
    from training.datasets import MicrostructureEvolutionDataset
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_z1_degeneracy.py")
    assert "encode_both_streams=True" in src, (
        "the deriv stream is never encoded, so there is no z1 to measure"
    )
    assert "window0, window1, dt_window, _theta = dataset[int(i)]" in src

    # and that flag really is what gates the 4-tuple
    getitem = inspect.getsource(MicrostructureEvolutionDataset.__getitem__)
    assert "if self.encode_both_streams:" in getitem
    assert "return window, window_deriv, dt_window, theta" in getitem
