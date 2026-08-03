"""
Standard errors from a near-singular fit must be an explicit NaN, not a
warning-and-NaN by-product.

Reported as:

    RuntimeWarning: invalid value encountered in sqrt
      coef_stderr = np.sqrt(np.diag(cov))

The `except np.linalg.LinAlgError` already there catches an EXACTLY singular
matrix. It does not catch the common case: on a NEAR-singular matrix inv()
succeeds and returns something that is not positive semi-definite, so a
variance comes back negative. The error bar then vanishes from the
Taylor-residual report while the coefficient beside it still looks
authoritative -- and those error bars are what decides whether eps, eps', C
and D mean anything.

_fits.py had TWO independent copies of the block and only one had ever been
hardened, so the joint fit still warned. They now share one helper.
"""

import numpy as np

from conftest import source_without_comments
from evaluation import _fits


def _near_singular_normal_equations():
    """Two basis columns that are almost collinear -- what a dt range too
    narrow to separate the Taylor terms actually produces."""
    x = np.linspace(1.0, 1.0 + 1e-9, 50)
    X = np.column_stack([x, x ** 2])
    return X.T @ X


def test_a_near_singular_fit_emits_no_warning():
    """THE regression: the warning, not the NaN, is what was wrong. A NaN
    error bar is the honest answer for an undetermined coefficient; a
    RuntimeWarning from inside numpy is noise that says nothing useful and
    trains the reader to ignore the warnings summary."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        stderr = _fits._stderr_from_normal_equations(
            _near_singular_normal_equations(), 1.0, 2)
    assert np.isnan(stderr).any(), (
        "a near-singular fit should still REPORT undetermined errors as NaN"
    )


def test_undefined_errors_are_reported_not_silent(capsys):
    """
    A NaN error bar beside a finite coefficient is easy to read straight past.
    The note names the condition number, which is the diagnostic: the usual
    cause is a dt range too narrow to separate the basis terms.
    """
    _fits._stderr_from_normal_equations(_near_singular_normal_equations(), 1.0, 2)
    out = capsys.readouterr().out
    assert "near-singular" in out
    assert "condition number" in out


def test_a_well_conditioned_fit_gives_finite_errors_and_says_nothing(capsys):
    x = np.linspace(1.0, 100.0, 50)
    XtWX = np.column_stack([np.ones_like(x), x]).T @ np.column_stack([np.ones_like(x), x])
    stderr = _fits._stderr_from_normal_equations(XtWX, 2.0, 2)
    assert np.all(np.isfinite(stderr)) and np.all(stderr > 0)
    assert capsys.readouterr().out == ""


def test_an_exactly_singular_matrix_does_not_raise():
    """pinv, not an all-NaN row: the least-norm covariance is finite and
    honestly enormous along the unidentifiable directions."""
    XtWX = np.zeros((3, 3))
    XtWX[0, 0] = 1.0
    stderr = _fits._stderr_from_normal_equations(XtWX, 1.0, 3)
    assert stderr.shape == (3,)
    assert np.isfinite(stderr[0])


def test_BOTH_fits_use_the_shared_helper():
    """
    GUARDS the second copy drifting back. Two independent implementations of
    this block existed and only one was hardened -- which is exactly why the
    joint fit still warned after the first fix.
    """
    src = source_without_comments(_fits)
    assert src.count("np.sqrt(np.diag(cov))") == 0, (
        "an inlined covariance-diagonal sqrt is back -- route it through "
        "_stderr_from_normal_equations instead"
    )
    assert src.count("_stderr_from_normal_equations(") >= 3, (
        "expected the definition plus a call from each of the two robust fits"
    )


def test_the_helper_is_reachable_from_the_public_fits():
    for name in ("fit_taylor_residual_coefficients",):
        assert hasattr(_fits, name)
        assert "_stderr_from_normal_equations" in source_without_comments(_fits)
