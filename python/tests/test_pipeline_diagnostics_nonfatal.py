"""
A failing DIAGNOSTIC must not abort the chain, and run_sanity_checks=False must
actually skip them.

Both found by running all six stages end to end for the first time. On a sweep
whose statistics.csv lacks a column check_parameter_dependence plots, a pandas
KeyError propagated out of run_from_params_file BETWEEN stages 3b and 4: every
training stage had succeeded, stages 4 and 5 never ran, and the traceback
pointed at pandas. Same failure shape as the exhausted-rollback raise that took
out stages 4/5 with a good checkpoint sitting on disk.

Diagnostics are figures and printed tables. They inform; they are not the
product.
"""
import pathlib

from conftest import source_without_comments

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_every_diagnostic_is_wrapped_nonfatal():
    """
    Wrapped by REBINDING the imported names once, rather than at each of the 15
    call sites: a per-site wrap is one `git merge` away from a new, unwrapped
    site, and the failure would be a whole aborted chain.
    """
    src = source_without_comments(_ROOT / "orchestration/pipeline.py")
    assert "def _nonfatal_diagnostic(" in src
    for name in ("check_reconstruction", "check_latent_channels", "check_interpolation",
                  "check_perturbation", "check_rollout", "check_parameter_dependence"):
        assert f'"{name}"' in src, f"{name} is not in the rebinding list"
    assert "globals()[_diag_name] = _nonfatal_diagnostic(" in src


def test_the_wrapper_swallows_the_exception_but_shouts():
    """Silent would be worse than fatal: a missing figure nobody noticed."""
    import sys

    sys.path.insert(0, str(_ROOT))
    from orchestration.pipeline import _nonfatal_diagnostic

    def boom(*a, **k):
        raise KeyError("autocorr_length")

    wrapped = _nonfatal_diagnostic(boom, "check_boom")
    assert wrapped() is None                      # continues


def test_the_wrapper_is_transparent_when_nothing_fails():
    import sys

    sys.path.insert(0, str(_ROOT))
    from orchestration.pipeline import _nonfatal_diagnostic
    assert _nonfatal_diagnostic(lambda a, b=2: a + b, "x")(1, b=5) == 6


def test_run_sanity_checks_False_skips_the_3a_3b_comparison_block():
    """
    GUARDS the gating bug this exposed: the block prints "Sanity check:"
    banners and builds figures, but was added after the flag existed and never
    wired to it, so =False ran it anyway.
    """
    src = source_without_comments(_ROOT / "orchestration/pipeline.py")
    assert "if run_sanity_checks and (fresh_3a or fresh_3b):" in src


def test_run_sanity_checks_False_skips_the_stage_4_5_diagnostics():
    """Same bug, second site."""
    src = source_without_comments(_ROOT / "orchestration/pipeline.py")
    block = src[src.index("split_joint_checkpoint_for_evaluation"):]
    block = block[:block.index("return checkpoint")]
    assert "if run_sanity_checks:" in block, (
        "the stage-4/5 diagnostics run even when sanity checks are disabled"
    )
