"""
Tests for evaluation.check_normalization_factor -- the tool that plots the phi
normalization factor A(T) = sqrt(-a0*(T-T0)/b) to explain why normalize_phi hurts
the dynamics. The load side needs the dataset; the MATH (_amplitude) is pure and is
what the whole normalization finding rests on, so it is worth pinning:

  - A(T) shrinks toward 0 as T -> T0 (the gain 1/A blows up near-critical);
  - A(T) is None for T >= T0 (no equilibrium amplitude -- exactly the steps
    normalize_phi cannot handle);
  - the value matches the closed form on known inputs.

The module imports matplotlib + the dataset loader at top level; we stub those so
the pure-math function can be imported and exercised without a dataset or a display.
"""
import sys
import types
import importlib.util
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


@pytest.fixture
def cnf(monkeypatch):
    """Import check_normalization_factor with its heavy deps stubbed.

    CRITICAL: stubs are installed with monkeypatch.setitem so sys.modules is
    RESTORED after the test. An earlier version assigned directly to
    sys.modules["training.datasets"] = <stub> and never cleaned up -- under
    pytest-xdist that poisoned the whole worker, so a LATER test's real
    `from training.datasets import MicrostructureSnapshotDataset` found the stub
    (no such class, no __file__ -> "cannot import name ... (unknown location)").
    A test fixture must never mutate global sys.modules without restoring it.
    """
    import numpy  # noqa: F401  (real, light dependency)
    mpl = types.ModuleType("matplotlib")
    plt = types.ModuleType("matplotlib.pyplot")
    mpl.pyplot = plt
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)

    td = types.ModuleType("training.datasets")
    td.report_save_step_distribution = lambda run_dirs: None
    tr = types.ModuleType("training"); tr.datasets = td
    monkeypatch.setitem(sys.modules, "training", tr)
    monkeypatch.setitem(sys.modules, "training.datasets", td)

    ld = types.ModuleType("load_datasets")
    ld.enumerate_run_dirs_from_metadata = lambda *a, **k: []
    ld.read_metadata = lambda p: None
    up = types.ModuleType("utils"); up.load_datasets = ld
    monkeypatch.setitem(sys.modules, "utils", up)
    monkeypatch.setitem(sys.modules, "utils.load_datasets", ld)

    for cand in (_HERE / "check_normalization_factor.py",
                 _HERE.parent / "evaluation" / "check_normalization_factor.py"):
        if cand.exists():
            spec = importlib.util.spec_from_file_location("cnf", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("check_normalization_factor.py not found")


class _Meta:
    def __init__(self, a0, T, T0, b):
        self.a0, self.temperature, self.T0, self.b = a0, T, T0, b


def test_amplitude_matches_closed_form(cnf):
    # A(T) = sqrt(-a0*(T-T0)/b); a0=1,b=1,T0=1,T=0.75 -> sqrt(0.25) = 0.5
    A = cnf._amplitude(_Meta(a0=1.0, T=0.75, T0=1.0, b=1.0))
    assert A == pytest.approx(0.5)


def test_amplitude_shrinks_toward_T0(cnf):
    a_far = cnf._amplitude(_Meta(1.0, 0.55, 1.0, 1.0))
    a_near = cnf._amplitude(_Meta(1.0, 0.99, 1.0, 1.0))
    assert a_far > a_near > 0.0           # closer to T0 => smaller amplitude
    # gain 1/A therefore grows near-critical (the mechanism the plot shows)
    assert (1.0 / a_near) > (1.0 / a_far)


def test_amplitude_none_at_and_above_T0(cnf):
    assert cnf._amplitude(_Meta(1.0, 1.0, 1.0, 1.0)) is None    # T == T0
    assert cnf._amplitude(_Meta(1.0, 1.01, 1.0, 1.0)) is None   # T > T0


def test_amplitude_uses_per_run_a0_b_T0_not_hardcoded(cnf):
    # different b changes the amplitude -> it reads metadata, not constants
    a_b1 = cnf._amplitude(_Meta(1.0, 0.75, 1.0, 1.0))
    a_b4 = cnf._amplitude(_Meta(1.0, 0.75, 1.0, 4.0))
    assert a_b4 == pytest.approx(a_b1 / 2.0)   # sqrt(0.25/4) = 0.25 = 0.5/2
