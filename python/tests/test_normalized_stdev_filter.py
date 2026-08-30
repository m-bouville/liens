"""min_normalized_stdev_phi thresholds stdev_phi / sqrt(-a(T)/b) (a=a0(T-T0)),
the equilibrium amplitude cspt.py normalizes by, so the surviving FRACTION of a
run is temperature-independent rather than biased by the ground state shrinking
like sqrt(T0-T). Off by default -> the existing min_stdev_phi path is unchanged.
"""
import math
import types
import pandas as pd
import pytest

from training.datasets import _filtered_steps


def _meta(T, save_steps, a0=1.0, b=1.0, T0=1.0):
    return types.SimpleNamespace(a0=a0, b=b, T0=T0, temperature=T,
                                 save_steps=save_steps)


def _df(steps, stdevs):
    return pd.DataFrame({"stdev_phi": stdevs}, index=steps)


def test_default_off_is_identical_to_min_stdev_phi():
    m = _meta(0.75, [1000, 2000])
    df = _df([1000, 2000], [0.2, 0.4])
    assert _filtered_steps(m, set(), 0, 0.3, df) == \
           _filtered_steps(m, set(), 0, 0.3, df, None)   # None -> no-op


def test_normalizes_by_equilibrium_amplitude():
    # T=0.75, a0=b=T0=1 -> amplitude = sqrt(0.25) = 0.5
    m = _meta(0.75, [1000, 2000, 3000, 4000])
    df = _df([1000, 2000, 3000, 4000], [0.20, 0.20, 0.40, 0.40])
    # normalized = [0.4,0.4,0.8,0.8]; threshold 0.5 keeps the 0.40 steps
    assert _filtered_steps(m, set(), 0, None, df, 0.5) == [3000, 4000]


def test_removes_temperature_bias():
    """Same raw stdev, two temperatures: the near-critical frame (large fraction
    of its small equilibrium) survives; the cold one (small fraction of its
    large equilibrium) does not -- the exact bias a raw threshold introduces."""
    hot = _meta(0.99, [1000])    # amplitude sqrt(0.01)=0.1 -> 0.15/0.1 = 1.5
    cold = _meta(0.50, [1000])   # amplitude sqrt(0.5)=0.707 -> 0.15/0.707 = 0.21
    df = _df([1000], [0.15])
    assert _filtered_steps(hot, set(), 0, None, df, 0.5) == [1000]
    assert _filtered_steps(cold, set(), 0, None, df, 0.5) == []
    # a RAW threshold at 0.15 keeps both -- the bias
    assert _filtered_steps(hot, set(), 0, 0.15, df) == [1000]
    assert _filtered_steps(cold, set(), 0, 0.15, df) == [1000]


def test_supercritical_has_no_equilibrium_and_is_excluded():
    # T >= T0: a(T) >= 0, amplitude 0, nothing to normalize by -> excluded
    m = _meta(1.0, [1000])       # T == T0
    df = _df([1000], [0.5])
    assert _filtered_steps(m, set(), 0, None, df, 0.1) == []


def test_min_passing_steps_accepts_either_filter():
    """The guard must accept min_passing_steps with EITHER threshold (this is
    what stage 4 crashed on before the inherit fix), and still refuse it with
    neither -- there'd be no 'passing' criterion to count against."""
    from training.datasets import build_good_steps
    # normalized-only: fine (empty run list -> just exercises the guard)
    assert build_good_steps([], min_passing_steps=12,
                            min_normalized_stdev_phi=0.5) == {}
    # raw-only: fine
    assert build_good_steps([], min_passing_steps=12, min_stdev_phi=0.005) == {}
    # neither: refused
    with pytest.raises(ValueError, match="min_stdev_phi or"):
        build_good_steps([], min_passing_steps=12)


def test_drop_message_names_the_actual_filter(tmp_run_dir, capsys):
    """Under the normalized filter the drop line must say
    min_normalized_stdev_phi=..., not 'min_stdev_phi=None' (which read as
    'no filter' -- the opposite of the truth). A statistics.csv with stdev
    far below threshold makes every step fail, so the run drops entirely."""
    from training.datasets import build_good_steps
    run_dir, steps = tmp_run_dir
    with open(run_dir / "statistics.csv", "w") as f:
        f.write("step,stdev_phi\n")
        for s_ in steps:
            f.write(f"{s_},0.001\n")     # << threshold*amplitude for T=0.8
    build_good_steps([run_dir], min_step=0, min_passing_steps=1,
                     min_normalized_stdev_phi=0.5)
    out = capsys.readouterr().out
    assert "min_normalized_stdev_phi=0.5" in out
    assert "min_stdev_phi=None" not in out
