"""
The (dt, theta) corner every spike attribution names must be reachable.

`dt_max=125, mean theta[0]~=-0.28` has appeared in stage 3a's loss spikes,
stage 3b's loss AND gradient spikes, stage 5's gradient spikes, and in the
off-diagonal rollout figures where exactly those windows degraded 2.4-2.5x
under z1 propagation. Investigating it means running a diagnostic on those
windows -- and --fixed-windows takes explicit
`<run_dir>:<step1>:<step2>:<step3>` strings that nobody can write by hand,
because dt is `(step_b - step_a) * metadata.dt` and theta[0] is
`temperature - T0`, both per-run values only the metadata knows.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_train_lds import _build_sweep

from evaluation._window_parsing import parse_fixed_window
from evaluation.find_windows import find_windows


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    return _build_sweep(tmp_path_factory.mktemp("fw"), n_runs=6, size=32)


def test_finds_windows_at_a_requested_dt_and_theta(sweep):
    found = find_windows(sweep, 32, dt=50.0, theta0=-0.20, theta_tol=0.01, limit=3)
    assert found, "no window found in a corner this sweep definitely contains"
    for _, actual_dt, actual_theta in found:
        assert actual_dt == pytest.approx(50.0, rel=0.05)
        assert actual_theta == pytest.approx(-0.20, abs=0.01)


def test_the_emitted_strings_PARSE_as_fixed_windows(sweep):
    """
    The whole point is paste-ability. A string the tool prints but
    check_rollout rejects is worse than no tool -- which is exactly the
    situation this replaces.
    """
    found = find_windows(sweep, 32, dt=50.0, theta0=-0.20, theta_tol=0.01, limit=3)
    for window, _, _ in found:
        run_dir, steps = parse_fixed_window(window)
        assert run_dir.exists(), run_dir
        assert len(steps) == 3, "n_steps=2 must give THREE step numbers"


def test_n_steps_controls_the_window_length(sweep):
    """n_steps is TRANSITIONS; a checkpoint at n_rollout_steps=N needs N+1
    step numbers, and passing the wrong count is silently truncated by
    check_rollout rather than reported."""
    found = find_windows(sweep, 32, dt=50.0, theta0=-0.20, theta_tol=0.01,
                          n_steps=1, limit=2)
    assert found
    for window, _, _ in found:
        _, steps = parse_fixed_window(window)
        assert len(steps) == 2


def _mixed_dt_corner(monkeypatch, steps, temperature=0.72, T0=1.0, dt_scale=0.05):
    """A synthetic run with NON-UNIFORM gaps.

    The real test sweep has uniform spacing, so first-hop-only matching and
    no-tolerance matching give identical answers on it -- both mutations
    passed against it, verified. Discriminating needs a window whose hops
    differ, which is exactly the "dt_max=125" case: a MAXIMUM over the window,
    so a window mixing dt=25 and dt=125 is a different population that merely
    touches the corner.
    """
    import evaluation.find_windows as fw

    run = pathlib.Path("/synthetic/T720_n010_s0")

    class _Meta:
        pass
    meta = _Meta()
    meta.temperature, meta.T0, meta.dt = temperature, T0, dt_scale

    monkeypatch.setattr(fw, "complete_run_dirs", lambda *a, **k: [run])
    monkeypatch.setattr(fw, "build_good_steps", lambda *a, **k: {run: steps})
    monkeypatch.setattr(fw.load, "read_metadata", lambda p: meta)
    return fw


def test_a_window_mixing_dts_is_REJECTED(monkeypatch):
    """steps 0, 500, 3000 -> hops of 25 and 125. Asking for dt=125 must not
    match it on the strength of its second hop alone."""
    fw = _mixed_dt_corner(monkeypatch, [0, 500, 3000])
    assert fw.find_windows("x", 32, dt=125.0, theta0=-0.28) == []


def test_a_window_whose_FIRST_hop_matches_but_later_hop_does_not_is_rejected(monkeypatch):
    """
    steps 0, 2500, 3000 -> hops of 125 then 25. Checking only dts[0] accepts
    this; checking all() rejects it.

    The ORDER matters and my first attempt had it backwards: a window of
    25-then-125 is rejected by both rules, so it cannot tell them apart.
    Verified -- the first-hop mutation passed against that case.
    """
    fw = _mixed_dt_corner(monkeypatch, [0, 2500, 3000])
    assert fw.find_windows("x", 32, dt=125.0, theta0=-0.28) == []


def test_a_window_with_ALL_hops_in_the_corner_is_accepted(monkeypatch):
    """steps 0, 2500, 5000 -> both hops exactly 125."""
    fw = _mixed_dt_corner(monkeypatch, [0, 2500, 5000])
    found = fw.find_windows("x", 32, dt=125.0, theta0=-0.28)
    assert len(found) == 1
    assert found[0][1] == pytest.approx(125.0)


def test_the_dt_tolerance_actually_excludes(monkeypatch):
    """
    GUARDS a tolerance read but not applied. steps giving hops of 50 must not
    satisfy a request for dt=125.
    """
    fw = _mixed_dt_corner(monkeypatch, [0, 1000, 2000])
    assert fw.find_windows("x", 32, dt=125.0, theta0=-0.28) == []
    assert fw.find_windows("x", 32, dt=50.0, theta0=-0.28), "the same steps ARE a dt=50 corner"


def test_EVERY_hop_must_be_in_the_corner(sweep):
    """
    GUARDS matching a window on its first transition alone. "dt_max=125" is a
    MAXIMUM over the window, so a window mixing dt=25 and dt=125 is not the
    corner being investigated -- it is a different population that happens to
    touch it.
    """
    found = find_windows(sweep, 32, dt=50.0, theta0=-0.20, theta_tol=0.01, limit=6)
    import utils.load_datasets as load
    for window, _, _ in found:
        run_dir, steps = parse_fixed_window(window)
        scale = load.read_metadata(run_dir / "metadata.txt").dt
        hops = [(steps[i + 1] - steps[i]) * scale for i in range(len(steps) - 1)]
        assert all(h == pytest.approx(50.0, rel=0.05) for h in hops), hops


def test_a_corner_that_does_not_exist_returns_empty_not_garbage(sweep):
    """A miss must be a miss. Returning the nearest windows instead would send
    the reader to investigate a population they did not ask about."""
    assert find_windows(sweep, 32, dt=125.0, theta0=-0.28) == []


def test_theta_tolerance_actually_excludes(sweep):
    """GUARDS a tolerance that is read but not applied."""
    assert find_windows(sweep, 32, dt=50.0, theta0=+0.40, theta_tol=0.01) == []
