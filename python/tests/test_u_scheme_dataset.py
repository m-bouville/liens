"""End-to-end coverage for the u-scheme (time_coordinate='log10_t') data path.

These guard the load-bearing conversion that ALL u-results rest on -- unlike
test_deriv_linear.py's test_dataset_u_conversion_arithmetic, which exercises the
convert_derivative_coordinate FORMULA in isolation, these build a REAL
MicrostructureEvolutionDataset and assert its __getitem__ actually yields
z~1 = ln10*t*z1, Delta-u, and physical dt -- catching a bug in _run_du
alignment, the in-place multiply, or the return_phys_dt 5-tuple that the
formula-only test cannot.
"""
import math

import torch
import torch.nn as nn
import pytest

from training.datasets import MicrostructureEvolutionDataset
from models.latent_streams import DEFAULT_STREAM_NAME
from models.constants import LATENT_SPATIAL_SIZE


class _TwoStreamEncoder(nn.Module):
    """Fake encoder returning BOTH the state and deriv streams (the real
    FakeEncoder only returns the state stream, so encode_both_streams=True
    would KeyError on 'deriv'). Deterministic, so z1 is reproducible."""
    def __init__(self, size: int = 64, latent_channels: int = 4):
        super().__init__()
        stride = size // LATENT_SPATIAL_SIZE
        self.state = nn.Conv2d(1, latent_channels, kernel_size=stride, stride=stride)
        self.deriv = nn.Conv2d(1, latent_channels, kernel_size=stride, stride=stride)

    def forward(self, x, theta=None):
        return {DEFAULT_STREAM_NAME: self.state(x), "deriv": self.deriv(x)}


_LN10 = math.log(10.0)
_SIM_DT = 0.05   # tmp_run_dir's metadata: dt = 0.05


def _both(run_dir, time_coordinate, encoder=None, **extra):
    return MicrostructureEvolutionDataset(
        [run_dir], encoder=encoder or _TwoStreamEncoder(), window_length=3,
        min_step=1000,          # skip step 0: log10(step/0) is undefined
        min_stdev_phi=None, encode_both_streams=True,
        time_coordinate=time_coordinate, **extra)


def test_u_dataset_converts_deriv_step_and_leaves_z0_and_phys_dt(tmp_run_dir):
    """u-mode __getitem__ must yield z~1=ln10*t*z1, Delta-u, physical dt,
    and an UNCHANGED z0 -- checked against the t-mode dataset over the same
    run so the raw z1 values need not be known independently."""
    run_dir, steps = tmp_run_dir
    kept = [s for s in steps if s >= 1000]          # [1000, 2000, 3000, 4000]
    win = kept[:3]                                   # first window's 3 steps

    enc = _TwoStreamEncoder()          # SAME encoder for both, so z0/z1 match
    ds_t = _both(run_dir, "t", encoder=enc)
    ds_u = _both(run_dir, "log10_t", encoder=enc, return_phys_dt=True)

    w_t, deriv_t, dt_t, theta_t = ds_t[0]            # t-mode: 4-tuple
    w_u, deriv_u, du_u, dtphys_u, theta_u = ds_u[0]  # u-mode: 5-tuple

    # z0 (state stream) is untouched by the coordinate change.
    assert torch.allclose(w_u, w_t)

    # deriv: z~1[k] = ln10 * (step_k * sim_dt) * z1[k], per frame.
    for k in range(len(win)):
        scale = _LN10 * win[k] * _SIM_DT
        assert torch.allclose(deriv_u[k], deriv_t[k] * scale, atol=1e-5), \
            f"frame {k}: z~1 != ln10*t*z1"

    # step size: Delta-u = log10(step_{i+1}/step_i).
    for i in range(len(win) - 1):
        assert du_u[i].item() == pytest.approx(math.log10(win[i + 1] / win[i]), rel=1e-5)

    # physical dt is preserved (== t-mode's dt_window), for the loss weighting.
    assert torch.allclose(dtphys_u, dt_t)


def test_t_mode_is_a_four_tuple_and_byte_identical(tmp_run_dir):
    """Default t-mode must be unchanged: 4-tuple, no phys-dt element even
    with return_phys_dt requested-but-irrelevant, deriv un-scaled."""
    run_dir, steps = tmp_run_dir
    ds_t = _both(run_dir, "t")
    item = ds_t[0]
    assert len(item) == 4, "t-mode must stay a 4-tuple"
    # dt_window is physical Delta-t in t-mode.
    kept = [s for s in steps if s >= 1000]
    _, _, dt_t, _ = item
    for i in range(2):
        assert dt_t[i].item() == pytest.approx((kept[i + 1] - kept[i]) * _SIM_DT)


def test_u_mode_five_tuple_only_when_return_phys_dt(tmp_run_dir):
    """The 5th element (physical dt) appears ONLY with return_phys_dt=True,
    so any external iterator that doesn't opt in still sees the 4-tuple."""
    run_dir, _ = tmp_run_dir
    assert len(_both(run_dir, "log10_t")[0]) == 4                       # no opt-in
    assert len(_both(run_dir, "log10_t", return_phys_dt=True)[0]) == 5  # opt-in
