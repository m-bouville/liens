"""Coverage for derivative_source='previous_quotient' (the q-scheme data path).

Builds a REAL MicrostructureEvolutionDataset and asserts __getitem__ serves the
backward quotient q_k = (z0_k - z0_{k-1})/du_{k-1} as window_deriv -- in the
model's own coordinate -- with the step-0 seed taken from the real predecessor
frame (or from z1 at a run start). This is what closes the two q-scheme gaps:
teacher-forced training on q, and a rollout step-0 seed that is the true
previous derivative rather than z1.
"""
import math

import torch
import torch.nn as nn
import pytest

from training.datasets import MicrostructureEvolutionDataset
from models.constants import LATENT_SPATIAL_SIZE


class _TwoStreamEncoder(nn.Module):
    def __init__(self, size: int = 64, latent_channels: int = 4):
        super().__init__()
        stride = size // LATENT_SPATIAL_SIZE
        self.state = nn.Conv2d(1, latent_channels, kernel_size=stride, stride=stride)
        self.deriv = nn.Conv2d(1, latent_channels, kernel_size=stride, stride=stride)

    def forward(self, x):
        return {"state": self.state(x), "deriv": self.deriv(x)}


def _ds(run_dir, time_coordinate, derivative_source, encoder=None, **extra):
    return MicrostructureEvolutionDataset(
        [run_dir], encoder=encoder or _TwoStreamEncoder(), window_length=3,
        min_step=1000, min_stdev_phi=None, encode_both_streams=True,
        time_coordinate=time_coordinate, derivative_source=derivative_source,
        **extra)


def test_quotient_matches_z0_backward_difference(tmp_run_dir):
    """window_deriv[k] must equal (z0_k - z0_{k-1})/dt for k>=1, straight from
    the returned window -- the derivative in the model's own coordinate."""
    run_dir, _ = tmp_run_dir
    enc = _TwoStreamEncoder()
    ds = _ds(run_dir, "t", "previous_quotient", encoder=enc)
    window, window_deriv, dt_window, _ = ds[0]           # first window has a predecessor? see below
    for k in range(1, window.shape[0]):
        expected = (window[k] - window[k - 1]) / dt_window[k - 1]
        assert torch.allclose(window_deriv[k], expected, atol=1e-5), \
            f"window_deriv[{k}] is not the backward quotient of z0"


def test_step0_seed_uses_predecessor_not_z1(tmp_run_dir):
    """For a window with a real predecessor, window_deriv[0] must be the
    predecessor-based quotient -- DIFFERENT from the encoder's z1 at frame 0."""
    run_dir, _ = tmp_run_dir
    enc = _TwoStreamEncoder()
    ds_q = _ds(run_dir, "t", "previous_quotient", encoder=enc)
    ds_z1 = _ds(run_dir, "t", "z1", encoder=enc)
    # kept steps are [1000,2000,3000,4000]; window_length=3. The window whose
    # start is NOT the first kept frame has a predecessor. Index 1 starts at
    # kept[1]=2000, predecessor kept[0]=1000.
    if len(ds_q) < 2:
        pytest.skip("need >=2 windows to reach one with a predecessor")
    w_q, deriv_q, dt_q, _ = ds_q[1]
    w_z1, deriv_z1, _, _ = ds_z1[1]
    # the predecessor quotient is not the z1 head output
    assert not torch.allclose(deriv_q[0], deriv_z1[0], atol=1e-4), \
        "step-0 seed should be the predecessor quotient, not z1"
    # and the state stream is identical between the two sources (only deriv changes)
    assert torch.allclose(w_q, w_z1)


def test_quotient_coordinate_is_du_in_log10_t(tmp_run_dir):
    """In log10_t mode the quotient divides by Delta-u (not Delta-t), so it is
    dz0/du -- the same coordinate z~1 lives in."""
    run_dir, _ = tmp_run_dir
    enc = _TwoStreamEncoder()
    ds = _ds(run_dir, "log10_t", "previous_quotient", encoder=enc, return_phys_dt=True)
    window, window_deriv, du_window, _dtphys, _ = ds[0]
    for k in range(1, window.shape[0]):
        expected = (window[k] - window[k - 1]) / du_window[k - 1]
        assert torch.allclose(window_deriv[k], expected, atol=1e-5)


def test_previous_quotient_requires_both_streams(tmp_run_dir):
    run_dir, _ = tmp_run_dir
    with pytest.raises(ValueError, match="encode_both_streams"):
        MicrostructureEvolutionDataset(
            [run_dir], encoder=_TwoStreamEncoder(), window_length=3,
            min_step=1000, min_stdev_phi=None, encode_both_streams=False,
            derivative_source="previous_quotient")


def test_previous_quotient_forbids_augment(tmp_run_dir):
    run_dir, _ = tmp_run_dir
    with pytest.raises(ValueError, match="augment"):
        MicrostructureEvolutionDataset(
            [run_dir], encoder=_TwoStreamEncoder(), window_length=3,
            min_step=1000, min_stdev_phi=None, encode_both_streams=True,
            augment=True, derivative_source="previous_quotient")


def test_z1_source_unchanged(tmp_run_dir):
    """derivative_source='z1' (default) must be byte-identical to not passing it."""
    run_dir, _ = tmp_run_dir
    enc = _TwoStreamEncoder()
    a = _ds(run_dir, "t", "z1", encoder=enc)[0]
    b = MicrostructureEvolutionDataset(
        [run_dir], encoder=enc, window_length=3, min_step=1000,
        min_stdev_phi=None, encode_both_streams=True, time_coordinate="t")[0]
    for x, y in zip(a, b):
        assert torch.allclose(x, y) if torch.is_tensor(x) else x == y


def test_run_start_predecessor_is_encoded_and_used(tmp_run_dir):
    """The 'compute before filter' fix: a window at the run's FIRST kept frame
    gets q_0 from the ENCODED predecessor (the filtered save_steps frame before
    it), a real backward quotient -- not the z1 fallback. min_step=2000 keeps
    [2000,3000,4000]; the predecessor 1000 was filtered but must still seed q_0."""
    run_dir, _ = tmp_run_dir
    enc = _TwoStreamEncoder()
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=enc, window_length=3, min_step=2000, min_stdev_phi=None,
        encode_both_streams=True, time_coordinate="t",
        derivative_source="previous_quotient")
    assert ds._run_pred_z0[0] is not None, "run-start predecessor was not encoded"
    window, window_deriv, dt_window, _ = ds[0]          # start == 0 (first kept frame)

    # q_0 is the predecessor quotient, exactly (z0_first - z0_pred)/du_pred
    pred_z0, du_pred = ds._run_pred_z0[0], ds._run_pred_du[0]   # (2000-1000)*0.05 = 50
    assert abs(du_pred - 50.0) < 1e-6
    expected = (window[0] - pred_z0) / du_pred
    assert torch.allclose(window_deriv[0], expected, atol=1e-5), \
        "q_0 is not the encoded-predecessor backward quotient"

    # ... and it is NOT the z1 fallback the old code would have used
    z1_ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=enc, window_length=3, min_step=2000, min_stdev_phi=None,
        encode_both_streams=True, time_coordinate="t")
    _, z1_deriv, _, _ = z1_ds[0]
    assert not torch.allclose(window_deriv[0], z1_deriv[0], atol=1e-4), \
        "run-start q_0 fell back to z1 instead of using the predecessor"


def test_absolute_first_snapshot_falls_back_to_z1(tmp_run_dir):
    """When the first kept frame's predecessor would be step 0 (t=0, singular),
    there is genuinely no predecessor -> z1 fallback for that one frame."""
    run_dir, _ = tmp_run_dir
    enc = _TwoStreamEncoder()
    # min_step=1 keeps [1000,2000,3000,4000]; predecessor of 1000 is step 0 -> guarded
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=enc, window_length=3, min_step=1, min_stdev_phi=None,
        encode_both_streams=True, time_coordinate="t",
        derivative_source="previous_quotient")
    assert ds._run_pred_z0[0] is None, "step-0 predecessor should be skipped, not encoded"


def test_run_start_predecessor_log10_t(tmp_run_dir):
    """log10_t is the production coordinate: the predecessor's du must be
    log10(first_kept/pred_step), and q_0 the z0 difference over THAT du. This is
    the path the t-mode test cannot exercise (du = physical dt there)."""
    run_dir, _ = tmp_run_dir
    enc = _TwoStreamEncoder()
    # min_step=2000 keeps [2000,3000,4000]; predecessor 1000 (nonzero -> valid in log10_t)
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=enc, window_length=3, min_step=2000, min_stdev_phi=None,
        encode_both_streams=True, time_coordinate="log10_t",
        derivative_source="previous_quotient")
    assert ds._run_pred_z0[0] is not None
    du_pred = ds._run_pred_du[0]
    assert abs(du_pred - math.log10(2000 / 1000)) < 1e-9, \
        "predecessor du must be log10(t_first/t_pred) in log10_t mode"
    window, window_deriv, du_window, _ = ds[0]
    expected = (window[0] - ds._run_pred_z0[0]) / du_pred
    assert torch.allclose(window_deriv[0], expected, atol=1e-5)


def test_log10_t_step0_predecessor_guarded(tmp_run_dir):
    """min_step=1 keeps [1000,...]; predecessor would be step 0 -> log10 singular
    -> guarded to None, z1 fallback for that run's first frame only."""
    run_dir, _ = tmp_run_dir
    ds = MicrostructureEvolutionDataset(
        [run_dir], encoder=_TwoStreamEncoder(), window_length=3, min_step=1,
        min_stdev_phi=None, encode_both_streams=True, time_coordinate="log10_t",
        derivative_source="previous_quotient")
    assert ds._run_pred_z0[0] is None
