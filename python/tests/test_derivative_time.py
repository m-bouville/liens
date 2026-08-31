"""derivative_time selects WHEN the rollout's derivative is evaluated, giving
the four (source x time) options. 'initial' freezes the seed z1_sequence[:,0]
(the encoder z1 for source='z1', the real backward quotient q_0 for
'previous_quotient') for every step -- no feedback; 'previous' recomputes it
(live quotient, or teacher-forced z1). It is a _MEANING_FIELDS field, so it
round-trips through checkpoints and defaults to 'previous' for old ones."""
import torch
import pytest

from models.constants import N_THETA
from models.latent_dynamics import (LatentDynamics, integration_kwargs_from_config,
                                     _MEANING_FIELDS)


def _model(derivative_time, derivative_source="previous_quotient", perturb_f=False):
    m = LatentDynamics(latent_channels=8, dynamics_mode="deriv_linear",
                       derivative_source=derivative_source,
                       derivative_time=derivative_time, time_coordinate="t")
    if perturb_f:
        with torch.no_grad():
            m.net[-1].weight.normal_(0, 0.05); m.net[-1].bias.normal_(0, 0.05)
    m.eval()
    return m


def _inputs(n=4, B=2, C=8):
    torch.manual_seed(0)
    return (torch.randn(B, C, 8, 8), torch.randn(B, n + 1, C, 8, 8),
            torch.full((B, n), 0.5), torch.zeros(B, N_THETA))


def test_initial_freezes_seed_for_every_step():
    z0, z1seq, dts, th = _inputs()
    out = _model("initial").rollout(z0, z1seq, dts, th, z1_resync=False)
    seed = z1seq[:, 0]
    expected = z0.clone()
    for k in range(1, dts.shape[1] + 1):          # f==0 -> z0_k = z0 + k*seed*dt
        expected = expected + seed * 0.5
        assert torch.allclose(out[:, k], expected, atol=1e-5)


def test_initial_and_previous_diverge_once_f_nonzero():
    z0, z1seq, dts, th = _inputs()
    mi, mp = _model("initial", perturb_f=True), _model("previous", perturb_f=True)
    mp.load_state_dict(mi.state_dict())           # identical f
    oi = mi.rollout(z0, z1seq, dts, th, z1_resync=False)
    op = mp.rollout(z0, z1seq, dts, th, z1_resync=False)
    assert torch.allclose(oi[:, 1], op[:, 1], atol=1e-6)      # step 1: both seed
    assert not torch.allclose(oi[:, 2], op[:, 2], atol=1e-5)  # step 2: live re-quotients


def test_initial_works_for_both_sources():
    z0, z1seq, dts, th = _inputs()
    for src in ("z1", "previous_quotient"):
        out = _model("initial", derivative_source=src).rollout(
            z0, z1seq, dts, th, z1_resync=False)
        assert torch.allclose(out[:, 1], z0 + z1seq[:, 0] * 0.5, atol=1e-5)


def test_z1_previous_autonomous_still_raises():
    z0, z1seq, dts, th = _inputs()
    with pytest.raises(ValueError, match="no z1-update equation"):
        _model("previous", derivative_source="z1").rollout(
            z0, z1seq, dts, th, z1_resync=False)


def test_meaning_field_roundtrip_and_default():
    assert "derivative_time" in _MEANING_FIELDS
    assert integration_kwargs_from_config({})["derivative_time"] == "previous"
    assert integration_kwargs_from_config(
        {"derivative_time": "initial"})["derivative_time"] == "initial"


def test_invalid_value_rejected():
    with pytest.raises(ValueError, match="derivative_time must be"):
        LatentDynamics(latent_channels=8, dynamics_mode="deriv_linear",
                       derivative_time="bogus")
