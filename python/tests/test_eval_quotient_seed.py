"""compute_trajectory must build the eval derivative channel the SAME way the
q-scheme dataset builds it for training: the backward quotient of z0 at every
frame (q_0 from the real predecessor), not the encoder z1. Otherwise a q-model
is plotted outside its trained regime and the step-1 number is an artefact.
Metadata reading is stubbed so the test exercises the arithmetic, not disk I/O.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import evaluation.compare_f_theta as cf


class _Meta:
    save_steps = [0, 1000, 2000, 3000, 4000]
    temperature, T0, dt, nx, ny = 0.8, 1.0, 0.05, 8, 8


class _Enc(nn.Module):
    def __init__(self):
        super().__init__(); self.c = nn.Conv2d(1, 2, 1, bias=False)
        nn.init.constant_(self.c.weight, 1.0)
    def forward(self, x, theta=None):
        z = self.c(x); return {"state": z, "deriv": torch.full_like(z, 7.0)}


class _AE:
    def __init__(self): self.encoder = _Enc(); self.decoder = lambda z: z[:, :1]


class _FT:
    def __init__(self, src): self.derivative_source = src; self.time_coordinate = "t"
    def rollout(self, z0, z1_sequence, dts, theta, z1_resync=False):
        self.seen = z1_sequence.clone()
        return z0.unsqueeze(1).expand(-1, dts.shape[1] + 1, -1, -1, -1).clone()


def _patch(monkeypatch):
    monkeypatch.setattr(cf.load, "read_metadata", lambda p: _Meta())
    monkeypatch.setattr(cf.load, "snapshot_filename", lambda s: f"t{s:08d}")
    monkeypatch.setattr(cf.load, "read_phi_half",
                        lambda path, nx, ny: np.full(
                            (ny, nx), int(str(path).split("t")[-1]) / 1000.0, dtype=np.float32))
    monkeypatch.setattr(cf, "resolve_stream_configs_from_checkpoint_config",
                        lambda c: (None, "state"))


def test_q_model_eval_uses_backward_quotient_every_frame(monkeypatch, tmp_path):
    _patch(monkeypatch)
    ft = _FT("previous_quotient")
    cf.compute_trajectory(tmp_path, [1000, 2000, 3000, 4000], _AE(), ft, {"size": 8}, "cpu")
    z1 = ft.seen[0]                       # (n, C, 8, 8)
    # every frame is (z0_k - z0_{k-1})/dt = 1/50 = 0.02 here; NONE is the deriv head 7.0
    assert torch.allclose(z1, torch.full_like(z1, 0.02), atol=1e-4)
    assert not torch.any((z1 - 7.0).abs() < 1e-3)


def test_z1_model_eval_untouched(monkeypatch, tmp_path):
    _patch(monkeypatch)
    ft = _FT("z1")
    cf.compute_trajectory(tmp_path, [1000, 2000, 3000, 4000], _AE(), ft, {"size": 8}, "cpu")
    # encoder deriv head passes through unchanged
    assert torch.allclose(ft.seen[0], torch.full_like(ft.seen[0], 7.0), atol=1e-4)


def test_q_model_eval_log10_t_coordinate(monkeypatch, tmp_path):
    """log10_t (the production coordinate): the eval quotient divides the z0
    difference by du = log10(t_{k+1}/t_k), matching z~1's coordinate."""
    _patch(monkeypatch)
    ft = _FT("previous_quotient"); ft.time_coordinate = "log10_t"
    cf.compute_trajectory(tmp_path, [1000, 2000, 4000], _AE(), ft, {"size": 8}, "cpu")
    z1 = ft.seen[0]
    # z0(step)=step/1000 (encoder weight 1, constant field). du_0=log10(2), du_1=log10(2).
    # q_1=(2-1)/log10(2), q_2=(4-2)/log10(2). q_0's predecessor is step 0 --
    # SINGULAR in log10_t -> falls back to the encoder deriv, WHICH IS z~1 HERE:
    # in log10_t the whole channel lives in the du-coordinate, so the fallback is
    # the converted z~1_0 = z1 * ln10 * dt * t_0 = 7 * ln10 * 0.05 * 1000, not the
    # raw head output. (Exactly what the dataset's run-start fallback serves.)
    q1 = (2.0 - 1.0) / math.log10(2.0)
    q2 = (4.0 - 2.0) / math.log10(2.0)
    z1_tilde_0 = 7.0 * math.log(10.0) * 0.05 * 1000
    assert abs(z1[0, 0, 0, 0].item() - z1_tilde_0) < 1e-2, \
        "singular step-0 pred -> z~1 (du-coordinate) fallback"
    assert abs(z1[1, 0, 0, 0].item() - q1) < 1e-3
    assert abs(z1[2, 0, 0, 0].item() - q2) < 1e-3


def test_q_model_run_start_falls_back_to_encoder_z1(monkeypatch, tmp_path):
    """A window starting at the run's first saved frame has no predecessor:
    step 0 keeps the encoder z1 (matching the dataset's run-start fallback);
    later frames are still the quotient."""
    _patch(monkeypatch)
    ft = _FT("previous_quotient")
    cf.compute_trajectory(tmp_path, [0, 1000, 2000], _AE(), ft, {"size": 8}, "cpu")
    z1 = ft.seen[0]
    assert abs(z1[0, 0, 0, 0].item() - 7.0) < 1e-4          # fallback
    assert abs(z1[1, 0, 0, 0].item() - 0.02) < 1e-4         # (1-0)/50 quotient
