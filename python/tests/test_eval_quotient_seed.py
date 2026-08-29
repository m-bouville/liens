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
