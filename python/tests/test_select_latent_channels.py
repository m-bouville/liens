"""
Tests for select_latent_channels: pruning latent channels in a stage-1/2 checkpoint.

This is checkpoint SURGERY -- a wrong slice silently corrupts a checkpoint whose
only later validation is a strict load. So the tests pin the three things that
matter: (1) the right dimension is sliced on every tensor role (encoder proj OUT,
decoder IN, stats-head flattened input), (2) the kept-channel CONTENT is exactly
the selected indices (not a wrong permutation), (3) tensors with no latent-channel
dim are untouched -- including the channels==spatial==8 clash that a naive
"slice every dim of size 8" would get wrong.
"""
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from evaluation.select_latent_channels import select_latent_channels


def _make_checkpoint(C=8, S=8, trunk=32, hidden=64, Ns=10, include_stats1=False):
    ck = {
        "config": {"latent_channels": C, "latent_spatial_size": S,
                   "latent_channels_decoder": C,
                   "stream_configs": {"state": {"channels": C}, "deriv": {"channels": C}}},
        "model_state": {
            "encoder.proj_state.weight": torch.randn(C, trunk, 1, 1),   # OUT = channels
            "encoder.proj_state.bias":   torch.randn(C),
            "encoder.proj_deriv.weight": torch.randn(C, trunk, 1, 1),
            "encoder.proj_deriv.bias":   torch.randn(C),
            "decoder.expand.weight":     torch.randn(trunk, C, 1, 1),   # IN = channels
            "encoder.trunk.0.weight":    torch.randn(trunk, 1, 3, 3),   # NO channel dim
        },
        "stats_head_state": {
            "net.0.weight": torch.randn(hidden, C * S * S),             # flattened latent input
            "net.0.bias":   torch.randn(hidden),
            "net.2.weight": torch.randn(Ns, hidden),                    # output -- untouched
            "net.2.bias":   torch.randn(Ns),
        },
        "epoch": 6, "val_loss": 2.9, "val_loss_ema": 3.4, "test_dirs": ["/x"],
    }
    if include_stats1:
        ck["stats_head1_state"] = {"net.0.weight": torch.randn(hidden, C * S * S),
                                   "net.0.bias": torch.randn(hidden)}
    return ck


def _roundtrip(tmp_path, keep, **kw):
    ck = _make_checkpoint(**kw)
    inp, out = tmp_path / "in.pt", tmp_path / "out.pt"
    torch.save(ck, inp)
    select_latent_channels(inp, out, keep)
    return ck, torch.load(out, map_location="cpu", weights_only=False)


def test_encoder_projection_sliced_on_out_axis_with_correct_content(tmp_path):
    keep = [1, 2, 5, 6]
    src, r = _roundtrip(tmp_path, keep)
    w = r["model_state"]["encoder.proj_state.weight"]
    assert w.shape == (4, 32, 1, 1)                       # OUT axis 8 -> 4
    assert torch.allclose(w, src["model_state"]["encoder.proj_state.weight"][keep])
    assert torch.allclose(r["model_state"]["encoder.proj_state.bias"],
                          src["model_state"]["encoder.proj_state.bias"][keep])


def test_decoder_sliced_on_IN_axis_not_out(tmp_path):
    keep = [1, 2, 5, 6]
    src, r = _roundtrip(tmp_path, keep)
    w = r["model_state"]["decoder.expand.weight"]
    assert w.shape == (32, 4, 1, 1)                       # IN axis sliced, OUT (32) kept
    assert torch.allclose(w, src["model_state"]["decoder.expand.weight"][:, keep])


def test_stats_head_flattened_input_sliced_across_all_spatial(tmp_path):
    keep = [1, 2, 5, 6]
    S = 8
    src, r = _roundtrip(tmp_path, keep)
    w = r["stats_head_state"]["net.0.weight"]
    assert w.shape == (64, 4 * S * S)                     # 512 -> 256
    expect = (src["stats_head_state"]["net.0.weight"]
              .reshape(64, 8, S, S)[:, keep].reshape(64, 4 * S * S))
    assert torch.allclose(w, expect)
    assert torch.allclose(r["stats_head_state"]["net.2.weight"],
                          src["stats_head_state"]["net.2.weight"])   # output untouched


def test_non_channel_tensors_and_spatial_untouched(tmp_path):
    """The trunk conv (no size-8 dim) is copied verbatim; nothing keys off the
    spatial 8 (channels==spatial clash)."""
    src, r = _roundtrip(tmp_path, [1, 2, 5, 6])
    assert torch.equal(r["model_state"]["encoder.trunk.0.weight"],
                       src["model_state"]["encoder.trunk.0.weight"])


def test_config_updated_all_streams_and_decoder(tmp_path):
    _src, r = _roundtrip(tmp_path, [0, 3, 5, 7])
    assert r["config"]["latent_channels"] == 4
    assert r["config"]["latent_channels_decoder"] == 4
    assert r["config"]["stream_configs"]["state"]["channels"] == 4
    assert r["config"]["stream_configs"]["deriv"]["channels"] == 4


def test_stale_val_loss_blanked(tmp_path):
    """The 8-channel val_loss is not a fair bar for the pruned model's resume."""
    _src, r = _roundtrip(tmp_path, [1, 2, 5, 6])
    assert r["val_loss"] == float("inf")
    assert r["val_loss_ema"] == float("inf")
    assert r["channels_selected_from"] == {"old_channels": 8, "kept": [1, 2, 5, 6]}


def test_second_stats_head_also_sliced_when_present(tmp_path):
    _src, r = _roundtrip(tmp_path, [1, 2, 5, 6], include_stats1=True)
    assert r["stats_head1_state"]["net.0.weight"].shape == (64, 4 * 8 * 8)


def test_rejects_out_of_range_and_full_selection(tmp_path):
    ck = _make_checkpoint()
    inp = tmp_path / "in.pt"; torch.save(ck, inp)
    with pytest.raises(ValueError, match="out of range"):
        select_latent_channels(inp, tmp_path / "o1.pt", [1, 2, 8])          # 8 >= C
    with pytest.raises(ValueError, match="selects all"):
        select_latent_channels(inp, tmp_path / "o2.pt", list(range(8)))     # nothing pruned


def test_kept_indices_normalised_sorted_unique(tmp_path):
    """Order-independent + dedup: [6,2,2,1,5] -> keep {1,2,5,6}."""
    src, r = _roundtrip(tmp_path, [6, 2, 2, 1, 5])
    assert r["model_state"]["encoder.proj_state.weight"].shape == (4, 32, 1, 1)
    assert torch.allclose(r["model_state"]["encoder.proj_state.weight"],
                          src["model_state"]["encoder.proj_state.weight"][[1, 2, 5, 6]])
