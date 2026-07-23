"""
Tests for theta (temperature, centered at T0) conditioning of the
"deriv" latent stream -- see models/encoder.py's own docstring for the
full design rationale (in short: the driving force a(T)=a0*(T-T0)
genuinely vanishes near T0, so a state-only encoder can get the
DIRECTION of change right from the image alone but has no way to know
the correct MAGNITUDE without T; deliberately NOT also conditioned on
dt -- a materially different proposal, see that same docstring for why).

Covers two levels: Encoder's own FiLM mechanism in isolation, and its
threading through MicrostructureEvolutionDataset's bulk, cross-run
encoding pipeline (training/datasets.py) -- the latter matters
specifically because that pipeline batches frames from SEVERAL runs
(each with its own, different theta) into the same encode call, which
is exactly the kind of place a per-frame theta/data misalignment bug
would hide.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_encoder_theta_conditioning.py -v
"""
import numpy as np
import torch
import pytest

from models.encoder import Encoder
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from training.datasets import MicrostructureEvolutionDataset
from utils import load_datasets as load


def _make_encoder(seed=0, base_channels=8, size=64):
    """A real (not fake/stand-in) Encoder with one unconditioned stream
    ("state") and one theta-conditioned stream ("deriv") -- the actual
    production configuration train_stage1b builds, at test scale."""
    torch.manual_seed(seed)
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER, condition_on_theta=False),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.DECODER, condition_on_theta=True),
    }
    return Encoder(input_size=size, stream_configs=stream_configs, in_channels=1,
                    base_channels=base_channels, n_theta=1)


def _perturb_film(encoder, scale=0.5, seed=1):
    """FiLM's own final layer is zero-initialized (see
    _ThetaFiLMConditioner's own docstring) -- an exact no-op until
    perturbed away from that, which every test below that wants to
    observe a REAL theta effect needs to do first."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in encoder.theta_conditioners["deriv"].net.parameters():
            p.add_(torch.randn_like(p) * scale)
    return encoder


def _write_run(base_path, name, n_steps, temperature, size=64, seed=0):
    """A fake run directory in the format load.read_metadata/
    read_phi_half/snapshot_filename expect, with a controllable
    temperature (unlike test_datasets_construction_performance.py's own
    _write_run, which hardcodes one) -- every test here needs several
    runs at DIFFERENT temperatures specifically."""
    rng = np.random.RandomState(seed)
    run_dir = base_path / name
    run_dir.mkdir()
    steps = list(range(0, n_steps * 1000, 1000))
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", f"steps = {steps[-1]}",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", f"temperature = {temperature}", "kappa = 0.2",
        "mobility = 0.05", "phi0 = 0.0", "noise = 0.01", "seed = 1",
        "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    for step in steps:
        arr = rng.randn(size, size).astype("<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))
    return run_dir


# ---------------------------------------------------------------------
# Encoder's own FiLM mechanism, in isolation (no dataset involved)
# ---------------------------------------------------------------------

def test_film_is_exact_noop_at_zero_init():
    """The whole point of zero-initializing FiLM's own final layer (see
    _ThetaFiLMConditioner's own docstring): a freshly-constructed
    encoder must produce IDENTICAL output regardless of theta, for
    every stream -- training starts numerically identical to the
    unconditioned baseline."""
    encoder = _make_encoder()
    encoder.eval()
    x = torch.randn(3, 1, 64, 64)
    theta_a = torch.full((3, 1), 0.1)
    theta_b = torch.full((3, 1), 5.0)
    with torch.no_grad():
        z_a = encoder(x, theta=theta_a)
        z_b = encoder(x, theta=theta_b)
    assert torch.equal(z_a["state"], z_b["state"])
    assert torch.allclose(z_a["deriv"], z_b["deriv"], atol=1e-6)


def test_state_stream_isolated_from_theta_after_training():
    """After FiLM is perturbed away from zero-init (simulating real
    training), "state" must STILL be completely unaffected by theta --
    its own bottleneck reads the shared trunk features directly, never
    through any conditioner. Isolation between streams, not just at
    init, is the actual property that matters."""
    encoder = _make_encoder()
    _perturb_film(encoder)
    encoder.eval()
    x = torch.randn(3, 1, 64, 64)
    theta_a = torch.full((3, 1), 0.1)
    theta_b = torch.full((3, 1), 5.0)
    with torch.no_grad():
        z_a = encoder(x, theta=theta_a)
        z_b = encoder(x, theta=theta_b)
    assert torch.equal(z_a["state"], z_b["state"]), "state must stay theta-invariant even after FiLM is trained"
    assert not torch.allclose(z_a["deriv"], z_b["deriv"], atol=1e-6), \
        "deriv should now genuinely differ with theta -- otherwise this test isn't exercising anything"


def test_missing_theta_raises_when_a_stream_needs_it():
    encoder = _make_encoder()
    x = torch.randn(2, 1, 64, 64)
    with pytest.raises(ValueError, match="theta"):
        encoder(x)  # theta=None, but 'deriv' requires it


def test_unconditioned_only_model_needs_no_theta():
    """A model with zero conditioned streams (e.g. a stage-1a, state-only
    checkpoint) must work exactly as before -- no theta required at
    all, not even an error about it being missing."""
    torch.manual_seed(0)
    stream_configs = {"state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                                    mode=LatentStreamMode.AUTOENCODER)}
    encoder = Encoder(input_size=64, stream_configs=stream_configs, in_channels=1, base_channels=8)
    x = torch.randn(2, 1, 64, 64)
    z = encoder(x)  # no theta at all
    assert "state" in z


# ---------------------------------------------------------------------
# Threading through MicrostructureEvolutionDataset's bulk, cross-run
# encoding pipeline
# ---------------------------------------------------------------------

def test_theta_correctly_aligned_across_cross_run_buffer_flushes(tmp_path):
    """The critical integration property: MicrostructureEvolutionDataset
    buffers and encodes frames from SEVERAL runs together (see its own
    _flush_buffer docstring) -- with different run sizes and a small
    encode_batch_size, buffer flushes straddle run boundaries, mixing
    frames from multiple DIFFERENT-temperature runs into the same
    encode call. Confirms every run's own result matches independent
    per-run encoding (with that run's own explicit theta) exactly --
    the direct test for a per-frame theta/data misalignment bug."""
    run_dirs = [
        _write_run(tmp_path, "run_lowT", n_steps=7, temperature=0.3, seed=1),
        _write_run(tmp_path, "run_midT", n_steps=3, temperature=0.6, seed=2),
        _write_run(tmp_path, "run_highT", n_steps=5, temperature=0.9, seed=3),
    ]
    encoder = _make_encoder()
    _perturb_film(encoder)
    encoder.eval()

    ds = MicrostructureEvolutionDataset(
        run_dirs, encoder=encoder, window_length=2, min_step=0, min_stdev_phi=None,
        encode_batch_size=6, encode_both_streams=True,  # small, to force flushes mid-run-sequence
    )

    for run_idx, run_dir in enumerate(ds._run_dirs):
        metadata = load.read_metadata(run_dir / "metadata.txt")
        frames = torch.stack([
            torch.from_numpy(load.read_phi_half(run_dir / load.snapshot_filename(s),
                                                  metadata.nx, metadata.ny)).unsqueeze(0)
            for s in metadata.save_steps
        ])
        theta = torch.full((frames.size(0), 1), metadata.temperature - metadata.T0)
        with torch.no_grad():
            expected = encoder(frames, theta=theta)
        assert torch.allclose(ds._run_data[run_idx], expected["state"], atol=1e-6), \
            f"{run_dir.name}: state mismatch"
        assert torch.allclose(ds._run_data_deriv[run_idx], expected["deriv"], atol=1e-6), \
            f"{run_dir.name}: deriv mismatch -- likely a theta/frame alignment bug across the buffer"


def test_encoder_without_theta_support_still_works(tmp_path):
    """Backward compatibility: an encoder whose forward() has no theta
    parameter at all (e.g. any pre-existing test fixture, or a
    hypothetical future simplified stand-in) must keep working through
    the SAME dataset pipeline, completely unaffected -- the
    inspect.signature check in _flush_buffer must correctly detect this
    and never attempt to pass theta to it."""
    import torch.nn as nn
    from models.latent_streams import DEFAULT_STREAM_NAME

    class NoThetaEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, kernel_size=8, stride=8)

        def forward(self, x):  # deliberately NO theta parameter
            return {DEFAULT_STREAM_NAME: self.conv(x)}

    run_dirs = [_write_run(tmp_path, "run_a", n_steps=5, temperature=0.5, seed=1)]
    encoder = NoThetaEncoder()
    ds = MicrostructureEvolutionDataset(
        run_dirs, encoder=encoder, window_length=2, min_step=0, min_stdev_phi=None,
    )
    assert len(ds) > 0


def test_theta_conditioning_saved_and_restored_via_stream_config(tmp_path):
    """condition_on_theta must round-trip through the same
    channels/spatial_size/mode dict-serialization pattern
    stream_configs already uses for checkpoints (see
    resolve_stream_configs_from_checkpoint_config) -- a checkpoint
    saved with it must resolve back with it still True, and one saved
    without it (or predating the field entirely) must resolve back
    False, not raise a KeyError."""
    from models.latent_streams import resolve_stream_configs_from_checkpoint_config

    model_cfg_with = {
        "latent_channels": 4, "latent_spatial_size": 8, "recon_stream_name": "state",
        "stream_configs": {
            "state": {"channels": 4, "spatial_size": 8, "mode": "autoencoder"},
            "deriv": {"channels": 4, "spatial_size": 8, "mode": "decoder", "condition_on_theta": True},
        },
    }
    resolved, _ = resolve_stream_configs_from_checkpoint_config(model_cfg_with)
    assert resolved["deriv"].condition_on_theta is True
    assert resolved["state"].condition_on_theta is False

    # Predates the field entirely (no "condition_on_theta" key at all) --
    # must default to False, not raise.
    model_cfg_without = {
        "latent_channels": 4, "latent_spatial_size": 8, "recon_stream_name": "state",
        "stream_configs": {
            "state": {"channels": 4, "spatial_size": 8, "mode": "autoencoder"},
            "deriv": {"channels": 4, "spatial_size": 8, "mode": "decoder"},
        },
    }
    resolved2, _ = resolve_stream_configs_from_checkpoint_config(model_cfg_without)
    assert resolved2["deriv"].condition_on_theta is False
