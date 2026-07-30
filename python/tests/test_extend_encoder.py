"""
Tests for training/extend_encoder.py -- the standalone replacement for
stage 1b's own checkpoint-saving setup phase (see extend_encoder.py's
own module docstring for the full rationale; stage 1b itself has since
been removed entirely). Compares the loaded (not randomly initialized)
parts of its own output directly against stage 1a's own saved
checkpoint on disk, since the actual regression risk here is "does
this correctly transfer stage 1a's own trained weights unchanged",
not "is the math right in isolation" (already covered by
models/encoder.py's, models/autoencoder.py's, and
training/stats_head.py's own test files).

Deliberately does NOT assert on the deriv bottleneck/theta_conditioner
weights matching anything -- freshly, randomly initialized, with no
prior counterpart to compare against at all (stage 1a never had a
deriv stream). What's actually asserted is the strictly-more-robust
invariant: every WEIGHT THAT WAS LOADED (not randomly initialized)
matches byte-for-byte, and the freshly-initialized parts exist with
the right shape/location and nothing else.
"""
import pytest

from conftest import cached_sweep
import torch
import pandas as pd

from training.train_stage1 import train_autoencoder
from training.extend_encoder import extend_state_checkpoint_with_deriv_stream
from models.latent_streams import LatentStreamMode
from utils import load_datasets as load


def _build_run_dir(base_dir, name, size=32):
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = [0, 1000, 2000, 3000]
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 3000",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        "seed = 1", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    for step in steps:
        value = step / 10000.0 + hash(name) % 100 / 10000.0
        arr = torch.full((size, size), value, dtype=torch.float16).numpy()
        arr.tofile(run_dir / load.snapshot_filename(step))
    df = pd.DataFrame({"avg_phi": [s / 10000.0 for s in steps]}, index=steps)
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")
    (run_dir / "COMPLETE").touch()
    return run_dir


def _build_sweep_uncached(tmp_path, n_runs=6, size=32):
    base_dir = tmp_path / f"{size}x{size}"
    run_names = [f"T800_n010_s{i}" for i in range(n_runs)]
    for name in run_names:
        _build_run_dir(base_dir, name, size=size)
    sweep_meta = "\n".join([
        f"Nx = {size}", f"Ny = {size}", "temperatures = 0.8", "noises = 0.01",
        f"seeds = {','.join(str(i) for i in range(n_runs))}", "subdirs =", *run_names,
    ])
    (base_dir / "metadata.txt").write_text(sweep_meta)
    return tmp_path


@pytest.fixture
def stage1a_checkpoint(tmp_path):
    base_path = _build_sweep(tmp_path)
    return train_autoencoder(
        size=32, base_path=base_path, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.3, test_fraction=0.2, num_workers=0, min_step=0, min_stdev_phi=None,
        stats0_weight=0.01, stat_names=["avg_phi"], checkpoint_path=tmp_path / "s1a.pt",
        device="cpu", seed=0, log_every_epoch=False, loss_curve_path=tmp_path / "c1a.png",
    )


def test_weight_transfer_matches_stage1a_source_byte_for_byte(tmp_path, stage1a_checkpoint):
    """The one thing that ISN'T random (loaded, not initialized) --
    trunk, state bottleneck, D0 -- must match stage 1a's own saved
    weights exactly, since nothing should change them between being
    saved there and loaded here. Compared directly against the raw
    stage 1a checkpoint on disk (not via an intermediate stage 1b pass
    -- see this module's own docstring: that comparison was useful
    while migrating off stage 1b, but stage 1b no longer exists at
    all, and this is actually the more direct test of the same
    property anyway)."""
    stage1a_state = torch.load(stage1a_checkpoint, map_location="cpu", weights_only=True)["model_state"]

    ext = extend_state_checkpoint_with_deriv_stream(resume_from=stage1a_checkpoint, device="cpu")
    new_state = ext.ae.state_dict()

    # stage 1a is a plain Autoencoder ("encoder."/"decoder." prefixes);
    # the extended model is a MultiStreamAutoencoder ("encoders.shared."/
    # "decoders.D0." -- see autoencoder.py's own docstring on why the
    # container holds a NAMED DICT, not a bare .encoder attribute).
    # Different prefixes for the SAME underlying weights -- translated
    # here for the comparison, not a sign anything is actually different.
    prefix_map = {"encoders.shared.": "encoder.", "decoders.D0.": "decoder."}
    checked = 0
    for new_key in new_state:
        for new_prefix, old_prefix in prefix_map.items():
            if new_key.startswith(new_prefix):
                # skip the deriv bottleneck/theta_conditioner -- freshly,
                # randomly initialized, has no counterpart in stage 1a at all
                if "bottlenecks.deriv" in new_key or "theta_conditioners.deriv" in new_key:
                    break
                old_key = old_prefix + new_key[len(new_prefix):]
                checked += 1
                assert old_key in stage1a_state, f"{old_key} (mapped from {new_key}) missing from stage 1a's own checkpoint"
                assert torch.equal(new_state[new_key], stage1a_state[old_key]), (
                    f"{new_key}: value mismatch vs stage 1a's own saved {old_key}"
                )
                break
    assert checked > 0, "test didn't actually check anything -- prefix map is stale"


def test_stats_head0_matches_stage1a_source_byte_for_byte(tmp_path, stage1a_checkpoint):
    stage1a_stats = torch.load(stage1a_checkpoint, map_location="cpu", weights_only=True)["stats_head_state"]

    ext = extend_state_checkpoint_with_deriv_stream(resume_from=stage1a_checkpoint, device="cpu")
    new_stats0 = ext.stats_head0.state_dict()

    assert set(new_stats0) == set(stage1a_stats)
    for k in new_stats0:
        assert torch.equal(new_stats0[k], stage1a_stats[k]), f"stats_head0.{k}: value mismatch"


def test_no_d1_or_deriv_pathway_exists(tmp_path, stage1a_checkpoint):
    """The actual point of this whole refactor -- confirmed structurally,
    not just by absence of a construction call. Both checks matter:
    "D1" isn't among the decoders (nothing built), AND "deriv" isn't
    among the pathways (MultiStreamAutoencoder's own PURE_LATENT filter
    correctly excluded it) -- see extend_encoder.py's own module
    docstring on why a real, previously-live crash (KeyError on
    ae.pathways["deriv"]) is exactly what this state prevents."""
    ext = extend_state_checkpoint_with_deriv_stream(resume_from=stage1a_checkpoint, device="cpu")
    assert "D1" not in ext.ae.decoders
    assert "deriv" not in ext.ae.pathways
    assert list(ext.ae.decoders.keys()) == ["D0"]


def test_deriv_stream_is_pure_latent_not_decoder(tmp_path, stage1a_checkpoint):
    """Distinguishes this from stage 1b's own choice explicitly -- an
    accurate structural claim (no decoder exists), not just an unused
    DECODER-mode stream sitting around."""
    ext = extend_state_checkpoint_with_deriv_stream(resume_from=stage1a_checkpoint, device="cpu")
    assert ext.stream_configs["deriv"].mode == LatentStreamMode.PURE_LATENT


def test_condition_on_theta_is_set_correctly(tmp_path, stage1a_checkpoint):
    ext_true = extend_state_checkpoint_with_deriv_stream(
        resume_from=stage1a_checkpoint, condition_on_theta=True, device="cpu")
    assert ext_true.stream_configs["deriv"].condition_on_theta is True
    assert any("theta_conditioners.deriv" in k for k in ext_true.ae.state_dict())

    ext_false = extend_state_checkpoint_with_deriv_stream(
        resume_from=stage1a_checkpoint, condition_on_theta=False, device="cpu")
    assert ext_false.stream_configs["deriv"].condition_on_theta is False


def test_latent_channels_override(tmp_path, stage1a_checkpoint):
    """None (default) matches state's own channel count; an explicit
    override genuinely changes the deriv bottleneck's own shape --
    identical contract to stage 1b's own latent_channels parameter."""
    ext_default = extend_state_checkpoint_with_deriv_stream(resume_from=stage1a_checkpoint, device="cpu")
    assert ext_default.stream_configs["deriv"].channels == ext_default.stream_configs[
        ext_default.state_name].channels

    ext_override = extend_state_checkpoint_with_deriv_stream(
        resume_from=stage1a_checkpoint, latent_channels=6, device="cpu")
    assert ext_override.stream_configs["deriv"].channels == 6


def test_rejects_a_multi_stream_ancestor(tmp_path, stage1a_checkpoint):
    """Identical contract to stage 1b's own equivalent check -- this
    function's whole point is extending a SINGLE-stream checkpoint;
    resuming from something that already has a deriv stream isn't a
    sensible input."""
    already_extended = extend_state_checkpoint_with_deriv_stream(
        resume_from=stage1a_checkpoint, device="cpu")
    fake_multi_stream_path = tmp_path / "fake_multi.pt"
    torch.save({
        "model_state": already_extended.ae.state_dict(),
        "config": {
            "size": already_extended.size, "base_channels": already_extended.base_channels,
            "stream_configs": {
                name: {"channels": cfg.channels, "spatial_size": cfg.spatial_size,
                       "mode": cfg.mode.value, "condition_on_theta": cfg.condition_on_theta}
                for name, cfg in already_extended.stream_configs.items()
            },
            "recon_stream_name": already_extended.state_name,
        },
        "stats_config": {"stat_names": already_extended.stat_names,
                          "stats_mean": already_extended.mean, "stats_std": already_extended.std},
    }, fake_multi_stream_path)

    with pytest.raises(ValueError, match="SINGLE-stream"):
        extend_state_checkpoint_with_deriv_stream(resume_from=fake_multi_stream_path, device="cpu")


def test_raises_clearly_when_ancestor_has_no_stats_head(tmp_path):
    """Identical contract to stage 1b's own equivalent check --
    L_stats1 needs stats_head0's own stat_names/normalization, not
    available if stage 1a never built one."""
    base_path = _build_sweep(tmp_path)
    stage1a_no_stats = train_autoencoder(
        size=32, base_path=base_path, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.3, test_fraction=0.2, num_workers=0, min_step=0, min_stdev_phi=None,
        stats0_weight=0.0, stat_names=None, checkpoint_path=tmp_path / "s1a_nostats.pt",
        device="cpu", seed=0, log_every_epoch=False, loss_curve_path=tmp_path / "c1a_nostats.png",
    )
    with pytest.raises(ValueError, match="stats_head"):
        extend_state_checkpoint_with_deriv_stream(resume_from=stage1a_no_stats, device="cpu")


def _build_sweep(tmp_path, *args, **kwargs):
    """
    Memoized wrapper around this module's own _build_sweep_uncached --
    see conftest.cached_sweep for the full rationale and the read-only
    justification. tmp_path is accepted for call-site compatibility and
    deliberately IGNORED: the sweep lives in a shared, longer-lived
    directory so repeated calls with the same arguments reuse one build
    instead of rewriting the same synthetic snapshots per test. Anything
    a test WRITES (checkpoints, figures, logs) still goes to its own
    tmp_path, which this never touches.
    """
    return cached_sweep((__name__, args, tuple(sorted(kwargs.items()))),
                        lambda d: _build_sweep_uncached(d, *args, **kwargs))
