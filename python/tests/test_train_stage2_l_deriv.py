import torch
from pathlib import Path
from utils import load_datasets as load
from training.train_stage1 import train_autoencoder
from training.train_stage2 import train_stage2
import pytest


def _build_run_dir(base_dir, name, temperature=0.8, noise=0.01, seed_val=1, size=32):
    run_dir = base_dir / name
    run_dir.mkdir()
    steps = [0, 1000, 2000, 3000, 4000]
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 4000",
        f"save_steps = {' '.join(str(s) for s in steps)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", f"temperature = {temperature}",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", f"noise = {noise}",
        f"seed = {seed_val}", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)
    for step in steps:
        value = step / 10000.0 + hash(name) % 100 / 10000.0
        arr = torch.full((size, size), value, dtype=torch.float16).numpy()
        arr.tofile(run_dir / load.snapshot_filename(step))
    # statistics.csv, needed for stage 1's stats_head AND stage 2
    import pandas as pd
    df = pd.DataFrame({"avg_phi": [v / 10000.0 for v in steps]}, index=steps)
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")
    return run_dir


def _build_sweep(tmp_path, n_runs=6, size=32):
    base_dir = tmp_path / "datasets" / f"{size}x{size}"
    base_dir.mkdir(parents=True)
    run_names = [f"T800_n010_s{i}" for i in range(n_runs)]
    for name in run_names:
        _build_run_dir(base_dir, name, size=size)
    sweep_metadata_text = "\n".join([
        f"Nx = {size}", f"Ny = {size}", "temperatures = 0.8", "noises = 0.01",
        f"seeds = {','.join(str(i) for i in range(n_runs))}",
        "subdirs =", *run_names,
    ])
    (base_dir / "metadata.txt").write_text(sweep_metadata_text)
    for name in run_names:
        (base_dir / name / "COMPLETE").touch()
    return tmp_path / "datasets"


def test_stage2_rejects_ambiguous_multi_deriv_ancestor(tmp_path, isolated_project_root):
    """L_deriv has no meaning without EXACTLY one deriv-role stream to
    compare z0's own trajectory against -- must raise clearly, not
    silently pick one. A single-stream (stage 1a) ancestor is no
    longer the right example of an invalid one (see
    test_stage2_accepts_single_stream_ancestor_and_builds_deriv below
    -- train_stage2 now builds the deriv stream itself from exactly
    that input, replacing what used to require a separate stage 1b
    pass first). The genuinely-still-invalid case is an ancestor with
    MORE than one non-recon stream -- built synthetically here
    (matching test_checkpoint_components_multi_stream.py's own
    approach), since no real training stage in this pipeline produces
    a 3-stream checkpoint to resume from."""
    from models.autoencoder import MultiStreamAutoencoder
    from models.encoder import Encoder
    from models.latent_streams import LatentStreamConfig, LatentStreamMode
    from training.stats_head import StatsHead

    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv_a": LatentStreamConfig(name="deriv_a", channels=4, spatial_size=8,
                                       mode=LatentStreamMode.PURE_LATENT),
        "deriv_b": LatentStreamConfig(name="deriv_b", channels=4, spatial_size=8,
                                       mode=LatentStreamMode.PURE_LATENT),
    }
    encoder = Encoder(input_size=32, in_channels=1, base_channels=4, stream_configs=stream_configs)
    from models.decoder import Decoder
    decoder = Decoder(output_size=32, out_channels=1, base_channels=4, latent_channels=4, latent_spatial_size=8)
    model = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"D0": decoder},
                                    stream_configs=stream_configs, decoder_for_stream={"state": "D0"})
    stats_head = StatsHead(latent_channels=4, stat_names=["avg_phi"], latent_spatial=8)

    checkpoint_path = tmp_path / "fake_ambiguous_multi_stream.pt"
    torch.save({
        "model_state": model.state_dict(),
        "stats_head_state": stats_head.state_dict(),
        "epoch": 1, "val_loss": 0.5,
        "config": {
            "size": 32, "base_channels": 4, "stats_weight": 0.01,
            "stream_configs": {n: {"channels": c.channels, "spatial_size": c.spatial_size,
                                    "mode": c.mode.value, "condition_on_theta": c.condition_on_theta}
                                for n, c in stream_configs.items()},
            "recon_stream_name": "state",
            "decoder_for_stream": {"state": "D0"},
        },
        "stats_config": {"stat_names": ["avg_phi"], "stats_mean": torch.zeros(1), "stats_std": torch.ones(1)},
    }, checkpoint_path)

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    with pytest.raises(ValueError, match="exactly one"):
        train_stage2(
            base_path=base_path, resume_from=checkpoint_path,
            deriv_weight=1.0, stats0_weight=0.0,
            epochs=1, batch_size=4, num_workers=0,
            min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / "stage2_should_fail.pt", device="cpu",
        )


def test_stage2_accepts_single_stream_ancestor_and_builds_deriv(tmp_path, isolated_project_root):
    """The new, intended behavior this test file's own removed test used
    to explicitly forbid: train_stage2() now resumes DIRECTLY from a
    stage 1a (single-stream) checkpoint, building the deriv stream
    itself in memory -- see extend_encoder.py's own module docstring
    for the full rationale (stage 1b's own training loop had been
    inert since it started running at epochs=0; D1, the one thing
    genuinely built only by that loop's own surrounding setup, is
    confirmed permanently unnecessary). Checked here at train_stage2's
    own integration level, not just extend_encoder.py's own isolated
    unit tests -- confirms the two are actually wired together
    correctly, end to end, including a real training epoch."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)

    stage1_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1_single.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1.png",
    )

    stage2_path = train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, stats0_weight=0.01,
        epochs=1, batch_size=4, num_workers=0,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_from_1a.pt", device="cpu",
        log_every_epoch=False, loss_curve_path=tmp_path / "curve2.png",
    )
    assert stage2_path.exists()

    saved = torch.load(stage2_path, map_location="cpu", weights_only=True)
    from models.latent_streams import LatentStreamMode
    assert saved["config"]["stream_configs"]["deriv"]["mode"] == LatentStreamMode.PURE_LATENT.value
    assert "deriv" not in saved["config"]["decoder_for_stream"]
    assert not any(k.startswith("decoders.D1") for k in saved["model_state"])
    assert saved["stats_head1_state"] is not None


def test_stats_head1_genuinely_trains_when_stats1_weight_nonzero(tmp_path, isolated_project_root):
    """Regression test for a real, confirmed bug: stats_head1 is a
    SEPARATE nn.Module from ae (not one of its submodules), left with
    requires_grad=True in the Stage-1a-direct branch (deliberately, so
    it gets its own first chance to learn something) -- but its own
    parameters were never actually added to the optimizer anywhere,
    so total.backward() genuinely computed gradients for it (the
    printed stats1 loss term visibly decreases epoch over epoch) while
    optimizer.step() never applied them. Confirmed directly: with the
    bug present, stats_head1's true initial state_dict (captured via a
    monkeypatch on extend_state_checkpoint_with_deriv_stream, since
    it's constructed INSIDE train_stage2 -- a fresh, separately-built
    comparison object would just be a different random draw, not the
    actual initial state) was byte-identical to its state after 3 real
    training epochs with stats1_weight=1.0, despite the loss
    decreasing throughout (which came entirely from the shared trunk
    moving via L_deriv/L_recon, not from stats_head1 itself)."""
    import training.train_stage2 as train_stage2_module
    from training.extend_encoder import extend_state_checkpoint_with_deriv_stream as real_extend

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )

    captured = {}

    def _recording_extend(*args, **kwargs):
        ext = real_extend(*args, **kwargs)
        captured["stats_head1_initial"] = {k: v.clone() for k, v in ext.stats_head1.state_dict().items()}
        return ext

    monkeypatch_target = train_stage2_module.extend_state_checkpoint_with_deriv_stream
    train_stage2_module.extend_state_checkpoint_with_deriv_stream = _recording_extend
    try:
        stage2_path = train_stage2_module.train_stage2(
            base_path=base_path, resume_from=stage1a_path, stats0_weight=0.01,
            stats1_weight=1.0, deriv_weight=1.0,
            epochs=3, batch_size=4, num_workers=0,
            val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / "stage2.pt", device="cpu", seed=0,
            log_every_epoch=False, loss_curve_path=tmp_path / "curve2.png",
        )
    finally:
        train_stage2_module.extend_state_checkpoint_with_deriv_stream = monkeypatch_target

    saved = torch.load(stage2_path, map_location="cpu", weights_only=True)
    sh1_after = saved["stats_head1_state"]
    sh1_before = captured["stats_head1_initial"]

    changed = any(not torch.equal(sh1_before[k], sh1_after[k]) for k in sh1_before)
    assert changed, (
        "stats_head1's weights are byte-identical to their true initial values after 3 real "
        "training epochs with stats1_weight=1.0 -- its own parameters aren't reaching the "
        "optimizer (check train_stage2's own `params = [...]` construction)"
    )
