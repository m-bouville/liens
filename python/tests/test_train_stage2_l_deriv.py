import torch
from pathlib import Path
from utils import load_datasets as load
from training.train_ae import train_autoencoder, train_stage2
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


@pytest.mark.skip(reason="Fixture built a multi-stream checkpoint via train_autoencoder's "
                         "now-removed latent_names/latent_modes syntax (see the C0/C1 alternation "
                         "removal). train_stage2 itself is also genuinely incompatible with stage "
                         "1b's own output (separate D0/D1 via decoder_for_stream, not one shared "
                         "decoder) -- pending stage 2's own redesign to match the new stage 1a/1b "
                         "split; not fixed here per explicit instruction not to touch stage 2 yet.")
def test_stage2_trains_deriv_via_l_deriv(tmp_path):
    """Real end-to-end: stage 1 (multi-stream, with stats_head) -> stage 2
    resuming from it. Confirms L_deriv genuinely moves the deriv stream's
    weights FURTHER (not just that stage 1 alone did), and that the
    checkpoint's own saved config correctly reflects deriv_weight, not
    the old interp_weight."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)

    stage1_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4,
        latent_names=["state", "deriv"], latent_modes=["autoencoder", "decoder"],
        latent_channels_decoder=4, latent_spatial_decoder=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve1.png",
    )
    stage1_checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=True)
    deriv_weight_after_stage1 = stage1_checkpoint["model_state"]["encoders.shared.bottlenecks.deriv.weight"].clone()

    stage2_path = train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, stats_weight=0.0,
        epochs=2, batch_size=4, num_workers=0,
        val_fraction=0.34, test_fraction=0.17,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve2.png",
    )
    stage2_checkpoint = torch.load(stage2_path, map_location="cpu", weights_only=True)
    deriv_weight_after_stage2 = stage2_checkpoint["model_state"]["encoders.shared.bottlenecks.deriv.weight"]

    assert not torch.allclose(deriv_weight_after_stage1, deriv_weight_after_stage2), (
        "deriv bottleneck weights UNCHANGED between stage 1 and stage 2 -- "
        "L_deriv is not actually reaching this stream's parameters in stage 2"
    )
    print(f"deriv bottleneck weight moved further in stage 2: max abs diff = "
          f"{(deriv_weight_after_stage2 - deriv_weight_after_stage1).abs().max().item():.6f}")

    assert "deriv_weight" in stage2_checkpoint["stage2_config"]
    assert "interp_weight" not in stage2_checkpoint["stage2_config"]
    print("stage2_config:", stage2_checkpoint["stage2_config"])


def test_stage2_rejects_single_stream_ancestor(tmp_path):
    """L_deriv has no meaning without a deriv stream -- must raise
    clearly, not silently do something else."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)

    stage1_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1_single.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1.png",
    )

    with pytest.raises(ValueError, match="exactly one"):
        train_stage2(
            base_path=base_path, resume_from=stage1_path,
            deriv_weight=1.0, stats_weight=0.0,
            epochs=1, batch_size=4, num_workers=0,
            min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / "stage2_should_fail.pt", device="cpu",
        )


@pytest.mark.skip(reason="Fixture built a multi-stream checkpoint via train_autoencoder's "
                         "now-removed latent_names/latent_modes syntax (see the C0/C1 alternation "
                         "removal). train_stage2 itself is also genuinely incompatible with stage "
                         "1b's own output (separate D0/D1 via decoder_for_stream, not one shared "
                         "decoder) -- pending stage 2's own redesign to match the new stage 1a/1b "
                         "split; not fixed here per explicit instruction not to touch stage 2 yet.")
def test_stage2_freeze_outer_layers_works_with_multi_stream(tmp_path):
    """Point 3's actual claim: freeze_outer_layers() needs no new code
    for the multi-stream EncoderDecoderPair case. Verify directly --
    frozen blocks show exactly 0 parameter drift, trainable ones don't."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)

    stage1_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4,
        latent_names=["state", "deriv"], latent_modes=["autoencoder", "decoder"],
        latent_channels_decoder=4, latent_spatial_decoder=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1.png",
    )
    stage1_checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=True)
    frozen_key = "encoders.shared.down_blocks.0.conv.block.0.weight"
    frozen_before = stage1_checkpoint["model_state"][frozen_key].clone()

    stage2_path = train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, stats_weight=0.0,
        epochs=2, batch_size=4, num_workers=0,
        val_fraction=0.34, test_fraction=0.17,
        min_step=0, min_stdev_phi=None,
        n_frozen_stages=1,
        checkpoint_path=tmp_path / "stage2_frozen.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve2.png",
    )
    stage2_checkpoint = torch.load(stage2_path, map_location="cpu", weights_only=True)
    frozen_after = stage2_checkpoint["model_state"][frozen_key]
    deriv_after = stage2_checkpoint["model_state"]["encoders.shared.bottlenecks.deriv.weight"]
    deriv_before = stage1_checkpoint["model_state"]["encoders.shared.bottlenecks.deriv.weight"]

    assert torch.equal(frozen_before, frozen_after), (
        "frozen outer layer's weights CHANGED despite n_frozen_stages=1 -- "
        "freeze_outer_layers() is not correctly freezing this multi-stream model"
    )
    assert not torch.allclose(deriv_before, deriv_after), (
        "deriv bottleneck (should stay TRAINABLE -- it's bottleneck-adjacent, "
        "not an outer layer) didn't move at all"
    )
    print("frozen outer layer: exactly unchanged (correct)")
    print(f"trainable deriv bottleneck: moved (correct), max abs diff = "
          f"{(deriv_after - deriv_before).abs().max().item():.6f}")
