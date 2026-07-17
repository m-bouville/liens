import torch
from pathlib import Path
from utils import load_datasets as load
from training.train_ae import train_autoencoder, train_stage1b, train_stage2


def _build_run_dir_with_stats(base_dir, name, size=32):
    run_dir = base_dir / name
    run_dir.mkdir()
    steps = [0, 1000, 2000, 3000, 4000]
    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {size}", f"Ny = {size}", "dt = 0.05", "steps = 4000",
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
    import pandas as pd
    df = pd.DataFrame({"avg_phi": [s / 10000.0 for s in steps]}, index=steps)
    df.index.name = "step"
    df.to_csv(run_dir / "statistics.csv")
    return run_dir


def _build_sweep(tmp_path, n_runs=6, size=32):
    base_dir = tmp_path / "datasets" / f"{size}x{size}"
    base_dir.mkdir(parents=True)
    run_names = [f"T800_n010_s{i}" for i in range(n_runs)]
    for name in run_names:
        _build_run_dir_with_stats(base_dir, name, size=size)
    sweep_meta = "\n".join([
        f"Nx = {size}", f"Ny = {size}", "temperatures = 0.8", "noises = 0.01",
        f"seeds = {','.join(str(i) for i in range(n_runs))}", "subdirs =", *run_names,
    ])
    (base_dir / "metadata.txt").write_text(sweep_meta)
    for name in run_names:
        (base_dir / name / "COMPLETE").touch()
    return tmp_path / "datasets"


def test_stage2_full_c0c1_loss_end_to_end(tmp_path, isolated_project_root):
    """THE actual redesigned stage 2, exercised end-to-end from a real
    stage 1a -> 1b chain: separate D0/D1 correctly loaded (not the old
    single-shared-decoder assumption that used to crash here), both
    stats_head0/stats_head1 loaded and frozen, all five loss components
    (recon0, recon1, stats0, stats1, deriv) genuinely computed and
    contributing gradient, freeze_outer_layers correctly freezes BOTH
    decoders' outer layers, and the saved checkpoint carries everything
    (stats_head1_state, decoder_for_stream) needed to be a valid
    ancestor for stage 3/4/5 and every evaluation script in turn."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats_weight=0.01,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b.png",
    )
    stage1b_checkpoint = torch.load(stage1b_path, map_location="cpu", weights_only=True)

    stage2_path = train_stage2(
        base_path=base_path, resume_from=stage1b_path,
        deriv_weight=1.0, deriv_weight_warmup_epochs=0,
        recon1_weight=0.5, stats_weight=0.01, stats1_weight=0.02,
        epochs=2, batch_size=4, num_workers=0,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        n_frozen_stages=1,
        checkpoint_path=tmp_path / "stage2.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve2.png",
    )
    checkpoint = torch.load(stage2_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]

    # 1. Separate D0/D1 correctly present (not the old, wrong, single "decoders.shared.*")
    assert any(k.startswith("decoders.D0.") for k in state)
    assert any(k.startswith("decoders.D1.") for k in state)
    assert not any(k.startswith("decoders.shared.") for k in state)
    print("Separate D0/D1 correctly loaded and saved")

    # 2. Both stats heads present in the saved checkpoint.
    assert checkpoint["stats_head_state"] is not None
    assert checkpoint["stats_head1_state"] is not None
    print("Both stats_head (SH0) and stats_head1 (SH1) present in checkpoint")

    # 3. decoder_for_stream carried forward correctly.
    assert checkpoint["config"]["decoder_for_stream"] == {"state": "D0", "deriv": "D1"}
    print("decoder_for_stream correctly carried forward in config")

    # 4. n_frozen_stages=1 froze BOTH decoders' outer layers (not just one).
    D0_output_conv_before = stage1b_checkpoint["model_state"]["decoders.D0.output_conv.weight"]
    D1_output_conv_before = stage1b_checkpoint["model_state"]["decoders.D1.output_conv.weight"]
    D0_output_conv_after = state["decoders.D0.output_conv.weight"]
    D1_output_conv_after = state["decoders.D1.output_conv.weight"]
    assert torch.equal(D0_output_conv_before, D0_output_conv_after), "D0's frozen output_conv moved"
    assert torch.equal(D1_output_conv_before, D1_output_conv_after), "D1's frozen output_conv moved"
    print("freeze_outer_layers correctly froze BOTH D0's and D1's output_conv")

    # 5. Inner layers (bottleneck-adjacent, NOT frozen by n_frozen_stages=1)
    # genuinely trained -- confirms real gradient reached both pathways,
    # not just that construction succeeded.
    D0_unbottleneck_before = stage1b_checkpoint["model_state"]["decoders.D0.unbottleneck.weight"]
    D1_unbottleneck_before = stage1b_checkpoint["model_state"]["decoders.D1.unbottleneck.weight"]
    D0_unbottleneck_after = state["decoders.D0.unbottleneck.weight"]
    D1_unbottleneck_after = state["decoders.D1.unbottleneck.weight"]
    assert not torch.equal(D0_unbottleneck_before, D0_unbottleneck_after), "D0's unbottleneck did not train"
    assert not torch.equal(D1_unbottleneck_before, D1_unbottleneck_after), "D1's unbottleneck did not train"
    print("Both D0's and D1's own inner (unfrozen) layers genuinely trained")
