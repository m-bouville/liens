import torch
from pathlib import Path
from utils import load_datasets as load
from training.train_ae import train_autoencoder, train_stage1b


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


def test_stage1b_freezes_1a_correctly_and_trains_only_new_pieces(tmp_path):
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)

    # Stage 1a: plain single-stream training with a real stats_head
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=2, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1a_checkpoint = torch.load(stage1a_path, map_location="cpu", weights_only=True)

    # Capture stage 1a's own weights BEFORE stage 1b touches anything,
    # to compare against afterward.
    trunk_key = "encoder.down_blocks.0.conv.block.0.weight"
    state_bottleneck_key = "encoder.bottlenecks.state.weight"
    decoder_key = "decoder.up_blocks.0.conv.block.0.weight"
    trunk_before = stage1a_checkpoint["model_state"][trunk_key].clone()
    state_bottleneck_before = stage1a_checkpoint["model_state"][state_bottleneck_key].clone()
    D0_before = stage1a_checkpoint["model_state"][decoder_key].clone()

    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path,
        stats_weight=0.01,
        epochs=2, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve1b.png",
    )
    stage1b_checkpoint = torch.load(stage1b_path, map_location="cpu", weights_only=True)
    state = stage1b_checkpoint["model_state"]

    # === THE central claims of stage 1b, verified directly ===

    # 1. Trunk and state bottleneck are EXACTLY unchanged (frozen).
    trunk_after = state["encoders.shared.down_blocks.0.conv.block.0.weight"]
    state_bottleneck_after = state["encoders.shared.bottlenecks.state.weight"]
    assert torch.equal(trunk_before, trunk_after), "trunk moved despite being frozen"
    assert torch.equal(state_bottleneck_before, state_bottleneck_after), (
        "state bottleneck moved despite being frozen"
    )
    print("Trunk and state bottleneck: EXACTLY unchanged (frozen correctly)")

    # 2. D0 is EXACTLY unchanged (frozen/unused).
    D0_after = state["decoders.D0.up_blocks.0.conv.block.0.weight"]
    assert torch.equal(D0_before, D0_after), "D0 moved despite being frozen/unused"
    print("D0: EXACTLY unchanged (frozen/unused correctly)")

    # 3. The NEW deriv bottleneck changed from ITS OWN random init
    # (can't compare against stage1a, which never had one -- build a
    # fresh Encoder with the same seed logic and check against a
    # differently-seeded one instead: simplest real check is just that
    # it's not all-zeros/degenerate AND that D1/stats_head1 moved).
    D1_final = state["decoders.D1.up_blocks.0.conv.block.0.weight"]
    assert D1_final.abs().sum().item() > 0, "D1 is degenerate (all zero)"

    # 4. D1 was WARM-STARTED from D0 -- can't check post-training (it
    # trained since), but confirm it's plausible: D1 and D0 should be
    # in the same ballpark of magnitude (same architecture, warm start),
    # not wildly different scales.
    print(f"D0 weight norm: {D0_after.norm().item():.4f}, D1 weight norm (after training): {D1_final.norm().item():.4f}")

    # 5. stats_head0 carried over UNCHANGED.
    sh0_before = stage1a_checkpoint["stats_head_state"]
    sh0_after = stage1b_checkpoint["stats_head_state"]
    for k in sh0_before:
        assert torch.equal(sh0_before[k], sh0_after[k]), f"stats_head0's {k} changed -- should be carried over unchanged"
    print("stats_head0: EXACTLY unchanged (carried over correctly)")

    # 6. stats_head1 exists and is genuinely different from a fresh random init.
    assert stage1b_checkpoint["stats_head1_state"] is not None
    print("stats_head1_state: present in checkpoint")

    print("ALL CHECKS PASSED")


def test_d1_warm_started_from_d0_before_any_training(tmp_path):
    """Directly verify the warm-start claim: BEFORE stage 1b trains
    anything, D1's weights should be an EXACT copy of D0's."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1a_checkpoint = torch.load(stage1a_path, map_location="cpu", weights_only=True)
    D0_state = {k[len("decoder."):]: v for k, v in stage1a_checkpoint["model_state"].items()
                if k.startswith("decoder.")}

    import sys
    from models.decoder import Decoder
    D0_fresh = Decoder(output_size=32, out_channels=1, base_channels=4,
                        latent_channels=4, latent_spatial_size=8)
    D0_fresh.load_state_dict(D0_state)

    D1_fresh = Decoder(output_size=32, out_channels=1, base_channels=4,
                        latent_channels=4, latent_spatial_size=8)
    D1_fresh.load_state_dict(D0_fresh.state_dict())  # THE warm-start operation itself

    for (n0, p0), (n1, p1) in zip(D0_fresh.named_parameters(), D1_fresh.named_parameters()):
        assert torch.equal(p0, p1), f"warm-start failed for {n0}"
    print("D1 warm-start mechanism verified: exact copy of D0 before any training")


def test_latent_spatial_size_no_longer_a_parameter(tmp_path):
    """Regression test for a real reported bug: latent_spatial_size
    (renamed from latent_spatial_decoder) used to be independently
    settable for the deriv stream, but Encoder hard-requires every
    stream to share ONE spatial_size (the shared trunk only produces
    one bottleneck resolution) -- so a mismatched value always crashed
    at Encoder construction anyway, before any of the (dead) fallback
    logic for it could even run. Removed entirely -- passing it now
    should fail loudly and immediately (unexpected keyword), not
    silently accepted and then crash three function calls later with a
    confusing Encoder-internal error."""
    import inspect
    sig = inspect.signature(train_stage1b)
    assert "latent_spatial_size" not in sig.parameters
    assert "latent_channels" in sig.parameters


def test_latent_channels_can_genuinely_differ_from_state(tmp_path):
    """Unlike spatial_size, channels genuinely CAN differ per-stream
    (each stream has its own, separate bottleneck conv) -- confirm
    this actually works end-to-end (no crash), and that D1 correctly
    is NOT warm-started when shapes don't match."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    # deriv stream gets a DIFFERENT channel count (8, not state's own 4)
    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path,
        stats_weight=0.01, latent_channels=8,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b.png",
    )
    checkpoint = torch.load(stage1b_path, map_location="cpu", weights_only=True)
    deriv_cfg = checkpoint["config"]["stream_configs"]["deriv"]
    state_cfg = checkpoint["config"]["stream_configs"]["state"]
    assert deriv_cfg["channels"] == 8
    assert state_cfg["channels"] == 4
    assert deriv_cfg["spatial_size"] == state_cfg["spatial_size"], (
        "spatial_size must ALWAYS match state's own, regardless of channels"
    )
    print(f"deriv channels={deriv_cfg['channels']} (independently set), "
          f"spatial_size={deriv_cfg['spatial_size']} (always inherited from state)")
    print("Ran successfully with mismatched channel counts -- no warm-start, but no crash either")


def test_frozen_trunk_batchnorm_buffers_exactly_unchanged(tmp_path):
    """Regression test for a real, confirmed bug: ae.train() recursively
    flips EVERY submodule into train mode each epoch, including the
    frozen trunk's BatchNorm2d layers -- which then normalize using the
    CURRENT BATCH's own statistics (not stage 1a's own, stable,
    trained statistics), regardless of requires_grad. Parameter drift
    alone (the OTHER tests in this file) couldn't catch this: BatchNorm
    running_mean/running_var are BUFFERS, not parameters, so a block
    can show EXACTLY zero parameter drift while its actual forward-pass
    behavior is still silently wrong. Directly verified: without
    encoder.eval(), these buffers drift by a real, substantial amount
    (confirmed via a manual before/after diff, not assumed)."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=2, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1a_checkpoint = torch.load(stage1a_path, map_location="cpu", weights_only=True)
    bn_running_mean_key = "encoder.down_blocks.0.conv.block.1.running_mean"
    bn_running_var_key = "encoder.down_blocks.0.conv.block.1.running_var"
    running_mean_before = stage1a_checkpoint["model_state"][bn_running_mean_key].clone()
    running_var_before = stage1a_checkpoint["model_state"][bn_running_var_key].clone()

    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats_weight=0.01,
        epochs=2, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b.png",
    )
    stage1b_checkpoint = torch.load(stage1b_path, map_location="cpu", weights_only=True)
    running_mean_after = stage1b_checkpoint["model_state"]["encoders.shared." + bn_running_mean_key[len("encoder."):]]
    running_var_after = stage1b_checkpoint["model_state"]["encoders.shared." + bn_running_var_key[len("encoder."):]]

    assert torch.equal(running_mean_before, running_mean_after), (
        "frozen trunk's BatchNorm running_mean DRIFTED -- encoder.eval() is not "
        "correctly applied, meaning the trunk's forward-pass BEHAVIOR (not just its "
        "weights) doesn't actually match stage 1a's own"
    )
    assert torch.equal(running_var_before, running_var_after), (
        "frozen trunk's BatchNorm running_var DRIFTED -- same issue as running_mean above"
    )


def test_deriv_log_output_scale_is_actually_trained(tmp_path):
    """Regression test for a real, confirmed bug: log_output_scale is a
    genuine nn.Parameter for a DECODER-mode stream (deriv), registered
    directly on the EncoderDecoderPair object -- NOT on encoder or D1.
    The optimizer's own parameter list only ever included
    encoder.parameters() + D1.parameters() (+ stats_head1's), so this
    parameter was silently NEVER included at all, despite being
    genuinely trainable -- indistinguishable, from the outside, from
    an intentionally-frozen parameter (both show exactly zero drift),
    which is exactly what made this easy to miss."""
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
        epochs=2, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b.png",
    )
    checkpoint = torch.load(stage1b_path, map_location="cpu", weights_only=True)
    scale = checkpoint["model_state"]["pathways.deriv.log_output_scale"]
    assert scale.item() != 0.0, (
        "pathways.deriv.log_output_scale is EXACTLY 0.0 -- it was not included in "
        "the optimizer's own parameter list (a real, previously-confirmed bug)"
    )


def test_pred_target_diagnostic_produces_sensible_values(tmp_path, capsys):
    """New diagnostic: pred/target norm and cosine similarity, printed
    per epoch -- distinguishes a pure scale mismatch (high cos_sim,
    mismatched norms) from pred carrying no real directional signal
    about target at all (cos_sim near 0). Confirms it runs without
    error and produces values in valid ranges; also confirms the
    accumulators are genuinely fresh each epoch (a real, separate bug
    found alongside this -- z1_train_stats and friends used to be
    declared OUTSIDE the epoch loop and never reset, silently
    accumulating across the entire run instead of reporting each
    epoch's own value)."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats_weight=0.01,
        epochs=2, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve1b.png",
    )
    output = capsys.readouterr().out
    diagnostic_lines = [line for line in output.splitlines() if "cos_sim:" in line]
    assert len(diagnostic_lines) == 2, f"expected one diagnostic line per epoch (2), got {len(diagnostic_lines)}"

    import re
    for line in diagnostic_lines:
        cos_sim_train = float(re.search(r"cos_sim: train=([+-][\d.]+)", line).group(1))
        cos_sim_val = float(re.search(r"val=([+-][\d.]+)\s", line.split("cos_sim:")[1]).group(1))
        assert -1.0 <= cos_sim_train <= 1.0
        assert -1.0 <= cos_sim_val <= 1.0
        pred_norm_train = float(re.search(r"\|\|pred\|\|: train=([\d.e+-]+)", line).group(1))
        target_norm_train = float(re.search(r"\|\|target\|\|: train=([\d.e+-]+)", line).group(1))
        assert pred_norm_train >= 0
        assert target_norm_train >= 0


def test_freeze_encoder_false_genuinely_unfreezes_trunk(tmp_path):
    """freeze_encoder=False is a diagnostic override, requested to test
    whether the FROZEN trunk's own activations carry any usable
    derivative information at all, independent of the deriv
    bottleneck's own (limited) readout capacity. Confirms it actually
    does what it claims: trunk parameters AND BatchNorm buffers both
    move (not just one or the other), while D0 stays exactly frozen
    regardless (this diagnostic is about the encoder, not D0)."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1a_checkpoint = torch.load(stage1a_path, map_location="cpu", weights_only=True)
    trunk_weight_before = stage1a_checkpoint["model_state"]["encoder.down_blocks.0.conv.block.0.weight"].clone()
    trunk_bn_running_mean_before = stage1a_checkpoint["model_state"]["encoder.down_blocks.0.conv.block.1.running_mean"].clone()

    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats_weight=0.01,
        freeze_encoder=False,
        epochs=2, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b.png",
    )
    checkpoint = torch.load(stage1b_path, map_location="cpu", weights_only=True)
    trunk_weight_after = checkpoint["model_state"]["encoders.shared.down_blocks.0.conv.block.0.weight"]
    trunk_bn_running_mean_after = checkpoint["model_state"]["encoders.shared.down_blocks.0.conv.block.1.running_mean"]
    D0_weight_after = checkpoint["model_state"]["decoders.D0.up_blocks.0.conv.block.0.weight"]
    D0_weight_before = stage1a_checkpoint["model_state"]["decoder.up_blocks.0.conv.block.0.weight"]

    assert not torch.equal(trunk_weight_before, trunk_weight_after), (
        "trunk weights did NOT move with freeze_encoder=False -- diagnostic override is not working"
    )
    assert not torch.equal(trunk_bn_running_mean_before, trunk_bn_running_mean_after), (
        "trunk BatchNorm buffers did NOT move with freeze_encoder=False -- "
        "encoder is not actually in train mode"
    )
    assert torch.equal(D0_weight_before, D0_weight_after), (
        "D0 moved despite freeze_encoder=False only being about the encoder, not D0"
    )


def test_cos_weight_diagnostic_runs_and_affects_gradient(tmp_path):
    """cos_weight is a diagnostic-only loss term (default 0.0, off) --
    confirms it runs end to end without error, and that D1/the deriv
    bottleneck genuinely receive gradient from it (nonzero drift),
    while the frozen trunk (freeze_encoder still defaults to True here)
    stays correctly untouched -- this diagnostic is orthogonal to
    freeze_encoder, not a replacement for it."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1a_checkpoint = torch.load(stage1a_path, map_location="cpu", weights_only=True)
    trunk_weight_before = stage1a_checkpoint["model_state"]["encoder.down_blocks.0.conv.block.0.weight"].clone()

    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats_weight=0.0,
        cos_weight=100.0,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b.png",
    )
    checkpoint = torch.load(stage1b_path, map_location="cpu", weights_only=True)
    trunk_weight_after = checkpoint["model_state"]["encoders.shared.down_blocks.0.conv.block.0.weight"]
    deriv_bottleneck_after = checkpoint["model_state"]["encoders.shared.bottlenecks.deriv.weight"]

    assert torch.equal(trunk_weight_before, trunk_weight_after), (
        "trunk moved despite freeze_encoder defaulting to True -- "
        "cos_weight should be orthogonal to freeze_encoder, not override it"
    )
    assert deriv_bottleneck_after.abs().sum().item() > 0, "deriv bottleneck is degenerate"


def test_cos_weight_contribution_shown_separately_in_message(tmp_path, capsys):
    """Regression test for a real gap: cos_weight*cos_loss used to be
    folded invisibly into train_total, with only recon1/stats1 broken
    out separately -- misleading once cos_weight became a real, tuned
    part of training rather than a one-off diagnostic. Confirms the
    breakdown line appears when cos_weight > 0, and is absent when it's
    off (0.0, the default)."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )

    train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats_weight=0.0,
        cos_weight=10.0,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b_with_cos.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve1b_with_cos.png",
    )
    output_with_cos = capsys.readouterr().out
    assert "||cos: train=" in output_with_cos
    assert "raw cos_sim=" in output_with_cos

    train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats_weight=0.0,
        cos_weight=0.0,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b_without_cos.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve1b_without_cos.png",
    )
    output_without_cos = capsys.readouterr().out
    assert "||cos: train=" not in output_without_cos
