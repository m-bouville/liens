import torch
import pytest
from pathlib import Path
from utils import load_datasets as load
from training.train_stage1 import train_autoencoder
from training.train_stage2 import train_stage2


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


@pytest.fixture(scope="module")
def shared_stage1a_ancestor(tmp_path_factory):
    """No stage 1b pass at all -- train_stage2() now builds the deriv
    stream itself, directly from stage 1's own checkpoint (see
    training/extend_encoder.py's own module docstring for the full
    rationale: stage 1b's own training loop had been inert since it
    started running at epochs=0, and D1 -- the one thing genuinely
    built only by that loop's own surrounding setup -- is confirmed
    permanently unnecessary)."""
    root = tmp_path_factory.mktemp("shared_stage1a")
    base_path = _build_sweep(root, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=root / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=root / "curve1a.png",
    )
    return base_path, stage1a_path


def test_stage2_full_c0c1_loss_end_to_end(shared_stage1a_ancestor, tmp_path, isolated_project_root):
    """The redesigned stage 2, exercised end-to-end from a real stage 1a
    ancestor directly (no stage 1b pass at all -- see
    training/extend_encoder.py's own module docstring for why: D1 is
    confirmed permanently unnecessary). recon1_weight/L_recon1 no
    longer exist as a concept at all -- there's nothing left to set or
    exercise for that term. Confirms: the deriv stream built fresh (PURE_LATENT,
    no D1), stats_head0/stats_head1 both loaded/built and frozen or
    available, the remaining four loss components (recon0, stats0,
    stats1, deriv) genuinely computed and contributing gradient,
    freeze_outer_layers correctly freezes D0's own outer layers, and the
    saved checkpoint carries everything (stats_head1_state,
    decoder_for_stream) needed to be a valid ancestor for stage 3/4/5
    and every evaluation script in turn."""
    base_path, stage1a_path = shared_stage1a_ancestor
    stage1a_checkpoint = torch.load(stage1a_path, map_location="cpu", weights_only=True)

    stage2_path = train_stage2(
        base_path=base_path, resume_from=stage1a_path,
        deriv_weight=1.0, deriv_weight_warmup_epochs=0,
        stats0_weight=0.01, stats1_weight=0.02,
        epochs=2, batch_size=4, num_workers=0,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        n_frozen_stages=1,
        checkpoint_path=tmp_path / "stage2.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve2.png",
    )
    checkpoint = torch.load(stage2_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]

    # 1. Single, named D0 decoder present -- no D1, no "decoders.shared.*"
    assert any(k.startswith("decoders.D0.") for k in state)
    assert not any(k.startswith("decoders.D1.") for k in state)
    assert not any(k.startswith("decoders.shared.") for k in state)
    print("Single, named D0 decoder correctly loaded and saved -- no D1")

    # 2. Both stats heads present in the saved checkpoint (stats_head1
    # kept available even though nothing here trains it via L_recon1
    # -- see extend_encoder.py's own ExtendedStateCheckpoint docstring).
    assert checkpoint["stats_head_state"] is not None
    assert checkpoint["stats_head1_state"] is not None
    print("Both stats_head (SH0) and stats_head1 (SH1) present in checkpoint")

    # 3. decoder_for_stream carried forward correctly -- no "deriv" entry
    # at all (PURE_LATENT streams are never looked up there -- see
    # MultiStreamAutoencoder's own pathway construction).
    assert checkpoint["config"]["decoder_for_stream"] == {"state": "D0"}
    assert checkpoint["config"]["stream_configs"]["deriv"]["mode"] == "pure_latent"
    print("decoder_for_stream correctly carried forward, no 'deriv' entry")

    # 4. n_frozen_stages=1 froze D0's own outer layers.
    D0_output_conv_before = stage1a_checkpoint["model_state"]["decoder.output_conv.weight"]
    D0_output_conv_after = state["decoders.D0.output_conv.weight"]
    assert torch.equal(D0_output_conv_before, D0_output_conv_after), "D0's frozen output_conv moved"
    print("freeze_outer_layers correctly froze D0's own output_conv")

    # 5. Inner layers (bottleneck-adjacent, NOT frozen by n_frozen_stages=1)
    # genuinely trained -- confirms real gradient reached the pathway,
    # not just that construction succeeded.
    D0_unbottleneck_before = stage1a_checkpoint["model_state"]["decoder.unbottleneck.weight"]
    D0_unbottleneck_after = state["decoders.D0.unbottleneck.weight"]
    assert not torch.equal(D0_unbottleneck_before, D0_unbottleneck_after), "D0's unbottleneck did not train"
    print("D0's own inner (unfrozen) layers genuinely trained")


def test_zero_weight_terms_omitted_from_console_output(shared_stage1a_ancestor, tmp_path, isolated_project_root, capsys):
    """Regression test for a real, reported clutter issue: with
    stats1_weight=0.0 (the new default -- pure L_deriv training), the
    header/per-epoch breakdown used to still print terms like
    "+0.0*stats1_diag" every single epoch for terms that structurally
    cannot contribute anything. Confirms zero-weight terms are omitted
    entirely, and that a nonzero one still shows correctly.
    recon1_weight/L_recon1 no longer exist as a concept at all (D1 is
    confirmed permanently unnecessary -- see extend_encoder.py's own
    module docstring), not just defaulted off, so there's nothing left
    to exercise or omit for that term specifically."""
    base_path, stage1a_path = shared_stage1a_ancestor

    capsys.readouterr()  # clear stage 1's own output first
    train_stage2(
        base_path=base_path, resume_from=stage1a_path, stats0_weight=0.01,
        stats1_weight=0.0, deriv_weight=0.02, deriv_weight_warmup_epochs=0,
        epochs=1, batch_size=4, num_workers=0,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_zero.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve2_zero.png",
    )
    output = capsys.readouterr().out
    formula_line = next(line for line in output.splitlines() if line.startswith("/"))
    per_epoch_line = next(line for line in output.splitlines() if line.startswith("   1|"))
    assert formula_line == "/  1 train = recon0/1.0 +0.01*stats0/1.0 +0.02*deriv/1.0 | valid = ...  | ema"
    assert per_epoch_line.count("+") == 4, "expected 2 '+' terms (stats0, deriv) on EACH of train/val"

    capsys.readouterr()  # clear again before the second call
    train_stage2(
        base_path=base_path, resume_from=stage1a_path, stats0_weight=0.01,
        stats1_weight=0.02, deriv_weight=1.0, deriv_weight_warmup_epochs=0,
        epochs=1, batch_size=4, num_workers=0,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_nonzero.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve2_nonzero.png",
    )
    output = capsys.readouterr().out
    assert "recon0/1.0 +0.01*stats0/1.0 +0.02*stats1/1.0 +1.0*deriv/1.0" in output


def test_epochs_zero_actually_writes_a_checkpoint_stage2(shared_stage1a_ancestor, tmp_path, isolated_project_root, capsys):
    """Same regression test as Stage 1a/1b's own, for train_stage2."""
    base_path, stage1a_path = shared_stage1a_ancestor

    checkpoint_path = tmp_path / "stage2_ablation.pt"
    assert not checkpoint_path.exists()

    result = train_stage2(
        base_path=base_path, resume_from=stage1a_path, stats0_weight=0.01,
        epochs=0, batch_size=4, num_workers=0,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=checkpoint_path, device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve2_ablation.png",
    )

    assert result == checkpoint_path
    assert checkpoint_path.exists(), "epochs=0 must still write a valid checkpoint"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 0
    output = capsys.readouterr().out
    assert "train_set: skipped" in output, "train_set must be skipped entirely at epochs=0"


def test_augment_is_actually_threaded_through_to_the_train_set_only(
    shared_stage1a_ancestor, tmp_path, isolated_project_root, monkeypatch,
):
    """Regression test for a real gap: augment was previously accepted by
    the params-file parsing layer but silently never passed to the
    dataset here at all (train_stage2 had no augment parameter). Patches
    MicrostructureEvolutionDataset itself to record what augment value
    each construction call actually received -- confirms train_set gets
    augment=True and val_set does NOT (val_loss must stay a clean
    measure of real, unaugmented performance), rather than just trusting
    that passing augment=True doesn't raise."""
    import training.train_stage2 as train_stage2_module

    base_path, stage1a_path = shared_stage1a_ancestor
    real_dataset_cls = train_stage2_module.MicrostructureEvolutionDataset
    recorded_augment_values = []

    class _RecordingDataset(real_dataset_cls):
        def __init__(self, *args, augment=False, **kwargs):
            recorded_augment_values.append(augment)
            super().__init__(*args, augment=augment, **kwargs)

    monkeypatch.setattr(train_stage2_module, "MicrostructureEvolutionDataset", _RecordingDataset)

    train_stage2(
        base_path=base_path, resume_from=stage1a_path, stats0_weight=0.01,
        epochs=1, batch_size=4, num_workers=0, augment=True,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_augment.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve2_augment.png",
    )

    # train_set constructed first, val_set second (see train_stage2's own
    # construction order) -- exactly one True (train), one False (val).
    assert recorded_augment_values == [True, False], (
        f"expected train_set augment=True, val_set augment=False (in that "
        f"construction order), got {recorded_augment_values}"
    )


def test_gradient_scaling_trick_preserves_forward_value_exactly():
    """Isolated unit test of the exact math z0_from_deriv_weight relies
    on: x.detach() + weight*(x - x.detach()) must have the SAME forward
    value as x itself, for ANY weight -- if this weren't exactly true,
    the feature would silently change what deriv_loss's own target
    actually IS, not just how much gradient flows through it."""
    torch.manual_seed(0)
    x = torch.randn(4, 3, 8, 8, requires_grad=True)
    for weight in [0.0, 0.05, 0.2, 0.5, 1.0, 2.0]:
        scaled = x.detach() + weight * (x - x.detach())
        assert torch.allclose(scaled, x, atol=1e-6), (
            f"forward value changed at weight={weight}: max diff "
            f"{(scaled - x).abs().max().item()}"
        )


def test_gradient_scaling_trick_scales_gradient_exactly():
    """The other half: gradient w.r.t. x must be exactly `weight`, not
    0 (fully blocked) or 1 (fully un-detached) regardless of what
    weight is requested."""
    torch.manual_seed(0)
    for weight in [0.0, 0.1, 0.3, 1.0]:
        x = torch.randn(4, 3, 8, 8, requires_grad=True)
        scaled = x.detach() + weight * (x - x.detach())
        scaled.sum().backward()
        expected_grad = torch.full_like(x, weight)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6), (
            f"expected gradient == {weight} everywhere, got range "
            f"[{x.grad.min().item()}, {x.grad.max().item()}]"
        )


def test_z0_from_deriv_weight_runs_end_to_end_across_its_range(
    shared_stage1a_ancestor, tmp_path, isolated_project_root,
):
    """End-to-end smoke test across z0_from_deriv_weight's own valid
    range, including the boundary values (0.0 -- today's original
    behavior; 1.0 -- fully un-detached). The precise mechanism itself
    (forward value exactly unchanged, gradient exactly scaled by
    weight) is already verified in isolation by the two tests above --
    this only confirms the new code path (the removed no_grad(), the
    new branch computing z0_next WITH gradient tracking) doesn't break
    anything when actually exercised through the real training loop.
    Does NOT attempt to prove encoder weights stay identical across
    weight values -- the encoder's own bottleneck is a SINGLE, shared
    module across streams (confirmed directly: checkpoint state_dict
    keys are "encoders.shared.bottlenecks", not split per-stream), so
    z1's own gradient path through those SAME weights legitimately
    differs with deriv_weight regardless of z0_from_deriv_weight --
    that isn't a bug, just a fact about this architecture that made an
    earlier, weight-identity version of this test provably wrong."""
    base_path, stage1a_path = shared_stage1a_ancestor

    for weight in [0.0, 0.1, 1.0]:
        ckpt_path = tmp_path / f"stage2_zfd_range_{weight}.pt"
        train_stage2(
            base_path=base_path, resume_from=stage1a_path, stats0_weight=0.01,
            deriv_weight=1.0, deriv_weight_warmup_epochs=0, z0_from_deriv_weight=weight,
            epochs=1, batch_size=4, num_workers=0,
            val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
            checkpoint_path=ckpt_path, device="cpu", seed=0,
            log_every_epoch=False, loss_curve_path=tmp_path / f"curve_zfd_range_{weight}.png",
        )
        assert ckpt_path.exists(), f"training failed to complete at z0_from_deriv_weight={weight}"
