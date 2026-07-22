import numpy as np
import torch
from pathlib import Path
from utils import load_datasets as load
from training.train_ae import train_autoencoder, train_stage1b
from training.train_lds import train_lds


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


def test_train_lds_loads_stage1b_checkpoint_correctly(tmp_path):
    """Regression test for a real bug: train_lds (stage 3) had the exact
    same construction gap check_reconstruction.py did -- assumed a
    single shared decoder, couldn't load stage 1b's separate-D0/D1
    checkpoint at all. Only the encoder is ever actually USED here
    (decoder weights are inert), but load_state_dict still needs the
    right key structure to succeed in the first place."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats1_weight=0.01,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b.png",
    )

    # THE actual test: this must not raise the old "decoders.shared.*
    # missing" RuntimeError.
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage1b_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        checkpoint_path=tmp_path / "stage3.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3.png",
    )
    assert lds_path.exists()
    print("train_lds successfully loaded a stage 1b checkpoint and completed training")


def test_l_1step_display_uses_the_same_rollout_scale_as_the_main_loss(tmp_path, capsys):
    """Regression test for a real, reported bug: l_1step (the "(1step)"
    figure shown alongside train_loss/val_loss every epoch, and the
    loss_curve.png secondary line) was returned RAW, never divided by
    rollout_scale -- while train_loss/val_loss themselves (=total) DO
    get that division. Whenever rollout_scale != 1, this made the two
    numbers shown side by side genuinely incomparable (differing by
    exactly 1/rollout_scale), not just hard to read -- reported
    symptom was l_1step appearing "orders of magnitude smaller" than
    the main loss. Confirms they're now on the same scale (a small,
    O(1) ratio, not 1/rollout_scale) for a real, multi-step run."""
    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats1_weight=0.01,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b.png",
    )

    train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage1b_path,
        ae_stats_weight=0.01,
        rollout_scale=0.0001,  # deliberately small, matching this session's own '_scale' convention
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=2, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        checkpoint_path=tmp_path / "stage3b.pt", device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve3b.png",
    )
    output = capsys.readouterr().out
    epoch_line = next(line for line in output.splitlines() if line.strip().startswith("1 "))

    # Parse "   1   <train_loss> (<train_1step>),  <val_loss> (<val_1step>) |..."
    import re
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", epoch_line)
    epoch, train_loss, train_1step, val_loss, val_1step = (float(n) for n in numbers[:5])

    assert train_1step != 0.0, "should not have collapsed to a misleadingly-truncated 0.000"
    ratio = train_loss / train_1step
    assert 0.01 < ratio < 100, (
        f"train_loss/train_1step ratio ({ratio}) should be O(1) for genuinely comparable "
        f"quantities -- a ratio anywhere near 1/rollout_scale ({1/0.0001}) would mean the "
        f"old bug (l_1step never divided by rollout_scale) is back"
    )


def test_epochs_zero_actually_writes_a_checkpoint_stage3(tmp_path, capsys):
    """Same regression test as Stage 1a/1b/2's own, for train_lds."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats1_weight=0.01,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b.png",
    )

    checkpoint_path = tmp_path / "stage3_ablation.pt"
    assert not checkpoint_path.exists()

    result = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage1b_path, ae_stats_weight=0.01,
        epochs=0, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        checkpoint_path=checkpoint_path, device="cpu", seed=0,
        log_every_epoch=True, loss_curve_path=tmp_path / "curve3_ablation.png",
    )

    assert result == checkpoint_path
    assert checkpoint_path.exists(), "epochs=0 must still write a valid checkpoint"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 0
    output = capsys.readouterr().out
    # Especially worth confirming here specifically: this dataset's own
    # construction runs the frozen AE's forward pass on every snapshot
    # (encoder=encoder, not None) -- likely the most expensive dataset
    # build in the whole pipeline, so skipping it at epochs=0 matters
    # more here than anywhere else.
    assert "train_set: skipped" in output, "train_set must be skipped entirely at epochs=0"


def test_use_dt_decade_weights_false_never_computes_or_calls_the_weights_fn(tmp_path, monkeypatch):
    """Default (False) must leave today's behavior completely
    untouched: compute_dt_decade_weights should never even be CALLED,
    not just "called but ignored" -- confirmed by patching it to raise
    if invoked at all."""
    import training.train_lds as train_lds_module

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("compute_dt_decade_weights must not be called when "
                              "use_dt_decade_weights=False (the default)")

    monkeypatch.setattr(train_lds_module, "compute_dt_decade_weights", _raise_if_called)

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats1_weight=0.01,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b_off.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b_off.png",
    )
    # THE actual test: must complete without the patched function ever firing.
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage1b_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        use_dt_decade_weights=False,
        checkpoint_path=tmp_path / "stage3_off.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3_off.png",
    )
    assert lds_path.exists()


def test_use_dt_decade_weights_true_is_computed_and_actually_used(tmp_path, monkeypatch):
    """Confirms the full, genuine integration: compute_dt_decade_weights
    is called with BOTH the measured dt array AND the measured raw-loss
    array (see compute_euler_only_losses -- the corrected scheme needs
    both, not dt alone; see losses.py's own "BUG THIS FIXES" docstring
    for why dt-alone was the original bug), AND the resulting
    weights_fn is actually invoked (with the batch's own dt_window)
    during a real training step -- not just that the flag exists and
    training doesn't crash with it set."""
    import training.train_lds as train_lds_module
    from training.losses import compute_dt_decade_weights as real_compute_dt_decade_weights

    calls = {"compute_called_with_dts": None, "compute_called_with_losses": None, "weights_fn_called": 0}

    def _recording_compute(all_dts, all_losses):
        calls["compute_called_with_dts"] = all_dts
        calls["compute_called_with_losses"] = all_losses
        real_weights_fn = real_compute_dt_decade_weights(all_dts, all_losses)

        def _recording_weights_fn(dt):
            calls["weights_fn_called"] += 1
            return real_weights_fn(dt)

        _recording_weights_fn.decade_weight = real_weights_fn.decade_weight
        return _recording_weights_fn

    monkeypatch.setattr(train_lds_module, "compute_dt_decade_weights", _recording_compute)

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a.png",
    )
    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats1_weight=0.01,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b_on.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b_on.png",
    )
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage1b_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        use_dt_decade_weights=True,
        checkpoint_path=tmp_path / "stage3_on.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3_on.png",
    )
    assert lds_path.exists()
    assert calls["compute_called_with_dts"] is not None, "compute_dt_decade_weights must be called"
    assert len(calls["compute_called_with_dts"]) > 0, "must be given a genuinely non-empty dt array"
    assert calls["compute_called_with_losses"] is not None, (
        "compute_dt_decade_weights must be given a measured raw-loss array, not dt alone -- "
        "dt-alone weighting is the exact bug this scheme fixes (see losses.py)"
    )
    assert len(calls["compute_called_with_losses"]) == len(calls["compute_called_with_dts"]), (
        "one measured loss value is required per dt"
    )
    assert calls["weights_fn_called"] > 0, (
        "the weights function itself must actually be invoked during training, not just built"
    )


def test_compute_euler_only_losses_matches_independent_manual_computation(tmp_path, monkeypatch):
    """Directly verifies compute_euler_only_losses' own arithmetic
    against an INDEPENDENT computation over a real dataset -- not by
    calling compute_euler_only_losses again, but by iterating the same
    train_set with a completely separate DataLoader and calling
    f_theta.rollout ourselves, then comparing element-for-element.

    Captured from inside train_lds itself (via a recording wrapper
    around compute_euler_only_losses), specifically BEFORE any training
    step or resume_from load has touched f_theta -- exactly the
    freshly-initialized state compute_euler_only_losses' own docstring
    requires, which a call made after train_lds() returns could not
    reproduce (f_theta's weights would already be trained by then)."""
    import training.train_lds as train_lds_module
    from training.train_lds import compute_euler_only_losses as real_compute_euler_only_losses

    captured = {}

    def _recording(f_theta, train_set, device, **kwargs):
        all_dts, all_losses = real_compute_euler_only_losses(f_theta, train_set, device, **kwargs)

        # Independent manual computation: separate DataLoader, direct
        # f_theta.rollout() call -- not a second call to the function
        # under test -- over the exact same (still freshly-initialized)
        # f_theta and train_set.
        was_training = f_theta.training
        f_theta.eval()
        manual_dts_parts, manual_losses_parts = [], []
        loader = torch.utils.data.DataLoader(train_set, batch_size=3, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                window0, window1, dt_window, theta = batch
                window0 = window0.to(device)
                window1 = window1.to(device)
                dt_window = dt_window.to(device)
                theta = theta.to(device)
                z0 = window0[:, 0]
                z0_true = window0[:, 1:]
                z0_hat_full = f_theta.rollout(z0, window1, dt_window, theta)
                z0_hat = z0_hat_full[:, 1:]
                diff = z0_hat - z0_true
                per_window_step = diff.pow(2).mean(dim=(2, 3, 4))
                manual_dts_parts.append(dt_window.cpu().numpy().reshape(-1))
                manual_losses_parts.append(per_window_step.cpu().numpy().reshape(-1))
        if was_training:
            f_theta.train()

        captured["all_dts"] = all_dts
        captured["all_losses"] = all_losses
        captured["manual_dts"] = np.concatenate(manual_dts_parts)
        captured["manual_losses"] = np.concatenate(manual_losses_parts)
        return all_dts, all_losses

    monkeypatch.setattr(train_lds_module, "compute_euler_only_losses", _recording)

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a_euler.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a_euler.png",
    )
    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats1_weight=0.01,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b_euler.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b_euler.png",
    )
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage1b_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        use_dt_decade_weights=True,
        checkpoint_path=tmp_path / "stage3_euler.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3_euler.png",
    )
    assert lds_path.exists()
    assert "all_dts" in captured, "the recording wrapper must have actually fired"
    assert len(captured["all_dts"]) > 0

    assert len(captured["all_dts"]) == len(captured["manual_dts"])
    np.testing.assert_allclose(captured["all_dts"], captured["manual_dts"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(captured["all_losses"], captured["manual_losses"], rtol=1e-5, atol=1e-8)


def test_z0_noise_scale_perturbs_train_only_not_val(tmp_path, monkeypatch):
    """z0_noise_scale is meant to fix a specific, diagnosed failure mode
    (see train_lds()'s own docstring): re-running train_lds with the
    SAME seed, differing only in z0_noise_scale (0.0 vs 0.5), must
    produce a GENUINELY different z0 for every TRAIN call (perturbation
    applied) but the EXACT SAME z0 for every VAL call (val_loss must
    stay computed against clean, unperturbed inputs, same rationale as
    use_dt_decade_weights being train-only).

    Relies on a real property of this setup, not a coincidence: with an
    identical seed and epochs=1, the train DataLoader's shuffle order is
    drawn ONCE, at the start of epoch 1's iteration, from a torch RNG
    stream whose state up to that point is identical between the two
    runs (same seed -> same f_theta init -> same everything, since
    z0_noise_scale only ever consumes extra random draws INSIDE the
    per-batch loop, strictly after that epoch's shuffle order was
    already fixed) -- so both runs see the exact same batches, in the
    exact same order, and any difference in z0 is attributable ONLY to
    the perturbation itself, not to a different draw of data.
    """
    from models.latent_dynamics import LatentDynamics

    calls = []
    original_rollout = LatentDynamics.rollout

    def _recording_rollout(self, z0, window1, dt_window, theta):
        calls.append((z0.detach().clone(), self.training))
        return original_rollout(self, z0, window1, dt_window, theta)

    monkeypatch.setattr(LatentDynamics, "rollout", _recording_rollout)

    base_path = _build_sweep(tmp_path, n_runs=6, size=32)
    stage1a_path = train_autoencoder(
        size=32, base_path=base_path,
        epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1a_noise.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1a_noise.png",
    )
    stage1b_path = train_stage1b(
        base_path=base_path, resume_from=stage1a_path, stats1_weight=0.01,
        epochs=1, batch_size=4, num_workers=0, augment=False,
        val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage1b_noise.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve1b_noise.png",
    )

    def _run(z0_noise_scale, suffix):
        calls.clear()
        train_lds(
            size=32, base_path=base_path, ae_checkpoint_path=stage1b_path, ae_stats_weight=0.01,
            epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
            val_fraction=0.34, test_fraction=0.17, num_workers=0,
            n_rollout_steps=1, min_step=0, min_stdev_phi=None,
            encode_batch_size=4, ema_warmup_epochs=0,
            z0_noise_scale=z0_noise_scale,
            checkpoint_path=tmp_path / f"stage3_noise_{suffix}.pt", device="cpu", seed=0,
            log_every_epoch=False, loss_curve_path=tmp_path / f"curve3_noise_{suffix}.png",
        )
        return list(calls)

    baseline_calls = _run(0.0, "baseline")  # default/off -- must match pre-existing behavior
    noisy_calls = _run(0.5, "noisy")

    assert len(baseline_calls) > 0 and len(baseline_calls) == len(noisy_calls), (
        "both runs must produce the exact same number/order of rollout() calls -- if "
        "this fails, batch order itself differs between runs and the comparison below "
        "isn't meaningful (see this test's own docstring for why it shouldn't)"
    )

    saw_train, saw_val = False, False
    for (z0_base, training_base), (z0_noisy, training_noisy) in zip(baseline_calls, noisy_calls):
        assert training_base == training_noisy, "train/eval mode must match call-for-call"
        if training_base:
            saw_train = True
            assert not torch.allclose(z0_base, z0_noisy), (
                "z0_noise_scale>0 must genuinely perturb z0 during training"
            )
        else:
            saw_val = True
            assert torch.equal(z0_base, z0_noisy), (
                "z0 passed during VALIDATION must be byte-identical regardless of "
                "z0_noise_scale -- perturbation must never leak into val_loss"
            )
    assert saw_train and saw_val, "test must actually exercise both train and val calls"
