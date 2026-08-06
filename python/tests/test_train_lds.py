import numpy as np
import torch
from conftest import cached_artifact, copy_cached_files
import pytest
from pathlib import Path
from utils import load_datasets as load
from training.train_stage1 import train_autoencoder
from training.train_stage2 import train_stage2
from training.checkpoint_criterion import grace_epochs_for_ema
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


def _cached_stage2_ancestor(tmp_path, **stage2_overrides):
    """A (base_path, stage2_checkpoint) pair, built ONCE per distinct config.

    Every test in this file needs a stage-2 checkpoint before it can exercise
    train_lds at all, and building one costs a sweep plus a stage-1 run plus a
    stage-2 run -- roughly three quarters of each test's runtime, for an
    artifact that is identical across all 11 of them (one stage-1 config, two
    stage-2 configs).

    The returned checkpoint is a fresh COPY in the caller's own tmp_path, so a
    test may resume from it, overwrite it, or delete it without affecting any
    other. base_path is the SHARED sweep directory, which is read-only by
    construction (see cached_sweep).

    The cache key is the full stage-2 kwargs, so any test needing a different
    ancestor transparently gets its own -- see cached_artifact on why the key
    is mechanical rather than a hand-written label.
    """
    def _build(cache_dir):
        base_path = _build_sweep(cache_dir, n_runs=6, size=32)
        stage1a_path = train_autoencoder(
            size=32, base_path=base_path,
            epochs=1, batch_size=4, base_channels=4, latent_channels=4,
            val_fraction=0.34, test_fraction=0.17, num_workers=0,
            min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
            checkpoint_path=cache_dir / "stage1a.pt", device="cpu", seed=0,
            log_every_epoch=False, loss_curve_path=cache_dir / "curve1a.png",
        )
        stage2_path = train_stage2(
            base_path=base_path, resume_from=stage1a_path,
            epochs=1, batch_size=4, num_workers=0, augment=False,
            val_fraction=0.34, test_fraction=0.17, min_step=0, min_stdev_phi=None,
            checkpoint_path=cache_dir / "stage2.pt", device="cpu", seed=0,
            log_every_epoch=False, loss_curve_path=cache_dir / "curve2.png",
            **stage2_overrides,
        )
        return base_path, stage2_path

    key = ("stage2_ancestor", tuple(sorted(stage2_overrides.items())))
    base_path, cached_stage2 = cached_artifact(key, _build)
    return base_path, copy_cached_files(cached_stage2, tmp_path)



def test_train_lds_loads_a_named_decoder_checkpoint_correctly(tmp_path, isolated_project_root):
    """Regression test for a real bug: train_lds (stage 3) had the exact
    same construction gap check_reconstruction.py did -- assumed a
    single shared decoder, couldn't load a checkpoint with a separate,
    NAMED decoder key ("decoders.D0.", not "decoders.shared.") at all.
    Only the encoder is ever actually USED here (decoder weights are
    inert), but load_state_dict still needs the right key structure to
    succeed in the first place.

    Stage 2 resumes directly from stage 1a here -- no stage 1b pass at
    all (see training/extend_encoder.py's own module docstring for the
    full rationale). Still the right regression case: the original bug
    was about a NAMED (non-"shared") decoder key existing at all, which
    "decoders.D0." alone still is, not specifically about there being
    two of them (D0 and D1 together)."""
    base_path, stage2_path = _cached_stage2_ancestor(tmp_path, stats0_weight=0.01)
    saved = torch.load(stage2_path, map_location="cpu", weights_only=True)
    assert saved["config"]["decoder_for_stream"] == {"state": "D0"}, (
        "test's own premise broke -- expected a single, NAMED (non-'shared') decoder key"
    )

    # THE actual test: this must not raise the old "decoders.shared.*
    # missing" RuntimeError.
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        checkpoint_path=tmp_path / "stage3.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3.png",
    )
    assert lds_path.exists()
    print("train_lds successfully loaded a named-decoder checkpoint and completed training")


def test_l_1step_display_uses_the_same_rollout_scale_as_the_main_loss(tmp_path, capsys, isolated_project_root):
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
    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)

    train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path,
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


def test_epochs_zero_actually_writes_a_checkpoint_stage3(tmp_path, capsys, isolated_project_root):
    """Same regression test as Stage 1a/1b/2's own, for train_lds."""
    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)

    checkpoint_path = tmp_path / "stage3_ablation.pt"
    assert not checkpoint_path.exists()

    result = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
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


def test_use_dt_decade_weights_false_never_computes_or_calls_the_weights_fn(tmp_path, monkeypatch, isolated_project_root):
    """Default (False) must leave today's behavior completely
    untouched: compute_dt_decade_weights should never even be CALLED,
    not just "called but ignored" -- confirmed by patching it to raise
    if invoked at all."""
    import training.train_lds as train_lds_module

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("compute_dt_decade_weights must not be called when "
                              "use_dt_decade_weights=False (the default)")

    monkeypatch.setattr(train_lds_module, "compute_dt_decade_weights", _raise_if_called)

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    # THE actual test: must complete without the patched function ever firing.
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        use_dt_decade_weights=False,
        checkpoint_path=tmp_path / "stage3_off.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3_off.png",
    )
    assert lds_path.exists()


def test_use_dt_decade_weights_true_is_computed_and_actually_used(tmp_path, monkeypatch, isolated_project_root):
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

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
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


def test_compute_euler_only_losses_matches_independent_manual_computation(tmp_path, monkeypatch, isolated_project_root):
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

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
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


def test_z0_noise_scale_perturbs_train_only_not_val(tmp_path, monkeypatch, isolated_project_root):
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

    def _recording_rollout(self, z0, window1, dt_window, theta, **kwargs):
        # **kwargs, not an explicit list: this wrapper only cares about z0 and
        # the training flag, so it should not have to be edited every time
        # rollout() gains an option (it broke once when z1_resync was added).
        calls.append((z0.detach().clone(), self.training))
        return original_rollout(self, z0, window1, dt_window, theta, **kwargs)

    monkeypatch.setattr(LatentDynamics, "rollout", _recording_rollout)

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)

    def _run(z0_noise_scale, suffix):
        calls.clear()
        train_lds(
            size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
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


# ---- exponent_deriv / dt_cap ---------------------------------------------
#
# exponent_deriv's own mechanism (RolloutLoss's q-weighted reduction of
# dt-dependence) already has thorough, dedicated unit coverage in
# test_losses.py -- unchanged by this session's work. What's new and
# untested here is specifically train_lds()'s OWN usage of it: this was
# a real, previously-reported regression (a comment here used to
# explicitly argue AGAINST using exponent_deriv at all, on reasoning
# that turned out to be incomplete) -- confirms train_lds() actually
# constructs RolloutLoss with exponent_deriv=0.0 (not left at the
# class's own default of 1.0) and actually passes dt at the call site
# (without which exponent_deriv would silently have no effect, per
# RolloutLoss's own requirement that dt be given whenever
# exponent_deriv != 1.0).
#
# dt_cap tests below similarly focus on train_lds()'s OWN plumbing
# (construction, checkpoint round-trip, resume-time consistency
# checking) -- LatentDynamics' own dt_cap behavior (does the forward()
# math work correctly) is already thoroughly covered in
# test_latent_dynamics.py and not re-tested here.

def test_rollout_loss_constructed_with_exponent_deriv_zero_and_dt_passed_at_call(tmp_path, monkeypatch, isolated_project_root):
    """The two-part fix, both parts required together: exponent_deriv=0.0
    at RolloutLoss's own construction does NOTHING on its own unless dt
    is also actually passed at the call site (RolloutLoss raises if
    exponent_deriv != 1.0 and dt is None) -- confirms BOTH halves are
    wired correctly, not just one."""
    import training.train_lds as train_lds_module
    from training.losses import RolloutLoss as RealRolloutLoss

    calls = {"init_exponent_deriv": "NOT_CALLED", "call_count": 0, "any_call_missing_dt": False}

    class _RecordingRolloutLoss(RealRolloutLoss):
        def __init__(self, *args, **kwargs):
            calls["init_exponent_deriv"] = kwargs.get("exponent_deriv")
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            calls["call_count"] += 1
            if kwargs.get("dt") is None:
                calls["any_call_missing_dt"] = True
            return super().__call__(*args, **kwargs)

    monkeypatch.setattr(train_lds_module, "RolloutLoss", _RecordingRolloutLoss)

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        checkpoint_path=tmp_path / "stage3.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3.png",
    )
    assert lds_path.exists()
    assert calls["init_exponent_deriv"] == 0.0, (
        f"RolloutLoss constructed with exponent_deriv={calls['init_exponent_deriv']!r}, expected 0.0"
    )
    assert calls["call_count"] > 0, "RolloutLoss must actually be called during training"
    assert not calls["any_call_missing_dt"], (
        "at least one RolloutLoss call was missing dt -- exponent_deriv=0.0 would silently "
        "have no effect on that call, since RolloutLoss's own dt-weighting requires it"
    )


def test_dt_cap_default_round_trips_as_inf(tmp_path, isolated_project_root):
    """Not specifying dt_cap at all must produce a saved checkpoint
    whose own config still explicitly records dt_cap=inf (not a
    missing key) -- so anything reading the checkpoint later gets an
    unambiguous answer via direct key lookup, not needing to know a
    separate, external default to interpret a missing key correctly."""
    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        # dt_cap deliberately NOT passed -- confirms the default itself
        checkpoint_path=tmp_path / "stage3.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3.png",
    )
    saved = torch.load(lds_path, map_location="cpu", weights_only=False)
    assert saved["config"]["dt_cap"] == float("inf")


def test_dt_cap_finite_value_round_trips_and_is_actually_applied(tmp_path, isolated_project_root):
    """The full, genuine chain: a finite dt_cap given to train_lds() must
    (1) be recorded in the saved checkpoint's own config, and (2) the
    RELOADED model (a fresh LatentDynamics, constructed from that saved
    config exactly as model_assembly.py/check_parameter_dependence.py's
    own loading paths do) must actually exhibit the capped forward()
    behavior -- not just that the number made it into a dict somewhere,
    but that it genuinely changes what the reloaded model computes."""
    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    dt_cap_value = 50.0
    lds_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0, dt_cap=dt_cap_value,
        checkpoint_path=tmp_path / "stage3.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3.png",
    )
    saved = torch.load(lds_path, map_location="cpu", weights_only=False)
    assert saved["config"]["dt_cap"] == dt_cap_value

    from models.latent_dynamics import LatentDynamics
    f_theta = LatentDynamics(latent_channels=saved["config"]["latent_channels"],
                              n_theta=saved["config"]["n_theta"],
                              latent_spatial=saved["config"]["latent_spatial_size"],
                              hidden_dim=saved["config"]["hidden_dim"],
                              n_hidden_layers=saved["config"]["n_hidden_layers"],
                              dt_cap=saved["config"]["dt_cap"])
    f_theta.load_state_dict(saved["model_state"])
    with torch.no_grad():
        f_theta.net[-1].bias.fill_(0.1)  # force a nonzero f, so capping has something to act on
    torch.manual_seed(0)
    z0 = torch.randn(2, saved["config"]["latent_channels"], saved["config"]["latent_spatial_size"],
                      saved["config"]["latent_spatial_size"])
    z1 = torch.randn_like(z0)
    theta = torch.randn(2, saved["config"]["n_theta"])
    f_val = f_theta.f(z0, z1, theta)

    dt_below = torch.full((2,), dt_cap_value - 10)
    dt_above = torch.full((2,), dt_cap_value + 500)
    out_below = f_theta(z0, z1, dt_below, theta)
    out_above = f_theta(z0, z1, dt_above, theta)

    dt_below_r = dt_below.view(-1, 1, 1, 1)
    dt_above_r = dt_above.view(-1, 1, 1, 1)
    expected_below = z0 + z1 * dt_below_r + f_val * (dt_below_r ** 2 / 2)  # below cap: uncapped formula
    expected_above_capped = z0 + z1 * dt_above_r + f_val * (dt_cap_value ** 2 / 2)  # above cap: frozen
    expected_above_if_uncapped = z0 + z1 * dt_above_r + f_val * (dt_above_r ** 2 / 2)  # what it'd be WITHOUT capping

    assert torch.allclose(out_below, expected_below, atol=1e-4)
    assert torch.allclose(out_above, expected_above_capped, atol=1e-4)
    assert not torch.allclose(out_above, expected_above_if_uncapped, atol=1e-2), (
        "reloaded model's dt_cap has no actual effect on forward() -- round-tripped correctly "
        "in the config dict, but not genuinely applied by the reconstructed model"
    )


def test_dt_cap_mismatch_on_resume_raises_a_clear_error(tmp_path, isolated_project_root):
    """dt_cap isn't a weight-shape mismatch load_state_dict would ever
    catch on its own (it's a plain float attribute, not a learnable
    parameter) -- resuming under a DIFFERENT dt_cap than the loaded
    weights were actually trained under is exactly the kind of silent,
    dangerous inconsistency the existing architecture-mismatch check
    exists to catch. Confirms dt_cap was actually added to that check,
    not just to construction/saving."""
    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    stage3a_path = train_lds(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0, dt_cap=100.0,
        checkpoint_path=tmp_path / "stage3a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3a.png",
    )

    with pytest.raises(ValueError, match="dt_cap"):
        train_lds(
            size=32, base_path=base_path, resume_from=stage3a_path,
            ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
            epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
            val_fraction=0.34, test_fraction=0.17, num_workers=0,
            n_rollout_steps=2, min_step=0, min_stdev_phi=None,
            encode_batch_size=4, ema_warmup_epochs=0,
            dt_cap=200.0,  # DIFFERENT from stage3a's own 100.0
            checkpoint_path=tmp_path / "stage3b.pt", device="cpu", seed=0,
            log_every_epoch=False, loss_curve_path=tmp_path / "curve3b.png",
        )


def _train_3a(tmp_path, base_path, stage2_path, **overrides):
    """A minimal stage-3a checkpoint to resume from."""
    kwargs = dict(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
        epochs=1, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0,
        checkpoint_path=tmp_path / "stage3a.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3a.png",
    )
    kwargs.update(overrides)
    return train_lds(**kwargs)


def _train_3b(tmp_path, base_path, stage2_path, stage3a_path, capsys=None, **overrides):
    kwargs = dict(
        size=32, base_path=base_path, resume_from=stage3a_path,
        ae_checkpoint_path=stage2_path, ae_stats_weight=0.01,
        epochs=4, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=2, min_step=0, min_stdev_phi=None,
        encode_batch_size=4, ema_warmup_epochs=0, val_ema_decay=0.7,
        checkpoint_path=tmp_path / "stage3b.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve3b.png",
    )
    kwargs.update(overrides)
    return train_lds(**kwargs)


def test_a_non_comparable_resume_gets_a_grace_period(tmp_path, isolated_project_root, capsys):
    """
    3a -> 3b changes the objective (n_rollout_steps 1 -> 2), so the ancestor's
    val_loss is not a bar this run should clear -- and, less obviously, the
    EMA now seeds on a val_loss measured under the NEW objective and relaxes
    toward its true level, minting a "best" almost every epoch on the way
    down. Observed in a real 3b log: the EMA entered at 211.7 against ~85 and
    saved on ten of its first eleven epochs while both val components were
    flat or worsening.

    BEHAVIORAL: no save may happen during the grace window, whatever the
    numbers do. Asserted through the checkpoint's own recorded epoch rather
    than through log text.
    """
    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    stage3a_path = _train_3a(tmp_path, base_path, stage2_path)
    saved_epochs = []
    _train_3b(tmp_path, base_path, stage2_path, stage3a_path,
              on_checkpoint_saved=lambda path, epoch: saved_epochs.append(epoch))

    grace = grace_epochs_for_ema(0.7)
    assert grace >= 2
    early = [e for e in saved_epochs if 1 <= e <= grace]
    assert not early, (
        f"saved at epoch(s) {early} inside the {grace}-epoch grace window -- the EMA is "
        f"still relaxing from its seed under the new objective, so those 'improvements' "
        f"are the tracker settling, not the model learning"
    )


def test_a_COMPARABLE_resume_gets_no_grace_period(tmp_path, isolated_project_root, capsys):
    """
    The counterpart, and the reason this is gated on the ancestor's
    COMPARABILITY rather than on resume_from alone: when n_rollout_steps and
    n_substeps match, the ancestor's val_loss measures the same quantity, the
    reference ceiling applies, and a grace period would only delay a save the
    run is entitled to make.

    The two branches are mutually exclusive, which is what makes this
    checkable without depending on whether the run happens to improve: a
    comparable resume announces its ceiling and no grace, a non-comparable one
    the reverse. An earlier version of this test asserted only that the
    ceiling still held -- true under BOTH branches, so widening the gate to
    `if resume_from is not None` kept it green. Verified.
    """
    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    stage3a_path = _train_3a(tmp_path, base_path, stage2_path)
    capsys.readouterr()

    _train_3b(tmp_path, base_path, stage2_path, stage3a_path,
              n_rollout_steps=1,  # SAME as the ancestor -> comparable
              checkpoint_path=tmp_path / "stage3b-same.pt")
    out = capsys.readouterr().out
    assert "reference ceiling" in out, (
        "a comparable resume must apply the ancestor's val_loss as a ceiling"
    )
    assert "grace period" not in out, (
        "a comparable resume was given a grace period -- it measures the same "
        "quantity as its ancestor, so there is nothing for the EMA to re-seed on, "
        "and the delay only blocks saves the run has earned"
    )


def test_the_grace_and_the_ceiling_are_the_same_verdict(tmp_path, isolated_project_root, capsys):
    """
    Both branches read the SAME comparability judgement, made once inside
    _resume_f_theta_from_checkpoint. Pinning that they never both fire (or
    both stay silent) on a resume keeps them from drifting into two
    independent notions of "is this ancestor comparable".
    """
    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    stage3a_path = _train_3a(tmp_path, base_path, stage2_path)
    capsys.readouterr()

    _train_3b(tmp_path, base_path, stage2_path, stage3a_path,
              checkpoint_path=tmp_path / "stage3b-diff.pt")
    out = capsys.readouterr().out
    assert ("grace period" in out) != ("reference ceiling" in out), (
        "exactly one of the ceiling and the grace period must apply on a resume -- "
        "they are two consequences of one comparability verdict"
    )
    assert "grace period" in out, "a non-comparable resume (1 -> 2 steps) must get the grace"


def test_the_grace_period_cannot_swallow_every_epoch(tmp_path, isolated_project_root):
    """
    clamp_grace_epochs, at the one new call site. A short 3b run must still
    produce a file: a grace period covering every epoch means no checkpoint at
    all, which downstream sees as a confusing FileNotFoundError rather than a
    worse model. Same failure this clamp was added for twice before.
    """
    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    stage3a_path = _train_3a(tmp_path, base_path, stage2_path)
    out = _train_3b(tmp_path, base_path, stage2_path, stage3a_path,
                    epochs=1,  # shorter than the derived grace of 3
                    checkpoint_path=tmp_path / "stage3b-short.pt")
    assert out.exists(), "a short non-comparable resume produced no checkpoint at all"
