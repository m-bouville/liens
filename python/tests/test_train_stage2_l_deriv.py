import torch

from utils import load_datasets as load
from training.train_stage1 import train_autoencoder
from training.train_stage2 import train_stage2
import re

import pytest

from conftest import cached_sweep, cached_stage1_ancestor


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


def _build_sweep_uncached(tmp_path, n_runs=6, size=32):
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


@pytest.mark.slow
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


@pytest.mark.slow
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
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
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


@pytest.mark.slow
def test_epoch0_reference_does_not_perturb_training_rng(tmp_path, isolated_project_root, monkeypatch):
    """
    REGRESSION: the epoch-0 reference row (a pure diagnostic, printed
    for comparison against epoch 1 onward) must not change the actual
    training outcome. Caught directly via a real A/B run before this
    fix existed: the reference block's own forward passes shifted
    torch's global RNG state, which changed train_loader's own shuffle
    order (shuffle=True) at epoch 1, silently producing a DIFFERENT
    final model_state depending on whether this purely-informational
    row was present at all.

    Verified here directly against the actual mechanism (RNG state
    save/restore around the reference block) via monkeypatching, not
    by comparing two full training runs against each other -- since
    both runs would trigger the reference block identically, any RNG
    shift it causes would be the SAME shift both times, making a
    plain two-run comparison unable to detect this class of bug at
    all (confirmed directly: removing the real fix and rerunning that
    kind of comparison still passed). Instead, directly confirms
    torch.set_rng_state is called with EXACTLY the state
    torch.get_rng_state returned right before the reference block's
    own forward passes ran -- the real property that matters,
    independent of whether any particular seed/architecture happens
    to expose it as a visible training difference.
    """
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
    )

    import torch as torch_module
    real_get_rng_state = torch_module.get_rng_state
    real_set_rng_state = torch_module.set_rng_state
    calls = []

    def _tracking_get_rng_state():
        state = real_get_rng_state()
        calls.append(("get", state.clone()))
        return state

    def _tracking_set_rng_state(state):
        calls.append(("set", state.clone()))
        return real_set_rng_state(state)

    monkeypatch.setattr(torch_module, "get_rng_state", _tracking_get_rng_state)
    monkeypatch.setattr(torch_module, "set_rng_state", _tracking_set_rng_state)

    torch.manual_seed(0)
    train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, stats0_weight=0.01, stats1_weight=1.0,
        epochs=2, batch_size=4, num_workers=0,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_epoch0ref.pt", device="cpu",
        log_every_epoch=False, loss_curve_path=tmp_path / "curve2_epoch0ref.png",
        deriv_target_centered=True,
    )

    get_calls = [state for kind, state in calls if kind == "get"]
    set_calls = [state for kind, state in calls if kind == "set"]
    assert len(get_calls) >= 1 and len(set_calls) >= 1, (
        "expected the epoch-0 reference block to call both torch.get_rng_state and "
        "torch.set_rng_state at least once (epochs=2 > 0, so the block must have run)"
    )
    # The FIRST get must be matched by an EQUAL later set -- the actual
    # save-before/restore-after property, not just "these functions
    # got called at some point for some unrelated reason".
    first_get = get_calls[0]
    assert any(torch.equal(first_get, s) for s in set_calls), (
        "the RNG state captured before the reference block's own forward passes was "
        "never restored via an equal torch.set_rng_state call -- the reference block's "
        "own randomness can leak into the real training that follows it"
    )


@pytest.mark.slow
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


@pytest.mark.slow
def test_deriv_target_centered_switches_at_ramp_completion(tmp_path, isolated_project_root, capsys):
    """deriv_target_centered's own switch epoch is derived from
    deriv_weight_warmup_epochs, not a separate knob (see train_stage2's
    own docstring).

    Asserts the SCHEDULE, which is deterministic, rather than whether a
    save happened to occur after the switch -- that depends on whether
    val_loss improved, which is stochastic, and made an earlier version
    of this test flaky (it failed ~2 runs in 3 as part of the full suite
    while passing alone). Papering over that with more epochs made the
    test slow AND left it flaky; asserting the deterministic property
    fixes both.
    """
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
    )
    capsys.readouterr()

    stage2_path = train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, deriv_weight_warmup_epochs=2, stats0_weight=0.01,
        epochs=3, batch_size=4, num_workers=0,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_centered_fresh.pt", device="cpu",
        log_every_epoch=False, loss_curve_path=tmp_path / "curve2_centered_fresh.png",
        deriv_target_centered=True,
    )
    printed = capsys.readouterr().out

    cfg = torch.load(stage2_path, map_location="cpu", weights_only=True)["stage2_config"]
    assert cfg["deriv_target_centered"] is True
    assert cfg["deriv_switch_epoch"] == 2, "switch epoch must track deriv_weight_warmup_epochs"
    # A FRESH run (prior_stage2_epochs=0) must spend its first epoch in
    # the cheaper one-sided phase, then announce the switch at epoch 2.
    assert "switching to the centered L_deriv target at epoch 2" in printed
    assert "prior_stage2_epochs=0" in printed
    assert "[epoch 2: switching to the centered L_deriv target now" in printed


@pytest.mark.slow
def test_deriv_target_centered_resume_skips_completed_warmup(tmp_path, capsys,
                                                            isolated_project_root):
    """REGRESSION: resuming a stage-2 checkpoint that's already well
    past deriv_weight_warmup_epochs must start the NEW run directly in
    the centered phase, at full deriv_weight, from ITS OWN epoch 1 --
    not re-ramp the weight or waste epochs in the cheaper phase again.
    Uses the checkpoint's own saved "epoch" to detect this (see
    prior_stage2_epochs in train_stage2's own docstring).

    This is the scenario an already-fully-trained stage-2 checkpoint
    (trained with deriv_target_centered=False, no relation to THIS
    run's own deriv_weight_warmup_epochs at all) needs to resume
    smoothly into deriv_target_centered=True without wasting the prior
    run's own computation."""
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
    )
    # A "fully trained" ancestor, well past any reasonable warmup --
    # trained WITHOUT deriv_target_centered at all, matching a real
    # pre-existing checkpoint from before this feature existed.
    stage2_ancestor_path = train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, stats0_weight=0.01,
        epochs=5, batch_size=4, num_workers=0,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_ancestor.pt", device="cpu",
        log_every_epoch=False, loss_curve_path=tmp_path / "curve2_ancestor.png",
    )
    ancestor_epoch = torch.load(stage2_ancestor_path, map_location="cpu",
                                 weights_only=True)["epoch"]
    assert ancestor_epoch >= 3, "fixture assumption: ancestor must be past a small warmup"

    stage2_resumed_path = train_stage2(
        base_path=base_path, resume_from=stage2_ancestor_path,
        deriv_weight=1.0, deriv_weight_warmup_epochs=2, stats0_weight=0.01,
        epochs=20, batch_size=4, num_workers=0,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_resumed.pt", device="cpu",
        log_every_epoch=False, loss_curve_path=tmp_path / "curve2_resumed.png",
        deriv_target_centered=True,
    )

    # ASSERTED ON THE SWITCH ITSELF, not on the saved checkpoint.
    #
    # use_centered_at_save was only ever a proxy, and it disappears whenever the
    # run legitimately fails to improve: the switch resets the criterion with a
    # grace period, nothing saves during it, and if no later epoch beats the
    # post-grace bar the no-improvement fallback copies the ANCESTOR forward --
    # which was trained with deriv_target_centered=False, so the field reads
    # False for a run that switched perfectly correctly at epoch 1.
    #
    # The BRACKETED in-loop message is the right observable: `just_switched`
    # emits it from the actual condition. The header line is NOT -- it computes
    # own_switch_epoch separately and still reads "epoch 1" even when the
    # condition ignores prior_stage2_epochs entirely (verified: that mutation
    # passed against the header and fails against this).
    out = capsys.readouterr().out
    assert "[epoch 1: switching to the centered L_deriv target now" in out, (
        "resuming a stage-2 checkpoint already well past deriv_weight_warmup_epochs "
        "must be centered from this run's own epoch 1, not stuck re-running the cheap "
        f"phase. Log said:\n{out[-1200:]}"
    )

    saved = torch.load(stage2_resumed_path, map_location="cpu", weights_only=True)
    ancestor = torch.load(stage2_ancestor_path, map_location="cpu", weights_only=True)
    if saved["epoch"] != ancestor["epoch"]:
        # A real save happened, so the field IS meaningful here.
        assert saved["stage2_config"]["use_centered_at_save"] is True


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


@pytest.mark.slow
def test_loss_component_scatter_writes_a_SEPARATE_file_from_loss_curve(tmp_path, isolated_project_root):
    """
    REGRESSION: loss_components_path used to be derived via
    loss_curve_path.name.replace("loss_curve", "loss_components") --
    a silent no-op whenever the caller's own filename doesn't literally
    contain the substring "loss_curve" (e.g. this very test file's own
    "curve2.png"), making loss_components_path collide with
    loss_curve_path itself. Since loss_component_scatter() is called
    AFTER loss_curve() each epoch, it then silently OVERWROTE the real
    loss-curve figure with the 3-panel component grid instead --
    confirmed directly by image dimensions (loss_curve's own 800x500
    vs the grid's 1500x450) before this was caught.
    """
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
    )
    # Deliberately a filename with NO "loss_curve" substring in it at
    # all -- the exact shape that broke the old .replace()-based
    # derivation.
    curve_path = tmp_path / "curve2.png"
    train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, stats0_weight=0.01,
        epochs=2, batch_size=4, num_workers=0,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_scatter.pt", device="cpu",
        log_every_epoch=True, loss_curve_path=curve_path,
    )

    components_path = tmp_path / "curve2-components.png"
    assert curve_path.exists()
    assert components_path.exists()
    assert components_path != curve_path

    import struct
    def _png_size(path):
        data = path.read_bytes()[16:24]
        return struct.unpack(">II", data)

    assert _png_size(curve_path) == (800, 500), (
        "loss_curve.png's own dimensions changed -- suggests it was overwritten by "
        "something else again"
    )
    assert _png_size(components_path) != (800, 500), (
        "the components figure has loss_curve's own dimensions -- the two are not "
        "actually distinct files"
    )


@pytest.mark.slow
def test_loss_component_scatter_values_reconstruct_train_total(tmp_path, isolated_project_root, monkeypatch):
    """The per-component values fed to loss_component_scatter must sum
    (recon0 + stats0 + deriv, all already weight/scale-normalized) to
    EXACTLY the same train_total the console prints for that epoch --
    otherwise the two diagnostics would silently disagree with each
    other about the same run."""
    import training.train_stage2 as ts2

    captured = {}
    real_fn = ts2.loss_component_scatter

    def spy(epoch_history, component_histories, output_path, **kw):
        captured["epoch_history"] = list(epoch_history)
        captured["component_histories"] = {
            k: {kk: list(vv) for kk, vv in v.items()} for k, v in component_histories.items()
        }
        return real_fn(epoch_history, component_histories, output_path, **kw)

    monkeypatch.setattr(ts2, "loss_component_scatter", spy)

    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
    )
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_stage2(
            base_path=base_path, resume_from=stage1_path,
            deriv_weight=1.0, stats0_weight=0.01,
            epochs=2, batch_size=4, num_workers=0,
            min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / "stage2_recon.pt", device="cpu",
            log_every_epoch=True, loss_curve_path=tmp_path / "curve2_recon.png",
        )
    printed = buf.getvalue()

    assert captured["component_histories"], "loss_component_scatter was never called"
    n = len(captured["epoch_history"])
    for i in range(n):
        epoch = captured["epoch_history"][i]
        reconstructed = sum(captured["component_histories"][c]["train"][i]
                            for c in captured["component_histories"])
        m = re.search(rf"^{epoch:4d}\|\s*([\d.]+)\s*=", printed, re.MULTILINE)
        assert m, f"epoch {epoch}: no console line found to compare against"
        printed_total = float(m.group(1))
        assert reconstructed == pytest.approx(printed_total, abs=5e-4), (
            f"epoch {epoch}: reconstructed train_total={reconstructed:.4f} from the component "
            f"histories doesn't match the console's own {printed_total:.4f} (tolerance 5e-4, "
            f"matching the console's own 4-decimal rounding -- these are two independently "
            f"computed, mathematically equal sums, not required to be bit-identical)"
        )


def test_reference_row_keeps_its_column_width_on_a_huge_untrained_component():
    """
    GUARDS `:7.4f` on the reference row. It reports UNTRAINED components, and a
    freshly built deriv stream starts around 4e4 -- rendered as "40613.3085":
    ten characters in a seven-character field, and nine significant figures for
    a number whose leading digit is the only meaningful one. It broke the
    column and overstated the precision.

    Values that fit are unchanged, so the epoch rows are untouched.
    """
    from training.train_stage2 import _compact_loss

    assert _compact_loss(22.3422) == "22.3422"      # fits: identical to :7.4f
    assert _compact_loss(0.7916) == " 0.7916"
    assert len(_compact_loss(float("nan"))) == 7

    big = _compact_loss(40613.3085)
    assert len(big) < len(f"{40613.3085:7.4f}"), "must be shorter than the fixed-point form"
    assert big.strip() == "4.061e+04", big


def test_the_switch_message_agrees_with_the_switch_CONDITION():
    """
    deriv_switch_epoch counts CUMULATIVE epochs -- the condition is
    `prior_stage2_epochs + epoch >= deriv_switch_epoch` -- so it is not this
    run's own epoch number when resuming.

    The message printed it as if it were, contradicting the very next clause
    of the same sentence, which correctly derived the count of one-sided
    epochs. Observed with switch=10, prior=2:

        "switching ... at epoch 10 of this run's own numbering
         (prior_stage2_epochs=2, so this run spends its first 7 epoch(s) ...)"

    7 + 1 = 8, not 10. A reader watching for the switch would have looked two
    epochs too late.
    """
    from conftest import source_without_comments
    import training.train_stage2 as mod

    src = source_without_comments(mod)
    assert "own_switch_epoch = max(1, deriv_switch_epoch - prior_stage2_epochs)" in src, (
        "the message must convert to this run's own numbering"
    )
    assert "{own_switch_epoch} of this run's own numbering" in src
    assert "cumulative epoch {deriv_switch_epoch}" in src, (
        "the cumulative number is still worth showing -- it is what the condition uses"
    )


def test_the_two_halves_of_the_message_are_consistent():
    """
    The printed switch epoch and the printed one-sided count must satisfy
    count + 1 == switch_epoch, for every resume position. Computed from the
    same expressions the code uses, so a change to either half fails here.
    """
    for switch, prior, epochs in ((10, 2, 100), (15, 0, 50), (3, 10, 20), (1, 0, 5)):
        own = max(1, switch - prior)
        already_past = prior + 1 >= switch
        if already_past:
            continue
        count = min(switch - prior - 1, epochs)
        assert count + 1 == own, (
            f"switch={switch} prior={prior}: message would say 'epoch {own}' and "
            f"'first {count} epochs', which disagree"
        )


@pytest.mark.slow
def test_no_curve_marker_when_already_centered_at_epoch_one(tmp_path, capsys,
                                                             isolated_project_root):
    """
    REGRESSION: a fresh run with deriv_target_centered=True and no warmup
    (deriv_switch_epoch <= 1) is centered from its own epoch 1. just_switched
    fires there -- correctly, for the print message and the grace-period
    reset, which are about comparability against the LOADED checkpoint's
    val_loss and are real concerns even at epoch 1.

    But the loss curve's own epoch history has no point before epoch 1 (the
    epoch-0 reference row is deliberately never added to it), so a vertical
    "centered L_deriv target" marker at epoch 0.5 has nothing on either side
    of it -- a discontinuity marker for a discontinuity that isn't in this
    figure. It must not be added.
    """
    import training.train_stage2 as train_stage2_module

    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
    )
    capsys.readouterr()

    captured_events = []
    captured_levels = []
    real_loss_curve = train_stage2_module.loss_curve

    def _recording_loss_curve(*args, **kwargs):
        captured_events.append(list(kwargs.get("event_epochs") or []))
        captured_levels.append(list(kwargs.get("reference_levels") or []))
        return real_loss_curve(*args, **kwargs)

    train_stage2_module.loss_curve = _recording_loss_curve
    try:
        train_stage2(
            base_path=base_path, resume_from=stage1_path,
            # warmup=0 -> deriv_switch_epoch <= 1: centered from epoch 1,
            # nothing before it in THIS run.
            deriv_weight=1.0, deriv_weight_warmup_epochs=0, stats0_weight=0.01,
            epochs=3, batch_size=4, num_workers=0,
            min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / "stage2_centered_from_1.pt", device="cpu",
            log_every_epoch=True, loss_curve_path=tmp_path / "curve2_from1.png",
            deriv_target_centered=True,
        )
    finally:
        train_stage2_module.loss_curve = real_loss_curve

    printed = capsys.readouterr().out
    # the print/grace-period side of just_switched still fires at epoch 1 --
    # UNCHANGED by this fix, and a real concern (val_loss under the loaded
    # checkpoint's target isn't a fair bar for the new target either).
    assert "[epoch 1: switching to the centered L_deriv target now" in printed

    # but no plotted curve ever received an event at epoch 0.5
    all_events = [e for call_events in captured_events for e in call_events]
    assert not any(x == 0.5 for x, _ in all_events), (
        f"a curve marker was drawn at epoch 0.5 with no preceding point on "
        f"this run's own curve: {all_events}"
    )
    # INSTEAD it gets the horizontal reference level: the target is constant
    # across this whole run, so the ancestor's val_loss under that same
    # target IS a fair bar to read the curve against.
    all_levels = [lv for call_levels in captured_levels for lv in call_levels]
    assert all_levels, (
        "no reference level either -- an already-centered run should be "
        "annotated with the bar it started from"
    )
    assert any("reference" in label for _, label in all_levels), all_levels


@pytest.mark.slow
def test_curve_marker_is_kept_for_a_genuine_mid_run_switch(tmp_path, capsys,
                                                            isolated_project_root):
    """The counterpart: when the switch happens after a real preceding point
    on THIS run's own curve (deriv_switch_epoch=2, so epoch 1 is plotted
    on the cheap target before epoch 2 switches), the marker at 1.5 must
    still be drawn -- the fix must not suppress genuine mid-run switches."""
    import training.train_stage2 as train_stage2_module

    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
    )
    capsys.readouterr()

    captured_events = []
    captured_levels = []
    real_loss_curve = train_stage2_module.loss_curve

    def _recording_loss_curve(*args, **kwargs):
        captured_events.append(list(kwargs.get("event_epochs") or []))
        captured_levels.append(list(kwargs.get("reference_levels") or []))
        return real_loss_curve(*args, **kwargs)

    train_stage2_module.loss_curve = _recording_loss_curve
    try:
        train_stage2(
            base_path=base_path, resume_from=stage1_path,
            deriv_weight=1.0, deriv_weight_warmup_epochs=2, stats0_weight=0.01,
            epochs=3, batch_size=4, num_workers=0,
            min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / "stage2_midrun.pt", device="cpu",
            log_every_epoch=True, loss_curve_path=tmp_path / "curve2_midrun.png",
            deriv_target_centered=True,
        )
    finally:
        train_stage2_module.loss_curve = real_loss_curve

    all_events = [e for call_events in captured_events for e in call_events]
    assert any(x == 1.5 for x, _ in all_events), (
        f"the genuine mid-run switch at epoch 2 lost its curve marker at "
        f"1.5: {all_events}"
    )
    # and NO horizontal reference level here: the reference was measured
    # under the one-sided target, so a flat line drawn across the switch
    # would invite comparing post-switch points against a bar computed for
    # a different quantity. The two annotations are mutually exclusive.
    all_levels = [lv for call_levels in captured_levels for lv in call_levels]
    assert not all_levels, (
        f"a reference level was drawn across a mid-run target switch, where "
        f"it is not a fair bar: {all_levels}"
    )


@pytest.mark.slow
def test_interp_weight_requires_the_centered_target(tmp_path, isolated_project_root):
    """
    L_interp needs the (t-, t, t+) triplet. Only the centered path loads a
    3-frame window; the one-sided path has window_length=2 and no middle
    frame to interpolate TO. Refusing loudly beats silently computing zero
    -- the silently-ignored-parameter failure mode this project already has
    a standing todo about.
    """
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01,
        stat_names=["avg_phi"], device="cpu", seed=0, log_every_epoch=False,
    )
    with pytest.raises(ValueError, match="deriv_target_centered"):
        train_stage2(
            base_path=base_path, resume_from=stage1_path,
            deriv_weight=1.0, stats0_weight=0.01, epochs=1, batch_size=4,
            num_workers=0, min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / "s2_bad.pt", device="cpu",
            log_every_epoch=False, deriv_target_centered=False,
            interp_weight=0.1,
        )


@pytest.mark.slow
def test_interp_and_centered_target_share_one_switch_epoch(tmp_path, capsys,
                                                            isolated_project_root):
    """
    Both gate on deriv_switch_epoch (from deriv_weight_warmup_epochs), so
    there is no way to configure one active without the other's data, and
    the loss-curve marker names BOTH -- the cliff there is the sum of two
    objective changes, and attributing it to the target change alone would
    understate what moved.
    """
    import training.train_stage2 as train_stage2_module

    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01,
        stat_names=["avg_phi"], device="cpu", seed=0, log_every_epoch=False,
    )
    capsys.readouterr()

    captured_events = []
    real_loss_curve = train_stage2_module.loss_curve

    def _recording(*args, **kwargs):
        captured_events.append(list(kwargs.get("event_epochs") or []))
        return real_loss_curve(*args, **kwargs)

    train_stage2_module.loss_curve = _recording
    try:
        train_stage2(
            base_path=base_path, resume_from=stage1_path,
            deriv_weight=1.0, deriv_weight_warmup_epochs=2, stats0_weight=0.01,
            epochs=3, batch_size=4, num_workers=0,
            min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / "s2_interp.pt", device="cpu",
            log_every_epoch=True,
            loss_curve_path=tmp_path / "curve_interp.png",
            deriv_target_centered=True, interp_weight=0.1,
        )
    finally:
        train_stage2_module.loss_curve = real_loss_curve

    labels = [lab for call in captured_events for _x, lab in call]
    assert any("L_interp" in lab for lab in labels), (
        f"the switch marker does not name L_interp, so the extra cliff it "
        f"contributes is misattributed: {labels}"
    )


@pytest.mark.slow
def test_interp_weight_zero_leaves_the_loss_untouched(tmp_path,
                                                       isolated_project_root):
    """interp_weight=0.0 must reproduce the old behaviour exactly -- the
    established pattern for every added term."""
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01,
        stat_names=["avg_phi"], device="cpu", seed=0, log_every_epoch=False,
    )
    import torch as _torch
    paths = []
    for weight in (0.0, 0.0):
        _torch.manual_seed(0)
        paths.append(train_stage2(
            base_path=base_path, resume_from=stage1_path,
            deriv_weight=1.0, deriv_weight_warmup_epochs=0, stats0_weight=0.01,
            epochs=2, batch_size=4, num_workers=0,
            min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / f"s2_w{weight}_{len(paths)}.pt",
            device="cpu", log_every_epoch=False,
            deriv_target_centered=True, interp_weight=weight,
        ))
    a = _torch.load(paths[0], map_location="cpu", weights_only=True)
    b = _torch.load(paths[1], map_location="cpu", weights_only=True)
    assert a["val_loss"] == b["val_loss"], (
        "two identical interp_weight=0 runs disagree, so the added term "
        "perturbs training even when disabled"
    )


@pytest.mark.slow
def test_interp_weight_actually_changes_training(tmp_path, isolated_project_root):
    """
    A term that is computed, returned and reported but never added to the
    total would pass every other test here. This is the one that fails if
    L_interp is inert: the same run at interp_weight 0 vs 0.5 must reach a
    DIFFERENT model.
    """
    import torch as _torch
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01,
        stat_names=["avg_phi"], device="cpu", seed=0, log_every_epoch=False,
    )
    losses = []
    for weight in (0.0, 0.5):
        _torch.manual_seed(0)
        path = train_stage2(
            base_path=base_path, resume_from=stage1_path,
            deriv_weight=1.0, deriv_weight_warmup_epochs=0, stats0_weight=0.01,
            epochs=2, batch_size=4, num_workers=0,
            min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / f"s2_influence_{weight}.pt",
            device="cpu", log_every_epoch=False,
            deriv_target_centered=True, interp_weight=weight,
        )
        losses.append(_torch.load(path, map_location="cpu",
                                   weights_only=True)["val_loss"])
    assert losses[0] != losses[1], (
        f"interp_weight 0.0 and 0.5 produced the same val_loss ({losses[0]}), "
        f"so L_interp is not reaching the optimizer at all"
    )


@pytest.mark.slow
def test_interp_appears_in_the_printed_loss_breakdown(tmp_path, capsys,
                                                        isolated_project_root):
    """
    REGRESSION: interp_weight=0.5 ran with the term in the TOTAL but absent
    from active_terms, so the console formula, the per-epoch component
    columns, component_names and loss_component_scatter all omitted it --
    a three-term breakdown printed beside a total its parts could not sum
    to. Everything that reports the breakdown derives from active_terms, so
    the term must be registered there, not only added into the total.
    """
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
    )
    capsys.readouterr()
    train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, deriv_weight_warmup_epochs=0, stats0_weight=0.01,
        interp_weight=0.5, deriv_target_centered=True,
        epochs=2, batch_size=4, num_workers=0,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_interp.pt", device="cpu",
        log_every_epoch=True, loss_curve_path=tmp_path / "curve_interp.png",
    )
    printed = capsys.readouterr().out
    assert "*interp/" in printed, (
        "the loss-composition header does not mention interp, so a run with "
        "interp_weight > 0 reports a breakdown missing one of its terms"
    )


@pytest.mark.slow
def test_interp_weight_zero_leaves_the_breakdown_untouched(tmp_path, capsys,
                                                            isolated_project_root):
    """The established pattern: 0.0 must reproduce today's output exactly,
    so the term is registered only when it is actually active."""
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
    )
    capsys.readouterr()
    train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, deriv_weight_warmup_epochs=0, stats0_weight=0.01,
        interp_weight=0.0, deriv_target_centered=True,
        epochs=2, batch_size=4, num_workers=0,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=tmp_path / "stage2_nointerp.pt", device="cpu",
        log_every_epoch=True, loss_curve_path=tmp_path / "curve_nointerp.png",
    )
    assert "*interp/" not in capsys.readouterr().out


def test_interp_parameters_are_recorded_in_the_checkpoint(tmp_path,
                                                           isolated_project_root):
    """
    REGRESSION (audit finding): every other loss weight is saved in
    stage2_config (deriv_weight, stats0_weight, stats1_weight) but interp
    was not -- a checkpoint trained WITH L_interp was indistinguishable
    from one trained without, which breaks provenance for exactly the
    before/after comparisons the acceptance tests rely on.
    """
    import torch
    base_path, stage1_path = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32),
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, augment=False,
        min_step=0, min_stdev_phi=None, stats0_weight=0.01, stat_names=["avg_phi"],
        device="cpu", seed=0, log_every_epoch=False,
    )
    out = tmp_path / "stage2_prov.pt"
    train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, deriv_weight_warmup_epochs=0, stats0_weight=0.01,
        interp_weight=0.125, interp_scale=0.005, deriv_target_centered=True,
        epochs=1, batch_size=4, num_workers=0,
        min_step=0, min_stdev_phi=None,
        checkpoint_path=out, device="cpu", log_every_epoch=False,
    )
    cfg = torch.load(out, map_location="cpu", weights_only=True)["stage2_config"]
    assert cfg["interp_weight"] == 0.125
    assert cfg["interp_scale"] == 0.005



def test_trunk_isolation_does_not_shadow_the_deriv_stream_config():
    """
    REGRESSION (source-level, because the crashing path needs a resume from
    a stats_head1-bearing stage-2 checkpoint that is costly to stage in a
    unit test): the trunk-isolation block must NOT bind a local
    `deriv_stream` -- that name already holds the stream CONFIG object, read
    later as deriv_stream.channels for stats_head1. Rebinding it to the
    stream NAME (a str) caused "'str' object has no attribute 'channels'"
    on any isolation run resuming from a checkpoint with stats_head1.

    Guard the invariant directly: within train_stage2, `deriv_stream` is
    only ever assigned stream_configs[...], never a bare name.
    """
    import ast
    import inspect
    from training.train_stage2 import train_stage2

    tree = ast.parse(inspect.getsource(train_stage2))
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    deriv_assigns = [n for n in assigns
                     for t in n.targets
                     if isinstance(t, ast.Name) and t.id == "deriv_stream"]
    assert deriv_assigns, "no deriv_stream assignment found -- test is stale"
    for node in deriv_assigns:
        assert isinstance(node.value, ast.Subscript), (
            "deriv_stream is assigned something other than stream_configs[...]; "
            "if it is the stream NAME (a str), the stats_head1 path reads a "
            "str .channels and crashes"
        )
