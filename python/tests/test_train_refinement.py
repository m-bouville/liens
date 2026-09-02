"""
Integration test for train_refinement() -- unlike the other test files,
which test one piece in isolation, this exercises the full wiring:
dataset -> checkpoint loading -> model assembly -> loss ->
criterion tracker -> checkpoint save, on a tiny but real, on-disk sweep.

The individual pieces already have their own thorough, isolated tests
(test_datasets.py, test_checkpoint_components.py, test_model_assembly.py,
test_refinement_loss.py, test_checkpoint_criterion.py) -- this test's
job is specifically to catch INTEGRATION bugs those can't: wrong
argument order/names between pieces, shape mismatches only visible once
real data flows through everything at once, etc.

Run from python/ (imports rely on that root being on sys.path):
    pytest tests/test_train_refinement.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import math
import torch
import pytest

from conftest import cached_sweep

from models.autoencoder import MultiStreamAutoencoder
from models.encoder import Encoder
from models.decoder import Decoder
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from models.latent_dynamics import LatentDynamics
from training.stats_head import StatsHead
from training.train_refinement import train_refinement
from utils import load_datasets as load
from models.constants import N_THETA


SIZE = 64
LATENT_CHANNELS = 4
STAT_NAMES = ["angle", "avg_phi", "stdev_phi"]
STEPS = [0, 1000, 2000, 3000, 4000, 5000]


def _build_run_dir(sweep_dir: Path, name: str, temperature: float, seed: int) -> Path:
    """One run directory: real metadata.txt, real binary snapshots, real
    statistics.csv -- same format established in tests/conftest.py's
    tmp_run_dir/tmp_run_dir_with_stats, just parametrized for building
    several at once."""
    run_dir = sweep_dir / name
    run_dir.mkdir()

    metadata_text = "\n".join([
        f"directory = {name}", "code version = test", "status = complete",
        f"Nx = {SIZE}", f"Ny = {SIZE}", "dt = 0.05", "steps = 5000",
        f"save_steps = {' '.join(str(s) for s in STEPS)}",
        "a0 = 1.0", "b = 1.0", "T0 = 1.0", f"temperature = {temperature}",
        "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
        f"seed = {seed}", "equation = allen_cahn", "solver = explicit", "",
    ])
    (run_dir / "metadata.txt").write_text(metadata_text)

    for step in STEPS:
        arr = np.full((SIZE, SIZE), step / 10000.0, dtype="<f2")
        arr.tofile(run_dir / load.snapshot_filename(step))

    rows = [{"step": s, **{name: s / 1000.0 for name in STAT_NAMES}} for s in STEPS]
    pd.DataFrame(rows).to_csv(run_dir / "statistics.csv", index=False)

    (run_dir / "COMPLETE").touch()  # required by load.is_complete(), or complete_run_dirs
                                     # filters this run out entirely

    return run_dir


def _build_sweep_uncached(tmp_path: Path, n_runs: int = 6) -> Path:
    """base_path such that base_path/64x64/ holds n_runs real run
    directories plus the sweep-level metadata.txt complete_run_dirs()
    needs (see utils.load_datasets.read_sweep_metadata)."""
    sweep_dir = tmp_path / f"{SIZE}x{SIZE}"
    sweep_dir.mkdir(parents=True)

    names = []
    for i in range(n_runs):
        name = f"T800_n010_s{i}"
        _build_run_dir(sweep_dir, name, temperature=0.8, seed=i)
        names.append(name)

    sweep_metadata = "\n".join([
        f"Nx = {SIZE}", f"Ny = {SIZE}", "temperatures = 0.8", "noises = 0.01",
        f"seeds = {','.join(str(i) for i in range(n_runs))}",
        "subdirs =", *names, "",
    ])
    (sweep_dir / "metadata.txt").write_text(sweep_metadata)

    return tmp_path


def _build_ae_checkpoint(path: Path, include_stats_head: bool = True):
    stream_configs = {
        "state": LatentStreamConfig(name="state", channels=LATENT_CHANNELS, spatial_size=8,
                                     mode=LatentStreamMode.AUTOENCODER),
        "deriv": LatentStreamConfig(name="deriv", channels=LATENT_CHANNELS, spatial_size=8,
                                     mode=LatentStreamMode.DECODER),
    }
    encoder = Encoder(input_size=SIZE, in_channels=1, base_channels=4, stream_configs=stream_configs)
    decoder = Decoder(output_size=SIZE, out_channels=1, base_channels=4,
                       latent_channels=LATENT_CHANNELS, latent_spatial_size=8)
    ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                 stream_configs=stream_configs)
    checkpoint = {
        "model_state": ae.state_dict(), "epoch": 1, "val_loss": 0.01,
        "val_loss_ema": 0.01, "test_dirs": [],
        "config": {"size": SIZE, "base_channels": 4, "latent_channels": LATENT_CHANNELS,
                   "latent_spatial_size": 8, "stats_weight": 0.01,
                   "stream_configs": {
                       "state": {"channels": LATENT_CHANNELS, "spatial_size": 8, "mode": "autoencoder"},
                       "deriv": {"channels": LATENT_CHANNELS, "spatial_size": 8, "mode": "decoder"},
                   },
                   "recon_stream_name": "state"},
    }
    if include_stats_head:
        stats_head = StatsHead(latent_channels=LATENT_CHANNELS, stat_names=STAT_NAMES, hidden_dim=8)
        checkpoint["stats_head_state"] = stats_head.state_dict()
        checkpoint["stats_config"] = {
            "stat_names": STAT_NAMES, "stats_mean": torch.zeros(len(STAT_NAMES)),
            "stats_std": torch.ones(len(STAT_NAMES)),
        }
    torch.save(checkpoint, path)


def _build_lds_checkpoint(path: Path, n_rollout_steps: int = 1):
    f_theta = LatentDynamics(latent_channels=LATENT_CHANNELS, n_theta=N_THETA, hidden_dim=8, n_hidden_layers=1)
    checkpoint = {
        "model_state": f_theta.state_dict(), "epoch": 1, "val_loss": 0.05,
        "val_loss_ema": 0.05, "ae_checkpoint": "fake", "test_dirs": [],
        "config": {"latent_channels": LATENT_CHANNELS, "n_theta": N_THETA, "hidden_dim": 8,
                   "n_hidden_layers": 1},
        "data_config": {"min_step": 0, "min_stdev_phi": None,
                        "window_length": n_rollout_steps + 1,
                        "n_rollout_steps": n_rollout_steps},
    }
    torch.save(checkpoint, path)


@pytest.mark.slow
def test_train_refinement_stage4_runs_end_to_end(tmp_path, isolated_project_root):
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)
    _build_lds_checkpoint(lds_checkpoint_path)

    checkpoint_path = tmp_path / "stage4_out.pt"
    result_path = train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
        rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.1,
        epochs=2, batch_size=4, n_rollout_steps=1,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=checkpoint_path, device="cpu", log_every_epoch=True,
    )

    assert result_path == checkpoint_path
    assert checkpoint_path.exists()

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ["ae_state", "f_theta_state", "stats_head_state", "epoch", "val_loss",
                "val_loss_ema", "ae_checkpoint", "lds_checkpoint", "config", "lds_config",
                "stats_config", "stage45_config"]:
        assert key in saved, f"missing expected key '{key}' in saved checkpoint"
    assert saved["stage45_config"]["freeze_decoder"] is True
    assert saved["stats_head_state"] is not None


@pytest.mark.slow
def test_train_refinement_stage5_trainable_decoder(tmp_path, isolated_project_root):
    """freeze_decoder=False (stage 5) should still run fine, and the
    saved checkpoint should correctly record that D was trainable."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)
    _build_lds_checkpoint(lds_checkpoint_path)

    checkpoint_path = tmp_path / "stage5_out.pt"
    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=False,
        rollout_weight=0.1, recon0_weight=1.0, stats0_weight=0.0,
        epochs=2, batch_size=4, n_rollout_steps=1,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=checkpoint_path, device="cpu",
    )

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["stage45_config"]["freeze_decoder"] is False


def test_stage5_recon_predict_threads_through_and_is_recorded(tmp_path, isolated_project_root):
    """End-to-end guard for the stage-5 recon_predict term. The loss math is
    covered in test_refinement_loss and the ramp math in
    test_stage45_regime_and_scales, but nothing checked that
    train_refinement(recon_predict_weight=..., recon_predict_weight_warmup_epochs=...)
    actually wires those through -- the 5-tuple step return, the component
    histories, and the SAVED provenance. A revert of the threading or of the
    stage45_config additions would otherwise pass every test."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)
    _build_lds_checkpoint(lds_checkpoint_path)

    checkpoint_path = tmp_path / "stage5_rp.pt"
    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=False,
        rollout_weight=0.2, recon0_weight=0.2, stats0_weight=0.05,
        recon_predict_weight=1.0, recon_predict_scale=0.05,
        recon_predict_weight_warmup_epochs=2,
        rollout_scale=0.15, recon0_scale=5e-4, stats0_scale=0.15,
        epochs=2, batch_size=4, n_rollout_steps=2,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=checkpoint_path, device="cpu",
    )

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = saved["stage45_config"]
    # provenance: the term that DEFINES a stage-5 objective, and every scale
    # (weight*raw/scale is not reproducible from weights alone) must be recorded.
    assert cfg["recon_predict_weight"] == 1.0
    for k in ("rollout_scale", "recon0_scale", "stats0_scale", "recon_predict_scale"):
        assert k in cfg, f"stage45_config must record {k} to reproduce the objective"
    assert cfg["recon_predict_scale"] == 0.05


def test_stage4_omits_recon_predict_from_the_saved_config_value_but_records_the_key(
        tmp_path, isolated_project_root):
    """A stage-4 run (recon_predict_weight defaults to 0) still RECORDS the key,
    so a reader can tell "recon_predict was off" apart from "this checkpoint
    predates the field". The value is 0.0, not absent."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)
    _build_lds_checkpoint(lds_checkpoint_path)

    checkpoint_path = tmp_path / "stage4_out.pt"
    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
        rollout_weight=1.0, recon0_weight=0.2, stats0_weight=0.05,
        epochs=1, batch_size=4, n_rollout_steps=2,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=checkpoint_path, device="cpu",
    )
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["stage45_config"]["recon_predict_weight"] == 0.0


@pytest.mark.slow
def test_train_refinement_without_ancestor_stats_head_warns_and_skips(tmp_path, capsys, isolated_project_root):
    """If the ancestor AE has no stats_head at all, asking for
    stats0_weight>0 should print a warning and skip L_stats gracefully,
    not crash."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2-nostats.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=False)
    _build_lds_checkpoint(lds_checkpoint_path)

    checkpoint_path = tmp_path / "stage4_nostats_out.pt"
    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
        rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.5,  # requested, but unavailable
        epochs=1, batch_size=4, n_rollout_steps=1,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=checkpoint_path, device="cpu",
    )
    captured = capsys.readouterr()
    assert "WARNING" in captured.out and "no stats_head" in captured.out

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["stats_head_state"] is None
    assert saved["stats_config"] is None


@pytest.mark.slow
def test_train_refinement_mismatched_ancestors_raises_before_training(tmp_path, isolated_project_root):
    """The cross-ancestor validation from checkpoint_components should
    fire here too, at load time -- not partway through training."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3-mismatched.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)

    # Deliberately mismatched latent_channels
    f_theta = LatentDynamics(latent_channels=LATENT_CHANNELS + 4, n_theta=N_THETA,
                              hidden_dim=8, n_hidden_layers=1)
    torch.save({
        "model_state": f_theta.state_dict(), "epoch": 1, "val_loss": 0.05,
        "val_loss_ema": 0.05, "ae_checkpoint": "fake", "test_dirs": [],
        "config": {"latent_channels": LATENT_CHANNELS + 4, "n_theta": N_THETA, "hidden_dim": 8,
                   "n_hidden_layers": 1},
        "data_config": {"min_step": 0, "min_stdev_phi": None, "window_length": 2,
                         "n_rollout_steps": 1},
    }, lds_checkpoint_path)

    with pytest.raises(ValueError, match="disagree on latent_channels"):
        train_refinement(
            base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
            lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
            epochs=1, batch_size=4, n_rollout_steps=1,
            min_step=0, min_stdev_phi=None, device="cpu",
        )


@pytest.mark.slow
def test_train_refinement_requires_min_step(tmp_path, isolated_project_root):
    """Only min_step is genuinely required to be non-None -- see the
    next test for confirmation that min_stdev_phi=None is fine."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path)
    _build_lds_checkpoint(lds_checkpoint_path)

    with pytest.raises(ValueError, match="requires min_step"):
        train_refinement(
            base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
            lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
            epochs=1, device="cpu",
        )


@pytest.mark.slow
def test_train_refinement_min_stdev_phi_none_does_not_raise(tmp_path, isolated_project_root):
    """
    Regression test for the exact bug the test suite caught: min_step
    and min_stdev_phi were both treated as 'must not be None', but
    min_stdev_phi=None is a genuinely valid, meaningful value (no
    stdev-based filtering at all) -- matching
    MicrostructureEvolutionDataset's own float|None semantics -- not
    'forgotten'. Only min_step (which has no meaningful None value at
    the dataset level) should actually be required.
    """
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path)
    _build_lds_checkpoint(lds_checkpoint_path)

    # Should NOT raise -- this is exactly what four earlier tests
    # tripped over before this fix.
    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
        epochs=1, batch_size=4, n_rollout_steps=1,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        device="cpu",
    )


@pytest.mark.slow
def test_epochs_zero_actually_writes_a_checkpoint_stage4(tmp_path, isolated_project_root, capsys):
    """Same regression test as every earlier stage's own, for
    train_refinement -- this is the stage that would have been hit
    NEXT in the reported scenario (Stage 4 resuming from an
    incorrectly-still-real Stage 1b/2 ancestor), had epochs=0 ever
    been tried there too."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)
    _build_lds_checkpoint(lds_checkpoint_path)

    checkpoint_path = tmp_path / "stage4_ablation.pt"
    assert not checkpoint_path.exists()

    result = train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
        epochs=0, batch_size=4, min_step=0, min_stdev_phi=None,
        checkpoint_path=checkpoint_path, device="cpu", log_every_epoch=True,
    )

    assert result == checkpoint_path
    assert checkpoint_path.exists(), "epochs=0 must still write a valid checkpoint"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 0
    output = capsys.readouterr().out
    assert "train_set: skipped" in output, "train_set must be skipped entirely at epochs=0"


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


def _build_lds_checkpoint_u(path: Path):
    """Like _build_lds_checkpoint but a log10_t (u-scheme) f_theta -- its
    config carries time_coordinate='log10_t', so build_models_from_components
    rebuilds a u-model and stage 4's guard should fire."""
    f_theta = LatentDynamics(latent_channels=LATENT_CHANNELS, n_theta=N_THETA,
                             hidden_dim=8, n_hidden_layers=1,
                             dynamics_mode="deriv_linear", time_coordinate="log10_t",
                             dt_cap=float("inf"))
    checkpoint = {
        "model_state": f_theta.state_dict(), "epoch": 1, "val_loss": 0.05,
        "val_loss_ema": 0.05, "ae_checkpoint": "fake", "test_dirs": [],
        "config": {"latent_channels": LATENT_CHANNELS, "n_theta": N_THETA, "hidden_dim": 8,
                   "n_hidden_layers": 1, "dynamics_mode": "deriv_linear",
                   "time_coordinate": "log10_t", "dt_cap": float("inf")},
        "data_config": {"min_step": 0, "min_stdev_phi": None, "window_length": 2,
                         "n_rollout_steps": 1},
    }
    torch.save(checkpoint, path)


@pytest.mark.slow
def test_stage4_handles_log10_t_f_theta(tmp_path, isolated_project_root):
    """Stage 4/5 now SUPPORTS a log10_t (u-scheme) f_theta: the dataset emits
    per-frame physical t (return_frame_t) and compute_stage45_loss converts
    z1->z̃1=ln10*t*z1 and dt->Delta-u before the rollout, so the u-model is fed
    its own coordinate instead of physical dt (which previously NaN'd).

    NOTE min_step=1000 (not 0): the u-coordinate u=log10(t) is SINGULAR at t=0
    -- a window starting at step 0 gives t0=0, Delta-u=log10(t1/0)=inf, NaN loss.
    Real runs never hit this (min_step=2000); the STEPS fixture starts at 0, so
    the test must exclude that first frame. This is a property of log-time, not
    a conversion bug: t>0 is required for the coordinate to be defined."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3-u.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)
    _build_lds_checkpoint_u(lds_checkpoint_path)

    checkpoint_path = tmp_path / "stage4_u_out.pt"
    result_path = train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=True,
        rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.1,
        epochs=2, batch_size=4, n_rollout_steps=1,
        min_step=1000, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=checkpoint_path, device="cpu", log_every_epoch=True,
    )

    # It ran to completion (no refusal, no NaN-driven never-saved) and produced
    # a real, reloadable checkpoint with a FINITE val_loss -- the end-to-end
    # guard that the u-conversion did not diverge on the large-dt windows.
    assert result_path == checkpoint_path
    assert checkpoint_path.exists()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert math.isfinite(saved["val_loss"]), saved["val_loss"]
    assert saved["stage45_config"]["freeze_decoder"] is True


@pytest.mark.slow
def test_printed_component_columns_sum_to_the_totals_beside_them(tmp_path, capsys,
                                                                 isolated_project_root):
    """Behavioral guard for the effective-weight rule the source-assertion in
    test_stage45_regime_and_scales pins: during a ramp the TRAIN columns must be
    computed with the effective (ramped) weight, because the train_total beside
    them is. If a column printed the full weight while the total used the ramped
    one, the columns would not sum to the total and the discrepancy would read
    as a loss bug. Checking the sum proves the columns and the total use the same
    weights -- without needing the raw magnitudes. Run mid-ramp
    (recon_predict_weight_warmup_epochs=3, epochs=2) so the effective weight is
    strictly below full on the epochs printed."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)
    _build_lds_checkpoint(lds_checkpoint_path)

    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_checkpoint_path,
        lds_checkpoint_path=lds_checkpoint_path, freeze_decoder=False,
        rollout_weight=0.2, recon0_weight=0.2, stats0_weight=0.05,
        recon_predict_weight=1.0, recon_predict_scale=0.05,
        recon_predict_weight_warmup_epochs=3,   # ramp still active on epochs 1-2
        rollout_scale=0.15, recon0_scale=5e-4, stats0_scale=0.15,
        epochs=2, batch_size=4, n_rollout_steps=2,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=tmp_path / "s5.pt", device="cpu",
    )
    out = capsys.readouterr().out

    import re
    # epoch lines look like:  "   1| T_tot =c0 +c1 +c2 +c3 | V_tot =d0 +d1 +d2 +d3 |   ema ..."
    checked = 0
    for line in out.splitlines():
        m = re.match(r"\s*\d+\|\s*(-?\d+\.\d+)\s*=(.+?)\|\s*(-?\d+\.\d+)\s*=(.+?)\|", line)
        if not m:
            continue
        for total_str, cols_str in ((m.group(1), m.group(2)), (m.group(3), m.group(4))):
            total = float(total_str)
            cols = [float(x) for x in re.findall(r"-?\d+\.\d+", cols_str)]
            assert cols, f"no component columns parsed from {cols_str!r}"
            # 4-decimal rounding per column -> tolerance scales with column count
            assert abs(sum(cols) - total) < 0.001 * len(cols) + 1e-4, (
                f"columns {cols} sum to {sum(cols):.4f}, not the printed total {total:.4f} "
                f"-- a column used a different weight than the total (the ramped-weight bug)")
            checked += 1
    assert checked >= 2, f"expected to check both train and val on >=1 epoch, checked {checked}"


@pytest.mark.slow
def test_the_grace_message_fires_at_the_ramp_completion_epoch(tmp_path, capsys,
                                                              isolated_project_root):
    """Behavioral form of test_the_criterion_is_reset_when_the_ramp_completes:
    val_loss is never ramped, so during the warmup it grades a model not yet
    trained on the full objective -- terrible numbers that dominate the EMA and
    make the save criterion blind. The tracker is reset (grace) when the ramp
    completes. Here we RUN a ramped stage 5 and assert the grace message fires
    at exactly the completion epoch and not before -- the observable proof the
    reset happened at the right time, rather than the source string."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_p, lds_p = tmp_path / "s2.pt", tmp_path / "s3.pt"
    _build_ae_checkpoint(ae_p, include_stats_head=True)
    _build_lds_checkpoint(lds_p)
    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_p, lds_checkpoint_path=lds_p,
        freeze_decoder=False, rollout_weight=0.2, recon0_weight=0.2, stats0_weight=0.05,
        recon_predict_weight=1.0, recon_predict_scale=0.05,
        recon_predict_weight_warmup_epochs=2,   # completes at epoch 2
        rollout_scale=0.15, recon0_scale=5e-4, stats0_scale=0.15,
        epochs=4, batch_size=4, n_rollout_steps=2,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=tmp_path / "s5.pt", device="cpu",
    )
    out = capsys.readouterr().out
    import re
    fired = [int(m.group(1)) for m in
             re.finditer(r"\[epoch (\d+): [^\]]*reached full strength", out)]
    assert fired == [2], (
        f"grace/reset must fire once, at the ramp-completion epoch 2, got {fired}")
    assert "grace period" in out

    # Observe the reset's EFFECT, not just the message (the two are separable --
    # a refactor could keep the print and drop the reset). reset_with_grace
    # re-baselines the val EMA to the current val_loss, so at the completion
    # epoch the printed EMA equals val_loss exactly; without the reset the EMA
    # would be the smoothed continuation (0.7*prev_ema + 0.3*val), a different
    # number. (Epoch 1 also has ema==val from EMA seeding, so we check epoch 2.)
    rows = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\|[^|]*\|\s*(-?\d+\.\d+)[^|]*\|\s*(-?\d+\.\d+)", line)
        if m:
            rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    assert 2 in rows, f"no epoch-2 row parsed from:\n{out}"
    val2, ema2 = rows[2]
    assert abs(ema2 - val2) < 1e-3, (
        f"at the ramp-completion epoch the EMA ({ema2}) must be re-baselined to "
        f"val_loss ({val2}) by reset_with_grace; it wasn't -- the reset did not fire")
    # and epoch 3 must have RESUMED smoothing (ema != val), confirming the reset
    # was a one-off re-baseline, not a permanent ema=val
    if 3 in rows:
        val3, ema3 = rows[3]
        assert abs(ema3 - val3) > 1e-4, "EMA never resumed smoothing after the reset"


@pytest.mark.slow
def test_no_grace_message_when_there_is_no_warmup(tmp_path, capsys, isolated_project_root):
    """Behavioral form of test_no_reset_when_there_is_no_warmup: with no ramp,
    val_loss is comparable from epoch 1, so the criterion needs no reset and the
    grace machinery must stay silent."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_p, lds_p = tmp_path / "s2.pt", tmp_path / "s3.pt"
    _build_ae_checkpoint(ae_p, include_stats_head=True)
    _build_lds_checkpoint(lds_p)
    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_p, lds_checkpoint_path=lds_p,
        freeze_decoder=True, rollout_weight=1.0, recon0_weight=0.2, stats0_weight=0.05,
        # no warmup on either weight
        rollout_scale=0.15, recon0_scale=5e-4, stats0_scale=0.15,
        epochs=3, batch_size=4, n_rollout_steps=2,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=tmp_path / "s4.pt", device="cpu",
    )
    out = capsys.readouterr().out
    assert "reached full strength" not in out and "grace period" not in out, (
        "the grace machinery fired without any warmup to complete")


@pytest.mark.slow
def test_n_rollout_steps_is_inherited_from_the_lds_checkpoint(tmp_path, isolated_project_root):
    """Behavioral form of test_n_rollout_steps_is_inherited_from_f_theta: build
    an f_theta checkpoint that recorded n_rollout_steps=2, run stage 4 WITHOUT
    passing n_rollout_steps, and assert the run actually used 2 (inherited) --
    not the old signature default of 1, which silently applied f_theta 2-step-
    trained at 1 step across a whole real chain before this was caught. Read
    back from the saved stage45_config (which records the resolved value)."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_p, lds_p = tmp_path / "s2.pt", tmp_path / "s3.pt"
    _build_ae_checkpoint(ae_p, include_stats_head=True)
    _build_lds_checkpoint(lds_p, n_rollout_steps=2)    # f_theta trained at 2
    ckpt = tmp_path / "s4.pt"
    train_refinement(
        base_path=base_path, ae_checkpoint_path=ae_p, lds_checkpoint_path=lds_p,
        freeze_decoder=True, rollout_weight=1.0, recon0_weight=0.2, stats0_weight=0.05,
        rollout_scale=0.15, recon0_scale=5e-4, stats0_scale=0.15,
        # n_rollout_steps NOT passed -> must inherit 2 from the checkpoint
        epochs=1, batch_size=4,
        min_step=0, min_stdev_phi=None, val_fraction=0.3, test_fraction=0.0,
        checkpoint_path=ckpt, device="cpu",
    )
    saved = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert saved["stage45_config"]["n_rollout_steps"] == 2, (
        "stage 4 did not inherit f_theta's n_rollout_steps=2 -- it used the default")
