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
import torch
import pytest

from models.autoencoder import Autoencoder
from models.latent_dynamics import LatentDynamics
from training.stats_head import StatsHead
from training.train_refinement import train_refinement
from utils import load_datasets as load


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


def _build_sweep(tmp_path: Path, n_runs: int = 6) -> Path:
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
    ae = Autoencoder(size=SIZE, channels=1, base_channels=4, latent_channels=LATENT_CHANNELS)
    checkpoint = {
        "model_state": ae.state_dict(), "epoch": 1, "val_loss": 0.01,
        "val_loss_ema": 0.01, "test_dirs": [],
        "config": {"size": SIZE, "base_channels": 4, "latent_channels": LATENT_CHANNELS,
                   "stats_weight": 0.01},
    }
    if include_stats_head:
        stats_head = StatsHead(latent_channels=LATENT_CHANNELS, stat_names=STAT_NAMES, hidden_dim=8)
        checkpoint["stats_head_state"] = stats_head.state_dict()
        checkpoint["stats_config"] = {
            "stat_names": STAT_NAMES, "stats_mean": torch.zeros(len(STAT_NAMES)),
            "stats_std": torch.ones(len(STAT_NAMES)),
        }
    torch.save(checkpoint, path)


def _build_lds_checkpoint(path: Path):
    f_theta = LatentDynamics(latent_channels=LATENT_CHANNELS, n_theta=1, hidden_dim=8, n_hidden_layers=1)
    checkpoint = {
        "model_state": f_theta.state_dict(), "epoch": 1, "val_loss": 0.05,
        "val_loss_ema": 0.05, "ae_checkpoint": "fake", "test_dirs": [],
        "config": {"latent_channels": LATENT_CHANNELS, "n_theta": 1, "hidden_dim": 8,
                   "n_hidden_layers": 1},
        "data_config": {"min_step": 0, "min_stdev_phi": None, "window_length": 2,
                         "n_rollout_steps": 1},
    }
    torch.save(checkpoint, path)


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


def test_train_refinement_mismatched_ancestors_raises_before_training(tmp_path, isolated_project_root):
    """The cross-ancestor validation from checkpoint_components should
    fire here too, at load time -- not partway through training."""
    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_checkpoint_path = tmp_path / "fake-stage2.pt"
    lds_checkpoint_path = tmp_path / "fake-stage3-mismatched.pt"
    _build_ae_checkpoint(ae_checkpoint_path, include_stats_head=True)

    # Deliberately mismatched latent_channels
    f_theta = LatentDynamics(latent_channels=LATENT_CHANNELS + 4, n_theta=1,
                              hidden_dim=8, n_hidden_layers=1)
    torch.save({
        "model_state": f_theta.state_dict(), "epoch": 1, "val_loss": 0.05,
        "val_loss_ema": 0.05, "ae_checkpoint": "fake", "test_dirs": [],
        "config": {"latent_channels": LATENT_CHANNELS + 4, "n_theta": 1, "hidden_dim": 8,
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
