"""
Tests for evaluation/check_deriv_temperature.py -- specifically, that it
shares evaluation._fits.robust_polynomial_fit rather than carrying its
own copy (as it used to). See _fits.py's own module docstring for why
that duplication was a real problem, not just untidy: the two copies
were used to fit the SAME underlying quantity (eps/eps') against
DIFFERENT bases while reporting the shared coefficients under the same
names, producing directly-contradictory numbers from the same checkpoint.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from test_compare_rollout_training import (
    LATENT_CHANNELS, SIZE, _build_ae_checkpoint, _build_run_dir,
)
from test_train_stage2_l_deriv import _build_sweep

import evaluation.check_deriv_temperature as cdt
from evaluation._fits import robust_polynomial_fit
from evaluation.check_deriv_temperature import check_deriv_temperature


def test_shares_fits_module_rather_than_redefining_it():
    """
    REGRESSION: check_deriv_temperature.py must import
    robust_polynomial_fit from evaluation._fits, not define its own
    module-level copy again. `is`, not just numeric equality -- the
    whole point is ONE implementation, not two that happen to agree
    right now and can silently drift apart later (exactly how this
    duplication originally happened).
    """
    assert cdt.robust_polynomial_fit is robust_polynomial_fit


def test_fit_still_recovers_known_coefficients_through_the_shared_import():
    """Exercises the IRLS/Huber iteration (not just import wiring) via
    the path check_deriv_temperature.py actually calls it through --
    same shape of problem this module fits: y = eps/dt + eps', with
    outliers the Huber reweighting must down-weight to recover the
    true coefficients."""
    rng = np.random.default_rng(0)
    dt = rng.uniform(10, 5000, 300)
    true_eps, true_eps_prime = 3.1e-3, -8e-5
    y = true_eps / dt + true_eps_prime + rng.normal(0, 1e-4, 300)
    y[::20] += rng.normal(0, 5e-2, 15)  # outliers

    basis = [lambda d: 1.0 / d, lambda d: np.ones_like(d)]
    coefs, stderr = cdt.robust_polynomial_fit(dt, y, basis)

    assert coefs[0] == pytest.approx(true_eps, rel=0.1)
    assert coefs[1] == pytest.approx(true_eps_prime, abs=5e-5)
    assert np.all(np.isfinite(stderr))


def _stage2_checkpoint_with_test_dirs(tmp_path: Path, run_dirs) -> Path:
    """test_compare_rollout_training's own _build_ae_checkpoint, but with
    test_dirs actually populated -- check_deriv_temperature reads them
    from the checkpoint itself (it takes no base_path argument)."""
    path = tmp_path / "fake-stage2-deriv-temp.pt"
    _build_ae_checkpoint(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck["test_dirs"] = [str(d) for d in run_dirs]
    torch.save(ck, path)
    return path


@pytest.mark.parametrize("centered", [False, True])
def test_check_deriv_temperature_runs_end_to_end(tmp_path, isolated_project_root, centered):
    """
    This module's ENTRY POINT had no end-to-end coverage at all -- only
    the two tests above, which check that it shares _fits'
    robust_polynomial_fit but never actually run the diagnostic. That
    gap mattered specifically because this module's setup phase was
    refactored (its own duplicate copy of the fitter removed), and a
    sharing test alone can't catch a break in the surrounding wiring.

    Both target modes exercised: deriv_target_centered=False uses
    window_length=2 (one-sided target), True uses window_length=3
    (centered) -- a real branch in the dataset construction, not just a
    flag passed through.
    """
    run_dir = _build_run_dir(tmp_path)
    checkpoint_path = _stage2_checkpoint_with_test_dirs(tmp_path, [run_dir])

    result = check_deriv_temperature(
        stage2_checkpoint_path=checkpoint_path, min_step=0, device="cpu",
        deriv_target_centered=centered,
    )
    assert isinstance(result, dict) and result, "expected a non-empty dict of fitted coefficients"


@pytest.mark.slow
def test_check_deriv_temperature_rejects_a_checkpoint_with_no_deriv_stream(
    tmp_path, isolated_project_root,
):
    """A stage-1 (single-stream) checkpoint has no z1 at all -- must
    raise a clear error naming the problem, not fail obscurely deeper in
    the encode loop.

    Builds the ancestor via the real train_autoencoder rather than
    hand-rolling a state_dict: a single-stream config is loaded back as
    the LEGACY flat Autoencoder (flat "encoder.*"/"decoder.*" keys), not
    MultiStreamAutoencoder ("encoders.shared.*"), so a hand-built
    two-stream-shaped state_dict fails with a state_dict key mismatch
    long before reaching the check this test is actually about.
    """
    from training.train_stage1 import train_autoencoder

    base_path = _build_sweep(tmp_path, n_runs=6, size=SIZE)
    stage1_path = train_autoencoder(
        size=SIZE, base_path=base_path, epochs=1, batch_size=4, base_channels=4,
        latent_channels=LATENT_CHANNELS, val_fraction=0.34, test_fraction=0.17,
        num_workers=0, augment=False, min_step=0, min_stdev_phi=None,
        stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "stage1_only.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "curve_s1_only.png",
    )
    # train_autoencoder already saves real test_dirs; no patching needed.

    with pytest.raises(ValueError, match="no 'deriv' stream"):
        check_deriv_temperature(stage2_checkpoint_path=stage1_path, min_step=0, device="cpu")


@pytest.mark.slow
def test_filtering_round_trips_from_the_checkpoints_own_data_config(tmp_path, isolated_project_root,
                                                                    monkeypatch, capsys):
    """
    REGRESSION: stage 2 saved NO data_config at all (only stage 3+ did),
    so this diagnostic -- which reads a stage-2 checkpoint -- had no way
    to reproduce the window population its checkpoint was trained on.
    Its own CLI defaults (min_step=None -> 0, min_stdev_phi=None) meant
    NO filtering, silently a much larger and different set than training
    ever saw.

    min_std_deriv is the sharpest case: applied ONLY in stage 2, saved
    nowhere, and re-applied by no diagnostic -- on real 64x64 data it
    discards tens of thousands of windows (33683 train / 13090 val in
    one real run), so omitting it was not a marginal difference.

    Asserts what the DATASET is actually constructed with, not just what
    gets printed -- the print could be right while the dataset is built
    from something else.
    """
    from training.train_stage1 import train_autoencoder
    from training.train_stage2 import train_stage2
    import evaluation.check_deriv_temperature as cdt

    base_path = _build_sweep(tmp_path, n_runs=6, size=SIZE)
    stage1_path = train_autoencoder(
        size=SIZE, base_path=base_path, epochs=1, batch_size=4, base_channels=4,
        latent_channels=LATENT_CHANNELS, val_fraction=0.34, test_fraction=0.17,
        num_workers=0, augment=False, min_step=0, min_stdev_phi=None,
        stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "s1_rt.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "c1_rt.png",
    )
    # Distinctive, non-default values: min_step=1000 (vs the diagnostic's
    # own default of 0) and min_std_deriv=0.0 (vs None). 0.0 is chosen
    # deliberately -- it is APPLIED (not None) yet excludes nothing,
    # since std < 0.0 is never true, so it round-trips visibly without
    # emptying this tiny synthetic dataset.
    stage2_path = train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, stats0_weight=0.01, epochs=1, batch_size=4, num_workers=0,
        min_step=1000, min_stdev_phi=None, min_std_deriv=0.0,  # noqa: E501
        checkpoint_path=tmp_path / "s2_rt.pt", device="cpu",
        log_every_epoch=False, loss_curve_path=tmp_path / "c2_rt.png",
    )

    saved = torch.load(stage2_path, map_location="cpu", weights_only=True)
    assert "data_config" in saved, "stage 2 must save a data_config at all"
    assert saved["data_config"]["min_step"] == 1000
    assert saved["data_config"]["min_std_deriv"] == 0.0

    seen = {}
    real_ds = cdt.MicrostructureEvolutionDataset

    def spy(*args, **kwargs):
        seen.update({k: kwargs.get(k) for k in
                     ("min_step", "min_stdev_phi", "min_passing_steps", "min_std_deriv")})
        return real_ds(*args, **kwargs)

    monkeypatch.setattr(cdt, "MicrostructureEvolutionDataset", spy)
    # Deliberately passes NO filter arguments -- the whole point is that
    # they come from the checkpoint.
    cdt.check_deriv_temperature(stage2_checkpoint_path=stage2_path, device="cpu")

    assert seen["min_step"] == 1000, (
        f"dataset built with min_step={seen['min_step']}, not the checkpoint's own 1000"
    )
    # min_std_deriv is NOT passed through: the dataset rejects it in
    # cached-latent mode (raw-pixel-only filter). It must instead be
    # SURFACED, so the population difference vs training isn't silent.
    assert seen["min_std_deriv"] is None
    printed = capsys.readouterr().out
    assert "cannot be reproduced here" in printed, (
        f"the un-reproducible min_std_deriv filter must be reported, not silently dropped. "
        f"Output:\n{printed}"
    )


@pytest.mark.slow
def test_explicitly_passed_filtering_still_overrides_the_checkpoint(tmp_path, isolated_project_root,
                                                                     monkeypatch):
    """The checkpoint is a DEFAULT, not a lock-in -- an explicit argument
    must still win, same convention as _latent_eval.py's own stage-3
    path ("from checkpoint's own data_config unless overridden above")."""
    from training.train_stage1 import train_autoencoder
    from training.train_stage2 import train_stage2
    import evaluation.check_deriv_temperature as cdt

    base_path = _build_sweep(tmp_path, n_runs=6, size=SIZE)
    stage1_path = train_autoencoder(
        size=SIZE, base_path=base_path, epochs=1, batch_size=4, base_channels=4,
        latent_channels=LATENT_CHANNELS, val_fraction=0.34, test_fraction=0.17,
        num_workers=0, augment=False, min_step=0, min_stdev_phi=None,
        stats0_weight=0.01, stat_names=["avg_phi"],
        checkpoint_path=tmp_path / "s1_ov.pt", device="cpu", seed=0,
        log_every_epoch=False, loss_curve_path=tmp_path / "c1_ov.png",
    )
    stage2_path = train_stage2(
        base_path=base_path, resume_from=stage1_path,
        deriv_weight=1.0, stats0_weight=0.01, epochs=1, batch_size=4, num_workers=0,
        min_step=1000, min_stdev_phi=None,
        checkpoint_path=tmp_path / "s2_ov.pt", device="cpu",
        log_every_epoch=False, loss_curve_path=tmp_path / "c2_ov.png",
    )

    seen = {}
    real_ds = cdt.MicrostructureEvolutionDataset

    def spy(*args, **kwargs):
        seen["min_step"] = kwargs.get("min_step")
        return real_ds(*args, **kwargs)

    monkeypatch.setattr(cdt, "MicrostructureEvolutionDataset", spy)
    cdt.check_deriv_temperature(stage2_checkpoint_path=stage2_path, min_step=0, device="cpu")
    assert seen["min_step"] == 0, "an explicit min_step must override the checkpoint's own 1000"
