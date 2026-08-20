"""
Integration tests for the evaluation/comparison scripts' model-
reconstruction logic -- the part that broke, repeatedly, across a long
session of real pipeline runs. Every one of these bugs was invisible
to unit tests of the underlying pure functions (resolve_stream_configs_
from_checkpoint_config, cross_check_stream_configs_against_state_dict,
etc. -- see test_latent_streams.py) because the bug was in HOW each
script WIRED those functions together, not in the functions
themselves. The only thing that actually catches a wiring bug is
calling the real function end-to-end against a real checkpoint --
which is what these tests do.

Two realistic checkpoint shapes get exercised throughout, matching the
two actual failure modes hit in production:

1. NON-DEFAULT latent_spatial_size (4, not the default 8) -- catches
   any reconstruction site that forgot to pass latent_spatial/
   latent_spatial_size at all (silently defaulting to 8 instead of
   reading the checkpoint's real value; the exact bug that broke
   LatentDynamics reconstruction in four separate files this session).

2. STALE multi-stream metadata (config claims one stream, encoder
   weights genuinely contain two) -- catches any reconstruction site
   that resolves stream_configs without also cross-checking against
   the real weights (the exact bug that broke model_assembly.py and
   several evaluation scripts this session).

A real Autoencoder/LatentDynamics is used throughout (not mocked) --
these tests call torch.save/torch.load and the real construction code,
so a wiring mistake shows up as an actual RuntimeError/NameError/
AttributeError, not something a mock could paper over.
"""
from pathlib import Path

import torch
import re

import pytest

from conftest import assert_figure_was_really_written

from models.autoencoder import Autoencoder, MultiStreamAutoencoder
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_dynamics import LatentDynamics
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from training.stats_head import StatsHead

from evaluation.check_latent_channels import check_latent_channels
from evaluation.check_reconstruction import check_reconstruction
from evaluation.check_interpolation import check_interpolation
from evaluation.check_perturbation import check_perturbation
from evaluation.check_rollout import check_rollout
from evaluation.check_parameter_dependence import check_parameter_dependence
from models.constants import N_THETA

pytestmark = [
    pytest.mark.filterwarnings("ignore:Polyfit may be poorly conditioned"),
    pytest.mark.filterwarnings("ignore:invalid value encountered in divide"),
    pytest.mark.filterwarnings("ignore:Data has no positive values, and therefore cannot be log-scaled"),
]


_NON_DEFAULT_SPATIAL = 4  # anything != models.constants.LATENT_SPATIAL_SIZE (8)


def _save_ae_checkpoint(path, run_dirs, size=64, base_channels=4, latent_channels=4,
                         latent_spatial_size=_NON_DEFAULT_SPATIAL, multi_stream=False,
                         stale_metadata=False, include_stats_head=True, stat_names=None):
    """
    Builds and saves a REAL Autoencoder (or MultiStreamAutoencoder, for
    multi_stream) checkpoint -- real weights, real state_dict, exactly
    what train_stage1.py/train_stage2.py actually produce. multi_stream + stale_metadata
    together reproduces the exact "self_heals" scenario: config claims
    only one stream, but the encoder's actual weights have two.
    """
    if multi_stream:
        stream_configs = {
            "state": LatentStreamConfig(name="state", channels=latent_channels,
                                         spatial_size=latent_spatial_size,
                                         mode=LatentStreamMode.AUTOENCODER),
            "deriv": LatentStreamConfig(name="deriv", channels=latent_channels,
                                         spatial_size=latent_spatial_size,
                                         mode=LatentStreamMode.DECODER),
        }
        encoder = Encoder(input_size=size, in_channels=1, base_channels=base_channels,
                           stream_configs=stream_configs)
        decoder = Decoder(output_size=size, out_channels=1, base_channels=base_channels,
                           latent_channels=latent_channels, latent_spatial_size=latent_spatial_size)
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                     stream_configs=stream_configs)
    else:
        ae = Autoencoder(size=size, channels=1, base_channels=base_channels,
                          latent_channels=latent_channels, latent_spatial_size=latent_spatial_size)

    config = {"size": size, "base_channels": base_channels, "latent_channels": latent_channels,
              "latent_spatial_size": latent_spatial_size, "stats_weight": 0.01}
    if not stale_metadata:
        streams_for_config = (
            {"state": {"channels": latent_channels, "spatial_size": latent_spatial_size,
                       "mode": "autoencoder"},
             "deriv": {"channels": latent_channels, "spatial_size": latent_spatial_size,
                       "mode": "decoder"}}
            if multi_stream else
            {"state": {"channels": latent_channels, "spatial_size": latent_spatial_size,
                       "mode": "autoencoder"}}
        )
        config["stream_configs"] = streams_for_config
        config["recon_stream_name"] = "state"
    # stale_metadata=True: deliberately omit stream_configs/recon_stream_name
    # entirely, even though multi_stream=True means the ACTUAL weights
    # (saved below) genuinely have both streams -- exactly the
    # intermediate-version-of-the-code checkpoint shape this session
    # hit for real.

    checkpoint = {
        "model_state": ae.state_dict(),
        "epoch": 5,
        "val_loss": 0.05,
        "val_loss_ema": 0.05,
        "test_dirs": [str(d) for d in run_dirs],
        "config": config,
    }
    if include_stats_head:
        names = stat_names or ["angle", "avg_phi"]
        stats_head = StatsHead(latent_channels=latent_channels, stat_names=names,
                                latent_spatial=latent_spatial_size)
        checkpoint["stats_head_state"] = stats_head.state_dict()
        checkpoint["stats_config"] = {
            "stat_names": names, "stats_mean": torch.zeros(len(names)),
            "stats_std": torch.ones(len(names)),
        }
    torch.save(checkpoint, path)
    return checkpoint


def _save_lds_checkpoint(path, ae_checkpoint_path, latent_channels=4,
                          latent_spatial_size=_NON_DEFAULT_SPATIAL, run_dirs=None):
    """Real LatentDynamics checkpoint, pointing at a real AE ancestor."""
    f_theta = LatentDynamics(latent_channels=latent_channels, n_theta=N_THETA,
                              latent_spatial=latent_spatial_size, hidden_dim=16, n_hidden_layers=1)
    checkpoint = {
        "model_state": f_theta.state_dict(),
        "epoch": 5,
        "val_loss": 0.05,
        "test_dirs": [str(d) for d in (run_dirs or [])],
        "ae_checkpoint": str(Path(ae_checkpoint_path).resolve()),
        "config": {"latent_channels": latent_channels, "n_theta": N_THETA,
                   "latent_spatial_size": latent_spatial_size, "hidden_dim": 16, "n_hidden_layers": 1},
    }
    torch.save(checkpoint, path)
    return checkpoint


# ---- check_latent_channels -------------------------------------------------
# THE function that actually had a real NameError (rank_channel_importance
# referencing recon_stream_name without receiving it) -- this is the most
# direct regression test for that specific bug: if the scoping bug came
# back, this would fail with NameError, not a normal assertion failure.

def test_check_latent_channels_non_default_spatial_size(tmp_path, tmp_run_dir):
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL)

    output_path = check_latent_channels(
        ae_checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert_figure_was_really_written(output_path)


@pytest.mark.filterwarnings("ignore:checkpoint's saved config only described streams")
def test_check_latent_channels_stale_multi_stream_metadata(tmp_path, tmp_run_dir):
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2-stale.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         multi_stream=True, stale_metadata=True)

    # Must not raise -- neither the old "unexpected keys" RuntimeError
    # (stale metadata undercounting streams) nor a NameError (the
    # scoping bug) -- both real failure modes this session hit.
    output_path = check_latent_channels(
        ae_checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert_figure_was_really_written(output_path)


# ---- check_reconstruction ---------------------------------------------------

def test_check_reconstruction_non_default_spatial_size(tmp_path, tmp_run_dir, capsys):
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL)

    output_path = check_reconstruction(
        checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert_figure_was_really_written(output_path)
    # A script that found NO data still writes its figure, so the blank check
    # above cannot see that case; the count it prints can.
    printed = capsys.readouterr().out
    counts = [int(n) for n in re.findall(r"(\d+) samples", printed)]
    assert counts and max(counts) > 0, (
        f"the script reported no samples -- it wrote a figure without processing "
        f"any data:\n{printed[-400:]}"
    )

@pytest.mark.filterwarnings("ignore:checkpoint's saved config only described streams")
def test_check_reconstruction_stale_multi_stream_metadata(tmp_path, tmp_run_dir, capsys):
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2-stale.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         multi_stream=True, stale_metadata=True)

    output_path = check_reconstruction(
        checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert_figure_was_really_written(output_path)


# ---- check_interpolation / check_perturbation -------------------------------
# Both need real statistics.csv data (stat_names must match real columns)
# -- tmp_run_dir_with_stats provides that.
    # A script that found NO data still writes its figure, so the blank check
    # above cannot see that case; the count it prints can.
    printed = capsys.readouterr().out
    counts = [int(n) for n in re.findall(r"(\d+) samples", printed)]
    assert counts and max(counts) > 0, (
        f"the script reported no samples -- it wrote a figure without processing "
        f"any data:\n{printed[-400:]}"
    )

def test_check_interpolation_non_default_spatial_size(tmp_path, tmp_run_dir_with_stats):
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    ae_path = tmp_path / "fake-stage2.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         stat_names=stat_names)

    output_path = check_interpolation(
        checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
    )
    assert_figure_was_really_written(output_path)


def test_check_perturbation_non_default_spatial_size(tmp_path, tmp_run_dir_with_stats):
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    ae_path = tmp_path / "fake-stage2.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         stat_names=stat_names)

    output_path = check_perturbation(
        checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
        n_samples=2, n_repeats=2,
    )
    assert_figure_was_really_written(output_path)


# ---- check_rollout / check_parameter_dependence -----------------------------
# Both need a real (AE, LDS) pair, LDS pointing back at the AE via
# "ae_checkpoint" -- the exact shape train_lds.py itself produces.

def test_check_rollout_non_default_spatial_size(tmp_path, tmp_run_dir, capsys):
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2.pt"
    lds_path = tmp_path / "fake-stage3b.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         include_stats_head=False, multi_stream=True)
    _save_lds_checkpoint(lds_path, ae_path, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                          run_dirs=[run_dir])

    output_path, windows = check_rollout(
        lds_checkpoint_path=lds_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
        n_samples=1,
    )
    assert_figure_was_really_written(output_path)
    # check_rollout RETURNS its windows, so assert on those rather than on
    # parsed console text: the figure is written unconditionally, so "it
    # exists" cannot distinguish a real run from one that found nothing.
    assert len(windows) > 0, "check_rollout wrote a figure without processing any window"

def test_check_parameter_dependence_non_default_spatial_size(tmp_path, tmp_run_dir_with_stats, capsys):
    run_dir, steps, stat_names = tmp_run_dir_with_stats
    # check_parameter_dependence specifically needs "autocorr_length"
    # for its length-scale panel -- the shared fixture's stat set
    # doesn't include it (it's a smaller, generic set other tests
    # rely on staying fixed), so it's added here, local to this test,
    # rather than changing the shared fixture for everyone.
    import pandas as pd
    stats_path = run_dir / "statistics.csv"
    df = pd.read_csv(stats_path)
    df["autocorr_length"] = 5.0
    df.to_csv(stats_path, index=False)

    ae_path = tmp_path / "fake-stage2.pt"
    lds_path = tmp_path / "fake-stage3b.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         include_stats_head=False, multi_stream=True)
    _save_lds_checkpoint(lds_path, ae_path, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                          run_dirs=[run_dir])

    output_path = check_parameter_dependence(
        lds_checkpoint_path=lds_path, device="cpu", min_step=0, output_path=tmp_path / "out.png", latent_cache_dir=tmp_path / "latent_cache"
    )
    assert_figure_was_really_written(output_path)
    # A script that found NO data still writes its figure, so the blank check
    # above cannot see that case; the count it prints can.
    printed = capsys.readouterr().out
    counts = [int(n) for n in re.findall(r"(\d+) windows", printed)]
    assert counts and max(counts) > 0, (
        f"the script reported no windows -- it wrote a figure without processing "
        f"any data:\n{printed[-400:]}"
    )
    # the end-of-run copy-pasteable SUMMARY block must appear and be POPULATED
    # by the real run (not just render when hand-fed): its headline fields come
    # from _summary_put calls scattered through the analysis, so an unreached
    # call site would show here as a missing row.
    assert "SUMMARY " in printed, "the summary block was not emitted"
    assert "fake-stage3b.pt" in printed, "summary missing the checkpoint name"
    assert "err_actual" in printed, "summary missing err_actual -- its _summary_put didn't run"
    assert "bias fraction" in printed, "summary missing bias fraction"

@pytest.mark.filterwarnings("ignore:checkpoint's saved config only described streams")
def test_check_rollout_stale_multi_stream_metadata(tmp_path, tmp_run_dir):
    """THE combination that broke model_assembly.py originally, exercised
    here for check_rollout.py's own (separate) reconstruction path."""
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2-stale.pt"
    lds_path = tmp_path / "fake-stage3b.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         multi_stream=True, stale_metadata=True, include_stats_head=False)
    _save_lds_checkpoint(lds_path, ae_path, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                          run_dirs=[run_dir])

    output_path, _windows = check_rollout(
        lds_checkpoint_path=lds_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
        n_samples=1,
    )
    assert_figure_was_really_written(output_path)


def test_check_reconstruction_min_stdev_phi_is_actually_enforced(tmp_path, tmp_run_dir_with_stats):
    """Regression test for a real, reported bug: min_stdev_phi was
    HARDCODED to None inside check_reconstruction's own dataset
    construction, with no parameter to override it at all -- so every
    stage's own reconstruction sanity check always drew from the FULL
    test set regardless of what min_stdev_phi was used during
    training, letting degenerate, fully-coarsened frames through
    unfiltered. tmp_run_dir_with_stats gives stdev_phi = step/1000.0
    (0.0 through 4.0 across its 5 steps) -- min_stdev_phi=100 should
    exclude every single window, and the old (bugged) code -- which
    silently ignored min_stdev_phi entirely -- would NOT raise here."""
    run_dir, steps, _ = tmp_run_dir_with_stats
    ae_path = tmp_path / "fake-stage2.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL)

    # Sanity check first: with no filtering, this must succeed (real
    # windows genuinely exist in the fixture).
    output_path = check_reconstruction(
        checkpoint_path=ae_path, device="cpu", min_step=0, min_stdev_phi=None,
        output_path=tmp_path / "out_unfiltered.png",
    )
    assert_figure_was_really_written(output_path)

    # The actual fix under test: a min_stdev_phi higher than every
    # stdev_phi value in the fixture must exclude every window,
    # raising -- not silently ignored.
    with pytest.raises(ValueError, match="No consecutive pairs found"):
        check_reconstruction(
            checkpoint_path=ae_path, device="cpu", min_step=0, min_stdev_phi=100.0,
            output_path=tmp_path / "out_filtered.png",
        )


def test_check_reconstruction_derivative_panel_uses_6_cols_with_symmetric_d0(tmp_path, tmp_run_dir):
    """Regression test for a real, reported design fix, in two stages:

    (1) The predicted-derivative panel used to be D1(z1) directly --
    meaningless checkerboard noise once Stage 2 actually runs, since D1
    is never trained past Stage 1b (Stage 2's default
    recon1_weight=0). Fixed to D0(z0+z1*dt) - D0(z0).

    (2) That first fix left real_deriv computed from RAW PIXELS
    ((x_next-x_t)/dt) while pred_deriv went entirely through D0 --
    different natural scales (D0's own reconstruction noise inflates
    one side but not the other), which visually squashed real_deriv
    nearly to blank under a shared color scale sized for pred_deriv's
    own larger range. Fixed by computing real_deriv AS
    D0(z0(t+dt))-D0(z0(t)) too -- z1 is STILL never sent to a decoder
    on either side, but now BOTH sides share the same D0
    reconstruction-error component, so they're on a comparable scale
    AND a real error panel between them is meaningful again (unlike
    the old raw-pixel-vs-D0-decode pairing, where a naive difference
    would have conflated z1's own predictive error with D0's own,
    separate reconstruction error). n_cols should be 6 now, not 5."""
    run_dir, steps = tmp_run_dir
    ae_path = tmp_path / "fake-stage2.pt"
    _save_ae_checkpoint(ae_path, [run_dir], size=64, latent_spatial_size=_NON_DEFAULT_SPATIAL,
                         multi_stream=True)

    captured = {}
    import matplotlib.pyplot as plt
    real_subplots = plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    import evaluation.check_reconstruction as check_reconstruction_module
    check_reconstruction_module.plt.subplots = spy_subplots
    try:
        output_path = check_reconstruction(
            checkpoint_path=ae_path, device="cpu", min_step=0, output_path=tmp_path / "out.png",
        )
    finally:
        check_reconstruction_module.plt.subplots = real_subplots

    assert_figure_was_really_written(output_path)
    axes = captured["axes"]
    assert axes.shape[1] == 6, f"expected 6 columns with a derivative panel, got {axes.shape[1]}"


def test_second_derivative_is_the_deriv_streams_own_difference():
    """
    d2z0/dt2 = (z1(t+dt) - z1(t)) / dt, from the encoder's deriv stream.

    WHY IT MATTERS. |dz0| saturates with dt, so dz0/dt falls as 1/dt BY
    CONSTRUCTION and the first-order term z1*dt is bounded -- that argues for
    no max_dt at all. The dt^2/2 term is multiplied by the SECOND derivative,
    so it is that curve, not dz0/dt, which can justify a limit.
    """
    import inspect

    from evaluation import _latent_eval
    src = inspect.getsource(_latent_eval._evaluate_windows)
    assert "z1_next = window1[:, 1]" in src, (
        "the second derivative is not taken from the deriv stream's own "
        "next frame"
    )
    # UNDIVIDED here: the figure divides by the GROUP's dt, matching how it
    # forms dz0/dt from dz0.
    assert "dz1_signed_batch = _per_sample_signed_mean(z1_next, z1_t)" in src
    assert "dz1_abs_batch = _per_sample_l1(z1_next, z1_t)" in src
    assert "_per_sample_l1(z1_next, z1_t) / dt" not in src


def test_second_derivative_arrays_are_numpy_like_the_others():
    """The plotting path indexes these with boolean masks; leaving them as
    Python lists raised 'only integer scalar arrays can be converted'."""
    import inspect

    from evaluation import _latent_eval
    src = inspect.getsource(_latent_eval._evaluate_windows)
    assert "dz1_signed = np.array(dz1_signed)" in src
    assert "dz1_abs = np.array(dz1_abs)" in src


def test_the_dz0_figure_has_a_third_row():
    import inspect
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_parameter_dependence.py")
    assert "plt.subplots(3, 2" in src, "the dz0 figure still has only two rows"
    assert '"d2z0/dt2", 2' in src


def test_the_ratio_row_divides_grouped_dz0_by_the_group_dt():
    """
    |mean dz0| / dt and mean|dz0| / dt -- the row above's two curves, each
    divided by the group's dt. NOT mean(dz0/dt): averaging per-window ratios
    is a different quantity whenever dt varies within a group, which it does
    across every t group.
    """
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_parameter_dependence.py")
    assert "t_signed_y = np.abs(_t_sig) / _t_dt_mean" in src
    assert "t_abs_y = t_abs_y / _t_dt_mean" in src
    assert 'signed_label = f"|mean {name}|"' in src
    assert 'abs_label = f"mean|{name}|"' in src


def test_the_ratio_row_uses_one_log_axis_not_a_twin_pair():
    """Both curves are positive and share a unit, so a twin pair on two
    linear scales would hide the ratio between them -- which is how much of
    the motion cancels across windows, the point of the row."""
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_parameter_dependence.py")
    assert "twin_t = ax_t if one_axis else ax_t.twinx()" in src
    assert "one_axis = True" in src, "not every panel uses the single axis"
    # and a log axis cannot take a symmetric-about-zero range, so the
    # alignment helper must be reached only on the twin-axis rows
    assert "left_ylim, right_ylim = _symmetric_left_zero_right_ylim(" in src
    # The dz0dt figure's per-row ratio panels use ONE log axis (one_axis=True)
    # rather than a twin pair; the symmetric-about-zero alignment (whose lower
    # bound is -inf on a log axis) must be reached only on the twin-axis
    # branch. Scope the check to _figure_dz0dt's own body -- the main figure
    # has its own, legitimately-twinned call that does not use one_axis.
    dz0dt_start = src.index("def _figure_dz0dt")
    dz0dt_end = src.index("\ndef ", dz0dt_start + 1)
    dz0dt_body = src[dz0dt_start:dz0dt_end]
    assert "one_axis = True" in dz0dt_body, "the dz0dt ratio row is not on a single axis"
    call_in_dz0dt = dz0dt_body.rindex("_symmetric_left_zero_right_ylim(")
    context = dz0dt_body[:call_in_dz0dt][-1200:]
    assert "if one_axis:" in context and "else:" in context, (
        "the symmetric-about-zero alignment is applied unconditionally in the "
        "dz0dt ratio row; on a log axis its lower bound is -inf"
    )


def test_axis_limits_are_data_driven_not_hardcoded():
    """
    The fixed left edges (10 for dt, 1000 for t) left two empty decades on
    every panel: min_stdev_phi drops the near-uniform early frames, so the
    first kept step is already ~1e5 and the smallest dt ~700. Empty decades
    make one decade of data look like a plateau.
    """
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_parameter_dependence.py")
    assert "_x_left = float(_positive_dts.min()) / 1.5 if len(_positive_dts) else 10" in src
    assert "_t_left = float(_positive_t.min()) / 1.5 if len(_positive_t) else 1000" in src
    assert "if dedimensionalize and len(_positive_dts)" not in src


def test_every_dz0_panel_uses_the_single_log_axis_form():
    """
    All six panels: |mean X| and mean|X| on ONE log axis. Not just the
    ratio row -- the same comparison (how much cancels across windows) is
    the point of every row, and a twin linear pair hides it everywhere.
    """
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_parameter_dependence.py")
    assert "one_axis = True" in src
    assert "t_signed_y, dt_signed_y = np.abs(_t_sig), np.abs(_dt_sig)" in src, (
        "the signed curve is not folded to its magnitude, so it cannot share "
        "a log axis"
    )


def test_legend_entries_are_not_duplicated_on_a_single_axis():
    """twin_t IS ax_t on these rows, so gathering handles from both listed
    every curve twice."""
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_parameter_dependence.py")
    assert "([], []) if twin_t is ax_t" in src
    assert "([], []) if twin_dt is ax_dt" in src


def test_dz0_panel_x_limits_come_from_the_plotted_curve():
    """
    A global minimum over all windows let one stray small-t window set the
    left edge, leaving two empty decades on a panel whose curve starts at
    ~1e5. Each panel is bounded by its own x values.
    """
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_parameter_dependence.py")
    assert "_finite_t = t_x[np.isfinite(t_x) & (t_x > 0)]" in src
    assert "_finite_dt = dt_x[np.isfinite(dt_x) & (dt_x > 0)]" in src
    assert "ax_t.set_xlim(left=_t_left)" not in src
    assert "ax_dt.set_xlim(left=_x_left)" not in src


def test_both_derivative_rows_use_the_same_reduction():
    """
    THE INCONSISTENCY. dz0/dt was formed by grouping dz0 and dividing by the
    GROUP's dt, while d2z0/dt2 was formed by dividing per window and then
    averaging -- mean|dz0|/mean(dt) against mean(dz1/dt). Those differ
    whenever dt varies within a group, which it does across every t group:
    on a group holding dt in {500, 1000, 2000, 4000} with a saturated
    increment, the two reductions differ by 1.76x.

    Two rows whose entire purpose is to be read against each other cannot be
    computed two different ways.
    """
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/check_parameter_dependence.py")
    assert 'if name in ("dz0/dt", "d2z0/dt2"):' in src, (
        "only one of the two derivative rows is divided by the group's dt"
    )
    # and the rows must feed on the RAW increments, not pre-divided arrays
    assert '(results.dz0_signed, results.dz0_abs, "dz0/dt", 1)' in src
    assert '(results.dz1_signed, results.dz1_abs, "d2z0/dt2", 2)' in src
    assert "results.dz0dt_signed, results.dz0dt_abs" not in src.split(
        "panel_rows = [")[1][:400]


def test_the_two_reductions_really_differ():
    """Guards the premise: if they agreed, the fix above would be cosmetic."""
    import numpy as np
    dt = np.array([500.0, 1000.0, 2000.0, 4000.0])
    dz0 = np.full(4, 0.13)
    per_window = float(np.mean(dz0 / dt))
    grouped = float(np.mean(dz0) / np.mean(dt))
    assert abs(per_window / grouped - 1.0) > 0.5


def test_dt_dependence_left_column_yrange_excludes_converged_regime():
    """
    The left-column (signed / |error|) y-range of the dt_dependence figure
    must be set from only the dt below the saturation cutoff -- the first dt
    where mean|dz0| drops below 1/3 of its max. Past that, the field has
    converged (dz0 -> 0), so error/dt explodes to meaningless values that,
    if allowed to set the range, crush the real signal into a thin band.
    The post-cutoff points are still plotted (off-chart) and x-range is
    unchanged; only the y-range computation excludes them.

    This tests the two pieces of that logic directly (they live as nested
    helpers inside check_parameter_dependence's plotting function, so they
    are reproduced here at the exact form used there).
    """
    import numpy as np

    def find_cutoff(dz0x, dz0abs):
        dz0abs = np.asarray(dz0abs, float)
        finite = np.isfinite(dz0abs)
        dz0max = float(np.nanmax(dz0abs[finite]))
        # anchor at the peak, take first sub-(max/3) bin to its RIGHT (the
        # convergence cliff), NOT the rising left edge which is also sub-1/3.
        peak = int(np.nanargmax(np.where(finite, dz0abs, -np.inf)))
        after = np.arange(len(dz0abs)) > peak
        below = np.where(finite & after & (dz0abs < dz0max / 3.0))[0]
        return float(np.asarray(dz0x)[below[0]]) if len(below) else float("inf")

    def ylim_below(xy, dt_cutoff, fallback):
        vals = []
        for xs, ys in xy:
            xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            m = np.isfinite(xs) & np.isfinite(ys) & (xs < dt_cutoff)
            vals.extend(ys[m].tolist())
        if not vals:
            return fallback
        lo, hi = min(vals), max(vals)
        if lo == hi:
            return fallback
        pad = 0.05 * (hi - lo)
        return lo - pad, hi + pad

    # mean|dz0|: RISES from a small value at short dt (12), climbs to a
    # plateau (~75), then CLIFFS at convergence. The rising left edge (12) is
    # below max/3=25 too -- the detector must NOT stop there.
    dt = np.array([7, 25, 100, 500, 2000, 5000, 1e4, 5e4])
    dz0 = np.array([12, 35, 75, 76, 70, 60, 1.2, 0.8])
    cutoff = find_cutoff(dt, dz0)
    assert cutoff == 1e4, (
        f"cutoff should be the convergence cliff (1e4), not the rising left "
        f"edge (7); got {cutoff}")

    # signed error: modest pre-cliff, explodes post-cliff (the meaningless part)
    # signed error, aligned with the 8-element dt above: small at the rising
    # edge and through the plateau, then explodes post-cliff (meaningless).
    err = np.array([0.02, 0.0, 0.1, 0.3, 0.5, 0.68, 500.0, 900.0])
    lo, hi = ylim_below([(dt, err)], cutoff, (-1.0, 1000.0))
    assert hi < 1.0, f"post-cliff explosion leaked into y-range (hi={hi})"
    assert lo < 0.1, f"pre-cliff minimum not captured (lo={lo})"

    # no convergence detected -> cutoff inf -> all data used (fallback semantics)
    lo2, hi2 = ylim_below([(dt, err)], float("inf"), (-5.0, 5.0))
    assert hi2 > 900, "an inf cutoff must include every point"

    # RANGE MUST SPAN ALL CURVES, not just euler-only: a second curve (causal)
    # dips below euler's floor, and the range has to reach it. Excluding large
    # dt, but excluding no curve.
    euler = np.array([0.10, 0.12, 0.15, 0.30, 0.50, 0.68, 500.0, 900.0])
    causal = np.array([0.01, 0.02, -0.05, 0.05, 0.03, 0.02, 400.0, 800.0])  # dips to -0.05, below euler
    lo_all, _ = ylim_below([(dt, euler), (dt, causal)], cutoff, (-1.0, 1000.0))
    lo_euler, _ = ylim_below([(dt, euler)], cutoff, (-1.0, 1000.0))
    assert lo_all < lo_euler, (
        f"range from all curves ({lo_all}) must reach below euler-only's floor "
        f"({lo_euler}) -- the causal curve dips lower and must not be clipped")
    # the -0.05 causal point (pre-cliff) must be inside the range
    assert lo_all <= -0.05 + 1e-9, "range does not reach the lowest pre-cliff causal point"
    # |error| PANEL (log magnitude) -- the cutoff governs a DIFFERENT bound
    # than on the signed panel. The converged tail here UNDERFLOWS (accurate
    # curves -> ~0 as the field stops moving), dragging y_MIN down; meanwhile
    # the euler curve CLIMBS at long dt and that rise is real. So:
    #   y_min: from sub-cutoff data only (underflow excluded)
    #   y_max: from ALL data (the long-dt climb kept)
    def abs_bounds(curves, dt_cutoff):
        # y_min from below-cutoff, with the real code's positivity guard: on a
        # log axis a non-positive padded lower edge is invalid, so fall back.
        lo, _ = ylim_below(curves, dt_cutoff, (1e-9, 1.0))
        if lo <= 0:
            # fall back to the smallest positive pre-cutoff value itself
            subvals = []
            for xs, ys in curves:
                xs = np.asarray(xs, float); ys = np.asarray(ys, float)
                m = np.isfinite(xs) & np.isfinite(ys) & (xs < dt_cutoff) & (ys > 0)
                subvals.extend(ys[m].tolist())
            lo = min(subvals) if subvals else 1e-9
        # y_max from all positive data
        allv = []
        for xs, ys in curves:
            ys = np.asarray(ys, float)
            allv.extend(ys[np.isfinite(ys) & (ys > 0)].tolist())
        hi = max(allv) * 1.1 if allv else 1.0
        return lo, hi

    dt_abs = np.array([7, 25, 100, 500, 2000, 5000, 1e4, 5e4])
    abs_cutoff = 1e4
    # euler: mid-range, CLIMBS at the largest dt (2.06) -- must set y_max
    euler_abs = np.array([2.2, 1.9, 1.3, 1.0, 1.1, 1.5, 1.8, 2.06])
    # causal: underflows in the converged tail (1e-4) -- must NOT set y_min
    causal_abs = np.array([0.9, 0.6, 0.3, 0.05, 0.01, 0.001, 1e-4, 5e-5])
    lo_ab, hi_ab = abs_bounds([(dt_abs, euler_abs), (dt_abs, causal_abs)], abs_cutoff)
    # y_max includes the long-dt euler climb (>= 2.06)
    assert hi_ab >= 2.06, f"y_max ({hi_ab}) must include the long-dt euler climb 2.06"
    # y_min excludes the converged-tail underflow (1e-4, 5e-5 at dt >= cutoff)
    assert lo_ab > 1e-4, f"y_min ({lo_ab}) must exclude the post-cutoff underflow (1e-4)"
    # but y_min still reaches the lowest PRE-cutoff causal point (0.001 at dt=5000)
    assert lo_ab <= 0.001 + 1e-6, f"y_min ({lo_ab}) must reach the pre-cutoff causal floor 0.001"


def test_figure_dz0dt_renders_standalone_after_extraction():
    """_figure_dz0dt was extracted verbatim from _build_and_save_figures (it
    is the fully-independent ground-truth dz0/dz0dt/d2z0dt2 figure). Lock that
    it is a real, independently-callable function producing a valid 3x2
    figure from just the results fields + grouping helper it receives -- so a
    future edit to the parent cannot silently re-inline or break it."""
    import numpy as np
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import evaluation.check_parameter_dependence as cpd

    class _R:
        pass
    r = _R()
    rng = np.random.default_rng(0)
    n = 400
    r.dts = np.geomspace(10, 5e4, n)
    r.abs_steps = np.geomspace(1e3, 2e6, n)
    r.dz0_signed = rng.normal(0, 0.05, n)
    r.dz0_abs = np.abs(rng.normal(0.05, 0.02, n))
    r.dz0dt_signed = r.dz0_signed / r.dts
    r.dz0dt_abs = r.dz0_abs / r.dts
    r.dz1_signed = rng.normal(0, 1e-4, n)        # the d2z0/dt2 row
    r.dz1_abs = np.abs(rng.normal(1e-4, 5e-5, n))

    def _grp(x_raw, y_signed, y_abs, run_dirs=None):
        return cpd._mean_curves_by_unique_value(x_raw, y_signed, y_abs)

    out = Path(tempfile.mkdtemp()) / "dz0dt.png"
    cpd._figure_dz0dt(r, r.abs_steps, r.dts, _grp, "t", "dt", "title", out)
    assert out.exists() and out.stat().st_size > 10000, \
        "_figure_dz0dt did not produce a valid figure"
