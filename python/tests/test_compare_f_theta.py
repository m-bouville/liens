"""
Tests for evaluation/compare_f_theta.py -- two checkpoints, identical
windows, six columns.

The tool exists because three A-vs-B judgements in a row stumbled on
population mismatches (each checkpoint's own config filtered different
windows, truncated to different horizons, over different dt ranges). So
what the tests pin is precisely the shared-ness: same steps to both
models, same z1_resync, shared color scales, no per-checkpoint
truncation.
"""
import numpy as np
import pytest
import torch

import evaluation.compare_f_theta as cf


@pytest.fixture(autouse=True)
def _never_write_outside_tmp(monkeypatch, tmp_path):
    """
    THE DEFAULT OUTPUT PATH IS THE REAL output/ DIRECTORY. Three tests here
    exercise the default filename on purpose, so they cannot pass
    output_path -- that would bypass the very logic under test. Without this
    they wrote 8x8 stub figures into the project's own
    output/rollout_check_png/, where they sat looking exactly like a real run
    that had somehow produced latent-sized images.

    Redirecting the anchor keeps the naming logic under test while sending
    the bytes to tmp_path.
    """
    monkeypatch.setattr(cf, "_PYTHON_ROOT", tmp_path / "python")


def _stub_models(monkeypatch, pred_noise=(0.05, 0.20)):
    calls = []

    def fake_compute_sample(run_dir, steps, ae, f_theta, ae_config, device,
                             z1_resync):
        calls.append({"steps": list(steps), "model": f_theta,
                       "z1_resync": z1_resync})
        rng = np.random.default_rng(len(calls))
        x_t = rng.normal(size=(16, 16))
        x_real = x_t + rng.normal(0, 0.1, (16, 16))
        noise = pred_noise[0] if f_theta == "A" else pred_noise[1]
        x_pred = x_real + noise * rng.normal(size=(16, 16))
        return x_t, x_real, x_pred, x_real, 500.0, [250.0, 250.0]

    monkeypatch.setattr(cf, "compute_sample", fake_compute_sample)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: {
        "path": p, "ck": {}, "config": {}, "ae": None, "ae_encoder": None,
        "ae_config": {}, "ae_path": "shared.pt", "f_theta": str(p)[-1],
        "prefix": "", "label": f"model{str(p)[-1]}"})
    return calls


def test_both_models_integrate_the_exact_same_steps(monkeypatch, tmp_path):
    """
    THE POINT OF THE TOOL. check_rollout truncates a fixed window to each
    checkpoint's own window_length -- correct for judging one checkpoint on
    its own terms, fatal for an A/B row, where it silently puts a 1-step
    prediction beside a 2-step one. Here both models must receive every step
    of every window, untruncated.
    """
    calls = _stub_models(monkeypatch)
    cf.compare_f_theta("A", "B", fixed_windows=["runX:100:200:300",
                                                  "runY:5:10:15"],
                        output_path=tmp_path / "f.png", device="cpu")
    assert len(calls) == 4
    assert calls[0]["steps"] == calls[1]["steps"] == [100, 200, 300]
    assert calls[2]["steps"] == calls[3]["steps"] == [5, 10, 15]


def test_z1_resync_is_forced_equal_for_both(monkeypatch, tmp_path):
    """One model resyncing while the other propagates is a different
    experiment per column; the flag applies to both or neither."""
    calls = _stub_models(monkeypatch)
    cf.compare_f_theta("A", "B", fixed_windows=["r:1:2:3"],
                        output_path=tmp_path / "f.png", device="cpu")
    assert all(c["z1_resync"] is False for c in calls)
    calls.clear()
    cf.compare_f_theta("A", "B", fixed_windows=["r:1:2:3"], z1_resync=True,
                        output_path=tmp_path / "f2.png", device="cpu")
    assert all(c["z1_resync"] is True for c in calls)


def test_mixed_length_fixed_windows_are_refused(monkeypatch, tmp_path):
    """A 2-step and a 3-step window in one figure is two horizons in one
    comparison -- exactly the mismatch the uploads had."""
    _stub_models(monkeypatch)
    with pytest.raises(ValueError):
        cf.compare_f_theta("A", "B", fixed_windows=["r:1:2:3", "q:1:2"],
                            output_path=tmp_path / "f.png", device="cpu")


def test_the_error_columns_share_one_scale_set_by_the_worse_model():
    """Which error panel is fuller is the figure's entire question; per-panel
    autoscaling would make both look equally full."""
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/compare_f_theta.py")
    assert "np.concatenate([per[\"a\"][\"error\"].ravel(), per[\"b\"][\"error\"].ravel()])" in src, (
        "the error scale is not derived from BOTH models' errors"
    )
    # and the prediction scale comes from real alone, so it flatters neither
    assert "_padded_bounds(real_delta, 1.0, symmetric=True)" in src


def test_the_shared_windows_and_summary_are_reported(monkeypatch, tmp_path, capsys):
    _stub_models(monkeypatch)
    _, ws = cf.compare_f_theta("A", "B", fixed_windows=["runX:100:200:300"],
                                output_path=tmp_path / "f.png", device="cpu")
    out = capsys.readouterr().out
    assert ws == ["runX:100:200:300"]
    assert "median end-to-end loss" in out
    assert "indicative only" in out, (
        "a 6-window median is presented without the caveat, inviting a "
        "conclusion the sample cannot support"
    )


def test_differing_ae_checkpoints_are_called_out(monkeypatch, tmp_path, capsys):
    """Each model decodes through its own AE; if those differ, part of the
    gap is the AEs' and the figure must say so."""
    calls = _stub_models(monkeypatch)

    def load(p, d):
        return {"path": p, "ck": {}, "config": {}, "ae": None,
                "ae_encoder": None, "ae_config": {},
                "ae_path": f"ae_{p}.pt", "f_theta": str(p)[-1],
                "prefix": "", "label": f"model{str(p)[-1]}"}

    monkeypatch.setattr(cf, "_load_model", load)
    cf.compare_f_theta("A", "B", fixed_windows=["r:1:2:3"],
                        output_path=tmp_path / "f.png", device="cpu")
    out = capsys.readouterr().out
    assert "DIFFERENT AEs" in out


def test_stems_parse_into_prefix_and_readable_stage_label():
    """'error A' forces the reader to keep a mapping the filename already
    knows. Timestamps stay in the label -- two checkpoints of the SAME stage
    must not both become 'stage 3b' -- and stems outside the naming scheme
    fall back whole rather than refusing the comparison."""
    from evaluation.compare_f_theta import _parse_stem
    assert _parse_stem("128x128-stage3a") == ("128x128", "stage 3a")
    assert _parse_stem("32x32-stage2") == ("32x32", "stage 2")
    assert _parse_stem("128x128-stage3b-20260808_13h29") == (
        "128x128", "stage 3b-20260808_13h29")
    assert _parse_stem("oddname") == ("", "oddname")


def test_the_seventh_column_is_b_minus_a_on_the_error_scale():
    """
    The B-A column equals error_B - error_A exactly (the shared real_delta
    cancels), so it belongs with the error columns and shares their scale --
    a seventh independent scale on a seven-column figure is one more number
    per row for the reader to hold. The direction must be B minus A,
    matching 'error = model - real'.
    """
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/compare_f_theta.py")
    assert 'diff = per["b"]["pred_delta"] - per["a"]["pred_delta"]' in src, (
        "the diff is not b - a; the sign convention breaks the analogy with "
        "error = model - real"
    )
    assert "f_lo, f_hi = e_lo, e_hi" in src, (
        "the diff column does not share the error scale"
    )
    assert "_padded_bounds(diff" not in src, (
        "the diff column still computes a scale of its own"
    )
    assert "plt.subplots(n_rows, 7" in src


def test_title_and_filename_use_the_parsed_names(monkeypatch, tmp_path):
    import numpy as np

    import evaluation.compare_f_theta as cf

    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        rng = np.random.default_rng(0)
        x_t = rng.normal(size=(8, 8))
        x_real = x_t + 0.1
        return x_t, x_real, x_real + 0.05, x_real, 500.0, [250.0]

    def load(p, d):
        pre, lab = cf._parse_stem(str(p))
        return {"path": p, "ck": {}, "config": {}, "ae": None,
                "ae_encoder": None, "ae_config": {}, "ae_path": "x",
                "f_theta": str(p), "prefix": pre, "label": lab}

    monkeypatch.setattr(cf, "compute_sample", fake)
    monkeypatch.setattr(cf, "_load_model", load)
    titles = []
    import matplotlib.figure
    real_suptitle = matplotlib.figure.Figure.suptitle

    def spy(self, t, *a, **k):
        titles.append(t)
        return real_suptitle(self, t, *a, **k)

    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", spy)
    out, _ = cf.compare_f_theta("128x128-stage3a", "128x128-stage3b",
                                 fixed_windows=["r:1:2"], device="cpu")
    # "r:1:2" is a 2-element window = ONE transition, hence the singular
    assert titles == ["128x128: stage 3a vs. stage 3b\n"
                       "1 chained step, z1 propagated"]
    # NO seed in the name for --fixed-windows (check_rollout's convention):
    # the seed played no part in selecting them, and a stamped seed would
    # suggest a rerun with another seed changes the windows.
    assert out.name == "128x128-stage3a_vs_stage3b-1step-propagated.png"


def _stats_stub(monkeypatch, corr_none_for=()):
    """
    compute_TRAJECTORY stub -- collect_stats reads whole trajectories now, so
    that it can report per-step series as well as the endpoint. Error grows
    with dt, and chosen runs return a frozen real trajectory (undefined
    correlation, as float16 storage produces at short dt).
    """
    seen = []

    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        is_a = str(f_theta).endswith("a")
        seen.append((str(run_dir), tuple(steps), is_a))
        rng = np.random.default_rng(abs(hash((str(run_dir), steps[0]))) % 9973)
        dt = 50.0 * (1 + steps[0] % 40)
        n = len(steps)
        x_t = rng.normal(size=(8, 8))
        if str(run_dir) in corr_none_for:
            real = [x_t.copy() for _ in range(n)]        # never changes
            pred = [x_t + 0.01 for _ in range(n)]        # constant delta
            return real, pred, [dt / max(n - 1, 1)] * (n - 1)
        real = [x_t + i * rng.normal(0, 0.1, (8, 8)) for i in range(n)]
        err = 0.03 if is_a else 0.03 + 0.0004 * dt
        pred = [f + err * rng.normal(size=(8, 8)) for f in real]
        return real, pred, [dt / max(n - 1, 1)] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)

    # The six-column image path still uses compute_sample; derive it from the
    # same trajectory so a test that exercises both cannot see them disagree.
    def fake_sample(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        real, pred, dts = fake(run_dir, steps, ae, f_theta, ae_config, device,
                                z1_resync)
        return real[0], real[-1], pred[-1], real[-1], sum(dts), dts

    monkeypatch.setattr(cf, "compute_sample", fake_sample)
    return seen


def _model(stem):
    prefix, label = cf._parse_stem(stem)
    return {"path": stem, "ck": {}, "config": {}, "ae": None, "ae_encoder": None,
            "ae_config": {}, "ae_path": "x", "f_theta": stem,
            "prefix": prefix, "label": label}


def test_statistics_use_the_same_windows_for_both_models(monkeypatch):
    """The verdict rests on these numbers, so the population identity that
    the image panels guarantee has to hold here too."""
    seen = _stats_stub(monkeypatch)
    windows = [(f"run{i}", [i, i + 1, i + 2]) for i in range(20)]
    stats = cf.collect_stats(_model("128x128-stage3a"), _model("128x128-stage3b"),
                              windows, "cpu", False)
    assert len(stats["dt"]) == 20
    assert len(seen) == 40
    a_windows = {(r, s) for r, s, is_a in seen if is_a}
    b_windows = {(r, s) for r, s, is_a in seen if not is_a}
    assert a_windows == b_windows, "the two models were given different windows"


def test_undefined_correlations_are_dropped_for_BOTH_models_and_counted(monkeypatch):
    """
    A quiet window's real dx has ~zero std, so correlation is undefined --
    the low-|z1| population the gradient profiler identified. Dropping such a
    window for one model only would leave the two curves describing different
    populations, which is the error this tool exists to prevent.
    """
    _stats_stub(monkeypatch, corr_none_for={"run3", "run7"})
    windows = [(f"run{i}", [i, i + 1, i + 2]) for i in range(20)]
    stats = cf.collect_stats(_model("128x128-stage3a"), _model("128x128-stage3b"),
                              windows, "cpu", False)
    assert stats["n_corr_undefined"] == 2
    for key in ("a", "b"):
        assert sum(c is None for c in stats[f"corr_{key}"]) == 2

    # And when only ONE model's correlation is undefined the window must
    # STILL be counted -- otherwise the two curves are drawn from different
    # populations. A one-sided check passes the test above, where both models
    # go undefined together, so it needs its own case.
    def only_b_undefined(run_dir, steps, ae, f_theta, ae_config, device,
                          z1_resync):
        rng = np.random.default_rng(1)
        x_t = rng.normal(size=(8, 8))
        real = [x_t, x_t + rng.normal(0, 0.1, (8, 8)),
                x_t + rng.normal(0, 0.2, (8, 8))]
        if not str(f_theta).endswith("a") and str(run_dir) == "run1":
            # B's predicted DELTA is constant, so its std is zero and the
            # correlation is undefined. (Predicting zeros would not do it:
            # the delta would be -x_0, which varies.)
            pred = [x_t, x_t + 0.5, x_t + 0.5]
        else:
            pred = [f + 0.03 for f in real]
        return real, pred, [50.0, 50.0]

    monkeypatch.setattr(cf, "compute_trajectory", only_b_undefined)
    stats = cf.collect_stats(_model("128x128-stage3a"), _model("128x128-stage3b"),
                              [("run0", [0, 1, 2]), ("run1", [1, 2, 3])],
                              "cpu", False)
    assert stats["n_corr_undefined"] == 1, (
        "a window undefined for B only was not counted, so the correlation "
        "panels would show B on fewer windows than A"
    )
    # the losses are still there: only the correlation is undefined
    assert all(np.isfinite(stats["loss_a"]))


def test_dt_bins_use_the_median_not_the_mean():
    """
    Losses span six decades on real data (1e-3 to 4e6). A mean over that is
    the largest sample restated -- one diverged decade-4 window would set
    every bin it landed in.
    """
    dt = np.full(101, 100.0)
    values = np.concatenate([np.ones(100), [1e6]])
    centres, med, lo, hi = cf._binned(dt, values, n_bins=1)
    assert med[0] == 1.0, f"bin statistic is {med[0]}, not the median"
    assert hi[0] < 10, "the quartile band is being dragged by the outlier"


def test_empty_dt_bins_are_skipped_not_plotted_as_zero():
    """A gap in the dt distribution must leave a gap, not a line to zero."""
    dt = np.concatenate([np.full(20, 10.0), np.full(20, 10000.0)])
    values = np.ones(40)
    centres, med, _, _ = cf._binned(dt, values, n_bins=8)
    assert len(centres) == len(med) < 8
    assert all(np.isfinite(med))


def test_the_stats_figure_has_six_panels_and_reports_the_window_count(monkeypatch, tmp_path):
    _stats_stub(monkeypatch)
    windows = [(f"run{i}", [i, i + 1, i + 2]) for i in range(30)]
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    stats = cf.collect_stats(a, b, windows, "cpu", False)

    import matplotlib.pyplot as plt
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    out = cf._stats_figure(stats, a, b, "t", tmp_path / "s.png")
    assert out.exists()
    assert captured["axes"].shape == (2, 3)
    titles = [captured["axes"][r, c].get_title()
              for r in range(2) for c in range(3)]
    assert any("loss distribution" in t for t in titles)
    assert any("correlation distribution" in t for t in titles)
    assert any("loss vs dt" in t for t in titles)
    assert any("correlation vs dt" in t for t in titles)


def test_n_stats_is_wired_and_off_by_default():
    import inspect
    import pathlib

    from conftest import source_without_comments
    sig = inspect.signature(cf.compare_f_theta).parameters
    assert "n_stats" in sig and sig["n_stats"].default == 0
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/compare_f_theta.py")
    assert "--n-stats" in src and "n_stats=args.n_stats" in src
    assert "collect_stats(a, b, stat_windows" in src


def test_one_colorbar_per_row_for_the_three_columns_that_share_a_scale(monkeypatch, tmp_path):
    """
    Columns 4, 5 and 6 (error A, error B, B-A) share one scale, so a second
    bar repeating the same range is clutter on an already-wide figure. There
    must be exactly one colorbar per row.
    """
    import matplotlib.pyplot as plt
    _stub_models(monkeypatch)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("fig", fig)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf.compare_f_theta("128x128-stage3a", "128x128-stage3b",
                        fixed_windows=["r:1:2:3", "q:4:5:6"],
                        output_path=tmp_path / "f.png", device="cpu")
    fig, axes = captured["fig"], captured["axes"]
    n_rows = axes.shape[0]
    colorbars = [ax for ax in fig.axes if not ax.get_images()]
    assert len(colorbars) == n_rows, (
        f"{len(colorbars)} colorbars for {n_rows} rows -- the shared scale is "
        f"being drawn more than once per row"
    )
    # and the three really do share it, which is what makes one bar enough
    clims = [axes[0, c].get_images()[0].get_clim() for c in (4, 5, 6)]
    assert clims[0] == clims[1] == clims[2]


def test_the_stats_figure_is_titled_with_the_model_names(monkeypatch, tmp_path):
    """
    SHADOWING BUG. The cell loop unpacked into a variable named `title`,
    rebinding the figure title, so the statistics figure came out headed
    "stage 3b - stage 3a (= error stage 3b - error stage 3a)" -- the last
    image panel's caption -- instead of "128x128: stage 3a vs. stage 3b".
    """
    _stats_stub(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    titles = []
    import matplotlib.figure
    real_suptitle = matplotlib.figure.Figure.suptitle

    def spy(self, t, *args, **kwargs):
        titles.append(t)
        return real_suptitle(self, t, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", spy)
    monkeypatch.setattr(cf, "_select_windows",
                         lambda *a, **k: [(f"run{i}", [i, i + 1, i + 2])
                                          for i in range(12)])
    cf.compare_f_theta("128x128-stage3a", "128x128-stage3b",
                        fixed_windows=["r:1:2:3"], n_stats=12,
                        output_path=tmp_path / "f.png", device="cpu")
    assert len(titles) == 2, "expected one title per figure"
    for t in titles:
        assert t.startswith("128x128: stage 3a vs. stage 3b\n"
                             "2 chained steps, z1 propagated"), (
            f"figure titled {t!r} -- a panel caption has leaked into it"
        )


def test_undefined_correlations_do_not_become_zero():
    """nan, not 0: a quiet window with undefined correlation is not a window
    with no skill, and plotting it at zero would drag every median down."""
    from evaluation.compare_f_theta import _binned
    dt = np.array([100.0, 100.0, 100.0])
    values = np.array([90.0, np.nan, 92.0])
    centres, med, _, _ = _binned(dt, values, n_bins=1)
    assert med[0] == 91.0, f"median {med[0]} -- the undefined window counted"


def test_the_regime_is_in_the_title_and_the_filename(monkeypatch, tmp_path):
    """
    --steps 4 without --z1-resync and --steps 2 with it are DIFFERENT
    EXPERIMENTS that gave opposite readings of the same two checkpoints.
    Nothing on the figure said which was which, and worse, they shared a
    filename: the second run silently overwrote the first.
    """
    _stats_stub(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    titles = []
    import matplotlib.figure
    real_suptitle = matplotlib.figure.Figure.suptitle

    def spy(self, t, *args, **kwargs):
        titles.append(t)
        return real_suptitle(self, t, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", spy)

    two, _ = cf.compare_f_theta("128x128-stage3a", "128x128-stage3b",
                                 fixed_windows=["r:1:2:3"], device="cpu")
    assert "2 chained steps, z1 propagated" in titles[-1]

    four, _ = cf.compare_f_theta("128x128-stage3a", "128x128-stage3b",
                                  fixed_windows=["r:1:2:3:4:5"],
                                  z1_resync=True, device="cpu")
    assert "4 chained steps, z1 resynced at each real frame" in titles[-1]

    assert two.name != four.name, (
        "two different experiments write to the same file; one overwrites "
        "the other"
    )
    assert "2steps-propagated" in two.name and "4steps-resync" in four.name


def _traj_stub(monkeypatch, b_scale=3.0):
    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        n = len(steps)
        rng = np.random.default_rng(0)
        real = [rng.normal(size=(8, 8)) for _ in range(n)]
        scale = 1.0 if str(f_theta).endswith("a") else b_scale
        pred = [real[0]] + [real[i] * scale for i in range(1, n)]
        return real, pred, [250.0] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)


def test_the_trajectory_panel_is_three_rows_by_N_plus_one_columns(monkeypatch, tmp_path):
    """rows real / A / B, columns t, t+dt, t+2dt, ... -- so a divergence can
    be located in TIME, not only measured at the end."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    _traj_stub(monkeypatch)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._trajectory_figure(Path("runX"), [0, 10, 20, 30, 40],
                           _model("128x128-stage3a"), _model("128x128-stage3b"),
                           "cpu", False, "T", tmp_path / "tr.png")
    axes = captured["axes"]
    assert axes.shape == (3, 5), f"grid is {axes.shape}, expected 3 x 5"
    assert [axes[r, 0].get_ylabel() for r in range(3)] == [
        "real", "stage 3a", "stage 3b"]
    assert [axes[0, c].get_title() for c in range(5)] == [
        "t", "t + 250", "t + 500", "t + 750", "t + 1000"]


def test_every_trajectory_panel_shares_one_scale(monkeypatch, tmp_path):
    """
    The states are the same physical field at every frame, so a per-panel
    scale would renormalise away exactly the amplitude blow-up that marks a
    model leaving the decoder's manifold -- the thing this panel exists to
    show. B's frames are 3x here and must LOOK 3x.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    _traj_stub(monkeypatch, b_scale=3.0)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._trajectory_figure(Path("runX"), [0, 10, 20], _model("128x128-stage3a"),
                           _model("128x128-stage3b"), "cpu", False, "T",
                           tmp_path / "tr.png")
    axes = captured["axes"]
    clims = {axes[r, c].get_images()[0].get_clim()
             for r in range(3) for c in range(3)}
    assert len(clims) == 1, f"{len(clims)} different scales across the panels"

    # And the scale comes from the REAL row alone. Taking it from all rows
    # would let B's 3x blow-up set the range, renormalising away the very
    # thing the panel is meant to reveal: B would look normal and the REAL
    # row would look washed out.
    real = [np.random.default_rng(0).normal(size=(8, 8)) for _ in range(3)]
    rng = np.random.default_rng(0)
    real = [rng.normal(size=(8, 8)) for _ in range(3)]
    expected = cf._padded_bounds(np.concatenate([f.ravel() for f in real]),
                                  1.0, symmetric=True)
    assert clims.pop() == expected, (
        "the colour range is not the real row's own -- a model's blow-up is "
        "setting the scale"
    )


def test_the_prediction_rows_start_from_the_AE_RECONSTRUCTION(monkeypatch):
    """
    Column 0 of a model row is the decoded z0 of the starting state, not the
    raw snapshot: that is where the model actually begins, so decoder error
    shows in the column it belongs to instead of being absorbed into step 1.
    """
    import inspect
    src = inspect.getsource(cf.compute_trajectory)
    assert "pred_frames = [ae_decoder(z0_t)[0, 0].cpu().numpy()]" in src, (
        "the prediction row does not start from the AE reconstruction"
    )


def test_trajectory_is_off_by_default_and_wired(monkeypatch):
    import inspect
    import pathlib

    from conftest import source_without_comments
    sig = inspect.signature(cf.compare_f_theta).parameters
    assert "trajectory" in sig and sig["trajectory"].default is False
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/compare_f_theta.py")
    assert "--trajectory" in src and "trajectory=args.trajectory" in src
    assert "_trajectory_figure(" in src


def test_compute_trajectory_decodes_EVERY_frame(monkeypatch):
    """
    len(steps) frames out, not just the endpoint. rollout() already computes
    every intermediate z0; decoding only the last would give a panel with
    nothing between start and finish -- exactly what compute_sample already
    provides and what this function exists to go beyond.
    """
    from pathlib import Path
    import types

    n_steps = 5
    decoded = []

    class FakeMeta:
        dt, temperature, T0 = 1.0, 900.0, 800.0

    monkeypatch.setattr(cf.load, "read_metadata", lambda p: FakeMeta())
    monkeypatch.setattr(cf.load, "snapshot_filename", lambda s: f"s{s}")
    monkeypatch.setattr(cf.load, "read_phi_half",
                         lambda p, nx, ny: np.zeros((nx, ny), dtype=np.float32))
    monkeypatch.setattr(cf, "resolve_stream_configs_from_checkpoint_config",
                         lambda cfg: (None, "recon"))

    def encoder(x, theta=None):
        n = x.shape[0]
        return {"recon": torch.zeros(n, 2, 4, 4), "deriv": torch.zeros(n, 2, 4, 4)}

    def decoder(z):
        decoded.append(z)
        return torch.zeros(z.shape[0], 1, 8, 8)

    ae = types.SimpleNamespace(encoder=encoder, decoder=decoder)
    f_theta = types.SimpleNamespace(
        rollout=lambda z0, z1, dts, theta, z1_resync: torch.zeros(
            1, dts.shape[1], 2, 4, 4))

    real, pred, dt_per_step = cf.compute_trajectory(
        Path("run"), list(range(n_steps)), ae, f_theta, {"size": 8}, "cpu")
    assert len(real) == n_steps
    assert len(pred) == n_steps, (
        f"{len(pred)} predicted frames for {n_steps} columns -- the "
        f"intermediate states are not being decoded"
    )
    assert len(dt_per_step) == n_steps - 1
    # one decode for the start + one per transition
    assert len(decoded) == n_steps


def test_a_trajectory_figure_is_written_for_EVERY_window(monkeypatch, tmp_path):
    """One per row of the comparison figure, not just the first: a collapse
    that happens at frame 3 in one run and frame 2 in another is exactly the
    variation worth seeing, and it is invisible from a single window."""
    _stub_models(monkeypatch)
    _traj_stub(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    out, _ = cf.compare_f_theta(
        "128x128-stage3a", "128x128-stage3b",
        fixed_windows=["T925_n035_s5:1:2:3", "T750_n015_s401:1:2:3"],
        trajectory=True, device="cpu")
    written = sorted(p.name for p in out.parent.iterdir())
    assert len(written) == 3, f"expected 2 trajectories + 1 comparison, got {written}"


def test_trajectory_filenames_carry_T_noise_and_seed(monkeypatch, tmp_path):
    """
    The run directory name IS the parameters -- T925_n035_s5 is temperature,
    noise and sim seed -- so a single-window figure should be identified by
    them. The SELECTION seed chose the window set, not the physics, and has
    no place in the name of one window's plot.
    """
    _stub_models(monkeypatch)
    _traj_stub(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    out, _ = cf.compare_f_theta(
        "128x128-stage3a", "128x128-stage3b",
        fixed_windows=["T925_n035_s5:1:2:3"], trajectory=True, device="cpu")
    traj = [p.name for p in out.parent.iterdir() if p.name != out.name]
    assert len(traj) == 1
    assert "T925_n035_s5" in traj[0], f"{traj[0]} does not name its run"

    # And on the SEEDED path, where a "-seedN" suffix actually exists to be
    # stripped -- with --fixed-windows there is none, so that route cannot
    # detect the seed leaking through.
    # Path, not str: that is what the real _select_windows returns, and the
    # figure code reads run_dir.name.
    from pathlib import Path as _Path
    monkeypatch.setattr(cf, "_select_windows",
                         lambda *a, **k: [(_Path("T800_n020_s7"), [1, 2, 3])])
    out2, _ = cf.compare_f_theta("128x128-stage3a", "128x128-stage3b",
                                  n_samples=1, n_steps=2, seed=4,
                                  trajectory=True, device="cpu")
    assert "-seed4" in out2.name, "fixture no longer exercises the seeded path"
    traj2 = [p.name for p in out2.parent.iterdir()
             if "T800_n020_s7" in p.name]
    assert len(traj2) == 1
    assert "-seed" not in traj2[0], (
        f"{traj2[0]} carries the selection seed, which chose the window set "
        f"rather than the physics"
    )
    # the regime still distinguishes experiments on the same window
    assert "4steps" not in traj[0] and "2steps" in traj[0]
    assert "propagated" in traj[0]


def _traj_stub_collapsing(monkeypatch, collapse_at=3):
    """A tracks the real trajectory; B collapses to noise from `collapse_at`."""
    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        n = len(steps)
        rng = np.random.default_rng(0)
        base = rng.normal(size=(8, 8))
        real = [base + 0.1 * i * rng.normal(size=(8, 8)) for i in range(n)]
        if str(f_theta).endswith("a"):
            pred = [f + 0.01 * rng.normal(size=(8, 8)) for f in real]
        else:
            pred = [real[i] if i < collapse_at
                    else rng.normal(size=(8, 8)) * 3 for i in range(n)]
        return real, pred, [250.0] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)


def _titles(monkeypatch, n_frames=5, collapse_at=3):
    import matplotlib.pyplot as plt
    from pathlib import Path
    _traj_stub_collapsing(monkeypatch, collapse_at)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    import tempfile
    cf._trajectory_figure(Path("T925_n020_s79"), list(range(n_frames)),
                           _model("128x128-stage3a"), _model("128x128-stage3b"),
                           "cpu", False, "T",
                           Path(tempfile.mkdtemp()) / "t.png")
    return captured["axes"]


def test_every_model_panel_carries_its_own_loss_and_correlation(monkeypatch):
    """So the frame at which each model collapses can be READ OFF rather than
    eyeballed -- the whole point of the panel is locating the break in time."""
    axes = _titles(monkeypatch)
    for row in (1, 2):
        for col in range(5):
            t = axes[row, col].get_title()
            assert "loss=" in t and "corr=" in t, f"panel [{row},{col}]: {t!r}"
    # the real row keeps the time header, not metrics
    assert axes[0, 1].get_title() == "t + 250"


def test_the_metrics_show_the_collapse_frame(monkeypatch):
    """B is exact until frame 3 and noise after: its numbers must say so."""
    axes = _titles(monkeypatch, collapse_at=3)
    b_corr = [axes[2, c].get_title().split("corr=")[1] for c in range(5)]
    assert b_corr[1] == "100%" and b_corr[2] == "100%"
    assert b_corr[3] not in ("100%", "99%"), (
        f"frame 3 reports corr={b_corr[3]} for a collapsed prediction"
    )
    a_corr = [axes[1, c].get_title().split("corr=")[1] for c in range(5)]
    assert a_corr[3] in ("99%", "100%"), "A is tracking and should say so"


def test_each_model_measures_its_delta_from_ITS_OWN_start(monkeypatch):
    """
    Subtracting A's starting reconstruction from B's frames would fold the
    difference between the two AEs into B's correlation at every frame. Here
    B equals the real trajectory plus a constant offset, so its correlation
    must be 100% throughout -- it is only wrong if the wrong baseline is used.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import tempfile

    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        n = len(steps)
        rng = np.random.default_rng(0)
        real = [rng.normal(size=(8, 8)) * (1 + 0.1 * i) for i in range(n)]
        offset = 0.0 if str(f_theta).endswith("a") else 5.0
        return real, [f + offset for f in real], [250.0] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._trajectory_figure(Path("r"), list(range(4)), _model("128x128-stage3a"),
                           _model("128x128-stage3b"), "cpu", False, "T",
                           Path(tempfile.mkdtemp()) / "t.png")
    axes = captured["axes"]
    for col in (1, 2, 3):
        assert axes[2, col].get_title().split("corr=")[1] == "100%", (
            f"frame {col}: B tracks the real trajectory exactly up to a "
            f"constant, so its correlation is only below 100% if the baseline "
            f"subtracted is not B's own start"
        )


# (superseded by test_frame_zero_reports_reconstruction_fidelity_not_na:
#  frame 0 now reports the AE reconstruction fidelity, which is the
#  meaningful number there, rather than n/a)


def test_frame_zero_reports_reconstruction_fidelity_not_na(monkeypatch):
    """
    At frame 0 there is no delta yet, so the meaningful correlation is
    between the STATES -- which is exactly the AE reconstruction fidelity,
    and the baseline every later frame is measured against. Reporting n/a
    wasted the one column that says how good the starting point was.
    """
    axes = _titles(monkeypatch)
    for row in (1, 2):
        corr = axes[row, 0].get_title().split("corr=")[1]
        assert corr != "n/a", f"frame 0 of row {row} still reports n/a"
        assert corr.endswith("%")


def test_a_constant_real_delta_still_reports_na(monkeypatch):
    """
    Snapshots are stored as float16, so at a short dt and a low temperature
    the real state can be UNCHANGED at storage precision -- the real delta is
    then exactly constant and the correlation genuinely undefined. That n/a
    is a fact about the data and must survive.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import tempfile

    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(8, 8))
        # frame 1 is IDENTICAL to frame 0 in the real trajectory
        real = [base, base.copy(), base + rng.normal(0, 0.3, (8, 8))]
        pred = [f + 0.01 * rng.normal(size=(8, 8)) for f in real]
        return real, pred, [125.0, 125.0]

    monkeypatch.setattr(cf, "compute_trajectory", fake)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._trajectory_figure(Path("r"), [0, 1, 2], _model("128x128-stage3a"),
                           _model("128x128-stage3b"), "cpu", False, "T",
                           Path(tempfile.mkdtemp()) / "t.png")
    axes = captured["axes"]
    assert axes[1, 1].get_title().endswith("corr=n/a"), (
        "an unchanged real state must give an undefined correlation, not a "
        "fabricated number"
    )
    assert axes[1, 2].get_title().split("corr=")[1] != "n/a"


def test_stats_collect_per_step_series(monkeypatch):
    """Index k holds every window's value after k chained applications --
    the axis the collapse actually lives on."""
    _traj_stub_collapsing(monkeypatch, collapse_at=3)
    windows = [(f"run{i}", list(range(6))) for i in range(8)]
    stats = cf.collect_stats(_model("128x128-stage3a"), _model("128x128-stage3b"),
                              windows, "cpu", False)
    for key in ("a", "b"):
        series = stats[f"step_loss_{key}"]
        assert len(series) == 8 and all(len(r) == 6 for r in series)
        assert len(stats[f"step_corr_{key}"]) == 8

    # B collapses at 3, so its per-step loss must jump there and A's must not
    b_med = [np.median([r[k] for r in stats["step_loss_b"]]) for k in range(6)]
    a_med = [np.median([r[k] for r in stats["step_loss_a"]]) for k in range(6)]
    assert b_med[3] > 100 * b_med[2], "the collapse is not visible per step"
    assert a_med[3] < 10 * a_med[2]


def test_the_endpoint_agrees_with_the_last_per_step_value(monkeypatch):
    """Both come from ONE trajectory, so the two views of the same run cannot
    disagree -- computing them separately would let them drift apart."""
    _traj_stub_collapsing(monkeypatch)
    windows = [("run0", list(range(5)))]
    stats = cf.collect_stats(_model("128x128-stage3a"), _model("128x128-stage3b"),
                              windows, "cpu", False)
    for key in ("a", "b"):
        assert stats[f"loss_{key}"][0] == stats[f"step_loss_{key}"][0][-1]
        assert stats[f"corr_{key}"][0] == stats[f"step_corr_{key}"][0][-1]


def test_the_stats_figure_has_a_steps_column(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    _traj_stub_collapsing(monkeypatch)
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    windows = [(f"run{i}", list(range(5))) for i in range(10)]
    stats = cf.collect_stats(a, b, windows, "cpu", False)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._stats_figure(stats, a, b, "T", tmp_path / "s.png")
    axes = captured["axes"]
    assert axes.shape == (2, 3), f"grid is {axes.shape}, expected 2 x 3"
    assert "number of steps" in axes[0, 2].get_title()
    assert "number of steps" in axes[1, 2].get_title()
    assert axes[0, 2].get_xlabel() == "chained steps applied"


def test_n_samples_zero_runs_statistics_without_the_panel(monkeypatch, tmp_path):
    """
    "Statistics only" is a legitimate request -- the panel is for looking at a
    few windows, the statistics are what a verdict rests on, and at
    --n-stats 1000 the panel is just 1000x slower to no purpose. It used to
    fail outright.
    """
    from pathlib import Path
    _stats_stub(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    monkeypatch.setattr(
        cf, "_select_windows",
        lambda m, n, ns, seed, mx, dev: [(Path(f"T{500 + i}_n020_s{i}"),
                                          list(range(ns + 1)))
                                         for i in range(max(n, 1))])
    out, _ = cf.compare_f_theta("128x128-stage3a", "128x128-stage3b",
                                 n_samples=0, n_steps=4, seed=2, n_stats=8,
                                 device="cpu")
    written = sorted(p.name for p in out.parent.iterdir())
    assert written == [out.stem + "-stats.png"], (
        f"expected only the stats figure, got {written}"
    )


def test_n_samples_zero_still_honours_trajectory(monkeypatch, tmp_path):
    """The horizon still has to come from somewhere, so one window is drawn
    even at n_samples=0 -- and --trajectory can follow it."""
    from pathlib import Path
    _stats_stub(monkeypatch)
    _traj_stub_collapsing(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    monkeypatch.setattr(
        cf, "_select_windows",
        lambda m, n, ns, seed, mx, dev: [(Path("T925_n020_s79"),
                                          list(range(ns + 1)))])
    out, _ = cf.compare_f_theta("128x128-stage3a", "128x128-stage3b",
                                 n_samples=0, n_steps=4, seed=2,
                                 trajectory=True, device="cpu")
    written = sorted(p.name for p in out.parent.iterdir())
    assert any("T925_n020_s79" in n for n in written)
    assert not any(n == out.name for n in written), "a panel was drawn anyway"


def test_a_wholly_undefined_step_is_reported(monkeypatch, capsys):
    """
    A step index missing from the correlation panel means EVERY window was
    undefined there, for both models. No property of a single window explains
    that, so it must be said out loud rather than left as a gap in a curve.
    """
    from pathlib import Path

    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        rng = np.random.default_rng(abs(hash(str(run_dir))) % 997)
        n = len(steps)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * i) for i in range(n)]
        pred = [f + 0.01 for f in real]
        pred[1] = pred[0]            # constant delta at k=1, EVERY window
        return real, pred, [250.0] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    monkeypatch.setattr(
        cf, "_select_windows",
        lambda m, n, ns, seed, mx, dev: [(Path(f"r{i}"), list(range(ns + 1)))
                                         for i in range(max(n, 1))])
    cf.compare_f_theta("128x128-stage3a", "128x128-stage3b", n_samples=0,
                        n_steps=4, n_stats=6, device="cpu")
    out = capsys.readouterr().out
    assert "UNDEFINED for EVERY window at step(s) [1]" in out, out[-400:]


def test_fixed_windows_draw_the_panel_even_at_n_samples_zero(monkeypatch):
    """n_samples is how MANY to choose; --fixed-windows says exactly WHICH.
    Naming windows explicitly and getting no panel would be surprising."""
    _stats_stub(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    out, _ = cf.compare_f_theta("128x128-stage3a", "128x128-stage3b",
                                 n_samples=0, fixed_windows=["r:1:2:3"],
                                 device="cpu")
    assert out.exists(), "an explicitly named window produced no panel"


def test_a_partly_undefined_step_is_NOT_flagged(monkeypatch, capsys):
    """
    Only a step undefined for EVERY window is anomalous. Quiet windows give
    scattered undefined correlations all the time -- flagging those would
    bury the one message that matters in noise.
    """
    from pathlib import Path

    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        rng = np.random.default_rng(abs(hash(str(run_dir))) % 997)
        n = len(steps)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * i) for i in range(n)]
        pred = [f + 0.01 for f in real]
        if str(run_dir) == "r0":          # ONE window only
            pred[1] = pred[0]
        return real, pred, [250.0] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    monkeypatch.setattr(
        cf, "_select_windows",
        lambda m, n, ns, seed, mx, dev: [(Path(f"r{i}"), list(range(ns + 1)))
                                         for i in range(6)])
    cf.compare_f_theta("128x128-stage3a", "128x128-stage3b", n_samples=0,
                        n_steps=4, n_stats=6, device="cpu")
    out = capsys.readouterr().out
    assert "UNDEFINED for EVERY window" not in out


def _stats_for_figure(monkeypatch, blow_up_every=10, max_dt_a=2000.0,
                       max_dt_b=1000.0):
    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        n = len(steps)
        i = abs(hash(str(run_dir))) % 997
        rng = np.random.default_rng(i)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * k) for k in range(n)]
        blow = blow_up_every and (i % blow_up_every == 0)
        pred = [real[k] + (1e15 if (blow and k > 2) else 0.02 * k)
                for k in range(n)]
        return real, pred, [250.0 * (1 + i % 4)] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    a["max_dt"], b["max_dt"] = max_dt_a, max_dt_b
    windows = [(f"T{500 + i}_n020_s{i}", list(range(9))) for i in range(40)]
    return cf.collect_stats(a, b, windows, "cpu", False), a, b


def _figure_axes(monkeypatch, stats, a, b, tmp_path):
    import matplotlib.pyplot as plt
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._stats_figure(stats, a, b, "T", tmp_path / "s.png")
    return captured["axes"]


def test_max_dt_is_marked_on_both_dt_panels(monkeypatch, tmp_path):
    """Each model's own training limit, in its own colour, on every panel
    whose x axis is dt."""
    stats, a, b = _stats_for_figure(monkeypatch)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    for panel in (axes[1, 0], axes[1, 1]):
        vlines = sorted(line.get_xdata()[0] for line in panel.get_lines()
                        if len(set(line.get_xdata())) == 1)
        assert vlines == [1000.0, 2000.0], (
            f"panel marks {vlines}, expected both models' max_dt"
        )
    labels = [t.get_text() for t in axes[1, 0].get_legend().get_texts()]
    assert any("max_dt" in t and "per transition" in t for t in labels), (
        "the marker does not say it bounds a single transition, while the "
        "axis is the summed dt_total"
    )


def test_the_two_max_dt_lines_collapse_when_equal(monkeypatch, tmp_path):
    """A second line exactly on top of the first is clutter."""
    stats, a, b = _stats_for_figure(monkeypatch, max_dt_a=1000.0,
                                     max_dt_b=1000.0)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    vlines = [line.get_xdata()[0] for line in axes[1, 0].get_lines()
              if len(set(line.get_xdata())) == 1]
    assert vlines == [1000.0], f"{len(vlines)} lines drawn for one value"


def test_the_loss_panels_are_scaled_by_medians_not_quartiles(monkeypatch, tmp_path):
    """
    A single window diverging to 1e15 pushed the quartile band -- and with it
    the axis -- to 1e15, flattening the decades where the two curves actually
    differ into a line at the bottom. The bands stay drawn and simply run off
    the top.
    """
    # every 3rd window diverges, so the 75th percentile IS the blown value
    # and the band genuinely reaches 1e15. At 10% it sat below the upper
    # quartile and the fixture proved nothing.
    stats, a, b = _stats_for_figure(monkeypatch, blow_up_every=3)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    band_top = max(seg[:, 1].max() for panel in (axes[1, 0], axes[0, 2])
                    for seg in [panel.collections[0].get_paths()[0].vertices])
    assert band_top > 1e9, (
        f"the quartile band only reaches {band_top:.3g}; the fixture does "
        f"not exercise a band that would blow up the axis"
    )
    for panel in (axes[1, 0], axes[0, 2]):
        top = panel.get_ylim()[1]
        assert top < 1e6, (
            f"y range reaches {top:.3g} -- the diverging tail is setting the "
            f"scale, not the medians"
        )
        # the band itself is still plotted (it just leaves the view)
        assert panel.collections, "the quartile band was removed rather than clipped"


def test_a_missing_max_dt_is_simply_not_drawn(monkeypatch, tmp_path):
    """Older checkpoints may have no data_config; that must not raise."""
    stats, a, b = _stats_for_figure(monkeypatch)
    a["max_dt"] = b["max_dt"] = None
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    vlines = [line.get_xdata()[0] for line in axes[1, 0].get_lines()
              if len(set(line.get_xdata())) == 1]
    assert vlines == []
