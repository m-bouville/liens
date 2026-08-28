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
import zlib

import numpy as np
import pytest
import torch

import evaluation.compare_f_theta as cf
from models.constants import N_THETA


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
        x_t = rng.normal(size=(8, 8))
        x_real = x_t + rng.normal(0, 0.1, (8, 8))
        noise = pred_noise[0] if f_theta == "A" else pred_noise[1]
        x_pred = x_real + noise * rng.normal(size=(8, 8))
        return x_t, x_real, x_pred, x_real, 500.0, [250.0, 250.0]

    monkeypatch.setattr(cf, "compute_sample", fake_compute_sample)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
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
    _stub_models(monkeypatch)

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
    assert "plt.subplots(n_rows, 8" in src


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
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
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
                       "1 chained step, z1 not resynchronized"]
    # NO seed in the name for --fixed-windows (check_rollout's convention):
    # the seed played no part in selecting them, and a stamped seed would
    # suggest a rerun with another seed changes the windows.
    assert out.name == "128x128-stage3a_vs_stage3b-1step-propagated.png"


def _stats_stub(monkeypatch, corr_none_for=()):
    _stub_metadata(monkeypatch)
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
        rng = np.random.default_rng(
            zlib.crc32(f"{run_dir}:{steps[0]}".encode()) % 9973)
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
    # collect_stats now also calls the causal baseline; stub it off unless a
    # test specifically exercises it, so the stats tests stay filesystem-free.
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    return seen


def _model(stem):
    prefix, label = cf._parse_stem(stem)
    return {"path": stem, "ck": {}, "config": {}, "ae": None, "ae_encoder": None,
            "ae_config": {}, "ae_path": "x", "f_theta": stem,
            "prefix": prefix, "label": label}


def _flat_stage2(run_dir, steps, ae, ae_config, device, time_coordinate="t"):
    """A stage-2 trajectory of the right length, VARYING frame to frame.

    The row is present for every window, so tests that do not care about its
    values still need it to exist -- but a constant field has an undefined
    correlation, which would make every frame report n/a and trip the tests
    that check frame 0 is a real number.
    """
    rng = np.random.default_rng(0)
    sz = (ae_config or {}).get("size", 8)   # match the paired compute_sample stub's grid
    return [rng.normal(size=(sz, sz)) * (1 + 0.05 * k)
            for k in range(len(steps))]


def _stub_metadata(monkeypatch):
    """collect_stats reads each window's temperature from metadata; parse it
    from the run name (T{ddd}) so tests need no real metadata file on disk."""
    import re as _re
    import types as _types

    def _meta(path):
        m = _re.search(r"T(\d+)", str(path))
        T = int(m.group(1)) / 1000.0 if m else 0.7
        return _types.SimpleNamespace(temperature=T, T0=1.0, dt=1.0)

    monkeypatch.setattr(cf.load, "read_metadata", _meta)


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


def test_the_stats_figure_has_eight_panels_and_reports_the_window_count(monkeypatch, tmp_path):
    _stats_stub(monkeypatch)
    windows = [(f"run{i}", [i, i + 1, i + 2]) for i in range(30)]
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    _stub_metadata(monkeypatch)
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
    assert captured["axes"].shape == (2, 5)   # 4 base columns + the rate column
    titles = [captured["axes"][r, c].get_title()
              for r in range(2) for c in range(5)]
    assert any("loss distribution" in t for t in titles)
    assert any("correlation distribution" in t for t in titles)
    assert any("loss vs dt" in t for t in titles)
    assert any("correlation vs dt" in t for t in titles)
    assert any("loss vs temperature" in t for t in titles)
    assert any("correlation vs temperature" in t for t in titles)


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
    Columns 5, 6 and 7 (error A, error B, B-A) share one scale, so a second
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
    clims = [axes[0, c].get_images()[0].get_clim() for c in (5, 6, 7)]
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
                             "2 chained steps, z1 not resynchronized"), (
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
    assert "2 chained steps, z1 not resynchronized" in titles[-1]

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
    _stub_metadata(monkeypatch)

    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        n = len(steps)
        rng = np.random.default_rng(0)
        real = [rng.normal(size=(8, 8)) for _ in range(n)]
        scale = 1.0 if str(f_theta).endswith("a") else b_scale
        pred = [real[0]] + [real[i] * scale for i in range(1, n)]
        return real, pred, [250.0] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)
    # The causal row reads the run directory from disk; stub it off by
    # default so trajectory tests stay filesystem-free. Tests that care about
    # the causal row override this.
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)


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
    # real, stage 2, 3a, 3b (causal stubbed out in this fixture)
    assert axes.shape == (4, 5), f"grid is {axes.shape}, expected 4 x 5"
    labels = [axes[r, 0].get_ylabel().replace("\n", " ") for r in range(4)]
    assert labels[0] == "real"
    assert labels[1].startswith("stage 2"), labels
    assert labels[2] == "stage 3a" and labels[3] == "stage 3b"
    # ABSOLUTE time, with the offset from the window start in brackets: a
    # frame can be placed in the run without going back to the step numbers.
    assert [axes[0, c].get_title() for c in range(5)] == [
        "t = 0 (t0)", "t = 250 (t0 + 250)", "t = 500 (t0 + 500)",
        "t = 750 (t0 + 750)", "t = 1000 (t0 + 1000)"]


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

    rollout() already provides it as column 0 of its own output, so it is
    read from there rather than decoded separately and prepended -- which
    duplicated frame 0 and shifted every panel one frame behind.
    """
    import inspect
    src = inspect.getsource(cf.compute_trajectory)
    assert "for i in range(z0_hat_full.shape[1])" in src
    assert "pred_frames = [ae_decoder(z0_t)[0, 0].cpu().numpy()]" not in src, (
        "the start is decoded separately AND included by rollout, so frame 0 "
        "appears twice and every later column is off by one"
    )


def test_the_trajectory_length_matches_the_window(monkeypatch):
    """
    THE OFF-BY-ONE. rollout returns n_steps+1 columns including the input, so
    prepending a separately-decoded start gave n_steps+2 frames against a
    window of n_steps+1. The extra entry was silently unplotted: the figure
    showed columns 0..n_steps of a list that began with a duplicate, so every
    model panel was one frame BEHIND its real column and the final frame was
    never shown at all.
    """
    from pathlib import Path
    import types

    n_steps = 5

    class FakeMeta:
        dt, temperature, T0 = 1.0, 0.8, 1.0

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
        return torch.zeros(z.shape[0], 1, 8, 8)

    ae = types.SimpleNamespace(encoder=encoder, decoder=decoder)
    f_theta = types.SimpleNamespace(
        rollout=lambda z0, z1, dts, theta, z1_resync: torch.zeros(
            1, dts.shape[1] + 1, 2, 4, 4))

    real, pred, _ = cf.compute_trajectory(
        Path("run"), list(range(n_steps)), ae, f_theta, {"size": 8}, "cpu")
    assert len(pred) == len(real) == n_steps, (
        f"{len(pred)} predicted frames for a {len(real)}-frame window"
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
        dt, temperature, T0 = 1.0, 0.8, 1.0

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
    # (B, n_steps+1, ...) with [:, 0] == z0 -- rollout's REAL contract. The
    # stub previously returned n_steps columns, which is why a test asserting
    # len(pred) == len(steps) passed while the real code produced one frame
    # too many and shifted every panel.
    f_theta = types.SimpleNamespace(
        rollout=lambda z0, z1, dts, theta, z1_resync: torch.zeros(
            1, dts.shape[1] + 1, 2, 4, 4))

    real, pred, dt_per_step = cf.compute_trajectory(
        Path("run"), list(range(n_steps)), ae, f_theta, {"size": 8}, "cpu")
    assert len(real) == n_steps
    assert len(pred) == n_steps, (
        f"{len(pred)} predicted frames for {n_steps} columns -- the "
        f"intermediate states are not being decoded"
    )
    assert len(dt_per_step) == n_steps - 1
    # one decode per returned column, which already includes the start
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
    _stub_metadata(monkeypatch)
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
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    _stub_metadata(monkeypatch)


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
    assert axes[0, 1].get_title() == "t = 250 (t0 + 250)"


def test_the_metrics_show_the_collapse_frame(monkeypatch):
    """B is exact until frame 3 and noise after: its numbers must say so."""
    axes = _titles(monkeypatch, collapse_at=3)
    b_corr = [axes[3, c].get_title().split("corr=")[1] for c in range(5)]
    assert b_corr[1] == "100%" and b_corr[2] == "100%"
    assert b_corr[3] not in ("100%", "99%"), (
        f"frame 3 reports corr={b_corr[3]} for a collapsed prediction"
    )
    # stage 3a is row 2: real, stage 2, 3a, 3b
    a_corr = [axes[2, c].get_title().split("corr=")[1] for c in range(5)]
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
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    _stub_metadata(monkeypatch)
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
        assert axes[3, col].get_title().split("corr=")[1] == "100%", (
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
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    _stub_metadata(monkeypatch)
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
    assert axes[2, 2].get_title().split("corr=")[1] != "n/a"


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
    _stub_metadata(monkeypatch)
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
    assert axes.shape == (2, 5), f"grid is {axes.shape}, expected 2 x 5"
    assert "number of steps" in axes[0, 3].get_title()
    assert "number of steps" in axes[1, 3].get_title()
    assert axes[0, 3].get_xlabel() == "chained steps applied"
    # 5th column: the normalized per-step rate panels (ln[loss(n)/loss(0)]/n and
    # (1-corr(n))/n), sharing the vs-steps x-axis.
    assert "per step" in axes[0, 4].get_title()
    assert "per step" in axes[1, 4].get_title()
    assert axes[0, 4].get_xlabel() == "chained steps applied"
    assert axes[1, 4].get_xlabel() == "chained steps applied"
    # the new temperature column
    assert "temperature" in axes[0, 2].get_title()
    assert "temperature" in axes[1, 2].get_title()
    assert axes[0, 2].get_xlabel() == "temperature (SMA)"


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
        lambda m, n, ns, seed, mx, dev, t0_range=None: [(Path(f"T{500 + i}_n020_s{i}"),
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
        lambda m, n, ns, seed, mx, dev, t0_range=None: [(Path("T925_n020_s79"),
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
        rng = np.random.default_rng(zlib.crc32(str(run_dir).encode()) % 997)
        n = len(steps)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * i) for i in range(n)]
        pred = [f + 0.01 for f in real]
        pred[1] = pred[0]            # constant delta at k=1, EVERY window
        return real, pred, [250.0] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    _stub_metadata(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    monkeypatch.setattr(
        cf, "_select_windows",
        lambda m, n, ns, seed, mx, dev, t0_range=None: [(Path(f"r{i}"), list(range(ns + 1)))
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
        rng = np.random.default_rng(zlib.crc32(str(run_dir).encode()) % 997)
        n = len(steps)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * i) for i in range(n)]
        pred = [f + 0.01 for f in real]
        if str(run_dir) == "r0":          # ONE window only
            pred[1] = pred[0]
        return real, pred, [250.0] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    _stub_metadata(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    monkeypatch.setattr(
        cf, "_select_windows",
        lambda m, n, ns, seed, mx, dev, t0_range=None: [(Path(f"r{i}"), list(range(ns + 1)))
                                         for i in range(6)])
    cf.compare_f_theta("128x128-stage3a", "128x128-stage3b", n_samples=0,
                        n_steps=4, n_stats=6, device="cpu")
    out = capsys.readouterr().out
    assert "UNDEFINED for EVERY window" not in out


def _stats_for_figure(monkeypatch, blow_up_every=10, max_dt_a=2000.0,
                       max_dt_b=1000.0):
    def fake(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        n = len(steps)
        # crc32, NOT hash(): Python salts string hashing per process, so
        # hash() made this fixture assign different dt values and blow-up
        # flags on every run. It passed here and failed on another machine
        # for no reason but the seed -- a flaky test is worse than none.
        i = zlib.crc32(str(run_dir).encode()) % 997
        rng = np.random.default_rng(i)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * k) for k in range(n)]
        # dt comes from i % 8 and the blow-up from i // 8, so the two are
        # INDEPENDENT. Deriving both from i % 3 and i % 8 correlated them,
        # and whole dt bins came out entirely diverged -- the medians then
        # legitimately reached 1e30 and the fixture no longer isolated
        # "median small, upper quartile huge".
        blow = blow_up_every and ((i // 8) % blow_up_every == 0)
        pred = [real[k] + (1e15 if (blow and k > 2) else 0.02 * k)
                for k in range(n)]
        return real, pred, [250.0 * (1 + i % 8)] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    _stub_metadata(monkeypatch)

    # collect_stats reads each window's temperature from metadata; parse it
    # from the run name (T{ddd}) so the vs-T column has real spread.
    import re as _re
    import types as _types

    def _meta(path):
        m = _re.search(r"T(\d+)", str(path))
        T = int(m.group(1)) / 1000.0 if m else 0.7
        return _types.SimpleNamespace(temperature=T, T0=1.0, dt=1.0)

    monkeypatch.setattr(cf.load, "read_metadata", _meta)
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    a["max_dt"], b["max_dt"] = max_dt_a, max_dt_b
    # 160 windows over 8 dt values = ~20 per bin, so a 1-in-3 blow-up rate
    # puts the MEDIAN safely below it and the UPPER QUARTILE safely above.
    # With 40 windows the bins held ~7 each and sampling noise pushed some
    # bins past half diverged, making the medians themselves 1e30.
    windows = [(f"T{500 + i}_n020_s{i}", list(range(9))) for i in range(160)]
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


def test_the_loss_row_is_scaled_by_medians_not_the_diverging_tail(monkeypatch, tmp_path):
    """
    loss-vs-dt sets the loss row's range from its MEDIANS, and the other
    three panels adopt it: a single window diverging to 1e15 would otherwise
    push a quartile band -- and the axis -- to 1e15, flattening the decades
    where the curves differ. The bands stay drawn and run off the top; the
    diverging tail does NOT set the scale for any panel in the row.
    """
    # every 3rd window diverges, so the 75th percentile IS the blown value.
    stats, a, b = _stats_for_figure(monkeypatch, blow_up_every=3)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    # the band on vs-dt genuinely blows up... (max over ALL bands: which
    # collection is first depends on the legend order, and the flat baseline
    # curves are drawn before the diverging models)
    band_top = max(coll.get_paths()[0].vertices[:, 1].max()
                    for coll in axes[0, 1].collections)
    assert band_top > 1e9, (
        f"the quartile band only reaches {band_top:.3g}; the fixture does "
        f"not exercise a band that would blow up the axis"
    )
    # ...yet no loss panel's axis follows it there
    for c in range(4):
        top = axes[0, c].get_ylim()[1]
        assert top < 1e6, (
            f"loss panel col {c} y range reaches {top:.3g} -- the diverging "
            f"tail is setting the scale, not the medians"
        )
    assert axes[0, 1].collections, "the quartile band was removed rather than clipped"


def test_the_whole_loss_row_shares_one_y_range_set_by_vs_dt(monkeypatch, tmp_path):
    """
    All FOUR loss panels (CDF, vs-dt, vs-T, vs-steps) share one y range so
    the row reads across. loss-vs-dt is the REFERENCE -- scaled to its own
    medians so the decades where the curves differ stay legible -- and the
    others adopt it. The CDF therefore loses its diverged tail off the top,
    which is the reversed match (it used to set the range and show the tail).
    """
    stats, a, b = _stats_for_figure(monkeypatch, blow_up_every=3)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    ref = axes[0, 1].get_ylim()
    for c in (0, 2):
        assert axes[0, c].get_ylim() == ref, (
            f"loss panel col {c} does not share loss-vs-dt's y range"
        )
        assert axes[0, c].get_yscale() == "log"
    # vs-steps is NOT in the shared range: it plots every step, starting from
    # the near-zero step-0 loss, so it spans decades the endpoint panels
    # never reach and keeps its own median scaling.
    assert axes[0, 3].get_ylim()[0] < ref[0], (
        "loss-vs-steps was forced onto the endpoint panels' range, crushing "
        "its low end"
    )
def test_correlation_axes_always_span_zero_and_one_hundred(monkeypatch, tmp_path):
    """
    Correlation has fixed, meaningful endpoints -- no skill and perfect. A
    panel autoscaled to 40..92% invites reading a small real difference as a
    large one, and cannot be compared against the panel beside it.
    """
    stats, a, b = _stats_for_figure(monkeypatch, blow_up_every=0)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    lo, hi = axes[1, 0].get_ylim()          # CDF now has correlation on Y
    assert lo <= 0.0 and hi >= 100.0, f"distribution panel spans {lo}..{hi}"
    for panel in (axes[1, 1], axes[1, 2], axes[1, 3]):  # correlation on y
        lo, hi = panel.get_ylim()
        assert lo <= 0.0 and hi >= 100.0, f"panel spans {lo}..{hi}"


def test_dt_axes_are_numbered_more_than_once_per_decade(monkeypatch, tmp_path):
    """
    A log axis spanning ~200 to ~5000 gets exactly ONE major tick from the
    default locator, so the axis carried a single number and no point on it
    could be placed.
    """
    stats, a, b = _stats_for_figure(monkeypatch, blow_up_every=0)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    axes[0, 0].figure.canvas.draw()
    for panel in (axes[0, 1], axes[1, 1]):
        lo, hi = panel.get_xlim()
        labelled = [float(t.get_text().replace(",", ""))
                    for t in panel.get_xticklabels(which="both")
                    if t.get_text()]
        inside = [v for v in labelled if lo <= v <= hi]
        assert len(inside) >= 4, (
            f"only {len(inside)} numbers on the dt axis between {lo:.0f} and "
            f"{hi:.0f}: {sorted(inside)}"
        )


def test_no_max_dt_markers_are_drawn(monkeypatch, tmp_path):
    """
    ROLLED BACK. max_dt bounds a single transition while the axis is
    dt_total, the sum over the window, so the line separated nothing: an
    8-step window at 8 x max_dt is entirely in-distribution.
    """
    stats, a, b = _stats_for_figure(monkeypatch, blow_up_every=0)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    for panel in (axes[0, 1], axes[1, 1]):
        vlines = [line.get_xdata()[0] for line in panel.get_lines()
                  if len(set(line.get_xdata())) == 1]
        assert vlines == [], f"a vertical marker is still drawn at {vlines}"


def test_dt_axis_tick_density_adapts_to_the_span():
    """
    A narrow range needs every sub-tick numbered to be readable at all; a
    four-decade range numbered the same way is a solid black line. The
    default minor locator already places sub-ticks, so this adaptive choice
    is what stops a wide axis from being unreadable -- removing it is
    invisible on a narrow one.
    """
    import matplotlib.pyplot as plt

    def n_labels(lo, hi):
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        cf._label_dt_axis(ax)
        fig.canvas.draw()
        labels = [t.get_text() for t in ax.get_xticklabels(which="both")
                  if t.get_text()]
        inside = [v for v in (float(t.replace(",", "")) for t in labels)
                  if lo <= v <= hi]
        plt.close(fig)
        return len(inside)

    narrow = n_labels(200.0, 2000.0)
    wide = n_labels(100.0, 1e6)
    assert narrow >= 4, f"a narrow axis carries only {narrow} numbers"
    assert wide <= 20, (
        f"a four-decade axis carries {wide} numbers -- at that density the "
        f"labels overlap into a solid line"
    )


def test_the_layout_is_metric_by_row_and_view_by_column(monkeypatch, tmp_path):
    """
    Rows are the METRIC, columns the VIEW, so every column shares one x axis:
    the two dt panels sit one above the other and can be read together, as can
    the two step panels. Before the transpose the dt panels were diagonal from
    each other.
    """
    stats, a, b = _stats_for_figure(monkeypatch, blow_up_every=0)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    assert "loss" in axes[0, 0].get_title() and "loss" in axes[0, 1].get_title()
    assert "correlation" in axes[1, 0].get_title()
    assert "correlation" in axes[1, 1].get_title()
    # each column shares an x axis, top row over bottom
    assert (axes[0, 1].get_xlabel() == axes[1, 1].get_xlabel()
            == "dt_total (binned)")
    assert (axes[0, 2].get_xlabel() == axes[1, 2].get_xlabel()
            == "temperature (SMA)")
    assert (axes[0, 3].get_xlabel() == axes[1, 3].get_xlabel()
            == "chained steps applied")


def test_the_fixtures_do_not_depend_on_hash_randomisation():
    """
    THE FLAKE. Python's string hash is salted per process unless
    PYTHONHASHSEED is set, so a fixture keyed on it assigned different dt
    values and blow-up flags on every run: the y-limit test passed here and
    failed on another machine for no reason but the seed. Fixtures use
    zlib.crc32 instead.
    """
    import pathlib

    # The pattern is ASSEMBLED rather than written out: spelled literally it
    # would appear in this test's own source and match itself. (Comments are
    # stripped by source_without_comments; docstrings are not.)
    pattern = "abs(" + "hash("
    src = pathlib.Path(__file__).resolve().read_text(encoding="utf-8")
    assert pattern not in src, (
        "a fixture is keyed on Python's salted string hash, so its data "
        "changes between runs"
    )


def test_the_causal_baseline_is_the_second_row(monkeypatch, tmp_path):
    """
    Immediately under the truth, because it is the bar every model row has to
    clear: what the PAST alone predicts, with no model. Measured on real
    data, it beats z1 outright beyond dt ~ 1e3.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    _traj_stub(monkeypatch)
    monkeypatch.setattr(cf, "compute_causal_trajectory",
                         lambda *a, **k: [np.zeros((8, 8)) for _ in range(5)])
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._trajectory_figure(Path("T625_n050_s599"), list(range(5)),
                           _model("128x128-stage3a"), _model("128x128-stage3b"),
                           "cpu", False, "T", tmp_path / "t.png")
    axes = captured["axes"]
    assert axes.shape == (5, 5)
    labels = [axes[r, 0].get_ylabel().replace("\n", " ") for r in range(5)]
    assert labels[0] == "real"
    assert labels[1].startswith("previous derivative")
    assert labels[2].startswith("stage 2"), labels
    assert labels[3] == "stage 3a" and labels[4] == "stage 3b"
    # and it carries its own loss/corr like the model rows
    assert "loss=" in axes[1, 2].get_title()


def test_a_window_at_a_runs_first_step_drops_the_causal_row(monkeypatch, tmp_path):
    """A backward difference needs an EARLIER real frame. Without one the row
    is omitted rather than faked."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    _traj_stub(monkeypatch)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._trajectory_figure(Path("r"), list(range(5)), _model("128x128-stage3a"),
                           _model("128x128-stage3b"), "cpu", False, "T",
                           tmp_path / "t.png")
    axes = captured["axes"]
    assert axes.shape == (4, 5), "the causal row was fabricated without data"
    labels = [axes[r, 0].get_ylabel().replace("\n", " ") for r in range(4)]
    assert labels[0] == "real"
    assert labels[1].startswith("stage 2"), labels
    assert labels[2] == "stage 3a" and labels[3] == "stage 3b"


def test_the_causal_estimate_is_backward_not_centered():
    """
    A centered difference sees z0(t+dt) and would be scored against that same
    frame -- it is a smoothness probe, not an achievable predictor. The
    baseline must use only frames available at prediction time.
    """
    import inspect
    src = inspect.getsource(cf.compute_causal_trajectory)
    assert "z0_dot_back = (z0_t - z0_prev) / dt_minus" in src
    assert "saved.index(steps[0]) - 1" in src, (
        "the earlier frame is not the run's own previous SAVED step"
    )
    # frozen and extrapolated: a causal derivative cannot be re-estimated
    # once the trajectory leaves the data
    assert "z0_t + z0_dot_back * elapsed" in src


def test_metric_keys_travel_with_their_row():
    """The row->key mapping was derived from the row INDEX, which stopped
    being readable once the causal row could be present or absent."""
    import inspect
    src = inspect.getsource(cf._trajectory_figure)
    assert "for row, (label, frames, metric_key) in enumerate(rows):" in src
    assert "metrics[metric_key][col]" in src


def _causal_io_stub(monkeypatch, save_steps, seen_reads):
    import types

    class FakeMeta:
        dt, temperature, T0 = 1.0, 0.8, 1.0

    FakeMeta.save_steps = save_steps
    monkeypatch.setattr(cf.load, "read_metadata", lambda p: FakeMeta())
    monkeypatch.setattr(cf.load, "snapshot_filename", lambda s: f"s{s}")

    def read(path, nx, ny):
        seen_reads.append(str(path).rsplit("s", 1)[-1])
        return np.full((nx, ny), float(len(seen_reads)), dtype=np.float32)

    monkeypatch.setattr(cf.load, "read_phi_half", read)
    monkeypatch.setattr(cf, "resolve_stream_configs_from_checkpoint_config",
                         lambda cfg: (None, "recon"))

    def encoder(x, theta=None):
        # z0 = frame index, so the backward difference is exactly 1 per frame
        n = x.shape[0]
        vals = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1)
        return {"recon": vals.expand(n, 1, 2, 2).clone()}

    decoded = []

    def decoder(z):
        decoded.append(float(z.flatten()[0]))
        return torch.zeros(z.shape[0], 1, 8, 8)

    return types.SimpleNamespace(encoder=encoder, decoder=decoder), decoded


def test_causal_returns_None_at_a_runs_first_saved_step(monkeypatch):
    """A backward difference needs an EARLIER saved frame; there is none at
    the start of a run, and inventing one would fabricate the baseline."""
    from pathlib import Path
    reads = []
    ae, _ = _causal_io_stub(monkeypatch, [100, 200, 300, 400], reads)
    out = cf.compute_causal_trajectory(Path("r"), [100, 200, 300], ae,
                                        {"size": 8}, "cpu")
    assert out is None, "a window at the run's first step produced a baseline"
    assert reads == [], "frames were read despite there being no earlier one"


def test_causal_differences_the_PREVIOUS_saved_step(monkeypatch):
    """Not the next one, and not a centered pair: only frames available at
    prediction time."""
    from pathlib import Path
    reads = []
    ae, _ = _causal_io_stub(monkeypatch, [100, 200, 300, 400], reads)
    out = cf.compute_causal_trajectory(Path("r"), [300, 400], ae,
                                        {"size": 8}, "cpu")
    assert out is not None
    assert reads == ["200", "300"], (
        f"read frames {reads}; the baseline must difference 300 against the "
        f"PREVIOUS saved step 200"
    )


def test_causal_freezes_the_slope_and_extrapolates(monkeypatch):
    """
    The slope is estimated ONCE from (prev_step, t0) and held: a causal
    estimate needs two real PAST frames, and there are none once the
    trajectory leaves the data. Frame k is z0(t0) + slope * elapsed_k, so
    growth is linear in elapsed time from the start -- NOT re-read from
    later real frames (that would be teacher-forced, and the frozen baseline
    already beats both models, so strengthening it is pointless).
    """
    from pathlib import Path
    reads = []
    ae, decoded = _causal_io_stub(monkeypatch, [0, 100, 200, 300, 400], reads)
    out = cf.compute_causal_trajectory(Path("r"), [100, 200, 300, 400], ae,
                                        {"size": 8}, "cpu")
    assert out is not None and len(out) == 4
    # Only (prev_step, t0) are read -- the later window frames are NOT, which
    # is what "frozen" means and what keeps it a fair rollout baseline.
    assert reads == ["0", "100"], (
        f"reads {reads}: a frozen causal baseline reads only the two frames "
        f"before the window, not the window's own later frames"
    )
    # z0(prev)=0, z0(t0)=1 over dt_minus=100 -> slope 0.01; elapsed 0,100,200,
    # 300 -> 1.0, 2.0, 3.0, 4.0
    assert decoded == pytest.approx([1.0, 2.0, 3.0, 4.0], rel=1e-5), decoded


def test_causal_is_a_frozen_extrapolation_not_teacher_forced(monkeypatch):
    """The slope must not be re-estimated from later real frames: the frozen
    baseline is the honest apples-to-apples one, and it already wins."""
    import inspect
    src = inspect.getsource(cf.compute_causal_trajectory)
    assert "z0_dot_back = (z0_t - z0_prev) / dt_minus" in src
    assert "z0_t + z0_dot_back * elapsed" in src
    assert "z0_real[k - 1:k]" not in src, (
        "the slope is being re-read from later real frames -- teacher-forced, "
        "not the frozen rollout baseline"
    )


def _stats_with_causal(monkeypatch, n=40):
    """a/b collapse; causal predicts a small varying delta on every window."""
    def traj(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        m = len(steps)
        i = zlib.crc32(str(run_dir).encode()) % 997
        rng = np.random.default_rng(i)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * k) for k in range(m)]
        return real, [real[k] + 0.02 * k for k in range(m)], [250.0] * (m - 1)

    def caus(run_dir, steps, ae, ae_config, device, time_coordinate="t"):
        m = len(steps)
        i = zlib.crc32(str(run_dir).encode()) % 997
        rng = np.random.default_rng(i)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * k) for k in range(m)]
        return [real[k] + 0.05 * rng.normal(size=(8, 8)) for k in range(m)]

    monkeypatch.setattr(cf, "compute_trajectory", traj)
    monkeypatch.setattr(cf, "compute_causal_trajectory", caus)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    windows = [(f"T{500 + i}_n020_s{i}", list(range(9))) for i in range(n)]
    _stub_metadata(monkeypatch)
    return cf.collect_stats(a, b, windows, "cpu", False), a, b


def test_stats_collect_a_causal_series_with_its_own_dt(monkeypatch):
    """The baseline is absent on windows with no pre-window frame, so it
    carries its own endpoint and dt arrays rather than aligning to a/b."""
    stats, a, b = _stats_with_causal(monkeypatch)
    assert len(stats["step_loss_causal"]) == 40
    assert len(stats["loss_causal"]) == len(stats["dt_causal"]) == 40


def test_the_causal_curve_is_on_all_six_stats_panels(monkeypatch, tmp_path):
    """It is the bar the models must clear, so it belongs on every view --
    both distributions, both-vs-dt, and both-vs-steps."""
    import matplotlib.pyplot as plt
    stats, a, b = _stats_with_causal(monkeypatch)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._stats_figure(stats, a, b, "T", tmp_path / "s.png")
    axes = captured["axes"]
    for r in range(2):
        for c in range(3):
            legend = axes[r, c].get_legend()
            labels = [t.get_text() for t in legend.get_texts()]
            assert any("previous derivative" in t for t in labels), (
                f"panel [{r},{c}] has no causal curve: {labels}"
            )


def test_the_causal_curve_is_absent_when_no_window_had_a_baseline(monkeypatch, tmp_path):
    """If every window starts at its run's first step there is no baseline,
    and the figure must not invent an empty green line."""
    import matplotlib.pyplot as plt
    _traj_stub_collapsing(monkeypatch)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    windows = [(f"run{i}", list(range(5))) for i in range(10)]
    _stub_metadata(monkeypatch)
    stats = cf.collect_stats(a, b, windows, "cpu", False)
    assert stats["step_loss_causal"] == []
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._stats_figure(stats, a, b, "T", tmp_path / "s.png")
    labels = [t.get_text()
              for t in captured["axes"][0, 2].get_legend().get_texts()]
    assert not any("previous derivative" in t for t in labels)


def test_causal_vs_dt_uses_its_OWN_dt_array(monkeypatch, tmp_path):
    """
    The baseline is absent on some windows, so its endpoint list is shorter
    than a/b's. Binning it against the a/b dt array (one entry per window)
    would pair causal losses with the wrong dt -- an index shift that
    silently mislabels every causal point. Its own dt_causal array is the
    only correct partner.
    """
    def traj(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        n = len(steps)
        i = zlib.crc32(str(run_dir).encode()) % 997
        rng = np.random.default_rng(i)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * k) for k in range(n)]
        # dt rises steeply with the run index, so a one-window shift moves a
        # causal point into a visibly different dt bin
        dt = 100.0 * (i % 20 + 1)
        return real, [real[k] + 0.02 * k for k in range(n)], [dt] * (n - 1)

    def caus(run_dir, steps, ae, ae_config, device, time_coordinate="t"):
        # No baseline for the first few runs -> causal arrays are SHORTER
        if str(run_dir).endswith(("s0", "s1", "s2")):
            return None
        n = len(steps)
        rng = np.random.default_rng(1)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * k) for k in range(n)]
        return [real[k] + 0.05 * rng.normal(size=(8, 8)) for k in range(n)]

    monkeypatch.setattr(cf, "compute_trajectory", traj)
    monkeypatch.setattr(cf, "compute_causal_trajectory", caus)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    windows = [(f"T500_n020_s{i}", list(range(9))) for i in range(20)]
    _stub_metadata(monkeypatch)
    stats = cf.collect_stats(a, b, windows, "cpu", False)

    assert len(stats["loss_causal"]) < len(stats["loss_a"]), (
        "fixture failed to make the causal list shorter"
    )
    assert len(stats["loss_causal"]) == len(stats["dt_causal"]), (
        "causal endpoints and their dt array have drifted out of step"
    )
    # the binning must consume exactly as many dt as losses
    from evaluation.compare_f_theta import _binned
    c, med, _, _ = _binned(np.array(stats["dt_causal"]),
                            np.array(stats["loss_causal"]))
    assert c.size > 0


def test_the_cdf_panels_put_the_quantity_on_the_y_axis(monkeypatch, tmp_path):
    """
    Flipped so loss and correlation sit on the SAME y axis as the vs-dt and
    vs-steps panels beside them: the three views of one metric can then be
    read straight across. Cumulative fraction moves to x.
    """
    stats, a, b = _stats_for_figure(monkeypatch, blow_up_every=0)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    assert axes[0, 0].get_xlabel() == "cumulative fraction of windows"
    assert axes[0, 0].get_ylabel() == "end-to-end loss"
    assert axes[0, 0].get_yscale() == "log"
    assert axes[1, 0].get_xlabel() == "cumulative fraction of windows"
    assert axes[1, 0].get_ylabel().startswith("correlation")
    # and the loss CDF shares its y scale with loss-vs-dt and loss-vs-steps
    assert axes[0, 1].get_yscale() == "log" and axes[0, 2].get_yscale() == "log"


def test_the_two_last_step_aggregations_are_over_different_populations():
    """
    The apparent loss-vs-dt / loss-vs-steps disagreement at the last step is
    NOT a bug: vs-steps medians over ALL windows at step k, vs-dt bins the
    same endpoints by dt_total, so the rightmost points are medians over
    different subpopulations. The titles say so, and the endpoint feeding
    both is provably the same value.
    """
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "evaluation/compare_f_theta.py")
    # endpoint used by the CDF / vs-dt panels is literally the last per-step
    assert 'row[key] = {"loss": losses_k[-1], "corr": corrs_k[-1]}' in src
    assert "all windows per step" in src


def test_the_whole_correlation_row_shares_one_y_range(monkeypatch, tmp_path):
    """
    The CDF [1,0], vs-dt [1,1] and vs-steps [1,2] are all correlation, so
    they share ONE y range and read straight across -- including the y-min.
    _corr_axis pins each to span 0..100, but the vs-dt/vs-steps panels dip
    negative (a model anticorrelated with truth) while the CDF, bounded by
    its own worst window, need not, so without the union they misaligned at
    the bottom.
    """
    # A fixture that drives correlation NEGATIVE, so the shared floor is < 0
    # and a panel left out of the union would visibly stay pinned at 0.
    def traj(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        n = len(steps)
        i = zlib.crc32(str(run_dir).encode()) % 997
        rng = np.random.default_rng(i)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * k) for k in range(n)]
        pred = [real[k] if k < 3 else -real[k] * (1 + 0.01 * (i % 30))
                for k in range(n)]
        return real, pred, [400.0 * (1 + i % 6)] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", traj)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    _stub_metadata(monkeypatch)
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    windows = [(f"T{550 + i * 5}_n020_s{i}", list(range(9))) for i in range(60)]
    stats = cf.collect_stats(a, b, windows, "cpu", False)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    ranges = [axes[1, c].get_ylim() for c in range(4)]
    assert ranges[0][0] < 0, "fixture did not drive the shared floor negative"
    assert len(set(ranges)) == 1, (
        f"correlation panels have y ranges {ranges} -- the row does not read "
        f"across (a panel left out of the union stays pinned at 0)"
    )


def test_a_negative_correlation_min_reaches_the_cdf_panel(monkeypatch, tmp_path):
    """
    When a model goes anticorrelated the vs-dt panel's y-min is negative; the
    CDF must inherit that same floor, not stay pinned at 0, or the two panels
    in the row cannot be compared at the bottom.
    """
    def traj(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        n = len(steps)
        i = zlib.crc32(str(run_dir).encode()) % 997
        rng = np.random.default_rng(i)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * k) for k in range(n)]
        # An INTERMEDIATE step is the most anticorrelated -- the endpoint
        # recovers. The endpoint-based CDF [1,0] therefore never sees the
        # most negative value, only the vs-steps panel [1,3] does, so the
        # panels have DIFFERENT natural floors and min-vs-max matters.
        pred = []
        for k in range(n):
            if k < 2:
                pred.append(real[k])
            elif k == n // 2:
                pred.append(-real[k] * 3.0)          # deep negative, mid-window
            else:
                pred.append(real[k] * 0.9)           # endpoint back near +1
        return real, pred, [400.0 * (1 + i % 6)] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", traj)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    windows = [(f"T500_n020_s{i}", list(range(9))) for i in range(60)]
    _stub_metadata(monkeypatch)
    stats = cf.collect_stats(a, b, windows, "cpu", False)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    # the vs-steps panel dips well below the CDF's own endpoint floor
    steps_floor = axes[1, 3].get_ylim()[0]
    assert steps_floor < 0, "fixture did not drive an intermediate step negative"
    # every panel shares that MOST-negative floor -- a max() union would have
    # clipped the vs-steps band instead
    shared_lo = axes[1, 0].get_ylim()[0]
    assert shared_lo == steps_floor, (
        f"CDF floor {shared_lo} != vs-steps floor {steps_floor}: the union "
        f"took max() and the deepest panel was clipped"
    )
    # Bands may now run OFF the bottom: the axis is floored at -20% so the
    # 0..100 range, where every meaningful difference lives, is not squeezed
    # into a sliver by one wildly anticorrelated window. What must hold is
    # that the floor is the floor.
    assert shared_lo >= -20.0 - 1e-6, (
        f"the correlation axis reaches {shared_lo:.1f}% -- below the -20% "
        f"floor, so the useful range is compressed"
    )
    for panel in (axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]):
        assert panel.get_ylim()[0] == shared_lo


def test_collect_stats_records_a_temperature_per_window(monkeypatch):
    """The vs-T column needs each window's temperature, read from its own
    metadata -- and a matching temp_causal for the windows that have a
    baseline, so the causal curve can be binned by T too."""
    stats, a, b = _stats_with_causal(monkeypatch)
    assert len(stats["temperature"]) == len(stats["loss_a"])
    assert len(stats["temp_causal"]) == len(stats["loss_causal"])


def test_the_temperature_column_is_present_on_both_rows(monkeypatch, tmp_path):
    """A third view of the SAME endpoints, against temperature -- dt and T are
    collinear here, so binning by T shows how much of the dt trend is a T
    trend. Loss-vs-T and corr-vs-T, with the causal baseline on both."""
    stats, a, b = _stats_with_causal(monkeypatch)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    assert "temperature" in axes[0, 2].get_title()
    assert "temperature" in axes[1, 2].get_title()
    assert axes[0, 2].get_xlabel() == "temperature (SMA)"
    for r in (0, 1):
        labels = [t.get_text() for t in axes[r, 2].get_legend().get_texts()]
        assert any("previous derivative" in t for t in labels), (
            f"the causal baseline is missing from the vs-T panel [{r},2]"
        )


def test_temperature_uses_a_moving_window_not_bins(monkeypatch, tmp_path):
    """
    One point per DISTINCT temperature, each averaging that T plus the two
    sweep values either side. Bins would impose arbitrary edges on a narrow,
    densely-sampled range; a sliding window keeps every point on a real
    temperature and lets neighbours overlap smoothly.
    """
    stats, a, b = _stats_with_causal(monkeypatch, n=80)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    xs = axes[0, 2].get_lines()[0].get_xdata()
    distinct = np.unique(np.array(stats["temperature"], dtype=float))
    # one point per distinct T that has a FULL window: the two at each end
    # are dropped
    assert len(xs) == len(distinct) - 4, (
        f"{len(xs)} plotted points for {len(distinct)} distinct temperatures "
        f"-- expected {len(distinct) - 4} after trimming the partial windows"
    )
    # and every plotted x IS one of the sweep's own temperatures
    assert np.allclose(np.sort(xs), distinct[2:-2])


def test_the_moving_window_spans_T_plus_and_minus_two_values():
    """
    {T, previous 2, next 2} -- and ONLY full windows. The first and last two
    values can only draw on a truncated set (3 temperatures, not 5), so their
    point would be a different, noisier statistic on the same line, right at
    the ends of the range where the interesting behaviour is. Dropped.
    """
    T = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    v = np.arange(7, dtype=float)      # value == index, so membership reads off
    x, med, lo, hi, n = cf._moving_window(T, v, half_width=2)
    assert np.allclose(x, [0.7, 0.8, 0.9]), (
        f"plotted {x}; the two values at each end have no full window"
    )
    # each is the median of 5 consecutive indices, i.e. the centre index
    assert np.allclose(med, [2.0, 3.0, 4.0])


def test_the_moving_window_reports_the_MEDIAN_inside_its_quartile_band():
    """
    MEDIAN, not mean. The loss distribution is heavy-tailed enough that one
    diverged window in six drags the mean ~17 decades above the 75th
    percentile, putting the plotted line outside its own band. The median is
    bracketed by construction.
    """
    T = np.repeat([0.5, 0.6, 0.7, 0.8, 0.9], 6)
    v = np.tile(np.array([3.0, 5.0, 8.0, 12.0, 20.0, 1e19]), 5)
    _, med, lo, hi, _n = cf._moving_window(T, v, half_width=2)
    assert med[0] == pytest.approx(10.0), (
        f"reported {med[0]:.3g}; the mean would be ~1.7e18, far outside the "
        f"quartile band"
    )
    assert lo[0] <= med[0] <= hi[0], "the line falls outside its own band"


def test_the_header_carries_ABSOLUTE_time_not_only_the_offset(monkeypatch, tmp_path):
    """
    "t = 650 (t0)" then "t = 750 (t0 + 100)". The bare offset gave no way to
    place a frame in the run without going back to the title's step numbers.

    Uses a window that does NOT start at step 0 -- with t0 = 0 the absolute
    and offset numbers coincide and dropping the absolute term is invisible.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    steps = [13000, 15000, 17500, 20000]
    sim_dt = 0.05                       # -> t0 = 650, offsets 0/100/225/350

    def traj(run_dir, s, ae, f_theta, ae_config, device, z1_resync):
        rng = np.random.default_rng(0)
        real = [rng.normal(size=(8, 8)) for _ in range(len(s))]
        return (real, [f + 0.01 for f in real],
                [(s[i + 1] - s[i]) * sim_dt for i in range(len(s) - 1)])

    monkeypatch.setattr(cf, "compute_trajectory", traj)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._trajectory_figure(Path("T625_n050_s599"), steps,
                           _model("128x128-stage3a"), _model("128x128-stage3b"),
                           "cpu", False, "T", tmp_path / "tr.png")
    axes = captured["axes"]
    assert [axes[0, c].get_title() for c in range(4)] == [
        "t = 650 (t0)", "t = 750 (t0 + 100)", "t = 875 (t0 + 225)",
        "t = 1000 (t0 + 350)"]


def test_sim_dt_is_recovered_from_the_step_spacing(monkeypatch, tmp_path):
    """
    metadata.dt is derived from data already in hand -- dt_per_step[i] is
    (steps[i+1] - steps[i]) * metadata.dt -- rather than re-reading the
    metadata file, so the header needs no extra I/O. A single-frame window
    has no spacing to divide by and must not raise.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    def traj(run_dir, s, ae, f_theta, ae_config, device, z1_resync):
        rng = np.random.default_rng(0)
        real = [rng.normal(size=(8, 8)) for _ in range(len(s))]
        return real, [f + 0.01 for f in real], []

    monkeypatch.setattr(cf, "compute_trajectory", traj)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._trajectory_figure(Path("T625_n050_s599"), [13000],
                           _model("128x128-stage3a"), _model("128x128-stage3b"),
                           "cpu", False, "T", tmp_path / "tr.png")
    assert captured["axes"][0, 0].get_title() == "t = 0 (t0)"


def test_the_moving_window_does_not_trim_away_every_point():
    """
    Trimming the partial windows must not empty the panel when there are too
    few distinct temperatures to have a full window at all (a single-T sweep,
    or a small fixture). An empty panel hides the data entirely, which is
    worse than a partial window -- matplotlib also warns about the empty
    legend, which is how this was caught.
    """
    T = np.repeat(0.7, 5)
    x, med, lo, hi, _n = cf._moving_window(T, np.arange(5.0), half_width=2)
    assert x.size == 1, "a single-temperature sweep produced no plotted point"
    # and with enough distinct values the trimming DOES apply
    T7 = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    x7, _, _, _, _ = cf._moving_window(T7, np.arange(7.0), half_width=2)
    assert np.allclose(x7, [0.7, 0.8, 0.9])


def test_the_figure_reports_how_many_distinct_dt_and_T_values(monkeypatch, tmp_path):
    """
    How many DISTINCT values each smoothed axis has is the thing a reader
    cannot recover from the panels, and it decides whether 8 bins is over- or
    under-resolving dt_total. Reported in the figure's own header.
    """
    import matplotlib.pyplot as plt
    stats, a, b = _stats_with_causal(monkeypatch, n=40)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("fig", fig)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._stats_figure(stats, a, b, "T", tmp_path / "s.png")
    header = captured["fig"]._suptitle.get_text()
    n_dt = len(np.unique(np.array(stats["dt"], dtype=float)))
    n_temp = len(np.unique(np.array(stats["temperature"], dtype=float)))
    assert f"{n_dt} distinct dt_total" in header, header
    assert f"{n_temp} distinct temperatures" in header, header
    # and the RANGE covered: a sample that never reached the sweep's hot end
    # produced a panel stopping at 0.98 with nothing to say it was truncated
    temps = np.array(stats["temperature"], dtype=float)
    assert f"({temps.min():g}-{temps.max():g})" in header, header


def test_the_moving_window_reports_its_population_per_point():
    """
    A quartile band over 2-3 windows is not a quartile band, and the panel
    cannot show how thin its tails get. The count travels with the curve so
    a suspicious band can be checked against the population behind it.
    """
    T = np.concatenate([np.repeat(0.5, 10), np.repeat(0.6, 10),
                         np.repeat(0.7, 10), np.repeat(0.8, 10),
                         np.repeat(0.9, 10), np.repeat(1.0, 1)])
    v = np.arange(T.size, dtype=float)
    x, med, lo, hi, n = cf._moving_window(T, v, half_width=2)
    assert n.size == x.size
    # the point whose window reaches the sparse T=1.0 end has fewer windows
    assert n.min() < n.max(), "every point reports the same population"
    assert n.sum() > 0


def test_temperature_axis_always_shows_T_equals_one(monkeypatch, tmp_path):
    """
    T = 1 is the sweep's physical endpoint, and the SMA trims the last two
    values, so the points stop short of it. Without the endpoint pinned a
    panel ending at 0.97 looks like it covers the whole range.
    """
    stats, a, b = _stats_with_causal(monkeypatch, n=80)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    for r in (0, 1):
        assert axes[r, 2].get_xlim()[1] >= 1.0, (
            f"temperature panel [{r},2] stops at {axes[r, 2].get_xlim()[1]:.3f}"
            f" -- T = 1 is off the axis"
        )


def test_stage2_is_causal_and_reads_only_the_starting_frame(monkeypatch):
    """
    THE COMPATIBILITY REQUIREMENT. 3a/3b see only t0 and advance z1
    internally; stage 2 must do the same or the comparison is void. It
    advances z1 by decoding its own predicted z0 and re-encoding it:

        z0(t+(n+1)dt) = z0(t+n dt) + z1(t+n dt) dt
        x (t+(n+1)dt) = D[z0(t+(n+1)dt)]
        z1(t+(n+1)dt) = E[x(t+(n+1)dt)]

    An earlier version read z1 from the REAL frame at each step. It looked
    stable for a reason that had nothing to do with stage 2 being good: with
    a correct derivative supplied every step, error cannot compound.
    """
    from pathlib import Path
    import types

    reads = []

    class FakeMeta:
        dt, temperature, T0 = 1.0, 0.9, 1.0

    monkeypatch.setattr(cf.load, "read_metadata", lambda p: FakeMeta())
    monkeypatch.setattr(cf.load, "snapshot_filename", lambda s: f"s{s}")

    def read(path, nx, ny):
        reads.append(str(path).rsplit("s", 1)[-1])
        return np.full((nx, ny), 1.0, dtype=np.float32)

    monkeypatch.setattr(cf.load, "read_phi_half", read)
    monkeypatch.setattr(cf, "resolve_stream_configs_from_checkpoint_config",
                         lambda cfg: (None, "recon"))

    def encoder(x, theta=None):
        m = x.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
        return {"recon": m.expand(-1, 1, 2, 2).clone(),
                "deriv": 0.1 * m.expand(-1, 1, 2, 2).clone()}

    def decoder(z):
        v = z.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
        return v.expand(-1, 1, 8, 8).clone()

    ae = types.SimpleNamespace(encoder=encoder, decoder=decoder)
    out = cf.compute_stage2_trajectory(Path("run"), [100, 200, 300, 400, 500],
                                        ae, {"size": 8}, "cpu")
    assert reads == ["100"], (
        f"read frames {reads}; a causal row may read ONLY the starting frame"
    )
    # z1 is re-derived from the model's own drifting state, so error COMPOUNDS
    growth = [out[i + 1].mean() - out[i].mean() for i in range(len(out) - 1)]
    assert growth[-1] > growth[0], (
        "growth is not compounding, so z1 is not being re-derived from the "
        "predicted state"
    )


def test_stage2_uses_no_f_theta_and_no_resync():
    """The whole point of the row: what the latent pair does with no
    correction network and no help from the real frames."""
    import inspect
    src = inspect.getsource(cf.compute_stage2_trajectory)
    code = src.split('"""')[2]
    assert "f_theta" not in code, "stage 2 must not touch f_theta"
    assert "z0 = z0 + z1_use * dt" in code, "the step is not z0 += z1*dt (z1_use == z1 in t-mode)"
    assert 'ae_encoder(x_pred, theta=theta_encode)["deriv"]' in code, (
        "z1 is not re-derived from the predicted state"
    )


def test_the_stage2_row_sits_between_causal_and_the_models(monkeypatch, tmp_path):
    """
    Ordered by how much machinery each row uses: backward difference, then
    z1 alone, then z1 + f_theta. Reading down a column shows what each layer
    buys.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    _traj_stub(monkeypatch)
    monkeypatch.setattr(cf, "compute_causal_trajectory",
                         lambda *a, **k: [np.zeros((8, 8)) for _ in range(5)])
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plt, "subplots", spy)
    cf._trajectory_figure(Path("T625_n050_s599"), list(range(5)),
                           _model("128x128-stage3a"), _model("128x128-stage3b"),
                           "cpu", False, "T", tmp_path / "t.png")
    axes = captured["axes"]
    labels = [axes[r, 0].get_ylabel().replace("\n", " ") for r in range(5)]
    assert labels == ["real", "previous derivative (linear extrapolation)",
                       "stage 2 (z0 + z1 dt)", "stage 3a", "stage 3b"], labels
    # and it carries its own per-frame numbers
    assert "loss=" in axes[2, 1].get_title()


def test_stage2_appears_on_every_stats_panel(monkeypatch, tmp_path):
    """Present for every window (unlike causal), so it needs no separate
    dt/T arrays and belongs on all eight panels."""
    stats, a, b = _stats_with_causal(monkeypatch, n=60)
    assert len(stats["loss_stage2"]) == len(stats["loss_a"])
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    for r in range(2):
        for c in range(4):
            labels = [t.get_text() for t in axes[r, c].get_legend().get_texts()]
            assert any("stage 2" in t for t in labels), (
                f"panel [{r},{c}] has no stage-2 curve: {labels}"
            )


def test_asking_for_more_windows_than_exist_uses_all_of_them(monkeypatch, tmp_path):
    """
    "Give me 5000" means "give me as many as you have". Aborting after the
    dataset is already built throws that work away, and a short sample is
    what silently truncated the temperature axis: 50 windows cannot cover 33
    sweep temperatures, so the vs-T panel stopped at 0.98 with no hint that
    the hot end was missing entirely.
    """
    import inspect
    src = inspect.getsource(cf._select_windows)
    assert "n_samples = len(dataset)" in src, (
        "an over-large request still aborts instead of using the whole "
        "population"
    )
    assert "raise ValueError(f\"only {len(dataset)} windows available" not in src
    # and it says so, so the window count is never a surprise
    assert "using all of them" in src


def test_stats_legends_are_ordered_like_the_trajectory_rows(monkeypatch, tmp_path):
    """
    causal, stage 2, 3a, 3b -- ordered by how much machinery each uses, the
    same order the trajectory figure stacks its rows, so the two figures'
    legends read alike.
    """
    stats, a, b = _stats_with_causal(monkeypatch, n=60)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    expected = ["previous derivative", "stage 2", "stage 3a", "stage 3b"]
    for r in range(2):
        for c in range(4):
            labels = [t.get_text()
                      for t in axes[r, c].get_legend().get_texts()]
            assert len(labels) == 4, f"panel [{r},{c}]: {labels}"
            for got, want in zip(labels, expected):
                assert got.startswith(want), (
                    f"panel [{r},{c}] legend order {labels}, expected "
                    f"{expected}"
                )


def test_the_correlation_axis_never_drops_below_minus_twenty():
    """
    A model that leaves the manifold can go arbitrarily anticorrelated, and
    letting one such window set the axis to -300% squeezes the 0..100 band --
    where every meaningful difference lives -- into the top sliver. Curves
    below -20% run off the bottom instead.
    """
    import matplotlib.pyplot as plt
    for data_lo, expected_floor in ((-5.0, None), (-50.0, -20.0),
                                     (-300.0, -20.0)):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [data_lo, 100.0])
        cf._corr_axis(ax)
        floor = ax.get_ylim()[0]
        assert floor >= -20.0 - 1e-9, (
            f"data reaching {data_lo}% produced a floor of {floor}"
        )
        if expected_floor is not None:
            assert floor == pytest.approx(expected_floor)
        plt.close(fig)


def test_the_shared_correlation_floor_is_also_clamped(monkeypatch, tmp_path):
    """The row union takes the minimum across panels, so it needs the same
    clamp -- otherwise one panel's dive re-imposes the range the per-panel
    pin just refused."""
    def traj(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        n = len(steps)
        i = zlib.crc32(str(run_dir).encode()) % 997
        rng = np.random.default_rng(i)
        real = [rng.normal(size=(8, 8)) * (1 + 0.05 * k) for k in range(n)]
        # violently anticorrelated after a few steps
        pred = [real[k] if k < 2 else -real[k] * 8.0 for k in range(n)]
        return real, pred, [400.0 * (1 + i % 6)] * (n - 1)

    monkeypatch.setattr(cf, "compute_trajectory", traj)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _flat_stage2)
    _stub_metadata(monkeypatch)
    a, b = _model("128x128-stage3a"), _model("128x128-stage3b")
    windows = [(f"T{550 + i * 5}_n020_s{i}", list(range(9))) for i in range(60)]
    stats = cf.collect_stats(a, b, windows, "cpu", False)
    axes = _figure_axes(monkeypatch, stats, a, b, tmp_path)
    for c in range(4):
        assert axes[1, c].get_ylim()[0] >= -20.0 - 1e-6, (
            f"correlation panel {c} floor is {axes[1, c].get_ylim()[0]:.1f}%"
        )


def _lds_with_nonzero_f(seed=0):
    """A LatentDynamics whose f_theta is clearly nonzero -- at init the net
    outputs ~0, so a scale sweep on a fresh model would prove nothing.

    Seeded through its OWN generator, not torch.manual_seed: the global RNG
    is shared, so seeding it here made these fixtures depend on how many
    random draws earlier tests had made. One ordering produced a model whose
    f happened to satisfy a later assertion, and a real mutation slipped
    through the batch run while failing in isolation.
    """
    from models.latent_dynamics import LatentDynamics
    gen = torch.Generator().manual_seed(seed)
    lds = LatentDynamics(latent_channels=2, n_theta=N_THETA, latent_spatial=4,
                          hidden_dim=16, n_hidden_layers=1)
    lds.eval()
    with torch.no_grad():
        for p in lds.net.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * 0.5)
    return lds


def test_scale_zero_reproduces_stage_two_exactly():
    """
    THE PREMISE OF THE SWEEP. Stage 2 is nested inside stage 3 at f == 0:
    z0 += z1*h + f*h^2/2 and z1 += (f+f')*h/2 both collapse to the pure
    first-order step. If scale 0 were not exactly stage 2, the sweep would
    not connect the two models and its shape would mean nothing.
    """
    lds = _lds_with_nonzero_f()
    B, n = 1, 4
    z0 = torch.randn(B, 2, 4, 4)
    z1 = torch.randn(B, 2, 4, 4) * 0.01
    z1_seq = z1.unsqueeze(1).repeat(1, n + 1, 1, 1, 1)
    dts = torch.full((B, n), 100.0)
    theta = torch.zeros(B, N_THETA)
    with torch.no_grad():
        assert lds.f(z0, z1, theta).abs().mean() > 1e-3, "fixture f is ~0"
        with cf.scaled_f_theta(lds, 0.0):
            got = lds.rollout(z0, z1_seq, dts, theta, z1_resync=False)
        want = [z0]
        cur = z0.clone()
        for i in range(n):
            cur = cur + z1 * dts[:, i].view(-1, 1, 1, 1)
            want.append(cur)
        want = torch.stack(want, dim=1)
    assert torch.allclose(got, want, atol=1e-6), (
        f"scale 0 is not the first-order step (max diff "
        f"{(got - want).abs().max():.3g})"
    )


def test_the_scale_sweep_moves_between_the_two_models():
    """Intermediate scales must land strictly between stage 2 and the trained
    model -- otherwise the sweep has only two usable points."""
    lds = _lds_with_nonzero_f()
    B, n = 1, 4
    z0 = torch.randn(B, 2, 4, 4)
    z1 = torch.randn(B, 2, 4, 4) * 0.01
    z1_seq = z1.unsqueeze(1).repeat(1, n + 1, 1, 1, 1)
    dts = torch.full((B, n), 100.0)
    theta = torch.zeros(B, N_THETA)
    outs = {}
    with torch.no_grad():
        for scale in (0.0, 0.5, 1.0):
            with cf.scaled_f_theta(lds, scale):
                outs[scale] = lds.rollout(z0, z1_seq, dts, theta,
                                           z1_resync=False)
    d = lambda x: float((x - outs[0.0]).abs().max())
    assert 0.0 == d(outs[0.0]) < d(outs[0.5]) < d(outs[1.0]), (
        f"departure from stage 2 is not monotone in the scale: "
        f"{d(outs[0.0]):.3g}, {d(outs[0.5]):.3g}, {d(outs[1.0]):.3g}"
    )


def test_scaling_f_theta_is_undone_afterwards():
    """A leaked scaling would silently corrupt every later figure in the
    same run."""
    lds = _lds_with_nonzero_f()
    z0 = torch.randn(1, 2, 4, 4)
    z1 = torch.randn(1, 2, 4, 4) * 0.01
    theta = torch.zeros(1, N_THETA)
    with torch.no_grad():
        before = lds.f(z0, z1, theta).clone()
        with cf.scaled_f_theta(lds, 0.0):
            assert float(lds.f(z0, z1, theta).abs().max()) == 0.0
        after = lds.f(z0, z1, theta)
    assert torch.allclose(before, after), "f_theta was not restored"


def test_the_scaling_survives_an_exception():
    """Restored via finally, so a failure mid-sweep cannot leave the model
    scaled for the rest of the run."""
    lds = _lds_with_nonzero_f()
    z0 = torch.randn(1, 2, 4, 4)
    z1 = torch.randn(1, 2, 4, 4) * 0.01
    theta = torch.zeros(1, N_THETA)
    with torch.no_grad():
        before = lds.f(z0, z1, theta).clone()
    try:
        with cf.scaled_f_theta(lds, 0.0):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    with torch.no_grad():
        assert torch.allclose(before, lds.f(z0, z1, theta))


def test_scaling_f_theta_to_zero_reproduces_stage_2_exactly():
    """
    THE NESTING, made operational. Stage 2 is stage 3 with f == 0: the
    integrator's z0 += z1*h + f*h^2/2 and z1 += (f+f')*h/2 both collapse to
    the plain Euler step. So a stage-3 model losing to stage 2 under
    propagation cannot be a hypothesis-class problem, and sweeping the scale
    measures whether the damage is monotone in how much f_theta is applied.
    """
    import torch
    from models.latent_dynamics import LatentDynamics

    m = LatentDynamics(latent_channels=2, n_theta=N_THETA, hidden_dim=8,
                        n_hidden_layers=1)
    m.eval()
    # a non-trivial, state-dependent curvature (an untrained net outputs 0,
    # which would make this test vacuous)
    m.f = lambda z0, z1, theta: 1e-4 * torch.tanh(z0)
    z0 = torch.randn(1, 2, 8, 8)
    z1 = torch.randn(1, 2, 8, 8)
    th = torch.zeros(1, N_THETA)
    dts = torch.full((1, 3), 100.0)
    z1s = z1.unsqueeze(1).expand(-1, 4, -1, -1, -1).contiguous()

    euler = [z0]
    for i in range(3):
        euler.append(euler[-1] + z1 * dts[0, i])
    euler = torch.stack(euler, dim=1)

    with torch.no_grad():
        with cf.scaled_f_theta(m, 0.0):
            at_zero = m.rollout(z0, z1s, dts, th, z1_resync=False)
        assert torch.allclose(at_zero, euler, atol=1e-5), (
            "scale 0 does not reproduce the plain Euler step"
        )
        # and the deviation from stage 2 grows with the scale
        devs = []
        for lam in (0.25, 0.5, 1.0):
            with cf.scaled_f_theta(m, lam):
                r = m.rollout(z0, z1s, dts, th, z1_resync=False)
            devs.append(float((r - at_zero).abs().mean()))
    assert devs[0] < devs[1] < devs[2], (
        f"deviation from stage 2 is not monotone in the scale: {devs}"
    )


def _lds_alpha(alpha=1.5, max_substeps=512, seed=1):
    from models.latent_dynamics import LatentDynamics
    torch.manual_seed(0)
    lds = LatentDynamics(latent_channels=2, n_theta=N_THETA, latent_spatial=4,
                          hidden_dim=16, n_hidden_layers=1, alpha=alpha,
                          max_substeps=max_substeps)
    lds.eval()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in lds.net.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * 0.5)
    return lds


def test_smaller_alpha_takes_more_substeps_and_is_restored():
    """
    The h -> 0 limit at fixed field: n ~ f*dt/(alpha*|z1|), so a smaller
    alpha must produce MORE substeps -- and the original alpha and cap must
    come back afterwards, or every later figure in the run integrates with
    the sweep's last setting.
    """
    lds = _lds_alpha()
    z0 = torch.randn(1, 2, 4, 4)
    z1 = torch.randn(1, 2, 4, 4) * 0.01
    z1_seq = z1.unsqueeze(1).repeat(1, 5, 1, 1, 1)
    dts = torch.full((1, 4), 500.0)
    theta = torch.zeros(1, N_THETA)

    def substeps_used():
        lds._substep_total = 0
        with torch.no_grad():
            lds.rollout(z0, z1_seq, dts, theta, z1_resync=False)
        return lds._substep_total

    base = substeps_used()
    with cf.overridden_alpha(lds, 0.15, max_substeps=4096):
        fine = substeps_used()
    assert fine > 2 * base, (
        f"alpha 1.5 -> 0.15 went from {base} to {fine} substeps; the "
        f"override is not shrinking h"
    )
    assert lds.alpha == 1.5 and lds.max_substeps == 512, "not restored"
    assert substeps_used() == base, "behaviour after the context differs"


def test_alpha_override_is_restored_on_exception():
    lds = _lds_alpha()
    try:
        with cf.overridden_alpha(lds, 0.05, max_substeps=4096):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert lds.alpha == 1.5 and lds.max_substeps == 512


def test_alpha_sweep_refuses_a_fixed_substep_model():
    """
    Established earlier in the project: a 3a trained at n_substeps=1 treats
    f as a dt-AVERAGED corrector, not a vector field, and sub-stepping it is
    meaningless (monotonically worse). The sweep must refuse rather than
    print a table that would be read as an h -> 0 limit.
    """
    from models.latent_dynamics import LatentDynamics
    torch.manual_seed(0)
    lds = LatentDynamics(latent_channels=2, n_theta=N_THETA, latent_spatial=4,
                          hidden_dim=16, n_hidden_layers=1, n_substeps=1)
    out = cf.sweep_alpha({"f_theta": lds, "label": "m", "ae": None,
                           "ae_config": {}}, [], "cpu", False)
    assert out == {}, "a fixed-n_substeps model was swept anyway"


def test_alpha_sweep_reports_the_substep_cap_hits(monkeypatch):
    """
    At small alpha the substep count saturates at max_substeps and h STOPS
    shrinking -- a 'converged' verdict at a saturated point would be an
    artifact. The clamp count must travel with each alpha's numbers.
    """
    lds = _lds_alpha()
    calls = []

    def fake_traj(run_dir, steps, ae, f_theta, ae_config, device, z1_resync):
        calls.append(f_theta.alpha)
        rng = np.random.default_rng(0)
        real = [rng.normal(size=(8, 8)) * (1 + 0.1 * k) for k in range(len(steps))]
        return real, [f + 0.01 for f in real], [100.0] * (len(steps) - 1)

    monkeypatch.setattr(cf, "compute_trajectory", fake_traj)
    # poison the counter BEFORE the sweep: a sweep that fails to reset it per
    # alpha reports this stale number instead of each alpha's own clamps.
    lds.n_substeps_clamped = 12345
    out = cf.sweep_alpha({"f_theta": lds, "label": "m", "ae": None,
                           "ae_config": {}},
                          [("r0", list(range(5)))], "cpu", False,
                          alphas=(1.5, 0.15))
    assert out["alphas"] == [1.5, 0.15]
    assert len(out["clamped"]) == 2
    # the fake trajectory never calls the integrator, so each alpha's own
    # clamp count is exactly 0 -- any 12345 leaking through is the stale one
    assert out["clamped"] == [0, 0], (
        f"clamp counts {out['clamped']} include a stale pre-sweep value; "
        f"the counter is not reset per alpha"
    )
    assert sorted(set(calls)) == [0.15, 1.5], (
        f"the trajectory ran at alphas {sorted(set(calls))}, not the sweep's"
    )


def test_compare_statistics_runs_without_touching_the_panel_tool(monkeypatch, tmp_path):
    """
    The split's point: the statistics tool is independently callable and
    NEVER draws a panel. It takes n_steps directly, so no panel window is
    selected just to define a horizon.
    """
    from pathlib import Path
    _stats_stub(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    monkeypatch.setattr(
        cf, "_select_windows",
        lambda m, n, ns, seed, mx, dev, t0_range=None: [(Path(f"T{500 + i}_n020_s{i}"),
                                          list(range(ns + 1)))
                                         for i in range(max(n, 1))])
    # if the panel tool were reached it would call plt.subplots(_, 8); make
    # that a failure
    import matplotlib.pyplot as plt
    real_subplots = plt.subplots

    def guard(*a, **k):
        ncols = (a[1] if len(a) > 1 else k.get("ncols"))
        assert ncols != 8, "compare_statistics drew the 8-column panel figure"
        return real_subplots(*a, **k)

    monkeypatch.setattr(plt, "subplots", guard)
    out, _ = cf.compare_statistics("128x128-stage3a", "128x128-stage3b",
                                    n_stats=8, n_steps=4, seed=2, device="cpu")
    written = sorted(p.name for p in out.parent.iterdir())
    assert written == [out.stem + "-stats.png"], written


def test_compare_panels_runs_without_computing_statistics(monkeypatch, tmp_path):
    """The panel tool draws its figure and never runs collect_stats or the
    sweeps -- it has no n_stats parameter at all."""
    from pathlib import Path
    _stats_stub(monkeypatch)
    monkeypatch.setattr(cf, "_load_model", lambda p, d: _model(str(p)))
    monkeypatch.setattr(
        cf, "_select_windows",
        lambda m, n, ns, seed, mx, dev, t0_range=None: [(Path(f"T{500 + i}_n020_s{i}"),
                                          list(range(ns + 1)))
                                         for i in range(max(n, 1))])
    called = {"stats": False}
    monkeypatch.setattr(cf, "collect_stats",
                         lambda *a, **k: called.__setitem__("stats", True))
    out, _ = cf.compare_panels("128x128-stage3a", "128x128-stage3b",
                                n_samples=2, n_steps=4, seed=2, device="cpu")
    assert not called["stats"], "compare_panels computed statistics"
    assert out.name.endswith(".png") and "-stats" not in out.name


def test_stats_only_and_panels_only_flags_are_exclusive(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv",
                        ["x", "a", "b", "--panels-only", "--stats-only"])
    with __import__("pytest").raises(SystemExit):
        cf.main()


def test_t0_range_selects_only_windows_starting_in_band(monkeypatch):
    """--t0-range must keep ONLY windows whose starting step t0 is in [lo,hi]
    and raise when the band is empty. Every t0-split verdict in the project
    rests on this filter selecting the right band, so it is tested against the
    REAL _select_windows (the dataset is stubbed, the filter logic is not)."""
    from pathlib import Path

    class _FakeDS:
        # window i starts at step t0s[i]; window_info(i) -> (run_dir, steps)
        t0s = [50, 100, 150, 200, 250]
        def __len__(self): return len(self.t0s)
        def window_info(self, i):
            t0 = self.t0s[i]
            return (Path(f"run{i}"), [t0, t0 + 10])

    monkeypatch.setattr(cf, "MicrostructureEvolutionDataset",
                        lambda *a, **k: _FakeDS())
    monkeypatch.setattr(cf.load, "validate_run_dirs", lambda dirs, **k: dirs)
    monkeypatch.setattr(cf, "default_latent_cache_dir", lambda *a, **k: None)
    model = {"path": "p", "ae_encoder": None, "ae_config": {},
             "ck": {"data_config": {}, "test_dirs": ["d"]}}

    # in-band [100, 200] -> only t0 in {100,150,200} survive
    got = cf._select_windows(model, n_samples=10, n_steps=1, seed=0,
                             max_dt=None, device="cpu", t0_range=(100, 200))
    starts = sorted(steps[0] for _run, steps in got)
    assert starts == [100, 150, 200], f"out-of-band windows leaked in: {starts}"

    # empty band -> raise (not silently return nothing)
    import pytest
    with pytest.raises(ValueError, match="no windows start in t0-range"):
        cf._select_windows(model, n_samples=10, n_steps=1, seed=0,
                           max_dt=None, device="cpu", t0_range=(1000, 2000))


def test_stage2_dx_column_uses_its_own_scale_not_the_shared_one(monkeypatch, tmp_path):
    """The stage-2 dx column (col 2) must get its OWN scale, wider than real dx
    (col 1) when z1 diverges -- else a diverged z1 saturates the whole panel to
    a flat block. A regression putting it back on the shared d_lo/d_hi would
    make the two clims equal; this asserts they differ when stage 2 blows up."""
    import numpy as np
    import matplotlib.pyplot as plt
    _stub_models(monkeypatch)

    # stage-2 trajectory that DIVERGES (>> real dx), like z1 past its horizon
    def _big_stage2(run_dir, steps, ae, ae_config, device, time_coordinate="t"):
        rng = np.random.default_rng(0)
        sz = (ae_config or {}).get("size", 8)
        return [rng.normal(size=(sz, sz)) * 1e6 for _ in range(len(steps))]
    monkeypatch.setattr(cf, "compute_stage2_trajectory", _big_stage2)

    captured = {}
    orig = plt.subplots
    def spy(*a, **k):
        fig, axes = orig(*a, **k)
        captured.setdefault("axes", axes)
        return fig, axes
    monkeypatch.setattr(plt, "subplots", spy)

    cf.compare_f_theta("128x128-stage3a", "128x128-stage3b",
                       fixed_windows=["r:1:2:3", "q:4:5:6"],
                       output_path=tmp_path / "f.png", device="cpu")
    axes = captured["axes"]
    real_clim = axes[0, 1].get_images()[0].get_clim()      # col 1 = real dx
    stage2_clim = axes[0, 2].get_images()[0].get_clim()     # col 2 = stage 2 dx
    assert stage2_clim[1] > real_clim[1] * 100, (
        "stage-2 dx column is not on its own scale -- a diverged z1 should give "
        "it a far wider range than real dx, not the shared one")
