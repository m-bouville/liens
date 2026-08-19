"""Option-A stage-2 rollout comparison: N AE-family checkpoints, one stage-2
rollout curve each, shared windows. Stubs the loaders/trajectory so the test
exercises the ORCHESTRATION (N checkpoints, shared window set, label
disambiguation, figure written) without real checkpoints."""
import numpy as np
import pytest

import evaluation.compare_f_theta as cf


@pytest.fixture(autouse=True)
def _never_write_outside_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(cf, "_PYTHON_ROOT", tmp_path / "python")


def _stub(monkeypatch, n_steps=10):
    """Stub loading + windows + trajectory. Each checkpoint gets a distinct
    error level so the curves differ; windows are shared."""
    def fake_load(path, device):
        return {"path": path, "ck": {}, "ae": str(path), "ae_encoder": None,
                "ae_config": {"size": 8}, "label": cf._parse_stem(path.stem)[1]}

    def fake_windows(model, n, steps, seed, max_dt, device):
        return [("run_A", list(range(steps + 1))),
                ("run_B", list(range(steps + 1)))]

    def fake_traj(run_dir, steps, ae, ae_config, device):
        # compute_stage2_trajectory returns the predicted trajectory ONLY --
        # a single list of frames, NOT a (real, pred, dt) tuple. The stub must
        # match that, or it hides real unpacking bugs.
        base = (abs(hash(ae)) % 10) / 100.0
        rng = np.random.default_rng(abs(hash((ae, run_dir))) % (2**32))
        real = [rng.normal(size=(8, 8)) for _ in range(len(steps))]
        pred = [r + base * (k + 1) * rng.normal(size=(8, 8))
                for k, r in enumerate(real)]
        return pred

    def fake_read_phi(path, nx, ny):
        # real frames are read from disk in the real function; stub returns a
        # deterministic frame per snapshot path.
        rng = np.random.default_rng(abs(hash(str(path))) % (2**32))
        return rng.normal(size=(8, 8))

    def fake_causal(run_dir, steps, ae, ae_config, device):
        # causal baseline: also a pred-frame list, or None when undefined.
        base = (abs(hash(("causal", ae))) % 10) / 100.0
        rng = np.random.default_rng(abs(hash(("c", ae, run_dir))) % (2**32))
        return [base * (k + 1) * rng.normal(size=(8, 8))
                for k in range(len(steps))]

    class _Meta:
        dt = 0.05
        temperature = 0.8
        T0 = 1.0

    monkeypatch.setattr(cf, "_load_stage2_ae", fake_load)
    monkeypatch.setattr(cf, "_select_windows", fake_windows)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", fake_traj)
    monkeypatch.setattr(cf, "compute_causal_trajectory", fake_causal)
    monkeypatch.setattr(cf.load, "read_phi_half", fake_read_phi)
    monkeypatch.setattr(cf.load, "snapshot_filename", lambda s: f"t{s:07d}")
    monkeypatch.setattr(cf.load, "read_metadata", lambda p: _Meta())


def test_compares_three_checkpoints_one_curve_each(monkeypatch, tmp_path, capsys):
    from pathlib import Path
    _stub(monkeypatch)
    paths = [Path("checkpoints/128x128-stage2-20260812_20h08.pt"),
             Path("checkpoints/128x128-stage2-20260818_13h54.pt"),
             Path("checkpoints/128x128-stage2-20260819_11h20.pt")]
    out = cf.compare_stage2_rollouts(paths, n_stats=2, n_steps=10,
                                     output_path=tmp_path / "cmp.png")
    assert out.exists()
    printed = capsys.readouterr().out
    # one "final median" line per checkpoint
    assert printed.count("final median loss") == 3
    # the three distinct timestamps survive into labels, now prettified to
    # "(DD/MM at HH:MM)" -- all 2026 (current year) so no year shown
    assert "20:08" in printed and "13:54" in printed and "11:20" in printed
    assert "12/08" in printed and "18/08" in printed and "19/08" in printed
    # and the corr is printed with a decimal, matching the loss precision
    import re
    assert re.search(r"final median corr -?\d+\.\d%", printed), printed


def test_single_checkpoint_is_allowed(monkeypatch, tmp_path):
    from pathlib import Path
    _stub(monkeypatch)
    out = cf.compare_stage2_rollouts([Path("x-stage2-20260819_11h20.pt")],
                                     n_stats=2, n_steps=5,
                                     output_path=tmp_path / "one.png")
    assert out.exists()


def test_empty_paths_rejected(monkeypatch, tmp_path):
    _stub(monkeypatch)
    with pytest.raises(ValueError, match="at least one"):
        cf.compare_stage2_rollouts([], output_path=tmp_path / "n.png")


def test_shared_windows_from_first_checkpoint(monkeypatch, tmp_path):
    """Every checkpoint must be measured on the SAME windows -- selected once
    from the first. Assert _select_windows is called exactly once."""
    from pathlib import Path
    _stub(monkeypatch)
    calls = []
    orig = cf._select_windows
    def counting(model, *a, **k):
        calls.append(model["path"])
        return orig(model, *a, **k)
    monkeypatch.setattr(cf, "_select_windows", counting)
    cf.compare_stage2_rollouts(
        [Path("a-stage2-20260812_20h08.pt"), Path("b-stage2-20260818_13h54.pt")],
        n_stats=2, n_steps=4, output_path=tmp_path / "s.png")
    assert len(calls) == 1, "windows must be selected once, from the first checkpoint"


def test_compute_stage2_trajectory_returns_a_plain_list_not_a_tuple():
    """Contract guard: compare_stage2_rollouts relies on
    compute_stage2_trajectory returning the predicted-frame LIST only (a real
    call previously crashed because the caller assumed a 3-tuple). This pins
    the real function's return shape so a future signature change is caught
    here rather than at runtime on a 1h50 checkpoint."""
    import inspect
    src = inspect.getsource(cf.compute_stage2_trajectory)
    # the function's sole `return` is `return out`, and `out` is built as a
    # list of frames -- assert it is not returning a tuple.
    assert "return out" in src
    # and the docstring/structure never packs real frames or dt into it
    returns = [l.strip() for l in src.splitlines() if l.strip().startswith("return")]
    assert returns == ["return out"], (
        f"compute_stage2_trajectory return shape changed: {returns} -- "
        f"compare_stage2_rollouts unpacks it as a plain frame list")


def test_end_to_end_on_real_ae_checkpoints(tmp_path, monkeypatch):
    """The REAL, un-stubbed path: build actual AE checkpoints and run the
    whole comparison. This is what catches signature-mismatch bugs (like the
    compute_stage2_trajectory return shape) that a stubbed test can hide."""
    import sys
    from pathlib import Path
    # reuse the real AE-checkpoint + run-dir builders from the integration test
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_evaluation_reconstruction_integration import _save_ae_checkpoint

    # Build two run dirs with enough frames for window_length=steps+1.
    import numpy as np
    from utils import load_datasets as load

    def make_run(base, name, n_steps=6, size=8):
        d = base / name
        d.mkdir()
        steps = [i * 1000 for i in range(n_steps)]
        (d / "metadata.txt").write_text("\n".join([
            f"directory = {name}", "code version = t", "status = complete",
            f"Nx = {size}", f"Ny = {size}", "dt = 0.05", f"steps = {steps[-1]}",
            f"save_steps = {' '.join(map(str, steps))}",
            "a0 = 1.0", "b = 1.0", "T0 = 1.0", "temperature = 0.8",
            "kappa = 0.2", "mobility = 0.05", "phi0 = 0.0", "noise = 0.01",
            "seed = 1", "equation = allen_cahn", "solver = explicit", ""]))
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        import pandas as pd
        for s in steps:
            rng.standard_normal((size, size)).astype("<f2").tofile(
                d / load.snapshot_filename(s))
        pd.DataFrame({"stdev_phi": [0.5] * len(steps),
                      "avg_phi": [0.0] * len(steps)}, index=steps
                     ).rename_axis("step").to_csv(d / "statistics.csv")
        return d

    runs = [make_run(tmp_path, f"T800_n010_s{i}") for i in (1, 2, 3)]
    ck1 = tmp_path / "128x128-stage2-20260812_20h08.pt"
    ck2 = tmp_path / "128x128-stage2-20260818_13h54.pt"
    for ck in (ck1, ck2):
        _save_ae_checkpoint(ck, runs, size=8, base_channels=4,
                            latent_channels=4, latent_spatial_size=2,
                            multi_stream=True)

    out = cf.compare_stage2_rollouts([ck1, ck2], n_stats=3, n_steps=4,
                                     output_path=tmp_path / "e2e.png",
                                     device="cpu")
    assert out.exists()


def test_diverged_loss_is_capped_not_inf(monkeypatch, tmp_path):
    """A rollout that blows up makes ReconLoss (squared) overflow float32 to
    inf; inf poisons the quartile aggregation. The per-window loss must be
    capped to a large FINITE sentinel so 'diverged' reads as 'very bad and
    finite', keeping medians/quartiles meaningful."""
    from pathlib import Path
    import numpy as np

    def fake_load(path, device):
        return {"path": path, "ck": {}, "ae": str(path), "ae_encoder": None,
                "ae_config": {"size": 8}, "label": cf._parse_stem(path.stem)[1]}

    def fake_windows(model, n, steps, seed, max_dt, device):
        return [("r1", list(range(steps + 1)))]

    def blown_up_traj(run_dir, steps, ae, ae_config, device):
        # frames escalate to a magnitude whose square overflows float32
        return [np.full((8, 8), 10.0 ** (20 + k), dtype=np.float64)
                for k in range(len(steps))]

    class _Meta:
        dt = 0.05; temperature = 0.8; T0 = 1.0

    monkeypatch.setattr(cf, "_load_stage2_ae", fake_load)
    monkeypatch.setattr(cf, "_select_windows", fake_windows)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", blown_up_traj)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a: None)
    monkeypatch.setattr(cf.load, "read_phi_half",
                        lambda p, nx, ny: np.zeros((8, 8)))
    monkeypatch.setattr(cf.load, "snapshot_filename", lambda s: f"t{s:07d}")
    monkeypatch.setattr(cf.load, "read_metadata", lambda p: _Meta())

    # capture the models via a spy on the figure so we can inspect the losses
    captured = {}
    orig_fig = cf._stage2_rollout_figure
    def spy(models, dt_totals, temps, n_steps, output_path):
        captured["models"] = models
        return orig_fig(models, dt_totals, temps, n_steps, output_path)
    monkeypatch.setattr(cf, "_stage2_rollout_figure", spy)

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cf.compare_stage2_rollouts([Path("x-stage2-20260819_11h20.pt")],
                                   n_stats=1, n_steps=3,
                                   output_path=tmp_path / "d.png", device="cpu")
    overflow = [w for w in caught if "overflow" in str(w.message).lower()]
    assert not overflow, f"overflow warning leaked to the user: {overflow}"
    losses = captured["models"][0]["step_losses"]
    assert np.isfinite(losses).all(), "diverged losses must be finite (capped)"
    assert losses.max() >= 1e11, "the diverged window should read as very-bad"


def test_progress_bar_silent_for_small_runs(monkeypatch, tmp_path, capsys):
    """Below the 500-eval threshold, no progress line is written."""
    from pathlib import Path
    _stub(monkeypatch)
    cf.compare_stage2_rollouts([Path("a-stage2-20260812_20h08.pt")],
                               n_stats=2, n_steps=3,
                               output_path=tmp_path / "s.png")
    out = capsys.readouterr().out
    assert "rollout progress" not in out


def test_pretty_label_formats_timestamp():
    from evaluation.compare_f_theta import _pretty_label, _labels_need_year
    labs = ["stage 2-20260812_20h08", "stage 2-20260819_11h20"]
    ny = _labels_need_year(labs)  # both 2026 -> current year -> no year shown
    assert ny is False
    assert _pretty_label(labs[0], ny) == "stage 2 (12/08 at 20:08)"
    # a non-current year forces the year in
    mixed = labs + ["stage 2-20251201_09h00"]
    assert _labels_need_year(mixed) is True
    assert _pretty_label("stage 2-20251201_09h00", True) == \
        "stage 2 (01/12/2025 at 09:00)"
    # a label with no timestamp is returned unchanged
    assert _pretty_label("stage 3a", False) == "stage 3a"


def test_loss_panels_y_range_is_capped_to_medians(monkeypatch, tmp_path):
    """A diverged window must NOT blow up the loss axis -- the y-range comes
    from the median curves (via _ylim_from_medians), same as _stats_figure.
    Without this the axis spans to 1e30 and flattens the real structure."""
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")

    def fake_load(path, device):
        return {"path": path, "ck": {}, "ae": str(path), "ae_encoder": None,
                "ae_config": {"size": 8}, "label": cf._parse_stem(path.stem)[1]}

    def fake_windows(model, n, steps, seed, max_dt, device):
        return [(f"r{i}", list(range(steps + 1))) for i in range(6)]

    def fake_traj(run_dir, steps, ae, ae_config, device):
        # most windows fine (~small), ONE window diverges hugely
        rng = np.random.default_rng(abs(hash(run_dir)) % (2**32))
        scale = 1e15 if run_dir == "r0" else 0.1
        base = rng.normal(size=(8, 8))
        return [base + scale * (k + 1) * rng.normal(size=(8, 8))
                for k in range(len(steps))]

    class _Meta:
        dt = 0.05; temperature = 0.8; T0 = 1.0

    captured = {}
    orig = cf._stage2_rollout_figure
    def spy(models, dt_totals, temps, n_steps, output_path):
        orig(models, dt_totals, temps, n_steps, output_path)
        captured["done"] = True
    monkeypatch.setattr(cf, "_load_stage2_ae", fake_load)
    monkeypatch.setattr(cf, "_select_windows", fake_windows)
    monkeypatch.setattr(cf, "compute_stage2_trajectory", fake_traj)
    monkeypatch.setattr(cf, "compute_causal_trajectory", lambda *a: None)
    monkeypatch.setattr(cf.load, "read_phi_half",
                        lambda p, nx, ny: np.zeros((8, 8)))
    monkeypatch.setattr(cf.load, "snapshot_filename", lambda s: f"t{s:07d}")
    monkeypatch.setattr(cf.load, "read_metadata", lambda p: _Meta())

    # patch _ylim_from_medians to record what range it sets
    ranges = []
    orig_ylim = cf._ylim_from_medians
    def rec_ylim(ax, medians):
        orig_ylim(ax, medians)
        ranges.append(ax.get_ylim())
    monkeypatch.setattr(cf, "_ylim_from_medians", rec_ylim)

    cf.compare_stage2_rollouts([Path("x-stage2-20260819_11h20.pt")],
                               n_stats=6, n_steps=3,
                               output_path=tmp_path / "cap.png", device="cpu")
    # the loss y-range top must be far below the 1e15-diverged window (the cap
    # keeps it near the median ~0.1..few, not 1e15)
    assert ranges, "_ylim_from_medians was not applied to the loss panels"
    top = max(hi for lo, hi in ranges)
    assert top < 1e10, f"loss axis not capped to medians: top={top:g}"
