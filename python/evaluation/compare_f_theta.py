"""
Side-by-side f_theta comparison: TWO checkpoints, IDENTICAL windows.

Seven columns per sample row, labelled with the parsed stage names:

    state(t) | real dx | stage 2 dx (z0+z1 dt) | pred dx (3a) | pred dx (3b)
             | error stage 3a | error stage 3b | stage 3b - stage 3a

The last column is the DIFFERENCE OF PREDICTIONS -- "is there a pattern
in what B changed?" -- on the SAME scale as the two error columns, which
it reads directly against: since both predictions share one real dx, it
equals error_B - error_A exactly, so it is also the map of where B is
worse.

Exists because every A-vs-B judgement so far has stumbled on the same
trap, three times: the two models' own figures sample DIFFERENT windows
(each checkpoint's config filters its own population, 3a truncates
3-step windows to its window_length=2, dt ranges differ), so the panels
invite a comparison they cannot support. Here:

  * ONE window list, chosen once, fed to both models UNTRUNCATED --
    both chain the SAME transitions over the same horizon. A 3a model
    chaining 2 steps at inference is legitimate; that is precisely what
    "using 3a for rollout" means.
  * z1_resync forced to the SAME value for both (default False, the
    inference regime), never read per-checkpoint.
  * real dx and BOTH predictions share one symmetric color scale
    (derived from real dx alone, so it cannot flatter either model);
    BOTH error columns share another, from the worse of the two -- the
    honest scale, since the whole point is which error panel is fuller.

Each model decodes through its OWN AE checkpoint; normally that is the
shared ancestor, and the header says if the two differ.

Usage (from python/):

    python -m evaluation.compare_f_theta \\
        checkpoints/stage3a/128x128-stage3a.pt \\
        checkpoints/stage3b/128x128-stage3b.pt \\
        --n-samples 6 --steps 2 --seed 0

--fixed-windows takes the exact 'run:step0:step1:step2' strings the
tool itself prints, for a repeatable rerun on later checkpoints.
"""
import argparse
import re
import math
import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import torch

from evaluation._window_parsing import parse_fixed_window
from evaluation.check_rollout import (
    _correlation_pct, _format_small, _padded_bounds, compute_sample,
)
from models.latent_streams import resolve_stream_configs_from_checkpoint_config

# Default output paths are anchored to the file's own location (python/..),
# not the CWD -- the project's path policy, so the tool lands its figure in
# output/ wherever it is invoked from.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent
from models.constants import LATENT_SPATIAL_SIZE, theta_coordinates, N_THETA
from orchestration.paths import default_latent_cache_dir
from models.latent_dynamics import LatentDynamics, integration_kwargs_from_config
from training.checkpoint_components import build_ae_from_checkpoint
from training.datasets import MicrostructureEvolutionDataset
from training.losses import ReconLoss
from utils import load_datasets as load
from utils.logging_utils import format_progress_count
import sys
import time


def _fmt_corr_pct(corr: float | None) -> str:
    """Format a correlation that is ALREADY IN PERCENT (this module's
    _correlation_pct convention) -- hence the _pct name, distinguishing it
    from evaluation._plot_helpers.fmt_corr which takes raw [-1, 1].
    None happens on real data: a quiet window's real dx has ~zero std --
    the low-|z1| population -- and correlation is undefined there. 'n/a'
    rather than a crash in the title formatting."""
    return "n/a" if corr is None else f"{corr:.0f}%"


def _parse_stem(stem: str) -> tuple[str, str]:
    """'128x128-stage3a[-20260808_13h]' -> ('128x128', 'stage 3a[-...]').

    The prefix titles the figure ('128x128: stage 3a vs. stage 3b') and the
    labels replace the anonymous A/B in every column -- 'error A' forces the
    reader to keep a mapping in their head that the filename already knows.
    Any trailing timestamp is kept in the label: two checkpoints of the SAME
    stage (yesterday's 3b against today's) must not both become 'stage 3b'.
    Stems that do not match the naming scheme fall back to themselves whole,
    so the tool never refuses a comparison over a filename.
    """
    m = re.match(r"(?P<prefix>.+?)-(?P<stage>stage\d+\w*)(?P<rest>.*)", stem)
    if not m:
        return "", stem
    label = re.sub(r"^stage(\d)", r"stage \1", m.group("stage")) + m.group("rest")
    return m.group("prefix"), label


def _load_model(lds_checkpoint_path: Path, device) -> dict:
    """One checkpoint -> its f_theta, its OWN AE, and its configs.

    A stage 4/5 JOINT checkpoint (E, D, f in one file, keys ae_state/
    f_theta_state) is transparently split into the standalone lds-view this
    function already knows how to read -- whose ae_checkpoint points at the
    REFINED encoder, not the pre-refinement stage-2 ancestor. So `compare_f_theta
    128x128-stage4.pt ...` just works, scoring stage 4's refined encoder against
    whatever it is compared with (the f_theta is frozen through stage 4, so the
    comparison isolates the encoder change)."""
    ck = torch.load(lds_checkpoint_path, map_location=device, weights_only=True)
    if "f_theta_state" in ck and "ae_state" in ck:
        # Stage 4/5 joint checkpoint (E, D, f in one file). Load via the SAME
        # component path train_refinement uses to RESUME it, not the
        # split->build_ae_from_checkpoint route: that route rebuilds the AE from
        # config alone and drops the deriv RESIDUAL heads the refined encoder
        # actually carries (cross_check reconciles stream NAMES, not head_kind),
        # so it raises "Unexpected key(s) ...residual_heads.deriv". The component
        # builder constructs the encoder that matches the saved weights, and the
        # f_theta keeps its own time_coordinate (u-scheme stays u-scheme).
        from training.checkpoint_components import load_joint_refinement_checkpoint
        from training.model_assembly import build_models_from_components
        components = load_joint_refinement_checkpoint(lds_checkpoint_path, device=device)
        ae, _stats_head, f_theta, _frozen, _sc, _rsn = build_models_from_components(
            components, device=device, freeze_decoder=True)
        ae.eval()
        f_theta.eval()
        ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
        return {"path": lds_checkpoint_path, "ck": ck,
                "config": ck.get("lds_config", ck.get("config")),
                "ae": ae, "ae_encoder": ae_encoder,
                "ae_config": components["encoder"].config,
                "ae_path": lds_checkpoint_path,   # the REFINED encoder is in this file
                "f_theta": f_theta,
                "max_dt": (ck.get("data_config") or {}).get("max_dt"),
                "prefix": _parse_stem(lds_checkpoint_path.stem)[0],
                "label": _parse_stem(lds_checkpoint_path.stem)[1]}
    cfg = ck["config"]
    ae_path = Path(ck["ae_checkpoint"])
    ae, ae_encoder, ae_ck, _, _ = build_ae_from_checkpoint(ae_path, device)
    f_theta = LatentDynamics(
        latent_channels=cfg["latent_channels"], n_theta=N_THETA,
        latent_spatial=cfg.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
        hidden_dim=cfg["hidden_dim"], n_hidden_layers=cfg["n_hidden_layers"],
        **integration_kwargs_from_config(cfg),
    ).to(device)
    from models.encoder import zero_pad_theta_columns
    f_theta.load_state_dict(zero_pad_theta_columns(ck["model_state"], f_theta))
    f_theta.eval()
    return {"path": lds_checkpoint_path, "ck": ck, "config": cfg, "ae": ae,
            "ae_encoder": ae_encoder, "ae_config": ae_ck["config"],
            "ae_path": ae_path, "f_theta": f_theta,
            # Each checkpoint's own training limit, for marking on dt axes.
            "max_dt": (ck.get("data_config") or {}).get("max_dt"),
            "prefix": _parse_stem(lds_checkpoint_path.stem)[0],
            "label": _parse_stem(lds_checkpoint_path.stem)[1]}


def _select_windows(model: dict, n_samples: int, n_steps: int, seed: int,
                     max_dt: float | None, device,
                     t0_range: tuple | None = None) -> list[tuple]:
    """Pick windows ONCE, from model A's test split, at the COMMON horizon.

    window_length = n_steps + 1 regardless of either checkpoint's own
    trained length -- the whole comparison is void unless both models
    integrate the same transitions.
    """
    data_config = model["ck"].get("data_config") or {}
    test_dirs = model["ck"].get("test_dirs") or []
    if not test_dirs:
        raise ValueError(f"{model['path']} has no saved test_dirs")
    test_dirs = load.validate_run_dirs(
        [Path(d) for d in test_dirs], source=str(model["path"]),
        min_stdev_phi=data_config.get("min_stdev_phi"))
    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=model["ae_encoder"], device=device,
        window_length=n_steps + 1,
        max_dt=max_dt if max_dt is not None else data_config.get("max_dt"),
        min_step=data_config.get("min_step", 0),
        min_stdev_phi=data_config.get("min_stdev_phi"),
        # Share the trainers' latent cache (keyed on encoder fingerprint + run +
        # step list), so a repeated diagnostic re-uses its own encodings instead
        # of re-encoding all runs every invocation. Omitting this disables the
        # cache entirely (datasets.py: latent_cache_dir=None -> no cache).
        latent_cache_dir=default_latent_cache_dir(_PYTHON_ROOT),
    )
    if len(dataset) < n_samples:
        # CLAMP, don't abort. "Give me 5000" means "give me as many as you
        # have" -- the whole population is the best answer to that, and
        # aborting after the dataset is already built throws the work away.
        # The count is printed so the figure's window count is never a
        # surprise.
        print(f"  only {len(dataset)} windows available at "
              f"window_length={n_steps + 1} (asked for {n_samples}); "
              f"using all of them")
        n_samples = len(dataset)
    # --t0-range LO HI: restrict to windows whose STARTING step (t0) is in
    # [LO, HI], sampling within that band. Lets an early-t0 and a late-t0 run at
    # the SAME step count (== matched Delta-u span) be compared: equal coherence
    # across bands => the stability box is set by log-time span, not absolute dt.
    eligible = list(range(len(dataset)))
    if t0_range is not None:
        lo, hi = t0_range
        eligible = [i for i in eligible
                    if lo <= dataset.window_info(i)[1][0] <= hi]
        print(f"  --t0-range [{lo:g}, {hi:g}]: {len(eligible)} of {len(dataset)} "
              f"windows start in band")
        if not eligible:
            raise ValueError(
                f"no windows start in t0-range [{lo:g}, {hi:g}] -- widen the band "
                f"or check it is in STEP units (e.g. 7000 30000), not physical time")
    if len(eligible) < n_samples:
        n_samples = len(eligible)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(eligible), generator=generator)[:n_samples].tolist()
    windows = []
    for j in perm:
        run_dir, steps = dataset.window_info(eligible[j])
        windows.append((run_dir, list(steps)))
    return windows


def collect_stats(a: dict, b: dict, windows: list[tuple], device,
                   z1_resync: bool) -> dict:
    """Loss and correlation for BOTH models over every window in `windows`.

    No images: the six-panel figure shows a handful of windows chosen to be
    looked at, which is the wrong basis for a verdict -- the seed-0 and seed-1
    figures disagreed on the size of the gap because six windows is six
    windows. This is the same quantities over hundreds.
    """
    recon_loss = ReconLoss()
    out = {"loss_a": [], "loss_b": [], "corr_a": [], "corr_b": [],
            "dt": [], "temperature": [], "n_corr_undefined": 0,
            # PER-STEP: index k holds every window's value after k chained
            # applications. The collapse is counted in APPLICATIONS, not in
            # elapsed time -- two runs with different per-step dt broke at the
            # same frame index -- so this is the axis the failure lives on.
            "step_loss_a": [], "step_loss_b": [],
            "step_corr_a": [], "step_corr_b": [],
            # The frozen causal baseline, per step -- the same curve the
            # trajectory panels draw as their second row, over the whole
            # sample. Absent where the window has no pre-window frame, so it
            # carries its OWN endpoint and dt arrays rather than aligning to
            # the a/b ones, which cover every window.
            "step_loss_causal": [], "step_corr_causal": [],
            "loss_causal": [], "corr_causal": [], "dt_causal": [],
            "temp_causal": [],
            # Stage 2 alone (z0 + z1*dt, no f_theta) -- present for every
            # window, unlike causal, so it needs no separate dt/T arrays.
            "step_loss_stage2": [], "step_corr_stage2": [],
            "loss_stage2": [], "corr_stage2": []}

    def _series(real_frames, pred_frames):
        losses_k, corrs_k = [], []
        for k in range(len(real_frames)):
            losses_k.append(recon_loss(
                torch.from_numpy(pred_frames[k])[None, None],
                torch.from_numpy(real_frames[k])[None, None]).item())
            corrs_k.append(
                _correlation_pct(pred_frames[0], real_frames[0]) if k == 0
                else _correlation_pct(pred_frames[k] - pred_frames[0],
                                       real_frames[k] - real_frames[0]))
        return losses_k, corrs_k

    for run_dir, steps in windows:
        row = {}
        real_frames = None
        for key, m in (("a", a), ("b", b)):
            real_frames, pred_frames, dt_per_step = compute_trajectory(
                run_dir, steps, m["ae"], m["f_theta"], m["ae_config"], device,
                z1_resync=z1_resync)
            losses_k, corrs_k = _series(real_frames, pred_frames)
            out[f"step_loss_{key}"].append(losses_k)
            out[f"step_corr_{key}"].append(corrs_k)
            dt_total = sum(dt_per_step)
            # The ENDPOINT is the last frame of this same trajectory, so the
            # two views of the same run cannot disagree.
            row[key] = {"loss": losses_k[-1], "corr": corrs_k[-1]}

        # Per-window temperature, read once from the run's own metadata (the
        # same file the trajectory paths came from), used by the vs-T panels.
        meta = load.read_metadata(Path(run_dir) / "metadata.txt")
        temperature = meta.temperature

        _tc = getattr(a["f_theta"], "time_coordinate", "t")
        stage2 = compute_stage2_trajectory(run_dir, steps, a["ae"],
                                            a["ae_config"], device,
                                            time_coordinate=_tc)
        s2_loss, s2_corr = _series(real_frames, stage2)
        out["step_loss_stage2"].append(s2_loss)
        out["step_corr_stage2"].append(s2_corr)
        out["loss_stage2"].append(s2_loss[-1])
        out["corr_stage2"].append(s2_corr[-1])

        causal = compute_causal_trajectory(run_dir, steps, a["ae"],
                                            a["ae_config"], device,
                                            time_coordinate=_tc)
        if causal is not None:
            c_loss, c_corr = _series(real_frames, causal)
            out["step_loss_causal"].append(c_loss)
            out["step_corr_causal"].append(c_corr)
            out["loss_causal"].append(c_loss[-1])
            out["corr_causal"].append(c_corr[-1])
            out["dt_causal"].append(dt_total)
            out["temp_causal"].append(temperature)
        out["dt"].append(dt_total)
        out["temperature"].append(temperature)
        for key in ("a", "b"):
            out[f"loss_{key}"].append(row[key]["loss"])
            out[f"corr_{key}"].append(row[key]["corr"])
        # A window counts as undefined if EITHER model's correlation is --
        # dropping it for one model only would compare two populations, the
        # error this whole tool exists to avoid.
        if row["a"]["corr"] is None or row["b"]["corr"] is None:
            out["n_corr_undefined"] += 1

    # WHERE the undefined correlations sit, per step index. A step missing
    # entirely from the correlation panel means EVERY window was undefined
    # there, for both models -- which no property of any single window
    # explains, and which points at a CONSTANT delta (pred[k] == pred[0], or
    # identical real frames) rather than at a quiet window.
    n_frames = max((len(r) for r in out["step_corr_a"]), default=0)
    out["n_corr_undefined_per_step"] = {
        key: [sum(1 for row in out[f"step_corr_{key}"]
                   if len(row) > k and row[k] is None)
              for k in range(n_frames)]
        for key in ("a", "b")}
    return out


def _binned(dt: np.ndarray, values: np.ndarray, n_bins: int = 8):
    """Median per geometric dt bin, with quartiles.

    MEDIAN, not mean. The losses here span six decades (1e-3 to 4e6 on real
    data), and a mean over that is just the largest sample restated -- one
    diverged decade-4 window would set every bin it fell in. The quartile
    band carries the spread the median drops.
    """
    good = np.isfinite(dt) & np.isfinite(values) & (dt > 0)
    dt, values = dt[good], values[good]
    if dt.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges = np.geomspace(dt.min(), dt.max() * 1.001, n_bins + 1)
    centres, med, lo, hi = [], [], [], []
    for i in range(n_bins):
        sel = (dt >= edges[i]) & (dt < edges[i + 1])
        if not sel.any():
            continue        # empty bins are SKIPPED, not plotted as zero
        centres.append(float(np.sqrt(edges[i] * edges[i + 1])))
        med.append(float(np.median(values[sel])))
        lo.append(float(np.percentile(values[sel], 25)))
        hi.append(float(np.percentile(values[sel], 75)))
    return (np.array(centres), np.array(med), np.array(lo), np.array(hi))


def _moving_window(x: np.ndarray, values: np.ndarray, half_width: int = 2):
    """Moving window over the DISTINCT x values -- no binning.

    For each distinct temperature T, the reported set is every window whose
    temperature is T or one of the `half_width` distinct values either side
    of it. Bins would impose arbitrary edges on a narrow, densely-sampled
    range and let a bin's population change discontinuously; a window that
    slides one sweep value at a time keeps every point on a real temperature
    and makes neighbouring points overlap smoothly.

    Returns (x, MEDIAN, q25, q75, N) -- N being how many windows each point
    rests on, so a quartile band computed from two or three windows can be
    told apart from one computed from fifty. MEDIAN, not mean: the loss distribution
    is heavy-tailed enough that one diverged window in six drags the mean
    ~17 decades above the 75th percentile, so a mean would be plotted
    outside its own quartile band. The median is bracketed by the band by
    construction, and matches the reduction every other panel uses.
    """
    good = np.isfinite(x) & np.isfinite(values)
    x, values = x[good], values[good]
    empty = np.array([])
    if x.size == 0:
        # FIVE values on every path -- the count array is part of the
        # contract, and returning four here raised only when a panel had no
        # finite data at all.
        return empty, empty, empty, empty, empty
    counts = []
    uniq = np.unique(x)
    centres, med, lo, hi = [], [], [], []
    # ONLY FULL WINDOWS. The first and last `half_width` values can only draw
    # on a truncated set (3 temperatures instead of 5), so their point is a
    # different, noisier statistic plotted on the same line -- exactly at the
    # ends of the range where the interesting behaviour is. Dropped rather
    # than shown alongside full-window points.
    # If trimming would leave nothing (fewer than 2*half_width+1 distinct
    # values -- e.g. a single-temperature sweep), fall back to every value
    # with whatever window it has: an empty panel hides the data entirely,
    # which is worse than a partial window honestly labelled.
    trim = half_width if len(uniq) > 2 * half_width else 0
    for i in range(trim, len(uniq) - trim):
        v = uniq[i]
        window = uniq[max(0, i - half_width):i + half_width + 1]
        sel = np.isin(x, window)
        if not sel.any():
            continue
        vals = values[sel]
        centres.append(float(v))
        counts.append(int(vals.size))
        med.append(float(np.median(vals)))
        lo.append(float(np.percentile(vals, 25)))
        hi.append(float(np.percentile(vals, 75)))
    return (np.array(centres), np.array(med), np.array(lo), np.array(hi),
            np.array(counts))


def _label_dt_axis(ax) -> None:
    """Number the dt axis at 2/3/5 x each decade, not once per decade.

    A log axis spanning ~200 to ~5000 gets exactly ONE major tick (1000) from
    the default locator, so the axis carries a single number and the reader
    cannot place any point on it.
    """
    # Density chosen from the SPAN. A range under a decade gets only two or
    # three numbers from a fixed (2,3,5); a range of four decades labelled at
    # every sub-tick is unreadable. Measured ranges here run from ~0.6 of a
    # decade (one seed's windows) to ~4 decades.
    lo, hi = ax.get_xlim()
    decades = np.log10(max(hi, 1e-30) / max(lo, 1e-30))
    if decades <= 1.5:
        subs = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    elif decades <= 3.0:
        subs = (2.0, 3.0, 5.0)
    else:
        subs = (3.0,)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=subs))
    for axis_formatter in (ax.xaxis.set_major_formatter,
                            ax.xaxis.set_minor_formatter):
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis_formatter(formatter)
    ax.tick_params(axis="x", which="minor", labelsize=7)


_CORR_FLOOR = -20.0


def _corr_axis(ax, vertical: bool = True) -> None:
    """Pin the correlation axis to span 0 and 100%.

    Correlation has fixed, meaningful endpoints -- no skill and perfect --
    and a panel autoscaled to 40..92% invites reading a small real difference
    as a large one, and cannot be compared against the next panel at all.
    """
    setter = ax.set_ylim if vertical else ax.set_xlim
    getter = ax.get_ylim if vertical else ax.get_xlim
    lo, hi = getter()
    # FLOOR AT -20%. A model can go arbitrarily anticorrelated once it leaves
    # the manifold, and letting one such window set the axis to -300%
    # squeezes the 0..100 band -- where every meaningful difference lives --
    # into the top sliver of the panel. Curves below -20% run off the bottom.
    setter(max(min(lo, -2.0), _CORR_FLOOR), max(hi, 100.0))


def _temperature_axis(ax) -> None:
    """Always show T = 1 on the axis.

    T = 1 is the physically meaningful endpoint of the sweep, and the SMA
    trims the last two sweep values, so the plotted points stop short of it.
    Without the endpoint pinned, a panel ending at 0.97 looks like it covers
    the whole range and the gap to T = 1 is invisible.
    """
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, max(hi, 1.0))


def _pretty_label(label: str, include_year: bool) -> str:
    """'stage 2-20260812_20h08' -> 'stage 2 (12/08 at 20:08)', adding the year
    ('12/08/2026') only when include_year is True (i.e. the compared
    checkpoints do not all share the current year). Labels without a
    -YYYYMMDD_HHhMM timestamp are returned unchanged."""
    m = re.search(r"(?P<stage>.*?)-?(?P<Y>\d{4})(?P<M>\d{2})(?P<D>\d{2})"
                  r"_(?P<h>\d{2})h(?P<min>\d{2})", label)
    if not m:
        return label
    stage = m.group("stage").rstrip("-").strip()
    date = f"{m.group('D')}/{m.group('M')}"
    if include_year:
        date += f"/{m.group('Y')}"
    stamp = f"{date} at {m.group('h')}:{m.group('min')}"
    return f"{stage} ({stamp})" if stage else f"({stamp})"


def _labels_need_year(labels: list[str]) -> bool:
    """True when the checkpoints' timestamps do not all fall in the current
    year -- then the year must be shown to disambiguate."""
    import datetime
    years = set()
    for label in labels:
        m = re.search(r"-?(\d{4})\d{4}_\d{2}h\d{2}", label)
        if m:
            years.add(m.group(1))
    if not years:
        return False
    return years != {str(datetime.date.today().year)}


def _ylim_from_medians(ax, medians) -> None:
    """Range set by the MEDIAN curves, not the quartile bands.

    A single window diverging to 1e30 pushed the band to 1e30, and with it
    the axis, flattening the six decades where the two curves actually differ
    into a line at the bottom. The bands stay drawn -- they simply run off
    the top, which is the honest depiction of an unbounded tail.
    """
    values = np.array([v for series in medians for v in series
                        if np.isfinite(v) and v > 0], dtype=float)
    if values.size:
        ax.set_ylim(values.min() / 3.0, values.max() * 3.0)


def _stats_figure(stats: dict, a: dict, b: dict, title: str,
                   output_path: Path, *, n_steps: int = 1) -> Path:
    """Four panels: loss and correlation, as distributions and against dt."""
    dt = np.array(stats["dt"], dtype=float)
    # The stored "dt" is sum(dt_per_step) in the MODEL's time coordinate: a
    # u-scheme (log10_t) model steps in Delta-u=log10(t_{i+1}/t_i), so the axis
    # is du_total, not dt_total. Label it truthfully so the span is not misread
    # as a narrow physical-dt range.
    _dt_axis_label = ("du_total" if getattr(a["f_theta"], "time_coordinate", "t") == "log10_t"
                      else "dt_total")
    fig, axes = plt.subplots(2, 5, figsize=(31, 9))
    colours = {"a": "tab:blue", "b": "tab:red", "causal": "tab:green",
                "stage2": "tab:purple"}
    # Stage 2 RE-ENCODES the predicted state each step (the models never do), so
    # its rollout is a same-methodology baseline only at 1 step. At multi-step it
    # uses information the models cannot; draw it DOTTED so "3a below stage 2"
    # there reads as the re-encoding advantage, not f_theta worsening things.
    _stage2_comparable = (n_steps == 1)
    # Tag the baselines with WHICH stage-2 encoder produced them (a's frozen
    # encoder; a and b share it). Compact timestamp from a["ae_path"] so the
    # legend says e.g. "causal (frozen dz0/dt) [26/08 14:55]" instead of a
    # generic label that could be any stage-2 run.
    # tag with the stage-2 encoder's timestamp in the SAME "DD/MM at HH:MM"
    # style as every other label (via _pretty_label on its stem).
    _enc_pretty = _pretty_label(Path(a.get("ae_path", "")).stem, include_year=False)
    _es = re.search(r"\((.*?)\)", _enc_pretty)   # pull just the "26/08 at 14:55"
    _enc = f"  [{_es.group(1)}]" if _es else ""
    # Keep the legend short: the "(z0 + z1 dt)" / "(frozen dz0/dt)" descriptors
    # are already in the suptitle, and the encoder tag made these overflow.
    # Drop the descriptor, keep the encoder provenance tag.
    labels = {"a": a["label"], "b": b["label"],
              "causal": "previous derivative",
              "stage2": "stage 2" + _enc + (
                  "" if _stage2_comparable else "  [not comparable]")}
    linestyles = {"a": "-", "b": "-", "causal": "-",
                  "stage2": "-" if _stage2_comparable else ":"}
    has_causal = bool(stats.get("step_loss_causal"))
    has_stage2 = bool(stats.get("step_loss_stage2"))

    def _keys():
        """Curves in the SAME order as the trajectory figure's rows:
        causal, stage 2, 3a, 3b -- ordered by how much machinery each uses,
        so the legends of the two figures read alike."""
        out = []
        if has_causal:
            out.append("causal")
        if has_stage2:
            out.append("stage2")
        return tuple(out) + ("a", "b")

    # LAYOUT: rows are the METRIC (loss, then correlation), columns the
    # VIEW (distribution, against dt, against number of steps). Transposed
    # from the original arrangement so that every column shares one x axis --
    # the two dt panels now sit one above the other and can be read together,
    # as can the two step panels.
    #
    # (0,0) loss distribution, as a CDF -- bin-free, so no bin width has to
    # be chosen for a quantity spanning six decades, and the two curves can
    # be read against each other at any quantile.
    ax = axes[0, 0]
    # QUANTITY ON Y, cumulative fraction on X -- so loss shares the y axis
    # with the loss-vs-dt and loss-vs-steps panels beside it and the three
    # can be read across at a glance. (A CDF with the value on y is the
    # quantile function; same curve, transposed.)
    for key in _keys():
        v = np.sort(np.array(stats[f"loss_{key}"], dtype=float))
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            ax.plot(np.arange(1, v.size + 1) / v.size, v,
                    color=colours[key], linestyle=linestyles[key], label=labels[key])
    ax.set_yscale("log")
    ax.set_ylabel("end-to-end loss")
    ax.set_xlabel("cumulative fraction of windows")
    ax.set_title("loss distribution (lower is better)")
    ax.grid(alpha=0.3)
    ax.legend()

    # (1,0) correlation distribution
    ax = axes[1, 0]
    for key in _keys():
        v = np.array([c for c in stats[f"corr_{key}"] if c is not None],
                      dtype=float)
        if v.size:
            v = np.sort(v)
            ax.plot(np.arange(1, v.size + 1) / v.size, v,
                    color=colours[key], linestyle=linestyles[key], label=labels[key])
    ax.set_ylabel("correlation of predicted vs real dx (%)")
    ax.set_xlabel("cumulative fraction of windows")
    ax.set_title("correlation distribution (higher is better)")
    _corr_axis(ax, vertical=True)
    ax.grid(alpha=0.3)
    ax.legend()

    # (0,1) loss against dt
    ax = axes[0, 1]
    dt_loss_medians = []
    for key in _keys():
        key_dt = dt if key != "causal" else np.array(stats["dt_causal"],
                                                       dtype=float)
        c, med, lo, hi = _binned(key_dt, np.array(stats[f"loss_{key}"],
                                                   dtype=float))
        if c.size:
            ax.plot(c, med, "o", color=colours[key], linestyle=linestyles[key], label=labels[key])
            ax.fill_between(c, lo, hi, color=colours[key], linestyle=linestyles[key], alpha=0.15)
            dt_loss_medians.append(med)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(_dt_axis_label + " (binned)")
    ax.set_ylabel("median loss (band: quartiles)")
    ax.set_title("loss vs dt")
    # loss-vs-dt is the REFERENCE range for the whole loss row: it is scaled
    # to its medians (via the shared _ylim call at the end of the row), so
    # the decades where the curves actually differ stay legible instead of
    # being flattened by one window's diverged tail. The CDF and loss-vs-T
    # adopt THIS range -- the row is matched to col 1 (vs dt), even though
    # the CDF then loses its own tail off the top.
    axes[0, 1].set_yscale("log")
    _label_dt_axis(ax)
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # (1,1) correlation against dt
    ax = axes[1, 1]
    for key in _keys():
        corr = np.array([np.nan if c is None else c
                          for c in stats[f"corr_{key}"]], dtype=float)
        key_dt = dt if key != "causal" else np.array(stats["dt_causal"],
                                                       dtype=float)
        c, med, lo, hi = _binned(key_dt, corr)
        if c.size:
            ax.plot(c, med, "o", color=colours[key], linestyle=linestyles[key], label=labels[key])
            ax.fill_between(c, lo, hi, color=colours[key], linestyle=linestyles[key], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel(_dt_axis_label + " (binned)")
    ax.set_ylabel("median correlation (%) (band: quartiles)")
    ax.set_title("correlation vs dt")
    _corr_axis(ax)
    _label_dt_axis(ax)
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # (0,2) and (1,2): the SAME endpoints against TEMPERATURE. dt and T are
    # collinear in this sweep (later frames are both larger dt and, through
    # the run, a different T), so binning the identical endpoints by T
    # instead of dt shows how much of the dt trend is really a T trend. The
    # 40x swing in the Taylor-residual dt* across the T<0.9 / T>=0.9 split is
    # the reason to look: error is a strong function of T here.
    temp = np.array(stats["temperature"], dtype=float)
    ax = axes[0, 2]
    for key in _keys():
        key_temp = (temp if key != "causal"
                    else np.array(stats["temp_causal"], dtype=float))
        c, med, lo, hi, n_win = _moving_window(
            key_temp, np.array(stats[f"loss_{key}"], dtype=float))
        if c.size:
            ax.plot(c, med, "o", color=colours[key], linestyle=linestyles[key], label=labels[key])
            ax.fill_between(c, lo, hi, color=colours[key], linestyle=linestyles[key], alpha=0.15)
            # How thin the tails get is not visible on the panel, and a
            # quartile band over 2-3 windows is not a quartile band. Report
            # it once per model rather than cluttering the figure.
            print(f"  vs-T {labels[key]}: {n_win.min()}-{n_win.max()} windows "
                  f"per point (median {int(np.median(n_win))}); "
                  f"thinnest at T={c[int(np.argmin(n_win))]:.3f}")
    ax.set_xlabel("temperature (SMA)")
    _temperature_axis(ax)
    ax.set_ylabel("median loss (band: quartiles)")
    ax.set_title("loss vs temperature (moving window: T +/- 2 sweep values)")
    # Same y range as the other loss panels in the row, so all three read
    # across; the CDF sets it (see below, after that panel's range is fixed).
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[1, 2]
    for key in _keys():
        corr = np.array([np.nan if c is None else c
                          for c in stats[f"corr_{key}"]], dtype=float)
        key_temp = (temp if key != "causal"
                    else np.array(stats["temp_causal"], dtype=float))
        c, med, lo, hi, _n = _moving_window(key_temp, corr)
        if c.size:
            ax.plot(c, med, "o", color=colours[key], linestyle=linestyles[key], label=labels[key])
            ax.fill_between(c, lo, hi, color=colours[key], linestyle=linestyles[key], alpha=0.15)
    ax.set_xlabel("temperature (SMA)")
    _temperature_axis(ax)
    ax.set_ylabel("median correlation (%) (band: quartiles)")
    ax.set_title("correlation vs temperature (moving window: T +/- 2 sweep values)")
    _corr_axis(ax)
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # (0,3) and (1,3): against the NUMBER OF CHAINED STEPS. The trajectory
    # panels showed two runs with different per-step dt collapsing at the
    # same frame INDEX, so the failure is counted in applications rather
    # than elapsed time -- this is that axis, over the whole sample.
    step_loss_medians = []
    step_keys = _keys()
    step_med = {}  # (kind, key) -> {step_index: median}; feeds the 5th column
    for panel, kind in ((axes[0, 3], "loss"), (axes[1, 3], "corr")):
        for key in step_keys:
            series = stats.get(f"step_{kind}_{key}") or []
            if not series:
                continue
            n_frames = max(len(row) for row in series)
            centres, med, lo, hi = [], [], [], []
            for k in range(n_frames):
                vals = np.array([row[k] for row in series
                                  if len(row) > k and row[k] is not None],
                                 dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                centres.append(k)
                med.append(float(np.median(vals)))
                lo.append(float(np.percentile(vals, 25)))
                hi.append(float(np.percentile(vals, 75)))
            if centres:
                panel.plot(centres, med, "o", color=colours[key], linestyle=linestyles[key],
                            label=labels[key])
                panel.fill_between(centres, lo, hi, color=colours[key], linestyle=linestyles[key],
                                    alpha=0.15)
                step_med[(kind, key)] = dict(zip(centres, med))
                if kind == "loss":
                    step_loss_medians.append(med)
        panel.set_xlabel("chained steps applied")
        panel.grid(alpha=0.3, which="both")
        panel.legend()
    axes[0, 3].set_yscale("log")
    axes[0, 3].set_ylabel("median loss (band: quartiles)")
    # NOTE for the reader: this x axis is the CHAINED-STEP INDEX over ALL
    # windows, whereas loss-vs-dt bins the SAME endpoints by dt_total. The
    # rightmost point here (step 8, every window) and the rightmost point
    # there (top dt bin only) are medians over DIFFERENT subpopulations, so
    # they need not match even though both are "the 8-step endpoint".
    axes[0, 3].set_title("loss vs number of steps (all windows per step)")
    # OWN median scaling, NOT the loss row's shared range: this panel starts
    # at the near-zero step-0 loss and spans decades the endpoint-based
    # panels never reach, so forcing it onto their scale crushed its low end.
    _ylim_from_medians(axes[0, 3], step_loss_medians)
    axes[1, 3].set_ylabel("median correlation (%) (band: quartiles)")
    axes[1, 3].set_title("correlation vs number of steps (all windows per step)")
    _corr_axis(axes[1, 3])

    # (0,4) and (1,4): the vs-steps medians NORMALIZED to expose the per-step
    # law, from n=1 (both are undefined at n=0: 1/n, and ln(1)/0).
    #   [0,4] loss:  g(n) = ln[loss(n)/loss(0)] / n  -- the per-step log-growth
    #                rate; FLAT across n => loss grows exponentially at rate g
    #                (the compounding signature), rising/falling => the rate
    #                itself changes with n.
    #   [1,4] corr:  (1 - corr(n)) / n  (corr as a FRACTION, so _corr_axis's %
    #                is divided by 100) -- the per-step decorrelation; FLAT =>
    #                correlation falls linearly (constant coherence lost per
    #                step), rising => accumulation accelerates.
    # OWN y-ranges, deliberately NOT joined to the loss-row (log loss) or the
    # correlation-row (shared 0..100) unions below: these are rates in
    # different units, and inheriting either scale would make them unreadable.
    for key in step_keys:
        ml = step_med.get(("loss", key), {})
        if ml and ml.get(0, 0.0) > 0.0:
            ks = sorted(n for n in ml if n >= 1 and ml.get(n, 0.0) > 0.0)
            g = [np.log(ml[n] / ml[0]) / n for n in ks]
            if ks:
                axes[0, 4].plot(ks, g, "o", color=colours[key], linestyle=linestyles[key], label=labels[key])
        mc = step_med.get(("corr", key), {})
        if mc:
            ks = sorted(n for n in mc if n >= 1)
            d = [(100.0 - mc[n]) / n for n in ks]  # percent per step
            if ks:
                axes[1, 4].plot(ks, d, "o", color=colours[key], linestyle=linestyles[key], label=labels[key])
    axes[0, 4].set_xlabel("chained steps applied")
    axes[0, 4].set_ylabel("ln[loss(n)/loss(0)] / n")
    axes[0, 4].set_title("loss log-growth rate per step")
    axes[0, 4].grid(alpha=0.3, which="both")
    axes[0, 4].legend(fontsize=7)
    axes[1, 4].set_xlabel("chained steps applied")
    axes[1, 4].set_ylabel("(1 - corr(n)) / n  [% per step]")
    axes[1, 4].set_title("decorrelation rate per step")
    axes[1, 4].grid(alpha=0.3, which="both")
    axes[1, 4].legend(fontsize=7)
    # (1-corr)/n is non-negative -- pin the floor to 0% (top stays auto).
    axes[1, 4].set_ylim(bottom=0.0)
    # Series start at n=1, but share the x range with the vs-steps column
    # beside them (which starts at step 0) so the two columns align.
    axes[0, 4].set_xlim(left=0)
    axes[1, 4].set_xlim(left=0)

    # The three ENDPOINT loss panels -- CDF [0,0], vs-dt [0,1], vs-T [0,2] --
    # share one y range, set by loss-vs-dt's MEDIANS. The CDF used to set the
    # range and show its full tail; now it takes the median-scaled range and
    # loses the tail off the top, so the decades where the curves differ are
    # legible across all three. loss-vs-steps [0,3] is NOT included: it plots
    # every step, starting from the near-zero step-0 loss, so it spans
    # decades these three never reach and keeps its own scale.
    _ylim_from_medians(axes[0, 1], dt_loss_medians)
    loss_ref = axes[0, 1].get_ylim()
    for ax in (axes[0, 0], axes[0, 2]):
        ax.set_yscale("log")
        ax.set_ylim(loss_ref)

    # CORRELATION ROW shares one y range across ALL FOUR panels. _corr_axis
    # pins each to span 0..100, but the vs-dt/vs-T/vs-steps panels dip
    # negative (a model anticorrelated with the truth) while the CDF, bounded
    # below by its own worst window, need not, so without the union they
    # misaligned at the bottom. Take the union so the row reads across.
    corr_panels = [axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]]
    # The clamp here is belt-and-braces: every panel above already went
    # through _corr_axis, so the minimum is >= _CORR_FLOOR already. It stays
    # so that a panel added later without _corr_axis cannot silently
    # re-impose a -300% range on the whole row.
    corr_lo = max(min(ax.get_ylim()[0] for ax in corr_panels), _CORR_FLOOR)
    corr_hi = max(ax.get_ylim()[1] for ax in corr_panels)
    for ax in corr_panels:
        ax.set_ylim(corr_lo, corr_hi)

    n = len(stats["dt"])
    note = f"{n} windows"
    # How many DISTINCT values each binned/smoothed axis actually has -- the
    # thing a reader cannot see from the panels, and what decides whether 8
    # bins is over- or under-resolving dt_total.
    n_dt = len(np.unique(np.array(stats["dt"], dtype=float)))
    n_temp = len(np.unique(np.array(stats["temperature"], dtype=float)))
    temps_all = np.array(stats["temperature"], dtype=float)
    note += (f"; {n_dt} distinct dt_total, {n_temp} distinct temperatures "
              f"({temps_all.min():g}-{temps_all.max():g})")
    if stats["n_corr_undefined"]:
        note += (f"; {stats['n_corr_undefined']} dropped from the correlation "
                  f"panels (undefined: a quiet window's real dx has ~zero std)")
    fig.suptitle(f"{title}\n{note}", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


def sweep_f_theta_scale(model: dict, windows: list[tuple], device,
                         z1_resync: bool, scales=(0.0, 0.25, 0.5, 1.0)) -> dict:
    """End-to-end loss and correlation as f_theta is scaled from 0 to 1.

    scale=0 IS stage 2 (see scaled_f_theta), scale=1 is the model as trained,
    and the trajectory is otherwise identical -- same windows, same z0 start,
    same integrator. Any difference is attributable to f_theta's magnitude
    alone.

    Reported as medians over the shared window set, so one diverged window
    cannot decide the shape of the curve.
    """
    recon_loss = ReconLoss()
    out = {"scales": list(scales), "loss": [], "corr": []}
    for scale in scales:
        losses, corrs = [], []
        with scaled_f_theta(model["f_theta"], scale):
            for run_dir, steps in windows:
                real_frames, pred_frames, _dts = compute_trajectory(
                    run_dir, steps, model["ae"], model["f_theta"],
                    model["ae_config"], device, z1_resync=z1_resync)
                losses.append(recon_loss(
                    torch.from_numpy(pred_frames[-1])[None, None],
                    torch.from_numpy(real_frames[-1])[None, None]).item())
                c = _correlation_pct(pred_frames[-1] - pred_frames[0],
                                      real_frames[-1] - real_frames[0])
                if c is not None:
                    corrs.append(c)
        finite = [v for v in losses if np.isfinite(v)]
        out["loss"].append(float(np.median(finite)) if finite else float("nan"))
        out["corr"].append(float(np.median(corrs)) if corrs else float("nan"))
        print(f"  f_theta x {scale:<5g} median loss {out['loss'][-1]:.4g}  "
              f"median corr {out['corr'][-1]:.1f}%")
    return out


@contextlib.contextmanager
def overridden_alpha(f_theta, alpha: float, max_substeps: int | None = None):
    """Temporarily replace the alpha substep criterion (and optionally the
    substep cap).

    Smaller alpha means more substeps: n ~ f*dt/(alpha*|z1|), so sweeping
    alpha downward is the h -> 0 limit at FIXED learned field. That limit
    separates the two explanations for the rollout blowup:

      scheme instability  -> the blowup VANISHES as h shrinks (the explicit
                             Heun step moves inside its stability region);
      unstable f_theta    -> the blowup PERSISTS: the exact solution of
                             z1' = f_theta diverges, and refining h just
                             computes that divergence more accurately.

    Only valid because THIS f_theta was trained with alpha-substepping, i.e.
    as a vector field -- for a model trained at fixed n_substeps=1, f is a
    dt-averaged algebraic corrector and h -> 0 is not a meaningful limit
    (established when sub-stepping a 3a trained that way made things
    monotonically worse).

    max_substeps matters: the count saturates there and h stops shrinking,
    so the sweep must either raise it or report the clamp rate -- a
    "converged" verdict at saturated substeps would be an artifact.
    """
    orig_alpha = f_theta.alpha
    orig_max = f_theta.max_substeps
    f_theta.alpha = alpha
    if max_substeps is not None:
        f_theta.max_substeps = max_substeps
    try:
        yield
    finally:
        f_theta.alpha = orig_alpha
        f_theta.max_substeps = orig_max


def sweep_alpha(model: dict, windows: list[tuple], device,
                z1_resync: bool, alphas=(1.5, 0.5, 0.15, 0.05),
                max_substeps: int = 4096) -> dict:
    """Endpoint loss/correlation as alpha shrinks -- the h -> 0 limit.

    Medians over the shared window set. The substep-clamp counter is reset
    per alpha and reported: a clamp rate near 100% means h has stopped
    shrinking and that alpha's point does not probe a smaller step.
    """
    f_theta = model["f_theta"]
    if f_theta.alpha is None:
        print(f"  {model['label']}: trained at fixed n_substeps="
              f"{f_theta.n_substeps}, not with alpha -- h -> 0 is not "
              f"meaningful for it (f is a dt-averaged corrector, not a "
              f"vector field); skipping")
        return {}
    recon_loss = ReconLoss()
    out = {"alphas": list(alphas), "loss": [], "corr": [], "clamped": []}
    for alpha in alphas:
        losses, corrs = [], []
        with overridden_alpha(f_theta, alpha, max_substeps=max_substeps):
            f_theta.n_substeps_clamped = 0
            for run_dir, steps in windows:
                real_frames, pred_frames, _dts = compute_trajectory(
                    run_dir, steps, model["ae"], f_theta,
                    model["ae_config"], device, z1_resync=z1_resync)
                losses.append(recon_loss(
                    torch.from_numpy(pred_frames[-1])[None, None],
                    torch.from_numpy(real_frames[-1])[None, None]).item())
                c = _correlation_pct(pred_frames[-1] - pred_frames[0],
                                      real_frames[-1] - real_frames[0])
                if c is not None:
                    corrs.append(c)
            clamped = f_theta.n_substeps_clamped
        finite = [v for v in losses if np.isfinite(v)]
        out["loss"].append(float(np.median(finite)) if finite else float("nan"))
        out["corr"].append(float(np.median(corrs)) if corrs else float("nan"))
        out["clamped"].append(int(clamped))
        print(f"  alpha {alpha:<5g} median loss {out['loss'][-1]:.4g}  "
              f"median corr {out['corr'][-1]:.1f}%  "
              f"substep-cap hits {clamped}")
    return out


@contextlib.contextmanager
def scaled_f_theta(f_theta, scale: float):
    """Temporarily multiply f_theta's output by `scale`.

    scale=0 recovers STAGE 2 exactly -- the integrator's z0 += z1*h + f*h^2/2
    and z1 += (f+f')*h/2 both collapse to the first-order step -- and scale=1
    is the model as trained. Sweeping between them interpolates along the
    line in function space that connects the two, WITHOUT retraining.

    That is the whole point: stage 2 is nested inside stage 3 (f == 0), so a
    stage-3 model doing worse than stage 2 under propagation cannot be
    explained by the hypothesis class. The sweep asks whether the damage is
    monotone in how much f_theta is applied -- if it is, f_theta is pure harm
    at rollout and belongs out of the propagation path; if there is an
    interior minimum, a damped f_theta is worth having and the fix is
    regularization toward zero rather than deletion.

    Wraps f() rather than the integrator so the sub-stepping, the alpha
    criterion and the trapezoidal corrector all see the scaled value and
    stay self-consistent.
    """
    original = f_theta.f
    f_theta.f = lambda z0, z1, theta: original(z0, z1, theta) * scale
    try:
        yield
    finally:
        f_theta.f = original


def compute_stage2_trajectory(run_dir: Path, steps: list[int], ae,
                               ae_config: dict, device, time_coordinate: str = "t"):
    """Stage 2 alone, CAUSAL -- only frame t0 is ever read.

        z0(t+(n+1)dt) = z0(t+n dt) + z1(t+n dt) dt
        x (t+(n+1)dt) = D[z0(t+(n+1)dt)]
        z1(t+(n+1)dt) = E[x(t+(n+1)dt)]        (deriv stream)

    No f_theta and no resync: the derivative for the next step is re-derived
    from the model's OWN predicted state by decoding and re-encoding it, not
    read from the real frame at that time. That is what puts this row on the
    same footing as 3a/3b, which likewise see only t0 and advance z1
    internally -- the difference being that they advance it with f_theta
    while this advances it through the autoencoder.

    An earlier version took z1 from the real frame at each step. It looked
    stable for a reason that had nothing to do with stage 2 being good: with
    a correct derivative supplied every step, errors cannot compound, so the
    curve was flat by construction. This version can drift, and the drift is
    the honest measurement.

    The decode/encode round trip is the cost of the causal version, and its
    error is part of what is being measured: stage 2 has no way to advance
    z1 without going back through pixel space.
    """
    metadata = load.read_metadata(run_dir / "metadata.txt")
    nx, ny = ae_config["size"], ae_config["size"]
    theta_vec = theta_coordinates(metadata.temperature, metadata.T0)
    with torch.no_grad():
        _, recon_stream_name = resolve_stream_configs_from_checkpoint_config(ae_config)
        ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
        ae_decoder = (ae.pathways[recon_stream_name].decoder if hasattr(ae, "pathways")
                      else ae.decoder)
        # ONLY the starting frame is read from disk.
        x0 = torch.from_numpy(
            load.read_phi_half(run_dir / load.snapshot_filename(steps[0]), nx, ny)
        ).unsqueeze(0).unsqueeze(0).to(device)
        theta_encode = torch.tensor([theta_vec], dtype=torch.float32,
                                   device=device)
        encoded = ae_encoder(x0, theta=theta_encode)
        z0 = encoded[recon_stream_name]
        z1 = encoded["deriv"]

        out = [ae_decoder(z0)[0, 0].cpu().numpy()]
        for k in range(len(steps) - 1):
            # u-scheme: z0 + z̃1*Delta-u (z̃1=ln10*t*z1) instead of z0 + z1*dt,
            # so the baseline integrates in the SAME coordinate as the model --
            # the only difference measured is then the correction network, not
            # the coordinate.
            if time_coordinate == "log10_t":
                dt = math.log10(steps[k + 1] / steps[k])
                z1_use = z1 * (math.log(10.0) * metadata.dt * steps[k])
            else:
                dt = (steps[k + 1] - steps[k]) * metadata.dt
                z1_use = z1
            z0 = z0 + z1_use * dt
            x_pred = ae_decoder(z0)
            out.append(x_pred[0, 0].cpu().numpy())
            # z1 for the NEXT step, from the predicted state alone.
            z1 = ae_encoder(x_pred, theta=theta_encode)["deriv"]
    return out


def compute_causal_trajectory(run_dir: Path, steps: list[int], ae,
                               ae_config: dict, device, time_coordinate: str = "t"):
    """Backward-difference baseline: what the PAST alone predicts.

    z0_dot_back = (z0(t) - z0(t - dt_minus)) / dt_minus, from the two real
    frames at and before the window start -- both available at prediction
    time, unlike a centered estimate. Same definition as
    check_parameter_dependence's err_causal.

    The derivative is FROZEN at the start and extrapolated linearly, because
    a causal estimate needs two real past frames and there are none once the
    trajectory leaves the data. So this is a BASELINE, not a rollout: it is
    what you get with no model at all, chained the same way the models are,
    and it already beats both of them by the end of an 8-step window -- a
    model row that cannot clear it is not earning its keep. (A teacher-forced
    version that re-reads a real frame each step would only widen a gap that
    is already decisive, so the honest apples-to-apples baseline is the
    frozen one.)

    Returns None when the window starts at the run's first saved step (no
    earlier frame to difference against).
    """
    metadata = load.read_metadata(run_dir / "metadata.txt")
    saved = list(metadata.save_steps)
    if steps[0] not in saved or saved.index(steps[0]) == 0:
        return None
    prev_step = saved[saved.index(steps[0]) - 1]
    # Backward gap in the model's coordinate: Delta-u=log10(t0/t_prev) for a
    # log10_t model (dividing the backward difference by it yields dz0/du, the
    # u-derivative -- equivalent to ln10*t0*dz0/dt to leading order), else Delta-t.
    if time_coordinate == "log10_t":
        dt_minus = math.log10(steps[0] / prev_step)
    else:
        dt_minus = (steps[0] - prev_step) * metadata.dt
    if dt_minus <= 0:
        return None

    nx, ny = ae_config["size"], ae_config["size"]
    theta_vec = theta_coordinates(metadata.temperature, metadata.T0)
    with torch.no_grad():
        _, recon_stream_name = resolve_stream_configs_from_checkpoint_config(ae_config)
        ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
        ae_decoder = (ae.pathways[recon_stream_name].decoder if hasattr(ae, "pathways")
                      else ae.decoder)
        frames = [load.read_phi_half(run_dir / load.snapshot_filename(s), nx, ny)
                   for s in (prev_step, steps[0])]
        x = torch.stack([torch.from_numpy(f) for f in frames]).unsqueeze(1).to(device)
        theta_encode = torch.tensor(theta_vec, dtype=torch.float32,
                                   device=device).expand(2, -1)
        encoded = ae_encoder(x, theta=theta_encode)[recon_stream_name]
        z0_prev, z0_t = encoded[0:1], encoded[1:2]

        z0_dot_back = (z0_t - z0_prev) / dt_minus
        elapsed = 0.0
        out = [ae_decoder(z0_t)[0, 0].cpu().numpy()]
        for i in range(len(steps) - 1):
            # elapsed accumulates in the model's coordinate; summing Delta-u
            # increments telescopes to log10(t_{i+1}/t0), the cumulative log-time.
            if time_coordinate == "log10_t":
                elapsed += math.log10(steps[i + 1] / steps[i])
            else:
                elapsed += (steps[i + 1] - steps[i]) * metadata.dt
            out.append(ae_decoder(z0_t + z0_dot_back * elapsed)[0, 0].cpu().numpy())
    return out


def compute_trajectory(run_dir: Path, steps: list[int], ae, f_theta,
                        ae_config: dict, device, z1_resync: bool = False):
    """Decoded state at EVERY frame: (real_frames, pred_frames, dt_per_step).

    compute_sample returns only the endpoint, which answers "how wrong is it
    at the end" but not "where did it go wrong". rollout() already computes
    every intermediate z0 -- this decodes them all.

    Both lists have len(steps) entries. Frame 0 of the prediction is the AE
    RECONSTRUCTION of the starting state, not the raw snapshot: that is where
    the model actually starts, so any decoder error is visible in the column
    it belongs to rather than being silently absorbed into step 1.
    """
    metadata = load.read_metadata(run_dir / "metadata.txt")
    nx, ny = ae_config["size"], ae_config["size"]
    theta_vec = theta_coordinates(metadata.temperature, metadata.T0)
    real_frames = [load.read_phi_half(run_dir / load.snapshot_filename(s), nx, ny)
                    for s in steps]
    # u-scheme: a log10_t model steps in Delta-u=log10(t_{i+1}/t_i) and consumes
    # z̃1=dz0/du=ln10*t*z1, not Delta-t and z1=dz0/dt. Coordinate read straight
    # off the model (a non-model f_theta -- e.g. a test stub -- defaults to "t").
    _tc = getattr(f_theta, "time_coordinate", "t")
    if _tc == "log10_t":
        dt_per_step = [math.log10(steps[i + 1] / steps[i])
                        for i in range(len(steps) - 1)]
    else:
        dt_per_step = [(steps[i + 1] - steps[i]) * metadata.dt
                        for i in range(len(steps) - 1)]

    with torch.no_grad():
        _, recon_stream_name = resolve_stream_configs_from_checkpoint_config(ae_config)
        ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
        ae_decoder = (ae.pathways[recon_stream_name].decoder if hasattr(ae, "pathways")
                      else ae.decoder)
        x_all = torch.stack([torch.from_numpy(f) for f in real_frames]
                             ).unsqueeze(1).to(device)
        theta_encode = torch.tensor(theta_vec,
                                   dtype=torch.float32, device=device).expand(len(steps), -1)
        encoded = ae_encoder(x_all, theta=theta_encode)
        z0_t = encoded[recon_stream_name][0:1]
        z1_sequence = encoded["deriv"].unsqueeze(0)
        if _tc == "log10_t":
            _sc = torch.tensor([math.log(10.0) * metadata.dt * s for s in steps],
                               device=device, dtype=z1_sequence.dtype)
            z1_sequence = z1_sequence * _sc[None, :, None, None, None]  # z1 -> z̃1
        dts = torch.tensor([dt_per_step], dtype=torch.float32, device=device)
        theta = torch.tensor([theta_vec], dtype=torch.float32, device=device)

        z0_hat_full = f_theta.rollout(z0_t, z1_sequence, dts, theta,
                                       z1_resync=z1_resync)
        # rollout() returns (B, n_steps+1, ...) with [:, 0] == z0 ALREADY --
        # the initial state is its first entry, not something to prepend.
        # Prepending a separately-decoded z0_t duplicated frame 0, shifted
        # every model panel one frame BEHIND the real row it sits under, and
        # dropped the final frame off the right-hand edge. It was silent:
        # the extra entry simply went unplotted, and column 1's delta was
        # identically zero, which surfaced only as "correlation undefined at
        # step 1 for every window".
        pred_frames = [ae_decoder(z0_hat_full[:, i])[0, 0].cpu().numpy()
                        for i in range(z0_hat_full.shape[1])]
    return real_frames, pred_frames, dt_per_step


def _trajectory_figure(run_dir: Path, steps: list[int], a: dict, b: dict,
                        device, z1_resync: bool, title: str,
                        output_path: Path) -> Path:
    """3 rows (real, A, B) x len(steps) columns (t, t+dt, t+2dt, ...).

    One window, followed frame by frame, so a divergence can be located in
    TIME rather than only measured at the end.
    """
    real_frames, pred_a, _ = compute_trajectory(
        run_dir, steps, a["ae"], a["f_theta"], a["ae_config"], device, z1_resync)
    _, pred_b, dt_per_step = compute_trajectory(
        run_dir, steps, b["ae"], b["f_theta"], b["ae_config"], device, z1_resync)

    # CAUSAL BASELINE as the second row, immediately under the truth: the
    # bar every model row has to clear. What "no model at all, just the last
    # two real frames" achieves -- check_parameter_dependence measured it
    # beating z1 outright beyond dt ~ 1e3.
    #
    # TEACHER-FORCED, unlike the model rows: it sees a real frame at every
    # step. That makes it an upper bound on a causal one-step predictor
    # rather than a rollout competitor, and the row label says so.
    _tc = getattr(a["f_theta"], "time_coordinate", "t")
    causal = compute_causal_trajectory(run_dir, steps, a["ae"], a["ae_config"],
                                        device, time_coordinate=_tc)
    # STAGE 2 alone -- z0 + z1*dt, no f_theta. Sits between the causal
    # baseline and the models: it uses the encoder's own derivative rather
    # than a backward difference, but has no dt^2/2 correction.
    stage2 = compute_stage2_trajectory(run_dir, steps, a["ae"], a["ae_config"],
                                        device, time_coordinate=_tc)
    n_cols = len(steps)
    # (label, frames, metrics-key) -- the key is carried WITH the row rather
    # than derived from its index, which stopped being readable the moment
    # the causal row could be present or absent.
    rows = [("real", real_frames, None)]
    if causal is not None:
        rows.append(("previous derivative\n(linear extrapolation)", causal, "causal"))
    rows.append(("stage 2\n(z0 + z1 dt)", stage2, "stage2"))
    rows += [(a["label"], pred_a, "a"), (b["label"], pred_b, "b")]

    # Per-frame loss and correlation, against the SAME real frame the panel
    # sits above. Both are measured on the DELTA from the start (x - x_0),
    # not on the state: the states are 95% identical background at every
    # frame, so a correlation between states would read ~99% right through
    # the collapse. The delta is what the model is actually predicting, and
    # it is what the six-column figure reports, so the numbers are
    # comparable between the two figures.
    recon_loss = ReconLoss()
    metrics = {}
    metric_sources = [("a", pred_a), ("b", pred_b), ("stage2", stage2)]
    if causal is not None:
        metric_sources.append(("causal", causal))
    for key, frames in metric_sources:
        per_frame = []
        for col in range(n_cols):
            loss = recon_loss(torch.from_numpy(frames[col])[None, None],
                               torch.from_numpy(real_frames[col])[None, None]).item()
            if col == 0:
                # No delta exists yet, so correlate the STATES: at frame 0
                # that is exactly the AE reconstruction fidelity, which is
                # the meaningful number there and the baseline every later
                # frame is measured against. Reporting n/a wasted the one
                # column that says how good the starting point was.
                corr = _correlation_pct(frames[0], real_frames[0])
            else:
                # frames[0], not pred_a[0]: each model's delta is measured
                # from ITS OWN starting reconstruction. Subtracting A's start
                # from B's frames would fold the difference between the two
                # AEs into B's correlation at every frame.
                #
                # This still returns n/a when the REAL delta is constant --
                # snapshots are stored as float16, so at a short dt and a low
                # temperature the state can be unchanged at storage
                # precision. That n/a is a fact about the data, not a defect.
                corr = _correlation_pct(frames[col] - frames[0],
                                         real_frames[col] - real_frames[0])
            per_frame.append((loss, corr))
        metrics[key] = per_frame

    # ONE scale for the whole figure, taken from the REAL row: the states are
    # the same physical field at every frame, so a per-panel scale would hide
    # exactly the amplitude blow-up that marks a model leaving the manifold.
    lo, hi = _padded_bounds(np.concatenate([f.ravel() for f in real_frames]),
                             1.0, symmetric=True)

    # A little taller than 3 x 3.1: every model panel now carries a caption.
    fig, axes = plt.subplots(len(rows), n_cols,
                              figsize=(3.1 * n_cols, 3.4 * len(rows)),
                              squeeze=False)
    # ABSOLUTE time in the header, with the offset in brackets: "t = 650 (t0)"
    # then "t = 750 (t0 + 100)". The bare "t + 100" gave no way to place a
    # frame in the run without going back to the title's step numbers.
    # metadata.dt is recovered from the data already in hand -- dt_per_step[i]
    # is (steps[i+1] - steps[i]) * metadata.dt -- rather than re-reading the
    # metadata file.
    sim_dt = (dt_per_step[0] / (steps[1] - steps[0])
              if len(steps) > 1 and steps[1] != steps[0] else 0.0)
    t0 = steps[0] * sim_dt
    elapsed = 0.0
    for col in range(n_cols):
        for row, (label, frames, metric_key) in enumerate(rows):
            ax = axes[row, col]
            im = ax.imshow(frames[col], cmap="RdBu_r", vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if metric_key is None:
                offset = "t0" if col == 0 else f"t0 + {elapsed:g}"
                head = f"t = {t0 + elapsed:g} ({offset})"
                ax.set_title(head, fontsize=9)
            else:
                # Model rows carry their own numbers per frame, so the frame
                # at which each collapses can be read off rather than eyeballed.
                loss, corr = metrics[metric_key][col]
                corr_txt = "n/a" if corr is None else f"{corr:.0f}%"
                ax.set_title(f"loss={_format_small(loss)}, corr={corr_txt}",
                              fontsize=8)
            if col == n_cols - 1 and row == len(rows) - 1:
                fig.colorbar(im, ax=axes[:, n_cols - 1].tolist(), fraction=0.03)
        if col < len(dt_per_step):
            elapsed += dt_per_step[col]

    fig.suptitle(f"{title}\n{run_dir.name}:{steps[0]}\u2192{steps[-1]}  "
                  f"(column 0 of the model rows is the AE reconstruction of "
                  f"the start, which is where the model actually begins)",
                  fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _setup_comparison(path_a, path_b, device, n_samples, n_steps, seed,
                       fixed_windows, max_dt, z1_resync, t0_range=None):
    """Shared prologue for the panel and statistics tools: load both models,
    resolve the window set and the title/prefix, print the z1_resync banner.

    Returns everything both tools need so neither reloads a checkpoint. The
    ONE cross-tool coupling that used to exist -- the stats path reaching
    into the panel window set only to read its horizon length -- is gone:
    the horizon is n_steps, passed explicitly, so `compare_statistics` no
    longer needs a panel window drawn just to define it.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    a = _load_model(path_a, device)
    b = _load_model(path_b, device)
    # readable legend/row labels: 'stage 3a-20260826_17h09' -> 'stage 3a
    # (26/08 at 17:09)', same style as --stage2-compare; year only if the two
    # checkpoints span different years.
    _need_year = _labels_need_year([a["label"], b["label"]])
    a["label"] = _pretty_label(a["label"], _need_year)
    b["label"] = _pretty_label(b["label"], _need_year)

    if a["ae_path"] != b["ae_path"]:
        print(f"NOTE: the two checkpoints decode through DIFFERENT AEs\n"
              f"  A: {a['ae_path']}\n  B: {b['ae_path']}\n"
              f"  -- each model is judged through its own decoder, so part of "
              f"any difference below may be the AEs', not f_theta's.")

    if fixed_windows:
        windows = [parse_fixed_window(sw) for sw in fixed_windows]
        lens = {len(steps) for _, steps in windows}
        if len(lens) != 1:
            raise ValueError(f"fixed windows have mixed lengths {sorted(lens)}; "
                              f"the comparison needs one common horizon")
    else:
        windows = _select_windows(a, max(n_samples, 1), n_steps, seed, max_dt,
                                   device, t0_range=t0_range)
    window_strings = [f"{run_dir}:{':'.join(str(sp) for sp in steps)}"
                       for run_dir, steps in windows]
    if not z1_resync:
        _cant = [m["label"] for m in (a, b)
                 # getattr default True: a non-model f_theta (e.g. a test stub,
                 # or any path where it is a name not a LatentDynamics) is
                 # assumed autonomous-capable, so it never forces teacher-forcing
                 # -- only a real model that reports False triggers the override.
                 if not getattr(m["f_theta"], "supports_autonomous_rollout", True)]
        if _cant:
            print(f"NOTE: {', '.join(_cant)} cannot roll out autonomously (no "
                  f"z1-update equation -- deriv_linear with derivative_source='z1'); "
                  f"forcing z1_resync=True for BOTH so the comparison runs "
                  f"teacher-forced. To read an AUTONOMOUS rollout, compare only "
                  f"models that support it (z1_taylor or "
                  f"derivative_source='previous_quotient').")
            z1_resync = True
    print(f"z1_resync={z1_resync} for BOTH models (forced equal; the comparison "
          f"is void if one resyncs and the other propagates).")
    for sw in window_strings:
        print(f"  {sw}")

    prefix = a["prefix"] or b["prefix"]
    if a["prefix"] and b["prefix"] and a["prefix"] != b["prefix"]:
        prefix = f"{a['prefix']} vs {b['prefix']}"
    n_steps_used = len(windows[0][1]) - 1
    # Show the RESOLVED per-transition max_dt: when the CLI gives none,
    # _select_windows silently falls back to model A's TRAINING max_dt
    # (data_config), so two runs can share a filter without the invoker
    # realizing -- e.g. --max-dt 10000 on a checkpoint trained at max_dt=10000
    # is a no-op that looks like a cap. Note it is PER-TRANSITION: the loss-vs-
    # dt axis is dt_total (the sum over the horizon), which reaches ~n_steps
    # times higher under the geometric schedule.
    resolved_max_dt = max_dt if max_dt is not None else a.get("max_dt")
    regime = (f"{n_steps_used} chained step{'s' if n_steps_used != 1 else ''}, "
              f"z1 {'resynced at each real frame' if z1_resync else 'not resynchronized'}, "
              f"max_dt={resolved_max_dt:g}/transition"
              if resolved_max_dt is not None else
              f"{n_steps_used} chained step{'s' if n_steps_used != 1 else ''}, "
              f"z1 {'resynced at each real frame' if z1_resync else 'not resynchronized'}")
    title = (f"{prefix}: {a['label']} vs. {b['label']}" if prefix
              else f"{a['label']} vs. {b['label']}")
    title = f"{title}\n{regime}"
    return device, a, b, windows, window_strings, prefix, title, n_steps_used, z1_resync


def _default_figure_path(prefix, a, b, seed, n_steps_used, z1_resync,
                          fixed_windows):
    # Build the FILENAME from the checkpoint stem, never from the (now
    # prettified) legend label: the pretty label contains "/" and ":"
    # (e.g. "stage 3a (27/08 at 22:44)"), which are a directory separator and
    # an illegal-on-Windows char -- they crashed mkdir. The stem is filesystem-
    # safe by construction. _sanitize is a final guard for any stray character.
    def _sanitize(text):
        for ch in ' /:()':
            text = text.replace(ch, '')
        return text
    # The checkpoint stems are the source of truth and already carry the size
    # prefix (e.g. "128x128-stage3a-..."); join them directly. Do NOT prepend
    # `prefix` (it would duplicate the size -> "128x128-128x128-...") and do NOT
    # build from the prettified label (its "/" and ":" crashed mkdir, and a
    # filename should be reversible to its source, not a pretty display).
    sa = _sanitize(Path(a["path"]).stem)
    sb = _sanitize(Path(b["path"]).stem)
    # Drop a shared leading "<prefix>-" (the size, e.g. "128x128-") from the
    # SECOND name so it is not repeated: "128x128-stage3a_vs_stage3b", not
    # "...stage3a_vs_128x128-stage3b". Still the raw stems, still reversible.
    _shared = f"{prefix}-" if prefix else ""
    if _shared and sb.startswith(_shared):
        sb = sb[len(_shared):]
    name = f"{sa}_vs_{sb}"
    suffix = "" if fixed_windows else f"-seed{seed}"
    regime_tag = f"-{n_steps_used}step{'s' if n_steps_used != 1 else ''}"
    regime_tag += "-resync" if z1_resync else "-propagated"
    out = (_PYTHON_ROOT.parent / "output" / "rollout_check_png"
           / f"{name}{regime_tag}{suffix}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def compare_panels(path_a: Path, path_b: Path, n_samples: int = 6,
                    t0_range=None,
                    n_steps: int = 2, seed: int = 0,
                    fixed_windows: list[str] | None = None,
                    max_dt: float | None = None, z1_resync: bool = False,
                    trajectory: bool = False, output_path: Path | None = None,
                    device: str | None = None) -> tuple[Path, list[str]]:
    """The IMAGE side: the 7-column per-window panel figure, and (with
    trajectory=True) one frame-by-frame trajectory figure per window. Both
    operate on the n_samples window set -- the windows chosen to be looked
    at. For the numbers a verdict rests on, use compare_statistics."""
    (device, a, b, windows, window_strings, prefix, title,
     n_steps_used, z1_resync) = _setup_comparison(path_a, path_b, device, n_samples,
                                         n_steps, seed, fixed_windows, max_dt,
                                         z1_resync, t0_range=t0_range)
    recon_loss = ReconLoss()
    n_rows = len(windows)
    fig, axes = plt.subplots(n_rows, 8, figsize=(33, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    fig.suptitle(title, fontsize=13)

    losses = {"a": [], "b": []}
    for row, (run_dir, steps) in enumerate(windows):
        per = {}
        for key, m in (("a", a), ("b", b)):
            x_t, x_real, x_pred, _x_ae, dt_total, dt_per_step = compute_sample(
                run_dir, steps, m["ae"], m["f_theta"], m["ae_config"], device,
                z1_resync=z1_resync,
            )
            per[key] = {"pred_delta": x_pred - x_t,
                         "loss": recon_loss(
                             torch.from_numpy(x_pred)[None, None],
                             torch.from_numpy(x_real)[None, None]).item()}
            losses[key].append(per[key]["loss"])
        real_delta = x_real - x_t
        for key in ("a", "b"):
            per[key]["error"] = per[key]["pred_delta"] - real_delta
            per[key]["corr"] = _correlation_pct(per[key]["pred_delta"], real_delta)

        state_scale = max(abs(x_t.min()), abs(x_t.max()), 0.1)
        d_lo, d_hi = _padded_bounds(real_delta, 1.0, symmetric=True)
        e_lo, e_hi = _padded_bounds(
            np.concatenate([per["a"]["error"].ravel(), per["b"]["error"].ravel()]),
            1.0, symmetric=True)
        diff = per["b"]["pred_delta"] - per["a"]["pred_delta"]
        f_lo, f_hi = e_lo, e_hi

        # Stage 2 = pure z0 + z1 dt (no f_theta), propagated -- the AE-only
        # prediction whose z1 diverges past its skill horizon at late t. Same
        # encoder for a and b, so compute once (uses a's coordinate). Its OWN
        # robust symmetric scale (99th pct, floored at the real-dx range): a
        # shared scale would saturate the whole panel once z1 blows up, so here
        # divergence shows as a few saturated pixels while the printed scale
        # reports the magnitude.
        s2_tc = a.get("config", {}).get("time_coordinate", "t")
        stage2_frames = compute_stage2_trajectory(
            run_dir, steps, a["ae"], a["ae_config"], device, time_coordinate=s2_tc)
        stage2_delta = stage2_frames[-1] - x_t
        s2_corr = _correlation_pct(stage2_delta, real_delta)
        s2_scale = max(float(np.percentile(np.abs(stage2_delta), 99.0)), d_hi)

        steps_txt = f"{run_dir.name}:{steps[0]}\u2192{steps[-1]} ({len(steps) - 1} steps)"
        cells = [
            (x_t, -state_scale, state_scale,
             f"state(t)\n{steps_txt}\ndt_total={dt_total:g}"),
            (real_delta, d_lo, d_hi, f"real dx\nscale=[{d_lo:.3f}, {d_hi:.3f}]"),
            (stage2_delta, -s2_scale, s2_scale,
             f"stage 2 dx (z0+z1 dt)\nscale=[-{s2_scale:.2g}, {s2_scale:.2g}], "
             f"corr={_fmt_corr_pct(s2_corr)}"),
            (per["a"]["pred_delta"], d_lo, d_hi,
             f"pred dx ({a['label']})\nloss={_format_small(per['a']['loss'])}, "
             f"corr={_fmt_corr_pct(per['a']['corr'])}"),
            (per["b"]["pred_delta"], d_lo, d_hi,
             f"pred dx ({b['label']})\nloss={_format_small(per['b']['loss'])}, "
             f"corr={_fmt_corr_pct(per['b']['corr'])}"),
            (per["a"]["error"], e_lo, e_hi, f"error {a['label']}"),
            (per["b"]["error"], e_lo, e_hi, f"error {b['label']}"),
            (diff, f_lo, f_hi,
             f"{b['label']} \u2212 {a['label']}\n(= error {b['label']} "
             f"\u2212 error {a['label']})"),
        ]
        for col, (img, lo, hi, cell_title) in enumerate(cells):
            ax = axes[row, col]
            im = ax.imshow(img, cmap="RdBu_r", vmin=lo, vmax=hi)
            ax.set_title(cell_title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 7:
                fig.colorbar(im, ax=axes[row, 7], fraction=0.046)

    med_a = float(np.median(losses["a"]))
    med_b = float(np.median(losses["b"]))
    print(f"\nmedian end-to-end loss over {n_rows} shared windows: "
          f"{a['label']}={med_a:.5f}  {b['label']}={med_b:.5f}  "
          f"({(a['label'] if med_a < med_b else b['label'])} better by "
          f"{max(med_a, med_b) / max(min(med_a, med_b), 1e-30):.2f}x)")
    print(f"Medians over {n_rows} windows are indicative only -- rerun "
          f"with --n-samples 24 (or the RMS tools) before concluding.")

    if output_path is None:
        output_path = _default_figure_path(prefix, a, b, seed, n_steps_used,
                                            z1_resync, fixed_windows)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    print(f"saved {output_path}")

    if trajectory:
        for run_dir, steps in windows:
            run_dir = Path(run_dir)
            traj_path = output_path.with_name(
                f"{output_path.stem.rsplit('-seed', 1)[0]}-{run_dir.name}.png")
            _trajectory_figure(run_dir, steps, a, b, device, z1_resync,
                                title, traj_path)
            print(f"saved {traj_path}")

    return output_path, window_strings


def _load_stage2_ae(path: Path, device) -> dict:
    """Load a stage-2 (AE-family) checkpoint as a plain model dict for the
    stage-2 rollout comparison. Unlike _load_model (which expects an f_theta
    checkpoint and reads its ae_checkpoint pointer), this treats the file
    itself as the autoencoder -- which is what a stage-2 checkpoint is. Carries
    just enough (ck for test_dirs/data_config, ae, ae_encoder, ae_config,
    label) for _select_windows and compute_stage2_trajectory."""
    ae, ae_encoder, ae_ck, _, _ = build_ae_from_checkpoint(path, device)
    ck = torch.load(path, map_location=device, weights_only=True)
    return {"path": path, "ck": ck, "ae": ae, "ae_encoder": ae_encoder,
            "ae_config": ae_ck["config"], "label": _parse_stem(path.stem)[1]}


def compare_stage2_rollouts(paths: list[Path], n_stats: int = 200,
                             t0_range=None,
                            n_steps: int = 10, seed: int = 0,
                            max_dt: float | None = None,
                            output_path: Path | None = None,
                            device: str | None = None) -> Path:
    """Compare the STAGE-2 rollout (z0+z1 dt, causal, no f_theta) of an
    arbitrary set of AE-family checkpoints -- one curve per checkpoint. This
    is the tool for "which stage-2 checkpoint rolls out best" (e.g. comparing
    training snapshots), independent of the f_theta a/b machinery.

    Windows are chosen ONCE from the FIRST checkpoint's test split at the
    common horizon, and every checkpoint's rollout is measured on that same
    window set, so the curves are directly comparable. Loss and correlation
    are collected per rollout step and plotted as median +/- quartile bands
    against step count.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if len(paths) < 1:
        raise ValueError("need at least one checkpoint")
    models = [_load_stage2_ae(Path(p), device) for p in paths]
    # duplicate labels (two snapshots parsing to the same stem) would collide
    # in the legend; disambiguate with the filename stem where needed.
    all_labels = [x["label"] for x in models]
    for m in models:
        if all_labels.count(m["label"]) > 1:
            m["label"] = f"{m['label']} [{Path(m['path']).stem}]"
    # readable legend labels: 'stage 2-20260812_20h08' -> 'stage 2 (12/08 at
    # 20:08)', with the year added only if the checkpoints span multiple years.
    _need_year = _labels_need_year([m["label"] for m in models])
    for m in models:
        m["label"] = _pretty_label(m["label"], _need_year)

    windows = _select_windows(models[0], n_stats, n_steps, seed, max_dt, device,
                               t0_range=t0_range)
    print(f"\nstage-2 rollout comparison over {len(windows)} windows "
          f"(same windows for all {len(models)} checkpoints), "
          f"horizon {n_steps} steps...")

    # Per-window dt_total and temperature are shared across checkpoints (same
    # windows), collected once for the vs-dt / vs-temperature panels.
    dt_totals, temps = [], []
    for run_dir, steps in windows:
        run_dir = Path(run_dir)
        meta = load.read_metadata(run_dir / "metadata.txt")
        dt_totals.append(sum((steps[i + 1] - steps[i]) * meta.dt
                             for i in range(len(steps) - 1)))
        temps.append(meta.temperature)
    dt_totals = np.array(dt_totals, dtype=float)
    temps = np.array(temps, dtype=float)

    # Total rollout evaluations for the progress bar: each model runs both the
    # stage-2 and the causal trajectory over every window.
    _total_evals = 2 * len(models) * len(windows)
    _progress = {"done": 0, "t0": None}
    _show_progress = _total_evals >= 500   # silent for small/test runs

    def _tick():
        _progress["done"] += 1
        if _progress["t0"] is None:
            _progress["t0"] = time.monotonic()
        if _show_progress and (_progress["done"] % 50 == 0
                               or _progress["done"] == _total_evals):
            _done = _progress["done"]
            _el = time.monotonic() - _progress["t0"]
            if _done > 0 and _el > 0:
                _rem = _el / _done * (_total_evals - _done)
                _m, _s = divmod(int(_rem + 0.5), 60)
                _eta = f"~{_m}m{_s:02d}s left" if _m else f"~{_s}s left"
            else:
                _eta = "estimating"
            sys.stdout.write(
                f"\r  rollout progress: "
                f"{format_progress_count(_done, _total_evals)}  ({_eta})   ")
            sys.stdout.flush()

    # A diverged rollout produces frames large enough that ReconLoss's squaring
    # overflows float32 to inf; inf then poisons np.nanpercentile/median across
    # windows (the "overflow encountered in reduce" warning). The divergence is
    # real information -- the window blew up -- but it should read as "very bad
    # and finite", not inf, so quartiles stay meaningful. Cap at a large
    # sentinel well above any converged loss (which top out ~1e3 here).
    _LOSS_CAP = 1e12

    def _rollout_series(traj_fn, ae, ae_config):
        """Per-window per-step loss/corr for a trajectory function that returns
        a pred-frame LIST (stage-2 or causal) or None (causal, no past frame).
        Returns (step_losses (W,S+1) with nan rows where undefined, step_corrs,
        final_losses, final_corrs)."""
        recon_loss = ReconLoss()
        step_losses, step_corrs, fin_loss, fin_corr = [], [], [], []
        for run_dir, steps in windows:
            _tick()
            run_dir = Path(run_dir)
            nx = ny = ae_config["size"]
            real = [load.read_phi_half(run_dir / load.snapshot_filename(s), nx, ny)
                    for s in steps]
            pred = traj_fn(run_dir, steps, ae, ae_config, device)
            if pred is None:                       # causal: no frame before start
                step_losses.append([np.nan] * len(steps))
                step_corrs.append([np.nan] * len(steps))
                fin_loss.append(np.nan)
                fin_corr.append(np.nan)
                continue
            lk, ck = [], []
            for k in range(len(real)):
                with np.errstate(over="ignore", invalid="ignore"):
                    loss_val = recon_loss(
                        torch.from_numpy(pred[k])[None, None],
                        torch.from_numpy(real[k])[None, None]).item()
                lk.append(min(loss_val, _LOSS_CAP) if np.isfinite(loss_val)
                          else _LOSS_CAP)
                c = (_correlation_pct(pred[0], real[0]) if k == 0
                     else _correlation_pct(pred[k] - pred[0], real[k] - real[0]))
                ck.append(np.nan if c is None else c)
            step_losses.append(lk)
            step_corrs.append(ck)
            fin_loss.append(lk[-1])
            fin_corr.append(ck[-1])
        return (np.array(step_losses, dtype=float),
                np.array(step_corrs, dtype=float),
                np.array(fin_loss, dtype=float),
                np.array(fin_corr, dtype=float))

    for m in models:
        (m["step_losses"], m["step_corrs"],
         m["final_loss"], m["final_corr"]) = _rollout_series(
            compute_stage2_trajectory, m["ae"], m["ae_config"])
        (m["causal_step_losses"], m["causal_step_corrs"],
         m["causal_final_loss"], m["causal_final_corr"]) = _rollout_series(
            compute_causal_trajectory, m["ae"], m["ae_config"])
    if _show_progress:
        sys.stdout.write("\r" + " " * 50 + "\r")   # erase the progress line
        sys.stdout.flush()
    import warnings
    for m in models:
        with np.errstate(over="ignore", invalid="ignore"), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med_loss = np.nanmedian(m["step_losses"], axis=0)
            med_corr = np.nanmedian(m["step_corrs"], axis=0)
        print(f"  {m['label']}: final median loss {med_loss[-1]:.5g}, "
              f"final median corr {med_corr[-1]:.1f}%")
    n_causal = int(np.isfinite(models[0]["causal_final_loss"]).sum())
    print(f"  causal baseline defined on {n_causal}/{len(windows)} windows "
          f"(needs a frame before the window start)")

    if output_path is None:
        prefix = _parse_stem(models[0]["path"].stem)[0] or "compare"
        output_path = (_PYTHON_ROOT.parent / "output" / "stage2" /
                       f"{prefix}-stage2_rollout-{len(models)}ckpts-seed{seed}"
                       f"-{n_steps}steps.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _stage2_rollout_figure(models, dt_totals, temps, n_steps, output_path)
    print(f"saved {output_path}")
    return output_path


def _stage2_rollout_figure(models, dt_totals, temps, n_steps, output_path):
    """2x4: loss (top) and correlation (bottom), each as distribution /
    vs-dt / vs-temperature / vs-steps. One colour per checkpoint. The causal
    baseline (frozen backward dz0/dt -- the best a local derivative can do) is
    drawn once in grey as the reference every stage-2 curve is judged against;
    it is checkpoint-independent in spirit (each uses its own AE) so the first
    model's causal is plotted, with the others' left implicit to avoid clutter.
    """
    import warnings
    steps_axis = np.arange(n_steps + 1)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 4, figsize=(25, 9))

    def _dist(ax, arrays_labels_colours, is_loss):
        for arr, label, colour in arrays_labels_colours:
            v = np.sort(arr[np.isfinite(arr)])
            if v.size == 0:
                continue
            ax.plot(np.linspace(0, 1, v.size), v, color=colour, label=label)
        if is_loss:
            ax.set_yscale("log")
        ax.set_xlabel("cumulative fraction of windows")

    def _vs(ax, x, arrays_labels_colours, binned, is_loss):
        meds = []
        for arr, label, colour in arrays_labels_colours:
            fn = _binned if binned else _moving_window
            res = fn(x, arr)
            c, med, lo, hi = res[0], res[1], res[2], res[3]
            if len(c) == 0:
                continue
            ax.plot(c, med, "-o", color=colour, label=label, markersize=3)
            ax.fill_between(c, lo, hi, color=colour, alpha=0.12)
            meds.append(med)
        if is_loss:
            ax.set_yscale("log")
        if binned:
            ax.set_xscale("log")
        return meds

    def _vs_steps(ax, per_model_step, causal_step, is_loss):
        meds = []
        for i, m in enumerate(models):
            data = m[per_model_step]
            with np.errstate(over="ignore", invalid="ignore"), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(data, axis=0)
                lo = np.nanpercentile(data, 25, axis=0)
                hi = np.nanpercentile(data, 75, axis=0)
            ax.plot(steps_axis, med, "-o", color=cmap(i % 10),
                    label=m["label"], markersize=3)
            ax.fill_between(steps_axis, lo, hi, color=cmap(i % 10), alpha=0.12)
            meds.append(med)
        cdata = models[0][causal_step]
        if np.isfinite(cdata).any():
            with np.errstate(over="ignore", invalid="ignore"), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _cmed = np.nanmedian(cdata, axis=0)
            ax.plot(steps_axis, _cmed, "--",
                    color="grey", label="previous derivative")
            meds.append(_cmed)
        if is_loss:
            ax.set_yscale("log")
        ax.set_xlabel("chained steps applied")
        return meds

    # colour/label bundles for the final-step panels (cols 0-2)
    def _bundle(final_key, causal_key):
        b = [(m[final_key], m["label"], cmap(i % 10))
             for i, m in enumerate(models)]
        b.append((models[0][causal_key], "previous derivative", "grey"))
        return b

    # TOP ROW: loss. Collect the MEDIAN curves so the y-range is set by them,
    # not by the diverging quartile bands (a single 1e30 window otherwise
    # flattens the decades that matter) -- same discipline as _stats_figure.
    loss_dist_meds = [(m["final_loss"][np.isfinite(m["final_loss"])])
                      for m in models]
    _dist(axes[0, 0], _bundle("final_loss", "causal_final_loss"), True)
    axes[0, 0].set_title("loss distribution (lower is better)")
    axes[0, 0].set_ylabel("end-to-end loss (final step)")
    axes[0, 0].legend(fontsize=7)
    dt_loss_meds = _vs(axes[0, 1], dt_totals,
                       _bundle("final_loss", "causal_final_loss"),
                       binned=True, is_loss=True)
    axes[0, 1].set_title("loss vs dt")
    axes[0, 1].set_xlabel("dt_total (binned)")
    _vs(axes[0, 2], temps, _bundle("final_loss", "causal_final_loss"),
        binned=False, is_loss=True)
    axes[0, 2].set_title("loss vs temperature")
    axes[0, 2].set_xlabel("temperature")
    step_loss_meds = _vs_steps(axes[0, 3], "step_losses",
                               "causal_step_losses", True)
    axes[0, 3].set_title("loss vs number of steps")
    axes[0, 3].set_ylabel("median loss (band: quartiles)")
    axes[0, 3].legend(fontsize=7)

    # BOTTOM ROW: correlation
    _dist(axes[1, 0], _bundle("final_corr", "causal_final_corr"), False)
    axes[1, 0].set_title("correlation distribution (higher is better)")
    axes[1, 0].set_ylabel("correlation (%) (final step)")
    _vs(axes[1, 1], dt_totals, _bundle("final_corr", "causal_final_corr"),
        binned=True, is_loss=False)
    axes[1, 1].set_title("correlation vs dt")
    axes[1, 1].set_xlabel("dt_total (binned)")
    _vs(axes[1, 2], temps, _bundle("final_corr", "causal_final_corr"),
        binned=False, is_loss=False)
    axes[1, 2].set_title("correlation vs temperature")
    axes[1, 2].set_xlabel("temperature")
    _vs_steps(axes[1, 3], "step_corrs", "causal_step_corrs", False)
    axes[1, 3].set_title("correlation vs number of steps")
    axes[1, 3].set_ylabel("median correlation (%) (band: quartiles)")

    # y-caps + alignment (mirrors _stats_figure's end block):
    # the endpoint loss panels (CDF [0,0], vs-dt [0,1], vs-T [0,2]) share one
    # median-driven range so their differing decades read together. loss-vs-
    # steps [0,3] is EXCLUDED: it plots every step from the near-zero step-0
    # loss and spans decades the endpoint panels never reach, so it keeps its
    # own median-driven scale (identical rationale to _stats_figure).
    # Correlation panels share a floored range.
    _ylim_from_medians(axes[0, 1], dt_loss_meds)
    _ylim_from_medians(axes[0, 3], step_loss_meds)
    _ylim_from_medians(axes[0, 0], loss_dist_meds)
    loss_ref = axes[0, 1].get_ylim()
    for ax in (axes[0, 0], axes[0, 2]):
        ax.set_ylim(loss_ref)
    corr_panels = [axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]]
    corr_lo = max(min(ax.get_ylim()[0] for ax in corr_panels), _CORR_FLOOR)
    corr_hi = max(ax.get_ylim()[1] for ax in corr_panels)
    for ax in corr_panels:
        ax.set_ylim(corr_lo, corr_hi)

    fig.suptitle("stage-2 rollout comparison (z0 + z1 dt, causal, no f_theta) "
                 "-- grey dashed: causal backward-dz0/dt baseline")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def compare_statistics(path_a: Path, path_b: Path, n_stats: int = 200,
                       t0_range=None,
                        n_steps: int = 2, seed: int = 0,
                        max_dt: float | None = None, z1_resync: bool = False,
                        f_scale_sweep: bool = False, alpha_sweep: bool = False,
                        trajectory: bool = False,
                        output_path: Path | None = None,
                        device: str | None = None) -> tuple[Path, list[str]]:
    """The STATISTICS side: loss/correlation over n_stats windows (the 2x4
    figure), plus the optional f-scale and alpha sweeps, which share that
    same window set and the same loaded models. No per-window images. This
    is what a verdict should rest on; the panels are for looking at.

    n_steps is the shared rollout horizon, passed directly -- unlike the old
    combined function, no panel window is drawn to define it, so n_samples
    has no meaning here."""
    (device, a, b, windows, window_strings, prefix, title,
     n_steps_used, z1_resync) = _setup_comparison(path_a, path_b, device, 0, n_steps,
                                         seed, None, max_dt, z1_resync, t0_range=t0_range)
    if n_stats:
        stat_windows = _select_windows(a, n_stats, n_steps, seed, max_dt, device,
                                        t0_range=t0_range)
        print(f"\ncollecting statistics over {len(stat_windows)} windows "
              f"(both models, same windows)...")
        stats = collect_stats(a, b, stat_windows, device, z1_resync)
    else:
        stats = None  # trajectory-only: no stats figure, just the traj plots
    if alpha_sweep and stats is not None:
        for key, m in (("a", a), ("b", b)):
            print(f"\nalpha sweep -- {m['label']} (smaller alpha == "
                  f"smaller substeps, h -> 0):")
            sweep_alpha(m, stat_windows, device, z1_resync)
    if f_scale_sweep and stats is not None:
        for key, m in (("a", a), ("b", b)):
            print(f"\nf_theta scale sweep -- {m['label']} "
                  f"(scale 0 == stage 2):")
            sweep_f_theta_scale(m, stat_windows, device, z1_resync)
    for key, m in ((("a", a), ("b", b)) if stats is not None else ()):
        losses_k = np.array(stats[f"loss_{key}"], dtype=float)
        corrs_k = np.array([c for c in stats[f"corr_{key}"]
                             if c is not None], dtype=float)
        print(f"  {m['label']}: median loss {np.median(losses_k):.5g}, "
              f"median corr {np.median(corrs_k):.1f}%")
    if stats is not None:
        # Causal (frozen dz0/dt extrapolation) never re-encodes -- same rollout
        # methodology as the models -- so it is comparable at every step count.
        _cl = np.array([x for x in stats["loss_causal"] if np.isfinite(x)], dtype=float)
        _cc = np.array([c for c in stats["corr_causal"] if c is not None], dtype=float)
        if _cl.size:
            print(f"  previous derivative: median loss {np.median(_cl):.5g}, "
                  f"median corr {np.median(_cc):.1f}%")
        # Stage 2 (z0+z1*dt) RE-ENCODES the predicted state every step -- the
        # models never do -- so it is a fair, same-methodology baseline ONLY at
        # 1 step (no re-encode yet). At multi-step it uses information the models
        # cannot, so "3a below stage 2" there is the re-encoding advantage, not
        # f_theta worsening things: the figure draws it DOTTED and its median is
        # printed as reference-only, not as a comparable baseline.
        _sl = np.array([x for x in stats["loss_stage2"] if np.isfinite(x)], dtype=float)
        _sc = np.array([c for c in stats["corr_stage2"] if c is not None], dtype=float)
        if _sl.size and n_steps == 1:
            print(f"  stage 2 (z0+z1*dt): median loss {np.median(_sl):.5g}, "
                  f"median corr {np.median(_sc):.1f}%  [comparable: 1 step]")
        elif _sl.size:
            print(f"  stage 2 (z0+z1*dt): median corr {np.median(_sc):.1f}% "
                  f"-- NOT comparable at {n_steps} steps (re-encodes each step; "
                  f"dotted in the figure), reference only")
    for key, m in ((("a", a), ("b", b)) if stats is not None else ()):
        per_step = stats["n_corr_undefined_per_step"][key]
        flagged = [k for k, n in enumerate(per_step)
                    if k > 0 and n == len(stats["dt"])]
        if flagged:
            print(f"  {m['label']}: correlation UNDEFINED for EVERY window "
                  f"at step(s) {flagged}. No property of a single window "
                  f"explains that -- the predicted delta is constant there "
                  f"(pred[k] == pred[0]) or the real frames are identical. "
                  f"Those steps are absent from the correlation panel.")
    if output_path is None:
        output_path = _default_figure_path(prefix, a, b, seed, n_steps_used,
                                            z1_resync, None)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if trajectory:
        # trajectory without any panel figure: follow the SETUP windows (the
        # same set the panel tool would draw), naming each after its run.
        for run_dir, steps in windows:
            run_dir = Path(run_dir)
            traj_path = output_path.with_name(
                f"{output_path.stem.rsplit('-seed', 1)[0]}-{run_dir.name}.png")
            _trajectory_figure(run_dir, steps, a, b, device, z1_resync,
                                title, traj_path)
            print(f"saved {traj_path}")
    if stats is not None:
        stats_path = output_path.with_name(output_path.stem + "-stats.png")
        _stats_figure(stats, a, b, title, output_path=stats_path, n_steps=n_steps)
        print(f"saved {stats_path}")
    # Return the BASE path (not the -stats one): callers derive both the
    # stats and trajectory names from this stem, as the panel tool's return
    # is also the base figure the extras hang off.
    return output_path, window_strings


def compare_f_theta(path_a: Path, path_b: Path, n_samples: int = 6,
                    t0_range=None,
                     n_steps: int = 2, seed: int = 0,
                     fixed_windows: list[str] | None = None,
                     max_dt: float | None = None, z1_resync: bool = False,
                     f_scale_sweep: bool = False, alpha_sweep: bool = False,
                     n_stats: int = 0, trajectory: bool = False,
                     output_path: Path | None = None,
                     device: str | None = None) -> tuple[Path, list[str]]:
    """Thin orchestrator kept for backward compatibility: runs the panel
    tool when n_samples != 0 (or fixed windows are given) and the statistics
    tool when n_stats > 0. The two are independently callable as
    compare_panels and compare_statistics; this just preserves the single
    entry point and its return (the panel figure's path, or the stats
    figure's path when panels are skipped).

    The two tools load the checkpoints separately -- a deliberate trade: the
    load is trivial beside a 200-window stats pass, and keeping them
    decoupled is worth one extra torch.load when both are asked for.
    """
    draw_panels = n_samples != 0 or bool(fixed_windows)
    result = None
    if draw_panels:
        # The panel tool draws its own trajectories when asked; only hand it
        # trajectory when panels are actually being drawn.
        result = compare_panels(
            path_a, path_b, n_samples=n_samples, n_steps=n_steps, seed=seed,
            fixed_windows=fixed_windows, max_dt=max_dt, z1_resync=z1_resync,
            trajectory=trajectory, output_path=output_path, device=device)
    elif not n_stats:
        print("\n--n-samples 0: no comparison panel drawn.")

    # Trajectory or stats at n_samples=0 both route through the stats tool,
    # which can draw trajectories without a panel. Entered when EITHER is
    # requested (n_stats>0, or trajectory-only with no panels).
    if n_stats or (trajectory and not draw_panels):
        want_traj_here = trajectory and not draw_panels
        stats_result = compare_statistics(
            path_a, path_b, n_stats=n_stats, n_steps=n_steps, seed=seed,
            t0_range=t0_range,
            max_dt=max_dt, z1_resync=z1_resync, f_scale_sweep=f_scale_sweep,
            alpha_sweep=alpha_sweep, trajectory=want_traj_here,
            output_path=output_path, device=device)
        if result is None:
            result = stats_result

    if result is None:
        device_r = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        (_d, a, b, _w, window_strings, prefix, _t,
         n_steps_used, z1_resync) = _setup_comparison(path_a, path_b, device_r, n_samples,
                                            n_steps, seed, fixed_windows,
                                            max_dt, z1_resync, t0_range=t0_range)
        out = output_path or _default_figure_path(prefix, a, b, seed,
                                                   n_steps_used, z1_resync,
                                                   fixed_windows)
        return out, window_strings
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+",
                        help="checkpoint(s). Default a/b modes use the first "
                             "two (an f_theta 3a and 3b). With "
                             "--stage2-compare, ALL of them are treated as "
                             "stage-2 (AE-family) checkpoints and their "
                             "stage-2 rollouts are compared, one curve each.")
    parser.add_argument("--stage2-compare", action="store_true",
                        help="compare the STAGE-2 rollout (z0+z1 dt, causal, "
                             "no f_theta) of every positional checkpoint -- "
                             "one curve per checkpoint, on shared windows. For "
                             "'which stage-2 snapshot rolls out best'. Ignores "
                             "the f_theta a/b machinery and its sweeps.")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--steps", type=int, default=2,
                         help="transitions BOTH models chain, whatever either "
                              "was trained at -- the shared horizon is the "
                              "point of the tool")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-windows", type=str, nargs="+", default=None)
    parser.add_argument("--max-dt", type=float, default=None)
    parser.add_argument("--t0-range", type=float, nargs=2, default=None,
                         metavar=("LO", "HI"),
                         help="restrict to windows whose STARTING step t0 is in "
                              "[LO, HI] (step units, e.g. 7000 30000). Run twice "
                              "-- an early-t0 and a late-t0 band at the SAME "
                              "--steps (== matched Delta-u span) -- to test "
                              "whether coherence tracks log-time span or absolute "
                              "dt: equal corr across bands => the box is log-time.")
    parser.add_argument("--trajectory", action="store_true",
                         help="ALSO save a 3 x (N+1) panel PER WINDOW, "
                              "following it frame by frame: rows real / A / B, "
                              "columns t, t+dt, t+2dt, ... -- for locating a "
                              "divergence in TIME rather than measuring it at "
                              "the end. Each is named after its own run "
                              "(T<temperature>_n<noise>_s<seed>)")
    parser.add_argument("--n-stats", type=int, default=0,
                         help="ALSO compute loss/correlation statistics over "
                              "this many windows (both models, same windows) "
                              "and save a second four-panel figure. The six "
                              "image rows are for looking at; this is what a "
                              "verdict should rest on")
    parser.add_argument("--alpha-sweep", action="store_true",
                         help="with --n-stats, re-run the rollout at alpha "
                              "1.5, 0.5, 0.15 and 0.05 (the h -> 0 limit at "
                              "fixed f_theta). Scheme instability vanishes "
                              "as h shrinks; an unstable learned field "
                              "persists. Substep-cap hits are reported -- a "
                              "clamped point does not probe a smaller step")
    parser.add_argument("--f-scale-sweep", action="store_true",
                         help="with --n-stats, also report end-to-end loss and "
                              "correlation with f_theta scaled by 0, 0.25, "
                              "0.5 and 1. Scale 0 IS stage 2, so this walks "
                              "the line between stage 2 and the trained model "
                              "without retraining: monotone damage means "
                              "f_theta does not belong in the propagation "
                              "path, an interior minimum means a damped one "
                              "is worth keeping")
    parser.add_argument("--z1-resync", action="store_true",
                         help="resync z1 at each real frame for BOTH models; "
                              "default is propagation, the inference regime")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--panels-only", action="store_true",
                         help="run ONLY the image tool (compare_panels): the "
                              "7-column figure and optional trajectories, no "
                              "statistics. Ignores --n-stats and the sweeps.")
    parser.add_argument("--stats-only", action="store_true",
                         help="run ONLY the statistics tool "
                              "(compare_statistics): the 2x4 figure and the "
                              "sweeps, no per-window panels. --n-stats "
                              "defaults to 200 here if left at 0.")
    args = parser.parse_args()
    if args.panels_only and args.stats_only:
        parser.error("--panels-only and --stats-only are mutually exclusive")

    if args.stage2_compare:
        compare_stage2_rollouts(
            args.checkpoints, n_stats=args.n_stats or 200,
            n_steps=args.steps, seed=args.seed, max_dt=args.max_dt,
            t0_range=tuple(args.t0_range) if args.t0_range else None,
            output_path=args.output, device=args.device)
        return

    # a/b modes use the first two positionals (an f_theta 3a and 3b).
    if len(args.checkpoints) < 2:
        parser.error("the default (f_theta a/b) comparison needs two "
                     "checkpoints; give two, or use --stage2-compare for "
                     "one-or-more stage-2 checkpoints")
    checkpoint_a, checkpoint_b = args.checkpoints[0], args.checkpoints[1]

    if args.stats_only:
        compare_statistics(
            checkpoint_a, checkpoint_b,
            n_stats=args.n_stats or 200, n_steps=args.steps, seed=args.seed,
            max_dt=args.max_dt, z1_resync=args.z1_resync,
            f_scale_sweep=args.f_scale_sweep, alpha_sweep=args.alpha_sweep,
            t0_range=tuple(args.t0_range) if args.t0_range else None,
            trajectory=args.trajectory, output_path=args.output,
            device=args.device)
    elif args.panels_only:
        compare_panels(
            checkpoint_a, checkpoint_b,
            n_samples=args.n_samples, n_steps=args.steps, seed=args.seed,
            fixed_windows=args.fixed_windows, max_dt=args.max_dt,
            z1_resync=args.z1_resync, trajectory=args.trajectory,
            t0_range=tuple(args.t0_range) if args.t0_range else None,
            output_path=args.output, device=args.device)
    else:
        compare_f_theta(checkpoint_a, checkpoint_b,
                         n_samples=args.n_samples, n_steps=args.steps,
                         seed=args.seed, fixed_windows=args.fixed_windows,
                         max_dt=args.max_dt, z1_resync=args.z1_resync,
                         f_scale_sweep=args.f_scale_sweep,
                         alpha_sweep=args.alpha_sweep,
                         t0_range=tuple(args.t0_range) if args.t0_range else None,
                         n_stats=args.n_stats, trajectory=args.trajectory,
                         output_path=args.output, device=args.device)


if __name__ == "__main__":
    main()
