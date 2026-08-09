"""
Side-by-side f_theta comparison: TWO checkpoints, IDENTICAL windows.

Seven columns per sample row, labelled with the parsed stage names:

    state(t) | real dx | pred dx (stage 3a) | pred dx (stage 3b)
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
from pathlib import Path

import matplotlib.pyplot as plt
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
from models.constants import LATENT_SPATIAL_SIZE
from models.latent_dynamics import LatentDynamics, integration_kwargs_from_config
from training.checkpoint_components import build_ae_from_checkpoint
from training.datasets import MicrostructureEvolutionDataset
from training.losses import ReconLoss
from utils import load_datasets as load


def _fmt_corr(corr: float | None) -> str:
    """None happens on real data: a quiet window's real dx has ~zero std --
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
    """One checkpoint -> its f_theta, its OWN AE, and its configs."""
    ck = torch.load(lds_checkpoint_path, map_location=device, weights_only=True)
    cfg = ck["config"]
    ae_path = Path(ck["ae_checkpoint"])
    ae, ae_encoder, ae_ck, _, _ = build_ae_from_checkpoint(ae_path, device)
    f_theta = LatentDynamics(
        latent_channels=cfg["latent_channels"], n_theta=cfg["n_theta"],
        latent_spatial=cfg.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
        hidden_dim=cfg["hidden_dim"], n_hidden_layers=cfg["n_hidden_layers"],
        **integration_kwargs_from_config(cfg),
    ).to(device)
    f_theta.load_state_dict(ck["model_state"])
    f_theta.eval()
    return {"path": lds_checkpoint_path, "ck": ck, "config": cfg, "ae": ae,
            "ae_encoder": ae_encoder, "ae_config": ae_ck["config"],
            "ae_path": ae_path, "f_theta": f_theta,
            # Each checkpoint's own training limit, for marking on dt axes.
            "max_dt": (ck.get("data_config") or {}).get("max_dt"),
            "prefix": _parse_stem(lds_checkpoint_path.stem)[0],
            "label": _parse_stem(lds_checkpoint_path.stem)[1]}


def _select_windows(model: dict, n_samples: int, n_steps: int, seed: int,
                     max_dt: float | None, device) -> list[tuple]:
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
    )
    if len(dataset) < n_samples:
        raise ValueError(f"only {len(dataset)} windows available for "
                          f"{n_samples} samples at window_length={n_steps + 1}")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n_samples].tolist()
    windows = []
    for idx in indices:
        run_dir, steps = dataset.window_info(idx)
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
            "dt": [], "n_corr_undefined": 0,
            # PER-STEP: index k holds every window's value after k chained
            # applications. The collapse is counted in APPLICATIONS, not in
            # elapsed time -- two runs with different per-step dt broke at the
            # same frame index -- so this is the axis the failure lives on.
            "step_loss_a": [], "step_loss_b": [],
            "step_corr_a": [], "step_corr_b": []}
    for run_dir, steps in windows:
        row = {}
        for key, m in (("a", a), ("b", b)):
            real_frames, pred_frames, dt_per_step = compute_trajectory(
                run_dir, steps, m["ae"], m["f_theta"], m["ae_config"], device,
                z1_resync=z1_resync)
            losses_k, corrs_k = [], []
            for k in range(len(real_frames)):
                losses_k.append(recon_loss(
                    torch.from_numpy(pred_frames[k])[None, None],
                    torch.from_numpy(real_frames[k])[None, None]).item())
                corrs_k.append(
                    _correlation_pct(pred_frames[0], real_frames[0]) if k == 0
                    else _correlation_pct(pred_frames[k] - pred_frames[0],
                                           real_frames[k] - real_frames[0]))
            out[f"step_loss_{key}"].append(losses_k)
            out[f"step_corr_{key}"].append(corrs_k)
            dt_total = sum(dt_per_step)
            # The ENDPOINT is the last frame of this same trajectory, so the
            # two views of the same run cannot disagree.
            row[key] = {"loss": losses_k[-1], "corr": corrs_k[-1]}
        out["dt"].append(dt_total)
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


def _mark_max_dt(ax, a: dict, b: dict, colours: dict) -> None:
    """Dotted vertical line at each model's own training max_dt.

    CAVEAT WORTH KNOWING: max_dt bounds a SINGLE TRANSITION, while the x axis
    is dt_total, the sum over the window. So the line does not separate
    "trained for" from "extrapolation" -- a 4-step window at dt_total = 4 x
    max_dt is entirely within distribution. It marks the per-transition
    limit, which is what the label says, and is most directly readable on a
    1-step figure.

    Only drawn when the two differ or there is a single value to show; a
    second line exactly on top of the first is clutter.
    """
    seen = {}
    for key, m in (("a", a), ("b", b)):
        value = m.get("max_dt")
        if value:
            seen.setdefault(float(value), []).append((key, m["label"]))
    for value, holders in seen.items():
        key, label = holders[0]
        names = " = ".join(lbl for _, lbl in holders)
        ax.axvline(value, color=colours[key] if len(holders) == 1 else "grey",
                    linestyle=":", linewidth=1.4,
                    label=f"max_dt {names} ({value:g}, per transition)")


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
                   output_path: Path) -> Path:
    """Four panels: loss and correlation, as distributions and against dt."""
    dt = np.array(stats["dt"], dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    colours = {"a": "tab:blue", "b": "tab:red"}
    labels = {"a": a["label"], "b": b["label"]}

    # (0,0) loss distribution, as a CDF -- bin-free, so no bin width has to
    # be chosen for a quantity spanning six decades, and the two curves can
    # be read against each other at any quantile.
    ax = axes[0, 0]
    for key in ("a", "b"):
        v = np.sort(np.array(stats[f"loss_{key}"], dtype=float))
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            ax.plot(v, np.arange(1, v.size + 1) / v.size,
                    color=colours[key], label=labels[key])
    ax.set_xscale("log")
    ax.set_xlabel("end-to-end loss")
    ax.set_ylabel("cumulative fraction of windows")
    ax.set_title("loss distribution (CDF -- lower/left is better)")
    ax.grid(alpha=0.3)
    ax.legend()

    # (0,1) correlation distribution
    ax = axes[0, 1]
    for key in ("a", "b"):
        v = np.array([c for c in stats[f"corr_{key}"] if c is not None],
                      dtype=float)
        if v.size:
            v = np.sort(v)
            ax.plot(v, np.arange(1, v.size + 1) / v.size,
                    color=colours[key], label=labels[key])
    ax.set_xlabel("correlation of predicted vs real dx (%)")
    ax.set_ylabel("cumulative fraction of windows")
    ax.set_title("correlation distribution (CDF -- right is better)")
    ax.grid(alpha=0.3)
    ax.legend()

    # (1,0) loss against dt
    ax = axes[1, 0]
    loss_medians = []
    for key in ("a", "b"):
        c, med, lo, hi = _binned(dt, np.array(stats[f"loss_{key}"], dtype=float))
        if c.size:
            ax.plot(c, med, "o-", color=colours[key], label=labels[key])
            ax.fill_between(c, lo, hi, color=colours[key], alpha=0.15)
            loss_medians.append(med)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _mark_max_dt(ax, a, b, colours)
    ax.set_xlabel("dt_total")
    ax.set_ylabel("median loss (band: quartiles)")
    ax.set_title("loss vs dt")
    _ylim_from_medians(ax, loss_medians)
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # (1,1) correlation against dt
    ax = axes[1, 1]
    for key in ("a", "b"):
        corr = np.array([np.nan if c is None else c
                          for c in stats[f"corr_{key}"]], dtype=float)
        c, med, lo, hi = _binned(dt, corr)
        if c.size:
            ax.plot(c, med, "o-", color=colours[key], label=labels[key])
            ax.fill_between(c, lo, hi, color=colours[key], alpha=0.15)
    ax.set_xscale("log")
    _mark_max_dt(ax, a, b, colours)
    ax.set_xlabel("dt_total")
    ax.set_ylabel("median correlation (%) (band: quartiles)")
    ax.set_title("correlation vs dt")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # (0,2) and (1,2): against the NUMBER OF CHAINED STEPS. The trajectory
    # panels showed two runs with different per-step dt collapsing at the
    # same frame INDEX, so the failure is counted in applications rather
    # than elapsed time -- this is that axis, over the whole sample.
    step_loss_medians = []
    for panel, kind in ((axes[0, 2], "loss"), (axes[1, 2], "corr")):
        for key in ("a", "b"):
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
                panel.plot(centres, med, "o-", color=colours[key],
                            label=labels[key])
                panel.fill_between(centres, lo, hi, color=colours[key],
                                    alpha=0.15)
                if kind == "loss":
                    step_loss_medians.append(med)
        panel.set_xlabel("chained steps applied")
        panel.grid(alpha=0.3, which="both")
        panel.legend()
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_ylabel("median loss (band: quartiles)")
    axes[0, 2].set_title("loss vs number of steps")
    _ylim_from_medians(axes[0, 2], step_loss_medians)
    axes[1, 2].set_ylabel("median correlation (%) (band: quartiles)")
    axes[1, 2].set_title("correlation vs number of steps")

    n = len(stats["dt"])
    note = f"{n} windows"
    if stats["n_corr_undefined"]:
        note += (f"; {stats['n_corr_undefined']} dropped from the correlation "
                  f"panels (undefined: a quiet window's real dx has ~zero std)")
    fig.suptitle(f"{title}\n{note}", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


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
    theta_val = metadata.temperature - metadata.T0
    real_frames = [load.read_phi_half(run_dir / load.snapshot_filename(s), nx, ny)
                    for s in steps]
    dt_per_step = [(steps[i + 1] - steps[i]) * metadata.dt
                    for i in range(len(steps) - 1)]

    with torch.no_grad():
        _, recon_stream_name = resolve_stream_configs_from_checkpoint_config(ae_config)
        ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
        ae_decoder = (ae.pathways[recon_stream_name].decoder if hasattr(ae, "pathways")
                      else ae.decoder)
        x_all = torch.stack([torch.from_numpy(f) for f in real_frames]
                             ).unsqueeze(1).to(device)
        theta_encode = torch.full((len(steps), 1), theta_val,
                                   dtype=torch.float32, device=device)
        encoded = ae_encoder(x_all, theta=theta_encode)
        z0_t = encoded[recon_stream_name][0:1]
        z1_sequence = encoded["deriv"].unsqueeze(0)
        dts = torch.tensor([dt_per_step], dtype=torch.float32, device=device)
        theta = torch.tensor([[theta_val]], dtype=torch.float32, device=device)

        z0_hat_full = f_theta.rollout(z0_t, z1_sequence, dts, theta,
                                       z1_resync=z1_resync)
        pred_frames = [ae_decoder(z0_t)[0, 0].cpu().numpy()]
        for i in range(z0_hat_full.shape[1]):
            pred_frames.append(ae_decoder(z0_hat_full[:, i])[0, 0].cpu().numpy())
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

    n_cols = len(steps)
    rows = [("real", real_frames), (a["label"], pred_a), (b["label"], pred_b)]

    # Per-frame loss and correlation, against the SAME real frame the panel
    # sits above. Both are measured on the DELTA from the start (x - x_0),
    # not on the state: the states are 95% identical background at every
    # frame, so a correlation between states would read ~99% right through
    # the collapse. The delta is what the model is actually predicting, and
    # it is what the six-column figure reports, so the numbers are
    # comparable between the two figures.
    recon_loss = ReconLoss()
    metrics = {}
    for key, frames in (("a", pred_a), ("b", pred_b)):
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
    fig, axes = plt.subplots(3, n_cols, figsize=(3.1 * n_cols, 10.2),
                              squeeze=False)
    elapsed = 0.0
    for col in range(n_cols):
        for row, (label, frames) in enumerate(rows):
            ax = axes[row, col]
            im = ax.imshow(frames[col], cmap="RdBu_r", vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            if row == 0:
                head = "t" if col == 0 else f"t + {elapsed:g}"
                ax.set_title(head, fontsize=9)
            else:
                # Model rows carry their own numbers per frame, so the frame
                # at which each collapses can be read off rather than eyeballed.
                loss, corr = metrics["a" if row == 1 else "b"][col]
                corr_txt = "n/a" if corr is None else f"{corr:.0f}%"
                ax.set_title(f"loss={_format_small(loss)}, corr={corr_txt}",
                              fontsize=8)
            if col == n_cols - 1 and row == 2:
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


def compare_f_theta(path_a: Path, path_b: Path, n_samples: int = 6,
                     n_steps: int = 2, seed: int = 0,
                     fixed_windows: list[str] | None = None,
                     max_dt: float | None = None, z1_resync: bool = False,
                     n_stats: int = 0, trajectory: bool = False,
                     output_path: Path | None = None,
                     device: str | None = None) -> tuple[Path, list[str]]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    a = _load_model(path_a, device)
    b = _load_model(path_b, device)

    if a["ae_path"] != b["ae_path"]:
        print(f"NOTE: the two checkpoints decode through DIFFERENT AEs\n"
              f"  A: {a['ae_path']}\n  B: {b['ae_path']}\n"
              f"  -- each model is judged through its own decoder, so part of "
              f"any difference below may be the AEs', not f_theta's.")

    if fixed_windows:
        windows = [parse_fixed_window(s) for s in fixed_windows]
        # NO per-checkpoint truncation -- unlike check_rollout, which trims a
        # 3-step window to a 3a checkpoint's own window_length=2. Correct for
        # judging one checkpoint on its own terms; fatal for an A/B figure,
        # where it would silently put a 1-step prediction beside a 2-step one
        # in the same row. Both models get every step.
        lens = {len(steps) for _, steps in windows}
        if len(lens) != 1:
            raise ValueError(f"fixed windows have mixed lengths {sorted(lens)}; "
                              f"the comparison needs one common horizon")
    else:
        # n_samples=0 is legitimate: "statistics only, no image figure". The
        # image path is skipped below, but the horizon still has to come from
        # somewhere, so a single window is drawn to define it (and to give
        # --trajectory something to follow if asked).
        windows = _select_windows(a, max(n_samples, 1), n_steps, seed, max_dt,
                                   device)
    window_strings = [f"{run_dir}:{':'.join(str(s) for s in steps)}"
                       for run_dir, steps in windows]
    print(f"z1_resync={z1_resync} for BOTH models (forced equal; the comparison "
          f"is void if one resyncs and the other propagates).")
    for s in window_strings:
        print(f"  {s}")

    recon_loss = ReconLoss()
    draw_panels = n_samples != 0 or bool(fixed_windows)
    n_rows = len(windows)
    if draw_panels:
        fig, axes = plt.subplots(n_rows, 7, figsize=(29, 3.2 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]
    prefix = a["prefix"] or b["prefix"]
    if a["prefix"] and b["prefix"] and a["prefix"] != b["prefix"]:
        # e.g. a 32x32 checkpoint against a 128x128 one: say so in the title
        # rather than silently keeping one side's prefix.
        prefix = f"{a['prefix']} vs {b['prefix']}"
    # THE REGIME BELONGS IN THE TITLE. --steps 4 without --z1-resync and
    # --steps 2 with it are different experiments that produced opposite
    # readings of the same two checkpoints, and nothing on the figure said
    # which was which.
    n_steps_used = len(windows[0][1]) - 1
    regime = (f"{n_steps_used} chained step{'s' if n_steps_used != 1 else ''}, "
              f"z1 {'resynced at each real frame' if z1_resync else 'propagated'}")
    title = (f"{prefix}: {a['label']} vs. {b['label']}" if prefix
              else f"{a['label']} vs. {b['label']}")
    title = f"{title}\n{regime}"
    if draw_panels:
        fig.suptitle(title, fontsize=13)

    losses = {"a": [], "b": []}
    for row, (run_dir, steps) in enumerate(windows if draw_panels else []):
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
        # One symmetric scale for real AND both predictions, from real alone:
        # derived from either prediction it would saturate differently per
        # column and the visual comparison would lie.
        d_lo, d_hi = _padded_bounds(real_delta, 1.0, symmetric=True)
        # BOTH error columns on ONE scale, set by the worse model -- which
        # error panel is fuller is the figure's entire question.
        e_lo, e_hi = _padded_bounds(
            np.concatenate([per["a"]["error"].ravel(), per["b"]["error"].ravel()]),
            1.0, symmetric=True)
        # The B-A column shares the ERROR scale. It equals error_B - error_A
        # exactly (the shared real_delta cancels), so it belongs to the same
        # family of quantities and reads directly against the two error
        # panels beside it -- a seventh independent scale on a seven-column
        # figure is one more number per row for the reader to hold. The cost
        # is that a subtle structured change is flattened whenever one
        # model's error is large; the row's own printed error scale says when
        # that is happening.
        diff = per["b"]["pred_delta"] - per["a"]["pred_delta"]
        f_lo, f_hi = e_lo, e_hi

        steps_txt = f"{run_dir.name}:{steps[0]}\u2192{steps[-1]} ({len(steps) - 1} steps)"
        cells = [
            (x_t, -state_scale, state_scale,
             f"state(t)\n{steps_txt}\ndt_total={dt_total:g}"),
            (real_delta, d_lo, d_hi, f"real dx\nscale=[{d_lo:.3f}, {d_hi:.3f}]"),
            (per["a"]["pred_delta"], d_lo, d_hi,
             f"pred dx ({a['label']})\nloss={_format_small(per['a']['loss'])}, "
             f"corr={_fmt_corr(per['a']['corr'])}"),
            (per["b"]["pred_delta"], d_lo, d_hi,
             f"pred dx ({b['label']})\nloss={_format_small(per['b']['loss'])}, "
             f"corr={_fmt_corr(per['b']['corr'])}"),
            (per["a"]["error"], e_lo, e_hi, f"error {a['label']}"),
            (per["b"]["error"], e_lo, e_hi, f"error {b['label']}"),
            (diff, f_lo, f_hi,
             f"{b['label']} \u2212 {a['label']}\n(= error {b['label']} "
             f"\u2212 error {a['label']})"),
        ]
        # cell_title, NOT title: `title` is the FIGURE's title, and reusing
        # the name here rebound it to the last panel's caption -- the stats
        # figure was then headed "stage 3b - stage 3a (= error ...)".
        for col, (img, lo, hi, cell_title) in enumerate(cells):
            ax = axes[row, col]
            im = ax.imshow(img, cmap="RdBu_r", vmin=lo, vmax=hi)
            ax.set_title(cell_title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 6:
                # ONE colorbar for columns 4, 5 and 6: they share a scale, so
                # a second bar repeating the same range is clutter on an
                # already-wide figure. fraction=0.046 attached to a single
                # axis, as check_rollout does -- a colorbar spanning
                # axes[row, 4:6] is the construct tight_layout cannot place,
                # and it warned on every run.
                fig.colorbar(im, ax=axes[row, 6], fraction=0.046)

    if draw_panels:
        med_a = float(np.median(losses["a"]))
        med_b = float(np.median(losses["b"]))
        print(f"\nmedian end-to-end loss over {n_rows} shared windows: "
              f"{a['label']}={med_a:.5f}  {b['label']}={med_b:.5f}  "
              f"({(a['label'] if med_a < med_b else b['label'])} better by "
              f"{max(med_a, med_b) / max(min(med_a, med_b), 1e-30):.2f}x)")
        print(f"Medians over {n_rows} windows are indicative only -- rerun "
              f"with --n-samples 24 (or the RMS tools) before concluding.")

    if output_path is None:
        # '128x128-stage3a_vs_stage3b-seed0.png' -- the parsed pieces, with
        # the label's space compacted back out for the filesystem.
        sa = a["label"].replace(" ", "")
        sb = b["label"].replace(" ", "")
        name = f"{prefix}-{sa}_vs_{sb}" if prefix and " vs " not in prefix \
            else f"{sa}_vs_{sb}"
        # No seed in the name for --fixed-windows: the seed played no part in
        # selecting them, and a stamped seed would suggest a rerun with
        # another seed changes the windows. check_rollout's convention.
        suffix = "" if fixed_windows else f"-seed{seed}"
        # The regime goes in the FILENAME too: without it a --steps 4 run
        # silently overwrote the --steps 2 run it should be compared against,
        # leaving two different experiments under one name.
        regime_tag = f"-{n_steps_used}step{'s' if n_steps_used != 1 else ''}"
        regime_tag += "-resync" if z1_resync else "-propagated"
        output_path = (_PYTHON_ROOT.parent / "output" / "rollout_check_png"
                       / f"{name}{regime_tag}{suffix}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if draw_panels:
        fig.tight_layout()
        fig.savefig(output_path, dpi=110)
        plt.close(fig)
        print(f"saved {output_path}")
    else:
        # --n-samples 0: statistics only. output_path was still constructed,
        # because the stats and trajectory figures derive their names from it.
        print("\n--n-samples 0: no comparison panel drawn.")

    if trajectory:
        # ONE FIGURE PER WINDOW, each named by the run it came from. The run
        # directory name already carries the parameters that identify it --
        # T925_n035_s5 is temperature, noise and sim seed -- so the figure is
        # self-describing, and the selection seed (which chose the window set,
        # not the physics) has no place in the name of a single-window plot.
        for run_dir, steps in windows:
            run_dir = Path(run_dir)
            traj_path = output_path.with_name(
                f"{output_path.stem.rsplit('-seed', 1)[0]}-{run_dir.name}.png")
            _trajectory_figure(run_dir, steps, a, b, device, z1_resync,
                                title, traj_path)
            print(f"saved {traj_path}")

    if n_stats:
        # A SEPARATE, LARGER sample: the panel windows were chosen to be
        # looked at, and six of them disagreed between seed 0 and seed 1 on
        # the size of the gap. Same selection machinery, same shared windows,
        # no images.
        stat_windows = _select_windows(a, n_stats, len(windows[0][1]) - 1,
                                        seed, max_dt, device)
        print(f"\ncollecting statistics over {len(stat_windows)} windows "
              f"(both models, same windows)...")
        stats = collect_stats(a, b, stat_windows, device, z1_resync)
        for key, m in (("a", a), ("b", b)):
            losses_k = np.array(stats[f"loss_{key}"], dtype=float)
            corrs_k = np.array([c for c in stats[f"corr_{key}"]
                                 if c is not None], dtype=float)
            print(f"  {m['label']}: median loss {np.median(losses_k):.5g}, "
                  f"median corr {np.median(corrs_k):.0f}%")
        for key, m in (("a", a), ("b", b)):
            per_step = stats["n_corr_undefined_per_step"][key]
            flagged = [k for k, n in enumerate(per_step)
                        if k > 0 and n == len(stats["dt"])]
            if flagged:
                print(f"  {m['label']}: correlation UNDEFINED for EVERY window "
                      f"at step(s) {flagged}. No property of a single window "
                      f"explains that -- the predicted delta is constant there "
                      f"(pred[k] == pred[0]) or the real frames are identical. "
                      f"Those steps are absent from the correlation panel.")
        stats_path = output_path.with_name(output_path.stem + "-stats.png")
        _stats_figure(stats, a, b, title, stats_path)
        print(f"saved {stats_path}")

    return output_path, window_strings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_a", type=Path)
    parser.add_argument("checkpoint_b", type=Path)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--steps", type=int, default=2,
                         help="transitions BOTH models chain, whatever either "
                              "was trained at -- the shared horizon is the "
                              "point of the tool")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-windows", type=str, nargs="+", default=None)
    parser.add_argument("--max-dt", type=float, default=None)
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
    parser.add_argument("--z1-resync", action="store_true",
                         help="resync z1 at each real frame for BOTH models; "
                              "default is propagation, the inference regime")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    compare_f_theta(args.checkpoint_a, args.checkpoint_b,
                     n_samples=args.n_samples, n_steps=args.steps,
                     seed=args.seed, fixed_windows=args.fixed_windows,
                     max_dt=args.max_dt, z1_resync=args.z1_resync,
                     n_stats=args.n_stats, trajectory=args.trajectory,
                     output_path=args.output, device=args.device)


if __name__ == "__main__":
    main()
