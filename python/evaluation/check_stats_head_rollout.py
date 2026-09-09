#!/usr/bin/env python
"""Diagnostic for the stats0_predict term: is the FROZEN stats head a trustworthy
latent-space read of the statistics (energy foremost) along a PREDICTED rollout?

Run BEFORE spending a training run on stats0_predict. For a few test windows it
rolls f_theta out and plots, per statistic, vs rollout step:

  * stats_head(z_true)  -- the head's read of the REAL trajectory's latents
  * stats_head(z_hat)   -- the head's read of f_theta's PREDICTED latents
  * the REAL statistic from statistics.csv (ground truth), when available

Three questions, one figure:

  1. OFF-DISTRIBUTION RELIABILITY. stats_head was trained on real latents; along a
     predicted rollout the latent drifts off that distribution. If stats_head(z_hat)
     stays in the CSV's plausible range, the head reads sensibly there and the
     latent-space term is on solid ground. If it goes WILD, the term is on sand --
     compute the free energy from the DECODED field instead (decoder + the
     phase_field.md formula F = integral[ f(phi,T) + kappa/2 |grad phi|^2 ]).

  2. HEAD ACCURACY on real latents: does stats_head(z_true) track the CSV curve?
     If not, the head itself is a poor readout even before any rollout drift.

  3. THE MONOTONICITY / MOTH SIGNATURE (energy specifically). phase_field.md line
     31: the solver's total free energy DECREASES monotonically. If the REAL energy
     falls while the PREDICTED energy (stats_head(z_hat)) RISES, that is the moths
     forming -- a Lyapunov violation -- and it is exactly what stats0_predict aims
     to penalize. If predicted energy also falls, monotonically, there is little for
     the term to fix on this sample.

Usage (from python/, run as a module so imports resolve):
    python -m evaluation.check_stats_head_rollout \
        checkpoints/stage3b/128x128-stage3b-....pt \
        --n-samples 6 --steps 16 --seed 0 --stat energy

The AE ancestor (encoder + stats head) is read from the LDS checkpoint's own
"ae_checkpoint" field unless --ae-checkpoint overrides it.

NOTE: written against the current APIs (build_ae_from_checkpoint, LatentDynamics +
integration_kwargs_from_config, MicrostructureEvolutionDataset, and train_lds's
_load_frozen_stats_head). It reuses, never reimplements, the rollout -- verify the
dataset-stats access on your tree (see the marked block); everything else is the
same call sites eval already uses.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

_PYTHON_ROOT = Path(__file__).resolve().parent.parent

from models.constants import LATENT_SPATIAL_SIZE, N_THETA  # noqa: E402
from models.latent_dynamics import LatentDynamics, integration_kwargs_from_config  # noqa: E402
from training.checkpoint_components import build_ae_from_checkpoint  # noqa: E402
from training.datasets import (  # noqa: E402
    MicrostructureEvolutionDataset, complete_run_dirs, split_run_dirs,
)
from training.train_lds import _load_frozen_stats_head  # noqa: E402  (the loader we added)
from utils.plots import _save_figure  # noqa: E402  (retry-safe, non-fatal figure write)


def _load(lds_ckpt_path: Path, ae_ckpt_path: Path | None, device):
    # Accept a stage-3 LDS checkpoint OR a stage-4/5 JOINT checkpoint. The joint
    # format stores f_theta as f_theta_state / lds_config (not model_state / config),
    # and carries the REFINED encoder -- the right one to pair with its co-trained
    # f_theta. split_joint_checkpoint_for_evaluation is the designed adapter: it
    # writes a stage-3-shaped LDS view and an AE view, so the stage-3 loading below
    # runs unchanged (this is why it exists -- see its docstring).
    _raw = torch.load(lds_ckpt_path, map_location=device, weights_only=True)
    from orchestration.checkpoint_identification import identify_checkpoint_stage
    _stage = identify_checkpoint_stage(_raw)
    if _stage.startswith("stage 4") or _stage.startswith("stage 5"):
        import tempfile
        from training.checkpoint_components import split_joint_checkpoint_for_evaluation
        _views = Path(tempfile.mkdtemp(prefix="stats_head_rollout_views_"))
        _ae_view, _lds_view = split_joint_checkpoint_for_evaluation(lds_ckpt_path, _views)
        print(f"  {_stage} joint checkpoint: split into LDS + (refined) AE views for evaluation")
        lds_ckpt_path = _lds_view
        if ae_ckpt_path is None:
            ae_ckpt_path = _ae_view          # the refined encoder, not the stage-2 ancestor
    lds_checkpoint = torch.load(lds_ckpt_path, map_location=device, weights_only=True)
    lds_config = lds_checkpoint["config"]
    if ae_ckpt_path is None:
        ae_ckpt_path = Path(lds_checkpoint["ae_checkpoint"])
    ae, ae_encoder, ae_checkpoint, _stream_configs, _recon = build_ae_from_checkpoint(
        ae_ckpt_path, device)
    ae_config = ae_checkpoint["config"]
    # n_theta=N_THETA, NOT lds_config["n_theta"]: stage-4 checkpoints do not store the
    # key (only train_lds writes it) and old 1-theta ancestors upgrade by zero-padding
    # -- the same contract as model_assembly.build_models_from_components.
    f_theta = LatentDynamics(
        latent_channels=lds_config["latent_channels"], n_theta=N_THETA,
        latent_spatial=lds_config.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
        hidden_dim=lds_config["hidden_dim"], n_hidden_layers=lds_config["n_hidden_layers"],
        **integration_kwargs_from_config(lds_config),
    ).to(device)
    from models.encoder import zero_pad_theta_columns
    f_theta.load_state_dict(zero_pad_theta_columns(lds_checkpoint["model_state"], f_theta))
    f_theta.eval()
    stats_head, _stats_loss = _load_frozen_stats_head(ae_checkpoint, ae_config, device)
    if stats_head is None:
        raise SystemExit("The AE ancestor has no stats head (trained with stats0_weight=0); "
                         "stats0_predict and this diagnostic need one.")
    return (f_theta, ae_encoder, stats_head, lds_config, ae_ckpt_path,
            lds_checkpoint.get("data_config", {}), ae_config)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lds_checkpoint", type=Path)
    ap.add_argument("--ae-checkpoint", type=Path, default=None)
    ap.add_argument("--base", type=Path, default=_PYTHON_ROOT.parent / "datasets")
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--n-scatter", type=int, default=500,
                    help="number of windows in the step_vs_dt scatter (cheap: single "
                         "step each, latents cached)")
    ap.add_argument("--n-windows", "--n-samples", dest="n_windows", type=int, default=6,
                    help="how many windows to plot. ONLY this many runs are encoded (one "
                         "window each) -- this is a plot-only diagnostic, it aggregates "
                         "nothing over the full set, so encoding more would be pure waste.")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stat", type=str, default=None,
                    help="focus one statistic (default: all the head predicts). "
                         "'energy' is the free-energy Lyapunov quantity. 'latent_norm' is a "
                         "PSEUDO-STAT: plots ||z_hat|| vs ||z_true|| directly (no head) -- the "
                         "measurement that tells DIVERGENCE from mere DRIFT. 'step_vs_dt': scatter "
                         "of one-step ||added term|| vs dt -- tells a dt-driven blowup (cap dt) "
                         "from a broken f_theta.")
    ap.add_argument("--min-step", type=int, default=None,
                    help="override the checkpoint's min_step (default: use the checkpoint's)")
    ap.add_argument("--min-normalized-stdev-phi", type=float, default=None)
    ap.add_argument("--max-dt", type=float, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    (f_theta, ae_encoder, stats_head, lds_config, ae_ckpt,
     lds_checkpoint_data_config, ae_config) = _load(args.lds_checkpoint, args.ae_checkpoint, device)
    # MATCH TRAINING's coordinate. The model steps in dt_window, which is Delta-u in
    # log10_t mode -- and the dataset only produces that (plus a physical dt) when
    # return_phys_dt is set the SAME way train_lds set it. Feeding a Delta-u model a
    # physical dt (return_phys_dt=False on a log10_t run) makes it step in the wrong
    # coordinate, so the blowup would be a diagnostic artefact, not the model's.
    # RECONSTRUCT THE TRAINING DATASET FROM THE CHECKPOINT. This is the crux: the
    # model was trained on windows built with a specific time_coordinate (log10_t
    # -> dt_window is Delta-u, ~0.1) and derivative_source (previous_quotient -> the
    # deriv stream is the backward quotient q, not the encoder's z1 head). Building
    # the diagnostic dataset with the DEFAULTS ('t', 'z1') instead fed f_theta a
    # physical dt 10^2-10^4x larger than it ever saw, plus the wrong derivative --
    # which produced a spurious 10^15 "divergence" that was entirely this mismatch.
    # Every param below is read from the checkpoint; CLI flags only OVERRIDE.
    _time_coord = lds_config.get("time_coordinate", "t")
    _deriv_src = lds_config.get("derivative_source", "z1")
    _deriv_time = lds_config.get("derivative_time", "initial")
    _dcfg = lds_checkpoint_data_config
    _return_phys = (_time_coord == "log10_t")
    print(f"dataset reconstructed from checkpoint: time_coordinate={_time_coord}, "
          f"derivative_source={_deriv_src}, derivative_time={_deriv_time}, "
          f"min_step={_dcfg.get('min_step')}, "
          f"min_normalized_stdev_phi={_dcfg.get('min_normalized_stdev_phi')}, "
          f"max_dt={_dcfg.get('max_dt')}")

    stat_names = list(stats_head.stat_names)
    window_length = args.steps + 1

    run_dirs = complete_run_dirs(args.base, args.size, args.size)
    _, _, test_dirs = split_run_dirs(run_dirs, 0.2, 0.1, seed=args.seed)
    # Encode ONLY as many runs as windows we will plot -- one window per run, for
    # temperature variety. Nothing here aggregates over the full test set, so
    # encoding the other ~430 runs (the old behaviour) was pure waste.
    # step_vs_dt needs many windows for a scatter -> encode more runs (cached, ~free);
    # the per-window trajectory modes only plot a few, so cap tightly there.
    _n_runs = max(args.n_windows, 12) if args.stat == "step_vs_dt" else args.n_windows
    test_dirs = test_dirs[:_n_runs]
    print(f"encoding {len(test_dirs)} run(s) (one plotted window each)...", flush=True)
    _min_step = args.min_step if args.min_step is not None else _dcfg.get("min_step", 0)
    _mnsp = (args.min_normalized_stdev_phi if args.min_normalized_stdev_phi is not None
             else _dcfg.get("min_normalized_stdev_phi"))
    _max_dt = args.max_dt if args.max_dt is not None else _dcfg.get("max_dt")
    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=ae_encoder, device=device, window_length=window_length,
        min_step=_min_step, min_normalized_stdev_phi=_mnsp,
        max_dt=_max_dt, encode_both_streams=True,
        normalize_phi=ae_config.get("normalize_phi", False),
        time_coordinate=_time_coord,           # MATCH TRAINING (was defaulting to 't')
        derivative_source=_deriv_src,          # MATCH TRAINING (was defaulting to 'z1')
        return_phys_dt=_return_phys,
        # ---- VERIFY ON YOUR TREE: request the real statistics so the CSV curve
        # can be drawn. If your MicrostructureEvolutionDataset exposes stats via a
        # different kwarg/return, adjust here; the head-vs-head curves work without it.
        # include_stats=True, stat_names=stat_names,
    )
    if len(dataset) == 0:
        raise SystemExit("No test windows after filtering -- loosen the filters.")
    print(f"{len(dataset)} windows available; plotting {min(args.n_windows, len(dataset))}.")

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(dataset), size=min(args.n_windows, len(dataset)), replace=False)

    if args.stat == "step_vs_dt":
        # Single-step diagnostic: is the norm blowup driven by dt (a config fix --
        # dt_cap/max_dt) or by f_theta itself (a model fix)? Plot ||z0_hat_step1 - z0||
        # (the ADDED term of ONE f_theta application: z0 + deriv*dt + correction - z0)
        # vs that step's dt, across many windows, log-log, with propto dt and propto
        # dt^2 reference slopes. Follows dt/dt^2 -> the update is multiplying a large dt
        # through, so cap it. Large even at small dt -> f_theta's output is broken.
        _n = min(len(dataset), args.n_scatter)
        _sel = rng.choice(len(dataset), size=_n, replace=False)
        dts, added, z0n, dphys = [], [], [], []
        for wi in _sel:
            item = dataset[int(wi)]
            w0, w1, dtw, th = item[0], item[1], item[2], item[-1]
            _dp = item[3] if len(item) == 5 else item[2]   # physical dt (5-tuple) else dt_window
            w0 = w0.unsqueeze(0).to(device); w1 = w1.unsqueeze(0).to(device)
            dtw = dtw.unsqueeze(0).to(device); th = th.unsqueeze(0).to(device)
            z0 = w0[:, 0]
            with torch.no_grad():
                z1hat = f_theta.rollout(z0, w1, dtw, th, z1_resync=False)[:, 1]  # 1st predicted step
            dts.append(float(dtw[0, 0]))                                         # dt_window
            dphys.append(float(np.asarray(_dp).reshape(-1)[0]))                  # physical dt
            added.append(float((z1hat - z0).flatten().pow(2).mean().sqrt()))     # RMS added term
            z0n.append(float(z0.flatten().pow(2).mean().sqrt()))                 # ||z0|| (RMS)
        dts = np.asarray(dts); added = np.asarray(added)
        z0n = np.asarray(z0n); dphys = np.asarray(dphys)
        ok = (np.isfinite(dts) & np.isfinite(added) & np.isfinite(z0n)
              & (dts > 0) & (added > 0) & (z0n > 0))
        dts, added, z0n, dphys = dts[ok], added[ok], z0n[ok], dphys[ok]
        rel = added / z0n                                       # ||added|| / ||z0|| (dimensionless)

        _xlabel = ("step-1 Delta-u (dt_window)" if _return_phys else "step-1 dt (dt_window)")
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        # LEFT: absolute added magnitude
        ax.scatter(dts, added, s=12, alpha=0.5, color="tab:red", label="||z0_hat_step1 - z0||")
        if len(dts):
            x0, y0 = np.median(dts), np.median(added)
            xr = np.array([dts.min(), dts.max()])
            ax.plot(xr, y0 * (xr / x0), "--", color="gray", lw=1, label="propto dt")
            ax.plot(xr, y0 * (xr / x0) ** 2, ":", color="gray", lw=1, label="propto dt^2")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(_xlabel); ax.set_ylabel("||added term|| (RMS)")
        ax.set_title("added magnitude vs step")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
        # RIGHT: RELATIVE -- ||added||/||z0||. Above 1 (grey) the step exceeds the
        # latent itself: one f_theta application already blows the norm up.
        ax2.scatter(dts, rel, s=12, alpha=0.5, color="tab:purple", label="||added|| / ||z0||")
        ax2.axhline(1.0, color="grey", lw=1, ls="--", label="added = ||z0|| (blowup threshold)")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel(_xlabel); ax2.set_ylabel("||added|| / ||z0||")
        ax2.set_title("RELATIVE step size (>1 = one-step blowup)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")
        fig.suptitle(f"one-step added term vs step size -- {args.lds_checkpoint.name} "
                     f"(time_coordinate={_time_coord})", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = args.output or (_PYTHON_ROOT.parent / "output"
                              / args.lds_checkpoint.resolve().parent.name
                              / (args.lds_checkpoint.stem + "-step_vs_dt.png"))
        out.parent.mkdir(parents=True, exist_ok=True)
        if _save_figure(fig, out, dpi=110):
            print(f"wrote {out}")
        if len(dts) > 2:
            _slope = np.polyfit(np.log(dts), np.log(added), 1)[0]
            _same = (len(dts) == len(dphys) and np.allclose(dts, dphys))
            print(f"\\n  dt_window range: "
                  f"{dts.min():.3g} .. {dts.max():.3g};  physical dt range: "
                  f"{dphys.min():.3g} .. {dphys.max():.3g}"
                  + ("   <- IDENTICAL: dt_window IS physical dt (not Delta-u) in this pipeline"
                     if _same else "   (differ: dt_window is a transformed coordinate)"))
            print(f"  log-log slope ||added|| vs dt_window: {_slope:.2f}  "
                  f"(~1 => propto dt [dt-driven]; ~0 => dt-independent [f_theta])")
            print(f"  ||added||/||z0||: median {np.median(rel):.3g}, max {rel.max():.3g}  "
                  f"({(rel > 1).mean() * 100:.0f}% of steps EXCEED ||z0||)")
        return

    # latent_norm is a two-row view: the norm ||z0|| itself, and its per-step
    # GROWTH FACTOR ||z_{k+1}|| / ||z_k|| (>1 = diverging). derivative_source=
    # previous_quotient means the ONLY latent stream is z0 (state) -- the deriv fed
    # to f_theta is the backward quotient of z0's own history, not a separate z1 --
    # so ||z0|| IS the norm that matters; there is no z1 stream to split out.
    if args.stat == "latent_norm":
        focus = ["latent_norm", "growth_factor"]
    else:
        focus = [args.stat] if args.stat else stat_names
    # `latent_norm` is a PSEUDO-STAT (not one the head predicts): it plots the raw
    # per-window RMS magnitude of the latent, ||z_hat|| vs ||z_true||, straight from
    # the rollout -- the measurement that adjudicates DIVERGENCE (||z_hat|| explodes)
    # vs mere DRIFT (||z_hat|| bounded, only its direction wrong). The head plots
    # show latent behaviour only THROUGH the head's own distortion; this shows it
    # directly, so it can't lie about which failure is happening.
    fig, axes = plt.subplots(len(focus), len(idxs),
                             figsize=(3.2 * len(idxs), 3.4 * len(focus)), squeeze=False)
    _row_vals = {r: [] for r in range(len(focus))}   # all plotted y per stat-row, for y-saturation
    _drew_real = False
    _true_ratios = []  # per-step ||z0_true_{k+1}||/||z0_true_k|| on the ACTUAL rollout, pooled
                       # across windows. Reported as |ln ratio| (geometric/symmetric) -> the
                       # physical band that sets L_z0_growth's z0_growth_scale and tolerance eps.
                       # TRUE (finite) values only; the predicted rollout overflows to inf/nan.

    for si, wi in enumerate(idxs):
        col = si
        item = dataset[int(wi)]
        window0, window1, dt_window, theta = item[0], item[1], item[2], item[-1]
        window0 = window0.unsqueeze(0).to(device)   # (1, L, C, H, W)
        window1 = window1.unsqueeze(0).to(device)
        dt_window = dt_window.unsqueeze(0).to(device)
        theta = theta.unsqueeze(0).to(device)

        z0 = window0[:, 0]
        z_true = window0[:, 1:]                       # (1, n, C, H, W)
        with torch.no_grad():
            z_hat = f_theta.rollout(z0, window1, dt_window, theta, z1_resync=False)[:, 1:]
            n = z_hat.shape[1]
            sh_true = stats_head(z_true.reshape(n, *z_true.shape[2:])).cpu().numpy()   # (n, Ns)
            sh_hat = stats_head(z_hat.reshape(n, *z_hat.shape[2:])).cpu().numpy()
            # RMS latent magnitude per rollout step (the raw thing, no head)
            ln_true = z_true.reshape(n, -1).pow(2).mean(dim=1).sqrt().cpu().numpy()     # (n,)
            ln_hat = z_hat.reshape(n, -1).pow(2).mean(dim=1).sqrt().cpu().numpy()
            # per-step growth factor ||z_{k+1}|| / ||z_k||, from z0 BEFORE the window
            # (window0[:,0]) so step 1 has a predecessor. Prepend ||z0|| to the hat
            # series; ratios are then aligned to steps 1..n.
            _z0n = z0.reshape(1, -1).pow(2).mean(dim=1).sqrt().cpu().numpy()             # (1,)
            _hat_chain = np.concatenate([_z0n, ln_hat])                                  # (n+1,)
            _true_chain = np.concatenate([_z0n, ln_true])
            gf_hat = _hat_chain[1:] / np.maximum(_hat_chain[:-1], 1e-30)                 # (n,)
            gf_true = _true_chain[1:] / np.maximum(_true_chain[:-1], 1e-30)
            _true_ratios.extend([r for r in gf_true.tolist() if np.isfinite(r)])
        steps = np.arange(1, n + 1)

        for row, sname in enumerate(focus):
            ax = axes[row][col]
            if sname == "latent_norm":
                ax.plot(steps, ln_true, "-o", ms=3, color="tab:green", label="||z_true||")
                ax.plot(steps, ln_hat, "--s", ms=3, color="tab:red", label="||z_hat||")
                _row_vals[row].extend(ln_true.tolist())
                _row_vals[row].extend(ln_hat.tolist())
            elif sname == "growth_factor":
                ax.plot(steps, gf_true, "-o", ms=3, color="tab:green", label="||z_true|| ratio")
                ax.plot(steps, gf_hat, "--s", ms=3, color="tab:red", label="||z_hat|| ratio")
                ax.axhline(1.0, color="grey", lw=1.0, ls=":", alpha=0.8)  # stability boundary
                _row_vals[row].extend(gf_hat.tolist())
            else:
                j = stat_names.index(sname)
                ax.plot(steps, sh_true[:, j], "-o", ms=3, color="tab:green",
                        label="stats_head(z_true)")
                ax.plot(steps, sh_hat[:, j], "--s", ms=3, color="tab:red",
                        label="stats_head(z_hat)")
                _row_vals[row].extend(sh_true[:, j].tolist())
                _row_vals[row].extend(sh_hat[:, j].tolist())
                # real CSV curve, if the dataset returned stats (see VERIFY block above)
                if len(item) >= 5 and torch.is_tensor(item[-2]):
                    real = item[-2].cpu().numpy()          # (n, Ns) if include_stats
                    if real.ndim == 2 and real.shape[1] == len(stat_names):
                        ax.plot(steps, real[:, j], ":", color="black", alpha=0.7, label="real (CSV)")
                        _row_vals[row].extend(real[:, j].tolist())
                        _drew_real = True
            if row == len(focus) - 1:
                ax.set_xlabel("rollout step")   # x is the number of f_theta applications
            ax.margins(x=0)
            ax.set_xlim(0, args.steps + 0.5)     # hard left edge at 0 (data unchanged)
            if row == 0:
                ax.set_title(f"window {int(wi)}", fontsize=8)
            if col == 0:
                ax.set_ylabel(sname, fontsize=9)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)
    # Y-SATURATION: the off-distribution blowups (1e36) flatten everything else to
    # zero. Clip each stat-row's y so the bottom (and top) come from the 10th/90th
    # PERCENTILE of plotted points -- 90% of points stay in frame; the handful of
    # extreme excursions are allowed to run off, with a small margin so they are
    # still visibly "off the chart" rather than clipped exactly at the edge.
    for row, sname in enumerate(focus):
        v = np.asarray([x for x in _row_vals[row] if np.isfinite(x)])
        if not v.size:
            continue
        a = np.abs(v)
        on_dist = np.median(a[a > 0]) if np.any(a > 0) else 1.0   # typical magnitude
        blows_up = a.max() > 1e3 * on_dist
        if sname in ("latent_norm", "growth_factor"):
            # strictly positive -> plain log. latent_norm: a flat line = bounded
            # rollout. growth_factor: flat at 1 = stable; a CONSTANT level > 1 =
            # linear instability (one growth rate); RISING = accelerating divergence.
            for col in range(len(idxs)):
                axes[row][col].set_yscale("log")
                axes[row][col].tick_params(axis="y", labelsize=7)
        elif sname == "energy" or blows_up:
            # symlog: sign-preserving, linear near 0 (linthresh = the on-distribution
            # magnitude), log in both tails so the sign AND the many-decade blowup are
            # BOTH visible, instead of a flat line at zero.
            lt = max(on_dist, np.finfo(float).tiny)
            for col in range(len(idxs)):
                _ax = axes[row][col]
                _ax.set_yscale("symlog", linthresh=lt)
                _ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
                # Show only every 3rd decade (10^0, 10^3, ...) each side of 0 --
                # a tick per power packs 25+ labels into a short panel, smearing
                # them into an unreadable stripe.
                _ax.yaxis.set_major_locator(
                    mticker.SymmetricalLogLocator(base=1000.0, linthresh=lt))
                _ax.yaxis.set_minor_locator(mticker.NullLocator())
                _ax.tick_params(axis="y", labelsize=7)
        else:
            lo, hi = np.percentile(v, 10), np.percentile(v, 90)
            if hi > lo:
                pad = 0.1 * (hi - lo)
                for col in range(len(idxs)):
                    axes[row][col].set_ylim(lo - pad, hi + pad)

    _curves = ("green ||z_true|| / red ||z_hat||" if args.stat == "latent_norm"
               else "green z_true / red z_hat" + (" / black real" if _drew_real else ""))
    # The energy-specific guidance (symlog axis, "should DECREASE", moth signature)
    # applies ONLY when the plotted stat is energy; it was previously appended
    # unconditionally, so a latent_norm/step_vs_dt figure carried an irrelevant
    # energy caption. Stat-aware now.
    _stat_note = ("; energy on SYMLOG (sign-preserving, linthresh=on-distribution scale). "
                  "energy should DECREASE (phase_field.md l.31) — predicted RISING where "
                  "real falls = moth signature" if args.stat == "energy" else "")
    fig.suptitle(f"stats head on predicted vs real rollout — {args.lds_checkpoint.name}\n"
                 f"{_curves}{_stat_note}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    # Mirror the checkpoint's OWN stage subfolder (stage3a / stage3b), not a
    # hardcoded "stage3" -- otherwise 3a and 3b figures collide under one name.
    _stage_dir = args.lds_checkpoint.resolve().parent.name  # e.g. "stage3b"
    out = args.output or (_PYTHON_ROOT.parent / "output" / _stage_dir
                          / (args.lds_checkpoint.stem + "-stats_head_rollout.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    # Retry-safe, non-fatal: on Windows a just-written PNG is intermittently held
    # for a few ms by an AV/Defender scan or a viewer, so a bare savefig can raise
    # OSError [Errno 22]. _save_figure retries, warns-and-skips, and closes the fig.
    if _save_figure(fig, out, dpi=110):
        print(f"wrote {out}")
    # z_growth_scale / tolerance from the ACTUAL rollout's growth ratio
    # ||z_true_{k+1}||/||z_true_k||. The real trajectory is stable (ratio ~ 1), so
    # L_z_growth = ReLU(ratio - (1+eps)) is ~0 on it BY CONSTRUCTION -- its "scale"
    # is therefore not a magnitude but the PHYSICAL FLUCTUATION BAND of the real
    # ratio: how far from 1 the norm legitimately wanders step to step. That band
    # sets both eps (the tolerance below which growth is real dynamics, not
    # instability) and z_growth_scale. Estimated on TRUE (finite) values -- the
    # predicted rollout overflows to inf/nan and is the pathology, not a reference.
    if _true_ratios:
        r = np.asarray([x for x in _true_ratios if x > 0])
        lr = np.abs(np.log(r))   # |ln(ratio)| -- geometric, symmetric (x10 and /10 equal)
        print("\n--- actual rollout LOG growth |ln(||z0_true_k+1||/||z0_true_k||)| "
              f"({len(r)} steps) ---")
        print(f"  ratio median {np.median(r):.4g};  |ln ratio|: median {np.median(lr):.3g}, "
              f"90th pct {np.percentile(lr, 90):.3g}, max {lr.max():.3g}")
        print(f"  L_z0_growth = |ln(||z0_hat_k+1||/||z0_hat_k||)|  (symmetric: collapse is as "
              f"bad as explosion; the real trajectory sits near 0).")
        print(f"  -> tolerance  eps ~ {np.percentile(lr, 90):.3g}  "
              f"(real |ln ratio| stays within this; penalize |ln ratio| beyond it)")
        print(f"  -> z0_growth_scale ~ {max(np.percentile(lr, 90), 1e-3):.3g}  "
              f"(the physical log-band; weight*raw/scale ~ O(1) once the term fires).")
        print("  NOTE minimized by f_theta->0 (predict no change): pair with L_rollout "
              "(unforced) or L_stats0_predict for a correct DIRECTION, or it goes inert.")


if __name__ == "__main__":
    main()
