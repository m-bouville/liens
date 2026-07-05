"""
Scatter one-step LDS prediction error against dt, across every window
in the test set (not just a handful) -- to check whether error
systematically worsens with dt (as a few examples in check_rollout.py
suggested) or whether that was a small-sample coincidence, and whether
the degradation is smooth or a sharp threshold.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_dt_dependence --lds-checkpoint ../output/lds_checkpoint.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.autoencoder import Autoencoder
from models.latent_dynamics import LatentDynamics
from training.datasets import MicrostructureEvolutionDataset
from training.losses import OneStepLoss, ReconLoss


def fit_power_law(dt: np.ndarray, error: np.ndarray):
    """
    log(error) = a*log(dt) + b via least squares. Returns (a, b, r2_log,
    sse_real, pred_real) -- sse_real is the fit's error IN REAL (non-log)
    space, so it can be compared directly against fit_saturating_exponential's
    sse, which is fit in real space to begin with. Comparing R^2 values
    computed in DIFFERENT spaces (log vs real) would not be a fair comparison.
    """
    log_dt = np.log(dt)
    log_err = np.log(np.clip(error, 1e-12, None))
    a, b = np.polyfit(log_dt, log_err, 1)
    pred_log = a * log_dt + b
    ss_res_log = np.sum((log_err - pred_log) ** 2)
    ss_tot_log = np.sum((log_err - log_err.mean()) ** 2)
    r2_log = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else float("nan")
    pred_real = np.exp(pred_log)
    sse_real = np.sum((error - pred_real) ** 2)
    return a, b, r2_log, sse_real, pred_real


def fit_saturating_exponential(dt: np.ndarray, error: np.ndarray, n_grid: int = 200):
    """
    error = c*(1 - exp(-dt/tau)) -- a smooth, fully DETERMINISTIC
    relaxation toward an asymptote c, with timescale tau. This is a
    genuinely different mechanism from "error grows without bound" or
    "irreducible unpredictability": it's ordinary exponential relaxation,
    which can look deceptively like a decelerating power law in a log-log
    plot over a limited dt range -- exactly why this needs an explicit
    fit-and-compare rather than eyeballing curvature in binned means.

    Fit via a tau grid search (log-spaced across the observed dt range)
    with closed-form c at each tau -- error is LINEAR in c for fixed tau,
    so c has a direct least-squares solution, avoiding a scipy dependency.
    """
    tau_grid = np.logspace(np.log10(dt.min() / 10), np.log10(dt.max() * 10), n_grid)
    best_sse, best_tau, best_c = np.inf, None, None
    for tau in tau_grid:
        basis = 1 - np.exp(-dt / tau)
        denom = np.sum(basis ** 2)
        if denom < 1e-12:
            continue
        c = np.sum(error * basis) / denom
        pred = c * basis
        sse = np.sum((error - pred) ** 2)
        if sse < best_sse:
            best_sse, best_tau, best_c = sse, tau, c
    pred_real = best_c * (1 - np.exp(-dt / best_tau))
    ss_tot = np.sum((error - error.mean()) ** 2)
    r2_real = 1 - best_sse / ss_tot if ss_tot > 0 else float("nan")
    return best_c, best_tau, r2_real, best_sse, pred_real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lds-checkpoint", type=Path, required=True,
            help="no default -- multiple LDS variants can now coexist under "
                 "../../output/lds_checkpoint_pt/")
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--min-stdev-phi", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None,
            help="default: ../../output/dt_dependence_png/<lds checkpoint name>.png")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"../../output/dt_dependence_png/{args.lds_checkpoint.stem}.png")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    lds_checkpoint = torch.load(args.lds_checkpoint, map_location=device, weights_only=True)
    lds_config = lds_checkpoint["config"]

    data_config = lds_checkpoint.get("data_config")
    if data_config is None:
        print("WARNING: checkpoint has no saved data_config -- falling back to "
              "min_step=0, min_stdev_phi=None, window_length=2 (may not match training).")
        data_config = {"min_step": 0, "min_stdev_phi": None, "window_length": 2}
    min_step = args.min_step if args.min_step is not None else data_config["min_step"]
    min_stdev_phi = args.min_stdev_phi if args.min_stdev_phi is not None else data_config["min_stdev_phi"]
    window_length = data_config["window_length"]

    test_dirs = lds_checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(f"{args.lds_checkpoint} has no saved test_dirs")
    test_dirs = [Path(d) for d in test_dirs]

    ae_checkpoint_path = Path(lds_checkpoint["ae_checkpoint"])
    ae_checkpoint = torch.load(ae_checkpoint_path, map_location=device, weights_only=True)
    ae_config = ae_checkpoint["config"]
    ae = Autoencoder(
        size=ae_config["size"], channels=1,
        base_channels=ae_config["base_channels"], latent_channels=ae_config["latent_channels"],
    ).to(device)
    ae.load_state_dict(ae_checkpoint["model_state"])
    ae.eval()

    f_theta = LatentDynamics(
        latent_channels=lds_config["latent_channels"], n_theta=lds_config["n_theta"],
        hidden_dim=lds_config["hidden_dim"], n_hidden_layers=lds_config["n_hidden_layers"],
    ).to(device)
    f_theta.load_state_dict(lds_checkpoint["model_state"])
    f_theta.eval()

    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=ae.encoder, device=device, window_length=window_length,
        min_step=min_step, min_stdev_phi=min_stdev_phi,
    )
    print(f"Evaluating {len(dataset)} test windows...")

    one_step_loss = OneStepLoss(kind="l1")  # per-sample, unreduced-ish (mean over one sample)
    recon_loss = ReconLoss(kind="l1")

    dts, latent_losses, pixel_losses = [], [], []

    with torch.no_grad():
        for idx in range(len(dataset)):
            z_window, dt_window, theta = dataset[idx]
            z_t = z_window[0:1].to(device)
            z_next_true = z_window[1:2].to(device)
            dt = dt_window[0:1].to(device)
            theta_b = theta.unsqueeze(0).to(device)

            dz = f_theta(z_t, dt, theta_b)
            z_next_pred = z_t + dz

            latent_loss = one_step_loss(z_next_pred, z_next_true).item()

            x_next_pred = ae.decoder(z_next_pred)
            x_next_true = ae.decoder(z_next_true)
            pixel_loss = recon_loss(x_next_pred, x_next_true).item()

            dts.append(dt.item())
            latent_losses.append(latent_loss)
            pixel_losses.append(pixel_loss)

    dts = np.array(dts)
    latent_losses = np.array(latent_losses)
    pixel_losses = np.array(pixel_losses)

    # Correlation in log-space (both span orders of magnitude) -- a
    # positive correlation here is the quantitative version of "error
    # gets worse with dt" seen visually in check_rollout.py.
    log_dt = np.log10(dts)
    log_latent = np.log10(np.clip(latent_losses, 1e-12, None))
    log_pixel = np.log10(np.clip(pixel_losses, 1e-12, None))
    corr_latent = np.corrcoef(log_dt, log_latent)[0, 1]
    corr_pixel = np.corrcoef(log_dt, log_pixel)[0, 1]

    print(f"\ncorrelation(log10(dt), log10(latent_loss)) = {corr_latent:.3f}")
    print(f"correlation(log10(dt), log10(pixel_loss))  = {corr_pixel:.3f}")
    print("(near 0 = no dt dependence; positive = error grows with dt)")

    # Head-to-head model comparison: does a smooth, fully deterministic
    # saturating exponential (ordinary relaxation, timescale tau) describe
    # the data better than a power law -- rather than assuming either.
    # Both SSEs computed in REAL (non-log) space for a fair comparison.
    print("\nModel comparison (latent_loss vs dt): power law vs saturating exponential")
    a, b, r2_log, sse_power, pred_power = fit_power_law(dts, latent_losses)
    c, tau, r2_sat, sse_sat, pred_sat = fit_saturating_exponential(dts, latent_losses)
    print(f"  power law:     error ~ dt^{a:.3f}, SSE(real space)={sse_power:.6f}")
    print(f"  saturating exp: error -> {c:.4f} with timescale tau={tau:.1f}, "
          f"SSE(real space)={sse_sat:.6f}")
    better = "saturating exponential" if sse_sat < sse_power else "power law"
    print(f"  -> {better} fits better (lower SSE). "
          f"{'This means the apparent deceleration is consistent with ordinary, ' + chr(10) + '     deterministic relaxation -- not necessarily evidence of irreducible unpredictability.' if better == 'saturating exponential' else ''}")

    # Binned summary: average loss per decade of dt, to see whether the
    # trend is smooth or has a sharp threshold.
    print("\ndt decade      n       mean latent_loss   mean pixel_loss")
    edges = np.floor(log_dt.min()), np.ceil(log_dt.max())
    for lo in np.arange(edges[0], edges[1] + 1):
        mask = (log_dt >= lo) & (log_dt < lo + 1)
        if mask.sum() == 0:
            continue
        print(f"1e{lo:.0f} - 1e{lo+1:.0f}   {mask.sum():4d}   "
              f"{latent_losses[mask].mean():.6f}         {pixel_losses[mask].mean():.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, losses, name, corr in [
        (axes[0], latent_losses, "latent-space (L1)", corr_latent),
        (axes[1], pixel_losses, "pixel-space (L1, decoded)", corr_pixel),
    ]:
        ax.scatter(dts, losses, alpha=0.3, s=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("dt")
        ax.set_ylabel(f"{name} one-step error")
        ax.set_title(f"{name}\ncorr(log dt, log error) = {corr:.3f}")

    # Overlay both candidate fits on the latent-space panel specifically,
    # since that's what the model comparison above was computed on.
    order = np.argsort(dts)
    axes[0].plot(dts[order], pred_power[order], "--", color="tab:red",
                 label=f"power law (SSE={sse_power:.4f})")
    axes[0].plot(dts[order], pred_sat[order], "--", color="tab:green",
                 label=f"saturating exp, tau={tau:.0f} (SSE={sse_sat:.4f})")
    axes[0].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"\nSaved scatter plot to {args.output}")


if __name__ == "__main__":
    main()
