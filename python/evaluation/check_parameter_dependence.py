"""
Scatter one-step LDS prediction error against dt, temperature, and
noise, across every window in the test set (not just a handful) -- to
check whether error systematically depends on any of these (as a few
examples in check_rollout.py suggested for dt), and specifically to
help decide WHERE in (temperature, noise) space more simulation data
would help most. Motivation: a rare, hard case like hourglass-shaped
grain snapping is exactly the kind of thing under-represented in some
region of parameter space, not something more training epochs would
fix -- if error is concentrated in an identifiable (temperature, noise)
region, that's a direct, actionable signal for where to run more
simulations, rather than a diffuse "the model needs to be better"
conclusion.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_parameter_dependence \
        --lds-checkpoint ../checkpoints/stage3/64x64.pt
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
from utils import load_datasets as load


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


def _print_binned_summary(name: str, values: np.ndarray, latent_losses: np.ndarray,
                           pixel_losses: np.ndarray, n_bins: int = 8):
    """Linear-space binned summary -- unlike dt (which spans orders of
    magnitude and gets log-decade bins below), temperature/noise are
    each a narrow, bounded range in a typical sweep, so linear bins are
    the more natural choice here."""
    edges = np.linspace(values.min(), values.max(), n_bins + 1)
    print(f"\n{name} bin              n       mean latent_loss   mean pixel_loss")
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (values >= lo) & (values <= hi if i == n_bins - 1 else values < hi)
        if mask.sum() == 0:
            continue
        print(f"{lo:8.4f} - {hi:8.4f}   {mask.sum():4d}   "
              f"{latent_losses[mask].mean():.6f}         {pixel_losses[mask].mean():.6f}")


def _boxplot_by_x(ax, x_values: np.ndarray, y_values: np.ndarray, log_x: bool = False):
    """
    Boxplot of y_values grouped by each unique value in x_values.
    Appropriate here specifically because dt/temperature/noise each
    take a small, discrete set of values within one sweep (unlike a
    genuinely continuous variable) -- at a given x, a scatter's
    overlapping points obscure the actual distribution (median, IQR,
    outliers), which a boxplot shows directly.

    log_x: widths are set PROPORTIONAL to position (not constant) so
    boxes look visually consistent across a log-scale axis -- a
    constant width would be imperceptibly thin at small x and absurdly
    wide at large x. Verified numerically: width=position*k gives an
    identical hi/lo ratio at every position, i.e. genuinely constant
    apparent width on a log axis, not just at a glance.
    """
    unique_x = np.unique(x_values)
    groups = [y_values[x_values == x] for x in unique_x]

    if log_x:
        widths = unique_x * 0.15
    else:
        min_gap = np.min(np.diff(unique_x)) if len(unique_x) > 1 else 1.0
        widths = np.full(len(unique_x), min_gap * 0.6)

    ax.boxplot(groups, positions=unique_x, widths=widths, showfliers=True,
               patch_artist=True, boxprops=dict(facecolor="tab:blue", alpha=0.4),
               medianprops=dict(color="black"),
               flierprops=dict(markersize=3, alpha=0.3, markeredgecolor="tab:blue"))
    if log_x:
        ax.set_xscale("log")
    else:
        # boxplot's default behavior is ALWAYS one labeled tick per
        # position, regardless of how many there are -- fine for a
        # handful of values, unreadable once a finer sweep grid (more
        # runs revealing more distinct values) produces dozens of them.
        # log-scale panels don't need this: set_xscale("log") replaces
        # boxplot's explicit tick locator with a log-appropriate one as
        # a side effect; nothing does that for the linear case, so it's
        # handled explicitly here instead.
        #
        # Ticks are spaced evenly across the VALUE range, not the index
        # range -- striding unique_x by index caps the tick COUNT but not
        # their spread: if unique_x is non-uniformly distributed (many
        # closely-spaced values in one region, few elsewhere), an
        # index-based stride picks nearly all ticks from the dense
        # region, leaving them visually clustered despite being capped
        # at max_labeled_ticks. Confirmed this reproduces the exact
        # "still compressed" symptom with a synthetic dense-then-sparse
        # value set before switching to this approach.
        max_labeled_ticks = 15
        if len(unique_x) > max_labeled_ticks:
            targets = np.linspace(unique_x.min(), unique_x.max(), max_labeled_ticks)
            indices = np.clip(np.searchsorted(unique_x, targets), 0, len(unique_x) - 1)
            for i, t in enumerate(targets):
                idx = indices[i]
                if idx > 0 and abs(unique_x[idx - 1] - t) < abs(unique_x[idx] - t):
                    indices[i] = idx - 1
            tick_positions = np.unique(unique_x[indices])
        else:
            tick_positions = unique_x
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{v:.3g}" for v in tick_positions], rotation=90, fontsize=8)


def check_parameter_dependence(
    lds_checkpoint_path: Path, min_step: int | None = None, min_stdev_phi: float | None = None,
    output_path: Path | None = None, device: str | None = None,
) -> Path:
    """Saves the dt/temperature/noise-vs-error figure and returns its path."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if output_path is None:
        output_path = Path(f"../../output/stage3/{lds_checkpoint_path.stem}-parameter_dependence.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lds_checkpoint = torch.load(lds_checkpoint_path, map_location=device, weights_only=True)
    lds_config = lds_checkpoint["config"]

    data_config = lds_checkpoint.get("data_config")
    if data_config is None:
        print("WARNING: checkpoint has no saved data_config -- falling back to "
              "min_step=0, min_stdev_phi=None, window_length=2 (may not match training).")
        data_config = {"min_step": 0, "min_stdev_phi": None, "window_length": 2}
    min_step = min_step if min_step is not None else data_config["min_step"]
    min_stdev_phi = min_stdev_phi if min_stdev_phi is not None else data_config["min_stdev_phi"]
    window_length = data_config["window_length"]

    test_dirs = lds_checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(f"{lds_checkpoint_path} has no saved test_dirs")
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

    one_step_loss = OneStepLoss(kind="l1")
    recon_loss = ReconLoss(kind="l1")

    # metadata.txt read once per run_dir, not once per window -- most
    # runs contribute several windows, and temperature/noise are
    # constant across all of them (unlike dt, which varies per window
    # even within the same run).
    metadata_cache: dict[Path, object] = {}

    dts, temperatures, noises, run_dirs = [], [], [], []
    latent_losses, pixel_losses = [], []

    with torch.no_grad():
        for idx in range(len(dataset)):
            z_window, dt_window, theta = dataset[idx]
            run_dir, _steps = dataset.window_info(idx)
            if run_dir not in metadata_cache:
                metadata_cache[run_dir] = load.read_metadata(run_dir / "metadata.txt")
            metadata = metadata_cache[run_dir]

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
            temperatures.append(metadata.temperature)
            noises.append(metadata.noise)
            run_dirs.append(run_dir)
            latent_losses.append(latent_loss)
            pixel_losses.append(pixel_loss)

    dts = np.array(dts)
    temperatures = np.array(temperatures)
    noises = np.array(noises)
    latent_losses = np.array(latent_losses)
    pixel_losses = np.array(pixel_losses)

    # ---- dt: unchanged from check_dt_dependence.py -- log-log, with
    # the power-law vs saturating-exponential model comparison. dt
    # genuinely spans orders of magnitude within one sweep; temperature
    # and noise (below) typically don't, so they get linear treatment
    # instead rather than forcing the same log-space analysis on them.
    log_dt = np.log10(dts)
    log_latent = np.log10(np.clip(latent_losses, 1e-12, None))
    log_pixel = np.log10(np.clip(pixel_losses, 1e-12, None))
    corr_dt_latent = np.corrcoef(log_dt, log_latent)[0, 1]
    corr_dt_pixel = np.corrcoef(log_dt, log_pixel)[0, 1]

    print(f"\ncorrelation(log10(dt), log10(latent_loss)) = {corr_dt_latent:.3f}")
    print(f"correlation(log10(dt), log10(pixel_loss))  = {corr_dt_pixel:.3f}")
    print("(near 0 = no dt dependence; positive = error grows with dt)")

    print("\nModel comparison (latent_loss vs dt): power law vs saturating exponential")
    a, b, r2_log, sse_power, pred_power = fit_power_law(dts, latent_losses)
    c, tau, r2_sat, sse_sat, pred_sat = fit_saturating_exponential(dts, latent_losses)
    print(f"  power law:     error ~ dt^{a:.3f}, SSE(real space)={sse_power:.6f}")
    print(f"  saturating exp: error -> {c:.4f} with timescale tau={tau:.1f}, "
          f"SSE(real space)={sse_sat:.6f}")
    better = "saturating exponential" if sse_sat < sse_power else "power law"
    print(f"  -> {better} fits better (lower SSE).")

    print("\ndt decade      n       mean latent_loss   mean pixel_loss")
    edges = np.floor(log_dt.min()), np.ceil(log_dt.max())
    for lo in np.arange(edges[0], edges[1] + 1):
        mask = (log_dt >= lo) & (log_dt < lo + 1)
        if mask.sum() == 0:
            continue
        print(f"1e{lo:.0f} - 1e{lo+1:.0f}   {mask.sum():4d}   "
              f"{latent_losses[mask].mean():.6f}         {pixel_losses[mask].mean():.6f}")

    # ---- temperature and noise: linear-space correlation + binned summary
    corr_temp_latent = np.corrcoef(temperatures, log_latent)[0, 1]
    corr_noise_latent = np.corrcoef(noises, log_latent)[0, 1]
    print(f"\ncorrelation(temperature, log10(latent_loss)) = {corr_temp_latent:.3f}")
    print(f"correlation(noise, log10(latent_loss))       = {corr_noise_latent:.3f}")
    _print_binned_summary("temperature", temperatures, latent_losses, pixel_losses)
    _print_binned_summary("noise", noises, latent_losses, pixel_losses)

    # ---- per-run aggregation for the 2D (temperature, noise) view --
    # the most directly actionable panel: raw per-WINDOW points at the
    # same (temperature, noise) would just overplot (many windows share
    # one run's fixed temperature/noise), so this averages per run first,
    # giving one point per run, colored by that run's mean one-step error.
    per_run: dict[Path, dict] = {}
    for run_dir, t, n, ll in zip(run_dirs, temperatures, noises, latent_losses):
        entry = per_run.setdefault(run_dir, {"temperature": t, "noise": n, "losses": []})
        entry["losses"].append(ll)
    run_temps = np.array([v["temperature"] for v in per_run.values()])
    run_noises = np.array([v["noise"] for v in per_run.values()])
    run_mean_loss = np.array([np.mean(v["losses"]) for v in per_run.values()])
    run_n_windows = np.array([len(v["losses"]) for v in per_run.values()])

    print(f"\n{len(per_run)} distinct runs contributing windows. "
          f"Runs with the highest mean one-step error:")
    order = np.argsort(run_mean_loss)[::-1]
    run_dir_list = list(per_run.keys())
    for i in order[:10]:
        print(f"  {run_dir_list[i].name}: T={run_temps[i]:.3f}  noise={run_noises[i]:.4f}  "
              f"mean_loss={run_mean_loss[i]:.6f}  ({run_n_windows[i]} windows)")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    for ax, losses, name, corr in [
        (axes[0, 0], latent_losses, "latent-space (L1)", corr_dt_latent),
        (axes[0, 1], pixel_losses, "pixel-space (L1, decoded)", corr_dt_pixel),
    ]:
        _boxplot_by_x(ax, dts, losses, log_x=True)
        ax.set_yscale("log")
        ax.set_xlabel("dt")
        ax.set_ylabel(f"{name} one-step error")
        ax.set_title(f"{name}\ncorr(log dt, log error) = {corr:.3f}")

    order_dt = np.argsort(dts)
    axes[0, 0].plot(dts[order_dt], pred_power[order_dt], "--", color="tab:red",
                     label=f"power law (SSE={sse_power:.4f})")
    axes[0, 0].plot(dts[order_dt], pred_sat[order_dt], "--", color="tab:green",
                     label=f"saturating exp, tau={tau:.0f} (SSE={sse_sat:.4f})")
    axes[0, 0].legend(fontsize=8)

    # The key actionable panel: which (temperature, noise) region has
    # the highest error, aggregated per run so points don't overplot.
    # Left as a scatter, not a boxplot -- this is genuinely 2D (two
    # discrete axes at once), not a single discrete-x-vs-y relationship.
    sc = axes[0, 2].scatter(run_temps, run_noises, c=run_mean_loss, s=30 + 10 * run_n_windows,
                             cmap="viridis", edgecolors="black", linewidths=0.3)
    axes[0, 2].set_xlabel("temperature")
    axes[0, 2].set_ylabel("noise")
    axes[0, 2].set_title("mean one-step latent error per run\n(point size ~ windows contributed)")
    fig.colorbar(sc, ax=axes[0, 2], label="mean latent_loss")

    for ax, values, name, corr in [
        (axes[1, 0], temperatures, "temperature", corr_temp_latent),
        (axes[1, 1], noises, "noise", corr_noise_latent),
    ]:
        _boxplot_by_x(ax, values, latent_losses, log_x=False)
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("latent-space one-step error")
        ax.set_title(f"{name}\ncorr({name}, log error) = {corr:.3f}")

    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"\nSaved figure to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lds-checkpoint", type=Path, required=True,
            help="no default -- multiple LDS variants can now coexist under "
                 "../checkpoints/stage3/")
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--min-stdev-phi", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None,
            help="default: ../../output/stage3/<lds checkpoint name>-parameter_dependence.png")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    check_parameter_dependence(
        lds_checkpoint_path=args.lds_checkpoint, min_step=args.min_step,
        min_stdev_phi=args.min_stdev_phi, output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()