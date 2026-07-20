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
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder, EncoderDecoderPair, MultiStreamAutoencoder
from models.constants import LATENT_SPATIAL_SIZE
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_dynamics import LatentDynamics
from models.latent_streams import (
    cross_check_stream_configs_against_state_dict, resolve_stream_configs_from_checkpoint_config,
)
from training.datasets import MicrostructureEvolutionDataset
from utils import load_datasets as load

# GENERAL POLICY (matches training/train_refinement.py's own
# _PYTHON_ROOT): every default checkpoint/output path is built from
# THIS anchor, never from a bare relative string like "../../output/...".
# Relative strings resolve against the process's CWD at invocation
# time, which silently differs across bare CLI, `python -m`, and being
# imported and called from another module (e.g. main.py calling this
# function) -- exactly the recurring "output ended up in the wrong
# place" bug hit repeatedly on this project. Path(__file__) is anchored
# to THIS FILE's own on-disk location instead, which is invariant
# regardless of how/from-where the process was launched.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/


def max_autocorr_dist(nx: int, ny: int) -> int:
    """
    The C++ simulation caps autocorr_length's search at
    min(Nx*2/3, Ny*2/3) (integer division) -- distances beyond that are
    deemed artifacts of the periodic-boundary autocorrelation wrapping
    around on itself, not a genuine length scale. Any window whose
    autocorrelation never decays within that search range gets this
    exact value back as a SENTINEL, not a real measurement -- and it's
    common enough (near-critical/smooth microstructures in particular)
    to distort both the plot and any regression fit through it if left
    in as if it were real data. Mirrors the C++ integer-division
    formula exactly (Python's // matches C++'s truncating int division
    for non-negative operands), so this returns the same sentinel value
    the simulation actually produced, not an approximation of it.
    """
    return min(nx * 2 // 3, ny * 2 // 3)


def robust_linear_fit(x: np.ndarray, y: np.ndarray, n_iter: int = 10, huber_delta_scale: float = 1.345):
    """
    Robust linear fit (y = slope*x + intercept) via IRLS with Huber
    weights -- self-contained (no scipy/sklearn dependency), since
    np.polyfit already supports weighted least-squares directly.

    Ordinary least squares (a single np.polyfit call) is heavily
    influenced by a small number of extreme-magnitude points -- exactly
    the failure mode motivating this function: a handful of small-dt
    windows with unusually large |residual| can dominate an OLS fit
    entirely, even though most of the data disagrees with the line
    those few points imply. IRLS refits repeatedly, each time
    DOWN-WEIGHTING points whose current residual is large (a Huber-type
    weight: full weight within `huber_delta_scale` MAD-scaled residuals
    of zero, decreasing weight beyond that) -- points that are
    genuinely outliers relative to the bulk of the data end up
    contributing little to the final fit, rather than being fit exactly
    at the expense of everything else.

    huber_delta_scale=1.345 is the standard choice for Huber's own
    estimator (95% efficiency under a purely Gaussian residual
    distribution) -- not tuned specifically for this application, just
    the well-established default.
    """
    slope, intercept = np.polyfit(x, y, deg=1)
    for _ in range(n_iter):
        residuals = y - (slope * x + intercept)
        # MAD (median absolute deviation), scaled to be a consistent
        # estimator of the standard deviation under a Gaussian residual
        # distribution (the standard 1.4826 factor) -- robust to the
        # same outliers the weights themselves are meant to down-weight,
        # unlike using the residuals' own (outlier-sensitive) std.
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = 1.4826 * mad if mad > 0 else np.std(residuals) + 1e-12
        huber_delta = huber_delta_scale * scale
        abs_resid = np.abs(residuals)
        weights = np.where(abs_resid <= huber_delta, 1.0, huber_delta / np.maximum(abs_resid, 1e-12))
        slope, intercept = np.polyfit(x, y, deg=1, w=weights)
    return slope, intercept


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


def fit_exponential(x: np.ndarray, error: np.ndarray):
    """
    log(error) = a*x + b via least squares (x itself, NOT log(x)) --
    i.e. error = exp(b) * exp(a*x). The semi-log analogue of
    fit_power_law: appropriate for a panel with a LINEAR x-axis and
    log-scaled error axis (like length_scale's), where fit_power_law's
    form would plot as a curve rather than a straight line and so
    wouldn't give the same at-a-glance visual fit-quality check that it
    does on a genuinely log-log panel (like dt's). Same
    sse_real/pred_real convention as fit_power_law, for direct SSE
    comparison against fit_saturating_exponential.
    """
    log_err = np.log(np.clip(error, 1e-12, None))
    a, b = np.polyfit(x, log_err, 1)
    pred_log = a * x + b
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


def _boxplot_by_x(ax, x_values: np.ndarray, y_values: np.ndarray, log_x: bool = False,
                   round_decimals: int = 6):
    """
    Boxplot of y_values grouped by each unique value in x_values, with
    boxes positioned at their LITERAL value -- honest, in both the
    log_x=True (dt) and log_x=False (temperature/noise) cases. An
    earlier version of this positioned temperature/noise boxes at
    evenly-spaced CATEGORICAL indices instead, discarding their real
    value spacing entirely to avoid crowding -- that's misleading, not
    a readability trade-off: it makes two boxes close in value and two
    boxes far apart in value look identically spaced, which is a false
    impression of the underlying sweep, not just a cosmetic
    simplification. Reverted.

    dt (log_x=True) was never the problem: its sweep is naturally close
    to geometric (roughly x2 between consecutive values), so an honest
    log axis at literal values already looks close to evenly spaced --
    nothing special is done for it, it just doesn't need anything
    special. temperature/noise are swept roughly linearly, so a linear
    axis at literal values will only look evenly spread if the sweep
    itself actually is -- if it isn't (or if the ACTUAL unique-value
    count is much larger than the intended sweep grid; see
    round_decimals below), no amount of axis trickery makes that
    honest AND uniformly readable at the same time. The two things
    below address the two real, non-misleading ways to reduce crowding:

    1. round_decimals: values are rounded to this many decimals before
       computing uniqueness. Temperature/noise round-trip through a
       text metadata file -- float parsing can turn what's meant to be
       the same nominal sweep value (e.g. 0.55) into many
       bit-distinct floats (0.5500000001, 0.5499999998, ...) across
       different runs, which np.unique would count as genuinely
       different x positions, silently exploding the apparent sweep
       size far beyond the real number of distinct settings. Rounding
       merges those back into one honest value rather than plotting
       dozens of near-duplicate boxes that were never meant to be
       distinguishable. This does NOT round away real distinctions --
       a human-designed sweep grid isn't going to have two genuinely
       different settings within 1e-6 of each other.
    2. Per-box width from that box's own LOCAL neighbor gaps, not a
       single global minimum gap. The old width = min(all gaps) * 0.6
       meant one closely-spaced pair anywhere in the sweep forced every
       box everywhere to be that thin, including ones with plenty of
       room -- using each box's own local spacing gives genuinely
       sparse boxes their actual available width instead.

    Tick labels are still thinned (not every value gets text) once
    there are more than max_labeled_ticks distinct values, since even
    honestly-positioned boxes can have more values than can be legibly
    labeled -- but the BOXES themselves are never merged or
    repositioned, only which ones get a text label underneath.
    """
    x_values = np.round(x_values, round_decimals)
    unique_x = np.unique(x_values)
    groups = [y_values[x_values == x] for x in unique_x]
    print(f"  ({len(unique_x)} distinct x values after rounding to {round_decimals} "
          f"decimals -- if this looks far larger than the intended sweep grid, "
          f"floating-point round-trip noise is the likely cause)")

    if log_x:
        widths = unique_x * 0.15
    else:
        widths = _local_widths(unique_x)

    ax.boxplot(groups, positions=unique_x, widths=widths, showfliers=True,
               patch_artist=True, boxprops=dict(facecolor="tab:blue", alpha=0.4),
               medianprops=dict(color="black"),
               flierprops=dict(markersize=3, alpha=0.3, markeredgecolor="tab:blue"))

    # matplotlib's boxplot() applies its OWN default x margin -- +-0.5
    # around the position range -- regardless of what scale the
    # positions are actually on. That's a reasonable margin for
    # categorical integer positions (0, 1, 2, ...), and a wildly
    # disproportionate one for small real-valued positions: verified
    # numerically for a noise-like sweep spanning [0.005, 0.05] (range
    # 0.045), the default autoscale gives xlim=(-0.495, 0.55) -- a
    # range over 23x wider than the actual data, squeezing every box
    # into a sliver in the middle of mostly blank axis. Overriding xlim
    # explicitly, proportional to the REAL data range, fixes this for
    # any parameter's scale rather than relying on matplotlib's
    # categorical-position assumption.
    if log_x:
        ax.set_xlim(unique_x.min() / 1.5, unique_x.max() * 1.5)
    else:
        data_range = unique_x.max() - unique_x.min() if len(unique_x) > 1 else widths[0]
        pad = max(data_range * 0.1, widths.max() * 0.75)
        ax.set_xlim(unique_x.min() - pad, unique_x.max() + pad)

    if log_x:
        ax.set_xscale("log")
        return

    # Value-evenly-spaced tick TARGETS, nearest-snapped to real data --
    # only decides which of the (honestly, literally positioned) boxes
    # get a text label, never moves or merges a box itself.
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


def _local_widths(unique_x: np.ndarray, factor: float = 0.6) -> np.ndarray:
    """
    Per-position box width from that position's OWN nearest-neighbor
    gap (left or right, whichever is smaller), not a single global
    min-gap applied everywhere. A box in a sparse region gets a width
    that reflects the room it actually has, instead of being forced
    down to match the tightest pair anywhere else in the sweep.
    """
    if len(unique_x) == 1:
        return np.array([max(abs(unique_x[0]) * 0.1, 1e-3)])
    gaps = np.diff(unique_x)
    left_gap = np.concatenate([[gaps[0]], gaps])
    right_gap = np.concatenate([gaps, [gaps[-1]]])
    return np.minimum(left_gap, right_gap) * factor


def check_parameter_dependence(
    lds_checkpoint_path: Path, min_step: int | None = None, min_stdev_phi: float | None = None,
    output_path: Path | None = None, device: str | None = None,
) -> Path:
    """Saves the dt/temperature/noise-vs-error figure and returns its path."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        # See this function's own docstring on why this re-check exists,
        # rather than trusting device unconditionally.
        print("WARNING: device='cuda' was requested (or defaulted to, from an "
              "argparse default computed at a DIFFERENT time than this actual "
              "run), but torch.cuda.is_available() is False right now -- falling "
              "back to CPU instead of letting torch.load() fail with a confusing "
              "deserialization error. If this is unexpected, check that CUDA is "
              "actually usable from THIS environment specifically (e.g. running "
              "from the command line vs. an IDE's own kernel can pick up a "
              "different Python/CUDA environment).")
        device = torch.device("cpu")

    if output_path is None:
        output_path = (_PYTHON_ROOT.parent / "output" / "stage3"
                       / f"{lds_checkpoint_path.stem}-parameter_dependence.png")
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
    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(ae_config)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, ae_checkpoint["model_state"],
    )
    recon_stream = stream_configs[recon_stream_name]
    decoder_for_stream = ae_config.get("decoder_for_stream")
    is_flat_checkpoint = any(k.startswith("encoder.") for k in ae_checkpoint["model_state"])
    if is_flat_checkpoint:
        encoder = Encoder(input_size=ae_config["size"], in_channels=1,
                           base_channels=ae_config["base_channels"], stream_configs=stream_configs)
        decoder = Decoder(output_size=ae_config["size"], out_channels=1,
                           base_channels=ae_config["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size)
        ae = EncoderDecoderPair(encoder, decoder, stream_name=recon_stream_name,
                                 mode=recon_stream.mode).to(device)
    elif len(stream_configs) == 1:
        ae = Autoencoder(
            size=ae_config["size"], channels=1,
            base_channels=ae_config["base_channels"], latent_channels=recon_stream.channels,
            latent_spatial_size=recon_stream.spatial_size,
        ).to(device)
    elif decoder_for_stream is None:
        encoder = Encoder(input_size=ae_config["size"], in_channels=1,
                           base_channels=ae_config["base_channels"], stream_configs=stream_configs)
        decoder = Decoder(output_size=ae_config["size"], out_channels=1,
                           base_channels=ae_config["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size)
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                     stream_configs=stream_configs).to(device)
    else:
        encoder = Encoder(input_size=ae_config["size"], in_channels=1,
                           base_channels=ae_config["base_channels"], stream_configs=stream_configs)
        decoders = {}
        for stream_name, decoder_key in decoder_for_stream.items():
            stream_cfg = stream_configs[stream_name]
            decoders[decoder_key] = Decoder(
                output_size=ae_config["size"], out_channels=1,
                base_channels=ae_config["base_channels"], latent_channels=stream_cfg.channels,
                latent_spatial_size=stream_cfg.spatial_size,
            )
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders=decoders,
                                     stream_configs=stream_configs,
                                     decoder_for_stream=decoder_for_stream).to(device)
    ae.load_state_dict(ae_checkpoint["model_state"])
    ae.eval()
    ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
    ae_decoder = (ae.pathways[recon_stream_name].decoder if hasattr(ae, "pathways")
                  else ae.decoder)

    f_theta = LatentDynamics(
        latent_channels=lds_config["latent_channels"], n_theta=lds_config["n_theta"],
        latent_spatial=lds_config.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
        hidden_dim=lds_config["hidden_dim"], n_hidden_layers=lds_config["n_hidden_layers"],
    ).to(device)
    f_theta.load_state_dict(lds_checkpoint["model_state"])
    f_theta.eval()

    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=ae_encoder, device=device, window_length=window_length,
        min_step=min_step, min_stdev_phi=min_stdev_phi, encode_both_streams=True,
    )
    print(f"Evaluating {len(dataset)} test windows...")

    # metadata.txt read once per run_dir, not once per window -- most
    # runs contribute several windows, and temperature/noise are
    # constant across all of them (unlike dt, which varies per window
    # even within the same run). statistics.csv similarly cached per
    # run_dir, for the autocorr_length lookup below -- unlike
    # temperature/noise, autocorr_length is NOT constant across a run's
    # windows (a run's own dominant length scale coarsens over time as
    # the microstructure evolves), so it's looked up per-window at that
    # window's own starting step, not cached at the run level itself.
    metadata_cache: dict[Path, object] = {}
    stats_cache: dict[Path, object] = {}

    dts, temperatures, noises, run_dirs, length_scales = [], [], [], [], []
    latent_losses, pixel_losses, euler_losses = [], [], []
    latent_losses_signed, euler_losses_signed = [], []

    def _per_sample_l1(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """(B, ...) -> (B,) mean absolute error per sample -- matches
        OneStepLoss/ReconLoss's own L1 definition (mean over all
        non-batch dims), but WITHOUT their forward()'s own further
        reduction over the batch dim too, which would collapse an
        entire batch to one scalar -- exactly what per-window
        correlation against dt/temperature/etc needs to NOT happen."""
        return (pred - true).abs().flatten(start_dim=1).mean(dim=1)

    def _per_sample_signed_mean(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """(B, ...) -> (B,) mean SIGNED residual per sample -- same
        reduction as _per_sample_l1 but WITHOUT the .abs(), so positive
        and negative components can cancel within a window. Used
        specifically for the linear panel: the formula it checks is
        written in terms of the signed residual, not |residual|."""
        return (pred - true).flatten(start_dim=1).mean(dim=1)

    signed_residual_sum = None  # accumulated (C, H, W) sum of z0_euler_pred - z0_next_true,
                                 # summed (not yet averaged) across ALL windows -- see the
                                 # bias-vs-variance analysis after the loop for why this needs
                                 # to stay SIGNED, unlike everything else computed here.
    n_total = 0

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    idx = 0
    with torch.no_grad():
        for window0, window1, dt_window, theta in loader:
            batch_size = window0.shape[0]
            window0 = window0.to(device)
            window1 = window1.to(device)
            dt_window = dt_window.to(device)
            theta = theta.to(device)

            z0_t = window0[:, 0]
            z1_t = window1[:, 0]
            z0_next_true = window0[:, 1]
            dt = dt_window[:, 0]
            theta_b = theta

            z0_next_pred = f_theta(z0_t, z1_t, dt, theta_b)
            # The pure, hard-coded Euler term ALONE -- z0(t) + z1(t)*dt,
            # no f_theta correction at all. Comparing this against
            # z0_next_pred's own error (both against the SAME
            # z0_next_true, same dt, same window) is what actually
            # disentangles the two Taylor orders: the FIRST-order term
            # (z1*dt) is hard-coded, never learned, so its own error is
            # entirely a property of z1's own quality and the physics'
            # own curvature -- f_theta (the SECOND-order, TRAINED
            # correction) can only ever act on top of it. If f_theta is
            # adding real value, its own (full) error should fall off
            # FASTER with dt (a higher power-law exponent) than this
            # Euler-only baseline does -- see the dedicated panel below
            # for the direct visual/numeric comparison.
            dt_r = dt.view(-1, 1, 1, 1)
            z0_euler_pred = z0_t + z1_t * dt_r

            latent_loss_batch = _per_sample_l1(z0_next_pred, z0_next_true)
            euler_loss_batch = _per_sample_l1(z0_euler_pred, z0_next_true)
            latent_loss_signed_batch = _per_sample_signed_mean(z0_next_pred, z0_next_true)
            euler_loss_signed_batch = _per_sample_signed_mean(z0_euler_pred, z0_next_true)

            x_next_pred = ae_decoder(z0_next_pred)
            x_next_true = ae_decoder(z0_next_true)
            pixel_loss_batch = _per_sample_l1(x_next_pred, x_next_true)

            latent_losses.extend(latent_loss_batch.cpu().tolist())
            euler_losses.extend(euler_loss_batch.cpu().tolist())
            latent_losses_signed.extend(latent_loss_signed_batch.cpu().tolist())
            euler_losses_signed.extend(euler_loss_signed_batch.cpu().tolist())
            pixel_losses.extend(pixel_loss_batch.cpu().tolist())
            dts.extend(dt.cpu().tolist())

            # SIGNED (not .abs()'d) euler-only residual, summed over the
            # batch dim only -- keeps the full (C, H, W) shape, so
            # element-wise cancellation across DIFFERENT windows is what
            # actually happens here (a random +/- residual at any given
            # element genuinely cancels when summed across many
            # windows; .abs() before summing would never let it).
            batch_signed_residual = (z0_euler_pred - z0_next_true).sum(dim=0)
            signed_residual_sum = (batch_signed_residual if signed_residual_sum is None
                                    else signed_residual_sum + batch_signed_residual)
            n_total += batch_size

            # Per-window metadata: cheap, CPU-bound, inherently
            # per-index -- not a tensor op, so batching it wouldn't
            # help; stays in its own loop, synchronized to the same
            # dataset ordering the DataLoader above preserves
            # (shuffle=False).
            for i in range(batch_size):
                run_dir, steps = dataset.window_info(idx)
                if run_dir not in metadata_cache:
                    metadata_cache[run_dir] = load.read_metadata(run_dir / "metadata.txt")
                metadata = metadata_cache[run_dir]
                if run_dir not in stats_cache:
                    stats_cache[run_dir] = load.read_statistics_csv(run_dir / "statistics.csv")
                stats_df = stats_cache[run_dir]

                temperatures.append(metadata.temperature)
                noises.append(metadata.noise)
                run_dirs.append(run_dir)
                # Ground-truth length scale (first peak in the autocorrelation
                # function), read from the SIMULATION's own precomputed
                # statistics.csv -- not re-derived from the (possibly
                # decoder-distorted) reconstructed frame -- at the window's
                # starting step, i.e. the length scale of the microstructure
                # this rollout step is actually predicting FROM.
                length_scales.append(stats_df.loc[steps[0], "autocorr_length"])
                idx += 1

    dts = np.array(dts)
    temperatures = np.array(temperatures)
    noises = np.array(noises)
    length_scales = np.array(length_scales, dtype=float)
    latent_losses = np.array(latent_losses)
    pixel_losses = np.array(pixel_losses)
    euler_losses = np.array(euler_losses)
    latent_losses_signed = np.array(latent_losses_signed)
    euler_losses_signed = np.array(euler_losses_signed)

    # Bias vs variance in z1's own error: euler_losses (E[|residual|],
    # mean ABSOLUTE error) can never distinguish "z1 is wrong by the
    # same amount, in the same direction, every time" (a bias -- in
    # principle correctable, e.g. by retraining z1 differently) from
    # "z1 is wrong by that much, but in a random direction each time"
    # (variance -- an irreducible floor no retraining on the SAME kind
    # of data could remove). |E[residual]| (mean of the SIGNED
    # residual's own norm, NOT mean of the norm) answers this directly:
    # random, cancelling errors average toward zero across many
    # windows; a genuine, consistent bias does not.
    mean_signed_residual = signed_residual_sum / n_total  # (C, H, W), SIGN preserved
    bias_magnitude = mean_signed_residual.abs().mean().item()
    total_magnitude = float(euler_losses.mean())
    bias_fraction = bias_magnitude / total_magnitude if total_magnitude > 0 else float("nan")
    print(f"\nz1's own euler-only error: bias vs variance (n={n_total} windows):")
    print(f"  E[|residual|]  (total error magnitude, already reported above as euler-only): "
          f"{total_magnitude:.6e}")
    print(f"  |E[residual]|  (the part that does NOT cancel across windows -- the bias): "
          f"{bias_magnitude:.6e}")
    print(f"  bias fraction = |E[residual]| / E[|residual|] = {bias_fraction:.3f}")
    print(f"  (near 1.0 -> error is mostly a consistent, SYSTEMATIC bias in the same "
          f"direction every time -- in principle correctable by retraining z1 differently. "
          f"near 0.0 -> error is mostly VARIANCE/NOISE, cancelling across windows -- an "
          f"irreducible floor that retraining on the same KIND of data is unlikely to fix.)")

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

    # Euler-only (hard-coded first-order term alone, no f_theta) vs the
    # full, f_theta-corrected prediction -- SAME z0_next_true, SAME dt,
    # SAME window for both, so this is a genuine, point-by-point
    # decomposition, not just two separately-fit trends.
    #
    # THEORY, if z1 were an EXACT derivative and f_theta an EXACT
    # curvature: raw euler_error ~ dt^2 (euler_error/dt ~ dt^1), raw
    # full_error ~ dt^3 (full_error/dt ~ dt^2) -- z1 being hard-coded
    # first-order means its error is entirely the curvature term
    # f_theta's own trained correction should be cancelling.
    #
    # WHAT WE ACTUALLY SEE (confirmed on real 64x64 data): euler-only
    # exponent ~1.0, not ~2.0. euler_error/dt is then ~dt^0, i.e. a
    # NON-VANISHING CONSTANT as dt->0 -- exactly z1(t) - z0_dot(t), z1's
    # own systematic bias against the true derivative. euler_error/dt
    # is measuring z1's own error directly, not curvature at all.
    #
    # A higher exponent alone does NOT mean full is actually better --
    # it only describes the asymptotic trend toward dt=0, which may lie
    # well below every dt actually observed. The direct magnitude
    # comparison below is what actually answers "which one is smaller,
    # in practice, over the range that matters."
    a_euler, b_euler, r2_log_euler, sse_power_euler, pred_power_euler = fit_power_law(dts, euler_losses)
    print(f"\nEuler-only (z0+z1*dt, no f_theta) vs full (f_theta-corrected) prediction, "
          f"same windows/dt:")
    print(f"  euler-only:  error ~ dt^{a_euler:.3f}")
    print(f"  full (f_theta): error ~ dt^{a:.3f}")
    print(f"  (a higher exponent only describes the asymptotic trend toward dt->0 -- "
          f"see the direct magnitude comparison below for which is ACTUALLY smaller "
          f"over the observed dt range)")

    ratio = np.array(latent_losses) / np.maximum(np.array(euler_losses), 1e-12)
    frac_full_worse = float((ratio > 1).mean())
    print(f"\nDirect magnitude comparison (full / euler-only), per window:")
    print(f"  mean(full)={np.mean(latent_losses):.6f}  mean(euler-only)={np.mean(euler_losses):.6f}  "
          f"ratio of means={np.mean(latent_losses) / np.mean(euler_losses):.4f}")
    print(f"  mean(full/euler-only ratio)={ratio.mean():.4f}  median={np.median(ratio):.4f}")
    print(f"  full prediction is WORSE than euler-only on {frac_full_worse:.1%} of windows")
    if frac_full_worse > 0.5:
        print(f"  -> f_theta's own trained correction is making the prediction WORSE on "
              f"most windows, despite a higher fit exponent -- it is NOT currently adding "
              f"value in practice, whatever its asymptotic behavior would eventually be.")

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

    # ---- length scale (first peak in autocorrelation): DIAGNOSTIC for
    # whether error tracks the microstructure's own dominant length
    # scale rather than (or in addition to) dt/temperature/noise --
    # motivated by rollout figures showing large-scale, visually "easy"
    # microstructures failing outright while a much finer-grained
    # texture reconstructs cleanly, raising the question of whether the
    # 8x8 latent bottleneck is under-resolving a specific length-scale
    # regime rather than "coarse features = easy".
    #
    # SATURATED VALUES EXCLUDED: the C++ simulation caps autocorr_length
    # at max_autocorr_dist(Nx, Ny) -- distances beyond that are an
    # artifact of periodic-boundary wraparound, not a real length scale.
    # This sentinel value is common (near-critical/smooth microstructures
    # in particular never show a decaying autocorrelation within range),
    # and left in as if real it distorts both the correlation and the
    # fit below -- a cluster of fake "very large length scale" points
    # that are actually "the true length scale is unknown/unbounded".
    # Treated as N/A: excluded from the correlation, the binned summary,
    # BOTH fits, and the scatter -- not just visually hidden.
    max_dist = max_autocorr_dist(ae_config["size"], ae_config["size"])
    saturated = length_scales >= max_dist
    n_saturated = int(saturated.sum())
    print(f"\n{n_saturated}/{len(length_scales)} windows have autocorr_length >= "
          f"{max_dist} (the C++ search cap) -- treated as N/A (not a real length "
          f"scale, just 'never decayed within range') and excluded below.")
    length_scales_valid = length_scales[~saturated]
    latent_losses_for_length = latent_losses[~saturated]
    log_latent_for_length = log_latent[~saturated]

    corr_length_latent = np.corrcoef(length_scales_valid, log_latent_for_length)[0, 1]
    print(f"correlation(length_scale, log10(latent_loss)) = {corr_length_latent:.3f} "
          f"(n={len(length_scales_valid)}, excluding saturated)")
    print("(if this is the dominant driver, error should track length_scale more "
          "cleanly than it tracks dt/temperature/noise individually above)")
    _print_binned_summary("length_scale", length_scales_valid, latent_losses_for_length,
                           pixel_losses[~saturated])

    # UNLIKE dt (log-log panel, where fit_power_law's straight-line form
    # is the natural visual fit check), the length_scale panel is
    # semi-log: length_scale itself is plotted on a LINEAR axis (it's a
    # continuous, per-window-computed quantity -- not a small discrete
    # sweep grid like temperature/noise -- so it's shown as a raw
    # scatter, not a boxplot), against a log-scaled error axis. A power
    # law would plot as a curve there, not a line, and wouldn't give the
    # same at-a-glance fit-quality read. fit_exponential (log(error)
    # linear in length_scale itself, not log(length_scale)) is the
    # correct semi-log analogue -- it's what actually draws straight on
    # this panel. Compared against the same saturating-exponential
    # candidate as dt (that model isn't tied to either axis convention).
    print("\nModel comparison (latent_loss vs length_scale): exponential vs saturating exponential")
    a_len, b_len, r2_len, sse_exp_len, pred_exp_len = fit_exponential(
        length_scales_valid, latent_losses_for_length
    )
    c_len, tau_len, r2_sat_len, sse_sat_len, pred_sat_len = fit_saturating_exponential(
        length_scales_valid, latent_losses_for_length
    )
    print(f"  exponential:    error ~ exp({a_len:.4f} * length_scale), SSE(real space)={sse_exp_len:.6f}")
    print(f"  saturating exp: error -> {c_len:.4f} with timescale tau={tau_len:.1f}, "
          f"SSE(real space)={sse_sat_len:.6f}")
    better_len = "saturating exponential" if sse_sat_len < sse_exp_len else "exponential"
    print(f"  -> {better_len} fits better (lower SSE).")

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

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for ax, losses, name, corr in [
        (axes[0, 0], latent_losses, "latent-space (L1)", corr_dt_latent),
        (axes[0, 1], pixel_losses, "pixel-space (L1, decoded)", corr_dt_pixel),
    ]:
        _boxplot_by_x(ax, dts, losses, log_x=True)
        ax.set_yscale("log")
        ax.set_xlabel("dt")
        ax.set_ylabel(f"{name} one-step error")
        if ax is axes[0, 0]:
            # Exponent shown here (not pixel-space) since this is the
            # ONE panel with the power-law fit actually drawn below --
            # directly answers "is this first-order (~dt^1) or
            # second-order (~dt^2) Taylor remainder behavior" without
            # cross-referencing the printed console output.
            ax.set_title(f"{name}\ncorr(log dt, log error) = {corr:.3f}, "
                         f"power-law exponent = {a:.3f}")
        else:
            ax.set_title(f"{name}\ncorr(log dt, log error) = {corr:.3f}")

    order_dt = np.argsort(dts)
    axes[0, 0].plot(dts[order_dt], pred_power[order_dt], "--", color="tab:red",
                     label=f"power law (SSE={sse_power:.4f})")
    axes[0, 0].plot(dts[order_dt], pred_sat[order_dt], "--", color="tab:green",
                     label=f"saturating exp, tau={tau:.0f} (SSE={sse_sat:.4f})")
    axes[0, 0].legend(fontsize=8)

    # Euler-only (hard-coded first-order term alone) vs full
    # (f_theta-corrected) prediction, SAME windows/dt for both -- see
    # this quantity's own computation above for the full rationale.
    # Scatter, not boxplot: the two series need to stay visually
    # distinguishable at each dt, which a side-by-side boxplot pair
    # would clutter given dt's own wide, non-uniform sampling.
    axes[0, 3].scatter(dts, euler_losses / dts, s=8, alpha=0.35, color="tab:orange", label="euler-only / dt")
    axes[0, 3].scatter(dts, latent_losses / dts, s=8, alpha=0.35, color="tab:blue", label="full (f_theta) / dt")
    axes[0, 3].plot(dts[order_dt], pred_power_euler[order_dt] / dts[order_dt], "--", color="tab:orange",
                     label=f"euler-only fit: dt^{a_euler:.3f}, /dt -> dt^{a_euler - 1:.3f}")
    axes[0, 3].plot(dts[order_dt], pred_power[order_dt] / dts[order_dt], "--", color="tab:blue",
                     label=f"full fit: dt^{a:.3f}, /dt -> dt^{a - 1:.3f}")
    axes[0, 3].set_xscale("log")
    axes[0, 3].set_yscale("log")
    axes[0, 3].set_xlabel("dt")
    axes[0, 3].set_ylabel("euler-only/dt, full/dt")
    axes[0, 3].set_title("Euler-only/dt vs full/dt\n(same normalization for both -- direct, "
                          "same-footing comparison)")
    axes[0, 3].legend(fontsize=7)

    # The key actionable panel: which (temperature, noise) region has
    # the highest error, aggregated per run so points don't overplot.
    # Left as a scatter, not a boxplot -- this is genuinely 2D (two
    # discrete axes at once), not a single discrete-x-vs-y relationship.
    # alpha=0.7 (not 1.0): different SEEDS at the same (temperature,
    # noise) setting are still separate points here (aggregation is
    # PER RUN, i.e. per seed -- see the comment above -- not per T/noise
    # pair), so multiple points can genuinely share a coordinate. Without
    # transparency, whichever one matplotlib happens to draw last fully
    # hides the other(s) -- alpha lets overlapping points blend visibly
    # instead. Deliberately NOT jittered: jittering would hide the same
    # problem by DISTORTING the coordinates instead of revealing it, and
    # the exact (temperature, noise) value is the actual thing this
    # panel exists to let someone read off precisely.
    sc = axes[0, 2].scatter(run_temps, run_noises, c=run_mean_loss, s=30 + 10 * run_n_windows,
                             cmap="viridis", edgecolors="black", linewidths=0.3, alpha=0.7)
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

    # Scatter, NOT a boxplot: temperature/noise are discrete because
    # they're SWEEP INPUTS (a handful of fixed values chosen for the
    # simulation grid); length_scale is continuous because it's
    # CALCULATED per window from the actual microstructure, with a
    # different value nearly every time. Binning it first (as an
    # earlier version of this plot did) manufactures artificial
    # discreteness that isn't really there in the underlying data.
    #
    # Uses length_scales_valid/latent_losses_for_length (saturated
    # points already excluded above) -- NOT the full arrays -- so the
    # plot matches the correlation and fit exactly, rather than showing
    # points the regression itself was never fit against.
    axes[1, 2].scatter(length_scales_valid, latent_losses_for_length, s=8, alpha=0.35,
                        color="tab:blue", edgecolors="none")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xlabel("length scale (autocorr_length)")
    axes[1, 2].set_ylabel("latent-space one-step error")
    axes[1, 2].set_title("length scale\n"
                          f"corr(length_scale, log error) = {corr_length_latent:.3f}")
    order_len = np.argsort(length_scales_valid)
    axes[1, 2].plot(length_scales_valid[order_len], pred_exp_len[order_len], "--", color="tab:red",
                     label=f"exponential (SSE={sse_exp_len:.4f})")
    axes[1, 2].plot(length_scales_valid[order_len], pred_sat_len[order_len], "--", color="tab:green",
                     label=f"saturating exp, tau={tau_len:.1f} (SSE={sse_sat_len:.4f})")
    axes[1, 2].legend(fontsize=8)
    # y-axis explicitly matched to the temperature/noise panels (rather
    # than left to its own autoscale) -- all three show the same
    # quantity (latent-space one-step error), and excluding the
    # saturated length_scale points above narrows this panel's own y
    # range slightly, which would otherwise make the panels subtly
    # harder to visually compare side by side.
    axes[1, 2].set_ylim(axes[1, 0].get_ylim())
    # Reference line at the latent bottleneck's own cell size in pixels
    # (Nx / latent_spatial_size). Uses THIS checkpoint's own actual
    # latent_spatial_size (falling back to the shared default,
    # models.constants.LATENT_SPATIAL_SIZE, for older checkpoints saved
    # before that field existed) -- no longer an independently
    # duplicated magic number, see models/constants.py. Purely visual:
    # makes it possible to see by eye whether error spikes specifically
    # for length scales at or below one latent cell, rather than
    # reading that off the correlation coefficient alone (which -- like
    # temperature's -- may not be trustworthy if the true relationship
    # isn't linear/monotonic).
    latent_spatial_size = ae_config.get("latent_spatial_size", LATENT_SPATIAL_SIZE)
    cell_size_px = ae_config["size"] / latent_spatial_size
    axes[1, 2].axvline(cell_size_px, color="black", linestyle=":", linewidth=1)
    axes[1, 2].text(cell_size_px, axes[1, 2].get_ylim()[1], " latent cell size",
                     fontsize=7, ha="left", va="top", rotation=90)

    # LINEAR-scale version of the same two quantities, to directly
    # check the formula's own predicted structure and read off its
    # coefficients numerically -- a log-log plot obscures this: the
    # derivation (dz1 = z1-true_derivative, df = f_theta-true_curvature)
    # gives, dividing BOTH residuals by dt (not dt^2 -- removes the
    # divergent 1/dt term entirely, a cleaner comparison):
    #   euler_residual/dt ~  dz1 + B*dt          (B = -true_curvature/2)
    #   full_residual/dt  ~  dz1 + (df/2)*dt      (SAME intercept dz1 --
    #                                              both reduce to the
    #                                              same first-order term
    #                                              as dt->0, since z1 is
    #                                              the SAME z1 either way)
    # so on a straight, LINEAR x/y scale, each should show up as an
    # actual straight line, with the fitted Y-INTERCEPT directly giving
    # dz1 (from EITHER fit -- a genuine check is whether the two
    # intercepts actually agree with each other), and the fitted SLOPE
    # giving B (euler) or df/2 (full) directly -- readable as real
    # numbers, not just a power-law exponent.
    #
    # Uses the SIGNED residual (euler_losses_signed/latent_losses_signed),
    # not the L1 magnitude used everywhere else in this figure -- the
    # formula above is written in terms of the signed residual (positive
    # and negative components genuinely cancelling), and |residual|
    # would silently answer a different question.
    #
    # Robust (IRLS/Huber) fit, not plain least-squares: a small number
    # of small-dt windows with unusually large residuals can otherwise
    # dominate the line entirely (most visible on euler-only), even
    # though the bulk of the data disagrees with the slope/intercept
    # that a handful of outliers would imply -- see robust_linear_fit's
    # own docstring for the mechanism.
    euler_over_dt = euler_losses_signed / dts
    full_over_dt = latent_losses_signed / dts
    euler_lin_slope, euler_lin_intercept = robust_linear_fit(dts, euler_over_dt)
    full_lin_slope, full_lin_intercept = robust_linear_fit(dts, full_over_dt)

    axes[1, 3].scatter(dts, euler_over_dt, s=8, alpha=0.35, color="tab:orange", label="euler-only / dt")
    axes[1, 3].scatter(dts, full_over_dt, s=8, alpha=0.35, color="tab:blue", label="full (f_theta) / dt")
    # Fit line drawn across the FULL dt range (all data used for the
    # fit itself) -- only the axis's own displayed window is cut below,
    # via set_xlim, so the line doesn't visually stop short even though
    # the view does.
    dt_lin_range = np.array([dts.min(), dts.max()])
    axes[1, 3].plot(dt_lin_range, euler_lin_intercept + euler_lin_slope * dt_lin_range, "--",
                     color="tab:orange",
                     label=f"euler-only robust fit: intercept={euler_lin_intercept:.4e} (~dz1), "
                           f"slope={euler_lin_slope:.4e} (~B)")
    axes[1, 3].plot(dt_lin_range, full_lin_intercept + full_lin_slope * dt_lin_range, "--",
                     color="tab:blue",
                     label=f"full robust fit: intercept={full_lin_intercept:.4e} (~dz1 too, "
                           f"if theory holds), slope={full_lin_slope:.4e} (~df/2)")
    # Cut the DISPLAYED x-axis to the 90th percentile of dt -- dt spans
    # several orders of magnitude but is sampled densely at the low end,
    # so an axis scaled to the true max leaves most of its own width
    # showing only a handful of the largest-dt points. The regression
    # above is entirely unaffected -- it already used every point,
    # before this view was ever applied.
    dt_90 = np.percentile(dts, 90)
    if dt_90 > dts.min():
        axes[1, 3].set_xlim(dts.min(), dt_90)
    axes[1, 3].set_xlabel("dt (linear, x-axis cut at 90th percentile -- fit itself uses ALL data)")
    axes[1, 3].set_ylabel("euler-only/dt, full/dt (linear, signed)")
    axes[1, 3].set_title("Linear-scale version of the panel above\n(checking the formula's own "
                          "predicted linear structure directly)")
    axes[1, 3].legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
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
            help="default: <repo root>/output/stage3/<lds checkpoint name>-parameter_dependence.png")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    check_parameter_dependence(
        lds_checkpoint_path=args.lds_checkpoint, min_step=args.min_step,
        min_stdev_phi=args.min_stdev_phi, output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()