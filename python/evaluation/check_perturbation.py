"""
Stage 3 latent validation: perturbation robustness, quantified via an
explicit power-law fit rather than a visual "looks proportional" check.

Two perturbation directions (per docs):
  - LATENT perturbation: z_eps = z + eps*eta, decode, measure real-space
    ||D(z_eps) - x|| as a function of eps -- tests the DECODER's local
    sensitivity/linearity around real data.
  - REAL-space perturbation: x_eps = x + eps*eta, re-encode, measure
    latent-space ||E(x_eps) - z|| as a function of eps -- tests the
    ENCODER's local sensitivity/linearity instead.

For a well-behaved (locally linear) map, error should scale as eps^1 for
small eps -- fit log(error) = a*log(eps) + b via least squares and report
the fitted slope `a` (expect ~1.0) and R^2 (how well it follows ANY power
law over the tested range -- deviation at large eps is EXPECTED, since
eventually a large enough perturbation must leave the locally-linear
regime; the diagnostic value is seeing where that happens, not assuming
it shouldn't).

The fitted slope from the LATENT-perturbation direction also serves as
an empirical calibration between latent-space and real-space distance
(see module docstring discussion) -- printed explicitly so other
latent-distance metrics elsewhere can be interpreted through it, rather
than left as raw, unitless numbers.

Isotropic (eta ~ N(0,1)) and directional (eta = a fixed real trajectory
direction, via --direction-window) variants are both supported.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_perturbation \
        --checkpoint ../../output/ae_checkpoint_pt/64x64-4latent-stats_weight_0p01.pt
    python -m evaluation.check_perturbation \
        --checkpoint ../../output/ae_checkpoint_pt/64x64-4latent-stats_weight_0p01.pt \
        --direction-window "../../datasets/64x64/T800_n050_s79:100000:120000"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.autoencoder import Autoencoder
from training.datasets import MicrostructureEvolutionDataset
from utils import load_datasets as load


def power_law_fit(eps_values: np.ndarray, errors: np.ndarray) -> tuple[float, float, float]:
    """
    Least-squares fit of log(error) = a*log(eps) + b. Returns (a, b, r_squared).
    r_squared measures how well the data follows ANY power law over the
    tested range -- not whether a=1 specifically.
    """
    log_eps = np.log(eps_values)
    log_err = np.log(np.clip(errors, 1e-12, None))
    a, b = np.polyfit(log_eps, log_err, 1)
    pred = a * log_eps + b
    ss_res = np.sum((log_err - pred) ** 2)
    ss_tot = np.sum((log_err - log_err.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r_squared


def parse_fixed_window(s: str) -> tuple[Path, int, int]:
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"expected 'run_dir:step_t:step_next', got '{s}'")
    return Path(parts[0]), int(parts[1]), int(parts[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=8,
                         help="number of real test-set frames to perturb and average over")
    parser.add_argument("--n-repeats", type=int, default=16,
                         help="random eta draws per (sample, eps) -- averaged, to separate "
                              "genuine eps-scaling from single-draw direction noise")
    parser.add_argument("--eps-values", type=float, nargs="+",
                         default=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                         help="sweep of perturbation magnitudes, log-spaced by default")
    parser.add_argument("--direction-window", type=str, default=None,
                         help="'run_dir:step_t:step_next' -- if given, perturb along this "
                              "real trajectory's normalized (z(t+dt)-z(t)) direction instead "
                              "of isotropic random noise, for each sample independently "
                              "reusing the SAME direction (tests sensitivity along a "
                              "physically meaningful axis vs generic isotropic robustness)")
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
            help="default: ../../output/perturbation_check_png/<checkpoint name>.png")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"../../output/perturbation_check_png/{args.checkpoint.stem}.png")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, config={model_cfg}")

    ae = Autoencoder(
        size=model_cfg["size"], channels=1,
        base_channels=model_cfg["base_channels"], latent_channels=model_cfg["latent_channels"],
    ).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()

    test_dirs = checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(f"{args.checkpoint} has no saved test_dirs")
    test_dirs = [Path(d) for d in test_dirs]

    # Sample real frames from the held-out test set, at the AE's own size.
    nx = ny = model_cfg["size"]
    frames = []
    generator = torch.Generator().manual_seed(args.seed)
    for run_dir in test_dirs:
        metadata = load.read_metadata(run_dir / "metadata.txt")
        check = load.check_snapshots_saved(run_dir, metadata)
        bad_steps = set(check["missing"]) | set(check["bad_size"])
        for step in metadata.save_steps:
            if step in bad_steps or step < args.min_step:
                continue
            frames.append((run_dir, step))
    if len(frames) == 0:
        raise ValueError("No usable test frames found")
    idx = torch.randperm(len(frames), generator=generator)[:args.n_samples].tolist()
    chosen = [frames[i] for i in idx]
    print(f"Perturbing {len(chosen)} real test-set frames")

    # Optional fixed direction, computed once, reused for every sample
    # (same direction vector applied everywhere -- tests sensitivity
    # along ONE physically meaningful axis consistently, not a
    # per-sample-local one, which would conflate "is this direction
    # special" with "is this a different direction each time").
    fixed_direction = None
    if args.direction_window:
        run_dir, step_t, step_next = parse_fixed_window(args.direction_window)
        metadata = load.read_metadata(run_dir / "metadata.txt")
        x_t = load.read_phi_half(run_dir / load.snapshot_filename(step_t), nx, ny)
        x_next = load.read_phi_half(run_dir / load.snapshot_filename(step_next), nx, ny)
        with torch.no_grad():
            z_t = ae.encoder(torch.from_numpy(x_t).unsqueeze(0).unsqueeze(0).to(device))
            z_next = ae.encoder(torch.from_numpy(x_next).unsqueeze(0).unsqueeze(0).to(device))
        direction = (z_next - z_t)
        direction = direction / direction.norm()
        fixed_direction = direction
        print(f"Using fixed direction from {args.direction_window} "
              f"(directional/structured perturbation mode)")

    eps_values = np.array(args.eps_values)
    decoder_errors = np.zeros(len(eps_values))  # ||D(z+eps*eta) - x||, real space
    encoder_errors = np.zeros(len(eps_values))  # ||E(x+eps*eta) - z||, latent space

    with torch.no_grad():
        for run_dir, step in chosen:
            nx_i, ny_i = nx, ny
            x_np = load.read_phi_half(run_dir / load.snapshot_filename(step), nx_i, ny_i)
            x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
            z = ae.encoder(x)

            for i, eps in enumerate(eps_values):
                d_err_sum, e_err_sum = 0.0, 0.0
                for _ in range(args.n_repeats):
                    if fixed_direction is not None:
                        eta_latent = fixed_direction
                    else:
                        eta_latent = torch.randn_like(z)
                    z_eps = z + eps * eta_latent
                    x_from_z_eps = ae.decoder(z_eps)
                    d_err_sum += (x_from_z_eps - x).pow(2).mean().sqrt().item()

                    eta_real = torch.randn_like(x)  # directional real-space perturbation
                                                     # not defined without a decode step,
                                                     # so real-space variant stays isotropic
                    x_eps = x + eps * eta_real
                    z_from_x_eps = ae.encoder(x_eps)
                    e_err_sum += (z_from_x_eps - z).pow(2).mean().sqrt().item()

                decoder_errors[i] += d_err_sum / args.n_repeats
                encoder_errors[i] += e_err_sum / args.n_repeats

    decoder_errors /= len(chosen)
    encoder_errors /= len(chosen)

    a_dec, b_dec, r2_dec = power_law_fit(eps_values, decoder_errors)
    a_enc, b_enc, r2_enc = power_law_fit(eps_values, encoder_errors)

    print(f"\nDecoder response (latent perturbation -> real-space error):")
    print(f"  fitted slope a = {a_dec:.3f} (expect ~1.0 for locally linear), R^2 = {r2_dec:.4f}")
    print(f"  CALIBRATION: exp(b) = {np.exp(b_dec):.4f} real-space units per 1 unit of "
          f"latent eps -- use this to convert other raw latent distances into an "
          f"interpretable real-space scale")

    print(f"\nEncoder response (real-space perturbation -> latent-space error):")
    print(f"  fitted slope a = {a_enc:.3f} (expect ~1.0 for locally linear), R^2 = {r2_enc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, errors, a, b, r2, title, ylabel in [
        (axes[0], decoder_errors, a_dec, b_dec, r2_dec,
         "Decoder: latent eps -> real-space error", "||D(z+eps*eta) - x|| (real space)"),
        (axes[1], encoder_errors, a_enc, b_enc, r2_enc,
         "Encoder: real-space eps -> latent error", "||E(x+eps*eta) - z|| (latent space)"),
    ]:
        ax.scatter(eps_values, errors, s=40, label="measured")
        fit_line = np.exp(b) * eps_values ** a
        ax.plot(eps_values, fit_line, "--", color="gray",
                label=f"fit: slope={a:.2f}, R2={r2:.3f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("eps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
