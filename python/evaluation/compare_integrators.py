"""
Direct empirical comparison: forward-Euler (LatentDynamics.forward) vs
Adams-Bashforth 2-step (LatentDynamics.forward_ab2), both using the
SAME trained rate function, evaluated on real test-set windows -- to
settle whether AB2 actually helps on this model/data, rather than
reasoning about it in the abstract. Toy-ODE testing suggested AB2 can
be WORSE than Euler for large steps (comparable to or bigger than the
dynamics' own timescale), which is exactly the regime
check_parameter_dependence.py found to be worst -- this script checks whether
that caution applies here.

Needs window_length=3 (t-dt_prev, t, t+dt_curr triplets), unlike the
window_length=2 used for one-step training/checking.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.compare_integrators --lds-checkpoint ../output/lds_checkpoint.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.autoencoder import Autoencoder
from models.latent_dynamics import LatentDynamics
from training.datasets import MicrostructureEvolutionDataset
from training.losses import OneStepLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lds-checkpoint", type=Path, default=Path("../../output/lds_checkpoint.pt"))
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--min-stdev-phi", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("../../output/integrator_comparison.png"))
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    lds_checkpoint = torch.load(args.lds_checkpoint, map_location=device, weights_only=True)
    lds_config = lds_checkpoint["config"]

    data_config = lds_checkpoint.get("data_config")
    if data_config is None:
        print("WARNING: checkpoint has no saved data_config -- falling back to "
              "min_step=0, min_stdev_phi=None (may not match training).")
        data_config = {"min_step": 0, "min_stdev_phi": None}
    min_step = args.min_step if args.min_step is not None else data_config["min_step"]
    min_stdev_phi = args.min_stdev_phi if args.min_stdev_phi is not None else data_config["min_stdev_phi"]

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

    # window_length=3: need (t-dt_prev, t, t+dt_curr) triplets, one more
    # step than one-step training/checking used.
    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=ae.encoder, device=device, window_length=3,
        min_step=min_step, min_stdev_phi=min_stdev_phi,
    )
    print(f"Evaluating {len(dataset)} test triplets...")

    one_step_loss = OneStepLoss(kind="l1")

    dt_currs, euler_errors, ab2_errors = [], [], []

    with torch.no_grad():
        for idx in range(len(dataset)):
            z_window, dt_window, theta = dataset[idx]
            z_prev = z_window[0:1].to(device)
            z_curr = z_window[1:2].to(device)
            z_next_true = z_window[2:3].to(device)
            dt_prev = dt_window[0:1].to(device)
            dt_curr = dt_window[1:2].to(device)
            theta_b = theta.unsqueeze(0).to(device)

            z_euler = z_curr + f_theta(z_curr, dt_curr, theta_b)
            z_ab2 = f_theta.forward_ab2(z_prev, z_curr, dt_prev, dt_curr, theta_b)

            euler_errors.append(one_step_loss(z_euler, z_next_true).item())
            ab2_errors.append(one_step_loss(z_ab2, z_next_true).item())
            dt_currs.append(dt_curr.item())

    dt_currs = np.array(dt_currs)
    euler_errors = np.array(euler_errors)
    ab2_errors = np.array(ab2_errors)

    ab2_wins = ab2_errors < euler_errors
    print(f"\nAB2 beats Euler on {ab2_wins.mean():.1%} of all {len(dataset)} triplets overall\n")

    print("dt decade      n     Euler wins   AB2 wins    mean Euler err   mean AB2 err")
    log_dt = np.log10(dt_currs)
    edges = np.floor(log_dt.min()), np.ceil(log_dt.max())
    for lo in np.arange(edges[0], edges[1] + 1):
        mask = (log_dt >= lo) & (log_dt < lo + 1)
        n = mask.sum()
        if n == 0:
            continue
        print(f"1e{lo:.0f} - 1e{lo+1:.0f}   {n:4d}   "
              f"{(~ab2_wins[mask]).sum():4d} ({(~ab2_wins[mask]).mean():.0%})   "
              f"{ab2_wins[mask].sum():4d} ({ab2_wins[mask].mean():.0%})   "
              f"{euler_errors[mask].mean():.6f}       {ab2_errors[mask].mean():.6f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(dt_currs, euler_errors, alpha=0.3, s=10, label="Euler", color="tab:blue")
    ax.scatter(dt_currs, ab2_errors, alpha=0.3, s=10, label="AB2", color="tab:orange")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dt")
    ax.set_ylabel("one-step error (latent L1)")
    ax.set_title("Euler vs AB2 prediction error by dt")
    ax.legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"\nSaved comparison plot to {args.output}")


if __name__ == "__main__":
    main()
