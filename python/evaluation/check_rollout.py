"""
Load a trained LDS checkpoint (and the AE checkpoint it was trained
against), predict z(t+dt) for real held-out test-set transitions,
decode the prediction, and compare against the actual x(t+dt).

Shown as explicit CHANGE rather than raw next-states -- state(t), the
real delta x(t+dt)-x(t), the predicted delta, and the error -- since at
small dt the raw states can look nearly identical, making the actual
predicted dynamics hard to see directly. Also shows the AE's OWN
reconstruction of the true next state alongside the loss numbers, so
LDS prediction error can be told apart from AE reconstruction error
rather than the two being conflated into one number.

REPRODUCIBLE COMPARISON: by default, samples are randomly picked from
the (min_step/min_stdev_phi-filtered) test set -- but since changing
those filters changes WHICH snapshots exist in the dataset at all, the
same --seed does NOT guarantee the same underlying snapshots across
runs with different filter settings, making before/after comparisons
across parameter changes impossible. Every random run prints its exact
picks as 'run_dir:step_t:step_next' triples; pass those back in via
--fixed-windows (repeatable) to see the EXACT SAME snapshots again,
computed fresh from the raw files and the frozen encoder/decoder --
entirely bypassing dataset filtering, so it works regardless of what
min_step/min_stdev_phi the comparison run uses.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_rollout \
        --size 64 --latent-channels 4 --stats-weight 0.01 --n-rollout-steps 3
    python -m evaluation.check_rollout \
        --size 64 --latent-channels 4 --stats-weight 0.01 --n-rollout-steps 3 \
        --fixed-windows "../../datasets/64x64/T800_n050_s79:100000:120000" ...
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from models.autoencoder import Autoencoder
from models.latent_dynamics import LatentDynamics
from training.datasets import MicrostructureEvolutionDataset
from training.losses import ReconLoss
from utils import load_datasets as load
from utils.naming import lds_checkpoint_name


def parse_fixed_window(s: str) -> tuple[Path, int, int]:
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"--fixed-windows entry must be 'run_dir:step_t:step_next', got '{s}'")
    run_dir, step_t, step_next = parts
    return Path(run_dir), int(step_t), int(step_next)


def compute_sample(run_dir: Path, step_t: int, step_next: int, ae, f_theta,
                    ae_config: dict, device: torch.device):
    """
    Everything needed for one row of the comparison figure, computed
    FRESH from the raw snapshot files and the frozen encoder/decoder --
    no dependence on any MicrostructureEvolutionDataset filtering, so
    this works identically regardless of what min_step/min_stdev_phi a
    training run used.
    """
    metadata = load.read_metadata(run_dir / "metadata.txt")
    nx, ny = ae_config["size"], ae_config["size"]

    x_t_raw = load.read_phi_half(run_dir / load.snapshot_filename(step_t), nx, ny)
    x_next_raw = load.read_phi_half(run_dir / load.snapshot_filename(step_next), nx, ny)

    dt_val = (step_next - step_t) * metadata.dt
    theta_val = metadata.temperature - metadata.T0  # see LatentDynamics/dataset docstrings

    with torch.no_grad():
        x_t = torch.from_numpy(x_t_raw).unsqueeze(0).unsqueeze(0).to(device)
        x_next_true_t = torch.from_numpy(x_next_raw).unsqueeze(0).unsqueeze(0).to(device)

        z_t = ae.encoder(x_t)
        z_next_true = ae.encoder(x_next_true_t)

        dt = torch.tensor([dt_val], dtype=torch.float32, device=device)
        theta = torch.tensor([[theta_val]], dtype=torch.float32, device=device)

        dz = f_theta(z_t, dt, theta)
        z_next_pred = z_t + dz

        x_next_pred = ae.decoder(z_next_pred)[0, 0].cpu().numpy()
        x_next_ae_baseline = ae.decoder(z_next_true)[0, 0].cpu().numpy()

    return x_t_raw, x_next_raw, x_next_pred, x_next_ae_baseline, dt_val


def check_rollout(
    lds_checkpoint_path: Path, n_samples: int = 6, seed: int = 0,
    fixed_windows: list[str] | None = None,
    min_step: int | None = None, min_stdev_phi: float | None = None,
    output_path: Path | None = None, device: str | None = None,
) -> Path:
    """Saves a visual rollout-comparison figure and returns its path."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if output_path is None:
        suffix = "" if fixed_windows else f"-seed{seed}"
        output_path = Path(f"../../output/rollout_check_png/{lds_checkpoint_path.stem}{suffix}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lds_checkpoint = torch.load(lds_checkpoint_path, map_location=device, weights_only=True)
    lds_config = lds_checkpoint["config"]
    print(f"Loaded LDS checkpoint from epoch {lds_checkpoint['epoch']}, "
          f"val_loss={lds_checkpoint['val_loss']:.6f}, config={lds_config}")

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

    if fixed_windows:
        windows = [parse_fixed_window(s) for s in fixed_windows]
        print(f"Using {len(windows)} fixed windows (dataset filtering bypassed entirely)")
    else:
        data_config = lds_checkpoint.get("data_config")
        if data_config is None:
            print("WARNING: this checkpoint has no saved data_config -- falling back to "
                  "min_step=0, min_stdev_phi=None, window_length=2, which may NOT match "
                  "what training actually used.")
            data_config = {"min_step": 0, "min_stdev_phi": None, "window_length": 2}
        min_step = min_step if min_step is not None else data_config["min_step"]
        min_stdev_phi = min_stdev_phi if min_stdev_phi is not None else data_config["min_stdev_phi"]
        window_length = data_config["window_length"]

        test_dirs = lds_checkpoint.get("test_dirs") or []
        if not test_dirs:
            raise ValueError(
                f"{lds_checkpoint_path} has no saved test_dirs -- it was likely trained "
                f"with --test-fraction 0."
            )
        test_dirs = [Path(d) for d in test_dirs]

        # Only used to PICK representative (run_dir, step_t, step_next) triples
        # from the actual filtered test distribution -- compute_sample() then
        # does the real work fresh, independent of this dataset object.
        dataset = MicrostructureEvolutionDataset(
            test_dirs, encoder=ae.encoder, device=device, window_length=window_length,
            min_step=min_step, min_stdev_phi=min_stdev_phi,
        )
        if len(dataset) == 0:
            raise ValueError(f"No windows found in the checkpoint's {len(test_dirs)} "
                              f"test_dirs (after min_step={min_step}/"
                              f"min_stdev_phi={min_stdev_phi} filtering)")

        generator = torch.Generator().manual_seed(seed)
        n_samples = min(n_samples, len(dataset))
        indices = torch.randperm(len(dataset), generator=generator)[:n_samples].tolist()

        windows = []
        for idx in indices:
            run_dir, steps = dataset.window_info(idx)
            windows.append((run_dir, steps[0], steps[1]))

        print("\nSelected windows -- reuse via --fixed-windows for reproducible comparison:")
        for run_dir, step_t, step_next in windows:
            print(f"  {run_dir}:{step_t}:{step_next}")
        print()

    recon_loss = ReconLoss()
    n_samples = len(windows)

    fig, axes = plt.subplots(n_samples, 4, figsize=(17, 3.2 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    for row, (run_dir, step_t, step_next) in enumerate(windows):
        x_t_raw, x_next_raw, x_next_pred, x_next_ae_baseline, dt_val = compute_sample(
            run_dir, step_t, step_next, ae, f_theta, ae_config, device,
        )

        x_next_pred_t = torch.from_numpy(x_next_pred).unsqueeze(0).unsqueeze(0)
        x_next_raw_t = torch.from_numpy(x_next_raw).unsqueeze(0).unsqueeze(0)
        x_next_baseline_t = torch.from_numpy(x_next_ae_baseline).unsqueeze(0).unsqueeze(0)

        end_to_end_loss = recon_loss(x_next_pred_t, x_next_raw_t).item()
        ae_baseline_loss = recon_loss(x_next_baseline_t, x_next_raw_t).item()

        # Explicit deltas rather than raw next-states: at small dt the
        # raw x(t) and x(t+dt) can look nearly identical, making the
        # actual predicted dynamics hard to see directly. real_delta
        # and predicted_delta share ONE scale (not each auto-scaled
        # independently) so they're directly visually comparable --
        # that comparison is the whole point of this panel.
        real_delta = x_next_raw - x_t_raw
        predicted_delta = x_next_pred - x_t_raw
        error = predicted_delta - real_delta

        state_scale = max(abs(x_t_raw.min()), abs(x_t_raw.max()), 0.1)
        delta_scale = max(abs(real_delta.min()), abs(real_delta.max()),
                           abs(predicted_delta.min()), abs(predicted_delta.max()), 0.02)
        error_scale = max(abs(error.min()), abs(error.max()), 1e-6)

        axes[row, 0].imshow(x_t_raw, cmap="RdBu", vmin=-state_scale, vmax=state_scale)
        axes[row, 0].set_title(f"state(t)\n{run_dir.name}:{step_t}\ndt={dt_val:.1f}" if row == 0
                                else f"{run_dir.name}:{step_t}\ndt={dt_val:.1f}", fontsize=9)
        axes[row, 1].imshow(real_delta, cmap="RdBu", vmin=-delta_scale, vmax=delta_scale)
        axes[row, 1].set_title(f"real \u0394x\nscale=+-{delta_scale:.3f}"
                                if row == 0 else f"scale=+-{delta_scale:.3f}", fontsize=10)
        axes[row, 2].imshow(predicted_delta, cmap="RdBu", vmin=-delta_scale, vmax=delta_scale)
        axes[row, 2].set_title(
            f"predicted \u0394x\nloss={end_to_end_loss:.4f} (AE={ae_baseline_loss:.4f})"
            if row == 0 else
            f"loss={end_to_end_loss:.4f} (AE={ae_baseline_loss:.4f})", fontsize=10
        )
        im_error = axes[row, 3].imshow(error, cmap="RdBu", vmin=-error_scale, vmax=error_scale)
        axes[row, 3].set_title(f"error\nscale=+-{error_scale:.3f}" if row == 0
                                else f"scale=+-{error_scale:.3f}", fontsize=10)
        fig.colorbar(im_error, ax=axes[row, 3], fraction=0.046)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(output_path, dpi=120)
    print(f"Saved rollout comparison to {output_path} ({n_samples} samples)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True,
                         help="grid size (square only) -- config.txt is never read")
    parser.add_argument("--latent-channels", type=int, default=None, help="see --size")
    parser.add_argument("--stats-weight", type=float, default=None)
    parser.add_argument("--n-rollout-steps", type=int, default=None, help="see --size")
    parser.add_argument("--lds-checkpoint", type=Path, default=None,
            help="direct path override, if you'd rather specify the checkpoint this way "
                 "instead of by --size/--latent-channels/--stats-weight/--n-rollout-steps")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0,
                         help="which test-set windows to display (ignored if --fixed-windows "
                              "is given), not the train/val/test split itself, which is "
                              "fixed and loaded from the checkpoint")
    parser.add_argument("--fixed-windows", type=str, nargs="+", default=None,
                         help="exact 'run_dir:step_t:step_next' triples to display "
                              "(repeatable), bypassing random dataset-based selection "
                              "entirely -- for reproducible before/after comparisons across "
                              "different --min-step/--min-stdev-phi training runs. Every "
                              "random run prints its picks in this exact format.")
    parser.add_argument("--min-step", type=int, default=None,
                         help="override the checkpoint's recorded min_step, if given "
                              "(ignored if --fixed-windows is given)")
    parser.add_argument("--min-stdev-phi", type=float, default=None,
                         help="override the checkpoint's recorded min_stdev_phi, if given "
                              "(ignored if --fixed-windows is given)")
    parser.add_argument("--output", type=Path, default=None,
            help="default: ../../output/rollout_check_png/<lds checkpoint name>"
                 "[-seed<N> if random-sampling].png -- named after the checkpoint "
                 "(and seed) so different checks don't collide")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.lds_checkpoint is None:
        missing = [n for n, v in [("--latent-channels", args.latent_channels),
                                   ("--stats-weight", args.stats_weight),
                                   ("--n-rollout-steps", args.n_rollout_steps)] if v is None]
        if missing:
            raise ValueError(
                f"Provide either --lds-checkpoint directly, or --latent-channels, "
                f"--stats-weight, and --n-rollout-steps (missing: {', '.join(missing)})."
            )
        name = lds_checkpoint_name(args.size, args.latent_channels, args.stats_weight,
                                    args.n_rollout_steps)
        args.lds_checkpoint = Path(f"../../checkpoints/stage3/{name}.pt")
        print(f"Reconstructed checkpoint path: {args.lds_checkpoint}")

    check_rollout(
        lds_checkpoint_path=args.lds_checkpoint, n_samples=args.n_samples, seed=args.seed,
        fixed_windows=args.fixed_windows, min_step=args.min_step,
        min_stdev_phi=args.min_stdev_phi, output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()
