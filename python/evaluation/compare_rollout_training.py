"""
Compare two LDS checkpoints (e.g. one-step-trained vs rollout-trained)
on their ACTUAL target: chained multi-step rollout error, where each
model's own prediction feeds into the next step. check_rollout.py ALSO
chains a full rollout (see its own docstring) -- the difference is
scope: check_rollout.py reports only the FINAL step, visually, in pixel
space, for ONE checkpoint at a time; this script tracks the full
PER-STEP error progression, in latent space, for TWO checkpoints side
by side, specifically to see whether/how fast error accumulates
differently between them over the rollout horizon.

For each fixed window (run_dir + a list of consecutive steps), both
models' rollout() is run from the same true starting latent, and the
per-step error against the true trajectory is compared -- this is the
comparison that actually tells you whether training on chained rollouts
paid off in reduced drift, or whether it just cost one-step accuracy
for nothing.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.compare_rollout_training \
        --checkpoint-a ../../output/lds_checkpoint_onestep.pt --label-a onestep \
        --checkpoint-b ../../output/lds_checkpoint_rollout3.pt --label-b rollout3 \
        --fixed-windows "../../datasets/64x64/T750_n030_s97:10000:20000:30000:40000"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.constants import LATENT_SPATIAL_SIZE, theta_coordinates, N_THETA
from models.latent_dynamics import LatentDynamics, integration_kwargs_from_config
from utils.window_parsing import parse_fixed_window
from training.checkpoint_components import build_ae_from_checkpoint
from utils import load_datasets as load

# GENERAL POLICY (matches training/train_refinement.py's own
# _PYTHON_ROOT): every default checkpoint/output path is built from
# THIS anchor, never from a bare relative string like "../../output/...".
# Relative strings resolve against the process's CWD at invocation
# time, which silently differs across bare CLI and `python -m`.
# Path(__file__) is anchored to THIS FILE's own on-disk location
# instead, which is invariant regardless of how/from-where the process
# was launched.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/compare_rollout_training.py -> python/



def load_lds(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    f_theta = LatentDynamics(
        latent_channels=config["latent_channels"], n_theta=N_THETA,
        latent_spatial=config.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
        hidden_dim=config["hidden_dim"], n_hidden_layers=config["n_hidden_layers"],
        # EVERY meaning-changing field, from ONE list -- see
        # integration_kwargs_from_config. The comments this replaces recorded,
        # three times independently, that "fixing dt_cap in either of those did
        # NOT fix it here": the same field had to be added by hand at each of
        # five separate reconstruction sites. alpha then reached four of them
        # and not model_assembly, and stage 4 ran an adaptive f_theta one-shot
        # at dt=500. One list, one call, and a new field cannot miss a site.
        **integration_kwargs_from_config(config),
    ).to(device)
    from models.encoder import zero_pad_theta_columns
    f_theta.load_state_dict(zero_pad_theta_columns(checkpoint["model_state"], f_theta))
    f_theta.eval()
    return f_theta, checkpoint


def rollout_errors(run_dir: Path, steps: list[int], encoder, f_theta,
                    ae_config: dict, device: torch.device,
                    recon_stream_name: str = "state") -> np.ndarray:
    """
    Encode the TRUE trajectory at every step, then chain f_theta's own
    predictions forward from steps[0] only -- comparing against the
    true latents at steps[1:]. Returns per-step L1 error, (len(steps)-1,).
    """
    metadata = load.read_metadata(run_dir / "metadata.txt")
    nx, ny = ae_config["size"], ae_config["size"]

    frames = torch.stack([
        torch.from_numpy(load.read_phi_half(
            run_dir / load.snapshot_filename(step), nx, ny)).unsqueeze(0)
        for step in steps
    ]).to(device)  # (n_steps+1, 1, ny, nx)

    # theta computed BEFORE the encoder call (it only depends on
    # metadata, not on frames/z_true at all) -- Encoder.forward()
    # requires theta whenever any of its own streams are
    # theta-conditioned (raises ValueError otherwise, not a silent
    # skip); a single (1, n_theta) theta broadcasts correctly against
    # the (n_steps+1, ...) batch of frames via standard FiLM
    # broadcasting, confirmed directly -- no need to expand/repeat it
    # to match frames' own batch size.
    theta = torch.tensor([theta_coordinates(metadata.temperature, metadata.T0)],
                          dtype=torch.float32, device=device)

    with torch.no_grad():
        z_all = encoder(frames, theta=theta)  # every stream, one pass
        z_true = z_all[recon_stream_name]  # (n_steps+1, C, 8, 8)
        # rollout() teacher-forces z1 at every step from REAL, encoded
        # values (see its own docstring) -- "deriv" is the deriv stream's
        # own, fixed name throughout this project (train_stage2.py's own
        # convention; any checkpoint that reached stage 3 at all must
        # have it, since f_theta is trained against real z1 values).
        # unsqueeze(0): rollout() expects (B, n_steps+1, C, 8, 8); B=1
        # here, one window at a time.
        z1_sequence = z_all["deriv"].unsqueeze(0)

    dts = torch.tensor(
        [[(steps[i + 1] - steps[i]) * metadata.dt for i in range(len(steps) - 1)]],
        dtype=torch.float32, device=device,
    )  # (1, n_steps)

    z0 = z_true[0:1]
    with torch.no_grad():
        z_hat = f_theta.rollout(z0, z1_sequence, dts, theta)[0]  # (n_steps+1, C, 8, 8), z_hat[0]==z0 exactly

    return (z_hat[1:] - z_true[1:]).abs().mean(dim=(1, 2, 3)).cpu().numpy()


def compare_rollout_training(
    checkpoint_a: Path, checkpoint_b: Path, fixed_windows: list[str],
    label_a: str = "A", label_b: str = "B",
    output_path: Path | None = None, device: str | None = None,
) -> Path:
    """Saves the per-step chained-rollout comparison plot and returns
    its path. See this module's own docstring for what this compares
    and why, versus check_rollout.py."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if output_path is None:
        output_path = _PYTHON_ROOT.parent / "output" / "rollout_vs_onestep.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    f_theta_a, checkpoint_a_data = load_lds(checkpoint_a, device)
    f_theta_b, checkpoint_b_data = load_lds(checkpoint_b, device)

    ae_checkpoint_path_a = Path(checkpoint_a_data["ae_checkpoint"])
    ae_checkpoint_path_b = Path(checkpoint_b_data["ae_checkpoint"])
    if ae_checkpoint_path_a != ae_checkpoint_path_b:
        print(f"WARNING: checkpoints were trained against DIFFERENT autoencoders "
              f"({ae_checkpoint_path_a} vs {ae_checkpoint_path_b}) -- this comparison "
              f"may not be apples-to-apples.")

    ae, ae_encoder, ae_checkpoint, stream_configs, recon_stream_name = build_ae_from_checkpoint(
        ae_checkpoint_path_a, device,
    )
    ae_config = ae_checkpoint["config"]

    windows = [parse_fixed_window(s) for s in fixed_windows]
    n_steps = len(windows[0][1]) - 1
    for run_dir, steps in windows:
        if len(steps) - 1 != n_steps:
            raise ValueError("all fixed_windows entries must have the same number of steps "
                              "for the per-step comparison plot to line up")

    all_errors_a, all_errors_b = [], []
    for run_dir, steps in windows:
        errors_a = rollout_errors(run_dir, steps, ae_encoder, f_theta_a, ae_config, device,
                                   recon_stream_name=recon_stream_name)
        errors_b = rollout_errors(run_dir, steps, ae_encoder, f_theta_b, ae_config, device,
                                   recon_stream_name=recon_stream_name)
        all_errors_a.append(errors_a)
        all_errors_b.append(errors_b)
        print(f"{run_dir}:{':'.join(map(str, steps))}")
        print(f"  {label_a}: {np.array2string(errors_a, precision=4)}")
        print(f"  {label_b}: {np.array2string(errors_b, precision=4)}")

    all_errors_a = np.array(all_errors_a)  # (n_windows, n_steps)
    all_errors_b = np.array(all_errors_b)

    print(f"\nMean per-step error across {len(windows)} windows:")
    print(f"  {label_a}: {np.array2string(all_errors_a.mean(axis=0), precision=4)}")
    print(f"  {label_b}: {np.array2string(all_errors_b.mean(axis=0), precision=4)}")

    final_a = all_errors_a[:, -1].mean()
    final_b = all_errors_b[:, -1].mean()
    print(f"\nFinal-step mean error -- {label_a}: {final_a:.4f}, {label_b}: {final_b:.4f}")
    print(f"({'B' if final_b < final_a else 'A'} drifts less over the full horizon)")

    fig, ax = plt.subplots(figsize=(8, 6))
    steps_axis = np.arange(1, n_steps + 1)
    for errors, label, color in [(all_errors_a, label_a, "tab:blue"),
                                  (all_errors_b, label_b, "tab:orange")]:
        for row in errors:
            ax.plot(steps_axis, row, color=color, alpha=0.25)
        ax.plot(steps_axis, errors.mean(axis=0), color=color, linewidth=3, label=label)
    ax.set_xlabel("rollout step (chained from the same true starting point)")
    ax.set_ylabel("latent L1 error vs true trajectory")
    ax.set_title("Chained rollout error: does it accumulate faster for one model?")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)  # a figure left open is held by pyplot's global
    # registry forever -- in a persistent kernel that is a real leak, and
    # every other check_*.py already closes its own.
    print(f"\nSaved comparison plot to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument("--fixed-windows", type=str, nargs="+", required=True,
                         help="'run_dir:step0:step1:...:stepN' (repeatable)")
    parser.add_argument("--output", type=Path,
                         default=_PYTHON_ROOT.parent / "output" / "rollout_vs_onestep.png")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    compare_rollout_training(
        checkpoint_a=args.checkpoint_a, checkpoint_b=args.checkpoint_b,
        fixed_windows=args.fixed_windows, label_a=args.label_a, label_b=args.label_b,
        output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()