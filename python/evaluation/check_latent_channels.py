"""
Visualizes what each of the AE's latent bottleneck channels actually
responds to: for a handful of representative input microstructures,
shows the raw input frame alongside every one of that frame's latent
channels (each an 8x8 map -- the bottleneck's spatial size regardless
of Nx/Ny; see training/datasets.py's _LATENT_SPATIAL_SIZE docstring).

Motivated by today's parameter-dependence diagnostics turning up a
puzzling length_scale trend (error correlates POSITIVELY with length
scale -- the opposite of what "8x8 too coarse for fine detail" would
predict, but too weak/noisy a trend to draw a firm conclusion from
alone). Seeing each channel's own spatial activation pattern directly,
across a few structurally different inputs (a thin sharp interface, a
dense fine texture, a curved front, ...), is a more direct way to ask
"what does each channel actually encode" than any further correlation
plot can answer on its own.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_latent_channels --ae-checkpoint ../checkpoints/stage2/64x64.pt

    # Reproduce a specific set of frames (e.g. ones already known to be
    # interesting from a rollout figure), instead of a random draw:
    python -m evaluation.check_latent_channels --ae-checkpoint ../checkpoints/stage2/64x64.pt \\
        --fixed-frames ../datasets/64x64/T950_n020_s79:400000 \\
        --fixed-frames ../datasets/64x64/T575_n050_s131:5000 \\
        --fixed-frames ../datasets/64x64/T625_n005_s191:100000
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.check_rollout import _format_small, _padded_bounds
from models.autoencoder import Autoencoder
from models.latent_streams import DEFAULT_STREAM_NAME
from training.datasets import MicrostructureSnapshotDataset
from training.losses import ReconLoss
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


def parse_fixed_frame(s: str) -> tuple[Path, int]:
    """
    'run_dir:step' -> (Path(run_dir), step).

    NOT a naive split(':') -- same reasoning as
    check_rollout.parse_fixed_window (not reused directly: that
    function requires >=2 trailing step numbers for a whole window,
    this needs exactly 1 for a single frame). run_dir itself can
    contain a colon on Windows (e.g. 'D:\\work\\...\\T950_n020_s79'),
    which a naive split(':') would misread as if it were the step
    number -- split from the right instead, taking only the final
    colon-separated part as the step.
    """
    parts = s.split(":")
    if len(parts) < 2:
        raise ValueError(f"--fixed-frames entry must be 'run_dir:step', got '{s}'")
    try:
        step = int(parts[-1])
    except ValueError:
        raise ValueError(f"--fixed-frames entry must end in an integer step, got '{s}'")
    return Path(":".join(parts[:-1])), step


def rank_channel_importance(
    ae, dataset, device, n_samples: int = 200, seed: int = 0,
) -> np.ndarray:
    """
    Per-channel ABLATION importance: for a random sample of the real
    test set (NOT just the handful of frames shown in the figure --
    those are chosen for visual variety, not statistical coverage), zero
    out one channel at a time in z, decode, and measure how much
    reconstruction loss increases relative to the unablated decode.
    Averaged over samples, per channel, then returned as a
    (latent_channels,) array -- higher means the decoder relies on that
    channel MORE (removing it hurts reconstruction more), which is a
    direct causal measure of importance, unlike cross-frame std (which
    only says a channel varies across inputs, not that the variation
    matters to the output). A channel could in principle have high std
    but low ablation importance (encoding something the decoder barely
    uses), or vice versa -- std and this ranking are asking genuinely
    different questions, not two readings of the same thing.
    """
    recon_loss = ReconLoss(kind="l1")
    n_samples = min(n_samples, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n_samples].tolist()

    latent_channels = None
    total_delta = None
    with torch.no_grad():
        for idx in indices:
            x = dataset[idx].unsqueeze(0).to(device)
            z = ae.encoder(x)[DEFAULT_STREAM_NAME]
            if latent_channels is None:
                latent_channels = z.shape[1]
                total_delta = np.zeros(latent_channels)
            x_recon_full = ae.decoder(z)
            base_loss = recon_loss(x_recon_full, x).item()
            for c in range(latent_channels):
                z_ablated = z.clone()
                z_ablated[:, c] = 0.0
                x_recon_ablated = ae.decoder(z_ablated)
                ablated_loss = recon_loss(x_recon_ablated, x).item()
                total_delta[c] += ablated_loss - base_loss
    return total_delta / n_samples


def check_latent_channels(
    ae_checkpoint_path: Path, fixed_frames: list[str] | None = None,
    n_frames: int = 12, seed: int = 0, min_step: int = 0,
    n_importance_samples: int = 200, skip_importance: bool = False,
    output_path: Path | None = None, device: str | None = None,
) -> Path:
    """Saves a figure showing, for each selected input frame, the raw
    input alongside every one of the AE's latent channels, and returns
    its path."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if output_path is None:
        output_path = (_PYTHON_ROOT.parent / "output" / "stage2"
                       / f"{ae_checkpoint_path.stem}-latent_channels.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(ae_checkpoint_path, map_location=device, weights_only=True)
    ae_config = checkpoint["config"]
    ae = Autoencoder(
        size=ae_config["size"], channels=1,
        base_channels=ae_config["base_channels"], latent_channels=ae_config["latent_channels"],
    ).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()

    nx, ny = ae_config["size"], ae_config["size"]
    latent_channels = ae_config["latent_channels"]

    # Test dataset built regardless of whether --fixed-frames was given
    # -- fixed_frames only decides which frames get VISUALIZED; ablation
    # importance below always needs the real test set for statistical
    # coverage, not just the handful of frames chosen for visual variety.
    test_dirs = checkpoint.get("test_dirs") or []
    test_dataset = None
    if test_dirs:
        test_dirs = [Path(d) for d in test_dirs]
        test_dataset = MicrostructureSnapshotDataset(test_dirs, augment=False, min_step=min_step)
        if len(test_dataset) == 0:
            print(f"WARNING: no snapshots found in the checkpoint's {len(test_dirs)} "
                  f"test_dirs (after min_step={min_step} filtering) -- skipping "
                  f"importance ranking.")
            test_dataset = None
    elif not fixed_frames:
        raise ValueError(f"{ae_checkpoint_path} has no saved test_dirs -- pass "
                          f"--fixed-frames explicitly instead.")

    if fixed_frames:
        frames = [parse_fixed_frame(s) for s in fixed_frames]
        print(f"Using {len(frames)} fixed frames (dataset filtering bypassed entirely)")
    else:
        generator = torch.Generator().manual_seed(seed)
        n_frames = min(n_frames, len(test_dataset))
        indices = torch.randperm(len(test_dataset), generator=generator)[:n_frames].tolist()
        frames = [test_dataset.frame_info(i) for i in indices]
        print(f"Selected {len(frames)} random frames (seed={seed}) -- reuse via "
              f"--fixed-frames for reproducible comparison:")
        for run_dir, step in frames:
            print(f"  {run_dir}:{step}")

    # All z maps computed FIRST, so each channel's color scale can be
    # set ACROSS every shown frame (not per-frame) -- makes the SAME
    # channel directly comparable frame-to-frame (down a column), which
    # is the whole point ("what triggers this channel"). A per-frame
    # scale would auto-normalize away genuine differences in how
    # strongly a channel responds to different inputs.
    all_x, all_z = [], []
    with torch.no_grad():
        for run_dir, step in frames:
            x_raw = load.read_phi_half(run_dir / load.snapshot_filename(step), nx, ny)
            x = torch.from_numpy(x_raw).unsqueeze(0).unsqueeze(0).to(device)
            z = ae.encoder(x)[DEFAULT_STREAM_NAME][0].cpu().numpy()  # (latent_channels, 8, 8)
            all_x.append(x_raw)
            all_z.append(z)
    all_z_arr = np.stack(all_z, axis=0)  # (n_frames, latent_channels, 8, 8)

    # symmetric=True (see check_rollout._padded_bounds' own docstring):
    # a channel's activation can be positive or negative with no
    # particular reason to expect one side dominates, so a symmetric
    # +-M scale keeps both signs equally resolvable rather than an
    # asymmetric scale (appropriate for real_delta, which check_rollout
    # uses this same helper for) potentially collapsing one side.
    channel_bounds = [
        _padded_bounds(all_z_arr[:, c], factor=1.1, symmetric=True)
        for c in range(latent_channels)
    ]
    # Cross-frame std per channel -- a quick, at-a-glance way to spot
    # channels that vary a lot across inputs vs staying near-constant.
    # NOT the same question as importance below: a channel can vary a
    # lot without the decoder actually relying on that variation, or
    # vary little while still being load-bearing (e.g. a near-constant
    # offset the decoder needs to get right). See rank_channel_importance.
    channel_std = all_z_arr.std(axis=(0, 2, 3))
    print("\nChannel activity (std across shown frames' 8x8 maps, sorted descending):")
    for c in np.argsort(channel_std)[::-1]:
        print(f"  channel {c:2d}: std={channel_std[c]:.4f}")

    channel_importance = None
    if not skip_importance and test_dataset is not None:
        channel_importance = rank_channel_importance(
            ae, test_dataset, device, n_samples=n_importance_samples, seed=seed,
        )
        print(f"\nChannel importance (mean recon-loss increase from zero-ablation, "
              f"n={min(n_importance_samples, len(test_dataset))} test frames, sorted descending):")
        for c in np.argsort(channel_importance)[::-1]:
            print(f"  channel {c:2d}: delta_loss={_format_small(channel_importance[c])}")
    elif not skip_importance:
        print("\n(skipping channel importance ranking -- no test set available)")

    n_cols = latent_channels + 1
    n_rows = len(frames)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, ((run_dir, step), x_raw, z) in enumerate(zip(frames, all_x, all_z)):
        state_scale = max(abs(x_raw.min()), abs(x_raw.max()), 0.1)
        axes[row, 0].imshow(x_raw, cmap="RdBu", vmin=-state_scale, vmax=state_scale)
        axes[row, 0].set_ylabel(f"{run_dir.name}:{step}", fontsize=8)
        if row == 0:
            axes[row, 0].set_title("input", fontsize=9)

        for c in range(latent_channels):
            vmin, vmax = channel_bounds[c]
            axes[row, c + 1].imshow(z[c], cmap="RdBu", vmin=vmin, vmax=vmax,
                                     interpolation="nearest")
            if row == 0:
                title = f"ch{c}\nstd={channel_std[c]:.3f}"
                if channel_importance is not None:
                    title += f"\nimp={_format_small(channel_importance[c])}"
                axes[row, c + 1].set_title(title, fontsize=8)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"\nSaved latent channel visualization to {output_path} "
          f"({len(frames)} frames, {latent_channels} channels)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--fixed-frames", action="append", default=None,
            help="'run_dir:step', repeatable. Default: n_frames random frames from "
                 "the checkpoint's own held-out test_dirs.")
    parser.add_argument("--n-frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--n-importance-samples", type=int, default=200,
            help="test frames used for ablation-based channel importance ranking "
                 "(separate from --n-frames, which only controls the figure)")
    parser.add_argument("--skip-importance", action="store_true",
            help="skip ablation importance ranking (it's cheap, but this is here "
                 "for a quick run without it)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    check_latent_channels(
        ae_checkpoint_path=args.ae_checkpoint, fixed_frames=args.fixed_frames,
        n_frames=args.n_frames, seed=args.seed, min_step=args.min_step,
        n_importance_samples=args.n_importance_samples, skip_importance=args.skip_importance,
        output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()
