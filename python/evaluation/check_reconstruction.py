"""
Load a trained autoencoder checkpoint and visually compare its held-out
TEST set (saved in the checkpoint by train_ae.py, never touched during
training or checkpoint selection) against their reconstructions.

If the checkpoint has a second decodable stream beyond the
reconstruction ("autoencoder"-mode) one -- see the project's own C0/C1
design doc and models/latent_streams.py -- this ALSO shows that
stream's decode compared against the real finite-difference time
derivative, (x(t+dt)-x(t))/dt: real state | predicted state | error |
real derivative | predicted derivative | error, six columns in ONE
figure rather than two separate three-column ones, since state and
derivative are closely related and worth reading side by side. Falls
back to the original three-column layout when there's no second
stream to show (every checkpoint saved before this redesign, and any
single-stream run since).

check_reconstruction() is importable -- see main.py, which calls it
automatically after stages 1, 2, and 4/5 with the checkpoint path it
already has in hand (never stage 3, which freezes the encoder/decoder
entirely -- there's nothing new to check there). The CLI below is for
standalone use.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_reconstruction --latent-channels 4
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from evaluation.check_rollout import _format_small
from models.autoencoder import Autoencoder, MultiStreamAutoencoder
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_streams import (
    LatentStreamMode, cross_check_stream_configs_against_state_dict,
    resolve_stream_configs_from_checkpoint_config,
)
from training.datasets import MicrostructureEvolutionDataset
from training.losses import ReconLoss
from utils.naming import ae_checkpoint_name

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


def check_reconstruction(
    checkpoint_path: Path, n_samples: int = 6, seed: int = 0, min_step: int = 0,
    output_path: Path | None = None, device: str | None = None,
) -> Path:
    """Saves a visual comparison figure and returns its path."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if output_path is None:
        output_path = (_PYTHON_ROOT.parent / "output" / "reconstruction_check_png"
                       / f"{checkpoint_path.stem}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f}, config={model_cfg}")

    test_dirs = checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(
            f"{checkpoint_path} has no saved test_dirs -- it was likely trained with "
            f"--test-fraction 0, or with an older version of train_ae.py."
        )
    test_dirs = [Path(d) for d in test_dirs]

    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, checkpoint["model_state"],
    )

    # Any OTHER decodable stream (not the recon one, not pure_latent)
    # gets shown as the derivative panel -- picked generically by
    # ROLE, not by assuming it's literally named "deriv": the params-
    # file syntax lets someone name streams however they like.
    other_decodable = [name for name, cfg in stream_configs.items()
                        if name != recon_stream_name and cfg.mode != LatentStreamMode.PURE_LATENT]
    deriv_stream_name = other_decodable[0] if other_decodable else None
    if len(other_decodable) > 1:
        print(f"NOTE: {len(other_decodable)} decodable streams besides '{recon_stream_name}' "
              f"({other_decodable}) -- showing only '{deriv_stream_name}'; this diagnostic "
              f"only has a derivative panel for one.")

    recon_stream = stream_configs[recon_stream_name]
    if len(stream_configs) == 1:
        ae = Autoencoder(
            size=model_cfg["size"], channels=1,
            base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
            latent_spatial_size=recon_stream.spatial_size,
        ).to(device)
        encoder, decoder = ae.encoder, ae.decoder
    else:
        encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                           base_channels=model_cfg["base_channels"], stream_configs=stream_configs).to(device)
        decoder = Decoder(output_size=model_cfg["size"], out_channels=1,
                           base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size).to(device)
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                     stream_configs=stream_configs).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()

    def _pathway_scale(stream_name):
        return ae.pathways[stream_name].log_output_scale if hasattr(ae, "pathways") else ae.log_output_scale

    # window_length=2 (a real consecutive PAIR, not a lone snapshot):
    # needed regardless of whether a derivative panel ends up shown,
    # since resolving that requires reading model_cfg first -- used
    # uniformly rather than branching to MicrostructureSnapshotDataset
    # for the no-second-stream case, so there's one data path, not two.
    # Deliberately unaugmented and encoder=None (raw pixels): we want
    # real frames and a real elapsed dt for the finite-difference
    # target, not rotated/translated synthetic views or a frozen-
    # encoder's own cached latents. Uses the checkpoint's own saved
    # test_dirs, so this is guaranteed to be the exact same held-out
    # set that training never touched.
    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=None, window_length=2, min_step=min_step, min_stdev_phi=None,
    )
    if len(dataset) == 0:
        raise ValueError(f"No consecutive pairs found in the checkpoint's {len(test_dirs)} "
                          f"test_dirs (after min_step={min_step} filtering)")

    generator = torch.Generator().manual_seed(seed)
    n_samples = min(n_samples, len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:n_samples].tolist()

    recon_loss = ReconLoss()
    n_cols = 6 if deriv_stream_name is not None else 3

    fig, axes = plt.subplots(len(indices), n_cols, figsize=(3 * n_cols, 3 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]  # keep 2D indexing uniform for a single sample

    def _scale(arr, floor=0.1):
        # Auto-scale symmetric around 0, from the actual data range of
        # THIS sample/panel, floored (avoids amplifying near-noise
        # samples/panels into looking like real signal).
        return max(abs(arr.min()), abs(arr.max()), floor)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            window, dt_window, _theta = dataset[idx]
            x_t = window[0:1].to(device)     # (1, 1, H, W)
            x_next = window[1:2].to(device)
            dt = dt_window[0].item()

            z = encoder(x_t)
            x_recon = decoder(z[recon_stream_name]) * torch.exp(_pathway_scale(recon_stream_name))
            loss = recon_loss(x_recon, x_t).item()

            x_np = x_t[0, 0].cpu().numpy()
            x_recon_np = x_recon[0, 0].cpu().numpy()
            diff_np = x_recon_np - x_np

            scale = _scale(x_np)
            diff_scale = _scale(diff_np, floor=1e-6)

            axes[row, 0].imshow(x_np, cmap="RdBu", vmin=-scale, vmax=scale)
            axes[row, 0].set_title(f"real state (idx={idx}, scale=+-{scale:.3f})" if row == 0
                                    else f"scale=+-{scale:.3f}")
            axes[row, 1].imshow(x_recon_np, cmap="RdBu", vmin=-scale, vmax=scale)
            axes[row, 1].set_title(f"predicted state (loss={loss:.4f})" if row == 0 else
                                    f"loss={loss:.4f}")
            im_diff = axes[row, 2].imshow(diff_np, cmap="RdBu", vmin=-diff_scale, vmax=diff_scale)
            axes[row, 2].set_title(f"error (scale=+-{diff_scale:.3f})" if row == 0
                                    else f"scale=+-{diff_scale:.3f}")
            fig.colorbar(im_diff, ax=axes[row, 2], fraction=0.046)

            if deriv_stream_name is not None:
                real_deriv_np = ((x_next - x_t) / dt)[0, 0].cpu().numpy()
                pred_deriv = decoder(z[deriv_stream_name]) * torch.exp(_pathway_scale(deriv_stream_name))
                pred_deriv_np = pred_deriv[0, 0].cpu().numpy()
                deriv_diff_np = pred_deriv_np - real_deriv_np
                deriv_loss = recon_loss(pred_deriv, (x_next - x_t) / dt).item()

                # Real and predicted get SEPARATE, independent scales
                # (not one shared scale, unlike the state columns) --
                # while the deriv stream is untrained (this scoped
                # step never gives it gradient -- see train_ae.py's own
                # docstring on that), a random-weight decode of it has
                # no reason to land anywhere near the real signal's
                # magnitude, and forcing them onto one shared scale
                # just saturates whichever one is smaller into a flat
                # block of color instead of showing its own real
                # structure. Once training actually touches this
                # stream, the two should naturally converge in scale;
                # until then, independent scales are more informative,
                # not less.
                real_deriv_scale = _scale(real_deriv_np, floor=1e-6)
                pred_deriv_scale = _scale(pred_deriv_np, floor=1e-6)
                deriv_diff_scale = _scale(deriv_diff_np, floor=1e-6)

                axes[row, 3].imshow(real_deriv_np, cmap="RdBu", vmin=-real_deriv_scale, vmax=real_deriv_scale)
                axes[row, 3].set_title(f"real derivative ('{deriv_stream_name}', dt={dt:.1f}, "
                                        f"scale=+-{_format_small(real_deriv_scale)})" if row == 0
                                        else f"scale=+-{_format_small(real_deriv_scale)}")
                axes[row, 4].imshow(pred_deriv_np, cmap="RdBu", vmin=-pred_deriv_scale, vmax=pred_deriv_scale)
                axes[row, 4].set_title(f"predicted derivative (loss={_format_small(deriv_loss)}, "
                                        f"scale=+-{_format_small(pred_deriv_scale)})" if row == 0
                                        else f"loss={_format_small(deriv_loss)}, "
                                             f"scale=+-{_format_small(pred_deriv_scale)}")
                im_deriv_diff = axes[row, 5].imshow(deriv_diff_np, cmap="RdBu",
                                                      vmin=-deriv_diff_scale, vmax=deriv_diff_scale)
                axes[row, 5].set_title(f"error (scale=+-{_format_small(deriv_diff_scale)})" if row == 0
                                        else f"scale=+-{_format_small(deriv_diff_scale)}")
                fig.colorbar(im_deriv_diff, ax=axes[row, 5], fraction=0.046)

            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"Saved comparison figure to {output_path} ({n_samples} samples from "
          f"{len(test_dirs)} held-out test dirs"
          f"{', with derivative panel' if deriv_stream_name is not None else ''})")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True,
                         help="grid size (square only) -- config.txt is never read")
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--stats-weight", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
            help="direct path override, instead of --size/--latent-channels/--stats-weight")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint is None:
        if args.latent_channels is None or args.stats_weight is None:
            raise ValueError(
                "Provide either --checkpoint directly, or both --latent-channels and "
                "--stats-weight so the expected path can be reconstructed."
            )
        name = ae_checkpoint_name(args.size, args.latent_channels, args.stats_weight)
        args.checkpoint = _PYTHON_ROOT / "checkpoints" / "stage2" / f"{name}.pt"
        print(f"Reconstructed checkpoint path: {args.checkpoint}")

    check_reconstruction(
        checkpoint_path=args.checkpoint, n_samples=args.n_samples, seed=args.seed,
        min_step=args.min_step, output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()
