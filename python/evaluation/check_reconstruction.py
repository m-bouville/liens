"""
Load a trained autoencoder checkpoint and visually compare its held-out
TEST set (saved in the checkpoint by train_ae.py, never touched during
training or checkpoint selection) against their reconstructions.

If the checkpoint has a second decodable stream beyond the
reconstruction ("autoencoder"-mode) one -- see the project's own C0/C1
design doc and models/latent_streams.py -- this ALSO shows that
stream's PREDICTED delta (D0(z0(t)+z1(t)*dt) - D0(z0(t))) compared
against the REAL delta (D0(z0(t+dt)) - D0(z0(t))) -- both sides
decoded through D0, z1 never sent to a decoder at all (D1 is never
trained past Stage 1b -- see the inline comment below): real state |
predicted state | error | real derivative | predicted derivative |
error, six columns in ONE figure rather than two separate three-column
ones, since state and derivative are closely related and worth reading
side by side. Falls back to the original three-column layout when
there's no second stream to show (every checkpoint saved before this
redesign, and any single-stream run since).

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
from models.autoencoder import Autoencoder, EncoderDecoderPair, MultiStreamAutoencoder
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
    min_stdev_phi: float | None = None,
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
    decoder_for_stream = model_cfg.get("decoder_for_stream")
    # Whether this checkpoint is actually flat ("encoder."/"decoder.",
    # stage 4/5's own EncoderDecoderPair -- see model_assembly.py's
    # build_models_from_components) or nested ("encoders.shared."/
    # "decoders.*", stage 1/1b/2's own MultiStreamAutoencoder) is a
    # property of the ACTUAL saved keys, not of how many streams
    # stream_configs happens to list -- a stage 4/5 checkpoint keeps
    # the full, inherited multi-stream stream_configs in its own
    # config even though model_assembly.py only ever saves a flat,
    # single-pathway EncoderDecoderPair (it only needs the recon
    # stream), regardless of how many streams that config lists.
    is_flat_checkpoint = any(k.startswith("encoder.") for k in checkpoint["model_state"])
    if is_flat_checkpoint and deriv_stream_name is not None:
        print(f"NOTE: this checkpoint's own ancestor had a '{deriv_stream_name}' stream, but "
              f"this checkpoint itself only kept '{recon_stream_name}''s own decoder (stage 4/5 "
              f"only ever needs the reconstruction stream's own pathway, see "
              f"model_assembly.py's build_models_from_components) -- no derivative panel is "
              f"possible from this checkpoint alone.")
        deriv_stream_name = None

    if is_flat_checkpoint:
        # Mirrors model_assembly.py's own construction exactly (the
        # SAME code that produced this checkpoint) -- encoder built
        # with the FULL stream_configs (every bottleneck, even ones
        # with no decoder here), wrapped in a single-pathway
        # EncoderDecoderPair for just the reconstruction stream.
        encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                           base_channels=model_cfg["base_channels"], stream_configs=stream_configs).to(device)
        decoder = Decoder(output_size=model_cfg["size"], out_channels=1,
                           base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size).to(device)
        ae = EncoderDecoderPair(encoder, decoder, stream_name=recon_stream_name,
                                 mode=recon_stream.mode).to(device)
    elif len(stream_configs) == 1:
        ae = Autoencoder(
            size=model_cfg["size"], channels=1,
            base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
            latent_spatial_size=recon_stream.spatial_size,
        ).to(device)
        encoder, decoder = ae.encoder, ae.decoder
    elif decoder_for_stream is None:
        # Stage 2's own format: every stream shares ONE decoder.
        encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                           base_channels=model_cfg["base_channels"], stream_configs=stream_configs).to(device)
        decoder = Decoder(output_size=model_cfg["size"], out_channels=1,
                           base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size).to(device)
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                     stream_configs=stream_configs).to(device)
    else:
        # Stage 1b's own format: a SEPARATE decoder per stream (see
        # autoencoder.py's MultiStreamAutoencoder -- decoder_for_stream
        # maps each stream name to which decoder key its own pathway
        # reads from). One Decoder built per UNIQUE decoder key
        # referenced, sized from whichever stream actually uses it
        # (today, always exactly one stream per decoder key, but this
        # doesn't assume that -- the last stream seen for a given key
        # wins if more than one ever mapped to the same decoder, same
        # as any dict-building loop).
        encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                           base_channels=model_cfg["base_channels"], stream_configs=stream_configs).to(device)
        decoders = {}
        for stream_name, decoder_key in decoder_for_stream.items():
            stream_cfg = stream_configs[stream_name]
            decoders[decoder_key] = Decoder(
                output_size=model_cfg["size"], out_channels=1,
                base_channels=model_cfg["base_channels"], latent_channels=stream_cfg.channels,
                latent_spatial_size=stream_cfg.spatial_size,
            ).to(device)
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders=decoders,
                                     stream_configs=stream_configs,
                                     decoder_for_stream=decoder_for_stream).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()

    def _pathway_scale(stream_name):
        return ae.pathways[stream_name].log_output_scale if hasattr(ae, "pathways") else ae.log_output_scale

    def _pathway_decoder(stream_name):
        return ae.pathways[stream_name].decoder if hasattr(ae, "pathways") else ae.decoder

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
        test_dirs, encoder=None, window_length=2, min_step=min_step, min_stdev_phi=min_stdev_phi,
    )
    if len(dataset) == 0:
        raise ValueError(f"No consecutive pairs found in the checkpoint's {len(test_dirs)} "
                          f"test_dirs (after min_step={min_step}, min_stdev_phi={min_stdev_phi} "
                          f"filtering)")

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
            x_recon = _pathway_decoder(recon_stream_name)(z[recon_stream_name]) * torch.exp(_pathway_scale(recon_stream_name))
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
                # D0(z0(t+dt)) - D0(z0(t)), NOT the raw pixel
                # difference (x_next - x_t): z0(t+dt) is encoded and
                # decoded through D0 here for the SAME reason
                # pred_deriv (below) never touches D1 -- so that BOTH
                # sides of this comparison share the same D0
                # reconstruction-error component, rather than one side
                # being raw ground truth and the other being a decoded
                # approximation. Without this, the two panels have
                # genuinely different natural scales (D0's own
                # reconstruction noise inflates one side but not the
                # other), which is what made real_deriv look nearly
                # blank under a shared color scale sized for
                # pred_deriv's own, larger range -- not a sign real_deriv
                # itself was somehow wrong.
                z_next = encoder(x_next)
                x_next_recon = (_pathway_decoder(recon_stream_name)(z_next[recon_stream_name])
                                 * torch.exp(_pathway_scale(recon_stream_name)))
                real_deriv_np = ((x_next_recon - x_recon) / dt)[0, 0].cpu().numpy()

                # D0(z0 + z1*dt) - D0(z0), NOT D1(z1): D1 is never
                # trained past Stage 1b (Stage 2's default
                # recon1_weight=0 means pure L_deriv training, D1
                # itself gets no gradient there -- see train_ae.py's
                # own docstring), so a direct D1(z1) decode goes stale
                # and produces meaningless checkerboard noise once
                # Stage 2 has actually run. z1's real, trained role is
                # as a RATE to be added to z0 and decoded through D0 --
                # exactly how the rollout mechanism itself uses it
                # (LatentDynamics: z(t+dt) = z(t) + dt*g_theta(z(t))),
                # and D0 is the decoder that's actually kept reliable
                # throughout. x_recon (already computed above) IS
                # D0(z0) -- reused directly, not recomputed.
                z0 = z[recon_stream_name]
                z1 = z[deriv_stream_name]
                x_pred_next = (_pathway_decoder(recon_stream_name)(z0 + z1 * dt)
                                * torch.exp(_pathway_scale(recon_stream_name)))
                pred_deriv_np = ((x_pred_next - x_recon) / dt)[0, 0].cpu().numpy()

                deriv_err_np = pred_deriv_np - real_deriv_np

                # ONE shared scale, now genuinely meaningful: both
                # panels are D0-decoded deltas built from the same
                # baseline (D0(z0(t))), so they're directly comparable
                # in a way the old raw-pixel-vs-D0-decode pairing
                # never was. A real error panel is included below for
                # the same reason -- see the docstring note above.
                deriv_scale = max(_scale(real_deriv_np, floor=1e-6),
                                   _scale(pred_deriv_np, floor=1e-6))
                deriv_err_scale = _scale(deriv_err_np, floor=1e-6)

                axes[row, 3].imshow(real_deriv_np, cmap="RdBu", vmin=-deriv_scale, vmax=deriv_scale)
                axes[row, 3].set_title(f"real derivative (D0(z0(t+dt))-D0(z0(t)), dt={dt:.1f}, "
                                        f"scale=+-{_format_small(deriv_scale, precision=3)})" if row == 0
                                        else f"scale=+-{_format_small(deriv_scale, precision=3)}")
                axes[row, 4].imshow(pred_deriv_np, cmap="RdBu", vmin=-deriv_scale, vmax=deriv_scale)
                axes[row, 4].set_title("predicted derivative\n(D0(z0+z1dt) - D0(z0))" if row == 0 else "")
                im_deriv_err = axes[row, 5].imshow(deriv_err_np, cmap="RdBu",
                                                     vmin=-deriv_err_scale, vmax=deriv_err_scale)
                axes[row, 5].set_title(f"deriv error (scale=+-{_format_small(deriv_err_scale, precision=3)})"
                                        if row == 0 else f"scale=+-{_format_small(deriv_err_scale, precision=3)}")
                fig.colorbar(im_deriv_err, ax=axes[row, 5], fraction=0.046)

            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
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
    parser.add_argument("--min-stdev-phi", type=float, default=None)
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
        min_step=args.min_step, min_stdev_phi=args.min_stdev_phi,
        output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()
