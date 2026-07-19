"""
Stage 2 latent validation: interpolation, using stats_head(z) directly
throughout -- comparing stats_head's output to ITSELF (via a different z),
never mixing it with raw statistics.csv values.

This sidesteps a real bug an earlier version of this script had: stats_head
is trained (via StatsLoss) to output ALREADY-NORMALIZED values -- i.e.
stats_head(z) directly predicts (true_raw - mean) / std, not a raw
physical value. Normalizing stats_head's output a SECOND time (treating
it as if it were still raw) produces a large, spurious, roughly-constant
error dominated by the raw scale of `mean`, unrelated to actual prediction
quality. Comparing stats_head(a) to stats_head(b) directly -- both already
in the same, consistent normalized units -- has no such risk.

Given three real, CONSECUTIVE kept steps (t1 < t2 < t3) from one run:
  x1 = x(t1), x2 = x(t2), x3 = x(t3)
  z1 = E(x1), z2 = E(x2), z3 = E(x3)

alpha is weighted by REAL elapsed time, not the midpoint -- since
save_steps are irregular, alpha=0.5 would compare against the wrong
physical instant whenever the two gaps differ:
    alpha = (t2 - t1) / (t3 - t1)
    z_tilde = (1-alpha)*z1 + alpha*z3

Metric: ||stats(z_tilde) - stats(z2)|| / ||stats(z2)||, expected close to
0 for a sensible representation -- a large deviation means interpolation
doesn't track the real trajectory, independent of overall AE
reconstruction accuracy (both z_tilde and z2 are compared through the
identical stats_head map, so this isolates representation GEOMETRY, not
accuracy).

check_interpolation() is importable -- see main.py, which calls it
automatically after stage 2 AND after stage 4/5 (skipped gracefully
there if the ancestor AE has no stats_head at all) with the checkpoint
path it already has in hand. The CLI below is for standalone use.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_interpolation --latent-channels 8
    python -m evaluation.check_interpolation --latent-channels 8 \
        --fixed-triples "../../datasets/64x64/T800_n050_s79:100000:120000:150000" ...
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.autoencoder import Autoencoder, EncoderDecoderPair, MultiStreamAutoencoder
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_streams import (
    cross_check_stream_configs_against_state_dict, resolve_stream_configs_from_checkpoint_config,
)
from training.stats_head import StatsHead
from utils import load_datasets as load
from utils.naming import ae_checkpoint_name

# GENERAL POLICY (matches training/train_refinement.py's own
# _PYTHON_ROOT): every default checkpoint/output path is built from
# THIS anchor, never from a bare relative string like "../../output/...".
# Relative strings resolve against the process's CWD at invocation
# time, which silently differs across bare CLI, `python -m`, and being
# imported and called from another module (e.g. main.py or train_ae.py
# calling this function) -- exactly the recurring "output ended up in
# the wrong place" bug hit repeatedly on this project. Path(__file__)
# is anchored to THIS FILE's own on-disk location instead, which is
# invariant regardless of how/from-where the process was launched.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/check_interpolation.py -> python/


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def parse_fixed_triple(s: str) -> tuple[Path, int, int, int]:
    """
    'run_dir:t1:t2:t3' -> (Path(run_dir), t1, t2, t3).

    NOT a naive split(':') -- same reasoning as
    check_rollout.parse_fixed_window / check_latent_channels.
    parse_fixed_frame (this had the identical bug independently, not
    fixed alongside them originally): run_dir itself can contain a
    colon on Windows (e.g. 'D:\\work\\...\\T800_n050_s79'), which a
    naive split(':') misreads as if it were one of the step numbers --
    split from the right instead, taking the final 3 colon-separated
    parts as the step numbers regardless of how many colons run_dir
    itself contains.
    """
    parts = s.split(":")
    if len(parts) < 4:
        raise ValueError(f"expected 'run_dir:t1:t2:t3', got '{s}'")
    *path_parts, t1_str, t2_str, t3_str = parts
    if not path_parts or not all(_is_int(x) for x in (t1_str, t2_str, t3_str)):
        raise ValueError(f"expected 'run_dir:t1:t2:t3', got '{s}'")
    run_dir = Path(":".join(path_parts))
    return run_dir, int(t1_str), int(t2_str), int(t3_str)


def find_all_triples(test_dirs: list[Path], min_step: int) -> list[tuple[Path, int, int, int]]:
    """Every consecutive (t1,t2,t3) triple of kept steps across all test_dirs."""
    triples = []
    for run_dir in test_dirs:
        metadata = load.read_metadata(run_dir / "metadata.txt")
        check = load.check_snapshots_saved(run_dir, metadata)
        bad_steps = set(check["missing"]) | set(check["bad_size"])
        kept = [s for s in metadata.save_steps if s not in bad_steps and s >= min_step]
        for i in range(len(kept) - 2):
            triples.append((run_dir, kept[i], kept[i + 1], kept[i + 2]))
    return triples


def check_interpolation(
    checkpoint_path: Path, n_samples: int | None = None, min_step: int = 4000,
    seed: int = 0, fixed_triples: list[str] | None = None,
    output_path: Path | None = None, device: str | None = None,
) -> Path:
    """Saves the interpolation-consistency histogram and returns its path."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if output_path is None:
        output_path = (_PYTHON_ROOT.parent / "output" / "interpolation_check_png"
                       / f"{checkpoint_path.stem}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, config={model_cfg}")

    stats_config = checkpoint.get("stats_config")
    if stats_config is None:
        raise ValueError(f"{checkpoint_path} has no stats_head (trained with --stats-weight 0)")

    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, checkpoint["model_state"],
    )
    recon_stream = stream_configs[recon_stream_name]
    decoder_for_stream = model_cfg.get("decoder_for_stream")
    is_flat_checkpoint = any(k.startswith("encoder.") for k in checkpoint["model_state"])
    if is_flat_checkpoint:
        encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                           base_channels=model_cfg["base_channels"], stream_configs=stream_configs)
        decoder = Decoder(output_size=model_cfg["size"], out_channels=1,
                           base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size)
        ae = EncoderDecoderPair(encoder, decoder, stream_name=recon_stream_name,
                                 mode=recon_stream.mode).to(device)
    elif len(stream_configs) == 1:
        ae = Autoencoder(
            size=model_cfg["size"], channels=1,
            base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
            latent_spatial_size=recon_stream.spatial_size,
        ).to(device)
    elif decoder_for_stream is None:
        encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                           base_channels=model_cfg["base_channels"], stream_configs=stream_configs)
        decoder = Decoder(output_size=model_cfg["size"], out_channels=1,
                           base_channels=model_cfg["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size)
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                     stream_configs=stream_configs).to(device)
    else:
        # Stage 1b's own format: a SEPARATE decoder per stream (see
        # autoencoder.py's MultiStreamAutoencoder).
        encoder = Encoder(input_size=model_cfg["size"], in_channels=1,
                           base_channels=model_cfg["base_channels"], stream_configs=stream_configs)
        decoders = {}
        for stream_name, decoder_key in decoder_for_stream.items():
            stream_cfg = stream_configs[stream_name]
            decoders[decoder_key] = Decoder(
                output_size=model_cfg["size"], out_channels=1,
                base_channels=model_cfg["base_channels"], latent_channels=stream_cfg.channels,
                latent_spatial_size=stream_cfg.spatial_size,
            )
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders=decoders,
                                     stream_configs=stream_configs,
                                     decoder_for_stream=decoder_for_stream).to(device)
    ae.load_state_dict(checkpoint["model_state"])
    ae.eval()
    ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]

    stats_head = StatsHead(
        latent_channels=recon_stream.channels, stat_names=stats_config["stat_names"],
        latent_spatial=recon_stream.spatial_size,
    ).to(device)
    stats_head.load_state_dict(checkpoint["stats_head_state"])
    stats_head.eval()
    print(f"stats_head covers: {stats_config['stat_names']}")

    if fixed_triples:
        triples = [parse_fixed_triple(s) for s in fixed_triples]
    else:
        test_dirs = checkpoint.get("test_dirs") or []
        if not test_dirs:
            raise ValueError(f"{checkpoint_path} has no saved test_dirs")
        test_dirs = [Path(d) for d in test_dirs]
        triples = find_all_triples(test_dirs, min_step)
        if not triples:
            raise ValueError("No consecutive (t1,t2,t3) triples found in test_dirs")
        if n_samples is not None and n_samples < len(triples):
            generator = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(triples), generator=generator)[:n_samples].tolist()
            triples = [triples[i] for i in idx]
        print(f"Using {len(triples)} (t1,t2,t3) triples from {len(test_dirs)} test dirs")

    nx = ny = model_cfg["size"]
    n = len(triples)
    relative_error = np.zeros(n)
    elapsed_span = np.zeros(n)

    with torch.no_grad():
        for i, (run_dir, t1, t2, t3) in enumerate(triples):
            alpha = (t2 - t1) / (t3 - t1)
            metadata = load.read_metadata(run_dir / "metadata.txt")
            elapsed_span[i] = (t3 - t1) * metadata.dt

            x1_np = load.read_phi_half(run_dir / load.snapshot_filename(t1), nx, ny)
            x2_np = load.read_phi_half(run_dir / load.snapshot_filename(t2), nx, ny)
            x3_np = load.read_phi_half(run_dir / load.snapshot_filename(t3), nx, ny)
            x1 = torch.from_numpy(x1_np).unsqueeze(0).unsqueeze(0).to(device)
            x2 = torch.from_numpy(x2_np).unsqueeze(0).unsqueeze(0).to(device)
            x3 = torch.from_numpy(x3_np).unsqueeze(0).unsqueeze(0).to(device)

            z1 = ae_encoder(x1)[recon_stream_name]
            z2 = ae_encoder(x2)[recon_stream_name]
            z3 = ae_encoder(x3)[recon_stream_name]
            z_tilde = (1 - alpha) * z1 + alpha * z3

            stats_z_tilde = stats_head(z_tilde)
            stats_z2 = stats_head(z2)

            diff_norm = (stats_z_tilde - stats_z2).norm(dim=1).item()
            target_norm = stats_z2.norm(dim=1).item()

            relative_error[i] = diff_norm / target_norm if target_norm > 1e-3 else np.nan

    valid = ~np.isnan(relative_error)
    n_dropped = n - valid.sum()
    if n_dropped:
        print(f"\nDropped {n_dropped}/{n} triples with ||z2|| too close to 0 (<1e-3) -- "
              f"a relative error isn't meaningful when the reference value itself is ~0")
    rel_err = relative_error[valid]
    span_valid = elapsed_span[valid]

    print(f"\n||stats(z_tilde) - stats(z2)|| / ||stats(z2)||, across {valid.sum()} triples:")
    print(f"  mean:   {rel_err.mean():.4f}")
    print(f"  median: {np.median(rel_err):.4f}   (mean >> median signals outlier-driven skew)")
    print(f"  std:    {rel_err.std():.4f}")
    if len(rel_err) >= 4:
        corr = np.corrcoef(np.log(span_valid), rel_err)[0, 1]
        print(f"  corr(log elapsed_span, error) = {corr:.3f} "
              f"(positive -> worse over longer spans)")

    lo, hi = 0.0, np.percentile(rel_err, 99.5)
    if hi <= lo:
        hi = lo + 1e-6  # degenerate (near-)zero-width range -- give the histogram a minimal, non-zero span
    n_outside = int((rel_err > hi).sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(rel_err, bins=np.linspace(lo, hi, 40), color="tab:blue")
    ax.axvline(0.0, color="red", linestyle="--", label="0 = perfect")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("||stats(z_tilde) - stats(z2)|| / ||stats(z2)||")
    ax.set_ylabel("count")
    ax.set_title(f"Distribution across {len(rel_err)} triples "
                 f"(up to 99.5th percentile shown, {n_outside} outliers excluded from view)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved plot to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True,
                         help="grid size (square only) -- config.txt is never read")
    parser.add_argument("--latent-channels", type=int, default=None,
            help="required -- not a sweep parameter, so config.txt has no value for this")
    parser.add_argument("--stats-weight", type=float, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
            help="direct path override, instead of --size/--latent-channels/--stats-weight")
    parser.add_argument("--n-samples", type=int, default=None,
            help="default: use EVERY available (t1,t2,t3) triple in the test set")
    parser.add_argument("--min-step", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0,
                         help="only matters if --n-samples subsamples; ignored otherwise")
    parser.add_argument("--fixed-triples", type=str, nargs="+", default=None,
                         help="'run_dir:t1:t2:t3' (repeatable) for reproducible comparison")
    parser.add_argument("--output", type=Path, default=None,
            help="default: <repo root>/output/interpolation_check_png/<checkpoint name>.png")
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

    if args.output is None:
        args.output = (_PYTHON_ROOT.parent / "output" / "interpolation_check_png"
                       / f"{args.checkpoint.stem}.png")

    check_interpolation(
        checkpoint_path=args.checkpoint, n_samples=args.n_samples, min_step=args.min_step,
        seed=args.seed, fixed_triples=args.fixed_triples, output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
