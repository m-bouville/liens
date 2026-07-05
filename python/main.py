"""
Orchestrates the full LIENS pipeline: stage 2 (autoencoder) -> stage 3
(interpolation-consistency fine-tuning) -> stage 4 (latent dynamics
surrogate), each stopping via early-stopping on its own val_ema rather
than a guessed epoch count, and each stage resuming from the previous
one's checkpoint.

Also retains the original sweep-status scan (--scan-only), useful as a
quick sanity check before committing to a long training run.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m main --latent-channels 8
    python main.py --scan-only --config config.txt --base ../datasets
"""

import argparse
from pathlib import Path

from training.train_ae import train_autoencoder, train_stage3
from training.train_lds import train_lds
from utils import load_datasets as load


def check_sweep_status(config_path: Path, base_path: Path) -> None:
    """Scan a sweep and report COMPLETE/INCOMPLETE/missing run directories."""
    config = load.read_config(config_path)
    dirs = load.enumerate_run_dirs(config, base=base_path)

    n_complete = n_incomplete = n_missing = 0
    for d in dirs:
        if not d.exists():
            n_missing += 1
            continue
        if load.is_complete(d):
            n_complete += 1
            print(f"COMPLETE    {d}")
            metadata = load.read_metadata(d / "metadata.txt")
            check = load.check_snapshots_saved(d, metadata)
            if check["missing"] or check["bad_size"]:
                print(f"            ! {len(check['missing'])} missing, "
                      f"{len(check['bad_size'])} bad size")
        else:
            n_incomplete += 1
            print(f"INCOMPLETE  {d}")

    print(f"\n{len(dirs)} possible runs in sweep -> "
          f"{n_complete} complete, {n_incomplete} incomplete, {n_missing} missing (ignored)")


def run_pipeline(
    config_path: Path, base_path: Path, latent_channels: int,
    stage2_epochs: int = 300, stage3_epochs: int = 300, stage4_epochs: int = 100,
    early_stopping_patience: int = 10, interp_weight: float = 1.0,
    n_rollout_steps: int = 1, seed: int = 0, device: str | None = None,
) -> Path:
    """Runs stage 2 -> 3 -> 4 in sequence. Returns the final (stage 4) checkpoint path."""
    print("=" * 70)
    print("STAGE 2: training autoencoder")
    print("=" * 70)
    stage2_checkpoint = train_autoencoder(
        config_path=config_path, base_path=base_path, latent_channels=latent_channels,
        epochs=stage2_epochs, early_stopping_patience=early_stopping_patience,
        seed=seed, device=device,
    )
    print(f"\nStage 2 complete: {stage2_checkpoint}\n")

    print("=" * 70)
    print("STAGE 3: interpolation-consistency fine-tuning")
    print("=" * 70)
    stage3_checkpoint = train_stage3(
        config_path=config_path, base_path=base_path, resume_from=stage2_checkpoint,
        interp_weight=interp_weight, epochs=stage3_epochs,
        early_stopping_patience=early_stopping_patience, seed=seed, device=device,
    )
    print(f"\nStage 3 complete: {stage3_checkpoint}\n")

    print("=" * 70)
    print("STAGE 4: latent dynamics surrogate (frozen encoder from stage 3)")
    print("=" * 70)
    stage4_checkpoint = train_lds(
        config_path=config_path, base_path=base_path, ae_checkpoint_path=stage3_checkpoint,
        epochs=stage4_epochs, n_rollout_steps=n_rollout_steps,
        early_stopping_patience=early_stopping_patience, seed=seed, device=device,
    )
    print(f"\nStage 4 complete: {stage4_checkpoint}\n")

    return stage4_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("../config.txt"))
    parser.add_argument("--base", type=Path, default=Path("../datasets"))
    parser.add_argument("--scan-only", action="store_true",
                         help="just report sweep status and exit, don't train anything")
    parser.add_argument("--latent-channels", type=int, default=8,
                         help="required unless --scan-only")
    parser.add_argument("--stage2-epochs", type=int, default=50,
                         help="upper bound; early stopping usually stops sooner")
    parser.add_argument("--stage3-epochs", type=int, default=50)
    parser.add_argument("--stage4-epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                         help="epochs without val_ema improvement before stopping a "
                              "stage -- the principled 'is this stage done' signal, "
                              "replacing a guessed epoch count")
    parser.add_argument("--interp-weight", type=float, default=1.0,
                         help="stage 3's L_interp weight -- expect this to need the same "
                              "kind of tuning effort stats_weight did")
    parser.add_argument("--n-rollout-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.scan_only:
        check_sweep_status(args.config, args.base)
        return

    if args.latent_channels is None:
        raise ValueError("--latent-channels is required unless --scan-only")

    check_sweep_status(args.config, args.base)
    print()

    run_pipeline(
        config_path=args.config, base_path=args.base, latent_channels=args.latent_channels,
        stage2_epochs=args.stage2_epochs, stage3_epochs=args.stage3_epochs,
        stage4_epochs=args.stage4_epochs, early_stopping_patience=args.early_stopping_patience,
        interp_weight=args.interp_weight, n_rollout_steps=args.n_rollout_steps,
        seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    main()
