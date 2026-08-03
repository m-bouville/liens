"""
CLI for porting a trained stage-1 checkpoint to a larger grid size.

    python -m training.port_checkpoint \\
        --checkpoint checkpoints/stage1/64x64-stage1.pt --to-size 128

Writes a new stage-1-shaped checkpoint that `train_autoencoder(resume_from=...)`
accepts directly, so the port needs no new pipeline stage: run stage 1 at the
new size as usual, then stage 2 and stage 3 exactly as they run natively.

What this does NOT decide for you
----------------------------------
It does not fine-tune. The written checkpoint has ~75% of its parameters at
fresh random init (every rung transfers exactly 25% -- the doubling-per-stage
channel rule makes each new deepest block pair about 3x everything below it
combined), so it is a STARTING POINT for stage 1 and nothing else. Its
`val_loss` is deliberately +inf rather than the source's, so no downstream
"is this better than before" comparison can accidentally treat it as trained.

The BatchNorm re-estimation is optional here and skipped by default, because
doing it properly needs the new sweep's own data. If `--base-path` is given,
this script re-estimates from it; otherwise it warns and leaves the old running
statistics in place, which makes the first val_loss look worse than the model
actually is. Stage 1's own first epochs will fix them either way -- the
re-estimation matters when you want to MEASURE the ported model before training
it.
"""
import argparse
from training.datasets import report_save_step_distribution
from pathlib import Path

import torch

from orchestration.paths import _STAGE_DIRS
from orchestration.stage_params import _backup_before_overwrite
from training.checkpoint_criterion import atomic_torch_save
from training.rescale_checkpoint import (
    describe_rescale, extract_stage1_checkpoint, reestimate_batchnorm_statistics,
    rescale_checkpoint_to_size,
)

_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/training/X.py -> python/


def _default_output_path(checkpoint_path: Path, to_size: int) -> Path:
    """`checkpoints/stage1/{to_size}x{to_size}-ported.pt`.

    Named and placed for what the file IS, not for where its source came from.
    A port always produces a SINGLE-STREAM, STAGE-1-SHAPED checkpoint whose only
    valid use is `train_autoencoder(resume_from=...)` -- porting from stage 2
    keeps that stage's trunk but still drops every non-recon stream.

    Deriving the name from the source instead put a stage-1 input at
    `checkpoints/stage2/128x128-stage2.pt`, which is actively dangerous rather
    than merely untidy: it is single-stream, so
    `extend_state_checkpoint_with_deriv_stream` would ACCEPT it, and stage 2
    would happily build a deriv head on a trunk that is 75% random init and has
    never been trained at this size. Nothing downstream would flag that.

    "-ported" rather than "-stage1" so it cannot be confused with a stage-1
    checkpoint that was actually trained; the two are byte-compatible and
    scientifically nothing alike.
    """
    return _STAGE_DIRS[1] / f"{to_size}x{to_size}-ported{checkpoint_path.suffix}"


def _batches_from_sweep(base_path: Path, size: int, batch_size: int, n_batches: int,
                         device: torch.device):
    """Yield input batches from the new sweep, for BatchNorm re-estimation.

    Imported lazily so that the common path -- port without re-estimation --
    does not pay for the dataset machinery or require a sweep to exist at all.
    """
    from training.datasets import MicrostructureDataset  # noqa: PLC0415
    from utils import load_datasets as load  # noqa: PLC0415

    run_dirs = load.enumerate_run_dirs_from_metadata(base_path, size, size)

    # This diagnostic discovers runs directly, bypassing

    # complete_run_dirs -- so it must ask for the save-step report itself.

    # A sweep with runs regenerated to pass tau_down is a MIXTURE, and every

    # count below pools populations evolved to different times.

    report_save_step_distribution(run_dirs)
    dataset = MicrostructureDataset(run_dirs, size=size, include_stats=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        yield batch


def port_checkpoint(
    checkpoint_path: Path, to_size: int, output_path: Path | None = None,
    base_path: Path | None = None, batch_size: int = 32, n_batches: int = 20,
    device: str | torch.device | None = None, dry_run: bool = False,
    keep_trunk_from_multi_stream: bool = False,
) -> Path:
    """Returns the path written (or the path that WOULD be written, if dry_run)."""
    device = torch.device(device or "cpu")
    if output_path is None:
        output_path = _default_output_path(checkpoint_path, to_size)

    rescaled = rescale_checkpoint_to_size(
        checkpoint_path, to_size=to_size, device=device,
        keep_trunk_from_multi_stream=keep_trunk_from_multi_stream)
    print(describe_rescale(rescaled))

    if base_path is not None:
        print(f"\nre-estimating BatchNorm running statistics from {base_path} "
              f"({n_batches} batches of {batch_size} at {to_size}x{to_size})...")
        theta_zeros = None

        def theta_for(batch):
            nonlocal theta_zeros
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            if theta_zeros is None or theta_zeros.shape[0] != x.shape[0]:
                theta_zeros = torch.zeros(x.shape[0], 1)
            return theta_zeros

        seen = reestimate_batchnorm_statistics(
            rescaled.encoder,
            _batches_from_sweep(base_path, to_size, batch_size, n_batches, device),
            device=device, theta_for=theta_for,
        )
        print(f"  re-estimated over {seen} batches")
        # The encoder's parameters were mutated in place, so the checkpoint
        # dict built before this point holds the PRE-re-estimation tensors for
        # any buffer that torch.save would have copied. Rebuild from the live
        # module rather than trusting them to be views.
        model_state = {f"encoder.{k}": v for k, v in rescaled.encoder.state_dict().items()}
        model_state.update({f"decoder.{k}": v for k, v in rescaled.decoder.state_dict().items()})
        rescaled.checkpoint["model_state"] = model_state
    else:
        print("\nBatchNorm running statistics NOT re-estimated (no --base-path). They were "
              "measured on the source size, so the ported model's val_loss will look worse than "
              "it is until stage 1 updates them. Fine if you are about to train; not fine if you "
              "are about to measure.")

    if dry_run:
        print(f"\ndry run -- would write {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Same protection the pipeline gives an explicit resume_from override: the
    # port writes into checkpoints/stage1/ alongside real trained checkpoints,
    # and a second port (say after fixing the source, or at a different size)
    # would otherwise silently destroy the first. _backup_before_overwrite
    # names the copy from the file's own mtime, so re-porting an unchanged
    # source does not leave redundant archives.
    if output_path.exists():
        # _backup_before_overwrite prints its own NOTE naming both paths, so
        # this deliberately adds nothing -- two messages for one action reads
        # like two backups happened.
        _backup_before_overwrite(output_path)
    atomic_torch_save(rescaled.checkpoint, output_path)
    print(f"\nwrote {output_path}")
    print(f"next: run STAGE 1 at {to_size}x{to_size} with resume_from={output_path.name}, "
          f"then stages 2 and 3 as normal.")
    print("      This file is a stage-1 INPUT, not a trained checkpoint: ~75% of its parameters "
          "are fresh random init. It is single-stream, so stage 2 would accept it directly -- "
          "do not let it.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="a stage-1 (single-stream) checkpoint to port")
    parser.add_argument("--to-size", type=int, default=None,
                         help="new grid size; must be the source size * 2^k. Omit it together "
                              "with --extract-stage1 to convert in place at the same size")
    parser.add_argument("--extract-stage1", action="store_true",
                         help="no rescale: turn a stage-2 checkpoint back into a stage-1-shaped "
                              "one AT THE SAME SIZE, keeping every trained weight the recon "
                              "pathway owns (stats_head included) and dropping only the deriv "
                              "stream -- for restarting stage 1 from the autoencoder training "
                              "stage 2 has already done")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-path", type=Path, default=None,
                         help="the NEW size's dataset root; if given, BatchNorm running "
                              "statistics are re-estimated from it before saving")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--from-stage2", action="store_true",
                         help="accept a stage-2+ (multi-stream) checkpoint and port its shared "
                              "trunk, which has had training the stage-1 checkpoint never saw. "
                              "The deriv bottleneck and f_theta are discarded either way -- they "
                              "read a feature space the new deepest block reinvents from scratch")
    parser.add_argument("--dry-run", action="store_true",
                         help="report what would happen and write nothing")
    args = parser.parse_args()

    if args.extract_stage1:
        source = args.checkpoint
        out = args.output or _STAGE_DIRS[1] / f"{source.stem}-as-stage1{source.suffix}"
        checkpoint = extract_stage1_checkpoint(source, device=args.device)
        if args.dry_run:
            print(f"dry run -- would write {out}")
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            _backup_before_overwrite(out)
        atomic_torch_save(checkpoint, out)
        print(f"\nwrote {out}")
        print(f"next: run STAGE 1 with resume_from={out.name}. Its epoch/val_loss are reset, so "
              f"the run starts a fresh save history rather than inheriting stage 2's.")
        return

    if args.to_size is None:
        parser.error("--to-size is required unless --extract-stage1 is given")
    port_checkpoint(
        checkpoint_path=args.checkpoint, to_size=args.to_size, output_path=args.output,
        base_path=args.base_path, batch_size=args.batch_size, n_batches=args.n_batches,
        device=args.device, dry_run=args.dry_run,
        keep_trunk_from_multi_stream=args.from_stage2,
    )


if __name__ == "__main__":
    main()
