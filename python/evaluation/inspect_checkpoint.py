"""
Print the config a checkpoint was trained with -- provenance without rerunning.

Reads the metadata blocks a checkpoint stores (`config`, `lds_config`,
`data_config`, `stage2_config`, and top-level scalars like epoch and
val_loss) and prints them flat, so "what was z0_noise_scale on this run?"
is one command instead of an interactive torch.load session.

Loads with `weights_only=True` and NEVER touches the weight tensors -- it is
safe to run on any .pt and does not construct a model.

CRUCIAL LIMITATION, printed in the output too: some training parameters are
used but NOT saved. For stage 3, `lr` and `use_dt_decade_weights` are logged
to the run's .log at startup but do not appear in the checkpoint at all, so
they cannot be recovered from the .pt -- only from the log. The tool flags
the fields it KNOWS are meaningful-but-unsaved rather than letting their
absence pass silently, which is exactly the trap that made one run's
provenance ambiguous.
"""

import argparse
from pathlib import Path

import torch

# (stage, field) pairs that matter for reproducing a run but are NOT written
# to the checkpoint -- so their absence is a fact to report, not an oversight.
# Recover these from the run's .log (startup parameter block) instead.
_KNOWN_UNSAVED = {
    "lr": "logged at startup; recover from the .log, not the .pt",
    "use_dt_decade_weights": "logged at startup; recover from the .log",
    "one_step_weight": "stage-3 loss weight; logged, not saved",
    "grad_clip": "logged, not saved",
}

# The metadata blocks a checkpoint may carry, in the order worth reading.
_CONFIG_BLOCKS = ("config", "lds_config", "stage2_config", "data_config")
# Top-level scalars worth surfacing before the nested blocks.
_SCALARS = ("epoch", "val_loss", "val_loss_ema", "n_substeps", "alpha",
            "z1_resync", "ae_checkpoint", "lds_checkpoint", "resumed_from")


def inspect_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    print(f"checkpoint: {path}")
    print("=" * 70)

    present_scalars = [(k, ckpt[k]) for k in _SCALARS if k in ckpt]
    if present_scalars:
        print("top-level:")
        for k, v in present_scalars:
            print(f"  {k:24} {v}")

    seen_keys = set()
    for block in _CONFIG_BLOCKS:
        if block not in ckpt or not isinstance(ckpt[block], dict):
            continue
        print(f"\n{block}:")
        for k, v in ckpt[block].items():
            # stream_configs is a nested dict; print it compactly rather than
            # as an unreadable one-liner
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk:22} {vv}")
            else:
                print(f"  {k:24} {v}")
            seen_keys.add(k)

    # Report meaningful-but-unsaved fields explicitly. Absence here is the
    # answer to a provenance question, so it must be stated, not implied.
    unsaved_relevant = [(k, why) for k, why in _KNOWN_UNSAVED.items()
                        if k not in seen_keys and k not in ckpt]
    if unsaved_relevant:
        print("\nNOT in this checkpoint (used at training time, recover from the .log):")
        for k, why in unsaved_relevant:
            print(f"  {k:24} -- {why}")

    # Anything else at top level, so nothing is hidden.
    other = sorted(set(ckpt) - set(_SCALARS) - set(_CONFIG_BLOCKS)
                   - {"model_state_dict", "optimizer_state_dict", "state_dict",
                      "f_theta_state_dict", "test_dirs"})
    if other:
        print("\nother top-level keys (not shown above):")
        print("  " + ", ".join(other))
    if "test_dirs" in ckpt:
        print(f"\ntest_dirs: {len(ckpt['test_dirs'])} run(s) recorded")

    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--key", type=str, default=None,
                        help="print ONLY this field's value (searches all "
                             "config blocks); exit 1 if absent. For scripting.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.key is not None:
        for container in (ckpt, *(ckpt[b] for b in _CONFIG_BLOCKS
                                  if isinstance(ckpt.get(b), dict))):
            if args.key in container:
                print(container[args.key])
                return
        # distinguish "known unsaved" from "unknown" in the exit message
        if args.key in _KNOWN_UNSAVED:
            raise SystemExit(
                f"'{args.key}' is not saved in checkpoints "
                f"({_KNOWN_UNSAVED[args.key]})")
        raise SystemExit(f"'{args.key}' not found in {args.checkpoint}")

    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
