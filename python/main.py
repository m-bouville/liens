"""
Orchestrates the LIENS pipeline from stage-PARAMETERS FILES, one per run
configuration, rather than a long list of CLI flags.

Stage numbering (see docs/neural_nets.md):
    0. Generate phase-field simulations (C++, not this pipeline)
    1. Train autoencoder (E, D) on individual microstructures -- real
       space, L_recon + lambda1*L_stats
    2. Latent-space validation: interpolation-consistency fine-tuning
       (E, D trainable; stats_head FROZEN, used only as a fixed
       measuring instrument for L_interp) -- L_recon + lambda1*L_interp,
       optionally + stats0_weight*L_stats0 as an anchor (gradient flows
       into E only, since stats_head is frozen) against the encoder
       drifting away from anything stats_head can still interpret --
       see train_stage2()'s docstring for why this is needed at all.
    3. Latent Dynamics Surrogate (f) on a FROZEN encoder -- latent space,
       L_rollout (or L_1step, identical at n_rollout_steps=1). Can run
       as ONE phase ('# Stage 3') or as a two-phase CURRICULUM
       ('# Stage 3a' + '# Stage 3b'): 3a trains at n_rollout_steps=1
       (stable, fast), 3b resumes from 3a's checkpoint at the target
       (larger) n_rollout_steps -- avoids the instability of jumping
       straight to multi-step rollout training from scratch. Both
       conventions run the exact same train_lds() function underneath
       (see run_lds_stage() below) -- 3a/3b is a params-file-level
       naming choice, not a separate code path.
    4. Encoder refinement -- E/D trainable (D frozen as an L_recon
       tether), f_theta trainable, on RAW PIXEL windows (encoding
       happens fresh every epoch, unlike stage 3's frozen-encoder
       cache) -- L_rollout (primary) + small L_recon anchor + optional
       L_stats. Takes TWO ancestors (stage 2's checkpoint for E/D/
       stats_head, stage 3's for f), unlike every earlier stage's
       single predecessor -- see
       checkpoint_components.assemble_joint_checkpoint.
    5. End-to-end refinement -- same mechanism as stage 4 (one shared
       train_refinement() function), but D also trainable and L_recon
       primary (L_rollout becomes the small anchor instead). Continues
       stage 4's own joint checkpoint (resume_from), not a fresh
       assembly from stage 2/3 -- see
       checkpoint_components.load_joint_refinement_checkpoint.

config.txt is NOT read by this module AT ALL -- not even for grid size.
Nx/Ny come directly from the stage-parameters file, and dataset directory
enumeration reads each size's OWN datasets/<nx>x<ny>/metadata.txt (see
utils/load_datasets.py's read_sweep_metadata), co-located with the
actual dataset rather than a separate, possibly-stale or
describing-a-different-sweep config.txt. This is what lets one
invocation process several resolutions (e.g. 64x64 and 128x128) while a
completely unrelated sweep (e.g. 256x256) is being generated in C++ at
the same time.

STAGE-PARAMETERS FILE FORMAT, e.g. 64x64_no_stage2.txt:

    Nx = 64                          # required -- config.txt is never read
    Ny = 64
    base = ../datasets               # optional; falls back to --base if omitted
    # Stage 1
    min_step      = 4000            # inline '#' comments are stripped
    min_stdev_phi = 0.01
    stats0_weight  = 0.01
    latent_channels = 8
    force = True                    # always retrain, even if a match already exists
    # Stage 2
    epochs = 0                      # 0 = SKIP this stage entirely
    min_stdev_phi = same            # inherit from the nearest preceding stage/global
    n_frozen_stages = 2             # optional: freeze E/D's outer layers (see train_stage2)
    stats0_weight = 0.01             # optional: L_stats0 anchor weight (0 = no anchor)
    # Stage 3
    epochs = 50
    patience = 10                   # renamed to early_stopping_patience internally

Stage 3's two-phase curriculum uses '# Stage 3a' + '# Stage 3b' INSTEAD
of '# Stage 3' (giving both, or neither -- 3a alone is just a warmup,
not a usable final result; mixing bare 3 with 3a/3b is an error too):

    # Stage 3a
    n_rollout_steps = 1
    epochs = 100
    # Stage 3b
    n_rollout_steps = 6             # the actual target
    epochs = 300
    min_step = same                 # 'same' looks at 3a first, then Stage 2, then Stage 1

Any key not specially handled (Nx, Ny, base, force, epochs=0 skip,
'same' inheritance) is passed straight through as a keyword argument to
that stage's training function (train_autoencoder/train_stage2/
train_lds), after best-effort str->int/float/bool conversion.

CACHING: two independent checks before training any stage, in order:
  1. Does THIS params file's own expected output
     (python/checkpoints/stage<N>/<stem>-stage<N>.pt) already exist?
  2. If not, does the PARAMETER REGISTRY (registry-stage<N>.csv, in the
     same directory) record any OTHER checkpoint -- under any name --
     whose recorded parameters for this stage match EXACTLY? This is
     what lets, e.g., 64x64_no_stage2.txt's stage 1 reuse 64x64.txt's
     stage-1 output without the two files needing matching names, as
     long as their stage-1 parameters are identical.
Either hit skips training. A stage's force=True skips BOTH checks and
always retrains (overwriting its own expected filename, with a warning
if that file already existed). Stage 3a/3b use this identical machinery
under their own names (checkpoints/stage3a/, checkpoints/stage3b/) --
not because reusing a 3a checkpoint saves much time (stage 3 is fast
enough that retraining it isn't a real cost), just because it's the
same generic per-stage mechanism every other stage already uses, so
there's no reason to special-case it away.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m main params/64x64_no_stage2.txt
    python -m main params/64x64.txt params/128x128.txt
"""

import argparse
import gc
from pathlib import Path

import torch


from orchestration.paths import _PYTHON_ROOT
from orchestration.pipeline import run_from_params_file
from orchestration.sweep_status import check_sweep_status


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("params_files", type=Path, nargs="*",
                         default=[_PYTHON_ROOT / "params" / "64x64.txt"],
                         help="one or more stage-parameters file paths -- the pipeline runs "
                              "once per file, in order given. Defaults to "
                              "params/64x64.txt if none given (e.g. hitting Run in an IDE "
                              "like Spyder, which doesn't pass CLI args) -- pass real paths "
                              "explicitly for anything else.")
    parser.add_argument("--base", type=Path, default=_PYTHON_ROOT.parent / "datasets",
                         help="fallback dataset base path, only used if a params file "
                              "doesn't specify its own 'base = ...' in its global section")
    parser.add_argument("--scan-only", action="store_true",
                         help="just report sweep status (scanning every size directory's "
                              "own metadata.txt under --base) and exit, don't train anything")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.scan_only:
        check_sweep_status(args.base)
        return


    for params_path in args.params_files:
        print("#" * 70)
        print(f"# {params_path}")
        print("#" * 70)
        run_from_params_file(params_path, default_base=args.base, device=args.device)
        print()


    if torch.cuda.is_available():
        # clear VRAM
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



if __name__ == "__main__":
    main()
