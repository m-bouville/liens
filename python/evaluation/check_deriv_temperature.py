"""
Stage 2 direct diagnostic: is z1's own error, measured against ITS OWN
training target, temperature-dependent -- the direct test of the
hypothesis raised by evaluation/check_parameter_dependence.py's own
T<0.9 vs T>=0.9 split (there, eps/eps'/z0_ddot all flipped sign and grew
5-9x in magnitude at T>=0.9, while f_theta's own bias relative to the
true curvature stayed comparatively stable across the split -- pointing
at z1's error itself, not f_theta, as the more likely root cause).

That earlier finding was INFERRED algebraically, through f_theta's own
downstream residual (see check_parameter_dependence.py's own module
docstring for the full derivation). This script instead measures z1's
error DIRECTLY: train_stage2.py's own L_deriv target is

    target_deriv(t) = (z0(t+dt) - z0(t)) / dt

-- i.e. EXACTLY the same eps/eps' decomposition applies here too, one
level up the pipeline, with NO f_theta or A(dt^3) term involved at all
(there's no z0_ddot/curvature concept at this level -- z1 vs
target_deriv is a first-order-only comparison):

    z1(t) - target_deriv(t) = eps/dt + eps'

Fits this 2-term model (via the SAME robust_polynomial_fit machinery
check_parameter_dependence.py uses) for ALL DATA, T<0.9, and T>=0.9
separately -- if THIS eps/eps' ALSO flips sign/grows at T>=0.9, that
confirms the root cause is upstream of f_theta entirely, in z1's own
training (stage 2) or in the encoder's own behavior near/above the
critical temperature this project's own metadata centers theta on.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_deriv_temperature --stage2-checkpoint checkpoints/stage2/64x64.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation._fits import robust_polynomial_fit
from training.checkpoint_components import build_ae_from_checkpoint
from training.datasets import MicrostructureEvolutionDataset
from training.losses import centered_deriv_target
from utils import load_datasets as load

# _latent_eval.py's own _load_models_and_dataset is NOT reused here,
# deliberately: it is coupled to the LDS/stage-3 conversion path
# (ensure_lds_checkpoint, f_theta, LatentDynamics) that check_parameter_
# dependence.py needs but this script never touches -- this diagnostic
# loads a stage-2 AE checkpoint directly, nothing else. Pulling that
# function in would reintroduce exactly the "heavier LDS-checkpoint-
# loading machinery" this module's own docstring already explains it
# was built to avoid. Only the numeric fitter (robust_polynomial_fit,
# _fits.py -- numpy-only, no torch/checkpoint dependency at all) is
# shared; see its own module docstring for why sharing IT specifically
# matters (it fixes a real eps/eps'-naming inconsistency between this
# script and check_parameter_dependence.py).

# Same anchor rationale as check_interpolation.py/check_parameter_dependence.py's
# own _PYTHON_ROOT (see either's docstring) -- never a bare relative path.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/



def check_deriv_temperature(
    stage2_checkpoint_path: Path, min_step: int | None = None, min_stdev_phi: float | None = None,
    min_passing_steps: int | None = None, deriv_target_centered: bool = False,
    device: str | None = None,
) -> dict:
    """
    Measures z1(t) - target_deriv(t), per window, then fits eps/dt +
    eps' (see this module's own docstring) for ALL DATA, T<0.9, and
    T>=0.9 separately.

    deriv_target_centered (default False): target_deriv(t) =
    (z0(t+dt)-z0(t))/dt -- EXACTLY train_stage2.py's own one-sided
    L_deriv target when this stage-2 checkpoint was trained without
    its own deriv_target_centered=True. If True instead, uses the
    second-order-accurate centered target from a (t-dt_minus, t,
    t+dt_plus) window -- see centered_deriv_target's own docstring
    (losses.py) for the derivation. Comparing eps/eps' between the two
    modes on the SAME checkpoint is a direct attribution test: eps'
    (the dt-independent term) collapsing under the centered target
    would confirm it was mostly z0's O(dt) truncation bias, not a
    genuine z1 modelling limitation; eps (the 1/dt term) is not
    expected to move much either way, since the centered target isn't
    designed to address it.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ae, encoder, checkpoint, stream_configs, recon_stream_name = build_ae_from_checkpoint(
        stage2_checkpoint_path, device,
    )
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, config={checkpoint['config']}")
    if "deriv" not in stream_configs:
        raise ValueError(
            f"{stage2_checkpoint_path} has no 'deriv' stream (stream_configs={list(stream_configs)}) -- "
            f"this diagnostic needs z1 (the deriv stream) to exist at all, which means a checkpoint "
            f"trained at or after stage 2, not stage 1/1b."
        )
    test_dirs = checkpoint.get("test_dirs")
    if not test_dirs:
        raise ValueError(f"{stage2_checkpoint_path} has no saved test_dirs")
    test_dirs = [Path(d) for d in test_dirs]

    # encode_both_streams=True gets z0 AND z1 cached together, in one
    # pass, with the SAME bounded-memory streaming/parallel-read
    # machinery MicrostructureEvolutionDataset already has (see its own
    # docstring) -- reused here rather than a bespoke encode loop.
    # window_length=3 (not 2) under deriv_target_centered -- same
    # reasoning as train_stage2.py's own switch: gives (before, middle,
    # after) triples, and MicrostructureEvolutionDataset's own window
    # index construction already only yields these for interior kept
    # steps, no separate edge-case filtering needed here either.
    # Filtering defaults to the checkpoint's OWN saved data_config, so
    # this evaluates the same window population training actually used;
    # an explicitly-passed value still overrides it. Same convention (and
    # same wording below) as _latent_eval.py's own stage-3 path.
    #
    # Without this, the CLI defaults (min_step=None -> 0,
    # min_stdev_phi=None) applied NO filtering at all -- silently a much
    # larger, different population than training saw. min_std_deriv
    # matters most: it is applied only in stage 2, and on real 64x64 data
    # discards tens of thousands of windows, so omitting it here was not
    # a marginal difference.
    data_config = checkpoint.get("data_config")
    if data_config is None:
        print("  WARNING: this checkpoint predates data_config being saved -- falling back to "
              "min_step=0, min_stdev_phi=None, min_passing_steps=None, min_std_deriv=None. "
              "That is almost certainly NOT what it was trained with, so the window population "
              "below won't match training's own; pass the values explicitly to be sure.")
        data_config = {}
    min_step = min_step if min_step is not None else (data_config.get("min_step") or 0)
    min_stdev_phi = min_stdev_phi if min_stdev_phi is not None else data_config.get("min_stdev_phi")
    min_passing_steps = (min_passing_steps if min_passing_steps is not None
                          else data_config.get("min_passing_steps"))
    print(f"min_step={min_step}  min_stdev_phi={min_stdev_phi}  "
          f"min_passing_steps={min_passing_steps} "
          f"(from checkpoint's own data_config unless overridden above)")
    # min_std_deriv is deliberately NOT re-applied, and cannot be:
    # MicrostructureEvolutionDataset rejects it outright in cached-latent
    # mode (its own constructor guard), because it filters on the RAW
    # PIXEL derivative's spatial std, which has no defined meaning
    # against a cached latent. train_stage2 can apply it only because it
    # trains E/D and therefore runs in raw-pixel mode; this diagnostic
    # uses a FROZEN encoder, so the two modes are structurally
    # different. Reported rather than silently dropped, since it means
    # the window population below genuinely differs from training's own
    # -- on real 64x64 data that filter removed tens of thousands of
    # windows, so the gap is not marginal.
    trained_min_std_deriv = data_config.get("min_std_deriv")
    if trained_min_std_deriv is not None:
        print(f"  NOTE: this checkpoint was TRAINED with min_std_deriv="
              f"{trained_min_std_deriv}, which cannot be reproduced here -- it filters on the "
              f"raw-pixel derivative, undefined against the cached latents this diagnostic uses. "
              f"The windows below therefore INCLUDE near-degenerate-derivative ones that training "
              f"itself excluded; treat any comparison against training's own numbers accordingly.")

    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=encoder, device=device, window_length=3 if deriv_target_centered else 2,
        min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
        encode_both_streams=True,
    )
    print(f"Evaluating {len(dataset)} "
          f"{'(t-dt_minus, t, t+dt_plus)' if deriv_target_centered else '(t, t+dt)'} windows...")

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    dts, temperatures, residuals_signed = [], [], []
    metadata_cache: dict[Path, object] = {}

    idx = 0
    with torch.no_grad():
        for window, window_deriv, dt_window, theta in loader:
            batch_size = window.shape[0]
            window = window.to(device)
            window_deriv = window_deriv.to(device)
            dt_window = dt_window.to(device)

            if deriv_target_centered:
                z0_before, z0_t, z0_after = window[:, 0], window[:, 1], window[:, 2]
                # z1's own frame is the MIDDLE one here (window_deriv[:,1]),
                # not window_deriv[:,0] -- z1(t) is compared against a
                # target built AROUND t, so it must be z1 evaluated AT t,
                # matching train_stage2.py's own step() (there too, z1_t
                # comes from x_t = the middle frame under this same mode).
                z1_t = window_deriv[:, 1]
                dt_minus = dt_window[:, 0]
                dt_plus = dt_window[:, 1]
                target_deriv = centered_deriv_target(
                    z0_before, z0_t, z0_after,
                    dt_minus.view(-1, 1, 1, 1), dt_plus.view(-1, 1, 1, 1),
                )
                # dt_minus+dt_plus (the TOTAL span this target draws on)
                # for the eps/dt fit below -- same convention as
                # train_stage2.py's own dt_for_weighting.
                dt = dt_minus + dt_plus
            else:
                z0_t = window[:, 0]
                z0_next = window[:, 1]
                z1_t = window_deriv[:, 0]
                dt = dt_window[:, 0]
                # EXACT match to train_stage2's own one-sided L_deriv
                # target -- see this module's own docstring.
                target_deriv = (z0_next - z0_t) / dt.view(-1, 1, 1, 1)

            residual = (z1_t - target_deriv).flatten(start_dim=1).mean(dim=1)  # (B,) signed, per-window

            residuals_signed.extend(residual.cpu().tolist())
            dts.extend(dt.cpu().tolist())

            for i in range(batch_size):
                run_dir, steps = dataset.window_info(idx)
                if run_dir not in metadata_cache:
                    metadata_cache[run_dir] = load.read_metadata(run_dir / "metadata.txt")
                temperatures.append(metadata_cache[run_dir].temperature)
                idx += 1

    dts = np.array(dts)
    temperatures = np.array(temperatures)
    residuals_signed = np.array(residuals_signed)

    basis_funcs = [lambda dt: 1.0 / dt, lambda dt: np.ones_like(dt)]

    def _fit_and_report(mask, label):
        n = mask.sum()
        if n < 20:
            print(f"\n  Skipping [{label}] -- only {n} windows, too few for a meaningful fit.")
            return None
        coefs, stderr = robust_polynomial_fit(dts[mask], residuals_signed[mask], basis_funcs)
        eps, eps_prime = coefs
        eps_se, eps_prime_se = stderr
        print(f"\n  [{label}] (n={n}):")
        print(f"    eps  = {eps: .6e}  (stderr {eps_se:.2e}, {abs(eps)/eps_se if eps_se > 0 else float('nan'):.1f} sigma)")
        print(f"    eps' = {eps_prime: .6e}  (stderr {eps_prime_se:.2e}, "
              f"{abs(eps_prime)/eps_prime_se if eps_prime_se > 0 else float('nan'):.1f} sigma)")
        return {"eps": eps, "eps_stderr": eps_se, "eps_prime": eps_prime, "eps_prime_stderr": eps_prime_se, "n": int(n)}

    print("\n" + "=" * 70)
    print(f"z1 direct error decomposition: z1(t) - target_deriv(t) = eps/dt + eps'  [{stage2_checkpoint_path.name}]")
    print("=" * 70)
    result_all = _fit_and_report(np.ones_like(dts, dtype=bool), "ALL DATA")
    t_mask = temperatures < 0.9
    result_lo = _fit_and_report(t_mask, "T < 0.9")
    result_hi = _fit_and_report(~t_mask, "T >= 0.9")
    print("=" * 70)

    return {"all": result_all, "T<0.9": result_lo, "T>=0.9": result_hi,
            "dts": dts, "temperatures": temperatures, "residuals_signed": residuals_signed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--min-stdev-phi", type=float, default=None)
    parser.add_argument("--min-passing-steps", type=int, default=None)
    parser.add_argument("--deriv-target-centered", action="store_true",
                         help="Use the second-order-accurate centered L_deriv target "
                              "instead of the default one-sided one -- see this module's "
                              "own docstring and check_deriv_temperature's own docstring.")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    check_deriv_temperature(
        stage2_checkpoint_path=args.stage2_checkpoint, min_step=args.min_step,
        min_stdev_phi=args.min_stdev_phi, min_passing_steps=args.min_passing_steps,
        deriv_target_centered=args.deriv_target_centered, device=args.device,
    )


if __name__ == "__main__":
    main()
