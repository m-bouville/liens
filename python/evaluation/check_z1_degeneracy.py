"""
Is the deriv stream z1 an independent derivative, or a rescaled copy of z0?

WHY THIS EXISTS. In the ground-truth dz0 figure the second-derivative row
(d2z0/dt2 = dz1/dt) and the first-derivative row (dz0/dt) fall with the SAME
slope (~1/dt each, measured -1.05 and -1.00) and their ratio is constant at
about 1.3e-3 across the whole dt range. A genuine second derivative should
fall as 1/dt^2, so two readings compete:

  (a) DEGENERACY: z1 ~= c * z0. Then dz1 = c * dz0 identically, the two rows
      differ by a constant factor by construction, and the second-derivative
      row is measuring the state, not the curvature. z1's causal/actual ratio
      of ~0.12 makes this a live possibility.

  (b) SATURATION: both increments reach their own ensemble scale beyond the
      decorrelation time, and the ratio of those two scales is a constant.
      No degeneracy -- just two saturated chords.

They differ in one measurable place: the per-window correlation between z1
and z0, and the residual left after the best-fit rescaling. Under (a) the
correlation is near 1 and the residual small; under (b) the correlation is
low and the constant ratio is a coincidence of scales.

The same two quantities are reported for the INCREMENTS (dz1 vs dz0), which
is what the figure's rows are actually built from -- z1 could track z0
without the increments doing so, or the reverse.

Usage (from python/):

    python -m evaluation.check_z1_degeneracy \\
        checkpoints/stage2/128x128-stage2.pt --size 128 --n-windows 512
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from training.checkpoint_components import build_ae_from_checkpoint
from training.datasets import MicrostructureEvolutionDataset
from utils import load_datasets as load

_PYTHON_ROOT = Path(__file__).resolve().parent.parent


def _corr_and_scale(u: torch.Tensor, v: torch.Tensor) -> tuple:
    """Per-sample (correlation, best-fit scale, residual fraction) for u ~ c*v.

    c is the least-squares scale <u,v>/<v,v>, and the residual fraction is
    ||u - c v|| / ||u|| -- what is left of u once everything proportional to
    v has been removed. Correlation alone is not enough: it is invariant to
    the scale, and the claim under test is specifically u = c*v.
    """
    u_flat = u.flatten(start_dim=1).double()
    v_flat = v.flatten(start_dim=1).double()
    u_c = u_flat - u_flat.mean(dim=1, keepdim=True)
    v_c = v_flat - v_flat.mean(dim=1, keepdim=True)
    denom = (u_c.norm(dim=1) * v_c.norm(dim=1)).clamp_min(1e-30)
    corr = (u_c * v_c).sum(dim=1) / denom
    scale = ((u_flat * v_flat).sum(dim=1)
             / (v_flat * v_flat).sum(dim=1).clamp_min(1e-30))
    residual = ((u_flat - scale[:, None] * v_flat).norm(dim=1)
                / u_flat.norm(dim=1).clamp_min(1e-30))
    return corr, scale, residual


def check_z1_degeneracy(lds_checkpoint_path: Path, base_path: Path | None = None,
                         size: int | None = None, device: str | None = None,
                         n_windows: int = 512, max_dt: float | None = None,
                         latent_cache_dir: Path | str | None = None) -> dict:
    # NO f_theta. This diagnostic is entirely about the ENCODER, so it loads
    # the AE and builds the dataset directly rather than going through
    # _load_ae_f_theta_and_dataset -- which, given a stage-2 checkpoint,
    # converts it into a stage-3 one and therefore demands a base_path it has
    # no use for here. An AE-family checkpoint is in fact the NATURAL input:
    # z1 is the encoder's own deriv stream.
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(lds_checkpoint_path, map_location=resolved_device,
                             weights_only=True)
    ae_path = Path(checkpoint.get("ae_checkpoint") or lds_checkpoint_path)
    _ae, ae_encoder, _ae_ck, _cfgs, _recon = build_ae_from_checkpoint(
        ae_path, resolved_device)

    data_config = checkpoint.get("data_config") or {}
    test_dirs = checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(
            f"{lds_checkpoint_path} has no saved test_dirs, so there is no "
            f"population to measure. Pass a checkpoint that recorded its own "
            f"split.")
    test_dirs = load.validate_run_dirs(
        [Path(d) for d in test_dirs], source=str(lds_checkpoint_path),
        min_stdev_phi=data_config.get("min_stdev_phi"))
    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=ae_encoder, device=resolved_device,
        window_length=2,
        max_dt=max_dt if max_dt is not None else data_config.get("max_dt"),
        min_step=data_config.get("min_step", 0),
        min_stdev_phi=data_config.get("min_stdev_phi"),
        # WITHOUT THIS the dataset yields (window, dt, theta) and never
        # encodes the deriv stream at all -- z1, the entire subject of this
        # diagnostic, would simply be absent.
        encode_both_streams=True,
        latent_cache_dir=latent_cache_dir,
    )
    print(f"encoder from {ae_path.name}; {len(dataset)} windows available")

    take = min(int(n_windows), len(dataset))
    idx = np.linspace(0, len(dataset) - 1, take).round().astype(int)
    print(f"check_z1_degeneracy: {take} windows.\n"
          f"  state:      is z1 ~= c*z0 ?\n"
          f"  increments: is dz1 ~= c*dz0 ? (what the dz0dt figure's rows are "
          f"built from)")

    out = {k: [] for k in ("corr_state", "scale_state", "resid_state",
                            "corr_incr", "scale_incr", "resid_incr", "dt")}
    with torch.no_grad():
        for i in idx:
            window0, window1, dt_window, _theta = dataset[int(i)]
            z0 = window0[0].unsqueeze(0).to(resolved_device)
            z1 = window1[0].unsqueeze(0).to(resolved_device)
            z0_next = window0[1].unsqueeze(0).to(resolved_device)
            z1_next = window1[1].unsqueeze(0).to(resolved_device)

            c, s, r = _corr_and_scale(z1, z0)
            out["corr_state"].append(float(c[0]))
            out["scale_state"].append(float(s[0]))
            out["resid_state"].append(float(r[0]))

            c, s, r = _corr_and_scale(z1_next - z1, z0_next - z0)
            out["corr_incr"].append(float(c[0]))
            out["scale_incr"].append(float(s[0]))
            out["resid_incr"].append(float(r[0]))
            out["dt"].append(float(dt_window[0]))

    for label, key in (("state (z1 vs z0)", "state"),
                        ("increments (dz1 vs dz0)", "incr")):
        corr = np.array(out[f"corr_{key}"])
        scale = np.array(out[f"scale_{key}"])
        resid = np.array(out[f"resid_{key}"])
        print(f"\n{label}:")
        print(f"  |correlation| median {np.median(np.abs(corr)):.3f}  "
              f"(10th pct {np.percentile(np.abs(corr), 10):.3f}, "
              f"90th {np.percentile(np.abs(corr), 90):.3f})")
        print(f"  best-fit scale c     median {np.median(scale):.4g}  "
              f"(spread {np.percentile(scale, 10):.3g} .. "
              f"{np.percentile(scale, 90):.3g})")
        print(f"  residual ||u-c*v||/||u||  median {np.median(resid):.3f}")

    med_corr = float(np.median(np.abs(out["corr_incr"])))
    med_resid = float(np.median(out["resid_incr"]))
    print("\nVERDICT (on the increments, which is what the figure plots):")
    if med_corr > 0.9 and med_resid < 0.3:
        print("  DEGENERATE. dz1 is essentially a rescaled dz0, so the "
              "'second derivative' row is the first-derivative row times a "
              "constant -- it carries no curvature information, and no "
              "max_dt argument can rest on it.")
    elif med_corr < 0.5:
        print("  NOT degenerate. dz1 is largely independent of dz0, so the "
              "constant ratio between the two rows is a coincidence of "
              "ensemble scales, not an identity.")
    else:
        print(f"  PARTIAL: |corr| {med_corr:.2f}, residual {med_resid:.2f}. "
              f"Some of dz1 is a rescaled dz0 and some is not; the "
              f"second-derivative row is contaminated but not empty.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--base-path", type=Path, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-windows", type=int, default=512)
    parser.add_argument("--max-dt", type=float, default=None)
    args = parser.parse_args()
    check_z1_degeneracy(args.checkpoint, base_path=args.base_path,
                         size=args.size, device=args.device,
                         n_windows=args.n_windows, max_dt=args.max_dt)


if __name__ == "__main__":
    main()
