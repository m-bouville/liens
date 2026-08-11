"""
Is a SECOND-derivative latent stream (z2) learnable from this data at all?

Before building a z2 stream -- an encoder head supervised on z0's curvature,
the way z1 is supervised on its velocity -- this asks whether the target
even exists above the noise. The stencil that would supply that target is a
finite second difference of z0, and its noise scales as ~sigma/dt^2 against
a signal measured at ~1e-7, so "unlearnable at every dt" is a live outcome
and a cheap one to rule out.

NO f_theta, NO training. This is a property of the ENCODER and the save
schedule, so it loads an AE-family checkpoint directly (the natural input:
z0 is the encoder's own recon stream) and never constructs a corrector.

TWO independent measurements, because either alone is ambiguous:

1. BIAS FRACTION, |E[z2]| / E[|z2|], the same decomposition
   check_parameter_dependence applies to z1's own residual. Near 0 means
   the estimate is noise cancelling across windows; near 1 means a
   consistent curvature in a consistent direction. Reported per dt decade
   and per temperature split, because the Taylor fit already showed the
   bias-optimal dt spanning 40x across the T=0.9 split -- an aggregate
   would average a real signal in one regime against noise in another.

   Ambiguity: curvature that is real but SIGN-VARYING across the ensemble
   also gives a low bias fraction. That is why (2) exists.

2. DISJOINT-STENCIL AGREEMENT. Two second differences over frames (0,1,2)
   and (3,4,5) -- sharing NO frame, hence no encoding noise. If a smooth
   curvature field exists they track each other; if both are noise their
   correlation is 0. No noise model, no assumption about sign structure.

   Why disjoint and not two stencils centred on the same frame (the obvious
   design, and the first one built here): a shared centre frame puts the
   SAME noise into both estimates with a large coefficient, correlating
   them even when no curvature exists at all. For uniform spacing the floor
   is 4/sqrt(6*6) = 2/3 -- measured at 0.667 on a synthetic pure-noise
   field, matching the algebra to three digits. A diagnostic whose "no
   signal" answer is 0.67 cannot report "unlearnable", which is the answer
   it exists to be able to give.

   The cost is that the two stencils sit at DIFFERENT centres, so they
   agree only if curvature is smooth across their separation. That is not a
   loophole: a z2 stream needs exactly that smoothness to be learnable, so
   the stricter test is the honest one.

A caveat the correlation does NOT remove: any finite stencil measures a
dt-AVERAGED curvature, not the pointwise one. That distinction is exactly
what made n_substeps mismatches fatal for f_theta, so the spacings are
reported alongside.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from training.checkpoint_components import build_ae_from_checkpoint
from training.datasets import MicrostructureEvolutionDataset
from utils import load_datasets as load


def second_difference(z_minus: torch.Tensor, z_centre: torch.Tensor,
                       z_plus: torch.Tensor, dt_minus: float,
                       dt_plus: float) -> torch.Tensor:
    """Nonuniform 3-point second derivative of z0 at the centre frame.

        z2 = 2 * [ dt_- * z(t+dt_+) - (dt_+ + dt_-) * z(t) + dt_+ * z(t-dt_-) ]
             / ( dt_+ * dt_- * (dt_+ + dt_-) )

    The nonuniform form is required, not a convenience: the save schedule is
    geometric (ratios ~1.15-1.35), so dt_+ != dt_- for essentially every
    window, and the uniform-spacing formula would carry an O(dt_+ - dt_-)
    first-derivative leak -- it would report z1's error as curvature.

    Exact for quadratics at ANY spacing, which is what the test asserts.
    """
    denom = dt_plus * dt_minus * (dt_plus + dt_minus)
    return 2.0 * (dt_minus * z_plus
                   - (dt_plus + dt_minus) * z_centre
                   + dt_plus * z_minus) / denom


def first_difference(z_minus: torch.Tensor, z_centre: torch.Tensor,
                      z_plus: torch.Tensor, dt_minus: float,
                      dt_plus: float) -> torch.Tensor:
    """Nonuniform 3-point FIRST derivative of z0 at the centre frame.

    The control for the curvature measurement: a trajectory smooth enough to
    carry curvature must first be smooth in its velocity. If the two
    stencils agree strongly on z1 but not on z2, the roughness lives in the
    second derivative -- an encoding wiggle, which L_interp targets. If they
    disagree on z1 too, something more basic is wrong with the latent
    trajectory and L_interp would not be sufficient.

    Exact for quadratics at any spacing, like its second-order counterpart.
    """
    denom = dt_plus * dt_minus * (dt_plus + dt_minus)
    return (dt_minus * dt_minus * z_plus
            + (dt_plus * dt_plus - dt_minus * dt_minus) * z_centre
            - dt_plus * dt_plus * z_minus) / denom


def _per_window_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Correlation computed WITHIN each window, then summarised across them.

    The pooled correlation over every element of every window is
    variance-weighted: stencil noise scales as sigma/dt^2, so small-dt
    windows carry enormously larger |z2| and dominate the pooled sums. On
    real data the pooled number landed essentially on top of the SMALLEST-dt
    decade's value while that decade held 8% of the sample -- i.e. it was
    reporting the noisiest slice, not the population.

    It is also grand-mean centred, so a window whose curvature field sits at
    a different overall level than the ensemble registers that offset as
    covariance -- a between-window difference leaking into a within-window
    question.

    Reported here: the MEDIAN per-window correlation (every window counts
    once), its quartiles and the fraction negative (a consistent weak effect
    looks nothing like a few dominant windows, and one scalar cannot tell
    them apart), and BOTH the centred correlation and the uncentred cosine
    similarity -- centring removes a spatially uniform curvature component,
    which for a coarsening field may be signal rather than nuisance, so the
    choice must not silently decide the verdict.
    """
    n = a.shape[0]
    corrs, cosines = [], []
    for i in range(n):
        u = a[i].ravel().astype(np.float64)
        v = b[i].ravel().astype(np.float64)
        good = np.isfinite(u) & np.isfinite(v)
        u, v = u[good], v[good]
        if u.size < 2:
            continue
        uc, vc = u - u.mean(), v - v.mean()
        d = float(np.sqrt((uc * uc).sum() * (vc * vc).sum()))
        if d > 0:
            corrs.append(float((uc * vc).sum() / d))
        d = float(np.sqrt((u * u).sum() * (v * v).sum()))
        if d > 0:
            cosines.append(float((u * v).sum() / d))
    corrs = np.array(corrs)
    cosines = np.array(cosines)
    if corrs.size == 0:
        return {"n": 0}
    return {
        "n": int(corrs.size),
        "median": float(np.median(corrs)),
        "q25": float(np.percentile(corrs, 25)),
        "q75": float(np.percentile(corrs, 75)),
        "frac_negative": float((corrs < 0).mean()),
        "median_cosine": float(np.median(cosines)) if cosines.size else float("nan"),
    }


def _bias_fraction(signed: np.ndarray) -> tuple[float, float, float]:
    """(|E[x]|, E[|x|], ratio) over the window axis.

    signed is (n_windows, ...) with the sign PRESERVED: the mean is taken
    over windows first and its magnitude second, so errors in random
    directions cancel while a consistent one survives. Taking the magnitude
    first -- E[|x|] -- cannot distinguish the two, which is the whole point.
    """
    mean_signed = signed.mean(axis=0)
    bias = float(np.abs(mean_signed).mean())
    total = float(np.abs(signed).mean())
    return bias, total, (bias / total if total > 0 else float("nan"))


def _corr(u: np.ndarray, v: np.ndarray) -> float:
    """Pearson correlation over flattened, per-window-centred arrays."""
    u = u.ravel()
    v = v.ravel()
    good = np.isfinite(u) & np.isfinite(v)
    u, v = u[good], v[good]
    if u.size < 2:
        return float("nan")
    u = u - u.mean()
    v = v - v.mean()
    denom = float(np.sqrt((u * u).sum() * (v * v).sum()))
    return float((u * v).sum() / denom) if denom > 0 else float("nan")


def check_z2_measurability(checkpoint_path: Path, base_path: Path | None = None,
                            device: str | None = None, n_windows: int = 512,
                            max_dt: float | None = None,
                            latent_cache_dir: Path | str | None = None) -> dict:
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device,
                             weights_only=True)
    ae_path = Path(checkpoint.get("ae_checkpoint") or checkpoint_path)
    _ae, ae_encoder, _ae_ck, _cfgs, _recon = build_ae_from_checkpoint(
        ae_path, resolved_device)

    data_config = checkpoint.get("data_config") or {}
    test_dirs = checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(
            f"{checkpoint_path} has no saved test_dirs, so there is no "
            f"population to measure. Pass a checkpoint that recorded its own "
            f"split.")
    test_dirs = load.validate_run_dirs(
        [Path(d) for d in test_dirs], source=str(checkpoint_path),
        min_stdev_phi=data_config.get("min_stdev_phi"))
    # window_length=6 so two DISJOINT stencils fit: frames (0,1,2) and
    # (3,4,5), sharing no frame and therefore no encoding noise. A 3-frame
    # window would give the bias fraction but not the agreement test, and a
    # 5-frame window only supports stencils that share a centre -- whose
    # shared noise alone correlates them at 2/3 (see module docstring).
    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=ae_encoder, device=resolved_device,
        window_length=6,
        max_dt=max_dt if max_dt is not None else data_config.get("max_dt"),
        min_step=data_config.get("min_step", 0),
        min_stdev_phi=data_config.get("min_stdev_phi"),
        latent_cache_dir=latent_cache_dir,
    )
    print(f"encoder from {ae_path.name}; {len(dataset)} 6-frame windows available")
    if len(dataset) == 0:
        raise ValueError(
            "no 6-frame windows survived the filters -- lower min_passing_steps "
            "or raise max_dt; z2 measurability cannot be assessed without them.")

    take = min(int(n_windows), len(dataset))
    idx = np.linspace(0, len(dataset) - 1, take).round().astype(int)

    narrow, wide, dt_narrow, dt_wide, temps = [], [], [], [], []
    d1_a, d1_b = [], []
    with torch.no_grad():
        for i in idx:
            window, dt_window, _theta = dataset[int(i)][:3]
            z = window.to(resolved_device)
            dts = [float(v) for v in dt_window]
            # DISJOINT stencils: centres 1 and 4, no shared frame.
            n_est = second_difference(z[0], z[1], z[2], dts[0], dts[1])
            w_est = second_difference(z[3], z[4], z[5], dts[3], dts[4])
            narrow.append(n_est.cpu().numpy())
            wide.append(w_est.cpu().numpy())
            # FIRST-derivative control on the same disjoint stencils
            d1_a.append(first_difference(z[0], z[1], z[2], dts[0], dts[1]).cpu().numpy())
            d1_b.append(first_difference(z[3], z[4], z[5], dts[3], dts[4]).cpu().numpy())
            dt_narrow.append(0.5 * (dts[0] + dts[1]))
            dt_wide.append(0.5 * (dts[3] + dts[4]))
            run_dir, _steps = dataset.window_info(int(i))
            temps.append(load.read_metadata(Path(run_dir) / "metadata.txt").temperature)

    narrow = np.stack(narrow)
    wide = np.stack(wide)
    d1_a = np.stack(d1_a)
    d1_b = np.stack(d1_b)
    dt_narrow = np.array(dt_narrow)
    temps = np.array(temps)

    print(f"\ncheck_z2_measurability: {take} windows")
    print(f"  first  stencil (frames 0-2) half-spacing: {dt_narrow.min():.4g} .. {dt_narrow.max():.4g}")
    print(f"  second stencil (frames 3-5) half-spacing: {min(dt_wide):.4g} .. {max(dt_wide):.4g}")

    bias, total, frac = _bias_fraction(narrow)
    agreement = _corr(narrow, wide)
    print("\n" + "=" * 70)
    print("1. BIAS FRACTION of the curvature estimate (first stencil)")
    print("=" * 70)
    print(f"  |E[z2]| (survives averaging across windows) = {bias:.6e}")
    print(f"  E[|z2|] (total magnitude)                   = {total:.6e}")
    print(f"  bias fraction                               = {frac:.3f}")
    print("  (near 0 -> cancels across windows: either noise, OR a real but "
          "sign-varying curvature. Measurement 2 separates those.)")

    print("\n" + "=" * 70)
    print("2. DISJOINT-STENCIL AGREEMENT (the decisive number)")
    print("=" * 70)
    pw = _per_window_agreement(narrow, wide)
    pw1 = _per_window_agreement(d1_a, d1_b)
    if pw.get("n"):
        print(f"  PER-WINDOW corr, median  = {pw['median']:+.3f}  "
              f"(quartiles {pw['q25']:+.3f} .. {pw['q75']:+.3f}; "
              f"{pw['frac_negative'] * 100:.0f}% of windows negative)")
        print(f"  PER-WINDOW cosine, median= {pw['median_cosine']:+.3f}  "
              f"(uncentred: keeps a spatially uniform curvature component, "
              f"which centring removes)")
    print(f"  pooled over all elements = {agreement:+.3f}   <- VARIANCE-WEIGHTED, "
          f"read with care")
    print(f"     (noise ~ sigma/dt^2, so small-dt windows carry far larger "
          f"|z2| and dominate the pooled sums; the median above counts every "
          f"window once. A large gap between the two IS the warning.)")
    if pw1.get("n"):
        print(f"\n  FIRST-derivative control, per-window median = "
              f"{pw1['median']:+.3f}   cosine = {pw1['median_cosine']:+.3f}")
        print(f"     (READ THE COSINE for this control. Velocity in a "
              f"coarsening field has a large coherent component -- the "
              f"causal baseline holds 93% correlation on exactly these "
              f"spans -- and per-window centring subtracts precisely that "
              f"component, so the centred number can sit near zero while "
              f"the velocity is in fact strongly persistent.)")
        print(f"     (the trajectory must be smooth in VELOCITY before it can "
              f"carry curvature. Strongly positive here with a negative z2 "
              f"agreement -> the roughness is confined to the second "
              f"derivative, i.e. an encoding wiggle, which is what L_interp "
              f"targets. Negative here too -> something more basic is wrong "
              f"with the latent trajectory and L_interp would not suffice.)")
    print("  (~0 -> the two estimates share no signal, so both are encoding "
          "noise and z2 is NOT measurable from this data.")
    print("   high -> a real curvature is present that both stencils see, so "
          "a z2 stream has a target to learn.")
    print("   CAVEAT: the two stencils sit at DIFFERENT centres, so this "
          "tests a SMOOTH curvature field -- which is what a z2 stream needs "
          "anyway. Any finite stencil also measures a dt-AVERAGED curvature, "
          "not the pointwise value.)")

    out = {"bias_fraction": frac, "agreement": agreement,
            "per_window": pw, "per_window_first_deriv": pw1,
            "n_windows": int(take), "narrow": narrow, "wide": wide,
            "dt_narrow": dt_narrow, "temperature": temps}
    _print_by_group(narrow, wide, dt_narrow, temps)
    return out


def _print_by_group(narrow: np.ndarray, wide: np.ndarray,
                     dt_narrow: np.ndarray, temps: np.ndarray) -> None:
    """Per dt decade and per T split.

    Aggregates mislead here by construction: the Taylor fit put the
    bias-optimal dt at ~5470 for T<0.9 against ~133 for T>=0.9, a 40x
    spread, so a single number averages a regime where curvature dominates
    against one where it does not.
    """
    print("\n" + "-" * 70)
    print("per dt decade (first stencil half-spacing)")
    print("-" * 70)
    # E|z2| AND E|z2|*dt^2. Noise in a second difference is
    # sqrt(6)*sigma/dt^2, so under the noise hypothesis the SCALED column is
    # CONSTANT across decades while the raw one falls as 1/dt^2. Real
    # curvature is dt-independent, so its scaled column GROWS as dt^2. That
    # contrast is the direct test of what the estimate is made of, and it is
    # independent of the correlation.
    # (The invariant is |z2|*dt^2, equivalently |z2|^2*dt^4 -- |z2|^2*dt^2
    #  still falls as 1/dt^2 and would look like a decaying signal.)
    print(f"  {'decade':<16}{'n':>7}{'bias frac':>11}{'med corr':>10}"
          f"{'E|z2|':>12}{'E|z2|*dt^2':>12}")
    if dt_narrow.size and dt_narrow.max() > 0:
        positive = dt_narrow[dt_narrow > 0]
        if positive.size:
            lo = int(np.floor(np.log10(positive.min())))
            hi = int(np.ceil(np.log10(positive.max())))
            for d in range(lo, hi):
                sel = (dt_narrow >= 10.0 ** d) & (dt_narrow < 10.0 ** (d + 1))
                if sel.sum() < 2:
                    continue
                _b, _t, frac = _bias_fraction(narrow[sel])
                pw = _per_window_agreement(narrow[sel], wide[sel])
                mag = float(np.abs(narrow[sel]).mean())
                dt_mean = float(dt_narrow[sel].mean())
                print(f"  1e{d:<3d}- 1e{d + 1:<7d}{int(sel.sum()):>7}"
                      f"{frac:>11.3f}{pw.get('median', float('nan')):>+10.3f}"
                      f"{mag:>12.3e}{mag * dt_mean ** 2:>12.3e}")

    print("\n" + "-" * 70)
    print("per temperature split (the Taylor fit put dt* 40x apart across it)")
    print("-" * 70)
    print(f"  {'subset':<16}{'n':>7}{'bias frac':>11}{'med corr':>10}"
          f"{'pooled':>10}")
    for label, sel in (("T < 0.9", temps < 0.9), ("T >= 0.9", temps >= 0.9)):
        if sel.sum() < 2:
            continue
        _b, _t, frac = _bias_fraction(narrow[sel])
        pw = _per_window_agreement(narrow[sel], wide[sel])
        print(f"  {label:<16}{int(sel.sum()):>7}{frac:>11.3f}"
              f"{pw.get('median', float('nan')):>+10.3f}"
              f"{_corr(narrow[sel], wide[sel]):>+10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--base-path", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-windows", type=int, default=512)
    parser.add_argument("--max-dt", type=float, default=None)
    parser.add_argument("--latent-cache-dir", type=Path, default=None)
    args = parser.parse_args()
    check_z2_measurability(
        args.checkpoint, base_path=args.base_path, device=args.device,
        n_windows=args.n_windows, max_dt=args.max_dt,
        latent_cache_dir=args.latent_cache_dir)


if __name__ == "__main__":
    main()
