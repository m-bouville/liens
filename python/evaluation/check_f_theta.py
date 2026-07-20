"""
Diagnose f_theta's own output magnitude directly -- specifically to
distinguish two different explanations for a multi-step rollout blowing
up:

  (1) intrinsic dt-dependence: f_theta produces large outputs whenever
      dt is large, REGARDLESS of whether its z0 input is real data or
      a chained prediction. If this is the whole story, ||f|| computed
      on REAL data at a given dt should already be large.

  (2) off-distribution blowup: f_theta was only ever trained on REAL
      (encoder-provided) z0 at step 1 of any window. At step 2+ of an
      actual multi-step rollout, its z0 INPUT is the model's own,
      once-compounded prediction -- never seen during training as an
      INPUT (only ever as this step's own TARGET). If f_theta is
      poorly behaved off its own training distribution, ||f|| on a
      chained z0 could be far larger than ||f|| on the REAL z0 at the
      exact same (dt, theta) -- a genuinely different failure mode
      from (1), and one dt-reweighting alone would not fix.

Computes, for every window with window_length >= 3 in the given
dataset:
  f1_real:     f_theta.f(z0(t0), z1(t0), theta)      -- step 1's real input
  z0_hat_1:    f_theta(z0(t0), z1(t0), dt1, theta)    -- step 1's OWN z0 prediction
  f2_chained:  f_theta.f(z0_hat_1, z1(t1), theta)     -- step 2, CHAINED z0 input
  f2_real:     f_theta.f(z0(t1), z1(t1), theta)       -- step 2, REAL z0 input (ablation)
z0_step1_error = ||z0_hat_1 - z0(t1)|| is also reported -- how far
off-distribution the chained input actually is.

The key comparison is f2_chained vs f2_real AT THE SAME dt2: if
f2_chained is systematically much larger, that's direct evidence for
explanation (2) -- f_theta misbehaving specifically on its own
predictions, not on large dt per se.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_f_theta \
        --lds-checkpoint ../checkpoints/stage3b/32x32-tiny-stage3b.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder, EncoderDecoderPair, MultiStreamAutoencoder
from models.constants import LATENT_SPATIAL_SIZE
from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_dynamics import LatentDynamics
from models.latent_streams import (
    cross_check_stream_configs_against_state_dict, resolve_stream_configs_from_checkpoint_config,
)
from training.datasets import MicrostructureEvolutionDataset

# GENERAL POLICY (matches training/train_refinement.py's own
# _PYTHON_ROOT): every default checkpoint/output path is built from
# THIS anchor, never from a bare relative string.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/


def _per_sample_l2(t: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B,) L2 norm per sample."""
    return t.flatten(start_dim=1).norm(dim=1)


def compute_f_diagnostics(dataset: MicrostructureEvolutionDataset, f_theta: LatentDynamics,
                           device: torch.device) -> dict[str, np.ndarray]:
    """
    Returns a dict of 1D arrays (one entry per window): f1_real_norm,
    f2_chained_norm, f2_real_norm, z0_step1_error, dt1, dt2, plus the
    ACTUAL per-step contribution to z0 (||f * dt^2/2||, what really
    lands in z0_hat, not just ||f|| alone) as f1_real_contrib,
    f2_chained_contrib, f2_real_contrib.

    ALSO computes the IDEAL f2 -- what f_theta SHOULD have output, given
    its own actual input, to exactly hit the real z0(t2). This is only
    possible because z0(t2) is available as ground truth here (unlike
    at actual training/inference time, where it's the unknown being
    predicted) -- it directly answers a question ||f|| alone cannot:
    is f_theta's output on a chained input too LARGE (overshooting the
    correction it needs to make) or too SMALL (undershooting -- still
    growing relative to real-data behavior, per f2_chained vs f2_real,
    but not growing ENOUGH to correct what a large-dt/off-distribution
    input actually requires)? These call for opposite fixes (bounding
    f_theta's output only helps the former, and actively hurts the
    latter), so the distinction matters before choosing one.
      f2_chained_ideal: [z0(t2) - z0_hat_1 - z1(t1)*dt2] / (dt2^2/2)
        -- the target implied by f_theta's OWN chained input (z0_hat_1,
        the same input f2_chained itself used).
      f2_real_ideal: [z0(t2) - z0(t1) - z1(t1)*dt2] / (dt2^2/2)
        -- the SAME kind of target, but from the real z0(t1) input --
        a baseline sanity check: this is close to exactly what f_theta
        was directly trained against at single-step (Stage 3a) time,
        so f2_real should already track ITS OWN ideal closely if
        f_theta was trained well at all. If it doesn't, the problem
        isn't specifically about off-distribution generalization --
        f_theta's basic single-step calibration is already off.
    Reported as ratio_chained/ratio_real (||actual||/||ideal||, >1 =
    overshooting, <1 = undershooting) and cos_sim_chained/cos_sim_real
    (directional correctness, independent of magnitude -- near +1 means
    at least pointing the right way, near 0 or negative is a
    qualitatively worse failure than a magnitude error alone).

    Requires dataset.window_length >= 3 -- step 2's own diagnostic
    needs a real z0(t1)/z1(t1)/dt2 beyond the first transition.
    """
    if dataset.window_length < 3:
        raise ValueError(
            f"window_length={dataset.window_length} -- this diagnostic needs >= 3 "
            f"(step 1 AND step 2, each with a real transition) to compare f_theta on "
            f"a chained vs real z0 input at step 2."
        )

    f1_real_norms, f2_chained_norms, f2_real_norms = [], [], []
    f1_real_contribs, f2_chained_contribs, f2_real_contribs = [], [], []
    z0_step1_errors, dt1s, dt2s = [], [], []
    ratio_chained_list, ratio_real_list = [], []
    cos_sim_chained_list, cos_sim_real_list = [], []

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    with torch.no_grad():
        for window0, window1, dt_window, theta in loader:
            window0 = window0.to(device)
            window1 = window1.to(device)
            dt_window = dt_window.to(device)
            theta = theta.to(device)

            z0_0, z0_1, z0_2 = window0[:, 0], window0[:, 1], window0[:, 2]
            z1_0, z1_1 = window1[:, 0], window1[:, 1]
            dt1, dt2 = dt_window[:, 0], dt_window[:, 1]

            f1_real = f_theta.f(z0_0, z1_0, theta)
            z0_hat_1 = f_theta(z0_0, z1_0, dt1, theta)  # step 1's own z0 prediction

            f2_chained = f_theta.f(z0_hat_1, z1_1, theta)
            f2_real = f_theta.f(z0_1, z1_1, theta)  # ablation: real z0 at the SAME step/dt2

            dt1_r = dt1.view(-1, 1, 1, 1)
            dt2_r = dt2.view(-1, 1, 1, 1)

            # Ideal targets -- solving forward()'s own update formula
            # for f, using the REAL z0(t2) in place of the (unknown,
            # at real inference time) prediction.
            f2_chained_ideal = (z0_2 - z0_hat_1 - z1_1 * dt2_r) / (dt2_r ** 2 / 2)
            f2_real_ideal = (z0_2 - z0_1 - z1_1 * dt2_r) / (dt2_r ** 2 / 2)

            f2_chained_flat = f2_chained.flatten(start_dim=1)
            f2_chained_ideal_flat = f2_chained_ideal.flatten(start_dim=1)
            f2_real_flat = f2_real.flatten(start_dim=1)
            f2_real_ideal_flat = f2_real_ideal.flatten(start_dim=1)

            ratio_chained = f2_chained_flat.norm(dim=1) / f2_chained_ideal_flat.norm(dim=1).clamp(min=1e-12)
            ratio_real = f2_real_flat.norm(dim=1) / f2_real_ideal_flat.norm(dim=1).clamp(min=1e-12)
            cos_sim_chained = torch.nn.functional.cosine_similarity(f2_chained_flat, f2_chained_ideal_flat, dim=1)
            cos_sim_real = torch.nn.functional.cosine_similarity(f2_real_flat, f2_real_ideal_flat, dim=1)

            f1_real_norms.append(_per_sample_l2(f1_real).cpu().numpy())
            f2_chained_norms.append(_per_sample_l2(f2_chained).cpu().numpy())
            f2_real_norms.append(_per_sample_l2(f2_real).cpu().numpy())

            f1_real_contribs.append(_per_sample_l2(f1_real * (dt1_r ** 2 / 2)).cpu().numpy())
            f2_chained_contribs.append(_per_sample_l2(f2_chained * (dt2_r ** 2 / 2)).cpu().numpy())
            f2_real_contribs.append(_per_sample_l2(f2_real * (dt2_r ** 2 / 2)).cpu().numpy())

            z0_step1_errors.append(_per_sample_l2(z0_hat_1 - z0_1).cpu().numpy())
            dt1s.append(dt1.cpu().numpy())
            dt2s.append(dt2.cpu().numpy())
            ratio_chained_list.append(ratio_chained.cpu().numpy())
            ratio_real_list.append(ratio_real.cpu().numpy())
            cos_sim_chained_list.append(cos_sim_chained.cpu().numpy())
            cos_sim_real_list.append(cos_sim_real.cpu().numpy())

    return {
        "f1_real_norm": np.concatenate(f1_real_norms),
        "f2_chained_norm": np.concatenate(f2_chained_norms),
        "f2_real_norm": np.concatenate(f2_real_norms),
        "f1_real_contrib": np.concatenate(f1_real_contribs),
        "f2_chained_contrib": np.concatenate(f2_chained_contribs),
        "f2_real_contrib": np.concatenate(f2_real_contribs),
        "z0_step1_error": np.concatenate(z0_step1_errors),
        "dt1": np.concatenate(dt1s),
        "dt2": np.concatenate(dt2s),
        "ratio_chained": np.concatenate(ratio_chained_list),
        "ratio_real": np.concatenate(ratio_real_list),
        "cos_sim_chained": np.concatenate(cos_sim_chained_list),
        "cos_sim_real": np.concatenate(cos_sim_real_list),
    }


def _log_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """None if either array is effectively constant (e.g. a
    toy/uniform-dt dataset) -- np.corrcoef would emit a RuntimeWarning
    and silently return nan otherwise, worse than an explicit N/A. Uses
    an epsilon threshold, not exact std()==0: values that round-trip
    through float32 (as these do, via torch) rarely come back
    bit-exactly identical even when genuinely constant -- std() lands
    near machine epsilon (~1e-7), not exactly 0.0, which an exact-zero
    check would miss."""
    log_x, log_y = np.log10(np.maximum(x, 1e-12)), np.log10(np.maximum(y, 1e-12))
    if log_x.std() < 1e-6 or log_y.std() < 1e-6:
        return None
    return np.corrcoef(log_x, log_y)[0, 1]


def _fmt_corr(c: float | None) -> str:
    return "N/A (zero variance)" if c is None else f"{c:.3f}"


def check_dead_relus(f_theta: LatentDynamics, dataset: MicrostructureEvolutionDataset,
                      device: torch.device, n_samples: int = 512) -> dict[str, np.ndarray]:
    """
    Diagnoses a specific, testable hypothesis for f_theta.f(...)
    returning a CONSTANT (zero-variance) output regardless of input: a
    dead (or stuck) trunk. If a unit's pre-activation is negative for
    EVERY realistic input, it's operating in the SAME regime for every
    sample regardless of what z0/z1/theta actually was -- for plain
    ReLU that means permanently zero output AND zero gradient (d/dx
    ReLU(x)=0 for x<0, so no future update can ever move it back --
    this is what LatentDynamics' own trunk used to use, until a real,
    confirmed collapse -- see that class's own docstring -- motivated
    switching to LeakyReLU). For LeakyReLU specifically (what the
    trunk now actually uses), a unit stuck on the negative side isn't
    fully dead -- d/dx LeakyReLU(x)=negative_slope for x<0, a small but
    nonzero gradient -- but it's still stuck in the SAME, heavily
    attenuated regime for every sample, still worth flagging as a much
    weaker version of the same underlying symptom.

    Checks the SIGN OF THE PRE-ACTIVATION (the layer's own INPUT), not
    whether the output is exactly 0 -- LeakyReLU's own output is never
    exactly 0 for a nonzero input (LeakyReLU(x) = negative_slope*x for
    x<0, itself nonzero whenever x is), so an exact-zero check would
    silently always report 0% and miss the real, if less catastrophic,
    version of this failure mode entirely.

    A unit counts as "stuck" only if its pre-activation is negative for
    EVERY sample in the batch, not just often-negative -- saturating on
    any given input is normal, expected behavior, not evidence of a
    problem.

    Returns a dict of {layer_name: fraction_stuck} for every
    ReLU/LeakyReLU layer in f_theta's own trunk, plus "trunk_output" --
    the fraction of the FINAL hidden layer's own output units that are
    stuck, i.e. what actually feeds the last Linear layer. If
    trunk_output is at or near 1.0 for a plain-ReLU trunk, the
    constant-output hypothesis is confirmed directly: the final layer's
    own input is (near-)entirely zero for every real sample, so its
    output can only ever be (approximately) its own bias term,
    regardless of what z0/z1/theta it was ever given.
    """
    n_samples = min(n_samples, len(dataset))
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False, num_workers=0)
    window0, window1, dt_window, theta = next(iter(loader))
    window0, window1, theta = window0.to(device), window1.to(device), theta.to(device)
    z0, z1 = window0[:, 0], window1[:, 0]

    pre_activations: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name):
        def _hook(module, inp, out):
            pre_activations[name] = inp[0].detach()  # the layer's own INPUT -- pre-activation, not output
        return _hook

    activation_types = (torch.nn.ReLU, torch.nn.LeakyReLU)
    act_layers = [(f"act_{i}", layer) for i, layer in enumerate(f_theta.net) if isinstance(layer, activation_types)]
    for name, layer in act_layers:
        hooks.append(layer.register_forward_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            f_theta.f(z0, z1, theta)  # triggers all hooks via the real forward pass
    finally:
        for h in hooks:
            h.remove()

    result = {}
    for name, _ in act_layers:
        pre_act = pre_activations[name]  # (n_samples, hidden_dim)
        stuck_mask = (pre_act < 0).all(dim=0)  # stuck iff negative pre-activation for EVERY sample
        result[name] = stuck_mask.float().mean().item()

    if act_layers:
        result["trunk_output"] = result[act_layers[-1][0]]  # last activation IS the trunk's own output, feeding net[-1]

    return result


def _print_summary(name: str, values: np.ndarray) -> None:
    print(f"  {name:20s}: mean={values.mean():14.4e}  median={np.median(values):14.4e}  "
          f"max={values.max():14.4e}  p90={np.percentile(values, 90):14.4e}")


def _print_binned_by_dt2(name: str, dt2: np.ndarray, values: np.ndarray) -> None:
    """Median of `values` within each dt2 decade -- median (not mean),
    deliberately: this is specifically about separating a genuine,
    widespread effect from a small-dt2 numerical-amplification
    artifact, and a heavy-tailed artifact would otherwise dominate the
    MEAN of any bin it's present in, defeating the point of splitting
    by decade in the first place. Matches
    check_parameter_dependence.py's own dt-decade convention."""
    log_dt2 = np.log10(np.maximum(dt2, 1e-12))
    decade_min, decade_max = int(np.floor(log_dt2.min())), int(np.ceil(log_dt2.max()))
    print(f"  {name}:")
    for decade in range(decade_min, decade_max):
        mask = (log_dt2 >= decade) & (log_dt2 < decade + 1)
        n = mask.sum()
        if n == 0:
            continue
        print(f"    1e{decade}-1e{decade + 1}  n={n:5d}  median={np.median(values[mask]):14.4e}")


def check_f_theta(
    lds_checkpoint_path: Path, min_step: int | None = None, min_stdev_phi: float | None = None,
    output_path: Path | None = None, device: str | None = None,
) -> Path:
    """Prints the f_theta diagnostic summary and saves a comparison figure; returns the figure's path."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if output_path is None:
        output_path = (_PYTHON_ROOT.parent / "output" / "stage3"
                        / f"{lds_checkpoint_path.stem}-f_theta_diagnostic.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lds_checkpoint = torch.load(lds_checkpoint_path, map_location=device, weights_only=True)
    lds_config = lds_checkpoint["config"]

    data_config = lds_checkpoint.get("data_config")
    if data_config is None:
        print("WARNING: checkpoint has no saved data_config -- falling back to "
              "min_step=0, min_stdev_phi=None, window_length=3 (may not match training).")
        data_config = {"min_step": 0, "min_stdev_phi": None, "window_length": 3}
    min_step = min_step if min_step is not None else data_config["min_step"]
    min_stdev_phi = min_stdev_phi if min_stdev_phi is not None else data_config["min_stdev_phi"]
    # window_length=3 regardless of what THIS checkpoint was trained at
    # (e.g. even a stage-3a, n_rollout_steps=1 checkpoint's own f_theta
    # can be probed this way) -- the diagnostic needs step 1 AND step
    # 2's own real transition, not however many steps training used.
    window_length = 3

    test_dirs = lds_checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(f"{lds_checkpoint_path} has no saved test_dirs")
    test_dirs = [Path(d) for d in test_dirs]

    ae_checkpoint_path = Path(lds_checkpoint["ae_checkpoint"])
    ae_checkpoint = torch.load(ae_checkpoint_path, map_location=device, weights_only=True)
    ae_config = ae_checkpoint["config"]
    stream_configs, recon_stream_name = resolve_stream_configs_from_checkpoint_config(ae_config)
    stream_configs, recon_stream_name = cross_check_stream_configs_against_state_dict(
        stream_configs, recon_stream_name, ae_checkpoint["model_state"],
    )
    recon_stream = stream_configs[recon_stream_name]
    decoder_for_stream = ae_config.get("decoder_for_stream")
    is_flat_checkpoint = any(k.startswith("encoder.") for k in ae_checkpoint["model_state"])
    if is_flat_checkpoint:
        encoder = Encoder(input_size=ae_config["size"], in_channels=1,
                           base_channels=ae_config["base_channels"], stream_configs=stream_configs)
        decoder = Decoder(output_size=ae_config["size"], out_channels=1,
                           base_channels=ae_config["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size)
        ae = EncoderDecoderPair(encoder, decoder, stream_name=recon_stream_name,
                                 mode=recon_stream.mode).to(device)
    elif len(stream_configs) == 1:
        ae = Autoencoder(
            size=ae_config["size"], channels=1,
            base_channels=ae_config["base_channels"], latent_channels=recon_stream.channels,
            latent_spatial_size=recon_stream.spatial_size,
        ).to(device)
    elif decoder_for_stream is None:
        encoder = Encoder(input_size=ae_config["size"], in_channels=1,
                           base_channels=ae_config["base_channels"], stream_configs=stream_configs)
        decoder = Decoder(output_size=ae_config["size"], out_channels=1,
                           base_channels=ae_config["base_channels"], latent_channels=recon_stream.channels,
                           latent_spatial_size=recon_stream.spatial_size)
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders={"shared": decoder},
                                     stream_configs=stream_configs).to(device)
    else:
        encoder = Encoder(input_size=ae_config["size"], in_channels=1,
                           base_channels=ae_config["base_channels"], stream_configs=stream_configs)
        decoders = {}
        for stream_name, decoder_key in decoder_for_stream.items():
            stream_cfg = stream_configs[stream_name]
            decoders[decoder_key] = Decoder(
                output_size=ae_config["size"], out_channels=1,
                base_channels=ae_config["base_channels"], latent_channels=stream_cfg.channels,
                latent_spatial_size=stream_cfg.spatial_size,
            )
        ae = MultiStreamAutoencoder(encoders={"shared": encoder}, decoders=decoders,
                                     stream_configs=stream_configs,
                                     decoder_for_stream=decoder_for_stream).to(device)
    ae.load_state_dict(ae_checkpoint["model_state"])
    ae.eval()
    ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]

    f_theta = LatentDynamics(
        latent_channels=lds_config["latent_channels"], n_theta=lds_config["n_theta"],
        latent_spatial=lds_config.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
        hidden_dim=lds_config["hidden_dim"], n_hidden_layers=lds_config["n_hidden_layers"],
    ).to(device)
    f_theta.load_state_dict(lds_checkpoint["model_state"])
    f_theta.eval()

    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=ae_encoder, device=device, window_length=window_length,
        min_step=min_step, min_stdev_phi=min_stdev_phi, encode_both_streams=True,
    )
    print(f"Evaluating {len(dataset)} test windows (window_length={window_length})...")

    d = compute_f_diagnostics(dataset, f_theta, device)

    print(f"\n{'=' * 70}\nf_theta diagnostic: {lds_checkpoint_path.name}\n{'=' * 70}")

    stuck_fractions = check_dead_relus(f_theta, dataset, device)
    if stuck_fractions:
        print("\nStuck-activation check (fraction of units with NEGATIVE pre-activation "
              "for EVERY sample in a real batch -- not just often-negative; for plain ReLU "
              "this means permanently dead/zero-gradient, for LeakyReLU it means stuck in "
              "the attenuated negative-slope regime for every input):")
        for name, frac in stuck_fractions.items():
            flag = "  <-- ENTIRE TRUNK OUTPUT IS STUCK" if name == "trunk_output" and frac > 0.99 else ""
            print(f"  {name:16s}: {frac:.1%}{flag}")
        if stuck_fractions.get("trunk_output", 0.0) > 0.99:
            print("  Trunk output is (near-)entirely stuck -- f_theta.f(...) can only ever "
                  "return (approximately) its own final-layer bias, regardless of z0/z1/theta. "
                  "This alone explains a constant, zero-variance f() output below, if seen.")

    print("\n||f(z0, z1, theta)|| -- the raw curvature prediction, independent of dt:")
    _print_summary("f1_real (step 1)", d["f1_real_norm"])
    _print_summary("f2_real (step 2, real z0)", d["f2_real_norm"])
    _print_summary("f2_chained (step 2, chained z0)", d["f2_chained_norm"])

    ratio = d["f2_chained_norm"] / np.maximum(d["f2_real_norm"], 1e-12)
    print(f"\n  f2_chained / f2_real ratio (same dt2, same z1, only z0 differs):")
    print(f"    mean={ratio.mean():.4f}  median={np.median(ratio):.4f}  max={ratio.max():.4f}  "
          f"(>>1 means f_theta blows up specifically on its OWN predicted z0, not on real data "
          f"at the same dt -- points at off-distribution behavior, not intrinsic dt-dependence)")

    print("\nActual contribution added to z0_hat (||f * dt^2/2||) -- what really lands in the prediction:")
    _print_summary("f1_real contrib", d["f1_real_contrib"])
    _print_summary("f2_real contrib", d["f2_real_contrib"])
    _print_summary("f2_chained contrib", d["f2_chained_contrib"])

    print(f"\nz0 step-1 prediction error ||z0_hat(t1) - z0(t1)|| (how far off-distribution "
          f"step 2's own chained input actually is):")
    _print_summary("z0_step1_error", d["z0_step1_error"])

    print(f"\n||f2_actual|| / ||f2_ideal|| -- OVER-shooting (>1) vs UNDER-shooting (<1) the "
          f"correction f_theta's own input actually calls for (see this function's own "
          f"docstring -- these call for OPPOSITE fixes, so this distinction matters):")
    _print_summary("ratio_real (baseline)", d["ratio_real"])
    _print_summary("ratio_chained", d["ratio_chained"])
    print(f"  ratio_real median far from 1.0 would mean f_theta's basic single-step "
          f"calibration is already off, independent of any chaining/off-distribution issue.")

    print(f"\ncos_sim(f2_actual, f2_ideal) -- directional correctness, independent of "
          f"magnitude (near +1 = at least pointing the right way; near 0 or negative is a "
          f"qualitatively worse failure than a magnitude error alone):")
    _print_summary("cos_sim_real (baseline)", d["cos_sim_real"])
    _print_summary("cos_sim_chained", d["cos_sim_chained"])

    print(f"\nSame ratio_real/cos_sim_real, broken down by dt2 decade -- the ideal-target "
          f"formula divides by dt2^2/2, so ANY noise in the numerator (including z1's own "
          f"approximation error -- z1 is never the exact derivative) gets massively amplified "
          f"for SMALL dt2. If the aggregate numbers above look bad mostly because of that "
          f"amplification, this breakdown should show badness concentrated at small dt2, NOT "
          f"uniform across all scales -- a genuine calibration problem would show up everywhere:")
    _print_binned_by_dt2("ratio_real (median)", d["dt2"], d["ratio_real"])
    _print_binned_by_dt2("cos_sim_real (median)", d["dt2"], d["cos_sim_real"])

    corr_f1_dt1 = _log_corr(d["dt1"], d["f1_real_norm"])
    corr_f2real_dt2 = _log_corr(d["dt2"], d["f2_real_norm"])
    corr_f2chained_dt2 = _log_corr(d["dt2"], d["f2_chained_norm"])
    corr_f2chained_err = _log_corr(d["z0_step1_error"], d["f2_chained_norm"])
    print(f"\ncorrelation(log10(dt1), log10(||f1_real||))        = {_fmt_corr(corr_f1_dt1)}")
    print(f"correlation(log10(dt2), log10(||f2_real||))        = {_fmt_corr(corr_f2real_dt2)}")
    print(f"correlation(log10(dt2), log10(||f2_chained||))     = {_fmt_corr(corr_f2chained_dt2)}")
    print(f"correlation(log10(z0_step1_error), log10(||f2_chained||)) = {_fmt_corr(corr_f2chained_err)}")
    print("(if the LAST correlation is much stronger than the dt-based ones above, ||f|| tracks "
          "how wrong the chained input is more than it tracks dt itself -- direct evidence for "
          "off-distribution blowup over intrinsic dt-dependence)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(d["f2_real_norm"], d["f2_chained_norm"], s=8, alpha=0.4)
    lims = [0, max(d["f2_real_norm"].max(), d["f2_chained_norm"].max()) * 1.05]
    axes[0].plot(lims, lims, "k--", linewidth=1, label="y = x (no difference)")
    axes[0].set_xlabel("||f2_real|| (real z0 input)")
    axes[0].set_ylabel("||f2_chained|| (chained z0 input)")
    axes[0].set_title("Same dt2/theta -- only z0 differs")
    axes[0].legend()

    axes[1].scatter(d["dt2"], d["f2_real_norm"], s=8, alpha=0.4, label="f2_real")
    axes[1].scatter(d["dt2"], d["f2_chained_norm"], s=8, alpha=0.4, label="f2_chained")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("dt2")
    axes[1].set_ylabel("||f||")
    axes[1].set_title("||f|| vs dt2, real vs chained z0")
    axes[1].legend()

    axes[2].scatter(d["z0_step1_error"], d["f2_chained_norm"], s=8, alpha=0.4)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("||z0_hat(t1) - z0(t1)|| (step-1 error)")
    axes[2].set_ylabel("||f2_chained||")
    axes[2].set_title("||f2_chained|| vs how off-distribution z0 is")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved figure to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lds-checkpoint", type=Path, required=True,
            help="a stage3a or stage3b checkpoint -- window_length=3 is used for the "
                 "diagnostic regardless of what this checkpoint was itself trained at")
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--min-stdev-phi", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None,
            help="default: <repo root>/output/stage3/<lds checkpoint name>-f_theta_diagnostic.png")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    check_f_theta(
        lds_checkpoint_path=args.lds_checkpoint, min_step=args.min_step,
        min_stdev_phi=args.min_stdev_phi, output_path=args.output, device=args.device,
    )


if __name__ == "__main__":
    main()
