"""
Load a trained LDS checkpoint (and the AE checkpoint it was trained
against), predict z(t+dt) for real held-out test-set transitions,
decode the prediction, and compare against the actual x(t+dt).

Shown as explicit CHANGE rather than raw next-states -- state(t), the
real delta x(t+dt)-x(t), the predicted delta, and the error -- since at
small dt the raw states can look nearly identical, making the actual
predicted dynamics hard to see directly. Also shows the AE's OWN
reconstruction of the true next state alongside the loss numbers, so
LDS prediction error can be told apart from AE reconstruction error
rather than the two being conflated into one number.

REPRODUCIBLE COMPARISON: by default, samples are randomly picked from
the (min_step/min_stdev_phi-filtered) test set -- but since changing
those filters changes WHICH snapshots exist in the dataset at all, the
same --seed does NOT guarantee the same underlying snapshots across
runs with different filter settings, making before/after comparisons
across parameter changes impossible. Every random run prints its exact
picks as 'run_dir:step0:step1:...:stepN' (the full window, e.g. 4 steps
for a checkpoint trained at n_rollout_steps=3 -- NOT just the first two
steps, which would silently test only 1-step quality regardless of how
many rollout steps the checkpoint was actually trained at); pass those
back in via --fixed-windows (repeatable) to see the EXACT SAME snapshots
again, computed fresh from the raw files and the frozen encoder/decoder --
entirely bypassing dataset filtering, so it works regardless of what
min_step/min_stdev_phi the comparison run uses.

COMPARING CHECKPOINTS AT DIFFERENT n_rollout_steps (e.g. 3a vs 3b): pass
the SAME --fixed-windows list to both runs. Each checkpoint truncates
every window down to its own window_length (n_rollout_steps+1) before
use, so a longer window (e.g. 3b's own 3-timestep picks) works directly
against a checkpoint that needs fewer steps (e.g. 3a's 2) -- both figures
then start from the identical (run_dir, step0), for a true side-by-side
comparison, rather than each stage picking its own independent random
samples.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_rollout \
        --size 64 --latent-channels 4 --stats-weight 0.01 --n-rollout-steps 3
    python -m evaluation.check_rollout \
        --size 64 --latent-channels 4 --stats-weight 0.01 --n-rollout-steps 3 \
        --fixed-windows "../../datasets/64x64/T800_n050_s79:100000:110000:120000" ...
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from models.constants import LATENT_SPATIAL_SIZE
from models.latent_dynamics import LatentDynamics
from models.latent_streams import resolve_stream_configs_from_checkpoint_config
from evaluation._window_parsing import parse_fixed_window
from training.checkpoint_components import build_ae_from_checkpoint
from training.datasets import MicrostructureEvolutionDataset
from training.losses import ReconLoss
from utils import load_datasets as load
from utils.naming import lds_checkpoint_name

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



def compute_sample(run_dir: Path, steps: list[int], ae, f_theta,
                    ae_config: dict, device: torch.device, z1_resync: bool = True):
    """
    Everything needed for one row of the comparison figure, computed
    FRESH from the raw snapshot files and the frozen encoder/decoder --
    no dependence on any MicrostructureEvolutionDataset filtering, so
    this works identically regardless of what min_step/min_stdev_phi a
    training run used.

    steps: the FULL window (len == window_length, e.g. 4 steps for a
    checkpoint trained at n_rollout_steps=3) -- chains f_theta.rollout()
    across every intermediate transition (steps[0]->steps[1]->...->
    steps[-1]), matching EXACTLY the multi-step prediction
    n_rollout_steps>1 training actually optimizes. Previously this
    always compared only steps[0]->steps[1] regardless of how many
    rollout steps a checkpoint was trained at -- silently testing
    1-step quality even for n_rollout_steps=3 checkpoints, never the
    actual compounding behavior those checkpoints exist to improve.
    Reports the comparison at the FINAL step (steps[0] -> steps[-1]),
    the full cumulative prediction, exactly as decided.
    """
    metadata = load.read_metadata(run_dir / "metadata.txt")
    nx, ny = ae_config["size"], ae_config["size"]

    x_t_raw = load.read_phi_half(run_dir / load.snapshot_filename(steps[0]), nx, ny)
    x_next_raw = load.read_phi_half(run_dir / load.snapshot_filename(steps[-1]), nx, ny)

    dt_total = (steps[-1] - steps[0]) * metadata.dt  # total elapsed span, for display only
    theta_val = metadata.temperature - metadata.T0  # see LatentDynamics/dataset docstrings

    with torch.no_grad():
        # ae.encoder(x) returns dict[str, Tensor] (one entry per latent
        # stream -- see models/latent_streams.py); resolved from
        # ae_config (already a parameter here) rather than assumed, so
        # this works regardless of what the recon stream is actually
        # named. ae here is the FULL container (Autoencoder exposes
        # .encoder/.decoder directly, MultiStreamAutoencoder only has
        # .encoders["shared"]/.decoders[...] -- see those classes'
        # own docstrings on why), resolved locally since this function
        # has its own scope, separate from check_rollout()'s own
        # ae_encoder/ae_decoder helpers. ae_decoder specifically must
        # be recon_stream_name's own pathway decoder (not a generic
        # "shared" fallback) -- z0_t below is THAT stream's own latent,
        # and decoders may now be genuinely separate per stream (stage
        # 1b's own format).
        _, recon_stream_name = resolve_stream_configs_from_checkpoint_config(ae_config)
        ae_encoder = ae.encoder if hasattr(ae, "encoder") else ae.encoders["shared"]
        ae_decoder = (ae.pathways[recon_stream_name].decoder if hasattr(ae, "pathways")
                      else ae.decoder)

        # EVERY step in the window gets encoded, not just the endpoints
        # -- rollout() teacher-forces the REAL z1 at every step (see
        # LatentDynamics' own class docstring), so the real z1 at each
        # intermediate step is needed here, not just at steps[0].
        x_all = torch.stack([
            torch.from_numpy(load.read_phi_half(run_dir / load.snapshot_filename(step), nx, ny))
            for step in steps
        ]).unsqueeze(1).to(device)  # (len(steps), 1, ny, nx)
        # theta broadcast across all len(steps) rows -- every frame here
        # comes from the SAME run_dir, hence the same theta_val, unlike
        # x_all's own per-frame variation. Passed unconditionally
        # (Encoder.forward accepts it regardless of whether any stream
        # actually needs it); needed because Encoder computes every
        # stream in one pass internally, so a theta-conditioned "deriv"
        # stream requires it even though z0_t/z0_next_true below only
        # ever read the recon stream's own output.
        theta_encode = torch.full((len(steps), 1), theta_val, dtype=torch.float32, device=device)
        x_all_encoded = ae_encoder(x_all, theta=theta_encode)
        z0_t = x_all_encoded[recon_stream_name][0:1]  # only the STARTING z0 is a rollout() input
        z1_sequence = x_all_encoded["deriv"].unsqueeze(0)  # (1, len(steps), C, 8, 8) -- every step
        z0_next_true = x_all_encoded[recon_stream_name][-1:]

        # Per-TRANSITION dts, chained via rollout() -- NOT one big dt
        # covering the whole span. A single f_theta call with a large
        # dt is a fundamentally different (and untrained-for) operation
        # from n_rollout_steps chained calls at the actual per-step dts.
        dt_per_step = [(steps[i + 1] - steps[i]) * metadata.dt for i in range(len(steps) - 1)]
        dts = torch.tensor([dt_per_step], dtype=torch.float32, device=device)
        theta = torch.tensor([[theta_val]], dtype=torch.float32, device=device)

        z0_hat_full = f_theta.rollout(z0_t, z1_sequence, dts, theta,
                                       z1_resync=z1_resync)
        z0_next_pred = z0_hat_full[:, -1]

        x_next_pred = ae_decoder(z0_next_pred)[0, 0].cpu().numpy()
        x_next_ae_baseline = ae_decoder(z0_next_true)[0, 0].cpu().numpy()

    return x_t_raw, x_next_raw, x_next_pred, x_next_ae_baseline, dt_total, dt_per_step


def _correlation_pct(predicted, real) -> float | None:
    """
    Pearson correlation (as a percentage) between predicted and real
    Delta x, flattened. Deliberately separate from the loss (L1,
    magnitude-sensitive): a prediction that gets the SHAPE right but
    the magnitude weak or wrong -- a real failure mode seen in practice
    ('right direction but weak') -- can show high correlation despite a
    middling loss, which the loss number alone doesn't distinguish from
    a prediction that's wrong in both shape and magnitude.

    Returns None (not nan or 0) if either array is numerically constant
    -- correlation is undefined there (zero std, division by zero),
    and a real number would misleadingly suggest a meaningful value.
    """
    predicted_flat, real_flat = predicted.flatten(), real.flatten()
    if np.std(predicted_flat) < 1e-12 or np.std(real_flat) < 1e-12:
        return None
    return float(np.corrcoef(predicted_flat, real_flat)[0, 1]) * 100


def _format_small(value: float, exponent: int = -3, precision: int = 1) -> str:
    """
    Formats a small loss value at a FIXED power of ten (default 1e-3),
    e.g. 0.0003 -> '0.3e-3' -- more legible than '%.4f' (0.0003) for
    values that are all clustered in the same 1e-4..1e-3 order of
    magnitude, which is what AE (autoencoder reconstruction baseline)
    loss values look like throughout this pipeline's logs. A fixed
    exponent (rather than each value picking its own, e.g. '%.1e') also
    keeps mantissas directly comparable across panels/figures at a
    glance, without each one needing to be re-normalized by eye first.

    precision: mantissa decimal digits, default 1. A value genuinely
    smaller than the fixed exponent (e.g. ~1e-5 displayed at the
    default 1e-3 exponent) rounds to a meaningless "0.0e-3" at
    precision=1 -- pass a higher precision for panels whose values
    routinely sit an order of magnitude or more below the exponent.
    """
    return f"{value / (10 ** exponent):.{precision}f}e{exponent}"


def _padded_bounds(values, factor: float, symmetric: bool = False) -> tuple[float, float]:
    """
    (vmin, vmax) padded by `factor` beyond values' own actual range,
    always including zero -- so a diverging colormap centered at 0
    stays meaningful even for a one-sided distribution (e.g. Delta x
    that happens to be mostly positive in a given window).

    Default (symmetric=False) is deliberately asymmetric (NOT
    +-max(abs(...))): if real Delta x ranges from -0.05 to +0.3, the
    padded range is [-0.06, +0.36], not a symmetric +-0.36 that wastes
    half the color range on a side the real data barely uses.

    symmetric=True instead returns (-M, +M) with M = factor *
    max(|lo|, |hi|). Needed specifically for the real/predicted-Delta-x
    scale these two panels share: when real Delta x is (near-)entirely
    one-signed -- e.g. a purely shrinking domain, real Delta x >= 0
    everywhere -- the default asymmetric bounds correctly reflect that
    real Delta x doesn't use the other side, but PREDICTED Delta x can
    and does land on that other side (over/undershoot), and every such
    value then saturates to the same extreme color regardless of
    magnitude -- a -0.01 miss and a -0.08 miss become visually
    identical. Symmetric bounds keep both signs fully resolved for the
    prediction panel, at the cost of the real-Delta-x panel visibly not
    using half the colormap when real Delta x genuinely is one-sided --
    an intentional trade favoring the prediction panel's readability,
    not an oversight.

    factor=1.2 gives 20% headroom beyond the real range, so a
    prediction that's slightly too high or too negative still shows up
    as a visible, readable color rather than being clipped at the very
    edge of the scale.
    """
    lo, hi = float(values.min()), float(values.max())
    if symmetric:
        magnitude = max(abs(lo), abs(hi), 1e-6)
        return -factor * magnitude, factor * magnitude
    vmin = factor * lo if lo < 0 else min(0.0, lo)
    vmax = factor * hi if hi > 0 else max(0.0, hi)
    # Guard against a degenerate all-one-sided (or all-zero) range,
    # which would make vmin/vcenter/vmax non-strictly-increasing and
    # break TwoSlopeNorm below.
    if vmin >= 0:
        vmin = -1e-6
    if vmax <= 0:
        vmax = 1e-6
    return vmin, vmax


def _error_bounds(real_delta, error, floor_factor: float = 0.25,
                   headroom_factor: float = 1.2) -> tuple[float, float]:
    """
    (vmin, vmax) for the error (predicted - real) panel: the wider of
    two candidates, not just one.

    floor_factor*real_delta's own range is kept as a FLOOR -- what
    keeps the error panel on a fixed, comparable scale across
    different images/checkpoints when the error is genuinely small
    (real_delta is the same fixed reference every image already uses
    for the other two panels, so a small, accurate prediction's error
    panel doesn't jump around from image to image for no reason).

    But deriving the scale ONLY from that floor, never checking the
    actual error array itself, saturates hard whenever a prediction is
    bad enough that its own error exceeds the floor -- exactly the
    case where seeing the true magnitude matters most (a genuinely
    wrong, noise-like prediction has error on the scale of the STATE
    itself, not on the scale of real_delta, e.g. dozens of times
    larger than the 0.25x-real_delta floor was ever sized for). Taking
    the wider of the floor and the actual error range (with its own
    headroom) fixes this without giving up the floor's own benefit for
    the common, small-error case.
    """
    floor_vmin, floor_vmax = _padded_bounds(real_delta, factor=floor_factor)
    actual_vmin, actual_vmax = _padded_bounds(error, factor=headroom_factor)
    return min(floor_vmin, actual_vmin), max(floor_vmax, actual_vmax)


def check_rollout(
    lds_checkpoint_path: Path, n_samples: int = 6, seed: int = 0,
    fixed_windows: list[str] | None = None,
    min_step: int | None = None, min_stdev_phi: float | None = None,
    max_dt: float | None = None, z1_resync: bool | None = None,
    output_path: Path | None = None, device: str | None = None,
) -> tuple[Path, list[str]]:
    """Saves a visual rollout-comparison figure and returns
    (output_path, window_strings) -- window_strings is the exact list of
    'run_dir:step0:...:stepN' windows actually used (whether freshly
    randomly selected or the given fixed_windows, after any truncation),
    in the same format --fixed-windows accepts, so a caller can pass it
    straight into a second check_rollout() call against a DIFFERENT
    checkpoint for a same-samples comparison (see module docstring)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if output_path is None:
        suffix = "" if fixed_windows else f"-seed{seed}"
        output_path = (_PYTHON_ROOT.parent / "output" / "rollout_check_png"
                       / f"{lds_checkpoint_path.stem}{suffix}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lds_checkpoint = torch.load(lds_checkpoint_path, map_location=device, weights_only=True)
    lds_config = lds_checkpoint["config"]
    print(f"Loaded LDS checkpoint from epoch {lds_checkpoint['epoch']}, "
          f"val_loss={lds_checkpoint['val_loss']:.6f}, config={lds_config}")

    ae_checkpoint_path = Path(lds_checkpoint["ae_checkpoint"])
    ae, ae_encoder, ae_checkpoint, stream_configs, recon_stream_name = build_ae_from_checkpoint(
        ae_checkpoint_path, device,
    )
    ae_config = ae_checkpoint["config"]

    f_theta = LatentDynamics(
        latent_channels=lds_config["latent_channels"], n_theta=lds_config["n_theta"],
        latent_spatial=lds_config.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
        hidden_dim=lds_config["hidden_dim"], n_hidden_layers=lds_config["n_hidden_layers"],
        # inf (exact no-op) for any checkpoint saved before dt_cap
        # existed -- same .get()-with-fallback pattern as
        # latent_spatial_size just above. A real, reported bug
        # otherwise: this LatentDynamics reconstruction is SEPARATE
        # from model_assembly.py's own build_models_from_components
        # (which already has this fix) and from
        # evaluation._latent_eval.py's own copy (ditto) -- fixing
        # dt_cap in either of those did NOT fix it here. A checkpoint
        # saved with a real, finite dt_cap would silently evaluate as
        # if dt_cap were still inf, with no error or warning anywhere
        # to indicate the mismatch.
        dt_cap=lds_config.get("dt_cap", float("inf")),
        # n_substeps from the checkpoint too, and for a SHARPER reason than
        # dt_cap: it changes what f_theta MEANS. Rebuilding a model trained
        # at n_substeps=N as n_substeps=1 applies a POINTWISE z1_dot as a
        # one-shot corrector over the whole dt -- the "NOT equivalent"
        # direction train_lds warns about on resume. The weights load
        # cleanly, so nothing else would catch it.
        n_substeps=lds_config.get("n_substeps", 1),
    ).to(device)
    f_theta.load_state_dict(lds_checkpoint["model_state"])
    f_theta.eval()

    data_config = lds_checkpoint.get("data_config")

    if fixed_windows:
        windows = [parse_fixed_window(s) for s in fixed_windows]
        # Lets the SAME --fixed-windows list (e.g. 3b's own 3-timestep
        # picks) be reused across checkpoints trained at DIFFERENT
        # n_rollout_steps for direct, same-starting-point comparison
        # (e.g. 3a vs 3b) -- rather than requiring separately-truncated
        # window lists per checkpoint. Truncates to *this* checkpoint's
        # own window_length so compute_sample() still chains exactly the
        # number of transitions this checkpoint was trained/evaluated at
        # -- e.g. a 3a checkpoint (window_length=2) sees only steps[0:2]
        # from a longer window, reporting its real 1-step quality from
        # that same starting point, not silently testing 2-step rollout
        # on a checkpoint that never trained for it.
        if data_config and "window_length" in data_config:
            target_len = data_config["window_length"]
            truncated = []
            for run_dir, steps in windows:
                if len(steps) > target_len:
                    print(f"  truncating {run_dir}:{':'.join(str(s) for s in steps)} to "
                          f"first {target_len} steps (this checkpoint's own window_length)")
                    steps = steps[:target_len]
                elif len(steps) < target_len:
                    raise ValueError(
                        f"--fixed-windows entry for {run_dir} has only {len(steps)} steps, "
                        f"but this checkpoint needs window_length={target_len} "
                        f"(n_rollout_steps={data_config.get('n_rollout_steps')})"
                    )
                truncated.append((run_dir, steps))
            windows = truncated
        print(f"Using {len(windows)} fixed windows (dataset filtering bypassed entirely)")
    else:
        if data_config is None:
            print("WARNING: this checkpoint has no saved data_config -- falling back to "
                  "min_step=0, min_stdev_phi=None, window_length=2, which may NOT match "
                  "what training actually used.")
            data_config = {"min_step": 0, "min_stdev_phi": None, "window_length": 2}
        min_step = min_step if min_step is not None else data_config["min_step"]
        min_stdev_phi = min_stdev_phi if min_stdev_phi is not None else data_config["min_stdev_phi"]
        # max_dt from the checkpoint too. Without it this picks windows from
        # the FULL dt range for a model trained on a restricted one, so the
        # rollout panels show f_theta extrapolating far outside anything it
        # saw -- its correction goes as f*dt^2/2, so at 125x the trained dt
        # that term is ~15000x too large and the comparison says nothing about
        # the model. Same defect that made check_parameter_dependence report
        # "f_theta is worse on 88% of windows".
        max_dt = max_dt if max_dt is not None else data_config.get("max_dt")
        # z1_resync from the checkpoint's own config unless overridden. This is
        # the ONLY diagnostic in which the flag means anything:
        # check_parameter_dependence evaluates a SINGLE forward() step with the
        # real z1 supplied, so there is nothing to resync. Only a chained
        # rollout can distinguish the two regimes.
        #
        # It is also the comparison the training losses cannot make. With
        # z1_resync=True the encoder hands back the true z1 at every real
        # frame, so errors cannot compound; with False they must. A model
        # trained under False therefore has a WORSE training loss while doing
        # the harder, inference-matching job -- and only evaluating both under
        # the SAME regime says which network is actually better.
        z1_resync = (z1_resync if z1_resync is not None
                      else lds_config.get("z1_resync", True))
        print(f"z1_resync={z1_resync} "
              f"({'from the checkpoint' if z1_resync is None else 'in effect'}): "
              f"{'z1 reset to the encoder value at each real frame' if z1_resync else 'z1 PROPAGATED throughout -- the inference regime'}")
        window_length = data_config["window_length"]

        test_dirs = lds_checkpoint.get("test_dirs") or []
        if not test_dirs:
            raise ValueError(
                f"{lds_checkpoint_path} has no saved test_dirs -- it was likely trained "
                f"with --test-fraction 0."
            )
        test_dirs = load.validate_run_dirs(
            [Path(d) for d in test_dirs], source=str(lds_checkpoint_path),
            min_stdev_phi=min_stdev_phi)

        # Only used to PICK representative (run_dir, steps) windows from the
        # actual filtered test distribution -- compute_sample() then does
        # the real work fresh, independent of this dataset object.
        dataset = MicrostructureEvolutionDataset(
            test_dirs, encoder=ae_encoder, device=device, window_length=window_length,
            max_dt=max_dt,
            min_step=min_step, min_stdev_phi=min_stdev_phi,
        )
        if len(dataset) == 0:
            raise ValueError(f"No windows found in the checkpoint's {len(test_dirs)} "
                              f"test_dirs (after min_step={min_step}/"
                              f"min_stdev_phi={min_stdev_phi} filtering)")

        generator = torch.Generator().manual_seed(seed)
        n_samples = min(n_samples, len(dataset))
        indices = torch.randperm(len(dataset), generator=generator)[:n_samples].tolist()

        windows = []
        for idx in indices:
            run_dir, steps = dataset.window_info(idx)
            windows.append((run_dir, steps))

        print("\nSelected windows -- reuse via --fixed-windows for reproducible comparison:")
        for run_dir, steps in windows:
            print(f"  {run_dir}:{':'.join(str(s) for s in steps)}")
        print()

    recon_loss = ReconLoss()
    n_samples = len(windows)

    fig, axes = plt.subplots(n_samples, 4, figsize=(17, 3.2 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    for row, (run_dir, steps) in enumerate(windows):
        x_t_raw, x_next_raw, x_next_pred, x_next_ae_baseline, dt_total, dt_per_step = compute_sample(
            run_dir, steps, ae, f_theta, ae_config, device, z1_resync=z1_resync,
        )

        x_next_pred_t = torch.from_numpy(x_next_pred).unsqueeze(0).unsqueeze(0)
        x_next_raw_t = torch.from_numpy(x_next_raw).unsqueeze(0).unsqueeze(0)
        x_next_baseline_t = torch.from_numpy(x_next_ae_baseline).unsqueeze(0).unsqueeze(0)

        end_to_end_loss = recon_loss(x_next_pred_t, x_next_raw_t).item()
        ae_baseline_loss = recon_loss(x_next_baseline_t, x_next_raw_t).item()

        # Explicit deltas rather than raw next-states: at small dt the
        # raw x(t) and x(t+dt) can look nearly identical, making the
        # actual predicted dynamics hard to see directly. real_delta
        # and predicted_delta share ONE scale (not each auto-scaled
        # independently) so they're directly visually comparable --
        # that comparison is the whole point of this panel.
        real_delta = x_next_raw - x_t_raw
        predicted_delta = x_next_pred - x_t_raw
        error = predicted_delta - real_delta

        state_scale = max(abs(x_t_raw.min()), abs(x_t_raw.max()), 0.1)
        # Scales derived ONLY from real_delta (never from predicted_delta
        # or error) -- so they're predictable and directly comparable
        # across different checkpoints/runs (e.g. with vs without stage
        # 2), and a prediction that's genuinely off shows up as visible
        # saturation against a fixed reference, rather than being hidden
        # by a scale that stretches to accommodate however wrong it is.
        #
        # symmetric=True: real Delta x is often (near-)entirely
        # one-signed for a given window (e.g. a purely shrinking
        # domain), but the PREDICTED Delta x can land on the other side
        # regardless -- an asymmetric scale then collapses that whole
        # side down near zero, and every such over/undershoot saturates
        # to the same extreme color no matter its actual magnitude.
        # Symmetric bounds cost the real-Delta-x panel visibly not
        # using half the colormap when real Delta x really is
        # one-sided -- an intentional trade for column 3's readability.
        # See _padded_bounds' own docstring for the full reasoning.
        delta_vmin, delta_vmax = _padded_bounds(real_delta, factor=1.2, symmetric=True)
        # Error is intrinsically signed (predicted - real can go either
        # way regardless of which side real_delta itself favors), so it
        # does NOT get the symmetric treatment above -- asymmetric
        # bounds remain the right shape here.
        error_vmin, error_vmax = _error_bounds(real_delta, error)
        delta_norm = TwoSlopeNorm(vmin=delta_vmin, vcenter=0.0, vmax=delta_vmax)
        error_norm = TwoSlopeNorm(vmin=error_vmin, vcenter=0.0, vmax=error_vmax)

        n_steps_shown = len(steps) - 1
        axes[row, 0].imshow(x_t_raw, cmap="RdBu", vmin=-state_scale, vmax=state_scale)
        span_label = f"{run_dir.name}:{steps[0]}\u2192{steps[-1]} ({n_steps_shown} step{'s' if n_steps_shown != 1 else ''})"
        dt_label = (f"dt={dt_total:.1f}" if n_steps_shown == 1
                    else f"dt_total={dt_total:.1f} (" + "+".join(f"{d:.1f}" for d in dt_per_step) + ")")
        axes[row, 0].set_title(f"state(t)\n{span_label}\n{dt_label}" if row == 0
                                else f"{span_label}\n{dt_label}", fontsize=9)
        axes[row, 1].imshow(real_delta, cmap="RdBu", norm=delta_norm)
        axes[row, 1].set_title(f"real \u0394x\nscale=[{delta_vmin:.3f}, {delta_vmax:.3f}]"
                                if row == 0 else f"scale=[{delta_vmin:.3f}, {delta_vmax:.3f}]",
                                fontsize=10)
        axes[row, 2].imshow(predicted_delta, cmap="RdBu", norm=delta_norm)
        corr_pct = _correlation_pct(predicted_delta, real_delta)
        corr_str = f"{corr_pct:.0f}%" if corr_pct is not None else "n/a"
        ae_str = _format_small(ae_baseline_loss)
        axes[row, 2].set_title(
            f"predicted \u0394x\nloss={end_to_end_loss:.4f} (AE={ae_str}), corr={corr_str}"
            if row == 0 else
            f"loss={end_to_end_loss:.4f} (AE={ae_str}), corr={corr_str}", fontsize=10
        )
        im_error = axes[row, 3].imshow(error, cmap="RdBu", norm=error_norm)
        axes[row, 3].set_title(f"error\nscale=[{error_vmin:.3f}, {error_vmax:.3f}]" if row == 0
                                else f"scale=[{error_vmin:.3f}, {error_vmax:.3f}]", fontsize=10)
        fig.colorbar(im_error, ax=axes[row, 3], fraction=0.046)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved rollout comparison to {output_path} ({n_samples} samples)")
    window_strings = [f"{run_dir}:{':'.join(str(s) for s in steps)}" for run_dir, steps in windows]
    return output_path, window_strings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=None,
                         help="grid size (square only) -- config.txt is never read. Needed ONLY "
                              "to RECONSTRUCT a checkpoint path from --latent-channels/"
                              "--stats-weight/--n-rollout-steps; with --lds-checkpoint given it "
                              "is unused, and requiring it there was pure friction")
    parser.add_argument("--latent-channels", type=int, default=None, help="see --size")
    parser.add_argument("--stats-weight", type=float, default=None)
    parser.add_argument("--n-rollout-steps", type=int, default=None, help="see --size")
    parser.add_argument("--lds-checkpoint", type=Path, default=None,
            help="direct path override, if you'd rather specify the checkpoint this way "
                 "instead of by --size/--latent-channels/--stats-weight/--n-rollout-steps")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0,
                         help="which test-set windows to display (ignored if --fixed-windows "
                              "is given), not the train/val/test split itself, which is "
                              "fixed and loaded from the checkpoint")
    parser.add_argument("--fixed-windows", type=str, nargs="+", default=None,
                         help="exact 'run_dir:step0:step1:...:stepN' windows to display "
                              "(repeatable), bypassing random dataset-based selection "
                              "entirely -- for reproducible before/after comparisons across "
                              "different --min-step/--min-stdev-phi training runs. Every "
                              "random run prints its picks in this exact format.")
    parser.add_argument("--min-step", type=int, default=None,
                         help="override the checkpoint's recorded min_step, if given "
                              "(ignored if --fixed-windows is given)")
    parser.add_argument("--min-stdev-phi", type=float, default=None,
                         help="override the checkpoint's recorded min_stdev_phi, if given "
                              "(ignored if --fixed-windows is given)")
    parser.add_argument("--output", type=Path, default=None,
            help="default: <repo root>/output/rollout_check_png/<lds checkpoint name>"
                 "[-seed<N> if random-sampling].png -- named after the checkpoint "
                 "(and seed) so different checks don't collide")
    parser.add_argument("--z1-resync", action=argparse.BooleanOptionalAction, default=None,
                         help="default: whatever the checkpoint was TRAINED with. "
                              "--no-z1-resync propagates z1 throughout the rollout instead of "
                              "resetting it to the encoder value at each real frame -- the "
                              "regime inference is actually in. Evaluating two models in the "
                              "SAME regime is the only way to compare them: a model trained "
                              "with z1_resync=False has a worse TRAINING loss purely because "
                              "its objective is harder. Meaningless in "
                              "check_parameter_dependence, which evaluates a single forward() "
                              "step with the real z1 supplied")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.lds_checkpoint is None:
        missing = [n for n, v in [("--size", args.size),
                                   ("--latent-channels", args.latent_channels),
                                   ("--stats-weight", args.stats_weight),
                                   ("--n-rollout-steps", args.n_rollout_steps)] if v is None]
        if missing:
            raise ValueError(
                f"Provide either --lds-checkpoint directly, or the pieces needed to "
                f"reconstruct its path: --size, --latent-channels, --stats-weight, "
                f"--n-rollout-steps (missing: {', '.join(missing)})."
            )
        name = lds_checkpoint_name(args.size, args.latent_channels, args.stats_weight,
                                    args.n_rollout_steps)
        args.lds_checkpoint = _PYTHON_ROOT / "checkpoints" / "stage3" / f"{name}.pt"
        print(f"Reconstructed checkpoint path: {args.lds_checkpoint}")

    check_rollout(
        lds_checkpoint_path=args.lds_checkpoint, n_samples=args.n_samples, seed=args.seed,
        fixed_windows=args.fixed_windows, min_step=args.min_step,
        min_stdev_phi=args.min_stdev_phi, output_path=args.output, device=args.device,
        z1_resync=args.z1_resync,
    )


if __name__ == "__main__":
    main()