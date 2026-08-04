"""
The shared data-collection layer for latent-space diagnostics: load a
checkpoint (AE-family or LDS), build the cached-latent window dataset,
and evaluate every window into flat per-window arrays.

Extracted from check_parameter_dependence.py so that other diagnostics
can reuse it instead of re-deriving it. check_deriv_temperature.py, for
one, builds a BYTE-IDENTICAL MicrostructureEvolutionDataset(...,
encode_both_streams=True) call and pays the whole encode cost a second
time when run alongside.

_EvaluationResults' per-window arrays are all aligned 1:1 and in dataset
order (the loader is unshuffled), which is what lets a consumer join
them against anything else indexed the same way -- see
check_parameter_dependence's own oracle attribution, which walks
dataset._index in that same order.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.constants import LATENT_SPATIAL_SIZE
from models.latent_dynamics import LatentDynamics
from training.checkpoint_components import build_ae_from_checkpoint
from training.datasets import MicrostructureEvolutionDataset
from utils import load_datasets as load

# ensure_lds_checkpoint is imported INSIDE _load_models_and_dataset, not
# here -- orchestration.checkpoint_identification imports from training/,
# and hoisting it to module scope risks a circular import. Kept as the
# original file had it.

# Same anchor as every evaluation/ script: paths are built from the file's
# own location, never the process CWD.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/


@dataclass
class _LoadedContext:
    """Everything check_parameter_dependence()'s own setup phase (checkpoint
    loading, model/dataset construction, output-path resolution) produces
    that the rest of the pipeline (_evaluate_windows/_print_summary_statistics/
    _build_and_save_figures) needs -- bundled rather than threaded through as
    a long, individually-named parameter list on each of those, since the
    full set is used piecemeal across all three."""
    device: torch.device
    euler_only: bool
    output_path: Path
    dz0dt_output_path: Path
    dt_dependence_output_path: Path
    lds_checkpoint_path: Path  # post ensure_lds_checkpoint conversion, NOT the caller's original
    ae_config: dict
    dataset: "MicrostructureEvolutionDataset"
    ae_decoder: torch.nn.Module
    f_theta: "LatentDynamics"


@dataclass
class _EvaluationResults:
    """Every per-window array (plus the two running-sum scalars) produced by
    _evaluate_windows()'s own pass over the dataset -- the shared input both
    _print_summary_statistics() and _build_and_save_figures() read from,
    after the euler_only substitution (see its own comment, preserved below)
    has already been applied to latent_losses/pixel_losses/latent_losses_signed."""
    dts: np.ndarray
    temperatures: np.ndarray
    noises: np.ndarray
    length_scales: np.ndarray
    run_dirs: list
    latent_losses: np.ndarray
    pixel_losses: np.ndarray
    euler_losses: np.ndarray
    euler_pixel_losses: np.ndarray
    latent_losses_signed: np.ndarray
    euler_losses_signed: np.ndarray
    abs_steps: np.ndarray
    dz0_signed: np.ndarray
    dz0_abs: np.ndarray
    dz0dt_signed: np.ndarray
    dz0dt_abs: np.ndarray
    signed_residual_sum: torch.Tensor
    n_total: int


@dataclass
class _DerivedStats:
    """A handful of values _print_summary_statistics() computes for its own
    console output that _build_and_save_figures()'s panels ALSO need (panel
    titles echo the same correlation percentages the console printed; the
    bubble panel [0,2] plots the same per-(temperature,noise)-point
    aggregation the runs-report is built alongside). Kept separate from
    _EvaluationResults -- these are DERIVED from it by
    _print_summary_statistics, not raw per-window data _evaluate_windows
    itself produces. corr_dt_pixel is None whenever decode=False (see its
    own assignment site below for why)."""
    corr_dt_pixel: float | None
    # Per-window oracle/causal residuals (NaN where a window has no
    # preceding frame), aligned 1:1 with results.* -- see
    # _print_oracle_z1_attribution. None when no dataset was passed.
    oracle_per_window: dict | None
    corr_noise_abs: float
    corr_noise_signed: float
    corr_temp_abs: float
    corr_temp_signed: float
    run_mean_loss: np.ndarray
    run_n_windows: np.ndarray
    run_noises: np.ndarray
    run_temps: np.ndarray


def _load_ae_f_theta_and_dataset(
    lds_checkpoint_path: Path, min_step: int | None, min_stdev_phi: float | None,
    min_passing_steps: int | None, base_path: Path | None, size: int | None,
    ae_stats_weight: float | None, hidden_dim: int, n_hidden_layers: int,
    condition_on_theta: bool | None, euler_only: bool | None, device: str | None,
    window_length_override: int | None = None, announce_euler_only: bool = True,
    max_dt: float | None = None, latent_cache_dir: Path | str | None = None,
):
    """
    The MODEL/DATASET half of check_parameter_dependence()'s own setup
    phase -- resolves device, converts an AE-family checkpoint via
    ensure_lds_checkpoint if needed, and builds the ae/f_theta/dataset
    this whole evaluation runs against. Split out from
    _load_models_and_dataset (which wraps this with THAT function's own
    three-figure output_path resolution) specifically so check_f_theta.py
    can reuse this half without being forced to also resolve
    check_parameter_dependence.py's own dz0dt_output_path/
    dt_dependence_output_path, which it has no use for at all (it saves
    exactly one figure, under its own, stage-folder-aware default --
    see check_f_theta() itself).

    window_length_override: check_f_theta.py's own diagnostic needs
    EXACTLY 2 real transitions (t0->t1->t2) regardless of what
    n_rollout_steps this checkpoint was actually trained at -- e.g. a
    stage-3a (n_rollout_steps=1, window_length=2) checkpoint's own
    f_theta can still be probed this way. None (the default) uses the
    checkpoint's own saved window_length unchanged, matching
    check_parameter_dependence()'s own, original behavior exactly.

    announce_euler_only: whether to PRINT the "EULER-ONLY MODE" banner
    when an AE-family checkpoint gets converted. True for
    check_parameter_dependence.py, whose report genuinely does swap
    every 'full' number for its euler-only equivalent and skip
    full-vs-euler-only comparisons. FALSE for check_f_theta.py, which
    has no full-vs-euler-only distinction at all -- printing it there
    was a real, reported regression: the banner actively misdescribes
    what that script does, claiming a substitution it never performs.
    ensure_lds_checkpoint's own NOTE (printed just above, unconditional)
    already explains the untrained-f_theta situation accurately for
    both callers, so suppressing this banner loses no information.

    Returns (device, euler_only, lds_checkpoint_path, ae_config,
    dataset, ae_decoder, f_theta) -- device (now RESOLVED to a real
    torch.device) and lds_checkpoint_path are echoed back since device
    resolution and ensure_lds_checkpoint's own conversion can both
    change them from what the caller passed in.
    """

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        # See this function's own docstring on why this re-check exists,
        # rather than trusting device unconditionally.
        print("WARNING: device='cuda' was requested (or defaulted to, from an "
              "argparse default computed at a DIFFERENT time than this actual "
              "run), but torch.cuda.is_available() is False right now -- falling "
              "back to CPU instead of letting torch.load() fail with a confusing "
              "deserialization error. If this is unexpected, check that CUDA is "
              "actually usable from THIS environment specifically (e.g. running "
              "from the command line vs. an IDE's own kernel can pick up a "
              "different Python/CUDA environment).")
        device = torch.device("cpu")

    from orchestration.checkpoint_identification import ensure_lds_checkpoint
    _original_checkpoint_path = lds_checkpoint_path
    lds_checkpoint_path = ensure_lds_checkpoint(
        lds_checkpoint_path, base_path=base_path, size=size, device=device,
        min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
        ae_stats_weight=ae_stats_weight, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
        condition_on_theta=condition_on_theta,
    )
    # ensure_lds_checkpoint returns its OWN input UNCHANGED for an
    # already-stage-3 checkpoint, and a NEW (tempfile) path only when it
    # actually converted something -- so this comparison is a reliable,
    # side-effect-free way to detect "did conversion actually happen"
    # without ensure_lds_checkpoint needing to return anything more than
    # a path.
    was_converted = lds_checkpoint_path != _original_checkpoint_path
    if euler_only is None:
        euler_only = was_converted
    if euler_only and announce_euler_only:
        print("EULER-ONLY MODE: every 'full' (f_theta-corrected) number below is replaced by "
              "its euler-only (z0+z1*dt, no f_theta) equivalent, and full-vs-euler-only "
              "comparisons are skipped -- " + ("f_theta is untrained (AE-family checkpoint "
              "converted above)." if was_converted else "requested explicitly."))

    lds_checkpoint = torch.load(lds_checkpoint_path, map_location=device, weights_only=True)
    lds_config = lds_checkpoint["config"]

    data_config = lds_checkpoint.get("data_config")
    if data_config is None:
        print("WARNING: checkpoint has no saved data_config -- falling back to "
              "min_step=0, min_stdev_phi=None, min_passing_steps=None, window_length=2 "
              "(may not match training).")
        data_config = {"min_step": 0, "min_stdev_phi": None, "window_length": 2}
    min_step = min_step if min_step is not None else data_config["min_step"]
    min_stdev_phi = min_stdev_phi if min_stdev_phi is not None else data_config["min_stdev_phi"]
    # .get(), not [...]: a checkpoint trained before min_passing_steps existed
    # at all (see train_lds.py's own history) has no such key in its saved
    # data_config -- None (its own default, meaning "no whole-run filtering")
    # is the correct fallback, not a KeyError.
    min_passing_steps = (min_passing_steps if min_passing_steps is not None
                          else data_config.get("min_passing_steps"))
    # max_dt from the checkpoint too. Omitting it was a REPORTED bug with a
    # large, silent effect: f_theta's contribution is f*dt^2/2, so evaluating a
    # model trained with max_dt=200 across dt up to 25000 inflates that term by
    # (25000/200)^2 = 15625. The diagnostic then reported "f_theta makes the
    # prediction WORSE on 88% of windows" -- and the observed ratio of means,
    # 35915, matched the extrapolation factor rather than anything about
    # f_theta's quality. The same run at stage 3a (max_dt=150) gave 24315
    # against a predicted 27778: two independent stages, both explained.
    #
    # Evaluating outside the trained dt range is a legitimate thing to WANT,
    # which is why an explicit override still wins -- but it must be chosen,
    # not inherited by omission.
    max_dt = max_dt if max_dt is not None else data_config.get("max_dt")
    window_length = data_config["window_length"]
    if window_length_override is not None:
        window_length = window_length_override
    # Say plainly whether the checkpoint's own value is in force. The old
    # wording, "unless overridden above", promised an override that did not
    # exist -- check_parameter_dependence had no max_dt parameter at all, so
    # the diagnostic could only ever look INSIDE the range f_theta was trained
    # on and therefore could not answer whether that range was set too tightly.
    _own = data_config.get("max_dt")
    _max_dt_provenance = (
        "(from the checkpoint's own data_config)" if max_dt == _own
        else f"(OVERRIDDEN -- the checkpoint's own is {_own}, so this evaluation is "
             f"deliberately OFF-DISTRIBUTION and is measuring the extrapolation)")
    print(f"min_step={min_step}  min_stdev_phi={min_stdev_phi}  min_passing_steps={min_passing_steps}"
          f"{'' if max_dt is None else f'  max_dt={max_dt}'} "
          f"{_max_dt_provenance}")

    test_dirs = lds_checkpoint.get("test_dirs") or []
    if not test_dirs:
        raise ValueError(f"{lds_checkpoint_path} has no saved test_dirs")
    test_dirs = [Path(d) for d in test_dirs]

    ae_checkpoint_path = Path(lds_checkpoint["ae_checkpoint"])
    ae, ae_encoder, ae_checkpoint, stream_configs, recon_stream_name = build_ae_from_checkpoint(
        ae_checkpoint_path, device,
    )
    ae_config = ae_checkpoint["config"]
    ae_decoder = (ae.pathways[recon_stream_name].decoder if hasattr(ae, "pathways")
                  else ae.decoder)

    f_theta = LatentDynamics(
        latent_channels=lds_config["latent_channels"], n_theta=lds_config["n_theta"],
        latent_spatial=lds_config.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
        hidden_dim=lds_config["hidden_dim"], n_hidden_layers=lds_config["n_hidden_layers"],
        # inf (exact no-op) for any checkpoint saved before dt_cap
        # existed -- same .get()-with-fallback pattern as
        # latent_spatial_size just above, and as model_assembly.py's
        # own build_models_from_components uses for the SAME parameter.
        # A real, reported bug otherwise: this is a SEPARATE LatentDynamics
        # construction from that one (this whole function evaluates a
        # raw Stage 3a/3b checkpoint directly, not via
        # build_models_from_components at all), so fixing dt_cap there
        # didn't fix it here -- a checkpoint saved with a real, finite
        # dt_cap would silently evaluate as if dt_cap were still inf,
        # with no error or warning anywhere to indicate the mismatch.
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

    dataset = MicrostructureEvolutionDataset(
        test_dirs, encoder=ae_encoder, device=device, window_length=window_length,
        min_step=min_step, min_stdev_phi=min_stdev_phi, min_passing_steps=min_passing_steps,
        max_dt=max_dt, encode_both_streams=True,
        # The diagnostics re-encoded their whole population on every run while
        # the trainers had been caching since the feature landed. Nothing about
        # the cache is training-specific: the key is the ENCODER's own
        # fingerprint plus the run and step list, and a diagnostic uses a
        # frozen encoder straight out of a checkpoint -- usually the very
        # checkpoint the trainer just wrote, so the entries are already there.
        #
        # Most visible on an off-distribution run: --max-dt large disables the
        # prefix truncation, so every frame of every run gets encoded.
        latent_cache_dir=latent_cache_dir,
    )
    print(f"Evaluating {len(dataset)} test windows...")

    return (device, euler_only, lds_checkpoint_path, ae_config, dataset, ae_decoder, f_theta)


def _grid_size_for_dataset_filename(size, lds_checkpoint_path) -> str:
    """The `{size}x{size}` part of a dataset-level figure name.

    `size` is a CLI argument, and it is OPTIONAL -- running

        python -m evaluation.check_parameter_dependence --lds-checkpoint ... \\
               --base-path ../datasets

    leaves it None, and the f-string then produced a file literally called
    `NonexNone-dz0dt.png` in output/datasets/. Reported.

    The size is not in the LDS checkpoint's own config (that records
    latent_channels/latent_spatial_size/hidden_dim, not the grid), and the AE
    checkpoint is not loaded yet where this path is built. But the checkpoint
    STEM carries it -- `128x128-stage3a` -- which is the same source
    _stage_folder_from_checkpoint_stem already reads.

    If the stem does not carry one either, falls back to the stem itself
    rather than inventing a size: a figure named after the checkpoint that
    produced it is worse-grouped but not WRONG, whereas `NonexNone` is a
    filename that will be mistaken for a bug in the physics.
    """
    if size is not None:
        return f"{size}x{size}"
    match = re.match(r"^(\d+)x(\d+)", Path(lds_checkpoint_path).stem)
    if match:
        return f"{match.group(1)}x{match.group(2)}"
    return Path(lds_checkpoint_path).stem


def _stage_folder_from_checkpoint_stem(checkpoint_path: Path) -> str:
    """
    Stage folder derived from the checkpoint's own stem (e.g.
    "64x64-stage3a" -> "stage3a"), NOT a hardcoded "stage3" -- matches
    the actual per-stage output layout (output/stage3a/, output/
    stage3b/, not a single shared output/stage3/) that this project's
    other stages already use. A hardcoded "stage3" was a real, reported
    bug in TWO separate places independently (check_parameter_dependence.py's
    own dt_dependence.png/dz0dt.png, and check_f_theta.py's own
    f_theta_diagnostic.png): a stage-3a and a stage-3b checkpoint's own
    figures would both land in output/stage3/, silently overwriting
    each other's output across runs. Shared here so this exact bug
    pattern can't recur a third time in some future script.
    """
    match = re.match(r"^\d+x\d+-stage(\w+)", checkpoint_path.stem)
    return f"stage{match.group(1)}" if match else "stage3"


def _load_models_and_dataset(
    lds_checkpoint_path: Path, min_step: int | None, min_stdev_phi: float | None,
    min_passing_steps: int | None, base_path: Path | None, size: int | None,
    ae_stats_weight: float | None, hidden_dim: int, n_hidden_layers: int,
    condition_on_theta: bool | None, euler_only: bool | None,
    output_path: Path | None, dz0dt_output_path: Path | None,
    dt_dependence_output_path: Path | None, device: str | None,
    max_dt: float | None = None, latent_cache_dir: Path | str | None = None,
) -> _LoadedContext:
    """check_parameter_dependence()'s own setup phase -- resolves this
    function's own three output paths, then delegates model/dataset
    loading to _load_ae_f_theta_and_dataset (see its own docstring for
    why that half is a separate, shared function). See
    check_parameter_dependence()'s own docstring for the parameters'
    meaning; unchanged here."""
    _stage_folder = _stage_folder_from_checkpoint_stem(lds_checkpoint_path)

    output_path_defaulted = output_path is None
    if output_path is None:
        output_path = (_PYTHON_ROOT.parent / "output" / _stage_folder
                       / f"{lds_checkpoint_path.stem}-parameter_dependence.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # dz0dt_output_path/dt_dependence_output_path default relative to
    # output_path's own (already-resolved) DIRECTORY, not independently
    # from _PYTHON_ROOT -- a real, reported bug otherwise: a caller that
    # overrides only output_path (e.g. every existing test that predates
    # these two newer parameters, pointing output_path at its own
    # isolated tmp_path) had no reason to know about these two siblings
    # at all, so they silently fell back to the real, live output/
    # tree -- leaving actual "fake-stage3b-dz0dt.png"/
    # "fake-stage3b-dt_dependence.png" files behind in the user's real
    # output/stage3b/ folder after every test run, not in the test's own
    # tmp_path. Cascading from output_path.parent means overriding just
    # output_path (the old, single-parameter calling convention) is
    # still enough to redirect all three consistently.
    if dz0dt_output_path is None:
        # dz0dt.png reports the GROUND-TRUTH dz0 / dz0dt of the encoded
        # data -- a property of the dataset at a given encoder, not of
        # f_theta -- so with everything left at its defaults it is
        # written next to the other dataset-level diagnostics
        # (output/datasets/{size}x{size}-dz0dt.png) rather than under the
        # per-stage folder.
        #
        # ONLY when output_path was itself defaulted, though. The cascade
        # described above exists because a caller overriding just
        # output_path (every test predating these two parameters) must
        # still redirect all three, or the run leaves real files in the
        # live output/ tree. Sending this one to a fixed absolute
        # location unconditionally would reintroduce exactly that
        # reported bug.
        #
        # NOTE the name drops the checkpoint stem, so two checkpoints for
        # the same grid size (e.g. stage 2 and stage 3) now write to ONE
        # file and the later run overwrites the earlier. Pass
        # dz0dt_output_path explicitly to keep both.
        dz0dt_output_path = ((_PYTHON_ROOT.parent / "output" / "datasets"
                               / f"{_grid_size_for_dataset_filename(size, lds_checkpoint_path)}"
                                 f"-dz0dt.png") if output_path_defaulted
                              else output_path.parent / f"{lds_checkpoint_path.stem}-dz0dt.png")
    dz0dt_output_path.parent.mkdir(parents=True, exist_ok=True)
    if dt_dependence_output_path is None:
        dt_dependence_output_path = output_path.parent / f"{lds_checkpoint_path.stem}-dt_dependence.png"
    dt_dependence_output_path.parent.mkdir(parents=True, exist_ok=True)

    device, euler_only, lds_checkpoint_path, ae_config, dataset, ae_decoder, f_theta = (
        _load_ae_f_theta_and_dataset(
            lds_checkpoint_path, min_step, min_stdev_phi, min_passing_steps, base_path, size,
            ae_stats_weight, hidden_dim, n_hidden_layers, condition_on_theta, euler_only, device,
            max_dt=max_dt, latent_cache_dir=latent_cache_dir,
        )
    )

    return _LoadedContext(
        device=device, euler_only=euler_only, output_path=output_path,
        dz0dt_output_path=dz0dt_output_path, dt_dependence_output_path=dt_dependence_output_path,
        lds_checkpoint_path=lds_checkpoint_path, ae_config=ae_config, dataset=dataset,
        ae_decoder=ae_decoder, f_theta=f_theta,
    )



def _evaluate_windows(dataset, f_theta, ae_decoder, device, decode: bool, euler_only: bool) -> _EvaluationResults:
    """The main per-window evaluation loop, extracted verbatim -- iterates
    every test window, computing the full/euler-only/pixel-space (if
    decode=True) predictions and losses, plus the ground-truth dz0/dz0dt
    tracking dz0dt.png needs. Ends with the euler_only substitution (see its
    own comment below, preserved unchanged) so every downstream consumer of
    latent_losses/pixel_losses/latent_losses_signed automatically reports on
    euler-only data in that mode without needing to know euler_only exists."""
    # metadata.txt read once per run_dir, not once per window -- most
    # runs contribute several windows, and temperature/noise are
    # constant across all of them (unlike dt, which varies per window
    # even within the same run). statistics.csv similarly cached per
    # run_dir, for the autocorr_length lookup below -- unlike
    # temperature/noise, autocorr_length is NOT constant across a run's
    # windows (a run's own dominant length scale coarsens over time as
    # the microstructure evolves), so it's looked up per-window at that
    # window's own starting step, not cached at the run level itself.
    metadata_cache: dict[Path, object] = {}
    stats_cache: dict[Path, object] = {}

    dts, temperatures, noises, run_dirs, length_scales = [], [], [], [], []
    latent_losses, pixel_losses, euler_losses, euler_pixel_losses = [], [], [], []
    latent_losses_signed, euler_losses_signed = [], []
    # abs_steps/dz0dt_* are NOT model-dependent at all -- dz0/dt is the
    # ground-truth finite-difference derivative, computed purely from
    # the encoder's own z0_t/z0_next_true, with no z1/f_theta involved
    # anywhere. Unlike everything else tracked here, these never need
    # the euler_only substitution -- there's only ever one "real" dz0/dt,
    # regardless of which prediction mode is being evaluated.
    abs_steps, dz0_signed, dz0_abs, dz0dt_signed, dz0dt_abs = [], [], [], [], []

    def _per_sample_l1(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """(B, ...) -> (B,) mean absolute error per sample -- matches
        OneStepLoss/ReconLoss's own L1 definition (mean over all
        non-batch dims), but WITHOUT their forward()'s own further
        reduction over the batch dim too, which would collapse an
        entire batch to one scalar -- exactly what per-window
        correlation against dt/temperature/etc needs to NOT happen."""
        return (pred - true).abs().flatten(start_dim=1).mean(dim=1)

    def _per_sample_signed_mean(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """(B, ...) -> (B,) mean SIGNED residual per sample -- same
        reduction as _per_sample_l1 but WITHOUT the .abs(), so positive
        and negative components can cancel within a window. Used
        specifically for the linear panel: the formula it checks is
        written in terms of the signed residual, not |residual|."""
        return (pred - true).flatten(start_dim=1).mean(dim=1)

    signed_residual_sum = None  # accumulated (C, H, W) sum of z0_euler_pred - z0_next_true,
                                 # summed (not yet averaged) across ALL windows -- see the
                                 # bias-vs-variance analysis after the loop for why this needs
                                 # to stay SIGNED, unlike everything else computed here.
    n_total = 0

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    idx = 0
    with torch.no_grad():
        for window0, window1, dt_window, theta in loader:
            batch_size = window0.shape[0]
            window0 = window0.to(device)
            window1 = window1.to(device)
            dt_window = dt_window.to(device)
            theta = theta.to(device)

            z0_t = window0[:, 0]
            z1_t = window1[:, 0]
            z0_next_true = window0[:, 1]
            dt = dt_window[:, 0]
            theta_b = theta

            z0_next_pred = f_theta(z0_t, z1_t, dt, theta_b)
            # The pure, hard-coded Euler term ALONE -- z0(t) + z1(t)*dt,
            # no f_theta correction at all. Comparing this against
            # z0_next_pred's own error (both against the SAME
            # z0_next_true, same dt, same window) is what actually
            # disentangles the two Taylor orders: the FIRST-order term
            # (z1*dt) is hard-coded, never learned, so its own error is
            # entirely a property of z1's own quality and the physics'
            # own curvature -- f_theta (the SECOND-order, TRAINED
            # correction) can only ever act on top of it. If f_theta is
            # adding real value, its own (full) error should fall off
            # FASTER with dt (a higher power-law exponent) than this
            # Euler-only baseline does -- see the dedicated panel below
            # for the direct visual/numeric comparison.
            dt_r = dt.view(-1, 1, 1, 1)
            z0_euler_pred = z0_t + z1_t * dt_r

            latent_loss_batch = _per_sample_l1(z0_next_pred, z0_next_true)
            euler_loss_batch = _per_sample_l1(z0_euler_pred, z0_next_true)
            latent_loss_signed_batch = _per_sample_signed_mean(z0_next_pred, z0_next_true)
            euler_loss_signed_batch = _per_sample_signed_mean(z0_euler_pred, z0_next_true)
            # Ground-truth dz0 = z0_next_true - z0_t -- note this is
            # (z0_next_true - z0_t), NOT (z0_t - z0_next_true) the way
            # _per_sample_signed_mean's usual (pred-true) argument order
            # would suggest -- there's no "pred" here at all, just the
            # actual finite difference in its own natural direction.
            # dz0dt derived from this SAME computation (not a second,
            # separate _per_sample_signed_mean call) -- dt is a positive
            # per-window scalar (not a per-pixel tensor), so dividing
            # dz0 by it after the mean reduction is equivalent to
            # dividing before. abs versions similarly derived from ONE
            # _per_sample_l1 call, not two.
            dz0_signed_batch = _per_sample_signed_mean(z0_next_true, z0_t)
            dz0dt_signed_batch = dz0_signed_batch / dt
            dz0_abs_batch = _per_sample_l1(z0_next_true, z0_t)
            dz0dt_abs_batch = dz0_abs_batch / dt

            # Decoder calls genuinely skipped, not just left unplotted,
            # when decode=False -- these are 3 extra decoder forward
            # passes per batch, purely for panel [1,0] (pixel-space),
            # which is itself now only built when decode=True. An
            # earlier version of this comment claimed the extra decoder
            # call was "cheap relative to everything else happening per
            # batch here" and ran it unconditionally regardless of
            # whether [1,0] even got drawn -- decode=False actually
            # avoids the cost now, rather than paying it either way.
            if decode:
                x_next_pred = ae_decoder(z0_next_pred)
                x_next_true = ae_decoder(z0_next_true)
                pixel_loss_batch = _per_sample_l1(x_next_pred, x_next_true)
                x_next_euler_pred = ae_decoder(z0_euler_pred)
                euler_pixel_loss_batch = _per_sample_l1(x_next_euler_pred, x_next_true)
                pixel_losses.extend(pixel_loss_batch.cpu().tolist())
                euler_pixel_losses.extend(euler_pixel_loss_batch.cpu().tolist())

            latent_losses.extend(latent_loss_batch.cpu().tolist())
            euler_losses.extend(euler_loss_batch.cpu().tolist())
            latent_losses_signed.extend(latent_loss_signed_batch.cpu().tolist())
            euler_losses_signed.extend(euler_loss_signed_batch.cpu().tolist())
            dts.extend(dt.cpu().tolist())
            dz0_signed.extend(dz0_signed_batch.cpu().tolist())
            dz0_abs.extend(dz0_abs_batch.cpu().tolist())
            dz0dt_signed.extend(dz0dt_signed_batch.cpu().tolist())
            dz0dt_abs.extend(dz0dt_abs_batch.cpu().tolist())

            # SIGNED (not .abs()'d) euler-only residual, summed over the
            # batch dim only -- keeps the full (C, H, W) shape, so
            # element-wise cancellation across DIFFERENT windows is what
            # actually happens here (a random +/- residual at any given
            # element genuinely cancels when summed across many
            # windows; .abs() before summing would never let it).
            batch_signed_residual = (z0_euler_pred - z0_next_true).sum(dim=0)
            signed_residual_sum = (batch_signed_residual if signed_residual_sum is None
                                    else signed_residual_sum + batch_signed_residual)
            n_total += batch_size

            # Per-window metadata: cheap, CPU-bound, inherently
            # per-index -- not a tensor op, so batching it wouldn't
            # help; stays in its own loop, synchronized to the same
            # dataset ordering the DataLoader above preserves
            # (shuffle=False).
            for i in range(batch_size):
                run_dir, steps = dataset.window_info(idx)
                if run_dir not in metadata_cache:
                    metadata_cache[run_dir] = load.read_metadata(run_dir / "metadata.txt")
                metadata = metadata_cache[run_dir]
                if run_dir not in stats_cache:
                    stats_cache[run_dir] = load.read_statistics_csv(run_dir / "statistics.csv")
                stats_df = stats_cache[run_dir]

                temperatures.append(metadata.temperature)
                noises.append(metadata.noise)
                run_dirs.append(run_dir)
                abs_steps.append(steps[0])
                # Ground-truth length scale (first peak in the autocorrelation
                # function), read from the SIMULATION's own precomputed
                # statistics.csv -- not re-derived from the (possibly
                # decoder-distorted) reconstructed frame -- at the window's
                # starting step, i.e. the length scale of the microstructure
                # this rollout step is actually predicting FROM.
                length_scales.append(stats_df.loc[steps[0], "autocorr_length"])
                idx += 1

    dts = np.array(dts)
    temperatures = np.array(temperatures)
    noises = np.array(noises)
    length_scales = np.array(length_scales, dtype=float)
    latent_losses = np.array(latent_losses)
    pixel_losses = np.array(pixel_losses)
    euler_losses = np.array(euler_losses)
    euler_pixel_losses = np.array(euler_pixel_losses)
    latent_losses_signed = np.array(latent_losses_signed)
    euler_losses_signed = np.array(euler_losses_signed)
    abs_steps = np.array(abs_steps, dtype=float)
    dz0_signed = np.array(dz0_signed)
    dz0_abs = np.array(dz0_abs)
    dz0dt_signed = np.array(dz0dt_signed)
    dz0dt_abs = np.array(dz0dt_abs)

    # euler_only substitution: EVERY panel/fit/print below this point
    # reads latent_losses/pixel_losses/latent_losses_signed as "the
    # thing worth showing" -- rather than threading an if euler_only
    # branch through each ~15 of those individually (panels [0,0],
    # [0,1], [0,2], [1,0], [1,1], [1,2], every fit/correlation/binned-
    # summary derived from them), aliasing these three names to their
    # own euler-only equivalents ONCE, here, makes every one of those
    # downstream consumers automatically report on euler-only data
    # without any of them needing to know euler_only exists at all. The
    # two panels that explicitly show BOTH euler-only and full
    # side-by-side ([0,3], [1,3]) are the only ones that still check
    # euler_only directly -- see their own comments below for why they
    # can't use this same trick (they'd become a redundant "euler-only
    # vs itself" comparison instead of correctly just dropping the
    # full/blue trace).
    if euler_only:
        latent_losses = euler_losses
        pixel_losses = euler_pixel_losses
        latent_losses_signed = euler_losses_signed

    return _EvaluationResults(
        dts=dts, temperatures=temperatures, noises=noises, length_scales=length_scales,
        run_dirs=run_dirs, latent_losses=latent_losses, pixel_losses=pixel_losses,
        euler_losses=euler_losses, euler_pixel_losses=euler_pixel_losses,
        latent_losses_signed=latent_losses_signed, euler_losses_signed=euler_losses_signed,
        abs_steps=abs_steps, dz0_signed=dz0_signed, dz0_abs=dz0_abs,
        dz0dt_signed=dz0dt_signed, dz0dt_abs=dz0dt_abs,
        signed_residual_sum=signed_residual_sum, n_total=n_total,
    )


