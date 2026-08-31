"""
PyTorch Dataset classes for loading phase-field runs off disk.
"""

import inspect
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import training.latent_cache as latent_cache
from utils.logging_utils import format_progress_count


def _progress_eta(done: int, total: int, t0: float) -> str:
    """Compact '~MmSSs left' from elapsed rate, for the in-place counters below.
    Empty until a rate exists (first tick)."""
    el = time.monotonic() - t0
    if done <= 0 or el <= 0:
        return "estimating"
    rem = el / done * (total - done)
    m, s = divmod(int(rem + 0.5), 60)
    return f"~{m}m{s:02d}s left" if m else f"~{s}s left"
from models.constants import (LATENT_SPATIAL_SIZE as _LATENT_SPATIAL_SIZE,
                               N_THETA, theta_coordinates)
from models.latent_streams import DEFAULT_STREAM_NAME
from utils import load_datasets as load


def _dihedral_transform(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """
    Apply one element of the square's symmetry group (dihedral group D4)
    to a (C, H, W) tensor: optional horizontal mirror, then rotate by
    k*90 degrees. The 8 (k, flip) combinations give exactly: identity,
    h-mirror, rot90, rot90+mirror (a diagonal transpose), rot180,
    rot180+mirror (v-mirror), rot270, rot270+mirror (the other diagonal
    transpose) -- i.e. mirror h/v, rotation +-90/180, and transpose +-45,
    with no double-counting or gaps (verified against numpy.rot90/flip).
    Exact grid permutations, no interpolation, so no numerical error.
    """
    if flip:
        x = torch.flip(x, dims=[-1])
    if k:
        x = torch.rot90(x, k=k, dims=[-2, -1])
    return x


def _translate(x: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    """
    Periodic (wraparound) translation. Confirmed valid: solver uses
    periodic boundary conditions.
    """
    if shift_y == 0 and shift_x == 0:
        return x
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(-2, -1))


def _transform_angle(angle: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """
    Transform the 'angle' statistic (local orientation, radians, defined
    mod pi since it describes an undirected line/interface normal --
    docs/neural_nets.md's arctan(v1y/v1x), range (-pi/2, pi/2]) to match
    the D4 transform _dihedral_transform applied to the image itself.
    Order matches that function: mirror first, then rotate by k*90.

    SOLID: mirroring negates the angle. torch.flip(dims=[-1]) mirrors
    along the column/x axis (confirmed elsewhere in this codebase), so
    (v1x, v1y) -> (-v1x, v1y), giving arctan(v1y/-v1x) = -arctan(v1y/v1x)
    regardless of any row/col-vs-xy convention question.

    SOLID: k=2 (180 degrees) is always a no-op after mod-pi wrapping,
    regardless of rotation sign convention, since a line's orientation
    is unchanged by a half turn.

    UNCONFIRMED: the SIGN of the k*90-degree term for k=1/k=3. This
    depends on how the C++ code's v1x/v1y map onto array axes, combined
    with whether torch.rot90's rotation direction is "visually
    counterclockwise" or "mathematically counterclockwise" once you
    account for image row-index increasing downward -- the same class
    of ambiguity as the original read_phi_half reshape-order question,
    which was only resolved by inspecting the actual C++ indexing code.
    No equivalent has been confirmed here. Assumed +k*90 degrees below;
    if empirical comparison against the real C++ angle computation on a
    rotated field disagrees, k=1 and k=3 simply swap -- flip this sign.
    """
    if flip:
        angle = -angle
    angle = angle + k * (torch.pi / 2)
    # wrap to (-pi/2, pi/2], since the angle is only defined mod pi
    angle = ((angle + torch.pi / 2) % torch.pi) - torch.pi / 2
    return angle


_N_AUGMENT_VARIANTS = 8 * 4  # N_DIHEDRAL(8) x 4 translations -- see _apply_augmentation

# A FIXED set of 4 augmentation variants for VALIDATION-side averaging
# (see MicrostructureEvolutionDataset's own fixed_aug_indices parameter).
# Chosen as a Latin-square spread: 4 DISTINCT dihedral transforms x 4
# DISTINCT translations, maximizing how decorrelated the model's own
# errors are across the 4 evaluations of each window -- the whole point
# being that these are exact symmetries (translations are periodic
# rolls, valid since the solver uses periodic BCs; see _translate), so
# the TRUE loss is identical across all of them and only the model's own
# equivariance error differs.
#
# Restricted to the dihedral transforms whose "angle" statistic
# correction is CONFIRMED correct -- k=0 (identity/mirror) and k=2
# (180 degrees) -- deliberately EXCLUDING every k=1/k=3 variant
# (aug_idx 8-15, 24-31). _transform_angle's own docstring flags the SIGN
# of its k*90 term for k=1/k=3 as UNCONFIRMED; baking a possibly-wrong
# angle target into a VALIDATION metric would put systematic error into
# exactly the stats0 term that's already the noisiest and
# worst-generalizing of the three, and unlike in training (where the
# model simply learns whatever it's shown) a systematic error here
# corrupts cross-epoch comparisons directly.
#
#   aug_idx  0 -> k=0, flip=False (identity),     shift (0, 0)
#   aug_idx  5 -> k=0, flip=True  (mirror),       shift (0, nx//2+third)
#   aug_idx 18 -> k=2, flip=False (180 deg),      shift (ny//2+third, 0)
#   aug_idx 23 -> k=2, flip=True  (mirror+180),   shift (ny//2+2/3, nx//2+2/3)
VAL_DECORRELATED_AUG_INDICES = (0, 5, 18, 23)

# Every aug_idx whose own "angle" stat correction is confirmed correct
# (see VAL_DECORRELATED_AUG_INDICES above for why this matters, and
# _transform_angle's own docstring for the underlying uncertainty).
_ANGLE_SAFE_AUG_INDICES = frozenset(
    d_idx * 4 + t_idx
    for d_idx in (0, 1, 4, 5)  # dihedral_idx = k*2 + flip, so these are k=0 and k=2 only
    for t_idx in range(4)
)


def _apply_augmentation(x: torch.Tensor, aug_idx: int, nx: int, ny: int) -> tuple[torch.Tensor, int, bool]:
    """
    One of the N_DIHEDRAL(8) x 4 augmentation variants, applied to an
    ALREADY-LOADED (C, H, W) frame (loading itself is the caller's
    job -- this is pure transform selection, shared by
    MicrostructureSnapshotDataset's per-frame augmentation and
    MicrostructureEvolutionDataset's per-window augmentation, which
    needs the SAME (k, flip, shift) applied to EVERY frame in a window
    to keep the window physically consistent -- flipping x(t) but not
    x(t+dt) would make the pair describe a physically meaningless
    "evolution").

    Sub-cell offset, used to spread the 4 translations across distinct
    sub-cell phases instead of the single phase the old Nx/2-only
    shifts all shared (see _LATENT_SPATIAL_SIZE's docstring for the
    underlying artifact this addresses).

    THIRDS, NOT HALVES: an earlier version of this used
    _LATENT_SPATIAL_SIZE // 2 (a literal half-cell offset) for all 3
    non-identity shifts, which seemed like the natural choice -- but
    half of a cell added to itself returns to 0 mod cell_size (half +
    half = cell_size == 0 mod cell_size), so the 4 shifts' residues --
    (0,0), (0,half), (half,0), (half,half) -- form a closed 2-element
    {0, half} subgroup under addition. That's a DEGENERATE,
    lower-diversity phase set in disguise (effectively period
    cell_size/2, not a genuine spread across the full cell), not the 4
    independent phases it looks like at a glance. Thirds avoid this:
    cell_size//3 and (cell_size*2)//3 aren't related by any such
    small-order closure, so the 4 shifts' residues don't accidentally
    collapse onto a smaller repeating subset the way the half-based
    ones did.

    NOTE -- this is only exactly a THIRD of a cell
    (nx/_LATENT_SPATIAL_SIZE pixels wide) when nx ==
    _LATENT_SPATIAL_SIZE**2 (64 for _LATENT_SPATIAL_SIZE=8), which is
    the only size actually run through this dataset today. For any
    other size (e.g. a future 128x128), the true cell size is nx //
    _LATENT_SPATIAL_SIZE and this constant offset would land at a
    different fraction of the cell instead -- still a distinct,
    non-degenerate phase per the reasoning above, just not exactly "a
    third" by name. Also worth noting: this uses the model's DEFAULT
    bottleneck size (see models/constants.py) -- if a particular
    checkpoint was built with a non-default latent_spatial_size (now
    genuinely configurable), this augmentation phase-spread wouldn't
    match ITS actual cell size. Augmentation happens before/independent
    of any specific model though, so there's no natural single "right"
    value to read here instead; the default is the reasonable
    general-purpose choice.
    """
    n_translations = 4
    dihedral_idx, translation_idx = divmod(aug_idx, n_translations)
    k, flip = divmod(dihedral_idx, 2)

    x = _dihedral_transform(x, k=k, flip=bool(flip))

    third = _LATENT_SPATIAL_SIZE // 3
    two_thirds = (_LATENT_SPATIAL_SIZE * 2) // 3
    shifts = [
        (0, 0),
        (0, nx // 2 + third),
        (ny // 2 + third, 0),
        (ny // 2 + two_thirds, nx // 2 + two_thirds),
    ]
    shift_y, shift_x = shifts[translation_idx]
    x = _translate(x, shift_y, shift_x)
    return x, k, bool(flip)


# Run/step selection lives in _dataset_filtering.py (extracted, no behaviour
# change). Re-exported here so `from training.datasets import build_good_steps`
# (and the other four names) keeps working everywhere it is already used.
from training._dataset_filtering import (  # noqa: F401  (re-export)
    _REPORTED_SWEEPS, _filtered_steps, build_good_steps, complete_run_dirs,
    report_save_step_distribution, split_run_dirs,
)












class MicrostructureSnapshotDataset(Dataset):
    """
    Flat collection of individual microstructure snapshots x(t), pooled
    across one or more complete run directories. For AE-only training
    (stage 1) -- no time dynamics, so cross-run/cross-time order doesn't
    matter, everything is just one big pool of frames.

    Each item is read from disk on access, not cached in memory unless
    cache_in_memory=True: one snapshot is small, but hundreds of runs x
    tens of steps at 256x256 is not something to hold entirely in RAM
    up front by default.

    augment: if True, expands the pooled dataset by the 8 elements of D4
    (see _dihedral_transform) combined with 4 periodic translations, for
    a 32x expansion of the effective frame count. The cache (if enabled)
    still stores only the base frames; augmentation is applied after
    cache lookup, so memory use is unaffected by the expansion factor.

    TRANSLATION PHASES: the bottleneck latent is always 8x8 regardless of
    Nx/Ny (encoder depth scales with input size specifically to always
    reach this exactly -- see models/encoder.py's own `input_size / 8`),
    which means the decoder reconstructs each (Nx/8, Ny/8)-pixel latent
    CELL through the same fixed sequence of non-overlapping 2x
    ConvTranspose2d upsamplings. A checkerboard/tiling reconstruction
    artifact was traced to this: for thin, sharp interfaces specifically,
    the artifact's period matches the latent cell size exactly (confirmed
    empirically: 8px period on 64x64, i.e. Nx/8), consistent with the
    decoder relying on a fixed within-cell spatial template rather than
    deriving sub-cell interface position purely from latent content.

    The old translation set -- (0,0), (Nx/2,0), (0,Ny/2), (Nx/2,Ny/2) --
    never exercised this: since Nx/2 (and Ny/2) are exact multiples of the
    8x8 grid's cell size for every size used in this project, all 4 shifts
    land on the IDENTICAL sub-cell phase relative to the latent grid (only
    large-scale position varies, never the phase within one cell). So 32x
    augmentation never once forced the decoder to be consistent under a
    sub-cell shift of the interface -- a plausible reason the fixed
    template could survive training undisturbed.

    The translations below keep the same large-scale repositioning (still
    offset by ~Nx/2, ~Ny/2, so features move well away from their
    original location) but add a HALF-CELL offset (cell_size // 2) to one
    or both axes on 3 of the 4 shifts, chosen so the 4 shifts land on 4
    distinct sub-cell phases -- (0,0), (0,half), (half,0), (half,half) --
    instead of 1 phase repeated 4 times. This doesn't fix the artifact by
    itself (the decoder architecture/training is unchanged), but it means
    augmentation actually samples the phase relationship the artifact
    depends on, which the previous grid-aligned-only shifts structurally
    could not do.

    Train/val/test independence: this class itself does NOT split --
    use split_run_dirs() on the run directory list BEFORE constructing
    one instance per split (train gets augment=True if desired, val/test
    should use augment=False). Each instance is then a fully independent
    dataset with its own stats_normalization(), with no shared state.
    """

    _N_DIHEDRAL = 8

    def __init__(self, run_dirs: list[str | Path], skip_bad: bool = True,
                 cache_in_memory: bool = False, augment: bool = False, min_step: int = 0,
                 min_stdev_phi: float | None = None, min_passing_steps: int | None = None,
                 include_stats: bool = False, stat_names: list[str] | None = None,
                 good_steps: dict[Path, list[int]] | None = None,
                 split_label: str = "", min_normalized_stdev_phi: float | None = None):
        """
        good_steps: a precomputed {run_dir: [kept_step, ...]} mapping
        from build_good_steps(), to skip re-scanning run_dirs when
        another dataset (e.g. a paired MicrostructureEvolutionDataset
        over the same run_dirs) already computed it with the same
        skip_bad/min_step/min_stdev_phi/min_passing_steps. Default None
        computes it here.

        min_step: skip snapshots earlier than this step. Kept as a cheap,
        always-available filter that needs no statistics.csv, but a fixed
        step count doesn't generalize across a sweep -- different (T,
        noise) combinations develop at very different rates, so one
        threshold that's safe for a fast-developing combo can still land
        inside the noise regime for a slow one. Default 0 keeps everything.

        min_stdev_phi: skip snapshots whose statistics.csv stdev_phi is
        below this value OR is NaN. This is a better criterion than
        min_step for excluding "uninteresting" smooth fields, since it
        catches BOTH ends of the evolution with one physically-meaningful
        check: near-t=0 noise (before microstructure develops) AND late,
        fully-coarsened single-grain states -- both have near-zero
        variance. The NaN case isn't incidental: stdev_phi is NaN
        specifically at very low true variance, consistent with
        floating-point cancellation in a one-pass variance computation
        (E[X^2] - E[X]^2 going slightly negative before the sqrt) --
        confirmed empirically to occur at both very early and very late
        (post-coarsening) steps, i.e. exactly the smooth-field regime
        this filter is meant to exclude. Reads statistics.csv even if
        include_stats=False, since this filter is about data relevance,
        not the stats-prediction feature. Default None disables it.

        include_stats: if True, __getitem__ returns (x, stats) instead
        of just x, where stats is a fixed-order vector of the columns in
        stat_names (or, if not given, every non-step column of the first
        run's statistics.csv -- every run must then have exactly the
        same columns, checked explicitly rather than silently mismatched).

        SCHEMA DRIFT WARNING: different batches of runs can have
        different statistics.csv columns (e.g. 'trace' dropped from
        newer 128x128 runs after being found identical to
        'gradient_sqr'). If stat_names=None, the columns are locked in
        from whichever run happens to be first in run_dirs -- if that's
        an old-schema run and a later run_dir lacks a column, this
        raises a clear error at that point. When pooling runs from
        different batches, pass stat_names explicitly rather than
        relying on auto-detection order.

        ORIENTATION HANDLING: most statistics (avg_phi, stdev_phi, energy,
        autocorr_length, ...) are exact aggregates over all pixel values
        and are therefore invariant under the D4 augmentation --
        rotating/mirroring the image doesn't change them. 'angle' (local
        orientation) is NOT invariant, so it is transformed to match each
        augmented view's actual rotation/mirror (see _transform_angle) --
        with one unconfirmed sign on the 90-degree rotation term, flagged
        in that function's docstring. Translation never needs adjustment
        (pure shift, no rotation/reflection of content).
        """
        self._index: list[tuple[Path, int, int, int]] = []  # (run_dir, step, nx, ny)
        self._stats_by_run = {}  # dict[Path, pd.DataFrame], populated below if include_stats

        run_dirs = [Path(d) for d in run_dirs]
        if good_steps is None:
            good_steps = build_good_steps(run_dirs, skip_bad, min_step, min_stdev_phi,
                                          min_passing_steps, split_label=split_label,
                                          min_normalized_stdev_phi=min_normalized_stdev_phi)

        # Per-run stats + index build. For a large sweep this parses a
        # statistics.csv per run (1000+ files) silently, right after
        # build_good_steps' own summary line -- a multi-second wall of no
        # output that looks hung. Show an in-place counter for a non-trivial
        # run set (raw counts below 10k, thousands above -- see
        # format_progress_count).
        _show_progress = len(run_dirs) >= 20
        _n_runs = len(run_dirs)
        _t0 = time.monotonic()

        for _run_pos, run_dir in enumerate(run_dirs):
            if _show_progress and (_run_pos % 25 == 0 or _run_pos == _n_runs - 1):
                sys.stdout.write(
                    f"\r  indexing runs: {format_progress_count(_run_pos + 1, _n_runs)}  "
                    f"({_progress_eta(_run_pos + 1, _n_runs, _t0)})   ")
                sys.stdout.flush()
            metadata = load.read_metadata(run_dir / "metadata.txt")
            kept_steps = good_steps[run_dir]

            # Still needed here (independent of build_good_steps) when
            # include_stats wants the full DataFrame, not just the
            # stdev_phi column build_good_steps may have already used
            # internally for min_stdev_phi filtering.
            stats_df = None
            if include_stats:
                stats_df = load.read_statistics_csv(run_dir / "statistics.csv")
                if stat_names is None:
                    stat_names = sorted(stats_df.columns)
                missing = set(stat_names) - set(stats_df.columns)
                if missing:
                    raise ValueError(
                        f"{run_dir}/statistics.csv is missing requested columns {missing}"
                    )
                self._stats_by_run[run_dir] = stats_df

            for step in kept_steps:
                if include_stats:
                    # ANY NaN in ANY requested column poisons stats_normalization()
                    # for the WHOLE dataset (mean()/std() over an array containing
                    # a single NaN returns NaN, not just for that one row) -- this
                    # must check every stat_names column, not just stdev_phi.
                    if step not in stats_df.index:
                        continue
                    if stats_df.loc[step, stat_names].isna().any():
                        continue
                self._index.append((run_dir, step, metadata.nx, metadata.ny))

        if _show_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

        self.augment = augment
        self.include_stats = include_stats
        self.stat_names = stat_names if include_stats else None

        if include_stats and self.stat_names is None:
            raise ValueError(
                "include_stats=True but no statistics were loaded -- run_dirs is "
                "likely empty (no complete runs found)"
            )

        # Optional: read every frame once up front instead of on every
        # access. Worth it when the whole dataset fits comfortably in RAM
        # (true for 64x64/128x128 pools of a few thousand frames -- tens
        # of MB) since it removes per-sample disk I/O entirely, which
        # otherwise bottlenecks small-image training (GPU finishes a
        # tiny forward/backward pass almost instantly and then waits on
        # disk). Leave off for large images/datasets where this would
        # actually blow up memory instead of saving time.
        self._cache: list[torch.Tensor] | None = None
        if cache_in_memory:
            self._cache = [self._load(i) for i in range(len(self._index))]

    @classmethod
    def from_sweep(cls, base: str | Path, size: int,
                    skip_bad: bool = True, cache_in_memory: bool = False,
                    augment: bool = False, min_step: int = 0, min_stdev_phi: float | None = None,
                    include_stats: bool = False, stat_names: list[str] | None = None,
                    ) -> "MicrostructureSnapshotDataset":
        """
        Convenience: pool every complete run for one grid size, skipping
        incomplete ones. NOTE: this does not split into train/val/test --
        for that, get the dir list via complete_run_dirs(), split it with
        split_run_dirs(), and construct one instance per split instead.
        """
        run_dirs = complete_run_dirs(base, size, size)
        return cls(run_dirs, skip_bad=skip_bad, cache_in_memory=cache_in_memory,
                   augment=augment, min_step=min_step, min_stdev_phi=min_stdev_phi,
                   include_stats=include_stats, stat_names=stat_names)

    @property
    def n_base_samples(self) -> int:
        """Count of real, distinct snapshots -- unaffected by augment."""
        return len(self._index)

    def _load(self, idx: int) -> torch.Tensor:
        if self._cache is not None:
            return self._cache[idx]
        run_dir, step, nx, ny = self._index[idx]
        path = run_dir / load.snapshot_filename(step)
        phi = load.read_phi_half(path, nx, ny)     # (ny, nx) float32 numpy
        return torch.from_numpy(phi).unsqueeze(0)  # (1, ny, nx)

    def _load_stats(self, idx: int, k: int = 0, flip: bool = False) -> torch.Tensor:
        run_dir, step, _, _ = self._index[idx]
        try:
            row = self._stats_by_run[run_dir].loc[step]
        except KeyError:
            raise KeyError(
                f"{run_dir}/statistics.csv has no row for step {step} "
                f"(present in metadata.txt save_steps, but missing from statistics.csv)"
            ) from None
        values = [row[name] for name in self.stat_names]
        stats = torch.tensor(values, dtype=torch.float32)

        if (k or flip) and "angle" in self.stat_names:
            angle_idx = self.stat_names.index("angle")
            stats[angle_idx] = _transform_angle(stats[angle_idx], k=k, flip=flip)

        return stats

    def frame_info(self, idx: int) -> tuple[Path, int]:
        """
        (run_dir, step) for a given __getitem__ index -- traces a
        sample back to its exact source file, mirroring
        MicrostructureEvolutionDataset.window_info's role for this
        dataset's own (single snapshot, not a window) indexing. Added
        specifically so callers like check_latent_channels.py's
        random-frame-selection path don't need to reach into the
        private _index themselves (which it previously did, since this
        accessor didn't exist yet).

        Correctly unwraps augmentation: idx is the AUGMENTED index (as
        passed to __getitem__) when augment=True -- same
        divmod(idx, n_aug) base_idx recovery __getitem__ itself uses --
        so this returns the true source frame regardless of which
        dihedral/translation variant idx happened to land on, not a
        meaningless direct index into _index.
        """
        if self.augment:
            n_aug = self._N_DIHEDRAL * 4
            base_idx, _aug_idx = divmod(idx, n_aug)
        else:
            base_idx = idx
        run_dir, step, _nx, _ny = self._index[base_idx]
        return run_dir, step

    def __len__(self) -> int:
        n = len(self._index)
        if self.augment:
            n *= self._N_DIHEDRAL * 4  # 8 D4 elements x 4 translations
        return n

    def _augment_item(self, base_idx: int, aug_idx: int) -> tuple[torch.Tensor, int, bool]:
        x = self._load(base_idx)
        _, _, nx, ny = self._index[base_idx]
        return _apply_augmentation(x, aug_idx, nx, ny)

    def __getitem__(self, idx: int):
        if not self.augment:
            x = self._load(idx)
            return (x, self._load_stats(idx)) if self.include_stats else x

        n_aug = self._N_DIHEDRAL * 4
        base_idx, aug_idx = divmod(idx, n_aug)
        x, k, flip = self._augment_item(base_idx, aug_idx)
        return (x, self._load_stats(base_idx, k=k, flip=flip)) if self.include_stats else x

    def stats_normalization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mean/std of each tracked statistic across all base samples in
        THIS instance. Since splitting now happens at the directory
        level before construction (see split_run_dirs), calling this on
        the train-split dataset is automatically correct -- there's no
        separate subset class to worry about leaking val statistics
        into normalization anymore.
        """
        if not self.include_stats:
            raise ValueError("include_stats=False -- no statistics loaded")
        values = torch.stack([self._load_stats(i) for i in range(len(self._index))])
        mean = values.mean(dim=0)
        std = values.std(dim=0).clamp_min(1e-8)

        # Defense in depth: __init__ already excludes any row with a NaN
        # in a requested column, so this should never fire. But a single
        # NaN slipping through (e.g. a future filtering gap) would
        # silently make mean/std NaN for that column, which then makes
        # StatsLoss NaN for EVERY sample, not just the bad one -- and
        # from there, a shared-encoder gradient update corrupts the
        # model permanently on the very first backward pass. Catching it
        # here, with the specific column named, beats debugging an
        # unexplained epoch-1 NaN in the training loop.
        bad = torch.isnan(mean) | torch.isnan(std)
        if bad.any():
            bad_cols = [name for name, is_bad in zip(self.stat_names, bad.tolist()) if is_bad]
            raise ValueError(
                f"stats_normalization() produced NaN for columns {bad_cols} -- at least "
                f"one pooled sample has NaN in that column despite row-level filtering. "
                f"This should not happen; check for a gap in the NaN-exclusion logic "
                f"in __init__, or that statistics.csv itself isn't corrupted."
            )

        return mean, std


def _truncate_to_max_dt(kept_steps, metadata, max_dt, window_length):
    """Drop the tail of `kept_steps` that no surviving window can reach.

    The max_dt window filter runs AFTER encoding, so without this every run
    has its whole trajectory encoded to keep a short prefix. Measured on the
    128x128 test set at max_dt=150: 1802 windows kept against 11587 dropped,
    i.e. ~90% of the encode discarded.

    The saved steps' gaps are non-decreasing after min_step, so once a
    transition exceeds max_dt every later one does too and everything beyond
    it is unreachable. But this works from the ACTUAL kept_steps gaps rather
    than assuming that: kept_steps is the schedule after min_stdev_phi
    filtering, and dropping an INTERIOR step merges two gaps into a larger
    one, which can break monotonicity. Dropping a prefix -- the usual case,
    early frames not yet phase-separated -- preserves it.

    So: scan forward, stop at the first transition exceeding max_dt. That is
    correct whether or not the gaps are monotonic, because a window starting
    after that point would have to cross it. It is CONSERVATIVE if
    monotonicity is broken (a later short gap is discarded along with the
    reachable window it might have supported), which costs a few windows and
    never keeps an invalid one -- the safe direction, and the window filter
    downstream is unchanged and still authoritative.

    `dt > max_dt` to cut, matching the window filter exactly: a transition of
    exactly max_dt is KEPT.
    """
    if max_dt is None or len(kept_steps) < 2:
        return kept_steps
    for i in range(len(kept_steps) - 1):
        if (kept_steps[i + 1] - kept_steps[i]) * metadata.dt > max_dt:
            # steps[0..i] remain reachable: the transition i -> i+1 is the
            # first that no window may cross.
            return kept_steps[:i + 1]
    return kept_steps


class MicrostructureEvolutionDataset(Dataset):
    """
    Sequences of consecutive frames for LDS-family training. Two modes,
    selected by whether encoder is given:

    - encoder given (stage 3, frozen encoder): z(t) is computed ONCE per
      kept step, upfront, and cached -- not re-encoded every batch.
      Latents are tiny compared to raw images (e.g. a 4x8x8 latent is
      ~200x smaller than a 64x64 snapshot), so caching every one is
      cheap even for a large sweep. __getitem__ returns latent windows.

    - encoder=None (stage 4/5, E is trainable): caching latents would be
      wrong here -- gradient must flow through E fresh every forward
      pass, and a value computed once under torch.no_grad() can't
      provide that. __getitem__ instead returns RAW PIXEL windows;
      encoding happens fresh in the training loop, every epoch. This is
      substantially more expensive per epoch than the cached-latent
      mode (a full E forward pass per frame per window, every epoch,
      rather than once ever) -- expect smaller batch sizes than stage 3
      needed.

    Windows are n_r+1 consecutive KEPT steps from a single run (never
    crossing run boundaries -- a "sequence" spanning two different runs
    isn't a real trajectory), in either mode.

    __getitem__ returns (window, dt_window, theta), or (window, dt_window,
    theta, true_stats) if stat_names was given at construction (see
    stat_names below -- only meaningful/available in raw-pixel mode), or
    (window, window_deriv, dt_window, theta) if encode_both_streams was
    given at construction (see that parameter below -- only meaningful
    with encoder given; window_deriv has the same shape as window, the
    "deriv" stream's own cached latents instead of "state"'s):
      window:    (n_r+1, latent_channels, 8, 8) if encoder was given
                 (a window of z's), or (n_r+1, 1, ny, nx) if encoder is
                 None (a window of raw frames x, encoding deferred to
                 the training loop).
      dt_window: (n_r,) float32, physical time between consecutive steps
                 in the window, i.e. (step_b - step_a) * metadata.dt.
      theta:     (N_THETA,) float32, [T - T0, log(T0 - T)] -- see
                 models.constants.theta_coordinates for why BOTH the linear
                 and the log coordinate are supplied (the log linearises the
                 power-law physical scales near T0). Per
                 docs/neural_nets.md, f_theta's theta subscript denotes
                 PHYSICAL parameters (e.g. temperature), not the neural
                 net's own learnable weights -- conditioning on theta is
                 what lets one LDS model generalize across a whole sweep
                 of different runs, rather than needing a separate model
                 per temperature. Centered on T0 (the Landau potential's
                 threshold temperature, metadata.T0) rather than raw
                 temperature: the whole sweep is subcritical (T < T0),
                 and distance from T0 is the physically meaningful
                 quantity governing dynamics, not distance from 0 --
                 centering puts that boundary at theta=0 for free rather
                 than requiring the network to discover T0's significance
                 from data. Only temperature (via this centering) is
                 included: noise only sets the INITIAL condition phi(0)
                 and never appears in the governing Allen-Cahn equation
                 itself (a0, b, T0, kappa, mobility are fixed constants
                 across the current sweep), so two runs at the same
                 temperature with different noise follow IDENTICAL
                 dynamics from different starting points -- noise is not
                 a dynamics-relevant conditioning variable. seed is pure
                 RNG choice, not physical, and is also excluded.
                 Constant across a window (one run), unlike dt_window.
      true_stats: (n_stats,) RAW (not normalized) statistics.csv row for
                 window[0] specifically -- the real STARTING frame, never
                 a predicted one, same anchoring convention stage 1/2
                 already use for L_stats. Only present if stat_names was
                 given.

    dt is always computed from the ACTUAL kept step numbers, never
    assumed uniform: your save_steps are already irregularly spaced, and
    min_step/min_stdev_phi filtering can additionally skip an
    intermediate step, widening the gap between the two steps that end
    up adjacent in the kept sequence. Both cases are handled identically
    here -- this matches the docs' explicit design intent that the LDS
    can operate at a coarser, non-uniform effective time resolution than
    the raw solver, rather than assuming one fixed dt throughout.

    Filtering (skip_bad/min_step/min_stdev_phi) is delegated to
    build_good_steps(), the same function MicrostructureSnapshotDataset
    uses, so the two classes are guaranteed to agree on which steps are
    usable for the same run_dirs/filter settings -- pass a precomputed
    good_steps= dict to skip re-scanning if you're building both from
    the same run_dirs.
    """

    def __init__(self, run_dirs: list[str | Path], encoder: torch.nn.Module | None,
                 device: str | torch.device = "cpu", window_length: int = 5,
                 skip_bad: bool = True, min_step: int = 0,
                 min_stdev_phi: float | None = None, min_passing_steps: int | None = None,
                 min_normalized_stdev_phi: float | None = None,
                 encode_batch_size: int = 256,
                 good_steps: dict[Path, list[int]] | None = None,
                 stat_names: list[str] | None = None, augment: bool = False,
                 min_std_deriv: float | None = None, encode_both_streams: bool = False,
                 max_dt: float | None = None, latent_cache_dir: Path | str | None = None,
                 stats_frame_index: int = 0,
                 fixed_aug_indices: tuple[int, ...] | list[int] | None = None,
                 read_workers: int = 16, split_label: str = "",
                 time_coordinate: str = "t", return_phys_dt: bool = False,
                 return_frame_t: bool = False, derivative_source: str = "z1",
                 cache_info: dict | None = None, require_consecutive: bool = True):
        """
        encoder: pass a frozen encoder for the cached-latent mode (stage
        3), or None for the raw-pixel mode (stage 4/5, E trainable) --
        see class docstring.

        encode_both_streams: False (default) -- exact prior behavior,
        only the "state" stream (z0) gets encoded and cached. If True,
        ALSO encodes and caches the "deriv" stream (z1) alongside it --
        needed by the split-latent LDS architecture (LatentDynamics),
        which requires z1(t+dt) as ground truth (to train g_theta's own
        target), not just z0(t+dt) the way the old single-stream
        architecture did. Only meaningful with encoder given (raises
        otherwise, same rationale as stat_names/augment/min_std_deriv's
        own encoder=None restrictions above) -- doubles this
        constructor's own encoding cost (already the most expensive
        dataset build in the pipeline), so left opt-in rather than
        default, for callers that only need z0.

        window_length: n_r + 1 (e.g. window_length=5 -> n_r=4 predicted steps).
        encode_batch_size: batch size for the upfront encoding pass when
        encoder is given; unused (raw frames aren't encoded here) when
        encoder is None. Chunks span RUN BOUNDARIES -- every kept frame
        across every run is concatenated first, then encoded in
        encode_batch_size chunks over that single combined tensor, not
        run-by-run -- so this can be sized for GPU/VRAM throughput
        without worrying about how many frames any individual run
        happens to have (a run with fewer frames than this no longer
        means an undersized, launch-overhead-bound encode batch for
        just that run).
        good_steps: a precomputed {run_dir: [kept_step, ...]} mapping
        from build_good_steps(), to skip re-scanning run_dirs when a
        paired MicrostructureSnapshotDataset over the same run_dirs
        already computed it with the same skip_bad/min_step/min_stdev_phi.
        Default None computes it here.

        stat_names: if given, ALSO loads statistics.csv per run and
        returns true_stats (see class docstring's __getitem__ section)
        for stage 4/5's L_stats, which -- like stage 1/2's -- is always
        anchored to a real, ground-truth-labeled frame, never a
        prediction. Only meaningful in raw-pixel mode (encoder=None);
        stage 3 never uses L_stats, so combining this with encoder-given
        is treated as a caller mistake worth catching at construction
        time, not something silently ignored. None (default): no
        statistics.csv reads at all, __getitem__ returns the plain
        3-tuple exactly as before this parameter existed. Like
        MicrostructureTripletDataset's identical parameter, must match
        the stage-2 checkpoint's stats_config exactly -- not
        auto-detected, since silently resolving a different schema than
        the already-trained stats_head expects would be a much worse
        failure mode than requiring it explicitly.

        augment: False (default) -- EXACT prior behavior, unchanged for
        every existing caller (stage 3, stage 4/5). If True, applies
        MicrostructureSnapshotDataset's own D4-dihedral-x-4-translation
        scheme (see _apply_augmentation, shared between both classes),
        multiplying __len__ by N_DIHEDRAL(8) x 4 = 32 -- but critically,
        the SAME (k, flip, shift) is applied to EVERY frame in a
        window, not independently per frame: flipping x(t) while
        leaving x(t+dt) unflipped would make the pair describe a
        physically meaningless "evolution", not a genuine augmented
        view of a real one. Restricted to raw-pixel mode (encoder=None)
        -- augmenting a cached LATENT has no well-defined meaning here
        (unlike a raw pixel grid, there's no established correspondence
        between D4/translating the 8x8 latent grid and any real
        symmetry of the underlying physics). If stat_names includes
        "angle", it's corrected the same way
        MicrostructureSnapshotDataset's own augmentation corrects it
        (_transform_angle, applied using the SAME (k, flip) the window
        itself was transformed by) -- so "angle" correctly describes
        the augmented frame's actual orientation, not the unaugmented
        one. Every other stat (avg_phi, stdev_phi, etc.) is a
        rotation/flip-invariant scalar, unaffected by which augmented
        variant produced the frame, and needs no correction at all.

        min_std_deriv: None (default) -- no filtering beyond
        min_step/min_stdev_phi. If given, ALSO excludes any candidate
        window whose first transition's (x(t+dt)-x(t))/dt has spatial
        std BELOW this threshold -- a DIFFERENT axis from
        min_stdev_phi, which only ever looks at a single frame's own
        spatial variance and says nothing about how much two
        consecutive frames actually differ. A microstructure can be
        genuinely spatially complex (comfortably passing min_stdev_phi
        at BOTH endpoints -- e.g. sharp, well-defined straight-strip
        interfaces) while being essentially stationary between those
        two specific saved steps (a straight interface has zero
        curvature, and curvature-driven interface motion is
        proportional to curvature -- zero curvature means zero
        velocity, regardless of how sharp or well-resolved the
        interface itself is). That's a real, physically legitimate
        state this class's other filters were never designed to catch,
        since they only ever examine one frame at a time, not a pair.
        Only meaningful in raw-pixel mode (encoder=None) -- raises
        otherwise, matching augment's own restriction and for the
        analogous reason (no well-defined meaning against a cached
        latent). Computed directly from each candidate window's own
        already-loaded frames (no extra disk I/O), so this is cheap
        even though it's a per-window check rather than a per-step one.

        fixed_aug_indices: None (default) -- every window is returned
        exactly once, untransformed. If given, every window is instead
        returned once PER LISTED VARIANT, each transformed by that
        variant's own (k, flip, shift) -- so __len__ is multiplied by
        len(fixed_aug_indices), the same way augment=True multiplies it
        by _N_AUGMENT_VARIANTS. See VAL_DECORRELATED_AUG_INDICES (this
        module) for the recommended 4-variant set and the full reasoning
        behind which variants are safe to use here.

        Purpose: VALIDATION-side averaging. Since augmentation applies
        to train only, a val set here is ~100x smaller in window count
        than its train counterpart, AND fixed -- so its epoch-to-epoch
        swing is not sampling noise, it's the model itself moving and
        interacting idiosyncratically with those specific windows.
        Evaluating each window under several exact-symmetry variants and
        averaging damps that, because the model isn't perfectly
        equivariant, so its errors on transformed copies are partially
        decorrelated. Strictly weaker than simply having more DISTINCT
        val windows (which are genuinely independent rather than
        symmetry-related), but far cheaper -- 4x on a val set that's
        already a rounding error against the augmented train set.

        Must be an explicit, FIXED list, never resampled per epoch:
        drawing fresh random variants each epoch would ADD a noise
        source on top of the model variation this exists to see
        through. Mutually exclusive with augment=True, and (like
        augment) requires encoder=None.

        read_workers: 16 (default) -- every raw snapshot file is read
        via its own read_phi_half() call (see that function's own
        docstring: a single, small np.fromfile per file, no header, no
        batching across files at the OS level). Read one at a time,
        that's thousands of small, independently-blocking file reads
        across a real sweep (hundreds of runs x dozens-to-hundreds of
        snapshots each) with NOTHING else happening while each one is
        in flight -- I/O-latency-bound, not CPU- or GPU-bound (measured
        directly: both sit well under saturated during this phase).
        Reading is spread across a ThreadPoolExecutor instead of a
        plain per-run list comprehension specifically because
        np.fromfile releases the GIL for the actual read syscall, so
        multiple threads can have reads genuinely in flight
        concurrently despite Python's GIL -- this is NOT compute
        parallelism (do not expect a read_workers-x speedup; expect a
        speedup bounded by how much read LATENCY, not CPU time, was
        actually being wasted). ONE pool is created for this whole
        constructor call (see its use below), not one per run --
        creating/tearing down a pool per run_dir would add its own
        fixed overhead on every single run, largely cancelling out the
        benefit for runs with only a handful of snapshots each. Set to
        1 to reproduce the exact prior sequential-read behavior (still
        correct, just slow) if concurrent reads are ever undesirable
        (e.g. a networked filesystem that penalizes concurrent access
        rather than benefiting from it -- worth checking directly for
        your own storage if this doesn't help as expected).
        """
        if stat_names is not None and encoder is not None:
            raise ValueError(
                "stat_names was given together with a real encoder (cached-latent, stage-3 "
                "mode) -- stage 3 never uses L_stats, so this combination is almost certainly "
                "a mistake. Pass encoder=None (raw-pixel mode) if you actually want true_stats."
            )
        if window_length < 2:
            raise ValueError(f"window_length must be >= 2 (got {window_length}), "
                              f"since a window needs at least one transition to predict")
        if augment and encoder is not None:
            raise ValueError(
                "augment=True was given together with a real encoder (cached-latent, stage-3 "
                "mode) -- augmenting a cached LATENT has no well-defined meaning (see this "
                "constructor's own docstring). Pass encoder=None if you want augmented raw-pixel "
                "windows."
            )
        if fixed_aug_indices is not None and encoder is not None:
            raise ValueError(
                "fixed_aug_indices was given together with a real encoder (cached-latent, "
                "stage-3 mode) -- same reason augment=True is rejected above (transforming a "
                "cached LATENT has no well-defined meaning), plus a second one specific to "
                "this mode: under encode_both_streams, __getitem__ transforms `window` but "
                "NOT `window_deriv`, so the two streams would describe DIFFERENTLY-oriented "
                "views of the same frame. Pass encoder=None for raw-pixel windows."
            )
        if min_std_deriv is not None and encoder is not None:
            raise ValueError(
                "min_std_deriv was given together with a real encoder (cached-latent, stage-3 "
                "mode) -- this filters on the RAW PIXEL derivative's spatial std, which has no "
                "well-defined meaning against a cached latent (see this constructor's own "
                "docstring). Pass encoder=None if you want this filter."
            )
        if encode_both_streams and encoder is None:
            raise ValueError(
                "encode_both_streams=True was given together with encoder=None (raw-pixel mode) "
                "-- there's no encoder to run the 'deriv' stream through at all in this mode. "
                "Pass a real encoder (cached-latent, stage-3 mode) if you want both streams."
            )

        if not 0 <= stats_frame_index < window_length:
            raise ValueError(
                f"stats_frame_index={stats_frame_index} is out of range for "
                f"window_length={window_length} -- must be a valid index INTO the window "
                f"(0 = the window's own first frame, the long-standing default; "
                f"window_length-1 = its last)."
            )

        self.window_length = window_length
        self.encoder_given = encoder is not None  # which mode __getitem__ operates in
        self.encode_both_streams = encode_both_streams
        self.stat_names = stat_names
        self.augment = augment
        # derivative_source: "z1" (default) serves the encoder's deriv stream as
        # window_deriv, exactly as before. "previous_quotient" instead serves the
        # backward quotient of z0, q_k = (z0_k - z0_{k-1})/du_{k-1} (the "previous
        # derivative" the causal baseline uses), so f trains on q, and the
        # autonomous rollout's step-0 seed (z1_sequence[:,0]) becomes the real
        # quotient rather than z1. Needs the deriv stream too (the run-start
        # fallback, where no predecessor frame exists); augmentation is not yet
        # handled for the quotient path.
        if derivative_source not in ("z1", "previous_quotient"):
            raise ValueError(
                f"derivative_source must be 'z1' or 'previous_quotient', got "
                f"{derivative_source!r}")
        if derivative_source == "previous_quotient":
            if not encode_both_streams:
                raise ValueError(
                    "derivative_source='previous_quotient' requires "
                    "encode_both_streams=True (it reads the z0 stream to form the "
                    "quotient and the z1 stream for the run-start fallback).")
            if augment:
                raise ValueError(
                    "derivative_source='previous_quotient' does not support "
                    "augment=True yet (the predecessor frame would need the same "
                    "augmentation as the window).")
        self.derivative_source = derivative_source
        self.min_std_deriv = min_std_deriv
        self.max_dt = max_dt
        self.require_consecutive = require_consecutive
        self._split_label = split_label
        # Latent cache. Only meaningful with a FROZEN encoder, which is what
        # stage 3 has by construction -- 3a and 3b load the same stage-2
        # checkpoint, and each diagnostic then encodes the test set again.
        # The key holds a hash of the encoder's WEIGHTS, not its checkpoint
        # path: paths get reused (stage 2 retrains in place, force=True
        # overwrites) and mtimes survive a copy, so neither says anything
        # about what the weights are, and a stale hit would train on latents
        # from a different encoder with no error anywhere.
        self._latent_cache_root = Path(latent_cache_dir) if latent_cache_dir else None
        self._cache_info = cache_info or {}
        self._cache_info_written: set = set()
        self._encoder_fingerprint = (
            latent_cache.encoder_fingerprint(encoder)
            if self._latent_cache_root is not None and encoder is not None else None)
        if self._latent_cache_root is not None and encoder is None:
            self._latent_cache_root = None  # nothing to cache: frames are returned raw
        # Which frame IN the window true_stats (and its own NaN guard)
        # refers to. 0 (default) is the long-standing behavior every
        # existing caller relies on: window[0], the window's own
        # starting frame. train_stage2.py's deriv_target_centered mode
        # needs 1 instead, since there the frame actually encoded (and
        # thus the frame stats_head/stats_head1 are asked to predict
        # statistics FOR) is the window's MIDDLE frame, not its first
        # -- leaving this at 0 in that mode silently trains both stats
        # anchors against a DIFFERENT frame's statistics than the one
        # they're predicting from, which is a real (measured, not
        # hypothetical) misalignment, not a rounding-level detail.
        self.stats_frame_index = stats_frame_index

        # fixed_aug_indices: evaluate EVERY window under this exact,
        # FIXED list of augmentation variants (see
        # VAL_DECORRELATED_AUG_INDICES for the recommended set and the
        # reasoning behind it), multiplying __len__ by len(list) the
        # same way augment=True multiplies it by _N_AUGMENT_VARIANTS.
        #
        # Intended for VALIDATION-side averaging: the val set here is
        # small (augmentation applies to train only, so val is ~100x
        # smaller in window count) and FIXED, so its epoch-to-epoch
        # swing isn't sampling noise -- it's the model itself moving,
        # interacting idiosyncratically with those specific windows.
        # Averaging each window over several exact-symmetry variants
        # damps that, since the model isn't perfectly equivariant and
        # its errors on transformed copies are partially decorrelated.
        # Strictly weaker than simply having more DISTINCT val windows
        # (those are genuinely independent), but far cheaper.
        #
        # MUST be a fixed list, never resampled per epoch -- drawing
        # random variants each epoch would ADD a fresh noise source on
        # top of the model variation this is meant to see through,
        # achieving the exact opposite of the goal. That's why this
        # takes an explicit list rather than a count.
        if fixed_aug_indices is not None:
            if augment:
                raise ValueError(
                    "fixed_aug_indices and augment=True are mutually exclusive -- augment "
                    "already expands every window across all "
                    f"{_N_AUGMENT_VARIANTS} variants (randomized coverage for TRAINING), "
                    "while fixed_aug_indices pins an exact, reproducible subset (for "
                    "deterministic VALIDATION averaging). Combining them would multiply "
                    "the index twice over, with no coherent meaning."
                )
            if len(fixed_aug_indices) == 0:
                raise ValueError("fixed_aug_indices must be non-empty (pass None to disable)")
            if len(set(fixed_aug_indices)) != len(fixed_aug_indices):
                raise ValueError(
                    f"fixed_aug_indices has duplicates: {list(fixed_aug_indices)} -- a repeated "
                    "variant just double-weights that one view, it doesn't add decorrelation"
                )
            bad_range = [i for i in fixed_aug_indices if not 0 <= i < _N_AUGMENT_VARIANTS]
            if bad_range:
                raise ValueError(
                    f"fixed_aug_indices out of range: {bad_range} -- must each be in "
                    f"[0, {_N_AUGMENT_VARIANTS})"
                )
            if stat_names is not None and "angle" in stat_names:
                unsafe = sorted(set(fixed_aug_indices) - _ANGLE_SAFE_AUG_INDICES)
                if unsafe:
                    print(
                        f"WARNING: fixed_aug_indices includes {unsafe}, which use k=1/k=3 "
                        f"dihedral rotations -- _transform_angle's own docstring flags the SIGN "
                        f"of its k*90 correction for those as UNCONFIRMED, and 'angle' IS in "
                        f"stat_names, so this may put SYSTEMATIC error into the stats loss. "
                        f"Consider VAL_DECORRELATED_AUG_INDICES ({list(VAL_DECORRELATED_AUG_INDICES)}) "
                        f"instead, which spans 4 distinct dihedrals and 4 distinct translations "
                        f"using only confirmed-correct angle handling."
                    )
        self.fixed_aug_indices = tuple(fixed_aug_indices) if fixed_aug_indices is not None else None
        self._run_dirs: list[Path] = []         # run_dir per run_idx, for tracing samples back
        if time_coordinate not in ("t", "log10_t"):
            raise ValueError(
                f"time_coordinate must be 't' or 'log10_t', got {time_coordinate!r}")
        self.time_coordinate = time_coordinate
        # u-mode is SINGULAR at t=0: u=log10(t) -> -inf and Delta-u=log10(t1/0)
        # -> inf, so a window containing step 0 emits NaN downstream. If a
        # log10_t dataset is built without excluding step 0 (min_step<1), raise
        # min_step to 1 and SAY SO, rather than silently producing NaN windows.
        if self.time_coordinate == "log10_t" and min_step < 1:
            print(f"MicrostructureEvolutionDataset: WARNING time_coordinate="
                  f"'log10_t' is singular at t=0 -- raising min_step from "
                  f"{min_step} to 1 to drop the step-0 frame (u=log10(t) is "
                  f"undefined there). Pass min_step>=1 explicitly to silence.")
            min_step = 1
        # return_phys_dt: when True (only the LDS trainer opts in), the
        # encode_both_streams __getitem__ appends PHYSICAL dt as a 5th element,
        # alongside dt_window (which is Delta-u in log10_t mode -- the model's
        # STEP). The loss weights by physical dt so a u-run's objective matches a
        # t-run's, isolating the coordinate change from the (dt-dependent) loss
        # weighting. Off by default => the 4-tuple is byte-identical, so every
        # diagnostic and test is untouched. In t-mode phys dt == dt_window.
        self.return_phys_dt = return_phys_dt
        # return_frame_t: raw-pixel path appends per-frame PHYSICAL time
        # t=(step*sim_dt) (n_r+1,) so a stage-4 u-scheme loss can build
        # z̃1=ln10*t*z1 and Delta-u. Gated -> default 4-tuple byte-identical.
        self.return_frame_t = return_frame_t
        self._run_steps: list[list[int]] = []   # kept step numbers per run, in order
        self._run_du: list[list[float] | None] = []  # per run: log10(t_{i+1}/t_i) when u-mode
        self._run_data: list[torch.Tensor] = []  # per run, on CPU: "state" latents (n_kept,C,8,8) if
                                                   # encoder given, else raw frames (n_kept,1,ny,nx)
        self._run_data_deriv: list[torch.Tensor] = []  # per run, on CPU: "deriv" latents (n_kept,C,8,8)
                                                   # -- ONLY populated if encode_both_streams=True
        self._run_dt_scale: list[float] = []    # metadata.dt per run
        self._run_theta: list[torch.Tensor] = []  # (n_theta,) physical params per run -- see class docstring
        self._run_nx: list[int] = []            # metadata.nx per run, only used if augment=True
        self._run_ny: list[int] = []            # metadata.ny per run, only used if augment=True
        self._stats_by_run = {}                 # run_dir -> statistics.csv DataFrame, only if stat_names given
        self._index: list[tuple[int, int]] = []  # (run_idx, window_start_position)

        run_dirs = [Path(d) for d in run_dirs]
        if good_steps is None:
            good_steps = build_good_steps(run_dirs, skip_bad, min_step, min_stdev_phi,
                                          min_passing_steps, split_label=split_label,
                                          min_normalized_stdev_phi=min_normalized_stdev_phi)

        if self.encoder_given:
            encoder = encoder.to(device).eval()
        # Checked ONCE, up front, not assumed: encoder is typed as any
        # torch.nn.Module, not specifically the real Encoder class (test
        # fixtures, older/simplified stand-ins, or any future variant
        # may have a forward(x) with no theta parameter at all) --
        # calling such an encoder with theta=... would raise TypeError,
        # not silently ignore it. inspect, not try/except: an explicit
        # signature check can't mask an unrelated TypeError from
        # elsewhere inside the real call the way a broad except would.
        encoder_accepts_theta = (
            self.encoder_given and "theta" in inspect.signature(encoder.forward).parameters
        )
        pending_meta, run_data_list, run_data_deriv_list = self._read_and_encode_all_runs(
            run_dirs, good_steps, encoder, device, window_length, encode_batch_size,
            read_workers, encoder_accepts_theta,
        )
        self._build_window_index(pending_meta, run_data_list, run_data_deriv_list)
        # q-scheme "compute before filter": give each run's FIRST kept frame a
        # real backward-quotient predecessor from the full save_steps sequence,
        # so q_0 there is the true previous derivative, not the z1 fallback. Done
        # AFTER the window index (needs the finalised per-run order) and only for
        # the quotient path with a real encoder.
        self._run_pred_z0: list = [None] * len(self._run_steps)
        self._run_pred_du: list = [None] * len(self._run_steps)
        self._quotient_precomputed = False
        if self.derivative_source == "previous_quotient" and self.encoder_given:
            self._encode_run_start_predecessors(encoder, device, encoder_accepts_theta)

    def _encode_run_start_predecessors(self, encoder, device,
                                       encoder_accepts_theta: bool) -> None:
        """For each run, read+encode the ONE save_steps frame immediately before
        its first kept frame -- the predecessor the filter dropped -- and store
        its z0 (plus the coordinate step to the first kept frame). This is the
        "compute the quotient on the full sequence, then filter" fix: the NaN
        that q_0 would be at a run's ABSOLUTE first snapshot never surfaces
        (min_step drops it), and every kept run's first frame gets a genuine
        backward quotient instead of z1. Runs whose first kept frame IS the
        first saved frame (no predecessor) keep None -> z1 fallback there only.

        Encodes are BATCHED (one frame per run, gathered into chunks) rather
        than 2750 sequential batch-1 forwards, and afterwards the per-run
        quotient is PRECOMPUTED once and swapped in place of the deriv stream
        (whose only remaining role -- the run-start fallback -- is baked into
        frame 0). __getitem__ then serves the quotient as a plain slice, the
        same cost as the z1 path, instead of rebuilding a du tensor from a
        Python list per window (~100k times per epoch on num_workers=0)."""
        pending: list[tuple[int, torch.Tensor, torch.Tensor | None, float]] = []
        for run_idx, run_dir in enumerate(self._run_dirs):
            metadata = load.read_metadata(run_dir / "metadata.txt")
            save_steps = list(metadata.save_steps)
            first_kept = self._run_steps[run_idx][0]
            try:
                pos = save_steps.index(first_kept)
            except ValueError:
                continue
            if pos == 0:
                continue                       # first kept == first saved: no predecessor
            pred_step = save_steps[pos - 1]
            if pred_step == 0:
                continue                       # predecessor is t=0 (the run's absolute
                                               # first snapshot): log10 singular / dt=0
                                               # -- the "not applicable at t=0" case; z1
                                               # fallback for this run's first frame only
            phi = load.read_phi_half(
                run_dir / load.snapshot_filename(pred_step), metadata.nx, metadata.ny)
            frame = torch.from_numpy(phi).unsqueeze(0)          # (1, ny, nx)
            theta = (torch.tensor(theta_coordinates(metadata.temperature, metadata.T0),
                                  dtype=torch.float32)
                     if encoder_accepts_theta else None)
            du_pred = (math.log10(first_kept / pred_step)
                       if self.time_coordinate == "log10_t"
                       else (first_kept - pred_step) * metadata.dt)
            pending.append((run_idx, frame, theta, du_pred))

        chunk = 256
        for lo in range(0, len(pending), chunk):
            part = pending[lo:lo + chunk]
            batch = torch.stack([f for _, f, _, _ in part]).to(device)   # (B,1,ny,nx)
            with torch.no_grad():
                if encoder_accepts_theta:
                    thetas = torch.stack([t for _, _, t, _ in part]).to(device)
                    encoded = encoder(batch, theta=thetas)
                else:
                    encoded = encoder(batch)
            z0 = encoded[DEFAULT_STREAM_NAME].cpu()
            for k, (run_idx, _, _, du_pred) in enumerate(part):
                self._run_pred_z0[run_idx] = z0[k]                        # (C, 8, 8)
                self._run_pred_du[run_idx] = du_pred

        # Precompute the per-run quotient ONCE and swap it in for the deriv
        # stream: q_k = (z0_k - z0_{k-1})/du_{k-1}, frame 0 from the predecessor
        # (or the old z1 where there is none). Same memory (replaces z1), and
        # __getitem__'s existing plain slice now serves the quotient directly.
        for run_idx in range(len(self._run_data)):
            z0 = self._run_data[run_idx]
            du = self._du_slice(run_idx, 0, z0.shape[0])                  # (n-1,)
            q = torch.empty_like(z0)
            q[1:] = (z0[1:] - z0[:-1]) / du.view(-1, 1, 1, 1)
            pred = self._run_pred_z0[run_idx]
            if pred is not None:
                q[0] = (z0[0] - pred) / self._run_pred_du[run_idx]
            else:
                q[0] = self._run_data_deriv[run_idx][0]                   # z1 fallback
            self._run_data_deriv[run_idx] = q
        self._quotient_precomputed = True

    def _read_and_encode_all_runs(
        self, run_dirs: list[Path], good_steps: dict, encoder: torch.nn.Module | None,
        device: str | torch.device, window_length: int, encode_batch_size: int,
        read_workers: int, encoder_accepts_theta: bool,
    ) -> tuple[list, list, list]:
        """
        Reads every kept snapshot for every run (via a shared
        ThreadPoolExecutor -- see read_workers' own docstring in
        __init__ for why), and, when self.encoder_given, encodes them
        through a bounded-memory streaming buffer that batches across
        run boundaries (see _flush_buffer's own docstring below for
        the full rationale) rather than accumulating every run's raw
        frames before encoding any of it. Extracted verbatim from
        __init__'s own former body -- same logic, same order, just
        named and callable on its own.

        Returns (pending_meta, run_data_list, run_data_deriv_list):
        pending_meta is (run_dir, metadata, kept_steps) for every run
        that passed the length check, aligned index-for-index with
        run_data_list (per-run "state" latents if self.encoder_given,
        else raw frames) and run_data_deriv_list (per-run "deriv"
        latents if self.encode_both_streams, else None per entry).
        """
        n_windowless_runs = 0

        pending_meta = []       # (run_dir, metadata, kept_steps), one per run that passed the length check
        # Keyed by POSITION in pending_meta rather than appended in flush
        # order. A cache hit skips the encode buffer entirely, so it would
        # otherwise land wherever the next flush happened to reach -- pairing
        # one run's latents with another run's steps, which no shape check
        # would catch.
        run_data_by_index = {}
        run_deriv_by_index = {}
        buffer_run_indices = []   # pending_meta positions for the frames currently buffered
        n_cache_hits = 0

        # Bounded-memory streaming buffer for cross-run batched encoding.
        # An EARLIER version of this accumulated every run's raw frames
        # into one big list BEFORE encoding any of it -- correct, but a
        # real problem at real sweep sizes, not just a theoretical one:
        # read_phi_half converts every frame to float32 in memory (see
        # its own docstring: raw storage is float16, read_phi_half's
        # return value is not), so a sweep with N total kept frames
        # needs 4*ny*nx*N bytes just for the raw frames -- for a 7GB
        # (on-disk, float16) 256x256 sweep, that's ~14GB of raw frames
        # alone if ever held all at once, on top of whatever else the
        # process needs. The buffer below is instead flushed (encoded,
        # then discarded) as soon as it reaches encode_batch_size,
        # keeping peak raw-frame memory bounded by roughly
        # encode_batch_size + one run's own frame count, REGARDLESS of
        # how large the full sweep is -- while still getting almost all
        # of cross-run batching's actual benefit (see the encode call
        # below): most flushes still span several runs at close to the
        # requested batch size, not one tiny call per run.
        buffer_frames = []  # raw frame tensors not yet encoded, one entry per buffered run
        buffer_sizes = []   # frame count per buffered entry, same order as buffer_frames
        buffer_thetas = []  # theta broadcast to (n_kept, n_theta) per buffered run, same order
        buffer_total = 0

        def _flush_buffer():
            """
            Encode everything currently buffered (spanning however many
            runs happen to be in it) in ONE pass, splitting the result
            back per run. Safe to batch across runs specifically because
            encoder.eval() was already called above, BEFORE this: the
            encoder here DOES contain BatchNorm2d (see train_ae.py's own
            extensive freeze/eval-mode handling, and
            test_train_stage2_c0c1.py's own drift tests), but in eval() mode it
            normalizes using its saved running_mean/running_var, not
            live per-batch statistics -- so a given frame's own encoded
            output is unaffected by which OTHER frames happen to share
            its batch (verified directly, not just assumed: a real
            BatchNorm2d layer in eval() mode gave byte-identical output
            whether encoded alone, per-run, or batched across run
            boundaries). This WOULD be unsafe in train() mode -- another
            reason .eval() being called first, and never reset to
            .train() anywhere in this constructor, matters. A no-op
            (returns empty lists) if the buffer is currently empty, so
            callers never need to check emptiness themselves.

            theta is passed to EVERY encode call unconditionally (not
            only when some stream is known to need it) -- Encoder.forward
            computes every one of its streams in a single pass (see its
            own docstring), so it needs theta if ANY of them requires
            conditioning, regardless of which stream(s) this dataset
            instance actually keeps; Encoder itself is the single place
            that knows which streams actually use it, and simply ignores
            theta for a model with no conditioned streams at all.
            """
            nonlocal buffer_frames, buffer_sizes, buffer_thetas, buffer_total
            if not buffer_frames:
                return [], []
            combined = torch.cat(buffer_frames, dim=0)
            combined_theta = torch.cat(buffer_thetas, dim=0) if encoder_accepts_theta else None
            latents = []
            latents_deriv = [] if self.encode_both_streams else None
            with torch.no_grad():
                for i in range(0, len(combined), encode_batch_size):
                    batch = combined[i:i + encode_batch_size].to(device)
                    if encoder_accepts_theta:
                        theta_batch = combined_theta[i:i + encode_batch_size].to(device)
                        encoded = encoder(batch, theta=theta_batch)
                    else:
                        encoded = encoder(batch)  # ONE forward pass, both streams available in the dict
                    latents.append(encoded[DEFAULT_STREAM_NAME].cpu())
                    if self.encode_both_streams:
                        latents_deriv.append(encoded["deriv"].cpu())
            del combined, combined_theta  # no longer needed -- everything from them is now in latents/latents_deriv
            latents = torch.cat(latents, dim=0)
            latents_deriv = torch.cat(latents_deriv, dim=0) if self.encode_both_streams else None

            result = list(torch.split(latents, buffer_sizes))
            result_deriv = (list(torch.split(latents_deriv, buffer_sizes))
                             if self.encode_both_streams else [None] * len(buffer_sizes))
            buffer_frames, buffer_sizes, buffer_thetas, buffer_total = [], [], [], 0
            return result, result_deriv

        # ONE pool for the whole constructor call -- see read_workers'
        # own docstring for why not one per run_dir. Reading and
        # (buffered) encoding now happen in the SAME per-run pass --
        # merging them back together, after the streaming buffer above
        # made accumulating a separate whole-dataset frame list
        # unnecessary, avoids ever building that list at all.
        # Progress for the read/encode pass -- the one long silent stretch of
        # dataset construction. At 128x128 on CPU, encoding hundreds of runs'
        # frames takes minutes with no output between "Loaded ... encoder" and
        # "Evaluating N windows", which looks hung. Print an in-place counter
        # (only when actually encoding, i.e. an encoder was given, and only for
        # a non-trivial run count) so a cold-cache run shows steady progress and
        # a warm-cache run visibly flies (cache hits skip the encode entirely).
        _show_progress = self.encoder_given and len(run_dirs) >= 20
        _n_total = len(run_dirs)
        _t0 = time.monotonic()

        with ThreadPoolExecutor(max_workers=read_workers) as read_pool:
            for _run_pos, run_dir in enumerate(run_dirs):
                if _show_progress and (_run_pos % 25 == 0 or _run_pos == _n_total - 1):
                    sys.stdout.write(
                        f"\r  encoding runs: {format_progress_count(_run_pos + 1, _n_total)} "
                        f"({n_cache_hits} cache hit{'' if n_cache_hits == 1 else 's'}, "
                        f"{_progress_eta(_run_pos + 1, _n_total, _t0)})   ")
                    sys.stdout.flush()
                metadata = load.read_metadata(run_dir / "metadata.txt")
                kept_steps = good_steps[run_dir]
                # BEFORE the read/encode below, not after: the window-level
                # max_dt filter further down runs on already-encoded frames,
                # so without this every run pays to encode a tail no surviving
                # window can reach.
                kept_steps = _truncate_to_max_dt(kept_steps, metadata, self.max_dt,
                                                  window_length)

                if len(kept_steps) < window_length:
                    n_windowless_runs += 1
                    continue  # not enough consecutive kept steps for even one window

                if self.stat_names is not None:
                    stats_df = load.read_statistics_csv(run_dir / "statistics.csv")
                    missing = set(self.stat_names) - set(stats_df.columns)
                    if missing:
                        raise ValueError(f"{run_dir}/statistics.csv is missing columns {missing}")
                    self._stats_by_run[run_dir] = stats_df

                # Read every kept step once (this is the only place raw
                # snapshots are touched -- everything after this is pure
                # bookkeeping over whichever of latents/frames we end up
                # with). read_pool.map preserves kept_steps' own order
                # (like the plain list comprehension it replaces) even
                # though the underlying reads complete out of order --
                # window construction below depends on that ordering.
                run_index = len(pending_meta)
                if self._latent_cache_root is not None:
                    # Once per (size, fingerprint) per BUILD -- not per run: the
                    # content is identical across a build's ~3000 runs, and the
                    # per-run version rewrote the same small file ~3000 times.
                    # "cache last accessed" therefore refreshes per build, which
                    # is the meaningful unit of access.
                    _ck = (metadata.nx, self._encoder_fingerprint)
                    if _ck not in self._cache_info_written:
                        self._cache_info_written.add(_ck)
                        latent_cache.write_cache_info(
                            self._latent_cache_root, self._encoder_fingerprint,
                            size=metadata.nx,
                            info={"streams cached": ("z0 (state) + z1 (deriv)"
                                                     if self.encode_both_streams
                                                     else "z0 (state)"),
                                  **self._cache_info})
                    cached = latent_cache.load_cached(
                        latent_cache.cache_path_for_run(
                            self._latent_cache_root, self._encoder_fingerprint, run_dir,
                            kept_steps, self.encode_both_streams,
                            size=metadata.nx))
                    if cached is not None:
                        # Skips the READ as well as the encode -- at 128x128 a
                        # frame is 32 KB against a 2 KB latent, so the disk
                        # traffic saved exceeds the compute saved.
                        pending_meta.append((run_dir, metadata, kept_steps))
                        run_data_by_index[run_index] = cached[0]
                        run_deriv_by_index[run_index] = cached[1]
                        n_cache_hits += 1
                        continue
                snapshot_paths = [run_dir / load.snapshot_filename(step) for step in kept_steps]
                frames = torch.stack([
                    torch.from_numpy(arr).unsqueeze(0)
                    for arr in read_pool.map(
                        lambda p: load.read_phi_half(p, metadata.nx, metadata.ny), snapshot_paths
                    )
                ])  # (n_kept, 1, ny, nx)
                pending_meta.append((run_dir, metadata, kept_steps))
                buffer_run_indices.append(run_index)

                if self.encoder_given:
                    buffer_frames.append(frames)
                    buffer_sizes.append(frames.size(0))
                    buffer_total += frames.size(0)
                    if encoder_accepts_theta:
                        # SAME convention as _run_theta below (theta =
                        # temperature centered at T0, see LatentDynamics'
                        # own docstring) -- computed here too, not only
                        # in the bookkeeping loop, because encoding
                        # happens BEFORE bookkeeping and needs it NOW.
                        # Broadcast to one row per frame in THIS run
                        # (theta is constant across a run, but the
                        # buffer mixes frames from several runs at once,
                        # so each frame needs its own copy to stay
                        # aligned with buffer_frames/buffer_sizes).
                        run_theta = torch.tensor(theta_coordinates(metadata.temperature, metadata.T0), dtype=torch.float32)
                        buffer_thetas.append(run_theta.unsqueeze(0).expand(frames.size(0), -1))
                    if buffer_total >= encode_batch_size:
                        flushed, flushed_deriv = _flush_buffer()
                        for pos, latents, deriv in zip(buffer_run_indices, flushed,
                                                        flushed_deriv, strict=True):
                            run_data_by_index[pos] = latents
                            run_deriv_by_index[pos] = deriv
                            if self._latent_cache_root is not None:
                                latent_cache.store_cached(
                                    latent_cache.cache_path_for_run(
                                        self._latent_cache_root, self._encoder_fingerprint,
                                        pending_meta[pos][0], pending_meta[pos][2],
                                        self.encode_both_streams,
                                        size=pending_meta[pos][1].nx),
                                    latents, deriv)
                        buffer_run_indices = []
                else:
                    # encoding deferred to the training loop
                    run_data_by_index[run_index] = frames  # (n_kept, 1, ny, nx)
                    run_deriv_by_index[run_index] = None

            if _show_progress:
                sys.stdout.write("\n")   # close the in-place counter line
                sys.stdout.flush()

            if self.encoder_given:
                flushed, flushed_deriv = _flush_buffer()  # final, possibly-partial buffer
                for pos, latents, deriv in zip(buffer_run_indices, flushed,
                                                flushed_deriv, strict=True):
                    run_data_by_index[pos] = latents
                    run_deriv_by_index[pos] = deriv
                    if self._latent_cache_root is not None:
                        latent_cache.store_cached(
                            latent_cache.cache_path_for_run(
                                self._latent_cache_root, self._encoder_fingerprint,
                                pending_meta[pos][0], pending_meta[pos][2],
                                self.encode_both_streams,
                                size=pending_meta[pos][1].nx),
                            latents, deriv)
                buffer_run_indices = []

        if self.encoder_given and torch.device(device).type == "cuda":
            # Returns the encoder's peak VRAM usage back to the CUDA
            # driver/allocator pool now that encoding is done, rather
            # than leaving it cached-but-idle in this process for
            # whatever comes next (e.g. training itself, or -- since
            # this constructor commonly runs 3x in a row for train/val/
            # test -- the NEXT MicrostructureEvolutionDataset built
            # right after this one). Not correctness-critical (PyTorch's
            # own caching allocator would reuse these blocks for later
            # allocations in this same process regardless), but matters
            # if anything else (another process, or just accurate
            # nvidia-smi output) needs to see that VRAM as free.
            torch.cuda.empty_cache()

        if n_windowless_runs:
            # The max_dt case is called out separately because the generic
            # advice ("shorter window_length or looser filtering") is actively
            # misleading for it: those runs are not badly filtered, they simply
            # have no transition short enough to be usable, and the count jumps
            # sharply once max_dt truncation is on -- 74/2466 to 1297/2594 on
            # the same sweep. Reading that as a filtering problem would send
            # someone to loosen min_stdev_phi for no reason.
            _finite_cap = self.max_dt is not None and math.isfinite(self.max_dt)
            cause = (f"kept steps (after max_dt={self.max_dt} truncated each run to the "
                     f"prefix its own transitions can reach)" if _finite_cap
                     else "kept steps")
            advice = ("expected when max_dt is well below the sweep's later dt values"
                      if _finite_cap
                      else "consider a shorter window_length or looser filtering if this "
                           "is a large fraction")
            _runs_word = f"{self._split_label} runs" if self._split_label else "runs"
            _pct = 100.0 * n_windowless_runs / len(run_dirs) if run_dirs else 0.0
            print(f"MicrostructureEvolutionDataset: {n_windowless_runs}/{len(run_dirs)} ({_pct:.1f}%) {_runs_word} "
                  f"had fewer than window_length={window_length} {cause} and were skipped "
                  f"entirely ({advice})")

        # Assembled in pending_meta order, the alignment every consumer
        # assumes. Asserted rather than assumed: a missing index means a run
        # reached pending_meta without its data ever being filled -- from the
        # cache, the buffer or the deferred path -- and a silently shorter
        # list would misalign every run after it.
        missing = set(range(len(pending_meta))) - set(run_data_by_index)
        assert not missing, f"run data missing for pending_meta indices {sorted(missing)}"
        run_data_list = [run_data_by_index[i] for i in range(len(pending_meta))]
        run_data_deriv_list = [run_deriv_by_index[i] for i in range(len(pending_meta))]
        if n_cache_hits:
            print(f"MicrostructureEvolutionDataset: {n_cache_hits}/{len(pending_meta)} run(s) "
                  f"read from the latent cache")
        return pending_meta, run_data_list, run_data_deriv_list


    def _build_window_index(self, pending_meta: list, run_data_list: list, run_data_deriv_list: list) -> None:
        """
        Populates self._run_dirs/_run_steps/_run_data/_run_data_deriv/
        _run_dt_scale/_run_theta (per-run bookkeeping) and self._index
        (the actual (run_idx, window_start_position) pairs __getitem__
        indexes into), applying stat_names' own NaN-guard and
        min_std_deriv's own near-degenerate-derivative filter per
        candidate window. Extracted verbatim from __init__'s own
        former body -- same logic, same order, just named and callable
        on its own.
        """
        n_degenerate_deriv_windows = 0
        n_max_dt_windows = 0
        n_nonconsecutive_windows = 0
        n_candidate_windows = 0

        for (run_dir, metadata, kept_steps), run_data, run_data_deriv in zip(
            pending_meta, run_data_list, run_data_deriv_list
        ):
            run_idx = len(self._run_steps)
            self._run_dirs.append(run_dir)
            self._run_steps.append(kept_steps)
            self._run_data.append(run_data)
            # u-scheme (time_coordinate="log10_t"): the cached deriv stream is
            # z1 = dz0/dt (the encoder's deriv head produces a PHYSICAL-time
            # derivative). Convert ONCE here to z̃1 = dz0/du = ln10 * t * z1,
            # t = step * sim_dt. ln10 and sim_dt are both run-invariant, so fold
            # them: scale = (ln10 * sim_dt) * step -- one multiply per frame. Done
            # at construction, not per __getitem__ draw, and in place (the t-form
            # is not needed in a u-run; the DISK cache stays dz0/dt, shared across
            # t/u runs -- this is the in-memory view only). Per-run kept_steps
            # suffice (a prefix of the global schedule). This is
            # convert_derivative_coordinate specialised to t -> log10_t with the
            # constants folded; that helper stays the canonical definition.
            if self.time_coordinate == "log10_t" and run_data_deriv is not None:
                _k = math.log(10.0) * metadata.dt
                _scale = torch.tensor([_k * s for s in kept_steps],
                                      dtype=run_data_deriv.dtype)
                run_data_deriv = run_data_deriv * _scale[:, None, None, None]
            self._run_data_deriv.append(run_data_deriv)
            self._run_dt_scale.append(metadata.dt)
            self._run_du.append(
                [math.log10(kept_steps[i + 1] / kept_steps[i])
                 for i in range(len(kept_steps) - 1)]
                if self.time_coordinate == "log10_t" else None)
            self._run_nx.append(metadata.nx)
            self._run_ny.append(metadata.ny)
            # T0 (metadata: "threshold temperature in Landau potential") is
            # the physically meaningful reference point, not 0 -- the whole
            # sweep is subcritical (T < T0), and how close a run sits to
            # T0 is what actually governs the dynamics' character, not the
            # raw temperature value. Centering here puts that boundary at
            # theta=0 instead of asking the network to discover T0's
            # significance implicitly from data.
            self._run_theta.append(
                torch.tensor(theta_coordinates(metadata.temperature, metadata.T0),
                              dtype=torch.float32)
            )

            # Consecutive-in-the-SAVE-SCHEDULE guard. A window is n+1 CONSECUTIVE
            # kept steps, but "consecutive kept" is not the same as "consecutive
            # SAVED": when a step-level filter (min_normalized_stdev_phi etc.)
            # drops a quiet frame, the two kept frames on either side become
            # adjacent in kept_steps while a real saved frame sits between them
            # in the simulation. A window spanning that seam is NOT n+1
            # consecutive frames of the trajectory -- it silently jumps over the
            # gap, manufacturing a large-dt transition (the du_total>=~2 tail and
            # the du_max=2.5e4 grad-spike windows are exactly these).
            # require_consecutive rejects any window whose kept steps are not a
            # contiguous run of the original save_steps. _adj[j] is True iff
            # kept_steps[j+1] is the IMMEDIATE successor of kept_steps[j] in the
            # save schedule (nothing filtered between them).
            _adj = None
            if self.require_consecutive:
                _save_pos = {s: i for i, s in enumerate(metadata.save_steps)}
                _kpos = [_save_pos[s] for s in kept_steps]
                _adj = [(_kpos[j + 1] == _kpos[j] + 1)
                        for j in range(len(kept_steps) - 1)]

            for start in range(len(kept_steps) - self.window_length + 1):
                n_candidate_windows += 1
                if self.require_consecutive and not all(
                        _adj[start + i] for i in range(self.window_length - 1)):
                    n_nonconsecutive_windows += 1
                    continue
                if self.stat_names is not None:
                    start_step = kept_steps[start + self.stats_frame_index]
                    stats_df = self._stats_by_run[run_dir]
                    if start_step not in stats_df.index or stats_df.loc[start_step, self.stat_names].isna().any():
                        continue  # same NaN-guard rationale as MicrostructureTripletDataset
                if self.min_std_deriv is not None:
                    first_dt = (kept_steps[start + 1] - kept_steps[start]) * metadata.dt
                    first_deriv = (run_data[start + 1] - run_data[start]) / first_dt
                    if first_deriv.std().item() < self.min_std_deriv:
                        n_degenerate_deriv_windows += 1
                        continue
                if self.max_dt is not None:
                    # Drops the window if ANY of its own transitions
                    # exceeds max_dt -- not just the first, and not the
                    # total span. A rollout window is only as usable as
                    # its worst single step: one transition past the
                    # horizon puts the whole chained prediction
                    # off-distribution from that point on, so keeping
                    # the window because its OTHER steps are fine would
                    # defeat the filter entirely.
                    if any((kept_steps[start + i + 1] - kept_steps[start + i]) * metadata.dt
                           > self.max_dt for i in range(self.window_length - 1)):
                        n_max_dt_windows += 1
                        continue
                self._index.append((run_idx, start))

        _split_suffix = f" in {self._split_label} runs" if self._split_label else ""
        if n_nonconsecutive_windows:
            _pct = 100.0 * n_nonconsecutive_windows / n_candidate_windows if n_candidate_windows else 0.0
            print(f"MicrostructureEvolutionDataset: {n_nonconsecutive_windows}/{n_candidate_windows} "
                  f"({_pct:.1f}%) candidate window(s){_split_suffix} skipped for NOT being "
                  f"{self.window_length} CONSECUTIVE saved steps -- a step-level filter dropped a "
                  f"frame inside the window, so its kept steps jump over a real saved frame and it "
                  f"is not {self.window_length} adjacent frames of the trajectory. Such a window "
                  f"silently carries a large-dt transition across the gap; require_consecutive "
                  f"excludes it at the definition of a window rather than clipping its dt tail.")

        if n_max_dt_windows:
            _pct = 100.0 * n_max_dt_windows / n_candidate_windows if n_candidate_windows else 0.0
            print(f"MicrostructureEvolutionDataset: {n_max_dt_windows}/{n_candidate_windows} "
                  f"({_pct:.1f}%) candidate window(s){_split_suffix} skipped "
                  f"for having a transition longer than max_dt={self.max_dt}. Beyond that dt the "
                  f"first-order (z0 + z1*dt) term's own error is large enough that f_theta can only "
                  f"add a correction on top of an already-wrong prediction -- excluding those "
                  f"windows trains f_theta only where it can actually help.")

        if n_degenerate_deriv_windows:
            _pct = 100.0 * n_degenerate_deriv_windows / n_candidate_windows if n_candidate_windows else 0.0
            print(f"MicrostructureEvolutionDataset: {n_degenerate_deriv_windows}/{n_candidate_windows} "
                  f"({_pct:.1f}%) candidate window(s){_split_suffix} "
                  f"skipped for having a near-degenerate first-transition derivative "
                  f"(std < min_std_deriv={self.min_std_deriv}) -- spatially complex but "
                  f"essentially stationary between those two specific steps (e.g. a straight, "
                  f"zero-curvature interface), not excluded by min_stdev_phi alone.")



    def __len__(self) -> int:
        n = len(self._index)
        if self.augment:
            n *= _N_AUGMENT_VARIANTS
        elif self.fixed_aug_indices is not None:
            n *= len(self.fixed_aug_indices)
        return n

    def all_dts(self) -> np.ndarray:
        """
        Every per-transition dt value across every BASE window in this
        dataset, flattened into a single 1D array -- (n_base_windows *
        (window_length-1),). Computed directly from the already-loaded,
        lightweight run metadata (self._index/_run_steps/_run_dt_scale)
        -- the SAME computation __getitem__ does for a single window's
        own dt_window, just without ever touching self._run_data (the
        actual frame tensors), which this has no need to load at all.

        Deliberately excludes augmentation's own multiplicity: augment
        repeats every base window by the SAME factor
        (_N_AUGMENT_VARIANTS), regardless of that window's own dt -- so
        it changes the ABSOLUTE window count uniformly, not the
        RELATIVE proportion of windows falling in each dt decade, which
        is the only thing a caller of this method (e.g. computing
        global per-decade loss weights) actually needs.
        """
        all_dts = []
        for run_idx, start in self._index:
            end = start + self.window_length
            steps = self._run_steps[run_idx][start:end]
            dt_scale = self._run_dt_scale[run_idx]
            all_dts.extend((steps[i + 1] - steps[i]) * dt_scale for i in range(len(steps) - 1))
        return np.array(all_dts, dtype=np.float32)

    def max_dt_per_window(self) -> np.ndarray:
        """The LARGEST transition in each base window -- shape (n_base_windows,).

        Exists for dt-homogeneous batching. Under adaptive sub-stepping the
        cost of a batch is its MAXIMUM sub-step count, because the integrator
        loops until the last sample in the batch has arrived and the arrived
        ones are masked (they still occupy the forward pass). A batch drawn
        uniformly from a population whose counts span 8x to 132x therefore
        pays close to the population maximum on every batch: measured on the
        128x128 stage-3b population, mean 17.7 against max 132, i.e. roughly
        7x the work the same windows would cost if grouped.

        Grouping by the window's own worst transition, rather than by its
        first or its total span, matches what the cost actually depends on --
        the same reasoning the max_dt window filter uses, and for the same
        reason: a window is as expensive as its most demanding step.

        Computed from run metadata only (never touching _run_data), on the
        same pattern as all_dts, so a sampler can call it before training
        starts without forcing any frame loads.
        """
        out = []
        for run_idx, start in self._index:
            end = start + self.window_length
            steps = self._run_steps[run_idx][start:end]
            dt_scale = self._run_dt_scale[run_idx]
            out.append(max((steps[i + 1] - steps[i]) * dt_scale
                            for i in range(len(steps) - 1)))
        return np.array(out, dtype=np.float32)


    def _du_slice(self, run_idx: int, lo: int, hi: int) -> torch.Tensor:
        """Coordinate step per transition for frames [lo, hi): Delta-u =
        log10(t_{i+1}/t_i) in log10_t mode, else physical Delta-t. This is the
        SAME quantity dt_window holds, so dividing a z0 backward difference by it
        yields dz0/du (or dz0/dt) -- the derivative in the model's own
        coordinate, matching z̃1. Length hi-lo-1."""
        if self.time_coordinate == "log10_t":
            return torch.tensor(self._run_du[run_idx][lo:hi - 1], dtype=torch.float32)
        steps = self._run_steps[run_idx]
        dt_scale = self._run_dt_scale[run_idx]
        return torch.tensor(
            [(steps[i + 1] - steps[i]) * dt_scale for i in range(lo, hi - 1)],
            dtype=torch.float32)

    def _backward_quotient(self, run_idx: int, start: int, end: int,
                           dt_window: torch.Tensor,
                           z1_fallback: torch.Tensor) -> torch.Tensor:
        """The 'previous quotient' derivative channel: q_k = (z0_k - z0_{k-1}) /
        du_{k-1}, one per window frame, in the model's coordinate. Step 0 uses
        the REAL predecessor frame (z0 at start-1) -- the value that makes the
        autonomous rollout's step-0 seed a true backward quotient instead of z1.
        At a run start (start==0) there is no predecessor, so step 0 falls back
        to the encoder's z1 (the only derivative available there); the note in
        the sweep log ('except t=0, not a great loss') is exactly this case.

        Shapes: window has L=window_length frames; returns (L, C, 8, 8), aligned
        with window_deriv so window_deriv[k] is the derivative AT frame k."""
        z0 = self._run_data[run_idx]
        if start > 0:
            z0_ext = z0[start - 1:end]                       # (L+1, C, 8, 8)
            du = self._du_slice(run_idx, start - 1, end)     # (L,)
            return (z0_ext[1:] - z0_ext[:-1]) / du.view(-1, 1, 1, 1)
        # start == 0: the window begins at the run's first kept frame. Its
        # predecessor was filtered out, but we encoded it (pre-filter) into
        # _run_pred_z0 -- use it so q_0 is a REAL backward quotient. Only when
        # there is genuinely no predecessor (the run's absolute first saved
        # frame, normally dropped by min_step) do we fall back to z1.
        q_rest = (z0[start + 1:end] - z0[start:end - 1]) / dt_window.view(-1, 1, 1, 1)
        pred = self._run_pred_z0[run_idx] if self._run_pred_z0 else None
        if pred is not None:
            q0 = ((z0[start] - pred) / self._run_pred_du[run_idx]).unsqueeze(0)
            return torch.cat([q0, q_rest], dim=0)
        return torch.cat([z1_fallback[:1], q_rest], dim=0)

    def __getitem__(self, idx: int):
        if self.augment:
            base_idx, aug_idx = divmod(idx, _N_AUGMENT_VARIANTS)
        elif self.fixed_aug_indices is not None:
            # Consecutive indices walk the variant list for one window
            # before advancing to the next window -- so all of a given
            # window's own variants land in the same neighbourhood of
            # the index space. Ordering is irrelevant to the mean this
            # feeds (see fixed_aug_indices' own comment in __init__),
            # and val loaders don't shuffle anyway.
            base_idx, slot = divmod(idx, len(self.fixed_aug_indices))
            aug_idx = self.fixed_aug_indices[slot]
        else:
            base_idx, aug_idx = idx, None

        run_idx, start = self._index[base_idx]
        end = start + self.window_length

        window = self._run_data[run_idx][start:end]  # (window_length, C, 8, 8) or (window_length, 1, ny, nx)

        aug_k = aug_flip = None
        if aug_idx is not None:
            # The SAME (k, flip, shift) applied to EVERY frame in the
            # window -- not independently per frame, which would make
            # the window describe a physically meaningless "evolution"
            # (see _apply_augmentation's own docstring). window is
            # (window_length, C, H, W); apply per-frame and re-stack.
            # k/flip captured here (not discarded) -- needed below to
            # correct the "angle" stat, if stat_names includes it, to
            # match the augmented frame's actual orientation.
            nx, ny = self._run_nx[run_idx], self._run_ny[run_idx]
            transformed = [_apply_augmentation(frame, aug_idx, nx, ny) for frame in window]
            window = torch.stack([t[0] for t in transformed])
            aug_k, aug_flip = transformed[0][1], transformed[0][2]  # identical across frames by construction

        steps = self._run_steps[run_idx][start:end]
        dt_scale = self._run_dt_scale[run_idx]
        if self.time_coordinate == "log10_t":
            # step size is Delta-u = log10(t_{i+1}/t_i), precomputed per run;
            # window_deriv is already z̃1 (converted in place at construction).
            dt_window = torch.tensor(
                self._run_du[run_idx][start:start + len(steps) - 1],
                dtype=torch.float32,
            )
        else:
            dt_window = torch.tensor(
                [(steps[i + 1] - steps[i]) * dt_scale for i in range(len(steps) - 1)],
                dtype=torch.float32,
            )  # (window_length - 1,)

        theta = self._run_theta[run_idx]  # (n_theta,) -- constant across the window (same run)

        if self.encode_both_streams:
            window_deriv = self._run_data_deriv[run_idx][start:end]
            if (self.derivative_source == "previous_quotient"
                    and not getattr(self, "_quotient_precomputed", False)):
                # Fallback path only (precompute normally replaces the deriv
                # stream with the quotient at build time, so the slice above
                # already IS the quotient).
                window_deriv = self._backward_quotient(
                    run_idx, start, end, dt_window, window_deriv)
            if self.return_phys_dt:
                # physical Delta-t = (step_{i+1}-step_i)*sim_dt, ALWAYS (== dt_window
                # in t-mode). The loss weights by this; the model steps in dt_window.
                dt_phys_window = torch.tensor(
                    [(steps[i + 1] - steps[i]) * dt_scale for i in range(len(steps) - 1)],
                    dtype=torch.float32,
                )
                return window, window_deriv, dt_window, dt_phys_window, theta
            return window, window_deriv, dt_window, theta

        if self.stat_names is None:
            return window, dt_window, theta

        run_dir = self._run_dirs[run_idx]
        # steps[stats_frame_index], not steps[0] -- see stats_frame_index's
        # own comment in __init__. Default 0 keeps the long-standing
        # "window[0] is the real starting frame" behavior exactly.
        stats_step = steps[self.stats_frame_index]
        true_stats = torch.tensor(
            self._stats_by_run[run_dir].loc[stats_step, self.stat_names].to_numpy(dtype=float),
            dtype=torch.float32,
        )
        if aug_idx is not None and "angle" in self.stat_names:
            angle_idx = self.stat_names.index("angle")
            true_stats[angle_idx] = _transform_angle(true_stats[angle_idx], aug_k, aug_flip)
        if self.return_frame_t:
            dt_scale = self._run_dt_scale[run_idx]
            t_window = torch.tensor([st * dt_scale for st in steps], dtype=torch.float32)
            return window, dt_window, theta, true_stats, t_window
        return window, dt_window, theta, true_stats

    def window_info(self, idx: int) -> tuple[Path, list[int]]:
        """
        (run_dir, [step, ...]) for a given __getitem__ index. Even in
        raw-pixel mode (encoder=None), this stays useful for tracing a
        sample back to its exact source files/steps (e.g. check_rollout.py
        decoding a prediction and comparing against the real x(t+dt)),
        independent of which mode built this dataset.

        Correctly unwraps augmentation the same way __getitem__ does
        (divmod by _N_AUGMENT_VARIANTS when augment=True, or by
        len(fixed_aug_indices) in that mode), so this returns the true
        source window regardless of which variant idx happened to land
        on -- matching MicrostructureSnapshotDataset.frame_info's
        identical pattern.
        """
        if self.augment:
            base_idx = idx // _N_AUGMENT_VARIANTS
        elif self.fixed_aug_indices is not None:
            base_idx = idx // len(self.fixed_aug_indices)
        else:
            base_idx = idx
        run_idx, start = self._index[base_idx]
        end = start + self.window_length
        return self._run_dirs[run_idx], self._run_steps[run_idx][start:end]


class MicrostructureTripletDataset(Dataset):
    """
    Real-pixel (t1, t2, t3) triplets for stage 2 (interpolation-consistency
    fine-tuning). UNLIKE MicrostructureEvolutionDataset, this does NOT
    precompute/cache latents under a frozen encoder: stage 2 continues
    training the encoder (and stats_head), so gradients must flow through
    z1/z2/z3 every forward pass. Returns raw pixels; encoding happens
    fresh in the training loop.

    __getitem__ returns (x1, x2, x3, alpha, true_stats_t2):
      x1, x2, x3: (1, H, W) raw snapshots at t1 < t2 < t3
      alpha: scalar, (t2-t1)/(t3-t1) -- dt-weighted, not the midpoint,
             since save_steps are irregular
      true_stats_t2: (n_stats,) RAW (not normalized) statistics.csv row
             at t2 -- StatsLoss handles normalization, matching how
             stage 2 already uses it
    """

    def __init__(self, run_dirs: list[str | Path], stat_names: list[str],
                 skip_bad: bool = True, min_step: int = 0,
                 min_stdev_phi: float | None = None, min_passing_steps: int | None = None,
                 good_steps: dict[Path, list[int]] | None = None):
        """
        stat_names: REQUIRED and must match the stage-2 checkpoint's
        stats_config exactly -- unlike MicrostructureSnapshotDataset,
        auto-detection isn't offered here, since silently resolving a
        different schema than the already-trained stats_head expects
        would be a much worse failure mode (wrong-shape predictions or
        silently-misaligned columns) than requiring it explicitly.
        """
        run_dirs = [Path(d) for d in run_dirs]
        if good_steps is None:
            good_steps = build_good_steps(run_dirs, skip_bad, min_step, min_stdev_phi, min_passing_steps)

        self.stat_names = stat_names
        self._index: list[tuple[Path, int, int, int]] = []  # (run_dir, t1, t2, t3)
        self._nx_ny: dict[Path, tuple[int, int]] = {}
        self._stats_by_run = {}

        for run_dir in run_dirs:
            metadata = load.read_metadata(run_dir / "metadata.txt")
            kept = good_steps[run_dir]
            if len(kept) < 3:
                continue
            stats_df = load.read_statistics_csv(run_dir / "statistics.csv")
            missing = set(stat_names) - set(stats_df.columns)
            if missing:
                raise ValueError(f"{run_dir}/statistics.csv is missing columns {missing}")
            self._stats_by_run[run_dir] = stats_df
            self._nx_ny[run_dir] = (metadata.nx, metadata.ny)

            for i in range(len(kept) - 2):
                t1, t2, t3 = kept[i], kept[i + 1], kept[i + 2]
                if t2 not in stats_df.index or stats_df.loc[t2, stat_names].isna().any():
                    continue  # same NaN-guard rationale as MicrostructureSnapshotDataset
                self._index.append((run_dir, t1, t2, t3))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        run_dir, t1, t2, t3 = self._index[idx]
        nx, ny = self._nx_ny[run_dir]

        x1 = torch.from_numpy(load.read_phi_half(run_dir / load.snapshot_filename(t1), nx, ny)).unsqueeze(0)
        x2 = torch.from_numpy(load.read_phi_half(run_dir / load.snapshot_filename(t2), nx, ny)).unsqueeze(0)
        x3 = torch.from_numpy(load.read_phi_half(run_dir / load.snapshot_filename(t3), nx, ny)).unsqueeze(0)

        alpha = torch.tensor((t2 - t1) / (t3 - t1), dtype=torch.float32)
        true_stats = torch.tensor(
            self._stats_by_run[run_dir].loc[t2, self.stat_names].to_numpy(dtype=float),
            dtype=torch.float32,
        )
        return x1, x2, x3, alpha, true_stats
