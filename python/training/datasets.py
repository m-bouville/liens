"""
PyTorch Dataset classes for loading phase-field runs off disk.
"""

import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from models.constants import LATENT_SPATIAL_SIZE as _LATENT_SPATIAL_SIZE
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


def complete_run_dirs(base: str | Path, nx: int, ny: int) -> list[Path]:
    """
    All directories for one grid size that exist on disk and are marked
    complete -- reads base/<nx>x<ny>/metadata.txt directly (see
    load.enumerate_run_dirs_from_metadata), NOT config.txt: metadata.txt
    is co-located with the actual dataset, so it's always correct for
    THIS directory, with no risk of describing an unrelated sweep (which
    a shared, possibly-currently-mutated config.txt could).
    """
    return [d for d in load.enumerate_run_dirs_from_metadata(base, nx, ny) if load.is_complete(d)]


def split_run_dirs(run_dirs: list[Path], val_fraction: float, test_fraction: float = 0.0,
                    seed: int | None = None) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Split whole run directories into train/val/test groups -- NOT
    individual frames, and not even "distinct base samples" pooled
    across runs (an earlier version of this code split at that level).

    Two frames from the same run, even at unrelated timesteps, are
    snapshots of one continuous physical evolution and can look very
    similar (adjacent saved steps especially, but not only those) --
    splitting below the directory level risks putting correlated frames
    from the same run on both sides, silently inflating val/test
    performance. Splitting whole directories is the only granularity
    where train and val/test are genuinely independent runs.

    Returns (train_dirs, val_dirs, test_dirs). test_dirs is empty if
    test_fraction=0.
    """
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction ({val_fraction}) + test_fraction ({test_fraction}) must be < 1.0"
        )

    n = len(run_dirs)
    n_val = max(1, int(n * val_fraction))
    n_test = max(1, int(n * test_fraction)) if test_fraction > 0 else 0

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()

    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    return (
        [run_dirs[i] for i in train_idx],
        [run_dirs[i] for i in val_idx],
        [run_dirs[i] for i in test_idx],
    )


def _filtered_steps(metadata: "load.RunMetadata", bad_steps: set[int],
                     min_step: int, min_stdev_phi: float | None, stats_df) -> list[int]:
    """
    Steps from metadata.save_steps passing the filters shared by every
    dataset in this module: not missing/corrupt (bad_steps), not earlier
    than min_step, and (if min_stdev_phi is set) not NaN/below threshold
    in statistics.csv's stdev_phi column. Factored out so
    MicrostructureSnapshotDataset and MicrostructureEvolutionDataset
    can't silently diverge in what counts as an excluded step.
    """
    kept = []
    for step in metadata.save_steps:
        if step in bad_steps or step < min_step:
            continue
        if min_stdev_phi is not None:
            if stats_df is None or step not in stats_df.index:
                continue  # can't verify the criterion -- exclude rather than assume
            stdev = stats_df.loc[step, "stdev_phi"]
            if math.isnan(stdev) or stdev < min_stdev_phi:
                continue
        kept.append(step)
    return kept


def build_good_steps(run_dirs: list[str | Path], skip_bad: bool = True,
                      min_step: int = 0, min_stdev_phi: float | None = None,
                      ) -> dict[Path, list[int]]:
    """
    Scan run_dirs ONCE and return a stable {run_dir: [kept_step, ...]}
    mapping -- the single source of truth for "which snapshots count as
    usable" under a given filter configuration.

    Both MicrostructureSnapshotDataset and MicrostructureEvolutionDataset
    call this exact function rather than each re-deriving their own
    filtered step list. That's a stronger guarantee than just sharing
    _filtered_steps's logic: two separate call sites invoking the same
    function can still drift in the bookkeeping around it (e.g. a future
    edit to one call site's loop), whereas both classes consuming the
    same precomputed dict cannot silently disagree.

    Pass the result of one call here into BOTH a Snapshot and an
    Evolution dataset built from the same run_dirs/filter settings (via
    their optional good_steps= argument) to skip the scan entirely on
    the second construction, rather than paying for it twice.
    """
    good_steps: dict[Path, list[int]] = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metadata = load.read_metadata(run_dir / "metadata.txt")

        check = load.check_snapshots_saved(run_dir, metadata)
        bad_steps = set(check["missing"]) | set(check["bad_size"])
        if bad_steps and not skip_bad:
            raise ValueError(
                f"{run_dir}: {len(bad_steps)} bad/missing snapshot(s) "
                f"({sorted(bad_steps)[:5]}...), and skip_bad=False"
            )

        stats_df = load.read_statistics_csv(run_dir / "statistics.csv") \
            if min_stdev_phi is not None else None
        good_steps[run_dir] = _filtered_steps(metadata, bad_steps, min_step, min_stdev_phi,
                                               stats_df)

    return good_steps


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
                 min_stdev_phi: float | None = None,
                 include_stats: bool = False, stat_names: list[str] | None = None,
                 good_steps: dict[Path, list[int]] | None = None):
        """
        good_steps: a precomputed {run_dir: [kept_step, ...]} mapping
        from build_good_steps(), to skip re-scanning run_dirs when
        another dataset (e.g. a paired MicrostructureEvolutionDataset
        over the same run_dirs) already computed it with the same
        skip_bad/min_step/min_stdev_phi. Default None computes it here.

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
            good_steps = build_good_steps(run_dirs, skip_bad, min_step, min_stdev_phi)

        for run_dir in run_dirs:
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
    stat_names below -- only meaningful/available in raw-pixel mode):
      window:    (n_r+1, latent_channels, 8, 8) if encoder was given
                 (a window of z's), or (n_r+1, 1, ny, nx) if encoder is
                 None (a window of raw frames x, encoding deferred to
                 the training loop).
      dt_window: (n_r,) float32, physical time between consecutive steps
                 in the window, i.e. (step_b - step_a) * metadata.dt.
      theta:     (1,) float32, currently [temperature - T0]. Per
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
                 min_stdev_phi: float | None = None, encode_batch_size: int = 256,
                 good_steps: dict[Path, list[int]] | None = None,
                 stat_names: list[str] | None = None, augment: bool = False,
                 min_std_deriv: float | None = None):
        """
        encoder: pass a frozen encoder for the cached-latent mode (stage
        3), or None for the raw-pixel mode (stage 4/5, E trainable) --
        see class docstring.

        window_length: n_r + 1 (e.g. window_length=5 -> n_r=4 predicted steps).
        encode_batch_size: batch size for the upfront encoding pass when
        encoder is given; unused (raw frames aren't encoded here) when
        encoder is None.
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
        if min_std_deriv is not None and encoder is not None:
            raise ValueError(
                "min_std_deriv was given together with a real encoder (cached-latent, stage-3 "
                "mode) -- this filters on the RAW PIXEL derivative's spatial std, which has no "
                "well-defined meaning against a cached latent (see this constructor's own "
                "docstring). Pass encoder=None if you want this filter."
            )

        self.window_length = window_length
        self.encoder_given = encoder is not None  # which mode __getitem__ operates in
        self.stat_names = stat_names
        self.augment = augment
        self.min_std_deriv = min_std_deriv
        self._run_dirs: list[Path] = []         # run_dir per run_idx, for tracing samples back
        self._run_steps: list[list[int]] = []   # kept step numbers per run, in order
        self._run_data: list[torch.Tensor] = []  # per run, on CPU: latents (n_kept,C,8,8) if
                                                   # encoder given, else raw frames (n_kept,1,ny,nx)
        self._run_dt_scale: list[float] = []    # metadata.dt per run
        self._run_theta: list[torch.Tensor] = []  # (n_theta,) physical params per run -- see class docstring
        self._run_nx: list[int] = []            # metadata.nx per run, only used if augment=True
        self._run_ny: list[int] = []            # metadata.ny per run, only used if augment=True
        self._stats_by_run = {}                 # run_dir -> statistics.csv DataFrame, only if stat_names given
        self._index: list[tuple[int, int]] = []  # (run_idx, window_start_position)

        run_dirs = [Path(d) for d in run_dirs]
        if good_steps is None:
            good_steps = build_good_steps(run_dirs, skip_bad, min_step, min_stdev_phi)

        if self.encoder_given:
            encoder = encoder.to(device).eval()
        n_windowless_runs = 0
        n_degenerate_deriv_windows = 0

        for run_dir in run_dirs:
            metadata = load.read_metadata(run_dir / "metadata.txt")
            kept_steps = good_steps[run_dir]

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
            # bookkeeping over whichever of latents/frames we end up with).
            frames = torch.stack([
                torch.from_numpy(load.read_phi_half(
                    run_dir / load.snapshot_filename(step), metadata.nx, metadata.ny
                )).unsqueeze(0)
                for step in kept_steps
            ])  # (n_kept, 1, ny, nx)

            if self.encoder_given:
                latents = []
                with torch.no_grad():
                    for i in range(0, len(frames), encode_batch_size):
                        batch = frames[i:i + encode_batch_size].to(device)
                        latents.append(encoder(batch)[DEFAULT_STREAM_NAME].cpu())
                run_data = torch.cat(latents, dim=0)  # (n_kept, latent_channels, 8, 8)
            else:
                run_data = frames  # (n_kept, 1, ny, nx) -- encoding deferred to the training loop

            run_idx = len(self._run_steps)
            self._run_dirs.append(run_dir)
            self._run_steps.append(kept_steps)
            self._run_data.append(run_data)
            self._run_dt_scale.append(metadata.dt)
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
                torch.tensor([metadata.temperature - metadata.T0], dtype=torch.float32)
            )

            for start in range(len(kept_steps) - window_length + 1):
                if self.stat_names is not None:
                    start_step = kept_steps[start]
                    stats_df = self._stats_by_run[run_dir]
                    if start_step not in stats_df.index or stats_df.loc[start_step, self.stat_names].isna().any():
                        continue  # same NaN-guard rationale as MicrostructureTripletDataset
                if self.min_std_deriv is not None:
                    first_dt = (kept_steps[start + 1] - kept_steps[start]) * metadata.dt
                    first_deriv = (run_data[start + 1] - run_data[start]) / first_dt
                    if first_deriv.std().item() < self.min_std_deriv:
                        n_degenerate_deriv_windows += 1
                        continue
                self._index.append((run_idx, start))

        if n_degenerate_deriv_windows:
            print(f"MicrostructureEvolutionDataset: {n_degenerate_deriv_windows} candidate window(s) "
                  f"skipped for having a near-degenerate first-transition derivative "
                  f"(std < min_std_deriv={self.min_std_deriv}) -- spatially complex but "
                  f"essentially stationary between those two specific steps (e.g. a straight, "
                  f"zero-curvature interface), not excluded by min_stdev_phi alone.")

        if n_windowless_runs:
            print(f"MicrostructureEvolutionDataset: {n_windowless_runs}/{len(run_dirs)} runs "
                  f"had fewer than window_length={window_length} kept steps and were skipped "
                  f"entirely (consider a shorter window_length or looser filtering if this "
                  f"is a large fraction)")

    def __len__(self) -> int:
        n = len(self._index)
        if self.augment:
            n *= _N_AUGMENT_VARIANTS
        return n

    def __getitem__(self, idx: int):
        if self.augment:
            base_idx, aug_idx = divmod(idx, _N_AUGMENT_VARIANTS)
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
        dt_window = torch.tensor(
            [(steps[i + 1] - steps[i]) * dt_scale for i in range(len(steps) - 1)],
            dtype=torch.float32,
        )  # (window_length - 1,)

        theta = self._run_theta[run_idx]  # (n_theta,) -- constant across the window (same run)

        if self.stat_names is None:
            return window, dt_window, theta

        run_dir = self._run_dirs[run_idx]
        start_step = steps[0]  # window[0] is the real starting frame -- true_stats is always FOR it
        true_stats = torch.tensor(
            self._stats_by_run[run_dir].loc[start_step, self.stat_names].to_numpy(dtype=float),
            dtype=torch.float32,
        )
        if aug_idx is not None and "angle" in self.stat_names:
            angle_idx = self.stat_names.index("angle")
            true_stats[angle_idx] = _transform_angle(true_stats[angle_idx], aug_k, aug_flip)
        return window, dt_window, theta, true_stats

    def window_info(self, idx: int) -> tuple[Path, list[int]]:
        """
        (run_dir, [step, ...]) for a given __getitem__ index. Even in
        raw-pixel mode (encoder=None), this stays useful for tracing a
        sample back to its exact source files/steps (e.g. check_rollout.py
        decoding a prediction and comparing against the real x(t+dt)),
        independent of which mode built this dataset.

        Correctly unwraps augmentation the same way __getitem__ does
        (divmod by _N_AUGMENT_VARIANTS) when augment=True, so this
        returns the true source window regardless of which augmented
        variant idx happened to land on -- matching
        MicrostructureSnapshotDataset.frame_info's identical pattern.
        """
        base_idx = idx // _N_AUGMENT_VARIANTS if self.augment else idx
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
                 min_stdev_phi: float | None = None,
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
            good_steps = build_good_steps(run_dirs, skip_bad, min_step, min_stdev_phi)

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
