"""
PyTorch Dataset classes for loading phase-field runs off disk.
"""

import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

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
    (stage 2) -- no time dynamics, so cross-run/cross-time order doesn't
    matter, everything is just one big pool of frames.

    Each item is read from disk on access, not cached in memory unless
    cache_in_memory=True: one snapshot is small, but hundreds of runs x
    tens of steps at 256x256 is not something to hold entirely in RAM
    up front by default.

    augment: if True, expands the pooled dataset by the 8 elements of D4
    (see _dihedral_transform) combined with 4 periodic translations --
    (0,0), (Nx/2,0), (0,Ny/2), (Nx/2,Ny/2) -- for a 32x expansion of the
    effective frame count. The cache (if enabled) still stores only the
    base frames; augmentation is applied after cache lookup, so memory
    use is unaffected by the expansion factor.

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

    def __len__(self) -> int:
        n = len(self._index)
        if self.augment:
            n *= self._N_DIHEDRAL * 4  # 8 D4 elements x 4 translations
        return n

    def _augment_item(self, base_idx: int, aug_idx: int) -> tuple[torch.Tensor, int, bool]:
        n_translations = 4
        dihedral_idx, translation_idx = divmod(aug_idx, n_translations)
        k, flip = divmod(dihedral_idx, 2)

        x = self._load(base_idx)
        x = _dihedral_transform(x, k=k, flip=bool(flip))

        _, _, nx, ny = self._index[base_idx]
        shifts = [(0, 0), (0, nx // 2), (ny // 2, 0), (ny // 2, nx // 2)]
        shift_y, shift_x = shifts[translation_idx]
        x = _translate(x, shift_y, shift_x)
        return x, k, bool(flip)

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

    __getitem__ returns (window, dt_window, theta):
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
                 good_steps: dict[Path, list[int]] | None = None):
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
        """
        if window_length < 2:
            raise ValueError(f"window_length must be >= 2 (got {window_length}), "
                              f"since a window needs at least one transition to predict")

        self.window_length = window_length
        self.encoder_given = encoder is not None  # which mode __getitem__ operates in
        self._run_dirs: list[Path] = []         # run_dir per run_idx, for tracing samples back
        self._run_steps: list[list[int]] = []   # kept step numbers per run, in order
        self._run_data: list[torch.Tensor] = []  # per run, on CPU: latents (n_kept,C,8,8) if
                                                   # encoder given, else raw frames (n_kept,1,ny,nx)
        self._run_dt_scale: list[float] = []    # metadata.dt per run
        self._run_theta: list[torch.Tensor] = []  # (n_theta,) physical params per run -- see class docstring
        self._index: list[tuple[int, int]] = []  # (run_idx, window_start_position)

        run_dirs = [Path(d) for d in run_dirs]
        if good_steps is None:
            good_steps = build_good_steps(run_dirs, skip_bad, min_step, min_stdev_phi)

        if self.encoder_given:
            encoder = encoder.to(device).eval()
        n_windowless_runs = 0

        for run_dir in run_dirs:
            metadata = load.read_metadata(run_dir / "metadata.txt")
            kept_steps = good_steps[run_dir]

            if len(kept_steps) < window_length:
                n_windowless_runs += 1
                continue  # not enough consecutive kept steps for even one window

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
                        latents.append(encoder(batch).cpu())
                run_data = torch.cat(latents, dim=0)  # (n_kept, latent_channels, 8, 8)
            else:
                run_data = frames  # (n_kept, 1, ny, nx) -- encoding deferred to the training loop

            run_idx = len(self._run_steps)
            self._run_dirs.append(run_dir)
            self._run_steps.append(kept_steps)
            self._run_data.append(run_data)
            self._run_dt_scale.append(metadata.dt)
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
                self._index.append((run_idx, start))

        if n_windowless_runs:
            print(f"MicrostructureEvolutionDataset: {n_windowless_runs}/{len(run_dirs)} runs "
                  f"had fewer than window_length={window_length} kept steps and were skipped "
                  f"entirely (consider a shorter window_length or looser filtering if this "
                  f"is a large fraction)")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        run_idx, start = self._index[idx]
        end = start + self.window_length

        window = self._run_data[run_idx][start:end]  # (window_length, C, 8, 8) or (window_length, 1, ny, nx)

        steps = self._run_steps[run_idx][start:end]
        dt_scale = self._run_dt_scale[run_idx]
        dt_window = torch.tensor(
            [(steps[i + 1] - steps[i]) * dt_scale for i in range(len(steps) - 1)],
            dtype=torch.float32,
        )  # (window_length - 1,)

        theta = self._run_theta[run_idx]  # (n_theta,) -- constant across the window (same run)

        return window, dt_window, theta

    def window_info(self, idx: int) -> tuple[Path, list[int]]:
        """
        (run_dir, [step, ...]) for a given __getitem__ index. Even in
        raw-pixel mode (encoder=None), this stays useful for tracing a
        sample back to its exact source files/steps (e.g. check_rollout.py
        decoding a prediction and comparing against the real x(t+dt)),
        independent of which mode built this dataset.
        """
        run_idx, start = self._index[idx]
        end = start + self.window_length
        return self._run_dirs[run_idx], self._run_steps[run_idx][start:end]


class MicrostructureTripletDataset(Dataset):
    """
    Real-pixel (t1, t2, t3) triplets for stage 3 (interpolation-consistency
    fine-tuning). UNLIKE MicrostructureEvolutionDataset, this does NOT
    precompute/cache latents under a frozen encoder: stage 3 continues
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