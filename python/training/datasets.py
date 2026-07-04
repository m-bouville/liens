"""
PyTorch Dataset classes for loading phase-field runs off disk.
"""

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


def complete_run_dirs(config: "load.SweepConfig", base: str | Path = "../datasets") -> list[Path]:
    """All directories implied by a sweep that exist on disk and are marked complete."""
    return [d for d in load.enumerate_run_dirs(config, base) if load.is_complete(d)]


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
                 include_stats: bool = False, stat_names: list[str] | None = None):
        """
        min_step: skip snapshots earlier than this step. Early steps
        (near t=0) are dominated by the initial noise (phi0 + noise in
        metadata.txt) rather than developed microstructure -- at low
        noise amplitude they're visually near-flat and arguably not
        useful/desirable training targets either. Default 0 keeps
        everything; pass e.g. 10000 to exclude the noise-dominated regime.

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

            # Read statistics.csv if EITHER include_stats needs the full
            # columns, OR min_stdev_phi needs just stdev_phi -- share one
            # read either way rather than reading the file twice.
            stats_df = None
            if include_stats or min_stdev_phi is not None:
                stats_df = load.read_statistics_csv(run_dir / "statistics.csv")

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

            for step in metadata.save_steps:
                if step in bad_steps or step < min_step:
                    continue
                if min_stdev_phi is not None:
                    if step not in stats_df.index:
                        continue  # can't verify the criterion -- exclude rather than assume
                    stdev = stats_df.loc[step, "stdev_phi"]
                    if math.isnan(stdev) or stdev < min_stdev_phi:
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
    def from_sweep(cls, config: "load.SweepConfig", base: str | Path = "../datasets",
                    skip_bad: bool = True, cache_in_memory: bool = False,
                    augment: bool = False, min_step: int = 0, min_stdev_phi: float | None = None,
                    include_stats: bool = False, stat_names: list[str] | None = None,
                    ) -> "MicrostructureSnapshotDataset":
        """
        Convenience: pool every complete run in a sweep, skipping
        incomplete ones. NOTE: this does not split into train/val/test --
        for that, get the dir list via complete_run_dirs(), split it with
        split_run_dirs(), and construct one instance per split instead.
        """
        run_dirs = complete_run_dirs(config, base)
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
        return values.mean(dim=0), values.std(dim=0).clamp_min(1e-8)
