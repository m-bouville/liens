"""Run/step selection shared by every dataset in this module.

Extracted verbatim from datasets.py (no behavioural change): the filters that
decide WHICH runs and WHICH saved steps are eligible -- completeness, the
min_step / min_stdev_phi / min_normalized_stdev_phi / min_passing_steps
thresholds, the train/val/test split, and the one-time save-schedule report.
These reference nothing else in datasets.py, so they live here as a leaf module;
datasets.py re-exports them, so `from training.datasets import build_good_steps`
(and the other four names) keeps working unchanged across the tree.
"""
import math
from pathlib import Path

import torch

from utils import load_datasets as load


# (sweep, nx, ny) already reported in this process -- see complete_run_dirs.
_REPORTED_SWEEPS: set[str] = set()


def report_save_step_distribution(run_dirs: list[Path]) -> None:
    """One-line-per-group summary of how far each run was actually evolved.

    Printed on FIRST discovery of a sweep because a sweep is not necessarily
    homogeneous: runs regenerated later to reach past tau_down carry many more
    saved steps than the originals, and nothing else in the pipeline says so.
    Every window count downstream is then a mixture of two populations, and a
    change in the mixture looks exactly like a change in the physics.

    This matters most where it is least visible. tau_down grows as ~L^2.5
    (measured: 6e5 at 64, 2.5e6 at 128, ~1.4e7 at 256), so at large sizes only
    the deliberately-extended runs reach coarsening completion at all -- and
    those runs are the only source of the frozen absorbing states that dominate
    the large-dt error. A sweep that is half-extended behaves differently from
    either of its halves.

    Grouped by final saved step rather than listed per run: the interesting
    structure is "two populations, 900 runs to 2.5e6 and 300 to 1e7", not 1200
    individual numbers.
    """
    if not run_dirs:
        return
    # Deduplicated HERE rather than by each caller: the diagnostics that
    # discover runs directly (check_stdev_phi_time, check_stdev_phi_temperature)
    # bypass complete_run_dirs entirely, so a dedup key owned by that one
    # function silences nothing for them and duplicates nothing either -- it
    # simply never runs. Keyed on the directory the runs live in, which is what
    # actually identifies a sweep+size.
    sweep_key = str(run_dirs[0].parent.resolve())
    if sweep_key in _REPORTED_SWEEPS:
        return
    _REPORTED_SWEEPS.add(sweep_key)

    counts: dict[tuple[int, int], int] = {}
    for run_dir in run_dirs:
        try:
            # read_metadata takes the FILE, not the directory -- the rest of
            # this module already calls it that way; passing the directory
            # raises IsADirectoryError, which the except below would have
            # swallowed as "no metadata", silently reporting nothing at all.
            metadata = load.read_metadata(run_dir / "metadata.txt")
        except (OSError, KeyError, ValueError):
            continue
        steps = metadata.save_steps
        if not steps:
            continue
        counts[(len(steps), steps[-1])] = counts.get((len(steps), steps[-1]), 0) + 1
    if not counts:
        return
    total = sum(counts.values())
    if len(counts) == 1:
        (n_steps, last), n_runs = next(iter(counts.items()))
        print(f"  save schedule: all {n_runs} runs have {n_steps} saved steps, "
              f"last at {last:,}")
        return
    print(f"  save schedule: {len(counts)} distinct lengths across {len(run_dirs)} runs "
          f"-- a MIXED sweep, so window counts below pool populations evolved to "
          f"different times")
    for (n_steps, last), n_runs in sorted(counts.items()):
        # Percentages of the runs that HAVE readable metadata, not of
        # len(run_dirs): an unreadable run contributes to neither, so dividing
        # by the directory count would make the column silently fail to reach
        # 100% with no indication why.
        print(f"    {n_runs:5d} runs ({100 * n_runs / total:4.1f}%): "
              f"{n_steps:3d} saved steps, last at {last:>12,}")


def complete_run_dirs(base: str | Path, nx: int, ny: int) -> list[Path]:
    """
    All directories for one grid size that exist on disk and are marked
    complete -- reads base/<nx>x<ny>/metadata.txt directly (see
    load.enumerate_run_dirs_from_metadata), NOT config.txt: metadata.txt
    is co-located with the actual dataset, so it's always correct for
    THIS directory, with no risk of describing an unrelated sweep (which
    a shared, possibly-currently-mutated config.txt could).
    """
    run_dirs = [d for d in load.enumerate_run_dirs_from_metadata(base, nx, ny)
                if load.is_complete(d)]
    # Once per sweep+size per process (the dedup lives in the reporter itself,
    # so the diagnostics that bypass this function get the same treatment).
    report_save_step_distribution(run_dirs)
    return run_dirs


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
                     min_step: int, min_stdev_phi: float | None, stats_df,
                     min_normalized_stdev_phi: float | None = None) -> list[int]:
    """
    Steps from metadata.save_steps passing the filters shared by every
    dataset in this module: not missing/corrupt (bad_steps), not earlier
    than min_step, and (if min_stdev_phi is set) not NaN/below threshold
    in statistics.csv's stdev_phi column. Factored out so
    MicrostructureSnapshotDataset and MicrostructureEvolutionDataset
    can't silently diverge in what counts as an excluded step.

    min_normalized_stdev_phi: like min_stdev_phi, but thresholds stdev_phi
    DIVIDED BY the theoretical equilibrium amplitude sqrt(-a(T)/b),
    a(T)=a0*(T-T0) -- the same normalizer cspt.py uses to put every
    temperature's plateau at 1.0. A raw stdev_phi threshold keeps a larger
    FRACTION of a low-T run than a near-critical one (the ground-state
    amplitude shrinks like sqrt(T0-T)), biasing the surviving sample by
    temperature; normalizing removes that. Steps at T>=T0 (no equilibrium
    amplitude to divide by) cannot satisfy it and are excluded.
    """
    amplitude = None
    if min_normalized_stdev_phi is not None:
        a_T = metadata.a0 * (metadata.temperature - metadata.T0)
        amplitude = math.sqrt(max(-a_T / metadata.b, 0.0))
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
        if min_normalized_stdev_phi is not None:
            if stats_df is None or step not in stats_df.index or not amplitude:
                continue  # no stats, or T>=T0 (no equilibrium amplitude to normalize by)
            stdev = stats_df.loc[step, "stdev_phi"]
            if math.isnan(stdev) or stdev / amplitude < min_normalized_stdev_phi:
                continue
        kept.append(step)
    return kept


def build_good_steps(run_dirs: list[str | Path], skip_bad: bool = True,
                      min_step: int = 0, min_stdev_phi: float | None = None,
                      min_passing_steps: int | None = None,
                      split_label: str = "",
                      min_normalized_stdev_phi: float | None = None,
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

    min_passing_steps: if set, an ENTIRE run is excluded (kept_steps=[])
    when FEWER than this many of its steps pass min_stdev_phi -- not
    just those individual under-threshold steps, which min_stdev_phi
    alone already excludes regardless of this parameter. Requires
    min_stdev_phi to also be set (there's nothing to "pass" otherwise).

    Rationale: min_stdev_phi alone still lets a run through on however
    few steps happen to individually clear the threshold, even if that's
    a tiny fraction of the run -- a real example from this project's own
    sweep: T990_n007_s599, 65 saved steps, only 2-4 EVER clear a 1%
    stdev_phi threshold (near T0, phase separation may simply never
    substantially develop within the simulated window -- critical
    slowing down, not a filtering artifact; confirmed by this project's
    own evaluation/check_stdev_phi_temperature.py, where the collapse
    persists unchanged even after excluding the first min_step steps).
    Those 2-4 survivors are themselves marginal BY CONSTRUCTION -- steps
    that just barely clear a threshold almost nothing else in the run
    clears are, definitionally, right at the boundary between genuine
    structure and noise floor, not confidently on the "real signal"
    side of it. And because so few steps survive, spread across the
    whole run, windows built from them span unusually large, irregular
    time gaps -- landing exactly in the large-dt regime this project's
    own diagnostics have separately shown to have the worst error
    (see losses.py's own DtDecadeWeights docstring). A per-run summary
    ("T990_n007_s599 performs poorly") built from a handful of such
    windows says little about general model quality -- it's dominated
    by whatever that one run's few borderline survivors happen to
    produce, not by anything representative.

    Applied consistently everywhere build_good_steps itself is used
    (train, val, test alike -- there is no special-cased subset), the
    same way min_step/min_stdev_phi already are: this changes what data
    EXISTS at all under the given filter configuration, not what a
    model sees at train time vs. what it's graded on at test time, so
    it introduces no train/test inconsistency by construction.
    """
    if min_passing_steps is not None and min_stdev_phi is None \
            and min_normalized_stdev_phi is None:
        raise ValueError("min_passing_steps requires min_stdev_phi or "
                          "min_normalized_stdev_phi to also be set -- there's no "
                          "'passing' threshold to count against otherwise")
    good_steps: dict[Path, list[int]] = {}
    n_runs_dropped_for_too_few_passing = 0
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

        _need_stats = (min_stdev_phi is not None
                       or min_normalized_stdev_phi is not None)
        stats_df = load.read_statistics_csv(run_dir / "statistics.csv") \
            if _need_stats else None
        kept_steps = _filtered_steps(metadata, bad_steps, min_step, min_stdev_phi,
                                     stats_df, min_normalized_stdev_phi)
        if min_passing_steps is not None and len(kept_steps) < min_passing_steps:
            n_runs_dropped_for_too_few_passing += 1
            kept_steps = []
        good_steps[run_dir] = kept_steps

    if min_passing_steps is not None and n_runs_dropped_for_too_few_passing:
        _what = f"{split_label} runs" if split_label else "runs"
        _pct = 100.0 * n_runs_dropped_for_too_few_passing / len(run_dirs) if run_dirs else 0.0
        # Name whichever threshold actually ran -- under --normalized the
        # criterion is min_normalized_stdev_phi, and printing "min_stdev_phi=None"
        # there read as "no filter", the opposite of the truth.
        _crit = (f"min_normalized_stdev_phi={min_normalized_stdev_phi}"
                 if min_normalized_stdev_phi is not None
                 else f"min_stdev_phi={min_stdev_phi}")
        print(f"build_good_steps: {n_runs_dropped_for_too_few_passing}/{len(run_dirs)} ({_pct:.1f}%) {_what} "
              f"dropped ENTIRELY -- fewer than min_passing_steps={min_passing_steps} steps "
              f"cleared {_crit}")

    return good_steps


