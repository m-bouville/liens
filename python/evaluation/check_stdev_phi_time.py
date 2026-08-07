"""
Checkpoint-free diagnostic for the t / Delta_t / temperature CONFOUND.

The problem this exists to address
----------------------------------
This project's save schedule is log-uniform: after the first few steps
it repeats {1, 1.15, 1.3, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 4.5, 5,
5.5, 6, 7, 8, 9} x 10^k, so Delta_t/t sits in [0.083, 0.167] (median
0.125) for every window in the sweep and corr(log t, log Delta_t) =
0.997. Delta_t is therefore a deterministic function of t to within a
factor of ~1.4, and NO stratification or reweighting of the existing
consecutive-step windows can separate "the model is bad at large
Delta_t" from "the model is bad at long times". Every per-dt-decade
table in evaluation/ is equally a per-time-decade table.

Temperature adds a THIRD variable to the same tangle. The driving force
a(T) = a0*(T-T0) vanishes as T -> T0, so near-critical runs evolve more
slowly: a microstructure state reached early at low T may only be
reached late near T0 -- and "late" means "at larger Delta_t", by the
schedule above. So an apparent temperature effect on model error (which
check_deriv_temperature.py and check_parameter_dependence.py both found,
around T >= 0.9 with T0 = 1) could be a Delta_t effect wearing a
temperature costume, with no way to tell them apart from the error
tables alone.

What this diagnostic does
-------------------------
Answers the DATA-side half of that question, with no model involved:

  1. stdev_phi as a function of t, ONE CURVE PER TEMPERATURE, averaged
     over noise and seed (the two sweep axes that are not under test).
     This is the raw picture: does the phase separation simply happen
     LATER near T0?

  2. The same curves normalized by each temperature's own equilibrium
     amplitude sqrt(-a(T)/b). Near T0 that amplitude shrinks toward
     zero (this is what check_stdev_phi_temperature.py established), so
     raw curves differ in HEIGHT for a reason that has nothing to do
     with timing. Dividing it out leaves timing alone.

  3. THE TEST: rescale each temperature's own time axis by a
     characteristic time tau(T) -- the time at which its normalized
     curve first reaches ref_fraction of its own plateau -- and check
     whether the curves COLLAPSE onto one master curve.

     collapse  -> the temperature difference IS a pure time rescaling.
                  "Poor near T0" and "poor at long times" are then the
                  same statement, and temperature is not an independent
                  factor: it is a clock speed.
     no collapse -> temperature does something beyond rescaling the
                  clock (different morphology, not just a slower version
                  of the same one), and must be treated as its own
                  variable.

  4. The bridge back to the confound: for each temperature, the Delta_t
     in force at t = tau(T). This is the quantitative statement of the
     problem -- the SAME physical state is sampled at systematically
     DIFFERENT step spacings depending on temperature, so a fixed
     Delta_t decade does not correspond to a fixed physical regime.

Deliberately NOT filtered by default: unlike
check_stdev_phi_temperature.py (whose question is about a min_stdev_phi
threshold, and which therefore warns loudly at min_step=0), the early
steps are the POINT here -- the transition is what carries the timing
information. --min-step exists to match a training population if wanted,
but using it discards the part of the curve tau is estimated from.

Reads statistics.csv and metadata.txt directly across an entire sweep --
no model, checkpoint, or GPU needed at all.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_stdev_phi_time --base-path ../datasets --size 64
"""

import argparse
from training.datasets import report_save_step_distribution
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from utils.plots import log_axis_ticks as _log_axis_ticks

from utils import load_datasets as load

_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/

# Curves are compared on a shared LOG-spaced grid; this is how many
# points that grid has. Only affects the reported spread numbers'
# resolution, never which conclusion they support.
_COLLAPSE_GRID_POINTS = 40

# How runs are combined, at BOTH aggregation levels (noise within a
# seed, then across seeds). Median by default: a single seed trapped in
# a metastable stationary state -- flat interfaces on a periodic box
# have zero curvature and so zero Allen-Cahn driving force, which is a
# genuine fixed point, not slow dynamics -- otherwise drags the centre
# of a 7-seed sample by its full share.
#
# KNOW THE TRADE. The late-time distribution is BIMODAL, with two
# absorbing states: trapped (~96% of the equilibrium amplitude,
# interfaces persist forever) and fully coarsened (exactly 0, one
# domain). A mean over seeds is then p*0.96, i.e. a direct readout of
# the trapping PROBABILITY and smooth in temperature. A median is not a
# compromise between the two states: it snaps to whichever holds the
# majority, so it steps sharply as p crosses 1/2 and reports no
# information about p at all. Median is the better statistic through the
# transition (robust to one odd seed) and the worse one in the tail
# (discards the mixture weight). Both are available; neither is right
# everywhere.
_AGGREGATORS = {"median": np.median, "mean": np.mean}


def _interp_crossing(x_lo: float, x_hi: float, y_lo: float, y_hi: float, y_target: float) -> float:
    """
    Where a curve crosses y_target between two bracketing samples,
    interpolated in LOG x (the sampling itself is log-spaced, so a
    linear-in-x interpolation would systematically bias the crossing
    toward the upper sample -- by up to the full 15% step spacing, which
    is the same order as the tau differences this is trying to resolve).

    x_lo must be > 0; callers skip the t=0 sample for exactly that
    reason.
    """
    if y_hi == y_lo:
        return x_lo
    frac = (y_target - y_lo) / (y_hi - y_lo)
    return float(np.exp(np.log(x_lo) + frac * (np.log(x_hi) - np.log(x_lo))))


def _characteristic_time(steps: np.ndarray, normalized: np.ndarray, ref_fraction: float) -> float:
    """
    tau: the first t at which the AMPLITUDE-NORMALIZED curve reaches
    ref_fraction -- an ABSOLUTE threshold, not a fraction of each curve's
    own observed plateau.

    This distinction decides whether the diagnostic works at all. Defining
    the target relative to each curve's own observed plateau sounds more
    robust, but the plateau can only be estimated from the part of the
    evolution that fits inside the sweep's fixed time span -- and a
    SLOWER run (the near-T0 case, i.e. exactly the one under test) gets
    less far along its own curve, so its estimated plateau comes out
    lower, so its target comes out lower, so its tau lands at a
    DIFFERENT feature of the curve than a fast run's does. Two curves
    that are exact time-rescalings of each other would then fail to
    collapse purely because of where the observation window ends. The
    normalization by sqrt(-a(T)/b) already puts every temperature's
    theoretical plateau at 1.0, so an absolute threshold on the
    normalized curve is the same physical milestone at every temperature
    by construction.

    First crossing, interpolated in log t. Returns NaN when the curve
    never reaches the target within the sweep's own time span -- reported,
    never silently substituted, since "this temperature never got half
    way to its own equilibrium amplitude" is itself a finding.
    """
    positive = steps > 0  # log interpolation, and t=0 carries no timing information anyway
    steps, normalized = steps[positive], normalized[positive]
    if len(steps) < 2:
        return float("nan")

    for i in range(1, len(normalized)):
        if normalized[i - 1] < ref_fraction <= normalized[i]:
            return _interp_crossing(steps[i - 1], steps[i], normalized[i - 1], normalized[i],
                                     ref_fraction)
    return float("nan")


def _characteristic_time_down(steps: np.ndarray, normalized: np.ndarray,
                                ref_fraction: float) -> float:
    """
    tau_down: the time at which the amplitude-normalized curve falls back
    THROUGH ref_fraction, i.e. the coarsening-completion time.

    The companion to _characteristic_time, and it measures something
    physically different. The upward crossing is set by the LINEAR
    growth rate sigma = M*a0*(T0-T), a purely local quantity -- which is
    why it comes out identical across box sizes to a few percent. The
    downward crossing is set by curvature-driven coarsening running out
    of domains to eliminate, which terminates only when the domain size
    reaches the BOX, so it scales with the box (R ~ sqrt(t) gives
    completion ~ L^2). One number is size-independent and one is not;
    reporting both separates the physics from the lattice.

    The LAST downward crossing, not the first: a curve can dip below the
    threshold and recover while domains reorganise, and what is wanted
    is when it finally leaves, not the first excursion. A downward
    crossing is necessarily past the peak, so no separate peak-finding
    is needed.

    NaN when the curve never comes back down -- which is the expected
    answer for a box large enough that coarsening does not complete
    inside the simulated window, and is itself the finding.
    """
    positive = steps > 0
    steps, normalized = steps[positive], normalized[positive]
    if len(steps) < 2:
        return float("nan")
    last = float("nan")
    for i in range(1, len(normalized)):
        if normalized[i - 1] >= ref_fraction > normalized[i]:
            last = _interp_crossing(steps[i - 1], steps[i], normalized[i - 1], normalized[i],
                                     ref_fraction)
    return last


def _observed_plateau(normalized: np.ndarray) -> float:
    """
    Median of the last quarter of the normalized curve: how close this
    temperature actually gets to its own Landau-predicted equilibrium
    amplitude within the sweep's time span.

    Reported, not used, deliberately. It is the check on whether
    ref_fraction is a sensible target: a temperature whose observed
    plateau sits below ref_fraction has a NaN tau, and this column is
    what tells the reader that the cause was "never got there", not a bug.
    Median rather than max because a single noisy high step would
    otherwise flatter the curve.
    """
    if len(normalized) == 0:
        return float("nan")
    tail_start = max(1, int(len(normalized) * 0.75))
    return float(np.median(normalized[tail_start:]))


def _curve_spread(curves: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """
    How far apart a set of curves is, as one number: the mean (over a
    shared log-spaced x grid) of the standard deviation ACROSS curves at
    each grid point.

    Restricted to the x range every curve actually covers. That
    restriction is what makes the before/after comparison fair: rescaling
    x by tau shifts each curve's own domain, so an unrestricted
    comparison would silently measure different regions of the curves
    before and after and attribute the difference to the rescaling.

    Returns NaN if fewer than two curves overlap at all -- a spread
    across one curve is not a number, and returning 0.0 there would look
    like a perfect collapse.
    """
    usable = [(x, y) for x, y in curves if len(x) >= 2 and np.all(np.isfinite(x))]
    if len(usable) < 2:
        return float("nan")
    lo = max(float(x.min()) for x, _ in usable)
    hi = min(float(x.max()) for x, _ in usable)
    if not (hi > lo > 0):
        return float("nan")
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), _COLLAPSE_GRID_POINTS))
    stacked = np.array([np.interp(grid, x, y) for x, y in usable])
    return float(np.mean(np.std(stacked, axis=0)))


def _delta_t_at(save_steps: list[int], t: float) -> float:
    """
    The step spacing in force at time t: the gap between the two saved
    steps bracketing t. This is the quantity that is confounded with t
    by the log-uniform schedule, so it is what has to be reported
    alongside tau to make the confound explicit rather than implied.
    """
    if not np.isfinite(t):
        return float("nan")
    steps = np.asarray(save_steps, dtype=float)
    for i in range(1, len(steps)):
        if steps[i - 1] <= t <= steps[i]:
            return float(steps[i] - steps[i - 1])
    return float("nan")


def _finite_number(value) -> float | None:
    """
    A CSV cell as a float, or None if it is missing or not numeric.

    Exists because `isinstance(value, (int, float, np.floating))` is a
    trap on pandas output: np.float64 DOES subclass float, but np.int64
    does NOT subclass int. A column pandas infers as int64 -- which
    autocorr_length is, being whole pixels -- therefore fails that test
    on every row, and the caller sees an empty result rather than an
    error. stdev_phi is float64 and passes, so the two columns behave
    differently for a reason that has nothing to do with the data.

    bool is excluded explicitly: Python bool subclasses int, so a
    mis-parsed boolean column would otherwise arrive as 0.0/1.0.
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return None if np.isnan(number) else number
    return None


def _format_temp(temperature: float) -> str:
    """
    Shortest label that does not collide with a neighbouring sweep
    temperature. The sweep uses 0.025 steps low down but 0.005 steps near
    T0 (..., 0.975, 0.980, 0.985, 0.990, 0.995), so a fixed 2-decimal
    format collapses three distinct temperatures onto "0.98" -- which is
    exactly the region every panel here is about. 3 decimals with
    trailing zeros stripped keeps "0.55" short where that is unambiguous
    and prints "0.975" where it is not.
    """
    return f"{temperature:.3f}".rstrip("0").rstrip(".")


def _normalized_ylim(ax) -> None:
    """
    Put the top of a normalized-quantity axis at 100%, the Landau
    equilibrium amplitude the curves are divided by -- so "how close did
    this temperature get to its own predicted plateau" is read off the
    same frame in every panel, instead of each panel auto-scaling to its
    own maximum and making a curve that reached 0.3 look like a curve
    that reached 0.9.

    max(100, current), never a hard set_ylim(top=100): if a curve ever
    exceeds its predicted amplitude, clamping would hide real data, and
    an axis silently cropping its own contents is precisely the kind of
    invisible change this project has been bitten by. 100% is a floor for
    the axis top, not a ceiling for the data.
    """
    ax.set_ylim(top=max(100.0, ax.get_ylim()[1]))


def _fit_driving_force_law(temperatures, taus, T0: float) -> tuple[float, float, int]:
    """
    Least-squares fit of tau = C / (T0 - T)^n, in log-log.

    Returns (C, n, n_points). C is in whatever units `taus` is passed in, so
    the caller must label it -- a coefficient with no stated units is the kind
    of number that gets quoted back later attached to the wrong axis.

    n is fitted rather than fixed at 1. The linear-instability growth rate is
    sigma = M*a0*(T0-T), which predicts exactly n = 1, so the FITTED value is
    the interesting output: a departure from 1 is a real measurement about the
    data, not a fitting artifact to be assumed away. Measured n has come out
    around 1.03-1.05, with the excess concentrated near T0.
    """
    x, y = [], []
    for temp in temperatures:
        drive, tau = T0 - temp, taus.get(temp, float("nan"))
        if drive > 0 and np.isfinite(tau) and tau > 0:
            x.append(np.log(drive))
            y.append(np.log(tau))
    if len(x) < 2:
        return float("nan"), float("nan"), len(x)
    slope, intercept = np.polyfit(np.array(x), np.array(y), 1)
    return float(np.exp(intercept)), float(-slope), len(x)


def _last_and_max(values: np.ndarray) -> tuple[float, float]:
    """Final and maximum value of a curve, NaN-safe (an all-NaN curve
    gives NaN rather than raising, so one bad temperature cannot take the
    whole figure down)."""
    finite = values[np.isfinite(values)] if values is not None else np.array([])
    if len(finite) == 0:
        return float("nan"), float("nan")
    return float(finite[-1]), float(np.max(finite))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """
    Ranks 0..n-1 with TIES SHARING THEIR MEAN RANK, rather than being
    ordered arbitrarily.

    argsort-of-argsort is the usual one-liner and is wrong here: it
    assigns tied values consecutive distinct ranks in whatever order they
    happen to sit in the array, so a set of seeds that behave IDENTICALLY
    gets spread across the entire rank range. Their spread then looks
    maximal, the MAD computed from it is large, and a genuinely
    anomalous seed disappears inside it -- the failure is silent and it
    gets worse the more the ordinary seeds agree.

    (scipy.stats.rankdata does this, but scipy is not a dependency of
    this project and one function does not justify making it one.)
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    ordered = values[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1] == ordered[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _saturation_step(run_max_steps: list[float], min_run_fraction: float) -> float:
    """
    The largest t that at least `min_run_fraction` of runs actually
    reach, i.e. the point past which the mean curve would start being an
    average over a SHRINKING population.

    Why this is needed at all: runs in this sweep do not all stop at the
    same step. Past the shortest run's own end, each (temperature, step)
    mean is taken over whichever runs happen to still have data there --
    so the curve steps discontinuously every time a run drops out, and
    the step is an artifact of the population changing, not of the
    physics. Truncating at a step nearly everyone reaches removes the
    artifact at the cost of a little late-time range.

    Exact rather than a quantile call: sorting the per-run maxima
    ascending, the k-th smallest is reached by exactly n-k runs, so
    k = floor((1 - fraction) * n) is the largest k whose step is still
    reached by at least fraction*n of them. np.quantile's interpolation
    would return a value BETWEEN two run maxima, which no run
    necessarily reaches -- the one thing this must not do.

    min_run_fraction <= 0 disables truncation entirely (returns inf),
    which is how to get the old, unfiltered view back.
    """
    if min_run_fraction <= 0 or not run_max_steps:
        return float("inf")
    ordered = sorted(run_max_steps)
    k = int(np.floor((1.0 - min_run_fraction) * len(ordered)))
    k = min(k, len(ordered) - 1)
    return float(ordered[k])


def characteristic_time_for_run(run_dir: Path, ref_fraction: float = 0.5) -> float:
    """
    tau for ONE run, in STEPS, from its own metadata.txt + statistics.csv.
    NaN if the run is incomplete, has no usable stdev_phi column, sits at
    T >= T0 (no equilibrium amplitude to normalize by), or never reaches
    ref_fraction within its own recorded span.

    Exists so that other diagnostics can de-dimensionalize their own dt
    axis by the SAME tau this module's collapse test is built on, rather
    than each re-deriving a characteristic time and slowly drifting
    apart -- the same single-implementation rule the project applies to
    any quantity used in two places.

    PER-RUN, not per-temperature: a window belongs to a run, so that is
    the granularity a per-window rescaling needs. It is noisier than the
    per-temperature curve check_stdev_phi_time() plots (which averages
    over noise and seed first), and the two will not agree exactly.
    Given that tau was measured to follow C/(T0-T) with no visible
    scatter, the difference should be small -- but it is a real
    difference, and a caller comparing numbers across the two must know
    which one it has.

    No model, checkpoint or GPU: reads two text files.
    """
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.txt"
    stats_path = run_dir / "statistics.csv"
    if not metadata_path.exists() or not stats_path.exists():
        return float("nan")
    metadata = load.read_metadata(metadata_path)
    if not metadata.is_complete:
        return float("nan")

    a_T = metadata.a0 * (metadata.temperature - metadata.T0)
    amplitude = float(np.sqrt(max(-a_T / metadata.b, 0.0)))
    if amplitude <= 0:
        return float("nan")

    stats_df = load.read_statistics_csv(stats_path)
    if "stdev_phi" not in stats_df.columns:
        return float("nan")

    steps, values = [], []
    for step in metadata.save_steps:
        if step <= 0 or step not in stats_df.index:
            continue
        value = _finite_number(stats_df.loc[step, "stdev_phi"])
        if value is not None:
            steps.append(float(step))
            values.append(value / amplitude)
    if len(steps) < 2:
        return float("nan")
    return _characteristic_time(np.array(steps), np.array(values), ref_fraction)


def _report_seed_breakdown(by_cell: dict, temp_amplitude: dict, seed: int,
                            aggregate=np.median) -> None:
    """
    Is a flagged seed anomalous as a SEED, or only in a few of its runs?

    The distinction decides what to do about it. A seed fixes the initial
    condition shared by all its runs, so if the anomaly is in the seed
    every one of its (temperature, noise) runs should look odd and the
    seed has to go. If instead a handful of runs look odd while the rest
    are ordinary, the initial condition is fine and those runs can be
    re-run individually.

    Method: rank this seed against the other seeds present in the SAME
    (temperature, noise, step) cell -- so runs are compared only against
    runs at identical parameters, and neither temperature nor noise
    coverage can bias the comparison. Aggregated two ways, by noise and
    by temperature, plus the individual (temperature, noise) runs with
    the most extreme median rank.
    """
    ranks_by_run: dict[tuple[float, float], list[float]] = defaultdict(list)
    cells: dict[tuple[float, float, int], dict[int, float]] = defaultdict(dict)
    for (temp, sd, noise, step), values in by_cell.items():
        amplitude = temp_amplitude.get(temp, 0.0)
        if amplitude > 0:
            cells[(temp, noise, step)][sd] = float(aggregate(values)) / amplitude

    for (temp, noise, _step), per_seed in cells.items():
        if seed not in per_seed or len(per_seed) < 2:
            continue
        seeds_here = list(per_seed)
        positions = _average_ranks(np.array([per_seed[sd] for sd in seeds_here]))
        normalized = dict(zip(seeds_here, positions / (len(seeds_here) - 1)))
        ranks_by_run[(temp, noise)].append(float(normalized[seed]))

    if not ranks_by_run:
        print(f"\nseed {seed}: no cell has another seed to compare against -- "
              f"cannot tell seed from run.")
        return

    per_run = {key: float(np.median(v)) for key, v in ranks_by_run.items()}
    print(f"\nseed {seed}: is the anomaly the SEED or a few RUNS of it?")
    print("  ranked against the other seeds at the SAME (temperature, noise, step); "
          "0.5 = typical, 1.0 = highest of the seeds present")

    by_noise: dict[float, list[float]] = defaultdict(list)
    by_temp: dict[float, list[float]] = defaultdict(list)
    for (temp, noise), rank in per_run.items():
        by_noise[noise].append(rank)
        by_temp[temp].append(rank)

    print("    by noise:  " + "  ".join(f"{noise:g}:{np.median(v):.2f}"
                                          for noise, v in sorted(by_noise.items())))
    print("    by temp :  " + "  ".join(f"{temp:g}:{np.median(v):.2f}"
                                          for temp, v in sorted(by_temp.items())))

    ranks = np.array(list(per_run.values()))
    # EITHER tail. A one-sided "rank > 0.9" test misses every anomaly that
    # runs slow rather than fast -- a seed whose runs lag the others sits
    # at rank 0.0 in every cell, which is exactly as extreme and exactly
    # as informative, but a top-decile test scores it as ordinary and
    # reports the wrong verdict with full confidence.
    high = np.abs(ranks - 0.5) > 0.4
    print(f"    {high.sum()}/{len(ranks)} of this seed's runs are extreme within their own cell "
          f"(median rank over its steps outside 0.1-0.9, in EITHER direction)")
    worst = sorted(per_run.items(), key=lambda kv: -abs(kv[1] - 0.5))[:8]
    print("    most extreme runs (T, noise, median rank): "
          + ", ".join(f"({t:g}, {n:g}, {r:.2f})" for (t, n), r in worst))

    fraction = high.mean()
    if fraction > 0.7:
        print(f"    -> SEED. {fraction:.0%} of its runs are extreme, across both noise and "
              f"temperature, which is what a shared initial condition looks like. Dropping the "
              f"seed (--exclude-seeds {seed}) is the right move; re-running individual runs "
              f"would reproduce it.")
    elif fraction < 0.3:
        print(f"    -> RUNS. Only {fraction:.0%} of its runs are extreme and the rest are "
              f"ordinary, so the initial condition is not the problem. Re-run the runs listed "
              f"above rather than dropping the seed, which would discard good data.")
    else:
        print(f"    -> AMBIGUOUS at {fraction:.0%}. Check whether the extreme runs cluster in "
              f"the by-noise or by-temperature rows above: clustering in one parameter points "
              f"at that parameter, not at the seed.")


def _report_autocorr_length(by_cell_ac: dict, taus: dict, nx: int, ny: int,
                             aggregate=np.median) -> None:
    """
    How far coarsening actually got, per temperature, in units the
    C++ measurement can express.

    autocorr_length is searched only out to min(Nx,Ny)*2/3 -- beyond that
    a periodic box starts correlating with itself -- so a returned value
    AT the cap means "never decayed within range", not a length. What
    fraction of samples sit at the cap is therefore a direct measure of
    how much of the sweep has coarsened past what the box can resolve,
    and it is the quantity that should fall as the box grows: domain
    size follows R ~ sqrt(t) independently of the box, so the time to
    reach a fixed FRACTION of the box scales as L^2 while the simulated
    window does not.

    The cap is taken from the DATA (the maximum value that actually
    occurs), not computed from nx/ny. The nominal formula is reported
    alongside and a mismatch is flagged: integer arithmetic in the C++
    can land a unit either side, and silently testing ">= 85" against
    data clipped at 84 would report 0% saturation for a sweep that is
    entirely saturated.
    """
    if not by_cell_ac:
        print("\nautocorr_length: not present in statistics.csv -- skipped.")
        return

    all_values = np.array([v for values in by_cell_ac.values() for v in values])
    cap = float(np.max(all_values))
    nominal = int(min(nx, ny) * 2 / 3)
    note = "" if abs(cap - nominal) < 0.5 else f"  (nominal min(nx,ny)*2/3 = {nominal} -- DIFFERS)"
    print(f"\nautocorr_length, search capped at {cap:g} px{note}")
    print("  a value AT the cap is 'never decayed within range', not a length")

    at_cap = float(np.mean(all_values >= cap))
    print(f"  overall {at_cap:.1%} of run-step samples sit at the cap "
          f"({len(all_values)} samples)")

    by_temp_step: dict[tuple[float, int], list[float]] = defaultdict(list)
    by_temp: dict[float, list[float]] = defaultdict(list)
    for (temp, _seed, _noise, step), values in by_cell_ac.items():
        by_temp_step[(temp, step)].append(float(aggregate(values)))
        by_temp[temp].extend(values)

    print("      T   median L   L/box   sat%     t_sat   t_sat/tau   n samples")
    for temp in sorted(by_temp):
        values = np.array(by_temp[temp])
        steps = sorted({st for (t, st) in by_temp_step if t == temp})
        # t_sat: first step whose across-seed median has reached the cap.
        # Median rather than any-run, so one early-saturating run cannot
        # date the whole temperature.
        t_sat = float("nan")
        for step in steps:
            if aggregate(by_temp_step[(temp, step)]) >= cap:
                t_sat = float(step)
                break
        tau = taus.get(temp, float("nan"))
        ratio = t_sat / tau if np.isfinite(t_sat) and np.isfinite(tau) and tau > 0 else float("nan")
        print(f"  {temp:5.3f}   {np.median(values):8.1f}   {np.median(values) / min(nx, ny):5.2f}   "
              f"{np.mean(values >= cap):5.1%}   {t_sat:7.4g}   {ratio:9.4g}   {len(values):9d}")

    print("  L/box is comparable across sweeps; sat% and t_sat are not directly comparable "
          "unless the sweeps ran for a similar number of steps.")


def _report_seed_ranks(by_temp_seed_step: dict, temp_amplitude: dict,
                        temperatures: list, aggregate=np.median) -> None:
    """Console table ranking each seed against the seeds it shares a cell with.

    Extracted from check_stdev_phi_time, which reached ~460 lines as each
    finding this session added a section to it. Everything below the curve
    construction is a pure function of already-computed data, so naming the
    blocks costs nothing and makes the top-level function readable as a
    sequence of steps rather than one long script.
    """
    # Per-seed comparison, STRATIFIED by (temperature, step).
    #
    # A seed's raw median over its own samples is NOT comparable across
    # seeds, because seeds do not all cover the same temperatures: this
    # sweep has general-purpose seeds present at every temperature and
    # others run only at the high-T end, where stdev_phi is small for
    # purely physical reasons. Ranking seeds by their raw medians
    # therefore sorts them mostly by WHICH TEMPERATURES THEY COVER, and
    # an outlier test built on that centre compares two different
    # populations.
    #
    # Fixed by comparing seeds only against the seeds they actually share
    # a cell with. Within each (temperature, step) cell every seed sees
    # identical physics, so a seed's position within that cell is
    # meaningful; aggregating those positions over its own cells gives a
    # statistic that does not care which cells it has. Two are reported:
    #   med rank -- median normalized rank within a cell (0 = lowest of
    #     the seeds present, 1 = highest, 0.5 = typical). Scale-free, so
    #     it is not dominated by the low-T cells where values are large.
    #   med dev  -- median (value - cell median), for magnitude, since a
    #     rank alone cannot say whether being highest matters.
    # The outlier flag keys on RANK; dev is context.
    cell_values: dict[tuple[float, int], dict[int, float]] = defaultdict(dict)
    for (temp, seed, step), values in by_temp_seed_step.items():
        amp = temp_amplitude.get(temp, 0.0)
        if amp > 0:
            cell_values[(temp, step)][seed] = float(aggregate(values)) / amp

    seed_ranks: dict[int, list[float]] = defaultdict(list)
    seed_devs: dict[int, list[float]] = defaultdict(list)
    seed_temps: dict[int, set] = defaultdict(set)
    for (temp, _step), per_seed in cell_values.items():
        if len(per_seed) < 2:
            continue  # a rank within a cell of one is not a comparison
        seeds_here = list(per_seed)
        vals = np.array([per_seed[sd] for sd in seeds_here])
        cell_median = float(np.median(vals))
        order = _average_ranks(vals)  # ties share their mean rank -- see _average_ranks
        for position, sd in zip(order, seeds_here):
            seed_ranks[sd].append(float(position) / (len(seeds_here) - 1))
            seed_devs[sd].append(per_seed[sd] - cell_median)
            seed_temps[sd].add(temp)

    if seed_ranks:
        print("\nper-seed comparison, stratified by (temperature, step) -- seeds are compared "
              "ONLY against seeds present in the same cell, so a seed run at a subset of "
              "temperatures is not penalised for its coverage:")
        print("     seed   n_temps          T range   med rank    med dev    p10 dev    p90 dev"
              "    n cells")
        rows = []
        for sd, ranks in seed_ranks.items():
            devs = np.array(seed_devs[sd])
            temps_here = sorted(seed_temps[sd])
            rows.append((sd, len(temps_here), temps_here[0], temps_here[-1],
                          float(np.median(ranks)), float(np.median(devs)),
                          float(np.percentile(devs, 10)), float(np.percentile(devs, 90)),
                          len(ranks)))
        rows.sort(key=lambda r: r[4])
        for sd, n_t, t_lo, t_hi, med_rank, med_dev, p10, p90, n_cells in rows:
            print(f"  {sd:7d}   {n_t:7d}   {t_lo:6.3f}-{t_hi:6.3f}   {med_rank:8.3f}   "
                  f"{med_dev:+8.4f}   {p10:+8.4f}   {p90:+8.4f}   {n_cells:8d}")

        coverages = {r[1] for r in rows}
        if len(coverages) > 1:
            print(f"  NOTE: seeds cover different numbers of temperatures {sorted(coverages)}. "
                  f"That is exactly why the flag below uses rank-within-cell and not the raw "
                  f"per-seed median, which would mostly rank seeds by their coverage.")

        med_ranks = np.array([r[4] for r in rows])
        if len(med_ranks) >= 3:
            centre = float(np.median(med_ranks))
            spread = float(np.median(np.abs(med_ranks - centre)))
            if spread > 0:
                outliers = [r[0] for r in rows if abs(r[4] - centre) > 3 * spread]
            else:
                # MAD collapses to exactly zero whenever a majority of
                # seeds agree to the last bit. A "> 3 * spread" test then
                # flags nothing at all -- silently failing in precisely
                # the case where the outlier is CLEAREST, since every
                # other seed agreeing is what drove the spread to zero.
                tolerance = 1e-9 + 0.01 * max(abs(centre), 1e-3)
                outliers = [r[0] for r in rows if abs(r[4] - centre) > tolerance]
            if outliers:
                print(f"  -> seed(s) {outliers} sit more than 3 MADs from the median seed RANK. "
                      f"Re-run with --exclude-seeds {' '.join(str(o) for o in outliers)} to see "
                      f"the effect on every number above.")


def _report_collapse_tests(curves: dict, taus: dict, taus_down: dict, temperatures: list):
    """The two collapse tests -- on tau (growth) and on tau_down (coarsening).

    Returns (collapse_temps, down_temps): the temperature subsets each test
    could actually use. _plot needs both, so that its panels show exactly the
    curves the printed ratios were computed from -- if the two ever disagreed,
    the figure would be illustrating a different population than the number
    beside it.
    """
    # THE TEST. Spread across temperature curves before vs after
    # rescaling t by tau. Both computed on normalized curves so the
    # amplitude difference (which rescaling t cannot possibly fix) is
    # not being counted as a failure to collapse.
    collapse_temps = [t for t in temperatures
                       if curves[t]["normalized"] is not None and np.isfinite(taus[t])]
    before = _curve_spread([(curves[t]["steps"], curves[t]["normalized"]) for t in collapse_temps])
    after = _curve_spread([(curves[t]["steps"] / taus[t], curves[t]["normalized"])
                            for t in collapse_temps])

    print(f"\ncollapse test over {len(collapse_temps)} temperature(s):")
    if not np.isfinite(before) or not np.isfinite(after):
        print("  UNDEFINED -- needs at least two temperatures with a well-defined tau and an "
              "overlapping time range. A single-temperature sweep cannot pose this question.")
    else:
        ratio = after / before if before > 0 else float("nan")
        print(f"  mean across-curve spread, normalized stdev_phi vs t        = {before:.4f}")
        print(f"  mean across-curve spread, normalized stdev_phi vs t/tau(T) = {after:.4f}")
        print(f"  ratio after/before = {ratio:.3f}")
        if ratio < 0.5:
            print("  -> COLLAPSES. The temperature dependence is largely a pure time rescaling: "
                  "near-T0 runs are a SLOWER version of the same evolution, not a different one. "
                  "'Poor near T0' and 'poor at long times' are then the same statement, and the "
                  "Delta_t column above shows the mechanism -- the same physical state is sampled "
                  "at a systematically larger step spacing near T0.")
        elif ratio < 0.9:
            print("  -> PARTIAL collapse. Time rescaling explains some of the temperature "
                  "dependence but not all of it; both effects are present and neither can be "
                  "dismissed.")
        else:
            print("  -> DOES NOT COLLAPSE. Temperature is doing something beyond setting the clock "
                  "speed, so it must be treated as an independent variable and cannot be explained "
                  "away as a Delta_t artifact.")

    down_temps = [t for t in temperatures
                   if curves[t]["normalized"] is not None and np.isfinite(taus_down[t])]
    before_d = _curve_spread([(curves[t]["steps"], curves[t]["normalized"]) for t in down_temps])
    after_d = _curve_spread([(curves[t]["steps"] / taus_down[t], curves[t]["normalized"])
                              for t in down_temps])
    print(f"\ncollapse test on tau_down (coarsening completion) over {len(down_temps)} "
          f"temperature(s):")
    if not down_temps:
        print("  NOT REACHED at any temperature: no curve falls back through the threshold inside "
              "the simulated window. Coarsening does not complete on this box in this many steps, "
              "so there is no late-time absorbing state to contaminate the tau collapse above.")
    elif not np.isfinite(before_d) or not np.isfinite(after_d):
        print("  UNDEFINED -- needs at least two temperatures with a finite tau_down and an "
              "overlapping time range.")
    else:
        ratio_d = after_d / before_d if before_d > 0 else float("nan")
        print(f"  mean across-curve spread vs t          = {before_d:.4f}")
        print(f"  mean across-curve spread vs t/tau_down = {after_d:.4f}")
        print(f"  ratio after/before = {ratio_d:.3f}")
        print("  tau_down is set by domains growing to the BOX size (R ~ sqrt(t), so completion "
              "~ L^2), not by the local growth rate -- unlike tau it is expected to depend on "
              "the system size.")
    return collapse_temps, down_temps


def check_stdev_phi_time(
    base_path: Path, size: int, min_step: int = 1, ref_fraction: float = 0.5,
    min_run_fraction: float = 0.9, exclude_seeds: set[int] | None = None,
    inspect_seed: int | None = None, statistic: str = "median",
    plot: bool = True, output_path: Path | None = None,
) -> Path:
    """
    Collects stdev_phi(t) for every complete run in the sweep, groups by
    temperature (averaging over noise and seed), and tests whether the
    temperature dependence is a pure time rescaling.

    plot (default True): render the figure. Set False for the console
    tables alone -- tau/tau_down, the collapse ratios, the per-seed
    comparison and the autocorr_length summary are all console output and
    do not need the figure. Rendering eight panels of ~24 curves at
    24x9in/120dpi costs ~1.5 s and is 99% of this function's total
    runtime on a small sweep, so it dominates any repeated or scripted
    use. The return value is still the path the figure WOULD occupy; no
    file is written.

    statistic ("median" by default, or "mean"): how runs are combined at
    both levels, and what the shaded band means (p25-p75 for the median,
    +-1 sd for the mean). See _AGGREGATORS for why neither is right
    everywhere -- in particular, only the mean carries the late-time
    trapping probability.

    inspect_seed: break one seed down into its individual runs, to decide
    whether an anomaly belongs to the SEED (shared initial condition, so
    the seed should be dropped) or to a handful of RUNS of it (which can
    simply be re-run). See _report_seed_breakdown.

    exclude_seeds: seed values to drop entirely, across every
    temperature. For removing a seed identified as anomalous by the
    per-seed table this function prints -- dropping it at ONE temperature
    only would leave a curve whose population differs from its
    neighbours', which is the same class of artifact the per-seed
    aggregation exists to remove.

    min_run_fraction (default 0.9): truncate every curve at the largest
    step at least this fraction of runs actually reach. Runs in this
    sweep do not all stop together, and past the shortest one's end each
    mean is taken over a shrinking population, which makes the curve
    step discontinuously as runs drop out. 0 disables it. Applied BEFORE
    tau is estimated, so tau and the collapse test see exactly the curves
    that are plotted -- in practice tau sits far below the cap, so this
    should not move it, but the two must not be allowed to disagree.

    A side benefit worth knowing: with the cap on, every temperature's
    curve ends at the SAME t, so the "last value vs T" panel compares
    values at one time rather than at whatever time each run happened to
    stop.

    ref_fraction: which fraction of its own plateau a normalized curve
    must reach to define tau. 0.5 by default -- the steepest part of the
    curve, so the crossing time is least sensitive to noise in the
    measurement. Exposed rather than hardcoded because the conclusion
    should not depend on it: if the collapse only appears at one
    particular value, that IS the finding, and it should be visible.

    Returns the path of the figure written.
    """
    if output_path is None:
        # output/datasets/: this diagnostic is checkpoint-free -- it reads
        # statistics.csv and metadata.txt only -- so it belongs with the
        # dataset-level outputs rather than under a per-stage folder. The
        # size prefix matters because every conclusion here (tau_down,
        # trapping, the near-T0 amplitude deficit) is size-dependent, and
        # an unprefixed filename silently overwrites one sweep's figure
        # with another's.
        output_path = (_PYTHON_ROOT.parent / "output" / "datasets"
                        / f"{size}x{size}-stdev_phi_time.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if statistic not in _AGGREGATORS:
        raise ValueError(f"statistic must be one of {sorted(_AGGREGATORS)}, got {statistic!r}")
    aggregate = _AGGREGATORS[statistic]

    run_dirs = load.enumerate_run_dirs_from_metadata(base_path, size, size)

    # This diagnostic discovers runs directly, bypassing

    # complete_run_dirs -- so it must ask for the save-step report itself.

    # A sweep with runs regenerated to pass tau_down is a MIXTURE, and every

    # count below pools populations evolved to different times.

    report_save_step_distribution(run_dirs)

    # (temperature, seed, step) -> [stdev_phi from each NOISE run].
    #
    # TWO-LEVEL on purpose. A seed fixes the initial condition, and every
    # noise value is run from the same seeds, so the runs sharing a seed
    # are not independent samples of the late-time outcome -- whether a
    # run finishes coarsening is largely decided by which domain
    # configuration it started from. Averaging over all runs at once
    # therefore treats n_noises x n_seeds as the sample size when the
    # real sample size is n_seeds, understating the spread by
    # sqrt(n_noises) (~3.6x on this sweep) and letting one anomalous seed
    # -- which carries n_noises runs with it -- move the mean as if it
    # were n_noises independent observations. Noise is collapsed first,
    # then statistics are taken across seeds.
    # Keyed on ROUNDED temperature so float formatting differences
    # between metadata files can never split one sweep temperature into
    # two near-identical curves.
    # Keyed by (temperature, seed, noise, step): one entry per RUN per
    # step. Noise is kept in the key rather than averaged away on
    # collection so that a flagged seed can be broken down into its own
    # runs afterwards -- deciding "is this the seed, or a few runs of it"
    # is impossible once noise has been collapsed.
    by_cell: dict[tuple[float, int, float, int], list[float]] = defaultdict(list)
    by_cell_ac: dict[tuple[float, int, float, int], list[float]] = defaultdict(list)
    grid_nx = grid_ny = 0
    T0_value = 1.0  # overwritten from the first run's own metadata; never assumed
    temp_amplitude: dict[float, float] = {}
    temp_save_steps: dict[float, list[int]] = {}
    temp_dt_scale: dict[float, float] = {}
    temp_run_count: dict[float, int] = defaultdict(int)
    temp_seeds: dict[float, set] = defaultdict(set)
    run_max_steps: list[float] = []  # per run, for the saturation cap below
    n_skipped_incomplete = n_skipped_no_stats = n_skipped_excluded_seed = 0

    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.txt"
        if not metadata_path.exists():
            continue
        metadata = load.read_metadata(metadata_path)
        if not metadata.is_complete:
            n_skipped_incomplete += 1
            continue
        stats_path = run_dir / "statistics.csv"
        if not stats_path.exists():
            n_skipped_no_stats += 1
            continue
        stats_df = load.read_statistics_csv(stats_path)
        if "stdev_phi" not in stats_df.columns:
            n_skipped_no_stats += 1
            continue

        temp = round(float(metadata.temperature), 6)
        # Same Landau construction check_stdev_phi_temperature.py uses,
        # and for the same reason read from THIS run's own metadata
        # rather than hardcoded: stable phases at +-sqrt(-a(T)/b) with
        # a(T) = a0*(T-T0), clipped at 0 for T >= T0 where phi=0 is the
        # only stable point.
        a_T = metadata.a0 * (metadata.temperature - metadata.T0)
        temp_amplitude[temp] = float(np.sqrt(max(-a_T / metadata.b, 0.0)))
        temp_save_steps[temp] = list(metadata.save_steps)
        temp_dt_scale[temp] = float(metadata.dt)
        if exclude_seeds and metadata.seed in exclude_seeds:
            n_skipped_excluded_seed += 1
            continue
        temp_run_count[temp] += 1
        temp_seeds[temp].add(metadata.seed)
        grid_nx, grid_ny = metadata.nx, metadata.ny
        T0_value = float(metadata.T0)

        has_autocorr = "autocorr_length" in stats_df.columns
        this_run_max = 0
        for step in metadata.save_steps:
            if step < min_step or step not in stats_df.index:
                continue
            value = _finite_number(stats_df.loc[step, "stdev_phi"])
            if value is not None:
                by_cell[(temp, metadata.seed, float(metadata.noise), int(step))].append(value)
                if has_autocorr:
                    ac = _finite_number(stats_df.loc[step, "autocorr_length"])
                    if ac is not None:
                        by_cell_ac[(temp, metadata.seed, float(metadata.noise),
                                     int(step))].append(ac)
                this_run_max = max(this_run_max, int(step))
        if this_run_max > 0:
            run_max_steps.append(float(this_run_max))

    if not by_cell:
        raise ValueError(f"No (temperature, step, stdev_phi) samples found under {base_path} "
                          f"for size={size} -- check the path and that statistics.csv files exist")

    t_cap = _saturation_step(run_max_steps, min_run_fraction)
    n_dropped = sum(1 for (_, _, _, step) in by_cell if step > t_cap)
    by_cell = defaultdict(list, {key: v for key, v in by_cell.items() if key[3] <= t_cap})
    by_cell_ac = defaultdict(list, {key: v for key, v in by_cell_ac.items() if key[3] <= t_cap})

    # LEVEL 0: noise collapsed within each (temperature, seed, step).
    by_temp_seed_step: dict[tuple[float, int, int], list[float]] = defaultdict(list)
    for (temp, seed, _noise, step), values in by_cell.items():
        by_temp_seed_step[(temp, seed, step)].extend(values)

    # LEVEL 1: collapse noise within each (temperature, seed, step), so
    # each seed contributes exactly one number per step. LEVEL 2 (the
    # mean and sd built from these lists further down) is then genuinely
    # across seeds.
    by_temp_step: dict[tuple[float, int], list[float]] = defaultdict(list)
    seed_of: dict[tuple[float, int], list[int]] = defaultdict(list)
    for (temp, seed, step), values in by_temp_seed_step.items():
        by_temp_step[(temp, step)].append(float(aggregate(values)))
        seed_of[(temp, step)].append(seed)
    if not by_temp_step:
        raise ValueError(f"Every sample lies past the saturation cap t={t_cap:g} -- "
                          f"min_run_fraction={min_run_fraction} is too strict for this sweep")

    temperatures = sorted({t for t, _ in by_temp_step})
    all_seeds = sorted({sd for seeds in temp_seeds.values() for sd in seeds})
    print(f"{sum(temp_run_count.values())} runs across {len(temperatures)} temperatures and "
          f"{len(all_seeds)} seeds ({n_skipped_incomplete} incomplete, {n_skipped_no_stats} "
          f"missing statistics.csv/stdev_phi"
          f"{f', {n_skipped_excluded_seed} excluded by --exclude-seeds' if n_skipped_excluded_seed else ''}"
          f", skipped)")
    print(f"  combined with the {statistic.upper()} at both levels (noise within a seed, then "
          f"across seeds); band = {'p25-p75' if statistic == 'median' else '+-1 sd'}, "
          f"a seed-to-seed spread, not a run-to-run one.")
    if statistic == "median":
        print("  NOTE: the late-time distribution is bimodal (trapped in a stationary two-phase "
              "state vs fully coarsened), and a median snaps to whichever state holds the "
              "majority rather than reporting the mixture. Use --statistic mean there: its value "
              "is the trapping probability times the trapped amplitude.")
    if np.isfinite(t_cap):
        print(f"saturation cap: t <= {t_cap:g}, the largest step reached by at least "
              f"{min_run_fraction:.0%} of runs (run extents span "
              f"{min(run_max_steps):g}..{max(run_max_steps):g}); {n_dropped} "
              f"(temperature, seed, noise, step) sample(s) -- i.e. per-RUN samples, not "
              f"curve points -- beyond it dropped.")
        if n_dropped:
            print("  Past that point a mean would be taken over a SHRINKING set of runs, so the "
                  "curve would step discontinuously as runs drop out -- an artifact of the "
                  "population changing, not of the physics. Pass min_run_fraction=0 to see the "
                  "untruncated curves.")
    else:
        print("saturation cap: DISABLED (min_run_fraction<=0) -- late-time curve segments may be "
              "averages over a shrinking set of runs.")

    # Per temperature: the centre curve across seeds, a band, and how
    # many seeds contributed at each step. The band is carried through to
    # the plot because combining seeds at all is an assumption -- that
    # they are draws from one population -- which the reader should be
    # able to check rather than take on trust. It is matched to the
    # statistic: p25-p75 around a median, +-1 sd around a mean. A sd
    # drawn around a median describes neither.
    curves: dict[float, dict[str, np.ndarray]] = {}
    for temp in temperatures:
        # t=0 excluded throughout: every axis here is logarithmic (the
        # schedule is log-spaced, so a linear axis would compress the
        # entire transition into the left edge), t=0 has no defined
        # position on one, and _curve_spread's own overlap range would
        # collapse to "starts at 0" for every curve and return NaN.
        steps = sorted(s for t, s in by_temp_step if t == temp and s > 0)
        centre = np.array([aggregate(by_temp_step[(temp, s)]) for s in steps])
        if statistic == "median":
            lo = np.array([np.percentile(by_temp_step[(temp, s)], 25) for s in steps])
            hi = np.array([np.percentile(by_temp_step[(temp, s)], 75) for s in steps])
        else:
            sd = np.array([np.std(by_temp_step[(temp, s)]) for s in steps])
            lo, hi = centre - sd, centre + sd
        n = np.array([len(by_temp_step[(temp, s)]) for s in steps])
        curves[temp] = {"steps": np.array(steps, dtype=float), "centre": centre,
                         "lo": lo, "hi": hi, "n": n}

    # Normalization by equilibrium amplitude. Undefined at T >= T0
    # (amplitude 0): those temperatures keep their raw curve, are
    # excluded from the collapse test, and are reported as excluded
    # rather than dropped silently.
    unnormalizable = [t for t in temperatures if temp_amplitude.get(t, 0.0) <= 0]
    for temp in temperatures:
        amp = temp_amplitude.get(temp, 0.0)
        curves[temp]["normalized"] = curves[temp]["centre"] / amp if amp > 0 else None

    taus, taus_down = {}, {}
    for temp in temperatures:
        norm = curves[temp]["normalized"]
        taus[temp] = (_characteristic_time(curves[temp]["steps"], norm, ref_fraction)
                       if norm is not None else float("nan"))
        taus_down[temp] = (_characteristic_time_down(curves[temp]["steps"], norm, ref_fraction)
                            if norm is not None else float("nan"))

    ragged: list[float] = []
    print(f"\nper-temperature summary (tau = first t at which stdev_phi reaches "
          f"{ref_fraction:g} x sqrt(-a(T)/b); tau_down = when it falls back through it):")
    print("      T   n_runs  n_seeds  seeds/step   equilib.ampl        tau   tau_down   "
          "Delta_t at tau   Delta_t (scaled)   obs. plateau")
    for temp in temperatures:
        tau = taus[temp]
        d_steps = _delta_t_at(temp_save_steps[temp], tau)
        d_scaled = d_steps * temp_dt_scale[temp]
        norm = curves[temp]["normalized"]
        plateau = _observed_plateau(norm) if norm is not None else float("nan")
        n_per_step = curves[temp]["n"]
        n_lo, n_hi = int(n_per_step.min()), int(n_per_step.max())
        # n_min..n_max: how many SEEDS actually contributed at each step of
        # THIS temperature's curve. Equal values mean every retained step
        # was averaged over the same population. Unequal means the curve
        # silently changes population along its own length -- the mean at
        # one step is then not comparable with the mean at another, and
        # any feature at the transition between them is an artifact. The
        # saturation cap fixes this for differing run EXTENTS; it cannot
        # fix differing save SCHEDULES, which is what this column detects.
        n_col = f"{n_lo}" if n_lo == n_hi else f"{n_lo}-{n_hi}"
        print(f"  {temp:5.3f}   {temp_run_count[temp]:6d}   {len(temp_seeds[temp]):6d}   "
              f"{n_col:>9s}   {temp_amplitude[temp]:12.4f}   "
              f"{tau:8.4g}   {taus_down[temp]:8.4g}   {d_steps:14.4g}   {d_scaled:16.4g}   "
              f"{plateau:12.3f}")
        if n_lo != n_hi:
            ragged.append(temp)

    if ragged:
        print(f"\n  WARNING: {len(ragged)} temperature(s) have a VARYING number of contributing "
              f"seeds across their own steps: {[round(t, 3) for t in ragged]}. Their mean curves "
              f"change population along the curve, so any feature there may be a composition "
              f"artifact rather than physics. Most often this means the runs at that temperature "
              f"do not share one save schedule.")

    if unnormalizable:
        print(f"\n  NOTE: T >= T0 for {unnormalizable} -- equilibrium amplitude is 0 there, so the "
              f"normalized curve and tau are undefined. Raw curves still plotted; excluded from "
              f"the collapse test below.")

    _report_seed_ranks(by_temp_seed_step, temp_amplitude, temperatures, aggregate)

    _report_autocorr_length(by_cell_ac, taus, grid_nx or size, grid_ny or size, aggregate)

    if inspect_seed is not None:
        _report_seed_breakdown(by_cell, temp_amplitude, inspect_seed, aggregate)

    collapse_temps, down_temps = _report_collapse_tests(
        curves, taus, taus_down, temperatures)

    if plot:
        _plot(curves, taus, taus_down, temperatures, temp_amplitude, collapse_temps, down_temps,
               ref_fraction, output_path, statistic, T0_value)
        print(f"\nSaved figure to {output_path}")
    return output_path


def _plot(curves, taus, taus_down, temperatures, temp_amplitude, collapse_temps, down_temps,
          ref_fraction, output_path: Path, statistic: str = "median",
          T0_value: float = 1.0) -> None:
    r"""
    Six panels, arranged so that panels sharing an x axis sit in the SAME
    COLUMN, one above the other, with their axes linked (both panels keep
    their own labelled x axis; the linking is for alignment, so a given x
    lands in the same place on the page in both):

        [0,0] raw vs t      [0,1] collapse: t/tau       [0,2] collapse: t/tau_down   [0,3] LAST vs T
        [1,0] norm. vs t    [1,1] tau vs -a(T)/b        [1,2] tau_down vs -a(T)/b    [1,3] MAX  vs T
              \_ x = t _/                                                                \_ x = T _/

    Columns 1 and 2 are deliberately the same pair of panels for the two
    characteristic times, so they can be read against each other: tau is
    the linear growth time and comes out size-independent, tau_down is
    the coarsening-completion time and scales with the box.

    Column 0 is the argument's first move (remove the amplitude effect)
    with the before/after vertically aligned, so a given t is the same
    place on the page in both. Column 2 is the same alignment for the
    late-time summaries. Column 1's two panels have genuinely different
    x quantities and are deliberately NOT linked.

    Markers are used only in column 1's lower panel and in column 2.
    Elsewhere there are ~24 curves of ~50 points each and markers merge
    into bands that obscure the lines; in those panels there is one point
    per temperature, so the markers are the data.
    """
    # 2x3. The two collapse panels (t/tau and t/tau_down) and the
    # LAST-value panel are gone; what the LAST-value panel carried is now an
    # overlay on the MAX-value panel, where the two can be read against each
    # other directly instead of across the figure.
    #
    # Five panels in six slots: [0, 2] is switched off rather than letting
    # something fill it, because the column pairing is the point -- column 0
    # is the same data raw and normalized, column 1 is the two timescales
    # against the same driving force -- and column 2 has only one member.
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    # Linked pairwise rather than with subplots(sharex="col"), which
    # would also link column 1 -- whose two panels have different x
    # quantities (tau against tau_down) and must not share a y scale.
    axes[0, 0].sharex(axes[1, 0])
    axes[0, 2].set_axis_off()

    colors = plt.cm.viridis(np.linspace(0, 1, max(len(temperatures), 2)))
    color_of = dict(zip(temperatures, colors))

    ax = axes[0, 0]
    for temp in temperatures:
        c = curves[temp]
        ax.plot(c["steps"], c["centre"], "-", color=color_of[temp],
                 label=f"T={_format_temp(temp)}")
        ax.fill_between(c["steps"], c["lo"], c["hi"],
                         color=color_of[temp], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("t (step)")
    ax.set_ylabel("stdev_phi")
    ax.set_title(f"raw: stdev_phi vs t, one curve per T\n({statistic} over noise then seeds, "
                  f"band = {'p25-p75' if statistic == 'median' else '1 sd'})")
    ax.legend(fontsize=7, ncol=2, loc="upper left")

    ax = axes[1, 0]
    for temp in temperatures:
        c = curves[temp]
        if c["normalized"] is None:
            continue
        ax.plot(c["steps"], 100.0 * c["normalized"], "-", color=color_of[temp],
                 label=f"T={_format_temp(temp)}")
    ax.axhline(100.0 * ref_fraction, color="k", ls=":", lw=1,
                label=f"tau threshold ({100 * ref_fraction:g}%)")
    ax.set_xscale("log")
    ax.set_xlabel("t (step)")
    ax.set_ylabel("stdev_phi / sqrt(-a(T)/b)  [%]")
    _normalized_ylim(ax)
    ax.set_title("amplitude removed: the height difference near T0\nis Landau scaling, not timing")
    ax.legend(fontsize=7, ncol=2, loc="upper left")

    ax = axes[0, 1]
    finite = [t for t in temperatures if np.isfinite(taus[t])]
    drive_xlim = None
    if finite:
        xs = [temp_amplitude[t] ** 2 for t in finite]
        ys = [taus[t] / 1000.0 for t in finite]  # thousands of steps; see the y label
        ax.plot(xs, ys, "o-", color="tab:red")
        for t, x, y in zip(finite, xs, ys):
            ax.annotate(_format_temp(t), (x, y), fontsize=6,
                         textcoords="offset points", xytext=(3, 3))
        ax.set_xscale("log")
        ax.set_yscale("log")
        _log_axis_ticks(ax.xaxis, min(xs), max(xs))
        _log_axis_ticks(ax.yaxis, min(ys), max(ys))
        # Remembered so [1,1] can share it: both panels plot against the same
        # x quantity, but tau_down is defined at fewer temperatures, so letting
        # it autoscale would silently show a narrower slice of the same axis
        # and invite a visual comparison of two different ranges.
        drive_xlim = ax.get_xlim()
    # x is -a(T)/b, i.e. the squared equilibrium amplitude, which is
    # LINEAR in the driving force a(T). Plotting tau against it directly
    # tests the "weaker driving force -> proportionally slower" reading:
    # a straight line of slope -1 here is exactly tau ~ 1/|T-T0|.
    ax.set_xlabel("-a(T)/b   (linear in driving force; labels are T)")
    ax.set_ylabel("tau (thousand steps)")
    # (T0 - T), not |T - T0|: below T0 the driving force a(T) = a0*(T-T0) is
    # negative and the ordered phase grows; at or above T0 there is no
    # instability at all and no tau to measure. The absolute value would imply
    # a symmetry that does not exist.
    coeff, exponent, n_fit = _fit_driving_force_law(temperatures, taus, T0_value)
    if np.isfinite(coeff):
        ax.set_title(f"does the driving force set the clock?\n"
                      f"fit: tau = {coeff / 1000:.3g}k / (T0-T)^{exponent:.3f}  ({n_fit} points)")
    else:
        ax.set_title("does the driving force set the clock?\n"
                      "tau ~ 1/(T0-T) would give exponent 1")

    ax = axes[1, 1]
    finite_down = [t for t in temperatures if np.isfinite(taus_down[t])]
    if finite_down:
        xs = [temp_amplitude[t] ** 2 for t in finite_down]
        ys = [taus_down[t] / 1e6 for t in finite_down]  # millions of steps
        ax.plot(xs, ys, "o-", color="tab:purple")
        for t, x, y in zip(finite_down, xs, ys):
            ax.annotate(_format_temp(t), (x, y), fontsize=6,
                         textcoords="offset points", xytext=(3, 3))
        ax.set_xscale("log")
        # LINEAR y, unlike [0,1]. tau spans ~90x across this axis and needs a
        # log scale to be readable at all; tau_down varies by well under a
        # factor of 2, and a log scale would stretch that near-constancy into
        # something that looks like a trend. The contrast between the two
        # panels IS the result -- one quantity follows the driving force, the
        # other essentially ignores it -- so the axes should not make them look
        # alike.
        _log_axis_ticks(ax.xaxis, min(xs), max(xs))
        ax.set_ylim(bottom=0.0)
    else:
        ax.text(0.5, 0.5, "tau_down undefined at every T", ha="center", va="center",
                 transform=ax.transAxes, fontsize=9)
    if drive_xlim is not None:
        ax.set_xscale("log")
        ax.set_xlim(drive_xlim)  # same x range as [0,1] -- see there
    # Same x as [0,1] so the two can be read directly against each other.
    # tau is a local growth time and lies on a clean -1 slope; tau_down
    # terminates when domains reach the BOX, so it need not, and its
    # VERTICAL position is the size-dependent quantity.
    ax.set_xlabel("-a(T)/b   (linear in driving force; labels are T)")
    ax.set_ylabel("tau_down (million steps)")
    ax.set_title("coarsening completion vs driving force\n(this one scales with the box)")

    # --- column 2: where the collapse breaks ----------------------------
    # ONE panel now, not two. The LAST-value and MAX-value panels plotted the
    # same three series against T and differed only in which point of each
    # curve they reduced to -- so reading one against the other meant looking
    # across the figure and holding a shape in mind. LAST is now an OVERLAY on
    # MAX, hollow and unjoined, so the gap between a curve's peak and where it
    # ends up is a vertical distance on one axis: the decay after the peak IS
    # the late-time coarsening/finite-size effect the pair existed to show.
    #
    # Twin axes because raw stdev_phi and the normalized curve are
    # different quantities: putting them on one axis made any horizontal
    # reference line ambiguous (it belongs to the normalized quantity
    # only). Left/right axes are colour-matched to their own series, the
    # same convention the project's dz0dt panels already use.
    #
    # Unjoined markers for the overlay, deliberately: a line would invite
    # reading a trend ACROSS temperatures, but each hollow point's meaning is
    # its distance from the filled point directly above it, at the same T.
    excluded = [t for t in temperatures if t not in set(collapse_temps)]
    ax = axes[1, 2]
    ax_norm = ax.twinx()

    def _raw_series(which: int):
        xs, ys = [], []
        for t in temperatures:
            value = _last_and_max(curves[t]["centre"])[which]
            if np.isfinite(value):
                xs.append(t)
                ys.append(value)
        return xs, ys

    max_x, max_y = _raw_series(1)
    ax.plot(max_x, max_y, "-", color="tab:blue", marker="o", ms=3,
             label="raw stdev_phi (max)")
    last_x, last_y = _raw_series(0)
    ax.plot(last_x, last_y, linestyle="none", marker="o", ms=6,
             markerfacecolor="none", markeredgecolor="tab:blue",
             label="raw stdev_phi (last)")

    norm_x, norm_y = [], []
    for t in temperatures:
        if curves[t]["normalized"] is None:
            continue
        value = _last_and_max(curves[t]["normalized"])[1]
        if np.isfinite(value):
            norm_x.append(t)
            norm_y.append(value)
    ax_norm.plot(norm_x, [100.0 * y for y in norm_y], "-", color="tab:orange",
                  marker="o", ms=3, label="normalized (max)")
    # Hollow overlay: no tau, so absent from the collapse test.
    miss_x = [t for t in norm_x if t in set(excluded)]
    miss_y = [y for t, y in zip(norm_x, norm_y) if t in set(excluded)]
    if miss_x:
        ax_norm.plot(miss_x, [100.0 * y for y in miss_y], linestyle="none", marker="o", ms=8,
                      markerfacecolor="none", markeredgecolor="tab:red",
                      label="no tau: excluded from collapse test")
    if norm_y and min(norm_y) < ref_fraction:
        # Against the NORMALIZED axis: tau exists iff the normalized curve
        # gets at least this high, so this line is exactly the boundary that
        # decides the hollow red markers above.
        #
        # Drawn ONLY when some temperature actually falls below it. When every
        # curve clears the threshold -- the usual case on a large box, where
        # maxima sit at 90-98% -- the line adds no information and, sitting at
        # 50%, stretches the axis over the empty half and flattens the real
        # variation into the top sliver.
        ax_norm.axhline(100.0 * ref_fraction, color="k", ls=":", lw=1,
                         label=f"tau threshold ({100 * ref_fraction:g}%)")

    # Anchored at 0 so the raw curves are read as magnitudes rather than as
    # shapes: autoscaling a monotone decay makes a 15% fall look like a
    # collapse. The normalized twin keeps its own scaling (0-100%).
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("raw stdev_phi", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax_norm.set_ylabel("stdev_phi / sqrt(-a(T)/b)  [%]", color="tab:orange")
    _normalized_ylim(ax_norm)  # twin axis only: the left axis is raw stdev_phi, not normalized
    ax_norm.tick_params(axis="y", labelcolor="tab:orange")
    ax.set_xlabel("T")
    ax.set_title("MAX value of each curve vs T\n(hollow blue = LAST value: the late-time decay)")
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax_norm.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, fontsize=7, loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-path", type=Path, default=_PYTHON_ROOT.parent / "datasets")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--min-step", type=int, default=1,
                         help="default 1, not 0: every axis here is logarithmic, so step 0 has no "
                              "position on one and is excluded regardless -- the default says so "
                              "rather than leaving it to a downstream filter. Unlike "
                              "check_stdev_phi_temperature, the early steps are what tau is "
                              "estimated from, so raising this further discards the signal this "
                              "diagnostic is built on")
    parser.add_argument("--ref-fraction", type=float, default=0.5)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                         help="--no-plot skips rendering the figure, which is ~99%% of the "
                              "runtime; the console tables are unaffected")
    parser.add_argument("--statistic", choices=sorted(_AGGREGATORS), default="median",
                         help="how runs are combined at both levels (default median). Only "
                              "'mean' carries the late-time trapping probability; see "
                              "_AGGREGATORS")
    parser.add_argument("--inspect-seed", type=int, default=None,
                         help="break this seed down into its individual runs, to decide whether "
                              "an anomaly is the seed (drop it) or a few runs (re-run them)")
    parser.add_argument("--exclude-seeds", type=int, nargs="*", default=None,
                         help="seed values to drop entirely, across every temperature")
    parser.add_argument("--min-run-fraction", type=float, default=0.9,
                         help="truncate curves at the largest step this fraction of runs reach "
                              "(default 0.9), removing the late-time discontinuities caused by "
                              "runs of differing length; 0 disables")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    check_stdev_phi_time(base_path=args.base_path, size=args.size, min_step=args.min_step,
                          ref_fraction=args.ref_fraction,
                          min_run_fraction=args.min_run_fraction,
                          exclude_seeds=set(args.exclude_seeds) if args.exclude_seeds else None,
                          inspect_seed=args.inspect_seed, statistic=args.statistic,
                          plot=args.plot,
                          output_path=args.output)


if __name__ == "__main__":
    main()
