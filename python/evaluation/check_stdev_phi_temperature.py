"""
Checkpoint-free diagnostic: does the natural (equilibrium) amplitude of
phi shrink toward zero as T -> T0 -- per this project's own Landau
free-energy setup, stable phases at +-sqrt(-a(T)/b), a(T)=a0*(T-T0) --
and if so, does a single GLOBAL min_stdev_phi threshold end up behaving
very differently near T0 than it does elsewhere, rather than
consistently across the whole sweep?

Motivation: evaluation/check_deriv_temperature.py and
check_parameter_dependence.py both found z1's own error changing sign
and magnitude specifically around T>=0.9 (T0=1 in this project's own
sweep) -- and neither the encoder nor ANY stage-2 loss term is ever
given temperature as an input (confirmed directly: Encoder.forward(x)
and Autoencoder.forward(x) take no theta argument at all; train_stage2
unpacks theta from its batch but never uses it again). If the natural
scale of genuine structure shrinks toward the SAME noise floor a fixed
min_stdev_phi threshold was calibrated against elsewhere in the sweep,
that's a direct, physically-motivated candidate explanation for why the
data (and therefore z1's training signal) could look qualitatively
different near T0 -- independent of anything the encoder itself does
wrong.

Reads statistics.csv directly across an entire sweep -- no model,
checkpoint, or GPU needed at all.

Usage (run as a module from python/, since imports rely on that root
being on sys.path):
    python -m evaluation.check_stdev_phi_temperature --base-path ../datasets --size 64 \
        --min-step 3000 --candidate-thresholds 0.05 0.1 0.2
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import load_datasets as load

_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/


def check_stdev_phi_temperature(
    base_path: Path, size: int, min_step: int = 0,
    candidate_thresholds: list[float] | None = None,
    min_passing_steps: int | None = None,
    n_temp_bins: int = 12, output_path: Path | None = None,
) -> Path:
    """
    Collects (temperature, stdev_phi, theoretical_equilibrium_amplitude)
    for every kept (run, step) in the sweep, then:
      - checks whether stdev_phi tracks the theoretical sqrt(-a(T)/b)
        curve (per-run, using THAT run's own a0/b/T0 -- not hardcoded,
        in case a future sweep ever uses different Landau parameters)
      - for each candidate_thresholds value, reports/plots what FRACTION
        of steps a min_stdev_phi filter at that value would exclude, as
        a function of temperature -- directly visualizing whether
        exclusion becomes disproportionately more (or less) aggressive
        near T0, rather than staying roughly uniform across the sweep.
      - for each candidate_thresholds value, ALSO reports/plots the
        RUN-level analog: what fraction of entire RUNS (not just
        individual steps) would have fewer than min_passing_steps steps
        clearing that threshold -- i.e. what build_good_steps' own
        min_passing_steps parameter would drop entirely (see its
        docstring in training/datasets.py for why whole-run exclusion,
        not just per-step, matters). If min_passing_steps is None, the
        median passing-step-COUNT per run is still reported/plotted per
        temperature bin (useful for choosing a value), just without a
        specific exclusion fraction computed against it.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.05, 0.1, 0.2]
    if min_step == 0:
        print("  WARNING: min_step=0 -- every run's EARLY steps (still near the near-uniform "
              "initial condition, before phase separation has had any chance to develop AT ANY "
              "TEMPERATURE) are included. That confounds exactly the question this diagnostic asks: "
              "a near-zero stdev_phi near T0 could mean either 'hasn't had time to separate yet' "
              "(true early on regardless of temperature, and not itself informative) or 'still hasn't "
              "separated even at times the actual training pipeline uses' (critical slowing down -- a "
              "real, temperature-specific effect). Rerun with --min-step set to whatever the actual "
              "training pipeline uses (e.g. 3000, matching this project's own stage-3 runs) to "
              "distinguish these -- results below are NOT reliable evidence for either explanation "
              "until that's done.")
    if output_path is None:
        output_path = _PYTHON_ROOT.parent / "output" / "stdev_phi_temperature.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = load.enumerate_run_dirs_from_metadata(base_path, size, size)

    temperatures, stdevs, theoretical_amplitudes = [], [], []
    # Per-RUN (not per-step) bookkeeping -- one entry per run, needed for
    # the run-level exclusion analysis below, which build_good_steps'
    # own min_passing_steps applies at the RUN granularity, not the step
    # granularity the arrays above are built at.
    run_temperatures = []
    run_passing_counts = {thr: [] for thr in candidate_thresholds}  # thr -> [count per run]
    n_runs_used = 0
    n_runs_skipped_incomplete = 0
    n_runs_skipped_no_stats = 0

    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.txt"
        if not metadata_path.exists():
            continue
        metadata = load.read_metadata(metadata_path)
        if not metadata.is_complete:
            n_runs_skipped_incomplete += 1
            continue
        stats_path = run_dir / "statistics.csv"
        if not stats_path.exists():
            n_runs_skipped_no_stats += 1
            continue
        stats_df = load.read_statistics_csv(stats_path)
        if "stdev_phi" not in stats_df.columns:
            n_runs_skipped_no_stats += 1
            continue

        # a(T) = a0*(T-T0); stable (subcritical, ordered) phases exist
        # only when a(T) < 0, i.e. T < T0 -- clipped at 0 at/above T0
        # (phi=0 is the only stable point there, per the free energy's
        # own construction, not an approximation). THIS run's own
        # a0/b/T0 (from its own metadata.txt), not hardcoded to
        # a0=b=T0=1 -- correct for the current sweep either way, but
        # doesn't silently break if a different sweep ever uses
        # different Landau parameters.
        a_T = metadata.a0 * (metadata.temperature - metadata.T0)
        equilibrium_amplitude = float(np.sqrt(max(-a_T / metadata.b, 0.0)))

        n_runs_used += 1
        run_stdevs_kept = []
        for step in metadata.save_steps:
            if step < min_step or step not in stats_df.index:
                continue
            stdev = stats_df.loc[step, "stdev_phi"]
            if isinstance(stdev, (int, float)) and not np.isnan(stdev):
                temperatures.append(metadata.temperature)
                stdevs.append(float(stdev))
                theoretical_amplitudes.append(equilibrium_amplitude)
                run_stdevs_kept.append(float(stdev))

        run_temperatures.append(metadata.temperature)
        run_stdevs_kept = np.array(run_stdevs_kept)
        for thr in candidate_thresholds:
            run_passing_counts[thr].append(int((run_stdevs_kept >= thr).sum()))

    if not temperatures:
        raise ValueError(f"No (step, temperature, stdev_phi) samples found under {base_path} "
                          f"for size={size} -- check the path and that statistics.csv files exist")

    temperatures = np.array(temperatures)
    stdevs = np.array(stdevs)
    theoretical_amplitudes = np.array(theoretical_amplitudes)
    run_temperatures = np.array(run_temperatures)
    for thr in candidate_thresholds:
        run_passing_counts[thr] = np.array(run_passing_counts[thr])

    print(f"{n_runs_used} runs used ({n_runs_skipped_incomplete} incomplete, "
          f"{n_runs_skipped_no_stats} missing statistics.csv/stdev_phi, skipped), "
          f"{len(temperatures)} (step, temperature) samples")

    corr = np.corrcoef(theoretical_amplitudes, stdevs)[0, 1]
    print(f"\ncorr(theoretical equilibrium amplitude, actual stdev_phi) = {corr:.3f} "
          f"(close to 1 -> stdev_phi tracks the Landau-predicted critical scaling closely)")

    # Binned view: median stdev_phi (and the SAME percentile of
    # theoretical amplitude, for a fair overlay) per temperature bin --
    # a scatter alone is dominated by within-bin spread (different
    # steps/noise/domain configurations at the SAME temperature), which
    # would obscure the across-temperature TREND this is actually about.
    bin_edges = np.linspace(temperatures.min(), temperatures.max(), n_temp_bins + 1)
    bin_centers, median_stdev, theoretical_at_bin = [], [], []
    exclusion_fractions = {thr: [] for thr in candidate_thresholds}
    for i in range(n_temp_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (temperatures >= lo) & (temperatures < hi if i < n_temp_bins - 1 else temperatures <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        median_stdev.append(np.median(stdevs[mask]))
        theoretical_at_bin.append(np.median(theoretical_amplitudes[mask]))
        for thr in candidate_thresholds:
            exclusion_fractions[thr].append((stdevs[mask] < thr).mean())

    bin_centers = np.array(bin_centers)
    median_stdev = np.array(median_stdev)
    theoretical_at_bin = np.array(theoretical_at_bin)

    print(f"\nmedian stdev_phi vs temperature (binned, n_temp_bins={n_temp_bins}):")
    for t, s, th in zip(bin_centers, median_stdev, theoretical_at_bin):
        print(f"  T~{t:.3f}:  median stdev_phi={s:.4f}   theoretical amplitude={th:.4f}   "
              f"ratio={s / th if th > 1e-9 else float('nan'):.3f}")

    print(f"\nfraction of steps EXCLUDED by min_stdev_phi, per candidate threshold, per temperature bin:")
    for thr in candidate_thresholds:
        fracs = exclusion_fractions[thr]
        print(f"  threshold={thr}:")
        for t, f in zip(bin_centers, fracs):
            print(f"    T~{t:.3f}:  {f:.1%} excluded")

    # RUN-level view: build_good_steps' own min_passing_steps drops an
    # entire run, not individual steps -- the step-level exclusion above
    # doesn't show this at all (a run could lose 95% of its steps and
    # still show up as "some steps kept" there). Binned by each RUN's
    # own temperature (one value per run, not one per step).
    run_bin_edges = np.linspace(run_temperatures.min(), run_temperatures.max(), n_temp_bins + 1)
    run_bin_centers = []
    median_passing_count = {thr: [] for thr in candidate_thresholds}
    run_exclusion_fractions = {thr: [] for thr in candidate_thresholds}
    for i in range(n_temp_bins):
        lo, hi = run_bin_edges[i], run_bin_edges[i + 1]
        mask = (run_temperatures >= lo) & (run_temperatures < hi if i < n_temp_bins - 1
                                            else run_temperatures <= hi)
        if mask.sum() == 0:
            continue
        run_bin_centers.append((lo + hi) / 2)
        for thr in candidate_thresholds:
            counts = run_passing_counts[thr][mask]
            median_passing_count[thr].append(np.median(counts))
            if min_passing_steps is not None:
                run_exclusion_fractions[thr].append((counts < min_passing_steps).mean())

    run_bin_centers = np.array(run_bin_centers)

    print(f"\nmedian passing-step COUNT per run (i.e. how many steps clear the threshold, "
          f"within EACH run), per candidate threshold, per temperature bin:")
    for thr in candidate_thresholds:
        print(f"  threshold={thr}:")
        for t, c in zip(run_bin_centers, median_passing_count[thr]):
            print(f"    T~{t:.3f}:  median {c:.1f} passing steps per run")

    if min_passing_steps is not None:
        print(f"\nfraction of ENTIRE RUNS dropped by min_passing_steps={min_passing_steps}, "
              f"per candidate threshold, per temperature bin:")
        for thr in candidate_thresholds:
            print(f"  threshold={thr}:")
            for t, f in zip(run_bin_centers, run_exclusion_fractions[thr]):
                print(f"    T~{t:.3f}:  {f:.1%} of runs dropped entirely")
    else:
        print(f"\nmin_passing_steps not given -- run-exclusion fractions not computed "
              f"(median passing-count-per-run above is still useful for choosing a value).")

    n_panels = 4 if min_passing_steps is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    axes[0].scatter(temperatures, stdevs, s=6, alpha=0.15, color="tab:blue", label="actual stdev_phi")
    order = np.argsort(temperatures)
    axes[0].plot(temperatures[order], theoretical_amplitudes[order], "--", color="tab:red",
                 label="theoretical sqrt(-a(T)/b)")
    axes[0].set_xlabel("temperature")
    axes[0].set_ylabel("stdev_phi")
    axes[0].set_title(f"Raw scatter\ncorr(theoretical, actual) = {corr:.3f}")
    axes[0].legend(fontsize=8)

    axes[1].plot(bin_centers, median_stdev, "o-", color="tab:blue", label="median stdev_phi")
    axes[1].plot(bin_centers, theoretical_at_bin, "--", color="tab:red", label="theoretical amplitude")
    for thr in candidate_thresholds:
        axes[1].axhline(thr, linestyle=":", alpha=0.5, label=f"threshold={thr}")
    axes[1].set_xlabel("temperature (binned)")
    axes[1].set_ylabel("stdev_phi")
    axes[1].set_title("Binned median vs. theoretical curve")
    axes[1].legend(fontsize=7)

    for thr in candidate_thresholds:
        axes[2].plot(bin_centers, exclusion_fractions[thr], "o-", label=f"threshold={thr}")
    axes[2].set_xlabel("temperature (binned)")
    axes[2].set_ylabel("fraction of STEPS excluded")
    axes[2].set_title("min_stdev_phi step-level exclusion\n"
                       "(flat = threshold behaves consistently; rising near T0 = it doesn't)")
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 1)

    if min_passing_steps is not None:
        for thr in candidate_thresholds:
            axes[3].plot(run_bin_centers, run_exclusion_fractions[thr], "o-", label=f"threshold={thr}")
        axes[3].set_xlabel("temperature (binned)")
        axes[3].set_ylabel("fraction of RUNS dropped entirely")
        axes[3].set_title(f"min_passing_steps={min_passing_steps} run-level exclusion\n"
                           "(what build_good_steps actually drops, not just individual steps)")
        axes[3].legend(fontsize=8)
        axes[3].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved figure to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, required=True,
                         help="sweep root (containing <size>x<size>/ subdirectories)")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--min-step", type=int, default=0,
                         help="EXCLUDE early transient steps before this -- see this module's own "
                              "docstring/warning for why min_step=0 (the default) confounds this "
                              "diagnostic's whole interpretation. Match whatever the real training "
                              "pipeline actually uses (e.g. 3000).")
    parser.add_argument("--candidate-thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--min-passing-steps", type=int, default=None,
                         help="if given, additionally reports/plots what fraction of ENTIRE RUNS "
                              "(not just individual steps) would be dropped by build_good_steps' "
                              "own min_passing_steps at this value, per candidate threshold, per "
                              "temperature bin. Omit to just see the median passing-step-count per "
                              "run instead (useful for choosing a value in the first place).")
    parser.add_argument("--n-temp-bins", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    check_stdev_phi_temperature(
        base_path=args.base_path, size=args.size, min_step=args.min_step,
        candidate_thresholds=args.candidate_thresholds, min_passing_steps=args.min_passing_steps,
        n_temp_bins=args.n_temp_bins, output_path=args.output,
    )


if __name__ == "__main__":
    main()
