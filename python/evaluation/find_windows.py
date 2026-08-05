"""Find windows in a specific (dt, theta) corner and print them as
--fixed-windows arguments, ready to paste into check_rollout /
compare_rollout_training.

WHY THIS EXISTS. Every spike attribution across this project names one corner:
`dt_max=125, mean theta[0]~=-0.28`. It has now appeared in stage 3a's loss
spikes, stage 3b's loss AND gradient spikes, stage 5's gradient spikes, and in
the off-diagonal rollout figures, where exactly those windows degraded 2.4-2.5x
under z1 propagation. Five independent sightings, two detection mechanisms.

Turning that into an actual investigation means running a diagnostic on those
specific windows -- and --fixed-windows takes explicit
`<run_dir>:<step1>:<step2>:<step3>` strings, which nobody can write by hand:
`dt` is `(step_b - step_a) * metadata.dt` and `theta[0]` is
`temperature - T0`, both per-run values only the metadata knows.

So: state the corner in physical terms, get the strings.

    python -m evaluation.find_windows --base ../datasets --size 128 \\
        --dt 125 --theta0 -0.28 --min-step 2000 --min-stdev-phi 0.01

then paste the printed block into

    python -m evaluation.check_rollout \\
        --lds-checkpoint checkpoints/stage3b/128x128-stage3b.pt \\
        --no-z1-resync --fixed-windows <paste here>

If those windows blow up with FROZEN weights, the instability is a property of
f_theta + z1 propagation on that corner, and the fix belongs upstream --
excluding or reweighting them -- rather than in the optimizer.
"""
import argparse
from pathlib import Path

import utils.load_datasets as load
from training.datasets import build_good_steps, complete_run_dirs


def find_windows(base: str | Path, size: int, dt: float, theta0: float,
                  n_steps: int = 2, dt_tol: float = 0.05, theta_tol: float = 0.02,
                  min_step: int = 0, min_stdev_phi: float | None = None,
                  min_passing_steps: int | None = None, limit: int = 6,
                  ) -> list[tuple[str, float, float]]:
    """Return [(window_string, actual_dt, actual_theta0), ...].

    Tolerances are RELATIVE for dt and ABSOLUTE for theta0, matching how the
    two are reported: dt as a magnitude ("dt_max=125") and theta as an offset
    from T0 ("mean theta[0]=-0.2817").

    n_steps is the number of TRANSITIONS, so a window string carries
    n_steps + 1 step numbers -- 3 for the n_rollout_steps=2 checkpoints, which
    is what check_rollout expects from a stage-3b/4/5 model.
    """
    run_dirs = complete_run_dirs(base, size, size)
    good = build_good_steps(run_dirs, min_step=min_step, min_stdev_phi=min_stdev_phi,
                             min_passing_steps=min_passing_steps)
    out: list[tuple[str, float, float]] = []
    for run_dir, steps in good.items():
        metadata = load.read_metadata(Path(run_dir) / "metadata.txt")
        this_theta0 = float(metadata.temperature - metadata.T0)
        if abs(this_theta0 - theta0) > theta_tol:
            continue
        # every consecutive run of n_steps+1 kept steps whose EVERY transition
        # is within tolerance of the target dt -- a window is only in the
        # corner if all of its hops are, which is what "dt_max=125" reported
        for i in range(len(steps) - n_steps):
            hop = steps[i:i + n_steps + 1]
            dts = [(hop[j + 1] - hop[j]) * metadata.dt for j in range(n_steps)]
            if all(abs(d - dt) <= dt_tol * dt for d in dts):
                out.append((":".join([str(run_dir)] + [str(s) for s in hop]),
                            sum(dts) / len(dts), this_theta0))
                break          # one window per run, for variety across runs
        if len(out) >= limit:
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="sweep root, e.g. ../datasets")
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--dt", type=float, required=True,
                    help="target dt PER TRANSITION, in simulation time units "
                         "(dt = (step_b - step_a) * metadata.dt)")
    p.add_argument("--theta0", type=float, required=True,
                    help="target theta[0] = temperature - T0")
    p.add_argument("--n-steps", type=int, default=2,
                    help="transitions per window; 2 matches a stage-3b/4/5 checkpoint")
    p.add_argument("--dt-tol", type=float, default=0.05, help="relative tolerance on dt")
    p.add_argument("--theta-tol", type=float, default=0.02, help="absolute tolerance on theta[0]")
    p.add_argument("--min-step", type=int, default=0)
    p.add_argument("--min-stdev-phi", type=float, default=None)
    p.add_argument("--min-passing-steps", type=int, default=None)
    p.add_argument("--limit", type=int, default=6,
                    help="how many windows to print (check_rollout draws one row each)")
    args = p.parse_args()

    found = find_windows(args.base, args.size, args.dt, args.theta0,
                          n_steps=args.n_steps, dt_tol=args.dt_tol,
                          theta_tol=args.theta_tol, min_step=args.min_step,
                          min_stdev_phi=args.min_stdev_phi,
                          min_passing_steps=args.min_passing_steps, limit=args.limit)
    if not found:
        print(f"No window found with dt~={args.dt} (+-{100*args.dt_tol:.0f}%) and "
              f"theta[0]~={args.theta0} (+-{args.theta_tol}).\n"
              f"Widen --dt-tol/--theta-tol, or check that this corner survives "
              f"your --min-step/--min-stdev-phi filters at all.")
        return

    print(f"\n{len(found)} window(s) at dt~={args.dt}, theta[0]~={args.theta0}:\n")
    for w, actual_dt, actual_theta in found:
        print(f"  dt={actual_dt:.1f}  theta[0]={actual_theta:+.4f}  {Path(w.split(':')[0]).name}")
    print("\nPaste after --fixed-windows :\n")
    print("  " + " ".join(w for w, _, _ in found) + "\n")


if __name__ == "__main__":
    main()
