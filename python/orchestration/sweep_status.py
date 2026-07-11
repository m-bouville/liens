"""
Reports simulation-sweep completeness (COMPLETE/INCOMPLETE/missing run
directories per grid size), independent of config.txt -- see
load.read_sweep_metadata. Extracted from main.py during its split into
orchestration/.
"""
from pathlib import Path

from utils import load_datasets as load


def check_sweep_status(base_path: Path) -> None:
    """
    Scan every <nx>x<ny> subdirectory under base_path -- each one has its
    own metadata.txt (see load.read_sweep_metadata), so this no longer
    depends on config.txt describing any particular sweep. Reports
    COMPLETE/INCOMPLETE/missing run directories per size found.
    """
    if not base_path.exists():
        print(f"{base_path} does not exist")
        return
    size_dirs = sorted(d for d in base_path.iterdir() if d.is_dir() and (d / "metadata.txt").exists())
    if not size_dirs:
        print(f"No <nx>x<ny> subdirectories with a metadata.txt found under {base_path}")
        return

    for size_dir in size_dirs:
        metadata = load.read_sweep_metadata(size_dir / "metadata.txt")
        dirs = [size_dir / subdir for subdir in metadata.subdirs]

        print(f"\n=== {size_dir.name} ===")
        n_complete = n_incomplete = n_missing = 0
        for d in dirs:
            if not d.exists():
                n_missing += 1
                continue
            if load.is_complete(d):
                n_complete += 1
                print(f"COMPLETE    {d}")
                run_metadata = load.read_metadata(d / "metadata.txt")
                check = load.check_snapshots_saved(d, run_metadata)
                if check["missing"] or check["bad_size"]:
                    print(f"            ! {len(check['missing'])} missing, "
                          f"{len(check['bad_size'])} bad size")
            else:
                n_incomplete += 1
                print(f"INCOMPLETE  {d}")

        print(f"{len(dirs)} runs listed in metadata.txt -> "
              f"{n_complete} complete, {n_incomplete} incomplete, {n_missing} missing (ignored)")
