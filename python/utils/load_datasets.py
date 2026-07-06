"""
I/O utilities for reading C++ phase-field solver output: binary snapshots,
sweep config, and per-run metadata/directory naming.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Snapshot files (one binary file per saved timestep, e.g. "t0100000")
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"^t(\d+)$")


def read_phi_half(path: str | Path, nx: int, ny: int) -> np.ndarray:
    """
    Read a phase-field snapshot saved by writer::save_phi_half.

    Format: raw IEEE 754 binary16, no header, little-endian, nx*ny values,
    row-major with __ny rows / __nx columns (confirmed via Field::save_as_png's
    cv::Mat(__ny, __nx, ...) construction -- x is the fastest-varying index).

    Returns
    -------
    np.ndarray, shape (ny, nx), dtype float32
    """
    path = Path(path)
    data = np.fromfile(path, dtype="<f2")

    expected = nx * ny
    if data.size != expected:
        raise ValueError(
            f"{path}: expected {expected} values ({nx}x{ny}), got {data.size}"
        )

    return data.reshape(ny, nx).astype(np.float32)


def parse_step(filename: str | Path) -> int:
    """Extract the integer step number from a snapshot filename like 't0100000'."""
    name = Path(filename).name
    m = _STEP_RE.match(name)
    if not m:
        raise ValueError(f"Filename '{name}' does not match expected pattern 't<digits>'")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Key-value text files (config.txt, metadata.txt)
# ---------------------------------------------------------------------------

def _parse_kv_file(path: str | Path) -> dict[str, str]:
    """
    Parse a `key = value  # comment` text file (shared by config.txt and
    metadata.txt) into a dict of raw string values. Caller does the typing.
    """
    kv = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            kv[key.strip()] = val.strip()
    return kv


def _parse_list(raw: str, cast=float) -> list:
    return [cast(x) for x in raw.split(",") if x.strip() != ""]


@dataclass
class SweepConfig:
    nx: int
    ny: int
    dt: float
    steps: int
    max_threads: int
    a0: float
    b: float
    T0: float
    kappa: float
    mobility: float
    temperatures: list[float]
    noises: list[float]
    seeds: list[int]
    phi0: float
    save_steps: list[int]


def read_config(path: str | Path) -> SweepConfig:
    """Parse the sweep config.txt written before a batch of simulations.
    Simulation-sweep parameters ONLY -- min_step/min_stdev_phi/
    stats_weight (ML training parameters) are no longer read from here;
    they belong in a stage-parameters file instead (see main.py)."""
    kv = _parse_kv_file(path)
    return SweepConfig(
        nx=int(kv["Nx"]),
        ny=int(kv["Ny"]),
        dt=float(kv["dt"]),
        steps=int(kv["steps"]),
        max_threads=int(kv["max_threads"]),
        a0=float(kv["a0"]),
        b=float(kv["b"]),
        T0=float(kv["T0"]),
        kappa=float(kv["kappa"]),
        mobility=float(kv["M"]),
        temperatures=_parse_list(kv["temperatures"], float),
        noises=_parse_list(kv["noises"], float),
        seeds=_parse_list(kv["seeds"], int),
        phi0=float(kv["phi0"]),
        save_steps=_parse_list(kv["save"], int),
    )


# ---------------------------------------------------------------------------
# Run directory naming (mirrors writer::make_dir_name)
# ---------------------------------------------------------------------------

def round_half_away_from_zero(x: float) -> int:
    """
    Replicate C++ std::round, which rounds half-away-from-zero.
    Python's built-in round() uses banker's rounding and can disagree
    on exact .5 ties (e.g. 22.5 -> 22 in Python, 23 in C++).
    """
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def make_dir_name(base: str | Path, nx: int, ny: int,
                   T: float, noise: float, seed: int) -> Path:
    """
    Reproduce writer::make_dir_name's directory naming convention:
    {base}/{nx}x{ny}/T{Ti}_n{ni:03d}_s{seed}, with Ti = round(T*1000)
    and ni = round(noise*1000).
    """
    Ti = round_half_away_from_zero(T * 1000)
    ni = round_half_away_from_zero(noise * 1000)
    return Path(base) / f"{nx}x{ny}" / f"T{Ti}_n{ni:03d}_s{seed}"


def enumerate_run_dirs(config: SweepConfig, base: str | Path = "../datasets") -> list[Path]:
    """
    All directory names implied by the sweep in config.txt (Cartesian
    product of temperatures x noises x seeds). Pure string generation --
    does not touch the filesystem, so it works even before any run exists.
    """
    return [
        make_dir_name(base, config.nx, config.ny, T, noise, seed)
        for T in config.temperatures
        for noise in config.noises
        for seed in config.seeds
    ]


def is_complete(run_dir: str | Path) -> bool:
    """A run is complete iff a COMPLETE marker file exists in its directory."""
    return (Path(run_dir) / "COMPLETE").exists()


@dataclass
class SweepMetadata:
    """
    Parsed from datasets/<nx>x<ny>/metadata.txt -- NOT the same as
    RunMetadata below, which is per-INDIVIDUAL-run. This describes the
    whole sweep for one grid size, co-located with the actual dataset
    directory rather than a separate, potentially-stale or
    describing-a-different-sweep config.txt. subdirs lists the actual
    run directory names directly -- no need to recompute the
    temperature x noise x seed cross-product or know the naming
    convention at all.
    """
    nx: int
    ny: int
    temperatures: list[float]
    noises: list[float]
    seeds: list[int]
    subdirs: list[str]


def read_sweep_metadata(path: str | Path) -> SweepMetadata:
    """
    Parses datasets/<nx>x<ny>/metadata.txt. Format: ordinary key=value
    lines for Nx/Ny/temperatures/noises/seeds, then a 'subdirs =' line
    (with nothing after the '=') followed by one subdirectory name per
    line for the rest of the file -- a different shape from the plain
    key=value files _parse_kv_file handles, so this needs its own parser.
    """
    kv: dict[str, str] = {}
    subdirs: list[str] = []
    in_subdirs = False
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if in_subdirs:
            subdirs.append(line)
            continue
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key == "subdirs" and value == "":
                in_subdirs = True
                continue
            kv[key] = value
    return SweepMetadata(
        nx=int(kv["Nx"]), ny=int(kv["Ny"]),
        temperatures=_parse_list(kv["temperatures"], float),
        noises=_parse_list(kv["noises"], float),
        seeds=_parse_list(kv["seeds"], int),
        subdirs=subdirs,
    )


def enumerate_run_dirs_from_metadata(base: str | Path, nx: int, ny: int) -> list[Path]:
    """
    All directory names for one grid size, read directly from that
    size's own datasets/<nx>x<ny>/metadata.txt -- replaces
    enumerate_run_dirs()'s config.txt-based cross-product entirely.
    Pure string construction from the metadata's subdirs list -- does
    not touch individual run directories, so it works even before any
    run in the list actually exists.
    """
    metadata = read_sweep_metadata(Path(base) / f"{nx}x{ny}" / "metadata.txt")
    return [Path(base) / f"{nx}x{ny}" / subdir for subdir in metadata.subdirs]


# ---------------------------------------------------------------------------
# Per-run metadata (metadata.txt)
# ---------------------------------------------------------------------------

@dataclass
class RunMetadata:
    directory:    str
    code_version: str
    status:       str
    nx:           int
    ny:           int
    dt:           float
    steps:        int
    save_steps:   list[int]
    a0:           float
    b:            float
    T0:           float
    temperature:  float
    kappa:        float
    mobility:     float
    phi0:         float
    noise:        float
    seed:         int
    equation:     str
    solver:       str

    @property
    def is_complete(self) -> bool:
        return self.status.strip().lower() == "complete"


def snapshot_filename(step: int, width: int = 7) -> str:
    """Filename for a saved snapshot at a given step, e.g. step=100000 -> 't0100000'."""
    return f"t{step:0{width}d}"


def check_snapshots_saved(run_dir: str | Path, metadata: RunMetadata) -> dict[str, list[int]]:
    """
    Verify that every step in metadata.save_steps has a corresponding
    snapshot file in run_dir, and that files present have the expected
    size (nx*ny*2 bytes for float16) -- catching truncated writes from
    an interrupted run, not just missing files.

    Returns
    -------
    dict with keys "missing" and "bad_size", each a list of step numbers.
    Both empty means the run's snapshots are all present and well-formed.
    """
    run_dir = Path(run_dir)
    expected_bytes = metadata.nx * metadata.ny * 2  # float16 = 2 bytes

    missing = []
    bad_size = []

    for step in metadata.save_steps:
        f = run_dir / snapshot_filename(step)
        if not f.exists():
            missing.append(step)
            continue
        if f.stat().st_size != expected_bytes:
            bad_size.append(step)

    return {"missing": missing, "bad_size": bad_size}


def read_metadata(path: str | Path) -> RunMetadata:
    """
    Parse a per-run metadata.txt. Unlike config.txt's comma-separated
    `save`, metadata.txt's `save_steps` is whitespace-separated -- the
    C++ side writes actual save times for this specific run, which can
    be shorter than the config.txt sweep list (e.g. if a run stopped early).
    """
    kv = _parse_kv_file(path)
    return RunMetadata(
        directory    = kv["directory"],
        code_version = kv["code version"],
        status       = kv["status"],
        nx           = int(kv["Nx"]),
        ny           = int(kv["Ny"]),
        dt           = float(kv["dt"]),
        steps        = int(kv["steps"]),
        save_steps   = [int(x) for x in kv["save_steps"].split()],
        a0           = float(kv["a0"]),
        b            = float(kv["b"]),
        T0           = float(kv["T0"]),
        temperature  = float(kv["temperature"]),
        kappa        = float(kv["kappa"]),
        mobility     = float(kv["mobility"]),
        phi0         = float(kv["phi0"]),
        noise        = float(kv["noise"]),
        seed         = int(kv["seed"]),
        equation     = kv["equation"],
        solver       = kv["solver"],
    )


# ---------------------------------------------------------------------------
# Per-run statistics (statistics.csv)
# ---------------------------------------------------------------------------

def read_statistics_csv(path: str | Path) -> pd.DataFrame:
    """
    Read a per-run statistics.csv into a DataFrame, indexed by step
    (for joining against snapshot filenames via snapshot_filename(step)).

    NOTE: different batches of runs can have different columns (e.g. a
    'trace' column dropped later because it was found to duplicate
    'gradient_sqr') -- this function doesn't enforce a fixed schema;
    callers pooling statistics across multiple runs need to check
    column consistency themselves (see training/datasets.py).
    """
    path = Path(path)
    df = pd.read_csv(path)

    if "step" not in df.columns:
        raise ValueError(f"{path}: missing 'step' column")

    df["step"] = df["step"].astype(int)
    return df.set_index("step")
