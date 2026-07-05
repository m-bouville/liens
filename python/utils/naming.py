"""
Shared naming conventions for checkpoint and output filenames, so
train_ae.py/train_lds.py/evaluation scripts agree on one canonical
format instead of each inventing its own.
"""


def format_float_for_filename(value: float) -> str:
    """
    Encode a float for use in a filename: replace '.' with 'p' (e.g.
    0.01 -> '0p01', 1.0 -> '1p0', 0.002 -> '0p002') rather than
    stripping/padding digits to some fixed width -- lossless and
    unambiguous for any magnitude, unlike stripping the '0.' prefix
    (which can't tell 0.01 apart from 0.001, and breaks for values >= 1,
    both of which occur across this project's actual stats_weight range
    of 0.002-1.0).
    """
    s = repr(float(value))
    if "e" in s or "E" in s:
        # Not expected in this project's realistic weight range, but
        # guard against Python's repr switching to scientific notation
        # for very small/large values.
        s = f"{value:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "neg")


def ae_checkpoint_name(size: int, latent_channels: int, stats_weight: float) -> str:
    """e.g. '64x64-4latent-stats_weight_0p01'"""
    weight_str = format_float_for_filename(stats_weight)
    return f"{size}x{size}-{latent_channels}latent-stats_weight_{weight_str}"


def lds_checkpoint_name(size: int, latent_channels: int, stats_weight: float,
                         n_rollout_steps: int) -> str:
    """e.g. '64x64-4latent-weight_0p01-rollout_3'"""
    weight_str = format_float_for_filename(stats_weight)
    return f"{size}x{size}-{latent_channels}latent-weight_{weight_str}-rollout_{n_rollout_steps}"
