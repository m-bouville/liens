"""
Single source of truth for architectural constants shared across
models/, training/, evaluation/, and tests/.

LATENT_SPATIAL_SIZE previously existed as 8 INDEPENDENT hardcoded
copies of the literal 8 (see the project's own magic-number audit,
this file's whole reason for existing): encoder.py's depth calculation,
decoder.py's depth calculation AND its separate runtime shape check,
latent_dynamics.py's and stats_head.py's default constructor
parameters, training/datasets.py's augmentation phase calculation,
evaluation/check_parameter_dependence.py's reference-line calculation,
and tests/conftest.py's FakeEncoder fixture -- silently correct only as
long as nobody ever needed to change it, with a shape-mismatch (not a
clear error) as the failure mode for any one missed on an update.

This is now a DEFAULT, not a hardcoded assumption: the actual bottleneck
size for any given model is a real, configurable value (see Encoder/
Decoder's own latent_spatial_size constructor parameter, and the
'latent_size' stage-parameters-file key, both defaulting to this
constant when not specified) -- LATENT_SPATIAL_SIZE is what "not
specified" falls back to, not a ceiling on what's possible.
"""

LATENT_SPATIAL_SIZE = 8


# ---------------------------------------------------------------------------
# theta conditioning: the physical coordinates every theta-conditioned module
# (encoder FiLM, f_theta) is fed. Historically theta was the single scalar
# [T - T0]. That coordinate crowds the near-critical regime: below T0 the
# dynamics' every physical scale is a POWER of (T0 - T) -- the well minima
# sit at +/- sqrt(a0(T0-T)/b), the barrier depth ~ (T0-T)^2, and the
# coarsening timescale tau ~ 1/(T0-T) -- so 0.95 and 0.99 (i.e. T-T0 =
# -0.05 vs -0.01) are physically 5x apart in timescale yet nearly identical
# to a bounded-weight MLP in the linear coordinate, and a gain large enough
# to resolve them extrapolates to nonsense at T=0.6. log(T0 - T) is the
# coordinate in which all those power-law scales become linear, so the
# conditioner sees a uniform-rate dependence and can resolve near-T0 without
# a destructive gain. We keep BOTH: [T - T0, log(T0 - T)] -- feature 0 the
# smooth global proximity coordinate (continuity with every pre-existing
# checkpoint), feature 1 the physically-linear near-critical one.
#
# T < T0 is guaranteed by the physics, not just the sweep: at or above T0 the
# Landau potential is a single well at phi=0 with nothing to coarsen, so any
# run that produced coarsening data is strictly subcritical and log(T0 - T)
# is always finite. T0 is a fixed normalization (=1 dedimensionalized), the
# same for every run family, so one transform serves the whole dataset.
#
# n_theta is now 2. Pre-2-feature checkpoints upgrade by ZERO-INITIALISING
# the new (feature-1) input column of each conditioner's first Linear, so a
# resumed model is bit-identical in function at load and training grows the
# new coordinate's use from silence -- the same backward-compatible pattern
# the residual head uses on its own output conv.
N_THETA = 2


def theta_coordinates(temperature: float, T0: float):
    """The n_theta physical coordinates fed to every theta-conditioned module,
    from a run's temperature and the (fixed) critical T0. Single source of
    truth so the dataset's per-frame theta and any diagnostic that rebuilds
    theta cannot drift. Returns a python list of length N_THETA.

    feature 0: T - T0        (smooth, signed proximity to criticality)
    feature 1: log(T0 - T)   (linearises the power-law physical scales;
                              finite because T < T0 strictly, see above)

    Standardisation (zero-mean/unit-variance over the sweep's temperature
    list) is applied by the caller that knows the sweep, NOT here -- this
    returns raw physical coordinates so the transform is explicit and
    storable in the checkpoint config.
    """
    import math
    gap = T0 - temperature
    if gap <= 0.0:
        raise ValueError(
            f"theta_coordinates requires T < T0 (subcritical, where coarsening "
            f"happens): got temperature={temperature}, T0={T0}, so T0-T={gap} "
            f"<= 0. log(T0-T) is undefined there; a supercritical run has no "
            f"phases to evolve and should not be in the dataset.")
    return [temperature - T0, math.log(gap)]
