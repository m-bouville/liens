"""
Tests for normalize_phi: rescaling each frame by 1/phi_eq(T) so the double-well
ground state sits at +/-1 regardless of temperature.

phi_eq(T) = sqrt(a0*(T0 - T)/b), with a0=b=1 for this sweep, so sqrt(T0 - T).
The SAME value is used two ways and they must agree, or the field is scaled on
the way in and not un-scaled on the way out:
  - datasets divide each frame by phi_eq before encoding (datasets.py);
  - the Allen-Cahn loss multiplies the decoded field back by phi_eq before its
    (physical) residual (_refinement_loss.py).

Three tiers here:
  1. CACHE  -- pure path logic, no torch: normalized and raw latents share a
     frozen encoder (same fingerprint) so they MUST land in different dirs, and
     normalize_phi=False must reproduce the pre-normalization names byte-for-byte.
  2. DATASET -- with the tmp_run_dir fixture: a normalized frame is exactly the
     raw frame / phi_eq(T), for both dataset classes; False leaves it untouched.
  3. LOSS   -- the un-normalization is INVARIANT: the AC residual on a decoded
     psi with normalize_phi=True equals the residual on psi*phi_eq with
     normalize_phi=False. This is the property that makes AC correct on
     normalized data.

Run from python/ (imports rely on that root being on sys.path).
"""
import math

import pytest

from training.latent_cache import cache_path_for_run, write_cache_info


# --------------------------------------------------------------------------- #
# 1. CACHE: dir separation + back-compat  (pure, no torch)
# --------------------------------------------------------------------------- #
def _args(tmp_path):
    """Common cache_path_for_run args; only normalize_phi varies per test."""
    from pathlib import Path
    return dict(cache_root=tmp_path, fingerprint="abc123", run_dir=Path("T850_n001_s0"),
                steps=[0, 1000, 2000], encode_both_streams=False, size=128)


def test_normalized_and_raw_caches_are_different_directories(tmp_path):
    """The frozen encoder is identical for a normalized and a raw run (same
    fingerprint), but the cached latent is E(phi/phi_eq) vs E(phi) -- different
    values. Without a dir marker they would collide and serve each other
    silently. The normalized path must live under a distinct directory."""
    raw = cache_path_for_run(**_args(tmp_path), normalize_phi=False)
    norm = cache_path_for_run(**_args(tmp_path), normalize_phi=True)
    assert raw.parent != norm.parent, "normalized and raw caches share a directory -- they will collide"
    assert norm.parent.name.endswith("-norm"), f"expected a -norm dir marker, got {norm.parent.name}"
    assert not raw.parent.name.endswith("-norm"), f"raw dir must not be marked, got {raw.parent.name}"
    # only the DIRECTORY differs -- the per-run filename (run, steps, streams) is the same
    assert raw.name == norm.name


def test_normalize_phi_false_is_byte_identical_to_pre_normalization(tmp_path):
    """Back-compat: normalize_phi=False (the default) must produce EXACTLY the
    path the pre-normalization code produced (no arg at all), so existing raw
    caches stay valid and readable."""
    without_arg = cache_path_for_run(**_args(tmp_path))
    explicit_false = cache_path_for_run(**_args(tmp_path), normalize_phi=False)
    assert without_arg == explicit_false


def test_write_cache_info_respects_the_norm_marker(tmp_path):
    """The human-readable _cache_info.txt must land in the SAME directory the
    latents do -- i.e. the -norm dir when normalize_phi -- or the note describes
    the wrong cache."""
    write_cache_info(tmp_path, "abc123", 128, normalize_phi=True)
    assert (tmp_path / "128x128-abc123-norm" / "_cache_info.txt").exists()
    assert not (tmp_path / "128x128-abc123" / "_cache_info.txt").exists()


def test_norm_marker_composes_with_theta_tag(tmp_path):
    """theta (a per-run encoder input) tags the FILENAME; normalize_phi marks the
    DIRECTORY. They are orthogonal and must both take effect at once."""
    p = cache_path_for_run(**_args(tmp_path), theta=[0.1, -2.3], normalize_phi=True)
    assert p.parent.name.endswith("-norm")     # directory marked
    assert "-t" in p.name                        # filename theta-tagged


# --------------------------------------------------------------------------- #
# 2. DATASET: a normalized frame is exactly raw / phi_eq(T)
# --------------------------------------------------------------------------- #
# These use the shared tmp_run_dir fixture (see conftest). tmp_run_dir writes
# constant-field snapshots and a metadata.txt; we read that metadata to derive
# the expected phi_eq rather than hardcoding the fixture's temperature.
torch = pytest.importorskip("torch")

from training.datasets import (  # noqa: E402  (after importorskip)
    MicrostructureEvolutionDataset, MicrostructureSnapshotDataset,
    _STAT_PHI_EQ_EXPONENT, _STATS_DROPPED_WHEN_NORMALIZED,
    normalized_stat_names, check_stat_names_normalizable, normalize_stats_vector,
)
from utils import load_datasets as load  # noqa: E402  (load_datasets lives in utils/)


def _expected_phi_eq(run_dir):
    md = load.read_metadata(run_dir / "metadata.txt")
    # sqrt(a0*(T0-T)/b) from the run's own metadata -- the same formula the datasets
    # use; asserting against metadata (not a hardcoded a0=b=1) is what makes the
    # test catch a regression to hardcoded constants.
    return math.sqrt(max(md.a0 * (md.T0 - md.temperature) / md.b, 1e-12))


def test_evolution_dataset_normalizes_by_phi_eq(tmp_run_dir):
    """A normalized window equals the raw window divided by phi_eq(T), element-
    wise, in raw (encoder=None) mode -- so the scaling is applied to the FIELD,
    before any encoding, exactly once."""
    run_dir, _steps = tmp_run_dir
    raw = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
        normalize_phi=False)
    norm = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
        normalize_phi=True)
    phi_eq = _expected_phi_eq(run_dir)
    raw_w, _, _ = raw[0]
    norm_w, _, _ = norm[0]
    assert torch.allclose(norm_w, raw_w / phi_eq, atol=1e-4), \
        f"normalized window != raw / phi_eq (phi_eq={phi_eq:.4f})"


def test_evolution_dataset_false_leaves_frames_untouched(tmp_run_dir):
    """normalize_phi=False (default) must not alter the field at all."""
    run_dir, _steps = tmp_run_dir
    default = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None)
    explicit = MicrostructureEvolutionDataset(
        [run_dir], encoder=None, window_length=3, min_step=0, min_stdev_phi=None,
        normalize_phi=False)
    assert torch.allclose(default[0][0], explicit[0][0])


def test_snapshot_dataset_normalizes_by_phi_eq(tmp_run_dir):
    """The AE-stage dataset (single frames) normalizes identically -- the
    encoder is trained on the SAME scaling stage 3+ will feed it."""
    run_dir, _steps = tmp_run_dir
    raw = MicrostructureSnapshotDataset([run_dir], min_step=0, min_stdev_phi=None,
                                         normalize_phi=False)
    norm = MicrostructureSnapshotDataset([run_dir], min_step=0, min_stdev_phi=None,
                                          normalize_phi=True)
    phi_eq = _expected_phi_eq(run_dir)
    assert torch.allclose(norm[0], raw[0] / phi_eq, atol=1e-4), \
        f"normalized frame != raw / phi_eq (phi_eq={phi_eq:.4f})"


# --------------------------------------------------------------------------- #
# 2b. STATS TARGETS: the preprocessing rule under normalize_phi
# --------------------------------------------------------------------------- #
# Under phi -> phi/phi_eq each statistics.csv column transforms by a power of
# phi_eq; the targets are divided by phi_eq**exponent so they sit in the same
# units as the field the stats head sees. The raw +/-0.1 thresholds have no
# clean transform and are dropped (n_stats 12 -> 10).
_ALL_12 = ["angle", "anisotropy", "autocorr_correl", "autocorr_length", "avg_gradient",
           "avg_phi", "energy", "gradient_sqr", "phi_above_10", "phi_below_-10",
           "phi_below_0", "stdev_phi"]


def test_normalized_stat_names_drops_exactly_the_two_thresholds_and_keeps_order():
    kept = normalized_stat_names(_ALL_12)
    assert len(kept) == 10
    assert set(_STATS_DROPPED_WHEN_NORMALIZED) == {"phi_above_10", "phi_below_-10"}
    assert not (set(kept) & set(_STATS_DROPPED_WHEN_NORMALIZED))
    # order preserved (the stats head's output index depends on it)
    assert kept == [n for n in _ALL_12 if n not in _STATS_DROPPED_WHEN_NORMALIZED]


def test_every_kept_stat_has_an_exponent_and_the_physics_is_right():
    """The exponent table is the physics: ratios / C(0)-normalized correlations /
    the sign threshold are scale-invariant (0); avg/stdev/avg_gradient are linear
    in phi (1); gradient_sqr and energy quadratic (2 -- energy's dominant term)."""
    kept = normalized_stat_names(_ALL_12)
    assert set(kept) <= set(_STAT_PHI_EQ_EXPONENT), "a kept stat has no exponent"
    for n in ("anisotropy", "angle", "autocorr_correl", "autocorr_length", "phi_below_0"):
        assert _STAT_PHI_EQ_EXPONENT[n] == 0, n
    for n in ("avg_phi", "stdev_phi", "avg_gradient"):
        assert _STAT_PHI_EQ_EXPONENT[n] == 1, n
    for n in ("gradient_sqr", "energy"):
        assert _STAT_PHI_EQ_EXPONENT[n] == 2, n


def test_normalize_stats_vector_divides_by_phi_eq_to_the_exponent():
    kept = normalized_stat_names(_ALL_12)
    phi_eq = 0.5
    out = normalize_stats_vector(torch.ones(len(kept)), kept, phi_eq)
    for n, v in zip(kept, out.tolist()):
        assert abs(v - 1.0 / phi_eq ** _STAT_PHI_EQ_EXPONENT[n]) < 1e-6, (n, v)
    # concretely: invariant x1, linear x2, quadratic x4 at phi_eq=0.5
    idx = {n: i for i, n in enumerate(kept)}
    assert abs(out[idx["anisotropy"]] - 1.0) < 1e-6
    assert abs(out[idx["stdev_phi"]] - 2.0) < 1e-6
    assert abs(out[idx["energy"]] - 4.0) < 1e-6


def test_explicit_threshold_stat_is_refused_under_normalization():
    """A caller asking for a dropped stat with normalize_phi=True gets a clear
    error, not a silently mis-scaled target."""
    with pytest.raises(ValueError, match="cannot be used with normalize_phi"):
        check_stat_names_normalizable(["stdev_phi", "phi_above_10"], "test")


def test_unknown_stat_is_refused_under_normalization():
    """A stat with no phi_eq exponent must never pass through un-normalized --
    a new column added to statistics.csv has to declare its power first."""
    with pytest.raises(ValueError, match="no phi_eq exponent"):
        check_stat_names_normalizable(["stdev_phi", "brand_new_stat"], "test")


def _write_statistics_csv(run_dir, steps):
    """The tmp_run_dir fixture writes snapshots + metadata but NO statistics.csv, so
    write one here with all 12 standard columns and KNOWN values (stat value =
    its column index + 1, constant over steps), so every assertion below is exact
    and fixture-independent. Format matches load.read_statistics_csv: a 'step'
    column plus one column per stat."""
    import pandas as pd
    rows = [{"step": st, **{n: float(i + 1) for i, n in enumerate(_ALL_12)}}
            for st in steps]
    pd.DataFrame(rows).to_csv(run_dir / "statistics.csv", index=False)


def test_snapshot_dataset_stats_targets_are_normalized(tmp_run_dir):
    """End to end through the dataset: with include_stats, the true-stats vector a
    normalized SnapshotDataset returns equals the raw one divided by phi_eq**exp
    per column -- and it has 10 entries, not 12."""
    run_dir, steps = tmp_run_dir
    _write_statistics_csv(run_dir, steps)
    raw = MicrostructureSnapshotDataset([run_dir], min_step=0, min_stdev_phi=None,
                                         include_stats=True, normalize_phi=False)
    norm = MicrostructureSnapshotDataset([run_dir], min_step=0, min_stdev_phi=None,
                                          include_stats=True, normalize_phi=True)
    assert len(norm.stat_names) == len(raw.stat_names) - 2
    assert not (set(norm.stat_names) & set(_STATS_DROPPED_WHEN_NORMALIZED))
    phi_eq = _expected_phi_eq(run_dir)
    _, raw_stats = raw[0]
    _, norm_stats = norm[0]
    raw_by = dict(zip(raw.stat_names, raw_stats.tolist()))
    for n, v in zip(norm.stat_names, norm_stats.tolist()):
        want = raw_by[n] / phi_eq ** _STAT_PHI_EQ_EXPONENT[n]
        assert abs(v - want) < 1e-4 * max(1.0, abs(want)), (n, v, want)


# --------------------------------------------------------------------------- #
# 2c. LATENT-SCALE ANCHOR: z0_scale_loss (the fix for the normalize_phi latent blow-up)
# --------------------------------------------------------------------------- #
from training.losses import z0_scale_loss  # noqa: E402


def test_z0_scale_loss_is_mean_squared_latent_element():
    """mean over batch of the mean-squared latent element -- independent of
    latent_channels/spatial size, so its magnitude does not shift when the latent
    shape changes."""
    # elements 1,1,4,0,0,0 -> mean square = 6/6 = 1.0
    z = torch.tensor([[1.0, -1.0, 2.0], [0.0, 0.0, 0.0]])
    assert abs(z0_scale_loss(z).item() - 1.0) < 1e-6
    # all 0.5 -> 0.25
    assert abs(z0_scale_loss(torch.full((4, 2), 0.5)).item() - 0.25) < 1e-6


def test_z0_scale_loss_grows_with_latent_magnitude():
    """The anchor must increase as the latent inflates -- that is what lets it
    push back on the blow-up. Scaling the latent by c scales the loss by c**2."""
    z = torch.randn(8, 4, 4)
    base = z0_scale_loss(z).item()
    assert abs(z0_scale_loss(3.0 * z).item() - 9.0 * base) < 1e-4 * max(1.0, 9.0 * base)


def test_z0_scale_loss_shape_invariant_magnitude():
    """A latent twice as large in CHANNELS but same element scale gives the same
    loss (mean, not sum) -- so z0_scale_scale need not be retuned per latent shape."""
    small = torch.full((8, 4, 8, 8), 0.7)
    big = torch.full((8, 16, 8, 8), 0.7)
    assert abs(z0_scale_loss(small).item() - z0_scale_loss(big).item()) < 1e-6


# --------------------------------------------------------------------------- #
# 3. LOSS: AC un-normalization is invariant
# --------------------------------------------------------------------------- #
# The residual on a decoded psi with normalize_phi=True must equal the residual
# on psi*phi_eq with normalize_phi=False. We drive compute_stage45_loss through
# its REAL model interfaces (ae.encoders["shared"](x, theta=...) -> dict of
# streams, ae.pathways[name].decoder(z), f_theta.rollout(...)) with minimal
# stand-ins, so only the Allen-Cahn component's math is actually exercised.
#
# HISTORY: this test used to be an xfail stub whose _StubAE exposed an
# ae.encode(x, theta=...) method -- but compute_stage45_loss never calls
# ae.encode() at all; it calls ae.encoders["shared"](x, theta=...) (see
# _refinement_loss.py). The stub's encode() was therefore dead code, and every
# run hit an AttributeError on ae.encoders before ever reaching the Allen-Cahn
# math the test exists to check. Fixed by giving _StubAE a real .encoders
# dict (a tiny stub "shared" encoder module returning a fixed-shape zero
# latent dict), matching the interface every other call site in this project
# actually uses. A SECOND interface mismatch survived that first fix: the stub
# named its recon stream "recon", but compute_stage45_loss looks streams up by
# recon_stream_name, whose DEFAULT is DEFAULT_STREAM_NAME ("state"), so the
# encoder's returned dict was indexed with a key it did not contain and the loss
# died with KeyError: 'state' before ever reaching the Allen-Cahn math. The stub
# now names its recon stream DEFAULT_STREAM_NAME, so it lines up with the real
# default (deriv_stream_name already defaulted to "deriv", which the stub used).
from training._refinement_loss import compute_stage45_loss  # noqa: E402
from models.latent_streams import DEFAULT_STREAM_NAME  # noqa: E402
import torch.nn as nn  # noqa: E402


class _StubPathway(nn.Module):
    """decoder returns a FIXED field (independent of the latent), so both loss
    calls see a decoded field we chose; log_output_scale=0 => exp()=1."""
    def __init__(self, field):
        super().__init__()
        self._field = field
        self.log_output_scale = nn.Parameter(torch.zeros(()), requires_grad=False)

    def decoder(self, z):
        # z is (N, C, h, w); return the fixed field broadcast to N rows
        n = z.shape[0]
        return self._field[:1].expand(n, *self._field.shape[1:])


class _StubSharedEncoder(nn.Module):
    """Stand-in for ae.encoders["shared"]: ignores the actual pixel content
    (the Allen-Cahn term never depends on what z0/z1 numerically ARE -- only
    the decoded field, which _StubPathway fixes independently of them) and
    returns a fixed-shape zero latent per stream. Real shape values don't
    matter here -- only that recon_stream_name/deriv_stream_name are both
    present with a shape _StubPathway.decoder can accept (any (N, C, h, w)
    works, since it ignores z entirely)."""
    def __init__(self, recon_name, deriv_name, latent_channels=8, latent_spatial=8):
        super().__init__()
        self._recon_name = recon_name
        self._deriv_name = deriv_name
        self._shape = (latent_channels, latent_spatial, latent_spatial)

    def forward(self, x, theta=None):
        b = x.shape[0]
        z = torch.zeros(b, *self._shape, dtype=x.dtype, device=x.device)
        return {self._recon_name: z, self._deriv_name: z.clone()}


class _StubAE(nn.Module):
    def __init__(self, field, recon_name, deriv_name="deriv"):
        super().__init__()
        self.pathways = {recon_name: _StubPathway(field)}
        self.encoders = {"shared": _StubSharedEncoder(recon_name, deriv_name)}


class _StubFTheta(nn.Module):
    time_coordinate = "t"
    derivative_source = "z1"

    def rollout(self, z0, z1_seq, dt, theta, **kw):
        # (B, n_r+1, C, h, w): hold z0 across steps -- the AC term only needs a
        # decoded field per frame, which the stub decoder fixes anyway.
        n_r = dt.shape[1]
        return z0[:, None].expand(z0.shape[0], n_r + 1, *z0.shape[1:])


def test_ac_unnormalization_is_invariant():
    """AC residual on (psi, normalize_phi=True) == residual on (psi*phi_eq,
    normalize_phi=False): un-normalizing inside the loss recovers the physical
    field, so a model trained/decoded in normalized units is graded by the SAME
    physics as one in raw units."""
    torch.manual_seed(0)
    B, n_r, H, W = 2, 2, 8, 8
    recon = DEFAULT_STREAM_NAME   # must match the loss's recon_stream_name default
    T, T0, a0, b = 0.75, 1.0, 1.0, 1.0
    phi_eq = math.sqrt(a0 * (T0 - T) / b)

    psi = torch.randn(B * (n_r + 1), 1, H, W) * 0.5      # a normalized-scale field
    theta = torch.tensor([[T - T0, math.log(T0 - T)]] * B)   # theta[:,0] = T-T0
    x_window = torch.randn(B, n_r + 1, 1, H, W)
    t_window = torch.arange(1, n_r + 2, dtype=torch.float32)[None].expand(B, -1).contiguous()
    dt_window = t_window[:, 1:] - t_window[:, :-1]

    def ac(field, normalize_phi):
        ae = _StubAE(field, recon)
        _, comps = compute_stage45_loss(
            ae, _StubFTheta(), None, x_window, dt_window, theta,
            rollout_weight=0.0, recon0_weight=0.0, stats0_weight=0.0,
            recon_predict_weight=0.0, grad_predict_weight=0.0,
            allen_cahn_weight=1.0, allen_cahn_scale=1.0,
            allen_cahn_a0=a0, allen_cahn_b=b, allen_cahn_phi_max=1e6,   # no clamp
            allen_cahn_all_steps=True, normalize_phi=normalize_phi,
            return_components=True, t_window=t_window,
        )
        return comps["allen_cahn"]

    r_norm = ac(psi, normalize_phi=True)            # loss multiplies psi by phi_eq
    r_raw = ac(psi * phi_eq, normalize_phi=False)   # already physical
    assert torch.allclose(r_norm, r_raw, atol=1e-5), \
        f"AC not invariant under normalization: norm={r_norm} raw={r_raw}"
