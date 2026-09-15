"""
Diagnostic + regression test for the latent-cache round-trip.

Context: two stage-4 runs with identical params, seed, and pinned frozen inputs
diverged on the ROLLOUT loss terms from epoch 1, while the frame-0 terms
(recon0/stats0) stayed bit-identical. One hypothesis was that the latent cache
loses precision on write/read, so a cache-WRITING run (uses the in-memory encoded
latents) and a cache-READING run (uses the torch.save/torch.load round-trip)
feed slightly different z0 into the rollout -- amplified by
derivative_source='previous_quotient', which DIFFERENCES consecutive latents.

These tests check that hypothesis directly, without a GPU or a dataset, by
exercising the exact functions the two paths differ on:
  - cache MISS path uses the in-memory `latents` (see datasets.py _flush_buffer);
  - cache HIT path uses load_cached(store_cached(latents)).
If the round-trip is bit-identical, the cache is NOT the source of the divergence
and datasets.py must NOT be changed for it. The tests assert bit-identity across
the realistic tensor variants (float32/float16, contiguous/not, state+deriv), so
a future change that DID narrow the cache (e.g. a float16 cast on write) would be
caught here as a real reproducibility regression.

Finding at the time of writing: the round-trip is EXACTLY identical -- the cache
is dtype- and value-preserving -- so it does not explain the rollout divergence.
"""
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

try:
    from training import latent_cache
except ImportError:                       # flat layout
    import latent_cache


def _roundtrip(state, deriv=None):
    d = Path(tempfile.mkdtemp())
    path = d / "cache.pt"
    latent_cache.store_cached(path, state, deriv)
    loaded = latent_cache.load_cached(path)
    assert loaded is not None, "store/load round-trip returned None"
    return loaded                          # (state, deriv)


def _prev_quotient(z):
    """The previous_quotient differencing the rollout applies: z(t) - z(t-1)."""
    return z[1:] - z[:-1]


# --------------------------------------------------------------------------- #
# the core claim: cache HIT (reloaded) == cache MISS (in-memory), bit for bit
# --------------------------------------------------------------------------- #
def test_cache_roundtrip_is_bit_identical_float32():
    torch.manual_seed(0)
    latents = torch.randn(71, 4, 8, 8, dtype=torch.float32)   # z0-like
    rl_state, _ = _roundtrip(latents)
    assert torch.equal(latents, rl_state)                     # exact, no tolerance
    assert rl_state.dtype == latents.dtype


def test_cache_roundtrip_identical_after_previous_quotient_differencing():
    """Even the DIFFERENCED latents (what previous_quotient feeds the rollout) are
    identical -- so the cache cannot be the source of a rollout-only divergence."""
    torch.manual_seed(1)
    latents = torch.randn(71, 4, 8, 8, dtype=torch.float32)
    rl_state, _ = _roundtrip(latents)
    assert torch.equal(_prev_quotient(latents), _prev_quotient(rl_state))


def test_cache_roundtrip_bit_identical_for_non_contiguous_source():
    """The in-memory latents on the miss path may be a non-contiguous view; the
    reloaded tensor is contiguous. Values must still match exactly."""
    torch.manual_seed(2)
    noncontig = torch.randn(71, 8, 4, 8).transpose(1, 2)      # non-contiguous
    assert not noncontig.is_contiguous()
    rl_state, _ = _roundtrip(noncontig)
    assert torch.equal(noncontig, rl_state)


def test_cache_roundtrip_preserves_float16_without_widening():
    torch.manual_seed(3)
    half = torch.randn(71, 4, 8, 8).half()
    rl_state, _ = _roundtrip(half)
    assert rl_state.dtype == torch.float16
    assert torch.equal(half, rl_state)


def test_cache_roundtrip_bit_identical_for_state_and_deriv():
    torch.manual_seed(4)
    state = torch.randn(71, 4, 8, 8)
    deriv = torch.randn(71, 4, 8, 8)
    rl_state, rl_deriv = _roundtrip(state, deriv)
    assert torch.equal(state, rl_state)
    assert rl_deriv is not None and torch.equal(deriv, rl_deriv)


def test_cache_does_not_narrow_precision_on_write():
    """GUARD: if a future change casts to a narrower dtype on write (e.g. .half()),
    the round-trip stops being identity and cache-hit vs cache-miss runs diverge.
    This test would then fail -- flagging a real reproducibility regression."""
    torch.manual_seed(5)
    latents = torch.randn(200, 4, 8, 8, dtype=torch.float32)
    rl_state, _ = _roundtrip(latents)
    assert rl_state.dtype == torch.float32                    # not narrowed
    assert (latents - rl_state).abs().max().item() == 0.0
