"""Guard test for evaluation.check_latent_channels._resolve_input_size.

Covers the stage-3b failure mode: a checkpoint whose config has no 'size'
key (it lives only in the stage-2 ancestor). The size must then be inferred
from the encoder's down-block count in the bundled AE state, prefix-agnostic.

Tests the helper directly on synthetic dicts -- no checkpoint or torch model
needed, so it stays fast and does not depend on any binary fixture.
"""

import pytest

from evaluation.check_latent_channels import _resolve_input_size


def test_recorded_config_size_is_preferred():
    # Stage 1/2: 'size' present -> used verbatim, state ignored even if it
    # would imply something else. Recorded value is authoritative.
    cfg = {"size": 128}
    misleading_state = {f"encoder.down_blocks.{i}.conv1.weight": None
                        for i in range(3)}  # would infer 64
    assert _resolve_input_size(cfg, misleading_state, 8) == 128


def test_size_inferred_when_absent_128():
    # Stage 3b: no 'size'. 4 down-blocks at 8x8 latent -> 128.
    cfg = {}
    state = {f"encoder.down_blocks.{i}.conv1.weight": None for i in range(4)}
    state["encoder.bottlenecks.z0.weight"] = None
    assert _resolve_input_size(cfg, state, 8) == 128


def test_inference_is_prefix_agnostic_64():
    # Whatever the encoder's prefix in the key path, the down_blocks.<i>.
    # segment is matched. 3 down-blocks at 8x8 -> 64.
    for prefix in ("", "encoder.", "ae.encoder."):
        cfg = {}
        state = {f"{prefix}down_blocks.{i}.conv1.weight": None
                 for i in range(3)}
        assert _resolve_input_size(cfg, state, 8) == 64


def test_inference_respects_latent_spatial_size():
    # 4 down-blocks, but a 16x16 latent -> 256 (size scales with the latent
    # spatial size, not just the block count).
    cfg = {}
    state = {f"encoder.down_blocks.{i}.conv1.weight": None for i in range(4)}
    assert _resolve_input_size(cfg, state, 16) == 256


def test_raises_when_neither_config_nor_blocks_available():
    # No 'size' and no down_blocks.* keys -> clear error, not a silent guess.
    with pytest.raises(KeyError):
        _resolve_input_size({}, {"some.other.key": None}, 8)
