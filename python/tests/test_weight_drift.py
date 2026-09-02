"""
Tests for training/_train_ae_common.py's weight-drift diagnostic.

Found by an audit: `freeze_outer_layers` is well covered (a stage-2 test
asserts both that a frozen block's weights are unchanged and that an unfrozen
one moved), but `compute_weight_drift` -- the diagnostic whose output is
actually READ to confirm freezing worked -- had no test at all. Its report is
the

    encoders.shared.down_blocks.0     0.0000  <- frozen, should be 0

line in every stage-2 log. A version that returned 0.0 unconditionally would
make that report look perfect while proving nothing, and nothing in the suite
would have noticed.

The tests below therefore pin both directions: zero for genuinely unchanged
blocks, and NON-zero for changed ones.
"""
import pytest
import torch

from training._train_ae_common import _drift_by_block, _param_group, compute_weight_drift


def _state(**values):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in values.items()}


# --------------------------------------------------------------------
# the property the report depends on
# --------------------------------------------------------------------

def test_unchanged_blocks_report_exactly_zero():
    initial = _state(**{"encoders.shared.down_blocks.0.conv.weight": [1.0, 2.0],
                         "decoders.D0.output_conv.weight": [3.0]})
    params, buffers = compute_weight_drift(initial, {}, dict(initial), {})
    assert params["encoders.shared.down_blocks.0"] == 0.0
    assert params["decoders.D0.output_conv"] == 0.0
    assert buffers == {}


def test_changed_blocks_report_NON_zero():
    """
    GUARDS a drift function that always reports 0. That is the failure the
    whole report is blind to: every line would read "0.0000  <- frozen, should
    be 0" and a freeze that silently did nothing would look like a success.
    """
    initial = _state(**{"encoders.shared.down_blocks.0.conv.weight": [1.0, 2.0]})
    final = _state(**{"encoders.shared.down_blocks.0.conv.weight": [1.0, 5.0]})
    params, _ = compute_weight_drift(initial, {}, final, {})
    assert params["encoders.shared.down_blocks.0"] == pytest.approx(3.0)


def test_drift_is_an_L2_norm_accumulated_across_the_whole_block():
    """
    A block's entries are summed in QUADRATURE, not reported per tensor: two
    tensors each drifting by 3 and 4 give 5, not 7. Worth pinning because a
    plain sum would make a block with many small honest changes look worse
    than one with a single large one.
    """
    initial = _state(**{"encoders.shared.down_blocks.1.a": [0.0],
                         "encoders.shared.down_blocks.1.b": [0.0]})
    final = _state(**{"encoders.shared.down_blocks.1.a": [3.0],
                       "encoders.shared.down_blocks.1.b": [4.0]})
    params, _ = compute_weight_drift(initial, {}, final, {})
    assert params["encoders.shared.down_blocks.1"] == pytest.approx(5.0)


def test_parameters_and_buffers_are_reported_SEPARATELY():
    """
    The two have different expectations, which is why the report prints two
    tables: a frozen block's PARAMETERS must be exactly 0, while its BUFFERS
    (BatchNorm running_mean/var) legitimately move whenever the module is in
    train() mode, frozen or not. Merging them would make a normal BatchNorm
    update indistinguishable from a broken freeze.
    """
    initial_p = _state(**{"encoders.shared.down_blocks.0.conv.weight": [1.0]})
    initial_b = _state(**{"encoders.shared.down_blocks.0.bn.running_mean": [0.0]})
    final_b = _state(**{"encoders.shared.down_blocks.0.bn.running_mean": [7.0]})
    params, buffers = compute_weight_drift(initial_p, initial_b, dict(initial_p), final_b)
    assert params["encoders.shared.down_blocks.0"] == 0.0
    assert buffers["encoders.shared.down_blocks.0"] == pytest.approx(7.0)


# --------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------

@pytest.mark.parametrize("key,expected", [
    ("encoders.shared.down_blocks.0.conv.block.0.weight", "encoders.shared.down_blocks.0"),
    ("encoders.shared.down_blocks.11.down.bias", "encoders.shared.down_blocks.11"),
    ("decoders.D0.up_blocks.2.conv.block.3.weight", "decoders.D0.up_blocks.2"),
    ("decoders.D0.output_conv.weight", "decoders.D0.output_conv"),
    ("encoders.shared.bottlenecks.state.weight", "encoders.shared.bottlenecks"),
    ("pathways.deriv.log_output_scale", "pathways.deriv.log_output_scale"),
])
def test_keys_group_by_containing_block(key, expected):
    assert _param_group(key) == expected


def test_indexed_blocks_do_not_collapse_into_one_group():
    """
    GUARDS truncating to three components for down_blocks/up_blocks too. That
    would merge every down_block into a single "encoders.shared.down_blocks"
    row -- and since the report's whole purpose is distinguishing the frozen
    OUTER blocks from the trainable inner ones, a merged row would hide
    exactly the distinction being checked.
    """
    initial = _state(**{"encoders.shared.down_blocks.0.w": [0.0],
                         "encoders.shared.down_blocks.1.w": [0.0]})
    final = _state(**{"encoders.shared.down_blocks.0.w": [0.0],     # frozen
                       "encoders.shared.down_blocks.1.w": [4.0]})    # trained
    params, _ = compute_weight_drift(initial, {}, final, {})
    assert params["encoders.shared.down_blocks.0"] == 0.0
    assert params["encoders.shared.down_blocks.1"] == pytest.approx(4.0)


def test_empty_state_reports_nothing_rather_than_raising():
    assert compute_weight_drift({}, {}, {}, {}) == ({}, {})


def test_drift_by_block_handles_a_dtype_change_without_raising():
    """Buffers can be integral (BatchNorm's num_batches_tracked); the
    difference must still be computed in floating point."""
    initial = {"encoders.shared.down_blocks.0.bn.num_batches_tracked":
               torch.tensor(0, dtype=torch.long)}
    final = {"encoders.shared.down_blocks.0.bn.num_batches_tracked":
             torch.tensor(9, dtype=torch.long)}
    assert _drift_by_block(initial, final)["encoders.shared.down_blocks.0"] == pytest.approx(9.0)
