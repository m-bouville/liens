"""accumulate_epoch: the shared per-epoch batch pass extracted from the trainers.

Tested against its OUTPUT (the reduced means and the batch count), since the
whole point of the extraction is that a fix to it lands once for all three
trainers -- so its contract must be pinned independently of any trainer."""
import torch

from training.training_loop import accumulate_epoch


def _loader(batch_sizes):
    """Batches whose only content is their size (batch[0].size(0) is what the
    sample-weighting reads)."""
    return [(torch.zeros(bs, 1),) for bs in batch_sizes]


def test_it_sample_weights_the_average_not_batch_weights():
    """The average is over SAMPLES, not batches: an unequal last batch must not
    be over-weighted. Two batches of sizes 6 and 2 carrying component values 1.0
    and 5.0 average to (6*1 + 2*5)/8 = 2.0, NOT the batch-mean (1+5)/2 = 3.0."""
    loader = _loader([6, 2])
    vals = iter([1.0, 5.0])
    means, n = accumulate_epoch(
        loader, lambda b: {"total": torch.tensor(next(vals))}, n_samples=8)
    assert means["total"] == 2.0, "must be sample-weighted, not batch-weighted"
    assert n == 2


def test_it_reduces_every_component_key_independently():
    loader = _loader([4, 4])
    seq = iter([{"total": 1.0, "a": 2.0}, {"total": 3.0, "a": 8.0}])
    means, _ = accumulate_epoch(
        loader, lambda b: {k: torch.tensor(v) for k, v in next(seq).items()},
        n_samples=8)
    assert means["total"] == 2.0 and means["a"] == 5.0


def test_it_ticks_the_progress_once_per_batch():
    class _P:
        def __init__(self): self.n = 0
        def tick(self): self.n += 1
    p = _P()
    accumulate_epoch(_loader([1, 1, 1]), lambda b: {"total": torch.tensor(0.0)},
                     n_samples=3, progress=p)
    assert p.n == 3


def test_an_empty_loader_reduces_to_no_components_and_zero_batches():
    means, n = accumulate_epoch([], lambda b: {"total": torch.tensor(0.0)}, n_samples=1)
    assert means == {} and n == 0


from training.training_loop import weighted_contributions


def test_weighted_contributions_is_weight_times_raw_over_scale():
    r = weighted_contributions(
        {"a": 2.0, "b": 10.0}, {"a": 0.5, "b": 0.1}, {"a": 4.0, "b": 2.0})
    assert r == {"a": 0.5 * 2.0 / 4.0, "b": 0.1 * 10.0 / 2.0}


def test_an_absent_weight_defaults_to_one_the_implicit_anchor():
    """stage 2's recon0 has no recon0_weight parameter -- it is the fixed anchor
    (coefficient 1). An absent weight must default to 1, not drop the term or
    zero it."""
    r = weighted_contributions({"recon0": 3.0}, {}, {"recon0": 1.5})
    assert r == {"recon0": 1.0 * 3.0 / 1.5}


def test_it_covers_exactly_the_raw_keys():
    r = weighted_contributions({"a": 1.0}, {"a": 1.0, "unused": 9.0}, {"a": 1.0, "unused": 1.0})
    assert set(r) == {"a"}, "only raw's keys are contributions; extra weights are ignored"
