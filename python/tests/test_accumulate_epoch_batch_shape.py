"""accumulate_epoch must sample-weight-average correctly for BOTH batch shapes:
the rollout trainers' tuple batches (size from element 0) and stage 1's
bare-tensor batches when there are no stats targets (size from the tensor
itself). The bare-tensor case was added when stage 1's val loop was migrated;
before the fix, batch[0].size(0) took the wrong dimension of a bare tensor."""
import torch
from training._training_loop import accumulate_epoch


def test_tuple_and_bare_tensor_batches_both_weight_by_true_batch_size():
    # two batches of different sizes so a wrong bs would skew the mean
    def make_loader(bare):
        b1 = torch.full((4, 3), 1.0)
        b2 = torch.full((2, 3), 4.0)          # value 4, half the samples
        return [b1, b2] if bare else [(b1, None), (b2, None)]

    # forward_fn returns the batch's mean value as "total"; sample-weighted mean
    # over 6 samples = (4*1 + 2*4)/6 = 2.0
    def fwd(batch):
        x = batch if isinstance(batch, torch.Tensor) else batch[0]
        return {"total": x.mean().detach()}

    for bare in (True, False):
        means, n_batches = accumulate_epoch(make_loader(bare), fwd, n_samples=6)
        assert n_batches == 2
        assert abs(means["total"] - 2.0) < 1e-6, (
            f"{'bare' if bare else 'tuple'} batch: sample-weighting wrong "
            f"(got {means['total']}, want 2.0)")


def test_empty_loader_returns_empty_means_and_zero_batches():
    means, n = accumulate_epoch([], lambda b: {"total": b}, n_samples=0)
    assert means == {} and n == 0
