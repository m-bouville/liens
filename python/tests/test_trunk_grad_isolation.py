"""Per-stream trunk-gradient isolation in the shared encoder.

The mechanism for stopping L_deriv from roughening z0's own trajectory:
the deriv stream reads the shared trunk forward but its loss's gradient
into the trunk can be scaled (1.0 = today, 0.0 = trunk frozen against it).
"""
import torch

from models.encoder import Encoder
from models.latent_streams import LatentStreamConfig


def _encoder():
    cfgs = {
        "state": LatentStreamConfig(name="state", channels=4, spatial_size=8,
                                     mode="autoencoder", condition_on_theta=False),
        "deriv": LatentStreamConfig(name="deriv", channels=4, spatial_size=8,
                                     mode="pure_latent", condition_on_theta=True),
    }
    return Encoder(input_size=32, base_channels=8, n_theta=1, stream_configs=cfgs)


def _trunk_grad(enc, x, th, stream):
    trunk = [p for n, p in enc.named_parameters() if "down_blocks" in n]
    enc.zero_grad()
    z = enc(x, th)
    z[stream].pow(2).sum().backward()
    return sum(p.grad.abs().sum().item() for p in trunk if p.grad is not None)


def test_forward_value_is_identical_at_every_scale():
    """Straight-through: scaling the gradient must not change the encoding.
    If it did, the fingerprint and every downstream latent would shift."""
    enc = _encoder()
    x, th = torch.randn(2, 1, 32, 32), torch.randn(2, 1)
    ref = enc(x, th)
    for scale in (0.0, 0.3, 1.0):
        enc.set_trunk_grad_scale("deriv", scale)
        z = enc(x, th)
        for k in ref:
            assert torch.allclose(ref[k], z[k]), (scale, k)


def test_scale_zero_blocks_the_stream_gradient_into_the_trunk():
    enc = _encoder()
    x, th = torch.randn(2, 1, 32, 32), torch.randn(2, 1)
    enc.set_trunk_grad_scale("deriv", 0.0)
    assert _trunk_grad(enc, x, th, "deriv") == 0.0


def test_scale_one_is_the_unchanged_default():
    """The default must leave trunk training exactly as before -- every
    prior run stays bit-identical."""
    enc = _encoder()
    x, th = torch.randn(2, 1, 32, 32), torch.randn(2, 1)
    # fresh encoder defaults to 1.0 without any setter call
    assert _trunk_grad(enc, x, th, "deriv") > 0.0


def test_partial_scale_is_linear():
    enc = _encoder()
    x, th = torch.randn(2, 1, 32, 32), torch.randn(2, 1)
    enc.set_trunk_grad_scale("deriv", 1.0)
    full = _trunk_grad(enc, x, th, "deriv")
    enc.set_trunk_grad_scale("deriv", 0.3)
    partial = _trunk_grad(enc, x, th, "deriv")
    assert abs(partial / full - 0.3) < 1e-5, partial / full


def test_isolating_one_stream_leaves_the_other_full():
    """State must still train the trunk fully while deriv is isolated -- the
    point is to keep z0's representation learning from reconstruction while
    denying L_deriv its trunk influence."""
    enc = _encoder()
    x, th = torch.randn(2, 1, 32, 32), torch.randn(2, 1)
    enc.set_trunk_grad_scale("deriv", 0.0)
    assert _trunk_grad(enc, x, th, "state") > 0.0


def test_unknown_stream_is_rejected():
    enc = _encoder()
    try:
        enc.set_trunk_grad_scale("nonesuch", 0.0)
    except KeyError:
        return
    raise AssertionError("silently accepted an unknown stream name")


def test_scale_is_not_persisted_in_state_dict():
    """It is a training-time control, not model state: it must not enter the
    checkpoint or it would change the architecture fingerprint."""
    enc = _encoder()
    enc.set_trunk_grad_scale("deriv", 0.0)
    assert not any("trunk_grad" in k for k in enc.state_dict())
