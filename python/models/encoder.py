"""
Convolutional encoder: maps a real-space microstructure to one or more
compact latent representations (streams) -- see the project's own
C0/C1 design doc for why more than one.
"""

import math

import torch
import torch.nn as nn

from .blocks import DownBlock
from .constants import LATENT_SPATIAL_SIZE
from .latent_streams import LatentStreamConfig


class _ThetaFiLMConditioner(nn.Module):
    """
    theta -> (gamma, beta), applied to a stream's OWN trunk features
    (BEFORE that stream's own bottleneck conv, not shared with any
    other stream) as x * (1 + gamma) + beta, per-channel -- standard
    FiLM (Feature-wise Linear Modulation).

    (1 + gamma), not gamma alone, and this module's own FINAL layer
    zero-initialized (both weight and bias): together, these make
    conditioning an EXACT no-op at initialization (gamma=0 -> scale=1,
    beta=0 -> no shift) -- training starts numerically IDENTICAL to the
    unconditioned baseline and has to actively learn to use theta,
    rather than being forced to depend on a randomly-initialized
    conditioning signal from step one (the same zero-init-the-part-that-
    changes-behavior pattern LatentDynamics' own f_theta uses on its own
    final layer, for the same reason).
    """

    def __init__(self, n_theta: int, n_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_theta, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * n_channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(theta).chunk(2, dim=-1)  # each (B, n_channels)
        gamma = gamma.view(*gamma.shape, 1, 1)
        beta = beta.view(*beta.shape, 1, 1)
        return x * (1 + gamma) + beta


class Encoder(nn.Module):
    """
    A SHARED trunk (repeated DownBlocks halving spatial resolution)
    feeding N separate 1x1-conv projections, one per declared latent
    stream (stream_configs) -- e.g. today's default single-stream case
    (backward-compatible with everything before the multi-stream
    redesign), or C0+C1 (two streams sharing the trunk, splitting only
    at this final projection). Two SEPARATE nn.Conv2d modules (an
    nn.ModuleDict keyed by stream name), not one wider conv sliced
    after the fact -- deliberately, so each stream's projection can be
    frozen/unfrozen independently (PyTorch has no clean way to
    partially freeze one module's output channels), which the training
    strategy for this redesign specifically needs.

    All streams share ONE spatial_size (the trunk only produces one
    bottleneck resolution) -- validated equal across stream_configs at
    construction. Channel count may differ per stream freely; the
    equal-channel-count constraint from earlier design discussion is a
    DECODER-sharing constraint (see decoder.py, latent_streams.py),
    not something Encoder itself needs to enforce.

    Depth is derived from input_size so the shared spatial_size is
    reached exactly: 3 stages for 64x64 at the default 8x8, 5 for
    256x256, matching docs/neural_nets.md.

    Skip connections (the pre-downsampling features at each level) are
    always computed by DownBlock at negligible extra cost, but are only
    *returned* when use_skips=True. Defaults to False: the plumbing is
    here for when skip_connections.py is trained later (encoder frozen,
    decoder given skips) -- for now the encoder runs skip-free. NOTE:
    with multiple streams, which stream(s) a future skip-connections
    path would feed is not yet decided -- flagged in the design doc,
    not resolved here.

    THETA CONDITIONING: any stream with stream_configs[name].
    condition_on_theta=True gets its OWN FiLM conditioner (applied to
    the shared trunk features, before that stream's own bottleneck --
    other streams' own projections are completely unaffected), built
    from theta alone -- deliberately NOT also from dt. theta is a
    property of the STATE (which run produced this image -- the same
    for every window that image participates in), so a stream
    conditioned on it stays a well-defined, single, cacheable function
    of a snapshot -- exactly like an unconditioned stream, just with one
    more (still snapshot-level) input. dt is a property of a WINDOW
    (the same snapshot is the start of many windows, each its own dt) --
    conditioning a stream on dt as well would mean that stream is no
    longer a function of a snapshot at all, breaking the "encode once,
    cache forever" premise every downstream consumer (training/datasets.py's
    own bulk-encoding, every evaluation/check_*.py script) relies on. Not
    done here; any dt-dependent correction belongs downstream, where dt
    already lives explicitly (f_theta, g_theta), not pushed back into the
    encoder itself.
    """

    def __init__(
        self,
        input_size: int,
        stream_configs: dict[str, LatentStreamConfig],
        in_channels: int = 1,
        base_channels: int = 32,
        norm: str = "batch",
        use_skips: bool = False,
        n_theta: int = 1,
    ):
        super().__init__()

        if not stream_configs:
            raise ValueError("Encoder requires at least one entry in stream_configs")

        spatial_sizes = {cfg.spatial_size for cfg in stream_configs.values()}
        if len(spatial_sizes) != 1:
            raise ValueError(
                f"all streams must share one spatial_size (the shared trunk only "
                f"produces one bottleneck resolution), got {spatial_sizes} across "
                f"streams {list(stream_configs)}"
            )
        latent_spatial_size = spatial_sizes.pop()

        n_stages = math.log2(input_size / latent_spatial_size)
        if not n_stages.is_integer() or n_stages < 1:
            raise ValueError(
                f"input_size must be latent_spatial_size * 2^k for integer k >= 1 "
                f"(latent_spatial_size={latent_spatial_size}), got input_size={input_size}"
            )
        n_stages = int(n_stages)

        self.input_size = input_size
        self.latent_spatial_size = latent_spatial_size
        self.n_stages = n_stages
        self.use_skips = use_skips
        self.stream_configs = dict(stream_configs)
        self.n_theta = n_theta

        # channels[0] = input channels, channels[i] = output channels of stage i.
        # Doubling per stage is a starting choice, not dictated by the docs --
        # exposed via base_channels so it's easy to sweep.
        channels = [in_channels] + [base_channels * 2**i for i in range(n_stages)]
        self.channels = channels

        self.down_blocks = nn.ModuleList([
            DownBlock(channels[i], channels[i + 1], norm=norm)
            for i in range(n_stages)
        ])

        self.bottlenecks = nn.ModuleDict({
            name: nn.Conv2d(channels[-1], cfg.channels, kernel_size=1)
            for name, cfg in stream_configs.items()
        })

        # ONE conditioner per theta-conditioned stream, not shared --
        # each stream's own bottleneck is already independent (see this
        # class's own docstring on why), so its own conditioner is too;
        # nothing here for a stream that doesn't request conditioning
        # (empty dict entry simply doesn't exist, not a no-op module).
        self.theta_conditioners = nn.ModuleDict({
            name: _ThetaFiLMConditioner(n_theta=n_theta, n_channels=channels[-1])
            for name, cfg in stream_configs.items() if cfg.condition_on_theta
        })

    def forward(self, x: torch.Tensor, theta: torch.Tensor | None = None):
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            raise ValueError(
                f"Encoder built for input_size={self.input_size}, "
                f"got input of shape {tuple(x.shape[-2:])}"
            )
        if self.theta_conditioners and theta is None:
            raise ValueError(
                f"stream(s) {list(self.theta_conditioners)} require theta conditioning "
                f"(condition_on_theta=True in their own stream_configs), but forward() "
                f"was called with theta=None"
            )

        skips = []
        for down in self.down_blocks:
            x, features = down(x)
            skips.append(features)

        # dict comprehension, iterating self.bottlenecks (an nn.ModuleDict,
        # so insertion order -- Python 3.7+ dict order -- matches
        # stream_configs' own order, keeping z's key order predictable):
        # EACH entry built directly from its OWN bottleneck conv, applied
        # to the SAME shared trunk features (or, for a theta-conditioned
        # stream, that same trunk conditioned by THAT stream's own FiLM
        # module first -- every OTHER stream's own features are entirely
        # unaffected) -- this is the one place correctness between a
        # stream's NAME and its VALUE is actually established (see
        # latent_streams.decode_stream's own docstring for why nothing
        # downstream can re-establish it if this is wrong).
        z = {}
        for name, bottleneck in self.bottlenecks.items():
            feat = self.theta_conditioners[name](x, theta) if name in self.theta_conditioners else x
            z[name] = bottleneck(feat)

        if self.use_skips:
            return z, skips
        return z
