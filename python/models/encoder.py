"""
Convolutional encoder: maps a real-space microstructure to one or more
compact latent representations (streams) -- see the project's own
C0/C1 design doc for why more than one.
"""

from models.constants import N_THETA

import math

import torch
import torch.nn as nn

from .blocks import DownBlock
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
        n_theta: int = N_THETA,
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

        # Optional nonlinear residual head H per stream: z = B(y) + H(y).
        # 3x3 -> GELU -> 3x3, the SECOND conv zero-initialised so H(y)=0 at
        # init and the stream is byte-identical to the pure-linear (1x1 B)
        # head until training grows the residual. GELU (not ReLU) keeps the
        # map differentiable everywhere, so perturbation linearity degrades
        # smoothly with ||H|| instead of fracturing at kinks. Built only for
        # streams whose config asks for it; absent entirely otherwise, so a
        # "linear" stream's state_dict has no H.* keys (old checkpoints load
        # unchanged, new "linear" streams stay identical to them).
        self.residual_heads = nn.ModuleDict()
        for name, cfg in stream_configs.items():
            if getattr(cfg, "head_kind", "linear") == "residual":
                h = cfg.head_hidden
                conv2 = nn.Conv2d(h, cfg.channels, kernel_size=3, padding=1)
                nn.init.zeros_(conv2.weight)
                nn.init.zeros_(conv2.bias)
                self.residual_heads[name] = nn.Sequential(
                    nn.Conv2d(channels[-1], h, kernel_size=3, padding=1),
                    nn.GELU(),
                    conv2,
                )

        # ONE conditioner per theta-conditioned stream, not shared --
        # each stream's own bottleneck is already independent (see this
        # class's own docstring on why), so its own conditioner is too;
        # nothing here for a stream that doesn't request conditioning
        # (empty dict entry simply doesn't exist, not a no-op module).
        self.theta_conditioners = nn.ModuleDict({
            name: _ThetaFiLMConditioner(n_theta=n_theta, n_channels=channels[-1])
            for name, cfg in stream_configs.items() if cfg.condition_on_theta
        })

        # Per-stream scale on the gradient each stream sends BACK into the
        # shared trunk. Forward value is untouched (straight-through); only
        # the backward flow is scaled. 1.0 = today's behaviour (every stream
        # trains the trunk fully). Set a stream to 0.0 to let it read the
        # trunk but not reshape it -- the mechanism for training the deriv
        # (z1) head against L_deriv WITHOUT that loss's frame-scale noise
        # roughening z0's own trajectory through the shared trunk (measured:
        # z0 velocity coherence falls from +0.39 at stage 1 to +0.04 after a
        # stage-2 run whose only trunk-touching temporal loss is L_deriv).
        # A plain python dict, not a buffer: it is a training-time control,
        # not model state, and must not enter the checkpoint or the
        # architecture fingerprint.
        self._trunk_grad_scale = {name: 1.0 for name in stream_configs}

    def set_trunk_grad_scale(self, stream_name: str, scale: float) -> None:
        """Scale the gradient `stream_name` contributes to the shared trunk.

        1.0 leaves training unchanged; 0.0 fully isolates the trunk from that
        stream's loss (the stream still reads the trunk forward). Intermediate
        values are a partial leak, for sweeping how much trunk plasticity the
        stream actually needs.
        """
        if stream_name not in self._trunk_grad_scale:
            raise KeyError(
                f"unknown stream {stream_name!r}; have "
                f"{sorted(self._trunk_grad_scale)}")
        self._trunk_grad_scale[stream_name] = float(scale)

    def head_nonlinearity(self) -> dict:
        """Per stream, a scale-free measure of how far the nonlinear residual
        head has moved from the pure-linear map. 0.0 means still linear (a
        linear head, or a residual head at init). This is the scalar the
        three smoothness properties are traded against -- report it, don't
        assume it stays small.

        Measured on the OUTPUT conv of H (the zero-initialised one): it is
        what actually injects nonlinearity into z, so its norm is 0 exactly
        when H(y)=0, whereas H's input conv is random-initialised and would
        make a fresh residual head look nonlinear when it is not.
        Normalised by ||B||, both Frobenius norms over weights and biases."""
        import torch as _torch
        out = {}
        with _torch.no_grad():
            for name, bottleneck in self.bottlenecks.items():
                if name not in self.residual_heads:
                    out[name] = 0.0
                    continue
                b_norm = _torch.sqrt(sum(p.pow(2).sum()
                                          for p in bottleneck.parameters()))
                out_conv = self.residual_heads[name][-1]  # the zero-init conv
                h_norm = _torch.sqrt(sum(p.pow(2).sum()
                                          for p in out_conv.parameters()))
                out[name] = float(h_norm / b_norm.clamp_min(1e-12))
        return out

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
            # Straight-through gradient scaling on the trunk features this
            # stream consumes: forward value == x exactly, backward gradient
            # == scale * (dL/dx). scale 1.0 is a no-op (the common path);
            # scale 0.0 makes x_stream = x.detach(), so this stream's loss
            # cannot move the trunk. See set_trunk_grad_scale.
            scale = self._trunk_grad_scale[name]
            x_stream = x if scale == 1.0 else x.detach() + scale * (x - x.detach())
            feat = self.theta_conditioners[name](x_stream, theta) \
                if name in self.theta_conditioners else x_stream
            out = bottleneck(feat)
            if name in self.residual_heads:
                # z = B(y) + H(y). H reads the SAME theta-conditioned,
                # trunk-grad-scaled features B does, so the leak knob governs
                # both branches identically.
                out = out + self.residual_heads[name](feat)
            z[name] = out

        if self.use_skips:
            return z, skips
        return z


def zero_pad_theta_columns(state_dict: dict, model: "nn.Module") -> dict:
    """Upgrade a checkpoint trained with FEWER theta features to a model that
    expects more (n_theta grew, e.g. 1 -> 2 when log(T0-T) was added).

    Every theta-conditioned module (encoder FiLM conditioners, f_theta's MLP)
    takes theta as the LAST n_theta columns of its FIRST Linear's weight. A
    model with more theta features has wider first-Linear weights; the extra
    columns are the new features. Zero-padding them makes the loaded model
    BIT-IDENTICAL in function to the checkpoint (the new feature contributes
    nothing until trained), the same backward-compatible contract the
    residual head uses on its own zero-init output conv.

    Returns a NEW state_dict with the relevant weights right-zero-padded to
    the model's widths. Keys whose width already matches are passed through
    untouched, so this is a no-op for same-n_theta loads. Only ever GROWS a
    weight (raises if the checkpoint is WIDER than the model -- that is a
    real mismatch, not an upgrade).
    """
    import torch
    model_sd = model.state_dict()
    out = dict(state_dict)
    for key, ckpt_w in state_dict.items():
        if key not in model_sd:
            continue
        model_w = model_sd[key]
        if ckpt_w.shape == model_w.shape:
            continue
        # only first-Linear weights differ, and only in their last (input) dim
        if (ckpt_w.dim() == 2 and model_w.dim() == 2
                and ckpt_w.shape[0] == model_w.shape[0]
                and ckpt_w.shape[1] < model_w.shape[1]):
            pad = model_w.shape[1] - ckpt_w.shape[1]
            zeros = torch.zeros(ckpt_w.shape[0], pad,
                                 dtype=ckpt_w.dtype, device=ckpt_w.device)
            out[key] = torch.cat([ckpt_w, zeros], dim=1)
        elif ckpt_w.shape[1] > model_w.shape[1]:
            raise ValueError(
                f"checkpoint weight {key} is WIDER ({tuple(ckpt_w.shape)}) than "
                f"the model ({tuple(model_w.shape)}) -- that is a real "
                f"mismatch, not a theta-feature upgrade (which only ever adds "
                f"columns). Refusing to silently truncate.")
    return out
