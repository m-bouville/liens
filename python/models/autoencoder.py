"""
Wraps a matched Encoder/Decoder pair built from one shared config, so
their shapes are guaranteed to line up.
"""

import torch
import torch.nn as nn

from .constants import LATENT_SPATIAL_SIZE
from .decoder import Decoder
from .encoder import Encoder
from .latent_streams import DEFAULT_STREAM_NAME, LatentStreamConfig, LatentStreamMode

_STREAM_NAME = DEFAULT_STREAM_NAME


class EncoderDecoderPair(nn.Module):
    """
    ONE decode PATHWAY: reads a single named stream's z out of
    encoder(x) (encoder may know about other streams too -- this
    pathway only ever touches its own), decodes it through decoder,
    and applies that stream's own output-scale correction.

    The scale exists because a decodable stream that ISN'T the
    reconstruction target (e.g. a time-derivative stream, C1) needs D
    to produce a very different-scale output than state -- state
    (~O(0.1-1)) vs a real physical derivative (~O(1e-4..1e-3),
    inherently far smaller than the state it's a rate of change OF) --
    while its INPUT to D is comparable in scale to any other stream's
    (both come from bottleneck convs built the same way). D itself has
    no way to know which stream produced whichever z it's holding, so
    its raw output is always state-scale regardless -- this pathway's
    own output-scale is what actually closes that gap, applied AFTER
    decode, not by distorting D's input.

    learnable_scale is decided ONCE, at construction, from the
    stream's own LatentStreamMode -- not a flag every call site has to
    remember to set correctly:
      - AUTOENCODER-mode (decoding a stream against ITS OWN input,
        i.e. a genuine reconstruction): scale is stored as a BUFFER,
        not a Parameter -- constant 1.0 (log_output_scale=0),
        structurally UNABLE to be trained, not merely defaulted there.
        A stream decoding as itself is, by definition, already at
        whatever scale D was built to produce; there is no other scale
        for it to calibrate to, so nothing should ever be able to move
        this away from 1.0, not even by accident.
      - DECODER-mode (decoded, but compared against something OTHER
        than its own input -- e.g. C1 vs a finite-difference
        derivative): scale is a real nn.Parameter, log-parameterized
        (guarantees positivity for free, and avoids the optimizer
        needing to approach zero asymptotically from the positive
        side -- there's no reason this scale would ever need to change
        sign, it's a pure magnitude correction).
      - PURE_LATENT streams are never decoded at all (see
        latent_streams.LatentStreamMode) -- constructing a pathway for
        one is a caller mistake, refused outright.

    Whether the scale is a buffer or a Parameter, callers can always
    read pathway.log_output_scale uniformly -- no branching on "is
    this one learnable" needed anywhere downstream.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, stream_name: str, mode: LatentStreamMode):
        super().__init__()
        if mode == LatentStreamMode.PURE_LATENT:
            raise ValueError(f"stream '{stream_name}' is pure_latent -- cannot build a decode "
                              f"pathway for a stream declared never-decodable")
        self.encoder = encoder
        self.decoder = decoder
        self.stream_name = stream_name
        self.mode = mode
        if mode == LatentStreamMode.AUTOENCODER:
            self.register_buffer("log_output_scale", torch.zeros(()))
        else:
            self.log_output_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, theta: torch.Tensor | None = None):
        # theta: only actually required if THIS pathway's own stream
        # requests conditioning (Encoder.forward's own check handles
        # that -- raises clearly if theta=None but the stream needs it,
        # silently ignored if given but the stream doesn't). Passed
        # through unconditionally rather than branching on self.mode
        # here too -- Encoder is the one place that actually knows
        # which streams are conditioned (via stream_configs), and is
        # already the single source of truth for that; duplicating the
        # check here would just be a second place it could drift out
        # of sync with the real config.
        z = self.encoder(x, theta=theta)[self.stream_name]
        x_recon = self.decoder(z) * torch.exp(self.log_output_scale)
        return x_recon, z


class Autoencoder(EncoderDecoderPair):
    """
    Constructs Encoder and Decoder together from a single set of
    hyperparameters -- unlike building them separately, where nothing
    enforces that input_size/output_size, base_channels, latent_channels,
    latent_spatial_size, and norm actually agree between the two.

    Deliberately the SIMPLE, single-stream case: same constructor
    signature as before the multi-stream (C0/C1) redesign -- a plain
    latent_channels: int, not a stream_configs dict. Internally this
    builds a single-entry stream_configs (one stream, named "state",
    mode=AUTOENCODER) and delegates to the same general Encoder every
    multi-stream model uses -- Autoencoder is the C0-only path (see the
    project's own design doc: C0 genuinely IS an autoencoder in the
    full sense; other streams, e.g. a time-derivative stream, are NOT,
    so they're never routed through this wrapper).

    IS an EncoderDecoderPair (a single AUTOENCODER-mode pathway, its
    own encoder/decoder) rather than a hand-maintained parallel
    implementation -- inheriting mode=AUTOENCODER from the parent is
    exactly what makes its log_output_scale a constant, non-trainable
    buffer (see EncoderDecoderPair's own docstring): a stream decoding
    as itself has no other scale to calibrate to, and that invariant
    is enforced by the type itself, not by every caller remembering to
    leave something unset.

    forward() returns (x_recon, z): both are needed downstream (z feeds
    LDS training -- stage 3 onward, refined jointly with the encoder in
    stage 4/5; x_recon feeds the reconstruction loss), so returning
    just one and forcing a second call would waste compute. Both are
    the single "state" stream's own tensors -- Autoencoder never
    exposes the underlying Encoder's dict-of-streams return shape;
    call self.encoder directly for that.
    """

    def __init__(
        self,
        size: int,
        channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 16,
        latent_spatial_size: int = LATENT_SPATIAL_SIZE,
        norm: str = "batch",
        use_skips: bool = False,
    ):
        self.size = size
        self.channels = channels
        self.latent_channels = latent_channels
        self.latent_spatial_size = latent_spatial_size
        self.use_skips = use_skips

        stream_configs = {
            _STREAM_NAME: LatentStreamConfig(
                name=_STREAM_NAME, channels=latent_channels,
                spatial_size=latent_spatial_size, mode=LatentStreamMode.AUTOENCODER,
            )
        }

        encoder = Encoder(
            input_size=size,
            in_channels=channels,
            base_channels=base_channels,
            stream_configs=stream_configs,
            norm=norm,
            use_skips=use_skips,
        )
        decoder = Decoder(
            output_size=size,
            out_channels=channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            latent_spatial_size=latent_spatial_size,
            norm=norm,
            use_skips=use_skips,
        )
        super().__init__(encoder, decoder, stream_name=_STREAM_NAME, mode=LatentStreamMode.AUTOENCODER)

    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.use_skips:
            z, skips = z
            return z[_STREAM_NAME], skips
        return z[_STREAM_NAME]

    def decode(self, z: torch.Tensor, skips: list[torch.Tensor] | None = None) -> torch.Tensor:
        return self.decoder(z, skips=skips)

    def forward(self, x: torch.Tensor):
        # Overrides EncoderDecoderPair's own forward() -- ONLY because
        # use_skips needs the separate encode()/decode() calls above
        # (skip tensors have to be threaded through explicitly);
        # log_output_scale is still applied, for consistency with
        # every other pathway, even though it's structurally always
        # exp(0)=1.0 here (see EncoderDecoderPair's own docstring) --
        # a harmless, always-identity multiply, not a special case.
        if self.use_skips:
            z, skips = self.encode(x)
            x_recon = self.decode(z, skips=skips)
        else:
            z = self.encode(x)
            x_recon = self.decode(z)
        return x_recon * torch.exp(self.log_output_scale), z


class MultiStreamAutoencoder(nn.Module):
    """
    Top-level container for the >1-stream case, where Autoencoder
    itself structurally cannot apply (it only ever knows about ONE
    AUTOENCODER-mode stream -- see Autoencoder's own docstring).

    Holds encoders/decoders as NAMED DICTS (nn.ModuleDict), not bare
    .encoder/.decoder attributes -- today there is exactly one shared
    trunk feeding every stream ({"shared": encoder}, {"shared":
    decoder}), but a dict-of-one is a trivial special case of a dict,
    not a different structure that would need revisiting if a future
    design ever wanted a separate trunk per stream (or per group of
    streams) -- the shape of this container doesn't have to change for
    that, only what's put in the dicts.

    pathways (nn.ModuleDict, one EncoderDecoderPair per stream) is
    where callers actually go for anything -- e.g.
    model.pathways["deriv"](x) -- rather than manually doing
    encoder(x)[name] then decoder(z) themselves at every call site the
    way pre-this-class code had to. That's the actual point of this
    class: the output-scale correction becomes automatic at the call
    site, not something every consumer has to remember to apply.

    decoder_for_stream: dict[str, str] | None, maps each stream NAME to
    which DECODER key its own pathway should read from. None (default)
    means every stream shares the single decoder in `decoders` --
    correct whenever there's genuinely only one decoder to share (the
    original design, still correct e.g. for Stage 2's C0/C1 sharing D).
    Explicit routing (e.g. {"state": "D0", "deriv": "D1"}, TWO separate
    decoder entries) is what a design with genuinely independent
    decoders per stream needs -- D1 trained without D0's own gradient
    ever touching it, and vice versa, rather than the two objectives
    fighting over the same decoder weights. The ENCODER side has no
    equivalent per-stream routing: the trunk is always genuinely
    shared, by design, across every stream this class has ever needed
    to support (a stream-specific decoder is a real, separate network;
    a stream-specific "trunk" would defeat the entire point of a
    shared-trunk architecture) -- so exactly one encoder key is
    required, not a dict of choices.

    Each pathway holds a direct reference to the SAME encoder/decoder
    objects also sitting in self.encoders/self.decoders -- reachable
    two ways through the module tree (model.encoders["shared"] and
    model.pathways["deriv"].encoder are the identical object when they
    share one decoder; with separate decoders, model.decoders["D1"]
    and model.pathways["deriv"].decoder are the identical object
    instead). This is correct, not a bug: PyTorch's own
    named_parameters()/state_dict() deduplicate by tensor/module
    identity, not by path, so a shared trunk (or a decoder reachable
    via decoders AND exactly one pathway) still only contributes its
    parameters once -- optimizer construction (model.parameters()),
    .train()/.eval(), and .to(device) all keep working as single calls
    on this container despite the redundant paths.
    """
    def __init__(self, encoders: dict[str, Encoder], decoders: dict[str, Decoder],
                 stream_configs: dict[str, LatentStreamConfig],
                 decoder_for_stream: dict[str, str] | None = None):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)

        if len(encoders) != 1:
            raise ValueError(
                f"MultiStreamAutoencoder requires exactly one shared encoder trunk -- got "
                f"{len(encoders)} encoder keys: {list(encoders)}. Per-stream encoder routing "
                f"isn't supported: the trunk is always genuinely shared by design, across every "
                f"stream this class has ever needed (see this class's own docstring)."
            )
        ((_encoder_key, shared_encoder),) = encoders.items()

        if decoder_for_stream is None:
            if len(decoders) != 1:
                raise ValueError(
                    f"decoder_for_stream was not given, but {len(decoders)} decoder keys exist "
                    f"({list(decoders)}) -- ambiguous which one each stream's pathway should use. "
                    f"Pass decoder_for_stream explicitly when there's more than one decoder."
                )
            ((only_decoder_key, _),) = decoders.items()
            decoder_for_stream = {name: only_decoder_key for name in stream_configs}

        self.pathways = nn.ModuleDict({
            name: EncoderDecoderPair(shared_encoder, decoders[decoder_for_stream[name]],
                                      stream_name=name, mode=cfg.mode)
            for name, cfg in stream_configs.items()
            if cfg.mode != LatentStreamMode.PURE_LATENT
        })
