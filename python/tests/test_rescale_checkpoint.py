"""
Tests for training/rescale_checkpoint.py.

The central test asserts the EXACT sets of transferred and freshly-initialised
parameters, not merely that loading raises no exception. That distinction is the
whole point: the dangerous outcome of a rescale is a silent partial success --
`load_state_dict(strict=False)` happily ignores anything that doesn't match, so
a test that only checks for absence of an error passes on a checkpoint that
transferred nothing at all.

Every fixture is a randomly-initialised model saved to a stage-1-shaped dict, so
these run on CPU in seconds with no trained checkpoint and no dataset.
"""
import math
from pathlib import Path

import pytest
import torch

from models.decoder import Decoder
from models.encoder import Encoder
from models.latent_streams import LatentStreamConfig, LatentStreamMode
from training.rescale_checkpoint import (
    describe_rescale, rescale_checkpoint_to_size, _shift_up_block_indices,
)
from training.stats_head import StatsHead

BASE_CHANNELS = 32
LATENT_CHANNELS = 8
LATENT_SPATIAL = 8
STAT_NAMES = ["avg_phi", "stdev_phi"]


def _stream_configs(name: str = "state"):
    """The recon stream, matching what `Autoencoder.__init__` builds.

    condition_on_theta=False is not a simplification: Autoencoder constructs its
    own LatentStreamConfig without that argument, so a stage-1 checkpoint's
    recon stream is NEVER theta-conditioned, and the real 32x32 checkpoint
    agrees (state: False, deriv: True). A fixture setting it True produces a
    checkpoint stage 1 cannot write, and one that fails the strict load into
    Autoencoder for a reason no real port would hit.
    """
    return {name: LatentStreamConfig(name=name, channels=LATENT_CHANNELS,
                                      spatial_size=LATENT_SPATIAL,
                                      mode=LatentStreamMode.AUTOENCODER,
                                      condition_on_theta=False)}


def _stage1_checkpoint(size: int, with_stats: bool = True, n_streams: int = 1) -> dict:
    """A stage-1-shaped checkpoint with random weights, matching train_stage1's
    own saved dict key for key (see its checkpoint-construction block).

    n_streams > 1 produces a STAGE-2-shaped checkpoint, and the key layout
    genuinely differs -- this is not cosmetic. Stage 1 saves a plain
    Autoencoder ("encoder."/"decoder." keys); stage 2 saves a
    MultiStreamAutoencoder, whose encoders/decoders are nn.ModuleDicts, giving
    "encoders.shared." and per-stream "decoders.D0.". Building the multi-stream
    fixture with stage-1 prefixes made the --from-stage2 tests pass against a
    layout no real checkpoint has, and hid a bug that failed immediately on the
    first real run.
    """
    configs = _stream_configs()
    if n_streams > 1:
        configs["deriv"] = LatentStreamConfig(name="deriv", channels=LATENT_CHANNELS,
                                               spatial_size=LATENT_SPATIAL,
                                               mode=LatentStreamMode.PURE_LATENT,
                                               condition_on_theta=True)
    encoder = Encoder(input_size=size, in_channels=1, base_channels=BASE_CHANNELS,
                       stream_configs=configs, n_theta=1)
    decoder = Decoder(output_size=size, out_channels=1, base_channels=BASE_CHANNELS,
                       latent_channels=LATENT_CHANNELS, latent_spatial_size=LATENT_SPATIAL)
    if n_streams > 1:
        model_state = {f"encoders.shared.{k}": v for k, v in encoder.state_dict().items()}
        model_state.update({f"decoders.D0.{k}": v for k, v in decoder.state_dict().items()})
    else:
        model_state = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
        model_state.update({f"decoder.{k}": v for k, v in decoder.state_dict().items()})

    stats_head = StatsHead(latent_channels=LATENT_CHANNELS, stat_names=STAT_NAMES,
                            latent_spatial=LATENT_SPATIAL)
    return {
        "model_state": model_state,
        "stats_head_state": stats_head.state_dict() if with_stats else None,
        "epoch": 42,
        "val_loss": 0.123,
        "val_loss_ema": 0.124,
        "normalized": False,
        "test_dirs": ["/data/64x64/T700_n010_s0"],
        "config": {
            "size": size, "base_channels": BASE_CHANNELS,
            "latent_channels": LATENT_CHANNELS, "latent_spatial_size": LATENT_SPATIAL,
            "stats_weight": 1.0,
            "stream_configs": {
                name: {"channels": cfg.channels, "spatial_size": cfg.spatial_size,
                        "mode": cfg.mode.value, "condition_on_theta": cfg.condition_on_theta}
                for name, cfg in configs.items()
            },
            "recon_stream_name": "state",
            **({"decoder_for_stream": {"state": "D0"}} if n_streams > 1 else {}),
        },
        "stats_config": {
            "stat_names": STAT_NAMES,
            "stats_mean": torch.zeros(len(STAT_NAMES)),
            "stats_std": torch.ones(len(STAT_NAMES)),
        } if with_stats else None,
    }


# --------------------------------------------------------------------
# the central claim: exactly these transfer, exactly those do not
# --------------------------------------------------------------------

def test_transferred_and_fresh_key_sets_are_exactly_as_predicted():
    """
    GUARDS a partial transfer that raises no error. load_state_dict(strict=False)
    silently ignores unmatched keys, so a test asserting only "no exception"
    would pass on a rescale that transferred nothing. Assert the SETS.
    """
    prev = _stage1_checkpoint(64)
    result = rescale_checkpoint_to_size(prev, to_size=128)

    assert result.from_size == 64 and result.to_size == 128
    assert result.n_stages_added == 1
    assert set(result.fresh) == {
        # no "encoder.theta_conditioners": the recon stream is unconditioned,
        # so the module does not exist (see _stream_configs)
        "encoder.down_blocks.3", "encoder.bottlenecks",
        "decoder.up_blocks.0", "decoder.unbottleneck",
    }
    assert {"encoder.down_blocks.0", "encoder.down_blocks.1",
             "encoder.down_blocks.2"} <= set(result.transferred)
    assert {"decoder.up_blocks.1", "decoder.up_blocks.2", "decoder.up_blocks.3",
             "decoder.output_conv"} <= set(result.transferred)
    # nothing may appear in both
    assert not (set(result.fresh) & set(result.transferred))


def test_transferred_weights_are_bit_identical_to_the_source():
    """
    The claim is that trained weights carry over UNCHANGED. Equality of shapes
    would not establish that -- a freshly initialised block has the same shape.
    """
    prev = _stage1_checkpoint(64)
    result = rescale_checkpoint_to_size(prev, to_size=128)
    new_state = result.checkpoint["model_state"]

    for i in (0, 1, 2):
        key = f"encoder.down_blocks.{i}.conv.block.0.weight"
        assert torch.equal(new_state[key], prev["model_state"][key]), key
    # decoder blocks are re-indexed by +1 (deepest-first ordering)
    for old_i, new_i in ((0, 1), (1, 2), (2, 3)):
        old_key = f"decoder.up_blocks.{old_i}.conv.block.0.weight"
        new_key = f"decoder.up_blocks.{new_i}.conv.block.0.weight"
        assert torch.equal(new_state[new_key], prev["model_state"][old_key]), new_key


def test_fresh_blocks_are_not_copies_of_anything_transferred():
    prev = _stage1_checkpoint(64)
    result = rescale_checkpoint_to_size(prev, to_size=128)
    new_state = result.checkpoint["model_state"]
    # the new deepest encoder block has no counterpart at 64x64 at all
    assert "encoder.down_blocks.3.conv.block.0.weight" in new_state
    assert "encoder.down_blocks.3.conv.block.0.weight" not in prev["model_state"]
    # the bottleneck changed shape, so it cannot have been copied
    old_bottleneck = prev["model_state"]["encoder.bottlenecks.state.weight"]
    new_bottleneck = new_state["encoder.bottlenecks.state.weight"]
    assert new_bottleneck.shape != old_bottleneck.shape
    assert new_bottleneck.shape[1] == 2 * old_bottleneck.shape[1]  # channels[-1] doubled


def test_up_block_reindexing_is_deepest_first():
    """
    GUARDS transferring decoder up_blocks by unchanged index. The decoder runs
    deepest-first, so a new stage inserts up_blocks[0] and pushes the rest
    later; keeping indices would silently attach each trained block to the wrong
    physical scale. Tested on the helper directly because the shapes at
    base_channels=32 happen to catch a one-off in the full path, and they would
    not for every channel configuration.
    """
    state = {"up_blocks.0.conv.w": torch.zeros(1), "up_blocks.1.conv.w": torch.ones(1),
             "output_conv.weight": torch.full((1,), 7.0)}
    shifted = _shift_up_block_indices(state, n_added=2, n_stages_new=4)
    assert set(shifted) == {"up_blocks.2.conv.w", "up_blocks.3.conv.w", "output_conv.weight"}
    assert torch.equal(shifted["up_blocks.2.conv.w"], torch.zeros(1))
    assert torch.equal(shifted["up_blocks.3.conv.w"], torch.ones(1))
    assert torch.equal(shifted["output_conv.weight"], torch.full((1,), 7.0))


def test_up_block_reindexing_refuses_to_overflow():
    state = {"up_blocks.3.conv.w": torch.zeros(1)}
    with pytest.raises(ValueError, match="more decoder blocks than the target size"):
        _shift_up_block_indices(state, n_added=1, n_stages_new=4)


# --------------------------------------------------------------------
# the rescaled model must actually run
# --------------------------------------------------------------------

@pytest.mark.parametrize("to_size", [128, 256])
def test_rescaled_model_round_trips_an_image_of_the_new_size(to_size):
    prev = _stage1_checkpoint(64)
    result = rescale_checkpoint_to_size(prev, to_size=to_size)
    result.encoder.eval(); result.decoder.eval()
    x = torch.randn(2, 1, to_size, to_size)
    theta = torch.zeros(2, 1)
    with torch.no_grad():
        latents = result.encoder(x, theta)
        z = latents["state"] if isinstance(latents, dict) else latents
        out = result.decoder(z)
    assert z.shape == (2, LATENT_CHANNELS, LATENT_SPATIAL, LATENT_SPATIAL)
    assert out.shape == x.shape


def test_returned_checkpoint_reloads_into_a_natively_built_model():
    """
    End-to-end: the returned dict must be loadable by a model built from scratch
    at the new size with strict=True. This is what makes it a drop-in
    resume_from for the existing pipeline rather than a special code path.
    """
    prev = _stage1_checkpoint(64)
    result = rescale_checkpoint_to_size(prev, to_size=128)

    encoder = Encoder(input_size=128, in_channels=1, base_channels=BASE_CHANNELS,
                       stream_configs=_stream_configs(), n_theta=1)
    decoder = Decoder(output_size=128, out_channels=1, base_channels=BASE_CHANNELS,
                       latent_channels=LATENT_CHANNELS, latent_spatial_size=LATENT_SPATIAL)
    state = result.checkpoint["model_state"]
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                              if k.startswith("encoder.")})
    decoder.load_state_dict({k[len("decoder."):]: v for k, v in state.items()
                              if k.startswith("decoder.")})


def test_parameter_count_matches_a_natively_built_model():
    prev = _stage1_checkpoint(64)
    result = rescale_checkpoint_to_size(prev, to_size=128)
    native_enc = Encoder(input_size=128, in_channels=1, base_channels=BASE_CHANNELS,
                          stream_configs=_stream_configs(), n_theta=1)
    native_dec = Decoder(output_size=128, out_channels=1, base_channels=BASE_CHANNELS,
                          latent_channels=LATENT_CHANNELS, latent_spatial_size=LATENT_SPATIAL)
    assert (sum(p.numel() for p in result.encoder.parameters())
            == sum(p.numel() for p in native_enc.parameters()))
    assert (sum(p.numel() for p in result.decoder.parameters())
            == sum(p.numel() for p in native_dec.parameters()))


# --------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------

def test_multi_stream_checkpoint_is_refused_by_default_with_the_reason():
    """
    GUARDS silently accepting a stage-2 checkpoint and quietly dropping its
    deriv stream. Dropping is the right behaviour when ASKED for (the trunk is
    worth porting), but doing it unasked would look like a full port and is not.
    """
    prev = _stage1_checkpoint(64, n_streams=2)
    with pytest.raises(ValueError, match="discarded"):
        rescale_checkpoint_to_size(prev, to_size=128)


@pytest.mark.parametrize("to_size", [64, 32, 96, 192])
def test_invalid_target_sizes_are_refused(to_size):
    prev = _stage1_checkpoint(64)
    with pytest.raises(ValueError):
        rescale_checkpoint_to_size(prev, to_size=to_size)


def test_stale_stats_normalisation_is_stripped_not_carried_forward():
    """
    GUARDS carrying stats_mean/stats_std forward. They are computed from the OLD
    sweep; at the new size autocorr_length's search cap moves (42 -> 84 for
    64 -> 128) and every stats target would be miscentred and mis-scaled with no
    error anywhere. Stripped rather than refused, because train_stage1
    recomputes them from its own dataset and never reads the resumed
    checkpoint's copy -- so the values must be UNAVAILABLE, not merely
    discouraged.
    """
    prev = _stage1_checkpoint(64, with_stats=True)
    assert prev["stats_config"]["stats_mean"] is not None  # the fixture really has them
    result = rescale_checkpoint_to_size(prev, to_size=128)
    stats = result.checkpoint["stats_config"]
    assert stats is not None
    assert stats["stat_names"] == STAT_NAMES      # names survive
    assert stats["stats_mean"] is None            # values do not
    assert stats["stats_std"] is None
    assert "STRIPPED" in describe_rescale(result)


def test_no_stats_config_needs_no_acknowledgement():
    prev = _stage1_checkpoint(64, with_stats=False)
    result = rescale_checkpoint_to_size(prev, to_size=128)
    assert result.checkpoint["stats_head_state"] is None


# --------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------

def test_checkpoint_records_where_it_came_from_and_drops_stale_fields():
    prev = _stage1_checkpoint(64)
    result = rescale_checkpoint_to_size(prev, to_size=128)
    cfg = result.checkpoint["config"]
    assert cfg["size"] == 128
    assert cfg["ported_from_size"] == 64
    # test_dirs point at 64x64 runs and would leak the wrong sweep downstream
    assert result.checkpoint["test_dirs"] == []
    # epoch/val_loss must not claim the old model's training history
    assert result.checkpoint["epoch"] == 0
    assert math.isinf(result.checkpoint["val_loss"])


def test_describe_rescale_names_both_categories_and_the_batchnorm_caveat():
    """
    The dangerous outcome here is a SILENT success: most of the load works and
    nothing says three quarters of the model is untrained. The summary is the
    only thing that will.
    """
    prev = _stage1_checkpoint(64)
    text = describe_rescale(rescale_checkpoint_to_size(prev, to_size=128))
    assert "64x64" in text and "128x128" in text
    assert "down_blocks.3" in text
    assert "BatchNorm" in text


def test_two_doublings_add_two_block_pairs():
    prev = _stage1_checkpoint(64)
    result = rescale_checkpoint_to_size(prev, to_size=256)
    assert result.n_stages_added == 2
    assert set(result.fresh) == {
        "encoder.down_blocks.3", "encoder.down_blocks.4", "encoder.bottlenecks",
        "decoder.up_blocks.0", "decoder.up_blocks.1", "decoder.unbottleneck",
    }


# --------------------------------------------------------------------
# BatchNorm re-estimation
# --------------------------------------------------------------------

def test_reestimate_batchnorm_recovers_the_new_size_statistics():
    """
    A transferred BatchNorm's running mean/var were estimated over 64x64
    batches. Feed data with a deliberately different mean and check the running
    estimate moves to it -- eval() mode uses these directly, so a stale value is
    a systematically wrong val_loss from epoch 0.
    """
    from training.rescale_checkpoint import reestimate_batchnorm_statistics
    prev = _stage1_checkpoint(64)
    result = rescale_checkpoint_to_size(prev, to_size=128)

    bn = [m for m in result.encoder.modules()
          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)][0]
    bn.running_mean.fill_(99.0)  # a value no data would produce

    theta = torch.zeros(2, 1)
    batches = [torch.randn(2, 1, 128, 128) for _ in range(4)]
    seen = reestimate_batchnorm_statistics(result.encoder, batches,
                                            theta_for=lambda _b: theta)
    assert seen == 4
    assert bn.running_mean.abs().max() < 10.0, "stale value should have been replaced"


def test_reestimate_batchnorm_is_order_independent():
    """
    GUARDS leaving `momentum` at its default, which makes the result an
    exponential moving average -- so the answer would depend on which batch
    happened to come last. Setting momentum=None gives a cumulative average,
    i.e. the exact mean over everything seen.
    """
    from training.rescale_checkpoint import reestimate_batchnorm_statistics
    prev = _stage1_checkpoint(64)
    torch.manual_seed(0)
    batches = [torch.randn(2, 1, 128, 128) * (i + 1) for i in range(4)]
    theta = torch.zeros(2, 1)

    means = []
    for order in (batches, list(reversed(batches))):
        result = rescale_checkpoint_to_size(prev, to_size=128)
        bn = [m for m in result.encoder.modules()
              if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)][0]
        reestimate_batchnorm_statistics(result.encoder, order, theta_for=lambda _b: theta)
        means.append(bn.running_var.clone())
    assert torch.allclose(means[0], means[1], rtol=1e-5)


def test_reestimate_batchnorm_restores_momentum_and_training_mode():
    from training.rescale_checkpoint import reestimate_batchnorm_statistics
    result = rescale_checkpoint_to_size(_stage1_checkpoint(64), to_size=128)
    bn = [m for m in result.encoder.modules()
          if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)][0]
    before = bn.momentum
    result.encoder.eval()
    theta = torch.zeros(2, 1)
    reestimate_batchnorm_statistics(result.encoder, [torch.randn(2, 1, 128, 128)],
                                     theta_for=lambda _b: theta)
    assert bn.momentum == before
    assert not result.encoder.training, "eval() mode must be restored"


def test_reestimate_batchnorm_refuses_to_finish_with_zero_batches():
    """
    GUARDS silently returning after an empty iterable. The statistics have
    already been RESET at that point, so every BatchNorm would be left at
    mean 0 / var 1 -- worse than the stale values, and undetectable downstream.
    """
    from training.rescale_checkpoint import reestimate_batchnorm_statistics
    result = rescale_checkpoint_to_size(_stage1_checkpoint(64), to_size=128)
    with pytest.raises(ValueError, match="zero batches"):
        reestimate_batchnorm_statistics(result.encoder, iter([]),
                                         theta_for=lambda _b: torch.zeros(2, 1))


# --------------------------------------------------------------------
# the CLI entry point
# --------------------------------------------------------------------

def test_port_writes_a_checkpoint_that_reloads(tmp_path):
    from training.port_checkpoint import port_checkpoint
    src = tmp_path / "64x64-stage1.pt"
    torch.save(_stage1_checkpoint(64), src)
    out = port_checkpoint(src, to_size=128, output_path=tmp_path / "ported.pt")
    assert out.exists()
    written = torch.load(out, map_location="cpu", weights_only=True)
    assert written["config"]["size"] == 128
    assert written["config"]["ported_from_size"] == 64


def test_default_output_is_named_for_what_it_is_not_for_its_source(tmp_path):
    """
    GUARDS deriving the output name from the source. Porting a stage-2
    checkpoint that way produced checkpoints/stage2/128x128-stage2.pt -- a
    stage-1 input sitting in the stage-2 directory under a stage-2 name. That
    is dangerous, not untidy: the file is single-stream, so
    extend_state_checkpoint_with_deriv_stream would accept it and stage 2 would
    build a deriv head on a 75%-random trunk with nothing flagging it.
    """
    from training.port_checkpoint import _default_output_path
    for src in ("ck/stage1/64x64-stage1.pt", "ck/stage2/64x64-stage2.pt", "ck/anything.pt"):
        out = _default_output_path(Path(src), 128)
        assert out.name == "128x128-ported.pt", src
        assert out.parent.name == "stage1", src
    assert _default_output_path(Path("ck/stage2/64x64-stage2.pt"), 256).name \
        == "256x256-ported.pt"


def test_dry_run_writes_nothing(tmp_path):
    from training.port_checkpoint import port_checkpoint
    src = tmp_path / "64x64-stage1.pt"
    torch.save(_stage1_checkpoint(64), src)
    out = port_checkpoint(src, to_size=128, output_path=tmp_path / "ported.pt", dry_run=True)
    assert not out.exists()


def test_port_warns_when_batchnorm_is_not_re_estimated(tmp_path, capsys):
    """
    Skipping re-estimation is the DEFAULT and is usually fine, but it makes the
    ported model's val_loss look worse than it is -- which is exactly what a
    failed transfer looks like. The warning is what separates the two.
    """
    from training.port_checkpoint import port_checkpoint
    src = tmp_path / "64x64-stage1.pt"
    torch.save(_stage1_checkpoint(64), src)
    port_checkpoint(src, to_size=128, dry_run=True)
    out = capsys.readouterr().out
    assert "NOT re-estimated" in out
    assert "val_loss will look worse" in out


def test_ported_checkpoint_does_not_inherit_a_trained_val_loss(tmp_path):
    """
    GUARDS carrying val_loss forward. A ported checkpoint is ~75% random init;
    if it claimed the source's val_loss, every "is this better than before"
    comparison downstream -- including the checkpoint-saving criterion -- would
    start from a number the model cannot reproduce.
    """
    from training.port_checkpoint import port_checkpoint
    src = tmp_path / "64x64-stage1.pt"
    prev = _stage1_checkpoint(64)
    torch.save(prev, src)
    written = torch.load(port_checkpoint(src, to_size=128, output_path=tmp_path / "p.pt"),
                          map_location="cpu", weights_only=True)
    assert written["val_loss"] != prev["val_loss"]
    assert math.isinf(written["val_loss"])


# --------------------------------------------------------------------
# porting the shared trunk from a later checkpoint
# --------------------------------------------------------------------

def test_multi_stream_is_refused_unless_explicitly_allowed():
    prev = _stage1_checkpoint(64, n_streams=2)
    with pytest.raises(ValueError, match="keep_trunk_from_multi_stream"):
        rescale_checkpoint_to_size(prev, to_size=128)


def test_trunk_ports_from_a_stage2_checkpoint_and_the_deriv_stream_is_dropped():
    """
    Stage 2 trains the shared trunk as well as the deriv head, so a stage-2
    checkpoint's down_blocks carry training the stage-1 checkpoint never saw --
    the one part of a later checkpoint that survives a rescale. The deriv
    bottleneck does not: it reads features produced by the newly-added,
    randomly-initialised deepest block, so new channel i has no relationship to
    old channel i.
    """
    prev = _stage1_checkpoint(64, n_streams=2)
    result = rescale_checkpoint_to_size(prev, to_size=128,
                                         keep_trunk_from_multi_stream=True)
    new_state = result.checkpoint["model_state"]

    # the trunk really came across, bit for bit -- note the SOURCE key uses the
    # multi-stream layout while the OUTPUT is stage-1-shaped
    for i in (0, 1, 2):
        key = f"encoder.down_blocks.{i}.conv.block.0.weight"
        src = f"encoders.shared.down_blocks.{i}.conv.block.0.weight"
        assert torch.equal(new_state[key], prev["model_state"][src]), key
    # and the output is single-stream, which is what stage 2 requires as input
    assert "encoder.bottlenecks.deriv.weight" not in new_state
    assert "encoder.bottlenecks.state.weight" in new_state
    assert list(result.checkpoint["config"]["stream_configs"]) == ["state"]


def test_stage2_port_and_stage1_port_differ_only_in_the_trunk():
    """
    Makes the actual benefit concrete: porting from stage 2 rather than stage 1
    changes the trunk weights and nothing else about the resulting checkpoint's
    shape. If the two were identical there would be no reason to offer the
    option.
    """
    stage1 = _stage1_checkpoint(64, n_streams=1)
    stage2 = _stage1_checkpoint(64, n_streams=2)  # independently initialised
    from_1 = rescale_checkpoint_to_size(stage1, to_size=128)
    from_2 = rescale_checkpoint_to_size(stage2, to_size=128,
                                         keep_trunk_from_multi_stream=True)
    assert set(from_1.checkpoint["model_state"]) == set(from_2.checkpoint["model_state"])
    key = "encoder.down_blocks.0.conv.block.0.weight"
    assert not torch.equal(from_1.checkpoint["model_state"][key],
                            from_2.checkpoint["model_state"][key])
    # and the output layout is stage-1-shaped in BOTH cases, whatever went in
    assert not any(k.startswith("encoders.") for k in from_2.checkpoint["model_state"])


def test_port_backs_up_an_existing_output_before_overwriting(tmp_path, capsys):
    """
    The port writes into checkpoints/stage1/ alongside real trained
    checkpoints, so a second port -- after fixing the source, or at a different
    size -- would otherwise destroy the first silently.
    """
    from training.port_checkpoint import port_checkpoint
    src = tmp_path / "64x64-stage1.pt"
    torch.save(_stage1_checkpoint(64), src)
    dest = tmp_path / "128x128-ported.pt"
    first = port_checkpoint(src, to_size=128, output_path=dest)
    assert first.exists()
    port_checkpoint(src, to_size=128, output_path=dest)
    assert "before this run" in capsys.readouterr().out  # printed by the shared helper
    backups = [p for p in tmp_path.iterdir() if p.name.startswith("128x128-ported-")]
    assert len(backups) == 1, [p.name for p in tmp_path.iterdir()]


def test_multi_stream_fixture_really_uses_the_multi_stream_key_layout():
    """
    GUARDS the fixture itself. Stage 1 and stage 2 save under DIFFERENT key
    prefixes, and a multi-stream fixture built with stage-1 prefixes makes every
    --from-stage2 test pass against a layout no real checkpoint has -- which is
    exactly what happened, and the resulting bug failed on the first real run.
    Assert the fixture's own shape, not only the behaviour it feeds.
    """
    stage1 = _stage1_checkpoint(64, n_streams=1)
    stage2 = _stage1_checkpoint(64, n_streams=2)
    assert any(k.startswith("encoder.") for k in stage1["model_state"])
    assert not any(k.startswith("encoders.") for k in stage1["model_state"])
    assert any(k.startswith("encoders.shared.") for k in stage2["model_state"])
    assert any(k.startswith("decoders.D0.") for k in stage2["model_state"])
    assert not any(k.startswith("encoder.") for k in stage2["model_state"])
    assert stage2["config"]["decoder_for_stream"] == {"state": "D0"}


def test_an_unrecognised_key_layout_is_reported_not_silently_empty():
    """
    _strip_prefix returns {} on a mismatch rather than raising, so a future
    third layout would otherwise surface as "every key is missing" at the load
    call, several frames from the cause.
    """
    prev = _stage1_checkpoint(64)
    prev["model_state"] = {f"backbone.{k}": v for k, v in prev["model_state"].items()}
    with pytest.raises(ValueError, match="No encoder weights found"):
        rescale_checkpoint_to_size(prev, to_size=128)


# --------------------------------------------------------------------
# against a REAL checkpoint, not a fixture
# --------------------------------------------------------------------

REAL_CHECKPOINT = Path(__file__).parent / "data" / "32x32-stage2.pt"
_needs_real = pytest.mark.skipif(not REAL_CHECKPOINT.exists(),
                                  reason="real 32x32 stage-2 checkpoint not present")


@_needs_real
def test_real_stage2_checkpoint_ports_and_the_output_is_single_stream():
    """
    The fixtures reconstruct what a checkpoint looks like; this asserts against
    one that a real training run actually wrote. Two bugs so far came from
    fixtures that were subtly unlike the real artifact (an int64 column typed as
    float, a multi-stream state_dict written with stage-1 prefixes), so a real
    file in the loop is worth its few hundred kB.
    """
    prev = torch.load(REAL_CHECKPOINT, map_location="cpu", weights_only=True)
    assert any(k.startswith("encoders.shared.") for k in prev["model_state"])
    assert any(k.startswith("pathways.") for k in prev["model_state"])  # duplicate views

    result = rescale_checkpoint_to_size(REAL_CHECKPOINT, to_size=64,
                                         keep_trunk_from_multi_stream=True)
    state = result.checkpoint["model_state"]
    assert list(result.checkpoint["config"]["stream_configs"]) == ["state"]
    # the output is stage-1-shaped: no ModuleDict wrapping, no pathway
    # duplicates, plus Autoencoder's own top-level log_output_scale buffer
    assert {k.split(".")[0] for k in state} == {"encoder", "decoder", "log_output_scale"}
    for i in (0, 1):
        assert torch.equal(state[f"encoder.down_blocks.{i}.conv.block.0.weight"],
                            prev["model_state"][f"encoders.shared.down_blocks.{i}."
                                                 f"conv.block.0.weight"])


@_needs_real
def test_fresh_list_omits_theta_conditioners_when_the_stream_is_unconditioned():
    """
    GUARDS an unconditional "encoder.theta_conditioners" in the fresh list. This
    real checkpoint's recon stream has condition_on_theta=False, so the module
    does not exist -- reporting it as reinitialised describes a module that is
    not there.
    """
    prev = torch.load(REAL_CHECKPOINT, map_location="cpu", weights_only=True)
    assert prev["config"]["stream_configs"]["state"]["condition_on_theta"] is False
    result = rescale_checkpoint_to_size(REAL_CHECKPOINT, to_size=64,
                                         keep_trunk_from_multi_stream=True)
    assert "encoder.theta_conditioners" not in result.fresh
    assert not any("theta_conditioners" in k for k in result.checkpoint["model_state"])


def test_ported_checkpoint_loads_STRICTLY_into_the_class_train_stage1_builds():
    """
    THE end-to-end guard, and the one that caught a real bug. train_stage1 does
    `ae.load_state_dict(prev["model_state"])` on an `Autoencoder` with
    strict=True by default, so the key set must match EXACTLY -- not merely be
    loadable by Encoder and Decoder separately, which is what the earlier
    reload test checked and what a hand-built encoder.*/decoder.* dict
    satisfies.

    Autoencoder owns `log_output_scale`, a buffer on EncoderDecoderPair that
    belongs to neither submodule. Enumerating the submodules by hand dropped it
    and the port failed at the first real resume with a bare
    "Missing key(s): log_output_scale".
    """
    from models.autoencoder import Autoencoder
    prev = _stage1_checkpoint(64)
    result = rescale_checkpoint_to_size(prev, to_size=128)
    ae = Autoencoder(size=128, channels=1, base_channels=BASE_CHANNELS,
                      latent_channels=LATENT_CHANNELS, latent_spatial_size=LATENT_SPATIAL)
    ae.load_state_dict(result.checkpoint["model_state"])  # strict
    assert "log_output_scale" in result.checkpoint["model_state"]


@_needs_real
def test_real_checkpoint_ports_and_loads_strictly_into_an_autoencoder():
    """Same, against a checkpoint a real training run wrote."""
    from models.autoencoder import Autoencoder
    prev = torch.load(REAL_CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = prev["config"]
    result = rescale_checkpoint_to_size(REAL_CHECKPOINT, to_size=64,
                                         keep_trunk_from_multi_stream=True)
    ae = Autoencoder(size=64, channels=1, base_channels=cfg["base_channels"],
                      latent_channels=cfg["latent_channels"],
                      latent_spatial_size=cfg["latent_spatial_size"])
    ae.load_state_dict(result.checkpoint["model_state"])  # strict
    with torch.no_grad():
        out = ae(torch.randn(2, 1, 64, 64))
    assert (out[0] if isinstance(out, tuple) else out).shape == (2, 1, 64, 64)


@_needs_real
def test_latent_channels_mismatch_fails_loudly_at_load():
    """
    A params file disagreeing with the ported checkpoint on latent_channels is
    a real trap: train_stage1 takes those from the PARAMS FILE, not from the
    checkpoint. Documented here as a test so the failure mode is on record --
    it is loud, which is the important part.
    """
    from models.autoencoder import Autoencoder
    cfg = torch.load(REAL_CHECKPOINT, map_location="cpu", weights_only=True)["config"]
    result = rescale_checkpoint_to_size(REAL_CHECKPOINT, to_size=64,
                                         keep_trunk_from_multi_stream=True)
    wrong = Autoencoder(size=64, channels=1, base_channels=cfg["base_channels"],
                         latent_channels=cfg["latent_channels"] + 2,
                         latent_spatial_size=cfg["latent_spatial_size"])
    with pytest.raises(RuntimeError, match="size mismatch"):
        wrong.load_state_dict(result.checkpoint["model_state"])


def test_fixture_recon_stream_matches_what_Autoencoder_actually_builds():
    """
    GUARDS the fixture. Autoencoder builds its recon stream WITHOUT
    condition_on_theta, so no stage-1 checkpoint has a conditioned recon stream
    -- a fixture that sets it True tests a shape the pipeline cannot produce.
    Verified against the class rather than asserted from memory.
    """
    from models.autoencoder import Autoencoder
    ae = Autoencoder(size=64, channels=1, base_channels=BASE_CHANNELS,
                      latent_channels=LATENT_CHANNELS, latent_spatial_size=LATENT_SPATIAL)
    assert len(ae.encoder.theta_conditioners) == 0
    assert not any(cfg.condition_on_theta for cfg in _stream_configs().values())
