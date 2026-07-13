"""
Componentized checkpoint structure for stage 4/5, which is the first
point in this pipeline where checkpoints from different lineages need
to be combined: stage 2's (E, D, stats_head) and stage 3's (f), merged
into one model, then carried forward together. Stages 1->2->3 are a
strict linear chain -- one ancestor each, never a merge -- so their
existing, monolithic checkpoint dicts (see train_ae.py/train_lds.py)
have been fine as-is and are NOT changed here.

This module is an ADAPTER, not a new save format: it reads checkpoints
in their existing, unchanged shapes and produces a componentized VIEW
for stage 4/5 to build from. Nothing about stage 1/2/3's own
checkpoint format, or the evaluation scripts that already read it
directly, needs to change.
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from models.constants import LATENT_SPATIAL_SIZE
from models.latent_streams import (
    cross_check_stream_configs_against_state_dict, resolve_stream_configs_from_checkpoint_config,
)


@dataclass
class ComponentCheckpoint:
    """
    One trained component's state, self-contained enough to be loaded
    into a fresh instance of its OWN model class, independent of
    whatever else was in the checkpoint dict it was extracted from.

    state_dict: ready to load directly into a standalone instance of
    this component's own class (e.g. Encoder(), not Autoencoder()) --
    any container prefix (e.g. "encoder.") has already been stripped.
    config: constructor arguments needed to instantiate that class
    correctly (e.g. {"size": 64, "latent_channels": 8, ...}).
    provenance: everything else worth keeping for traceability (source
    checkpoint path, epoch, val_loss, etc.) -- informational only, never
    read by anything that reconstructs the model itself.
    """
    state_dict: dict
    config: dict
    provenance: dict


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    """
    'encoder.down_blocks.0.conv1.weight' -> 'down_blocks.0.conv1.weight'
    for prefix='encoder' -- so the result loads directly into a
    standalone Encoder()/Decoder() instance, not just back into the
    combined Autoencoder it was saved from.
    """
    full_prefix = prefix + "."
    return {k[len(full_prefix):]: v for k, v in state_dict.items() if k.startswith(full_prefix)}


def load_ae_components(checkpoint_path: str | Path,
                        device: str | None = None) -> dict[str, ComponentCheckpoint]:
    """
    Reads an EXISTING stage-1 or stage-2 checkpoint (both share the same
    shape -- see train_autoencoder()/train_stage2()'s save calls) and
    splits it into separate components: 'encoder', 'decoder', and
    'stats_head' (omitted if the checkpoint has none, e.g. it was
    trained with stats_weight <= 0).
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=True)
    model_cfg = checkpoint["config"]
    ae_state = checkpoint["model_state"]

    shared_provenance = {
        "source_checkpoint": str(checkpoint_path.resolve()),
        "epoch": checkpoint["epoch"],
        "val_loss": checkpoint["val_loss"],
    }
    ae_component_config = {
        "size": model_cfg["size"], "base_channels": model_cfg["base_channels"],
        "latent_channels": model_cfg["latent_channels"],
        "latent_spatial_size": model_cfg.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
        "stream_configs": model_cfg.get("stream_configs"),
        "recon_stream_name": model_cfg.get("recon_stream_name"),
    }
    # Cross-checked against the ACTUAL encoder weights (ae_state, still
    # "encoder."-prefixed at this point), not just trusted from
    # model_cfg -- see cross_check_stream_configs_against_state_dict's
    # own docstring for the real failure mode this guards against.
    # Corrected back into ae_component_config as plain dicts/strings
    # (not LatentStreamConfig/LatentStreamMode objects), matching the
    # rest of this config's own serializable-values convention.
    _resolved_streams, _resolved_recon_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    _resolved_streams, _resolved_recon_name = cross_check_stream_configs_against_state_dict(
        _resolved_streams, _resolved_recon_name, ae_state,
    )
    ae_component_config["stream_configs"] = {
        name: {"channels": cfg.channels, "spatial_size": cfg.spatial_size, "mode": cfg.mode.value}
        for name, cfg in _resolved_streams.items()
    }
    ae_component_config["recon_stream_name"] = _resolved_recon_name

    components = {
        "encoder": ComponentCheckpoint(
            state_dict=_strip_prefix(ae_state, "encoder"),
            config=dict(ae_component_config),
            provenance=dict(shared_provenance),
        ),
        "decoder": ComponentCheckpoint(
            state_dict=_strip_prefix(ae_state, "decoder"),
            config=dict(ae_component_config),
            provenance=dict(shared_provenance),
        ),
    }

    stats_head_state = checkpoint.get("stats_head_state")
    stats_config = checkpoint.get("stats_config")
    if stats_head_state is not None and stats_config is not None:
        components["stats_head"] = ComponentCheckpoint(
            state_dict=stats_head_state,
            config={"latent_channels": model_cfg["latent_channels"],
                    "latent_spatial_size": model_cfg.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
                    "stat_names": stats_config["stat_names"]},
            provenance={**shared_provenance, "stats_mean": stats_config["stats_mean"],
                        "stats_std": stats_config["stats_std"]},
        )

    return components


def load_lds_component(checkpoint_path: str | Path, device: str | None = None) -> ComponentCheckpoint:
    """Reads an EXISTING stage-3 checkpoint (unchanged format) into one ComponentCheckpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=True)
    return ComponentCheckpoint(
        state_dict=checkpoint["model_state"],
        config=dict(checkpoint["config"]),
        provenance={
            "source_checkpoint": str(checkpoint_path.resolve()),
            "epoch": checkpoint["epoch"],
            "val_loss": checkpoint["val_loss"],
            "ae_checkpoint": checkpoint.get("ae_checkpoint"),
        },
    )


def validate_component_compatibility(components: dict[str, ComponentCheckpoint]) -> None:
    """
    Cross-checks that every component agrees on latent_channels AND
    latent_spatial_size -- together, the two things that MUST match for
    the encoder's output to be a valid input to f_theta, and for the
    decoder to be able to invert it. (latent_spatial_size defaults to
    models.constants.LATENT_SPATIAL_SIZE almost everywhere it's read,
    so in practice this mostly catches an OLD LDS checkpoint -- saved
    before this field existed, so load_lds_component's wholesale
    dict(checkpoint["config"]) copy genuinely has no key for it at all,
    unlike encoder/decoder/stats_head which always get it injected via
    a fallback default at read time -- being combined with a NEWER
    ancestor that used a non-default value; see the "in c.config" guard
    below, mirroring latent_channels' own.)

    This check is NEW: stages 1->2->3 are a strict linear chain (one
    ancestor each), so nothing before stage 4 ever needed to verify that
    two checkpoints from different lineages actually agree on anything.
    Without this, a mismatch would surface as a confusing shape error
    deep inside a forward pass instead of a clear message at load time.
    """
    latent_channels = {name: c.config["latent_channels"] for name, c in components.items()
                        if "latent_channels" in c.config}
    distinct = set(latent_channels.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{name}={n}" for name, n in sorted(latent_channels.items()))
        raise ValueError(
            f"Components disagree on latent_channels: {detail}. Check that the AE and LDS "
            f"checkpoints being combined actually came from the same pipeline run (or "
            f"deliberately compatible ones)."
        )

    latent_spatial_size = {name: c.config["latent_spatial_size"] for name, c in components.items()
                            if "latent_spatial_size" in c.config}
    distinct_spatial = set(latent_spatial_size.values())
    if len(distinct_spatial) > 1:
        detail = ", ".join(f"{name}={n}" for name, n in sorted(latent_spatial_size.items()))
        raise ValueError(
            f"Components disagree on latent_spatial_size: {detail}. Check that the AE and LDS "
            f"checkpoints being combined actually came from the same pipeline run (or "
            f"deliberately compatible ones)."
        )


def assemble_joint_checkpoint(ae_checkpoint_path: str | Path, lds_checkpoint_path: str | Path,
                               device: str | None = None) -> dict[str, ComponentCheckpoint]:
    """
    The stage-4 entry point: merges a stage-1/2 AE checkpoint and a
    stage-3 LDS checkpoint into one componentized structure, validating
    they're actually compatible before returning rather than deferring
    the failure to wherever the mismatch happens to first cause a crash.
    """
    ae_components = load_ae_components(ae_checkpoint_path, device=device)
    lds_component = load_lds_component(lds_checkpoint_path, device=device)
    components = {**ae_components, "lds": lds_component}
    validate_component_compatibility(components)
    return components


def load_joint_refinement_checkpoint(checkpoint_path: str | Path,
                                      device: str | None = None) -> dict[str, ComponentCheckpoint]:
    """
    The stage-5 entry point: unlike stage 4 (which always merges TWO
    separate ancestors via assemble_joint_checkpoint), stage 5 continues
    stage 4's own output -- a single checkpoint already holding E, D, f
    (and optionally stats_head) together, in train_refinement()'s own
    joint save format (ae_state/f_theta_state/stats_head_state -- see
    that function), not the separate stage-1/2/stage-3 shapes
    load_ae_components()/load_lds_component() expect.

    Produces the SAME componentized dict shape assemble_joint_checkpoint
    does (keys: encoder, decoder, lds, and stats_head if present), so
    build_models_from_components works identically regardless of which
    of these two adapters loaded the checkpoint -- train_refinement()
    itself doesn't need to know or care which path was taken once this
    returns.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=True)
    model_cfg = checkpoint["config"]
    lds_cfg = checkpoint["lds_config"]
    ae_state = checkpoint["ae_state"]

    shared_provenance = {
        "source_checkpoint": str(checkpoint_path.resolve()),
        "epoch": checkpoint["epoch"],
        "val_loss": checkpoint["val_loss"],
    }
    ae_component_config = {
        "size": model_cfg["size"], "base_channels": model_cfg["base_channels"],
        "latent_channels": model_cfg["latent_channels"],
        "latent_spatial_size": model_cfg.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
        "stream_configs": model_cfg.get("stream_configs"),
        "recon_stream_name": model_cfg.get("recon_stream_name"),
    }
    # Cross-checked against the ACTUAL encoder weights (ae_state, still
    # "encoder."-prefixed at this point), not just trusted from
    # model_cfg -- see cross_check_stream_configs_against_state_dict's
    # own docstring for the real failure mode this guards against.
    # Corrected back into ae_component_config as plain dicts/strings
    # (not LatentStreamConfig/LatentStreamMode objects), matching the
    # rest of this config's own serializable-values convention.
    _resolved_streams, _resolved_recon_name = resolve_stream_configs_from_checkpoint_config(model_cfg)
    _resolved_streams, _resolved_recon_name = cross_check_stream_configs_against_state_dict(
        _resolved_streams, _resolved_recon_name, ae_state,
    )
    ae_component_config["stream_configs"] = {
        name: {"channels": cfg.channels, "spatial_size": cfg.spatial_size, "mode": cfg.mode.value}
        for name, cfg in _resolved_streams.items()
    }
    ae_component_config["recon_stream_name"] = _resolved_recon_name

    components = {
        "encoder": ComponentCheckpoint(
            state_dict=_strip_prefix(ae_state, "encoder"),
            config=dict(ae_component_config),
            provenance=dict(shared_provenance),
        ),
        "decoder": ComponentCheckpoint(
            state_dict=_strip_prefix(ae_state, "decoder"),
            config=dict(ae_component_config),
            provenance=dict(shared_provenance),
        ),
        "lds": ComponentCheckpoint(
            state_dict=checkpoint["f_theta_state"],
            config=dict(lds_cfg),
            provenance={**shared_provenance, "ae_checkpoint": checkpoint.get("ae_checkpoint"),
                        "lds_checkpoint": checkpoint.get("lds_checkpoint")},
        ),
    }

    stats_head_state = checkpoint.get("stats_head_state")
    stats_config = checkpoint.get("stats_config")
    if stats_head_state is not None and stats_config is not None:
        components["stats_head"] = ComponentCheckpoint(
            state_dict=stats_head_state,
            config={"latent_channels": model_cfg["latent_channels"],
                    "latent_spatial_size": model_cfg.get("latent_spatial_size", LATENT_SPATIAL_SIZE),
                    "stat_names": stats_config["stat_names"]},
            provenance={**shared_provenance, "stats_mean": stats_config["stats_mean"],
                        "stats_std": stats_config["stats_std"]},
        )

    validate_component_compatibility(components)
    return components


def split_joint_checkpoint_for_evaluation(
    joint_checkpoint_path: str | Path, output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Derives two standalone-format checkpoint files from a stage 4/5
    joint checkpoint, so check_reconstruction()/check_rollout() -- which
    only know how to read a standalone AE checkpoint or a standalone LDS
    checkpoint, not the joint ae_state/f_theta_state format -- can run
    against stage 4/5's output completely unchanged. An adapter that
    writes real files, not a code change to either evaluation script:
    those scripts stay exactly as correct/tested as they already are for
    stages 1-3, and this doesn't need to know anything about their
    internals beyond the checkpoint SHAPE they expect.

    Returns (ae_view_path, lds_view_path). The LDS view's own
    "ae_checkpoint" field points at ae_view_path (NOT the original
    stage-2 ancestor) -- so check_rollout(), which follows that field to
    load the paired encoder/decoder, picks up stage 4/5's REFINED E/D,
    not the pre-refinement stage-2 ones.
    """
    joint_checkpoint_path = Path(joint_checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(joint_checkpoint_path, map_location="cpu", weights_only=True)

    ae_view_path = output_dir / f"{joint_checkpoint_path.stem}-ae_view.pt"
    ae_view = {
        "model_state": checkpoint["ae_state"],
        "epoch": checkpoint["epoch"],
        "val_loss": checkpoint["val_loss"],
        "val_loss_ema": checkpoint.get("val_loss_ema"),
        "test_dirs": checkpoint["test_dirs"],
        "config": dict(checkpoint["config"]),
    }
    if checkpoint.get("stats_head_state") is not None and checkpoint.get("stats_config") is not None:
        ae_view["stats_head_state"] = checkpoint["stats_head_state"]
        ae_view["stats_config"] = checkpoint["stats_config"]
    torch.save(ae_view, ae_view_path)

    data_config = checkpoint.get("data_config", {})
    lds_view_path = output_dir / f"{joint_checkpoint_path.stem}-lds_view.pt"
    lds_view = {
        "model_state": checkpoint["f_theta_state"],
        "epoch": checkpoint["epoch"],
        "val_loss": checkpoint["val_loss"],
        "val_loss_ema": checkpoint.get("val_loss_ema"),
        "ae_checkpoint": str(ae_view_path.resolve()),
        "test_dirs": checkpoint["test_dirs"],
        "config": dict(checkpoint["lds_config"]),
        "data_config": {
            "min_step": data_config.get("min_step", 0),
            "min_stdev_phi": data_config.get("min_stdev_phi"),
            "window_length": data_config.get("window_length", 2),
            "n_rollout_steps": data_config.get("n_rollout_steps", 1),
        },
    }
    torch.save(lds_view, lds_view_path)

    return ae_view_path, lds_view_path

