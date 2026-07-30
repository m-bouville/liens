"""
Shared checkpoint-loading setup for the two diagnostics built directly
around an AE-family checkpoint's own StatsHead: check_interpolation.py
and check_perturbation.py. Both had independently reimplemented the same
~25 lines -- device resolution, output_path defaulting, AE loading via
build_ae_from_checkpoint, stats_head construction/loading, and the same
two validation checks -- differing only in the output subdirectory name
and one error message's own extra sentence.

Deliberately NOT part of _latent_eval.py: that module's own
_load_models_and_dataset is specifically for the LDS/stage-3 evaluation
path (ensure_lds_checkpoint, f_theta, LatentDynamics) that neither of
these two diagnostics touches at all -- they load an AE-family
checkpoint and its StatsHead, nothing else. Crowding this into
_latent_eval.py would blur what that module is actually for.
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from training.checkpoint_components import build_ae_from_checkpoint
from training.stats_head import StatsHead

_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/evaluation/X.py -> python/


@dataclass
class AEStatsContext:
    device: torch.device
    output_path: Path
    ae: object
    ae_encoder: object
    checkpoint: dict
    stream_configs: dict
    recon_stream_name: str
    model_cfg: dict
    recon_stream: object
    stats_config: dict
    stats_head: StatsHead


def load_ae_and_stats_head(
    checkpoint_path: Path, output_subdir: str, output_path: Path | None,
    device: str | None, no_stats_head_context: str = "",
) -> AEStatsContext:
    """
    Resolves device, defaults output_path (under output/<output_subdir>/
    if not given explicitly -- callers pass their own subdirectory name,
    e.g. "interpolation_check_png"), loads the AE-family checkpoint, and
    builds+loads its StatsHead. Raises ValueError if the checkpoint has
    no stats_head at all (trained with --stats-weight 0) or no saved
    test_dirs -- both diagnostics need the former; test_dirs handling is
    left to each caller, since check_interpolation.py's own
    fixed_triples mode doesn't need it at all, while
    check_perturbation.py always does.

    no_stats_head_context: appended to the "no stats_head" error message
    -- check_perturbation.py's own call adds "-- this check is built
    entirely around stats_head." where check_interpolation.py's adds
    nothing; preserved here rather than silently unified, since it's a
    genuine (if small) difference in how each diagnostic depends on
    stats_head.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if output_path is None:
        output_path = (_PYTHON_ROOT.parent / "output" / output_subdir
                       / f"{checkpoint_path.stem}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ae, ae_encoder, checkpoint, stream_configs, recon_stream_name = build_ae_from_checkpoint(
        checkpoint_path, device,
    )
    model_cfg = checkpoint["config"]
    recon_stream = stream_configs[recon_stream_name]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, config={model_cfg}")

    stats_config = checkpoint.get("stats_config")
    if stats_config is None:
        raise ValueError(f"{checkpoint_path} has no stats_head (trained with --stats-weight 0)"
                          f"{(' ' + no_stats_head_context) if no_stats_head_context else ''}")

    stats_head = StatsHead(
        latent_channels=recon_stream.channels, stat_names=stats_config["stat_names"],
        latent_spatial=recon_stream.spatial_size,
    ).to(device)
    stats_head.load_state_dict(checkpoint["stats_head_state"])
    stats_head.eval()
    print(f"stats_head covers: {stats_config['stat_names']}")

    return AEStatsContext(
        device=device, output_path=output_path, ae=ae, ae_encoder=ae_encoder,
        checkpoint=checkpoint, stream_configs=stream_configs, recon_stream_name=recon_stream_name,
        model_cfg=model_cfg, recon_stream=recon_stream, stats_config=stats_config,
        stats_head=stats_head,
    )
