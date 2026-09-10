"""
Select (prune) latent channels in a stage-1 or stage-2 checkpoint.

Reduces latent_channels from N to len(channels_to_keep) by slicing, in place in
the saved state_dict, every tensor that carries the latent-channel dimension --
so the resulting checkpoint loads into a model built at the smaller channel count
and can be resumed (e.g. rerun stage 2 at 4 channels after the importance analysis
showed 8 was over-provisioned).

WHY this is safe to do by dimension-size rather than by hardcoded key names:
in the PARAMETER tensors, the latent-channel dim is unambiguous.
  - The AE latent path (encoder projection(s), decoder input conv, any deriv
    residual-head convs) is all 1x1 convs, so latent channels appear as a conv
    dimension of size == old_channels. A 1x1 conv has NO spatial dim in its
    weight (kernel 1x1), so latent_spatial_size (also 8!) never appears as a
    parameter dim -- the channels/spatial size clash exists only in ACTIVATIONS,
    never in the saved weights. And base_channels/hidden_dim are not 8, so a
    parameter dim of exactly old_channels is the latent-channel dim, full stop.
  - The ONE exception is StatsHead's first Linear, whose input is the FLATTENED
    latent (in_features == old_channels * spatial * spatial): there channels and
    spatial ARE entangled, so it is reshaped to (hidden, C, S, S), sliced on the
    channel axis, and re-flattened. Handled explicitly.

channels_to_keep is applied to EVERY stream (state and deriv share the count).
The deriv stream's specific kept channels barely matter when stage 2 will retrain
its head, but keeping the same indices is consistent.

VALIDATION: this does not rebuild a model to check itself -- the authoritative
check is that the DOWNSTREAM load (stage 2 building a len(keep)-channel model and
load_state_dict(strict=True) on this output) succeeds. A mis-slice raises there,
loudly, before any training -- it cannot silently corrupt. Run it and confirm the
resume loads without a shape/missing-key error.

Usage:
    python -m evaluation.select_latent_channels IN.pt OUT.pt --channels-to-keep 1 2 5 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _slice_dims_equal_to(tensor: torch.Tensor, size: int, keep: torch.Tensor,
                          name: str) -> torch.Tensor:
    """index_select `keep` on EVERY dim of `tensor` whose length == size.

    A latent tensor may carry the channel dim more than once is not expected here
    (encoder proj: out only; decoder: in only), but slicing all matching dims is
    correct and idempotent for the 1x1-conv params where size==old_channels is
    exclusively the latent dim. Reports each slice for eyeballing."""
    out = tensor
    for axis, length in enumerate(tensor.shape):
        if length == size:
            out = out.index_select(axis, keep)
    if out.shape != tensor.shape:
        print(f"    {name}: {tuple(tensor.shape)} -> {tuple(out.shape)}")
    return out


def _slice_stats_head(state: dict, old_channels: int, spatial: int,
                      keep: torch.Tensor, label: str) -> dict:
    """Slice a StatsHead state_dict's first Linear along the flattened channel
    axis. in_features == old_channels*spatial*spatial; reshape -> select channel
    -> reflatten. Everything else (hidden, output layer) is unchanged."""
    flat = old_channels * spatial * spatial
    out = {}
    for k, v in state.items():
        # the input Linear weight is the 2D tensor whose in_features == flat
        if v.ndim == 2 and v.shape[1] == flat:
            w = v.reshape(v.shape[0], old_channels, spatial, spatial)
            w = w.index_select(1, keep).reshape(v.shape[0], len(keep) * spatial * spatial)
            print(f"    {label}.{k}: {tuple(v.shape)} -> {tuple(w.shape)} "
                  f"(reshape/select channel/flatten)")
            out[k] = w
        else:
            out[k] = v
    return out


def _update_config(config: dict, new_channels: int) -> None:
    """latent_channels (top level), stream_configs[*].channels, and
    latent_channels_decoder if present -- set to the new count. Mutates in place."""
    if "latent_channels" in config:
        config["latent_channels"] = new_channels
    if "latent_channels_decoder" in config:
        config["latent_channels_decoder"] = new_channels
    sc = config.get("stream_configs")
    if isinstance(sc, dict):
        for stream in sc.values():
            if isinstance(stream, dict) and "channels" in stream:
                stream["channels"] = new_channels


def select_latent_channels(in_path: Path, out_path: Path,
                           channels_to_keep: list[int]) -> None:
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    old_channels = int(config["latent_channels"])
    spatial = int(config.get("latent_spatial_size", 8))

    keep = sorted(set(channels_to_keep))
    if keep != list(channels_to_keep):
        print(f"NOTE: channels_to_keep normalised to sorted-unique {keep}")
    if not keep or keep[0] < 0 or keep[-1] >= old_channels:
        raise ValueError(f"channels_to_keep {channels_to_keep} out of range for "
                         f"latent_channels={old_channels} (valid 0..{old_channels-1})")
    if len(keep) == old_channels:
        raise ValueError(f"channels_to_keep selects all {old_channels} channels -- "
                         f"nothing to prune")
    keep_t = torch.tensor(keep, dtype=torch.long)
    new_channels = len(keep)
    print(f"Selecting {new_channels}/{old_channels} latent channels: keep={keep} "
          f"(spatial={spatial}, applied to all streams)")

    # 1. AE model_state: slice every param dim == old_channels (see module docstring).
    print("  model_state:")
    ckpt["model_state"] = {
        k: _slice_dims_equal_to(v, old_channels, keep_t, k)
        for k, v in ckpt["model_state"].items()
    }

    # 2. stats head(s): reshape/slice the flattened-latent input Linear.
    for key, label in (("stats_head_state", "stats_head0"),
                       ("stats_head1_state", "stats_head1")):
        if ckpt.get(key) is not None:
            print(f"  {key}:")
            ckpt[key] = _slice_stats_head(ckpt[key], old_channels, spatial, keep_t, label)

    # 3. config: latent_channels / stream_configs / decoder count.
    _update_config(config, new_channels)

    # 4. the stored val_loss is for the OLD (larger) model -- it is not a fair bar
    #    for the pruned model's resume to clear. Blank it so the resuming stage
    #    treats this as a fresh baseline (its grace/comparability logic then applies)
    #    rather than chasing an unbeatable stale number.
    if "val_loss" in ckpt:
        ckpt["val_loss"] = float("inf")
    if "val_loss_ema" in ckpt:
        ckpt["val_loss_ema"] = float("inf")
    ckpt["channels_selected_from"] = {"old_channels": old_channels, "kept": keep}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved {new_channels}-channel checkpoint to {out_path}")
    print("  VALIDATE by resuming stage 2 from it: the model is built at "
          f"{new_channels} channels and load_state_dict(strict=True) will RAISE "
          "on any mis-slice before training -- a clean load confirms the surgery.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Select a subset of latent channels in a checkpoint.")
    ap.add_argument("in_path", type=Path)
    ap.add_argument("out_path", type=Path)
    ap.add_argument("--channels-to-keep", type=int, nargs="+", required=True,
                    help="Latent channel indices to KEEP (e.g. --channels-to-keep 1 2 5 6). "
                         "Applied to every stream.")
    args = ap.parse_args()
    select_latent_channels(args.in_path, args.out_path, args.channels_to_keep)


if __name__ == "__main__":
    main()
