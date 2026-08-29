"""
Disk cache for latents produced by a FROZEN encoder.

Stages 3a and 3b both load the same stage-2 checkpoint as their frozen
encoder and both encode the same runs; each stage-3 diagnostic then encodes
the test set again. The latents are a pure function of

    (encoder weights, run_dir, the exact list of steps encoded)

and stage 3 freezes the encoder by definition, so the second and subsequent
passes are recomputing a value that cannot have changed.

Storage is cheap relative to what it replaces: a latent is
latent_channels * 8 * 8 floats -- 2 KB at C=8 in float32 -- against 32 KB for
one 128x128 float16 frame. With the max_dt truncation in place only the
reachable prefix is ever encoded, shrinking it further.

THE HAZARD, and why the key is built the way it is. A stale hit is silent and
severe: training would proceed on latents from a DIFFERENT encoder, with no
shape error and no warning, and every downstream number would be wrong in a
way nothing else detects. So the key contains a hash of the encoder's own
weights rather than a checkpoint path and mtime. Paths get reused (stage 2 is
retrained in place, and `force=True` overwrites), mtimes survive a file copy,
and neither says anything about what the weights actually are. Hashing costs
one pass over ~2.6M parameters, ~10 ms, once per process.
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import torch


def encoder_fingerprint(encoder: torch.nn.Module) -> str:
    """A short hash of the encoder's weights AND buffers.

    Buffers included deliberately: BatchNorm running_mean/running_var change
    what the encoder outputs while leaving every parameter untouched, so a
    parameters-only hash would call two encoders identical when they are not.
    That is exactly the state a re-estimated port is in.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(name.encode("utf-8"))
        contiguous = tensor.detach().to("cpu").contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def write_cache_info(cache_root: Path, fingerprint: str, size: int | None,
                     info: dict | None = None) -> None:
    """Drop a human-readable `_cache_info.txt` in the fingerprint's subdir, so a
    directory whose name is an opaque blake2b hash says WHAT produced it. The
    fingerprint is a hash of the encoder's weights+buffers; `info` carries the
    identifying context the hash itself can't be read back into (source
    checkpoint, latent_channels, ...). Best-effort: the static block is written
    once, the 'cache last accessed' timestamp is refreshed on every call, and
    failure is swallowed (a missing note must never break caching)."""
    directory = f"{size}x{size}-{fingerprint}" if size is not None else fingerprint
    cache_dir = cache_root / directory
    info_path = cache_dir / "_cache_info.txt"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "Latent cache directory. Name = <size>x<size>-<encoder fingerprint>.",
            "The fingerprint is a blake2b hash of the encoder's weights AND",
            "buffers (BatchNorm running stats included) -- nothing else. Same",
            "encoder => same directory, independent of WHEN it was built. Files",
            "here are latents from THAT encoder; a different encoder gets a",
            "different dir.",
            "",
            f"fingerprint     = {fingerprint}",
            f"size            = {size}",
        ]
        for k, v in (info or {}).items():
            lines.append(f"{k:<15} = {v}")
        # The access time DOES belong to the cache (not the hash): separated by
        # a blank line so it reads as metadata about this directory, not as an
        # input to the fingerprint.
        lines.append("")
        lines.append(f"cache last accessed = "
                     f"{datetime.datetime.now().isoformat(timespec='seconds')}")
        info_path.write_text("\n".join(lines) + "\n")
    except OSError:
        pass


def cache_path_for_run(cache_root: Path, fingerprint: str, run_dir: Path,
                        steps: list[int], encode_both_streams: bool,
                        size: int | None = None) -> Path:
    """Where this run's latents live, for this encoder and this step list.

    `steps` is part of the key, not just the run: the max_dt truncation makes
    the encoded prefix depend on max_dt, and a shorter cached prefix must not
    satisfy a later request for a longer one. Hashed rather than spelled out
    because the list can hold 70+ entries and the result is a filename.

    The directory is named `<size>x<size>-<fingerprint>` when `size` is
    known. The fingerprint alone is opaque -- a directory listing said
    nothing about which resolution a cache belonged to, so a 32x32 test
    cache and a 128x128 training cache were indistinguishable without
    opening one. The size is a LABEL, not part of the key: the fingerprint
    already separates encoders, and two encoders at different resolutions
    cannot collide because their weight shapes differ and shapes are hashed.
    Omitting size falls back to the bare fingerprint, so existing caches
    stay readable.
    """
    directory = f"{size}x{size}-{fingerprint}" if size is not None else fingerprint
    step_digest = hashlib.blake2b(
        ",".join(str(s) for s in steps).encode("utf-8"), digest_size=8).hexdigest()
    suffix = "both" if encode_both_streams else "state"
    return cache_root / directory / f"{run_dir.name}-{step_digest}-{suffix}.pt"


def load_cached(path: Path) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """The cached (state, deriv) latents ON CPU, or None on ANY problem.

    ALWAYS CPU, never the training device. The encode path this substitutes
    for ends in `.cpu()` (see _flush_buffer), and a dataset whose runs are a
    MIXTURE of devices only fails later, in the DataLoader's collate:

        RuntimeError: Expected all tensors to be on the same device, but
        found at least two devices, cuda:0 and cpu!

    -- from a worker process, pointing at torch.stack rather than at the
    cache. Taking a `device` argument at all was the mistake: there is no
    correct value for it other than the one the encode path already uses.

    Deliberately total: a corrupt or half-written cache file must degrade to a
    recompute, never to a crash. A cache that can break the run it is meant to
    speed up is worse than no cache.
    """
    if not path.exists():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        state = blob["state"]
        deriv = blob.get("deriv")
        return state, deriv
    except Exception:  # noqa: BLE001 - see docstring: any failure means recompute
        return None


def store_cached(path: Path, state: torch.Tensor, deriv: torch.Tensor | None) -> None:
    """Write atomically, and never raise.

    Atomic because several stages (and, under pytest-xdist, several processes)
    can be building the same entry concurrently; a reader must see either the
    old file or the complete new one, never a partial write. Silent on failure
    for the same reason load_cached is: a full disk should slow the run down,
    not stop it.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"state": state.detach().to("cpu")}
        if deriv is not None:
            blob["deriv"] = deriv.detach().to("cpu")
        tmp = path.with_name(path.name + f".tmp{id(state):x}")
        torch.save(blob, tmp)
        tmp.replace(path)
    except Exception:  # noqa: BLE001
        pass
