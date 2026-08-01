"""
Golden-master regression test for the Autoencoder architecture, built
ahead of the C0/C1 latent-stream redesign specifically so every step
of that redesign can be checked against "does this still compute
exactly what today's code computes", not just "does it run".

Two roles in one file (deliberately -- see below for why not split):

1. CAPTURE (run ONCE, now, before any redesign changes):
       python tests/test_architecture_stability.py
   (from python/). Builds a small, fixed-seed Autoencoder, runs one
   forward pass and one optimizer step, and saves everything needed to
   reproduce and check that computation later -- initial weights,
   input, forward outputs, gradients, post-step weights -- to
   architecture_v1_golden_master.pt alongside this file. This is NOT
   run automatically by pytest (guarded behind __main__, and
   capture() is never called from the test_* function below) -- a
   fixture that regenerates itself on every run would defeat the
   entire point of a regression check.

2. COMPARE (runs as part of the normal test suite): loads that
   fixture, rebuilds a model against WHATEVER the current code is,
   loads the SAVED initial weights into it (not fresh random weights
   -- see _load_state_dict_into_current_model's own docstring for
   why), redoes the identical forward+backward+step, and asserts every
   captured value still matches exactly.

Single file rather than a shared-helper-module-plus-test-file split:
this project's existing test suite consistently keeps each test file
self-contained (no test-file-importing-another-test-file anywhere in
it), and tests/ has no __init__.py visible -- rather than assume a
cross-file import resolves cleanly under however this suite actually
gets collected, this stays self-contained like its siblings. The cost
is that CAPTURE logic ships inside a file that's collected by pytest
too; the __main__ guard is what keeps that safe.

Design choice worth being explicit about: this does NOT rely on
construction-order RNG coincidence to reproduce the same initial
weights across a refactored constructor -- it saves the actual initial
state_dict and loads it explicitly instead. That's a strictly more
robust invariant: it verifies "given IDENTICAL weights and IDENTICAL
input, does the (possibly refactored) code compute the IDENTICAL
forward output and gradients", which is what actually matters, without
also depending on construction-call-order details that could change
for reasons unrelated to any real bug.
"""
from pathlib import Path
import sys

# GENERAL POLICY (matches training/train_refinement.py's own
# _PYTHON_ROOT): needed here specifically because this file is also
# meant to be run directly (python tests/test_architecture_stability.py,
# for the CAPTURE step) -- conftest.py already adds python/ to
# sys.path for normal pytest collection, but that machinery never runs
# for a bare script invocation, which only puts THIS file's own
# directory (tests/) on sys.path, not python/ -- so `from models...`
# below fails with ModuleNotFoundError unless this is done explicitly.
_PYTHON_ROOT = Path(__file__).resolve().parent.parent  # python/tests/test_architecture_stability.py -> python/
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

import pytest
import contextlib
from datetime import datetime
import shutil

import torch
import torch.nn.functional as F

from models.autoencoder import Autoencoder
from models.latent_streams import remap_pre_multistream_state_dict_key as _remap_key

# Deliberately tiny: smallest valid size for the default
# latent_spatial_size=8 is 16 (n_stages = log2(16/8) = 1) -- small
# enough that the fixture is a small file and the test is fast, since
# this needs to be committed to the repo and rerun at every step of
# the redesign, not just once.
_SIZE = 16
_BASE_CHANNELS = 2
_LATENT_CHANNELS = 2
_BATCH_SIZE = 4  # >1 required for BatchNorm's training-mode statistics
_LR = 0.01
_SEED = 0

_FIXTURE_PATH = Path(__file__).resolve().parent / "architecture_v1_golden_master.pt"


def _build_reference_model(seed: int = _SEED) -> Autoencoder:
    torch.manual_seed(seed)
    return Autoencoder(
        size=_SIZE, channels=1, base_channels=_BASE_CHANNELS,
        latent_channels=_LATENT_CHANNELS,
    )


def _build_reference_input(seed: int = _SEED) -> torch.Tensor:
    # Separate Generator (not the global manual_seed used for model
    # weights) so this stays reproducible independent of exactly how
    # many random draws model construction consumes -- if a refactor
    # changes how many parameters get initialized (a different number
    # of RNG draws), the INPUT should still come out identical, since
    # weight LOADING (not fresh construction) is what's actually being
    # tested here, not construction-order coincidence (see module
    # docstring).
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(_BATCH_SIZE, 1, _SIZE, _SIZE, generator=generator)


@contextlib.contextmanager
def _deterministic_numerics():
    """Pin torch to ONE intra-op thread for the duration.

    This fixture is compared with torch.equal -- BIT-exact -- but the values
    it stores are not bit-stable across thread counts: torch parallelises the
    reductions inside convolutions and BatchNorm, and a different number of
    threads sums the same numbers in a different order. Float addition is not
    associative, so the result differs in the last ulp or two.

    That made the test silently MACHINE-DEPENDENT. It passed everywhere it had
    been run only because every run happened to use the same thread count; it
    surfaced the moment the suite was run under pytest-xdist with one thread
    per worker, as:

        x_recon: max abs diff = 8.345e-07
        z:       max abs diff = 1.192e-07
        loss:    1.254752e+00 vs 1.254752e+00     <- identical

    A loss that matches exactly while its own intermediates differ at 1e-7 is
    the signature of reassociation, not of an architecture change: the
    reduction that produces the scalar loss happens to be order-insensitive at
    this precision while the tensors feeding it are not.

    Pinning to one thread (rather than loosening the comparison to a
    tolerance) keeps the check at full strength -- the point of a golden
    master is to notice ANY change -- while removing the one source of
    variation that is not about the architecture. Applied to BOTH capture()
    and the comparison, so the two always agree.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _forward_backward_and_step(ae: Autoencoder, x: torch.Tensor) -> dict:
    """
    One forward pass, one backward pass, one optimizer step -- the
    smallest computation that exercises both the forward AND the
    gradient-flow/freeze-pattern behavior of the architecture (a
    subtly wrong freeze/parameter-sharing setup could produce
    identical forward values while silently routing gradient
    somewhere wrong -- forward-only comparison wouldn't catch that).

    Returns every intermediate value worth checking, all CPU tensors
    (forced .cpu() so this is comparable regardless of what device
    it's run on, and so the fixture doesn't require a GPU to load).
    """
    ae = ae.to("cpu")
    x = x.to("cpu")

    initial_state = {k: v.clone() for k, v in ae.state_dict().items()}

    x_recon, z = ae(x)

    optimizer = torch.optim.SGD(ae.parameters(), lr=_LR)
    optimizer.zero_grad()
    loss = F.mse_loss(x_recon, x)
    loss.backward()

    grads = {name: (p.grad.clone() if p.grad is not None else None)
             for name, p in ae.named_parameters()}

    optimizer.step()
    final_state = {k: v.clone() for k, v in ae.state_dict().items()}

    return {
        "initial_state_dict": initial_state,
        "input": x.clone(),
        "x_recon": x_recon.detach().clone(),
        "z": z.detach().clone(),
        "loss": loss.detach().clone(),
        "grads": grads,
        "final_state_dict": final_state,
    }


def capture(output_path: Path = _FIXTURE_PATH, force: bool = False) -> Path:
    """Builds today's reference Autoencoder, runs the standard
    forward/backward/step, and saves everything to output_path. Run
    this ONCE, deliberately, against a known-good reference state of
    the architecture -- see this module's __main__ block below.

    Refuses to overwrite an existing fixture unless force=True: this
    file's entire job is being a fixed reference to compare AGAINST,
    so silently regenerating it (e.g. by re-running this after the
    architecture has already changed) would make the comparison
    tautological -- new code checked against a "reference" that's
    ALSO just-generated new code, which can't ever fail regardless of
    what actually changed."""
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists -- refusing to overwrite a fixture "
            f"that's meant to be a fixed reference. If you genuinely mean to "
            f"replace it (e.g. establishing a NEW deliberate reference point), "
            f"call capture(force=True) or delete the file first."
        )
    ae = _build_reference_model()
    x = _build_reference_input()
    with _deterministic_numerics():
        result = _forward_backward_and_step(ae, x)
    result["config"] = {
        "size": _SIZE, "base_channels": _BASE_CHANNELS,
        "latent_channels": _LATENT_CHANNELS, "batch_size": _BATCH_SIZE,
        "lr": _LR, "seed": _SEED,
    }
    torch.save(result, output_path)
    print(f"Saved golden-master fixture to {output_path}")
    return output_path




def _load_state_dict_into_current_model(ae: Autoencoder, saved_state_dict: dict) -> None:
    """Loads the FIXTURE's saved initial weights into a freshly-built
    CURRENT-code model, rather than relying on fresh random
    initialization to happen to match (see module docstring). Renaming
    itself lives in _remap_key, not here -- see that function's own
    docstring for why.

    strict=False + explicit missing-key check (not just strict=True):
    log_output_scale (see autoencoder.py's EncoderDecoderPair) is a
    genuinely NEW buffer this golden-master fixture predates -- always
    "missing" here regardless of when the fixture was captured, not a
    sign of real architecture drift. Filtered out before re-raising so
    any OTHER missing/unexpected key (genuine drift, what this test
    exists to catch) still fails loudly.
    """
    remapped = {_remap_key(key): value for key, value in saved_state_dict.items()}
    result = ae.load_state_dict(remapped, strict=False)
    missing = [k for k in result.missing_keys if not k.endswith("log_output_scale")]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Error(s) in loading state_dict for {ae.__class__.__name__}: "
            f"missing keys: {missing}, unexpected keys: {result.unexpected_keys}"
        )


def _compare_against_fixture(fixture_path: Path = _FIXTURE_PATH) -> dict:
    """
    Rebuilds a model against WHATEVER the current code is, loads the
    fixture's saved initial weights into it, redoes the identical
    forward/backward/step against the fixture's saved input, and
    returns {check_name: (matches: bool, detail: str)} for every
    captured value.
    """
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    config = fixture["config"]

    ae = Autoencoder(
        size=config["size"], channels=1, base_channels=config["base_channels"],
        latent_channels=config["latent_channels"],
    )
    _load_state_dict_into_current_model(ae, fixture["initial_state_dict"])

    with _deterministic_numerics():
        result = _forward_backward_and_step(ae, fixture["input"])

    checks = {}

    checks["x_recon"] = (
        torch.equal(result["x_recon"], fixture["x_recon"]),
        f"max abs diff = {(result['x_recon'] - fixture['x_recon']).abs().max().item():.3e}",
    )
    checks["z"] = (
        torch.equal(result["z"], fixture["z"]),
        f"max abs diff = {(result['z'] - fixture['z']).abs().max().item():.3e}",
    )
    checks["loss"] = (
        torch.equal(result["loss"], fixture["loss"]),
        f"{result['loss'].item():.6e} vs {fixture['loss'].item():.6e}",
    )

    grad_mismatches = []
    for name, saved_grad in fixture["grads"].items():
        new_grad = result["grads"].get(_remap_key(name))
        if saved_grad is None and new_grad is None:
            continue
        if saved_grad is None or new_grad is None or not torch.equal(new_grad, saved_grad):
            grad_mismatches.append(name)
    checks["grads"] = (
        not grad_mismatches,
        f"mismatched: {grad_mismatches}" if grad_mismatches else "all matched",
    )

    state_mismatches = []
    for name, saved_value in fixture["final_state_dict"].items():
        new_value = result["final_state_dict"].get(_remap_key(name))
        if new_value is None or not torch.equal(new_value, saved_value):
            state_mismatches.append(name)
    checks["final_state_dict"] = (
        not state_mismatches,
        f"mismatched: {state_mismatches}" if state_mismatches else "all matched",
    )

    return checks


def test_architecture_matches_golden_master():
    if not _FIXTURE_PATH.exists():
        pytest.fail(
            f"golden-master fixture not found at {_FIXTURE_PATH}. Generate it "
            f"once with: python tests/test_architecture_stability.py (from python/)."
        )

    checks = _compare_against_fixture(_FIXTURE_PATH)

    # Printed unconditionally (not just embedded in the assert message
    # below) specifically because pytest's default summary output has
    # repeatedly truncated the assert message's actual detail in
    # practice -- pytest captures stdout separately and shows it under
    # "Captured stdout call" for a failing test, which isn't subject to
    # the same truncation, so this is belt-and-suspenders for actually
    # being able to see WHICH check failed and by how much.
    print("\nGolden-master comparison:")
    for name, (ok, detail) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    failed = {name: detail for name, (ok, detail) in checks.items() if not ok}
    assert not failed, (
        "architecture no longer matches the golden-master fixture:\n  "
        + "\n  ".join(f"{name}: {detail}" for name, detail in failed.items())
    )


if __name__ == "__main__":
    # Generate the fixture:
    #   python tests/test_architecture_stability.py            (first time)
    #   python tests/test_architecture_stability.py --force    (regenerate)
    # (from python/, so the models package import resolves).
    #
    # --force exists because capture() deliberately refuses to overwrite, and
    # without a way through that refusal the documented regeneration command
    # simply raises. Regeneration IS occasionally legitimate -- the numerics
    # are pinned to one thread now (see _deterministic_numerics), so a fixture
    # captured before that pin no longer matches on any machine.
    #
    # The existing fixture is ARCHIVED first, not replaced. Regenerating is the
    # one operation that can silently bless an architecture change: it
    # overwrites the only record of what the architecture used to compute. The
    # archive means a regeneration done for the wrong reason is recoverable,
    # and it can be diffed afterwards to see what actually moved.
    _force = "--force" in sys.argv
    if _force and _FIXTURE_PATH.exists():
        _stamp = datetime.fromtimestamp(_FIXTURE_PATH.stat().st_mtime).strftime("%Y%m%d_%Hh%M")
        _archive = _FIXTURE_PATH.with_name(f"{_FIXTURE_PATH.stem}-{_stamp}{_FIXTURE_PATH.suffix}")
        if not _archive.exists():
            shutil.copy2(_FIXTURE_PATH, _archive)
            print(f"archived the existing fixture to {_archive.name}")
        print("REGENERATING: verify the test passed BEFORE this change was applied -- a green "
              "run on the old fixture is what proves the architecture itself has not drifted. "
              "Regenerating on a red run would bless whatever caused it.")
    capture(force=_force)
