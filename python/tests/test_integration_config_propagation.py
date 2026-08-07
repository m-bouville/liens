"""
Every field that changes what f_theta MEANS must survive a save/rebuild.

THE BUG THIS EXISTS FOR. alpha was added to the checkpoint config and to
train_lds's own resume comparability, and NOT to model_assembly -- so stage 4
rebuilt an f_theta fitted at delta_t~36 and ran it ONE-SHOT at dt=500. It
skipped 39, then 277, then 324, then 325 of ~325 batches, in four epochs, on a
model that had been training cleanly. The weights load fine either way; nothing
fails; the run simply integrates with a step the model was never fitted for.

THE SHAPE. There were FIVE independent places rebuilding a LatentDynamics from
a saved config, each with its own hand-written list of fields. The comments
they carried recorded the history three separate times -- "fixing dt_cap in
either of those did NOT fix it here" -- which is the same defect arriving once
per field: dt_cap needed four manual fixes, n_substeps four, alpha reached four
of five, and compare_integrators had none of them.

So the tests below are enumeration tests. They do not check that any particular
site is right; they check that NO site is allowed to have its own opinion.
"""
import ast
import inspect
import pathlib

import pytest
import torch

from models.latent_dynamics import (
    LatentDynamics, _MEANING_FIELDS, integration_kwargs_from_config,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every module that rebuilds a LatentDynamics from a SAVED config. A sixth
# joins here -- and the test below fails if one appears and is not listed.
_REBUILD_SITES = [
    "training/model_assembly.py",
    "evaluation/check_rollout.py",
    "evaluation/_latent_eval.py",
    "evaluation/compare_rollout_training.py",
    "evaluation/compare_integrators.py",
]


def _construction_calls(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "LatentDynamics"]


@pytest.mark.parametrize("site", _REBUILD_SITES)
def test_every_rebuild_site_uses_the_shared_field_list(site):
    """
    Structural: the call must splat integration_kwargs_from_config, not spell
    the fields out. Spelling them out is what let each field be forgotten at a
    different site, and no amount of care at one site prevents it at another.
    """
    path = _ROOT / site
    calls = _construction_calls(path)
    assert calls, f"{site} no longer constructs a LatentDynamics -- update _REBUILD_SITES"
    for call in calls:
        rendered = ast.unparse(call)
        assert "integration_kwargs_from_config" in rendered, (
            f"{site} builds a LatentDynamics without the shared field list, so a new "
            f"meaning-changing field would silently miss it"
        )
        for field in _MEANING_FIELDS:
            assert f"{field}=" not in rendered, (
                f"{site} passes {field} by hand alongside the shared list -- two "
                f"sources for one field, and the hand-written one wins silently"
            )


# The WRITER: train_lds builds a LatentDynamics from its own PARAMETERS and
# saves the config every reader above rebuilds from. It is deliberately not a
# rebuild site -- it has nothing to read -- but it closes the loop, and a field
# it fails to save is one no reader can propagate however careful they are.
_WRITER_SITE = "training/train_lds.py"


def test_no_unlisted_rebuild_site_exists():
    """
    The enumeration itself. A sixth reconstruction added anywhere in the
    package must either use the helper or be added here deliberately -- the
    failure mode being fixed is precisely a site nobody remembered.
    """
    found = []
    for pkg in ("models", "training", "orchestration", "evaluation", "utils"):
        for path in (_ROOT / pkg).glob("*.py"):
            if path.name == "latent_dynamics.py":
                continue  # the class's own module
            if _construction_calls(path):
                found.append(f"{pkg}/{path.name}")
    expected = sorted(_REBUILD_SITES + [_WRITER_SITE])
    assert sorted(found) == expected, (
        f"the set of LatentDynamics construction sites changed: {sorted(found)}. "
        f"If it REBUILDS from a saved config, add it to _REBUILD_SITES and make it "
        f"use integration_kwargs_from_config; if it builds from parameters and saves "
        f"a config, it is a writer and needs the round-trip test below."
    )


def test_the_writer_accepts_and_saves_every_meaning_field():
    """
    Closing the loop. The readers can only propagate what the writer saved, so
    a field train_lds accepts but does not write is invisible downstream -- and
    a field it neither accepts nor writes cannot be set at all.

    This is the half that would have caught the stage-4 bug one step earlier:
    alpha WAS saved correctly here, and the break was on the reading side, but
    the same list has to hold at both ends or the next field breaks the other
    way round.
    """
    from training.train_lds import train_lds
    params = inspect.signature(train_lds).parameters
    for field in _MEANING_FIELDS:
        assert field in params, (
            f"train_lds does not accept {field}, so no run can set it and no "
            f"checkpoint can record it"
        )

    src = (_ROOT / _WRITER_SITE).read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    saved_keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    saved_keys.add(key.value)
    missing = sorted(set(_MEANING_FIELDS) - saved_keys)
    assert not missing, (
        f"train_lds never writes {missing} into any saved dict, so every reader "
        f"below will fall back to the default and integrate differently from the "
        f"run that produced the weights"
    )


def test_the_field_list_covers_every_meaning_changing_constructor_argument():
    """
    The list must not fall behind the constructor. Architecture arguments are
    excluded on purpose -- a mismatch there fails loudly at load_state_dict --
    but anything governing the STEP fails silently and belongs here.
    """
    params = set(inspect.signature(LatentDynamics.__init__).parameters) - {"self"}
    architecture = {"latent_channels", "n_theta", "latent_spatial", "hidden_dim",
                    "n_hidden_layers",
                    # truncate_bptt is NEITHER architecture nor meaning-changing,
                    # and is listed here as the deliberate third case. It alters
                    # how the GRADIENT was computed during training; the forward
                    # pass is bit-identical with and without it (pinned by
                    # test_truncation_leaves_the_forward_pass_bit_identical), so
                    # a checkpoint rebuilt without it evaluates exactly as it
                    # trained. Propagating it would also make the no_grad
                    # diagnostics pay detach calls for a graph they never build.
                    #
                    # This test firing on it is the machinery working: a new
                    # constructor argument must be classified deliberately, not
                    # default into silence.
                    "truncate_bptt"}
    uncovered = params - architecture - set(_MEANING_FIELDS)
    assert not uncovered, (
        f"LatentDynamics gained {sorted(uncovered)}, which is neither architecture "
        f"(caught loudly by load_state_dict) nor in _MEANING_FIELDS (propagated to "
        f"every rebuild site). If it changes what f_theta means, add it to "
        f"_MEANING_FIELDS; if not, add it to this test's `architecture` set."
    )


def test_defaults_reproduce_a_pre_feature_checkpoint():
    """
    Each default must be the value that makes an OLD checkpoint rebuild exactly
    as it always did -- otherwise adding a field silently changes how every
    existing checkpoint evaluates.
    """
    kwargs = integration_kwargs_from_config({})
    assert kwargs["dt_cap"] == float("inf"), "a finite default would cap old checkpoints"
    assert kwargs["n_substeps"] == 1
    assert kwargs["alpha"] is None, "a default alpha would make every old checkpoint adaptive"
    old = LatentDynamics(latent_channels=4, latent_spatial=8, hidden_dim=8,
                          n_hidden_layers=1, **kwargs)
    assert old.alpha is None and old.n_substeps == 1


def test_a_saved_config_round_trips_into_an_equivalent_model():
    """
    BEHAVIORAL end of it: build adaptive, save its config, rebuild, and require
    the rebuilt model to integrate IDENTICALLY. This is what stage 4 needed and
    did not get.
    """
    torch.manual_seed(0)
    original = LatentDynamics(latent_channels=4, latent_spatial=8, hidden_dim=8,
                               n_hidden_layers=1, alpha=0.05, max_substeps=64)
    with torch.no_grad():
        original.net[-1].weight.normal_(0.0, 0.02)
        original.net[-1].bias.normal_(0.0, 0.02)
    config = {"latent_channels": 4, "n_theta": 1, "latent_spatial_size": 8,
              "hidden_dim": 8, "n_hidden_layers": 1,
              "dt_cap": float("inf"), "n_substeps": 1,
              "alpha": 0.05, "max_substeps": 64}

    rebuilt = LatentDynamics(latent_channels=4, latent_spatial=8, hidden_dim=8,
                              n_hidden_layers=1,
                              **integration_kwargs_from_config(config))
    rebuilt.load_state_dict(original.state_dict())

    torch.manual_seed(3)
    z0 = torch.randn(4, 4, 8, 8)
    z1 = torch.randn(4, 4, 8, 8)
    dt = torch.full((4,), 500.0)
    theta = torch.zeros(4, 1)
    a, _, _ = original._integrate(z0.clone(), z1.clone(), dt, theta)
    b, _, _ = rebuilt._integrate(z0.clone(), z1.clone(), dt, theta)
    assert torch.equal(a, b), "the rebuilt model integrates differently from the original"


def test_dropping_alpha_from_a_config_is_visibly_different():
    """
    The counter-test, and the one that shows the stage-4 failure directly: a
    rebuild that LOSES alpha produces a materially different trajectory, so
    "the weights loaded fine" is no evidence at all that the model matches.
    """
    torch.manual_seed(0)
    model = LatentDynamics(latent_channels=4, latent_spatial=8, hidden_dim=8,
                            n_hidden_layers=1, alpha=0.05, max_substeps=64)
    with torch.no_grad():
        model.net[-1].weight.normal_(0.0, 0.02)
        model.net[-1].bias.normal_(0.0, 0.02)
    lost = LatentDynamics(latent_channels=4, latent_spatial=8, hidden_dim=8,
                           n_hidden_layers=1,
                           **integration_kwargs_from_config({"n_substeps": 1}))
    lost.load_state_dict(model.state_dict())  # loads cleanly -- that is the trap

    torch.manual_seed(3)
    z0 = torch.randn(4, 4, 8, 8)
    z1 = torch.randn(4, 4, 8, 8)
    dt = torch.full((4,), 500.0)
    theta = torch.zeros(4, 1)
    a, _, _ = model._integrate(z0.clone(), z1.clone(), dt, theta)
    b, _, _ = lost._integrate(z0.clone(), z1.clone(), dt, theta)
    assert not torch.allclose(a, b, rtol=1e-3), (
        "losing alpha on rebuild produced the same trajectory -- then this whole "
        "propagation machinery would be unnecessary, so something is wrong with "
        "the test rather than the code"
    )
