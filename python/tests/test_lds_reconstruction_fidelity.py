"""
Every place that rebuilds a LatentDynamics from a checkpoint must carry the
parameters that change what f_theta MEANS.

HISTORY, because it is the argument for how these tests are now written. There
were FIVE independent reconstructions, each with its own hand-written list of
fields, and each new field had to be added at all of them: dt_cap was fixed
three times before it was right (compare_rollout_training's own comment
recorded that fixing it elsewhere "did NOT fix it here"), n_substeps was missed
at all four sites at once, and alpha then reached four of five -- so stage 4
rebuilt an f_theta fitted at delta_t~36 and ran it ONE-SHOT at dt=500, skipping
325 of 325 batches by epoch 4.

The fields now come from ONE list (models.latent_dynamics._MEANING_FIELDS) via
integration_kwargs_from_config, so this file no longer asserts per-site
literals like '.get("dt_cap", float("inf"))'. Those assertions were themselves
a symptom: written once per site per field, they broke on the refactor that
made the whole class of bug impossible, while saying nothing about whether the
propagation was correct. The field-level checks live in
test_integration_config_propagation.py; what remains here is the SITE
enumeration and the historical-default guarantee, expressed against the shared
list.
"""
import ast
import pathlib

import pytest

from conftest import source_without_comments

_SITES = [
    "evaluation/_latent_eval.py",
    "evaluation/check_rollout.py",
    "training/model_assembly.py",
    "evaluation/compare_rollout_training.py",
]
_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Parameters that alter what the loaded weights MEAN or how they are applied.
# Not every constructor argument: only those a checkpoint records because the
# model behaves differently without them.
_SEMANTIC = ["dt_cap", "n_substeps", "latent_spatial_size"]


def _lds_call_kwargs(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "LatentDynamics"):
            # kw.arg is None for a ** splat -- recorded as "**" so a test can
            # require the shared list rather than individually named fields.
            return {kw.arg if kw.arg else "**" for kw in node.keywords}
    return None


@pytest.mark.parametrize("site", _SITES)
def test_every_reconstruction_site_exists(site):
    assert _lds_call_kwargs(_ROOT / site) is not None, f"no LatentDynamics(...) call in {site}"


@pytest.mark.parametrize("site", _SITES)
@pytest.mark.parametrize("param", ["dt_cap", "n_substeps"])
def test_semantic_parameters_are_restored_from_the_checkpoint(site, param):
    """
    Now satisfied by the SHARED list rather than by naming each field at each
    site: the call splats integration_kwargs_from_config, which supplies every
    entry of _MEANING_FIELDS. Kept parametrized over the fields anyway, so the
    failure message still names the one that went missing.
    """
    from models.latent_dynamics import _MEANING_FIELDS
    kwargs = _lds_call_kwargs(_ROOT / site)
    assert param in _MEANING_FIELDS, (
        f"{param} is no longer in the shared field list, so no site propagates it"
    )
    assert kwargs is not None and "**" in kwargs, (
        f"{site} does not splat the shared field list, so {param} reaches it only "
        f"if someone remembered to name it -- which is how {param} was missed before"
    )


@pytest.mark.parametrize("site", _SITES)
def test_the_latent_geometry_is_restored_too(site):
    kwargs = _lds_call_kwargs(_ROOT / site)
    assert "latent_spatial" in kwargs or "latent_spatial_size" in kwargs, site


def test_restored_with_a_historical_default_not_a_bare_lookup():
    """
    GUARDS `config["n_substeps"]`, which KeyErrors on every checkpoint written
    before the parameter existed. Now checked ONCE against the shared list --
    it uses .get(name, default) for every field, so the guarantee is structural
    rather than five copies of a literal.
    """
    from models.latent_dynamics import integration_kwargs_from_config
    restored = integration_kwargs_from_config({})   # a pre-everything checkpoint
    assert restored["dt_cap"] == float("inf")
    assert restored["n_substeps"] == 1
    assert restored["alpha"] is None
    # And it must not raise on a config missing every field, which is the
    # actual failure mode a bare lookup produces.
    assert set(restored) == set(integration_kwargs_from_config({"dt_cap": 1.0}))


# compare_integrators.py also rebuilds a LatentDynamics, but it is already
# BROKEN independently of any of this: it calls f_theta.forward_ab2(), a method
# that does not exist on LatentDynamics. It cannot run, so it is excluded here
# rather than held to a standard the rest of the file cannot meet -- but it is
# named explicitly so that reviving it means confronting this list.
_KNOWN_BROKEN = {"evaluation/compare_integrators.py"}


def test_the_known_broken_site_really_is_broken():
    """If compare_integrators is ever fixed, this fails and forces it into
    _SITES rather than letting it quietly rejoin the codebase unchecked."""
    from models.latent_dynamics import LatentDynamics
    src = source_without_comments(_ROOT / "evaluation/compare_integrators.py")
    assert "forward_ab2" in src
    assert not hasattr(LatentDynamics, "forward_ab2"), (
        "compare_integrators.py now runs -- add it to _SITES so its LatentDynamics "
        "reconstruction is checked for the semantic parameters too"
    )


def test_the_site_list_is_complete():
    """
    GUARDS a FIFTH reconstruction appearing and going unchecked -- the exact
    way n_substeps was missed at four places at once.
    """
    packages = ("evaluation", "training", "models", "orchestration", "utils")
    found = []
    for path in sorted(_ROOT.rglob("*.py")):
        if not path.parts or path.relative_to(_ROOT).parts[0] not in packages:
            continue
        if "test" in path.parts or path.name.startswith("test_"):
            continue
        try:
            src = source_without_comments(path)
        except (UnicodeDecodeError, OSError):
            continue
        if "LatentDynamics(" not in src or "class LatentDynamics" in src:
            continue
        if _lds_call_kwargs(path) is None:
            continue
        found.append(str(path.relative_to(_ROOT)).replace("\\", "/"))
    unexpected = set(found) - set(_SITES) - _KNOWN_BROKEN - {"training/train_lds.py"}
    assert not unexpected, (
        f"new LatentDynamics reconstruction(s) not covered by this test: {sorted(unexpected)}. "
        f"Add them to _SITES so they are checked for the semantic parameters too."
    )
