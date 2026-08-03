"""
Every place that rebuilds a LatentDynamics from a checkpoint must carry the
parameters that change what f_theta MEANS.

There are FOUR independent reconstructions -- evaluation/_latent_eval.py,
evaluation/check_rollout.py, training/model_assembly.py and
evaluation/compare_rollout_training.py -- and compare_rollout_training's own
comment records that fixing dt_cap in the others did not fix it there. So this
is checked collectively, once, rather than trusting four copies to stay in
step.

n_substeps was missed at all four when it was introduced. The consequence is
silent and severe: a checkpoint trained at n_substeps=N carries a POINTWISE
z1_dot, and rebuilding it at n_substeps=1 applies that as a one-shot corrector
over the whole dt -- the "NOT equivalent" direction train_lds explicitly warns
about on resume. The weights load cleanly, so no error is raised; the
diagnostic simply reports the wrong model.
"""
import ast
import pathlib

import pytest

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
            return {kw.arg for kw in node.keywords if kw.arg}
    return None


@pytest.mark.parametrize("site", _SITES)
def test_every_reconstruction_site_exists(site):
    assert _lds_call_kwargs(_ROOT / site) is not None, f"no LatentDynamics(...) call in {site}"


@pytest.mark.parametrize("site", _SITES)
@pytest.mark.parametrize("param", ["dt_cap", "n_substeps"])
def test_semantic_parameters_are_restored_from_the_checkpoint(site, param):
    """
    GUARDS a reconstruction that drops one of these and silently gets the
    constructor default. latent_spatial_size is excluded from the parametrize
    because one site names it `latent_spatial`; the check below covers it.
    """
    kwargs = _lds_call_kwargs(_ROOT / site)
    assert param in kwargs, (
        f"{site} rebuilds LatentDynamics without {param} -- it will silently get the "
        f"constructor default, which is not what the checkpoint was trained with"
    )


@pytest.mark.parametrize("site", _SITES)
def test_the_latent_geometry_is_restored_too(site):
    kwargs = _lds_call_kwargs(_ROOT / site)
    assert "latent_spatial" in kwargs or "latent_spatial_size" in kwargs, site


@pytest.mark.parametrize("site", _SITES)
def test_restored_with_a_historical_default_not_a_bare_lookup(site):
    """
    GUARDS `config["n_substeps"]`, which KeyErrors on every checkpoint written
    before the parameter existed. The project convention is
    .get(key, <the value that reproduces the old behaviour>) -- inf for dt_cap,
    1 for n_substeps.
    """
    src = (_ROOT / site).read_text(encoding="utf-8")
    for param, default in (("dt_cap", 'float("inf")'), ("n_substeps", "1")):
        assert f'.get("{param}", {default})' in src, (
            f"{site} must use .get(\"{param}\", {default}) so pre-{param} checkpoints "
            f"still load with their original behaviour"
        )


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
    src = (_ROOT / "evaluation/compare_integrators.py").read_text(encoding="utf-8")
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
            src = path.read_text(encoding="utf-8")
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
