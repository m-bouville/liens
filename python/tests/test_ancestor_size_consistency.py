"""
An ancestor checkpoint's recorded settings must match the run that loads it.

The incident this exists for: a params file for 128x128 with a mistyped
`resume_from` pointing at 64x64-stage2.pt. train_stage2 derives
`size = model_cfg["size"]` from the ancestor and uses that SAME size to locate
the dataset -- so the run built a 64x64 architecture, trained it on
datasets/64x64, and wrote the result to checkpoints/stage2/128x128-stage2.pt,
because the output filename comes from the params file rather than the model.

Nothing failed. Every printed number was internally consistent. The only
reason it was recoverable is that the stage-2 backup fired.

That makes this categorically worse than the dt_cap / n_substeps cross-checks:
those change what trained weights MEAN, this changes WHICH DATA IS READ.
"""
import inspect
import pathlib

import pytest

from conftest import source_without_comments

from training.checkpoint_components import cross_check_ancestor_config

_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------
# the helper
# --------------------------------------------------------------------

def test_a_size_mismatch_is_refused():
    with pytest.raises(ValueError, match="size"):
        cross_check_ancestor_config({"size": 64}, {"size": 128}, "anc.pt")


def test_the_message_names_both_values_and_the_way_out():
    """A bare 'mismatch' would leave the reader to work out which is which,
    and the fix (port, not resume) is not obvious from the error alone."""
    with pytest.raises(ValueError) as e:
        cross_check_ancestor_config({"size": 64}, {"size": 128}, "anc.pt")
    msg = str(e.value)
    assert "64" in msg and "128" in msg
    assert "port_checkpoint" in msg
    assert "dataset" in msg


def test_a_match_passes_and_int_float_do_not_spuriously_differ():
    cross_check_ancestor_config({"size": 128}, {"size": 128}, "anc.pt")
    cross_check_ancestor_config({"size": 128}, {"size": 128.0}, "anc.pt")


def test_expected_none_means_no_opinion():
    """Callers that genuinely do not know the size (a bare CLI invocation with
    no params file) must still work -- silence, not a false alarm."""
    cross_check_ancestor_config({"size": 64}, {"size": None}, "anc.pt")


def test_a_key_absent_from_the_checkpoint_is_not_a_mismatch():
    """A checkpoint predating a field cannot be said to disagree about it."""
    cross_check_ancestor_config({}, {"size": 128}, "anc.pt")


def test_every_mismatch_is_reported_not_just_the_first():
    with pytest.raises(ValueError) as e:
        cross_check_ancestor_config({"size": 64, "latent_channels": 8},
                                     {"size": 128, "latent_channels": 16}, "anc.pt")
    msg = str(e.value)
    assert "size" in msg and "latent_channels" in msg


# --------------------------------------------------------------------
# every stage that adopts an ancestor's size must check it first
# --------------------------------------------------------------------

_STAGES = {
    "training/train_stage2.py": "train_stage2",
    "training/train_refinement.py": "train_refinement",
    "training/train_lds.py": "train_lds",
}


@pytest.mark.parametrize("module_path,func_name", list(_STAGES.items()))
def test_stage_accepts_an_explicit_size(module_path, func_name):
    """
    Stages 2 and 4/5 INFERRED size from their ancestor and the pipeline passed
    them none, so no check was even possible. The parameter is optional (None =
    no opinion) so direct CLI use is unaffected.
    """
    import importlib
    mod = importlib.import_module(module_path.replace("/", ".").removesuffix(".py"))
    assert "size" in inspect.signature(getattr(mod, func_name)).parameters, (
        f"{func_name} cannot cross-check its ancestor without being told the expected size"
    )


@pytest.mark.parametrize("module_path", list(_STAGES))
def test_stage_actually_calls_the_check(module_path):
    src = source_without_comments(_ROOT / module_path)
    assert "cross_check_ancestor_config(" in src, (
        f"{module_path} adopts an ancestor's config without checking it"
    )


@pytest.mark.parametrize("module_path", list(_STAGES))
def test_the_check_precedes_adopting_the_size(module_path):
    """
    GUARDS checking AFTER `size = <ancestor>["size"]`, which would compare the
    ancestor against itself and always pass.
    """
    src = source_without_comments(_ROOT / module_path)
    check_at = src.index("cross_check_ancestor_config(")
    for adopt in ('size = model_cfg["size"]', 'size = components["encoder"].config["size"]'):
        if adopt in src:
            assert check_at < src.index(adopt), (
                f"{module_path} adopts the ancestor's size before checking it"
            )


def test_the_pipeline_passes_size_to_every_stage():
    """
    The check is inert unless the pipeline states what it expects. Stage 1 and
    stage 3 already passed size; stages 2 and 4/5 did not, which is why the
    64x64 ancestor went unnoticed.
    """
    src = source_without_comments(_ROOT / "orchestration/pipeline.py")
    for call, marker in (("train_stage2", "size=size, base_path=base_path, resume_from="),
                          ("train_refinement", "size=size, base_path=base_path, freeze_decoder=")):
        assert marker in src, f"pipeline does not pass size to {call}"
