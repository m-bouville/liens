"""
Tests for orchestration/stage_params.py -- parsing was already exercised
indirectly through real pipeline runs, but the global-defaults merge
mechanism in _prepare_stage_kwargs (a stage inheriting a value it never
specified from the file's preamble, while a stage that DOES specify its
own value still wins) had zero dedicated test coverage before this.
"""
from pathlib import Path

import pytest

from orchestration.stage_params import (
    _prepare_stage_kwargs, _resolve_same, parse_stage_params,
)


def _write(tmp_path, text):
    path = tmp_path / "params.txt"
    path.write_text(text)
    return path


# ---- _prepare_stage_kwargs: global-defaults merge --------------------

def test_stage_specific_value_wins_over_global_default():
    global_params = {"min_step": "4000", "min_stdev_phi": "0.01"}
    raw_stage = {"min_stdev_phi": "0.05"}  # this stage overrides
    kwargs = _prepare_stage_kwargs(raw_stage, global_params)
    assert kwargs["min_step"] == 4000        # inherited from global, untouched
    assert kwargs["min_stdev_phi"] == 0.05   # stage's own value wins


def test_stage_inherits_global_default_when_key_entirely_absent():
    """THE actual feature being added: a stage that never mentions a key
    at all (not even '= same') still gets the global value."""
    global_params = {"min_step": "4000", "min_stdev_phi": "0.01", "min_std_deriv": "0.0002"}
    raw_stage = {"epochs": "12"}  # doesn't mention any of the three at all
    kwargs = _prepare_stage_kwargs(raw_stage, global_params)
    assert kwargs["min_step"] == 4000
    assert kwargs["min_stdev_phi"] == 0.01
    assert kwargs["min_std_deriv"] == 0.0002
    assert kwargs["epochs"] == 12


def test_no_global_params_behaves_exactly_as_before():
    """Backward compat: omitting global_params entirely (every existing
    call site before this feature) must behave identically to before
    this parameter existed."""
    raw_stage = {"epochs": "12", "batches": "128"}
    kwargs = _prepare_stage_kwargs(raw_stage)
    assert kwargs == {"epochs": 12, "batch_size": 128}


def test_global_defaults_mechanism_is_general_not_key_specific():
    """Deliberately not scoped to min_step/min_stdev_phi/min_std_deriv/
    force specifically -- ANY key given globally applies to any stage
    that happens to accept a parameter by that name."""
    global_params = {"stats_weight": "0.02"}
    raw_stage = {"epochs": "5"}
    kwargs = _prepare_stage_kwargs(raw_stage, global_params)
    assert kwargs["stats_weight"] == 0.02


def test_key_renaming_still_applies_to_globally_inherited_keys():
    """A key inherited purely from the global section still goes
    through the same renaming (patience -> early_stopping_patience) as
    a stage-local one -- the merge happens on RAW keys, before renaming,
    so both paths are treated identically."""
    global_params = {"patience": "4"}
    kwargs = _prepare_stage_kwargs({}, global_params)
    assert kwargs == {"early_stopping_patience": 4}


def test_list_valued_keys_still_convert_correctly_when_inherited(monkeypatch):
    """latent_names/latent_modes (the ORIGINAL reason this mechanism
    existed) are gone now -- _LIST_VALUED_KEYS is empty by default (see
    its own comment) -- but the comma-split MECHANISM itself is still
    real, general infrastructure, tested here via a synthetic key
    rather than a dead one."""
    import orchestration.stage_params as stage_params_module
    monkeypatch.setattr(stage_params_module, "_LIST_VALUED_KEYS", {"some_list_param"})
    global_params = {"some_list_param": "a, b, c"}
    kwargs = _prepare_stage_kwargs({}, global_params)
    assert kwargs["some_list_param"] == ["a", "b", "c"]


# ---- End-to-end: parse_stage_params + _prepare_stage_kwargs together --

def test_end_to_end_global_preamble_and_per_stage_override(tmp_path):
    path = _write(tmp_path, """
base = ../datasets
Nx = 32
Ny = 32
min_step = 4000
min_stdev_phi = 0.01
min_std_deriv = 0.0002
force = True

# Stage 1
epochs = 12

# Stage 2
min_std_deriv = 0.0005
epochs = 20
""")
    global_params, stages = parse_stage_params(path)

    stage1_kwargs = _prepare_stage_kwargs(stages.get(1, {}), global_params)
    assert stage1_kwargs["min_step"] == 4000
    assert stage1_kwargs["min_stdev_phi"] == 0.01
    assert stage1_kwargs["min_std_deriv"] == 0.0002
    assert stage1_kwargs["force"] is True
    assert stage1_kwargs["epochs"] == 12

    stage2_kwargs = _prepare_stage_kwargs(stages.get(2, {}), global_params)
    assert stage2_kwargs["min_step"] == 4000          # still inherited
    assert stage2_kwargs["min_stdev_phi"] == 0.01      # still inherited
    assert stage2_kwargs["min_std_deriv"] == 0.0005    # stage 2's own override wins
    assert stage2_kwargs["force"] is True              # still inherited
    assert stage2_kwargs["epochs"] == 20

    # base/Nx/Ny are pipeline-level settings, not training-function
    # kwargs -- real callers (pipeline.py) pop them out of global_params
    # before this merge happens, specifically so they never show up
    # here (and never get flagged as "unrecognized" for every stage).
    # This test passes global_params UNMODIFIED (no pop simulated) to
    # confirm the base/Nx/Ny case really does need that pop upstream --
    # if this assertion ever fails, it means _prepare_stage_kwargs
    # itself started filtering them, which would make the pop
    # redundant, worth noticing either way.
    assert "base" in stage1_kwargs and "Nx" in stage1_kwargs


def test_same_still_requires_explicit_mention_unlike_global_default(tmp_path):
    """'= same' and omitting the key entirely are different mechanisms
    (see _prepare_stage_kwargs' own docstring) -- confirm '= same'
    still works standalone, independent of the new global-default path."""
    path = _write(tmp_path, """
min_stdev_phi = 0.01

# Stage 1
min_stdev_phi = 0.03

# Stage 2
min_stdev_phi = same
""")
    global_params, stages = parse_stage_params(path)
    # Stage 2's '= same' should resolve to stage 1's value (nearest
    # preceding stage), NOT the global section's -- _resolve_same
    # checks preceding stages before falling back to global.
    assert stages[2]["min_stdev_phi"] == "0.03"


def test_stage_1a_header_is_an_alias_for_plain_stage_1(tmp_path):
    """Regression test for a real reported bug: '# Stage 1a' (a more
    natural, consistent name pairing with '# Stage 1b') was silently
    never read at all -- the pipeline only ever looked for the bare
    '# Stage 1' key, so every param in a '# Stage 1a' section was
    dropped without any error, surfacing as a confusing "missing
    stats_weight" failure far from the actual cause. '1a' must
    normalize to the SAME int key (1) plain '# Stage 1' uses -- unlike
    '3a'/'3b', which stay their own distinct string keys (two genuinely
    different curriculum phases of stage 3, not aliases for anything)."""
    path = _write(tmp_path, """
# Stage 1a
stats_weight = 0.01
epochs = 20

# Stage 1b
stats_weight = 0.02
epochs = 5
""")
    global_params, stages = parse_stage_params(path)
    assert 1 in stages, "'# Stage 1a' should have been normalized to the int key 1"
    assert "1a" not in stages, "'1a' should NOT remain as its own separate string key"
    assert stages[1]["stats_weight"] == "0.01"
    assert stages[1]["epochs"] == "20"

    # '1b' must NOT be touched by this normalization -- it's a
    # genuinely distinct stage (train_stage1b), not an alias.
    assert "1b" in stages
    assert stages["1b"]["stats_weight"] == "0.02"
    assert stages["1b"]["epochs"] == "5"


def test_stage_1_and_1a_headers_are_interchangeable(tmp_path):
    """Whichever spelling is used, the SAME stage-1 section is
    populated -- confirms this isn't just 1a happening to work, but
    that both forms genuinely mean the same thing."""
    path_bare = tmp_path / "bare.txt"
    path_bare.write_text("# Stage 1\nstats_weight = 0.05\n")
    path_1a = tmp_path / "alt.txt"
    path_1a.write_text("# Stage 1a\nstats_weight = 0.05\n")

    _, stages_bare = parse_stage_params(path_bare)
    _, stages_1a = parse_stage_params(path_1a)
    assert stages_bare[1]["stats_weight"] == stages_1a[1]["stats_weight"] == "0.05"
