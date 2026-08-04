"""
A diagnostic must evaluate a model on the population it was TRAINED on, unless
explicitly told otherwise.

The demonstrated failure: check_parameter_dependence read min_step,
min_stdev_phi and min_passing_steps from the checkpoint's data_config but not
max_dt, so a stage-3 model trained with max_dt=200 was evaluated across dt up
to 25000. f_theta's contribution goes as f*dt^2/2, so that is a (25000/200)^2 =
15625x inflation of the correction term -- and the diagnostic duly reported
"f_theta makes the prediction WORSE on 88% of windows". The observed ratio of
means, 35915, matched the extrapolation factor rather than anything about
f_theta.

The same run at stage 3a (max_dt=150) gave 24315 against a predicted 27778:
two stages, independently, explained by the missing filter alone.

Evaluating outside the trained range is a legitimate thing to WANT, which is
why an explicit override still wins. It must be CHOSEN, not inherited by
omission.
"""
import inspect
import pathlib
import re

import pytest

from conftest import source_without_comments

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Everything that CONSUMES an f_theta and builds its own window population --
# diagnostics AND training stages. Stage 4/5 was originally absent, because
# this file was written for "diagnostics", and it duly went unfixed: stage 4
# built its dataset with no max_dt at all and reported val_loss = 2.7e29 with
# recon0 (0.21) and stats0 (25.1) sane beside it. The rule is not about
# diagnostics, it is about anything that applies f_theta.
#
# Stage-2 consumers such as check_deriv_temperature are deliberately absent:
# train_stage2 has no max_dt parameter at all, so there is nothing to inherit.
_STAGE3_DIAGNOSTICS = ["evaluation/_latent_eval.py", "evaluation/check_rollout.py"]
_F_THETA_CONSUMERS = _STAGE3_DIAGNOSTICS + ["training/train_refinement.py"]

_FILTERS = ["min_step", "min_stdev_phi", "max_dt"]


def _dataset_constructions(module):
    """Every MicrostructureEvolutionDataset(...) call, with balanced parens."""
    src = source_without_comments(_ROOT / module)
    out = []
    for m in re.finditer(r"MicrostructureEvolutionDataset\(", src):
        depth, i = 0, m.end() - 1
        while i < len(src):
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    out.append(src[m.start():i + 1])
                    break
            i += 1
    return out


@pytest.mark.parametrize("module", _F_THETA_CONSUMERS)
def test_EVERY_dataset_construction_gets_the_filters(module):
    """
    GUARDS filtering some datasets and not others.

    Reported: train_refinement has THREE constructions -- val, train, val --
    and a fix anchored on the val block reached two of them. Stage 4 then
    trained on 93,317 unfiltered windows against 4,776 filtered val ones, and
    the rollout term was NaN from epoch 1 while recon0 (170.9) was finite.

    Asserting the kwarg appears SOMEWHERE in the module passes in exactly that
    state, which is how it got through: the earlier version of this test did
    precisely that.
    """
    constructions = _dataset_constructions(module)
    assert constructions, f"no MicrostructureEvolutionDataset(...) in {module}"
    for i, call in enumerate(constructions):
        which = "train" if "train_dirs" in call else "val/other"
        assert "max_dt=max_dt" in call, (
            f"{module} construction #{i} ({which}) does not pass max_dt -- it will build "
            f"a window population f_theta never saw"
        )


@pytest.mark.parametrize("module", _F_THETA_CONSUMERS)
def test_every_f_theta_consumer_inherits_max_dt(module):
    """
    THE regression, generalised. Stage 4/5 reads its max_dt from the LDS
    component's provenance rather than from a data_config dict, so this checks
    for the inheritance PATTERN rather than one spelling of it.
    """
    src = source_without_comments(_ROOT / module)
    inherits = ("max_dt = max_dt if max_dt is not None else" in src)
    assert inherits, (
        f"{module} applies f_theta without inheriting max_dt -- it will build a "
        f"window population f_theta never saw, and f*dt^2/2 grows as the square"
    )


@pytest.mark.parametrize("module", _STAGE3_DIAGNOSTICS)
@pytest.mark.parametrize("filter_name", _FILTERS)
def test_stage3_diagnostics_inherit_every_dataset_filter(module, filter_name):
    src = (_ROOT / module).read_text(encoding="utf-8")
    inherits = re.search(rf'data_config\.get\("{filter_name}"|data_config\["{filter_name}"\]', src)
    assert inherits, (
        f"{module} does not take {filter_name} from the checkpoint's data_config -- it will "
        f"evaluate on a different population than the model was trained on"
    )


@pytest.mark.parametrize("module", _STAGE3_DIAGNOSTICS)
def test_the_inherited_filter_actually_reaches_the_dataset(module):
    """
    GUARDS resolving a filter from the config and then not passing it on --
    which looks correct at the resolution site and changes nothing. max_dt was
    in exactly that state for one commit.
    """
    src = (_ROOT / module).read_text(encoding="utf-8")
    # Balanced parens, not "up to the first )": an inner call such as
    # Path(...) closes first, truncating the slice before the kwargs and
    # failing on a file that is actually correct.
    calls = []
    for start in [m for m in range(len(src))
                  if src.startswith("MicrostructureEvolutionDataset(", m)]:
        depth, i = 0, start + len("MicrostructureEvolutionDataset")
        while i < len(src):
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    calls.append(src[start:i + 1])
                    break
            i += 1
    assert calls, f"no MicrostructureEvolutionDataset(...) found in {module}"
    assert any("max_dt=max_dt" in c for c in calls), (
        f"{module} resolves max_dt but never passes it to the dataset "
        f"({len(calls)} construction(s) checked)"
    )


@pytest.mark.parametrize("module", _STAGE3_DIAGNOSTICS)
def test_an_explicit_override_still_wins(module):
    """
    Inheritance must be a DEFAULT, not a lock: evaluating outside the trained
    dt range is a legitimate experiment, and the pattern
    `x = x if x is not None else data_config.get(...)` is what allows it.
    """
    src = (_ROOT / module).read_text(encoding="utf-8")
    assert re.search(r"max_dt = max_dt if max_dt is not None else", src), (
        f"{module} should prefer an explicitly-passed max_dt over the checkpoint's"
    )


def test_check_rollout_exposes_max_dt_to_its_caller():
    from evaluation.check_rollout import check_rollout
    assert "max_dt" in inspect.signature(check_rollout).parameters


def test_train_refinement_exposes_max_dt_and_min_passing_steps():
    """Stage 4/5 had NEITHER, so no caller could even correct it by hand."""
    from training.train_refinement import train_refinement
    params = inspect.signature(train_refinement).parameters
    for name in ("max_dt", "min_passing_steps"):
        assert name in params, f"train_refinement cannot filter on {name}"


def test_the_lds_component_carries_the_window_population_forward():
    """
    GUARDS load_lds_component dropping data_config. Stage 4/5 has no other
    route to f_theta's own filters -- it never opens the stage-3 checkpoint
    itself.
    """
    from training.checkpoint_components import load_lds_component
    src = source_without_comments(load_lds_component)
    assert '"data_config"' in src


def test_stage45_records_the_resolved_filters_for_stage5():
    """
    Stage 5 resumes stage 4's own joint checkpoint, so the population has to
    survive one more hop -- and the RESOLVED values, not the arguments, since
    they may have come from f_theta's data_config rather than the caller.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    # '"data_config": {' -- the SAVE. Matching '"data_config"' alone finds the
    # READ first (provenance.get("data_config")), which is a different thing
    # and made this test fail on correct code.
    saved = src[src.index('"data_config": {'):]
    saved = saved[:saved.index("}")]
    for key in ("max_dt", "min_passing_steps"):
        assert key in saved, f"stage 4/5 does not record {key} for stage 5"


def test_stage2_diagnostics_are_correctly_excluded():
    """
    Pins WHY check_deriv_temperature is not in the list, so a later reader does
    not "fix" it: train_stage2 has no max_dt parameter, so a stage-2 checkpoint
    cannot record one and there is nothing to inherit.
    """
    from training.train_stage2 import train_stage2
    assert "max_dt" not in inspect.signature(train_stage2).parameters
    from training.train_lds import train_lds
    assert "max_dt" in inspect.signature(train_lds).parameters


# --------------------------------------------------------------------
# z1_resync: meaningful in exactly one diagnostic
# --------------------------------------------------------------------

def test_check_rollout_accepts_and_inherits_z1_resync():
    """
    check_rollout CHAINS rollout() across transitions, so the two regimes
    genuinely differ there. Inherited from the checkpoint by default -- a
    diagnostic silently evaluating in a regime the model did not train in is
    the max_dt bug again.
    """
    from evaluation.check_rollout import check_rollout
    assert "z1_resync" in inspect.signature(check_rollout).parameters
    src = source_without_comments(_ROOT / "evaluation/check_rollout.py")
    assert 'lds_config.get("z1_resync", True)' in src, "must default to the checkpoint's own"
    # BOTH hops, checked separately. There are two: check_rollout ->
    # compute_sample, and compute_sample -> rollout(). Asserting the substring
    # once passes when only the inner hop survives, and compute_sample then
    # quietly uses its own default -- verified: dropping the outer hop left
    # this test green until it was split.
    assert "device, z1_resync=z1_resync" in src, (
        "the flag does not reach compute_sample -- it will use its default"
    )
    assert "z1_resync=z1_resync)" in src, "the flag does not reach rollout()"


def test_check_parameter_dependence_REFUSES_z1_resync_with_a_reason():
    """
    GUARDS silently accepting a flag that changes nothing. This diagnostic
    evaluates ONE forward() step per window with the real z1 supplied, so
    there is nothing propagated and nothing to resync -- and forward() does
    not sub-step either, so n_substeps is inert here for the same reason.

    Accepting it and doing nothing would be worse than the bare argparse
    error: two runs differing only in the flag would produce identical output
    and look like evidence that the regime does not matter.
    """
    src = source_without_comments(_ROOT / "evaluation/check_parameter_dependence.py")
    assert "--z1-resync" in src, "the flag must be accepted, to explain itself"
    assert "parser.error(" in src
    assert "check_rollout" in src, "the error must name the right tool"


def test_the_single_step_claim_is_true():
    """
    Pins the fact both tests above rest on: check_parameter_dependence's
    evaluation calls f_theta(...) directly -- one forward() step -- and never
    rollout(). If that ever changes, the refusal above becomes wrong.
    """
    src = source_without_comments(_ROOT / "evaluation/_latent_eval.py")
    assert "f_theta(z0_t, z1_t, dt, theta_b)" in src
    assert ".rollout(" not in src, (
        "_latent_eval now rolls out -- z1_resync has become meaningful there and "
        "check_parameter_dependence should stop refusing it"
    )


# --------------------------------------------------------------------
# max_dt must be OVERRIDABLE, or it cannot be evaluated
# --------------------------------------------------------------------

def test_check_parameter_dependence_accepts_a_max_dt_override():
    """
    max_dt is inherited from the checkpoint so the diagnostic reproduces the
    window population f_theta trained on -- correct, and the fix for a real
    bug. But with NO way to override it, the diagnostic can only ever look
    INSIDE that range, so it cannot answer whether the range was set too
    tightly. max_dt broke its own evaluation.

    Concretely: under max_dt=200 only three dt bins survive, which is why every
    Taylor coefficient came back consistent with zero, and within those bins
    the NORMALISED error FALLS with dt (2.15e-3 -> 0.80e-3 from dt=72 to 130)
    -- the opposite of an extrapolation limit, and a hint the cap is too low.
    Testing that hint requires looking outside.
    """
    from evaluation.check_parameter_dependence import check_parameter_dependence
    assert "max_dt" in inspect.signature(check_parameter_dependence).parameters
    src = source_without_comments(_ROOT / "evaluation/check_parameter_dependence.py")
    assert '"--max-dt"' in src, "no CLI flag"
    assert "max_dt=args.max_dt" in src, "the flag does not reach the function"


def test_the_override_still_defaults_to_inheriting():
    """
    GUARDS turning the override into a new default. Inheriting is right for
    every ordinary run -- the override exists for the deliberate, off-
    distribution question.
    """
    from evaluation.check_parameter_dependence import check_parameter_dependence
    assert (inspect.signature(check_parameter_dependence)
            .parameters["max_dt"].default is None)
    src = source_without_comments(_ROOT / "evaluation/_latent_eval.py")
    assert 'max_dt = max_dt if max_dt is not None else data_config.get("max_dt")' in src


def test_the_header_says_whether_an_override_is_in_force():
    """
    The old header read "(from checkpoint's own data_config unless overridden
    above)" -- promising an override that did not exist. Worse than silence:
    a reader would assume the range shown was a choice rather than a ceiling
    they could not lift.
    """
    src = source_without_comments(_ROOT / "evaluation/_latent_eval.py")
    assert "unless overridden above" not in src
    assert "_max_dt_provenance" in src
    assert "OFF-DISTRIBUTION" in src, (
        "an overridden run must say so -- its numbers are not comparable to an "
        "inherited one"
    )


# --------------------------------------------------------------------
# the diagnostics must use the shared latent cache
# --------------------------------------------------------------------

def test_the_shared_loader_passes_latent_cache_dir_to_the_dataset():
    """
    The trainers had been caching latents since the feature landed; the
    DIAGNOSTICS re-encoded their whole population on every run. Nothing about
    the cache is training-specific -- the key is the encoder's own fingerprint
    plus the run and step list -- and a diagnostic uses a frozen encoder
    straight out of a checkpoint, usually the one the trainer just wrote.

    Worst on an off-distribution run: --max-dt large disables the prefix
    truncation, so every frame of every run is encoded. That is the "unusually
    slow" the max_dt=2000 sweep reported.
    """
    import inspect

    from evaluation._latent_eval import _load_ae_f_theta_and_dataset
    assert "latent_cache_dir" in inspect.signature(_load_ae_f_theta_and_dataset).parameters
    # Scoped to the DATASET CONSTRUCTION, not the file: the same kwarg also
    # appears where _load_models_and_dataset forwards it, so a file-level
    # substring check passed even with the dataset call stripped -- verified.
    src = source_without_comments(_ROOT / "evaluation/_latent_eval.py")
    call = src[src.index("MicrostructureEvolutionDataset("):]
    call = call[:call.index(")\n")]
    assert "latent_cache_dir=latent_cache_dir" in call, (
        "the parameter must reach MicrostructureEvolutionDataset itself"
    )


def test_the_cache_root_has_ONE_definition():
    """
    GUARDS a second literal path. Trainer and diagnostic must land in the SAME
    cache or the sharing is theoretical -- and the pipeline had the only copy,
    written inline.
    """
    from orchestration.paths import default_latent_cache_dir
    root = _ROOT
    assert default_latent_cache_dir(root) == root / "checkpoints" / "latent_cache"
    pipeline_src = source_without_comments(_ROOT / "orchestration/pipeline.py")
    assert "default_latent_cache_dir(_PYTHON_ROOT)" in pipeline_src, (
        "pipeline must call the shared helper"
    )
    assert '"checkpoints" / "latent_cache"' not in pipeline_src, (
        "pipeline still builds the path inline -- a second definition to drift"
    )


def test_none_can_still_turn_the_cache_OFF():
    """
    GUARDS defaulting with `latent_cache_dir or default(...)`: None is a
    MEANINGFUL value (caching disabled), so it cannot double as "not
    specified". A sentinel is required, and --no-latent-cache depends on it.
    """
    import inspect

    from evaluation.check_parameter_dependence import check_parameter_dependence
    default = inspect.signature(check_parameter_dependence).parameters["latent_cache_dir"].default
    assert default is not None, "None as the default would make the cache unturnoffable"
    src = source_without_comments(_ROOT / "evaluation/check_parameter_dependence.py")
    assert "_UNSET_CACHE" in src
    assert '"--no-latent-cache"' in src
    assert "None if args.no_latent_cache else _UNSET_CACHE" in src


# --------------------------------------------------------------------
# the ephemeral stage-3 wrapper must not encode a full val set
# --------------------------------------------------------------------

def test_the_ephemeral_wrapper_forces_a_minimal_val_set():
    """
    Converting an AE-family checkpoint runs train_lds with epochs=0. That
    already skips the TRAIN set (never iterated), but the VAL set was still
    built and fully ENCODED -- ~30,000 windows on the 128 sweep, each pushed
    through the frozen AE's forward pass -- to produce one val_loss that
    ensure_lds_checkpoint's own NOTE calls uninformative.

    It was the dominant cost of every diagnostic run against a stage-1/2
    checkpoint.
    """
    src = source_without_comments(_ROOT / "orchestration/checkpoint_identification.py")
    assert '_ephemeral_kwargs["val_fraction"] = 1e-9' in src
    assert "**_ephemeral_kwargs," in src, "the override must actually be passed"


def test_shrinking_val_does_not_move_the_TEST_dirs():
    """
    The safety argument, exercised rather than asserted in prose: the
    diagnostic evaluates TEST windows, so a change to val_fraction would be
    silently corrupting if it repartitioned them.

    split_run_dirs carves test FIRST (perm[:n_test]), so val shrinks into
    train -- which epochs=0 skips anyway.
    """
    from pathlib import Path

    from training.datasets import split_run_dirs

    dirs = [Path(f"r{i}") for i in range(4277)]
    _, val_big, test_big = split_run_dirs(dirs, val_fraction=0.2, test_fraction=0.1, seed=0)
    _, val_small, test_small = split_run_dirs(dirs, val_fraction=1e-9, test_fraction=0.1, seed=0)

    assert test_big == test_small, "shrinking val repartitioned the TEST set"
    assert len(val_small) == 1 and len(val_big) > 100


def test_val_fraction_is_not_zero():
    """
    GUARDS 1e-9 -> 0. The epoch-0 evaluation still runs, and an empty val set
    divides by zero. One dir is enough for the structurally valid checkpoint
    the wrapper exists to produce.
    """
    from pathlib import Path

    from training.datasets import split_run_dirs
    _, val, _ = split_run_dirs([Path(f"r{i}") for i in range(4277)],
                                val_fraction=1e-9, test_fraction=0.1, seed=0)
    assert len(val) >= 1
