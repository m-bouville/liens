"""
Stage 4/5 must apply f_theta in the regime it was trained in, and must say so
when its loss scales are inherited from a stage where the same quantity had a
different magnitude.

Both found by auditing the 2 -> 3a -> 3b -> 4 -> 5 chain rather than any one
stage: individually every stage looked right, and the defects are entirely in
the handoffs.
"""
import inspect
import pathlib

import pytest

from conftest import source_without_comments

_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------
# regime: z1_resync
# --------------------------------------------------------------------

def test_stage45_rollout_takes_z1_resync():
    """
    REGRESSION: compute_stage45_loss called
    `f_theta.rollout(z0, z1_sequence, dt_window, theta)` with no z1_resync, so
    it used the default True (teacher-forced) whatever stage 3b trained with.

    An f_theta trained at z1_resync=False expects z1 to be PROPAGATED, not
    reset at each real frame -- the same "NOT equivalent" direction that
    n_substeps N -> 1 is, and n_substeps IS inherited. Missed because the
    rollout call lives in refinement_loss.py rather than beside the model
    construction in model_assembly.py.
    """
    from training.refinement_loss import compute_stage45_loss
    assert "z1_resync" in inspect.signature(compute_stage45_loss).parameters
    src = source_without_comments(_ROOT / "training/refinement_loss.py")
    assert "z1_resync=z1_resync" in src, "the parameter must reach rollout()"


def test_stage45_inherits_the_regime_from_the_lds_checkpoint():
    """Not a default, and not a params-file value: the regime is a property of
    the f_theta being refined."""
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert 'config.get("z1_resync", True)' in src, "must read it from the LDS component"
    assert "z1_resync=lds_z1_resync" in src, "must pass it to the loss"


def test_all_three_semantic_parameters_reach_stage45():
    """
    n_substeps, max_dt and z1_resync are one rule -- reproduce the conditions
    f_theta was trained under -- and each was fixed separately, at a different
    time, after a different symptom. Checked together so the next one is not
    a fourth incident.
    """
    refinement = source_without_comments(_ROOT / "training/train_refinement.py")
    assembly = source_without_comments(_ROOT / "training/model_assembly.py")
    assert 'lds_cfg.get("n_substeps", 1)' in assembly
    assert 'lds_data_config.get("max_dt")' in refinement
    assert 'config.get("z1_resync", True)' in refinement


# --------------------------------------------------------------------
# scales
# --------------------------------------------------------------------

def test_stage45_warns_when_one_component_dominates():
    """
    rollout_scale=1e-6 is right for stage 3, which FREEZES the encoder so
    f_theta sees the latents it was fitted to (raw rollout ~1e-6). Stage 4
    unfreezes it, f_theta is immediately off-distribution, and the same
    quantity is ~0.7 -- 6e5 times larger. A single global rollout_scale then
    makes stage 4's loss 99.997% rollout, with recon0+stats0 at 26 parts per
    million.

    Stage 4 exists for the balance between those terms, so this is not a
    cosmetic imbalance -- it removes the decoder tether the stage is built
    around, and it is invisible without undoing the scale arithmetic by hand.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "sh > 0.99" in src, "no dominance check"
    assert "WARNING" in src
    # Reported once the ramp completes, not at epoch 1 -- during a warmup the
    # imbalance is deliberate. See test_the_dominance_warning_waits_for_the_ramp.
    assert "epoch == max(1, rollout_weight_warmup_epochs)" in src, (
        "must report exactly once, at the first epoch the full objective is in effect"
    )


def test_the_dominance_warning_names_the_raw_magnitudes():
    """
    A percentage alone does not say what to set the scale TO. The raw values
    are exactly the numbers a corrected *_scale should equal.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    warning = src[src.index("sh > 0.99"):]
    warning = warning[:warning.index("if saved_this_epoch")]
    assert "Raw values" in warning
    assert "_raw_str" in warning and "_suggest" in warning, (
        "the report must name the raw magnitudes AND the scale they imply"
    )


@pytest.mark.parametrize("scale", ["rollout_scale", "recon0_scale", "stats0_scale"])
def test_every_scale_defaults_to_one(scale):
    """
    GUARDS baking a stage-specific scale into the signature. The defaults must
    be neutral so that a params file's value is the ONLY source -- which is
    also what makes a global value's misuse visible rather than silently
    overridden.
    """
    from training.train_refinement import train_refinement
    assert inspect.signature(train_refinement).parameters[scale].default == 1.0


# --------------------------------------------------------------------
# rollout_weight warmup
# --------------------------------------------------------------------

def test_stage45_has_a_rollout_weight_warmup():
    """
    Stage 2 has deriv_weight_warmup_epochs; stage 3 has lr_warmup_steps;
    stage 4/5 had neither -- and needs one for a sharper reason than either.

    Its f_theta is FROZEN. The encoder starts on exactly the distribution
    f_theta was fitted to, so a full-strength rollout gradient drives it off
    that distribution, which raises the rollout loss, which pulls harder. On a
    real run the raw rollout fell 6e9 between epoch 1 and epoch 10, so no
    single rollout_scale serves both ends: rollout_scale=1 stalled the run,
    10 was the largest the transient tolerated.
    """
    from training.train_refinement import train_refinement
    params = inspect.signature(train_refinement).parameters
    assert "rollout_weight_warmup_epochs" in params
    assert params["rollout_weight_warmup_epochs"].default == 0, (
        "must default to 0 -- an unrequested ramp would silently change every "
        "existing run's first epochs"
    )


def test_the_ramp_is_GEOMETRIC_not_linear():
    """
    Linear is what stage 2 uses for deriv_weight, and it is wrong here: the
    quantities behave completely differently. L_deriv is O(1) throughout;
    L_rollout collapses ~6e9 over ten epochs (1.76e9 -> 0.29, measured). A
    linear ramp at epoch 1 gives 0.10 * 1.76e9 = 1.76e8 -- eight orders of
    magnitude above the converged contribution, so it barely softens the
    transient it exists to absorb.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "start_fraction ** (1.0 - frac)" in src, "the ramp must be geometric"
    assert "min(1.0, epoch / rollout_weight_warmup_epochs)" not in src, (
        "the linear ramp is back"
    )
    # The no-op branch lives in the extracted helper now, under its own
    # parameter name -- checked behaviourally rather than by matching the
    # caller's variable name, which is what the earlier version did.
    assert "warmup_epochs <= 0" in src
    assert _ramp(1, 0) == 1.0 and _ramp(99, 0) == 1.0


def _ramp(epoch, warmup_epochs, start=1e-6, weight=1.0):
    """The PRODUCTION formula, imported -- not a re-implementation.

    An earlier version of this file reimplemented it, so an off-by-one
    mutation in train_refinement left every endpoint assertion green: the
    tests were checking their own arithmetic. That is why the ramp is a
    module-level function rather than an expression inline in the epoch loop.
    """
    from training.train_refinement import geometric_warmup_weight
    return geometric_warmup_weight(epoch, weight, warmup_epochs, start)


def test_the_ramp_endpoints_are_exact():
    """Epoch 1 must be the start value and the last warmup epoch exactly the
    full weight -- an off-by-one here either skips the protection entirely or
    never reaches full strength."""
    assert _ramp(1, 10) == pytest.approx(1e-6)
    assert _ramp(10, 10) == 1.0
    assert _ramp(11, 10) == 1.0
    assert _ramp(1, 0) == 1.0, "warmup disabled must be an exact no-op at every epoch"


def test_the_ramp_flattens_the_CONTRIBUTION_not_the_weight():
    """
    The point of geometric: weight * raw should be roughly constant across the
    ramp, because that is the gradient scale the optimiser actually sees. With
    the measured raw values a linear ramp spans ~9 decades of contribution and
    the geometric one spans ~1 (after epoch 1).
    """
    raw = {2: 7.3859e4, 3: 8.2262e3, 5: 7.3873e2, 10: 2.93e-1}
    geo = [_ramp(e, 10) * v for e, v in raw.items()]
    lin = [min(1.0, e / 10) * v for e, v in raw.items()]
    assert max(geo) / min(geo) < 100, f"geometric contributions span too much: {geo}"
    assert max(lin) / min(lin) > 1000, f"expected linear to span far more: {lin}"


def test_the_start_value_is_configurable():
    from training.train_refinement import train_refinement
    params = inspect.signature(train_refinement).parameters
    assert "rollout_weight_warmup_start" in params
    assert params["rollout_weight_warmup_start"].default == 1e-6


def test_only_the_TRAIN_step_is_ramped():
    """
    GUARDS ramping val too. The ramp exists to protect the training GRADIENT
    from a transient; val_loss is never backpropagated, so it never needed
    protection -- and a ramped val_loss is not comparable across epochs or
    against a run without a warmup, which would corrupt the save criterion.

    Same reasoning stage 2 records for val_deriv_weight.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "effective_rollout_weight=effective_rollout_weight" in src
    val_call = src[src.index("step(batch, train=False"):] if "step(batch, train=False" in src else ""
    assert "effective_rollout_weight" not in val_call[:200], (
        "the validation step must use the full rollout_weight"
    )


def test_the_printed_train_component_uses_the_EFFECTIVE_weight():
    """
    GUARDS printing rollout_weight while optimising a ramped one: the columns
    would not sum to the train_loss beside them, and the discrepancy reads as
    a bug in the loss rather than as the ramp working.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    import re
    assert "effective_rollout_weight*train_rollout/rollout_scale" in src
    # val keeps the FULL weight, so its columns stay comparable across epochs
    # and against runs without a warmup.
    #
    # Negative lookbehind, not a plain substring: "effective_rollout_weight*
    # val_rollout" CONTAINS "rollout_weight*val_rollout", so the plain check
    # passed even after val was wrongly ramped -- verified.
    assert re.search(r"(?<!effective_)rollout_weight\*val_rollout/rollout_scale", src), (
        "the validation column must use the unramped rollout_weight"
    )


# --------------------------------------------------------------------
# the ramp's interaction with the save criterion
# --------------------------------------------------------------------

def test_the_criterion_is_reset_when_the_ramp_completes():
    """
    val_loss is deliberately never ramped, so DURING the warmup it measures
    the full objective against a model not yet trained on it -- legitimately
    terrible numbers that then dominate the EMA for many epochs.

    Observed: epoch 1 val_loss 33.86 (a ramp transient); at epoch 13 the EMA
    was still 0.656, still forgetting it, while val_loss had stopped improving
    at epoch 11 and was oscillating in 0.10-0.12. Every epoch cleared the
    descending bar and saved, so the criterion was BLIND and early stopping
    could not fire -- the run would have kept whichever epoch happened to be
    last rather than the best.

    Same situation stage 2 handles with reset_with_grace when
    deriv_target_centered switches.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "tracker.reset_with_grace(grace)" in src
    assert "epoch == rollout_weight_warmup_epochs" in src, (
        "the reset must happen exactly when the ramp completes"
    )


def test_the_reset_grace_is_at_least_two_epochs():
    """
    GUARDS max(1, ...). A single-epoch grace is mathematically IDENTICAL to no
    grace: with only one epoch there is no second value for the EMA to blend
    with, so best_val_loss ends up as that epoch's own raw value, lucky or
    not. Stage 2 records the same derivation.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "max(2, round(1 / (1 - val_ema_decay)))" in src


def test_the_reset_is_clamped_against_the_remaining_epochs():
    """A grace covering every remaining epoch means NO checkpoint is written
    after the reset -- a missing file rather than a worse one."""
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "clamp_grace_epochs(grace, epochs - epoch + 1)" in src


def test_no_reset_when_there_is_no_warmup():
    """GUARDS resetting unconditionally, which would discard a perfectly good
    criterion history for every existing run."""
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    guard = src[src.index("if (rollout_weight_warmup_epochs > 0"):]
    guard = guard[:guard.index("tracker.reset_with_grace")]
    assert "rollout_weight_warmup_epochs > 0" in guard


def test_the_dominance_warning_waits_for_the_ramp():
    """
    GUARDS reporting at epoch 1. During a warmup the imbalance is DELIBERATE,
    so an epoch-1 reading describes the transient rather than the scales: on a
    real run it reported raw rollout=168 and implied rollout_scale~168 when the
    converged value was 0.053. Acting on that would have undone the ramp.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "epoch == max(1, rollout_weight_warmup_epochs)" in src
    assert "if epoch == 1:" not in src, "the warning must not fire during the ramp"


# --------------------------------------------------------------------
# the 2 -> 3a -> 3b -> 4 -> 5 chain
# --------------------------------------------------------------------

def test_the_joint_loader_carries_data_config_for_stage5():
    """
    REGRESSION at the 4 -> 5 handoff, the other half of a chain whose first
    half was already fixed.

    load_lds_component carries data_config so stage 4 can inherit f_theta's
    max_dt. load_joint_refinement_checkpoint -- which stage 5 uses when it
    RESUMES stage 4 -- did not, so stage 5 got max_dt=None and trained on the
    full dt range. That is exactly the defect that gave stage 4 a val_loss of
    2.7e29 before max_dt was inherited there.

    Stage 4 saved data_config correctly all along; nothing was reading it back.
    """
    from training.checkpoint_components import (
        load_joint_refinement_checkpoint, load_lds_component,
    )
    for loader in (load_lds_component, load_joint_refinement_checkpoint):
        src = source_without_comments(loader)
        assert '"data_config"' in src, (
            f"{loader.__name__} drops data_config -- the stage that consumes its output "
            f"cannot then reproduce the window population f_theta was trained on"
        )


def test_both_halves_of_the_chain_use_the_same_key():
    """
    GUARDS the two loaders diverging on where they put it. train_refinement
    reads exactly one path -- components["lds"].provenance["data_config"] --
    so both loaders must populate that same place.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert 'provenance.get("data_config")' in src
    for name in ("load_lds_component", "load_joint_refinement_checkpoint"):
        loader_src = source_without_comments(_ROOT / "training/checkpoint_components.py")
        block = loader_src[loader_src.index(f"def {name}"):]
        block = block[:block.index("\ndef ") if "\ndef " in block[1:] else len(block)]
        assert "provenance" in block and '"data_config"' in block, name


def test_non_inherited_population_filters_are_at_least_reported():
    """
    min_step/min_stdev_phi are params-supplied at every stage, never
    inherited -- deliberately, since unlike max_dt a mismatch only shifts
    which frames are eligible rather than how far f_theta is extrapolated.
    But a silent difference still means the encoder is refined against a
    different population than f_theta saw.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert 'lds_data_config.get(_field, "<absent>")' in src
    assert "min_step" in src and "min_stdev_phi" in src


# --------------------------------------------------------------------
# the scale check must be TWO-SIDED
# --------------------------------------------------------------------

def test_the_check_catches_a_STARVED_component_too():
    """
    The original check fired only on a component DOMINATING (>99%), and stage 4
    then hit the opposite end just as hard: rollout_scale=100 against a
    converged raw of 0.04 left L_rollout at 0.46% of the validation loss, and
    the warning stayed silent while the term stage 4 exists to balance had
    effectively left the objective.

    Both are the same defect -- a scale that is not the raw magnitude of its
    own component -- so both must report.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "sh > 0.99" in src, "the dominance branch is gone"
    assert "sh < 0.01" in src, "no starvation branch"
    assert "effectively OUT of the objective" in src


def test_starvation_is_keyed_on_a_NONZERO_weight():
    """
    GUARDS warning about a term the user deliberately switched off. Stage 5
    runs recon0_weight=1 with rollout_weight small; a weight of exactly 0 is a
    choice, not a mis-scaled component, and reporting it would train the
    reader to ignore the warning.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    starved = src[src.index("starved = ["):]
    starved = starved[:starved.index("\n\n")]
    assert "weights.get(k, 0.0) != 0.0" in starved


def test_the_report_suggests_the_scale_to_USE():
    """
    A percentage says something is wrong; the raw magnitude says what to set.
    Both ends of this took manual arithmetic to convert into a params change,
    twice -- the second time after the first had already been diagnosed.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "_scale~" in src, "the suggestion is not computed at all"
    # Scoped to the PRINTED message, not the file: an earlier version matched
    # the _suggest DEFINITION, so deleting it from the print left the test
    # green -- verified.
    block = src[src.index("if dominant:"):src.index("if saved_this_epoch")]
    assert block.count("Suggested: {_suggest}") == 2, (
        "both branches must print the suggested scales, not just compute them"
    )


def test_dominance_takes_priority_over_starvation():
    """
    With one component at 99.9%, the others are necessarily under 1% -- both
    branches would fire and say contradictory things about the same imbalance.
    The dominant reading is the useful one, so it wins.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert "if dominant:" in src and "elif starved:" in src, (
        "the branches must be exclusive, dominance first"
    )


def test_n_rollout_steps_is_inherited_from_f_theta():
    """
    The regime f_theta was tuned in, not a free choice -- same class as max_dt
    and z1_resync, which are already inherited.

    Found by running the WHOLE chain end to end: 3b recorded
    n_rollout_steps=2, stage 4 silently used 1 (the old signature default), and
    at one step z1_resync has nothing to propagate, so the inherited
    z1_resync=False went inert too. The encoder was refined in a regime f_theta
    was never tuned for, with no message at all. No unit test could see this:
    every one passes n_rollout_steps explicitly.
    """
    import inspect

    from training.train_refinement import train_refinement
    default = inspect.signature(train_refinement).parameters["n_rollout_steps"].default
    assert default is None, (
        "1 as the default silently overrides the ancestor; None means 'unspecified'"
    )
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert '_inherited_rollout = lds_data_config.get("n_rollout_steps")' in src
    # Contiguous substring only: the message is split across two f-string
    # lines, so the full phrase never appears in the source as written.
    assert "inherited from f_theta's own training \"" in src


def test_an_explicit_n_rollout_steps_still_overrides_and_is_flagged():
    """
    GUARDS turning inheritance into a lock. 1 is a meaningful explicit value
    and must stay settable -- but a mismatch with f_theta's own regime should
    say so rather than pass silently.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    block = src[src.index("_inherited_rollout ="):]
    block = block[:block.index("window_length")]
    assert "elif _inherited_rollout is not None and n_rollout_steps != _inherited_rollout:" in block
    assert "but f_theta was" in block


def test_window_length_is_computed_AFTER_the_resolution():
    """
    GUARDS the ordering. window_length = n_rollout_steps + 1 sat ~35 lines
    before the resolution, so the None sentinel was an immediate TypeError --
    caught only because a test actually ran the stage.
    """
    src = source_without_comments(_ROOT / "training/train_refinement.py")
    assert src.index("_inherited_rollout =") < src.index("window_length = n_rollout_steps + 1")
