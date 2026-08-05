"""
A resume that changes NOTHING must reproduce the ancestor's number.

Every stage boundary in this project is a handover: stage 2 extends stage 1's
encoder, stage 3b resumes stage 3a's f_theta, stage 5 unfreezes stage 4's
decoder. Each of those is supposed to CONTINUE from the ancestor, not restart
from something subtly different -- but nothing checked it, and the failures
that shape would produce look exactly like "the next stage is worse", which
this project has repeatedly (and sometimes wrongly) attributed to the physics.

Concretely: stage 3b resumed from a stage-3a checkpoint scored 3.966 on the
1-step metric where 3a itself scored 0.670 -- a 5.9x jump before any training.
That WAS real (n_substeps 1 -> 30 changes what f_theta means), but it took a
day of measurement to establish, because there was no baseline saying "with
everything held equal, the number is preserved".

The idiom throughout: run the successor with epochs=0, which evaluates the
resumed model without training it, and require the reported val_loss to match
the ancestor's. Any mismatch is a handover bug -- a differently-built model, a
differently-composed loss, or a differently-drawn split.
"""
import pathlib
import re
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _val_of(checkpoint_path):
    return float(torch.load(checkpoint_path, map_location="cpu",
                            weights_only=True)["val_loss"])


def test_stage3b_resuming_3a_with_3a_PARAMS_reproduces_3a(tmp_path, isolated_project_root):
    """
    3b IS 3a when n_rollout_steps=1 and n_substeps/z1_resync match. Resuming
    with those settings and epochs=0 must give back 3a's own val_loss.

    A mismatch here would mean the 3a->3b handover changes the model or the
    metric even before the rollout is switched on -- and every 3b run in this
    project would then be measured against a moved goalpost.
    """
    from test_train_lds import _cached_stage2_ancestor
    from training.train_lds import train_lds

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path, stats0_weight=0.01)
    common = dict(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path,
        ae_stats_weight=0.01, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, min_step=0,
        min_stdev_phi=None, encode_batch_size=4, ema_warmup_epochs=0,
        device="cpu", seed=0, log_every_epoch=False,
        n_rollout_steps=1, n_substeps=1, z1_resync=True,
    )
    stage3a = train_lds(epochs=2, checkpoint_path=tmp_path / "s3a.pt",
                         loss_curve_path=tmp_path / "a.png", **common)
    ancestor_val = _val_of(stage3a)

    # The resumed run CANNOT save: the reference ceiling seeds best_val_loss
    # from the ancestor, and an identity resume ties rather than beats it. That
    # refusal is itself part of the contract -- but it means the number has to
    # be read from the epoch-0 row, not from a checkpoint that was never
    # written.
    with pytest.raises(RuntimeError) as exc:
        train_lds(epochs=0, resume_from=stage3a,
                   checkpoint_path=tmp_path / "s3b.pt",
                   loss_curve_path=tmp_path / "b.png", **common)
    assert "without ever saving a checkpoint" in str(exc.value), (
        f"expected the identity resume to tie and refuse to save:\n{exc.value}"
    )
    m = re.search(r"best ([0-9.eE+-]+)", str(exc.value))
    assert m, f"could not read the achieved val_loss from:\n{exc.value}"
    resumed_val = float(m.group(1))
    assert resumed_val == pytest.approx(ancestor_val, rel=1e-5), (
        f"an identity resume moved the number: {ancestor_val} -> {resumed_val}"
    )


def test_stage3b_resume_DOES_move_when_n_substeps_changes(tmp_path, isolated_project_root):
    """
    The negative twin, and the reason the test above is not vacuous: changing
    n_substeps SHOULD move the number, because f_theta trained at
    n_substeps=1 is a dt-AVERAGED corrector and is then applied pointwise.

    A test that passed for any settings at all would pass the first test and
    fail this one.
    """
    from test_train_lds import _cached_stage2_ancestor
    from training.train_lds import train_lds

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path, stats0_weight=0.01)
    common = dict(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path,
        ae_stats_weight=0.01, batch_size=4, hidden_dim=8, n_hidden_layers=1,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, min_step=0,
        min_stdev_phi=None, encode_batch_size=4, ema_warmup_epochs=0,
        device="cpu", seed=0, log_every_epoch=False,
        n_rollout_steps=1, z1_resync=True,
    )
    stage3a = train_lds(epochs=2, n_substeps=1, checkpoint_path=tmp_path / "s3a.pt",
                         loss_curve_path=tmp_path / "a.png", **common)
    ancestor_val = _val_of(stage3a)
    try:
        resumed = train_lds(epochs=0, n_substeps=4, resume_from=stage3a,
                             checkpoint_path=tmp_path / "s3b.pt",
                             loss_curve_path=tmp_path / "b.png", **common)
        resumed_val = _val_of(resumed)
    except RuntimeError as exc:            # tied or worse -> refused to save
        m = re.search(r"best ([0-9.eE+-]+)", str(exc))
        assert m, str(exc)
        resumed_val = float(m.group(1))
    assert resumed_val != pytest.approx(ancestor_val, rel=1e-3), (
        "changing n_substeps left the metric untouched -- then sub-stepping is "
        "not doing anything, and the whole 3b design is inert"
    )


def test_stage5_resuming_stage4_with_STAGE4_params_reproduces_stage4(tmp_path,
                                                                      isolated_project_root):
    """
    Stage 5 is stage 4 with the decoder unfrozen. Held otherwise equal --
    same weights, same scales, freeze_decoder unchanged -- resuming with
    epochs=0 must give back stage 4's own val_loss.

    This boundary has already produced a real regression: a stage 5 that
    inherited stage 4's scales unchanged weighted rollout at 0.007% and
    DEGRADED the dynamics 38x. That was a scale bug, not a handover bug, but
    nothing distinguished the two at the time.
    """
    from test_train_refinement import (
        _build_ae_checkpoint, _build_lds_checkpoint, _build_sweep,
    )
    from training.train_refinement import train_refinement

    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_path, lds_path = tmp_path / "ae.pt", tmp_path / "lds.pt"
    _build_ae_checkpoint(ae_path, include_stats_head=True)
    _build_lds_checkpoint(lds_path)

    common = dict(
        base_path=base_path, freeze_decoder=True,
        rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.1,
        batch_size=4, n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        val_fraction=0.3, test_fraction=0.0, device="cpu", log_every_epoch=False,
    )
    stage4 = train_refinement(
        ae_checkpoint_path=ae_path, lds_checkpoint_path=lds_path, epochs=2,
        checkpoint_path=tmp_path / "s4.pt", **common)
    ancestor_val = _val_of(stage4)

    stage5 = train_refinement(
        resume_from=stage4, epochs=0, checkpoint_path=tmp_path / "s5.pt", **common)
    assert _val_of(stage5) == pytest.approx(ancestor_val, rel=1e-5), (
        f"an identity resume moved the number: {ancestor_val} -> {_val_of(stage5)}"
    )


def test_stage5_resume_DOES_move_when_the_SCALES_change(tmp_path, isolated_project_root):
    """
    The negative twin. Changing rollout_scale must move the reported loss --
    otherwise the scales are inert and the whole stage-4/5 balancing exercise
    was measuring nothing.
    """
    from test_train_refinement import (
        _build_ae_checkpoint, _build_lds_checkpoint, _build_sweep,
    )
    from training.train_refinement import train_refinement

    base_path = _build_sweep(tmp_path, n_runs=6)
    ae_path, lds_path = tmp_path / "ae.pt", tmp_path / "lds.pt"
    _build_ae_checkpoint(ae_path, include_stats_head=True)
    _build_lds_checkpoint(lds_path)

    common = dict(
        base_path=base_path, freeze_decoder=True,
        rollout_weight=1.0, recon0_weight=0.1, stats0_weight=0.1,
        batch_size=4, n_rollout_steps=1, min_step=0, min_stdev_phi=None,
        val_fraction=0.3, test_fraction=0.0, device="cpu", log_every_epoch=False,
    )
    stage4 = train_refinement(
        ae_checkpoint_path=ae_path, lds_checkpoint_path=lds_path, epochs=2,
        checkpoint_path=tmp_path / "s4.pt", rollout_scale=1.0, **common)
    moved = train_refinement(
        resume_from=stage4, epochs=0, checkpoint_path=tmp_path / "s5.pt",
        rollout_scale=0.1, **common)
    assert _val_of(moved) != pytest.approx(_val_of(stage4), rel=1e-3), (
        "a 10x change in rollout_scale left the reported loss untouched"
    )


def test_stage2_resuming_stage1_PRESERVES_the_trained_trunk(tmp_path, isolated_project_root):
    """
    Stage 1 and stage 2 val_loss are NOT comparable, even at deriv_weight=0:
    stage 1 evaluates single FRAMES, stage 2 evaluates WINDOWS (a centred
    deriv target needs three consecutive kept steps), so the two populations
    differ by construction. Measured: 53.13 vs 59.88, a 13% gap that is
    entirely population, not model.

    So the identity worth pinning is on the WEIGHTS, not the metric:
    extend_encoder must add the deriv head WITHOUT perturbing the trunk it
    inherited. If it did, stage 2 would be starting from a partly-random
    model while appearing to continue from a trained one -- and the val
    numbers, being incomparable anyway, would hide it.
    """
    from conftest import cached_stage1_ancestor
    from test_train_lds import _build_sweep
    from training.train_stage2 import train_stage2

    stage1_cfg = dict(
        size=32, epochs=1, batch_size=4, base_channels=4, latent_channels=4,
        val_fraction=0.34, test_fraction=0.17, num_workers=0, min_step=0,
        min_stdev_phi=None, stats0_weight=1.0, stat_names=["avg_phi"],
        recon0_scale=1e-4, stats0_scale=1e-2,
        device="cpu", seed=0, log_every_epoch=False,
    )
    base_path, stage1 = cached_stage1_ancestor(
        tmp_path, lambda d: _build_sweep(d, n_runs=6, size=32), **stage1_cfg)

    try:
        out = train_stage2(
            size=32, base_path=base_path, resume_from=stage1, epochs=0,
            batch_size=4, deriv_weight=0.0, stats0_weight=1.0,
            recon0_scale=1e-4, stats0_scale=1e-2,
            val_fraction=0.34, test_fraction=0.17,
            num_workers=0, min_step=0, min_stdev_phi=None,
            checkpoint_path=tmp_path / "s2.pt", device="cpu", seed=0,
            log_every_epoch=False, loss_curve_path=tmp_path / "c.png",
        )
    except RuntimeError:
        pytest.skip("stage 2 refused to save at epochs=0; nothing to compare")

    before = torch.load(stage1, map_location="cpu", weights_only=True)["model_state"]
    after = torch.load(out, map_location="cpu", weights_only=True)["model_state"]

    # Stage 2 RESTRUCTURES the naming for its multi-stream model, so a raw key
    # intersection is empty by design (71 keys vs 153, 0 shared). The mapping
    # is part of the contract: a rename on either side would silently break
    # the inheritance while every other test still passed.
    def _mapped(k):
        if k.startswith("encoder."):
            return "encoders.shared." + k[len("encoder."):]
        if k.startswith("decoder."):
            return "decoders.D0." + k[len("decoder."):]
        if k == "log_output_scale":
            return "pathways.state.log_output_scale"
        return None

    shared = [(k, _mapped(k)) for k in before if _mapped(k) in after]
    assert len(shared) >= 0.8 * len(before), (
        f"only {len(shared)} of {len(before)} stage-1 tensors map into stage 2 -- "
        f"the naming convention has changed on one side"
    )
    moved = [k for k, mk in shared
              if before[k].shape == after[mk].shape
              and not torch.equal(before[k].float(), after[mk].float())]
    assert not moved, (
        f"{len(moved)} of {len(shared)} inherited tensors changed during an "
        f"epochs=0 resume, e.g. {moved[:3]} -- extend_encoder is perturbing the "
        f"trained trunk, not just adding a head"
    )
