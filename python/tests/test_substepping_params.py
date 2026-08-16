"""
Step 1 of stage-3b sub-stepping: the parameters, before any integrator exists.

`n_substeps=1, z1_resync=True` must reproduce today's behaviour exactly -- and
that pair is also precisely stage 3a's configuration, so "the default is a
no-op" and "3a is unchanged" are the same claim, tested once here.

The checkpoint round-trip and resume cross-check matter more than they look.
n_substeps changes what f_theta was trained to BE: at 1 its exact target is
[z0(t+dt) - z0 - z1*dt]*2/dt^2, the dt-AVERAGED curvature over the interval;
as it grows that converges to the pointwise z1_dot the design doc defines.
Two checkpoints with identical shapes can therefore mean different things, and
load_state_dict cannot tell them apart -- exactly the hazard dt_cap's own
cross-check exists for, which is the pattern copied here.
"""
import inspect

import pytest
import torch

from models.latent_dynamics import LatentDynamics
from training.train_lds import train_lds
from models.constants import N_THETA

LC, LS = 4, 8


def _model(**kwargs):
    return LatentDynamics(latent_channels=LC, latent_spatial=LS, hidden_dim=8,
                           n_hidden_layers=1, **kwargs)


# --------------------------------------------------------------------
# the defaults ARE the current behaviour, and are 3a's configuration
# --------------------------------------------------------------------

def test_defaults_are_todays_behaviour():
    assert _model().n_substeps == 1
    assert inspect.signature(train_lds).parameters["n_substeps"].default == 1
    assert inspect.signature(train_lds).parameters["z1_resync"].default is True


def test_rollout_is_unchanged_at_the_defaults():
    """
    GUARDS the parameter silently altering the default path. Compared against
    an explicit re-implementation of the historical update rather than against
    a golden file, so it stays readable and cannot rot.
    """
    torch.manual_seed(0)
    model = _model().eval()
    B, n = 2, 3
    z0 = torch.randn(B, LC, LS, LS)
    z1_seq = torch.randn(B, n + 1, LC, LS, LS)
    dts = torch.rand(B, n) * 100 + 1
    theta = torch.zeros(B, N_THETA)

    with torch.no_grad():
        got = model.rollout(z0, z1_seq, dts, theta)
        # historical: repeated forward(), z1 teacher-forced at every step
        cur, expected = z0, [z0]
        for i in range(n):
            cur = model.forward(cur, z1_seq[:, i], dts[:, i], theta)
            expected.append(cur)
        expected = torch.stack(expected, dim=1)
    assert torch.equal(got, expected)


# --------------------------------------------------------------------
# n_substeps lives on the MODEL; z1_resync does not
# --------------------------------------------------------------------

def test_n_substeps_is_a_model_property_and_z1_resync_is_not():
    """
    GUARDS putting z1_resync on LatentDynamics. At inference there is no frame
    to resync to, so it is meaningless on the model -- it is a training policy.
    Putting it there would also imply it belongs in the architecture
    cross-check, where a mismatch is not an error.
    """
    assert "n_substeps" in inspect.signature(LatentDynamics.__init__).parameters
    assert "z1_resync" not in inspect.signature(LatentDynamics.__init__).parameters
    assert "z1_resync" in inspect.signature(train_lds).parameters


@pytest.mark.parametrize("bad", [0, -1])
def test_n_substeps_below_one_is_refused(bad):
    with pytest.raises(ValueError, match="n_substeps"):
        _model(n_substeps=bad)


def test_n_substeps_is_stored_verbatim():
    assert _model(n_substeps=4).n_substeps == 4


# --------------------------------------------------------------------
# the cross-check -- the reason any of this is checkpointed
# --------------------------------------------------------------------

def test_a_changed_n_substeps_is_REPORTED_and_not_refused():
    """
    n_substeps must be visible on resume -- weights trained at one value load
    cleanly into a model built at another, same shapes, different meaning --
    but it must NOT be fatal.

    An earlier version put it in the architecture mismatch list beside dt_cap.
    That blocked the one transition the parameter exists to enable: stage 3a is
    single-step BY DESIGN (that is what makes it ground-truth-conditioned) and
    stage 3b sub-steps, so the automatic 3a -> 3b handoff raised
    "n_substeps=1 (checkpoint) vs 3 (requested)" and stopped the pipeline.

    n_rollout_steps has exactly this shape -- it also changes legitimately at
    that boundary -- and is warned about rather than refused. Same treatment.
    """
    from training.train_lds import _resume_f_theta_from_checkpoint as resume
    assert "n_substeps" in inspect.signature(resume).parameters
    src = inspect.getsource(resume)
    assert 'mismatch.append(("n_substeps"' not in src, (
        "n_substeps must not be fatal -- it blocks the 3a -> 3b handoff"
    )
    assert 'prev_config.get("n_substeps", 1)' in src
    assert "NOTE: resuming from a checkpoint trained at n_substeps=" in src


def test_a_pre_n_substeps_checkpoint_does_not_spuriously_mismatch():
    """
    GUARDS a bare prev_config["n_substeps"], which would KeyError, or a default
    other than 1, which would flag every checkpoint written before the
    parameter existed. Same `.get(..., <historical default>)` reasoning dt_cap
    documents for inf.
    """
    import training.train_lds as mod
    src = inspect.getsource(mod)
    assert 'prev_config.get("n_substeps", 1)' in src


def test_both_parameters_are_saved_in_the_checkpoint():
    import training.train_lds as mod
    src = inspect.getsource(mod)
    assert '"n_substeps": n_substeps,' in src
    assert '"z1_resync": z1_resync,' in src


# --------------------------------------------------------------------
# step 3: z1_resync actually reaches rollout()
# --------------------------------------------------------------------

def test_z1_resync_is_forwarded_to_every_rollout_call():
    """
    GUARDS storing the parameter and never using it -- the state the code was
    in after step 1, where z1_resync was saved in the checkpoint but the
    training loop always passed rollout()'s default. The checkpoint would then
    claim a regime the run never trained in.
    """
    import inspect
    import training.train_lds as mod

    src = inspect.getsource(mod)
    # "= f_theta.rollout(", not just the substring: the module's prose
    # mentions f_theta.rollout() too, and matching that made the test fail on
    # a docstring rather than on any real call.
    rollout_calls = [l for l in src.splitlines() if "= f_theta.rollout(" in l]
    assert rollout_calls, "no rollout call found"
    # each call must be followed within a couple of lines by the kwarg
    for line in rollout_calls:
        idx = src.index(line)
        window = src[idx:idx + 220]
        assert "z1_resync=z1_resync" in window, f"not forwarded: {line.strip()}"


def test_the_euler_baseline_uses_the_same_regime_as_training():
    """
    GUARDS measuring the euler-only baseline teacher-forced while training runs
    unforced. The baseline exists to be compared against what training
    produces and is used to SCALE the loss, so a mismatch would silently
    rescale it -- a bug with no error message and no obviously wrong number.
    """
    import inspect
    from training.train_lds import compute_euler_only_losses

    assert "z1_resync" in inspect.signature(compute_euler_only_losses).parameters
    src = inspect.getsource(inspect.getmodule(compute_euler_only_losses))
    call = src[src.index("all_dts, all_losses = compute_euler_only_losses("):][:260]
    assert "z1_resync=z1_resync" in call


def test_cli_exposes_both_knobs():
    import inspect
    import training.train_lds as mod
    src = inspect.getsource(mod)
    assert '"--n-substeps"' in src
    assert '"--z1-resync"' in src
    assert "n_substeps=args.n_substeps" in src
    assert "z1_resync=args.z1_resync" in src


def test_the_3a_to_3b_handoff_is_not_blocked(tmp_path):
    """
    END TO END on the exact failure reported: a checkpoint written at
    n_substeps=1 (stage 3a) resumed by a run asking for n_substeps=3
    (stage 3b). This must load, and must say so.

    Exercises _resume_f_theta_from_checkpoint directly rather than through
    train_lds, so it needs no dataset and runs in milliseconds -- the bug was
    entirely in the config comparison, not in anything downstream of it.
    """
    import io
    import contextlib
    from training.train_lds import _resume_f_theta_from_checkpoint

    ae_config = {"latent_channels": LC, "latent_spatial_size": LS}
    written = tmp_path / "128x128-stage3a.pt"
    torch.save({
        "model_state": _model(n_substeps=1).state_dict(),
        "epoch": 7, "val_loss": 1.25,
        "config": {"hidden_dim": 8, "n_hidden_layers": 1, "dt_cap": float("inf"),
                    "n_substeps": 1, "latent_channels": LC, "latent_spatial_size": LS},
        "data_config": {"n_rollout_steps": 1},
    }, written)

    target = _model(n_substeps=3)          # stage 3b
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _resume_f_theta_from_checkpoint(
            target, written, ae_config, hidden_dim=8, n_hidden_layers=1,
            dt_cap=float("inf"), n_substeps=3, n_rollout_steps=4,
            device=torch.device("cpu"))
    out = buf.getvalue()
    assert "n_substeps=1" in out and "n_substeps=3" in out, out
    assert "NOTE" in out


def test_going_backwards_from_substepped_to_single_step_is_flagged(tmp_path):
    """
    The direction that is NOT the curriculum: a pointwise f_theta used as a
    one-shot corrector. Still allowed -- refusing it would repeat the mistake
    above -- but the note must say it is not equivalent.
    """
    import io
    import contextlib
    from training.train_lds import _resume_f_theta_from_checkpoint

    ae_config = {"latent_channels": LC, "latent_spatial_size": LS}
    written = tmp_path / "sub.pt"
    torch.save({
        "model_state": _model(n_substeps=8).state_dict(),
        "epoch": 3, "val_loss": 0.5,
        "config": {"hidden_dim": 8, "n_hidden_layers": 1, "dt_cap": float("inf"),
                    "n_substeps": 8, "latent_channels": LC, "latent_spatial_size": LS},
        "data_config": {"n_rollout_steps": 4},
    }, written)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _resume_f_theta_from_checkpoint(
            _model(n_substeps=1), written, ae_config, hidden_dim=8, n_hidden_layers=1,
            dt_cap=float("inf"), n_substeps=1, n_rollout_steps=4,
            device=torch.device("cpu"))
    assert "NOT equivalent" in buf.getvalue()


def test_max_substeps_none_is_legal_in_fixed_mode():
    """
    REGRESSION: max_substeps is the clamp on the ALPHA-derived count and is
    read only in the adaptive path. In fixed mode (alpha=None, n_substeps
    drives the count) it is never consulted, so None must be accepted --
    passing max_substeps=None crashed the constructor with
    'TypeError: < not supported between NoneType and int' before any
    substepping ran.
    """
    from models.latent_dynamics import LatentDynamics
    m = LatentDynamics(latent_channels=8, alpha=None, n_substeps=512,
                        max_substeps=None)
    # constructed without error; the fixed-count path does not consult it
    assert m.max_substeps >= 1  # kept as a harmless int, not None


def test_max_substeps_none_is_refused_in_adaptive_mode():
    """When alpha IS set, max_substeps is the clamp that bounds the derived
    count -- None there is a real misconfiguration and must raise, not be
    silently coerced."""
    from models.latent_dynamics import LatentDynamics
    import pytest
    with pytest.raises(ValueError, match="max_substeps"):
        LatentDynamics(latent_channels=8, alpha=0.5, n_substeps=1,
                        max_substeps=None)
