"""
Every trainer parameter must reach the log.

The bug this exists to prevent is not hypothetical: audited across the four
trainers, 24 of train_lds()'s 46 parameters were never printed anywhere, and
the three that mattered most were exactly the ones that change what f_theta
MEANS -- z1_resync, dt_cap, n_substeps. All three are written into the saved
checkpoint config, all three are warned about when they change on resume, and
none of them appeared in any log, so a stage-3b log could not say which
objective produced it. seed was missing from three of the four trainers,
making those runs unreproducible from their own logs.

The cause was structural, not carelessness: the preamble was a hand-maintained
list of print() calls, so the DEFAULT for a newly added parameter was silence.
print_run_parameters inverts that -- a parameter is printed unless deliberately
excluded -- and these tests enforce the inversion by enumerating every site.
"""
import inspect

import pytest

from utils.logging_utils import _PLUMBING, print_run_parameters
from training.train_lds import train_lds, _LDS_PREAMBLE_PARAMS
from training.train_refinement import train_refinement, _REFINEMENT_PREAMBLE_PARAMS
from training.train_stage1 import train_autoencoder, _STAGE1_PREAMBLE_PARAMS
from training.train_stage2 import train_stage2, _STAGE2_PREAMBLE_PARAMS

# THE SITE LIST. A fifth trainer joins here; every test below is parametrized
# over it, so adding the entry is all that is needed to get the same coverage.
TRAINERS = [
    pytest.param(train_lds, _LDS_PREAMBLE_PARAMS, id="train_lds"),
    pytest.param(train_refinement, _REFINEMENT_PREAMBLE_PARAMS, id="train_refinement"),
    pytest.param(train_autoencoder, _STAGE1_PREAMBLE_PARAMS, id="train_autoencoder"),
    pytest.param(train_stage2, _STAGE2_PREAMBLE_PARAMS, id="train_stage2"),
]


def _defaults(func):
    return {n: (p.default if p.default is not inspect.Parameter.empty else "<required>")
            for n, p in inspect.signature(func).parameters.items()}


@pytest.mark.parametrize("func,preamble", TRAINERS)
def test_every_parameter_is_logged_or_deliberately_excluded(func, preamble, capsys):
    """
    The core guarantee. A parameter is in the log, in the trainer's own
    preamble list, or in _PLUMBING -- there is no fourth option, so a new
    parameter cannot be silently unrecorded.
    """
    values = _defaults(func)
    print_run_parameters(func, values, preamble)
    printed = capsys.readouterr().out

    for name in inspect.signature(func).parameters:
        accounted = (name in _PLUMBING or name in preamble
                     or f"{name}=" in printed)
        assert accounted, (
            f"{func.__name__}({name}=...) is printed nowhere and is not declared in "
            f"either exclusion list -- it would not appear in the run's log, so the "
            f"log could not say what was actually run. Add it to the preamble prints, "
            f"or to that trainer's preamble list if it IS printed there, or to "
            f"_PLUMBING if it genuinely cannot change the result."
        )


@pytest.mark.parametrize("func,preamble", TRAINERS)
def test_the_meaning_changing_parameters_are_never_plumbing(func, preamble):
    """
    _PLUMBING is defined as "cannot change what the run means". Nothing that
    changes the objective may drift into it -- which is how these three went
    missing in the first place, and the reason a 3b diagnosis had to guess at
    z1_resync from a default rather than read it.
    """
    meaning_changing = {"z1_resync", "dt_cap", "n_substeps", "seed", "lr",
                        "hidden_dim", "n_hidden_layers", "rollout_scale",
                        "latent_channels", "min_step", "condition_on_theta"}
    for name in inspect.signature(func).parameters:
        if name in meaning_changing:
            assert name not in _PLUMBING, (
                f"{name} changes what the run means and must never be treated as "
                f"plumbing -- it has to be recoverable from the log."
            )


@pytest.mark.parametrize("func,preamble", TRAINERS)
def test_preamble_list_entries_are_real_parameters(func, preamble):
    """
    A stale name in the preamble list silently un-prints the parameter it was
    renamed from: the helper skips it as "already printed", and nothing prints
    it. Raising is the only safe behaviour, since the alternative is a log that
    looks complete and is not.
    """
    params = set(inspect.signature(func).parameters)
    stale = sorted(set(preamble) - params)
    assert not stale, (
        f"{func.__name__}'s preamble list names {stale}, which are not parameters "
        f"of it -- probably renamed. Until the list is updated they are excluded "
        f"from the block AND absent from the preamble, i.e. missing entirely."
    )


@pytest.mark.parametrize("func,preamble", TRAINERS)
def test_preamble_list_entries_are_actually_printed_in_the_preamble(func, preamble):
    """
    The list is a CLAIM ("the preamble already prints this"), and an unchecked
    claim is how the original silence happened. Adding a name to it excludes
    the parameter from the block, so if the claim is false the parameter is
    printed nowhere -- and every other test here still passes, since they all
    treat membership in the list as sufficient. Verified: adding hidden_dim and
    seed to train_lds's list without printing them kept the suite green.

    Checked structurally rather than by string match: the parameter must be
    referenced inside a print() call occurring BEFORE the epoch loop, which is
    what "in the preamble" means. Where the preamble prints it is free to
    change; that it prints it is not.
    """
    import ast
    source, first_line = inspect.getsourcelines(func)
    tree = ast.parse("".join(source).lstrip())
    fn = tree.body[0]
    loop_line = min((n.lineno for n in ast.walk(fn)
                     if isinstance(n, ast.For) and "epoch" in ast.unparse(n.target)),
                    default=10 ** 9)
    preamble_prints = "\n".join(
        ast.unparse(n) for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "print" and n.lineno < loop_line
    )
    for name in preamble:
        assert name in preamble_prints, (
            f"{func.__name__}'s preamble list claims {name} is printed before the "
            f"epoch loop, but no print() there references it -- so it is excluded "
            f"from the parameter block AND absent from the preamble, i.e. missing "
            f"from the log entirely."
        )


@pytest.mark.parametrize("func,preamble", TRAINERS)
def test_each_trainer_actually_calls_the_helper(func, preamble):
    """
    Every other test here calls print_run_parameters ITSELF, so they all pass
    against a trainer that never calls it -- verified by deleting the call from
    train_refinement, which changed nothing. The call site is the thing that
    puts the parameters in a real log, so it needs its own check.

    Also pinned: the call passes the trainer's OWN preamble list and happens
    before the epoch loop. Passing another trainer's list would silence a
    near-arbitrary subset, and calling it after the loop would put the
    parameters at the END of the log -- both technically "calling the helper".
    """
    import ast
    source, _ = inspect.getsourcelines(func)
    tree = ast.parse("".join(source).lstrip())
    fn = tree.body[0]
    loop_line = min((n.lineno for n in ast.walk(fn)
                     if isinstance(n, ast.For) and "epoch" in ast.unparse(n.target)),
                    default=10 ** 9)
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "print_run_parameters"]
    assert calls, (
        f"{func.__name__} never calls print_run_parameters -- its parameters reach "
        f"no log, however complete its exclusion lists look."
    )
    assert len(calls) == 1, f"{func.__name__} calls print_run_parameters {len(calls)} times"
    call = calls[0]
    assert call.lineno < loop_line, (
        f"{func.__name__} calls print_run_parameters after the epoch loop -- the "
        f"parameters would land at the end of the log, after the run they describe."
    )
    rendered = ast.unparse(call)
    assert func.__name__ in rendered, (
        f"{func.__name__}'s call does not pass its own function object, so the "
        f"signature introspected is some other trainer's."
    )
    assert "locals()" in rendered, (
        f"{func.__name__}'s call must pass locals() -- passing a hand-built dict "
        f"reintroduces exactly the hand-maintained list this replaces."
    )


def test_a_stale_preamble_name_raises_rather_than_dropping_it():
    """The helper's own guard, exercised directly."""
    with pytest.raises(ValueError, match="not parameters of"):
        print_run_parameters(train_lds, _defaults(train_lds), ("lr", "a_renamed_parameter"))


def test_values_are_printed_unambiguously(capsys):
    """
    None, an empty string and the string "None" must be distinguishable in a
    log -- otherwise a reader cannot tell "not set" from "set to the word
    None", and the log stops being evidence.
    """
    def f(a=None, b="None", c="", d=1.0):
        pass

    print_run_parameters(f, {"a": None, "b": "None", "c": "", "d": 1.0}, ())
    out = capsys.readouterr().out
    assert "a=None" in out
    assert "b='None'" in out
    assert "c=''" in out
    assert "d=1.0" in out


def test_nothing_is_printed_when_everything_is_already_covered(capsys):
    """No empty 'other parameters:' header on a trainer whose preamble covers
    everything -- a header with nothing under it reads like a bug."""
    def f(a=1, device=None):
        pass

    assert print_run_parameters(f, {"a": 1, "device": None}, ("a",)) == []
    assert capsys.readouterr().out == ""


def test_log_is_readable_before_the_run_ends(tmp_path):
    """
    REGRESSION: _Tee wrote to the log file without flushing, so output sat
    in Python's ~8 KiB buffer until close. A killed run -- Ctrl-C, an IDE
    stop button, an OOM kill -- discarded the whole log, which is exactly
    when it is most wanted: a run that ends normally could have been rerun,
    a run that was killed after 16 hours cannot.

    Also makes the log tail-able while training is in progress.
    """
    from utils.logging_utils import _log_to_file

    log_path = tmp_path / "run.log"
    with _log_to_file(log_path):
        print("epoch 1 | loss 0.5")
        mid_run = log_path.read_text()
    assert "epoch 1 | loss 0.5" in mid_run, (
        "the log was empty while the run was still going -- output is "
        "buffered, so a killed run loses everything"
    )


def test_log_survives_an_interrupt(tmp_path):
    """The realistic failure: KeyboardInterrupt unwinds through the
    contextmanager's finally, which must leave a complete log behind."""
    from utils.logging_utils import _log_to_file

    log_path = tmp_path / "run.log"
    try:
        with _log_to_file(log_path):
            print("epoch 1 | loss 0.5")
            print("epoch 2 | loss 0.4")
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    written = log_path.read_text()
    assert "epoch 1" in written and "epoch 2" in written, written
