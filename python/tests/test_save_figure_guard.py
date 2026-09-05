"""A figure write must never kill a training run. On Windows a just-written PNG
is intermittently held for a few ms by an antivirus/Defender scan (or a viewer),
so savefig's open(..., 'w+b') can raise OSError [Errno 22] on the next epoch's
rewrite despite a valid path -- which crashed a stage-4 run at epoch 3.
_save_figure retries to ride out the transient lock, then warns-and-skips if it
persists, and always closes the figure."""
import io
import contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plots import _save_figure


def test_transient_lock_is_retried_then_succeeds(tmp_path):
    fig = plt.figure()
    orig, calls = fig.savefig, {"n": 0}
    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError(22, "Invalid argument")   # the Windows scanner window
        return orig(*a, **k)
    fig.savefig = flaky
    assert _save_figure(fig, tmp_path / "a.png", retries=5) is True
    assert (tmp_path / "a.png").exists()
    assert calls["n"] == 3                            # retried, didn't give up early


def test_persistent_lock_warns_and_does_not_raise(tmp_path):
    fig = plt.figure()
    fig.savefig = lambda *a, **k: (_ for _ in ()).throw(OSError(22, "Invalid argument"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = _save_figure(fig, tmp_path / "b.png", retries=2)   # must NOT raise
    assert result is False
    assert "WARNING" in buf.getvalue() and "training continues" in buf.getvalue()


def test_figure_is_closed_even_when_the_save_fails(tmp_path):
    before = set(plt.get_fignums())
    fig = plt.figure()
    fig.savefig = lambda *a, **k: (_ for _ in ()).throw(OSError(22, "x"))
    with contextlib.redirect_stdout(io.StringIO()):
        _save_figure(fig, tmp_path / "c.png", retries=1)
    assert set(plt.get_fignums()) == before, "figure leaked on the failure path"
