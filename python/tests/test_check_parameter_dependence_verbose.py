"""The --verbose gate: explanatory prose goes through _vprint and appears only
when _VERBOSE is set; data lines (via plain print) always appear. Default is
quiet so the console output fits a paste-able size."""
import io
import contextlib

import evaluation.check_parameter_dependence as m


def test_vprint_silent_by_default():
    original = m._VERBOSE
    try:
        m._VERBOSE = False
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m._vprint("explanatory prose")
        assert buf.getvalue() == ""
    finally:
        m._VERBOSE = original


def test_vprint_prints_when_verbose():
    original = m._VERBOSE
    try:
        m._VERBOSE = True
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m._vprint("explanatory prose")
        assert "explanatory prose" in buf.getvalue()
    finally:
        m._VERBOSE = original


def test_module_default_is_quiet():
    """A fresh import must default to quiet -- the whole point is that a normal
    run is paste-able without passing any flag."""
    import importlib
    fresh = importlib.reload(m)
    assert fresh._VERBOSE is False
