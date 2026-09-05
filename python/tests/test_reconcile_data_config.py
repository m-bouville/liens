"""compare_f_theta must borrow a data_config filtering field that one checkpoint
recorded and another (older lineage) didn't -- they were trained on the same
filter, one just predates the field being saved. Warn (don't override) on genuine
disagreement; leave absent (so the usual error fires) when nobody has it."""
import io
import contextlib
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "cft", pathlib.Path(__file__).resolve().parent / "evaluation" / "compare_f_theta.py")


def _reconcile():
    # import just the function without executing argparse/main
    import evaluation.compare_f_theta as m
    return m._reconcile_data_config


def _mk(label, **dc):
    return {"label": label, "ck": {"data_config": dict(dc)}}


def test_missing_field_is_borrowed_from_the_sibling_with_a_note():
    reconcile = _reconcile()
    a = _mk("old", min_passing_steps=12)                          # missing the field
    b = _mk("new", min_passing_steps=12, min_normalized_stdev_phi=0.02)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reconcile([a, b])
    assert a["ck"]["data_config"]["min_normalized_stdev_phi"] == 0.02
    out = buf.getvalue()
    assert "NOTE" in out and "min_normalized_stdev_phi" in out and "old" in out


def test_disagreement_warns_and_does_not_override():
    reconcile = _reconcile()
    a = _mk("r1", min_normalized_stdev_phi=0.02)
    b = _mk("r2", min_normalized_stdev_phi=0.05)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reconcile([a, b])
    assert a["ck"]["data_config"]["min_normalized_stdev_phi"] == 0.02   # unchanged
    assert b["ck"]["data_config"]["min_normalized_stdev_phi"] == 0.05   # unchanged
    assert "WARNING" in buf.getvalue() and "disagree" in buf.getvalue()


def test_absent_from_all_is_left_absent():
    reconcile = _reconcile()
    a = _mk("r1", min_passing_steps=12)
    b = _mk("r2", min_passing_steps=12)
    with contextlib.redirect_stdout(io.StringIO()):
        reconcile([a, b])
    # not filled -> build_good_steps' own 'no threshold' error still fires downstream
    assert "min_normalized_stdev_phi" not in a["ck"]["data_config"]
    assert "min_normalized_stdev_phi" not in b["ck"]["data_config"]
