"""Tests for utils/plot_helpers.py -- the shared home of helpers that
had drifted into diverged per-file copies (see the module docstring)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.plot_helpers import fmt_corr, log_scale_if_positive


def test_explicit_arrays_mode_applies_log_when_positive():
    fig, ax = plt.subplots()
    assert log_scale_if_positive(ax, np.array([0.1, 2.0])) is True
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_explicit_arrays_mode_falls_back_on_all_zero():
    fig, ax = plt.subplots()
    assert log_scale_if_positive(ax, np.zeros(5)) is False
    assert ax.get_yscale() == "linear"
    plt.close(fig)


def test_introspection_mode_reads_lines_and_scatter():
    # no arrays passed -> decides from the axes' own artists
    fig, ax = plt.subplots()
    ax.plot([1, 2], [0.5, 3.0])
    assert log_scale_if_positive(ax) is True
    assert ax.get_yscale() == "log"
    plt.close(fig)

    # scatter lands in ax.collections, not ax.get_lines() -- the introspection
    # must look there too. POSITIVE scatter data so the assertion can only pass
    # if collections were actually inspected (a skipped-collections bug would
    # find no data and return False).
    fig, ax = plt.subplots()
    ax.scatter([1, 2], [0.5, 3.0])
    assert log_scale_if_positive(ax) is True
    assert ax.get_yscale() == "log"
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter([1, 2], [-1.0, 0.0])       # nothing positive anywhere
    assert log_scale_if_positive(ax) is False
    plt.close(fig)


def test_x_axis_selection_via_keyword():
    fig, ax = plt.subplots()
    assert log_scale_if_positive(ax, np.array([3.0]), axis="x") is True
    assert ax.get_xscale() == "log" and ax.get_yscale() == "linear"
    plt.close(fig)


def test_inf_only_data_does_not_log_scale():
    """The reconciliation's ONE deliberate behavior change: the explicit-array
    path now requires FINITE positives (the introspection copy always did).
    +inf compares > 0 but cannot anchor a log scale any more than zero can --
    the old check_f_theta copy would have tried and produced a broken axis."""
    fig, ax = plt.subplots()
    assert log_scale_if_positive(ax, np.array([np.inf, np.nan, -1.0])) is False
    assert ax.get_yscale() == "linear"
    plt.close(fig)


def test_annotate_writes_the_linear_note_only_on_fallback():
    fig, ax = plt.subplots()
    log_scale_if_positive(ax, np.zeros(3), annotate=True)
    texts = [t.get_text() for t in ax.texts]
    assert any("LINEAR" in t for t in texts), texts
    plt.close(fig)

    fig, ax = plt.subplots()
    log_scale_if_positive(ax, np.array([1.0]), annotate=True)
    assert not ax.texts          # success -> no note
    plt.close(fig)


def test_fmt_corr_raw_contract():
    assert fmt_corr(None) == "N/A (zero variance)"
    assert fmt_corr(0.123) == "12.3%"
    # raw [-1,1] input contract: 1.0 -> 100.0%, NOT 1.0%
    assert fmt_corr(1.0) == "100.0%"
