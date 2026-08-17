"""
Shared plotting helpers for the evaluation scripts.

Created to reconcile helpers that had drifted into DIVERGED per-file
copies (same name, same intent, different behavior -- so a fix to one
silently missed the other; that actually happened with
_log_scale_if_positive during the session-10 log-axis work). Every
helper here is the single source of truth; the per-file copies are gone.
"""
from __future__ import annotations

import numpy as np


def log_scale_if_positive(ax, *value_arrays, axis: str = "y",
                          annotate: bool = False) -> bool:
    """set_{x,y}scale("log") only when there is positive, finite data.

    matplotlib raises "Data has no positive values, and therefore cannot
    be log-scaled" -- from inside tight_layout()/draw, not from
    set_yscale() itself -- so an all-zero or all-negative panel kills the
    whole figure at save time, with a traceback pointing at the layout
    engine rather than at the offending panel.

    Degenerate populations reach here legitimately: a tiny run set, a
    single-dt slice, or a diagnostic on an untrained f_theta.
    LatentDynamics zero-initializes its final layer, so f_theta is
    EXACTLY zero everywhere until trained -- which is precisely the state
    ensure_lds_checkpoint produces when an eval script is pointed at an
    AE-family (stage-1/1b/2) checkpoint. Every ||f||-derived quantity is
    then identically 0. Losing one panel's log scaling beats losing the
    figure.

    Two ways to decide, matching the two historical call styles:
    - value_arrays given: decide from those arrays (the caller knows
      exactly what will be plotted; robust when data is drawn AFTER the
      scale is set).
    - no value_arrays: introspect the axes' own already-plotted artists
      (lines and scatter collections; robust when the caller doesn't
      have the arrays at hand).

    Positivity requires np.isfinite too: +inf compares > 0 but cannot
    anchor a log scale any more than zero can.

    annotate=True writes a small red note on the panel when falling back,
    so a reader is not left assuming an axis is log-scaled when it isn't.

    Returns whether log scale was applied, so a caller can say so.
    """
    if value_arrays:
        data = [np.asarray(v, dtype=float).ravel() for v in value_arrays]
    else:
        data = []
        for ln in ax.get_lines():
            arr = ln.get_ydata() if axis == "y" else ln.get_xdata()
            data.append(np.asarray(arr, dtype=float))
        for coll in ax.collections:        # scatter() lands here, not in lines
            off = coll.get_offsets()
            if off is not None and len(off):
                off = np.asarray(off, dtype=float)
                data.append(off[:, 1] if axis == "y" else off[:, 0])

    allv = np.concatenate(data) if len(data) > 1 else (data[0] if data else np.array([]))
    if allv.size and np.any(np.isfinite(allv) & (allv > 0)):
        (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
        return True
    if annotate:
        ax.text(0.5, 0.5, f"all-zero data\n({axis}-axis left LINEAR, not log)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="tab:red", alpha=0.7)
    return False


def fmt_corr(c: float | None) -> str:
    """Format a RAW correlation (in [-1, 1]) for a printed report line.
    None -> 'N/A (zero variance)': correlation is undefined when either
    series has zero variance (e.g. a quiet window's real dx), and the
    explicit message beats a bare 'nan%' that reads like a format bug.

    NOTE ON THE NAME-COLLISION THIS RESOLVES: compare_f_theta.py had a
    same-named local helper with a DIFFERENT input contract -- values
    already in percent (its module-wide _correlation_pct convention).
    That one is renamed _fmt_corr_pct in place, not merged here: two
    contracts must not share a name, but forcing one contract on the
    other would touch compare_f_theta's whole data flow for zero gain.
    """
    return "N/A (zero variance)" if c is None else f"{c * 100:.1f}%"
