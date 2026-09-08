"""Behavioral inspection of the matplotlib figures produced by utils.plots.

WHY THIS EXISTS
The plot functions in utils.plots save through plots._save_figure -- which also
CLOSES the figure -- and return only a Path, so a test cannot inspect the return
value to see what was drawn. Historically that pushed plot tests to grep the
function SOURCE ('"--"' in src, 'loc="best"' in src, '"tab:green"' in src): brittle
(a refactor that moves the code breaks the test with no bug), and shallow (it cannot
catch a colour that is set and then overridden, a duplicate palette entry, or a scale
applied to the wrong axis -- the source says one thing, the artist another).

capture_figure() intercepts _save_figure to grab the Figure BEFORE it is closed, and
FigureArtifacts exposes the actually-drawn artists via matplotlib's STABLE PUBLIC
getters (get_lines/get_color/get_linestyle/get_xscale/...). A test then asserts on
what was rendered, survives refactors of the plot code, and catches the artist-level
bugs a source grep cannot.

CAVEAT, stated honestly: legend_loc_code() reads matplotlib's PRIVATE Legend._loc
(there is no public getter for the requested location). It is the one method here that
trades source-fragility for matplotlib-version-fragility; everything else uses public
API. Prefer the public views; use legend_loc_code() knowingly.
"""

from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")                       # headless; no display needed for introspection
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.colors import to_hex        # noqa: E402

from utils import plots                     # noqa: E402


@contextmanager
def capture_figure():
    """Intercept the next plots._save_figure call(s) and yield a list that fills with
    the rendered Figure(s), WITHOUT closing them (so the test can inspect their
    artists). The real _save_figure is restored and every captured figure closed on
    exit -- so the test neither writes a file nor leaks a figure.

    Usage:
        with capture_figure() as figs:
            plots.loss_component_scatter(..., some_path, ...)
        art = FigureArtifacts(figs[0])
        assert art.line_styles(...)[-1] == "--"
    """
    captured = []
    original = plots._save_figure

    def _capturing(fig, output_path, *args, **kwargs):
        captured.append(fig)
        return True                          # report success; do NOT save or close

    plots._save_figure = _capturing
    try:
        yield captured
    finally:
        plots._save_figure = original
        for fig in captured:
            plt.close(fig)


def render_once(render_callable):
    """Run a zero-arg callable that draws via plots._save_figure, capture the Figure
    WITHOUT closing it or writing a file, and restore _save_figure IMMEDIATELY -- so
    this is safe to call from a module-scoped fixture (it does not leave _save_figure
    patched for the whole module the way holding capture_figure() open would). Returns
    FigureArtifacts; the caller closes the figure via .close() at fixture teardown.
    Use this to render one figure once and share it across several assertions, instead
    of re-rendering (each loss_component_scatter is ~0.7 s)."""
    captured = []
    original = plots._save_figure

    def _cap(fig, *args, **kwargs):
        captured.append(fig)
        return True

    plots._save_figure = _cap
    try:
        render_callable()
    finally:
        plots._save_figure = original
    if not captured:
        raise AssertionError("render_callable did not draw a figure via plots._save_figure")
    return FigureArtifacts(captured[0])


class FigureArtifacts:
    """Read-only view of what a captured Figure actually drew. Methods take an axis
    index (0 = first subplot); the figure may be an n x n grid (fig.axes is flat,
    row-major)."""

    def __init__(self, fig):
        self.fig = fig
        self.axes = fig.axes

    def close(self):
        """Close the underlying figure -- call at teardown when the figure was kept
        open (e.g. by render_once) for cross-test inspection."""
        plt.close(self.fig)

    def n_axes(self):
        return len(self.axes)

    def lines(self, ax=0):
        """The Line2D artists on an axis (plot() calls; scatter() is NOT a Line2D)."""
        return self.axes[ax].get_lines()

    def line_colors(self, ax=0):
        """Hex colours of the drawn lines -- normalised so 'tab:green', '#2ca02c' and
        (0.17,0.63,0.17) all compare equal."""
        return [to_hex(line.get_color()) for line in self.lines(ax)]

    def line_styles(self, ax=0):
        """Linestyles of the drawn lines ('-', '--', ...). matplotlib may normalise
        '--' to the tuple form, so compare against BOTH via is_dashed() when in doubt."""
        return [line.get_linestyle() for line in self.lines(ax)]

    def is_dashed(self, line):
        """True if a Line2D is dashed, robust to matplotlib returning '--' or the
        equivalent dash-tuple (offset, [on, off])."""
        ls = line.get_linestyle()
        if isinstance(ls, str):
            return ls in ("--", "dashed")
        # dash tuple: (offset, on_off_seq) -- non-empty on_off_seq means a dash pattern
        return bool(ls and ls[1])

    def scatter_colors(self, ax=0):
        """Hex colours of scatter() collections (drawn separately from lines)."""
        out = []
        for coll in self.axes[ax].collections:
            fc = coll.get_facecolor()
            if len(fc):
                out.append(to_hex(fc[0]))
        return out

    def scales(self, ax=0):
        """(xscale, yscale), e.g. ('linear', 'log') or ('log', 'symlog')."""
        a = self.axes[ax]
        return (a.get_xscale(), a.get_yscale())

    def legend_labels(self, ax=0):
        leg = self.axes[ax].get_legend()
        return [] if leg is None else [t.get_text() for t in leg.get_texts()]

    def has_legend(self, ax=0):
        return self.axes[ax].get_legend() is not None

    def legend_loc_code(self, ax=0):
        """The REQUESTED legend location as matplotlib's integer code (0 == 'best',
        1 == 'upper right', ...). Reads the private Legend._loc -- see the module
        caveat. Returns None when there is no legend."""
        leg = self.axes[ax].get_legend()
        return None if leg is None else getattr(leg, "_loc", None)
