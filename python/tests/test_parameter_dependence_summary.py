"""The end-of-run SUMMARY block gathers the headline numbers into one compact,
copy-pasteable, one-metric-per-line format so several runs stack into a table.
These tests pin: the collector records what the report printed, the block
renders those values, and it RESETS between runs (no leakage)."""
import io
from contextlib import redirect_stdout

import evaluation.check_parameter_dependence as m


def test_summary_block_renders_recorded_values():
    m._summary_reset()
    m._summary_put("err_actual", 1.176095)
    m._summary_put("bias_fraction", 0.430)
    m._summary_put("corr_length", 0.63)
    m._summary_put("dt_exponent", 0.818)
    buf = io.StringIO()
    with redirect_stdout(buf):
        m._print_run_summary("checkpoints/stage2/128x128-stage2-20260812_20h08.pt",
                             euler_only=True)
    out = buf.getvalue()
    assert "SUMMARY" in out
    assert "128x128-stage2-20260812_20h08.pt" in out   # basename, not full path
    assert "euler-only" in out
    assert "1.1761" in out           # err_actual, 4dp
    assert "0.430" in out            # bias fraction
    assert "63%" in out              # corr length as percent
    assert "0.818" in out            # dt exponent


def test_summary_resets_between_runs():
    """A field set in run A must not survive into run B's block."""
    m._summary_reset()
    m._summary_put("err_actual", 9.999)
    m._summary_reset()                      # start of run B
    m._summary_put("bias_fraction", 0.1)
    buf = io.StringIO()
    with redirect_stdout(buf):
        m._print_run_summary("B.pt", euler_only=False)
    out = buf.getvalue()
    assert "9.999" not in out               # run A's err_actual is gone
    assert "f_theta-corrected" in out       # euler_only=False label


def test_summary_omits_missing_fields_gracefully():
    """A run that didn't compute a field (e.g. no length scale) just omits that
    row rather than printing None or crashing."""
    m._summary_reset()
    m._summary_put("err_actual", 1.0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        m._print_run_summary("x.pt", euler_only=True)
    out = buf.getvalue()
    assert "err_actual" in out
    assert "corr length_scale" not in out   # not recorded -> not shown
    assert "None" not in out


def test_summary_shows_both_temperature_regimes_and_worst_T():
    """T is not monotonic in loss, so the summary reports the cold (<=0.65) and
    near-critical (>=0.95) tails separately, plus the robust worst-temperature
    aggregate (not a single fragile run)."""
    m._summary_reset()
    m._summary_put("loss_T_cold", 2.04)
    m._summary_put("loss_T_hot", 1.30)
    m._summary_put("worst_T", "0.550 (mean loss 2.04)")
    buf = io.StringIO()
    with redirect_stdout(buf):
        m._print_run_summary("x.pt", euler_only=True)
    out = buf.getvalue()
    assert "mean loss T<=0.65" in out and "2.040" in out
    assert "mean loss T>=0.95" in out and "1.300" in out
    assert "worst temperature" in out and "0.550" in out
    assert "worst-error run" not in out          # replaced, not kept


def test_summary_shows_three_oracle_decades():
    m._summary_reset()
    m._summary_put("oracle_ratio_1e1", 0.325)
    m._summary_put("oracle_ratio_1e2", 0.179)
    m._summary_put("oracle_ratio_1e3", 0.057)
    buf = io.StringIO()
    with redirect_stdout(buf):
        m._print_run_summary("x.pt", euler_only=True)
    out = buf.getvalue()
    assert "oracle/actual 1e1-1e2" in out and "0.325" in out
    assert "oracle/actual 1e2-1e3" in out and "0.179" in out
    assert "oracle/actual 1e3-1e4" in out and "0.057" in out


def test_summary_shows_raw_bias_variance_magnitudes():
    """The bias FRACTION alone hides whether a change is the error growing or
    the bias shrinking; the summary also shows the two raw magnitudes it's
    built from (E[|residual|] total, |E[residual]| bias)."""
    m._summary_reset()
    m._summary_put("err_total_magnitude", 1.6830)
    m._summary_put("err_bias_magnitude", 0.3956)
    m._summary_put("bias_fraction", 0.235)
    buf = io.StringIO()
    with redirect_stdout(buf):
        m._print_run_summary("x.pt", euler_only=True)
    out = buf.getvalue()
    assert "E[|residual|] (total)" in out and "1.6830e+00" in out
    assert "|E[residual]| (bias)" in out and "3.9560e-01" in out


def test_summary_shows_epsprime_asymptote():
    """The |Euler error|/dt floor -- the flat asymptote of the euler |error|/dt
    curve -- is the irreducible per-dt error that sets the usable-dt ceiling.
    Reported with a fixed e-3 suffix so it reads off the [1,0] panel."""
    m._summary_reset()
    m._summary_put("err_over_dt_floor", 0.227)
    buf = io.StringIO()
    with redirect_stdout(buf):
        m._print_run_summary("x.pt", euler_only=True)
    out = buf.getvalue()
    assert "|Euler err|/dt floor" in out and "0.227e-3" in out
    assert "eps'" not in out          # undefined symbol must not appear


def test_summary_header_names_the_checkpoint():
    """The header is 'SUMMARY <filename>' (was a generic 'copy-paste' line) so a
    stacked set of blocks is self-labeling. When a source (original) checkpoint
    is given, the header names IT, not the ephemeral-stage3 wrapper."""
    m._summary_reset()
    buf = io.StringIO()
    with redirect_stdout(buf):
        m._print_run_summary("path/to/128x128-stage2-20260818_13h54.pt",
                             euler_only=True)
    out = buf.getvalue()
    assert "SUMMARY 128x128-stage2-20260818_13h54.pt" in out
    # a rule separates the input params from the output metrics
    assert "-" * 66 in out


def test_summary_shows_training_params_from_stage2_config(tmp_path):
    """The key training params that DIFFER between runs (stage2a,
    z0_from_deriv_weight, deriv_dt_weight_exponent, ...) are read from the
    original checkpoint's stage2_config and printed, so the table shows what
    actually changed between checkpoints."""
    import torch
    ckpt = tmp_path / "src-stage2.pt"
    torch.save({
        "config": {"trunk_from_deriv_weight": 0.5},  # a param in a NON-stage2 dict
        "stage2_config": {
            "stage2a": True, "z0_from_deriv_weight": 0.05,
            "deriv_dt_weight_exponent": -0.5,
            "stats0_weight": 0.25, "deriv_target_centered": True}}, ckpt)
    m._summary_reset()
    buf = io.StringIO()
    with redirect_stdout(buf):
        m._print_run_summary("wrapper-ephemeral-stage3.pt", euler_only=True,
                             source_checkpoint_path=str(ckpt))
    out = buf.getvalue()
    assert "z0_from_deriv_weight" in out and "0.05" in out
    assert "deriv_dt_weight_exponent" in out and "-0.5" in out
    assert "stage2a" in out and "True" in out
    assert "trunk_from_deriv_weight" in out and "0.5" in out  # merged from config, not stage2_config
    # header names the SOURCE checkpoint, not the evaluated wrapper
    assert "SUMMARY src-stage2.pt" in out
    assert "ephemeral-stage3" not in out


def test_summary_params_silently_absent_without_stage2_config(tmp_path):
    """A real stage-3 checkpoint (or any file lacking stage2_config) prints no
    param rows -- the block degrades cleanly, no crash, no empty labels."""
    import torch
    ckpt = tmp_path / "real-stage3.pt"
    torch.save({"config": {"size": 128}}, ckpt)
    m._summary_reset()
    buf = io.StringIO()
    with redirect_stdout(buf):
        m._print_run_summary(str(ckpt), euler_only=False)
    out = buf.getvalue()
    assert "z0_from_deriv_weight" not in out
    assert "SUMMARY" in out          # block still renders


def test_parameter_dependence_has_abs_error_vs_param_row():
    """parameter_dependence.png gained a second row: raw |error| vs temperature
    and vs noise (the [1,0]-style multi-curve panel re-binned against a
    parameter). Guard the wiring so the row isn't silently lost in a refactor."""
    import inspect
    src = inspect.getsource(m._build_and_save_figures)
    # 2x3 grid, not the old 1x3
    assert "plt.subplots(2, 3" in src, "parameter_dependence is no longer a 2-row figure"
    # the helper and both calls
    assert "_plot_abs_error_vs_param" in src
    assert "_ax_err_temp, results.temperatures" in src
    assert "_ax_err_noise, results.noises" in src
    # raw |error|, not divided by dt (option B) -- the helper bins the raw
    # error arrays, never divides by results.dts
    helper = src[src.index("def _plot_abs_error_vs_param"):
                 src.index("_plot_abs_error_vs_param(_ax_err_temp")]
    assert "/ results.dts" not in helper and "/results.dts" not in helper, (
        "the |error| vs param panel must plot RAW |error|, not |error|/dt")
    # unused corner is hidden
    assert 'axes[1, 2].axis("off")' in src


def test_dt_dependence_has_raw_delta_z0_column():
    """dt_dependence.png gained a raw delta z0 column (col 0) alongside the
    delta z0 / dt column (col 1): signed [0,0]/[0,1], absolute [1,0]/[1,1].
    Labels use 'delta z0', defined explicitly, not 'error'."""
    import inspect
    src = inspect.getsource(m._build_and_save_figures)
    assert "_ax_dz0_signed, _ax_dz0_abs = axes_dt[0, 0], axes_dt[1, 0]" in src
    assert "delta z0 = z0_pred(t+dt) - z0_true(t+dt)" in src
    assert "delta z0 (mean, signed)" in src
    assert "|delta z0| / dt (mean, absolute)" in src
    # the raw curves are NOT divided by dt (that is the /dt column's job)
    helper = src[src.index("def _plot_raw_dz0"):src.index("_ax_dz0_signed.axhline")]
    assert "/ results.dts" not in helper and "/results.dts" not in helper
    # 'error' should be gone from the delta-z0 panel titles
    assert "error (mean, signed)" not in src and "|error| (mean, absolute)" not in src


def test_dt_dependence_has_dt_temperature_scatter():
    """dt_dependence.png has a (dt, T) bubble scatter colored by |delta z0|
    at [1,2], drawn via the shared _dtT_bubbles helper. Guard the wiring."""
    import inspect
    src = inspect.getsource(m._build_and_save_figures)
    # the scatter is now a _dtT_bubbles call into axes_dt[1, 2]
    assert "_dtT_bubbles(axes_dt[1, 2]" in src, (
        "the (dt,T) |delta z0| scatter should be drawn at [1,2] via _dtT_bubbles")
    block = src[src.index("_dtT_bubbles(axes_dt[1, 2]"):
                src.index("_dtT_bubbles(axes_dt[1, 2]") + 400]
    assert "results.temperatures" in block and "results.latent_losses" in block
    # the helper log-scales x through the shared guard (no raw set_xscale)
    helper = inspect.getsource(m._dtT_bubbles)
    assert "_shared_log_scale(ax" in helper


def test_parameter_figure_uses_delta_z0_not_error_in_labels():
    """Both figures use the 'delta z0' convention (delta z0 = z0_pred - z0_true),
    not 'error' / 'pred - true', in their VISIBLE labels. Guards against the
    parameter figure drifting back to 'error' while dt_dependence uses delta z0.
    (Comments may still say 'error' as a concept -- only display strings matter,
    so this checks the set_title/set_ylabel/label= calls, not the whole source.)"""
    import inspect, re
    src = inspect.getsource(m._build_and_save_figures)
    # pull the string literals passed to set_title / set_ylabel / label= / colorbar
    label_strs = re.findall(r'set_title\(\s*f?"([^"]*)"', src)
    label_strs += re.findall(r'set_ylabel\(\s*f?"([^"]*)"', src)
    label_strs += re.findall(r'label=f?"([^"]*)"', src)
    joined = " ".join(label_strs)
    assert "error" not in joined.lower(), (
        f"a visible label still says 'error' (use 'delta z0'): "
        f"{[s for s in label_strs if 'error' in s.lower()]}")
    assert "pred - true" not in joined, (
        f"a visible label still says 'pred - true' (use 'delta z0'): "
        f"{[s for s in label_strs if 'pred - true' in s]}")


def test_dt_dependence_has_minus_trivial_difference_panels():
    """dt_dependence.png col 3: (dt,T) bubble panels of causal - trivial [0,3]
    and euler - trivial [1,3], diverging colormap centered at 0 (blue beats
    trivial, red loses). Guards the wiring + the diverging map."""
    import inspect
    src = inspect.getsource(m._build_and_save_figures)
    assert "plt.subplots(2, 4" in src, "dt figure is not the 2x4 grid holding the difference panels"
    assert "axes_dt[0, 3]" in src and "axes_dt[1, 3]" in src
    assert "causal - trivial over (dt, T)" in src
    assert "euler - trivial over (dt, T)" in src
    # the difference panels must use the diverging path of the helper
    assert "diverging=True" in src
    # the shared bubble helper exists and centers the diverging map at 0
    helper = inspect.getsource(m._dtT_bubbles)
    assert 'cmap="coolwarm"' in helper and "vmin=-_amax" in helper
