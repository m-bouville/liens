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
