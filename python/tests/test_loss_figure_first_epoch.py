"""should_write_loss_figure must NOT write on a run's first epoch: a single
history point plots as an empty figure. The skip is by POINT COUNT (robust to
resumes, whose history restarts at 1), and the epochs=0 ablation is unaffected
because its figure comes from the caller's unconditional end-of-run write."""
from utils.plots import should_write_loss_figure, LOSS_FIGURE_EVERY


def test_first_epoch_one_point_is_skipped_both_modes():
    assert should_write_loss_figure(1, True, n_points=1) is False
    assert should_write_loss_figure(1, False, n_points=1) is False


def test_second_epoch_writes_once_plottable():
    assert should_write_loss_figure(2, True, n_points=2) is True
    # throttled mode still respects `every`, but only once there are >= 2 points
    assert should_write_loss_figure(LOSS_FIGURE_EVERY, False, n_points=LOSS_FIGURE_EVERY) is True


def test_no_n_points_is_backward_compatible():
    # callers that don't pass n_points get the pre-change behaviour exactly
    assert should_write_loss_figure(1, True) is True
    assert should_write_loss_figure(1, False) == (1 % LOSS_FIGURE_EVERY == 0)


def test_ablation_epoch0_periodic_skipped_but_covered_by_final_write():
    # epoch 0 with one point: periodic write skipped; the trainer's
    # unconditional final loss_curve (no n_points) still produces the figure
    assert should_write_loss_figure(0, False, n_points=1) is False
    assert should_write_loss_figure(0, False) is True     # the final-write path


def test_rollout_scatter_is_written_periodically_not_only_at_the_end():
    """The L_rollout-vs-L_1step scatter used to be written ONLY in the final
    block, so a still-running or early-stopped n_rollout_steps>1 run showed the
    loss curve (written periodically) but never the scatter -- the "missing
    plot". It must now be written on the loss-figure cadence too: both a
    PERIODIC call (under the should_write_loss_figure gate) and the FINAL one."""
    import inspect
    from training.train_lds import train_lds
    src = inspect.getsource(train_lds)
    # The scatter is now handed to write_epoch_figures as `extra`, which invokes
    # it on the SAME cadence: the periodic (gated) write and the final
    # (force=True) write. Two extra= sites == both the periodic and final scatter.
    assert src.count("extra=_write_rollout_scatter") >= 2, (
        "the rollout scatter is not handed to write_epoch_figures at BOTH the "
        "periodic and final sites -- a mid-run/early-stopped run will miss it"
    )
    # the scatter closure exists and is gated on show_1step (n_rollout_steps>1)
    assert "def _write_rollout_scatter()" in src
    assert "if show_1step:" in src
    # and it hands the epoch axis + rollout depth to the plot (ratio panel + n-line)
    assert "saved_epochs=saved_epoch_hist" in src
    assert "n_rollout_steps=n_rollout_steps" in src
