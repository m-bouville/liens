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
