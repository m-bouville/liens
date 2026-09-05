"""
The epoch loop shared by the stage trainers.

train_stage2(), train_lds() and train_refinement() each grew their own copy of
the same per-epoch machinery -- iterate the loader, run the step, accumulate the
weighted components, reduce to per-epoch means -- and the copies drifted (the
BatchNorm-restore-on-skip fix, the merged skip reporter and the stage45_config
provenance all landed in one trainer and not the others). This module holds the
pieces that are genuinely identical so a fix lands once.

The extraction is deliberately incremental: each piece moves here only once it
is exercised by a behavioural test that stays green across the move, so a
regression shows up as a failed assertion rather than as a silently different
number on the next hours-long run. `accumulate_epoch` is the first piece -- the
innermost train/val batch pass.

The stage-specific part each trainer keeps is its `forward_fn`: a callable that
takes one batch and returns `{component_name: scalar_tensor}` INCLUDING a
"total" key (the value that is back-propagated / compared), already guarded and
(for the train pass) already stepped. `accumulate_epoch` never touches the
model, the optimiser or the guards -- it only iterates and reduces -- so it is
agnostic to how many components a stage has and what they are called.
"""
from __future__ import annotations

from typing import Callable

import torch


def accumulate_epoch(
    loader,
    forward_fn: Callable[[object], dict[str, torch.Tensor]],
    n_samples: int,
    *,
    progress=None,
) -> tuple[dict[str, float], int]:
    """Run `forward_fn` over every batch in `loader`, sample-weighted-average the
    returned components, and return `({name: mean_float}, n_batches)`.

    forward_fn returns a dict of DETACHED scalar tensors (one per component,
    including "total"); the weighting/guarding/stepping happens inside it. The
    average is over SAMPLES, not batches (each batch contributes its own size),
    matching what every trainer's inline loop did -- an unequal last batch would
    otherwise be over-weighted.

    `progress`, if given, has a no-argument `.tick()` called once per batch (the
    trainers' EpochProgress ETA bar); it is the caller's to construct and close.

    An EMPTY loader returns `({}, 0)`: there is nothing to average and the
    component names are unknown, so the caller's `means["total"]` raises
    KeyError. This is deliberate. The inline loops it replaced computed
    `zeros/0` and silently produced nan for every metric, which then flowed into
    the save criterion as a nan val_loss; a degenerate split (an empty
    validation set) should fail at the point it happens, not train on nan.
    """
    sums: dict[str, torch.Tensor] = {}
    n_batches = 0
    for batch in loader:
        if progress is not None:
            progress.tick()
        n_batches += 1
        # Tuple/list batch (the rollout trainers: x_window, dt_window, theta) ->
        # size from element 0; a bare-tensor batch (stage 1 without stats) -> its
        # own size. Backward-compatible: a tuple still takes batch[0].size(0).
        bs = (batch[0] if isinstance(batch, (list, tuple)) else batch).size(0)
        components = forward_fn(batch)
        for name, value in components.items():
            sums[name] = sums.get(name, value.new_zeros(())) + value * bs
    means = {name: (s / n_samples).item() for name, s in sums.items()}
    return means, n_batches


def weighted_contributions(
    raw: dict[str, float],
    weights: dict[str, float],
    scales: dict[str, float],
) -> dict[str, float]:
    """`{name: weights[name] * raw[name] / scales[name]}` for every raw key.

    The single arithmetic behind a stage's objective decomposition: the total is
    the sum of these, the per-epoch loss line prints them, the component-history
    scatter plots them, and scale_balance_report diagnoses their shares. Each
    trainer built this comprehension three or four times inline with the same
    keys spelled out each time; one drifted (a term added to the objective but
    not the history) exactly the way this module exists to prevent. A term whose
    name is absent from `weights` defaults to weight 1 (the implicit anchor --
    stage 2's recon0 has no recon0_weight parameter); an absent scale is an
    error, because an unscaled term silently dominates.
    """
    return {name: weights.get(name, 1.0) * raw[name] / scales[name] for name in raw}



def format_component_side(total: float, contribs: list[float]) -> str:
    """One side of an epoch loss line: ``{total} =c0 +c1 +c2 ...`` with the shared
    7.4f width convention. `contribs` are the already-WEIGHTED per-component values
    in display order -- the caller builds that list, since which weight is
    effective vs full and which components are active is stage-specific. Extracted
    so the width/`+` convention is single-sourced across the component-decomposition
    stages (refinement, stage 2, stage 1); train_lds's rollout/1step line is a
    different format and does not use this.
    """
    body = f"{contribs[0]:7.4f}" + "".join(f" +{c:7.4f}" for c in contribs[1:])
    return f"{total:7.4f} ={body}"

def write_epoch_figures(
    epoch: int,
    log_every_epoch: bool,
    *,
    force: bool = False,
    epoch_history,
    train_loss_history,
    val_loss_history,
    best_so_far_history,
    loss_curve_path,
    title: str,
    event_epochs=None,
    secondary_train=None,
    secondary_val=None,
    secondary_label: str = "1step",
    reference_levels=None,
    train_full_weight=None,
    component_histories=None,
    loss_components_path=None,
    scale_ratios=None,
    scales_path=None,
    extra=None,
) -> None:
    """The per-epoch figure/CSV writes shared by every trainer: the loss curve
    and its CSV sidecar always, then the stage's decomposition figure.

    Gated by should_write_loss_figure (in-loop writes are throttled and skip a
    run's first epoch, which plots as one point) unless `force=True`, for the
    unconditional final write -- a run can end on a throttled or early-stopped
    epoch, leaving on-disk figures up to `every` epochs stale, the state a
    finished run gets judged from.

    `title` is the stage label ("Stage 4", "Stage 3"); " loss" / " loss
    components" are appended. The loss curve/CSV carry an optional secondary
    series (train_lds's 1step line, with `secondary_label`) and optional
    `reference_levels` (train_lds's ancestor bars) -- both no-ops for stages that
    pass neither. The stage's decomposition figure is EITHER the standard
    component-share scatter (pass `component_histories` + `loss_components_path`,
    stages 2/4/5) OR a stage-specific `extra` callable (train_lds's
    rollout-vs-1step scatter); a stage uses whichever fits, or neither.

    Imported lazily so the module's pure core carries no matplotlib dependency.
    """
    from utils.plots import (loss_component_scatter, loss_curve,
                             should_write_loss_figure, write_loss_history)
    if not (force or should_write_loss_figure(
            epoch, log_every_epoch, n_points=len(epoch_history))):
        return
    loss_curve(epoch_history, train_loss_history, val_loss_history, best_so_far_history,
               loss_curve_path, title=f"{title} loss", event_epochs=event_epochs,
               secondary_train=secondary_train, secondary_val=secondary_val,
               secondary_label=secondary_label, reference_levels=reference_levels,
               train_full_weight=train_full_weight)
    write_loss_history(loss_curve_path, epoch_history, train_loss_history,
                       val_loss_history, best_so_far_history,
                       secondary_train=secondary_train, secondary_val=secondary_val,
                       train_full_weight=train_full_weight)
    if component_histories is not None:
        loss_component_scatter(epoch_history, component_histories, loss_components_path,
                               title=f"{title} loss components")
    if scale_ratios is not None:
        from utils.plots import loss_scale_curve
        loss_scale_curve(epoch_history, scale_ratios, scales_path,
                         title=f"{title} loss/scale ratios (val)",
                         event_epochs=event_epochs)
    if extra is not None:
        extra()
