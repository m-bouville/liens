"""
dt-homogeneous batching for the adaptive sub-stepping integrator.

THE COST MODEL. `_integrate` masks samples that have already arrived and
loops to the batch's MAXIMUM sub-step count, so a batch costs

    batch_size * max(n_substeps over the batch)

rather than the sum of its samples' own counts. That is a property of the
masked implementation, not a defect of it: a ragged loop would need
per-sample control flow, which is far slower on a GPU than doing a few extra
no-op steps -- but it does mean a batch is priced by its worst member.

WHY IT MATTERS HERE. Under alpha the count varies per window by more than an
order of magnitude (measured on the 128x128 stage-3b population at alpha=0.1:
mean 17.7, p95 36, max 132). A batch of 2048 drawn uniformly from 31139
windows contains a near-population-maximum sample with near-certainty, so
essentially EVERY batch pays ~132 -- about 7x the mean. Grouping windows of
similar cost into the same batch recovers most of that, at full batch size.

The alternative -- shrinking batch_size so each batch is less likely to
contain an expensive window -- recovers much less (the batch maximum falls
only to a high quantile), costs GPU utilisation, and changes the optimisation
that the learning rate was tuned against. Bucketing changes only the ORDER in
which windows are grouped.

WHY NOT BUCKET BY dt, WHICH IS THE OBVIOUS CHOICE. Because it barely works.
The count is |f_theta|*Delta_t/(alpha*|z1|), and |f_theta|/|z1| varies by an
order of magnitude WITHIN a single Delta_t -- visible directly as the vertical
spread in check_alpha's drive panel. Measured on a population matched to the
128x128 stage-3b marginals, against the ideal of one pass per window:

    scatter in |f|/|z1|      shuffled   dt-bucketed   count-bucketed
      none (dt-driven)          1.91x         1.04x            1.04x
      realistic (sigma 0.45)    6.26x         4.16x            1.19x

So sorting by dt removes the dt-driven part of the variance and leaves the
part that actually dominates. Sorting by the ESTIMATED COUNT reaches 1.19x.
The estimate costs one f_theta evaluation per window per refresh -- about 9%
of one epoch's evaluations -- against a saving of roughly 5x, so it pays for
itself many times over even refreshed every epoch.

WHAT THIS COSTS IN EXCHANGE. Batches are no longer uniform samples of the
population: each is drawn from one dt band, so the gradient of a single batch
is biased toward that band. Over an epoch every window is still visited
exactly once, and the bucket ORDER is reshuffled every epoch, so the epoch's
gradient is unbiased -- but consecutive steps are correlated in a way they
were not before. That is the real trade, and it is why this is opt-in rather
than the default.
"""
import numpy as np
from torch.utils.data import Sampler


class CostBucketBatchSampler(Sampler):
    """Yields batches whose windows have similar SUB-STEP COST.

    Sorts by a per-window cost estimate, cuts the sorted order into
    contiguous batches, then shuffles the ORDER OF BATCHES (not their
    contents). Within a batch the cost is homogeneous -- which is what bounds
    the loop -- while across an epoch the model still sees every window
    exactly once, in an order that differs each epoch.

    The cost estimate is whatever the caller supplies: the sub-step count
    from estimate_window_costs (what actually works), or the per-window
    maximum dt (a metadata-only fallback that recovers far less -- see the
    module docstring's table).

    COUPLING TO THE MODEL, acknowledged rather than avoided. A count-sorted
    order depends on f_theta, so which windows share a batch changes as
    f_theta trains. That is a real feedback path and the reason to prefer dt
    if dt worked; it does not. The mitigations are that the order is
    refreshed on a schedule rather than continuously, every window is still
    visited exactly once per epoch, and the batch ORDER is reshuffled -- so
    the epoch's gradient is unbiased regardless of how the sort came out.
    """

    def __init__(self, costs, batch_size: int, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.update_costs(costs)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._epoch = 0
        self._seed = seed

    def update_costs(self, costs) -> None:
        """Re-sort for a new cost estimate (i.e. after f_theta has moved).

        Separate from __init__ so a trainer can refresh the order mid-run
        without rebuilding the DataLoader around it -- rebuilding would
        discard persistent workers, which on Windows costs seconds per epoch.
        """
        self._order = np.argsort(np.asarray(costs, dtype=np.float64), kind="stable")

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle the batch order for a new epoch.

        Optional: __iter__ advances the epoch itself, so a caller that simply
        iterates each epoch gets a fresh order. This exists for callers that
        need reproducibility across a resume.
        """
        self._epoch = int(epoch)

    def _batches(self) -> list[np.ndarray]:
        batches = [self._order[i:i + self.batch_size]
                   for i in range(0, len(self._order), self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        return batches

    def __iter__(self):
        batches = self._batches()
        if self.shuffle:
            # Seeded per epoch so a resumed run does not repeat the order it
            # last used, and two runs with the same seed agree.
            rng = np.random.default_rng(self._seed + self._epoch)
            rng.shuffle(batches)
            self._epoch += 1
        for batch in batches:
            yield [int(i) for i in batch]

    def __len__(self) -> int:
        return len(self._batches())


def budget_report(sampler, batch_size: int, n_rollout_steps: int,
                   peak_bytes: float | None = None,
                   reserved_bytes: float | None = None) -> str:
    """What the budget is actually bounding, and what it costs in bytes.

    THE FAILURE THIS PREVENTS: a budget that never binds. If batch_size caps
    every batch before the budget does, the budget is inert and peak memory is
    still batch_size * max_substeps -- the very quantity it was introduced to
    replace. That happened on a real run: budget=300000 with batch_size=1024
    and max_substeps=256 meant a batch would have needed 1172 windows to reach
    the budget, so every batch was 1024 and the peak sat at 1024*256=262144 by
    accident rather than by design. It then looked safe to raise max_substeps,
    which would have doubled the peak and gone straight back to OOM.

    So this states BOTH numbers: what bounds each batch, and the resulting
    peak. With a measured peak_bytes it also reports bytes per sample-substep,
    which is the constant needed to set the budget from VRAM directly:

        budget = usable_VRAM / (n_rollout_steps * bytes_per_sample_substep)
    """
    # _batches, NOT iter(sampler): iterating a shuffle=True sampler advances
    # its epoch counter, so each report call would shift every subsequent
    # epoch's batch order -- found in review, where the startup report plus
    # the epoch-1 report were consuming two epochs' worth of shuffles before
    # training took its first step.
    sizes = [len(b) for b in sampler._batches]
    if not sizes:
        return "cost-budgeted batches: dataset is EMPTY -- nothing to batch."
    peak = sampler.peak_cost()
    # DID THE BUDGET ACTUALLY CUT ANYTHING? Answered by looking for a batch
    # that stopped short of batch_size for cost reasons -- the final batch is
    # excluded because it is the remainder and is short for arithmetic
    # reasons, not budgetary ones.
    #
    # Two earlier versions of this test were wrong in opposite directions.
    # "Did any batch reach batch_size and is the peak under budget" reported
    # batch_size whenever ANY batch was cheap enough to fill. Comparing the
    # peak against batch_size * max(cost) reported BUDGET even when nothing
    # had been cut -- because on a skewed distribution the few deepest windows
    # land in the short remainder batch, so that product is a hypothetical the
    # sampler never realises.
    #
    # The distinction matters for exactly one question the reader has: if I
    # raise max_substeps, does memory go up? It does iff the budget is not
    # cutting.
    budget_cuts = any(len(b) < batch_size for b in sampler._batches[:-1])
    # What an UNBUCKETED fixed-size run would peak at: a random batch of
    # batch_size out of tens of thousands contains a near-deepest window with
    # near-certainty, so its peak really is batch_size * max(cost).
    unbucketed_peak = (batch_size * float(np.max(sampler._retained))
                       if len(sampler._retained) else 0.0)
    lines = [
        f"cost-budgeted batches: {len(sampler)} per epoch, {min(sizes)}-{max(sizes)} "
        f"windows each, budget {sampler.budget:.0f} sample-substeps.",
        f"  peak = {peak:.0f} retained sample-substeps x {n_rollout_steps} rollout "
        f"steps (a batch's size x its own max RETAINED depth"
        + (f", = min(sub-steps, truncate_bptt={sampler.truncate_bptt})"
           if sampler.truncate_bptt else "")
        + f"). Unbucketed at this batch_size it would be {unbucketed_peak:.0f}.",
        f"  compute is separate: {sampler.compute_cost():.3e} f_theta evaluations "
        f"per epoch, which scales with the FULL sub-step count -- every sub-step "
        f"is evaluated forward whether or not its graph is kept.",
    ]
    if budget_cuts:
        lines.append(
            "  The budget IS cutting batches: raising max_substeps costs smaller "
            "batches for the deep windows rather than more memory. NOTE the bound is "
            "on the ESTIMATED cost -- the estimate uses each window's first state and "
            "worst transition, while the integrator re-derives the count per "
            "transition from the state it actually reaches, so realised memory "
            "carries some slack over the budget. The measured bytes/sample-substep "
            "constant below absorbs that slack, which is why setting the budget from "
            "the MEASURED constant is reliable and setting it from first principles "
            "is not.")
    else:
        lines.append(
            f"  The budget is NOT cutting any batch -- every one is capped by "
            f"batch_size={batch_size} first. Peak memory therefore still scales with the "
            f"deepest window, and raising max_substeps WILL raise it. Set the budget "
            f"below {peak:.0f} to take control of the peak.")
    if peak_bytes and peak > 0:
        per_unit = peak_bytes / (peak * n_rollout_steps)
        lines.append(
            f"  measured {peak_bytes / 2**30:.2f} GiB peak = {per_unit:.0f} bytes per "
            f"RETAINED sample-substep. To fit V bytes: budget = V / "
            f"({n_rollout_steps} x {per_unit:.0f}).")
        if sampler.truncate_bptt:
            lines.append(
                "  (Before this was measured against the raw sub-step count, the "
                "constant wandered 33764 -> 8961 -> 6438 -> 13493 across four runs: "
                "it was dividing real memory by a cost that truncation had stopped "
                "making predictive. Against retained depth it should now hold "
                "steady across budgets and batch sizes -- if it still moves, the "
                "memory model is wrong again.)")
    if reserved_bytes and peak_bytes:
        overhead = reserved_bytes - peak_bytes
        lines.append(
            f"  allocator RESERVED {reserved_bytes / 2**30:.2f} GiB to hold that "
            f"{peak_bytes / 2**30:.2f} GiB of live tensors -- "
            f"{overhead / 2**30:.2f} GiB ({100 * overhead / reserved_bytes:.0f}%) is "
            f"cached free blocks, not data.")
        if overhead > peak_bytes:
            lines.append(
                "  MOST OF THE VRAM IS FRAGMENTATION, not activations. Cost-budgeted "
                "batching produces a different batch SHAPE for every batch, and the "
                "caching allocator keeps a separate pool of blocks per shape, so it "
                "reserves far more than it ever holds live. Shrinking the budget "
                "will barely move this. What does: PYTORCH_CUDA_ALLOC_CONF="
                "expandable_segments:True (lets one segment grow instead of "
                "hoarding per-size blocks), or fewer distinct batch sizes.")
    return "\n".join(lines)


def substep_cost(counts_per_window, batches) -> float:
    """Total f_theta evaluations a given batching would pay.

    sum over batches of len(batch) * max(count in batch) -- the masked loop's
    actual cost. Used to measure what bucketing saves rather than assert it.
    """
    counts = np.asarray(counts_per_window, dtype=np.float64)
    return float(sum(len(b) * counts[list(b)].max() for b in batches if len(b)))


def estimate_window_costs(dataset, f_theta, alpha: float, max_substeps: int,
                           device, batch_size: int = 4096) -> "np.ndarray":
    """Sub-step count each base window would need, at the CURRENT f_theta.

    One f_theta evaluation per window -- at the window's FIRST state, which is
    where _substeps_for evaluates the criterion for its first transition -- so
    the whole dataset costs about as much as 1/12th of an epoch's forward
    passes at the measured mean count of 17.7. No backward pass, no frame
    loads: z0/z1 come from the already-cached latents.

    APPROXIMATE, deliberately. A window's later transitions get their own
    counts at run time from their own states, which this cannot know without
    integrating. It only has to rank windows well enough for similar ones to
    share a batch; being wrong about a window's exact cost moves it a few
    places in the sort, which costs a little of the saving and nothing else.
    """
    import torch

    from models.latent_dynamics import LatentDynamics  # noqa: F401  (documented dependency)

    if len(dataset) != len(dataset._index):
        # AUGMENTATION GUARD. MicrostructureEvolutionDataset multiplies its
        # __len__ by the augmentation variants (x32 under augment=True, xK
        # under fixed_aug_indices) while _index holds only the BASE windows --
        # and this estimator, and the samplers fed from it, cover exactly
        # range(len(_index)). Feeding such a sampler to a DataLoader would
        # silently train on 1/32 of the dataset, every epoch, with nothing in
        # any loss curve to show it: the windows visited are real, the count
        # per epoch is plausible, and n_train's normalisation hides the
        # shortfall. train_lds never augments stage 3, so today this cannot
        # trigger -- the guard exists because "safe by accident" is one
        # refactor away from not.
        raise ValueError(
            f"cost bucketing does not support augmented datasets: len(dataset)="
            f"{len(dataset)} but only {len(dataset._index)} base windows are "
            f"indexable by the sampler. Disable bucket_batches or augmentation."
        )
    was_training = f_theta.training
    f_theta.eval()
    z0_rows, z1_rows, theta_rows, dt_rows = [], [], [], []
    for run_idx, start in dataset._index:
        steps = dataset._run_steps[run_idx]
        scale = dataset._run_dt_scale[run_idx]
        z0_rows.append(dataset._run_data[run_idx][start])
        z1_rows.append(dataset._run_data_deriv[run_idx][start])
        theta_rows.append(dataset._run_theta[run_idx])
        dt_rows.append(max((steps[start + i + 1] - steps[start + i]) * scale
                            for i in range(dataset.window_length - 1)))
    counts = []
    with torch.no_grad():
        for lo in range(0, len(z0_rows), batch_size):
            hi = min(lo + batch_size, len(z0_rows))
            z0 = torch.stack(z0_rows[lo:hi]).to(device)
            z1 = torch.stack(z1_rows[lo:hi]).to(device)
            theta = torch.stack(theta_rows[lo:hi]).to(device)
            dt = torch.tensor(dt_rows[lo:hi], dtype=torch.float32, device=device)
            f = f_theta.f(z0, z1, theta)
            b = z0.shape[0]
            f_norm = torch.linalg.vector_norm(f.reshape(b, -1), dim=1)
            z1_norm = torch.linalg.vector_norm(z1.reshape(b, -1), dim=1)
            tiny = torch.finfo(z1_norm.dtype).tiny
            raw = torch.where(f_norm > 0,
                               f_norm * dt / (alpha * z1_norm.clamp_min(tiny)),
                               torch.zeros_like(f_norm))
            n = torch.ceil(raw).clamp(min=1.0, max=float(max_substeps))
            counts.append(torch.nan_to_num(n, nan=float(max_substeps),
                                            posinf=float(max_substeps)).cpu())
    if was_training:
        f_theta.train()
    return torch.cat(counts).numpy() if counts else np.zeros(0, dtype=np.float32)


class BudgetedBatchSampler(Sampler):
    """Batches of a fixed COST, not a fixed size.

    THE MEMORY MODEL, which is the reason this exists. `_integrate` masks
    arrived samples by zeroing their h rather than indexing them out, so every
    one of the loop's n_max iterations allocates full-batch tensors and the
    autograd graph retains all of them. Peak backward memory is therefore

        batch_size * MAX(n_substeps over the batch) * n_rollout_steps * K

    -- the batch maximum, paid by every sample, with the mean playing no part.
    A fixed batch_size therefore has a peak set by whichever window in it
    happens to be deepest, which is why raising max_substeps 256 -> 512 caused
    an out-of-memory even after batch_size was halved 2048 -> 1024: the
    product went UP, 270k sample-substeps to 524k.

    Filling each batch to a fixed budget of `batch_size * depth` instead makes
    peak memory a constant of the run, chosen once, rather than an emergent
    property of the deepest window in the population. Deep windows get small
    batches; shallow ones get large ones, which is also where the throughput
    comes back -- under a fixed size the shallow batches were paying for
    headroom the deep ones needed.

    It also decouples the three knobs that were fighting: max_substeps goes
    back to being a correctness bound (how far alpha may be overridden),
    batch_size becomes a cap for cheap windows, and memory is the budget.
    Raising max_substeps to 1024 for genuinely hard windows then costs a
    smaller batch for those windows, not an OOM.

    THE COST: batch size varies, so gradient noise varies batch to batch. With
    grad_clip active every step is normalised anyway, which mutes it; without
    clipping this would need more thought.
    """

    def __init__(self, costs, batch_size: int, budget: float | None = None,
                 shuffle: bool = True, seed: int = 0,
                 truncate_bptt: int | None = None):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = int(batch_size)
        self._explicit_budget = budget
        # RETAINED DEPTH, not raw sub-step count. Under truncated BPTT the
        # graph is detached every truncate_bptt sub-steps, so a window needing
        # 900 sub-steps retains only ~64 -- and memory follows the retained
        # depth, not the count. Budgeting on the raw count then bounds a
        # quantity that no longer drives memory, which is exactly what was
        # observed: peak stayed at 0.82 GiB whether the budget was 50000 or
        # 100000, and the "bytes per sample-substep" constant wandered
        # 33764 -> 8961 -> 6438 -> 13493 across four runs because it was
        # dividing real memory by a cost that had stopped predicting it.
        self.truncate_bptt = truncate_bptt
        self.shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self.update_costs(costs)

    def _fill(self, order) -> list:
        """Greedy fill of `order` into batches, under three caps.

        budget      -- peak retained memory (batch_size x max retained depth)
        batch_size  -- the caller's cap, for cheap windows
        SPAN        -- counts within a batch differ by at most truncate_bptt

        THE SPAN CAP is the non-obvious one. Under PER-SAMPLE truncation each
        sample's FINAL SEGMENT is retained, so a batch costs

            batch_size x (segments containing an arrival) x truncate_bptt

        and NOT batch_size x truncate_bptt. A batch whose counts span 300-900
        at k=64 has arrivals in ~9 distinct segments and retains ~9k --
        measured at 35x the batch-wide graph on a (100,300,600,900) batch,
        which is the VRAM regression the per-sample gradient fix introduced.
        Capping the span at k bounds arrivals to at most two segments.

        Cheap in practice: the population is already sorted by count, so this
        splits only batches whose members were adjacent anyway. On a real
        max_dt=2000 population the MEDIAN batch span was 5 -- it is the single
        deepest batch, span 214, that this reins in.
        """
        batches, current, current_min = [], [], 0.0
        for idx in order:
            lo = current_min if current else float(self._costs[idx])
            depth = self._depth(lo, float(self._costs[idx]), len(current) + 1)
            if current and ((len(current) + 1) * depth > self.budget
                            or len(current) >= self.batch_size):
                batches.append(np.array(current))
                current = []
            if not current:
                current_min = float(self._costs[idx])
            current.append(int(idx))
        if current:
            batches.append(np.array(current))
        return batches

    def _depth(self, lo: float, hi: float, n: int = 1 << 30) -> float:
        """Retained depth of a batch whose counts run from `lo` to `hi`.

        Without truncation every sample keeps its whole history, so the batch
        is priced by its DEEPEST member: hi.

        With per-sample truncation each sample retains only its own final
        segment -- but those segments sit at different places in the loop, and
        the updates are full-batch tensor ops, so every segment in which SOME
        sample arrives is retained for the whole batch:

            depth ~ (number of arrival segments) x k ~ span + k

        A fixed span cap was the first attempt and it was the wrong shape: it
        split the deep tail into 3-window batches, each still taking a full
        clipped optimizer step. Pricing the span instead lets a small batch
        carry a wide span (few samples x many segments is cheap) while a
        4096-window batch is forced to stay narrow -- which is the actual
        constraint.
        """
        if self.truncate_bptt is None:
            return hi
        k = float(self.truncate_bptt)
        # Arrival segments are bounded THREE ways, and the tightest wins:
        #   n           -- at most one arrival segment per sample, so n*k
        #   span/k + 1  -- segments the counts actually straddle, so span + k
        #   hi          -- the loop is only hi steps long
        # Missing the n bound was my first model, and it charged a 2-window
        # batch spanning 1000-41000 as if it retained 41k steps when it can
        # retain at most 2 segments = 2k. That shattered the deep tail into
        # 1-window batches, each still taking a full clipped optimizer step.
        return min(hi, n * k, (hi - lo) + k)

    def update_costs(self, costs, hold_batch_count: bool = True) -> None:
        """Re-sort and re-batch for a new cost estimate.

        HOLDING THE BATCH COUNT is the default, and it exists because of a
        real incident. Every batch is one optimizer step, and with grad_clip
        active each step moves the weights by roughly lr regardless of batch
        size -- so the number of batches per epoch IS the per-epoch learning
        rate. A refresh that changes it rescales the optimisation mid-run,
        silently, with no parameter having changed.

        Observed: a run at 16 batches/epoch refreshed at epoch 26 and began
        reporting 27-43 SKIPS per epoch -- more skips than it previously had
        batches -- because the sharpened costs made the budget cut far harder.
        Everything epoch-indexed (patience, the refresh interval itself) was
        rescaled with it, and the run destabilised immediately afterwards
        having been descending cleanly for 25 epochs.

        So on a refresh the budget is rescaled to keep the batch count where
        it was, rather than keeping the budget fixed and letting the count
        move. Memory follows the budget, so this trades a bounded increase in
        peak memory for a stable optimisation -- the right way round, since
        peak memory is checkable in advance and a silent lr change is not.

        Pass hold_batch_count=False for the old behaviour (fixed budget,
        floating count), which is right when the batching is being rebuilt
        deliberately rather than refreshed.
        """
        previous_count = len(getattr(self, "_batches", []))
        self._costs = np.asarray(costs, dtype=np.float64)
        # What memory actually scales with. Compute still scales with the full
        # count (every sub-step is evaluated forward), so _costs is kept for
        # the compute reporting; _retained is what the budget bounds.
        self._retained = (np.minimum(self._costs, float(self.truncate_bptt))
                          if self.truncate_bptt else self._costs)
        if self._costs.size == 0:
            # budget must still exist: budget_report reads it, and an empty
            # dataset should produce an empty (but well-formed) report rather
            # than an AttributeError two calls later. Found in review.
            self.budget = float(self._explicit_budget) if self._explicit_budget else 0.0
            self._batches = []
            return
        # AUTO BUDGET: the batch_size the caller asked for, at the population's
        # MEDIAN cost. So a typical batch is exactly batch_size, a window
        # costing 4x the median gets a quarter of it, and the caller's number
        # keeps the meaning it had before budgeting existed.
        self.budget = (float(self._explicit_budget) if self._explicit_budget
                       else self.batch_size * float(np.median(self._retained)))
        # Sorted by FULL count. retained = min(count, truncate_bptt) is
        # monotone in count, so this groups by retained depth exactly as the
        # previous lexsort did -- and it ALSO groups by count SPAN, which is
        # what drives retained memory under per-sample truncation (see _fill).
        order = np.argsort(self._costs, kind="stable")
        self._batches = batches = self._fill(order)

        if (hold_batch_count and previous_count
                and len(batches) != previous_count
                and self._explicit_budget is None):
            # Rescale the budget so the epoch keeps its step count, then
            # re-batch once. One correction, not a search: the count is
            # monotone in the budget, and the residual mismatch is a batch or
            # two, which does not meaningfully move the effective lr.
            self.budget *= len(batches) / previous_count
            # SAME rule as the first pass -- shared so the two cannot drift.
            self._batches = batches = self._fill(order)

        # THE EXPLICIT-BUDGET CASE CANNOT BE HELD, and must not be silent.
        # An explicit budget is a memory limit, so rescaling it to preserve
        # the step count would trade an OOM for a stable lr -- the wrong way
        # round. But the step count then moves, which IS an lr change, so the
        # caller is told rather than left to infer it from a skip count that
        # suddenly exceeds the batch count.
        self.last_refresh_note = ""
        if previous_count and len(self._batches) != previous_count:
            ratio = len(self._batches) / previous_count
            if abs(ratio - 1.0) > 0.1:
                self.last_refresh_note = (
                    f"cost refresh changed the batch count {previous_count} -> "
                    f"{len(self._batches)} ({ratio:.2f}x). Every batch is one "
                    f"optimizer step and grad_clip normalises each step, so the "
                    f"per-epoch distance travelled changed by the same factor -- "
                    f"an effective learning-rate change, mid-run, with no "
                    f"parameter altered. Epoch-indexed settings (patience, this "
                    f"refresh interval) rescale with it."
                    + ("" if self._explicit_budget is None else
                       " The budget is EXPLICIT so it was not rescaled to "
                       "compensate: it bounds memory, and raising it to hold the "
                       "step count could exceed VRAM. Set bucket_refresh_epochs=0 "
                       "to freeze the batching, or use the auto budget, which "
                       "does hold the count."))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def compute_cost(self) -> float:
        """Total f_theta evaluations an epoch will pay -- the FULL counts.

        Distinct from peak_cost: every sub-step is evaluated in the forward
        pass whether or not its graph is retained, so compute scales with the
        raw count while memory scales with the retained depth. Conflating them
        is what made the memory constant wander across runs.
        """
        return float(sum(len(b) * self._costs[b].max() for b in self._batches))

    def peak_cost(self) -> float:
        """The largest batch_size * max RETAINED depth any batch will reach.

        What the budget is supposed to bound. Can exceed it only for a single
        window costing more than the whole budget, which must still run --
        reported rather than silently split, since it means max_substeps is
        set beyond what one sample of memory allows.
        """
        # SAME model as _fill budgets on -- reporting the plain retained max
        # here would understate a wide-span batch, which is precisely the case
        # that caused the VRAM regression.
        return max((len(b) * self._depth(float(self._costs[b].min()),
                                          float(self._costs[b].max()), len(b))
                    for b in self._batches), default=0.0)

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            rng = np.random.default_rng(self._seed + self._epoch)
            rng.shuffle(batches)
            self._epoch += 1
        for batch in batches:
            yield [int(i) for i in batch]

    def __len__(self) -> int:
        return len(self._batches)
