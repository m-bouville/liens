"""
Cost-bucketed batching: the masked sub-step loop costs the batch MAXIMUM.

_integrate zeroes h for arrived samples and runs to the batch's largest count,
so a batch of 2048 drawn uniformly from a population whose counts span 8x to
132x contains a near-maximum sample with near-certainty and pays for it. The
saving from grouping similar windows is the point; these tests pin that the
grouping is by the right quantity and that it does not change WHAT is trained
on, only the order.
"""
import numpy as np
import pytest
import torch

from training.dt_bucketing import (
    CostBucketBatchSampler, estimate_window_costs, substep_cost,
)


def _population(n=6000, seed=0, sigma=0.45):
    """dt over two decades, counts driven by dt AND by a per-window factor --
    the shape measured on the real stage-3b population."""
    rng = np.random.default_rng(seed)
    dt = np.exp(rng.uniform(np.log(12), np.log(500), n))
    counts = np.ceil(1.6 * dt ** 0.40 * np.exp(rng.normal(0, sigma, n)))
    return dt, counts


def test_bucketing_by_cost_beats_shuffled_by_a_wide_margin():
    """
    THE WHOLE POINT, measured rather than asserted. Shuffled pays ~6x the
    per-window ideal; bucketed by cost pays ~1.2x.
    """
    dt, counts = _population()
    n, B = len(counts), 512
    rng = np.random.default_rng(1)
    perm = rng.permutation(n)
    shuffled = [perm[i:i + B] for i in range(0, n, B)]
    bucketed = list(CostBucketBatchSampler(counts, B, shuffle=False))

    ideal = counts.sum()
    cost_shuffled = substep_cost(counts, shuffled) / ideal
    cost_bucketed = substep_cost(counts, bucketed) / ideal
    assert cost_shuffled > 3.0, f"the test population is not spread enough: {cost_shuffled:.2f}x"
    assert cost_bucketed < 1.35, f"bucketing did not concentrate the cost: {cost_bucketed:.2f}x"
    assert cost_bucketed < cost_shuffled / 2.5


def test_bucketing_by_dt_alone_does_not_work():
    """
    RECORDED BECAUSE IT WAS MY OWN RECOMMENDATION, and it was wrong. dt is the
    obvious sort key and is nearly useless: the count is |f|*dt/(alpha*|z1|)
    and |f|/|z1| varies ~10x WITHIN a single dt, so sorting by dt removes the
    part of the variance that does not dominate.

    Pinned so the cheaper-looking key cannot quietly come back.
    """
    dt, counts = _population()
    B = 512
    by_dt = list(CostBucketBatchSampler(dt, B, shuffle=False))
    by_cost = list(CostBucketBatchSampler(counts, B, shuffle=False))
    ideal = counts.sum()
    assert substep_cost(counts, by_dt) / ideal > 2.5, (
        "dt-bucketing worked on this population -- either the population lost "
        "its |f|/|z1| spread or the claim needs revisiting"
    )
    assert substep_cost(counts, by_cost) < substep_cost(counts, by_dt) / 2


def test_a_stale_estimate_still_helps():
    """
    The sort ages as f_theta sharpens, which is why the refresh is periodic
    rather than per-epoch. A 20% drift must still beat shuffled comfortably,
    or the refresh interval would have to be 1 and the estimator's cost would
    stop paying for itself.
    """
    dt, counts = _population()
    rng = np.random.default_rng(5)
    stale = counts * np.exp(rng.normal(0, 0.20, len(counts)))
    B, n = 512, len(counts)
    perm = rng.permutation(n)
    shuffled = [perm[i:i + B] for i in range(0, n, B)]
    bucketed = list(CostBucketBatchSampler(stale, B, shuffle=False))
    ideal = counts.sum()
    assert substep_cost(counts, bucketed) / ideal < 2.5
    assert substep_cost(counts, bucketed) < substep_cost(counts, shuffled) / 2


def test_every_window_is_visited_exactly_once_per_epoch():
    """
    Bucketing changes the ORDER, not the data. If it dropped or duplicated
    windows the loss would be computed on a different population than the one
    reported, which no loss curve would reveal.
    """
    _, counts = _population(n=1000)
    sampler = CostBucketBatchSampler(counts, 128, shuffle=True)
    for _ in range(3):
        seen = [i for batch in sampler for i in batch]
        assert sorted(seen) == list(range(len(counts)))


def test_the_batch_order_changes_between_epochs():
    """
    Within a batch the composition is fixed by the sort; across batches the
    order must vary, or every epoch presents the cost bands in the same
    sequence and the optimiser sees a systematic curriculum nobody asked for.
    """
    _, counts = _population(n=2000)
    sampler = CostBucketBatchSampler(counts, 256, shuffle=True, seed=3)
    first = [list(b) for b in sampler]
    second = [list(b) for b in sampler]
    assert first != second, "the batch order repeated exactly across epochs"
    assert sorted(map(tuple, first)) == sorted(map(tuple, second)), (
        "the batch CONTENTS changed, not just the order -- bucketing must not "
        "reshuffle across buckets"
    )


def test_shuffle_false_is_deterministic():
    _, counts = _population(n=500)
    a = [list(b) for b in CostBucketBatchSampler(counts, 64, shuffle=False)]
    b = [list(b) for b in CostBucketBatchSampler(counts, 64, shuffle=False)]
    assert a == b


def test_len_matches_the_number_of_batches_yielded():
    """DataLoader trusts __len__; a mismatch shows up as a progress bar that
    lies or an epoch that ends early."""
    _, counts = _population(n=1000)
    for B, drop in ((128, False), (128, True), (7, False), (1000, False)):
        s = CostBucketBatchSampler(counts, B, shuffle=False, drop_last=drop)
        assert len(s) == len(list(s)), (B, drop)


def test_drop_last_only_drops_a_short_final_batch():
    _, counts = _population(n=1000)
    kept = list(CostBucketBatchSampler(counts, 128, shuffle=False, drop_last=False))
    dropped = list(CostBucketBatchSampler(counts, 128, shuffle=False, drop_last=True))
    assert len(kept) == len(dropped) + 1
    assert all(len(b) == 128 for b in dropped)


def test_a_bad_batch_size_is_rejected():
    _, counts = _population(n=100)
    with pytest.raises(ValueError, match="batch_size"):
        CostBucketBatchSampler(counts, 0)


# --------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------

class _Field(torch.nn.Module):
    """f proportional to z1 with a per-window factor, so the true count is
    known in closed form."""

    def __init__(self, k=0.002):
        super().__init__()
        self.k = k
        self.training = False

    def f(self, z0, z1, theta):
        return z1 * self.k

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self


class _MiniEvolutionDataset:
    """The slice of MicrostructureEvolutionDataset the estimator touches --
    including __len__, which the augmentation guard compares against _index.
    (A SimpleNamespace cannot carry __len__: dunder lookup is on the type.)"""

    def __init__(self, window_length):
        self.window_length = window_length
        self._index = []
        self._run_data = []
        self._run_data_deriv = []
        self._run_steps = []
        self._run_dt_scale = []
        self._run_theta = []

    def __len__(self):
        return len(self._index)


def _dataset(n_runs=5, n_steps=8, window_length=3):
    torch.manual_seed(0)
    ds = _MiniEvolutionDataset(window_length)
    for r in range(n_runs):
        ds._run_data.append(torch.randn(n_steps, 4, 8, 8))
        ds._run_data_deriv.append(torch.randn(n_steps, 4, 8, 8))
        ds._run_steps.append([int(500 * 1.5 ** i) for i in range(n_steps)])
        ds._run_dt_scale.append(0.05)
        ds._run_theta.append(torch.tensor([-0.22]))
        for start in range(n_steps - window_length + 1):
            ds._index.append((r, start))
    return ds


def test_the_estimate_matches_the_closed_form_count():
    """With f = k*z1 the criterion is k*dt/alpha exactly, so the estimate is
    checkable against arithmetic rather than against itself."""
    ds = _dataset()
    k, alpha = 0.002, 0.1
    got = estimate_window_costs(ds, _Field(k), alpha, 256, torch.device("cpu"))
    expected = []
    for run_idx, start in ds._index:
        steps, scale = ds._run_steps[run_idx], ds._run_dt_scale[run_idx]
        dt = max((steps[start + i + 1] - steps[start + i]) * scale
                  for i in range(ds.window_length - 1))
        expected.append(min(np.ceil(k * dt / alpha), 256.0))
    assert np.allclose(got, expected)


def test_the_estimate_uses_the_windows_worst_transition():
    """
    A window is as expensive as its most demanding step -- the same reasoning
    the max_dt window filter uses. Using the FIRST transition instead would
    systematically under-rank windows whose later steps are the long ones,
    which on a geometric save schedule is most of them.
    """
    ds = _dataset(n_runs=1, n_steps=4, window_length=3)
    # k large enough that the first-vs-worst difference survives ceil(): at
    # k=0.01 both rounded to 2 and the test could not tell them apart, which
    # is a test that passes for the wrong reason rather than a real result.
    k, alpha = 1.0, 0.1
    got = estimate_window_costs(ds, _Field(k), alpha, 10_000, torch.device("cpu"))
    steps, scale = ds._run_steps[0], ds._run_dt_scale[0]
    checked = 0
    for idx, (run_idx, start) in enumerate(ds._index):
        first = (steps[start + 1] - steps[start]) * scale
        worst = max((steps[start + i + 1] - steps[start + i]) * scale for i in range(2))
        if np.ceil(k * worst / alpha) > np.ceil(k * first / alpha):
            assert got[idx] == pytest.approx(np.ceil(k * worst / alpha)), (
                f"window {idx}: estimate {got[idx]} matches the FIRST transition "
                f"({np.ceil(k * first / alpha)}), not the worst "
                f"({np.ceil(k * worst / alpha)})"
            )
            checked += 1
    assert checked, "no window had a worst transition beyond its first; test vacuous"


def test_the_estimate_respects_max_substeps():
    ds = _dataset()
    got = estimate_window_costs(ds, _Field(10.0), 0.001, 32, torch.device("cpu"))
    assert got.max() <= 32


def test_the_estimator_leaves_the_model_in_training_mode():
    """
    It runs under eval() to avoid touching anything mode-dependent, but a
    refresh happens INSIDE the epoch loop -- leaving f_theta in eval would
    silently disable dropout/BN updates for the rest of training.
    """
    ds = _dataset()
    field = _Field()
    field.train()
    estimate_window_costs(ds, field, 0.1, 256, torch.device("cpu"))
    assert field.training, "the estimator left f_theta in eval mode"


def test_a_zero_field_costs_one_substep_per_window():
    """Every fresh stage 3a: f is zero-initialised, so nothing needs
    sub-stepping and the sort is degenerate but harmless."""
    ds = _dataset()
    got = estimate_window_costs(ds, _Field(0.0), 0.1, 256, torch.device("cpu"))
    assert np.all(got == 1.0)


# --------------------------------------------------------------------
# Wiring into train_lds
# --------------------------------------------------------------------

def test_bucketing_is_only_built_under_alpha():
    """
    Fixed n_substeps gives every window the same count, so there is nothing to
    group and the estimator would be pure overhead. Guarding on alpha keeps
    the non-adaptive path byte-identical to before.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "epochs > 0 and alpha is not None and bucket_batches:" in src, (
        "bucketing is not gated on alpha; a fixed-n run would pay the estimator "
        "for a sort that cannot help it"
    )


def test_the_sampler_is_defined_on_the_epochs_zero_path():
    """
    epochs=0 builds no train loader, and the epoch loop's refresh check still
    reads the name. Initialising it only where the loader is built raised
    UnboundLocalError -- caught by the ablation's own test, and pinned here so
    the two stay tied together.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert src.count("_bucket_sampler = None") >= 2, (
        "_bucket_sampler is initialised on only one branch"
    )


def test_train_lds_accepts_the_bucketing_parameters():
    import inspect

    from training.train_lds import train_lds
    params = inspect.signature(train_lds).parameters
    assert params["bucket_batches"].default is True, (
        "bucketing should be on by default under alpha -- it is a pure cost "
        "saving that does not change what is trained on"
    )
    assert params["bucket_refresh_epochs"].default == 25


def test_bucketing_changes_the_batches_but_not_the_result(tmp_path, isolated_project_root):
    """
    END TO END, and the property that matters: bucketing reorders batches, so
    a run with it and a run without it visit the same windows and should reach
    comparable losses -- but they are NOT bit-identical, because batch
    composition changes the gradient sequence. Asserting equality would be
    wrong; asserting the run completes and lands in the same region is the
    honest check.
    """
    from test_train_lds import _build_sweep, _cached_stage2_ancestor
    from training.train_lds import train_lds

    base_path, stage2_path = _cached_stage2_ancestor(tmp_path)
    common = dict(
        size=32, base_path=base_path, ae_checkpoint_path=stage2_path,
        ae_stats_weight=0.01, epochs=2, batch_size=4, hidden_dim=8,
        n_hidden_layers=1, val_fraction=0.34, test_fraction=0.17, num_workers=0,
        n_rollout_steps=2, min_step=0, min_stdev_phi=None, encode_batch_size=4,
        ema_warmup_epochs=0, device="cpu", seed=0, log_every_epoch=False,
        z1_resync=False, alpha=0.05,
    )
    plain = train_lds(checkpoint_path=tmp_path / "plain.pt",
                       loss_curve_path=tmp_path / "plain.png",
                       bucket_batches=False, **common)
    bucketed = train_lds(checkpoint_path=tmp_path / "bucketed.pt",
                          loss_curve_path=tmp_path / "bucketed.png",
                          bucket_batches=True, **common)
    a = torch.load(bucketed, map_location="cpu", weights_only=True)
    b = torch.load(plain, map_location="cpu", weights_only=True)
    assert np.isfinite(a["val_loss"]) and np.isfinite(b["val_loss"])
    # same population, so the same order of magnitude -- a bucketing bug that
    # dropped or duplicated windows would show up here
    assert 0.2 < a["val_loss"] / b["val_loss"] < 5.0, (
        f"bucketed {a['val_loss']:.4g} vs plain {b['val_loss']:.4g}: the two "
        f"runs did not train on the same population"
    )


# --------------------------------------------------------------------
# Cost-BUDGETED batches: bounding peak memory rather than batch size
# --------------------------------------------------------------------

def _peak(costs, batches):
    """batch_size * max(cost in batch) -- the quantity peak backward memory
    is proportional to, because the masked loop allocates full-batch tensors
    on every one of its max-count iterations."""
    costs = np.asarray(costs, dtype=np.float64)
    return max((len(b) * costs[list(b)].max() for b in batches), default=0.0)


def test_budgeting_bounds_peak_memory_where_fixed_size_does_not():
    """
    THE OOM, reproduced as arithmetic. A fixed batch_size has a peak set by
    whichever window happens to be deepest, so raising max_substeps 256->512
    raised the peak even though batch_size was halved 2048->1024. A budget
    makes the peak a constant of the run.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    dt, counts = _population(n=8000, sigma=0.5)
    B = 1024
    rng = np.random.default_rng(2)
    perm = rng.permutation(len(counts))
    fixed = [perm[i:i + B] for i in range(0, len(counts), B)]
    budgeted = list(BudgetedBatchSampler(counts, B, shuffle=False))

    assert _peak(counts, budgeted) < _peak(counts, fixed) / 3, (
        f"budgeted peak {_peak(counts, budgeted):.0f} vs fixed "
        f"{_peak(counts, fixed):.0f} -- the budget is not bounding anything"
    )


def test_the_peak_never_exceeds_the_budget_except_for_a_single_oversized_window():
    """
    The bound has one unavoidable escape: a window costing more than the whole
    budget still has to run, as a batch of one. That is reported by peak_cost
    rather than hidden, because it means max_substeps is set beyond what one
    sample of memory allows.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=3000)
    s = BudgetedBatchSampler(counts, 256, budget=5000.0, shuffle=False)
    for batch in s:
        cost = len(batch) * counts[list(batch)].max()
        assert cost <= 5000.0 or len(batch) == 1, (
            f"a batch of {len(batch)} costs {cost:.0f}, over the 5000 budget"
        )
    assert s.peak_cost() == pytest.approx(_peak(counts, list(s)))


def test_the_default_budget_gives_a_typical_batch_the_requested_size():
    """
    The auto budget is batch_size x the population MEDIAN cost, so the
    caller's batch_size keeps the meaning it had before budgeting existed:
    a median-cost batch has exactly that many windows, and only deeper ones
    shrink.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=5000)
    B = 512
    s = BudgetedBatchSampler(counts, B, shuffle=False)
    assert s.budget == pytest.approx(B * np.median(counts))
    sizes = [len(b) for b in s]
    assert max(sizes) == B, "no batch reached the requested size"
    assert min(sizes) < B / 2, "nothing shrank; the population has no depth spread"


def test_the_budget_follows_the_population_as_f_theta_sharpens():
    """
    Costs grow as f_theta sharpens. A budget frozen at the initial median
    would shrink every batch over a long run, quietly changing the
    optimisation; the auto budget is recomputed on refresh so the typical
    batch keeps its size.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=4000)
    s = BudgetedBatchSampler(counts, 512, shuffle=False)
    before = s.budget
    s.update_costs(counts * 3.0)          # f_theta three times sharper
    assert s.budget == pytest.approx(3.0 * before)
    assert max(len(b) for b in s) == 512, "the typical batch lost its size"


def test_an_explicit_budget_is_not_rescaled():
    """A caller who set a budget did so because that is what fits in VRAM;
    it must not drift with the population."""
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=2000)
    s = BudgetedBatchSampler(counts, 512, budget=9000.0, shuffle=False)
    s.update_costs(counts * 5.0)
    assert s.budget == 9000.0


def test_budgeted_batches_still_visit_every_window_once():
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=2500)
    s = BudgetedBatchSampler(counts, 256, shuffle=True)
    for _ in range(3):
        seen = [i for batch in s for i in batch]
        assert sorted(seen) == list(range(len(counts)))


def test_budgeted_len_matches_what_is_yielded():
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=2500)
    for B in (64, 256, 4096):
        s = BudgetedBatchSampler(counts, B, shuffle=False)
        assert len(s) == len(list(s))


def test_train_lds_uses_the_budgeted_sampler():
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "BudgetedBatchSampler(" in src, (
        "train_lds still batches by fixed size under alpha, so peak memory is "
        "set by the deepest window in each batch"
    )
    import inspect

    from training.train_lds import train_lds
    assert "batch_cost_budget" in inspect.signature(train_lds).parameters


# --------------------------------------------------------------------
# The budget report: is the budget actually doing anything?
# --------------------------------------------------------------------

def test_the_report_says_when_the_budget_never_binds():
    """
    THE TRAP THIS EXISTS FOR. A budget larger than batch_size x max_cost is
    inert: batch_size caps every batch first, peak memory is still set by the
    deepest window, and raising max_substeps raises it. That was a real
    configuration -- budget=300000 with batch_size=1024 and max_substeps=256,
    where a batch would have needed 1172 windows to reach the budget -- and it
    looked safe to raise max_substeps precisely because the budget's presence
    suggested the peak was controlled.
    """
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    _, counts = _population(n=8000)
    huge = 1e9
    s = BudgetedBatchSampler(counts, 512, budget=huge, shuffle=False)
    report = budget_report(s, 512, 2)
    assert "NOT cutting any batch" in report, report
    assert "WILL raise it" in report


def test_the_report_says_when_the_budget_holds():
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    _, counts = _population(n=8000)
    tight = 4 * float(np.median(counts))          # ~4 windows per median batch
    s = BudgetedBatchSampler(counts, 512, budget=tight, shuffle=False)
    report = budget_report(s, 512, 2)
    assert "budget IS cutting batches" in report, report
    assert "rather than more memory" in report


def test_the_final_short_batch_is_not_mistaken_for_a_budget_cut():
    """
    The last batch is short because 37035 is not a multiple of 1024, not
    because the budget cut it. Counting it would report the budget as binding
    on every run, which is the false-reassurance direction -- the one that
    lets someone raise max_substeps into an OOM.
    """
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    _, counts = _population(n=1000 + 7)           # deliberately not a multiple
    s = BudgetedBatchSampler(counts, 500, budget=1e9, shuffle=False)
    sizes = [len(b) for b in s]
    assert sizes[-1] < 500 and all(x == 500 for x in sizes[:-1]), sizes
    assert "NOT cutting any batch" in budget_report(s, 500, 2)


def test_the_report_gives_the_bytes_per_sample_substep_constant():
    """
    The constant that turns budget-setting from arithmetic into a
    measurement: budget = usable_VRAM / (n_rollout_steps x bytes_per_unit).
    """
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    _, counts = _population(n=4000)
    s = BudgetedBatchSampler(counts, 512, shuffle=False)
    peak_bytes = 2.0 * 2 ** 30
    report = budget_report(s, 512, 2, peak_bytes=peak_bytes)
    expected = peak_bytes / (s.peak_cost() * 2)
    assert f"{expected:.0f} bytes per RETAINED sample-substep" in report, report
    assert "2.00 GiB" in report
    # and it is omitted when nothing was measured
    assert "bytes per RETAINED sample-substep" not in budget_report(s, 512, 2)


def test_the_unbucketed_peak_is_reported_for_comparison():
    """
    What the same batch_size would cost WITHOUT bucketing -- a random batch of
    hundreds contains a near-deepest window with near-certainty, so that peak
    really is batch_size x max(cost). It is the number that says what
    bucketing bought.
    """
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    _, counts = _population(n=6000)
    s = BudgetedBatchSampler(counts, 512, shuffle=False)
    report = budget_report(s, 512, 2)
    assert f"{512 * counts.max():.0f}" in report, report


def test_train_lds_reports_the_budget_and_resets_the_peak_counter():
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "budget_report(" in src
    assert "reset_peak_memory_stats" in src, (
        "the measured peak would include the dataset build and the cost "
        "estimate, not the training footprint the constant is meant to describe"
    )
    assert "max_memory_allocated" in src


# --------------------------------------------------------------------
# Review findings: augmentation, epoch hygiene, empty edges
# --------------------------------------------------------------------

def test_an_augmented_dataset_is_refused_loudly():
    """
    MicrostructureEvolutionDataset multiplies __len__ by the augmentation
    variants (x32) while _index holds the BASE windows -- exactly the range
    the samplers cover. Feeding a bucketed sampler to an augmented dataset
    would train on 1/32 of it, every epoch, with a plausible batch count and
    a correct-looking loss curve. Nothing downstream can detect it, so the
    estimator refuses upfront.

    train_lds never augments stage 3, which is why this has not bitten --
    "safe by accident", pinned into "safe by construction".
    """
    ds = _dataset()
    ds.__class__ = type("_Augmented", (_MiniEvolutionDataset,),
                         {"__len__": lambda self: 32 * len(self._index)})
    with pytest.raises(ValueError, match="augmented"):
        estimate_window_costs(ds, _Field(), 0.1, 256, torch.device("cpu"))

    plain = _dataset()
    got = estimate_window_costs(plain, _Field(), 0.1, 256, torch.device("cpu"))
    assert len(got) == len(plain._index)


def test_the_report_does_not_advance_the_shuffle_epoch():
    """
    Iterating a shuffle=True sampler advances its epoch counter. The report
    used to iterate, so startup + the epoch-1 measurement consumed two epochs
    of shuffles before training took a step -- every subsequent epoch's batch
    order silently shifted relative to a run without the report. Found in
    review; the report now reads _batches directly.
    """
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    _, counts = _population(n=2000)
    s = BudgetedBatchSampler(counts, 128, shuffle=True, seed=1)
    before = s._epoch
    budget_report(s, 128, 2)
    budget_report(s, 128, 2, peak_bytes=1e9)
    assert s._epoch == before, "the report consumed shuffle epochs"


def test_empty_costs_still_produce_a_wellformed_sampler_and_report():
    """An empty dataset should fail at dataset construction with its own
    message -- not two calls later with an AttributeError from the report."""
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    s = BudgetedBatchSampler(np.zeros(0), 64, shuffle=False)
    assert len(s) == 0 and list(s) == []
    assert s.peak_cost() == 0.0
    # The budget attribute must exist even on the empty path -- it is public
    # surface (read by the report on non-empty samplers and by callers), and
    # "exists after __init__, always" is a simpler invariant than "exists
    # unless the dataset was empty". Asserted directly because the report's
    # own empty-case early return means it never reads it, so this line is
    # the only thing keeping the invariant true.
    assert s.budget == 0.0
    assert BudgetedBatchSampler(np.zeros(0), 64, budget=9.0, shuffle=False).budget == 9.0
    report = budget_report(s, 64, 2, peak_bytes=1e9)
    assert "EMPTY" in report


# --------------------------------------------------------------------
# Retained depth, and holding the step count across a refresh
# --------------------------------------------------------------------

def test_the_budget_bounds_retained_depth_not_the_raw_count():
    """
    Under truncated BPTT the graph is detached every k sub-steps, so a window
    needing 900 sub-steps retains only ~k -- and memory follows the RETAINED
    depth. Budgeting on the raw count bounds a quantity that no longer drives
    memory: observed as peak staying at 0.82 GiB whether the budget was 50000
    or 100000, while the bytes-per-sample-substep constant wandered
    33764 -> 8961 -> 6438 -> 13493 across four runs.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=8000)
    counts = counts * 20                       # deep windows, far past k
    k = 16
    s = BudgetedBatchSampler(counts, 1024, budget=5000.0, shuffle=False,
                              truncate_bptt=k)
    for batch in s:
        retained = np.minimum(counts[list(batch)], k).max()
        assert len(batch) * retained <= 5000.0 or len(batch) == 1
    # and the raw-count sampler would have cut far harder for the same budget
    raw = BudgetedBatchSampler(counts, 1024, budget=5000.0, shuffle=False)
    assert len(raw) > len(s), (
        "ignoring truncation did not over-cut -- the fixture has no windows "
        "deeper than truncate_bptt, so the test proves nothing"
    )


def test_compute_and_memory_are_reported_separately():
    """
    Every sub-step is evaluated forward whether or not its graph is kept, so
    compute scales with the FULL count while memory scales with the retained
    depth. Conflating them is what made the constant wander.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=4000)
    counts = counts * 20
    s = BudgetedBatchSampler(counts, 512, shuffle=False, truncate_bptt=16)
    # Computed independently from the FULL counts, so a compute_cost that
    # secretly used the retained depth cannot match. My first version compared
    # compute_cost against peak_cost * len(s), which both quantities satisfy.
    expected = sum(len(b) * counts[list(b)].max() for b in s)
    assert s.compute_cost() == pytest.approx(expected), (
        f"compute_cost {s.compute_cost():.4g} != {expected:.4g} from the full "
        f"counts -- it is using the retained depth, which understates the "
        f"forward work by the truncation factor"
    )
    retained_version = sum(len(b) * min(counts[list(b)].max(), 16) for b in s)
    assert expected > 5 * retained_version, "fixture too shallow to separate them"


def test_a_refresh_holds_the_batch_count_under_the_auto_budget():
    """
    THE EPOCH-26 INCIDENT. Every batch is one optimizer step and grad_clip
    normalises each step, so batches-per-epoch IS the per-epoch learning rate.
    A refresh at epoch 26 re-batched a sharpened population, the count jumped,
    and the run -- descending cleanly for 25 epochs -- destabilised
    immediately, reporting more skips per epoch than it previously had
    batches.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=20000)
    # Sharpening must push a substantial fraction of the population THROUGH
    # the truncate_bptt ceiling for the retained depth -- and hence the batch
    # count -- to move materially. 4x left most windows still under 64 and the
    # unheld count barely budged; 12x clears it.
    sharpened = counts * 40.0
    # First establish that WITHOUT holding, the count moves materially --
    # otherwise the held case proves nothing. (My first version used a 2.5x
    # sharpening on a population where the unheld count barely moved, so the
    # test passed against a mutation that removed the holding entirely.)
    loose = BudgetedBatchSampler(counts, 2048, shuffle=False, truncate_bptt=64)
    unheld_before = len(loose)
    loose.update_costs(sharpened, hold_batch_count=False)
    assert abs(len(loose) / unheld_before - 1.0) > 0.25, (
        f"the unheld count moved only {unheld_before} -> {len(loose)}; this "
        f"fixture cannot detect whether holding works"
    )

    s = BudgetedBatchSampler(counts, 2048, shuffle=False, truncate_bptt=64)
    before = len(s)
    s.update_costs(sharpened)
    assert abs(len(s) / before - 1.0) < 0.15, (
        f"batch count moved {before} -> {len(s)} despite the auto budget, so "
        f"the effective learning rate changed mid-run"
    )
    assert s.budget > 0


def test_an_explicit_budget_is_never_rescaled_but_the_change_is_announced():
    """
    An explicit budget is a MEMORY limit: rescaling it to hold the step count
    would trade an OOM for a stable lr, which is the wrong way round. So the
    count moves -- and that must be announced, not left for the reader to
    infer from a skip count that suddenly exceeds the batch count.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=20000)
    s = BudgetedBatchSampler(counts, 2048, budget=4000.0, shuffle=False,
                              truncate_bptt=64)
    before, budget_before = len(s), s.budget
    s.update_costs(counts * 3.0)
    assert s.budget == budget_before, "an explicit memory limit was rescaled"
    assert len(s) != before
    assert "effective learning-rate change" in s.last_refresh_note
    assert "bucket_refresh_epochs=0" in s.last_refresh_note


def test_no_note_when_the_refresh_barely_moves_the_count():
    """The note must mean something -- firing on every refresh would train the
    reader to ignore it."""
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=20000)
    s = BudgetedBatchSampler(counts, 2048, shuffle=False, truncate_bptt=64)
    s.update_costs(counts * 1.02)
    assert s.last_refresh_note == "", s.last_refresh_note


def test_train_lds_passes_truncation_to_the_sampler_and_prints_the_note():
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "truncate_bptt=truncate_bptt)" in src, (
        "the sampler budgets on raw sub-step counts, which truncation has made "
        "non-predictive of memory"
    )
    assert "last_refresh_note" in src, "the mid-run lr change would go unreported"


def test_the_measured_report_is_a_strict_extension_of_the_dry_one():
    """
    The epoch-1 report prints only its NEW lines, computed as the suffix of
    the measured report past the dry one. That slicing is only valid if the
    dry report is a strict PREFIX -- if a measured run ever reworded an
    earlier line, the suffix would start mid-sentence and the log would show
    a fragment.

    Written because the original printed the whole report twice: four
    paragraphs repeated to add one number, with the epoch lines sandwiched
    between the copies.
    """
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    _, counts = _population(n=6000)
    for k in (None, 64):
        s = BudgetedBatchSampler(counts, 512, shuffle=False, truncate_bptt=k)
        dry = budget_report(s, 512, 2)
        measured = budget_report(s, 512, 2, peak_bytes=1.5 * 2 ** 30)
        assert measured.startswith(dry), (
            "the measured report is not an extension of the dry one, so "
            "printing the suffix would emit a fragment"
        )
        assert measured[len(dry):].strip(), "no new lines to print"


def test_the_column_header_is_printed_after_the_batching_report():
    """
    The header labels the epoch lines, so anything between it and the first
    epoch line reads as a caption for the wrong thing. The batching report is
    built after the sampler exists, which is after the "Starting N epochs"
    banner -- so the header has to come later still.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    header = src.index("train  (1step)   valid  (1step)     ema")
    report = src.index("print(budget_report(_bucket_sampler")
    assert report < header, (
        "the column header is emitted before the batching report, so the "
        "report lands under a legend it does not belong to"
    )
    # exactly one full-report print remains
    assert src.count("print(budget_report(") == 1
    # and the epoch-1 report prints only the SUFFIX past the dry one. Counting
    # print(budget_report( alone missed this: a mutation that assigned the
    # full report to a local and printed that passed, re-emitting four
    # paragraphs to add one number.
    assert "_full[len(_dry):]" in src, (
        "the epoch-1 report does not slice off the already-printed prefix, so "
        "the whole report is emitted a second time"
    )


# --------------------------------------------------------------------
# Retained memory under PER-SAMPLE truncation: the span matters
# --------------------------------------------------------------------

def _retained_depth(counts, k):
    """Arrival segments x k -- what a batch actually retains per sample.

    Bounded three ways: one arrival segment per sample (n*k), the segments the
    counts straddle (span + k), and the loop length itself (hi).
    """
    counts = np.asarray(counts)
    lo, hi = float(counts.min()), float(counts.max())
    return min(hi, len(counts) * float(k), (hi - lo) + float(k))


def test_the_budget_prices_the_count_span_not_just_the_depth():
    """
    THE VRAM REGRESSION. Per-sample truncation retains each sample's FINAL
    segment, and because the updates are full-batch tensor ops, every segment
    in which some sample arrives is retained for the whole batch. So a batch
    costs

        len x (arrival segments) x k   ~   len x (span + k)

    not len x k. Budgeting on retained depth alone let a batch spanning
    counts 300-900 through at k=64 -- ~9 arrival segments, measured at 35x
    the batch-wide graph on a (100,300,600,900) batch.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=20000)
    counts = counts * 30                       # deep, wide spread
    k, budget = 64, 50_000.0
    s = BudgetedBatchSampler(counts, 4096, budget=budget, shuffle=False,
                              truncate_bptt=k)
    for batch in s:
        cost = len(batch) * _retained_depth(counts[list(batch)], k)
        assert cost <= budget * 1.001 or len(batch) == 1, (
            f"a batch of {len(batch)} spanning "
            f"{counts[list(batch)].min():.0f}-{counts[list(batch)].max():.0f} "
            f"retains {cost:.0f}, over the {budget:.0f} budget"
        )


def test_pricing_the_span_beats_budgeting_on_retained_depth_alone():
    """The improvement is real on a population matching the max_dt=2000 run --
    not just on a constructed worst case."""
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=20000)
    counts = counts * 30
    k = 64
    retained = np.minimum(counts, k)
    order = np.lexsort((counts, retained))     # the previous rule
    old, cur = [], []
    for i in order:
        if cur and ((len(cur) + 1) * retained[i] > 50_000 or len(cur) >= 4096):
            old.append(np.array(cur))
            cur = []
        cur.append(i)
    if cur:
        old.append(np.array(cur))
    old_worst = max(len(b) * _retained_depth(counts[b], k) for b in old)

    s = BudgetedBatchSampler(counts, 4096, budget=50_000.0, shuffle=False,
                              truncate_bptt=k)
    new_worst = max(len(b) * _retained_depth(counts[list(b)], k) for b in s._batches)
    assert new_worst < old_worst / 1.3, (
        f"worst retained {new_worst:.0f} vs {old_worst:.0f} under the old rule "
        f"-- the span is not being priced"
    )


def test_a_wide_span_is_allowed_when_the_batch_is_small():
    """
    A fixed span CAP was the first attempt and it was the wrong shape: it
    split the deep tail into 3-window batches, each still taking a full
    clipped optimizer step. Few samples x many segments is cheap, so the
    span must be priced, not capped.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=20000)
    counts = counts * 30
    s = BudgetedBatchSampler(counts, 4096, budget=50_000.0, shuffle=False,
                              truncate_bptt=64)
    spans = [counts[list(b)].max() - counts[list(b)].min() for b in s._batches]
    assert max(spans) > 64, "no batch exceeds a span of k, so this is a cap not a price"
    # Excluding the FINAL batch, which is short because the population is not
    # a multiple of anything -- the same remainder-vs-cut distinction the
    # budget-binding test has to make. Counting it here would have this test
    # pass or fail on the population size.
    interior = [len(b) for b in s._batches[:-1]]
    assert min(interior) > 25, (
        f"smallest interior batch is {min(interior)} windows (bar is 25, and a "
        f"fixed span cap gave 3) -- the deep tail has been shattered into "
        f"batches too small to give a meaningful gradient direction. Full: "
        f"has been shattered into batches too small to give a meaningful "
        f"gradient direction"
    )


def test_without_truncation_the_depth_is_just_the_deepest_window():
    """No truncation means every sample keeps its whole history, so span is
    irrelevant and the batch is priced by its deepest member."""
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=5000)
    s = BudgetedBatchSampler(counts, 512, budget=20_000.0, shuffle=False)
    for batch in s:
        cost = len(batch) * counts[list(batch)].max()
        assert cost <= 20_000.0 * 1.001 or len(batch) == 1


def test_peak_cost_uses_the_same_model_as_the_fill():
    """Reporting retained-max while budgeting on span would understate exactly
    the batches that caused the regression."""
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=8000)
    counts = counts * 30
    s = BudgetedBatchSampler(counts, 4096, budget=50_000.0, shuffle=False,
                              truncate_bptt=64)
    expected = max(len(b) * _retained_depth(counts[list(b)], 64)
                    for b in s._batches)
    assert s.peak_cost() == pytest.approx(expected)


def test_a_sparse_deep_tail_is_not_shattered_into_tiny_batches():
    """
    The n bound, isolated. When the deepest windows are few and far apart --
    counts 2000, 6000, 20000 -- the span term alone charges a 2-window batch
    as if it retained 18000 steps, when it can retain at most TWO arrival
    segments = 2k. Without the n bound those windows each become their own
    batch, and a 1-window batch still takes a full clipped optimizer step.

    A dense tail cannot show this (the gaps are too small for the span term to
    dominate), which is why the general-population test passed against the
    mutation that removed the bound.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    counts = np.concatenate([
        np.full(4000, 50.0),                       # the bulk
        np.array([2000., 6000., 12000., 20000.]),  # a sparse, deep tail
    ])
    s = BudgetedBatchSampler(counts, 4096, budget=20_000.0, shuffle=False,
                              truncate_bptt=64)
    tail_batches = [b for b in s._batches
                    if counts[list(b)].max() >= 2000.0]
    assert tail_batches, "the deep windows vanished"
    assert max(len(b) for b in tail_batches) >= 4, (
        f"the four deep windows were split into batches of "
        f"{[len(b) for b in tail_batches]} -- the span term is charging for "
        f"arrival segments the batch is too small to have"
    )


def test_the_report_separates_reserved_from_allocated():
    """
    THE QUESTION THE ALLOCATED PEAK CANNOT ANSWER. Every run reported ~2.5 GiB
    allocated while the card showed 7.7 GB in use -- a 3x gap that no change
    to the batching model can close, because the difference is cached free
    blocks the allocator is holding, not live tensors.

    Reporting only the allocated peak sent two rounds of work at the wrong
    target: the per-sample retained-depth model was real but moved 2.5 GiB,
    not 7.7.
    """
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    _, counts = _population(n=6000)
    s = BudgetedBatchSampler(counts, 512, shuffle=False, truncate_bptt=64)
    report = budget_report(s, 512, 2, peak_bytes=2.5 * 2 ** 30,
                            reserved_bytes=7.2 * 2 ** 30)
    assert "RESERVED 7.20 GiB" in report
    assert "2.50 GiB of live tensors" in report
    assert "FRAGMENTATION" in report, "the diagnosis is not drawn"
    assert "expandable_segments" in report, "no actionable remedy given"

    # and it stays quiet when reservation is close to what is live
    tight = budget_report(s, 512, 2, peak_bytes=2.5 * 2 ** 30,
                           reserved_bytes=2.8 * 2 ** 30)
    assert "RESERVED" in tight, "the numbers should still be stated"
    assert "FRAGMENTATION" not in tight, (
        "the diagnosis fires when reservation is merely normal, so it would "
        "be ignored when it matters"
    )
    # omitted entirely when not measured
    assert "RESERVED" not in budget_report(s, 512, 2, peak_bytes=2.5 * 2 ** 30)


def test_train_lds_measures_reserved_as_well_as_allocated():
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "max_memory_reserved" in src, (
        "only the allocated peak is measured, so a fragmentation-dominated "
        "run reports 2.5 GiB while the card shows 7.7"
    )
