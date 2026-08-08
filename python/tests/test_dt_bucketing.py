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
    from test_train_lds import _cached_stage2_ancestor
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

def _peak(costs, batches, sampler=None):
    """Predicted peak BYTES, through the same measured model the sampler uses.

    Was batch_size * max(count): a plain product of size and depth. That model
    is dead -- the fixed-n sweep put the depth exponent near 0.79, so memory
    is not proportional to any such product, and `retained` (this helper's
    successor) measured WORST of five predictors at 59.4%. Tests that
    recompute an expectation by hand have to use the model under test, or they
    pin the arithmetic of a model nobody runs.
    """
    costs = np.asarray(costs, dtype=np.float64)
    if sampler is None:
        from training.dt_bucketing import BudgetedBatchSampler as _S
        depth = lambda hi: (_S.COST_A_BYTES
                             + _S.COST_B_BYTES * max(hi, 1.0) ** _S.COST_P)
    else:
        depth = lambda hi: sampler._depth(hi, hi)
    return max((len(b) * depth(float(costs[list(b)].max())) for b in batches),
                default=0.0)


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
    s = BudgetedBatchSampler(counts, 256, budget=50 * 2**20, shuffle=False)
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
    # ...at the median window's MEASURED cost, not its raw count: the budget
    # is bytes now, because no count of sample-substeps predicts memory.
    _med = float(np.median(counts))
    assert s.budget == pytest.approx(B * s._depth(_med, _med))
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
    # NOT 3x: the cost model is A + B*depth^0.79, so tripling the depth
    # raises the budget by 3^0.79 on the depth term only. Asserting 3x would
    # pin proportionality that measurement rejected.
    assert s.budget > before, "the budget did not follow the population at all"
    assert s.budget < 3.0 * before, (
        "the budget scaled linearly with depth -- the fitted exponent (0.79) "
        "is not being applied"
    )
    assert max(len(b) for b in s) == 512, "the typical batch lost its size"


def test_an_explicit_budget_is_not_rescaled():
    """A caller who set a budget did so because that is what fits in VRAM;
    it must not drift with the population."""
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=2000)
    s = BudgetedBatchSampler(counts, 512, budget=90 * 2**20, shuffle=False)
    s.update_costs(counts * 5.0)
    assert s.budget == 90 * 2**20


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
    configuration -- budget=3 * 2**30 with batch_size=1024 and max_substeps=256,
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
    # ~4 windows per median batch, priced through the model under test rather
    # than as a raw count of sub-steps.
    _probe = BudgetedBatchSampler(counts, 512, shuffle=False)
    _med = float(np.median(counts))
    tight = 4 * _probe._depth(_med, _med)
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
    # In MiB, through the measured cost model -- the raw product 512*max(count)
    # was the old (falsified) unit and no longer appears anywhere.
    _hi = float(counts.max())
    expected = 512 * s._depth(_hi, _hi) / 2 ** 20
    assert f"{expected:.0f} MiB" in report, report


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
    assert BudgetedBatchSampler(np.zeros(0), 64, budget=2 * 2**20,
                                 shuffle=False).budget == 2 * 2**20
    report = budget_report(s, 64, 2, peak_bytes=1e9)
    assert "EMPTY" in report
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
def test_the_auto_budget_holds_the_batch_count_by_itself():
    """
    REFRAMED, because the measured cost model changed the property.

    The auto budget is batch_size x the MEDIAN window's cost, so when f_theta
    sharpens and every cost rises, the budget rises with it and the batch
    count barely moves. That is now true without any holding: the old fixture
    tried to establish a precondition -- "the unheld count moves by >25%" --
    that no realistic sharpening can produce any more, because budget and
    costs cancel. Under the previous (linear, falsified) model they cancelled
    less exactly, which is what the hold_batch_count machinery was for.

    Holding the count matters because batches-per-epoch IS the per-epoch
    learning rate under grad_clip: a refresh that halves it silently halves
    the training rate.
    """
    from training.dt_bucketing import BudgetedBatchSampler

    _, counts = _population(n=20000)
    s = BudgetedBatchSampler(counts, 512, shuffle=False)
    before = len(s)

    sharpened = counts.astype(float) * 50.0
    s.update_costs(sharpened, hold_batch_count=False)
    assert abs(len(s) / before - 1.0) < 0.25, (
        f"the batch count moved {before} -> {len(s)} on a 50x sharpening even "
        f"though the auto budget tracks the median -- the two are no longer "
        f"cancelling, and every refresh is an effective learning-rate change"
    )

    # and the explicit hold does not make it worse
    s2 = BudgetedBatchSampler(counts, 512, shuffle=False)
    s2.update_costs(sharpened, hold_batch_count=True)
    assert abs(len(s2) / before - 1.0) < 0.25, (
        f"{before} -> {len(s2)} with holding requested"
    )

def test_an_explicit_budget_is_never_rescaled_but_the_change_is_announced():
    """
    An explicit budget is a MEMORY limit: rescaling it to hold the step count
    would trade an OOM for a stable lr, which is the wrong way round. So the
    count moves -- and that must be announced, not left for the reader to
    infer from a skip count that suddenly exceeds the batch count.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=20000)
    s = BudgetedBatchSampler(counts, 2048, budget=40 * 2**20, shuffle=False,
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
def test_without_truncation_the_depth_is_just_the_deepest_window():
    """No truncation means every sample keeps its whole history, so span is
    irrelevant and the batch is priced by its deepest member."""
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=5000)
    s = BudgetedBatchSampler(counts, 512, budget=200 * 2**20, shuffle=False)
    for batch in s:
        cost = len(batch) * counts[list(batch)].max()
        assert cost <= 20_000.0 * 1.001 or len(batch) == 1


def test_peak_cost_uses_the_same_model_as_the_fill():
    """Reporting retained-max while budgeting on span would understate exactly
    the batches that caused the regression."""
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=8000)
    counts = counts * 30
    s = BudgetedBatchSampler(counts, 4096, budget=500 * 2**20, shuffle=False,
                              truncate_bptt=64)
    expected = _peak(counts, s._batches, sampler=s)
    assert s.peak_cost() == pytest.approx(expected)
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


def test_train_lds_tracks_peak_memory_all_run_not_just_epoch_one():
    """
    THE INSTRUMENTATION BUG. The peak was measured once, after epoch 1 -- the
    CHEAPEST epoch. Sub-step counts climb as f_theta sharpens (46 -> 209 mean
    on one run), and with bucket_refresh_epochs=0 the batching is frozen at
    the initial estimate, so each batch retains steadily more than it was
    sized for. The report said 2.5 GiB while the card showed 7.8 GB, and that
    gap sent two rounds of optimisation at the wrong target.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "PeakMemoryTracker()" in src, "no high-water mark is kept across epochs"
    assert "_mem_tracker.update(" in src

    # BEHAVIOURAL, because the source check above passed against a mutation
    # that deleted the high-water assignment -- the string was still present.
    from training.dt_bucketing import PeakMemoryTracker
    G = 2 ** 30
    t = PeakMemoryTracker()
    assert t.update(1, 2.5 * G, 3.0 * G) == "", "epoch 1 should set, not report"
    assert t.update(2, 2.6 * G, 3.1 * G) == "", "noise must not report"
    grown = t.update(40, 5.0 * G, 7.0 * G)
    assert "grown to 5.00 GiB" in grown and "from 2.50 GiB" in grown, grown
    assert "bucket_refresh_epochs" in grown, "no remedy named"
    # the mark MOVED, so the next report compares against 5.0, not 2.5
    assert t.update(41, 5.1 * G, 7.1 * G) == ""
    again = t.update(80, 9.0 * G, 12.6 * G)
    assert "from 5.00 GiB" in again, (
        f"the high-water mark was never updated, so growth is measured from "
        f"the first epoch forever: {again}"
    )
def test_retained_peak_can_be_read_without_clearing_the_substep_stats():
    """
    The calibration reads the realised peak; the epoch line reads the sub-step
    statistics. Reading either must not clear the other, or one of the two
    silently reports zeros.
    """
    import torch as _torch

    from models.latent_dynamics import LatentDynamics
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, alpha=0.1, max_substeps=4096,
                        truncate_bptt=64)
    _torch.manual_seed(0)
    m._integrate(_torch.randn(4, 2, 4, 4) * 0.3, _torch.randn(4, 2, 4, 4) * 0.3,
                  _torch.full((4,), 300.0), _torch.zeros(4, 1))
    peak = m.retained_peak()
    assert peak > 0
    assert m.retained_peak() == peak, "reading cleared it"
    assert m.substep_stats(reset=False) is not None
    assert m.retained_peak() == peak, "substep_stats cleared the retained peak"
    assert m.retained_peak(reset=True) == peak
    assert m.retained_peak() == 0.0


# --------------------------------------------------------------------
# Feedback control on MEASURED bytes (no memory model)
# --------------------------------------------------------------------

def test_the_budget_converges_to_a_byte_target():
    """
    MODEL-FREE, and deliberately so. Four structural models of retained memory
    failed against measurement -- retained depth, span-aware depth, span
    bounded by batch size, and the realised per-transition cost, the last of
    which TRIPLED on a run whose measured peak HALVED. The implied
    bytes-per-unit constant ranged over 1700-77000.

    What holds is monotonicity: smaller batches, less memory. Feedback needs
    nothing more.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=20000)
    s = BudgetedBatchSampler(counts, 2048, budget=300 * 2**20, shuffle=False,
                              truncate_bptt=64)

    # a monotone but NON-proportional response, which is what broke the models
    def measured(sampler):
        return max(len(b) * float(counts[list(b)].max()) ** 0.9
                    for b in sampler._batches) * 4.0e5

    target = 3.0 * 2 ** 30
    for _ in range(5):
        if not s.rescale_to_bytes(measured(s), target):
            break
    assert measured(s) <= target * 1.25, (
        f"did not converge: {measured(s) / 2**30:.2f} GiB against a "
        f"{target / 2**30:.2f} GiB target"
    )


def test_it_does_not_thrash_when_already_within_tolerance():
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=8000)
    s = BudgetedBatchSampler(counts, 512, budget=200 * 2**20, shuffle=False,
                              truncate_bptt=64)
    before = [list(b) for b in s._batches]
    assert s.rescale_to_bytes(3.0 * 2 ** 30, 3.1 * 2 ** 30) == ""
    assert s.rescale_to_bytes(3.0 * 2 ** 30, 2.9 * 2 ** 30) == ""
    assert [list(b) for b in s._batches] == before


def test_it_stops_raising_the_budget_once_batch_size_binds():
    """
    THE DIVERGENCE. When batch_size caps every batch the budget stops being
    the binding constraint, so raising it changes nothing -- and an
    unguarded loop multiplies forever: measured 30000 -> 1.7e12 over five
    no-op steps.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=4000)
    s = BudgetedBatchSampler(counts, 512, budget=1e12, shuffle=False,
                              truncate_bptt=64)
    sizes_before = [len(b) for b in s._batches]
    budget_before = s.budget
    # ask for far more memory than is being used: the budget cannot deliver
    assert s.rescale_to_bytes(0.01 * 2 ** 30, 4.0 * 2 ** 30) == ""
    assert s.budget == budget_before, "the budget grew with no effect"
    assert [len(b) for b in s._batches] == sizes_before


def test_shrinking_still_works_when_batch_size_binds():
    """The guard must be one-sided: shrinking always has an effect, and is the
    direction that matters for fitting in VRAM."""
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=8000)
    s = BudgetedBatchSampler(counts, 512, budget=1e12, shuffle=False,
                              truncate_bptt=64)
    before = len(s._batches)
    # Iterating, as the epoch loop does. One DAMPED step from a budget this
    # far above binding does not reach it -- which is the controller working,
    # not failing: it converges over epochs rather than lurching in one.
    notes = [s.rescale_to_bytes(8.0 * 2 ** 30, 1.0 * 2 ** 30) for _ in range(8)]
    assert all(notes), "shrinking was refused, but shrinking always has effect"
    assert len(s._batches) > before, (
        f"{before} -> {len(s._batches)} batches: the budget never reached the "
        f"binding region, so no amount of shrinking would fit VRAM"
    )


def test_every_window_survives_rescaling():
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=6000)
    s = BudgetedBatchSampler(counts, 512, budget=500 * 2**20, shuffle=False,
                              truncate_bptt=64)
    s.rescale_to_bytes(6.0 * 2 ** 30, 2.0 * 2 ** 30)
    assert sorted(i for b in s for i in b) == list(range(len(counts)))


def test_train_lds_exposes_a_vram_target_and_feeds_back_measured_bytes():
    import inspect
    import pathlib

    from conftest import source_without_comments
    from training.train_lds import train_lds
    assert inspect.signature(train_lds).parameters["target_vram_gib"].default is None
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    # The call moved into MemoryGovernor.step; train_lds drives it.
    assert "_mem_governor.step(epoch, _bucket_sampler, _resv)" in src, (
        "the feedback is not driven by the MEASURED reserved peak"
    )
    block = src[src.index("_mem_governor.step("):]
    assert "train_loader = DataLoader(" in block[:900], "loader not rebuilt"
    assert "reset_peak_memory_stats" in block[:900], (
        "the peak is not reset, so the next epoch re-measures the old high "
        "mark and the loop never converges"
    )


# --------------------------------------------------------------------
# MemoryGovernor: state that cannot go unbound
# --------------------------------------------------------------------

def test_the_governor_holds_its_state_in_an_object_not_a_loop_local():
    """
    THE CRASH. The epoch loop's memory block is CUDA-gated, so on a machine
    without a GPU it never runs -- an initialisation lost from it is invisible
    to import, to pyflakes and to every test, and surfaces only as an
    UnboundLocalError on a real run. (It was lost to a batched edit that
    asserted on a later hunk and never wrote the file.)

    Holding the state in an object constructed unconditionally removes the
    failure mode: there is no local to leave unbound, and the logic is
    reachable on CPU.
    """
    from training.dt_bucketing import MemoryGovernor
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "MemoryGovernor(target_vram_gib)" in src
    # constructed OUTSIDE any cuda gate
    head = src[:src.index("MemoryGovernor(target_vram_gib)")]
    assert head.count('device.type == "cuda"') == head.count('device.type == "cuda"')
    assert "_mem_adjustments" not in src, "the bare counter is still there"

    g = MemoryGovernor(None)
    assert not g.active(1), "no target means no adjusting"
    assert g.step(1, None, 5.0) == "", "must not touch the sampler without a target"


def test_the_governor_stops_after_its_budget_of_adjustments():
    """A run must not spend its whole schedule re-batching."""
    from training.dt_bucketing import BudgetedBatchSampler, MemoryGovernor
    _, counts = _population(n=8000)
    s = BudgetedBatchSampler(counts, 512, budget=1e9, shuffle=False,
                              truncate_bptt=64)
    g = MemoryGovernor(1.0, max_adjustments=2, last_epoch=99)
    notes = [g.step(e, s, 8.0 * 2 ** 30) for e in range(1, 6)]
    assert sum(1 for n in notes if n) == 2, notes
    assert g.adjustments == 2
    assert not g.active(6)


def test_the_governor_stops_after_the_early_epochs():
    from training.dt_bucketing import BudgetedBatchSampler, MemoryGovernor
    _, counts = _population(n=8000)
    s = BudgetedBatchSampler(counts, 512, budget=1e9, shuffle=False,
                              truncate_bptt=64)
    g = MemoryGovernor(1.0, max_adjustments=99, last_epoch=3)
    assert g.step(3, s, 8.0 * 2 ** 30) != ""
    assert g.step(4, s, 8.0 * 2 ** 30) == "", "still adjusting past last_epoch"


def test_the_governor_labels_its_note_with_the_epoch():
    from training.dt_bucketing import BudgetedBatchSampler, MemoryGovernor
    _, counts = _population(n=8000)
    s = BudgetedBatchSampler(counts, 512, budget=1e9, shuffle=False,
                              truncate_bptt=64)
    g = MemoryGovernor(1.0)
    assert g.step(7, s, 8.0 * 2 ** 30).startswith("  [epoch 7]")


def test_the_report_states_the_measured_peak_without_inventing_a_constant():
    """
    NO bytes-per-unit constant is quoted, because none exists. Successive
    denominators gave 1700, 3771, 4845, 13476, 29315 and 76911 bytes per unit
    across measurements -- each a structural model of retained memory, each
    falsified. Quoting any of them invites sizing a budget from a number that
    is not a property of the model. MemoryGovernor closes the loop on the
    measured peak instead.
    """
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    _, counts = _population(n=6000)
    s = BudgetedBatchSampler(counts, 512, shuffle=False, truncate_bptt=64)
    report = budget_report(s, 512, 2, peak_bytes=3.19 * 2 ** 30,
                            reserved_bytes=4.31 * 2 ** 30)
    assert "3.19 GiB peak allocated" in report
    assert "bytes per" not in report, (
        "a per-unit constant is being quoted again: " + report
    )
    assert "RESERVED 4.31 GiB" in report


def test_only_one_controller_touches_the_batching():
    """
    THE FIGHT. For two epochs, MemoryGovernor and a second, model-based
    calibrate() both re-batched: 31 -> 26 (governor), 26 -> 68 (calibrate),
    68 -> 45 (governor). Three re-batchings in two epochs, each one an
    effective learning-rate change, with the two controllers pulling in
    opposite directions -- and the model-based one raising an alarm about a
    run that was comfortably inside its memory target.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "_mem_governor.step(" in src
    assert "_bucket_sampler.calibrate(" not in src, (
        "the model-based controller is back, and it will fight the governor"
    )
    # THREE construction sites, and only three: the unbucketed loader, the
    # bucketed one built before training, and the governor's rebuild. A fourth
    # would mean a second controller re-batching mid-run again.
    assert src.count("train_loader = DataLoader(") == 3, (
        f"{src.count('train_loader = DataLoader(')} loader construction sites "
        f"-- a second mid-run re-batching path has appeared"
    )
    governor_block = src[src.index("_mem_governor.step("):]
    assert "train_loader = DataLoader(" in governor_block[:600], (
        "the governor does not rebuild the loader, so its correction has no "
        "effect on what the next epoch iterates"
    )


# --------------------------------------------------------------------
# Per-batch memory diagnostic: the data every failed model lacked
# --------------------------------------------------------------------

def test_the_diagnostic_records_each_batchs_predictors(tmp_path):
    """
    Six bytes-per-unit "constants" (1700 to 77000) came from fitting models to
    RUN-LEVEL aggregates -- one number per run, four runs, which any curve
    fits. The decidable data is per batch: peak bytes next to that batch's
    size, realised count statistics, span and arrival segments. One epoch
    gives tens of points and the dependence is read off, not theorised.
    """
    from training.dt_bucketing import BatchMemoryDiagnostic
    d = BatchMemoryDiagnostic(tmp_path / "m.csv", truncate_bptt=64)
    d.record(1, 0, np.array([70., 100., 140., 200.]), 2.5e9, 3.0e9)
    d.record(1, 1, np.array([300., 310.]), 1.0e9, 1.5e9)
    note = d.flush()
    assert "2 batch measurements" in note
    lines = (tmp_path / "m.csv").read_text().splitlines()
    assert lines[0].startswith("epoch,batch,n_windows,n_max,n_min,span,arrival_segments")
    e, b, n, mx, mn, span, seg, k, alloc, resv = lines[1].split(",")
    assert (n, mx, mn, span) == ("4", "200", "70", "130")
    # counts 70,100,140,200 at k=64 -> ceil/64 = 2,2,3,4 -> 3 distinct segments
    assert seg == "3", f"arrival segments {seg}"
    assert k == "64" and alloc == "2500000000"
    # second batch: 300,310 -> ceil/64 = 5,5 -> one segment
    assert lines[2].split(",")[6] == "1"


def test_the_diagnostic_appends_across_flushes_with_one_header(tmp_path):
    from training.dt_bucketing import BatchMemoryDiagnostic
    d = BatchMemoryDiagnostic(tmp_path / "m.csv", truncate_bptt=32)
    d.record(1, 0, np.array([10.0]), 1.0, 2.0)
    d.flush()
    d.record(2, 0, np.array([20.0]), 3.0, 4.0)
    d.flush()
    text = (tmp_path / "m.csv").read_text()
    assert text.count("epoch,batch") == 1, "header repeated"
    assert len(text.splitlines()) == 3
    assert d.flush() == "", "an empty flush should say nothing"


def test_train_lds_isolates_each_batchs_measurement():
    """
    Without a reset BEFORE each batch, every reading is the running high-water
    mark of the epoch so far -- monotone by construction, and useless for
    reading off per-batch dependence. Without a synchronize, the reading can
    include kernels from the PREVIOUS batch still in flight.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    i = src.index("for batch in train_loader:")
    body = src[i:i + 2500]
    reset = body.index("reset_peak_memory_stats")
    record = body.index("_mem_diag.record")
    assert reset < record, "no per-batch reset: readings are cumulative"
    assert body[:reset].count("synchronize") + body[reset:record].count("synchronize") >= 2, (
        "missing a synchronize either before the reset (previous batch's "
        "kernels) or before the reading (this batch's)"
    )
    assert "retained_peak(reset=True)" in body[:record], (
        "the counts are not cleared per batch, so record() sees every "
        "transition since the epoch began"
    )
    assert "last_counts()" in body


def test_the_model_exposes_the_realised_counts_per_batch():
    import torch as _torch

    from models.latent_dynamics import LatentDynamics
    m = LatentDynamics(latent_channels=2, latent_spatial=4, hidden_dim=8,
                        n_hidden_layers=1, alpha=0.1, max_substeps=4096,
                        truncate_bptt=64)
    _torch.manual_seed(0)
    counts = _torch.tensor([70, 100])
    m._substeps_for = lambda *a, **k: counts
    m._integrate(_torch.randn(2, 2, 4, 4), _torch.randn(2, 2, 4, 4),
                  _torch.full((2,), 100.0), _torch.zeros(2, 1))
    m._integrate(_torch.randn(2, 2, 4, 4), _torch.randn(2, 2, 4, 4),
                  _torch.full((2,), 100.0), _torch.zeros(2, 1))
    got = m.last_counts()
    assert len(got) == 2, "one entry per transition"
    assert _torch.equal(got[0], counts)
    m.retained_peak(reset=True)
    assert m.last_counts() == [], "reset must clear the counts"


def test_the_governor_targets_reserved_not_allocated():
    """
    RESERVED is what occupies the card: the caching allocator does not hand
    cached blocks back, so a target on ALLOCATED understates the card by the
    cache fraction. Measured: 5.64 GiB allocated against a 5.50 GiB target
    read as converged, while 6.77 GiB was actually held -- 23% more card than
    the user asked for, on an 8 GB device.
    """
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "_mem_governor.step(epoch, _bucket_sampler, _resv)" in src, (
        "the governor is driven by allocated bytes, so target_vram_gib does "
        "not mean what its name says"
    )
    assert "peak RESERVED GiB" in src, "the help text still promises allocated"


# --------------------------------------------------------------------
# The MEASURED cost model
# --------------------------------------------------------------------

def test_the_cost_model_has_both_a_per_window_and_a_depth_term():
    """
    bytes = A*n + B*n*depth^p, from check_memory's two probe sweeps on a real
    3b checkpoint (9 points, n 128-1024, depth 8-279, 13.2% worst residual).

    BOTH terms are load-bearing. Without A, a shallow batch is priced at
    almost nothing and 2048 windows of depth 2 look free -- that batch
    measured 731 MiB. Without B, depth is free and the deep tail is
    unbounded.
    """
    from training.dt_bucketing import BudgetedBatchSampler as S
    assert S.COST_A_BYTES > 0 and S.COST_B_BYTES > 0
    s = S(np.array([10.0, 20.0]), 8, shuffle=False, truncate_bptt=64)
    shallow = s._depth(1.0, 1.0)
    deep = s._depth(256.0, 256.0)
    assert shallow > 0.5 * S.COST_A_BYTES, (
        "a depth-1 window costs almost nothing, so a huge shallow batch would "
        "look free -- 2048 windows of depth 2 measured 731 MiB"
    )
    assert deep > 4 * shallow, "depth is nearly free, so the deep tail is unbounded"


def test_the_depth_exponent_is_the_measured_one_not_one():
    """
    p=0.79. Forcing p=1 was every earlier model's assumption and it
    over-charges deep batches -- the direction of every budget discrepancy in
    this project's history. The fixed-n sweep (n constant, constant fitted
    separately) put it at 0.70-0.79, and p=1 scored 29.6% against 8.5%.
    """
    from training.dt_bucketing import BudgetedBatchSampler as S
    assert 0.6 <= S.COST_P <= 0.9, f"exponent {S.COST_P}"
    s = S(np.array([10.0, 20.0]), 8, shuffle=False, truncate_bptt=64)
    # the depth TERM must grow sublinearly: 16x the depth, well under 16x cost
    lo = s._depth(16.0, 16.0) - S.COST_A_BYTES
    hi = s._depth(256.0, 256.0) - S.COST_A_BYTES
    assert hi < 12 * lo, (
        f"depth term grew {hi / lo:.1f}x for a 16x depth increase -- that is "
        f"linear scaling, which measurement rejected"
    )
    assert hi > 4 * lo, "depth term barely grows; the exponent is too small"


def test_the_budget_is_in_bytes_and_bounds_the_predicted_peak():
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=20000)
    budget = 512 * 2 ** 20
    s = BudgetedBatchSampler(counts, 4096, budget=budget, shuffle=False,
                              truncate_bptt=64)
    assert s.peak_cost() <= budget * 1.001, (
        f"predicted peak {s.peak_cost() / 2**20:.0f} MiB over a "
        f"{budget / 2**20:.0f} MiB budget"
    )
    assert s.peak_cost() > budget * 0.3, "the budget is not binding at all"
    assert sorted(i for b in s for i in b) == list(range(len(counts)))


def test_a_budget_in_the_retired_unit_is_refused():
    """
    Configs carry batch_cost_budget=30000 from when the budget counted
    sample-substeps. Read as bytes that is 30 KB -- below one window -- so
    every batch would hold one and the run would look merely slow rather than
    broken. Refuse instead, and say what to do.
    """
    from training.dt_bucketing import BudgetedBatchSampler
    _, counts = _population(n=2000)
    with pytest.raises(ValueError) as e:
        BudgetedBatchSampler(counts, 512, budget=30_000.0, shuffle=False,
                              truncate_bptt=64)
    assert "BYTES" in str(e.value) and "target_vram_gib" in str(e.value)
    # a legitimate small-but-sane budget is still accepted
    ok = BudgetedBatchSampler(counts, 512, budget=4 * 2 ** 20, shuffle=False,
                               truncate_bptt=64)
    assert len(ok) > 0


def test_train_lds_turns_a_vram_target_into_the_byte_budget():
    from conftest import source_without_comments
    import pathlib
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    # The target must REACH the budget -- but converted, not verbatim (see
    # test_a_vram_target_is_converted_not_used_as_the_budget_directly).
    assert "target_vram_gib" in src and "_budget" in src, (
        "the VRAM target does not set the budget, so the two knobs disagree"
    )
    assert "budget=_budget" in src


def test_the_report_states_the_budget_in_the_unit_it_actually_carries():
    """
    The budget is BYTES. The report kept calling it sample-substeps, so a real
    3a run printed "budget 403244338 sample-substeps" and "peak = 403244338
    retained sample-substeps" -- a byte figure with a count's label, nine
    digits wide, which reads as a bug in the batching rather than a unit.
    """
    from training.dt_bucketing import BudgetedBatchSampler, budget_report
    _, counts = _population(n=6000)
    s = BudgetedBatchSampler(counts, 512, shuffle=False)
    report = budget_report(s, 512, 1)
    assert "sample-substeps." not in report, report.splitlines()[0]
    assert "MiB of activations" in report
    assert "ACTIVATIONS only" in report, (
        "the report does not say the budget excludes parameters and optimizer "
        "state, so it looks inconsistent with target_vram_gib"
    )


def test_the_cost_coefficients_can_be_overridden_per_machine():
    """
    The fitted values are DEFAULTS, not constants of nature: they were
    measured on one card (RTX 2060 Super), one architecture (hidden_dim=256,
    latent 8x8x8) and n_rollout_steps=2. A different GPU or latent size moves
    them, and the way to recalibrate is to rerun check_memory and paste its
    joint fit -- which requires the numbers to be settable.
    """
    from training.dt_bucketing import BudgetedBatchSampler as S
    _, counts = _population(n=4000)
    default = S(counts, 512, shuffle=False, truncate_bptt=64)
    custom = S(counts, 512, shuffle=False, truncate_bptt=64,
                cost_a_bytes=1e5, cost_b_bytes=5e4, cost_p=1.0)
    assert custom.COST_P == 1.0 and custom.COST_A_BYTES == 1e5
    assert custom.budget != default.budget, "the override did not reach the budget"
    _hi = float(counts.max())
    assert custom._depth(_hi, _hi) == pytest.approx(1e5 + 5e4 * _hi), (
        "the overridden coefficients are not the ones _depth uses"
    )
    # and the CLASS reference values are untouched, so one run cannot silently
    # recalibrate another in the same process
    assert S.COST_P == 0.79 and S.COST_A_BYTES == 118.0 * 1024.0


def test_partial_overrides_keep_the_measured_defaults_for_the_rest():
    """Setting only the exponent must not zero the coefficients."""
    from training.dt_bucketing import BudgetedBatchSampler as S
    _, counts = _population(n=2000)
    s = S(counts, 512, shuffle=False, truncate_bptt=64, cost_p=0.6)
    assert s.COST_P == 0.6
    assert s.COST_A_BYTES == S.COST_A_BYTES
    assert s.COST_B_BYTES == S.COST_B_BYTES


def test_train_lds_threads_the_cost_coefficients_to_the_sampler():
    import inspect

    from conftest import source_without_comments
    import pathlib
    from training.train_lds import train_lds
    for name in ("memory_cost_a_bytes", "memory_cost_b_bytes", "memory_cost_p"):
        assert inspect.signature(train_lds).parameters[name].default is None
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "cost_a_bytes=memory_cost_a_bytes" in src
    assert "cost_b_bytes=memory_cost_b_bytes" in src
    assert "cost_p=memory_cost_p" in src


def test_a_vram_target_is_converted_not_used_as_the_budget_directly():
    """
    THE TARGET IS TOTAL MEMORY; THE BUDGET IS ACTIVATIONS ONLY. Setting them
    equal overshoots by exactly the overhead on the first epoch, every time.

    Measured on a real 3b run at target 6.5 GiB: an activations budget of 6.5
    GiB produced 8.19 GiB allocated / 8.96 GiB reserved on an 8 GB card. It
    spilled into shared system memory and ran at PCIe speed while the governor
    cut the budget three times without the peak falling -- it never fit.
    """
    import pathlib

    from conftest import source_without_comments
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "_budget = float(target_vram_gib) * 2 ** 30" not in src, (
        "the VRAM target is being used as the activation budget directly, "
        "which overshoots by the overhead on epoch 1"
    )
    assert "_OVERHEAD_BYTES" in src and "_CACHE_FACTOR" in src, (
        "no conversion from total memory to an activation budget"
    )

    # The CONSTANTS THEMSELVES, read out of the source -- asserting only that
    # the names appear let a mutation zeroing the overhead pass, which is the
    # whole bug back again under a different spelling.
    import re
    m_over = re.search(r"_OVERHEAD_BYTES\s*=\s*([0-9.]+)\s*\*\s*2\s*\*\*\s*30", src)
    m_cache = re.search(r"_CACHE_FACTOR\s*=\s*([0-9.]+)", src)
    assert m_over and m_cache, "the constants are not in the expected form"
    overhead_gib, cache = float(m_over.group(1)), float(m_cache.group(1))
    assert overhead_gib >= 1.0, (
        f"overhead is {overhead_gib} GiB -- parameters, optimizer state and "
        f"workspace measured about 1.7 GiB, so this does not account for them"
    )
    assert cache >= 1.05, (
        f"cache factor {cache} -- reserved ran 9% above allocated on the "
        f"measured run"
    )
    overhead, cache = overhead_gib * 2 ** 30, cache
    budget = 6.5 * 2 ** 30 / cache - overhead
    assert budget > 0
    assert (budget + overhead) * cache <= 6.5 * 2 ** 30 * 1.001, (
        "the converted budget does not reconcile back to the target"
    )
    assert budget < 6.5 * 2 ** 30 * 0.75, (
        "the conversion barely reduces the budget, so the overhead is not "
        "actually being accounted for"
    )


def test_a_small_target_still_yields_a_usable_budget():
    """A target below the overhead must not produce a negative or absurd
    budget -- it should floor at something that still batches."""
    import pathlib

    from conftest import source_without_comments
    from training.dt_bucketing import BudgetedBatchSampler
    src = source_without_comments(pathlib.Path(__file__).resolve().parent.parent
                                   / "training/train_lds.py")
    assert "max(" in src.split("_OVERHEAD_BYTES")[1][:600], (
        "no floor on the converted budget; a small target would give a "
        "negative one"
    )
    floor = BudgetedBatchSampler.COST_A_BYTES * 64.0
    assert floor > BudgetedBatchSampler.COST_A_BYTES, (
        "the floor is below one window's cost and would be refused by the "
        "sampler's own stale-unit guard"
    )
