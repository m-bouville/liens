"""Guard: no source file may exceed its recorded code-size ceiling.

check_parameter_dependence.py motivated this: the project's own notes record
it being cut from 2566 to ~1722 lines via module extraction, and it has since
regrown to 2846 (1818 of comment/docstring-stripped CODE) -- larger than
BEFORE that refactor. Nothing held the extraction in place.

RATCHET, not a fixed limit: each ceiling below is that file's current
comment/docstring-stripped line count (conftest.source_without_comments, so
this project's own convention of heavy rationale docstrings doesn't itself
trip the guard) plus headroom (max(30, 10%), rounded up to the nearest 10).
Every file is GREEN today by construction; it fails only on further growth
past that headroom, or a brand-new file starting past _DEFAULT_CAP.

Keys are "{package}/{filename}", matching tests/_import_graph.py's own
modpaths convention -- this project has five package dirs (training, models,
evaluation, utils, orchestration) plus main.py sitting outside all of them
(see tests/_import_graph.py's "top-level entry scripts" handling, which this
mirrors). Measuring stripped code rather than raw wc -l matters here too: by
raw lines check_parameter_dependence.py (2846) looks largest, but by
stripped code evaluation/compare_f_theta.py (1977) actually is
(check_parameter_dependence.py is 1818 stripped) -- raw line count was
partly measuring documentation, not code.

NOT guarded (by design, not oversight): files under tests/ itself --
tests/_ast_helpers.py, tests/_figure_artifacts.py, tests/_import_graph.py
(all test/tooling infrastructure, imported by bare name only, confirmed via
grep before this list was built) and tests/data/dependency_graph.py, which
is a checked-in DATA snapshot (two dicts, imports nothing -- see its own
docstring) rather than code. An earlier version of this dict wrongly
included all four, discovered when _PYTHON_ROOT pointed at tests/ instead of
python/ and every ceiling came back "stale" -- glob-ing zero files, not
zero-length files.

On tripping this test: either split the file (the actual fix), or -- if the
growth is deliberate and reviewed, not creep -- bump that file's ceiling
explicitly in this dict, in the same commit, so the bump itself is visible
in review. Do not raise a ceiling to silence a failure without reading why
it grew.
"""
from pathlib import Path

from conftest import source_without_comments

# tests/test_source_file_size_guard.py -> python/ (this project's package
# dirs and main.py all sit directly under here; tests/ itself is a sibling,
# never a guard target -- see the NOT GUARDED note above for why that
# matters).
_PYTHON_ROOT = Path(__file__).resolve().parent.parent

_PACKAGES = ("training", "models", "evaluation", "utils", "orchestration")

# A file not in this dict is either new since this test was written, or one
# of the top-level scripts (currently just main.py, itself in the dict).
_DEFAULT_CAP = 500   # a new file this large from the start should be split
                      # before it grows further, not grandfathered by
                      # omission from the dict

_CEILINGS = {
    "evaluation/_ae_stats_eval.py": 120,   # current 83
    "evaluation/_latent_eval.py": 480,   # current 432
    "evaluation/backfill_eval_components.py": 760,   # current 689
    "evaluation/check_alpha.py": 670,   # current 609
    "evaluation/check_deriv_temperature.py": 230,   # current 192
    "evaluation/check_dt_vs_time.py": 350,   # current 314
    "evaluation/check_f_theta.py": 550,   # current 500
    "evaluation/check_grad_spikes.py": 260,   # current 228
    "evaluation/check_interpolation.py": 250,   # current 211
    "evaluation/check_latent_channels.py": 690,   # current 624
    "evaluation/check_memory.py": 620,   # current 562
    "evaluation/check_normalization_factor.py": 160,   # current 127
    "evaluation/check_parameter_dependence.py": 2000,   # current 1818
    "evaluation/check_perturbation.py": 230,   # current 199
    "evaluation/check_reconstruction.py": 310,   # current 279
    "evaluation/check_rollout.py": 540,   # current 488
    "evaluation/check_stats_head_rollout.py": 390,   # current 354
    "evaluation/check_stdev_phi_temperature.py": 310,   # current 279
    "evaluation/check_stdev_phi_time.py": 1210,   # current 1095
    "evaluation/check_substep_convergence.py": 240,   # current 202
    "evaluation/check_z1_degeneracy.py": 190,   # current 156
    "evaluation/check_z2_measurability.py": 390,   # current 354
    "evaluation/compare_f_theta.py": 2180,   # current 1977
    "evaluation/compare_integrators.py": 150,   # current 117
    "evaluation/compare_rollout_training.py": 200,   # current 161
    "evaluation/find_windows.py": 140,   # current 105
    "evaluation/inspect_checkpoint.py": 120,   # current 89
    "evaluation/lineage.py": 260,   # current 228
    "evaluation/select_latent_channels.py": 180,   # current 142
    "evaluation/sweep_min_passing_steps.py": 100,   # current 70
    "evaluation/sweep_min_std_deriv.py": 200,   # current 167
    "evaluation/sweep_min_stdev_phi.py": 130,   # current 95
    "main.py": 220,   # current 188
    "models/autoencoder.py": 280,   # current 241
    "models/blocks.py": 120,   # current 87
    "models/constants.py": 80,   # current 46
    "models/decoder.py": 130,   # current 92
    "models/encoder.py": 290,   # current 260
    "models/latent_dynamics.py": 600,   # current 543
    "models/latent_streams.py": 360,   # current 319
    "orchestration/checkpoint_identification.py": 200,   # current 164
    "orchestration/checkpoint_registry.py": 220,   # current 188
    "orchestration/pipeline.py": 600,   # current 539
    "orchestration/stage_params.py": 400,   # current 357
    "orchestration/sweep_status.py": 80,   # current 42
    "training/_checkpoint_criterion.py": 400,   # current 364
    "training/_dataset_filtering.py": 280,   # current 248
    "training/_refinement_loss.py": 210,   # current 172
    "training/_spike_guard.py": 510,   # current 462
    "training/_train_ae_common.py": 160,   # current 124
    "training/_training_loop.py": 220,   # current 190
    "training/checkpoint_components.py": 630,   # current 567
    "training/datasets.py": 1740,   # current 1579
    "training/dt_bucketing.py": 680,   # current 614
    "training/extend_encoder.py": 190,   # current 154
    "training/latent_cache.py": 190,   # current 158
    "training/losses.py": 560,   # current 502
    "training/model_assembly.py": 190,   # current 153
    "training/port_checkpoint.py": 210,   # current 179
    "training/rescale_checkpoint.py": 590,   # current 533
    "training/stats_head.py": 70,   # current 32
    "training/train_lds.py": 1540,   # current 1395
    "training/train_refinement.py": 790,   # current 713
    "training/train_stage1.py": 730,   # current 662
    "training/train_stage2.py": 1280,   # current 1155
    "utils/eval_log.py": 330,   # current 293
    "utils/fits.py": 490,   # current 439
    "utils/load_datasets.py": 360,   # current 325
    "utils/logging_utils.py": 310,   # current 275
    "utils/naming.py": 60,   # current 24
    "utils/paths.py": 80,   # current 41 -- utils/, confirmed by 9 "from utils.paths import" sites (check_alpha.py, check_dt_vs_time.py, check_parameter_dependence.py, checkpoint_registry.py, compare_f_theta.py, main.py, pipeline.py, port_checkpoint.py, conftest.py); NOT orchestration/, despite NN-code_structure.md's phrasing ("paths.py ... living in orchestration/") reading that way out of context -- doc text lost to grep evidence
    "utils/plot_helpers.py": 160,   # current 127
    "utils/plots.py": 840,   # current 758
    "utils/sweep_filters_common.py": 130,   # current 93
    "utils/window_parsing.py": 80,   # current 43
}


def _stripped_line_count(path: Path) -> int:
    stripped = source_without_comments(path)
    return len([ln for ln in stripped.splitlines() if ln.strip()])


def _source_files():
    """Every source file this guard covers: main.py (top-level) plus every
    *.py under each of the five package dirs (excluding __init__.py, which
    never has meaningful size). Yields (key, path) with key matching
    _CEILINGS's "{package}/{filename}" convention ("main.py" for the
    top-level file, unprefixed)."""
    main_py = _PYTHON_ROOT / "main.py"
    if main_py.exists():
        yield "main.py", main_py
    for pkg in _PACKAGES:
        d = _PYTHON_ROOT / pkg
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.py")):
            if p.stem != "__init__":
                yield f"{pkg}/{p.name}", p


def test_no_source_file_exceeds_its_recorded_ceiling():
    over = []
    for key, path in _source_files():
        n = _stripped_line_count(path)
        cap = _CEILINGS.get(key, _DEFAULT_CAP)
        if n > cap:
            over.append((key, n, cap))
    if over:
        detail = "\n".join(
            f"  {key}: {n} stripped lines > ceiling {cap} "
            f"({'not in _CEILINGS -- new file over the default cap' if cap == _DEFAULT_CAP and key not in _CEILINGS else 'grew past its recorded ceiling'})"
            for key, n, cap in over)
        raise AssertionError(
            "source file(s) exceeded their code-size ceiling -- split the "
            "file, or bump its ceiling deliberately in this test if the "
            "growth was reviewed:\n" + detail)


def test_ceilings_dict_has_no_stale_entries():
    """A file removed, renamed, or moved between packages should have its
    OLD ceiling key removed too, or the dict silently stops guarding
    anything for it. (This test itself caught a wrong _PYTHON_ROOT once:
    every key looked stale when the discovery glob was pointed at the wrong
    directory and found nothing at all -- an empty `existing` set makes
    EVERY entry stale, which is a discovery-path bug, not real staleness;
    double-check _source_files() finds anything before trusting this
    failure.)"""
    existing = {key for key, _ in _source_files()}
    assert existing, (
        "_source_files() found NOTHING -- _PYTHON_ROOT or _PACKAGES is "
        "almost certainly wrong (points at the wrong directory), not that "
        "every guarded file vanished. Fix discovery before trusting any "
        "'stale entries' result from this test.")
    stale = sorted(set(_CEILINGS) - existing)
    assert not stale, (
        f"_CEILINGS has entries for file(s) that no longer exist at their "
        f"recorded location: {stale} -- remove them, or update the key if "
        f"the file moved between packages")
