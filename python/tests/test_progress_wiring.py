"""Wiring tests: the progress/labeling plumbing (EpochProgress ticks,
format_progress_count, split_label handling in the datasets) are behaviorally
invisible to the numeric suites -- deleting a tick() call or a split_label=
argument would pass every existing test while silently degrading the logs.
These pin the call sites themselves.

NOTE: this file was reconstructed after an accidental overwrite (2026-08-20)
reduced it to a single test. One original test may be missing -- if version
control has the pre-2026-08-19-22h13 copy, prefer restoring from there and
re-adding the newer tests below. Known gap: the original file also had a
test_interpolation_check_wires_its_progress_counter -- it guarded an
in-place counter that the current check_interpolation.py does not have,
so it is omitted here rather than reconstructed against a missing feature.
"""
import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TRAINERS = ["training/train_stage1.py", "training/train_stage2.py",
             "training/train_lds.py", "training/train_refinement.py"]


def _dataset_split_labels(rel, class_name):
    """Collect the split_label keyword values of every `class_name(...)` call."""
    tree = ast.parse((_ROOT / rel).read_text())
    labels = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", getattr(fn, "attr", None))
            if name == class_name:
                found = None
                for kw in node.keywords:
                    if kw.arg == "split_label" and isinstance(kw.value, ast.Constant):
                        found = kw.value.value
                labels.append(found)
    return labels


def test_stage2_passes_split_labels_to_all_three_constructions():
    labels = _dataset_split_labels("training/train_stage2.py",
                                   "MicrostructureEvolutionDataset")
    assert len(labels) == 3 and set(labels) == {"training", "validation"}, (
        f"train_stage2 dataset constructions carry labels {labels}; expected one "
        f"'training' and two 'validation' -- "
        f"a train_stage2 dataset construction lost its split_label: {labels}")


def test_stage1_passes_split_labels_to_all_three_constructions():
    labels = _dataset_split_labels("training/train_stage1.py",
                                   "MicrostructureSnapshotDataset")
    assert len(labels) == 3 and set(labels) == {"training", "validation"}, (
        f"train_stage1 dataset constructions carry labels {labels}; expected one "
        f"'training' and two 'validation' -- "
        f"a train_stage1 dataset construction lost its split_label: {labels}")


def test_lds_passes_split_labels_to_all_three_constructions():
    """train_lds is the stage-3a/3b trainer -- it encodes with the frozen
    encoder, so its build_good_steps/encoding messages are the ones actually
    seen in stage-3 logs. Its three dataset constructions must carry labels
    (this was missed when split_label was first threaded through stages 1-2,
    which is why stage-3 logs showed the unlabeled 'runs dropped ENTIRELY')."""
    labels = _dataset_split_labels("training/train_lds.py",
                                   "MicrostructureEvolutionDataset")
    assert len(labels) == 3 and set(labels) == {"training", "validation"}, (
        f"train_lds dataset constructions carry labels {labels}; expected "
        f"three constructions labeled training/validation")


def test_refinement_passes_split_labels_to_all_three_constructions():
    labels = _dataset_split_labels("training/train_refinement.py",
                                   "MicrostructureEvolutionDataset")
    assert len(labels) == 3 and set(labels) == {"training", "validation"}, (
        f"train_refinement dataset constructions carry labels {labels}; "
        f"expected three constructions labeled training/validation")


def test_all_saving_trainers_stamp_the_save_time():
    """Every trainer that prints '-> saved' must also stamp the wall-clock
    time, so a saved-epoch line can be matched to the timestamped checkpoint
    filename it produced. This was applied to stage 2 first and initially
    forgotten in the other three -- pin all of them.

    A trainer that delegates the save to _checkpoint_criterion.save_checkpoint
    gets the "-> saved at HH:MM" suffix (strftime) from the helper -- covered by
    test_save_checkpoint_returns_the_saved_suffix -- so it satisfies this without
    an inline strftime. Comments are stripped so a mention of '-> saved' in prose
    (e.g. describing the epoch line) does not count as printing it.
    """
    from conftest import source_without_comments
    for rel in _TRAINERS:
        src = source_without_comments(_ROOT / rel)
        if "save_checkpoint(" in src:      # delegated -> helper stamps the time
            continue
        if "-> saved" not in src:
            continue
        assert "strftime" in src, (
            f"{rel}: prints '-> saved' but never stamps the time "
            f"(strftime missing) -- the checkpoint can't be matched to its epoch")


def test_no_duplicate_logging_utils_module():
    """logging_utils must live in exactly ONE place (utils/). A second copy
    under orchestration/ silently shadowed the r-filter for the pipeline's
    own Tee -- progress bars then leaked into stage logs even though
    utils/logging_utils.py was correct. Pin: no stray module, and nobody
    imports from a non-utils logging_utils."""
    copies = [p for p in _ROOT.rglob("logging_utils.py")
              if "__pycache__" not in str(p)]
    locations = {p.parent.name for p in copies}
    assert locations == {"utils"}, (
        f"logging_utils.py must exist only under utils/, found in {locations} "
        f"-- a duplicate shadows the canonical r-filtering _Tee")
    offenders = []
    for py in _ROOT.rglob("*.py"):
        if "__pycache__" in str(py) or py.name == "test_progress_wiring.py":
            continue
        for line in py.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped.startswith(("from ", "import ")):
                continue
            if "logging_utils" in stripped and "utils.logging_utils" not in stripped:
                offenders.append(f"{py.name}: {stripped}")
    assert not offenders, f"non-canonical logging_utils imports: {offenders}"
