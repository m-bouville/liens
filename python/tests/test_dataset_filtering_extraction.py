"""The run/step-selection helpers were extracted to
training._dataset_filtering. datasets.py re-exports them so the whole tree's
`from training.datasets import build_good_steps` (and the other four names)
keeps working. This locks that contract: the names must resolve through BOTH
modules and be the SAME object (a re-export, not a divergent copy)."""
import training.datasets as ds
import training._dataset_filtering as flt

_REEXPORTED = ["build_good_steps", "complete_run_dirs", "split_run_dirs",
               "_filtered_steps", "report_save_step_distribution"]


def test_all_names_resolve_through_datasets():
    for name in _REEXPORTED:
        assert hasattr(ds, name), f"training.datasets lost re-export of {name}"


def test_reexport_is_the_same_object_not_a_copy():
    for name in _REEXPORTED:
        assert getattr(ds, name) is getattr(flt, name), \
            f"{name} diverged between datasets and _dataset_filtering"


def test_filtering_module_is_a_leaf_no_datasets_import():
    """_dataset_filtering must NOT import datasets (that would be the circular
    import the leaf placement exists to avoid). Checked at the AST level so the
    module's own docstring mentioning datasets.py doesn't false-trigger."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(flt))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(m == "training.datasets" or m.endswith(".datasets")
                   for m in imported), f"leaf module imports datasets: {imported}"
