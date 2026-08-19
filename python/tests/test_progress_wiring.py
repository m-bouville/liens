

def test_no_duplicate_logging_utils_module():
    """logging_utils must live in exactly ONE place (utils/). A second copy
    under orchestration/ silently shadowed the r-filter for the pipeline's
    own Tee -- progress bars then leaked into stage logs even though
    utils/logging_utils.py was correct. Pin: no stray module, and nobody
    imports from a non-utils logging_utils."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    copies = [p for p in root.rglob("logging_utils.py")
              if "__pycache__" not in str(p)]
    locations = {p.parent.name for p in copies}
    assert locations == {"utils"}, (
        f"logging_utils.py must exist only under utils/, found in {locations} "
        f"-- a duplicate shadows the canonical r-filtering _Tee")
    # and no module imports it from anywhere but utils
    offenders = []
    for py in root.rglob("*.py"):
        if "__pycache__" in str(py) or py.name == "test_progress_wiring.py":
            continue  # skip this file: its own assertion text mentions the pattern
        for line in py.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped.startswith(("from ", "import ")):
                continue  # only real import statements, not strings mentioning it
            if "logging_utils" in stripped and "utils.logging_utils" not in stripped:
                offenders.append(f"{py.name}: {stripped}")
    assert not offenders, f"non-canonical logging_utils imports: {offenders}"
