"""
Structural test: train_stage1 records in its checkpoint config the
filter/preprocessing params that eval tools read back by default.

The eval tools (check_latent_channels, check_reconstruction, ...) default
normalize_phi/min_step/min_stdev_phi to the checkpoint's OWN recorded values --
that only works if the writer (stage 1) actually saves them. This is the writer
end of that contract, the same shape as the meaning-field writer guard: checked
structurally (does the saved config dict contain these keys?) so a refactor that
drops a key fails here rather than silently making eval fall back to raw defaults.
"""
import ast
from pathlib import Path


def _saved_config_string_keys(src: str) -> set[str]:
    """Every string key that appears in a dict literal under a \"config\" entry of
    train_stage1's checkpoint save. Structural, not a run: we assert the KEYS are
    written, not their values."""
    tree = ast.parse(src)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                # the checkpoint dict has a "config": { ... } entry; collect that
                # inner dict's string keys
                if (isinstance(k, ast.Constant) and k.value == "config"
                        and isinstance(v, ast.Dict)):
                    for ik in v.keys:
                        if isinstance(ik, ast.Constant) and isinstance(ik.value, str):
                            keys.add(ik.value)
    return keys


def test_stage1_config_records_filter_and_normalize_params():
    src = (Path(__file__).resolve().parent.parent / "training" / "train_stage1.py").read_text()
    keys = _saved_config_string_keys(src)
    for required in ("normalize_phi", "min_step", "min_stdev_phi"):
        assert required in keys, (
            f"train_stage1's saved config must record {required!r} so eval tools can "
            f"default to it; found config keys: {sorted(keys)}")
