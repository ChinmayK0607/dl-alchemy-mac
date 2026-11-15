"""Locate the optional, read-only course checkout without importing ML frameworks."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_URL = "https://github.com/deep-learning-alchemy/assignments"
REFERENCE_COMMIT = "baafc3f2d364e5ddbba75f0fe8d1cb29326920e5"
_REFERENCE_FILES = ("modeling.py", "model_config.py", "train.py", "optimizers.py", "lr_schedules.py", "utils.py", "data.py")


def reference_path():
    """Resolve env override > submodule > legacy sibling; fail on a bad override.

    Only parity checks and the PyTorch port call this function. Importing this
    module, or training with MLX, never requires the course repository.
    """
    override = os.environ.get("DL_ALCHEMY_REFERENCE")
    candidates = ([Path(override).expanduser()] if override is not None else
                  [REPO_ROOT / "reference", REPO_ROOT.parent / "dl-alchemy-a1"])
    for path in candidates:
        if (override is None or override.strip()) and all((path / name).is_file() for name in _REFERENCE_FILES):
            return path.resolve()
    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Course reference checkout missing or incomplete (checked: {checked}). "
        "Run `git submodule update --init --recursive` from the project root, "
        "or set DL_ALCHEMY_REFERENCE to a complete A1 checkout. "
        f"Expected {REFERENCE_URL} at {REFERENCE_COMMIT}. "
        "An invalid DL_ALCHEMY_REFERENCE is an error; unset it to use the submodule."
    )
