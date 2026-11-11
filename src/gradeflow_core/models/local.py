"""Where the language models of the optional ``ml`` extra are kept, and what to do without them."""

import os
from pathlib import Path

MISSING_MODEL_MESSAGE = (
    "Comparing by meaning requires gradeflow-core[ml]. "
    "Install it, then run python -m gradeflow_core setup."
)


def cache_directory() -> Path:
    if configured_directory := os.environ.get("GRADEFLOW_CACHE_DIR"):
        root = Path(configured_directory).expanduser()
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "gradeflow"
    return root / "models"
