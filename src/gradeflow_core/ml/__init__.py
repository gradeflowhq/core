"""The language models of the optional ``ml`` extra, run locally, and where they are kept."""

import os
from pathlib import Path

MISSING_MODEL = (
    "Comparing by meaning requires gradeflow-core[ml]. "
    "Install it, then run python -m gradeflow_core setup."
)


def cache_directory() -> Path:
    if configured := os.environ.get("GRADEFLOW_CACHE_DIR"):
        root = Path(configured).expanduser()
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "gradeflow"
    return root / "models"
