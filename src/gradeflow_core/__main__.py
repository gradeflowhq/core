"""Prepare optional resources for local or offline use."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .ml import cache_directory, embeddings, nli


def setup_models() -> Path:
    """Download the core's models without loading them for inference."""
    embeddings.setup()
    nli.setup()
    return cache_directory()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gradeflow_core")
    parser.add_argument("command", choices=["setup"], help="Download the core's language models.")
    parser.parse_args(argv)
    try:
        directory = setup_models()
    except ImportError as error:
        parser.exit(1, f"{error}\n")
    print(f"Models ready in {directory}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
