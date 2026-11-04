"""Prepare optional resources for local or offline use."""

import argparse
from collections.abc import Sequence

from .ml import cache_directory, embeddings, nli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["setup"], help="Download the core's language models.")
    parser.parse_args(argv)
    try:
        embeddings.setup()
        nli.setup()
    except ImportError as error:
        parser.exit(1, f"{error}\n")
    print(f"Models ready in {cache_directory()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
