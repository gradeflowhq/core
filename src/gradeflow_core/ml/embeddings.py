"""Answers as vectors, for grouping them by meaning, with a small language model run locally."""

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

from . import MISSING_MODEL_MESSAGE, cache_directory

if TYPE_CHECKING:
    import numpy as np
    from fastembed import TextEmbedding
    from numpy.typing import NDArray

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _load_model(*, lazy_load: bool = False) -> TextEmbedding:
    try:
        from fastembed import TextEmbedding
    except ImportError as error:
        raise ImportError(MISSING_MODEL_MESSAGE) from error
    return TextEmbedding(EMBEDDING_MODEL, cache_dir=str(cache_directory()), lazy_load=lazy_load)


def setup() -> None:
    """Download the model without loading it for inference."""
    _load_model(lazy_load=True)


@functools.cache
def _model() -> TextEmbedding:
    return _load_model()


def embed_texts(texts: Sequence[str]) -> NDArray[np.float32]:
    """Encode texts as normalized document vectors for comparing them by meaning, one row each."""
    import numpy as np

    return np.stack(list(_model().embed(list(texts))))
