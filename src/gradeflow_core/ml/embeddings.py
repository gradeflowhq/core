"""Answers as vectors, for grouping them by meaning, with a small language model run locally."""

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from . import MISSING_MODEL, cache_directory

if TYPE_CHECKING:
    from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _load_model(*, lazy_load: bool = False) -> TextEmbedding:
    try:
        from fastembed import TextEmbedding
    except ImportError as error:
        raise ImportError(MISSING_MODEL) from error
    return TextEmbedding(EMBEDDING_MODEL, cache_dir=str(cache_directory()), lazy_load=lazy_load)


def setup() -> None:
    """Download the model without loading it for inference."""
    _load_model(lazy_load=True)


@functools.cache
def _model() -> TextEmbedding:
    return _load_model()


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Encode texts as normalized document vectors for comparing them by meaning."""
    if not texts:
        return []
    model = _model()
    import numpy as np

    vectors = np.asarray(list(model.embed(list(texts))), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return cast(list[list[float]], vectors.tolist())
