import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from gradeflow_core.ml import embeddings


@pytest.fixture
def cleared_model_cache() -> Iterator[None]:
    load_model = embeddings._model
    load_model.cache_clear()
    yield
    load_model.cache_clear()


@pytest.mark.usefixtures("cleared_model_cache")
def test_setup_and_grouping_share_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GRADEFLOW_CACHE_DIR", str(tmp_path))
    calls: list[tuple[str, str, bool]] = []

    class FakeTextEmbedding:
        def __init__(self, model: str, *, cache_dir: str, lazy_load: bool = False) -> None:
            calls.append((model, cache_dir, lazy_load))

    fake_fastembed = types.ModuleType("fastembed")
    cast(Any, fake_fastembed).TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)
    embeddings.setup()
    embeddings._model()
    embeddings._model()
    directory = str(tmp_path / "models")
    assert calls == [
        (embeddings.EMBEDDING_MODEL, directory, True),
        (embeddings.EMBEDDING_MODEL, directory, False),
    ]


def test_grouping_embeddings_preserve_document_encoding_and_float32_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    embedded_texts: list[list[str]] = []

    class FakeEmbeddingModel:
        def embed(self, texts: list[str]) -> Iterator[Any]:
            embedded_texts.append(texts)
            return iter(np.array([[3.0, 4.0], [-2.0, 0.0]], dtype=np.float64))

    monkeypatch.setattr(embeddings, "_model", lambda: FakeEmbeddingModel())
    texts = ["first", "second"]
    vectors = embeddings.embed_texts(texts)
    expected = np.asarray([[3.0, 4.0], [-2.0, 0.0]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert vectors == expected.tolist()
    assert embedded_texts == [["first", "second"]]
    assert texts == ["first", "second"]


def test_empty_embedding_batch_does_not_load_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> Any:
        pytest.fail("An empty batch needs no model.")

    monkeypatch.setattr(embeddings, "_model", forbidden)
    assert embeddings.embed_texts([]) == []
