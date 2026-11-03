import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, call

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
    fastembed = Mock()
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    embeddings.setup()
    embeddings._model()
    embeddings._model()
    directory = str(tmp_path / "models")
    assert fastembed.TextEmbedding.call_args_list == [
        call(embeddings.EMBEDDING_MODEL, cache_dir=directory, lazy_load=True),
        call(embeddings.EMBEDDING_MODEL, cache_dir=directory, lazy_load=False),
    ]


def test_grouping_embeddings_are_the_model_vectors_one_row_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    rows = np.array([[0.6, 0.8], [-1.0, 0.0]], dtype=np.float32)
    model = Mock()
    model.embed.return_value = iter(rows)
    monkeypatch.setattr(embeddings, "_model", lambda: model)
    texts = ("first", "second")
    vectors = embeddings.embed_texts(texts)
    assert vectors.dtype == np.float32
    assert vectors.tolist() == rows.tolist()
    model.embed.assert_called_once_with(["first", "second"])


def test_empty_embedding_batch_does_not_load_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    load_model = Mock()
    monkeypatch.setattr(embeddings, "_model", load_model)
    assert embeddings.embed_texts([]).size == 0
    load_model.assert_not_called()
