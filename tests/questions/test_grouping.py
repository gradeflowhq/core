import json
import re
import subprocess
import sys
from collections.abc import Sequence
from itertools import combinations
from math import acos, cos, sin, sqrt
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from rapidfuzz.fuzz import ratio

import gradeflow_core
from gradeflow_core import AnswerComparison, GradeFlowError, compare_answers
from gradeflow_core.ml import MISSING_MODEL_MESSAGE, embeddings


def connected_groups(size: int, pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    remaining = set(range(size))
    groups = []
    while remaining:
        component = {min(remaining)}
        while neighbors := {
            other
            for left, right in pairs
            for answer, other in ((left, right), (right, left))
            if answer in component and other not in component
        }:
            component.update(neighbors)
        remaining.difference_update(component)
        groups.append(sorted(component))
    return groups


def test_exact_answers_preserve_text_and_original_indices() -> None:
    texts = ["Water", "water!", "Water", "", " ", "", "water!"]
    comparison = compare_answers(texts)
    assert comparison.texts == tuple(texts)
    assert comparison.similarity == "exact"
    assert comparison.groups() == [[0, 2], [1, 6], [3, 5], [4]]
    assert texts == ["Water", "water!", "Water", "", " ", "", "water!"]


def test_normalized_equality_ignores_case_spacing_and_unicode_width() -> None:
    comparison = compare_answers(
        ["  Ｈ₂Ｏ  IS\n wet!", "h2o is wet!", "h2o is wet", "water"], similarity="normalized"
    )
    assert comparison.groups() == [[0, 1], [2], [3]]


@pytest.mark.parametrize("similarity", ["exact", "normalized", "spelling", "meaning"])
def test_empty_and_single_answer_need_no_model(
    similarity: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    load_model = Mock()
    monkeypatch.setattr(embeddings, "_model", load_model)
    assert compare_answers([], similarity=similarity).groups() == []
    assert compare_answers(["single", "single"], similarity=similarity).groups() == [[0, 1]]
    load_model.assert_not_called()


def test_meaning_without_the_ml_extra_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ("numpy", "scipy", "scipy.cluster.hierarchy", "scipy.spatial.distance"):
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(GradeFlowError, match=re.escape(MISSING_MODEL_MESSAGE)):
        compare_answers(["ice", "frozen water"], similarity="meaning")


def test_spelling_overlooks_case_and_spacing() -> None:
    comparison = compare_answers(["H2O", " h2o ", "water"], similarity="spelling")
    assert comparison.groups(threshold=1) == [[0, 1], [2]]


def test_spelling_compares_numbers_as_text_and_never_joins_a_blank() -> None:
    comparison = compare_answers(["9.81", "9.8", "", " "], similarity="spelling")
    assert comparison.groups(threshold=0.85) == [[0, 1], [2, 3]]
    assert comparison.groups(threshold=0) == [[0, 1], [2, 3]]


def test_spelling_joins_transitively_at_the_requested_threshold() -> None:
    comparison = compare_answers(["abc", "abd", "bde", "abc"], similarity="spelling")
    assert comparison.groups(threshold=2 / 3) == [[0, 1, 2, 3]]
    assert comparison.groups(threshold=0.7) == [[0, 3], [1], [2]]
    assert comparison.groups(threshold=1) == [[0, 3], [1], [2]]


def test_spelling_compact_comparison_matches_all_pairwise_thresholds() -> None:
    texts = ["water", "waters", "waiter", "later", "taller", "salt", "zzz", "water"]
    comparison = compare_answers(texts, similarity="spelling")
    scores = {
        (i, j): ratio(texts[i], texts[j]) / 100 for i, j in combinations(range(len(texts)), 2)
    }
    for threshold in {0, 0.1, 0.85, 1, *scores.values()}:
        expected = connected_groups(
            len(texts), [pair for pair, score in scores.items() if score and score >= threshold]
        )
        assert comparison.groups(threshold=threshold) == expected


def test_meaning_prepares_documents_once_then_groups_without_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    embed = Mock(return_value=np.asarray([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]], np.float32))
    monkeypatch.setattr(embeddings, "embed_texts", embed)
    texts = ["First ", "middle", "last", "opposite", "FIRST"]
    comparison = compare_answers(texts, similarity="meaning")
    embed.assert_called_once_with(["first", "middle", "last", "opposite"])
    restored = AnswerComparison.model_validate_json(comparison.model_dump_json())
    assert restored.texts == tuple(texts)
    assert restored.similarity == "meaning"
    assert restored.groups(threshold=0.8) == [[0, 1, 4], [2], [3]]
    assert restored.groups(threshold=0.6) == [[0, 1, 4], [2], [3]]
    assert restored.groups(threshold=1) == [[0, 4], [1], [2], [3]]
    matrix = np.asarray([[1, 0], [0.8, 0.6], [0, 1], [-1, 0], [1, 0]], dtype=np.float32)
    scores = matrix @ matrix.T
    for threshold in {0, 0.1, 0.85, 1, *(float(score) for score in scores.flat if 0 <= score <= 1)}:
        for indices in restored.groups(threshold=threshold):
            assert all(scores[i, j] >= threshold for i, j in combinations(indices, 2))
    embed.assert_called_once()


def test_meaning_groups_only_answers_similar_to_every_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    first_angle, second_angle = acos(0.95), acos(0.9)
    vectors = [
        [1.0, 0.0],
        [cos(first_angle), sin(first_angle)],
        [cos(first_angle + second_angle), sin(first_angle + second_angle)],
    ]
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: np.asarray(vectors, np.float32))
    comparison = compare_answers(["cold", "chilly", "cool"], similarity="meaning")
    assert comparison.groups(threshold=0.85) == [[0, 1], [2]]
    assert comparison.groups(threshold=0.7) == [[0, 1, 2]]


@pytest.mark.parametrize("score", [0.7, 0.9])
def test_meaning_threshold_uses_the_same_precision_as_document_scores(
    score: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    matrix = np.asarray([[1.0, 0.0], [score, sqrt(1 - score**2)]], dtype=np.float32)
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: matrix)
    comparison = compare_answers(["first", "second"], similarity="meaning")
    restored = AnswerComparison.model_validate_json(comparison.model_dump_json())
    scores = matrix @ matrix.T
    for threshold in (score, score + 1e-9, score + 1e-6):
        expected = [[0, 1]] if (scores >= threshold)[0, 1] else [[0], [1]]
        assert restored.groups(threshold) == expected


def test_restoring_and_regrouping_work_without_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    spelling_json = compare_answers(
        ["water", "waters", "later"], similarity="spelling"
    ).model_dump_json()
    vectors = np.asarray([[1, 0], [0.8, 0.6]], np.float32)
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: vectors)
    meaning_json = compare_answers(["ice", "frozen water"], similarity="meaning").model_dump_json()
    code = f"""
import sys
sys.modules.update(dict.fromkeys(['numpy', 'scipy', 'fastembed', 'onnxruntime'], None))
from gradeflow_core import AnswerComparison, compare_answers
assert compare_answers(['water', 'waters'], similarity='spelling').groups() == [[0, 1]]
restored = AnswerComparison.model_validate_json({spelling_json!r})
assert restored.groups(threshold=1) == [[0], [1], [2]]
assert restored.groups(threshold=0.8) == [[0, 1, 2]]
restored = AnswerComparison.model_validate_json({meaning_json!r})
assert restored.groups(threshold=0.85) == [[0], [1]]
assert restored.groups(threshold=0.75) == [[0, 1]]
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("inf"), float("nan")])
def test_threshold_must_be_a_finite_fraction(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        compare_answers(["a", "b"]).groups(threshold=threshold)


def test_unrecognized_comparison_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="similarity"):
        compare_answers(["a", "b"], similarity="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key_indices", [0]),
        ("key_indices", [0, 2]),
        ("key_indices", [0, -1]),
        ("similarity_links", [[0, 2, 0.9]]),
        ("similarity_links", [[0, 0, 0.9]]),
        ("similarity_links", [[0, 1, float("nan")]]),
        ("similarity_links", [[0, 1, 0.9], [1, 0, 0.8]]),
    ],
)
def test_serialized_comparison_rejects_invalid_membership_or_links(
    field: str, value: object
) -> None:
    comparison_json = json.loads(
        compare_answers(["water", "waters"], similarity="spelling").model_dump_json()
    )
    comparison_json[field] = value
    with pytest.raises(ValidationError):
        AnswerComparison.model_validate(comparison_json)


def test_a_serialized_comparison_has_linear_links_and_no_vectors() -> None:
    comparison = compare_answers([f"answer {index}" for index in range(100)], similarity="spelling")
    comparison_json = json.loads(comparison.model_dump_json())
    assert len(comparison_json["similarity_links"]) <= 99
    assert len(comparison_json["key_indices"]) == 100
    assert not hasattr(gradeflow_core, "embed_texts")
    assert not hasattr(gradeflow_core, "EMBEDDING_MODEL")
