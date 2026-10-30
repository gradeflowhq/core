import json
import subprocess
import sys
from collections.abc import Sequence
from itertools import combinations
from math import acos, cos, sin, sqrt
from typing import Any

import pytest
from pydantic import ValidationError
from rapidfuzz.fuzz import ratio

import gradeflow_core
from gradeflow_core import AnswerComparison, compare_answers
from gradeflow_core.ml import embeddings


def connected(size: int, pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    remaining = set(range(size))
    groups = []
    while remaining:
        component = {min(remaining)}
        while neighbors := {
            other
            for left, right in pairs
            for here, other in ((left, right), (right, left))
            if here in component and other not in component
        }:
            component.update(neighbors)
        remaining.difference_update(component)
        groups.append(sorted(component))
    return groups


def test_exact_answers_preserve_text_and_original_indices() -> None:
    texts = ["Water", "water!", "Water", "", " ", "", "water!"]
    comparison = compare_answers(texts)
    assert comparison.texts == tuple(texts)
    assert comparison.similar == "exact"
    assert comparison.groups() == [[0, 2], [1, 6], [3, 5], [4]]
    assert texts == ["Water", "water!", "Water", "", " ", "", "water!"]


def test_normalized_equality_ignores_case_punctuation_and_unicode_width() -> None:
    comparison = compare_answers(
        ["  Ｈ₂Ｏ  IS\n wet!", "h₂o is wet", "water", "waters"], similar="normalized"
    )
    assert comparison.groups() == [[0, 1], [2], [3]]


@pytest.mark.parametrize("similar", ["normalized", "spelling", "meaning"])
def test_numbers_and_blanks_only_group_when_equal(similar: Any) -> None:
    texts = ["-9.81", "9.81", "+9.81", "9.8", "1e-3", "1e3", "", " ", "!!!", "Water", "water!"]
    comparison = compare_answers(texts, similar=similar)
    assert comparison.groups(threshold=0) == [[0], [1], [2], [3], [4], [5], [6, 7, 8], [9, 10]]


@pytest.mark.parametrize("similar", ["exact", "normalized", "spelling", "meaning"])
def test_empty_and_single_answer_need_no_model(
    similar: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Equal answers need no model.")

    monkeypatch.setattr(embeddings, "_model", forbidden)
    assert compare_answers([], similar=similar).groups() == []
    assert compare_answers(["single", "single"], similar=similar).groups() == [[0, 1]]


def test_spelling_overlooks_case_and_punctuation() -> None:
    comparison = compare_answers(["H2O", "h2o.", "water"], similar="spelling")
    assert comparison.groups(threshold=1) == [[0, 1], [2]]


def test_spelling_joins_transitively_at_the_requested_threshold() -> None:
    comparison = compare_answers(["abc", "abd", "bde", "abc"], similar="spelling")
    assert comparison.groups(threshold=2 / 3) == [[0, 1, 2, 3]]
    assert comparison.groups(threshold=0.7) == [[0, 3], [1], [2]]
    assert comparison.groups(threshold=1) == [[0, 3], [1], [2]]


def test_spelling_compact_comparison_matches_all_pairwise_thresholds() -> None:
    texts = ["water", "waters", "waiter", "later", "taller", "salt", "zzz", "water"]
    comparison = compare_answers(texts, similar="spelling")
    scores = {
        (i, j): ratio(texts[i], texts[j]) / 100 for i, j in combinations(range(len(texts)), 2)
    }
    for threshold in {0, 0.1, 0.85, 1, *scores.values()}:
        expected = connected(
            len(texts), [pair for pair, score in scores.items() if score and score >= threshold]
        )
        assert comparison.groups(threshold=threshold) == expected


def test_meaning_prepares_documents_once_then_groups_without_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    calls: list[list[str]] = []

    def embed(texts: Sequence[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[1, 0], [0.8, 0.6], [0, 1], [-1, 0]]

    monkeypatch.setattr(embeddings, "embed_texts", embed)
    texts = ["First!", "middle", "last", "opposite", "FIRST"]
    comparison = compare_answers(texts, similar="meaning")
    assert calls == [["first", "middle", "last", "opposite"]]
    restored = AnswerComparison.model_validate_json(comparison.model_dump_json())
    assert restored.texts == tuple(texts)
    assert restored.similar == "meaning"
    assert restored.groups(threshold=0.8) == [[0, 1, 4], [2], [3]]
    assert restored.groups(threshold=0.6) == [[0, 1, 4], [2], [3]]
    assert restored.groups(threshold=1) == [[0, 4], [1], [2], [3]]
    matrix = np.asarray([[1, 0], [0.8, 0.6], [0, 1], [-1, 0], [1, 0]], dtype=np.float32)
    scores = matrix @ matrix.T
    for threshold in {0, 0.1, 0.85, 1, *(float(score) for score in scores.flat if 0 <= score <= 1)}:
        for indices in restored.groups(threshold=threshold):
            assert all(scores[i, j] >= threshold for i, j in combinations(indices, 2))
    assert calls == [["first", "middle", "last", "opposite"]]


def test_meaning_groups_only_answers_similar_to_every_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    ab, bc = acos(0.95), acos(0.9)
    vectors = [[1.0, 0.0], [cos(ab), sin(ab)], [cos(ab + bc), sin(ab + bc)]]
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: vectors)
    comparison = compare_answers(["cold", "chilly", "cool"], similar="meaning")
    assert comparison.groups(threshold=0.85) == [[0, 1], [2]]
    assert comparison.groups(threshold=0.7) == [[0, 1, 2]]


@pytest.mark.parametrize("score", [0.7, 0.9])
def test_meaning_threshold_uses_the_same_precision_as_document_scores(
    score: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    vectors = [[1.0, 0.0], [score, sqrt(1 - score**2)]]
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: vectors)
    comparison = compare_answers(["first", "second"], similar="meaning")
    restored = AnswerComparison.model_validate_json(comparison.model_dump_json())
    matrix = np.asarray(vectors, dtype=np.float32)
    scores = matrix @ matrix.T
    for threshold in (score, score + 1e-9, score + 1e-6):
        expected = [[0, 1]] if (scores >= threshold)[0, 1] else [[0], [1]]
        assert restored.groups(threshold) == expected


def test_meaning_never_compares_numbers_or_blanks(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("scipy")
    seen: list[str] = []

    def embed(texts: Sequence[str]) -> list[list[float]]:
        seen.extend(texts)
        return [[1, 0] for _ in texts]

    monkeypatch.setattr(embeddings, "embed_texts", embed)
    comparison = compare_answers(["9", "word", "another", "", "9"], similar="meaning")
    assert comparison.groups(threshold=0) == [[0, 4], [1, 2], [3]]
    assert seen == ["word", "another"]


def test_restoring_and_regrouping_work_without_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    payload = compare_answers(["water", "waters", "later"], similar="spelling").model_dump_json()
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [[1, 0], [0.8, 0.6]])
    meaning = compare_answers(["ice", "frozen water"], similar="meaning").model_dump_json()
    code = f"""
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {{'numpy', 'scipy', 'fastembed', 'onnxruntime'}}:
        raise ImportError('Optional dependency unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from gradeflow_core import AnswerComparison, compare_answers
assert compare_answers(['water', 'waters'], similar='spelling').groups() == [[0, 1]]
restored = AnswerComparison.model_validate_json({payload!r})
assert restored.groups(threshold=1) == [[0], [1], [2]]
assert restored.groups(threshold=0.8) == [[0, 1, 2]]
restored = AnswerComparison.model_validate_json({meaning!r})
assert restored.groups(threshold=0.85) == [[0], [1]]
assert restored.groups(threshold=0.75) == [[0, 1]]
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("inf"), float("nan")])
def test_threshold_must_be_a_finite_fraction(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        compare_answers(["a", "b"]).groups(threshold=threshold)


def test_unrecognized_comparison_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="similar"):
        compare_answers(["a", "b"], similar="unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("members", [0]),
        ("members", [0, 2]),
        ("members", [0, -1]),
        ("links", [[0, 2, 0.9]]),
        ("links", [[0, 0, 0.9]]),
        ("links", [[0, 1, float("nan")]]),
        ("links", [[0, 1, 0.9], [1, 0, 0.8]]),
    ],
)
def test_serialized_comparison_rejects_invalid_membership_or_links(
    field: str, value: object
) -> None:
    data = json.loads(compare_answers(["water", "waters"], similar="spelling").model_dump_json())
    data[field] = value
    with pytest.raises(ValidationError):
        AnswerComparison.model_validate(data)


def test_a_serialized_comparison_has_linear_links_and_no_vectors() -> None:
    comparison = compare_answers([f"answer {index}" for index in range(100)], similar="spelling")
    data = json.loads(comparison.model_dump_json())
    assert len(data["links"]) <= 99
    assert len(data["members"]) == 100
    assert not hasattr(gradeflow_core, "embed_texts")
    assert not hasattr(gradeflow_core, "EMBEDDING_MODEL")
