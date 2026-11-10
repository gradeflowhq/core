"""Answers compared once by how alike they are, then grouped at any threshold without comparing
them again."""

import struct
import unicodedata
from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rapidfuzz.fuzz import ratio

from ..common.exceptions import GradeFlowError
from ..ml import MISSING_MODEL_MESSAGE, embeddings

type Similarity = Literal["exact", "normalized", "spelling", "meaning"]
type _Index = Annotated[int, Field(ge=0, strict=True)]
type _Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
type _Link = tuple[_Index, _Index, _Score]


class AnswerComparison(BaseModel):
    """Prepared comparisons of ordered input texts. Store and restore this value as JSON;
    ``groups`` returns input indices without loading a model or repeating comparisons.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    texts: tuple[str, ...]
    similarity: Similarity
    key_indices: tuple[_Index, ...] = Field(repr=False)
    similarity_links: tuple[_Link, ...] = Field(repr=False)

    @model_validator(mode="after")
    def _check_structure(self) -> Self:
        key_count = len(set(self.key_indices))
        if len(self.key_indices) != len(self.texts) or set(self.key_indices) != set(
            range(key_count)
        ):
            raise ValueError("Comparison membership must cover every input text.")
        if self.similarity in ("exact", "normalized") and self.similarity_links:
            raise ValueError("Equal-answer comparisons cannot contain similarity links.")
        disjoint_sets = _DisjointSets(key_count)
        for left, right, _ in self.similarity_links:
            if max(left, right) >= key_count or disjoint_sets.find(left) == disjoint_sets.find(
                right
            ):
                raise ValueError("Comparison links must join known answers without forming a loop.")
            disjoint_sets.union(left, right)
        return self

    def groups(self, threshold: float = 0.85) -> list[list[int]]:
        """Input indices in connected groups at ``threshold`` (0 to 1), in input order.

        Equal answers always group together. Similar spelling joins chains of similar answers; in a
        group of similar meaning, every answer is that similar to every other.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("The comparison threshold must be between 0 and 1.")
        if self.similarity == "meaning":
            # Cosine scores and their inclusive thresholds use the same float32 precision.
            threshold = struct.unpack("f", struct.pack("f", threshold))[0]
        disjoint_sets = _DisjointSets(len(set(self.key_indices)))
        for left, right, score in self.similarity_links:
            if score >= threshold:
                disjoint_sets.union(left, right)
        groups_by_root: dict[int, list[int]] = {}
        for index, key_index in enumerate(self.key_indices):
            groups_by_root.setdefault(disjoint_sets.find(key_index), []).append(index)
        return list(groups_by_root.values())


def compare_answers(
    answers: Sequence[str], *, similarity: Similarity = "exact"
) -> AnswerComparison:
    """Prepare ordered answers for repeated grouping at different thresholds.

    Supply canonical answers from ``Question.normalize``. Exact comparison preserves text; other
    modes overlook case and spacing. ``normalized`` compares equality only. Meaning comparison needs
    the ``ml`` extra, raising ``GradeFlowError`` without it, and performs document inference here.
    """
    index_of_key: dict[str, int] = {}
    key_indices = []
    for text in answers:
        key = text if similarity == "exact" else _normalized(text)
        key_indices.append(index_of_key.setdefault(key, len(index_of_key)))
    similarity_links: list[_Link] = []
    if similarity in ("spelling", "meaning") and len(index_of_key) > 1:
        keys = list(index_of_key)
        similarity_links = (
            _meaning_links(keys) if similarity == "meaning" else _spelling_links(keys)
        )
    return AnswerComparison(
        texts=tuple(answers),
        similarity=similarity,
        key_indices=tuple(key_indices),
        similarity_links=tuple(similarity_links),
    )


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _spelling_links(keys: list[str]) -> list[_Link]:
    """A maximum spanning forest preserves connected groups at every threshold, in n-1 links."""
    unlinked_indices = set(range(1, len(keys)))
    best_scores = [ratio(keys[0], key) / 100 for key in keys]
    nearest_indices = [0] * len(keys)
    links = []
    while unlinked_indices:
        selected_index = max(unlinked_indices, key=lambda index: (best_scores[index], -index))
        unlinked_indices.remove(selected_index)
        if best_scores[selected_index] > 0:
            links.append(
                (nearest_indices[selected_index], selected_index, best_scores[selected_index])
            )
        for index in unlinked_indices:
            score = ratio(keys[selected_index], keys[index]) / 100
            if score > best_scores[index]:
                best_scores[index] = score
                nearest_indices[index] = selected_index
    return links


def _meaning_links(keys: list[str]) -> list[_Link]:
    """Complete-linkage merges as links between the merged groups' first answers, scored by
    their least similar pair."""
    try:
        import numpy as np
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist

        vectors = embeddings.embed_texts(keys)
    except ImportError:
        raise GradeFlowError(MISSING_MODEL_MESSAGE) from None
    tree = linkage(pdist(vectors, "cosine"), method="complete")
    first_answer_indices = list(range(len(keys)))
    links = []
    for left, right, distance, _ in tree:
        first_answer_indices.append(first_answer_indices[int(left)])
        if distance <= 1:
            links.append(
                (
                    first_answer_indices[int(left)],
                    first_answer_indices[int(right)],
                    float(np.float32(1 - distance)),
                )
            )
    return links


class _DisjointSets:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(left)] = self.find(right)
