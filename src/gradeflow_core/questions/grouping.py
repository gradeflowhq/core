"""Answers compared once by how alike they are, then grouped at any threshold without comparing
them again."""

import re
import struct
import unicodedata
from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rapidfuzz.fuzz import ratio

from ..ml import embeddings

type Similarity = Literal["exact", "normalized", "spelling", "meaning"]
type _Index = Annotated[int, Field(ge=0, strict=True)]
type _Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
type _Link = tuple[_Index, _Index, _Score]

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


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
    def _valid_structure(self) -> Self:
        key_count = len(set(self.key_indices))
        if len(self.key_indices) != len(self.texts) or set(self.key_indices) != set(
            range(key_count)
        ):
            raise ValueError("Comparison membership must cover every input text.")
        if self.similarity in ("exact", "normalized") and self.similarity_links:
            raise ValueError("Equal-answer comparisons cannot contain similarity links.")
        sets = _DisjointSets(key_count)
        for left, right, _ in self.similarity_links:
            if max(left, right) >= key_count or sets.find(left) == sets.find(right):
                raise ValueError("Comparison links must join known answers without forming a loop.")
            sets.union(left, right)
        return self

    def groups(self, threshold: float = 0.85) -> list[list[int]]:
        """Input indices in connected groups at ``threshold`` (0 to 1), in input order.

        Equal answers always group together; numbers and blanks never join different answers.
        Similar spelling joins chains of similar answers; in a group of similar meaning, every
        answer is that similar to every other.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("The comparison threshold must be between 0 and 1.")
        if self.similarity == "meaning":
            # Cosine scores and their inclusive thresholds use the same float32 precision.
            threshold = struct.unpack("f", struct.pack("f", threshold))[0]
        sets = _DisjointSets(len(set(self.key_indices)))
        for left, right, score in self.similarity_links:
            if score >= threshold:
                sets.union(left, right)
        groups_by_root: dict[int, list[int]] = {}
        for index, key_index in enumerate(self.key_indices):
            groups_by_root.setdefault(sets.find(key_index), []).append(index)
        return list(groups_by_root.values())


def compare_answers(
    answers: Sequence[str], *, similarity: Similarity = "exact"
) -> AnswerComparison:
    """Prepare ordered answers for repeated grouping at different thresholds.

    Supply canonical answers from ``Question.normalize``. Exact comparison preserves text;
    other modes overlook case, punctuation and spacing. ``normalized`` compares equality
    only. Meaning comparison requires the ``ml`` extra and performs document inference here.
    """
    index_of_key: dict[str, int] = {}
    key_indices = []
    for text in answers:
        normalized = text if similarity == "exact" else _normalized(text)
        key_indices.append(index_of_key.setdefault(normalized, len(index_of_key)))
    similarity_links: list[_Link] = []
    if similarity in ("spelling", "meaning"):
        comparable = {
            key: index for key, index in index_of_key.items() if key and not _NUMBER.fullmatch(key)
        }
        if len(comparable) > 1:
            keys, indices = list(comparable), list(comparable.values())
            key_links = _meaning_links(keys) if similarity == "meaning" else _spelling_links(keys)
            similarity_links = [
                (indices[left], indices[right], score) for left, right, score in key_links
            ]
    return AnswerComparison(
        texts=tuple(answers),
        similarity=similarity,
        key_indices=tuple(key_indices),
        similarity_links=tuple(similarity_links),
    )


def _normalized(text: str) -> str:
    if _NUMBER.fullmatch(text.strip()):
        return text.strip()
    text = _PUNCTUATION.sub("", unicodedata.normalize("NFKC", text).casefold())
    return _SPACE.sub(" ", text).strip()


def _spelling_links(keys: list[str]) -> list[_Link]:
    """A maximum spanning forest preserves connected groups at every threshold, in n-1 links."""
    remaining = set(range(1, len(keys)))
    best_score = [ratio(keys[0], key) / 100 for key in keys]
    nearest = [0] * len(keys)
    links = []
    while remaining:
        selected = max(remaining, key=lambda index: (best_score[index], -index))
        remaining.remove(selected)
        if best_score[selected] > 0:
            links.append((nearest[selected], selected, best_score[selected]))
        for index in remaining:
            score = ratio(keys[selected], keys[index]) / 100
            if score > best_score[index]:
                best_score[index] = score
                nearest[index] = selected
    return links


def _meaning_links(keys: list[str]) -> list[_Link]:
    """Complete-linkage merges as links between the merged groups' first answers, scored by
    their least similar pair."""
    import numpy as np
    from scipy.cluster.hierarchy import linkage

    vectors = np.asarray(embeddings.embed_texts(keys), dtype=np.float32)
    scores = (vectors @ vectors.T)[np.triu_indices(len(keys), 1)]
    tree = linkage(np.clip(1 - scores.astype(np.float64), 0, 2), method="complete")
    first_answer = list(range(len(keys)))
    links = []
    for left, right, distance, _ in tree:
        first_answer.append(first_answer[int(left)])
        if distance <= 1:
            links.append(
                (first_answer[int(left)], first_answer[int(right)], float(np.float32(1 - distance)))
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
