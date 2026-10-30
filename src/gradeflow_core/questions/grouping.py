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

type Similar = Literal["exact", "normalized", "spelling", "meaning"]
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
    similar: Similar
    members: tuple[_Index, ...] = Field(repr=False)
    links: tuple[_Link, ...] = Field(repr=False)

    @model_validator(mode="after")
    def _valid_structure(self) -> Self:
        size = len(set(self.members))
        if len(self.members) != len(self.texts) or set(self.members) != set(range(size)):
            raise ValueError("Comparison membership must cover every input text.")
        if self.similar in ("exact", "normalized") and self.links:
            raise ValueError("Equal-answer comparisons cannot contain similarity links.")
        roots = _Roots(size)
        for left, right, _ in self.links:
            if max(left, right) >= size or roots.find(left) == roots.find(right):
                raise ValueError("Comparison links must form a forest over its members.")
            roots.join(left, right)
        return self

    def groups(self, threshold: float = 0.85) -> list[list[int]]:
        """Input indices in connected groups at ``threshold`` (0 to 1), in input order.

        Equal answers always group together; numbers and blanks never join different answers.
        Similar spelling joins chains of similar answers; in a group of similar meaning, every
        answer is that similar to every other.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("The comparison threshold must be between 0 and 1.")
        if self.similar == "meaning":
            # Cosine scores and their inclusive thresholds use the same float32 precision.
            threshold = struct.unpack("f", struct.pack("f", threshold))[0]
        roots = _Roots(len(set(self.members)))
        for left, right, score in self.links:
            if score >= threshold:
                roots.join(left, right)
        grouped: dict[int, list[int]] = {}
        for index, member in enumerate(self.members):
            grouped.setdefault(roots.find(member), []).append(index)
        return list(grouped.values())


def compare_answers(answers: Sequence[str], *, similar: Similar = "exact") -> AnswerComparison:
    """Prepare ordered answers for repeated grouping at different thresholds.

    Supply canonical answers from ``Question.normalize``. Exact comparison preserves text;
    other modes overlook case, punctuation and spacing. ``normalized`` compares equality
    only. Meaning comparison requires the ``ml`` extra and performs document inference here.
    """
    distinct: dict[str, int] = {}
    members = []
    for text in answers:
        key = text if similar == "exact" else _normalise(text)
        members.append(distinct.setdefault(key, len(distinct)))
    links: list[_Link] = []
    if similar in ("spelling", "meaning"):
        eligible = {
            key: index for key, index in distinct.items() if key and not _NUMBER.fullmatch(key)
        }
        if len(eligible) > 1:
            keys, indices = list(eligible), list(eligible.values())
            found = _meaning_links(keys) if similar == "meaning" else _spelling_links(keys)
            links = [(indices[left], indices[right], score) for left, right, score in found]
    return AnswerComparison(
        texts=tuple(answers), similar=similar, members=tuple(members), links=tuple(links)
    )


def _normalise(text: str) -> str:
    if _NUMBER.fullmatch(text.strip()):
        return text.strip()
    text = _PUNCTUATION.sub("", unicodedata.normalize("NFKC", text).casefold())
    return _SPACE.sub(" ", text).strip()


def _spelling_links(keys: list[str]) -> list[_Link]:
    """A maximum spanning forest preserves connected groups at every threshold, in n-1 links."""
    remaining = set(range(1, len(keys)))
    best = [ratio(keys[0], key) / 100 for key in keys]
    source = [0] * len(keys)
    links = []
    while remaining:
        selected = max(remaining, key=lambda index: (best[index], -index))
        remaining.remove(selected)
        if best[selected] > 0:
            links.append((source[selected], selected, best[selected]))
        for index in remaining:
            score = ratio(keys[selected], keys[index]) / 100
            if score > best[index]:
                best[index] = score
                source[index] = selected
    return links


def _meaning_links(keys: list[str]) -> list[_Link]:
    """Complete-linkage merges as links between the merged groups' first answers, scored by
    their least similar pair."""
    import numpy as np
    from scipy.cluster.hierarchy import linkage

    vectors = np.asarray(embeddings.embed_texts(keys), dtype=np.float32)
    scores = (vectors @ vectors.T)[np.triu_indices(len(keys), 1)]
    tree = linkage(np.clip(1 - scores.astype(np.float64), 0, 2), method="complete")
    first = list(range(len(keys)))
    links = []
    for left, right, distance, _ in tree:
        first.append(first[int(left)])
        if distance <= 1:
            links.append((first[int(left)], first[int(right)], float(np.float32(1 - distance))))
    return links


class _Roots:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, left: int, right: int) -> None:
        self.parent[self.find(left)] = self.find(right)
