from collections.abc import Iterator, Mapping
from itertools import batched
from typing import Any, ClassVar, Literal, Self

from pydantic import Field, model_validator

from ..common.hints import hint
from ..common.problems import Problem, problem_error
from ..common.schemas import JsonDict, type_field
from .base import Question
from .multipart import MultipartQuestion
from .number import NumberQuestion
from .parts import (
    MIN_PARTS,
    VALUE_TYPE_LABELS,
    AnswerParts,
    part_properties,
    part_types,
    read_parts,
    unbracketed,
    value_question,
)
from .reading import ListReading, is_blank, read_list, reading_field
from .text import TextQuestion
from .types import PartValues, Value

ItemType = Literal["text", "number", "parts"]
ITEM_TYPE_LABELS: dict[str, str] = {**VALUE_TYPE_LABELS, "parts": "Parts"}
ITEMS_KEY = "items"
"""The key of a list answer's values that holds its items."""


class ListQuestion(Question):
    label: ClassVar[str] = "List"
    summary: ClassVar[str] = "Any number of items of one kind, such as the stages of mitosis."

    type: Literal["list"] = type_field("list")
    reading: ListReading = reading_field(ListReading)
    item: ItemType = Field(
        default="text", title="Item type", json_schema_extra=hint(labels=ITEM_TYPE_LABELS)
    )
    parts: AnswerParts = Field(
        default_factory=list,
        title="Parts",
        description="What each item is made of, in the order it is written.",
        json_schema_extra=hint(item_label="part", applies_when={"item": ["parts"]}),
    )

    @model_validator(mode="after")
    def _check_item_parts(self) -> Self:
        if self.item == "parts" and len(self.parts) < MIN_PARTS:
            raise problem_error(Problem(f"Add at least {MIN_PARTS}.", field="/parts"))
        return self

    def item_question(self) -> TextQuestion | NumberQuestion | MultipartQuestion:
        """A question reading one item of answers as this one reads it."""
        if self.item == "parts":
            return MultipartQuestion(
                max_marks=self.max_marks, reading=self.reading, parts=self.parts
            )
        return value_question(self.item, self.reading, self.max_marks)

    def python_type(self) -> str:
        if self.item == "parts":
            return (
                "list[dict[str, str | int | float | None]] | None, each with "
                f"{part_types(self.parts)}"
            )
        return f"list[{'str' if self.item == 'text' else 'int | float'}] | None"

    def read(self, answer: str) -> list[Value] | list[PartValues] | None:
        values = read_list(unbracketed(answer), self.reading)
        if values is None:
            return None
        if self.item == "parts":
            items = [
                read_parts(self.parts, item_values, self.reading)
                for item_values in self._grouped_into_items(values)
            ]
            return [
                item for item in items if any(value is not None for value in item.values())
            ] or None
        read = value_question(self.item, self.reading, self.max_marks).read
        read_values: list[Value] = [value for value in map(read, values) if value is not None]
        return read_values or None

    def answer_schema(self) -> JsonDict:
        item_schema: JsonDict = (
            {"type": "object", "properties": part_properties(self.parts)}
            if self.item == "parts"
            else {"type": "string"}
        )
        return {
            "type": "object",
            "properties": {ITEMS_KEY: {"type": "array", "title": "Items", "items": item_schema}},
        }

    def answer_values(self, answer: str) -> dict[str, Any]:
        written_values = (
            [value.strip() for value in unbracketed(answer).split(self.reading.separator)]
            if answer.strip()
            else []
        )
        if self.item != "parts":
            return {
                ITEMS_KEY: [value for value in written_values if not is_blank(value, self.reading)]
            }
        items = [
            {part.name: value for part, value in zip(self.parts, item_values, strict=True)}
            for item_values in self._grouped_into_items(written_values)
            if not all(is_blank(value, self.reading) for value in item_values)
        ]
        return {ITEMS_KEY: items}

    def answer_from(self, values: Mapping[str, Any]) -> str:
        items = values.get(ITEMS_KEY) or []
        written_items = (
            [[str(item.get(part.name) or "").strip() for part in self.parts] for item in items]
            if self.item == "parts"
            else [[str(item or "").strip()] for item in items]
        )
        return self.reading.separator.join(
            value for item_values in written_items if any(item_values) for value in item_values
        )

    def _grouped_into_items(self, values: list[str]) -> Iterator[tuple[str, ...]]:
        """``values`` grouped into items of the question's parts."""
        part_count = len(self.parts)
        if len(values) % part_count:
            raise ValueError(
                f"Expected {part_count} parts for each item, but got {len(values)} values."
            )
        return batched(values, part_count, strict=True)
