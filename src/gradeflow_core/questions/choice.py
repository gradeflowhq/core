from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, Self

from pydantic import Field, ValidationInfo, field_validator

from ..common.schemas import JsonDict, type_field
from ..common.text import quoted
from .base import Question
from .reading import ListReading, is_blank, read_list, reading_field, sorted_options


class ChoiceQuestion(Question):
    label: ClassVar[str] = "Choice"
    summary: ClassVar[str] = "One or more options from a list, such as B, or A and C."
    new_question_data: ClassVar[dict[str, Any]] = {"options": ["a", "b"]}

    type: Literal["choice"] = type_field("choice")
    reading: ListReading = reading_field(ListReading)
    options: tuple[str, ...] = Field(
        min_length=1,
        strict=True,
        title="Options",
        description="The options students can choose from.",
    )
    allow_multiple: bool = Field(
        default=False,
        title="Allow more than one option",
        description="Students may choose several options.",
    )

    @field_validator("options", mode="before")
    @classmethod
    def _read_like_answers(cls, options: object, info: ValidationInfo) -> object:
        """Options are read like answers, so they compare equal to them."""
        if isinstance(options, list):
            options = tuple(options)
        if not isinstance(options, tuple) or not all(isinstance(option, str) for option in options):
            return options
        reading = info.data.get("reading") or ListReading()
        read_options: list[str] = []
        for option in options:
            if reading.separator in option:
                raise ValueError(
                    f"Options can't have the separator in them: {quoted(option)} has "
                    f"{quoted(reading.separator)}."
                )
            option = reading.normalized(option)
            if not option:
                raise ValueError("Options can't be blank.")
            if option in read_options:
                raise ValueError(f"Options must be different: {quoted(option)} appears twice.")
            read_options.append(option)
        return tuple(read_options)

    def python_type(self) -> str:
        return "set[str]"

    def read(self, answer: str) -> frozenset[str] | None:
        values = read_list(answer, self.reading)
        if values is None:
            return None
        return frozenset(
            value if is_blank(value, self.reading) else self.reading.normalized(value)
            for value in values
        )

    def answer_schema(self) -> JsonDict:
        option_schema = {"type": "string", "enum": list(self.options)}
        if self.allow_multiple:
            property_schema = {
                "type": "array",
                "title": "Options",
                "items": option_schema,
                "uniqueItems": True,
            }
        else:
            property_schema = {"title": "Option", **option_schema}
        return {"type": "object", "properties": {self._answer_property: property_schema}}

    def answer_values(self, answer: str) -> dict[str, Any]:
        answer_options = {
            option for option in self.read(answer) or () if not is_blank(option, self.reading)
        }
        if unlisted := sorted(answer_options - set(self.options)):
            raise ValueError(f"{quoted(unlisted[0])} isn't one of the options.")
        if len(answer_options) > 1 and not self.allow_multiple:
            raise ValueError("Only one option can be chosen.")
        chosen_options = [option for option in self.options if option in answer_options]
        if self.allow_multiple:
            return {self._answer_property: chosen_options}
        return {self._answer_property: chosen_options[0] if chosen_options else None}

    def answer_from(self, values: Mapping[str, Any]) -> str:
        property_value = values.get(self._answer_property) or []
        chosen_options = [property_value] if isinstance(property_value, str) else property_value
        return self.reading.separator.join(
            option for option in self.options if option in chosen_options
        )

    @property
    def _answer_property(self) -> str:
        return "options" if self.allow_multiple else "option"

    def accepting(self, answers: Sequence[str]) -> Self:
        answered_options = {
            option for answer in answers for option in self.read(answer) or () if option
        }
        new_options = sorted_options(answered_options.difference(self.options))
        return self.model_copy(update={"options": (*self.options, *new_options)})
