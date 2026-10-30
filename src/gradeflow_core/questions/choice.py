from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal, Self

from natsort import natsorted
from pydantic import Field, PlainSerializer, ValidationInfo, field_validator

from ..common.schemas import JsonDict, tag
from ..common.text import quoted
from .base import Question
from .reading import ListReading, read_list, reading_field

Options = Annotated[
    frozenset[str], PlainSerializer(sorted, return_type=list[str], when_used="json")
]


def choice_reading() -> ListReading:
    """How choices are read unless set otherwise: separated by commas, ignoring capitals."""
    return ListReading(ignore_case=True)


class ChoiceQuestion(Question):
    label: ClassVar[str] = "Choice"
    summary: ClassVar[str] = "One or more options from a list, such as B, or A and C."
    starting: ClassVar[dict[str, Any]] = {"options": ["a", "b"]}

    type: Literal["choice"] = tag("choice")
    reading: ListReading = reading_field(choice_reading)
    options: Options = Field(
        ..., min_length=1, title="Options", description="The options students can choose from."
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
        if not isinstance(options, list | tuple | set | frozenset) or not all(
            isinstance(option, str) for option in options
        ):
            return options
        reading = info.data.get("reading") or choice_reading()
        read: list[str] = []
        for option in map(reading.option, options):
            if not option:
                raise ValueError("Options can't be blank.")
            if option in read:
                raise ValueError(f"Options must be different: {quoted(option)} appears twice.")
            read.append(option)
        return frozenset(read)

    def python_type(self) -> str:
        return "set[str]"

    def read(self, answer: str) -> frozenset[str] | None:
        values = read_list(answer, self.reading)
        return None if values is None else frozenset(values)

    def answer_schema(self) -> JsonDict:
        options = {"type": "string", "enum": natsorted(self.options)}
        if self.allow_multiple:
            value = {"type": "array", "title": "Options", "items": options, "uniqueItems": True}
        else:
            value = {"title": "Option", **options}
        return {"type": "object", "properties": {self._key: value}}

    def answer_values(self, answer: str) -> dict[str, Any]:
        chosen = natsorted((self.read(answer) or frozenset()) & self.options)
        if self.allow_multiple:
            return {self._key: chosen}
        return {self._key: chosen[0] if chosen else None}

    def answer_from(self, values: Mapping[str, Any]) -> str:
        chosen = values.get(self._key) or []
        options = [chosen] if isinstance(chosen, str) else chosen
        return self.reading.separator.join(natsorted(options))

    @property
    def _key(self) -> str:
        return "options" if self.allow_multiple else "option"

    def accepting(self, answers: Sequence[str]) -> Self:
        seen = {option for answer in answers for option in self.read(answer) or () if option}
        return self.model_copy(update={"options": self.options | seen})
