"""Steps of a marking. A check turns an answer into marks; a transform turns it into another
answer for later steps to read."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    SerializeAsAny,
    ValidationError,
    field_validator,
)
from pydantic.json_schema import SkipJsonSchema

from ..common.error_formatting import REQUIRED_MESSAGE, UNKNOWN_MESSAGE
from ..common.hints import ADVANCED_HINT, hint
from ..common.paths import Location, is_within, json_pointer, value_at
from ..common.problems import InvalidError, Problem, validation_problem, validation_problems
from ..common.text import capitalized
from ..questions.types import QuestionId
from .input_sources import InputSource, OtherAnswerSource, input_source
from .method import Method
from .outcome import Outcome
from .settings import IDENTIFIER, Settings, step_cardinality

METHODS_CONTEXT_KEY = "methods"
"""The validation context key of the methods by ref, for settings that hold steps."""
NOT_A_STEP_MESSAGE = "A step needs uses: and a method's name."
FEEDBACK_INSERTS = {
    "{{ answer }}": "Their answer",
    "{{ feedback }}": "The method's feedback",
    "{{ proportion | percent }}": "Proportion of the marks",
}
"""What feedback may include, by label."""
CHECK_PROPERTIES = ("weight", "feedback")
"""The properties only a check has."""
type InnerSteps = Mapping[Location, Step | None]
"""By place in a step's settings, such as ``("steps", 0)``, the step written there; None where
one can be added, as after the last of a list."""


class OutcomeFeedback(BaseModel):
    """Feedback for each outcome. One left empty is the method's own."""

    model_config = ConfigDict(extra="forbid", title="Feedback")

    correct: str = Field(default="", title="When it's correct")
    partly_correct: str = Field(default="", title="When it's partly correct")
    incorrect: str = Field(default="", title="When it's incorrect")

    def template_for(self, outcome: Outcome) -> str:
        if outcome.correct:
            return self.correct
        return self.partly_correct if outcome.proportion > 0 else self.incorrect


@dataclass(frozen=True)
class StepResolution:
    """What a step's ``uses`` and ``with`` come to among the methods it was loaded with."""

    method: Method | None
    gives_marks: bool
    """Whether it is a check; otherwise a transform, which gives an answer for later steps."""
    runs_inner_steps_in_order: bool
    """Whether the steps in a list of its settings may read the results of those before them."""
    inner_steps: InnerSteps
    data_problems: tuple[Problem, ...]
    """What is wrong with its data and that of the steps it holds, each from this step."""
    scope: Mapping[str, Any]
    """What its templates also see when a method file made it: the file's settings and item."""


class Step(BaseModel):
    """One step of a marking, as YAML holds it: a method with its settings. A property that isn't
    valid is at its default, and settings that aren't valid are None."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str | None = Field(
        default=None,
        pattern=IDENTIFIER,
        title="ID",
        description="Lets later steps read this one.",
    )
    name: str | None = Field(
        default=None,
        title="Name",
        description="What lists and feedback call this step.",
        examples=["Sign error"],
        json_schema_extra=hint(advanced=True),
    )
    question: QuestionId | None = Field(
        default=None,
        title="Question",
        description="The question this step reads and marks.",
    )
    input: str | None = Field(
        default=None,
        title="Input",
        description="What this step reads, if not the answer as it is.",
    )
    if_: str | None = Field(
        default=None,
        alias="if",
        title="Only when",
        json_schema_extra=hint(kind="condition", advanced=True),
    )
    weight: float = Field(
        default=1,
        allow_inf_nan=False,
        title="Weight",
        description="How much it counts. Zero lets later steps read it without giving marks, and "
        "a negative weight is a penalty.",
        examples=[2, 1, 0.5, 0.25, -1],
        json_schema_extra=ADVANCED_HINT,
    )
    """0 for a transform."""
    uses: str
    settings: SerializeAsAny[Settings] | None = Field(default=None, alias="with")
    feedback: str | OutcomeFeedback | None = Field(
        default=None,
        title="Feedback",
        description="What students read, instead of the method's own feedback.",
        json_schema_extra=hint(kind="feedback", template_parts=FEEDBACK_INSERTS, advanced=True),
    )
    resolution: SkipJsonSchema[InstanceOf[StepResolution]] = Field(exclude=True, repr=False)
    """What loading worked out; never written."""

    @field_validator("input")
    @classmethod
    def _validate_input(cls, text: str | None) -> str | None:
        input_source(text)
        return text

    @property
    def method_label(self) -> str:
        """What its method is called: as ``uses`` names it, if there's no such method."""
        method = self.resolution.method
        return method.label if method else self.uses

    @property
    def source(self) -> InputSource:
        """What its ``input`` names."""
        return input_source(self.input)

    def describe(self) -> str:
        """What it does, in a sentence or two of markdown."""
        method, settings = method_and_settings(self)
        return method.describe(settings)

    def described_in_list(self, weight_notes: Sequence[str]) -> str:
        """What it does, as an item of a list. A check comes after its question and its name, with
        ``weight_notes`` after them: ``**Q2, Sign error** (50% of the marks): …``."""
        description = self.describe()
        if not self.resolution.gives_marks:
            return description
        heading = ", ".join(part for part in (self.question, self.name) if part)
        notes_text = ", ".join(weight_notes)
        if heading and notes_text:
            return f"**{heading}** ({notes_text}): {description}"
        if title := heading or capitalized(notes_text):
            return f"**{title}:** {description}"
        return description

    def questions_marked(self, question_id: QuestionId | None = None) -> set[QuestionId]:
        method, settings = method_and_settings(self)
        return method.questions_marked(settings, self.question or question_id)

    def questions_read(self, question_id: QuestionId | None = None) -> set[QuestionId]:
        method, settings = method_and_settings(self)
        question_id = self.question or question_id
        questions_read = method.questions_read(settings, question_id)
        if question_id:
            questions_read |= {question_id}
        if isinstance(self.source, OtherAnswerSource):
            questions_read |= {self.source.question_id}
        return questions_read

    def needs_model_endpoint(self) -> bool:
        """Whether running it asks a language model."""
        method = self.resolution.method
        return (method is not None and method.needs_model_endpoint) or any(
            inner_step is not None and inner_step.needs_model_endpoint()
            for inner_step in self.resolution.inner_steps.values()
        )

    def dump(self) -> dict[str, Any]:
        """The step as YAML or JSON holds it: without what it leaves at the defaults."""
        step_data = self.model_dump(
            mode="json",
            by_alias=True,
            exclude_defaults=True,
            exclude=None if self.resolution.gives_marks else {"weight"},
        )
        return {key: value for key, value in step_data.items() if key != "with" or value}


STEP_PROPERTIES = {
    field_info.alias or name: field_info
    for name, field_info in Step.model_fields.items()
    if name != "resolution"
}
"""A step's properties as YAML names them, in the order it shows them."""
STEP_OWN_PROPERTIES = {
    key: field_info for key, field_info in STEP_PROPERTIES.items() if key not in ("uses", "with")
}
"""The properties a step has beside its method and its settings. No setting may share a name
with one."""


def method_and_settings(step: Step) -> tuple[Method, Settings]:
    """The method and settings of a step with nothing wrong with its data."""
    method = step.resolution.method
    assert method is not None
    assert step.settings is not None
    return method, step.settings


def needs_workflow_problem(step: Step) -> Problem:
    """What is wrong with ``step``, a transform, where only a check can be."""
    return Problem(f"Only a workflow can use {step.method_label}.")


def valid_step(
    data: object,
    methods_by_ref: Mapping[str, Method],
    scope: Mapping[str, Any] = MappingProxyType({}),
) -> Step:
    """``data`` as a step using ``methods_by_ref``; its templates also see ``scope``. Raises
    ``InvalidError`` with what is wrong with it."""
    if isinstance(data, Step):
        data = data.dump()
    if not isinstance(data, Mapping):
        raise InvalidError([Problem(NOT_A_STEP_MESSAGE)])
    if "uses" not in data:
        raise InvalidError([Problem("Choose a method.", field=("uses",))])
    step = load_step(data, methods_by_ref, scope)
    if step.resolution.data_problems:
        raise InvalidError(list(step.resolution.data_problems))
    return step


def load_step(
    data: Mapping[str, Any],
    methods_by_ref: Mapping[str, Method],
    scope: Mapping[str, Any] = MappingProxyType({}),
) -> Step:
    """``data``, which names a method with ``uses``, as a step, whatever is wrong with it: a
    property that isn't valid is left at its default, and settings that aren't valid leave it
    none."""
    uses = data["uses"]
    method = methods_by_ref.get(str(uses)) if uses else None
    settings_data = settings_of(data)
    settings: Settings | None = None
    inner_steps: InnerSteps = {}
    if method is None:
        message = f"There's no method {uses}." if uses else "Choose a method."
        settings_problems = [Problem(message, field=("uses",))]
    else:
        settings, inner_steps, settings_problems = _loaded_settings(method, data, methods_by_ref)
    resolution = StepResolution(
        method=method,
        gives_marks=method is None or method.gives_marks(settings_data),
        runs_inner_steps_in_order=method is not None
        and method.runs_inner_steps_in_order(settings_data),
        inner_steps=inner_steps,
        data_problems=tuple(settings_problems),
        scope=scope,
    )
    return _step_of({**data, "uses": str(uses or ""), "with": settings}, resolution)


def _step_of(step_data: Mapping[str, Any], resolution: StepResolution) -> Step:
    """The step ``step_data`` holds, with ``resolution`` and what is wrong with its properties
    before the problems it has. A property that isn't valid is left at its default; a transform
    has no weight or feedback, and no step is written with a resolution."""
    refused_keys = ("resolution", *(() if resolution.gives_marks else CHECK_PROPERTIES))
    read_data = {key: value for key, value in step_data.items() if key not in refused_keys}
    if not resolution.gives_marks:
        read_data["weight"] = 0
    problems = [Problem(UNKNOWN_MESSAGE, field=(key,)) for key in refused_keys if key in step_data]

    def step(valid_data: Mapping[str, Any], property_problems: list[Problem]) -> Step:
        data_problems = (*property_problems, *resolution.data_problems)
        return Step.model_validate(
            {**valid_data, "resolution": replace(resolution, data_problems=data_problems)}
        )

    try:
        return step(read_data, problems)
    except ValidationError as error:
        invalid_keys = {error_details["loc"][0] for error_details in error.errors()}
        valid_data = {key: value for key, value in read_data.items() if key not in invalid_keys}
        return step(valid_data, validation_problems(error, read_data) + problems)


def _loaded_settings(
    method: Method,
    data: Mapping[str, Any],
    methods_by_ref: Mapping[str, Method],
) -> tuple[Settings | None, InnerSteps, list[Problem]]:
    """The settings of the step ``data`` holds, none if they aren't valid, the steps written in
    them, and what is wrong with the settings, then with each step."""
    inner_steps = _loaded_inner_steps(method, settings_of(data), methods_by_ref)
    needs_settings = any(
        field_info.is_required() for field_info in method.settings.model_fields.values()
    )
    if "with" not in data and needs_settings:
        return None, inner_steps, [Problem(REQUIRED_MESSAGE, field=("with",), is_incomplete=True)]
    settings_data = data.get("with", {})
    if isinstance(settings_data, Mapping):
        settings_data = _with_steps(settings_data, inner_steps)
    settings: Settings | None = None
    problems: list[Problem] = []
    try:
        settings = method.settings.model_validate(
            settings_data, context={METHODS_CONTEXT_KEY: methods_by_ref}
        )
    except ValidationError as error:
        for error_details in error.errors():
            location = tuple(error_details["loc"])
            problem = validation_problem(error_details, settings_data)
            inner_step = inner_steps.get(location)
            place = next((place for place in inner_steps if is_within(location, place)), ())
            if inner_step is not None:
                # What a setting refuses of a step is wrong with the step, if nothing else is.
                resolution = inner_step.resolution
                if not resolution.data_problems:
                    step_problem = replace(problem, field=())
                    inner_steps[location] = inner_step.model_copy(
                        update={"resolution": replace(resolution, data_problems=(step_problem,))}
                    )
            elif len(location) > len(place) > 0:
                step_path, field = json_pointer(("with", *place)), location[len(place) :]
                problems.append(replace(problem, step_path=step_path, field=field))
            else:
                problems.append(_at_settings(problem, error_details["type"]))
    problems += [
        replace(problem, step_path=json_pointer(("with", *location)) + problem.step_path)
        for location, inner_step in inner_steps.items()
        if inner_step is not None
        for problem in inner_step.resolution.data_problems
    ]
    return settings, inner_steps, list(dict.fromkeys(problems))


def _at_settings(problem: Problem, error_type: str) -> Problem:
    """A problem pydantic found in a step's settings, at the step's field. A rule over all the
    settings is the step's; anything else about them as a whole is about its ``with``."""
    if problem.field:
        return problem
    return replace(problem, field=() if error_type == "value_error" else ("with",))


def _loaded_inner_steps(
    method: Method,
    settings_data: Mapping[str, Any],
    methods_by_ref: Mapping[str, Method],
) -> dict[Location, Step | None]:
    def inner_step(value: object) -> Step | None:
        if isinstance(value, Step):
            return value
        if isinstance(value, Mapping) and "uses" in value:
            return load_step(value, methods_by_ref)
        return None

    inner_steps: dict[Location, Step | None] = {}
    for name, field_info in method.settings.model_fields.items():
        value = settings_data.get(name)
        match step_cardinality(field_info):
            case "one":
                inner_steps[name,] = inner_step(value)
            case "many":
                for index, item in enumerate([*(value if isinstance(value, list) else ()), None]):
                    inner_steps[name, index] = inner_step(item)
    return inner_steps


def _with_steps(settings_data: Mapping[str, Any], inner_steps: InnerSteps) -> dict[str, Any]:
    """Settings as YAML holds them, with the steps loaded from them in their places."""
    settings_with_steps = {
        name: list(value) if isinstance(value, list) else value
        for name, value in settings_data.items()
    }
    for location, inner_step in inner_steps.items():
        if inner_step is not None:
            value_at(settings_with_steps, json_pointer(location[:-1]))[location[-1]] = inner_step
    return settings_with_steps


def settings_of(step_data: Mapping[str, Any]) -> Mapping[str, Any]:
    """A step's settings as YAML holds them; none unless they are a mapping."""
    settings = step_data.get("with")
    return settings if isinstance(settings, Mapping) else {}
