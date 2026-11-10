"""Steps of a marking. A check turns an answer into marks; a transform turns it into another
answer for later steps to read."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic.fields import FieldInfo
from pydantic_core import InitErrorDetails, PydanticCustomError

from ..common.exceptions import InvalidError
from ..common.hints import ADVANCED_HINT, hint, hint_of
from ..common.model_service import ModelEndpoint
from ..common.problems import Problem, invalid_errors
from ..common.text import capitalized
from ..questions.base import Question
from ..questions.question_set import Answers, QuestionSet, value_of
from ..questions.reading import write_answer
from ..questions.types import Answer, QuestionId, Unreadable
from .input_sources import StepOutputSource, input_source
from .outcome import (
    CheckOutput,
    Detail,
    Outcome,
    QuestionOutcomes,
    SkippedStep,
    StepResult,
    TransformResult,
    unanswered_outcome,
)
from .position import Position
from .problems import located_problems
from .settings import IDENTIFIER, Settings, item_field
from .templates import TemplateError, condition_expression, evaluate, names_read, render

if TYPE_CHECKING:
    from .collection import Methods
    from .method import Method, Progress

METHODS_CONTEXT_KEY = "methods"
"""The validation context key of the prepared methods by ref."""
FEEDBACK_INSERTS = {
    "{{ answer }}": "Their answer",
    "{{ feedback }}": "The method's feedback",
    "{{ proportion | percent }}": "Proportion of the marks",
}
"""What feedback may include, by label."""


@dataclass(frozen=True)
class Context:
    """What marking one student needs: their answers, the questions and the results of earlier
    steps."""

    answers: Answers
    questions: QuestionSet
    written_answers: Mapping[QuestionId, str]
    """What the student wrote, for feedback to show as ``{{ answer }}``."""
    earlier_results: Mapping[str, StepResult] = field(default_factory=dict)
    source_answers: Mapping[QuestionId, tuple[Answer | Unreadable, Question]] = field(
        default_factory=dict
    )
    """The original answer and question of each answer replaced by a step's input."""
    model_endpoint: ModelEndpoint | None = None
    report_progress: Callable[[Progress], None] = lambda _: None

    def with_answer(
        self, question_id: QuestionId, answer: Answer | Unreadable, question: Question
    ) -> Context:
        """This context, with ``question_id`` answered by ``answer`` and read as ``question``, and
        feedback showing ``answer``."""
        marked_question = self.questions[question_id]
        input_question = question.model_copy(update={"max_marks": marked_question.max_marks})
        written_answer = "" if isinstance(answer, Unreadable) else write_answer(answer)
        return replace(
            self,
            answers={**self.answers, question_id: answer},
            written_answers={**self.written_answers, question_id: written_answer},
            questions=QuestionSet.model_construct(
                root={**self.questions.root, question_id: input_question}
            ),
            source_answers={
                question_id: (self.answers.get(question_id), marked_question),
                **self.source_answers,
            },
        )

    def with_value_as_answer(self, question_id: QuestionId, value_number: int) -> Context:
        """This context, with value ``value_number`` of the answer to ``question_id`` as its
        answer."""
        source_answer, source_question = self.source_answers.get(
            question_id, (self.answers.get(question_id), self.questions[question_id])
        )
        question = source_question.value_question(value_number)
        if question is None:
            return self
        return self.with_answer(question_id, value_of(source_answer, value_number), question)

    def with_step_result(self, step_id: str, result: StepResult) -> Context:
        """This context, with a step's result among the earlier ones."""
        return replace(self, earlier_results={**self.earlier_results, step_id: result})


def _check_result(outcomes: Mapping[QuestionId, Outcome]) -> StepResult:
    """A check's result over every question it marks."""
    output: CheckOutput = {
        "correct": bool(outcomes) and all(outcome.correct for outcome in outcomes.values()),
        "proportion": min((outcome.proportion for outcome in outcomes.values()), default=0.0),
        "feedback": "\n".join(
            outcome.feedback for outcome in outcomes.values() if outcome.feedback
        ),
        "marked": all(outcome.marked for outcome in outcomes.values()),
    }
    is_skipped = bool(outcomes) and all(outcome.is_skipped for outcome in outcomes.values())
    return StepResult(output, is_skipped)


@dataclass(frozen=True)
class ResolvedInput:
    """A step's input read for one student. ``short_circuit_outcome`` replaces running the step."""

    context: Context
    short_circuit_outcome: Outcome | None = None


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


class Step(BaseModel, ABC):
    """One step of a marking: a method with its settings."""

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    method: ClassVar[Method[Any]]

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
    uses: str = Field(title="Method")
    with_: Settings = Field(alias="with")

    _scope: dict[str, Any] = PrivateAttr(default_factory=dict)
    """What its templates also see when a method file made it: the file's settings and item."""
    _methods: Mapping[str, Method[Any]] = PrivateAttr(default_factory=dict)
    _cached_expansion: Step | None = PrivateAttr(default=None)
    """The workflow a method file expands it to, once worked out."""

    @classmethod
    def load(cls, data: object, *, methods: Methods | None = None) -> Step:
        """``data`` as a step, its nested steps too, using ``methods``: the built-ins unless
        given. Raises ``InvalidError`` if it isn't a step."""
        from .catalogue import BUILTINS  # built from steps, so not importable before them

        with invalid_errors(data):
            return load_step(data, (BUILTINS if methods is None else methods).by_ref)

    @field_validator("input")
    @classmethod
    def _validate_input(cls, text: str | None) -> str | None:
        input_source(text)
        return text

    @property
    def marks_weight(self) -> float:
        """How much it counts towards the marks of what holds it."""
        return 0.0

    @property
    def has_weight(self) -> bool:
        """Whether it counts towards the marks of what holds it."""
        return self.marks_weight != 0

    @abstractmethod
    def run(
        self, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> tuple[list[Context], list[QuestionOutcomes]]:
        """Each student's context after it, and the outcomes it gave: none if it doesn't mark."""

    def described_in_list(self, weight_notes: Sequence[str]) -> str:
        """What it does, as an item of a list, with a check's ``weight_notes``."""
        return self.describe()

    def problems_in_list(self, step_ids_read_later: Set[str], index: int) -> Iterator[Problem]:
        """What is wrong with it at ``index`` of a list, given the IDs later steps read."""
        return iter(())

    def describe(self) -> str:
        """What it does, in a sentence or two of markdown."""
        return self.method.describe(self)

    def questions_marked(self, question_id: QuestionId | None = None) -> set[QuestionId]:
        return self.method.questions_marked(self, self.question or question_id)

    def questions_read(self, question_id: QuestionId | None = None) -> set[QuestionId]:
        question_id = self.question or question_id
        questions_read = self.method.questions_read(self, question_id) | (
            {question_id} if question_id else set()
        )
        return questions_read | input_source(self.input).questions_read()

    def needs_model_endpoint(self) -> bool:
        """Whether running it asks a language model."""
        if self.method.needs_model_endpoint:
            return True
        try:
            return any(
                inner_step.needs_model_endpoint()
                for _, inner_step in self.method.expanded_steps(self)
            )
        except InvalidError, TemplateError:
            return False

    def problems(self, questions: QuestionSet) -> list[Problem]:
        """What stops it marking ``questions``, each where it is."""
        return located_problems(self, questions)

    def dump(self) -> dict[str, Any]:
        """The step as YAML or JSON holds it: without what it leaves at the defaults."""
        data = self.model_dump(mode="json", exclude_defaults=True)
        if data.get("with") == {}:
            data.pop("with")
        return {key: data[key] for key in STEP_KEYS if key in data}

    def marked_question_id(self, question_id: QuestionId | None) -> QuestionId | None:
        """The key of the answer it reads and marks, at ``question_id`` unless it names its own."""
        return input_source(self.input).marked_question_id(self.question or question_id)

    def _resolved_input(self, context: Context, question_id: QuestionId | None) -> ResolvedInput:
        """The step's input read and its condition tested, for one student."""
        input_context = input_source(self.input).resolved_context(context, question_id)
        if input_context is None:
            return ResolvedInput(context, Outcome(0.0, is_skipped=True))
        short_circuit_outcome = self._condition_short_circuit_outcome(input_context)
        if short_circuit_outcome is None and question_id is not None:
            answer = input_context.answers.get(question_id)
            short_circuit_outcome = unanswered_outcome(answer)
            if answer is None and self.method.marks_blank_answers:
                short_circuit_outcome = None
        return ResolvedInput(input_context, short_circuit_outcome)

    def _resolved_inputs_and_contexts_to_run(
        self, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> tuple[list[ResolvedInput], list[Context]]:
        """Each student's resolved input, and the contexts of those the method runs for."""
        resolved_inputs = [self._resolved_input(context, question_id) for context in contexts]
        return resolved_inputs, [
            resolved_input.context
            for resolved_input in resolved_inputs
            if resolved_input.short_circuit_outcome is None
        ]

    def _condition_short_circuit_outcome(self, context: Context) -> Outcome | None:
        if self.if_ is None:
            return None
        expression = condition_expression(self.if_)
        for step_id in sorted(names_read(self.if_, "steps")):
            if step_id not in context.earlier_results:
                raise TemplateError(f"There's no earlier step {step_id}.")
            if (outcome := context.earlier_results[step_id].short_circuit_outcome) is not None:
                return outcome
        if not evaluate(expression, {**self._scope, "steps": context.earlier_results}):
            return Outcome(0.0, is_skipped=True)
        return None


class Check(Step):
    """A step that gives marks."""

    method: ClassVar[Method[Any, QuestionOutcomes]]

    weight: float = Field(
        default=1,
        allow_inf_nan=False,
        title="Weight",
        description="How much it counts. Zero lets later steps read it without giving marks, and "
        "a negative weight is a penalty.",
        examples=[2, 1, 0.5, 0.25, -1],
        json_schema_extra=ADVANCED_HINT,
    )
    feedback: str | OutcomeFeedback | None = Field(
        default=None,
        title="Feedback",
        description="What students read, instead of the method's own feedback.",
        json_schema_extra=hint(kind="feedback", template_parts=FEEDBACK_INSERTS, advanced=True),
    )

    @classmethod
    def load(cls, data: object, *, methods: Methods | None = None) -> Check:
        """``data`` as a check, its nested steps too, using ``methods``: the built-ins unless
        given. Raises ``InvalidError`` if it isn't a check."""
        from .catalogue import BUILTINS  # built from steps, so not importable before them

        with invalid_errors(data):
            return load_check(data, (BUILTINS if methods is None else methods).by_ref)

    @property
    def marks_weight(self) -> float:
        return self.weight

    def run(
        self, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> tuple[list[Context], list[QuestionOutcomes]]:
        outcomes_by_student = self.mark(contexts, question_id)
        return [
            context.with_step_result(self.id, _check_result(outcomes)) if self.id else context
            for context, outcomes in zip(contexts, outcomes_by_student, strict=True)
        ], outcomes_by_student

    def described_in_list(self, weight_notes: Sequence[str]) -> str:
        """What it checks after its question and its name, with ``weight_notes`` after them, as a
        check among others: ``**Q2, Sign error** (50% of the marks): …``."""
        description = self.describe()
        heading = ", ".join(part for part in (self.question, self.name) if part)
        notes_text = ", ".join(weight_notes)
        if heading and notes_text:
            return f"**{heading}** ({notes_text}): {description}"
        if title := heading or capitalized(notes_text):
            return f"**{title}:** {description}"
        return description

    def mark(
        self, contexts: Sequence[Context], question_id: QuestionId | None = None
    ) -> list[dict[QuestionId, Outcome]]:
        """The outcomes of each student in ``contexts``, reading ``question_id`` unless the check
        names its own."""
        marked_question_id = self.marked_question_id(question_id)
        resolved_inputs, contexts_to_run = self._resolved_inputs_and_contexts_to_run(
            contexts, marked_question_id
        )
        method_results = iter(
            self.method.run(self, contexts_to_run, marked_question_id) if contexts_to_run else ()
        )
        outcomes_by_student: list[dict[QuestionId, Outcome]] = []
        for resolved_input in resolved_inputs:
            if resolved_input.short_circuit_outcome is None:
                outcomes = next(method_results)
                outcomes_by_student.append(
                    self._with_feedback_and_details(
                        outcomes, resolved_input.context, marked_question_id
                    )
                )
                continue
            marked_question_ids = (
                {marked_question_id}
                if marked_question_id
                else self.questions_marked(question_id) or self.questions_read(question_id)
            )
            outcomes_by_student.append(
                dict.fromkeys(marked_question_ids, resolved_input.short_circuit_outcome)
            )
        return [
            {
                marked_question_id: self._titled(
                    replace(outcome, method_label=self.method.label), marked_question_id
                )
                for marked_question_id, outcome in outcomes.items()
            }
            for outcomes in outcomes_by_student
        ]

    def _with_feedback_and_details(
        self,
        outcomes: Mapping[QuestionId, Outcome],
        context: Context,
        marked_question_id: QuestionId | None,
    ) -> dict[QuestionId, Outcome]:
        """``outcomes`` with the check's own feedback, and the transform's answer it read."""
        read_as_details: tuple[Detail, ...] = ()
        if marked_question_id is not None and isinstance(
            input_source(self.input), StepOutputSource
        ):
            answer = context.answers[marked_question_id]
            assert not isinstance(answer, Unreadable)  # marking deals with these first
            read_as_details = (Detail("Read as", write_answer(answer)),)
        return {
            question_id: replace(
                outcome,
                feedback=self._feedback(outcome, context, question_id)
                if self.feedback
                else outcome.feedback,
                details=(*read_as_details, *outcome.details),
            )
            for question_id, outcome in outcomes.items()
        }

    def _titled(self, outcome: Outcome, question_id: QuestionId) -> Outcome:
        """``outcome`` with what its breakdown says of the check: labelled by its name, and its
        question where that isn't the one marked. A named check with checks inside is titled by its
        name alone."""
        labels = tuple(
            label for label in (self.question, self.name) if label and label != question_id
        )
        title = (
            ""
            if self.name and outcome.children
            else outcome.title or self.describe().split("\n", 1)[0].removesuffix(":")
        )
        return replace(outcome, title=title, labels=(*labels, *outcome.labels))

    def _feedback(self, outcome: Outcome, context: Context, question_id: QuestionId) -> str:
        feedback = self.feedback
        template = (
            feedback.template_for(outcome) if isinstance(feedback, OutcomeFeedback) else feedback
        )
        if not template or not outcome.marked:
            return outcome.feedback
        scope = {**self._scope, "steps": dict(context.earlier_results)}
        return rendered_feedback(template, scope, outcome, context, question_id)


class Transform(Step):
    """A step that turns an answer into another for later steps to read, and gives no marks."""

    method: ClassVar[Method[Any, TransformResult | SkippedStep]]

    def transform(
        self, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> list[StepResult]:
        """What it gives each student in ``contexts``, reading ``question_id`` unless it names its
        own."""
        marked_question_id = self.marked_question_id(question_id)
        resolved_inputs, contexts_to_run = self._resolved_inputs_and_contexts_to_run(
            contexts, marked_question_id
        )
        method_results = iter(
            self.method.run(self, contexts_to_run, marked_question_id) if contexts_to_run else ()
        )
        output_question = self._output_question(marked_question_id, resolved_inputs)
        results = []
        for resolved_input in resolved_inputs:
            outcome = resolved_input.short_circuit_outcome
            transform_result: TransformResult | SkippedStep
            if outcome is None:
                transform_result = next(method_results)
            elif outcome.is_skipped:
                transform_result = SkippedStep()
            else:
                transform_result = None if outcome.marked else Unreadable(outcome.feedback)
            results.append(
                StepResult(None, skipped=True)
                if isinstance(transform_result, SkippedStep)
                else StepResult(transform_result, question=output_question)
            )
        return results

    def _output_question(
        self, question_id: QuestionId | None, resolved_inputs: Sequence[ResolvedInput]
    ) -> Question | None:
        """The question that reads what it gives."""
        input_context = next(
            (
                resolved_input.context
                for resolved_input in resolved_inputs
                if resolved_input.short_circuit_outcome is None
                or not resolved_input.short_circuit_outcome.is_skipped
            ),
            None,
        )
        if input_context is None or question_id is None:
            return None
        settings = self.dump().get("with", {})
        return self.method.output_question(
            settings, Position(input_context.questions, question_id), self._methods
        )

    def run(
        self, contexts: Sequence[Context], question_id: QuestionId | None
    ) -> tuple[list[Context], list[QuestionOutcomes]]:
        results = self.transform(contexts, question_id)
        return [
            context.with_step_result(self.id, result) if self.id else context
            for context, result in zip(contexts, results, strict=True)
        ], [{} for _ in contexts]

    def problems_in_list(self, step_ids_read_later: Set[str], index: int) -> Iterator[Problem]:
        if self.id is None:
            yield Problem("Give it an ID.", field=("steps", index, "id"))
        elif self.id not in step_ids_read_later:
            yield Problem(
                f"No later step reads {self.method.label}.", field=("steps",), is_incomplete=True
            )


def rendered_feedback(
    template: str,
    scope: Mapping[str, Any],
    outcome: Outcome,
    context: Context,
    question_id: QuestionId,
) -> str:
    """``template`` filled in with ``scope`` and how ``outcome`` marked the answer."""
    return render(
        template,
        {
            **scope,
            "answer": context.written_answers.get(question_id, "").strip(),
            "correct": outcome.correct,
            "proportion": outcome.proportion,
            "feedback": outcome.feedback,
            "parts": [part.template_values(question_id) for part in outcome.parts],
        },
    )


STEP_KEYS = (
    "id",
    "name",
    "question",
    "input",
    "if",
    "weight",
    "uses",
    "with",
    "feedback",
)
"""A step's keys, in the order YAML shows them."""
STEP_OWN_FIELDS = {
    field_info.alias or name: field_info
    for name, field_info in Check.model_fields.items()
    if name not in ("uses", "with_")
}
"""The keys a step has beside its method and its settings, which no setting may share."""


class StepData(BaseModel):
    """What a step looks like before its method is known, for its JSON schema."""

    model_config = ConfigDict(title="Step")

    uses: str
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")


def _methods_of(context: Any) -> Mapping[str, Method[Any]]:
    if not isinstance(context, Mapping) or METHODS_CONTEXT_KEY not in context:
        raise PydanticCustomError("missing_methods", "Supply a method collection.")
    methods_by_ref: Mapping[str, Method[Any]] = context[METHODS_CONTEXT_KEY]
    return methods_by_ref


def _validate_step(data: Any, info: ValidationInfo) -> Step:
    if isinstance(data, Step):
        if data._methods is _methods_of(info.context):
            return data
        data = data.dump()
    if not isinstance(data, Mapping):
        raise PydanticCustomError("step_type", "A step needs uses: and a method's name.")
    uses = data.get("uses")
    if not uses:
        raise _uses_error("missing_method", "Choose a method.", uses)
    methods_by_ref = _methods_of(info.context)
    method = methods_by_ref.get(str(uses))
    if method is None:
        raise _uses_error("unknown_method", f"There's no method {uses}.", uses)
    model = method.check_model if method.gives_marks(settings_of(data)) else method.transform_model
    step = model.model_validate(data, context=info.context)
    step._methods = methods_by_ref
    return step


def _validate_check(data: Any, info: ValidationInfo) -> Check:
    step = _validate_step(data, info)
    if not isinstance(step, Check):
        raise PydanticCustomError(
            "gives_no_marks",
            "Only a workflow can use {label}.",
            {"label": step.method.label},
        )
    return step


def _uses_error(kind: str, message: str, uses: Any) -> ValidationError:
    details = InitErrorDetails(type=PydanticCustomError(kind, message), loc=("uses",), input=uses)
    return ValidationError.from_exception_data("Step", [details])


AnyStep = Annotated[
    Step,
    PlainValidator(_validate_step, json_schema_input_type=StepData),
    PlainSerializer(Step.dump, return_type=dict[str, Any]),
]
"""One of a workflow's steps, a check or a transform, validated by the method it names."""
Marking = Annotated[
    Check,
    PlainValidator(_validate_check, json_schema_input_type=StepData),
    PlainSerializer(Step.dump, return_type=dict[str, Any]),
]
"""A check validated by the method its ``uses`` names: a marking's top check, or a nested one."""


_MARKING_ADAPTER: TypeAdapter[Check] = TypeAdapter(Marking)
_STEP_ADAPTER: TypeAdapter[Step] = TypeAdapter(AnyStep)


def step_cardinality(field_info: FieldInfo) -> Literal["one", "many"] | None:
    """Whether a setting holds a step, a list of steps, or neither."""
    if hint_of(field_info).kind == "step":
        return "one"
    item_field_info = item_field(field_info)
    return (
        "many" if item_field_info is not None and hint_of(item_field_info).kind == "step" else None
    )


def settings_of(step_data: Mapping[str, Any]) -> Mapping[str, Any]:
    """A step's settings as written; none unless they are a mapping."""
    settings = step_data.get("with")
    return settings if isinstance(settings, Mapping) else {}


def load_check(data: object, methods_by_ref: Mapping[str, Method[Any]]) -> Check:
    """``data`` as a check whose nested steps use only the methods of ``methods_by_ref``."""
    return _MARKING_ADAPTER.validate_python(data, context={METHODS_CONTEXT_KEY: methods_by_ref})


def load_step(data: object, methods_by_ref: Mapping[str, Method[Any]]) -> Step:
    """``data`` as a step, a check or a transform, using only the methods of ``methods_by_ref``."""
    return _STEP_ADAPTER.validate_python(data, context={METHODS_CONTEXT_KEY: methods_by_ref})
