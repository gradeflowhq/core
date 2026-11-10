"""Steps of a marking. A check turns an answer into marks; a transform turns it into another
answer for later steps to read."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, cast

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
from .input_sources import ResolvedInput, StepOutputSource, input_source
from .outcome import (
    CheckOutput,
    Detail,
    MethodResult,
    Outcome,
    QuestionOutcomes,
    Skipped,
    StepResult,
    TransformResult,
    unanswered_outcome,
)
from .problems import located_problems
from .settings import IDENTIFIER, Settings, item_field
from .templates import TemplateError, condition, evaluate, names_read, render

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
        self, key: QuestionId, answer: Answer | Unreadable, question: Question
    ) -> Context:
        """This context, with ``key`` answered by ``answer`` and read as ``question``, and
        feedback showing ``answer``."""
        marked_question = self.questions[key]
        input_question = question.model_copy(update={"max_marks": marked_question.max_marks})
        written_answer = "" if isinstance(answer, Unreadable) else write_answer(answer)
        return replace(
            self,
            answers={**self.answers, key: answer},
            written_answers={**self.written_answers, key: written_answer},
            questions=QuestionSet.model_construct(
                root={**self.questions.root, key: input_question}
            ),
            source_answers={key: (self.answers.get(key), marked_question), **self.source_answers},
        )

    def with_value_as_answer(self, key: QuestionId, number: int) -> Context:
        """This context, with value ``number`` of the answer to ``key`` as its answer."""
        source_answer, source_question = self.source_answers.get(
            key, (self.answers.get(key), self.questions[key])
        )
        question = source_question.value_question(number)
        if question is None:
            return self
        return self.with_answer(key, value_of(source_answer, number), question)

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
    skipped = bool(outcomes) and all(outcome.skipped for outcome in outcomes.values())
    return StepResult(output, skipped)


class OutcomeFeedback(BaseModel):
    """Feedback for each outcome. One left empty is the method's own."""

    model_config = ConfigDict(extra="forbid", title="Feedback")

    correct: str = Field(default="", title="When it's correct")
    partly_correct: str = Field(default="", title="When it's partly correct")
    incorrect: str = Field(default="", title="When it's incorrect")

    def of(self, outcome: Outcome) -> str:
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
        json_schema_extra=hint(is_name=True, advanced=True),
    )
    question: QuestionId | None = Field(
        default=None,
        title="Question",
        description="The question this step reads and marks.",
        json_schema_extra=hint(kind="question"),
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
        self, contexts: Sequence[Context], question: QuestionId | None = None
    ) -> list[tuple[Context, dict[QuestionId, Outcome] | None]]:
        """Each student's context after it, and the outcomes it gave if it marks."""

    def described_in_list(self, weight_notes: Sequence[str]) -> str:
        """What it does, as an item of a list, with a check's ``weight_notes``."""
        return self.describe()

    def problems_in_list(self, step_ids_read_later: Set[str], index: int) -> Iterator[Problem]:
        """What is wrong with it at ``index`` of a list, given the IDs later steps read."""
        return iter(())

    def describe(self) -> str:
        """What it does, in a sentence or two of markdown."""
        return self.method.describe(self)

    def questions_marked(self, question: QuestionId | None = None) -> set[QuestionId]:
        return self.method.questions_marked(self, self.question or question)

    def questions_read(self, question: QuestionId | None = None) -> set[QuestionId]:
        question = self.question or question
        questions_read = self.method.questions_read(self, question) | (
            {question} if question else set()
        )
        return questions_read | input_source(self.input).questions_read()

    def needs_model_endpoint(self) -> bool:
        """Whether running it asks a language model."""
        if self.method.needs_model_endpoint:
            return True
        try:
            return any(
                inner.needs_model_endpoint() for _, inner in self.method.expanded_steps(self)
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

    def question_key(self, question: QuestionId | None) -> QuestionId | None:
        """The key of the answer it reads and marks, at ``question`` unless it names its own."""
        return input_source(self.input).question_key(self.question or question)

    def _resolved_input(self, context: Context, key: QuestionId | None) -> ResolvedInput:
        """The step's input read and its condition tested, for one student."""
        resolved_input = input_source(self.input).resolve(context, key)
        if resolved_input.short_circuit_outcome is not None:
            return resolved_input
        short_circuit_outcome = self._condition_short_circuit(resolved_input.context)
        if short_circuit_outcome is None and key is not None:
            answer = resolved_input.context.answers.get(key)
            short_circuit_outcome = unanswered_outcome(answer)
            if answer is None and self.method.marks_blank_answers:
                short_circuit_outcome = None
        return replace(resolved_input, short_circuit_outcome=short_circuit_outcome)

    def _run_method(
        self, contexts: Sequence[Context], key: QuestionId | None
    ) -> list[tuple[ResolvedInput, MethodResult]]:
        """Each student's resolved input, and what the method gave or None if it didn't run."""
        resolved_inputs = [self._resolved_input(context, key) for context in contexts]
        inputs_to_run = [
            resolved_input
            for resolved_input in resolved_inputs
            if resolved_input.short_circuit_outcome is None
        ]
        method_results = iter(
            self.method.run(self, [resolved.context for resolved in inputs_to_run], key)
            if inputs_to_run
            else ()
        )
        return [
            (
                resolved_input,
                next(method_results) if resolved_input.short_circuit_outcome is None else None,
            )
            for resolved_input in resolved_inputs
        ]

    def _condition_short_circuit(self, context: Context) -> Outcome | None:
        if self.if_ is None:
            return None
        expression = condition(self.if_)
        for key in sorted(names_read(self.if_, "steps")):
            if key not in context.earlier_results:
                raise TemplateError(f"There's no earlier step {key}.")
            if (outcome := context.earlier_results[key].condition_short_circuit) is not None:
                return outcome
        if not evaluate(expression, {**self._scope, "steps": context.earlier_results}):
            return Outcome(0.0, skipped=True)
        return None


class Check(Step):
    """A step that gives marks."""

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
        self, contexts: Sequence[Context], question: QuestionId | None = None
    ) -> list[tuple[Context, dict[QuestionId, Outcome] | None]]:
        outcomes_by_student = self.mark(contexts, question)
        return [
            (
                context.with_step_result(self.id, _check_result(outcomes)) if self.id else context,
                outcomes,
            )
            for context, outcomes in zip(contexts, outcomes_by_student, strict=True)
        ]

    def described_in_list(self, weight_notes: Sequence[str]) -> str:
        return self.described_with_heading(weight_notes)

    def described_with_heading(self, notes: Sequence[str] = ()) -> str:
        """What it checks after its question and its name, with ``notes`` after them, as a check
        among others: ``**Q2, Sign error** (50% of the marks): …``."""
        description = self.describe()
        heading = ", ".join(part for part in (self.question, self.name) if part)
        notes_text = ", ".join(notes)
        if heading and notes_text:
            return f"**{heading}** ({notes_text}): {description}"
        if title := heading or capitalized(notes_text):
            return f"**{title}:** {description}"
        return description

    def mark(
        self, contexts: Sequence[Context], question: QuestionId | None = None
    ) -> list[dict[QuestionId, Outcome]]:
        """The outcomes of each student in ``contexts``, reading ``question`` unless the check
        names its own."""
        key = self.question_key(question)
        results: list[dict[QuestionId, Outcome]] = []
        for resolved_input, method_result in self._run_method(contexts, key):
            if resolved_input.short_circuit_outcome is None:
                outcomes = cast(QuestionOutcomes, method_result)
                results.append(self._with_feedback_and_details(outcomes, resolved_input, key))
                continue
            keys = (
                {key} if key else self.questions_marked(question) or self.questions_read(question)
            )
            results.append(dict.fromkeys(keys, resolved_input.short_circuit_outcome))
        return [
            {
                key: self._titled(replace(outcome, method=outcome.method or self.method.label), key)
                for key, outcome in outcomes.items()
            }
            for outcomes in results
        ]

    def _with_feedback_and_details(
        self,
        outcomes: Mapping[QuestionId, Outcome],
        resolved_input: ResolvedInput,
        key: QuestionId | None,
    ) -> dict[QuestionId, Outcome]:
        """``outcomes`` with the check's own feedback, and the transform's answer it read."""
        read_as_details: tuple[Detail, ...] = ()
        if key is not None and isinstance(input_source(self.input), StepOutputSource):
            answer = resolved_input.context.answers[key]
            assert not isinstance(answer, Unreadable)  # marking deals with these first
            read_as_details = (Detail("Read as", write_answer(answer)),)
        return {
            key: replace(
                outcome,
                feedback=self._feedback(outcome, resolved_input.context, key)
                if self.feedback
                else outcome.feedback,
                details=(*read_as_details, *outcome.details),
            )
            for key, outcome in outcomes.items()
        }

    def _titled(self, outcome: Outcome, key: QuestionId) -> Outcome:
        """``outcome`` with what its breakdown says of the check: labelled by its name, and its
        question where that isn't the one marked. A named check with checks inside is titled by its
        name alone."""
        labels = tuple(label for label in (self.question, self.name) if label and label != key)
        title = "" if self.name and outcome.children else outcome.title or self.method.title(self)
        return replace(outcome, title=title, labels=(*labels, *outcome.labels))

    def _feedback(self, outcome: Outcome, context: Context, question: QuestionId) -> str:
        feedback = self.feedback
        template = feedback.of(outcome) if isinstance(feedback, OutcomeFeedback) else feedback
        if not template or not outcome.marked:
            return outcome.feedback
        scope = {**self._scope, "steps": dict(context.earlier_results)}
        return rendered_feedback(template, scope, outcome, context, question)


class Transform(Step):
    """A step that turns an answer into another for later steps to read, and gives no marks."""

    def transform(
        self, contexts: Sequence[Context], question: QuestionId | None = None
    ) -> list[StepResult]:
        """What it gives each student in ``contexts``, reading ``question`` unless it names its
        own."""
        key = self.question_key(question)
        runs = self._run_method(contexts, key)
        output_question = self._output_question(key, [resolved for resolved, _ in runs])
        results = []
        for resolved_input, method_result in runs:
            outcome = resolved_input.short_circuit_outcome
            if outcome is None:
                value = cast(TransformResult | Skipped, method_result)
            elif outcome.skipped:
                value = Skipped()
            else:
                value = None if outcome.marked else Unreadable("", outcome.feedback)
            results.append(
                StepResult(None, skipped=True)
                if isinstance(value, Skipped)
                else StepResult(value, question=output_question)
            )
        return results

    def _output_question(
        self, key: QuestionId | None, resolved_inputs: Sequence[ResolvedInput]
    ) -> Question | None:
        """The question that reads what it gives."""
        read = next(
            (
                resolved_input.context
                for resolved_input in resolved_inputs
                if resolved_input.short_circuit_outcome is None
                or not resolved_input.short_circuit_outcome.skipped
            ),
            None,
        )
        if read is None or key is None:
            return None
        settings = self.dump().get("with", {})
        return self.method.output_question(settings, read.questions[key], self._methods)

    def run(
        self, contexts: Sequence[Context], question: QuestionId | None = None
    ) -> list[tuple[Context, dict[QuestionId, Outcome] | None]]:
        results = self.transform(contexts, question)
        return [
            (context.with_step_result(self.id, result) if self.id else context, None)
            for context, result in zip(contexts, results, strict=True)
        ]

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
    question: QuestionId,
) -> str:
    """``template`` filled in with ``scope`` and how ``outcome`` marked the answer."""
    written = context.written_answers.get(question)
    answer = written.strip() if written is not None else context.answers.get(question)
    return render(
        template,
        {
            **scope,
            "answer": answer.answer if isinstance(answer, Unreadable) else write_answer(answer),
            "correct": outcome.correct,
            "proportion": outcome.proportion,
            "feedback": outcome.feedback,
            "parts": [part.template_values(question) for part in outcome.parts],
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
    info.alias or name: info
    for name, info in Check.model_fields.items()
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
    settings = data.get("with")
    gives_marks = method.gives_marks(settings if isinstance(settings, Mapping) else {})
    model = method.check_model if gives_marks else method.transform_model
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


def step_cardinality(info: FieldInfo) -> Literal["one", "many"] | None:
    """Whether a setting holds a step, a list of steps, or neither."""
    if hint_of(info).kind == "step":
        return "one"
    item = item_field(info)
    return "many" if item is not None and hint_of(item).kind == "step" else None


def load_check(data: object, methods_by_ref: Mapping[str, Method[Any]]) -> Check:
    """``data`` as a check whose nested steps use only the methods of ``methods_by_ref``."""
    return _MARKING_ADAPTER.validate_python(data, context={METHODS_CONTEXT_KEY: methods_by_ref})


def load_step(data: object, methods_by_ref: Mapping[str, Method[Any]]) -> Step:
    """``data`` as a step, a check or a transform, using only the methods of ``methods_by_ref``."""
    return _STEP_ADAPTER.validate_python(data, context={METHODS_CONTEXT_KEY: methods_by_ref})
