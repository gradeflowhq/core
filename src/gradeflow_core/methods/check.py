"""Checks: a method named by ``uses`` and set by ``with``, plus the keys every check may have."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self

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
)
from pydantic.fields import FieldInfo
from pydantic_core import InitErrorDetails, PydanticCustomError

from ..common.hints import ADVANCED_HINT, hint, hint_of
from ..common.problems import Problem, invalid_errors
from ..common.text import capitalized
from ..questions.question_set import Answers, QuestionSet, read_answer
from ..questions.types import QuestionId
from .outcome import EarlierResult, Outcome, feedback_scope, unanswered_outcome
from .position import Position
from .problems import located_problems
from .settings import IDENTIFIER, Settings, item_field
from .templates import TemplateError, condition, evaluate, names_read, render

if TYPE_CHECKING:
    from .collection import Methods
    from .method import Method

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
    """What marking one student needs: their answers, the questions, and the results of earlier
    checks with an ``id``. ``written_answers`` holds each answer as the student wrote it, for
    feedback that shows it, and ``precomputed`` what methods worked out for everyone's answers, by
    method ref and question."""

    answers: Answers
    questions: QuestionSet
    written_answers: Mapping[QuestionId, str]
    precomputed: Mapping[str, Mapping[QuestionId, Any]]
    earlier_results: Mapping[str, EarlierResult] = field(default_factory=dict)


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


class Check(BaseModel):
    """A method with its settings. Each method validates checks with a subclass of this, whose
    ``with`` holds that method's settings. Fields are in the order YAML shows them, as in a
    workflow step: what the check is, then its method, then its feedback."""

    model_config = ConfigDict(extra="forbid", validate_by_name=True, serialize_by_alias=True)

    method: ClassVar[Method[Any]]

    id: str | None = Field(
        default=None,
        pattern=IDENTIFIER,
        title="ID",
        description="Lets later checks' conditions read this check's result.",
    )
    name: str | None = Field(
        default=None,
        title="Name",
        description="Comes before its feedback.",
        examples=["Sign error"],
        json_schema_extra=hint(is_name=True, advanced=True),
    )
    question: QuestionId | None = Field(
        default=None,
        title="Question",
        description="The question this checks.",
        json_schema_extra=hint(kind="question"),
    )
    value: int | None = Field(
        default=None, ge=1, title="Value", description="Which value of the answer this checks."
    )
    read_only: bool = Field(
        default=False, title="Read only", description="It decides, but gives no marks."
    )
    if_: str | None = Field(
        default=None,
        alias="if",
        title="Only when",
        description="This check counts only when the condition holds.",
        json_schema_extra=hint(kind="condition", advanced=True),
    )
    weight: float = Field(
        default=1,
        allow_inf_nan=False,
        title="Weight",
        description="How much it counts. A negative weight is a penalty.",
        examples=[2, 1, 0.5, 0.25, -1],
        json_schema_extra=ADVANCED_HINT,
    )
    uses: str = Field(title="Method")
    with_: Settings = Field(alias="with")
    feedback: str | OutcomeFeedback | None = Field(
        default=None,
        title="Feedback",
        description="What students read, instead of the method's own feedback.",
        json_schema_extra=hint(kind="feedback", template_parts=FEEDBACK_INSERTS, advanced=True),
    )

    _scope: dict[str, Any] = PrivateAttr(default_factory=dict)
    _methods: Mapping[str, Method[Any]] = PrivateAttr(default_factory=dict)

    @classmethod
    def load(cls, data: object, *, methods: Methods | None = None) -> Check:
        """``data`` as a check, its nested checks too, using ``methods``: the built-ins unless
        given. Raises ``InvalidError`` if it isn't a check."""
        from .catalogue import BUILTINS  # built from checks, so not importable before them

        with invalid_errors(data):
            return check_of(data, (BUILTINS if methods is None else methods).by_ref)

    @property
    def scope(self) -> Mapping[str, Any]:
        """The inputs and item of the method file that made this check, for its templates."""
        return self._scope

    def scoped(self, scope: Mapping[str, Any]) -> Self:
        """This check, its templates also seeing ``scope``."""
        check = self.model_copy()
        check._scope = dict(scope)
        return check

    def describe(self) -> str:
        """What it checks, in a sentence or two of markdown."""
        return self.method.describe(self)

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
        self, context: Context, question: QuestionId | None = None, value: int | None = None
    ) -> dict[QuestionId, Outcome]:
        """Reads ``question``, or its ``value``, unless the check names its own."""
        position = Position(context.questions, question, value).for_check(self.question, self.value)
        question, value = position.question, position.value
        label = self.method.label
        return {
            key: self._titled(replace(outcome, method=outcome.method or label), key)
            for key, outcome in self._outcomes(context, question, value).items()
        }

    def _titled(self, outcome: Outcome, key: QuestionId) -> Outcome:
        """``outcome`` titled after this check's name, and its question unless that is ``key``. A
        named check with checks inside is titled by its name alone."""
        labels = tuple(label for label in (self.question, self.name) if label and label != key)
        title = "" if self.name and outcome.children else outcome.title or self.method.title(self)
        return replace(outcome, title=title, labels=(*labels, *outcome.labels))

    def _outcomes(
        self, context: Context, question: QuestionId | None, value: int | None
    ) -> dict[QuestionId, Outcome]:
        if (condition_outcome := self._condition_outcome(context)) is not None:
            questions = self.questions_marked(question) or self.questions_read(question)
            return dict.fromkeys(questions, condition_outcome)
        if question is not None:
            answer = read_answer(context.answers, question, value)
            outcome = unanswered_outcome(answer)
            if outcome is not None and not (answer is None and self.method.marks_blank_answers):
                return {question: outcome}
        outcomes = self.method.mark(self, context, question, value)
        if not self.feedback:
            return outcomes
        return {
            key: replace(outcome, feedback=self._feedback(outcome, context, key, value))
            for key, outcome in outcomes.items()
        }

    def _condition_outcome(self, context: Context) -> Outcome | None:
        if self.if_ is None:
            return None
        expression = condition(self.if_)
        for key in sorted(names_read(self.if_, "checks")):
            if key not in context.earlier_results:
                raise TemplateError(f"There's no earlier check {key}.")
            earlier_result = context.earlier_results[key]
            if not earlier_result["marked"]:
                return Outcome(0.0, earlier_result["feedback"], marked=False)
        if not evaluate(expression, {**self._scope, "checks": context.earlier_results}):
            return Outcome(0.0, skipped=True)
        return None

    def _feedback(
        self, outcome: Outcome, context: Context, question: QuestionId, value: int | None
    ) -> str:
        feedback = self.feedback
        template = feedback.of(outcome) if isinstance(feedback, OutcomeFeedback) else feedback
        if not template or not outcome.marked:
            return outcome.feedback
        scope = {**self._scope, "checks": dict(context.earlier_results)}
        return rendered_feedback(template, scope, outcome, context, question, value)

    def questions_marked(self, question: QuestionId | None = None) -> set[QuestionId]:
        return self.method.questions_marked(self, self.question or question)

    def questions_read(self, question: QuestionId | None = None) -> set[QuestionId]:
        question = self.question or question
        return self.method.questions_read(self, question) | ({question} if question else set())

    def needs_model_endpoint(self) -> bool:
        """Whether marking with it asks a language model, so ``mark`` needs a model endpoint."""
        return any(check.method.needs_model_endpoint for check, _ in checks_in(self, None))

    def problems(self, questions: QuestionSet) -> list[Problem]:
        """What stops it marking ``questions`` as a marking, each where it is: a question's, or a
        question group's without one."""
        return located_problems(self, questions)

    def dump(self) -> dict[str, Any]:
        """The check as YAML or JSON holds it: without what it leaves at the defaults."""
        data = self.model_dump(mode="json", by_alias=True, exclude_defaults=True)
        if data.get("with") == {}:
            data.pop("with")
        return data


def rendered_feedback(
    template: str,
    scope: Mapping[str, Any],
    outcome: Outcome,
    context: Context,
    question: QuestionId,
    value: int | None,
) -> str:
    """``template`` filled in with ``scope`` and how ``outcome`` marked the answer."""
    # Students see their answer as they wrote it, not as it was read.
    written = context.written_answers.get(question) if value is None else None
    answer = (
        written.strip() if written is not None else read_answer(context.answers, question, value)
    )
    return render(template, {**scope, **feedback_scope(outcome, answer, question)})


CHECK_KEYS = tuple(info.alias or name for name, info in Check.model_fields.items())
"""A check's keys, in the order YAML shows them."""
CHECK_FIELDS = {
    info.alias or name: info
    for name, info in Check.model_fields.items()
    if name not in ("uses", "with_")
}
"""The keys a check has beside its method and its settings, which no setting may share."""


class CheckData(BaseModel):
    """What a check looks like before its method is known, for its JSON schema."""

    model_config = ConfigDict(validate_by_name=True, title="Check")

    uses: str
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")


def _methods_of(context: Any) -> Mapping[str, Method[Any]]:
    if not isinstance(context, Mapping) or METHODS_CONTEXT_KEY not in context:
        raise PydanticCustomError("missing_methods", "Supply a method collection.")
    return context[METHODS_CONTEXT_KEY]  # type: ignore[no-any-return]


def _validate(data: Any, info: ValidationInfo) -> Check:
    if isinstance(data, Check):
        if data._methods is _methods_of(info.context):
            return data
        data = data.dump()
    if not isinstance(data, Mapping):
        raise PydanticCustomError("check_type", "A check needs uses: and a method's name.")
    uses = data.get("uses")
    if not uses:
        raise _uses_error("missing_method", "Choose a method.", uses)
    methods_by_ref = _methods_of(info.context)
    method = methods_by_ref.get(str(uses))
    if method is None:
        raise _uses_error("unknown_method", f"There's no method {uses}.", uses)
    check = method.check_model.model_validate(data, context=info.context)
    check._methods = methods_by_ref
    return check


def _uses_error(kind: str, message: str, uses: Any) -> ValidationError:
    details = InitErrorDetails(type=PydanticCustomError(kind, message), loc=("uses",), input=uses)
    return ValidationError.from_exception_data("Check", [details])


Marking = Annotated[
    Check,
    PlainValidator(_validate, json_schema_input_type=CheckData),
    PlainSerializer(Check.dump, return_type=dict[str, Any]),
]
"""A question's whole marking, or a question group's: the check at its top. Nested checks of a
method's settings have this type too. Each is validated by the method its ``uses`` names, nested
ones with the same methods."""


_MARKING_ADAPTER: TypeAdapter[Check] = TypeAdapter(Marking)


def how_many_checks(info: FieldInfo) -> Literal["one", "many"] | None:
    """Whether a setting holds a check, a list of checks, or neither."""
    if hint_of(info).kind == "check":
        return "one"
    item = item_field(info)
    return "many" if item is not None and hint_of(item).kind == "check" else None


def check_of(data: object, methods_by_ref: Mapping[str, Method[Any]]) -> Check:
    """``data`` as a check whose nested checks use only the methods of ``methods_by_ref``."""
    return _MARKING_ADAPTER.validate_python(data, context={METHODS_CONTEXT_KEY: methods_by_ref})


def checks_in(
    check: Check, question: QuestionId | None
) -> Iterator[tuple[Check, QuestionId | None]]:
    """``check`` and every check inside it, generated ones included, each with the question it
    reads. Generated checks that can't be worked out, which marking reports, are left out."""
    yield check, question
    try:
        inner_checks = list(check.method.expanded_checks(check))
    except ValidationError, TemplateError:
        return
    for _, inner, _ in inner_checks:
        yield from checks_in(inner, inner.question or question)
