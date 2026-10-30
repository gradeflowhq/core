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

from ..common.hints import ADVANCED, hint, hint_of
from ..common.problems import Problem, as_invalid
from ..common.text import capitalised
from ..questions.question_set import Answers, QuestionSet, read_answer
from ..questions.types import QuestionId
from .outcome import CheckResult, Outcome, feedback_scope, unanswered
from .position import Position
from .problems import located_problems
from .settings import IDENTIFIER, Settings, item_field
from .templates import TemplateError, condition, evaluate, named, render

if TYPE_CHECKING:
    from .collection import Methods
    from .method import Method

METHODS = "methods"
"""The validation context key of the prepared method lookup."""
INSERTS = {
    "{{ answer }}": "Their answer",
    "{{ feedback }}": "Its own feedback",
    "{{ proportion | percent }}": "Proportion of the marks",
}
"""What feedback may include, by label."""


@dataclass(frozen=True)
class Context:
    """What marking one student needs: their answers, the questions, and the results of earlier
    checks with an ``id``. ``written`` holds each answer as the student wrote it, for feedback that
    shows it, and ``precomputed`` what methods worked out for everyone's answers, by method ref
    and question."""

    answers: Answers
    questions: QuestionSet
    written: Mapping[QuestionId, str]
    precomputed: Mapping[str, Mapping[QuestionId, Any]]
    earlier: Mapping[str, CheckResult] = field(default_factory=dict)


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

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

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
        json_schema_extra=hint(names=True, advanced=True),
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
        description="How much it counts. A negative weight takes marks off.",
        examples=[2, 1, 0.5, 0.25, -1],
        json_schema_extra=ADVANCED,
    )
    uses: str = Field(title="Method")
    with_: Settings = Field(alias="with")
    feedback: str | OutcomeFeedback | None = Field(
        default=None,
        title="Feedback",
        description="What students read, instead of the method's own feedback.",
        json_schema_extra=hint(kind="feedback", inserts=INSERTS, advanced=True),
    )

    _scope: dict[str, Any] = PrivateAttr(default_factory=dict)
    _methods: Mapping[str, Method[Any]] = PrivateAttr(default_factory=dict)

    @classmethod
    def load(cls, data: object, *, methods: Methods | None = None) -> Check:
        """``data`` as a check, its nested checks too, using ``methods``: the built-ins unless
        given. Raises ``InvalidError`` if it isn't a check."""
        from .catalogue import BUILTINS  # built from checks, so not importable before them

        with as_invalid(data):
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

    def headed(self, notes: Sequence[str] = ()) -> str:
        """What it checks after its question and its name, with ``notes`` after them, as a check
        among others: ``**Q2, Sign error** (50% of the marks): …``."""
        described = self.describe()
        own = ", ".join(part for part in (self.question, self.name) if part)
        noted = ", ".join(notes)
        if own and noted:
            return f"**{own}** ({noted}): {described}"
        if title := own or capitalised(noted):
            return f"**{title}:** {described}"
        return described

    def mark(
        self, context: Context, question: QuestionId | None = None, value: int | None = None
    ) -> dict[QuestionId, Outcome]:
        """Reads ``question``, or its ``value``, unless the check names its own."""
        method = self.method
        at = Position(context.questions, question, value).at(self.question, self.value)
        question, value = at.question, at.value
        if (held := self._condition(context)) is not None:
            keys = self.questions_marked(question) or self.questions_read(question)
            return {key: held.labelled(method.label) for key in keys}
        if question is not None:
            answer = read_answer(context.answers, question, value)
            unread = unanswered(answer)
            if unread is not None and not (answer is None and method.marks_blank):
                return {question: unread.labelled(method.label)}
        outcomes = method.mark(self, context, question, value)
        if self.feedback:
            outcomes = {
                key: replace(outcome, feedback=self._feedback(outcome, context, key, value))
                for key, outcome in outcomes.items()
            }
        return {key: outcome.labelled(method.label) for key, outcome in outcomes.items()}

    def _condition(self, context: Context) -> Outcome | None:
        if self.if_ is None:
            return None
        expression = condition(self.if_)
        for key in sorted(named(self.if_, "checks")):
            if key not in context.earlier:
                raise TemplateError(f"There's no earlier check {key}.")
            result = context.earlier[key]
            if not result["marked"]:
                return Outcome(0.0, result["feedback"], marked=False)
        if not evaluate(expression, {**self._scope, "checks": context.earlier}):
            return Outcome(0.0, skipped=True)
        return None

    def _feedback(
        self, outcome: Outcome, context: Context, question: QuestionId, value: int | None
    ) -> str:
        own = self.feedback
        template = own.of(outcome) if isinstance(own, OutcomeFeedback) else own
        if not template or not outcome.marked:
            return outcome.feedback
        scope = {**self._scope, "checks": dict(context.earlier)}
        return rendered_feedback(template, scope, outcome, context, question, value)

    def questions_marked(self, question: QuestionId | None = None) -> set[QuestionId]:
        return self.method.questions_marked(self, self.question or question)

    def questions_read(self, question: QuestionId | None = None) -> set[QuestionId]:
        own = self.question or question
        return self.method.questions_read(self, own) | ({own} if own else set())

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
    written = context.written.get(question) if value is None else None
    answer = (
        written.strip() if written is not None else read_answer(context.answers, question, value)
    )
    return render(template, {**scope, **feedback_scope(outcome, answer, question)})


KEYS = tuple(info.alias or name for name, info in Check.model_fields.items())
"""A check's keys, in the order YAML shows them."""
CHECK_FIELDS = {
    info.alias or name: info
    for name, info in Check.model_fields.items()
    if name not in ("uses", "with_")
}
"""The keys a check has beside its method and its settings, which no setting may share."""


class CheckData(BaseModel):
    """What a check looks like before its method is known, for its JSON schema."""

    model_config = ConfigDict(populate_by_name=True, title="Check")

    uses: str
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")


def _methods_of(context: Any) -> Mapping[str, Method[Any]]:
    if not isinstance(context, Mapping) or METHODS not in context:
        raise PydanticCustomError("missing_methods", "Supply a method collection.")
    return context[METHODS]  # type: ignore[no-any-return]


def _validate(data: Any, info: ValidationInfo) -> Check:
    if isinstance(data, Check):
        if data._methods is _methods_of(info.context):
            return data
        data = data.dump()
    if not isinstance(data, Mapping):
        raise PydanticCustomError("check_type", "A check needs uses: and a method's name.")
    uses = data.get("uses")
    if not uses:
        raise _error("missing_method", "Choose a method.", uses)
    lookup = _methods_of(info.context)
    method = lookup.get(str(uses))
    if method is None:
        raise _error("unknown_method", f"There's no method {uses}.", uses)
    check = method.check_model.model_validate(data, context=info.context)
    check._methods = lookup
    return check


def _error(kind: str, message: str, uses: Any) -> ValidationError:
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


_CHECK: TypeAdapter[Check] = TypeAdapter(Marking)


def holds_checks(info: FieldInfo) -> Literal["one", "many"] | None:
    """Whether a setting holds a check, a list of checks, or neither."""
    if hint_of(info).kind == "check":
        return "one"
    item = item_field(info)
    return "many" if item is not None and hint_of(item).kind == "check" else None


def check_of(data: object, lookup: Mapping[str, Method[Any]]) -> Check:
    """``data`` as a check whose nested checks use only the methods of ``lookup``."""
    return _CHECK.validate_python(data, context={METHODS: lookup})


def checks_in(
    check: Check, question: QuestionId | None
) -> Iterator[tuple[Check, QuestionId | None]]:
    """``check`` and every check inside it, generated ones included, each with the question it
    reads. A generated body that can't be worked out, which marking reports, is left out."""
    yield check, question
    try:
        children = list(check.method.validation_children(check))
    except ValidationError, TemplateError:
        return
    for _, child, _ in children:
        yield from checks_in(child, child.question or question)
