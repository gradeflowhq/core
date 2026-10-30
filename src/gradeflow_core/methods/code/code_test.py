from dataclasses import replace
from functools import lru_cache
from typing import Any

from pydantic import ConfigDict, Field

from ...common.hints import ADVANCED, CODE, hint
from ...common.text import markdown_code, sentence
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..method import AnswerMethod
from ..outcome import Outcome
from ..settings import Settings
from . import runner
from .fixes import fixed_code

MAX_TIME_LIMIT_S = 60
FIXES = Field(
    default=0,
    ge=0,
    le=10,
    title="Most fixes",
    description="Code that needs more fixes gets no second try.",
    json_schema_extra=ADVANCED,
)
PENALTY = Field(
    default=0.0,
    ge=0,
    le=1,
    title="Marks off for fixes",
    description="The share of the marks fixed code loses, from 0 to 1.",
    json_schema_extra=ADVANCED,
)


class Test(Settings):
    model_config = ConfigDict(title="Test")

    expression: str = Field(
        min_length=1,
        title="Expression",
        description="Python code to run after the student's code.",
        examples=["add(2, 3)"],
    )
    expected: str = Field(
        min_length=1,
        title="Expected value",
        description="The Python value the expression should give.",
        examples=["5"],
    )
    weight: float = Field(
        default=1,
        title="Weight",
        description="How much the test counts, when each test earns its share.",
        json_schema_extra=ADVANCED,
    )


class TestSetup(Settings):
    model_config = ConfigDict(title="Test setup", frozen=True)

    code_before: str = Field(
        default="",
        title="Code before",
        description="Runs before the student's code, such as imports or helper functions.",
        json_schema_extra=hint(**CODE, advanced=True),
    )
    code_after: str = Field(
        default="",
        title="Code after",
        description="Runs after the student's code.",
        json_schema_extra=hint(**CODE, advanced=True),
    )
    indent: int = Field(
        default=0,
        ge=0,
        title="Indent student code (spaces)",
        description="Spaces added to each line of the student's code, to fit it inside yours.",
        json_schema_extra=ADVANCED,
    )
    time_limit: int = Field(
        default=runner.TIME_LIMIT_S,
        ge=1,
        le=MAX_TIME_LIMIT_S,
        title="Time limit (seconds)",
        description="Longest time each test may run before it fails.",
        json_schema_extra=ADVANCED,
    )

    def around(self, code: str) -> str:
        indented = "\n".join(
            " " * self.indent + line if line.strip() else line for line in code.splitlines()
        )
        return f"{self.code_before}\n{indented}\n{self.code_after}"


class CodeTestSettings(Settings):
    test: Test = Field(
        title="Test", description="An expression to run and the value it should give."
    )
    setup: TestSetup = Field(
        default_factory=TestSetup,
        title="Test setup",
        description="Code before and after the student's, and each test's time limit.",
    )
    evaluate_expected: bool = Field(
        default=True,
        title="Work out expected values",
        description="Feedback shows what an expected expression gives, such as 5 for 2 + 3.",
        json_schema_extra=ADVANCED,
    )
    fixes: int = FIXES
    """With some, the test runs the code with up to that many small errors fixed."""
    penalty: float = PENALTY


@lru_cache(maxsize=4096)
def _worked_out(expected: str, setup: TestSetup) -> str:
    """The expected value worked out around no student code, or as written if that fails."""
    variables: dict[str, Any] = {}
    try:
        runner.run(
            f"{setup.around('pass')}\nexpected = {expected}",
            variables,
            time_limit_s=setup.time_limit,
        )
    except runner.CodeError:
        return expected
    return repr(variables.get("expected"))


class CodeTest(AnswerMethod[CodeTestSettings]):
    name = "gradeflow/code-test"
    version = 1
    label = "Code Test"
    summary = "Full marks if the student's code passes one test."
    level = "internal"
    question_types = frozenset({"code"})
    settings = CodeTestSettings

    def describe(self, check: Check) -> str:
        test = self._settings_of(check).test
        return f"Gives {markdown_code(test.expected)} for {markdown_code(test.expression)}."

    def _mark_answer(
        self, settings: CodeTestSettings, answer: Answer, question: Question
    ) -> Outcome:
        assert isinstance(answer, str)  # a code question reading one answer
        if settings.fixes:
            fixed = fixed_code(answer, settings.setup.around("pass"), settings.fixes)
            if fixed is None:
                return Outcome(0.0, skipped=True)
            outcome = self._tested(fixed.code, settings)
            return replace(outcome, proportion=outcome.proportion * (1 - settings.penalty))
        return self._tested(answer, settings)

    def _tested(self, code: str, settings: CodeTestSettings) -> Outcome:
        test, setup = settings.test, settings.setup
        expected = (
            _worked_out(test.expected, setup) if settings.evaluate_expected else test.expected
        )
        check = (
            f"output = {test.expression}\nexpected = {test.expected}\npassed = output == expected"
        )
        variables: dict[str, Any] = {}
        try:
            runner.run(f"{setup.around(code)}\n{check}", variables, time_limit_s=setup.time_limit)
        except runner.CodeError as error:
            failed = sentence(f"{test.expression} failed: {error.reason}")
            return Outcome(0.0, f"{failed} Expected {expected}.")
        output = variables.get("output")
        if variables.get("passed"):
            return Outcome(1.0, f"{test.expression} gave {output!r}, as expected.")
        return Outcome(0.0, f"{test.expression} gave {output!r}. Expected {expected}.")
