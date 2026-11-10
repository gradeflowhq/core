import textwrap
from collections.abc import Iterator
from dataclasses import replace

from pydantic import ConfigDict, Field

from ...common.hints import ADVANCED_HINT, PYTHON_CODE_HINT, hint
from ...common.problems import Problem
from ...common.text import (
    as_sentence,
    capitalized,
    code_span,
    plain_join,
    plain_markdown,
    quoted,
)
from ...questions.base import Question
from ...questions.code_files import python_files, split_answer
from ...questions.types import Answer
from ..example import examples_beside
from ..logic.workflow import (
    MARKS_BY_LABELS,
    MarksBy,
    ignored_weight_problems,
    proportion_and_parts,
    weight_notes,
)
from ..method import AnswerMethod
from ..outcome import CORRECT, Detail, Outcome
from ..position import Position
from ..settings import Settings
from ..step import Step
from . import runner
from .code_fixes import CodeFixes, best_fixed_outcome, fixes_field
from .custom_code import memory_limit_field, time_limit_field

MAX_SHOWN_LENGTH = 200
"""How many characters of a value feedback shows."""


class Test(Settings):
    model_config = ConfigDict(title="Test", str_strip_whitespace=True)

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
        json_schema_extra=ADVANCED_HINT,
    )


class TestSetup(Settings):
    model_config = ConfigDict(title="Test setup", frozen=True)

    code_before: str = Field(
        default="",
        title="Code before",
        description="Runs before the student's code, such as imports or helper functions.",
        json_schema_extra=hint(**PYTHON_CODE_HINT, advanced=True),
    )
    code_after: str = Field(
        default="",
        title="Code after",
        description="Runs after the student's code.",
        json_schema_extra=hint(**PYTHON_CODE_HINT, advanced=True),
    )
    indent_spaces: int = Field(
        default=0,
        ge=0,
        title="Indent student code (spaces)",
        description="Spaces added to each line of the student's code, to fit it inside yours.",
        json_schema_extra=ADVANCED_HINT,
    )
    time_limit: int = time_limit_field("Longest time each test may run before it fails.")
    memory_limit: int = memory_limit_field("Most memory each test may use before it fails.")

    def program_for(self, code: str) -> str:
        indented = textwrap.indent(code, " " * self.indent_spaces)
        return f"{self.code_before}\n{indented}\n{self.code_after}"


class CodeTestsSettings(Settings):
    model_config = ConfigDict(title="Code Tests")

    tests: list[Test] = Field(
        min_length=1,
        title="Tests",
        description="An expression to run, and the value it should give.",
    )
    setup: TestSetup = Field(
        default={},
        validate_default=True,
        title="Test setup",
        description="Code before and after the student's, and each test's time limit.",
        json_schema_extra=ADVANCED_HINT,
    )
    fixes: CodeFixes = fixes_field()
    marks_by: MarksBy = Field(
        default="all_correct",
        title="How tests count",
        json_schema_extra=hint(labels=MARKS_BY_LABELS),
    )


class CodeTests(AnswerMethod[CodeTestsSettings]):
    name = "gradeflow/code-tests"
    version = 1
    label = "Code Tests"
    summary = "Marks for each test the student's code passes."
    guide = (
        "**Use it when:** Students write Python, such as a function, and you want to check what "
        "it does.\n\n"
        "**How it marks:** The student's code runs, then each test runs an expression and "
        "compares what it gives with the expected value, so `add(2, 3)` should give `5`. **How "
        "tests count** sets whether all tests are needed, any one is enough, or each earns its "
        "share. **Fix code** also tests the code with common errors fixed, such as lost "
        "indentation. Each fix can carry a **Penalty**.\n\n"
        "**Good to know:** Test an ordinary case and an edge case, such as an empty list or "
        "zero. Each test has a time limit, under **Test setup**. In an answer of several files, "
        "the first runs and the others can be imported.\n\n"
        "**Related:** To check how the code is written, not what it does, use Code Uses."
    )
    level = "advanced"
    question_types = frozenset({"code"})
    settings = CodeTestsSettings
    examples = examples_beside(__file__)

    def describe(self, step: Step) -> str:
        settings = self._settings_of(step)
        tests, marks_by = settings.tests, settings.marks_by
        if len(tests) == 1:
            [test] = tests
            negated = marks_by == "none_correct"
            return f"{_weight_prefix(test, marks_by)}{_test_title(test, negated=negated)}"
        test_lines = "\n".join(
            f"- {_weight_prefix(test, marks_by)}{code_span(test.expression)} gives "
            f"{code_span(test.expected)}."
            for test in tests
        )
        return f"Passes {_which_tests(marks_by, len(tests))}:\n{test_lines}"

    def problems(self, step: Step, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(step)
        weights = [test.weight for test in settings.tests]
        yield from ignored_weight_problems(weights, settings.marks_by, "tests")

    def _mark_answer(
        self, settings: CodeTestsSettings, answer: Answer, question: Question
    ) -> Outcome:
        assert isinstance(answer, str)  # a code question's answer
        return best_fixed_outcome(
            answer,
            settings.fixes,
            settings.setup.program_for("pass"),
            lambda code: _tests_outcome(settings, code, question.max_marks),
        )


def _tests_outcome(settings: CodeTestsSettings, code: str, max_marks: float) -> Outcome:
    """The tests run on ``code``, combined."""
    (_, main_source), *module_files = python_files(split_answer(code)).items()
    weighted = []
    for test in settings.tests:
        outcome = _outcome_of_test(main_source, dict(module_files), test, settings.setup)
        weighted.append((test.weight, replace(outcome, title=_test_title(test))))
    marks_by = settings.marks_by
    proportion, test_outcomes = proportion_and_parts(marks_by, weighted, max_marks)
    feedback = _feedback(settings, test_outcomes, proportion)
    if marks_by == "none_correct" and (
        error := next(filter(None, (outcome.first_error() for outcome in test_outcomes)), None)
    ):
        proportion, feedback = 0.0, as_sentence(f"Couldn't test the code: {error}")
    if len(test_outcomes) == 1:  # one test is shown as the method itself
        return Outcome(proportion, feedback, parts=test_outcomes, details=test_outcomes[0].details)
    return Outcome(proportion, feedback, parts=test_outcomes, children=test_outcomes)


def _feedback(
    settings: CodeTestsSettings, test_outcomes: tuple[Outcome, ...], proportion: float
) -> str:
    """How many tests passed, and why the others failed. A test passes when it gives marks."""
    marks_by = settings.marks_by
    if marks_by == "none_correct":
        passed_titles = [
            as_sentence(plain_markdown(outcome.title))
            for outcome in test_outcomes
            if outcome.proportion > 0
        ]
        return f"Not allowed. {' '.join(passed_titles)}" if passed_titles else CORRECT
    if len(test_outcomes) == 1:
        return test_outcomes[0].feedback
    failed = [
        (test, outcome)
        for test, outcome in zip(settings.tests, test_outcomes, strict=True)
        if not outcome.proportion
    ]
    if not failed:
        return (
            "Passed both tests."
            if len(test_outcomes) == 2
            else f"Passed all {len(test_outcomes)} tests."
        )
    if len(failed) == len(test_outcomes) and len({outcome.feedback for _, outcome in failed}) == 1:
        return failed[0][1].feedback  # such as code that couldn't run
    passed_summary = f"Passed {len(test_outcomes) - len(failed)} of {len(test_outcomes)} tests."
    if marks_by == "any_correct" and proportion > 0:
        return passed_summary
    return "\n".join([passed_summary, *_failures(failed, len(test_outcomes))])


def _failures(failed: list[tuple[Test, Outcome]], count: int) -> list[str]:
    """Each failed test's feedback; tests the same error stopped share one line."""
    by_error: dict[str, list[tuple[Test, Outcome]]] = {}
    for test, outcome in failed:
        by_error.setdefault(outcome.first_error() or outcome.feedback, []).append((test, outcome))
    lines = []
    for error, failed_with_error in by_error.items():
        if len(failed_with_error) == 1:
            lines.append(failed_with_error[0][1].feedback)
        elif len(failed_with_error) == count:
            lines.append(as_sentence(f"All {count} failed: {error}"))
        else:
            expressions = plain_join(
                [test.expression for test, _ in failed_with_error], conjunction="and"
            )
            lines.append(as_sentence(f"{expressions} failed: {error}"))
    return lines


def _test_title(test: Test, *, negated: bool = False) -> str:
    """What ``test`` checks: ``Makes `add(2, 3)` give `5`.``"""
    makes = "Doesn't make" if negated else "Makes"
    return f"{makes} {code_span(test.expression)} give {code_span(test.expected)}."


def _weight_prefix(test: Test, marks_by: MarksBy) -> str:
    return "".join(f"**{capitalized(note)}:** " for note in weight_notes(test.weight, marks_by))


def _which_tests(marks_by: MarksBy, count: int) -> str:
    match marks_by:
        case "all_correct":
            return "both tests" if count == 2 else f"all {count} tests"
        case "any_correct":
            return "any one of these tests"
        case "none_correct":
            return "none of these tests"
        case "total":
            return "these tests, marks added up"
        case "proportional":
            return "these tests, a share for each"


_TEST_PROGRAM = """
import contextlib, io

def nested_types(value):
    types, seen, pending = set(), set(), [value]
    while pending:
        item = pending.pop()
        types.add(type(item))
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, dict):
            pending += [*item, *item.values()]
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending += item
    return types

scope = {}
exec(program, scope)
captured_stdout = io.StringIO()
with contextlib.redirect_stdout(captured_stdout):
    output = eval(expression, scope)
printed = captured_stdout.getvalue().rstrip()[:max_length]
output_repr = repr(output)[:max_length]
expected_value = eval(expected, scope)
expected_repr = repr(expected_value)[:max_length]
expected_types = nested_types(expected_value)
if expected_types & {int, float}:
    expected_types |= {int, float}
passed = bool(nested_types(output) <= expected_types and output == expected_value)
"""
"""Runs the student's code, then the test. It passes when the output equals the expected value and
holds only its types, ints and floats counting as one."""


def _outcome_of_test(
    code: str, module_files: dict[str, str], test: Test, setup: TestSetup
) -> Outcome:
    """How ``code``, able to import ``module_files``, does on ``test``, run inside ``setup``."""
    program = setup.program_for(code)
    test_details = (Detail("Expression", test.expression, "code"),)
    if (syntax_error := _syntax_error(program, setup)) is not None:
        return Outcome(
            0.0,
            f"Couldn't run the code: {syntax_error}.",
            details=(*test_details, Detail("Output", syntax_error, "error")),
        )
    try:
        returned = runner.run(
            _TEST_PROGRAM,
            {
                "program": program,
                "expression": test.expression,
                "expected": test.expected,
                # One more character than feedback shows, so _truncated can tell a longer value.
                "max_length": MAX_SHOWN_LENGTH + 1,
            },
            ("printed", "output_repr", "expected_repr", "passed"),
            module_files=module_files,
            time_limit_s=setup.time_limit,
            memory_limit_mb=setup.memory_limit,
        )
    except runner.CodeError as error:
        failure = as_sentence(f"{test.expression} failed: {error.reason}")
        return Outcome(
            0.0,
            f"{failure} Expected {test.expected}.",
            details=(
                *test_details,
                Detail("Expected", test.expected, "code"),
                Detail("Output", error.reason, "error"),
            ),
        )
    output, expected = (
        _truncated(str(returned.get(name))) for name in ("output_repr", "expected_repr")
    )
    output_details = (
        *test_details,
        Detail("Expected", expected, "code"),
        Detail("Output", output, "code"),
    )
    if returned.get("passed") is True:
        return Outcome(
            1.0, f"{test.expression} gave {output}, as expected.", details=output_details
        )
    feedback = f"{test.expression} gave {output}. Expected {expected}."
    if printed := _truncated(str(returned.get("printed", ""))):
        return Outcome(
            0.0,
            f"{feedback} It printed {quoted(printed)}.",
            details=(*output_details, Detail("Printed", printed)),
        )
    return Outcome(0.0, feedback, details=output_details)


def _truncated(value: str) -> str:
    """``value`` truncated to ``MAX_SHOWN_LENGTH`` characters."""
    return value if len(value) <= MAX_SHOWN_LENGTH else f"{value[:MAX_SHOWN_LENGTH]}…"


def _syntax_error(program: str, setup: TestSetup) -> str | None:
    """Why Python can't read ``program``, with the line of the student's code; None if it
    can."""
    try:
        compile(program, "<code>", "exec")
    except SyntaxError as error:
        return runner.syntax_error_text(
            error, (error.lineno or 0) - setup.code_before.count("\n") - 1
        )
    return None
