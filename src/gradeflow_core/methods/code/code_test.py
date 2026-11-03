from functools import lru_cache
from typing import Any

from pydantic import ConfigDict, Field

from ...common.hints import ADVANCED_HINT, PYTHON_CODE_HINT, hint
from ...common.text import as_sentence, quoted
from ..outcome import Detail, Outcome
from ..settings import Settings
from . import runner

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
    time_limit: int = Field(
        default=runner.TIME_LIMIT_S,
        ge=1,
        le=runner.MAX_TIME_LIMIT_S,
        title="Time limit (seconds)",
        description="Longest time each test may run before it fails.",
        json_schema_extra=ADVANCED_HINT,
    )
    memory_limit: int = Field(
        default=runner.MEMORY_LIMIT_MB,
        ge=runner.MIN_MEMORY_LIMIT_MB,
        le=runner.MAX_MEMORY_LIMIT_MB,
        title="Memory limit (MiB)",
        description="Most memory each test may use before it fails.",
        json_schema_extra=ADVANCED_HINT,
    )

    def program_for(self, code: str) -> str:
        indented = "\n".join(
            " " * self.indent_spaces + line if line.strip() else line for line in code.splitlines()
        )
        return f"{self.code_before}\n{indented}\n{self.code_after}"


@lru_cache(maxsize=4096)
def _evaluated_expected(expected: str, setup: TestSetup) -> str:
    """The expected value evaluated around no student code, or as written if that fails."""
    variables: dict[str, Any] = {}
    try:
        runner.run(
            f"{setup.program_for('pass')}\nexpected = {expected}",
            variables,
            time_limit_s=setup.time_limit,
            memory_limit_mb=setup.memory_limit,
        )
    except runner.CodeError:
        return expected
    return repr(variables.get("expected"))


def outcome_of_test(code: str, test: Test, setup: TestSetup, *, evaluate_expected: bool) -> Outcome:
    """How ``code`` does on ``test``, run inside ``setup``."""
    program = setup.program_for(code)
    expected = _truncated(
        _evaluated_expected(test.expected, setup) if evaluate_expected else test.expected
    )
    test_details = (
        Detail("Expression", test.expression, "code"),
        Detail("Expected", expected, "code"),
    )
    if (syntax_error := _syntax_error(program, setup)) is not None:
        return Outcome(
            0.0,
            f"Couldn't run the code: {syntax_error}.",
            details=(*test_details, Detail("Output", syntax_error, "error")),
        )
    test_code = (
        "import contextlib as _contextlib, io as _io\n"
        "_printed = _io.StringIO()\n"
        "with _contextlib.redirect_stdout(_printed):\n"
        f"    output = {test.expression}\n"
        "printed = _printed.getvalue()\n"
        f"expected = {test.expected}\n"
        "passed = output == expected"
    )
    variables: dict[str, Any] = {}
    try:
        runner.run(
            f"{program}\n{test_code}",
            variables,
            time_limit_s=setup.time_limit,
            memory_limit_mb=setup.memory_limit,
        )
    except runner.CodeError as error:
        failure = as_sentence(f"{test.expression} failed: {error.reason}")
        return Outcome(
            0.0,
            f"{failure} Expected {expected}.",
            details=(*test_details, Detail("Output", error.reason, "error")),
        )
    output = _truncated(repr(variables.get("output")))
    output_details = (*test_details, Detail("Output", output, "code"))
    if variables.get("passed"):
        return Outcome(
            1.0, f"{test.expression} gave {output}, as expected.", details=output_details
        )
    feedback = f"{test.expression} gave {output}. Expected {expected}."
    if printed := _truncated(str(variables.get("printed", "")).rstrip()):
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
        line = (error.lineno or 0) - setup.code_before.count("\n") - 1
        line_note = f" (line {line})" if line > 0 else ""
        return f"SyntaxError: {error.msg}{line_note}"
    return None
