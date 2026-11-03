from functools import lru_cache
from typing import Any

from pydantic import ConfigDict, Field

from ...common.hints import ADVANCED, CODE, hint
from ...common.text import quoted, sentence
from ..outcome import Detail, Outcome
from ..settings import Settings
from . import runner

LONGEST = 200
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
        le=runner.MAX_TIME_LIMIT_S,
        title="Time limit (seconds)",
        description="Longest time each test may run before it fails.",
        json_schema_extra=ADVANCED,
    )
    memory_limit: int = Field(
        default=runner.MEMORY_LIMIT_MB,
        ge=runner.MIN_MEMORY_LIMIT_MB,
        le=runner.MAX_MEMORY_LIMIT_MB,
        title="Memory limit (MiB)",
        description="Most memory each test may use before it fails.",
        json_schema_extra=ADVANCED,
    )

    def around(self, code: str) -> str:
        indented = "\n".join(
            " " * self.indent + line if line.strip() else line for line in code.splitlines()
        )
        return f"{self.code_before}\n{indented}\n{self.code_after}"


@lru_cache(maxsize=4096)
def _worked_out(expected: str, setup: TestSetup) -> str:
    """The expected value worked out around no student code, or as written if that fails."""
    variables: dict[str, Any] = {}
    try:
        runner.run(
            f"{setup.around('pass')}\nexpected = {expected}",
            variables,
            time_limit_s=setup.time_limit,
            memory_limit_mb=setup.memory_limit,
        )
    except runner.CodeError:
        return expected
    return repr(variables.get("expected"))


def tested(code: str, test: Test, setup: TestSetup, *, evaluate_expected: bool) -> Outcome:
    """How ``code`` does on ``test``, run inside ``setup``."""
    program = setup.around(code)
    expected = _cut(_worked_out(test.expected, setup) if evaluate_expected else test.expected)
    asked = (
        Detail("Expression", test.expression, "code"),
        Detail("Expected", expected, "code"),
    )
    if (unreadable := _unreadable(program, setup)) is not None:
        return Outcome(
            0.0,
            f"Couldn't run the code: {unreadable}.",
            details=(*asked, Detail("Output", unreadable, "error")),
        )
    check = (
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
            f"{program}\n{check}",
            variables,
            time_limit_s=setup.time_limit,
            memory_limit_mb=setup.memory_limit,
        )
    except runner.CodeError as error:
        failed = sentence(f"{test.expression} failed: {error.reason}")
        return Outcome(
            0.0,
            f"{failed} Expected {expected}.",
            details=(*asked, Detail("Output", error.reason, "error")),
        )
    output = _cut(repr(variables.get("output")))
    gave = (*asked, Detail("Output", output, "code"))
    if variables.get("passed"):
        return Outcome(1.0, f"{test.expression} gave {output}, as expected.", details=gave)
    feedback = f"{test.expression} gave {output}. Expected {expected}."
    if printed := _cut(str(variables.get("printed", "")).rstrip()):
        return Outcome(
            0.0,
            f"{feedback} It printed {quoted(printed)}.",
            details=(*gave, Detail("Printed", printed)),
        )
    return Outcome(0.0, feedback, details=gave)


def _cut(value: str) -> str:
    """``value`` cut to ``LONGEST`` characters."""
    return value if len(value) <= LONGEST else f"{value[:LONGEST]}…"


def _unreadable(program: str, setup: TestSetup) -> str | None:
    """Why Python can't read ``program``, with the line of the student's code; None if it
    can."""
    try:
        compile(program, "<code>", "exec")
    except SyntaxError as error:
        line = (error.lineno or 0) - setup.code_before.count("\n") - 1
        where = f" (line {line})" if line > 0 else ""
        return f"SyntaxError: {error.msg}{where}"
    return None
