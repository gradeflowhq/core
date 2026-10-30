import ast
from collections.abc import Iterator
from typing import Literal

from pydantic import Field

from ...common.hints import hint
from ...common.problems import Problem
from ...common.text import markdown_code, sentence
from ...questions.base import Question
from ...questions.types import Answer
from ..check import Check
from ..example import examples_beside
from ..method import AnswerMethod
from ..outcome import Outcome
from ..position import Position
from ..settings import IDENTIFIER, Settings

Construct = Literal["call", "import", "loop", "comprehension", "recursion"]
CONSTRUCT_LABELS = {
    "call": "A call to a function",
    "import": "An import",
    "loop": "A loop",
    "comprehension": "A comprehension",
    "recursion": "Recursion",
}
LOOPS = (ast.For, ast.AsyncFor, ast.While)
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


class CodeUsesSettings(Settings):
    looks_for: Construct = Field(title="Looks for", json_schema_extra=hint(labels=CONSTRUCT_LABELS))
    function: str | None = Field(
        default=None,
        pattern=IDENTIFIER,
        title="Function",
        examples=["sum"],
        json_schema_extra=hint(applies_when={"looks_for": "call"}),
    )
    module: str | None = Field(
        default=None,
        title="Module",
        examples=["numpy"],
        json_schema_extra=hint(applies_when={"looks_for": "import"}),
    )


class CodeUses(AnswerMethod[CodeUsesSettings]):
    name = "gradeflow/code-uses"
    version = 1
    label = "Code Uses"
    summary = "Full marks if the code uses a construct you choose."
    guide = (
        "**Use it when:** Code must use a construct, such as a loop, recursion or a call to a "
        "function.\n\n"
        "**How it marks:** The code is read as written, not run, so comments and strings don't "
        "count. Code that can't be read gets no marks and is left for a person to look at.\n\n"
        "**Related:** To forbid a construct, put it in Not Allowed. To check what the code does, "
        "use Code Tests."
    )
    level = "advanced"
    question_types = frozenset({"code"})
    settings = CodeUsesSettings
    examples = examples_beside(__file__)

    def describe(self, check: Check) -> str:
        settings = self._settings_of(check)
        match settings.looks_for:
            case "call":
                return f"Calls {_named(settings.function, 'a function')}."
            case "import":
                return f"Imports {_named(settings.module, 'a module')}."
            case "loop":
                return "Uses a loop."
            case "comprehension":
                return "Uses a comprehension."
            case "recursion":
                return "Uses recursion."

    def _mark_answer(
        self, settings: CodeUsesSettings, answer: Answer, question: Question
    ) -> Outcome:
        assert isinstance(answer, str)  # a code question reading one answer
        try:
            tree = ast.parse(answer)
        except SyntaxError as error:
            where = f" on line {error.lineno}" if error.lineno else ""
            return Outcome(
                0.0, sentence(f"Couldn't read the code: {error.msg}{where}"), marked=False
            )
        nodes = list(ast.walk(tree))
        match settings.looks_for:
            case "call":
                found = any(
                    isinstance(node, ast.Call) and _called(node.func) == settings.function
                    for node in nodes
                )
                said = (f"Calls {settings.function}.", f"Expected a call to {settings.function}.")
            case "import":
                module = settings.module or ""
                found = any(
                    name == module or name.startswith(f"{module}.")
                    for node in nodes
                    for name in _imported(node)
                )
                said = (f"Imports {module}.", f"Expected an import of {module}.")
            case "loop":
                found = any(isinstance(node, LOOPS) for node in nodes)
                said = ("Uses a loop.", "Expected a loop.")
            case "comprehension":
                found = any(isinstance(node, COMPREHENSIONS) for node in nodes)
                said = ("Uses a comprehension.", "Expected a comprehension.")
            case "recursion":
                found = any(isinstance(node, FUNCTIONS) and _recursive(node) for node in nodes)
                said = ("Uses recursion.", "Expected recursion.")
        return Outcome(float(found), said[0] if found else said[1])

    def problems(self, check: Check, position: Position) -> Iterator[Problem]:
        settings = self._settings_of(check)
        if settings.looks_for == "call" and not settings.function:
            yield Problem("Name the function.", field=("function",), incomplete=True)
        if settings.looks_for == "import" and not settings.module:
            yield Problem("Name the module.", field=("module",), incomplete=True)


def _named(name: str | None, unnamed: str) -> str:
    return markdown_code(name) if name else unnamed


def _called(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    return func.attr if isinstance(func, ast.Attribute) else None


def _imported(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and not node.level and node.module:
        return [node.module]
    return []


def _recursive(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``function`` calls itself by name, or as ``self.name`` or ``cls.name`` in a
    method."""
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == function.name)
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == function.name
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("self", "cls")
            )
        )
        for node in ast.walk(function)
    )
