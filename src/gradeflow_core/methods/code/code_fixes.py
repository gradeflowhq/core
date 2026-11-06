"""Fix code: common errors in Python code fixed before marking."""

import ast
import builtins
import io
import keyword
import symtable
import tokenize
import warnings
from collections.abc import Callable, Collection, Iterable, Iterator
from dataclasses import dataclass, replace
from tokenize import TokenInfo
from typing import Any, Literal, NamedTuple, get_args

from pydantic import ConfigDict, Field
from rapidfuzz.distance import OSA

from ...common.hints import ADVANCED_HINT, hint
from ...common.text import capitalized, format_number, quoted
from ...questions.layouts import BRACKETS, CLOSERS, COMPOUND, OPENERS, guess_layouts
from ..outcome import Detail, Outcome
from ..settings import Settings

type LineColumn = tuple[int, int]
FixKind = Literal["indentation", "colons", "brackets", "equals", "keywords", "names"]

MAX_INDENTATIONS_TRIED = 3
"""How many indentations to try for code with none."""
MAX_FIXES = 10
"""Code that needs more fixes is marked as written."""

STATEMENT_KEYWORDS = frozenset(
    {
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "nonlocal",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }
)
# A soft keyword can also be a name, as in "match = 1".
HEADER_KEYWORDS = COMPOUND - frozenset(keyword.softkwlist)
CONDITION_KEYWORDS = frozenset({"if", "elif", "while"})
STANDALONE_KEYWORDS = frozenset({"pass", "break", "continue", "return", "else", "try", "finally"})
KEYWORDS_BEFORE_COLON = frozenset({"else", "try", "finally", "except"})
# Keywords that can follow an expression, so "word if …" may be a valid statement.
EXPRESSION_KEYWORDS = frozenset({"and", "or", "not", "in", "is", "if", "else", "for", "as"})
LOWERCASE_CONSTANTS = {"true": "True", "false": "False", "none": "None"}
STRING_STARTS = frozenset({tokenize.STRING, tokenize.FSTRING_START, tokenize.TSTRING_START})
OPERAND_TOKEN_TYPES = frozenset(
    {tokenize.NUMBER, tokenize.STRING, tokenize.FSTRING_END, tokenize.TSTRING_END}
)
IGNORED_TOKEN_TYPES = frozenset(
    {tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
)
PYTHON_BUILTINS = frozenset(dir(builtins))


class Fix(NamedTuple):
    kind: FixKind
    description: str
    """The fix as feedback says it, such as: line 3, added “:” after “def add(a, b)”."""


@dataclass(frozen=True)
class FixedCode:
    code: str
    fixes: tuple[Fix, ...]


class _Edit(NamedTuple):
    start: LineColumn
    end: LineColumn
    text: str


type _LineFix = tuple[_Edit, Fix]


def _penalty_field(kind: FixKind) -> Any:
    return Field(
        default=0,
        ge=0,
        le=1,
        title="Penalty",
        examples=[0.1, 0.25, 0.5],
        json_schema_extra=hint(applies_when={kind: [True]}),
    )


class CodeFixes(Settings):
    model_config = ConfigDict(title="Fix code")

    indentation: bool = Field(
        default=False,
        title="Indentation",
        description="Lost indentation is rebuilt.",
    )
    indentation_penalty: float = _penalty_field("indentation")
    colons: bool = Field(
        default=False,
        title="Missing colons",
        description="“if x > 0” reads as “if x > 0:”.",
    )
    colons_penalty: float = _penalty_field("colons")
    brackets: bool = Field(
        default=False,
        title="Brackets and quotes",
        description="“print(x” reads as “print(x)”.",
    )
    brackets_penalty: float = _penalty_field("brackets")
    equals: bool = Field(
        default=False,
        title="Equals in conditions",
        description="“if x = 1” reads as “if x == 1”.",
    )
    equals_penalty: float = _penalty_field("equals")
    keywords: bool = Field(
        default=False,
        title="Misspelt keywords",
        description="“retrun” reads as “return”.",
    )
    keywords_penalty: float = _penalty_field("keywords")
    names: bool = Field(
        default=False,
        title="Misspelt names",
        description="“lenght” reads as “length”, and “true” as “True”.",
    )
    names_penalty: float = _penalty_field("names")

    @property
    def enabled_kinds(self) -> frozenset[FixKind]:
        return frozenset(kind for kind in get_args(FixKind) if getattr(self, kind))

    def penalty(self, kind: FixKind) -> float:
        """The share of the marks each fix of ``kind`` removes."""
        return float(getattr(self, f"{kind}_penalty"))

    def total_penalty(self, version: FixedCode) -> float:
        """The share of the marks the fixes in ``version`` remove, at most 1."""
        return min(1.0, sum(self.penalty(fix.kind) for fix in version.fixes))


def fixes_field() -> Any:
    """A code method's ``fixes`` field."""
    return Field(
        default={},
        validate_default=True,
        title="Fix code",
        description="Common errors fixed before marking.",
        json_schema_extra=ADVANCED_HINT,
    )


def best_fixed_outcome(
    code: str, fixes: CodeFixes, setup: str, mark: Callable[[str], Outcome]
) -> Outcome:
    """The best outcome ``mark`` gives ``code`` or one of its fixed versions, less the fixes'
    penalty: the most marks, then no error, then the earliest. ``setup`` is the code run before
    it."""
    best = mark(code)
    best_version: FixedCode | None = None
    for version in fixed_versions(code, fixes.enabled_kinds, setup):
        if best.correct:
            break
        outcome = mark(version.code)
        outcome = replace(
            outcome, proportion=outcome.proportion * (1 - fixes.total_penalty(version))
        )
        if (outcome.marked, outcome.proportion, outcome.first_error() is None) > (
            best.marked,
            best.proportion,
            best.first_error() is None,
        ):
            best, best_version = outcome, version
    if best_version is None:
        return best
    fix_lines = "\n".join(
        f"- {_fix_feedback(fix, fixes.penalty(fix.kind))}" for fix in best_version.fixes
    )
    return replace(
        best,
        feedback=f"Fixed before marking:\n{fix_lines}\n{best.feedback}",
        details=(Detail("Marked as", best_version.code, "code"), *best.details),
    )


def _fix_feedback(fix: Fix, penalty: float) -> str:
    suffix = f" (-{format_number(penalty * 100)}%)" if penalty else ""
    return f"{capitalized(fix.description)}{suffix}."


def fixed_versions(code: str, kinds: frozenset[FixKind], setup: str) -> tuple[FixedCode, ...]:
    """``code`` fixed with ``kinds`` of fixes, once for each likely indentation. Empty when it
    can't be fixed. Names defined in ``setup`` count as defined."""
    line_fixes: list[Fix] = []
    while next_fix := _first_fix(code, kinds):
        edit, fix = next_fix
        code = _with_edit(code, edit)
        line_fixes.append(fix)
        if len(line_fixes) > MAX_FIXES:
            return ()
    indented_versions = (
        guess_layouts(code, MAX_INDENTATIONS_TRIED)
        if "indentation" in kinds and _has_no_indentation(code)
        else [code]
    )
    setup_names = _names_defined_in(setup) if "names" in kinds else set()
    versions = []
    for indented in indented_versions:
        indentation_fix = (
            [Fix("indentation", "rebuilt the indentation")] if indented != code else []
        )
        with warnings.catch_warnings(action="ignore"):
            try:
                fixed_code, name_fixes = (
                    _fixed_names(indented, setup_names) if "names" in kinds else (indented, [])
                )
                compile(fixed_code, "<answer>", "exec", dont_inherit=True)
            except SyntaxError, ValueError:
                continue
        fixes = (*indentation_fix, *line_fixes, *name_fixes)
        if fixes:
            versions.append(FixedCode(fixed_code, fixes))
    return tuple(versions)


def _names_defined_in(setup: str) -> set[str]:
    try:
        return _bound_names(symtable.symtable(setup, "<setup>", "exec"))
    except SyntaxError:
        return set()


def _first_fix(code: str, kinds: frozenset[FixKind]) -> _LineFix | None:
    """The first fix of ``kinds`` in ``code``, whatever its indentation."""
    lines = code.split("\n")
    tokens, error = _tokens(lines)
    *statements, unfinished_statement = _statements(tokens)
    for statement in statements:
        for fix in (
            _bracket_fix(statement, code_ends=False),
            _keyword_fix(statement),
            _condition_fix(statement),
            _colon_fix(statement, lines),
        ):
            if fix is not None and fix[1].kind in kinds:
                return fix
    if error is None or "brackets" not in kinds:
        return None
    message, position = error.args
    if fix := _bracket_fix(unfinished_statement, code_ends=message.startswith("unexpected EOF")):
        return fix
    if message.startswith("unterminated") and "triple-quoted" not in message:
        return _quote_fix(unfinished_statement, lines, position)
    return None


def _tokens(lines: list[str]) -> tuple[list[TokenInfo], tokenize.TokenError | None]:
    """The tokens of ``lines`` read without indentation, up to the first error, each at its position
    in ``lines``; and that error, with the position of the token it is in."""
    indents = [len(line) - len(line.lstrip(" \t\f")) for line in lines]

    def original_position(position: LineColumn) -> LineColumn:
        row, column = position
        return row, column + indents[row - 1] if row <= len(lines) else column

    unindented = "\n".join(line[indent:] for line, indent in zip(lines, indents, strict=True))
    tokens: list[TokenInfo] = []
    try:
        # extend keeps the tokens read before an error.
        tokens.extend(
            token._replace(start=original_position(token.start), end=original_position(token.end))
            for token in tokenize.generate_tokens(io.StringIO(unindented).readline)
            if token.type not in IGNORED_TOKEN_TYPES
        )
    except tokenize.TokenError as error:
        message, (row, column) = error.args
        # Columns in tokenize's errors count from 1.
        return tokens, tokenize.TokenError(message, original_position((row, column - 1)))
    return tokens, None


def _statements(tokens: list[TokenInfo]) -> list[list[TokenInfo]]:
    """``tokens`` split into logical lines, the last one being whatever follows the last
    ``NEWLINE``."""
    statements: list[list[TokenInfo]] = [[]]
    for token in tokens:
        if token.type == tokenize.NEWLINE:
            statements.append([])
        else:
            statements[-1].append(token)
    return [statement for statement in statements[:-1] if statement] + statements[-1:]


def _bracket_fix(statement: list[TokenInfo], *, code_ends: bool) -> _LineFix | None:
    """The brackets left open at the end of a line that the next line, or the code ending there,
    shows was finished."""
    for index, token in enumerate(statement):
        following_tokens = statement[index + 1 :]
        if (following_tokens and following_tokens[0].start[0] == token.end[0]) or not (
            _starts_statement(following_tokens) if following_tokens else code_ends
        ):
            continue
        if not (opened := _open_brackets(statement[: index + 1])):
            continue
        if token.string == ":" and _statement_keyword(statement) in HEADER_KEYWORDS:
            position = token.start
        elif _is_operand(token):
            position = token.end
        else:
            continue
        closing = "".join(BRACKETS[bracket] for bracket in reversed(opened))
        description = f"line {position[0]}, closed {quoted(''.join(opened))}"
        return _Edit(position, position, closing), Fix("brackets", description)
    return None


def _starts_statement(tokens: list[TokenInfo]) -> bool:
    """Whether the line starting ``tokens`` can only be a statement of its own."""
    if tokens[0].type != tokenize.NAME:
        return False
    line_tokens = [token for token in tokens if token.start[0] == tokens[0].start[0]]
    word = line_tokens[0].string
    if keyword.iskeyword(word) and (
        word not in STATEMENT_KEYWORDS
        or (word in EXPRESSION_KEYWORDS and line_tokens[-1].string != ":")
    ):
        return False
    depth = 0
    for token in line_tokens:
        depth += (token.string in OPENERS) - (token.string in CLOSERS)
        if depth < 0:
            return False
    return True


def _open_brackets(tokens: list[TokenInfo]) -> list[str]:
    opened: list[str] = []
    for token in tokens:
        if token.string in OPENERS:
            opened.append(token.string)
        elif token.string in CLOSERS and opened:
            opened.pop()
    return opened


def _is_operand(token: TokenInfo) -> bool:
    if token.type == tokenize.NAME:
        return not keyword.iskeyword(token.string) or token.string in LOWERCASE_CONSTANTS.values()
    return token.type in OPERAND_TOKEN_TYPES or token.string in CLOSERS


def _quote_fix(
    unfinished_statement: list[TokenInfo], lines: list[str], position: LineColumn
) -> _LineFix | None:
    """The quote closing a string left open at its line's end, before the brackets and header
    ``:`` the line ends with."""
    row, column = position
    line = lines[row - 1].rstrip()
    start = next((index for index in range(column, len(line)) if line[index] in "'\""), None)
    if start is None:
        return None
    end = len(line)
    if (
        unfinished_statement
        and _statement_keyword(unfinished_statement) in HEADER_KEYWORDS
        and line.endswith(":")
        and end > start + 1
    ):
        end -= 1
    closing = "".join(
        BRACKETS[bracket] for bracket in reversed(_open_brackets(unfinished_statement))
    )
    end -= next(
        size
        for size in range(len(closing), -1, -1)
        if line[:end].endswith(closing[:size]) and end - size > start
    )
    quote = line[start]
    return _Edit((row, end), (row, end), quote), Fix(
        "brackets", f"line {row}, closed {quoted(quote)}"
    )


def _keyword_fix(statement: list[TokenInfo]) -> _LineFix | None:
    """A statement's first_token word as the keyword it misspells or capitalises, when the statement
    can't stand as written."""
    first_token = statement[0]
    word = first_token.string
    if first_token.type != tokenize.NAME or keyword.iskeyword(word) or keyword.issoftkeyword(word):
        return None
    is_capitalized = word.lower() in STATEMENT_KEYWORDS
    matches = [word.lower()] if is_capitalized else _one_edit_away(word, STATEMENT_KEYWORDS)
    if len(matches) != 1:
        return None
    [match] = matches
    next_token = statement[1] if len(statement) > 1 else None
    if next_token is None:
        valid_as_written = match not in STANDALONE_KEYWORDS
    elif next_token.string == ":":
        valid_as_written = match not in KEYWORDS_BEFORE_COLON
    elif next_token.string in ("(", "["):
        valid_as_written = not is_capitalized
    else:
        valid_as_written = next_token.type not in STRING_STARTS | {
            tokenize.NUMBER,
            tokenize.NAME,
        } or (next_token.string in EXPRESSION_KEYWORDS)
    if valid_as_written:
        return None
    description = f"line {first_token.start[0]}, {_read_as(word, match)}"
    return _Edit(first_token.start, first_token.end, match), Fix("keywords", description)


def _condition_fix(statement: list[TokenInfo]) -> _LineFix | None:
    """The one ``=`` in an ``if``, ``elif`` or ``while`` condition, as ``==``."""
    if statement[0].string not in CONDITION_KEYWORDS:
        return None
    equals = [token for token in _top_level_tokens(statement) if token.string == "="]
    if len(equals) != 1:
        return None
    [token] = equals
    description = f"line {token.start[0]}, {_read_as('=', '==')}"
    return _Edit(token.start, token.end, "=="), Fix("equals", description)


def _colon_fix(statement: list[TokenInfo], lines: list[str]) -> _LineFix | None:
    """The ``:`` a block header ends with."""
    if _statement_keyword(statement) not in HEADER_KEYWORDS or any(
        token.string == ":" for token in _top_level_tokens(statement)
    ):
        return None
    end = statement[-1].end
    header = lines[end[0] - 1][: end[1]].strip()
    description = f"line {end[0]}, added {quoted(':')} after {quoted(header)}"
    return _Edit(end, end, ":"), Fix("colons", description)


def _statement_keyword(statement: list[TokenInfo]) -> str:
    """The statement's first word, after any ``async``."""
    if statement[0].string == "async" and len(statement) > 1:
        return statement[1].string
    return statement[0].string


def _top_level_tokens(statement: list[TokenInfo]) -> Iterator[TokenInfo]:
    depth = 0
    for token in statement:
        depth -= token.string in CLOSERS
        if depth == 0:
            yield token
        depth += token.string in OPENERS


def _one_edit_away(word: str, candidates: Iterable[str]) -> list[str]:
    """The candidates one edit from ``word``, if it has three letters or more."""
    if len(word) < 3:
        return []
    return [candidate for candidate in candidates if OSA.distance(word, candidate) == 1]


def _fixed_names(code: str, setup_names: Collection[str]) -> tuple[str, list[Fix]]:
    """``code`` with each name it uses but never defines read as the one defined name or builtin a
    single edit away, or as the constant it writes in lower case; and a fix for each, in order
    of first use."""
    defined_names = (
        _bound_names(symtable.symtable(code, "<answer>", "exec"))
        | set(setup_names)
        | PYTHON_BUILTINS
    )
    undefined_uses: dict[str, list[LineColumn]] = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name) and node.id not in defined_names:
            undefined_uses.setdefault(node.id, []).append((node.lineno, node.col_offset))
    lines = code.split("\n")
    edits: list[_Edit] = []
    name_fixes: list[tuple[LineColumn, Fix]] = []
    for name, positions in undefined_uses.items():
        matches = (
            [LOWERCASE_CONSTANTS[name]]
            if name in LOWERCASE_CONSTANTS
            else _one_edit_away(name, defined_names)
        )
        if len(matches) != 1:
            continue
        description = f"line {min(positions)[0]}, {_read_as(name, matches[0])}"
        name_fixes.append((min(positions), Fix("names", description)))
        for row, offset in positions:
            column = len(lines[row - 1].encode()[:offset].decode())
            edits.append(_Edit((row, column), (row, column + len(name)), matches[0]))
    for edit in sorted(edits, reverse=True):
        code = _with_edit(code, edit)
    return code, [fix for _, fix in sorted(name_fixes)]


def _bound_names(table: symtable.SymbolTable) -> set[str]:
    names = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_parameter()
    }
    for child in table.get_children():
        names |= _bound_names(child)
    return names


def _read_as(written: str, meant: str) -> str:
    return f"{quoted(written)} read as {quoted(meant)}"


def _with_edit(code: str, edit: _Edit) -> str:
    lines = code.split("\n")

    def index(position: LineColumn) -> int:
        row, column = position
        return sum(len(line) + 1 for line in lines[: row - 1]) + column

    return code[: index(edit.start)] + edit.text + code[index(edit.end) :]


def _has_no_indentation(code: str) -> bool:
    """Whether ``code`` has lost its indentation: none of its lines starts with a space or tab."""
    return not any(line[:1] in (" ", "\t") for line in code.split("\n") if line.strip())
