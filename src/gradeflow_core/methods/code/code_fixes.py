"""Fix code: small errors in Python code fixed before marking."""

import ast
import builtins
import io
import keyword
import symtable
import tokenize
import warnings
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, replace
from functools import lru_cache
from tokenize import TokenInfo
from typing import Any, Literal, NamedTuple, get_args

from pydantic import ConfigDict, Field
from rapidfuzz.distance import OSA

from ...common.hints import ADVANCED, hint
from ...common.text import capitalised, format_number, quoted
from ...questions.layouts import BRACKETS, CLOSERS, OPENERS, guess_layouts
from ..outcome import Detail, Outcome
from ..settings import Settings

type Position = tuple[int, int]
FixKind = Literal["indentation", "colons", "brackets", "equals", "keywords", "names"]

LAYOUTS = 3
"""How many layouts to try for code with no indentation."""
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
HEADER_KEYWORDS = frozenset(
    {"class", "def", "elif", "else", "except", "finally", "for", "if", "try", "while", "with"}
)
CONDITION_KEYWORDS = frozenset({"if", "elif", "while"})
STANDALONE_KEYWORDS = frozenset({"pass", "break", "continue", "return", "else", "try", "finally"})
COLON_KEYWORDS = frozenset({"else", "try", "finally", "except"})
# Keywords that can follow an expression, so "word if …" may be a valid statement.
EXPRESSION_KEYWORDS = frozenset({"and", "or", "not", "in", "is", "if", "else", "for", "as"})
CONSTANTS = {"true": "True", "false": "False", "none": "None"}
STRING_STARTS = frozenset({tokenize.STRING, tokenize.FSTRING_START, tokenize.TSTRING_START})
OPERANDS = frozenset({tokenize.NUMBER, tokenize.STRING, tokenize.FSTRING_END, tokenize.TSTRING_END})
SKIPPED = frozenset(
    {tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
)
BUILTINS = frozenset(dir(builtins))


class Fix(NamedTuple):
    kind: FixKind
    phrase: str
    """The fix as feedback says it, such as: line 3, added “:” after “def add(a, b)”."""


@dataclass(frozen=True)
class FixedCode:
    code: str
    fixes: tuple[Fix, ...]


class _Edit(NamedTuple):
    start: Position
    end: Position
    text: str


type _LineFix = tuple[_Edit, Fix]


def _penalty(kind: FixKind) -> Any:
    return Field(
        default=0,
        ge=0,
        le=1,
        title="Penalty",
        examples=[0.1, 0.25, 0.5],
        json_schema_extra=hint(applies_when={kind: True}),
    )


class CodeFixes(Settings):
    model_config = ConfigDict(title="Fix code", frozen=True)

    indentation: bool = Field(
        default=False,
        title="Indentation",
        description="Lost indentation is rebuilt.",
    )
    indentation_penalty: float = _penalty("indentation")
    colons: bool = Field(
        default=False,
        title="Missing colons",
        description="“if x > 0” reads as “if x > 0:”.",
    )
    colons_penalty: float = _penalty("colons")
    brackets: bool = Field(
        default=False,
        title="Brackets and quotes",
        description="“print(x” reads as “print(x)”.",
    )
    brackets_penalty: float = _penalty("brackets")
    equals: bool = Field(
        default=False,
        title="Equals in conditions",
        description="“if x = 1” reads as “if x == 1”.",
    )
    equals_penalty: float = _penalty("equals")
    keywords: bool = Field(
        default=False,
        title="Misspelt keywords",
        description="“retrun” reads as “return”.",
    )
    keywords_penalty: float = _penalty("keywords")
    names: bool = Field(
        default=False,
        title="Misspelt names",
        description="“lenght” reads as “length”, and “true” as “True”.",
    )
    names_penalty: float = _penalty("names")

    @property
    def kinds(self) -> frozenset[FixKind]:
        return frozenset(kind for kind in get_args(FixKind) if getattr(self, kind))

    def penalty(self, kind: FixKind) -> float:
        """The share of the marks each fix of ``kind`` takes off."""
        return float(getattr(self, f"{kind}_penalty"))

    def total_penalty(self, fixed: FixedCode) -> float:
        """The share of the marks the fixes in ``fixed`` take off, at most 1."""
        return min(1.0, sum(self.penalty(fix.kind) for fix in fixed.fixes))


def fixes_field() -> Any:
    """A code method's ``fixes`` field."""
    return Field(
        default={},
        validate_default=True,
        title="Fix code",
        description="Small errors fixed before marking.",
        json_schema_extra=ADVANCED,
    )


def best_fixed_outcome(
    code: str, fixes: CodeFixes, setup: str, mark: Callable[[str, float], Outcome]
) -> Outcome:
    """The best outcome ``mark`` gives ``code`` or one of its fixed versions; the earliest on a tie.
    ``setup`` is the code run before it."""
    best = mark(code, 0)
    chosen: FixedCode | None = None
    for fixed in fixed_versions(code, fixes.kinds, setup):
        if best.correct:
            break
        outcome = mark(fixed.code, fixes.total_penalty(fixed))
        if (outcome.marked, outcome.proportion) > (best.marked, best.proportion):
            best, chosen = outcome, fixed
    if chosen is None:
        return best
    listed = "\n".join(f"- {_fix_line(fix, fixes.penalty(fix.kind))}" for fix in chosen.fixes)
    return replace(
        best,
        feedback=f"Fixed before marking:\n{listed}\n{best.feedback}",
        details=(Detail("Marked as", chosen.code, "code"), *best.details),
    )


def _fix_line(fix: Fix, penalty: float) -> str:
    suffix = f" (-{format_number(penalty * 100)}%)" if penalty else ""
    return f"{capitalised(fix.phrase)}{suffix}."


@lru_cache(maxsize=4096)
def fixed_versions(code: str, kinds: frozenset[FixKind], setup: str) -> tuple[FixedCode, ...]:
    """``code`` fixed with ``kinds`` of fixes, once for each likely layout. Empty when it can't be
    fixed. Names defined in ``setup`` count as defined."""
    line_fixes: list[Fix] = []
    while found := _line_fix(code, kinds):
        edit, fix = found
        code = _edit(code, edit)
        line_fixes.append(fix)
        if len(line_fixes) > MAX_FIXES:
            return ()
    layouts = guess_layouts(code, LAYOUTS) if "indentation" in kinds and _is_flat(code) else [code]
    defined = _defined(setup) if "names" in kinds else set()
    versions = []
    for layout in layouts:
        indentation = [Fix("indentation", "rebuilt the indentation")] if layout != code else []
        with warnings.catch_warnings(action="ignore"):
            try:
                fixed, renamed = _fixed_names(layout, defined) if "names" in kinds else (layout, [])
                compile(fixed, "<answer>", "exec", dont_inherit=True)
            except SyntaxError, ValueError:
                continue
        fixes = (*indentation, *line_fixes, *renamed)
        if fixes:
            versions.append(FixedCode(fixed, fixes))
    return tuple(versions)


def _defined(setup: str) -> set[str]:
    try:
        return _bound_names(symtable.symtable(setup, "<setup>", "exec"))
    except SyntaxError:
        return set()


def _line_fix(code: str, kinds: frozenset[FixKind]) -> _LineFix | None:
    """The first fix of ``kinds`` in ``code``, whatever its indentation."""
    lines = code.split("\n")
    tokens, error = _tokens(lines)
    *statements, partial = _statements(tokens)
    for statement in statements:
        for fix in (
            _bracket_fix(statement, at_end=False),
            _keyword_fix(statement),
            _condition_fix(statement),
            _colon_fix(statement, lines),
        ):
            if fix is not None and fix[1].kind in kinds:
                return fix
    if error is None or "brackets" not in kinds:
        return None
    message, place = error.args
    if fix := _bracket_fix(partial, at_end=message.startswith("unexpected EOF")):
        return fix
    if message.startswith("unterminated") and "triple-quoted" not in message:
        return _quote_fix(partial, lines, place)
    return None


def _tokens(lines: list[str]) -> tuple[list[TokenInfo], tokenize.TokenError | None]:
    """The tokens of ``lines`` read without indentation, up to the first error, each at its place
    in ``lines``; and that error, with the place of the token it is in."""
    indents = [len(line) - len(line.lstrip(" \t\f")) for line in lines]

    def placed(position: Position) -> Position:
        row, column = position
        return row, column + indents[row - 1] if row <= len(lines) else column

    flat = "\n".join(line[indent:] for line, indent in zip(lines, indents, strict=True))
    tokens: list[TokenInfo] = []
    try:
        # extend keeps the tokens read before an error.
        tokens.extend(
            token._replace(start=placed(token.start), end=placed(token.end))
            for token in tokenize.generate_tokens(io.StringIO(flat).readline)
            if token.type not in SKIPPED
        )
    except tokenize.TokenError as error:
        message, (row, column) = error.args
        # Columns in tokenize's errors count from 1.
        return tokens, tokenize.TokenError(message, placed((row, column - 1)))
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


def _bracket_fix(statement: list[TokenInfo], *, at_end: bool) -> _LineFix | None:
    """The brackets left open at the end of a line that the next line, or the code ending there,
    shows was finished."""
    for index, token in enumerate(statement):
        rest = statement[index + 1 :]
        if (rest and rest[0].start[0] == token.end[0]) or not (
            _starts_statement(rest) if rest else at_end
        ):
            continue
        if not (opened := _open_brackets(statement[: index + 1])):
            continue
        if token.string == ":" and statement[0].string in HEADER_KEYWORDS:
            place = token.start
        elif _is_operand(token):
            place = token.end
        else:
            continue
        closing = "".join(BRACKETS[bracket] for bracket in reversed(opened))
        phrase = f"line {place[0]}, closed {quoted(''.join(opened))}"
        return _Edit(place, place, closing), Fix("brackets", phrase)
    return None


def _starts_statement(tokens: list[TokenInfo]) -> bool:
    """Whether the line starting ``tokens`` can only be a statement of its own."""
    if tokens[0].type != tokenize.NAME:
        return False
    row = [token for token in tokens if token.start[0] == tokens[0].start[0]]
    word = row[0].string
    if keyword.iskeyword(word) and (
        word not in STATEMENT_KEYWORDS or (word in EXPRESSION_KEYWORDS and row[-1].string != ":")
    ):
        return False
    depth = 0
    for token in row:
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
        return not keyword.iskeyword(token.string) or token.string in CONSTANTS.values()
    return token.type in OPERANDS or token.string in CLOSERS


def _quote_fix(partial: list[TokenInfo], lines: list[str], place: Position) -> _LineFix | None:
    """The quote closing a string left open at its line's end, before the brackets and header
    ``:`` the line ends with."""
    row, column = place
    line = lines[row - 1].rstrip()
    start = next((index for index in range(column, len(line)) if line[index] in "'\""), None)
    if start is None:
        return None
    end = len(line)
    if partial and partial[0].string in HEADER_KEYWORDS and line.endswith(":") and end > start + 1:
        end -= 1
    closing = "".join(BRACKETS[bracket] for bracket in reversed(_open_brackets(partial)))
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
    """A statement's first word as the keyword it misspells or capitalises, when the statement
    can't stand as written."""
    first = statement[0]
    word = first.string
    if first.type != tokenize.NAME or keyword.iskeyword(word) or keyword.issoftkeyword(word):
        return None
    capitalised = word.lower() in STATEMENT_KEYWORDS
    matches = (
        [word.lower()]
        if capitalised
        else [
            match
            for match in STATEMENT_KEYWORDS
            if len(word) >= 3 and OSA.distance(word, match) == 1
        ]
    )
    if len(matches) != 1:
        return None
    [match] = matches
    following = statement[1] if len(statement) > 1 else None
    if following is None:
        stands = match not in STANDALONE_KEYWORDS
    elif following.string == ":":
        stands = match not in COLON_KEYWORDS
    elif following.string in ("(", "["):
        stands = not capitalised
    else:
        stands = following.type not in STRING_STARTS | {tokenize.NUMBER, tokenize.NAME} or (
            following.string in EXPRESSION_KEYWORDS
        )
    if stands:
        return None
    phrase = f"line {first.start[0]}, {_read_as(word, match)}"
    return _Edit(first.start, first.end, match), Fix("keywords", phrase)


def _condition_fix(statement: list[TokenInfo]) -> _LineFix | None:
    """The one ``=`` in an ``if``, ``elif`` or ``while`` condition, as ``==``."""
    if statement[0].string not in CONDITION_KEYWORDS:
        return None
    equals = [token for token in _outside_brackets(statement) if token.string == "="]
    if len(equals) != 1:
        return None
    [token] = equals
    phrase = f"line {token.start[0]}, {_read_as('=', '==')}"
    return _Edit(token.start, token.end, "=="), Fix("equals", phrase)


def _colon_fix(statement: list[TokenInfo], lines: list[str]) -> _LineFix | None:
    """The ``:`` a block header ends with."""
    if statement[0].string not in HEADER_KEYWORDS or any(
        token.string == ":" for token in _outside_brackets(statement)
    ):
        return None
    end = statement[-1].end
    header = lines[end[0] - 1][: end[1]].strip()
    phrase = f"line {end[0]}, added {quoted(':')} after {quoted(header)}"
    return _Edit(end, end, ":"), Fix("colons", phrase)


def _outside_brackets(statement: list[TokenInfo]) -> Iterator[TokenInfo]:
    depth = 0
    for token in statement:
        depth -= token.string in CLOSERS
        if depth == 0:
            yield token
        depth += token.string in OPENERS


def _fixed_names(code: str, defined: Collection[str]) -> tuple[str, list[Fix]]:
    """``code`` with each name it uses but never defines read as the one defined name or builtin a
    single edit away, or as the constant it writes in lower case; and a phrase for each, in order
    of first use."""
    known = _bound_names(symtable.symtable(code, "<answer>", "exec")) | set(defined) | BUILTINS
    uses: dict[str, list[Position]] = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name) and node.id not in known:
            uses.setdefault(node.id, []).append((node.lineno, node.col_offset))
    lines = code.split("\n")
    edits: list[_Edit] = []
    phrases: list[tuple[Position, Fix]] = []
    for name, places in uses.items():
        matches = (
            [CONSTANTS[name]]
            if name in CONSTANTS
            else [other for other in known if len(name) >= 3 and OSA.distance(name, other) == 1]
        )
        if len(matches) != 1:
            continue
        phrase = f"line {min(places)[0]}, {_read_as(name, matches[0])}"
        phrases.append((min(places), Fix("names", phrase)))
        for row, offset in places:
            column = len(lines[row - 1].encode()[:offset].decode())
            edits.append(_Edit((row, column), (row, column + len(name)), matches[0]))
    for edit in sorted(edits, reverse=True):
        code = _edit(code, edit)
    return code, [phrase for _, phrase in sorted(phrases)]


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


def _edit(code: str, edit: _Edit) -> str:
    lines = code.split("\n")

    def index(position: Position) -> int:
        row, column = position
        return sum(len(line) + 1 for line in lines[: row - 1]) + column

    return code[: index(edit.start)] + edit.text + code[index(edit.end) :]


def _is_flat(code: str) -> bool:
    """Whether ``code`` has lost its indentation: none of its lines starts with a space or tab."""
    return not any(line[:1] in (" ", "\t") for line in code.split("\n") if line.strip())
