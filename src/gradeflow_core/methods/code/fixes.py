"""Small errors in Python code fixed, so code that nearly runs can still be tested: a missing
colon, bracket or quote, ``=`` for ``==``, a misspelt keyword, or a misspelt name."""

import ast
import builtins
import io
import keyword
import symtable
import tokenize
import warnings
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from functools import lru_cache
from tokenize import TokenInfo
from typing import NamedTuple

from ...common.text import quoted
from ...questions.layouts import BRACKETS, CLOSERS, OPENERS, guess_layouts

type Position = tuple[int, int]

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


@dataclass(frozen=True)
class FixedCode:
    code: str
    fixes: tuple[str, ...]
    """Each fix as feedback says it, such as: line 3, added “:” after “def add(a, b)”."""


class _Fix(NamedTuple):
    start: Position
    end: Position
    text: str
    phrase: str = ""


def fix_small_errors(code: str, defined: Collection[str], limit: int) -> FixedCode | None:
    """``code`` with its small errors fixed, or None when it has none, needs more than ``limit``
    fixes, or still doesn't compile. ``defined`` are the names the setup code defines. Code typed
    without indentation comes back in its most likely layout."""
    fixes: list[str] = []
    while fix := _line_fix(code):
        code = _edit(code, fix)
        fixes.append(fix.phrase)
        if len(fixes) > limit:
            return None
    if _is_flat(code):
        layouts = guess_layouts(code, 1)
        if not layouts:
            return None
        code = layouts[0]
    with warnings.catch_warnings(action="ignore"):
        try:
            code, renamed = _fixed_names(code, defined)
            compile(code, "<answer>", "exec", dont_inherit=True)
        except SyntaxError, ValueError:
            return None
    fixes += renamed
    if not fixes or len(fixes) > limit:
        return None
    return FixedCode(code, tuple(fixes))


@lru_cache(maxsize=4096)
def fixed_code(code: str, setup: str, fixes: int) -> FixedCode | None:
    """``code`` with up to ``fixes`` small errors fixed, names the ``setup`` code defines counting
    as defined; None when it has none to fix, or more."""
    if not fixes:
        return None
    try:
        defined = _bound_names(symtable.symtable(setup, "<setup>", "exec"))
    except SyntaxError:
        defined = set()
    return fix_small_errors(code, defined, fixes)


def _line_fix(code: str) -> _Fix | None:
    """The first fix in ``code`` that its lines show, whatever their indentation."""
    lines = code.split("\n")
    tokens, error = _tokens(lines)
    *statements, partial = _statements(tokens)
    for statement in statements:
        if fix := (
            _bracket_fix(statement, at_end=False)
            or _keyword_fix(statement)
            or _condition_fix(statement)
            or _colon_fix(statement, lines)
        ):
            return fix
    if error is None:
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


def _bracket_fix(statement: list[TokenInfo], *, at_end: bool) -> _Fix | None:
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
        return _Fix(place, place, closing, f"line {place[0]}, closed {quoted(''.join(opened))}")
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


def _quote_fix(partial: list[TokenInfo], lines: list[str], place: Position) -> _Fix | None:
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
    return _Fix((row, end), (row, end), quote, f"line {row}, closed {quoted(quote)}")


def _keyword_fix(statement: list[TokenInfo]) -> _Fix | None:
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
        else [match for match in STATEMENT_KEYWORDS if len(word) >= 3 and _one_edit(word, match)]
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
    return _Fix(first.start, first.end, match, f"line {first.start[0]}, {_read_as(word, match)}")


def _condition_fix(statement: list[TokenInfo]) -> _Fix | None:
    """The one ``=`` in an ``if``, ``elif`` or ``while`` condition, as ``==``."""
    if statement[0].string not in CONDITION_KEYWORDS:
        return None
    equals = [token for token in _outside_brackets(statement) if token.string == "="]
    if len(equals) != 1:
        return None
    [token] = equals
    return _Fix(token.start, token.end, "==", f"line {token.start[0]}, {_read_as('=', '==')}")


def _colon_fix(statement: list[TokenInfo], lines: list[str]) -> _Fix | None:
    """The ``:`` a block header ends with."""
    if statement[0].string not in HEADER_KEYWORDS or any(
        token.string == ":" for token in _outside_brackets(statement)
    ):
        return None
    end = statement[-1].end
    header = lines[end[0] - 1][: end[1]].strip()
    return _Fix(end, end, ":", f"line {end[0]}, added {quoted(':')} after {quoted(header)}")


def _outside_brackets(statement: list[TokenInfo]) -> Iterator[TokenInfo]:
    depth = 0
    for token in statement:
        depth -= token.string in CLOSERS
        if depth == 0:
            yield token
        depth += token.string in OPENERS


def _fixed_names(code: str, defined: Collection[str]) -> tuple[str, list[str]]:
    """``code`` with each name it uses but never defines read as the one defined name or builtin a
    single edit away, or as the constant it writes in lower case; and a phrase for each, in order
    of first use."""
    known = _bound_names(symtable.symtable(code, "<answer>", "exec")) | set(defined) | BUILTINS
    uses: dict[str, list[Position]] = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name) and node.id not in known:
            uses.setdefault(node.id, []).append((node.lineno, node.col_offset))
    lines = code.split("\n")
    edits: list[_Fix] = []
    phrases: list[tuple[Position, str]] = []
    for name, places in uses.items():
        matches = (
            [CONSTANTS[name]]
            if name in CONSTANTS
            else [other for other in known if len(name) >= 3 and _one_edit(name, other)]
        )
        if len(matches) != 1:
            continue
        phrases.append((min(places), f"line {min(places)[0]}, {_read_as(name, matches[0])}"))
        for row, offset in places:
            column = len(lines[row - 1].encode()[:offset].decode())
            edits.append(_Fix((row, column), (row, column + len(name)), matches[0]))
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


def _one_edit(word: str, other: str) -> bool:
    """Whether ``word`` is ``other`` with one letter changed, added, removed, or swapped with the
    next."""
    if word == other or abs(len(word) - len(other)) > 1:
        return False
    same = next(
        (i for i, pair in enumerate(zip(word, other, strict=False)) if pair[0] != pair[1]),
        len(word),
    )
    rest, other_rest = word[same:], other[same:]
    return (
        rest[1:] in (other_rest[1:], other_rest)
        or rest == other_rest[1:]
        or (rest[:2] == other_rest[1::-1] and rest[2:] == other_rest[2:])
    )


def _read_as(written: str, meant: str) -> str:
    return f"{quoted(written)} read as {quoted(meant)}"


def _edit(code: str, fix: _Fix) -> str:
    lines = code.split("\n")

    def index(position: Position) -> int:
        row, column = position
        return sum(len(line) + 1 for line in lines[: row - 1]) + column

    return code[: index(fix.start)] + fix.text + code[index(fix.end) :]


def _is_flat(code: str) -> bool:
    """Whether ``code`` has lost its indentation: none of its lines starts with a space or tab."""
    return not any(line[:1] in (" ", "\t") for line in code.split("\n") if line.strip())
