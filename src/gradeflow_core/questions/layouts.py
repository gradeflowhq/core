"""Guessing the indentation exam software drops from Python code."""

import io
import keyword
import tokenize
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import NamedTuple

INDENT = "    "
BUDGET = 20_000  # each state searched costs a step, and each finished layout one per statement

LEAVES = {"return", "raise", "break", "continue"}
CONTINUES = {
    "elif": {"if", "elif"},
    "else": {"if", "elif", "for", "while", "except"},
    "except": {"try", "except"},
    "finally": {"try", "except", "else"},
}
COMPOUND = {*CONTINUES, "if", "for", "while", "try", "with", "def", "class", "match", "case"}
LOOPS = {"for", "while"}
SCOPES = {"def", "class"}
DEFINES = {*SCOPES, "@"}
BRACKETS = {"(": ")", "[": "]", "{": "}"}
OPENERS = tuple(BRACKETS)
CLOSERS = tuple(BRACKETS.values())
NOT_ASSIGNMENTS = {"==", "!=", "<=", ">=", ":="}

type Line = tuple[int | None, str]
"""How many levels deeper than its statement a line goes, or None to keep it as typed, and its
text."""


class _Token(NamedTuple):
    string: str
    kind: int
    depth: int

    @property
    def name(self) -> bool:
        return self.kind == tokenize.NAME

    def op(self, *strings: str) -> bool:
        return self.kind == tokenize.OP and self.string in strings


class _Names(NamedTuple):
    uses: frozenset[str]
    reads: frozenset[str]
    stores: frozenset[str]
    targets: frozenset[str] = frozenset()
    value: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _Statement:
    """A logical line: the lines it renders, its first keyword, whether it opens a block, the
    keyword it leaves its block with, and the block a return or break in it needs."""

    lines: tuple[Line, ...]
    keyword: str
    opens: bool
    leaves: str
    needs: str
    names: _Names


class _Frame(NamedTuple):
    """The latest statement at a depth. ``names`` keep a line in its block: a loop's variables
    and names first set in it, or a function's parameters and names set in it. ``assigns`` says
    a plain assignment sits directly in the block."""

    keyword: str
    start: int
    names: frozenset[str]
    assigns: bool = False


class _State(NamedTuple):
    depths: tuple[int, ...]
    path: tuple[_Frame, ...]
    known: frozenset[str]


def guess_layouts(code: str, limit: int) -> list[str]:
    """``code`` alone when it compiles as typed, otherwise up to ``limit`` layouts of its lines
    that compile and read as code people write, most likely first."""
    if _compiles(code):
        return [code]
    statements = _statements(code)
    return list(islice(_search(statements), limit)) if statements else []


def _compiles(code: str) -> bool:
    with warnings.catch_warnings(action="ignore"):
        try:
            compile(code, "<answer>", "exec", dont_inherit=True)
        except SyntaxError, ValueError:
            return False
    return True


def _statements(code: str) -> list[_Statement]:
    """The statements of ``code``, each with the lines it renders: its continuation lines, and
    the comments and blank lines before it. Lines after the last go with it."""
    rows = code.split("\n")
    tokens = _tokens(code, len(rows))
    if tokens is None:
        return []
    by_row, starts, in_string, ended = tokens
    statements: list[_Statement] = []
    lines: list[Line] = []
    pending: list[Line] = []
    statement_tokens: list[_Token] = []
    for row, text in enumerate(rows):
        stripped = text.lstrip()
        depth, kind = starts.get(row, (0, tokenize.NL))
        if row in in_string:
            lines.append((None, text))
        elif depth or row - 1 not in ended:
            lines.append(
                (0 if stripped.startswith(CLOSERS) else 1, stripped) if text else (None, "")
            )
        elif kind == tokenize.NL:
            pending.append((None, ""))
        elif kind == tokenize.COMMENT:
            pending.append((0, stripped))
        else:
            if lines:
                statements.append(_statement(lines, statement_tokens))
            lines, pending, statement_tokens = [*pending, (0, stripped)], [], []
        statement_tokens += by_row[row]
    if not lines:
        return []
    statements.append(_statement(lines + pending, statement_tokens))
    return statements


def _tokens(
    code: str, count: int
) -> tuple[list[list[_Token]], dict[int, tuple[int, int]], set[int], set[int]] | None:
    """Each row's code tokens, the bracket depth and type of its first token, the rows inside a
    string and the rows a line break ends."""
    by_row: list[list[_Token]] = [[] for _ in range(count)]
    starts: dict[int, tuple[int, int]] = {}
    in_string: set[int] = set()
    ended: set[int] = set()
    depth = 0
    # Inside a bracket the tokenizer ignores indentation, so none left or lost can trip it.
    readline = io.StringIO(f"(\n{code}\n)").readline
    try:
        for token in tokenize.generate_tokens(readline):
            row, end = token.start[0] - 2, token.end[0] - 2
            if not 0 <= row < count:
                ended.add(row)
                continue
            starts.setdefault(row, (depth, token.type))
            bracket = token.type == tokenize.OP and token.string
            depth -= bracket in CLOSERS
            if depth < 0:
                return None
            if token.type == tokenize.NL:
                ended.add(row)
            elif token.type != tokenize.COMMENT:
                by_row[row].append(_Token(token.string, token.type, depth))
            depth += bracket in OPENERS
            in_string.update(range(row + 1, end + 1))
    except SyntaxError, tokenize.TokenError:
        return None
    return by_row, starts, in_string, ended


def _statement(lines: list[Line], tokens: list[_Token]) -> _Statement:
    first = tokens[0].string
    keyword_ = tokens[1].string if first == "async" and len(tokens) > 1 else first
    compound = keyword_ in COMPOUND and any(t.op(":") and not t.depth for t in tokens)
    semicolons = [i for i, t in enumerate(tokens[:-1]) if t.op(";") and not t.depth]
    last = "" if compound else tokens[semicolons[-1] + 1 if semicolons else 0].string
    names = {token.string for token in tokens if token.name}
    # A yield in brackets may be a lambda's, so only a bare one needs a def.
    if keyword_ != "def" and (
        "return" in names or any(t.string == "yield" and not t.depth for t in tokens)
    ):
        needs = "def"
    elif keyword_ not in {*LOOPS, "def"} and names & {"break", "continue"}:
        needs = "loop"
    else:
        needs = ""
    return _Statement(
        lines=tuple(lines),
        keyword=keyword_,
        opens=tokens[-1].op(":"),
        leaves=last if last in LEAVES else "",
        needs=needs,
        names=_names(tokens, keyword_, compound),
    )


def _names(tokens: list[_Token], keyword_: str, compound: bool) -> _Names:
    """The names a statement uses, reads and stores, and for a plain assignment such as
    ``a = b = f(c)`` the names it sets and the names its value uses."""
    used = {
        i: token.string
        for i, token in enumerate(tokens)
        if token.name and not keyword.iskeyword(token.string) and not (i and tokens[i - 1].op("."))
    }
    operators = [] if compound else [i for i, t in enumerate(tokens) if _assigns(t)]
    if keyword_ == "for" and compound:
        end = next(
            (i for i, t in enumerate(tokens) if t.name and t.string == "in" and not t.depth), 0
        )
        stores = {i for i in used if i < end}
    elif keyword_ == "def":
        stores = {i for i in used if tokens[i].depth}
    else:
        stores = {
            i
            for i in used
            if operators
            and i < operators[-1]
            and not tokens[i].depth
            and not tokens[i + 1].op(".", "(", "[")
        }
    names = _Names(
        uses=frozenset(used.values()),
        reads=frozenset(name for i, name in used.items() if i not in stores),
        stores=frozenset(used[i] for i in stores),
    )
    segments = [tokens[a + 1 : b] for a, b in zip([-1, *operators], operators, strict=False)]
    if not segments or not all(
        tokens[i].string == "=" and len(segment) == 1 and segment[0].name
        for i, segment in zip(operators, segments, strict=True)
    ):
        return names
    return names._replace(
        targets=frozenset(segment[0].string for segment in segments),
        value=frozenset(name for i, name in used.items() if i > operators[-1]),
    )


def _assigns(token: _Token) -> bool:
    """Whether ``token`` is ``=`` or an augmented assignment such as ``+=``, outside brackets."""
    return (
        token.kind == tokenize.OP
        and token.string.endswith("=")
        and token.string not in NOT_ASSIGNMENTS
        and not token.depth
    )


def _search(statements: list[_Statement]) -> Iterator[str]:
    """The layouts of ``statements`` that compile and pass the checks, in order of preference,
    as far as the budget goes."""
    needed = _needed(statements)
    steps = 0
    stack = [_State((), (), frozenset())]
    while stack and steps < BUDGET:
        state = stack.pop()
        steps += 1
        if len(state.depths) == len(statements):
            steps += len(statements)
            if _closes(state, statements, 0):
                text = _render(statements, state.depths)
                if _compiles(text):
                    yield text
            continue
        stack.extend(reversed([*_children(state, statements, needed)]))


def _needed(statements: list[_Statement]) -> list[set[str]]:
    """What must stay open after each statement: a def for a later return with no def before it,
    and a loop for a later break with no loop before it."""
    needed: list[set[str]] = []
    later: set[str] = set()
    for statement in reversed(statements):
        needed.append(set(later))
        if statement.keyword == "def":
            later.discard("def")
        if statement.keyword in LOOPS:
            later.discard("loop")
        if statement.needs:
            later.add(statement.needs)
    return needed[::-1]


def _children(
    state: _State, statements: list[_Statement], needed: list[set[str]]
) -> Iterator[_State]:
    index = len(state.depths)
    statement = statements[index]
    previous = statements[index - 1] if index else None
    for depth in _depths(state.path, statement, previous):
        headers = state.path[:depth]
        opened = {frame.keyword for frame in headers}
        if statement.opens:
            opened.add(statement.keyword)
        if (
            _fits(headers, statement)
            and ("def" not in needed[index] or "def" in opened)
            and ("loop" not in needed[index] or opened & LOOPS)
            and _closes(state, statements, depth)
        ):
            yield _placed(state, statement, depth)


def _depths(
    path: tuple[_Frame, ...], statement: _Statement, previous: _Statement | None
) -> list[int]:
    """The depths ``statement`` can go to after ``previous``, most likely first."""
    top = len(path) - 1
    if previous is None or previous.opens:
        return [] if statement.keyword in CONTINUES else [top + 1]
    if previous.keyword == "@":
        return [top] if statement.keyword in DEFINES else []
    if previous.leaves:
        top -= 1
    if statement.keyword in CONTINUES:
        fits = CONTINUES[statement.keyword]
        return [depth for depth in range(top, -1, -1) if path[depth].keyword in fits]
    if statement.keyword in DEFINES:
        sibling = next((d for d in range(top, -1, -1) if path[d].keyword in SCOPES), 0)
        return [sibling, *(depth for depth in range(top, -1, -1) if depth != sibling)]
    uses = statement.names.uses
    inside = max((d + 1 for d in range(len(path) - 1) if path[d].names & uses), default=0)
    bound = min(inside, top)
    return [*range(bound, top + 1), *range(bound - 1, -1, -1)]


def _fits(headers: tuple[_Frame, ...], statement: _Statement) -> bool:
    """Whether ``statement`` can go in the blocks ``headers`` open."""
    keywords = [frame.keyword for frame in reversed(headers)]
    if statement.leaves in {"return", "raise", "break"} and keywords and keywords[0] in LOOPS:
        return False
    if statement.needs == "def":
        return next((k for k in keywords if k in SCOPES), "") == "def"
    if statement.needs == "loop":
        return next((k for k in keywords if k in SCOPES | LOOPS), "") in LOOPS
    return True


def _closes(state: _State, statements: list[_Statement], depth: int) -> bool:
    """Whether every loop a statement at ``depth`` closes changes something each pass."""
    index = len(state.depths)
    return all(
        _changes(statements, state.depths, frame.start, index)
        for frame in state.path[depth:-1]
        if frame.keyword in LOOPS and frame.assigns
    )


def _changes(statements: list[_Statement], depths: tuple[int, ...], start: int, end: int) -> bool:
    """Whether the loop from ``start`` to ``end`` reads each name it sets to the same value each
    pass, unlike ``row = []`` alone in a loop."""
    loop = [statements[i].names for i in range(start, end)]
    changed = {name for names in loop for name in names.stores}
    read = {name for names in loop for name in names.reads}
    return all(
        statements[i].names.value & changed or statements[i].names.targets & read
        for i in range(start + 1, end)
        if depths[i] == depths[start] + 1 and statements[i].names.targets
    )


def _placed(state: _State, statement: _Statement, depth: int) -> _State:
    path = list(state.path[:depth])
    stores = statement.names.stores
    binds = statement.keyword in {*LOOPS, "def"}
    if statement.keyword != "def":
        _keep(path, SCOPES, stores)
    if not binds:
        _keep(path, LOOPS, stores - state.known)
    if depth and statement.names.targets:
        path[-1] = path[-1]._replace(assigns=True)
    known = state.known if statement.keyword in LOOPS else state.known | stores
    frame = _Frame(statement.keyword, len(state.depths), stores if binds else frozenset())
    return _State((*state.depths, depth), (*path, frame), known)


def _keep(path: list[_Frame], keywords: set[str], names: frozenset[str]) -> None:
    """Add ``names`` to the innermost block in ``path`` opened by one of ``keywords``."""
    block = next((d for d in range(len(path) - 1, -1, -1) if path[d].keyword in keywords), None)
    if block is not None and names:
        path[block] = path[block]._replace(names=path[block].names | names)


def _render(statements: list[_Statement], depths: tuple[int, ...]) -> str:
    return "\n".join(
        text if offset is None else INDENT * (depth + offset) + text
        for statement, depth in zip(statements, depths, strict=True)
        for offset, text in statement.lines
    )
