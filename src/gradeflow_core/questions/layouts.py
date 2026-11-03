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
# Each state searched costs a step, and each finished layout one per statement.
MAX_SEARCH_STEPS = 20_000

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
    def is_name(self) -> bool:
        return self.kind == tokenize.NAME

    def is_operator(self, *strings: str) -> bool:
        return self.kind == tokenize.OP and self.string in strings


class _Names(NamedTuple):
    uses: frozenset[str]
    reads: frozenset[str]
    stores: frozenset[str]
    targets: frozenset[str] = frozenset()
    value_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _Statement:
    """A logical line: the lines it renders, its first keyword, whether it opens a block, the
    keyword it leaves its block with, and the block a return or break in it needs."""

    lines: tuple[Line, ...]
    keyword: str
    opens_block: bool
    exit_keyword: str
    needs_block: str
    names: _Names


class _OpenBlock(NamedTuple):
    """The latest statement at a depth. ``names`` keep a line in its block: a loop's variables
    and names first set in it, or a function's parameters and names set in it. ``has_assignment``
    says a plain assignment sits directly in the block."""

    keyword: str
    start: int
    names: frozenset[str]
    has_assignment: bool = False


class _State(NamedTuple):
    depths: tuple[int, ...]
    open_blocks: tuple[_OpenBlock, ...]
    assigned_names: frozenset[str]


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
    code_lines = code.split("\n")
    tokens = _tokens(code, len(code_lines))
    if tokens is None:
        return []
    tokens_by_row, first_tokens, rows_in_string, ended_rows = tokens
    statements: list[_Statement] = []
    lines: list[Line] = []
    pending: list[Line] = []
    statement_tokens: list[_Token] = []
    for row, text in enumerate(code_lines):
        stripped = text.lstrip()
        depth, kind = first_tokens.get(row, (0, tokenize.NL))
        if row in rows_in_string:
            lines.append((None, text))
        elif depth or row - 1 not in ended_rows:
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
        statement_tokens += tokens_by_row[row]
    if not lines:
        return []
    statements.append(_statement(lines + pending, statement_tokens))
    return statements


def _tokens(
    code: str, count: int
) -> tuple[list[list[_Token]], dict[int, tuple[int, int]], set[int], set[int]] | None:
    """Each row's code tokens, the bracket depth and type of its first token, the rows inside a
    string and the rows a line break ends."""
    tokens_by_row: list[list[_Token]] = [[] for _ in range(count)]
    first_tokens: dict[int, tuple[int, int]] = {}
    rows_in_string: set[int] = set()
    ended_rows: set[int] = set()
    depth = 0
    # Inside a bracket the tokenizer ignores indentation, so none left or lost can trip it.
    readline = io.StringIO(f"(\n{code}\n)").readline
    try:
        for token in tokenize.generate_tokens(readline):
            row, end = token.start[0] - 2, token.end[0] - 2
            if not 0 <= row < count:
                ended_rows.add(row)
                continue
            first_tokens.setdefault(row, (depth, token.type))
            bracket = token.type == tokenize.OP and token.string
            depth -= bracket in CLOSERS
            if depth < 0:
                return None
            if token.type == tokenize.NL:
                ended_rows.add(row)
            elif token.type != tokenize.COMMENT:
                tokens_by_row[row].append(_Token(token.string, token.type, depth))
            depth += bracket in OPENERS
            rows_in_string.update(range(row + 1, end + 1))
    except SyntaxError, tokenize.TokenError:
        return None
    return tokens_by_row, first_tokens, rows_in_string, ended_rows


def _statement(lines: list[Line], tokens: list[_Token]) -> _Statement:
    first = tokens[0].string
    first_keyword = tokens[1].string if first == "async" and len(tokens) > 1 else first
    compound = first_keyword in COMPOUND and any(
        token.is_operator(":") and not token.depth for token in tokens
    )
    semicolons = [
        i for i, token in enumerate(tokens[:-1]) if token.is_operator(";") and not token.depth
    ]
    last_word = "" if compound else tokens[semicolons[-1] + 1 if semicolons else 0].string
    names = {token.string for token in tokens if token.is_name}
    # A yield in brackets may be a lambda's, so only a bare one needs a def.
    if first_keyword != "def" and (
        "return" in names or any(token.string == "yield" and not token.depth for token in tokens)
    ):
        needs = "def"
    elif first_keyword not in {*LOOPS, "def"} and names & {"break", "continue"}:
        needs = "loop"
    else:
        needs = ""
    return _Statement(
        lines=tuple(lines),
        keyword=first_keyword,
        opens_block=tokens[-1].is_operator(":"),
        exit_keyword=last_word if last_word in LEAVES else "",
        needs_block=needs,
        names=_names(tokens, first_keyword, compound),
    )


def _names(tokens: list[_Token], first_keyword: str, compound: bool) -> _Names:
    """The names a statement uses, reads and stores, and for a plain assignment such as
    ``a = b = f(c)`` the names it sets and the names its value uses."""
    names_by_index = {
        i: token.string
        for i, token in enumerate(tokens)
        if token.is_name
        and not keyword.iskeyword(token.string)
        and not (i and tokens[i - 1].is_operator("."))
    }
    assignment_indices = (
        [] if compound else [i for i, token in enumerate(tokens) if _is_assignment(token)]
    )
    if first_keyword == "for" and compound:
        end = next(
            (
                i
                for i, token in enumerate(tokens)
                if token.is_name and token.string == "in" and not token.depth
            ),
            0,
        )
        stored_indices = {i for i in names_by_index if i < end}
    elif first_keyword == "def":
        stored_indices = {i for i in names_by_index if tokens[i].depth}
    else:
        stored_indices = {
            i
            for i in names_by_index
            if assignment_indices
            and i < assignment_indices[-1]
            and not tokens[i].depth
            and not tokens[i + 1].is_operator(".", "(", "[")
        }
    names = _Names(
        uses=frozenset(names_by_index.values()),
        reads=frozenset(name for i, name in names_by_index.items() if i not in stored_indices),
        stores=frozenset(names_by_index[i] for i in stored_indices),
    )
    segments = [
        tokens[start + 1 : end]
        for start, end in zip([-1, *assignment_indices], assignment_indices, strict=False)
    ]
    if not segments or not all(
        tokens[i].string == "=" and len(segment) == 1 and segment[0].is_name
        for i, segment in zip(assignment_indices, segments, strict=True)
    ):
        return names
    return names._replace(
        targets=frozenset(segment[0].string for segment in segments),
        value_names=frozenset(
            name for i, name in names_by_index.items() if i > assignment_indices[-1]
        ),
    )


def _is_assignment(token: _Token) -> bool:
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
    while stack and steps < MAX_SEARCH_STEPS:
        state = stack.pop()
        steps += 1
        if len(state.depths) == len(statements):
            steps += len(statements)
            if _closed_loops_change(state, statements, 0):
                text = _render(statements, state.depths)
                if _compiles(text):
                    yield text
            continue
        stack.extend(reversed([*_next_states(state, statements, needed)]))


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
        if statement.needs_block:
            later.add(statement.needs_block)
    return needed[::-1]


def _next_states(
    state: _State, statements: list[_Statement], needed: list[set[str]]
) -> Iterator[_State]:
    index = len(state.depths)
    statement = statements[index]
    previous = statements[index - 1] if index else None
    for depth in _depths(state.open_blocks, statement, previous):
        enclosing_blocks = state.open_blocks[:depth]
        open_keywords = {block.keyword for block in enclosing_blocks}
        if statement.opens_block:
            open_keywords.add(statement.keyword)
        if (
            _fits(enclosing_blocks, statement)
            and ("def" not in needed[index] or "def" in open_keywords)
            and ("loop" not in needed[index] or open_keywords & LOOPS)
            and _closed_loops_change(state, statements, depth)
        ):
            yield _state_after(state, statement, depth)


def _depths(
    open_blocks: tuple[_OpenBlock, ...], statement: _Statement, previous: _Statement | None
) -> list[int]:
    """The depths ``statement`` can go to after ``previous``, most likely first."""
    innermost = len(open_blocks) - 1
    if previous is None or previous.opens_block:
        return [] if statement.keyword in CONTINUES else [innermost + 1]
    if previous.keyword == "@":
        return [innermost] if statement.keyword in DEFINES else []
    if previous.exit_keyword:
        innermost -= 1
    if statement.keyword in CONTINUES:
        continuable = CONTINUES[statement.keyword]
        return [
            depth for depth in range(innermost, -1, -1) if open_blocks[depth].keyword in continuable
        ]
    if statement.keyword in DEFINES:
        definition_depth = next(
            (depth for depth in range(innermost, -1, -1) if open_blocks[depth].keyword in SCOPES), 0
        )
        return [
            definition_depth,
            *(depth for depth in range(innermost, -1, -1) if depth != definition_depth),
        ]
    uses = statement.names.uses
    using_depth = max(
        (depth + 1 for depth in range(len(open_blocks) - 1) if open_blocks[depth].names & uses),
        default=0,
    )
    least_depth = min(using_depth, innermost)
    return [*range(least_depth, innermost + 1), *range(least_depth - 1, -1, -1)]


def _fits(enclosing_blocks: tuple[_OpenBlock, ...], statement: _Statement) -> bool:
    """Whether ``statement`` can go in the blocks ``enclosing_blocks`` open."""
    keywords = [block.keyword for block in reversed(enclosing_blocks)]
    if statement.exit_keyword in {"return", "raise", "break"} and keywords and keywords[0] in LOOPS:
        return False
    if statement.needs_block == "def":
        return (
            next((block_keyword for block_keyword in keywords if block_keyword in SCOPES), "")
            == "def"
        )
    if statement.needs_block == "loop":
        return (
            next(
                (block_keyword for block_keyword in keywords if block_keyword in SCOPES | LOOPS), ""
            )
            in LOOPS
        )
    return True


def _closed_loops_change(state: _State, statements: list[_Statement], depth: int) -> bool:
    """Whether every loop a statement at ``depth`` closes changes something each pass."""
    index = len(state.depths)
    return all(
        _loop_changes_each_pass(statements, state.depths, block.start, index)
        for block in state.open_blocks[depth:-1]
        if block.keyword in LOOPS and block.has_assignment
    )


def _loop_changes_each_pass(
    statements: list[_Statement], depths: tuple[int, ...], start: int, end: int
) -> bool:
    """Whether the loop from ``start`` to ``end`` reads each name it sets to the same value each
    pass, unlike ``row = []`` alone in a loop."""
    loop = [statements[i].names for i in range(start, end)]
    changed = {name for names in loop for name in names.stores}
    names_read = {name for names in loop for name in names.reads}
    return all(
        statements[i].names.value_names & changed or statements[i].names.targets & names_read
        for i in range(start + 1, end)
        if depths[i] == depths[start] + 1 and statements[i].names.targets
    )


def _state_after(state: _State, statement: _Statement, depth: int) -> _State:
    open_blocks = list(state.open_blocks[:depth])
    stores = statement.names.stores
    binds_names = statement.keyword in {*LOOPS, "def"}
    if statement.keyword != "def":
        _add_names_to_block(open_blocks, SCOPES, stores)
    if not binds_names:
        _add_names_to_block(open_blocks, LOOPS, stores - state.assigned_names)
    if depth and statement.names.targets:
        open_blocks[-1] = open_blocks[-1]._replace(has_assignment=True)
    assigned_names = (
        state.assigned_names if statement.keyword in LOOPS else state.assigned_names | stores
    )
    block = _OpenBlock(statement.keyword, len(state.depths), stores if binds_names else frozenset())
    return _State((*state.depths, depth), (*open_blocks, block), assigned_names)


def _add_names_to_block(
    open_blocks: list[_OpenBlock], keywords: set[str], names: frozenset[str]
) -> None:
    """Add ``names`` to the innermost block in ``open_blocks`` opened by one of ``keywords``."""
    block = next(
        (
            depth
            for depth in range(len(open_blocks) - 1, -1, -1)
            if open_blocks[depth].keyword in keywords
        ),
        None,
    )
    if block is not None and names:
        open_blocks[block] = open_blocks[block]._replace(names=open_blocks[block].names | names)


def _render(statements: list[_Statement], depths: tuple[int, ...]) -> str:
    return "\n".join(
        text if offset is None else INDENT * (depth + offset) + text
        for statement, depth in zip(statements, depths, strict=True)
        for offset, text in statement.lines
    )
