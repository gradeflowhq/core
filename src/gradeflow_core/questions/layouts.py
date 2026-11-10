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

EXIT_KEYWORDS = {"return", "raise", "break", "continue"}
CONTINUED_KEYWORDS = {
    "elif": {"if", "elif"},
    "else": {"if", "elif", "for", "while", "except"},
    "except": {"try", "except"},
    "finally": {"try", "except", "else"},
}
COMPOUND_KEYWORDS = {
    *CONTINUED_KEYWORDS,
    "if",
    "for",
    "while",
    "try",
    "with",
    "def",
    "class",
    "match",
    "case",
}
LOOP_KEYWORDS = {"for", "while"}
SCOPE_KEYWORDS = {"def", "class"}
DEFINITION_STARTS = {*SCOPE_KEYWORDS, "@"}
BRACKETS = {"(": ")", "[": "]", "{": "}"}
OPENING_BRACKETS = tuple(BRACKETS)
CLOSING_BRACKETS = tuple(BRACKETS.values())
NOT_ASSIGNMENTS = {"==", "!=", "<=", ">=", ":="}

type Line = tuple[int | None, str]
"""How many levels deeper than its statement a line goes, or None to keep it as typed, and its
text."""


class _Token(NamedTuple):
    string: str
    type: int
    bracket_depth: int

    @property
    def is_name(self) -> bool:
        return self.type == tokenize.NAME

    def is_operator(self, *strings: str) -> bool:
        return self.type == tokenize.OP and self.string in strings


class _Names(NamedTuple):
    read_names: frozenset[str]
    stored_names: frozenset[str]
    targets: frozenset[str] = frozenset()
    value_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _Statement:
    """A logical line: the lines it renders, its first keyword, whether it opens a block, the
    keyword it leaves its block with, and the block a return or break in it needs."""

    lines: tuple[Line, ...]
    first_keyword: str
    opens_block: bool
    exit_keyword: str
    needed_block: str
    names: _Names


class _OpenBlock(NamedTuple):
    """The latest statement at a depth. ``names`` keep a line in its block: a loop's variables
    and names first set in it, or a function's parameters and names set in it."""

    first_keyword: str
    start_index: int
    names: frozenset[str]


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
    rows = _token_rows(code)
    if rows is None:
        return []
    statements: list[_Statement] = []
    lines: list[Line] = []
    pending_lines: list[Line] = []
    statement_tokens: list[_Token] = []
    for row, text in enumerate(code_lines):
        stripped_text = text.lstrip()
        bracket_depth, token_type = rows.first_tokens.get(row, (0, tokenize.NL))
        if row in rows.rows_in_string:
            lines.append((None, text))
        elif bracket_depth or row - 1 not in rows.rows_ended_by_newline:
            lines.append(
                (0 if stripped_text.startswith(CLOSING_BRACKETS) else 1, stripped_text)
                if text
                else (None, "")
            )
        elif token_type == tokenize.NL:
            pending_lines.append((None, ""))
        elif token_type == tokenize.COMMENT:
            pending_lines.append((0, stripped_text))
        else:
            if lines:
                statements.append(_statement(lines, statement_tokens))
            lines, pending_lines, statement_tokens = [*pending_lines, (0, stripped_text)], [], []
        statement_tokens += rows.tokens[row]
    if not lines:
        return []
    statements.append(_statement(lines + pending_lines, statement_tokens))
    return statements


class _TokenRows(NamedTuple):
    tokens: list[list[_Token]]
    first_tokens: dict[int, tuple[int, int]]
    rows_in_string: set[int]
    rows_ended_by_newline: set[int]


def _token_rows(code: str) -> _TokenRows | None:
    """Each row's code tokens, the bracket depth and type of its first token, the rows inside a
    string and the rows a line break ends."""
    row_count = code.count("\n") + 1
    tokens_by_row: list[list[_Token]] = [[] for _ in range(row_count)]
    first_tokens: dict[int, tuple[int, int]] = {}
    rows_in_string: set[int] = set()
    rows_ended_by_newline: set[int] = set()
    bracket_depth = 0
    # Inside a bracket the tokenizer ignores indentation, so none left or lost can trip it.
    readline = io.StringIO(f"(\n{code}\n)").readline
    try:
        for token in tokenize.generate_tokens(readline):
            row, end_row = token.start[0] - 2, token.end[0] - 2
            if not 0 <= row < row_count:
                rows_ended_by_newline.add(row)
                continue
            first_tokens.setdefault(row, (bracket_depth, token.type))
            operator = token.type == tokenize.OP and token.string
            bracket_depth -= operator in CLOSING_BRACKETS
            if bracket_depth < 0:
                return None
            if token.type == tokenize.NL:
                rows_ended_by_newline.add(row)
            elif token.type != tokenize.COMMENT:
                tokens_by_row[row].append(_Token(token.string, token.type, bracket_depth))
            bracket_depth += operator in OPENING_BRACKETS
            rows_in_string.update(range(row + 1, end_row + 1))
    except SyntaxError, tokenize.TokenError:
        return None
    return _TokenRows(tokens_by_row, first_tokens, rows_in_string, rows_ended_by_newline)


def _statement(lines: list[Line], tokens: list[_Token]) -> _Statement:
    first_word = tokens[0].string
    first_keyword = tokens[1].string if first_word == "async" and len(tokens) > 1 else first_word
    is_compound = first_keyword in COMPOUND_KEYWORDS and any(
        token.is_operator(":") and not token.bracket_depth for token in tokens
    )
    semicolon_indices = [
        i
        for i, token in enumerate(tokens[:-1])
        if token.is_operator(";") and not token.bracket_depth
    ]
    last_statement_keyword = (
        "" if is_compound else tokens[semicolon_indices[-1] + 1 if semicolon_indices else 0].string
    )
    words = {token.string for token in tokens if token.is_name}
    # A yield in brackets may be a lambda's, so only a bare one needs a def.
    if first_keyword != "def" and (
        "return" in words
        or any(token.string == "yield" and not token.bracket_depth for token in tokens)
    ):
        needed_block = "def"
    elif first_keyword not in {*LOOP_KEYWORDS, "def"} and words & {"break", "continue"}:
        needed_block = "loop"
    else:
        needed_block = ""
    return _Statement(
        lines=tuple(lines),
        first_keyword=first_keyword,
        opens_block=tokens[-1].is_operator(":"),
        exit_keyword=last_statement_keyword if last_statement_keyword in EXIT_KEYWORDS else "",
        needed_block=needed_block,
        names=_names(tokens, first_keyword, is_compound),
    )


def _names(tokens: list[_Token], first_keyword: str, is_compound: bool) -> _Names:
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
        [] if is_compound else [i for i, token in enumerate(tokens) if _is_assignment(token)]
    )
    if first_keyword == "for" and is_compound:
        in_index = next(
            (
                i
                for i, token in enumerate(tokens)
                if token.is_name and token.string == "in" and not token.bracket_depth
            ),
            0,
        )
        stored_indices = {i for i in names_by_index if i < in_index}
    elif first_keyword == "def":
        stored_indices = {i for i in names_by_index if tokens[i].bracket_depth}
    else:
        stored_indices = {
            i
            for i in names_by_index
            if assignment_indices
            and i < assignment_indices[-1]
            and not tokens[i].bracket_depth
            and not tokens[i + 1].is_operator(".", "(", "[")
        }
    names = _Names(
        read_names=frozenset(name for i, name in names_by_index.items() if i not in stored_indices),
        stored_names=frozenset(names_by_index[i] for i in stored_indices),
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
        token.type == tokenize.OP
        and token.string.endswith("=")
        and token.string not in NOT_ASSIGNMENTS
        and not token.bracket_depth
    )


def _search(statements: list[_Statement]) -> Iterator[str]:
    """The layouts of ``statements`` that compile and pass the checks, in order of preference,
    as far as the budget goes."""
    blocks_needed_after = _blocks_needed_after(statements)
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
        stack.extend(reversed([*_next_states(state, statements, blocks_needed_after)]))


def _blocks_needed_after(statements: list[_Statement]) -> list[set[str]]:
    """What must stay open after each statement: a def for a later return with no def before it,
    and a loop for a later break with no loop before it."""
    blocks_needed_after: list[set[str]] = []
    blocks_needed_later: set[str] = set()
    for statement in reversed(statements):
        blocks_needed_after.append(set(blocks_needed_later))
        if statement.first_keyword == "def":
            blocks_needed_later.discard("def")
        if statement.first_keyword in LOOP_KEYWORDS:
            blocks_needed_later.discard("loop")
        if statement.needed_block:
            blocks_needed_later.add(statement.needed_block)
    return blocks_needed_after[::-1]


def _next_states(
    state: _State, statements: list[_Statement], blocks_needed_after: list[set[str]]
) -> Iterator[_State]:
    index = len(state.depths)
    statement = statements[index]
    previous_statement = statements[index - 1] if index else None
    for depth in _possible_depths(state.open_blocks, statement, previous_statement):
        enclosing_blocks = state.open_blocks[:depth]
        open_keywords = {block.first_keyword for block in enclosing_blocks}
        if statement.opens_block:
            open_keywords.add(statement.first_keyword)
        if (
            _is_allowed_in_blocks(enclosing_blocks, statement)
            and ("def" not in blocks_needed_after[index] or "def" in open_keywords)
            and ("loop" not in blocks_needed_after[index] or open_keywords & LOOP_KEYWORDS)
            and _closed_loops_change(state, statements, depth)
        ):
            yield _state_after(state, statement, depth)


def _possible_depths(
    open_blocks: tuple[_OpenBlock, ...],
    statement: _Statement,
    previous_statement: _Statement | None,
) -> list[int]:
    """The depths ``statement`` can go to after ``previous_statement``, most likely first."""
    innermost_depth = len(open_blocks) - 1
    if previous_statement is None or previous_statement.opens_block:
        return [] if statement.first_keyword in CONTINUED_KEYWORDS else [innermost_depth + 1]
    if previous_statement.first_keyword == "@":
        return [innermost_depth] if statement.first_keyword in DEFINITION_STARTS else []
    if previous_statement.exit_keyword:
        innermost_depth -= 1
    if statement.first_keyword in CONTINUED_KEYWORDS:
        continued_keywords = CONTINUED_KEYWORDS[statement.first_keyword]
        return [
            depth
            for depth in range(innermost_depth, -1, -1)
            if open_blocks[depth].first_keyword in continued_keywords
        ]
    if statement.first_keyword in DEFINITION_STARTS:
        definition_depth = next(
            (
                depth
                for depth in range(innermost_depth, -1, -1)
                if open_blocks[depth].first_keyword in SCOPE_KEYWORDS
            ),
            0,
        )
        return [
            definition_depth,
            *(depth for depth in range(innermost_depth, -1, -1) if depth != definition_depth),
        ]
    used_names = statement.names.read_names | statement.names.stored_names
    used_names_depth = max(
        (
            depth + 1
            for depth in range(len(open_blocks) - 1)
            if open_blocks[depth].names & used_names
        ),
        default=0,
    )
    preferred_depth = min(used_names_depth, innermost_depth)
    return [*range(preferred_depth, innermost_depth + 1), *range(preferred_depth - 1, -1, -1)]


def _is_allowed_in_blocks(enclosing_blocks: tuple[_OpenBlock, ...], statement: _Statement) -> bool:
    """Whether ``statement`` can go in the blocks ``enclosing_blocks`` open."""
    keywords = [block.first_keyword for block in reversed(enclosing_blocks)]
    if (
        statement.exit_keyword in {"return", "raise", "break"}
        and keywords
        and keywords[0] in LOOP_KEYWORDS
    ):
        return False
    if statement.needed_block == "def":
        return (
            next(
                (block_keyword for block_keyword in keywords if block_keyword in SCOPE_KEYWORDS), ""
            )
            == "def"
        )
    if statement.needed_block == "loop":
        return (
            next(
                (
                    block_keyword
                    for block_keyword in keywords
                    if block_keyword in SCOPE_KEYWORDS | LOOP_KEYWORDS
                ),
                "",
            )
            in LOOP_KEYWORDS
        )
    return True


def _closed_loops_change(state: _State, statements: list[_Statement], depth: int) -> bool:
    """Whether every loop a statement at ``depth`` closes changes something each pass."""
    index = len(state.depths)
    return all(
        _loop_changes_each_pass(statements, state.depths, block.start_index, index)
        for block in state.open_blocks[depth:-1]
        if block.first_keyword in LOOP_KEYWORDS
    )


def _loop_changes_each_pass(
    statements: list[_Statement], depths: tuple[int, ...], start_index: int, end_index: int
) -> bool:
    """Whether the loop from ``start_index`` to ``end_index`` reads each name it sets to the same
    value each pass, unlike ``row = []`` alone in a loop."""
    loop_names = [statements[i].names for i in range(start_index, end_index)]
    names_stored = {name for names in loop_names for name in names.stored_names}
    names_read = {name for names in loop_names for name in names.read_names}
    return all(
        statements[i].names.value_names & names_stored or statements[i].names.targets & names_read
        for i in range(start_index + 1, end_index)
        if depths[i] == depths[start_index] + 1 and statements[i].names.targets
    )


def _state_after(state: _State, statement: _Statement, depth: int) -> _State:
    open_blocks = list(state.open_blocks[:depth])
    names_stored = statement.names.stored_names
    binds_names = statement.first_keyword in {*LOOP_KEYWORDS, "def"}
    if statement.first_keyword != "def":
        _add_names_to_block(open_blocks, SCOPE_KEYWORDS, names_stored)
    if not binds_names:
        _add_names_to_block(open_blocks, LOOP_KEYWORDS, names_stored - state.assigned_names)
    assigned_names = (
        state.assigned_names
        if statement.first_keyword in LOOP_KEYWORDS
        else state.assigned_names | names_stored
    )
    block = _OpenBlock(
        statement.first_keyword, len(state.depths), names_stored if binds_names else frozenset()
    )
    return _State((*state.depths, depth), (*open_blocks, block), assigned_names)


def _add_names_to_block(
    open_blocks: list[_OpenBlock], keywords: set[str], names: frozenset[str]
) -> None:
    """Add ``names`` to the innermost block in ``open_blocks`` opened by one of ``keywords``."""
    block_depth = next(
        (
            depth
            for depth in range(len(open_blocks) - 1, -1, -1)
            if open_blocks[depth].first_keyword in keywords
        ),
        None,
    )
    if block_depth is not None and names:
        open_blocks[block_depth] = open_blocks[block_depth]._replace(
            names=open_blocks[block_depth].names | names
        )


def _render(statements: list[_Statement], depths: tuple[int, ...]) -> str:
    return "\n".join(
        text if offset is None else INDENT * (depth + offset) + text
        for statement, depth in zip(statements, depths, strict=True)
        for offset, text in statement.lines
    )
