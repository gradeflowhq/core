"""Arithmetic expressions as people type them, such as 2(x + 1) or x², read without ``eval``."""

import math
import operator
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ...common.text import quoted

MAX_LENGTH = 200
FUNCTIONS: dict[str, Callable[[float], float]] = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "abs": math.fabs,
}
CONSTANTS = {"pi": math.pi, "π": math.pi, "e": math.e}
RESERVED = FUNCTIONS.keys() | CONSTANTS.keys()
"""Names a variable can't have."""

type Values = Mapping[str, float]
type Node = Callable[[Values], float]

_SYMBOLS = str.maketrans({"×": "*", "·": "*", "÷": "/", "\N{MINUS SIGN}": "-"})
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SUPERSCRIPT_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+")
_TOKEN = re.compile(
    r"\s*(?:(?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_\u03b1-\u03c9\u0391-\u03a9][A-Za-z0-9_\u03b1-\u03c9\u0391-\u03a9]*)"
    r"|\[(?P<bracketed>[^\[\]]+)\]|(?P<operator>\*\*|[-+*/^()])|(?P<other>\S))"
)
_LETTERS = re.compile(r"[A-Za-z\u03b1-\u03c9\u0391-\u03a9]+")
_OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


@dataclass(frozen=True)
class Expression:
    """An expression read from ``text``, and the variables it uses."""

    text: str
    names: frozenset[str]
    _node: Node = field(repr=False, compare=False)

    def value(self, values: Values) -> float:
        """Its value with ``values`` for its variables; a ValueError says why there is none."""
        try:
            result = self._node(values)
        except ZeroDivisionError:
            raise ValueError("It divides by zero.") from None
        except OverflowError:
            result = math.inf
        except ValueError:
            raise ValueError("A function gets a value it can't take, such as sqrt(-1).") from None
        if not math.isfinite(result):
            raise ValueError("It's too large to work out.")
        return result


def read(text: str, *, whole_names: bool = False) -> Expression:
    """``text`` as an expression. Each letter that isn't part of a function or constant is a
    variable, so xy is x times y. With ``whole_names``, each other name is a variable as written,
    such as Q1, and so is anything in square brackets, such as [1a]."""
    if len(text) > MAX_LENGTH:
        raise ValueError(f"An expression can't be longer than {MAX_LENGTH} characters.")
    try:
        parser = _Parser(_tokens(text, whole_names))
        node = parser.sum()
        if parser.peek() is not None:
            raise _Unreadable
    except _Unreadable:
        raise ValueError(f"Couldn't read {quoted(text)} as an expression.") from None
    return Expression(text, frozenset(parser.names), node)


class _Unreadable(Exception):
    pass


def _tokens(text: str, whole_names: bool) -> list[str]:
    """The tokens of ``text``; a variable named in brackets keeps them, so it can't be mistaken
    for a number, function or constant."""
    written = _SUPERSCRIPT_RUN.sub(
        lambda run: f"^({run[0].translate(_SUPERSCRIPTS)})", text.translate(_SYMBOLS)
    )
    tokens: list[str] = []
    for match in _TOKEN.finditer(written.rstrip()):
        if match["other"] or (match["bracketed"] and not whole_names):
            raise _Unreadable
        if match["bracketed"]:
            tokens.append(f"[{match['bracketed'].strip()}]")
        elif (name := match["name"]) is None:
            tokens.append("^" if match[0].strip() == "**" else match[0].strip())
        elif whole_names or name in RESERVED:
            tokens.append(name)
        elif parts := _split(name):
            tokens.extend(parts)
        else:
            raise ValueError(f"Unknown name {quoted(name)}.")
    return tokens


def _split(name: str) -> list[str]:
    """``name`` as functions, constants and single letters, longest first: pix is pi times x.
    Empty for a name with digits or underscores."""
    if not _LETTERS.fullmatch(name):
        return []
    parts: list[str] = []
    while name:
        part = next(
            (name[:end] for end in range(len(name), 1, -1) if name[:end] in RESERVED), name[0]
        )
        parts.append(part)
        name = name[len(part) :]
    return parts


class _Parser:
    """Recursive descent, building a function of the variables' values as it reads."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.at = 0
        self.names: set[str] = set()

    def peek(self) -> str | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise _Unreadable
        self.at += 1
        return token

    def expect(self, token: str) -> None:
        if self.take() != token:
            raise _Unreadable

    def sum(self) -> Node:
        node = self._product()
        while (token := self.peek()) in ("+", "-"):
            self.take()
            node = _binary(_OPERATORS[token], node, self._product())
        return node

    def _product(self) -> Node:
        node = self._signed()
        while (token := self.peek()) is not None:
            if token in ("*", "/"):
                self.take()
                node = _binary(_OPERATORS[token], node, self._signed())
            elif token == "(" or _is_name(token):
                node = _binary(operator.mul, node, self._power())
            else:
                break
        return node

    def _signed(self) -> Node:
        if (token := self.peek()) in ("+", "-"):
            self.take()
            inner = self._signed()
            return inner if token == "+" else lambda values: -inner(values)
        return self._power()

    def _power(self) -> Node:
        base = self._atom()
        if self.peek() == "^":
            self.take()
            return _binary(math.pow, base, self._signed())
        return base

    def _atom(self) -> Node:
        token = self.take()
        if token == "(":
            node = self.sum()
            self.expect(")")
            return node
        if token[0].isdigit() or token[0] == ".":
            number = float(token)
            return lambda values: number
        if not _is_name(token):
            raise _Unreadable
        if token.startswith("["):
            name = token[1:-1]
            self.names.add(name)
            return lambda values: values[name]
        if token in FUNCTIONS:
            function = FUNCTIONS[token]
            self.expect("(")
            argument = self.sum()
            self.expect(")")
            return lambda values: function(argument(values))
        if token in CONSTANTS:
            constant = CONSTANTS[token]
            return lambda values: constant
        self.names.add(token)
        return lambda values: values[token]


def _is_name(token: str) -> bool:
    return token[0].isalpha() or token[0] in "_["


def _binary(combine: Callable[[float, float], float], left: Node, right: Node) -> Node:
    return lambda values: combine(left(values), right(values))
