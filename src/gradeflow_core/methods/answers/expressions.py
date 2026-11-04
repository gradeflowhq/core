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
type Evaluator = Callable[[Values], float]

_OPERATOR_SYMBOLS = str.maketrans({"×": "*", "·": "*", "÷": "/", "\N{MINUS SIGN}": "-"})
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
    """An expression, and the variables it uses."""

    variables: frozenset[str]
    _node: Evaluator = field(repr=False, compare=False)

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


def parse(text: str, *, names_as_written: bool = False) -> Expression:
    """``text`` as an expression. Each letter that isn't part of a function or constant is a
    variable, so xy is x times y. With ``names_as_written``, each other name is a variable as
    written, such as Q1, and so is anything in square brackets, such as [1a]."""
    if len(text) > MAX_LENGTH:
        raise ValueError(f"An expression can't be longer than {MAX_LENGTH} characters.")
    try:
        parser = _Parser(_tokens(text, names_as_written))
        node = parser.sum()
        if parser.peek() is not None:
            raise _ParseError
    except _ParseError:
        raise ValueError(f"Couldn't read {quoted(text)} as an expression.") from None
    return Expression(frozenset(parser.variables), node)


class _ParseError(Exception):
    pass


def _tokens(text: str, names_as_written: bool) -> list[str]:
    """The tokens of ``text``; a variable named in brackets keeps them, so it can't be mistaken
    for a number, function or constant."""
    normalized_text = _SUPERSCRIPT_RUN.sub(
        lambda run: f"^({run[0].translate(_SUPERSCRIPTS)})", text.translate(_OPERATOR_SYMBOLS)
    )
    tokens: list[str] = []
    for match in _TOKEN.finditer(normalized_text.rstrip()):
        if match["other"] or (match["bracketed"] and not names_as_written):
            raise _ParseError
        if match["bracketed"]:
            tokens.append(f"[{match['bracketed'].strip()}]")
        elif (name := match["name"]) is None:
            tokens.append("^" if match[0].strip() == "**" else match[0].strip())
        elif names_as_written or name in RESERVED:
            tokens.append(name)
        elif parts := _split_name(name):
            tokens.extend(parts)
        else:
            raise ValueError(f"Unknown name {quoted(name)}.")
    return tokens


def _split_name(name: str) -> list[str]:
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
        self.position = 0
        self.variables: set[str] = set()

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise _ParseError
        self.position += 1
        return token

    def expect(self, token: str) -> None:
        if self.take() != token:
            raise _ParseError

    def sum(self) -> Evaluator:
        node = self._product()
        while (token := self.peek()) in ("+", "-"):
            self.take()
            node = _binary(_OPERATORS[token], node, self._product())
        return node

    def _product(self) -> Evaluator:
        node = self._unary()
        while (token := self.peek()) is not None:
            if token in ("*", "/"):
                self.take()
                node = _binary(_OPERATORS[token], node, self._unary())
            elif token == "(" or _is_name(token):
                node = _binary(operator.mul, node, self._power())
            else:
                break
        return node

    def _unary(self) -> Evaluator:
        if (token := self.peek()) in ("+", "-"):
            self.take()
            inner = self._unary()
            return inner if token == "+" else lambda values: -inner(values)
        return self._power()

    def _power(self) -> Evaluator:
        base = self._atom()
        if self.peek() == "^":
            self.take()
            return _binary(math.pow, base, self._unary())
        return base

    def _atom(self) -> Evaluator:
        token = self.take()
        if token == "(":
            node = self.sum()
            self.expect(")")
            return node
        if token[0].isdigit() or token[0] == ".":
            number = float(token)
            return lambda values: number
        if not _is_name(token):
            raise _ParseError
        if token.startswith("["):
            name = token[1:-1]
            self.variables.add(name)
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
        self.variables.add(token)
        return lambda values: values[token]


def _is_name(token: str) -> bool:
    return token[0].isalpha() or token[0] in "_["


def _binary(
    combine: Callable[[float, float], float], left: Evaluator, right: Evaluator
) -> Evaluator:
    return lambda values: combine(left(values), right(values))
