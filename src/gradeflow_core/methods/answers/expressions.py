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
RESERVED_NAMES = FUNCTIONS.keys() | CONSTANTS.keys()
"""Names a variable can't have."""

type VariableValues = Mapping[str, float]
type Evaluator = Callable[[VariableValues], float]

_ASCII_OPERATORS = str.maketrans({"×": "*", "·": "*", "÷": "/", "\N{MINUS SIGN}": "-"})
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_SUPERSCRIPT_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+")
_TOKEN = re.compile(
    r"\s*(?:(?P<number>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_\u03b1-\u03c9\u0391-\u03a9][A-Za-z0-9_\u03b1-\u03c9\u0391-\u03a9]*)"
    r"|\[(?P<bracketed>[^\[\]]+)\]|(?P<operator>\*\*|[-+*/^()])|(?P<invalid_character>\S))"
)
_LETTERS = re.compile(r"[A-Za-z\u03b1-\u03c9\u0391-\u03a9]+")
_OPERATOR_FUNCTIONS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


@dataclass(frozen=True)
class Expression:
    """An expression, and the variables it uses."""

    variables: frozenset[str]
    _evaluator: Evaluator = field(repr=False, compare=False)

    def value(self, variable_values: VariableValues) -> float:
        """Its value with ``variable_values`` for its variables; a ValueError says why there is
        none."""
        try:
            value = self._evaluator(variable_values)
        except ZeroDivisionError:
            raise ValueError("It divides by zero.") from None
        except OverflowError:
            value = math.inf
        except ValueError:
            raise ValueError("A function gets a value it can't take, such as sqrt(-1).") from None
        if not math.isfinite(value):
            raise ValueError("It's too large to work out.")
        return value


def parse(expression_text: str, *, names_as_written: bool = False) -> Expression:
    """``expression_text`` as an expression. Each letter that isn't part of a function or constant
    is a variable, so xy is x times y. With ``names_as_written``, each other name is a variable as
    written, such as Q1, and so is anything in square brackets, such as [1a]."""
    if len(expression_text) > MAX_LENGTH:
        raise ValueError(f"An expression can't be longer than {MAX_LENGTH} characters.")
    try:
        parser = _Parser(_tokens(expression_text, names_as_written))
        evaluator = parser.sum()
        if parser.peek() is not None:
            raise _ParseError
    except _ParseError:
        raise ValueError(f"Couldn't read {quoted(expression_text)} as an expression.") from None
    return Expression(frozenset(parser.variables), evaluator)


class _ParseError(Exception):
    pass


def _tokens(expression_text: str, names_as_written: bool) -> list[str]:
    """The tokens of ``expression_text``; a variable named in brackets keeps them, so it can't be
    mistaken for a number, function or constant."""
    normalized_text = _SUPERSCRIPT_RUN.sub(
        lambda run: f"^({run[0].translate(_SUPERSCRIPTS)})",
        expression_text.translate(_ASCII_OPERATORS),
    )
    tokens: list[str] = []
    for match in _TOKEN.finditer(normalized_text.rstrip()):
        if match["invalid_character"] or (match["bracketed"] and not names_as_written):
            raise _ParseError
        if match["bracketed"]:
            tokens.append(f"[{match['bracketed'].strip()}]")
        elif (name := match["name"]) is None:
            tokens.append("^" if match[0].strip() == "**" else match[0].strip())
        elif names_as_written or name in RESERVED_NAMES:
            tokens.append(name)
        elif names := _split_name(name):
            tokens.extend(names)
        else:
            raise ValueError(f"Unknown name {quoted(name)}.")
    return tokens


def _split_name(name: str) -> list[str]:
    """``name`` as functions, constants and single letters, longest first: pix is pi times x.
    Empty for a name with digits or underscores."""
    if not _LETTERS.fullmatch(name):
        return []
    names: list[str] = []
    while name:
        leading_name = next(
            (name[:end] for end in range(len(name), 1, -1) if name[:end] in RESERVED_NAMES), name[0]
        )
        names.append(leading_name)
        name = name[len(leading_name) :]
    return names


class _Parser:
    """Recursive descent, building a function of the variables' values as it reads."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.next_token_index = 0
        self.variables: set[str] = set()

    def peek(self) -> str | None:
        return (
            self.tokens[self.next_token_index] if self.next_token_index < len(self.tokens) else None
        )

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise _ParseError
        self.next_token_index += 1
        return token

    def expect(self, token: str) -> None:
        if self.take() != token:
            raise _ParseError

    def sum(self) -> Evaluator:
        evaluator = self._product()
        while (token := self.peek()) in ("+", "-"):
            self.take()
            evaluator = _binary_operation(_OPERATOR_FUNCTIONS[token], evaluator, self._product())
        return evaluator

    def _product(self) -> Evaluator:
        evaluator = self._unary()
        while (token := self.peek()) is not None:
            if token in ("*", "/"):
                self.take()
                evaluator = _binary_operation(_OPERATOR_FUNCTIONS[token], evaluator, self._unary())
            elif token == "(" or _is_name(token):
                evaluator = _binary_operation(operator.mul, evaluator, self._power())
            else:
                break
        return evaluator

    def _unary(self) -> Evaluator:
        if (token := self.peek()) in ("+", "-"):
            self.take()
            operand = self._unary()
            return operand if token == "+" else lambda variable_values: -operand(variable_values)
        return self._power()

    def _power(self) -> Evaluator:
        base = self._atom()
        if self.peek() == "^":
            self.take()
            return _binary_operation(math.pow, base, self._unary())
        return base

    def _atom(self) -> Evaluator:
        token = self.take()
        if token == "(":
            evaluator = self.sum()
            self.expect(")")
            return evaluator
        if token[0].isdigit() or token[0] == ".":
            number = float(token)
            return lambda variable_values: number
        if not _is_name(token):
            raise _ParseError
        if token.startswith("["):
            name = token[1:-1]
            self.variables.add(name)
            return lambda variable_values: variable_values[name]
        if token in FUNCTIONS:
            function = FUNCTIONS[token]
            self.expect("(")
            argument = self.sum()
            self.expect(")")
            return lambda variable_values: function(argument(variable_values))
        if token in CONSTANTS:
            constant = CONSTANTS[token]
            return lambda variable_values: constant
        self.variables.add(token)
        return lambda variable_values: variable_values[token]


def _is_name(token: str) -> bool:
    return token[0].isalpha() or token[0] in "_["


def _binary_operation(
    operation: Callable[[float, float], float], left: Evaluator, right: Evaluator
) -> Evaluator:
    return lambda variable_values: operation(left(variable_values), right(variable_values))
