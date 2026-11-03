"""MathML as LaTeX: ``s_{0}``, ``h^{*}(s)``, ``\\frac{1}{2}``, ``\\sqrt{x}``."""

import re
from xml.etree import ElementTree

from ..common.latex import MATRICES, SYMBOLS

_SCRIPTS = {
    "msub": "_",
    "msup": "^",
    "msubsup": "_^",
    "munder": "_",
    "mover": "^",
    "munderover": "_^",
}
"""What comes before each part after the first."""
_COMMANDS = {symbol: f"\\{name} " for name, symbol in reversed(SYMBOLS.items())} | {
    "\u2212": "-",
    "\u2032": "'",
}
_MATRICES = {brackets: matrix for matrix, brackets in MATRICES.items()}
"""The matrix a table between a pair of brackets is."""
_ESCAPED = {character: f"\\{character}" for character in "#$%&_{}"}
_MATH_ESCAPES = str.maketrans(
    _ESCAPED | {"\\": "\\backslash ", "^": "\\text{\\textasciicircum}", "~": "\\sim "}
)
_TEXT_ESCAPES = str.maketrans(
    _ESCAPED | {"\\": "\\textbackslash ", "^": "\\textasciicircum ", "~": "\\textasciitilde "}
)
_COMMAND_SPACE = re.compile(r"(\\[A-Za-z]+) (?=[_^{}']|$)")
"""The space ending a command, before a script, brace or prime."""
_LEADING_TRAILING_SPACE = re.compile(r"^(?:\s|\\ )+|(?:\s|\\ )+$")
_INVISIBLE_OPERATORS = dict.fromkeys(range(0x2061, 0x2065))
"""Function application, invisible times, separator and plus."""


def mathml_to_latex(mathml: str) -> str:
    latex = _COMMAND_SPACE.sub(r"\1", _latex(ElementTree.fromstring(mathml)))
    return _LEADING_TRAILING_SPACE.sub("", latex)


def _latex(element: ElementTree.Element) -> str:
    tag = _local_name(element)
    if tag in ("mi", "mn", "mo"):
        text = (element.text or "").strip().translate(_INVISIBLE_OPERATORS)
        if len(text) > 1 and text.isalpha():
            return f"\\operatorname{{{text}}}"
        return "".join(_COMMANDS.get(symbol, symbol.translate(_MATH_ESCAPES)) for symbol in text)
    if tag in ("mtext", "ms"):
        text = (element.text or "").translate(_TEXT_ESCAPES)
        return f"\\text{{{text}}}" if text else ""
    if tag == "mspace":
        return "\\ "
    if tag == "mtable":
        return _matrix(element, "matrix")
    if tag == "msqrt":
        return f"\\sqrt{{{_row(list(element))}}}"
    if tag in _SCRIPTS or tag in ("mfrac", "mroot"):
        return _scripts_or_fraction(element, tag)
    return _row(list(element))


def _scripts_or_fraction(element: ElementTree.Element, tag: str) -> str:
    """A script, fraction or root; just its parts if it has too few or many."""
    children = [_latex(child) for child in element]
    script_prefixes = _SCRIPTS.get(tag, "")
    if script_prefixes and len(children) == len(script_prefixes) + 1:
        base, *scripts = children
        if _local_name(element[0]) in _SCRIPTS:
            base = f"{{{base}}}"
        return base + "".join(
            script
            if prefix == "^" and script and not script.strip("'")
            else f"{prefix}{{{script}}}"
            for prefix, script in zip(script_prefixes, scripts, strict=True)
        )
    if tag == "mfrac" and len(children) == 2:
        return f"\\frac{{{children[0]}}}{{{children[1]}}}"
    if tag == "mroot" and len(children) == 2:
        return f"\\sqrt[{children[1]}]{{{children[0]}}}"
    return "".join(children)


def _row(elements: list[ElementTree.Element]) -> str:
    """The elements as LaTeX; a table between a pair of brackets is that bracket's matrix."""
    parts: list[str] = []
    index = 0
    while index < len(elements):
        if matrix := _bracketed_matrix(elements[index : index + 3]):
            parts.append(_matrix(elements[index + 1], matrix))
            index += 3
        else:
            parts.append(_latex(elements[index]))
            index += 1
    return "".join(parts)


def _bracketed_matrix(elements: list[ElementTree.Element]) -> str | None:
    """The matrix ``elements`` make if they are a table between a pair of brackets."""
    if [_local_name(element) for element in elements] != ["mo", "mtable", "mo"]:
        return None
    opening, _, closing = elements
    return _MATRICES.get(((opening.text or "").strip(), (closing.text or "").strip()))


def _matrix(table: ElementTree.Element, environment: str) -> str:
    rows = " \\\\ ".join(" & ".join(_latex(cell) for cell in row) for row in table)
    return f"\\begin{{{environment}}}{rows}\\end{{{environment}}}"


def _local_name(element: ElementTree.Element) -> str:
    return element.tag.rpartition("}")[2]
