"""LaTeX formulas as people type them in plain text: ``h^*(s_0)``, ``1/(2n)``, ``sqrt(x)``."""

import re

SYMBOLS = {
    "times": "×",
    "div": "÷",
    "cdot": "·",
    "pm": "±",
    "mp": "∓",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "ne": "≠",
    "neq": "≠",
    "approx": "≈",
    "equiv": "≡",
    "sim": "\u223c",
    "propto": "∝",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "leftrightarrow": "↔",
    "Rightarrow": "⇒",
    "Leftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "subseteq": "⊆",
    "supset": "⊃",
    "supseteq": "⊇",
    "cup": "\u222a",
    "cap": "∩",
    "emptyset": "∅",
    "infty": "∞",
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
    "partial": "∂",
    "nabla": "∇",
    "neg": "¬",
    "land": "∧",
    "wedge": "∧",
    "lor": "\u2228",
    "vee": "\u2228",
    "forall": "∀",
    "exists": "∃",
    "circ": "∘",
    "oplus": "⊕",
    "otimes": "⊗",
    "perp": "⊥",
    "ldots": "…",
    "cdots": "⋯",
}
"""Commands that write a symbol, and the symbol; the first command for a symbol is its usual one."""

_TOKEN = re.compile(r"(\\[A-Za-z]+) ?|(\\.|.)", re.DOTALL)
"""A command, without the space that ends its name, or a character."""
_TEXT_COMMANDS = frozenset(
    {
        "text",
        "textrm",
        "textbf",
        "textit",
        "mbox",
        "operatorname",
        "mathrm",
        "mathit",
        "mathbf",
        "mathsf",
        "mathtt",
        "mathcal",
        "mathbb",
        "boldsymbol",
    }
)
MATRIX_BRACKETS = {
    "pmatrix": ("(", ")"),
    "bmatrix": ("[", "]"),
    "Bmatrix": ("{", "}"),
    "vmatrix": ("|", "|"),
    "Vmatrix": ("\u2016", "\u2016"),
}
"""The brackets around each kind of matrix."""
_CHARACTER_COMMANDS = {
    "backslash": "\\",
    "textbackslash": "\\",
    "textasciicircum": "^",
    "textasciitilde": "~",
}
_SPACE_COMMANDS = frozenset({" ", ",", ":", ";", "quad", "qquad"})
_IGNORED_COMMANDS = frozenset(
    {"left", "right", "big", "Big", "bigg", "Bigg", "displaystyle", "limits", "!"}
)
_ASCII_SYMBOLS = str.maketrans({"≤": "<=", "≥": ">=", "≠": "!=", "→": "->", "\u2212": "-"})


def typed_latex(latex: str) -> str:
    """``latex``, a formula, as people type it: ``\\frac{1}{2}`` as ``1/2``, ``\\le`` as ``<=``;
    as it is if it nests too deeply to read."""
    try:
        return _LatexReader(latex).read_until("").strip().translate(_ASCII_SYMBOLS)
    except RecursionError:
        return latex


class _LatexReader:
    def __init__(self, latex: str) -> None:
        self.tokens: list[str] = [
            command or character for command, character in _TOKEN.findall(latex)
        ]
        self.position = 0
        self.is_in_text = False
        """Whether spaces count, as they do in ``\\text{…}``."""

    def read_until(self, closing_token: str) -> str:
        parts: list[str] = []
        while self.position < len(self.tokens):
            token = self._next()
            if token == closing_token:
                break
            parts.append(self._read_item(token))
        return "".join(parts)

    def _next(self) -> str:
        self.position += 1
        return self.tokens[self.position - 1]

    def _peek(self) -> str:
        return self.tokens[self.position] if self.position < len(self.tokens) else ""

    def _read_argument(self) -> str:
        while self._peek().isspace():
            self.position += 1
        return self._read_item(self._next()) if self._peek() else ""

    def _read_item(self, token: str) -> str:
        if token == "{":
            return self.read_until("}")
        if token in ("^", "_"):
            return token + _bracketed(self._read_argument())
        if token == "&":
            return ", "
        if token.isspace():
            return " " if self.is_in_text else ""
        if not token.startswith("\\") or len(token) == 1:
            return token
        command_text = self._read_command(token[1:])
        return (
            f"{command_text} "
            if command_text[-1:].isalpha() and self._letter_follows()
            else command_text
        )

    def _letter_follows(self) -> bool:
        next_token = next(
            (token for token in self.tokens[self.position :] if not token.isspace()), ""
        )
        return next_token[:1].isalpha()

    def _read_command(self, name: str) -> str:
        if name == "frac":
            numerator = _bracketed(self._read_argument())
            return f"{numerator}/{_bracketed(self._read_argument())}"
        if name == "sqrt":
            index = ""
            if self._peek() == "[":
                self.position += 1
                index = self.read_until("]")
            body = self._read_argument()
            return f"root({body}, {index})" if index else f"sqrt({body})"
        if name in ("begin", "end"):
            opening_bracket, closing_bracket = MATRIX_BRACKETS.get(self._read_argument(), ("", ""))
            return closing_bracket if name == "end" else opening_bracket
        if name in _TEXT_COMMANDS:
            was_in_text, self.is_in_text = self.is_in_text, True
            text = self._read_argument()
            self.is_in_text = was_in_text
            return text
        if name == "\\":
            return "; "
        if name in _SPACE_COMMANDS:
            return " "
        return (
            ""
            if name in _IGNORED_COMMANDS
            else SYMBOLS.get(name) or _CHARACTER_COMMANDS.get(name, name)
        )


def _bracketed(text: str) -> str:
    """``text`` in brackets unless it reads as one name or number."""
    is_single_term = len(text) <= 1 or text.isalpha() or text.replace(".", "", 1).isdigit()
    return text if is_single_term or (text.startswith("(") and text.endswith(")")) else f"({text})"
