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
_TEXT = frozenset(
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
MATRICES = {
    "pmatrix": ("(", ")"),
    "bmatrix": ("[", "]"),
    "Bmatrix": ("{", "}"),
    "vmatrix": ("|", "|"),
    "Vmatrix": ("\u2016", "\u2016"),
}
"""The brackets around each kind of matrix."""
_CHARACTERS = {
    "backslash": "\\",
    "textbackslash": "\\",
    "textasciicircum": "^",
    "textasciitilde": "~",
}
_SPACES = frozenset({" ", ",", ":", ";", "quad", "qquad"})
_IGNORED = frozenset({"left", "right", "big", "Big", "bigg", "Bigg", "displaystyle", "limits", "!"})
_TYPED = str.maketrans({"≤": "<=", "≥": ">=", "≠": "!=", "→": "->", "\u2212": "-"})


def latex_text(latex: str) -> str:
    """``latex``, a formula, as people type it: ``\\frac{1}{2}`` as ``1/2``, ``\\le`` as ``<=``;
    as it is if it nests too deeply to read."""
    try:
        return _Reader(latex).sequence("").strip().translate(_TYPED)
    except RecursionError:
        return latex


class _Reader:
    def __init__(self, latex: str) -> None:
        self.tokens: list[str] = [
            command or character for command, character in _TOKEN.findall(latex)
        ]
        self.position = 0
        self.text = False
        """Whether spaces count, as they do in ``\\text{…}``."""

    def sequence(self, closing: str) -> str:
        parts: list[str] = []
        while self.position < len(self.tokens):
            token = self._next()
            if token == closing:
                break
            parts.append(self._item(token))
        return "".join(parts)

    def _next(self) -> str:
        self.position += 1
        return self.tokens[self.position - 1]

    def _peek(self) -> str:
        return self.tokens[self.position] if self.position < len(self.tokens) else ""

    def _argument(self) -> str:
        while self._peek().isspace():
            self.position += 1
        return self._item(self._next()) if self._peek() else ""

    def _item(self, token: str) -> str:
        if token == "{":
            return self.sequence("}")
        if token in ("^", "_"):
            return token + _grouped(self._argument())
        if token == "&":
            return ", "
        if token.isspace():
            return " " if self.text else ""
        if not token.startswith("\\") or len(token) == 1:
            return token
        text = self._command(token[1:])
        return f"{text} " if text[-1:].isalpha() and self._letter_follows() else text

    def _letter_follows(self) -> bool:
        following = next(
            (token for token in self.tokens[self.position :] if not token.isspace()), ""
        )
        return following[:1].isalpha()

    def _command(self, name: str) -> str:
        if name == "frac":
            numerator = _grouped(self._argument())
            return f"{numerator}/{_grouped(self._argument())}"
        if name == "sqrt":
            index = ""
            if self._peek() == "[":
                self.position += 1
                index = self.sequence("]")
            body = self._argument()
            return f"root({body}, {index})" if index else f"sqrt({body})"
        if name in ("begin", "end"):
            opening, closing = MATRICES.get(self._argument(), ("", ""))
            return closing if name == "end" else opening
        if name in _TEXT:
            outer, self.text = self.text, True
            text = self._argument()
            self.text = outer
            return text
        if name == "\\":
            return "; "
        if name in _SPACES:
            return " "
        return "" if name in _IGNORED else SYMBOLS.get(name) or _CHARACTERS.get(name, name)


def _grouped(text: str) -> str:
    """``text`` in brackets unless it reads as one name or number."""
    single = len(text) <= 1 or text.isalpha() or text.replace(".", "", 1).isdigit()
    return text if single or (text.startswith("(") and text.endswith(")")) else f"({text})"
