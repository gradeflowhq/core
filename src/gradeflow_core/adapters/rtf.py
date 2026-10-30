"""
The paragraphs of an RTF document as text, with its pictures as their bytes.

Table cells are separated by `` | `` and each table row is a paragraph. Headers, footers, style
tables and other destinations that aren't body text are skipped.
"""

import re
from dataclasses import dataclass, replace

type Part = str | bytes
"""Text, or a picture's data."""

MAX_DEPTH = 256
_SKIPPED = frozenset(
    {
        "colortbl",
        "fonttbl",
        "footer",
        "footerf",
        "footerl",
        "footerr",
        "header",
        "headerf",
        "headerl",
        "headerr",
        "info",
        "listoverridetable",
        "listtable",
        "nonshppict",
        "stylesheet",
    }
)
_SPECIAL = {
    "line": " ",
    "tab": "\t",
    "lquote": "\u2018",
    "rquote": "\u2019",
    "ldblquote": "\u201c",
    "rdblquote": "\u201d",
    "endash": "\u2013",
    "emdash": "\u2014",
    "bullet": "\u2022",
}
_SYMBOLS = {"\\": "\\", "{": "{", "}": "}", "~": "\xa0", "_": "-"}
_TOKEN = re.compile(
    r"\\([a-z]+)(-?\d+)? ?|\\'([0-9a-f]{2})|\\(.)|([{}])|[\r\n]+|([^\\{}\r\n]+)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class _Group:
    skipping: bool = False
    picture: bool = False
    unicode_fallback: int = 1


def paragraphs(text: str) -> list[list[Part]]:
    """
    The paragraphs of ``text``, an RTF document read as Latin-1 so binary data keeps its bytes.
    Pictures given as binary data (``\\bin``) are kept; others are left out.
    """
    reader = _Reader()
    position = 0
    while match := _TOKEN.match(text, position):
        position = match.end()
        word, argument, code, symbol, brace, plain = match.groups()
        if word == "bin":
            size = max(int(argument or 0), 0)
            if size:
                reader.picture(text[position : position + size].encode("latin-1"))
            position += size
        elif word:
            reader.control(word, argument)
        elif brace:
            reader.group(brace == "{")
        elif code:
            reader.write(bytes([int(code, 16)]).decode("cp1252", "replace"), fallback=True)
        elif symbol:
            reader.symbol(symbol)
        elif plain:
            reader.write(plain, fallback=True)
    return reader.done()


class _Reader:
    def __init__(self) -> None:
        self.paragraphs: list[list[Part]] = []
        self.current: list[Part] = []
        self.text: list[str] = []
        self.cells = 0
        """Table cells ended whose separators wait for the next text."""
        self.state = _Group()
        self.stack: list[_Group] = []
        self.starred = False
        self.pending_fallback = 0

    def group(self, opening: bool) -> None:
        self.starred = False
        self.pending_fallback = 0
        if opening:
            if len(self.stack) >= MAX_DEPTH:
                raise ValueError("its groups are nested too deeply.")
            self.stack.append(self.state)
        elif self.stack:
            self.state = self.stack.pop()

    def control(self, word: str, argument: str | None) -> None:
        starred, self.starred = self.starred, False
        if word in _SKIPPED or (starred and word != "shppict"):
            self.state = replace(self.state, skipping=True)
        elif word == "pict":
            self.state = replace(self.state, picture=True)
        elif word == "uc":
            self.state = replace(self.state, unicode_fallback=int(argument or 1))
        elif word == "u":
            number = int(argument or 0)
            self.write(chr(number + 65536 if number < 0 else number))
            self.pending_fallback = self.state.unicode_fallback
        elif word in ("par", "row"):
            self.end_paragraph()
        elif word == "cell":
            self.cells += 1
        elif word in _SPECIAL:
            self.write(_SPECIAL[word])

    def symbol(self, symbol: str) -> None:
        if symbol == "*":
            self.starred = True
        elif symbol in _SYMBOLS:
            self.write(_SYMBOLS[symbol])

    def write(self, text: str, fallback: bool = False) -> None:
        self.starred = False
        if fallback and self.pending_fallback:
            skipped = min(self.pending_fallback, len(text))
            text, self.pending_fallback = text[skipped:], self.pending_fallback - skipped
        if self.state.skipping or self.state.picture or not text:
            return
        self._separate_cell()
        self.text.append(text)

    def picture(self, data: bytes) -> None:
        if not self.state.skipping:
            self._separate_cell()
            self._flush()
            self.current.append(data)

    def end_paragraph(self) -> None:
        if self.state.skipping:
            return
        self._flush()
        self.cells = 0
        self.paragraphs.append(self.current)
        self.current = []

    def done(self) -> list[list[Part]]:
        self._flush()
        if self.current:
            self.paragraphs.append(self.current)
        return self.paragraphs

    def _separate_cell(self) -> None:
        if self.cells:
            self.text.append(" | " * self.cells)
            self.cells = 0

    def _flush(self) -> None:
        """Ends the text so far. A character past U+FFFF comes as two surrogates, joined here."""
        if self.text:
            joined = "".join(self.text).encode("utf-16", "surrogatepass")
            self.current.append(joined.decode("utf-16", "replace"))
            self.text = []
