"""
The paragraphs of an RTF document as text, with its pictures as their bytes.

Table cells are separated by `` | `` and each table row is a paragraph. Headers, footers, style
tables and other destinations that aren't body text are skipped.
"""

import re
from dataclasses import dataclass, replace

type ParagraphPart = str | bytes
"""Text, or a picture's data."""

MAX_GROUP_DEPTH = 256
_SKIPPED_DESTINATIONS = frozenset(
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
_CHARACTER_WORDS = {
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
_ESCAPED_SYMBOLS = {"\\": "\\", "{": "{", "}": "}", "~": "\xa0", "_": "-"}
_TOKEN = re.compile(
    r"\\([a-z]+)(-?\d+)? ?|\\'([0-9a-f]{2})|\\(.)|([{}])|[\r\n]+|([^\\{}\r\n]+)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class _GroupState:
    is_skipped: bool = False
    is_picture: bool = False
    unicode_fallback: int = 1


def paragraphs(text: str) -> list[list[ParagraphPart]]:
    """
    The paragraphs of ``text``, an RTF document read as Latin-1 so binary data keeps its bytes.
    Pictures given as binary data (``\\bin``) are kept; others are left out.
    """
    reader = _RtfReader()
    position = 0
    while match := _TOKEN.match(text, position):
        position = match.end()
        word, argument, hex_code, symbol, brace, plain_text = match.groups()
        if word == "bin":
            size = max(int(argument or 0), 0)
            if size:
                reader.picture(text[position : position + size].encode("latin-1"))
            position += size
        elif word:
            reader.control(word, argument)
        elif brace:
            reader.brace(brace == "{")
        elif hex_code:
            reader.write(bytes([int(hex_code, 16)]).decode("cp1252", "replace"), fallback=True)
        elif symbol:
            reader.symbol(symbol)
        elif plain_text:
            reader.write(plain_text, fallback=True)
    return reader.finish()


class _RtfReader:
    def __init__(self) -> None:
        self.paragraphs: list[list[ParagraphPart]] = []
        self.paragraph: list[ParagraphPart] = []
        self.pending_text: list[str] = []
        self.pending_cell_separators = 0
        """Table cells ended whose separators wait for the next text."""
        self.group = _GroupState()
        self.outer_groups: list[_GroupState] = []
        self.after_star = False
        self.pending_fallback = 0

    def brace(self, is_opening: bool) -> None:
        self.after_star = False
        self.pending_fallback = 0
        if is_opening:
            if len(self.outer_groups) >= MAX_GROUP_DEPTH:
                raise ValueError("its groups are nested too deeply.")
            self.outer_groups.append(self.group)
        elif self.outer_groups:
            self.group = self.outer_groups.pop()

    def control(self, word: str, argument: str | None) -> None:
        after_star, self.after_star = self.after_star, False
        if word in _SKIPPED_DESTINATIONS or (after_star and word != "shppict"):
            self.group = replace(self.group, is_skipped=True)
        elif word == "pict":
            self.group = replace(self.group, is_picture=True)
        elif word == "uc":
            self.group = replace(self.group, unicode_fallback=int(argument or 1))
        elif word == "u":
            number = int(argument or 0)
            self.write(chr(number + 65536 if number < 0 else number))
            self.pending_fallback = self.group.unicode_fallback
        elif word in ("par", "row"):
            self.end_paragraph()
        elif word == "cell":
            self.pending_cell_separators += 1
        elif word in _CHARACTER_WORDS:
            self.write(_CHARACTER_WORDS[word])

    def symbol(self, symbol: str) -> None:
        if symbol == "*":
            self.after_star = True
        elif symbol in _ESCAPED_SYMBOLS:
            self.write(_ESCAPED_SYMBOLS[symbol])

    def write(self, text: str, fallback: bool = False) -> None:
        self.after_star = False
        if fallback and self.pending_fallback:
            skipped = min(self.pending_fallback, len(text))
            text, self.pending_fallback = text[skipped:], self.pending_fallback - skipped
        if self.group.is_skipped or self.group.is_picture or not text:
            return
        self._separate_cell()
        self.pending_text.append(text)

    def picture(self, data: bytes) -> None:
        if not self.group.is_skipped:
            self._separate_cell()
            self._flush()
            self.paragraph.append(data)

    def end_paragraph(self) -> None:
        if self.group.is_skipped:
            return
        self._flush()
        self.pending_cell_separators = 0
        self.paragraphs.append(self.paragraph)
        self.paragraph = []

    def finish(self) -> list[list[ParagraphPart]]:
        self._flush()
        if self.paragraph:
            self.paragraphs.append(self.paragraph)
        return self.paragraphs

    def _separate_cell(self) -> None:
        if self.pending_cell_separators:
            self.pending_text.append(" | " * self.pending_cell_separators)
            self.pending_cell_separators = 0

    def _flush(self) -> None:
        """Ends the text so far. A character past U+FFFF comes as two surrogates, joined here."""
        if self.pending_text:
            joined = "".join(self.pending_text).encode("utf-16", "surrogatepass")
            self.paragraph.append(joined.decode("utf-16", "replace"))
            self.pending_text = []
