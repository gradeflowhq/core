import pytest

from gradeflow_core.questions.reading import TextReading
from gradeflow_core.questions.text import TextQuestion


def test_text_returns_same_string() -> None:
    assert TextQuestion().read("hello world") == "hello world"


def test_text_trims_whitespace_by_default() -> None:
    assert TextQuestion().read("  leading and trailing  ") == "leading and trailing"


def test_text_keeps_whitespace_when_trimming_is_off() -> None:
    q = TextQuestion(reading=TextReading(trim_spaces=False))
    assert q.read("  leading and trailing  ") == "  leading and trailing  "


def test_text_ignores_capitals_unless_told_not_to() -> None:
    assert TextQuestion().read(" Hello World ") == "hello world"
    assert TextQuestion(reading=TextReading(ignore_case=False)).read("Hello") == "Hello"


def test_text_blank_marker_survives_trimming_and_case_normalization() -> None:
    q = TextQuestion()
    assert q.read("  N/A ") is None
    assert q.read("   ") is None
    assert TextQuestion(reading=TextReading(trim_spaces=False)).read(" ") == " "


def test_text_ignores_punctuation_when_configured() -> None:
    q = TextQuestion(reading=TextReading(ignore_punctuation=True))
    assert [q.read(a) for a in ("(LL)", "LL,", "L, L", "a.b-c!")] == ["ll", "ll", "l l", "abc"]
    assert q.read("N/A") is None
    assert TextQuestion().read("(LL)") == "(ll)"


def test_a_text_answer_is_written_whole() -> None:
    question = TextQuestion()
    assert question.answer_schema() == {"type": "string"}
    with pytest.raises(ValueError, match="written whole"):
        question.answer_from({"answer": "x"})
