import pytest

from gradeflow_core.steps.input_sources import (
    AnswerPartSource,
    AnswerSource,
    InputSource,
    OtherAnswerSource,
    StepOutputSource,
    input_source,
    input_text,
)


@pytest.mark.parametrize(
    ("text", "source"),
    [
        (None, AnswerSource()),
        ("{{ answer }}", AnswerSource()),
        ("{{answer.year}}", AnswerPartSource("year")),
        ("{{ answer['year'] }}", AnswerPartSource("year")),
        ("{{ answers.Q2 }}", OtherAnswerSource("Q2")),
        ("{{ answers['1 a'] }}", OtherAnswerSource("1 a")),
        ("{{ steps.final.output }}", StepOutputSource("final")),
        ("{{ steps['final'].output }}", StepOutputSource("final")),
    ],
)
def test_an_input_names_what_it_reads(text: str | None, source: InputSource) -> None:
    assert input_source(text) == source


@pytest.mark.parametrize(
    "text",
    [
        "answer",
        "{{ answer }} too",
        "{{ answer | upper }}",
        "{{ answer or answer.year }}",
        "{{ answer[0] }}",
        "{{ answer.__class__ }}",
        "{{ values[0] }}",
        "{{ answers[1] }}",
        "{{ steps.final }}",
        "{{ steps.final.output.correct }}",
        "{{ nothing }}",
        "{{ 'a' * 3 }}",
    ],
)
def test_an_input_that_names_no_one_thing_to_read_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="Write an input as one of"):
        input_source(text)


@pytest.mark.parametrize(
    "source",
    [
        AnswerSource(),
        AnswerPartSource("year"),
        OtherAnswerSource("Q2"),
        OtherAnswerSource("1 a"),
        StepOutputSource("final"),
    ],
)
def test_a_source_is_written_as_the_input_that_names_it(source: InputSource) -> None:
    assert input_source(input_text(source)) == source
