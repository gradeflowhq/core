import pytest

from gradeflow_core.steps.input_sources import (
    AnswerSource,
    AnswerValueSource,
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
        ("{{values[1]}}", AnswerValueSource(2)),
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
        "{{ answer or values[0] }}",
        "{{ values[-1] }}",
        "{{ values['first'] }}",
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
        AnswerValueSource(3),
        OtherAnswerSource("Q2"),
        OtherAnswerSource("1 a"),
        StepOutputSource("final"),
    ],
)
def test_a_source_is_written_as_the_input_that_names_it(source: InputSource) -> None:
    assert input_source(input_text(source)) == source
