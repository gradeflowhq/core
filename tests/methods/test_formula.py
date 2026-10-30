from typing import Any

from gradeflow_core import BUILTINS, Question, QuestionSet
from gradeflow_core.editing import Position
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

from .helpers import mark_group, scheme

METHODS = {method.name: method for method in BUILTINS}
QUESTIONS: dict[str, Question] = {
    "Q1": NumberQuestion(),
    "Q2": NumberQuestion(),
    "Q3": TextQuestion(),
    "Q10": NumberQuestion(),
    "1a": NumberQuestion(),
}


def formula(text: str, target: str = "Q2", tolerance: float = 0) -> dict[str, Any]:
    return {
        "uses": "gradeflow/formula-of-answers@1",
        "with": {"target": target, "formula": text, "tolerance": tolerance},
    }


def marked(marking: dict[str, Any], **answers: str) -> tuple[float, str, bool]:
    blank = dict.fromkeys(QUESTIONS, "")
    results = mark_group(marking, {**blank, **answers}, QUESTIONS)
    assert set(results) == {marking["with"]["target"]}
    result = results[marking["with"]["target"]]
    return result.proportion, result.feedback, result.marked


def test_follows_through_from_the_answers_it_names() -> None:
    area = formula("Q1^2")
    assert marked(area, Q1="4", Q2="16") == (1, "Correct.", True)
    assert marked(area, Q1="5", Q2="25") == (1, "Correct.", True)
    assert marked(area, Q1="5", Q2="16") == (0, "Expected 25, from the answer to Q1.", True)


def test_several_answers_and_a_tolerance() -> None:
    product = formula("12 Q1 Q2", target="Q10", tolerance=0.5)
    assert marked(product, Q1="2", Q2="0.5", Q10="12.4")[0] == 1
    assert marked(product, Q1="2", Q2="0.5", Q10="13") == (
        0,
        "Expected about 12, from the answers to Q1 and Q2.",
        True,
    )
    assert marked(formula("Q2 + Q10", target="Q1"), Q2="1", Q10="2", Q1="4")[1] == (
        "Expected 3, from the answers to Q2 and Q10."
    )


def test_brackets_name_any_question() -> None:
    assert marked(formula("2[1a] + [Q1]"), **{"1a": "3", "Q1": "1", "Q2": "7"})[0] == 1


def test_long_results_are_written_briefly() -> None:
    assert marked(formula("Q1 / 3"), Q1="1", Q2="1")[1] == (
        "Expected 0.3333333333, from the answer to Q1."
    )


def test_blank_and_unreadable_answers() -> None:
    area = formula("Q1^2")
    assert marked(area, Q1="4") == (0, "No answer.", True)
    assert marked(area, Q2="16") == (0, "No answer to Q1 to work from.", True)
    proportion, feedback, is_marked = marked(area, Q1="4", Q2="lots")
    assert (proportion, is_marked) == (0, False)
    assert feedback
    proportion, feedback, is_marked = marked(area, Q1="lots", Q2="16")
    assert (proportion, is_marked) == (0, False)
    assert feedback.startswith("Q1: ")


def test_a_formula_it_cant_work_out_gives_no_marks() -> None:
    assert marked(formula("1 / Q1"), Q1="0", Q2="1") == (
        0,
        "Can't work out the expected answer from the answer to Q1. It divides by zero.",
        True,
    )


def test_marks_its_question_and_reads_those_its_formula_names() -> None:
    [marking] = scheme(formula("Q1 + [1a]", target="Q10")).markings
    assert marking.questions_marked() == {"Q10"}
    assert marking.questions_read() == {"Q1", "1a", "Q10"}


def test_describes_the_formula() -> None:
    [marking] = scheme(formula("Q1^2")).markings
    assert marking.describe() == "**Q2** equals `Q1^2`, from the student's own answers."
    [marking] = scheme(formula("Q1 + Q2", target="Q10", tolerance=0.5)).markings
    assert marking.describe() == (
        "**Q10** is within `0.5` of `Q1 + Q2`, from the student's own answers."
    )


def found(marking: dict[str, Any]) -> list[tuple[str, tuple[Any, ...], str]]:
    [check] = scheme(marking).markings
    return [(p.check, p.field, p.message) for p in check.problems(QuestionSet(QUESTIONS))]


def test_problems_with_its_question_and_formula() -> None:
    assert found(formula("Q1^2")) == []
    assert found(formula("Q3 + Q9", target="Q8")) == [
        ("", ("target",), "There's no question Q8."),
        ("", ("formula",), "Q3 isn't a number question."),
        ("", ("formula",), "There's no question Q9."),
    ]
    assert found(formula("12")) == [("", ("formula",), "Use another answer, such as Q1 * 12.")]
    assert found(formula("Q2 + 1")) == [("", ("formula",), "Q2 can't follow from itself.")]
    assert found(formula("2Q1 +")) == [
        ("", ("formula",), "Couldn't read “2Q1 +” as an expression.")
    ]


def test_offered_only_with_two_number_questions_to_choose() -> None:
    method = METHODS["gradeflow/formula-of-answers"]
    assert method.fits(Position(QuestionSet(QUESTIONS), in_group=True))
    one = QuestionSet({"Q1": NumberQuestion(), "Q2": TextQuestion()})
    assert not method.fits(Position(one, in_group=True))
    assert not method.fits(Position(QuestionSet(QUESTIONS), "Q1"))


def test_settings_offer_the_number_questions() -> None:
    position = Position(QuestionSet(QUESTIONS), in_group=True)
    schema = METHODS["gradeflow/formula-of-answers"].settings_schema(position)
    assert schema["properties"]["target"]["enum"] == ["1a", "Q1", "Q2", "Q10"]
