from typing import Any

from gradeflow_core import BUILTINS, Question, QuestionSet
from gradeflow_core.editing import Position
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion

from .helpers import located, mark_group, mark_scheme_of

FORMULA_OF_ANSWERS = BUILTINS.find("gradeflow/formula-of-answers@1")
QUESTIONS: dict[str, Question] = {
    "Q1": NumberQuestion(),
    "Q2": NumberQuestion(),
    "Q3": TextQuestion(),
    "Q10": NumberQuestion(),
    "1a": NumberQuestion(),
}


def formula(text: str, question_to_mark: str = "Q2", tolerance: float = 0) -> dict[str, Any]:
    return {
        "uses": "gradeflow/formula-of-answers@1",
        "with": {"question_to_mark": question_to_mark, "formula": text, "tolerance": tolerance},
    }


def mark_question(marking: dict[str, Any], **answers: str) -> tuple[float, str, bool]:
    blank_answers = dict.fromkeys(QUESTIONS, "")
    results = mark_group(marking, {**blank_answers, **answers}, QUESTIONS)
    assert set(results) == {marking["with"]["question_to_mark"]}
    result = results[marking["with"]["question_to_mark"]]
    return result.proportion, result.feedback, result.is_marked


def test_follows_through_from_the_answers_it_names() -> None:
    area = formula("Q1^2")
    assert mark_question(area, Q1="4", Q2="16") == (1, "Correct.", True)
    assert mark_question(area, Q1="5", Q2="25") == (1, "Correct.", True)
    assert mark_question(area, Q1="5", Q2="16") == (0, "Expected 25, from the answer to Q1.", True)


def test_several_answers_and_a_tolerance() -> None:
    product = formula("12 Q1 Q2", question_to_mark="Q10", tolerance=0.5)
    assert mark_question(product, Q1="2", Q2="0.5", Q10="12.4")[0] == 1
    assert mark_question(product, Q1="2", Q2="0.5", Q10="13") == (
        0,
        "Expected about 12, from the answers to Q1 and Q2.",
        True,
    )
    assert mark_question(formula("Q2 + Q10", question_to_mark="Q1"), Q2="1", Q10="2", Q1="4")[
        1
    ] == ("Expected 3, from the answers to Q2 and Q10.")


def test_brackets_name_any_question() -> None:
    assert mark_question(formula("2[1a] + [Q1]"), **{"1a": "3", "Q1": "1", "Q2": "7"})[0] == 1


def test_long_results_are_rounded() -> None:
    assert mark_question(formula("Q1 / 3"), Q1="1", Q2="1")[1] == (
        "Expected 0.3333333333, from the answer to Q1."
    )


def test_blank_and_unreadable_answers() -> None:
    area = formula("Q1^2")
    assert mark_question(area, Q1="4") == (0, "No answer.", True)
    assert mark_question(area, Q2="16") == (0, "No answer to Q1 to work from.", True)
    proportion, feedback, is_marked = mark_question(area, Q1="4", Q2="lots")
    assert (proportion, is_marked) == (0, False)
    assert feedback
    proportion, feedback, is_marked = mark_question(area, Q1="lots", Q2="16")
    assert (proportion, is_marked) == (0, False)
    assert feedback.startswith("Q1: ")


def test_a_formula_it_cant_work_out_gives_no_marks() -> None:
    assert mark_question(formula("1 / Q1"), Q1="0", Q2="1") == (
        0,
        "Can't work out the expected answer from the answer to Q1. It divides by zero.",
        True,
    )


def test_marks_its_question_and_reads_those_its_formula_names() -> None:
    [marking] = mark_scheme_of(formula("Q1 + [1a]", question_to_mark="Q10")).markings
    assert marking.questions_marked() == {"Q10"}
    assert marking.questions_read() == {"Q1", "1a", "Q10"}


def test_describes_the_formula() -> None:
    [marking] = mark_scheme_of(formula("Q1^2")).markings
    assert marking.describe() == "**Q2** equals `Q1^2`, from the student's own answers."
    [marking] = mark_scheme_of(formula("Q1 + Q2", question_to_mark="Q10", tolerance=0.5)).markings
    assert marking.describe() == (
        "**Q10** is within `0.5` of `Q1 + Q2`, from the student's own answers."
    )


def group_problems(marking: dict[str, Any]) -> list[tuple[str, tuple[Any, ...], str]]:
    [check] = mark_scheme_of(marking).markings
    return located(check.problems(QuestionSet(QUESTIONS)))


def test_problems_with_its_question_and_formula() -> None:
    assert group_problems(formula("Q1^2")) == []
    assert group_problems(formula("Q3 + Q9", question_to_mark="Q8")) == [
        ("", ("question_to_mark",), "There's no question Q8."),
        ("", ("formula",), "Q3 isn't a number question."),
        ("", ("formula",), "There's no question Q9."),
    ]
    assert group_problems(formula("12")) == [
        ("", ("formula",), "Use another answer, such as Q1 * 12.")
    ]
    assert group_problems(formula("Q2 + 1")) == [("", ("formula",), "Q2 can't follow from itself.")]
    assert group_problems(formula("2Q1 +")) == [
        ("", ("formula",), "Couldn't read “2Q1 +” as an expression.")
    ]


def test_offered_only_with_two_number_questions_to_choose() -> None:
    assert FORMULA_OF_ANSWERS.can_mark_at(Position(QuestionSet(QUESTIONS), in_group=True))
    one_number_question = QuestionSet({"Q1": NumberQuestion(), "Q2": TextQuestion()})
    assert not FORMULA_OF_ANSWERS.can_mark_at(Position(one_number_question, in_group=True))
    assert not FORMULA_OF_ANSWERS.can_mark_at(Position(QuestionSet(QUESTIONS), "Q1"))


def test_settings_offer_the_number_questions() -> None:
    position = Position(QuestionSet(QUESTIONS), in_group=True)
    schema = FORMULA_OF_ANSWERS.settings_schema(position)
    assert schema["properties"]["question_to_mark"]["enum"] == ["Q1", "Q2", "Q10", "1a"]
