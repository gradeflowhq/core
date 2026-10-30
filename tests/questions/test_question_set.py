import pytest

from gradeflow_core import InvalidError, Problem, QuestionSet, Submission
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.reading import ListReading
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.types import Unreadable


def submission(answers: dict[str, str]) -> Submission:
    return Submission(student_id="s1", answers=answers)


def test_answers_are_read_as_their_questions_read_them() -> None:
    questions = QuestionSet(
        {
            "Q1": TextQuestion(),
            "Q2": ChoiceQuestion(
                options=frozenset({"a", "b", "c"}),
                reading=ListReading(separator="|"),
            ),
            "Q3": NumberQuestion(),
        }
    )
    answers = questions.read(submission({"Q1": " hello ", "Q2": "A|b| C ", "Q3": "N/A"}))
    assert answers == {"Q1": "hello", "Q2": {"a", "b", "c"}, "Q3": None}


def test_an_answer_a_question_cant_read_says_why_and_answers_to_other_questions_are_left_out() -> (
    None
):
    questions = QuestionSet({"Q1": NumberQuestion()})
    answers = questions.read(submission({"Q1": "approx 9", "Q9": "x"}))
    assert answers == {"Q1": Unreadable("approx 9", "Couldn't read “approx 9” as a number.")}


def test_questions_are_in_natural_order() -> None:
    questions = QuestionSet({"Q10": TextQuestion(), "Q2": TextQuestion()})
    assert list(questions) == ["Q2", "Q10"]


def test_updating_keeps_answered_questions_extends_choices_and_works_out_new_ones() -> None:
    questions = QuestionSet(
        {
            "Q1": TextQuestion(description="Custom prompt", max_marks=2.5),
            "Q2": ChoiceQuestion(options=frozenset({"A"}), max_marks=3.0),
            "Q3": TextQuestion(description="Remove me"),
        }
    )
    synced = questions.updated_from(
        [
            submission({"Q1": "alpha", "Q2": "A, B", "Q4": "1"}),
            submission({"Q1": "beta", "Q2": "C", "Q4": "2"}),
        ]
    )
    assert list(synced) == ["Q1", "Q2", "Q4"]
    assert synced["Q1"] == questions["Q1"]
    q2 = synced["Q2"]
    assert isinstance(q2, ChoiceQuestion)
    assert (q2.options, q2.allow_multiple, q2.max_marks) == (frozenset({"a", "b", "c"}), False, 3.0)
    assert isinstance(synced["Q4"], NumberQuestion)


def test_a_new_question_is_out_of_the_highest_marks_given_for_it() -> None:
    given = [
        Submission(student_id="s1", answers={"Q1": "a", "Q2": "b"}, marks={"Q1": 1.5, "Q2": 0}),
        Submission(student_id="s2", answers={"Q1": "b", "Q2": "a"}, marks={"Q1": 4}),
    ]
    old = QuestionSet({"Q3": TextQuestion(max_marks=2)})
    synced = old.updated_from(
        [*given, Submission(student_id="s3", answers={"Q3": "x"}, marks={"Q3": 5})]
    )
    assert [synced[key].max_marks for key in ("Q1", "Q2", "Q3")] == [4, 1, 2]


def test_questions_dump_with_their_types_and_load_back() -> None:
    questions = QuestionSet(
        {"Q1": TextQuestion(), "Q2": ChoiceQuestion(options=frozenset({"b", "a"}), max_marks=2)}
    )
    assert questions.dump() == {
        "Q1": {"type": "text"},
        "Q2": {"type": "choice", "max_marks": 2.0, "options": ["a", "b"]},
    }
    assert QuestionSet.load(questions.dump()) == questions


def test_loading_what_isnt_a_question_says_where() -> None:
    with pytest.raises(InvalidError) as caught:
        QuestionSet.load({"Q1": {"type": "number", "max_marks": "x"}})
    assert caught.value.problems == [Problem("Must be a number.", field=("Q1", "max_marks"))]


def test_questions_are_written_and_read_as_yaml() -> None:
    questions = QuestionSet({"Q1": NumberQuestion(max_marks=2)})
    assert questions.to_yaml() == "Q1:\n  type: number\n  max_marks: 2.0\n"
    assert QuestionSet.from_yaml(questions.to_yaml()) == questions
    assert QuestionSet() == QuestionSet({})
    with pytest.raises(InvalidError, match="Line 1: "):
        QuestionSet.from_yaml("Q1: {type: number")
