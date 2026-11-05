import pytest

from gradeflow_core import Question, QuestionSet, Submission
from gradeflow_core.editing import converted_question
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion


def submission(student_id: str, answers: dict[str, str]) -> Submission:
    return Submission(student_id=student_id, answers=answers)


def _inferred_question(answers: list[str], existing: Question | None = None) -> Question:
    """The question of ``answers``, inferred or updated from ``existing``."""
    submissions = [
        Submission(student_id=f"s{index}", answers={"Q": answer})
        for index, answer in enumerate(answers)
    ]
    return QuestionSet({"Q": existing} if existing else {}).updated_from(submissions)["Q"]


def test_infer_empty_submissions_returns_empty_question_set() -> None:
    assert QuestionSet({}).updated_from([]) == QuestionSet({})


def test_infer_choice_from_few_distinct_values() -> None:
    question = _inferred_question(
        ["red, blue, green", "blue , yellow", "green, purple", "red, blue", "yellow"]
    )
    assert isinstance(question, ChoiceQuestion)
    assert question.options == {"red", "blue", "green", "yellow", "purple"}
    options = question.read("blue,  yellow ,  purple")
    assert options == frozenset({"blue", "yellow", "purple"})


def test_infer_choice_ignores_capitals() -> None:
    question = _inferred_question(["A,B,C", "B,D", "C,E,A"])
    assert isinstance(question, ChoiceQuestion)
    assert question.options == {"a", "b", "c", "d", "e"}
    assert question.read("B,E,A") == frozenset({"b", "e", "a"})


def test_infer_multiple_values_when_the_number_of_values_is_consistent() -> None:
    question = _inferred_question(["x~ y", "a~ b", "  c ~  d  ", ""])
    assert isinstance(question, ValuesQuestion)
    values = question.read("m~ n")
    assert values == ["m", "n"]


def test_infer_no_multiple_values_when_the_number_of_values_varies() -> None:
    question = _inferred_question(["x~ y~ z", "a~ b", "c1", "c2", "c3", "c4"])
    assert isinstance(question, TextQuestion)


def test_infer_a_number_question_when_most_answers_are_numbers() -> None:
    question = _inferred_question(["42", "3.14", "100", "N/A", "text"])
    assert isinstance(question, NumberQuestion)
    assert question.read("15") == 15
    assert question.read("2.5") == 2.5


def test_infer_text_when_no_other_type_fits() -> None:
    question = _inferred_question(
        ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    )
    assert isinstance(question, TextQuestion)
    assert question.read("Hello World") == "hello world"


def test_infer_handles_multiple_questions_in_same_run() -> None:
    submissions = [
        submission(
            "s1",
            {
                "choice_question": "red,blue",
                "values_question": "1~2",
                "number_question": "10",
                "text_question": "foo",
            },
        ),
        submission(
            "s2",
            {
                "choice_question": "blue,green",
                "values_question": "3~4",
                "number_question": "20",
                "text_question": "bar",
            },
        ),
        submission(
            "s3",
            {
                "choice_question": "green",
                "values_question": "5~6",
                "number_question": "30",
                "text_question": "baz1",
            },
        ),
        submission(
            "s4",
            {
                "choice_question": "green",
                "values_question": "5~6",
                "number_question": "1",
                "text_question": "baz2",
            },
        ),
        submission(
            "s5",
            {
                "choice_question": "green",
                "values_question": "5~6",
                "number_question": "2",
                "text_question": "baz3",
            },
        ),
        submission(
            "s6",
            {
                "choice_question": "green",
                "values_question": "5~6",
                "number_question": "3",
                "text_question": "baz4",
            },
        ),
        submission(
            "s7",
            {
                "choice_question": "green",
                "values_question": "5~6",
                "number_question": "4",
                "text_question": "baz5",
            },
        ),
    ]
    questions = QuestionSet({}).updated_from(submissions)

    assert isinstance(questions["choice_question"], ChoiceQuestion)
    assert questions["choice_question"].options == {"red", "blue", "green"}

    assert isinstance(questions["values_question"], ValuesQuestion)
    assert isinstance(questions["number_question"], NumberQuestion)
    assert isinstance(questions["text_question"], TextQuestion)


def test_read_with_an_inferred_question_set() -> None:
    submissions = [
        submission("s1", {"Q1": "A,B,C"}),
        submission("s2", {"Q1": "B,C"}),
    ]
    questions = QuestionSet({}).updated_from(submissions)
    new_submissions = [
        submission("s10", {"Q1": "A,C"}),
        submission("s11", {"Q1": "C"}),
    ]
    assert [questions.read(submission)["Q1"] for submission in new_submissions] == [
        frozenset({"a", "c"}),
        frozenset({"c"}),
    ]


def test_infer_choice_allows_multiple_when_counts_not_all_single_token() -> None:
    question = _inferred_question(["red, blue", "blue", "red, green"])
    assert isinstance(question, ChoiceQuestion)
    assert question.allow_multiple is True
    assert question.read("red, blue") == frozenset({"red", "blue"})


@pytest.mark.parametrize(
    "answers",
    [
        ["9.81 m/s^2", "9.8 m/s^2", "10 m/s^2"],
        ["6.02 x 10^23", "6.0×10²³", "6e23"],
        ["37 °C", "36.6 °C", "N/A"],
        ["1/2", "0.5", "twelve"],
        ["3.5", "4.1", "", " ", "N/A"],
    ],
)
def test_numbers_are_inferred_as_a_number_question_reads_them(answers: list[str]) -> None:
    question = _inferred_question(answers)
    assert isinstance(question, NumberQuestion)
    assert all(question.read(answer) is not None for answer in answers[:2])


def test_values_are_numbers_as_a_number_question_reads_them() -> None:
    question = _inferred_question(["3 kg~red", "4 kg~blue"])
    assert isinstance(question, ValuesQuestion)
    assert question.value_types == ["number", "text"]


def test_words_that_are_not_numbers_are_not_inferred_as_numbers() -> None:
    assert isinstance(_inferred_question(["nan", "inf", "approx 9.8"]), TextQuestion)


BLANK_ANSWERS = ["", "N/A", " , ", "  "]


def test_infer_blank_answers_as_text_not_an_empty_choice() -> None:
    assert _inferred_question(BLANK_ANSWERS).type == "text"


def test_updating_keeps_choice_options_when_every_answer_is_blank() -> None:
    choice = ChoiceQuestion(options=frozenset({"a", "b"}))
    assert _inferred_question(BLANK_ANSWERS, choice) == choice


@pytest.mark.parametrize(
    "answers",
    [
        ["H2O", "h2o", "H20", "water", "H2O2"],  # formulas
        ["3, -2", "3,-2", "-2, 3", "3, 2"],  # coordinates
        ["1969-07-20", "20/07/1969", "1969-7-20"],  # dates
        ["negative", "positive", "-"],  # a sign among the words
    ],
)
def test_few_distinct_answers_are_choices_only_when_they_look_like_options(
    answers: list[str],
) -> None:
    assert not isinstance(_inferred_question(answers), ChoiceQuestion)


def test_a_chosen_type_takes_its_settings_from_the_answers() -> None:
    answers = ["3, -2", "3,-2", "2, -3", "", "3, -2.0"]
    question = converted_question(TextQuestion(), "values", answers)
    assert isinstance(question, ValuesQuestion)
    assert question.value_types == ["number", "number"]
    assert question.read("3, -2") == [3, -2]
    choice = converted_question(TextQuestion(), "choice", ["A", "b", "A,C"])
    assert isinstance(choice, ChoiceQuestion)
    assert choice.options == {"a", "b", "c"}
    assert choice.allow_multiple
    assert isinstance(converted_question(NumberQuestion(), "text", ["x"]), TextQuestion)


def test_a_chosen_type_keeps_what_every_question_has() -> None:
    choice = ChoiceQuestion(options=frozenset({"a"}), max_marks=3, description="Pick one")
    question = converted_question(choice, "number", ["1", "2"])
    assert (question.type, question.max_marks, question.description) == ("number", 3, "Pick one")


def test_answers_with_the_same_number_of_comma_separated_values_are_values() -> None:
    question = _inferred_question(["2, -3", "-3, 2", "(2, -3)", "3, -2", ""])
    assert isinstance(question, ValuesQuestion)
    assert question.value_types == ["number", "number"]
    assert question.read("3, -2") == [3, -2]


def test_options_listed_with_commas_stay_a_choice() -> None:
    assert isinstance(_inferred_question(["A, C", "B, D", "A, B"]), ChoiceQuestion)


def test_commas_in_prose_do_not_turn_text_into_multiple_values() -> None:
    answers = ["Ice floats, because it is less dense.", "Ice sinks, because it is denser."]
    assert isinstance(_inferred_question(answers), TextQuestion)


def test_multiple_values_chosen_are_read_with_the_default_separator() -> None:
    question = converted_question(TextQuestion(), "values", ["1, x, 2", "3, y, 4", "5"])
    assert isinstance(question, ValuesQuestion)
    assert question.reading.separator == ","
    assert question.value_types == ["number", "text", "number"]
    tilde_values = converted_question(TextQuestion(), "values", ["a~b", "c~d"])
    assert isinstance(tilde_values, ValuesQuestion)
    assert (tilde_values.reading.separator, tilde_values.value_types) == (",", ["text", "text"])


@pytest.mark.parametrize("answers", [["Paris", "London"], ["3, east"], ["A", "B", "C"]])
def test_answers_that_share_no_option_arent_a_choice(answers: list[str]) -> None:
    assert not isinstance(_inferred_question(answers), ChoiceQuestion)
