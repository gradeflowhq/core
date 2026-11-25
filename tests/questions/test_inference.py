import pytest

from gradeflow_core import Question, QuestionSet, Submission
from gradeflow_core.editing import converted_question
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.list import ListQuestion
from gradeflow_core.questions.multipart import MultipartQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion


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
    assert question.options == ("blue", "green", "purple", "red", "yellow")
    options = question.read("blue,  yellow ,  purple")
    assert options == frozenset({"blue", "yellow", "purple"})


def test_infer_choice_ignores_capitals() -> None:
    question = _inferred_question(["A,B,C", "B,D", "C,E,A"])
    assert isinstance(question, ChoiceQuestion)
    assert question.options == ("a", "b", "c", "d", "e")
    assert question.read("B,E,A") == frozenset({"b", "e", "a"})


def test_infer_a_multipart_answer_when_the_number_of_values_is_consistent() -> None:
    question = _inferred_question(["x~ y", "a~ b", "  c ~  d  ", ""])
    assert isinstance(question, MultipartQuestion)
    assert [part.name for part in question.parts] == ["part_1", "part_2"]
    assert question.read("m~ n") == {"part_1": "m", "part_2": "n"}


def test_infer_no_multipart_answer_when_the_number_of_values_varies() -> None:
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
                "multipart_question": "1~2",
                "number_question": "10",
                "text_question": "foo",
            },
        ),
        submission(
            "s2",
            {
                "choice_question": "blue,green",
                "multipart_question": "3~4",
                "number_question": "20",
                "text_question": "bar",
            },
        ),
        submission(
            "s3",
            {
                "choice_question": "green",
                "multipart_question": "5~6",
                "number_question": "30",
                "text_question": "baz1",
            },
        ),
        submission(
            "s4",
            {
                "choice_question": "green",
                "multipart_question": "5~6",
                "number_question": "1",
                "text_question": "baz2",
            },
        ),
        submission(
            "s5",
            {
                "choice_question": "green",
                "multipart_question": "5~6",
                "number_question": "2",
                "text_question": "baz3",
            },
        ),
        submission(
            "s6",
            {
                "choice_question": "green",
                "multipart_question": "5~6",
                "number_question": "3",
                "text_question": "baz4",
            },
        ),
        submission(
            "s7",
            {
                "choice_question": "green",
                "multipart_question": "5~6",
                "number_question": "4",
                "text_question": "baz5",
            },
        ),
    ]
    questions = QuestionSet({}).updated_from(submissions)

    assert isinstance(questions["choice_question"], ChoiceQuestion)
    assert questions["choice_question"].options == ("blue", "green", "red")

    assert isinstance(questions["multipart_question"], MultipartQuestion)
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


def test_parts_are_numbers_as_a_number_question_reads_them() -> None:
    question = _inferred_question(["3 kg~red", "4 kg~blue"])
    assert isinstance(question, MultipartQuestion)
    assert [part.type for part in question.parts] == ["number", "text"]


def test_words_that_are_not_numbers_are_not_inferred_as_numbers() -> None:
    assert isinstance(_inferred_question(["nan", "inf", "approx 9.8"]), TextQuestion)


BLANK_ANSWERS = ["", "N/A", " , ", "  "]


def test_infer_blank_answers_as_text_not_an_empty_choice() -> None:
    assert _inferred_question(BLANK_ANSWERS).type == "text"


def test_updating_keeps_choice_options_when_every_answer_is_blank() -> None:
    choice = ChoiceQuestion(options=("a", "b"))
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
    question = converted_question(TextQuestion(), "multipart", answers)
    assert isinstance(question, MultipartQuestion)
    assert [part.type for part in question.parts] == ["number", "number"]
    assert question.read("3, -2") == {"part_1": 3, "part_2": -2}
    choice = converted_question(TextQuestion(), "choice", ["A", "b", "A,C"])
    assert isinstance(choice, ChoiceQuestion)
    assert choice.options == ("a", "b", "c")
    assert choice.allow_multiple
    assert isinstance(converted_question(NumberQuestion(), "text", ["x"]), TextQuestion)


def test_a_chosen_type_keeps_what_every_question_has() -> None:
    choice = ChoiceQuestion(options=("a",), max_marks=3, description="Pick one")
    question = converted_question(choice, "number", ["1", "2"])
    assert (question.type, question.max_marks, question.description) == ("number", 3, "Pick one")


def test_answers_with_the_same_number_of_comma_separated_numbers_are_multipart() -> None:
    question = _inferred_question(["2, -3", "-3, 2", "(2, -3)", "3, -2", ""])
    assert isinstance(question, MultipartQuestion)
    assert [part.type for part in question.parts] == ["number", "number"]
    assert question.read("3, -2") == {"part_1": 3, "part_2": -2}


def test_options_listed_with_commas_stay_a_choice() -> None:
    assert isinstance(_inferred_question(["A, C", "B, D", "A, B"]), ChoiceQuestion)


def test_commas_in_prose_do_not_turn_text_into_a_multipart_answer() -> None:
    answers = ["Ice floats, because it is less dense.", "Ice sinks, because it is denser."]
    assert isinstance(_inferred_question(answers), TextQuestion)


def test_a_multipart_answer_chosen_is_read_with_the_separator_that_splits_its_answers() -> None:
    question = converted_question(TextQuestion(), "multipart", ["1, x, 2", "3, y, 4", "5"])
    assert isinstance(question, MultipartQuestion)
    assert question.reading.separator == ","
    assert [part.type for part in question.parts] == ["number", "text", "number"]
    blanks = converted_question(TextQuestion(), "multipart", ["Waterloo~1815", "Leipzig~1813"])
    assert isinstance(blanks, MultipartQuestion)
    assert (blanks.reading.separator, [part.type for part in blanks.parts]) == (
        "~",
        ["text", "number"],
    )
    unsplit = converted_question(TextQuestion(), "multipart", ["light", "water"])
    assert isinstance(unsplit, MultipartQuestion)
    assert unsplit.reading.separator == ","


def test_a_later_separator_is_chosen_only_where_earlier_ones_split_few() -> None:
    fractions = converted_question(TextQuestion(), "list", ["1/2, 3/4", "1/3, 2/3", "1/2"])
    assert isinstance(fractions, ListQuestion)
    assert fractions.reading.separator == ","
    slashes = converted_question(TextQuestion(), "list", ["red/green", "blue/red/green", "red"])
    assert isinstance(slashes, ListQuestion)
    assert slashes.reading.separator == "/"


def test_points_written_in_brackets_are_a_multipart_question_of_numbers() -> None:
    question = _inferred_question(["(3, -2)", "(3,-2)", "(-2, 3)", "3, -2", ""])
    assert isinstance(question, MultipartQuestion)
    assert [part.type for part in question.parts] == ["number", "number"]


def test_letters_in_brackets_arent_inferred_as_choice_options() -> None:
    assert not isinstance(_inferred_question(["(A)", "(B)", "(A)", "(C)"]), ChoiceQuestion)


def test_examsoft_blanks_count_when_they_are_left_empty() -> None:
    question = _inferred_question(["Waterloo~1815", "Austerlitz~1805", "Waterloo~", "~1815"])
    assert isinstance(question, MultipartQuestion)
    assert [part.type for part in question.parts] == ["text", "number"]
    assert question.read("Waterloo~") == {"part_1": "waterloo", "part_2": None}


def test_a_list_chosen_is_of_numbers_when_most_values_are() -> None:
    assert converted_question(TextQuestion(), "list", ["3, -2", "1, 4"]) == ListQuestion(
        item="number"
    )
    assert converted_question(TextQuestion(), "list", ["a, 2", "b"]) == ListQuestion()


@pytest.mark.parametrize("answers", [["Paris", "London"], ["3, east"], ["A", "B", "C"]])
def test_answers_that_share_no_option_arent_a_choice(answers: list[str]) -> None:
    assert not isinstance(_inferred_question(answers), ChoiceQuestion)
