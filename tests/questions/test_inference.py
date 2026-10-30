import pytest

from gradeflow_core import Question, QuestionSet, Submission
from gradeflow_core.editing import infer_question_as
from gradeflow_core.questions.choice import ChoiceQuestion
from gradeflow_core.questions.number import NumberQuestion
from gradeflow_core.questions.text import TextQuestion
from gradeflow_core.questions.values import ValuesQuestion


def make_rs(student_id: str, answers: dict[str, str]) -> Submission:
    return Submission(student_id=student_id, answers=answers)


def test_infer_empty_submissions_returns_empty_question_set() -> None:
    assert QuestionSet({}).updated_from([]) == QuestionSet({})


def test_infer_choice_from_few_distinct_values() -> None:
    subs = [
        make_rs("s1", {"q1": "red, blue, green"}),
        make_rs("s2", {"q1": "blue , yellow"}),
        make_rs("s3", {"q1": "green, purple"}),
        make_rs("s4", {"q1": "red, blue"}),
        make_rs("s5", {"q1": "yellow"}),
    ]
    qs = QuestionSet({}).updated_from(subs)
    q = qs["q1"]
    assert isinstance(q, ChoiceQuestion)
    assert q.options == {"red", "blue", "green", "yellow", "purple"}
    parsed = q.read("blue,  yellow ,  purple")
    assert parsed == frozenset({"blue", "yellow", "purple"})


def test_infer_choice_ignores_capitals() -> None:
    subs = [
        make_rs("s1", {"q1": "A,B,C"}),
        make_rs("s2", {"q1": "B,D"}),
        make_rs("s3", {"q1": "C,E,A"}),
    ]
    qs = QuestionSet({}).updated_from(subs)
    q = qs["q1"]
    assert isinstance(q, ChoiceQuestion)
    assert q.options == {"a", "b", "c", "d", "e"}
    assert q.read("B,E,A") == frozenset({"b", "e", "a"})


def test_infer_multi_valued_when_consistent_cardinality() -> None:
    subs = [
        make_rs("s1", {"q2": "x~ y"}),
        make_rs("s2", {"q2": "a~ b"}),
        make_rs("s3", {"q2": "  c ~  d  "}),
        make_rs("s4", {"q2": ""}),
    ]
    qs = QuestionSet({}).updated_from(subs)
    q = qs["q2"]
    assert isinstance(q, ValuesQuestion)
    parsed = q.read("m~ n")
    assert parsed == ["m", "n"]


def test_infer_not_multi_valued_when_inconsistent_cardinality() -> None:
    subs = [
        make_rs("s1", {"qmv": "x~ y~ z"}),
        make_rs("s2", {"qmv": "a~ b"}),
        make_rs("s3", {"qmv": "c1"}),
        make_rs("s4", {"qmv": "c2"}),
        make_rs("s5", {"qmv": "c3"}),
        make_rs("s6", {"qmv": "c4"}),
    ]
    qs = QuestionSet({}).updated_from(subs)
    q = qs["qmv"]
    assert isinstance(q, TextQuestion)


def test_infer_numeric_when_majority_single_tokens_are_numeric() -> None:
    subs = [
        make_rs("s1", {"q3": "42"}),
        make_rs("s2", {"q3": "3.14"}),
        make_rs("s3", {"q3": "100"}),
        make_rs("s4", {"q3": "N/A"}),
        make_rs("s5", {"q3": "text"}),
    ]
    qs = QuestionSet({}).updated_from(subs)
    q = qs["q3"]
    assert isinstance(q, NumberQuestion)
    assert q.read("15") == 15
    assert q.read("2.5") == 2.5


def test_infer_text_fallback_when_none_of_the_conditions_match() -> None:
    subs = [
        make_rs("s1", {"q4": "alpha"}),
        make_rs("s2", {"q4": "beta"}),
        make_rs("s3", {"q4": "gamma"}),
        make_rs("s4", {"q4": "delta"}),
        make_rs("s5", {"q4": "epsilon"}),
        make_rs("s6", {"q4": "zeta"}),
        make_rs("s7", {"q4": "eta"}),
        make_rs("s8", {"q4": "theta"}),
    ]
    qs = QuestionSet({}).updated_from(subs)
    q = qs["q4"]
    assert isinstance(q, TextQuestion)
    assert q.read("Hello World") == "hello world"


def test_infer_handles_multiple_questions_in_same_run() -> None:
    subs = [
        make_rs("s1", {"q_choice": "red,blue", "q_multi": "1~2", "q_num": "10", "q_text": "foo"}),
        make_rs("s2", {"q_choice": "blue,green", "q_multi": "3~4", "q_num": "20", "q_text": "bar"}),
        make_rs("s3", {"q_choice": "green", "q_multi": "5~6", "q_num": "30", "q_text": "baz1"}),
        make_rs("s4", {"q_choice": "green", "q_multi": "5~6", "q_num": "1", "q_text": "baz2"}),
        make_rs("s5", {"q_choice": "green", "q_multi": "5~6", "q_num": "2", "q_text": "baz3"}),
        make_rs("s6", {"q_choice": "green", "q_multi": "5~6", "q_num": "3", "q_text": "baz4"}),
        make_rs("s7", {"q_choice": "green", "q_multi": "5~6", "q_num": "4", "q_text": "baz5"}),
    ]
    qs = QuestionSet({}).updated_from(subs)

    assert isinstance(qs["q_choice"], ChoiceQuestion)
    assert qs["q_choice"].options == {"red", "blue", "green"}

    assert isinstance(qs["q_multi"], ValuesQuestion)
    assert isinstance(qs["q_num"], NumberQuestion)
    assert isinstance(qs["q_text"], TextQuestion)


def test_parse_with_inferred_question_set() -> None:
    subs = [
        make_rs("s1", {"qid": "A,B,C"}),
        make_rs("s2", {"qid": "B,C"}),
    ]
    qs = QuestionSet({}).updated_from(subs)
    new_subs = [
        make_rs("s10", {"qid": "A,C"}),
        make_rs("s11", {"qid": "C"}),
    ]
    assert [qs.read(submission)["qid"] for submission in new_subs] == [
        frozenset({"a", "c"}),
        frozenset({"c"}),
    ]


def test_infer_choice_allows_multiple_when_counts_not_all_single_token() -> None:
    subs = [
        make_rs("s1", {"qm": "red, blue"}),
        make_rs("s2", {"qm": "blue"}),
        make_rs("s3", {"qm": "red, green"}),
    ]
    qs = QuestionSet({}).updated_from(subs)
    q = qs["qm"]
    assert isinstance(q, ChoiceQuestion)
    assert q.allow_multiple is True
    assert q.read("red, blue") == frozenset({"red", "blue"})


def _inferred(answers: list[str]) -> Question:
    submissions = [Submission(student_id=str(i), answers={"Q": a}) for i, a in enumerate(answers)]
    return QuestionSet({}).updated_from(submissions)["Q"]


@pytest.mark.parametrize(
    "answers",
    [
        ["9.81 m/s^2", "9.8 m/s^2", "10 m/s^2"],
        ["6.02 x 10^23", "6.0×10²³", "6e23"],
        ["37 °C", "36.6 °C", "N/A"],
        ["1/2", "0.5", "twelve"],
    ],
)
def test_numbers_are_inferred_as_a_number_question_reads_them(answers: list[str]) -> None:
    question = _inferred(answers)
    assert isinstance(question, NumberQuestion)
    assert all(question.read(answer) is not None for answer in answers[:2])


def test_values_are_numbers_as_a_number_question_reads_them() -> None:
    question = _inferred(["3 kg~red", "4 kg~blue"])
    assert isinstance(question, ValuesQuestion)
    assert question.value_types == ["number", "text"]


def test_words_that_are_not_numbers_are_not_inferred_as_numbers() -> None:
    assert isinstance(_inferred(["nan", "inf", "approx 9.8"]), TextQuestion)


BLANK_ANSWERS = ["", "N/A", " , ", "  "]


def _blank_submissions() -> list[Submission]:
    return [
        Submission(student_id=f"s{i}", answers={"Q1": answer})
        for i, answer in enumerate(BLANK_ANSWERS)
    ]


def test_infer_blank_answers_as_text_not_an_empty_choice() -> None:
    assert QuestionSet({}).updated_from(_blank_submissions())["Q1"].type == "text"


def test_sync_infers_blank_new_questions_as_text() -> None:
    synced = QuestionSet({}).updated_from(_blank_submissions())
    assert synced["Q1"].type == "text"


def test_sync_keeps_choice_options_when_every_answer_is_blank() -> None:
    question_set = QuestionSet({"Q1": ChoiceQuestion(options=frozenset({"a", "b"}))})
    synced = question_set.updated_from(_blank_submissions())
    assert synced["Q1"] == question_set["Q1"]


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
    subs = [make_rs(f"s{n}", {"q": answer}) for n, answer in enumerate(answers)]
    assert not isinstance(QuestionSet({}).updated_from(subs)["q"], ChoiceQuestion)


def test_a_chosen_type_takes_its_settings_from_the_answers() -> None:
    answers = ["3, -2", "3,-2", "2, -3", "", "3, -2.0"]
    question = infer_question_as(TextQuestion(), "values", answers)
    assert isinstance(question, ValuesQuestion)
    assert question.value_types == ["number", "number"]
    assert question.read("3, -2") == [3, -2]
    choice = infer_question_as(TextQuestion(), "choice", ["A", "b", "A,C"])
    assert isinstance(choice, ChoiceQuestion)
    assert choice.options == {"a", "b", "c"}
    assert choice.allow_multiple
    assert isinstance(infer_question_as(NumberQuestion(), "text", ["x"]), TextQuestion)


def test_a_chosen_type_keeps_what_every_question_has() -> None:
    current = ChoiceQuestion(options=frozenset({"a"}), max_marks=3, description="Pick one")
    question = infer_question_as(current, "number", ["1", "2"])
    assert (question.type, question.max_marks, question.description) == ("number", 3, "Pick one")


def test_answers_with_the_same_number_of_comma_separated_values_are_values() -> None:
    subs = [
        make_rs(f"s{n}", {"q": answer})
        for n, answer in enumerate(["2, -3", "-3, 2", "(2, -3)", "3, -2", ""])
    ]
    question = QuestionSet({}).updated_from(subs)["q"]
    assert isinstance(question, ValuesQuestion)
    assert question.value_types == ["number", "number"]
    assert question.read("3, -2") == [3, -2]


def test_options_listed_with_commas_stay_a_choice() -> None:
    subs = [make_rs(f"s{n}", {"q": answer}) for n, answer in enumerate(["A, C", "B, D", "A, B"])]
    assert isinstance(QuestionSet({}).updated_from(subs)["q"], ChoiceQuestion)


def test_commas_in_prose_do_not_turn_text_into_multiple_values() -> None:
    subs = [
        make_rs("s1", {"q": "Ice floats, because it is less dense."}),
        make_rs("s2", {"q": "Ice sinks, because it is denser."}),
    ]
    assert isinstance(QuestionSet({}).updated_from(subs)["q"], TextQuestion)


def test_multiple_values_chosen_are_read_with_the_default_separator() -> None:
    question = infer_question_as(TextQuestion(), "values", ["1, x, 2", "3, y, 4", "5"])
    assert isinstance(question, ValuesQuestion)
    assert question.reading.separator == ","
    assert question.value_types == ["number", "text", "number"]
    unsplit = infer_question_as(TextQuestion(), "values", ["a~b", "c~d"])
    assert isinstance(unsplit, ValuesQuestion)
    assert (unsplit.reading.separator, unsplit.value_types) == (",", ["text", "text"])


@pytest.mark.parametrize("answers", [["Paris", "London"], ["3, east"], ["A", "B", "C"]])
def test_answers_that_share_no_option_arent_a_choice(answers: list[str]) -> None:
    subs = [make_rs(f"s{n}", {"q": answer}) for n, answer in enumerate(answers)]
    assert not isinstance(QuestionSet({}).updated_from(subs)["q"], ChoiceQuestion)
