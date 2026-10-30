# GradeFlow Core

A Python library for marking students' answers. You give it questions, a mark scheme and answers,
and it gives back marks and feedback. Mark schemes are built from methods such as Accepted Answers
and Key Points, and methods can be combined into new ones.

`gradeflow_core` is for scripts. `gradeflow_core.editing` adds what an editor of questions, markings
and methods needs ([Editing](#editing)). Before 1.0.0, minor releases may break the API
([Versioning](#versioning)).

## Installation

Python 3.14 or newer:

```bash
uv add "gradeflow-core @ git+https://github.com/gradeflowhq/core@v0.1.0"
```

Two extras add model-based marking:

- `ml` runs local models, for Same meaning matches and grouping answers by meaning.
- `llm` calls an OpenAI-compatible endpoint, for Answer Criteria.

The `ml` models download on first use, into `GRADEFLOW_CACHE_DIR` or `~/.cache/gradeflow`. To fetch
them ahead of time:

```bash
python -m gradeflow_core setup
```

## Marking

```python
from gradeflow_core import AnswersCsv, MarkScheme, QuestionSet, ResultsCsv, mark

students = AnswersCsv(student_id_column="student_id", answer_columns=["Q1", "Q2"]).read(
    "student_id,Q1,Q2\n10000001,red,10\n10000002,blue,8\n10000003,red,9\n"
)
questions = QuestionSet().updated_from(students)
scheme = MarkScheme.from_yaml("""
- question: Q1
  uses: gradeflow/multiple-choice@1
  with: {correct: [red]}
- question: Q2
  uses: gradeflow/number-range@1
  with: {min_value: 9, max_value: 10}
""")
assert scheme.problems(questions) == []

marked = list(mark(questions, scheme, students))
for student in marked:
    print(student.student_id, student.total_marks, "/", student.max_marks)
    print("\n".join(map(str, student.report())))

print(ResultsCsv().write(questions, marked))
```

`mark()` yields each sheet as it is marked. A `Submission` gives a `MarkedSubmission`, and any other
`AnswerSheet` a `MarkedSheet`. Each holds a `QuestionResult` per question, and `report()` gives a
printable `Report` per question.

A blank answer gets nothing, except from Full Marks. An answer the question can't read, a check that
fails to run, or a marking with problems leaves the answer unmarked, with the reason as feedback. A
question with no marking keeps the marks given on the sheet, if any.

Questions and mark schemes round-trip through `to_yaml()` and `from_yaml()`, or `dump()` and
`load()` for plain data:

```python
from gradeflow_core import NumberQuestion, QuestionSet

questions = QuestionSet({"Q1": NumberQuestion(max_marks=2)})
assert QuestionSet.from_yaml(questions.to_yaml()) == questions
```

## Language models

Answer Criteria asks a language model which of the lecturer's criteria an answer meets. The endpoint
is passed to `mark()`, not written into the marking:

```python
from gradeflow_core import (
    MarkScheme,
    ModelEndpoint,
    QuestionSet,
    Submission,
    choose_models,
    mark,
)

questions = QuestionSet.load({"Q1": {"type": "text", "max_marks": 2}})
criteria = [
    {"text": "Says ice is less dense than water", "marks": 1},
    {"text": "Explains that water expands as it freezes", "marks": 1},
]
scheme = MarkScheme.load(
    [{"question": "Q1", "uses": "gradeflow/answer-criteria@1", "with": {"criteria": criteria}}]
)
choice = choose_models(ModelEndpoint(url="https://models.example.com/v1", token="your-token"))
students = [Submission(student_id="1", answers={"Q1": "Because ice is less dense than water."})]
for marked in mark(questions, scheme, students, model_endpoint=choice.endpoint):
    assert marked.results["Q1"].proportion == 0.5
```

`choose_models` tries the endpoint's newest models on Answer Criteria's own examples and keeps the
faster of the first two that mark them all correctly, for each of its two roles. `choice.trials`
shows how each model did.

Answers go 12 to a request (`answers_per_request`) and sheets 60 at a time (`chunk_size`), with
progress reported to `on_progress`. Without an endpoint, these answers are left unmarked.
`MarkScheme.needs_model_endpoint()` tells you whether a scheme needs one.

## Questions

A `QuestionSet` maps question IDs to questions, in natural order (`Q2` before `Q10`).

| Type | Class | An answer is |
|---|---|---|
| `text` | `TextQuestion` | Text |
| `number` | `NumberQuestion` | A number, with fractions, units (`9.81 m/s^2`) and powers of ten (`6.02 x 10^23`) |
| `choice` | `ChoiceQuestion` | One or more options, like `B` or `A, C` |
| `values` | `ValuesQuestion` | Several values |
| `code` | `CodeQuestion` | Code, as written |

`updated_from(sheets)` adds questions for new answers, adds the options students chose, and drops
questions nobody answered. A code question with `guess_indentation` reads code exported without its
indentation as its likeliest layouts, and Best Answer marks the best of them.

Answers are always strings. A values question converts to and from its values:

```python
from gradeflow_core import ValuesQuestion

question = ValuesQuestion.load({"value_types": ["number", "text"]})
question.answer_values("3, east")  # {"value_1": "3", "value_2": "east"}
question.answer_from({"value_1": "4", "value_2": "west"})  # "4,west"
```

## Methods and checks

A `Method` is a way of marking, named with a version: `gradeflow/key-points@1`. A check is a method
with its settings, and a marking is a check on a question:

```yaml
question: Q3
uses: gradeflow/combination@1
with:
  combine: any
  checks:
    - name: Correct
      uses: gradeflow/accepted-answers@1
      with: {answers: ["9.81"]}
    - name: Sign error
      weight: 0.5
      uses: gradeflow/accepted-answers@1
      with: {answers: ["-9.81"]}
```

A check can also have an `id`, the `question` or `value` it reads, `read_only`, an `if` condition,
and a `feedback` template. A marking without a `question` marks a question group, and its checks
name the questions they read.

A `MarkScheme` is a list of markings. `problems(questions)` on a `MarkScheme` or a `Check` lists
what would stop it marking. Each `Problem` points at the check and field at fault, such as
`markings.0.with.checks.1`.

## Method collections

`BUILTINS` is a `Methods` collection of the built-in methods. Anything that reads checks takes
`methods=` to use others. `with_files` adds methods written in YAML, named `namespace/name`:

```python
from gradeflow_core import BUILTINS, MarkScheme

term = """
version: 1
label: Term
summary: Names the term.
question_types: [text]
inputs:
  term:
    title: Term
checks:
  - uses: gradeflow/accepted-answers@1
    with: {answers: ["{{ inputs.term }}"]}
"""
methods = BUILTINS.with_files([("intro-biology/term", term)])
scheme = MarkScheme.from_yaml(
    """
- question: Q1
  uses: intro-biology/term@1
  with: {term: osmosis}
""",
    methods=methods,
)
print(methods.find("gradeflow/key-points@1").guide)
for method in methods.latest():
    print(method.ref, method.label, sorted(method.question_types))
```

Files can use each other. A bad file raises `MethodFileError` with the line at fault.

The built-in Key Points is itself composed:

```yaml
version: 1
label: Key Points
summary: Marks for each point the answer includes.
question_types: [text]
inputs:
  points:
    title: Points
  combine:
    title: How points count
    default: proportional
combine: "{{ inputs.combine }}"
checks:
  - for_each: "{{ inputs.points }}"
    uses: gradeflow/key-point@1
    with:
      point: "{{ each }}"
```

An input takes the type of the setting it fills, and is required without a `default`. `for_each`
repeats a check with `each` and `index`. With `combine: proportional`, weights share out the marks.
With `combine: total`, each weight is a number of marks, capped at full marks. Templates are
sandboxed Jinja, and `{{ … }}` on its own keeps its value's type.

## Examples

Each method declares examples: questions, settings, and answers with the marks and feedback they
should get. `evaluate_examples` marks them and says whether each row matches:

```python
from gradeflow_core import evaluate_examples

for example in evaluate_examples("gradeflow/key-points@1"):
    print(example.description, [row.matches for row in example.rows])
```

## Comparing answers

`compare_answers` groups similar answers, by `exact`, `normalized`, `spelling` or `meaning`. The
comparison can be stored and regrouped at any threshold without the models:

```python
from gradeflow_core import AnswerComparison, compare_answers

comparison = compare_answers(["water", "waters", "salt"], similar="spelling")
stored = comparison.model_dump_json()
assert AnswerComparison.model_validate_json(stored).groups(threshold=0.85) == [[0, 1], [2]]
```

## Adapters

- `AnswersCsv` reads answers, and any marks already given, from CSV.
- `ResultsCsv` writes answers, marks, feedback, totals and reports to CSV.
- `read_answer_key` reads an ExamSoft answer key row as an `AnswerSheet`.
- `read_question_document` sets questions up from ExamSoft's Print/Export as DOC.

## Errors

The library raises `GradeFlowError`, with messages fit to show people. `InvalidError` carries the
`problems` with invalid data, and `MethodFileError` is for method files.

## Editing

`gradeflow_core.editing` is for tools that edit markings and methods field by field.

`MarkingDraft` edits a marking by check path. It gives each check's schema, values, fitting methods
and problems, and can add, update, retype, move and remove checks:

```python
from gradeflow_core import QuestionSet
from gradeflow_core.editing import Condition, ConditionRow, MarkingDraft

questions = QuestionSet.load({"Q1": {"type": "text"}})
draft = MarkingDraft.new("gradeflow/combination@1", questions, "Q1")
draft.add("with.checks.0", "gradeflow/accepted-answers@1")
draft.update("with.checks.0", {"answers": ["Paris"]})
draft.add("with.checks.1", "gradeflow/key-points@1")
incorrect = Condition((ConditionRow("with.checks.0", "incorrect"),))
draft.update("with.checks.1", {"points": ["capital"], "weight": 0.5, "if": incorrect})
assert draft.problems == []
print(draft.notes("with.checks.1"))
print(draft.finalized().dump())
```

`MethodDraft` writes a composed method from example questions. `expose` turns a setting into an
input, and `bind` and `fix` connect or disconnect them:

```python
from gradeflow_core import QuestionSet
from gradeflow_core.editing import MethodDraft

draft = MethodDraft.new(
    QuestionSet.load({"Q1": {"type": "number"}}),
    name="examples/within-range",
    version=1,
    label="Within range",
    summary="A number in a range.",
)
draft.checks.add("with.checks.0", "gradeflow/number-range@1")
draft.checks.update("with.checks.0", {"min_value": 10, "max_value": 20})
draft.expose("with.checks.0.with.min_value", label="Lowest")
assert draft.problems == []
print(draft.to_yaml())
```

`compatible_methods` lists the methods that fit at a `Position`, and `settings_schema` gives a
method's settings there:

```python
from gradeflow_core import BUILTINS, QuestionSet
from gradeflow_core.editing import Position, compatible_methods

questions = QuestionSet.load({"Q1": {"type": "number"}})
print([method.label for method in compatible_methods(Position(questions, "Q1"))])
number_range = BUILTINS.find("gradeflow/number-range@1")
print(list(number_range.settings_schema(Position(questions, "Q1"))["properties"]))
```

`evaluate_rows` marks example rows with given settings, and `as_example()` records the result as a
row:

```python
from gradeflow_core import QuestionSet
from gradeflow_core.editing import ExampleRow, evaluate_rows

questions = QuestionSet.load({"Q1": {"type": "number"}})
rows = [ExampleRow(answers={"Q1": "15"}), ExampleRow(answers={"Q1": "25"})]
results = evaluate_rows(
    "gradeflow/number-range@1",
    questions=questions,
    question="Q1",
    inputs={"min_value": 10, "max_value": 20},
    rows=rows,
)
recorded = [result.as_example() for result in results]
```

`mark_by_example` changes a marking so that given answers get full, part or no marks. The module
also has helpers for question types (`new_question`, `infer_question_as`), refs (`method_ref`,
`split_ref`, `rewrite_references`), models (`list_models`, `try_model`) and YAML (`load_yaml`,
`dump_yaml`).

## Running code

Custom Code and Code Tests run CPython 3.14 in a fresh [Eryx](https://github.com/eryx-org/eryx)
WebAssembly sandbox, with the standard library but no host files, packages or network. A run gets 5
seconds (Code Tests allows up to 60), 512 MiB of memory, and 1 MiB each of output and result.

## Adding to the library

Built-in methods are in `src/gradeflow_core/methods/`, in `answers/`, `text/`, `code/` and `logic/`.

- **Composed method:** a YAML file named after the method, and its name in `ORDER` in
  `methods/catalogue.py`.
- **Coded method:** a subclass of `AnswerMethod` or `Method`, an instance in `PYTHON`, its name in
  `ORDER`, and its examples in `<method-name>.examples.yaml` beside it.
- **Question type:** a subclass of `Question` implementing `read` and `python_type`, added to
  `AnyQuestion` in `questions/__init__.py` and to `QuestionType`.

The tests check that each built-in's examples get the results they declare.

## Versioning

Semantic versioning. Before 1.0.0, minor releases (`0.x.0`) may break compatibility and patch
releases (`0.x.y`) don't. Compatibility covers both public modules, the saved formats of questions,
mark schemes and method files, and documented marking behaviour. Release notes list breaking changes
and any change to awarded marks.

## Development

```bash
uv sync --all-extras
uv run ruff format && uv run ruff check && uv run mypy && uv run pytest
```

Tests that need the `ml` extra are marked `ml`.

## License

MIT.
