from concurrent.futures import ThreadPoolExecutor

import pytest

from gradeflow_core import QuestionResult

from .helpers import mark


def mark_with_code(code: str, answer: str = "x") -> QuestionResult:
    return mark({"uses": "gradeflow/custom-code@1", "with": {"code": code}}, answer)


def test_user_code_environ_excludes_host_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRADEFLOW_TEST_SECRET", "s3cret")
    result = mark_with_code(
        "import os\n"
        "correct = 'GRADEFLOW_TEST_SECRET' not in os.environ\n"
        "feedback = ', '.join(sorted(os.environ))\n"
    )
    assert result.correct, result.feedback


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("import sys\nsys.exit(3)\n", "SystemExit"),
        ("import sys\nsys.exit(0)\n", "SystemExit"),
        ("raise SystemExit\n", "SystemExit"),
        ("import os\nos._exit(0)\n", "stopped before returning a result"),
    ],
)
def test_user_code_exiting_fails_only_that_answer(code: str, message: str) -> None:
    result = mark_with_code(code)
    assert result.correct is False
    assert message in result.feedback
    assert mark_with_code("correct = True").correct


@pytest.mark.parametrize(
    "code",
    [
        "import sys\n"
        "for _ in range(200):\n"
        "    print('x' * 100_000)\n"
        "    print('y' * 100_000, file=sys.stderr)",
        "import os\nfor _ in range(64): os.write(1, b'x' * 65536)",
        "import sys\nsys.stdout.buffer.write(b'x' * (2 * 1024**2))",
        "import sys\nsys.stdout.write('x' * 600000)\nsys.stderr.write('x' * 600000)",
    ],
)
def test_output_over_the_limit_stops_only_that_answer(code: str) -> None:
    result = mark_with_code(f"{code}\ncorrect = True")
    assert not result.is_marked
    assert "output reached the 1 MiB limit" in result.feedback
    assert mark_with_code("correct = True").correct


def test_concurrent_marking_keeps_results_apart() -> None:
    answers = [f"answer-{i}" for i in range(16)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda answer: mark_with_code("feedback = answer\ncorrect = True", answer), answers
            )
        )
    assert [result.feedback for result in results] == answers
