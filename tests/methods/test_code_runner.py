import math
import socket
from pathlib import Path

import pytest

from gradeflow_core.methods.code import runner
from gradeflow_core.methods.code.runner import CodeError, run


class TestRun:
    def test_run_returns_the_named_variables_the_code_set(self) -> None:
        assert run("x = x * 2\ny = 1", {"x": 5}, ("x", "z")) == {"x": 10}

    def test_set_round_trip(self) -> None:
        assert run("s = s | {4}", {"s": {1, 2, 3}}, ("s",)) == {"s": {1, 2, 3, 4}}

    def test_only_named_values_must_be_json_values_or_sets(self) -> None:
        code = "o = object()\ndata = list(range(300_000))\nitems = {10, 20}"
        assert run(code, {}, ("items",)) == {"items": {10, 20}}
        with pytest.raises(CodeError) as exc_info:
            run(code, {}, ("items", "o"))
        assert str(exc_info.value) == "TypeError: o isn't a JSON value or a set"

    def test_variables_must_be_json_values(self) -> None:
        with pytest.raises(TypeError, match="JSON-serializable"):
            run("", {"obj": object()}, ())

    @pytest.mark.parametrize(
        "code",
        [
            "import json\njson.dumps = lambda *args, **kwargs: '{'\nx = 1",
            "import json\njson.dumps = lambda *args, **kwargs: '\"text\"'\nx = 1",
        ],
    )
    def test_invalid_returned_variables_fail_only_their_run(self, code: str) -> None:
        with pytest.raises(CodeError) as exc_info:
            run(code, {}, ("x",))
        assert str(exc_info.value) == "Code returned invalid variables."
        assert run("x = 1", {}, ("x",)) == {"x": 1}

    def test_a_failure_says_its_exception_without_the_traceback(self) -> None:
        with pytest.raises(CodeError) as exc_info:
            run("1 / 0", {"x": 1}, ())
        assert str(exc_info.value) == "ZeroDivisionError: division by zero"

    def test_a_timeout_says_how_long_the_code_had(self) -> None:
        with pytest.raises(CodeError) as exc_info:
            run("while True: pass", {}, (), time_limit_s=1)
        assert str(exc_info.value) == "it took longer than 1 second"

    def test_a_timeout_does_not_prevent_the_next_execution(self) -> None:
        with pytest.raises(CodeError):
            run("while True: pass", {}, (), time_limit_s=1)
        assert run("correct = True", {}, ("correct",), time_limit_s=1) == {"correct": True}

    def test_nan_and_infinity_round_trip(self) -> None:
        variables = run(
            "not_a_number = float('nan')\nnegative_infinity = float('-inf')",
            {},
            ("not_a_number", "negative_infinity"),
        )
        assert math.isnan(variables["not_a_number"])
        assert variables["negative_infinity"] == -math.inf

    def test_code_aborting_its_sandbox_fails_only_its_run(self) -> None:
        # Eryx 0.12 aborts the process running it when deep C recursion overflows its stack.
        with pytest.raises(CodeError) as exc_info:
            run("nested = []\nfor _ in range(5000): nested = [nested]", {}, ())
        assert str(exc_info.value) == "it crashed the sandbox"
        assert run("correct = True", {}, ("correct",)) == {"correct": True}

    def test_a_worker_that_doesnt_answer_in_time_is_replaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner, "_WORKER_GRACE_S", -0.99)
        with pytest.raises(CodeError, match="it took longer than 1 second"):
            run("while True: pass", {}, (), time_limit_s=1)
        monkeypatch.undo()
        assert run("correct = True", {}, ("correct",)) == {"correct": True}

    def test_nested_set_round_trip(self) -> None:
        code = "results = {'Q1': answer_map['Q1'] | {'b'}}"
        assert run(code, {"answer_map": {"Q1": {"a"}}}, ("results",)) == {
            "results": {"Q1": {"a", "b"}}
        }

    def test_memory_is_capped(self) -> None:
        with pytest.raises(CodeError) as exc_info:
            run("data = b'x' * 1024**3", {}, ())
        assert str(exc_info.value).startswith("MemoryError")

    def test_memory_limit_is_set_per_run(self) -> None:
        code = "data = b'x' * (96 * 1024**2)\nallocated = len(data)\ndel data"
        with pytest.raises(CodeError, match="MemoryError"):
            run(code, {}, ("allocated",), memory_limit_mb=64)
        assert run(code, {}, ("allocated",), memory_limit_mb=256) == {"allocated": 96 * 1024**2}

    def test_host_files_cannot_be_read(self, tmp_path: Path) -> None:
        secret = tmp_path / "secret"
        secret.write_text("host-only")
        with pytest.raises(CodeError):
            run(f"data = open({str(secret)!r}).read()", {}, ())

    def test_host_files_cannot_be_written(self, tmp_path: Path) -> None:
        output_file = tmp_path / "output"
        with pytest.raises(CodeError):
            run(f"open({str(output_file)!r}, 'w').write('changed')", {}, ())
        assert not output_file.exists()

    def test_network_access_is_denied(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with pytest.raises(CodeError):
                run(
                    f"import socket\nsocket.create_connection({listener.getsockname()!r}, 1)",
                    {},
                    (),
                )
            listener.settimeout(0.1)
            with pytest.raises(TimeoutError):
                listener.accept()

    def test_each_execution_has_fresh_globals_and_modules(self) -> None:
        run("import builtins, math\nbuiltins.leaked = True\nmath.leaked = True", {}, ())
        assert run(
            "import builtins, math\n"
            "nothing_leaked = not hasattr(builtins, 'leaked') and not hasattr(math, 'leaked')",
            {},
            ("nothing_leaked",),
        ) == {"nothing_leaked": True}

    def test_results_are_bounded(self) -> None:
        with pytest.raises(CodeError, match=r"result.*1 MiB"):
            run("feedback = 'x' * (2 * 1024**2)", {}, ("feedback",))

    def test_output_is_bounded(self) -> None:
        with pytest.raises(CodeError, match="output reached the 1 MiB limit"):
            run("print('x' * (1536 * 1024))", {}, ())

    def test_sandbox_result_does_not_reserve_a_user_variable(self) -> None:
        variables = {"result": "input", "code": "answer", "scope": {1}, "return_names": []}
        assert run("result += code\nscope.add(2)", variables, tuple(variables)) == {
            "result": "inputanswer",
            "code": "answer",
            "scope": {1, 2},
            "return_names": [],
        }

    @pytest.mark.parametrize(
        ("code", "module_files"),
        [
            ("from board import size", {"board.py": "size = 3"}),
            ("import pkg.board\nsize = pkg.board.size", {"pkg/board.py": "size = 3"}),
            ("from pkg import board\nsize = board.size", {"pkg/board.py": "size = 3"}),
            (
                "from pkg import size",
                {"pkg/__init__.py": "from .board import size", "pkg/board.py": "size = 3"},
            ),
        ],
    )
    def test_modules_can_be_imported_by_their_paths(
        self, code: str, module_files: dict[str, str]
    ) -> None:
        assert run(code, {}, ("size",), module_files=module_files) == {"size": 3}

    def test_a_module_that_doesnt_parse_is_named_in_the_error(self) -> None:
        with pytest.raises(CodeError) as excinfo:
            run("import board", {}, (), module_files={"board.py": "x = 1\n  y = 2"})
        assert str(excinfo.value) == "IndentationError: unexpected indent (board.py, line 2)"
