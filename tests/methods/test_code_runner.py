import math
import socket
from pathlib import Path
from typing import Any

import pytest

from gradeflow_core.methods.code import runner
from gradeflow_core.methods.code.runner import CodeError, run


class TestRun:
    def test_basic_execution(self) -> None:
        variables: dict[str, Any] = {"x": 5}
        run("x = x * 2", variables, time_limit_s=5)
        assert variables["x"] == 10

    def test_set_round_trip(self) -> None:
        variables: dict[str, Any] = {"s": {1, 2, 3}}
        run("s = s | {4}", variables, time_limit_s=5)
        assert variables["s"] == {1, 2, 3, 4}

    def test_values_json_cannot_hold_are_left_out(self) -> None:
        variables: dict[str, Any] = {"x": 1}
        run("o = object()\nitems = {10, 20}", variables, time_limit_s=5)
        assert variables == {"x": 1, "items": {10, 20}}

    def test_variables_must_be_json_values(self) -> None:
        with pytest.raises(TypeError, match="JSON-serializable"):
            run("", {"obj": object()}, time_limit_s=5)

    def test_a_failure_says_its_exception_without_the_traceback(self) -> None:
        with pytest.raises(CodeError) as error:
            run("1 / 0", {"x": 1}, time_limit_s=5)
        assert error.value.reason == "ZeroDivisionError: division by zero"

    def test_a_timeout_says_how_long_the_code_had(self) -> None:
        with pytest.raises(CodeError) as error:
            run("while True: pass", {}, time_limit_s=1)
        assert error.value.reason == "it took longer than 1 second"

    def test_a_timeout_does_not_prevent_the_next_execution(self) -> None:
        with pytest.raises(CodeError):
            run("while True: pass", {}, time_limit_s=1)
        variables: dict[str, Any] = {}
        run("correct = True", variables, time_limit_s=1)
        assert variables == {"correct": True}

    def test_nan_and_infinity_round_trip(self) -> None:
        variables: dict[str, Any] = {}
        run("missing = float('nan')\nunbounded = float('-inf')", variables)
        assert math.isnan(variables["missing"])
        assert variables["unbounded"] == -math.inf

    def test_code_aborting_its_sandbox_fails_only_its_run(self) -> None:
        # Eryx 0.12 aborts the process running it when deep C recursion overflows its stack.
        with pytest.raises(CodeError) as error:
            run("nested = []\nfor _ in range(5000): nested = [nested]", {})
        assert error.value.reason == "it crashed the sandbox"
        variables: dict[str, Any] = {}
        run("correct = True", variables)
        assert variables == {"correct": True}

    def test_a_worker_that_doesnt_answer_in_time_is_replaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner, "_WORKER_GRACE_S", -0.99)
        with pytest.raises(CodeError, match="it took longer than 1 second"):
            run("while True: pass", {}, time_limit_s=1)
        monkeypatch.undo()
        variables: dict[str, Any] = {}
        run("correct = True", variables)
        assert variables == {"correct": True}

    def test_nested_set_round_trip(self) -> None:
        variables: dict[str, Any] = {"answer_map": {"Q1": {"a"}}}
        run("results = {'Q1': answer_map['Q1'] | {'b'}}", variables, time_limit_s=5)
        assert variables["results"] == {"Q1": {"a", "b"}}

    def test_memory_is_capped(self) -> None:
        with pytest.raises(CodeError) as error:
            run("data = b'x' * 1024**3", {}, time_limit_s=5)
        assert error.value.reason.startswith("MemoryError")

    def test_memory_limit_is_configurable_per_execution(self) -> None:
        code = "data = b'x' * (96 * 1024**2)\nallocated = len(data)\ndel data"
        variables: dict[str, Any] = {}
        with pytest.raises(CodeError, match="MemoryError"):
            run(code, variables, time_limit_s=5, memory_limit_mb=64)
        assert variables == {}
        run(code, variables, time_limit_s=5, memory_limit_mb=256)
        assert variables == {"allocated": 96 * 1024**2}

    @pytest.mark.parametrize("limit", [0, -1])
    def test_memory_limit_must_be_positive(self, limit: int) -> None:
        with pytest.raises(ValueError, match="memory_limit_mb must be positive"):
            run("", {}, time_limit_s=5, memory_limit_mb=limit)

    def test_host_files_cannot_be_read(self, tmp_path: Path) -> None:
        secret = tmp_path / "secret"
        secret.write_text("host-only")
        with pytest.raises(CodeError):
            run(f"data = open({str(secret)!r}).read()", {}, time_limit_s=5)

    def test_host_files_cannot_be_written(self, tmp_path: Path) -> None:
        target = tmp_path / "output"
        with pytest.raises(CodeError):
            run(f"open({str(target)!r}, 'w').write('changed')", {}, time_limit_s=5)
        assert not target.exists()

    def test_network_access_is_denied(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with pytest.raises(CodeError):
                run(
                    f"import socket\nsocket.create_connection({listener.getsockname()!r}, 1)",
                    {},
                    time_limit_s=5,
                )
            listener.settimeout(0.1)
            with pytest.raises(TimeoutError):
                listener.accept()

    def test_each_execution_has_fresh_globals_and_modules(self) -> None:
        run("import builtins, math\nbuiltins.leaked = True\nmath.leaked = True", {}, time_limit_s=5)
        variables: dict[str, Any] = {}
        run(
            "import builtins, math\n"
            "clean = not hasattr(builtins, 'leaked') and not hasattr(math, 'leaked')",
            variables,
            time_limit_s=5,
        )
        assert variables == {"clean": True}

    def test_failed_code_does_not_update_inputs(self) -> None:
        variables: dict[str, Any] = {"answer": [1]}
        with pytest.raises(CodeError):
            run("answer.append(2)\nraise ValueError('failed')", variables, time_limit_s=5)
        assert variables == {"answer": [1]}

    def test_results_are_bounded(self) -> None:
        with pytest.raises(CodeError, match=r"result.*1 MiB"):
            run("feedback = 'x' * (2 * 1024**2)", {}, time_limit_s=5)

    def test_output_limit_is_configurable_independently(self) -> None:
        code = "print('x' * (1536 * 1024))"
        with pytest.raises(CodeError, match="output reached the 1 MiB limit"):
            run(code, {}, result_limit_mb=2)
        variables: dict[str, Any] = {}
        run(code, variables, output_limit_mb=2)
        assert variables == {}

    def test_result_limit_is_configurable_independently(self) -> None:
        code = "feedback = 'x' * (1536 * 1024)"
        variables: dict[str, Any] = {}
        with pytest.raises(CodeError, match="result exceeded 1 MiB"):
            run(code, variables, output_limit_mb=2)
        assert variables == {}
        run(code, variables, result_limit_mb=2)
        assert variables == {"feedback": "x" * (1536 * 1024)}

    @pytest.mark.parametrize("name", ["time_limit_s", "output_limit_mb", "result_limit_mb"])
    @pytest.mark.parametrize("limit", [0, -1])
    def test_limits_must_be_positive(self, name: str, limit: int) -> None:
        limits: dict[str, Any] = {name: limit}
        with pytest.raises(ValueError, match=f"{name} must be positive"):
            run("", {}, **limits)

    def test_sandbox_result_does_not_reserve_a_user_variable(self) -> None:
        variables: dict[str, Any] = {"result": "input", "code": "answer", "scope": {1}}
        run("result += code\nscope.add(2)", variables, time_limit_s=5)
        assert variables == {"result": "inputanswer", "code": "answer", "scope": {1, 2}}
