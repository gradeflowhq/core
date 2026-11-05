import atexit
import ctypes
import json
import os
import selectors
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from typing import IO, Any

import eryx

from ...common.error_formatting import shortened_reason
from ...common.exceptions import GradeFlowError
from ...common.text import plural

TIME_LIMIT_S = 5
"""How long code may run, in seconds, unless a check sets its own limit."""
MAX_TIME_LIMIT_S = 60
MEMORY_LIMIT_MB = 512
"""How much memory code may use, in MiB, unless a check sets its own limit."""
MIN_MEMORY_LIMIT_MB = 32
"""Enough for the sandbox to start, which takes about 16 MiB, and some code."""
MAX_MEMORY_LIMIT_MB = 1024
OUTPUT_LIMIT_MB = 1
"""For stdout and stderr together, as Eryx truncates each at 1 MiB."""
VARIABLES_LIMIT_MB = 1
_WORKER_GRACE_S = 5
"""How long the host waits beyond the time limit for a worker, which may first have to start."""
_GUEST_PROGRAM = """
import json

def encode_set(value):
    if isinstance(value, (set, frozenset)):
        return {"__set__": list(value)}
    raise TypeError()

def decode_set(value):
    return set(value["__set__"]) if value.keys() == {"__set__"} else value

def execute(code, variables, return_names):
    scope = json.loads(variables, object_hook=decode_set)
    exec(compile(code, "<code>", "exec"), scope)
    returned = {name: scope[name] for name in return_names if name in scope}
    for name, value in returned.items():
        try:
            json.dumps(value, default=encode_set)
        except (TypeError, ValueError, RecursionError):
            raise TypeError(f"{name} isn't a JSON value or a set") from None
    return json.dumps(returned, default=encode_set)
"""


class CodeError(GradeFlowError):
    """Code didn't run to the end; ``reason`` says why, without saying it was code."""

    def __init__(self, reason: str) -> None:
        self.reason = shortened_reason(reason)
        super().__init__(self.reason)


def syntax_error_text(error: SyntaxError, line: int | None) -> str:
    """``error`` as Python shows it, with ``line`` if it is 1 or more."""
    line_note = f" (line {line})" if line and line > 0 else ""
    return f"{type(error).__name__}: {error.msg}{line_note}"


def _encode_set(value: object) -> object:
    if isinstance(value, set | frozenset):
        return {"__set__": list(value)}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON-serializable")


def _decode_set(value: dict[str, Any]) -> Any:
    return set(value["__set__"]) if value.keys() == {"__set__"} else value


def _timeout_error(time_limit_s: int) -> CodeError:
    return CodeError(f"it took longer than {time_limit_s} {plural('second', time_limit_s)}")


def _returned_variables(execution: eryx.ExecuteResult) -> str:
    """The guest's JSON of the variables it returned. The guest encodes them itself, as Eryx's
    encoding refuses NaN and infinity."""
    if execution.result_error:
        raise CodeError(execution.result_error)
    if execution.result_json is None:
        # Eryx ends a run whose code raises SystemExit(None) or SystemExit(0) as a success.
        raise CodeError("SystemExit")
    if len(execution.result_json.encode()) > VARIABLES_LIMIT_MB * 1024**2:
        raise CodeError(f"result exceeded {VARIABLES_LIMIT_MB} MiB")
    return execution.result_json


def _decode(encoded: str) -> dict[str, Any]:
    """``encoded`` read with nested sets. Code can break the guest's encoding, so a failure is the
    code's own error."""
    try:
        variables = json.loads(json.loads(encoded), object_hook=_decode_set)
    except (TypeError, ValueError, RecursionError) as error:
        raise CodeError("Code returned invalid variables.") from error
    if not isinstance(variables, dict):
        raise CodeError("Code returned invalid variables.")
    return variables


def _execute(
    code: str, variables: str, return_names: list[str], time_limit_s: int, memory_limit_mb: int
) -> str:
    """Runs ``code`` in a fresh sandbox with ``variables``, as JSON, and returns those of
    ``return_names`` it set, as JSON. Runs in a worker, as a sandbox can abort the process running
    it."""
    source = f"{_GUEST_PROGRAM}\nresult = execute({code!r}, {variables!r}, {return_names!r})\n"
    cancel = eryx.CancelToken()
    output_bytes = 0

    def limit_output(chunk: bytes) -> None:
        nonlocal output_bytes
        output_bytes += len(chunk)
        # Reaching the limit counts: Eryx truncates a stream at 1 MiB, so it never passes it.
        if output_bytes >= OUTPUT_LIMIT_MB * 1024**2:
            cancel.cancel()

    try:
        sandbox = eryx.Sandbox(
            resource_limits=eryx.ResourceLimits(
                execution_timeout_ms=time_limit_s * 1000,
                max_memory_bytes=memory_limit_mb * 1024**2,
                max_callback_invocations=0,
            ),
            on_stdout=limit_output,
            on_stderr=limit_output,
        )
    except eryx.InitializationError as error:
        raise GradeFlowError("Couldn't start the Python sandbox.") from error
    try:
        execution = sandbox.execute(source, cancel=cancel)
    except eryx.CancelledError as error:
        raise CodeError(f"output reached the {OUTPUT_LIMIT_MB} MiB limit") from error
    except eryx.TimeoutError as error:
        raise _timeout_error(time_limit_s) from error
    except eryx.PythonExceptionError as error:
        raise CodeError(str(error).strip().splitlines()[-1]) from error
    except eryx.ResourceLimitError as error:
        raise CodeError("Code exceeded the sandbox's resource limits.") from error
    except eryx.ExecutionError as error:
        raise CodeError("Code stopped before returning a result.") from error
    if cancel.cancelled:
        raise CodeError(f"output reached the {OUTPUT_LIMIT_MB} MiB limit")
    return _returned_variables(execution)


def _send(stream: IO[bytes], message: dict[str, Any]) -> None:
    stream.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
    stream.flush()


def _receive(stream: IO[bytes], timeout_s: float) -> str:
    """The next line of ``stream``. Raises ``TimeoutError`` after ``timeout_s`` seconds and
    ``EOFError`` if the stream closes first."""
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout_s
    with selectors.DefaultSelector() as selector:
        selector.register(stream.fileno(), selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError
            chunk = os.read(stream.fileno(), 64 * 1024)
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            if b"\n" in chunk:
                return b"".join(chunks).split(b"\n", maxsplit=1)[0].decode()


def _disable_core_dumps() -> None:
    """Stops the kernel dumping a sandbox that aborts: a dump piped to a crash handler streams
    the whole address space Eryx reserved, many GiB, before the worker exits."""
    if sys.platform == "linux":
        pr_set_dumpable = 4
        ctypes.CDLL(None, use_errno=True).prctl(pr_set_dumpable, 0, 0, 0, 0)


def _serve() -> None:
    """Runs each request, a line of stdin, and writes its response to stdout."""
    _disable_core_dumps()
    for line in sys.stdin.buffer:
        try:
            response = {"variables": _execute(**json.loads(line))}
        except CodeError as error:
            response = {"code_error": error.reason}
        except GradeFlowError as error:
            response = {"sandbox_error": str(error)}
        _send(sys.stdout.buffer, response)


class _Worker:
    """A process running sandboxes, started on first use and again after a sandbox aborts it."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None

    def run(self, request: dict[str, Any], timeout_s: float) -> dict[str, str]:
        process = self._running_process()
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            _send(process.stdin, request)
            line = _receive(process.stdout, timeout_s)
        except TimeoutError as error:
            self.close()
            raise _timeout_error(request["time_limit_s"]) from error
        except (EOFError, OSError) as error:
            self.close()
            raise CodeError("it crashed the sandbox") from error
        response: dict[str, str] = json.loads(line)
        return response

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _running_process(self) -> subprocess.Popen[bytes]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self.close()
        self._process = subprocess.Popen(
            [sys.executable, "-m", __name__],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return self._process


_worker = _Worker()
_worker_lock = threading.Lock()
atexit.register(_worker.close)


def run(
    code: str,
    variables: Mapping[str, Any],
    return_names: tuple[str, ...],
    *,
    time_limit_s: int = TIME_LIMIT_S,
    memory_limit_mb: int = MEMORY_LIMIT_MB,
) -> dict[str, Any]:
    """Run Python in a sandbox, without network access, with ``variables``, and return those of
    ``return_names`` it set. Values in and out are JSON values and nested sets.

    Sandboxes run one at a time in a worker process, so one that aborts its process fails only
    its own run.
    """
    request = {
        "code": code,
        "variables": json.dumps(variables, default=_encode_set),
        "return_names": return_names,
        "time_limit_s": time_limit_s,
        "memory_limit_mb": memory_limit_mb,
    }
    with _worker_lock:
        response = _worker.run(request, time_limit_s + _WORKER_GRACE_S)
    if "sandbox_error" in response:
        raise GradeFlowError(response["sandbox_error"])
    if "code_error" in response:
        raise CodeError(response["code_error"])
    return _decode(response["variables"])


if __name__ == "__main__":
    _serve()
