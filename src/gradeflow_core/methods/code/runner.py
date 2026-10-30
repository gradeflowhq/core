import atexit
import ctypes
import json
import os
import selectors
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from typing import IO, Any

import eryx

from ...common.error_formatting import format_reason
from ...common.exceptions import GradeFlowError
from ...common.text import plural

TIME_LIMIT_S = 5
"""How long code may run, unless a check sets its own limit."""
MEMORY_LIMIT_MB = 512
"""Default guest memory in MiB."""
OUTPUT_LIMIT_MB = 1
RESULT_LIMIT_MB = 1
ALLOWED_HOSTS: tuple[str, ...] = ()
_WORKER_GRACE_S = 5
"""How long the host waits beyond the time limit for a worker, which may first have to start."""
_PROGRAM = """
import json

def tag_set(value):
    if isinstance(value, (set, frozenset)):
        return {"__set__": list(value)}
    raise TypeError()

def encodes(value):
    try:
        json.dumps(value, default=tag_set)
        return True
    except (TypeError, ValueError, RecursionError):
        return False

def execute(code, variables):
    scope = json.loads(variables, object_hook=lambda d: set(d["__set__"])
                       if d.keys() == {"__set__"} else d)
    exec(compile(code, "<code>", "exec"), scope)
    kept = {k: v for k, v in scope.items() if k != "__builtins__" and encodes(v)}
    return json.dumps(kept, default=tag_set)
"""


class CodeError(GradeFlowError):
    """Code didn't run to the end; ``reason`` says why, without saying it was code."""

    def __init__(self, reason: str) -> None:
        self.reason = format_reason(reason)
        super().__init__(self.reason)


def _tag_set(value: object) -> object:
    if isinstance(value, set | frozenset):
        return {"__set__": list(value)}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON-serializable")


def _untag_set(value: dict[str, Any]) -> Any:
    return set(value["__set__"]) if value.keys() == {"__set__"} else value


def _timed_out(time_limit_s: int) -> CodeError:
    return CodeError(f"it took longer than {time_limit_s} {plural('second', time_limit_s)}")


def _returned(execution: eryx.ExecuteResult, limit_mb: int) -> str:
    """The variables the guest returned, as JSON. The guest encodes them itself, as Eryx's
    encoding refuses NaN and infinity."""
    if execution.result_error:
        raise CodeError(execution.result_error)
    if execution.result_json is None:
        # Eryx ends a run whose code raises SystemExit(None) or SystemExit(0) as a success.
        raise CodeError("SystemExit")
    encoded = json.loads(execution.result_json)
    if not isinstance(encoded, str):
        raise CodeError("Code returned invalid variables.")
    if len(encoded.encode()) > limit_mb * 1024**2:
        raise CodeError(f"result exceeded {limit_mb} MiB")
    return encoded


def _decode(encoded: str) -> dict[str, Any]:
    try:
        result = json.loads(encoded, object_hook=_untag_set)
    except (TypeError, ValueError, RecursionError) as error:
        raise CodeError("Code returned invalid variables.") from error
    if not isinstance(result, dict):
        raise CodeError("Code returned invalid variables.")
    return result


def _validate_limits(**limits: int) -> None:
    for name, limit in limits.items():
        if limit <= 0:
            raise ValueError(f"{name} must be positive")


def _network(allowed_hosts: Sequence[str]) -> eryx.NetConfig | None:
    return eryx.NetConfig(allowed_hosts=allowed_hosts) if allowed_hosts else None


def _execute(
    code: str,
    variables: str,
    time_limit_s: int,
    memory_limit_mb: int,
    output_limit_mb: int,
    result_limit_mb: int,
    allowed_hosts: Sequence[str],
) -> str:
    """Runs ``code`` in a fresh sandbox with ``variables``, as JSON, and returns its variables as
    JSON. Runs in a worker, as a sandbox can abort the process running it."""
    source = f"{_PROGRAM}\nresult = execute({code!r}, {variables!r})\n"
    cancel = eryx.CancelToken()
    output_bytes = 0

    def limit_output(chunk: bytes) -> None:
        nonlocal output_bytes
        output_bytes += len(chunk)
        # Reaching the limit counts: Eryx truncates a stream at 1 MiB, so it never passes it.
        if output_bytes >= output_limit_mb * 1024**2:
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
            network=_network(allowed_hosts),
        )
    except eryx.InitializationError as error:
        raise GradeFlowError("Couldn't start the Python sandbox.") from error
    try:
        result = sandbox.execute(source, cancel=cancel)
    except eryx.CancelledError as error:
        raise CodeError(f"output reached the {output_limit_mb} MiB limit") from error
    except eryx.TimeoutError as error:
        raise _timed_out(time_limit_s) from error
    except eryx.PythonExceptionError as error:
        raise CodeError(str(error).strip().splitlines()[-1]) from error
    except eryx.ResourceLimitError as error:
        raise CodeError("Code exceeded the sandbox's resource limits.") from error
    except eryx.ExecutionError as error:
        raise CodeError("Code stopped before returning a result.") from error
    if cancel.cancelled:
        raise CodeError(f"output reached the {output_limit_mb} MiB limit")
    return _returned(result, result_limit_mb)


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
    """Runs each request, a line of stdin, and writes its outcome to stdout."""
    _disable_core_dumps()
    for line in sys.stdin.buffer:
        try:
            outcome = {"variables": _execute(**json.loads(line))}
        except CodeError as error:
            outcome = {"error": error.reason}
        except GradeFlowError as error:
            outcome = {"failure": str(error)}
        _send(sys.stdout.buffer, outcome)


class _Worker:
    """A process running sandboxes, started on first use and again after a sandbox aborts it.
    Only the thread that made it uses it."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None

    def run(self, request: dict[str, Any], timeout_s: float) -> dict[str, str]:
        process = self._started()
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            _send(process.stdin, request)
            line = _receive(process.stdout, timeout_s)
        except TimeoutError as error:
            self.close()
            raise _timed_out(request["time_limit_s"]) from error
        except (EOFError, OSError) as error:
            self.close()
            raise CodeError("it crashed the sandbox") from error
        outcome: dict[str, str] = json.loads(line)
        return outcome

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

    def _started(self) -> subprocess.Popen[bytes]:
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


_local = threading.local()
_workers: list[_Worker] = []
_workers_lock = threading.Lock()


def _worker() -> _Worker:
    worker: _Worker | None = getattr(_local, "worker", None)
    if worker is None:
        worker = _local.worker = _Worker()
        with _workers_lock:
            _workers.append(worker)
    return worker


@atexit.register
def _close_workers() -> None:
    with _workers_lock:
        workers = list(_workers)
        _workers.clear()
    for worker in workers:
        worker.close()


def run(
    code: str,
    variables: dict[str, Any],
    *,
    time_limit_s: int = TIME_LIMIT_S,
    memory_limit_mb: int = MEMORY_LIMIT_MB,
    output_limit_mb: int = OUTPUT_LIMIT_MB,
    result_limit_mb: int = RESULT_LIMIT_MB,
    allowed_hosts: Sequence[str] = ALLOWED_HOSTS,
) -> None:
    """Run Python in a sandbox and update variables with its JSON values and nested sets.

    Unsupported output values are omitted. Variables are updated only on success.
    Limits must be positive; size limits use MiB (1024**2 bytes).
    Eryx truncates raw file-descriptor output at 1 MiB per stream.
    Network access is limited to ``allowed_hosts``; private addresses are blocked.
    Sandboxes run in a worker process per thread, so one that aborts its process fails only
    its own run.
    """
    _validate_limits(
        time_limit_s=time_limit_s,
        memory_limit_mb=memory_limit_mb,
        output_limit_mb=output_limit_mb,
        result_limit_mb=result_limit_mb,
    )
    if isinstance(allowed_hosts, str):
        raise TypeError("allowed_hosts must be a sequence of hostnames")
    request = {
        "code": code,
        "variables": json.dumps(variables, default=_tag_set),
        "time_limit_s": time_limit_s,
        "memory_limit_mb": memory_limit_mb,
        "output_limit_mb": output_limit_mb,
        "result_limit_mb": result_limit_mb,
        "allowed_hosts": list(allowed_hosts),
    }
    outcome = _worker().run(request, time_limit_s + _WORKER_GRACE_S)
    if "failure" in outcome:
        raise GradeFlowError(outcome["failure"])
    if "error" in outcome:
        raise CodeError(outcome["error"])
    variables.update(_decode(outcome["variables"]))


if __name__ == "__main__":
    _serve()
