import socket
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import eryx
import pytest

from gradeflow_core.methods.code import runner


@pytest.fixture
def listener() -> Iterator[socket.socket]:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(2)
        yield server


@pytest.fixture
def local_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lets sandboxes reach localhost. It patches this process, so tests using it call the
    worker's ``_execute`` here rather than ``run``."""
    sandbox = eryx.Sandbox

    def create(**kwargs: Any) -> eryx.Sandbox:
        network = kwargs["network"]
        assert network is not None
        assert not network.allow_all_hosts
        # Use a local server to test the hostname allowlist.
        network.allow_localhost()
        return sandbox(**kwargs)

    monkeypatch.setattr(eryx, "Sandbox", create)


def _execute_locally(code: str, allowed_hosts: tuple[str, ...]) -> str:
    return runner._execute(
        code,
        "{}",
        time_limit_s=runner.TIME_LIMIT_S,
        memory_limit_mb=runner.MEMORY_LIMIT_MB,
        output_limit_mb=runner.OUTPUT_LIMIT_MB,
        result_limit_mb=runner.RESULT_LIMIT_MB,
        allowed_hosts=allowed_hosts,
    )


def test_allowed_host_can_exchange_data(listener: socket.socket, local_network: None) -> None:
    def respond() -> bytes:
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(2)
            message = connection.recv(4)
            connection.sendall(b"pong")
            return message

    port = listener.getsockname()[1]
    code = (
        "import socket\n"
        f"with socket.create_connection(('localhost', {port}), timeout=2) as connection:\n"
        "    connection.sendall(b'ping')\n"
        "    response = connection.recv(4).decode()"
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        reply = pool.submit(respond)
        variables = _execute_locally(code, allowed_hosts=("localhost",))
        assert reply.result(timeout=3) == b"ping"
    assert variables == '{"response": "pong"}'


def test_unlisted_host_is_denied(listener: socket.socket, local_network: None) -> None:
    port = listener.getsockname()[1]
    with pytest.raises(runner.CodeError, match="not-permitted"):
        _execute_locally(
            f"import socket\nsocket.create_connection(('localhost', {port}), timeout=1)",
            allowed_hosts=("example.test",),
        )
    listener.settimeout(0.1)
    with pytest.raises(TimeoutError):
        listener.accept()


def test_allowlist_does_not_enable_private_networks(listener: socket.socket) -> None:
    port = listener.getsockname()[1]
    with pytest.raises(runner.CodeError, match="blocked"):
        runner.run(
            f"import socket\nsocket.create_connection(('127.0.0.1', {port}), timeout=1)",
            {},
            allowed_hosts=("127.0.0.1",),
        )
    listener.settimeout(0.1)
    with pytest.raises(TimeoutError):
        listener.accept()


def test_allowlist_rejects_a_bare_string() -> None:
    with pytest.raises(TypeError, match="sequence of hostnames"):
        runner.run("", {}, allowed_hosts="example.test")
