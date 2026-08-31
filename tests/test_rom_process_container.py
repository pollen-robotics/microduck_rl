from __future__ import annotations

import json
import os
import select
import shlex
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from tests.test_rom_qualification import _docker_context_includes

_REPOSITORY_FILES: Final[frozenset[str]] = frozenset(
    {
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        "docker/rom-simulator/entrypoint.sh",
        "docker/rom-simulator/mjlab_microduck_rom.pth",
        "docker/rom-simulator/pid1_bootstrap.py",
        "schemas/microduck-policy-bundle-v1.schema.json",
        "schemas/microduck-simulator-api-v1.openapi.yaml",
        "schemas/microduck-v1-portability-fixtures.json",
        "src/mjlab_microduck/__init__.py",
        "src/mjlab_microduck/rom/__init__.py",
        "src/mjlab_microduck/rom/action_catalog.py",
        "src/mjlab_microduck/rom/action_specs.py",
        "src/mjlab_microduck/rom/api.py",
        "src/mjlab_microduck/rom/bundle.py",
        "src/mjlab_microduck/rom/contracts.py",
        "src/mjlab_microduck/rom/main.py",
        "src/mjlab_microduck/rom/mirroring.py",
        "src/mjlab_microduck/rom/model_semantics.py",
        "src/mjlab_microduck/rom/mujoco_runtime.py",
        "src/mjlab_microduck/rom/observation.py",
        "src/mjlab_microduck/rom/onnx_policy.py",
        "src/mjlab_microduck/rom/parent_death.py",
        "src/mjlab_microduck/rom/process_protocol.py",
        "src/mjlab_microduck/rom/process_service.py",
        "src/mjlab_microduck/rom/process_supervisor.py",
        "src/mjlab_microduck/rom/qualification.py",
        "src/mjlab_microduck/rom/runtime.py",
        "src/mjlab_microduck/rom/runtime_child.py",
        "src/mjlab_microduck/rom/runtime_identity.py",
        "src/mjlab_microduck/rom/service.py",
        "src/mjlab_microduck/rom/store.py",
        "src/mjlab_microduck/rom/supervisor_state.py",
    }
)
_IMAGE_RUNTIME_FILES: Final[frozenset[str]] = frozenset(
    {
        "LICENSE",
        "pyproject.toml",
        *(
            path
            for path in _REPOSITORY_FILES
            if path.startswith(("schemas/", "src/"))
        ),
    }
)
_TOKEN: Final = "container-release-gate-token"
_IMAGE_DISTRIBUTIONS: Final[frozenset[str]] = frozenset(
    {
        "absl-py",
        "annotated-doc",
        "annotated-types",
        "anyio",
        "click",
        "etils",
        "fastapi",
        "flatbuffers",
        "fsspec",
        "glfw",
        "h11",
        "idna",
        "importlib-resources",
        "ml-dtypes",
        "mpmath",
        "mujoco",
        "numpy",
        "onnx",
        "onnxruntime",
        "packaging",
        "protobuf",
        "pydantic",
        "pydantic-core",
        "pyopengl",
        "starlette",
        "sympy",
        "typing-extensions",
        "typing-inspection",
        "uvicorn",
        "zipp",
    }
)


@dataclass(frozen=True)
class _Container:
    name: str
    image: str
    base_url: str
    state_dir: Path
    parent_pid: int
    child_pid: int


def _release_inputs(tmp_path: Path) -> tuple[str, Path]:
    image = os.environ.get("MICRODUCK_ROM_CONTAINER_TEST_IMAGE")
    bundle_input = os.environ.get("MICRODUCK_ROM_CONTAINER_TEST_BUNDLE")
    if not image or not bundle_input:
        pytest.skip(
            "set MICRODUCK_ROM_CONTAINER_TEST_IMAGE and "
            "MICRODUCK_ROM_CONTAINER_TEST_BUNDLE for real lifecycle evidence"
        )
    source = Path(bundle_input).resolve()
    if source.is_dir():
        bundle = source
    else:
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        with zipfile.ZipFile(source) as archive:
            archive.extractall(bundle)
    assert (bundle / "microduck-policy-bundle.json").is_file()
    return image, bundle


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        capture_output=True,
        text=True,
    )


def _prepare_state(image: str, state_dir: Path) -> None:
    state_dir.mkdir(exist_ok=True)
    mount = f"type=bind,src={state_dir},dst=/state"
    _run(
        "docker",
        "run",
        "--rm",
        "--user",
        "0:0",
        "--mount",
        mount,
        "--entrypoint",
        "/bin/chown",
        image,
        "10001:10001",
        "/state",
    )
    _run(
        "docker",
        "run",
        "--rm",
        "--user",
        "0:0",
        "--mount",
        mount,
        "--entrypoint",
        "/bin/chmod",
        image,
        "0750",
        "/state",
    )


def _container_pids(name: str) -> tuple[int, tuple[int, ...]]:
    parent_pid = int(
        _run("docker", "inspect", "--format", "{{.State.Pid}}", name).stdout
    )
    top = _run("docker", "top", name, "-eo", "pid,ppid").stdout.splitlines()
    pids = tuple(
        int(line.split()[0])
        for line in top[1:]
        if line.split() and line.split()[0].isdigit()
    )
    descendants = tuple(pid for pid in pids if pid != parent_pid)
    return parent_pid, descendants


def _namespace_pid(host_pid: int) -> int:
    status = Path(f"/proc/{host_pid}/status").read_text()
    line = next(item for item in status.splitlines() if item.startswith("NSpid:"))
    return int(line.split()[-1])


def _container_proc_maps(name: str, namespace_pid: int) -> str:
    return _run(
        "docker",
        "exec",
        "--user",
        "10001:10001",
        name,
        "/bin/cat",
        f"/proc/{namespace_pid}/maps",
    ).stdout.lower()


def _signal_container_pid(
    container: _Container,
    host_pid: int,
    signum: signal.Signals,
    *,
    check: bool = True,
) -> None:
    """Signal one exact namespace PID as the same non-root container UID."""
    namespace_pid = _namespace_pid(host_pid)
    completed = _run(
        "docker",
        "exec",
        "--user",
        "10001:10001",
        container.name,
        "/app/.venv/bin/python",
        "-P",
        "-c",
        "import os,sys;os.kill(int(sys.argv[1]),int(sys.argv[2]))",
        str(namespace_pid),
        str(int(signum)),
        check=check,
    )
    if check:
        assert completed.returncode == 0


def _parent_wchans(container: _Container) -> tuple[str, ...]:
    completed = _run(
        "docker",
        "exec",
        "--user",
        "10001:10001",
        container.name,
        "/app/.venv/bin/python",
        "-P",
        "-c",
        (
            "from pathlib import Path;"
            "print('\\n'.join(p.read_text().strip() "
            "for p in Path('/proc/1/task').glob('*/wchan')))"
        ),
    )
    return tuple(completed.stdout.splitlines())


def _wait_parent_wchan_count(
    container: _Container,
    expected: str,
    expected_count: int,
    *,
    timeout: float = 5.0,
) -> None:
    """Wait for a kernel-visible parent thread state, not an elapsed-time guess."""
    deadline = time.monotonic() + timeout
    latest = ""
    while time.monotonic() < deadline:
        wchans = _parent_wchans(container)
        latest = "\n".join(wchans)
        if sum(expected in item for item in wchans) >= expected_count:
            return
        time.sleep(0.01)
    raise AssertionError(f"parent thread did not enter {expected}: {latest}")


def _launch_container(
    *, image: str, bundle: Path, state_dir: Path, suffix: str
) -> _Container:
    _prepare_state(image, state_dir)
    name = f"microduck-rom-task5-{os.getpid()}-{suffix}"
    _run("docker", "rm", "--force", name, check=False)
    _run(
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--user",
        "10001:10001",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        f"type=bind,src={bundle},dst=/bundle,readonly",
        "--mount",
        f"type=bind,src={state_dir},dst=/state",
        "--env",
        f"MICRODUCK_ROM_BEARER_TOKEN={_TOKEN}",
        "--publish",
        "127.0.0.1::8000",
        "--stop-timeout",
        "60",
        image,
    )
    deadline = time.monotonic() + 30
    base_url = ""
    last_error = "container did not publish a port"
    while time.monotonic() < deadline:
        port = _run("docker", "port", name, "8000/tcp", check=False).stdout.strip()
        if port:
            base_url = f"http://127.0.0.1:{port.rsplit(':', 1)[1]}"
            try:
                status, body = _request(base_url, "GET", "/v1/ready")
                if status == 200 and body.get("ready") is True:
                    break
                last_error = f"readiness={status}:{body}"
            except (OSError, ValueError) as exc:
                last_error = type(exc).__name__
        time.sleep(0.05)
    else:
        logs = _run("docker", "logs", name, check=False)
        _run("docker", "rm", "--force", name, check=False)
        _restore_state_ownership(image, state_dir)
        raise AssertionError(f"{last_error}; logs={logs.stderr[-2000:]}")
    parent_pid, descendants = _container_pids(name)
    assert len(descendants) == 1
    return _Container(name, image, base_url, state_dir, parent_pid, descendants[0])


def _launch_without_readiness_wait(
    *, image: str, bundle: Path, state_dir: Path, suffix: str
) -> tuple[str, int, tuple[int, ...]]:
    """Start the production entrypoint and return as soon as Docker owns PID 1."""
    _prepare_state(image, state_dir)
    name = f"microduck-rom-task5-{os.getpid()}-{suffix}"
    _run("docker", "rm", "--force", name, check=False)
    _run(
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--user",
        "10001:10001",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        f"type=bind,src={bundle},dst=/bundle,readonly",
        "--mount",
        f"type=bind,src={state_dir},dst=/state",
        "--env",
        f"MICRODUCK_ROM_BEARER_TOKEN={_TOKEN}",
        "--stop-timeout",
        "60",
        image,
    )
    deadline = time.monotonic() + 5.0
    marker = "/tmp/.microduck-pid1-sigterm-ready"
    while time.monotonic() < deadline:
        observed = _run(
            "docker",
            "exec",
            "--user",
            "10001:10001",
            name,
            "/usr/bin/test",
            "-f",
            marker,
            check=False,
        )
        if observed.returncode == 0:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("PID 1 did not publish its pre-import SIGTERM barrier")
    parent_pid, descendants = _container_pids(name)
    return name, parent_pid, descendants


def _request(
    base_url: str,
    method: str,
    path: str,
    document: object | None = None,
    *,
    raw: bytes | None = None,
    timeout: float = 20.0,
) -> tuple[int, dict[str, object]]:
    data = raw
    headers = {"Authorization": f"Bearer {_TOKEN}"}
    if document is not None:
        data = json.dumps(document, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    elif raw is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        base_url + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return exc.code, json.loads(body) if body else {}


def _post_after_slot_resolution(
    container: _Container,
    path: str,
    document: object,
    *,
    expected_status: int,
    timeout: float = 10.0,
) -> tuple[int, dict[str, object]]:
    """Retry only transient fail-closed readiness while delivery/reap completes."""
    deadline = time.monotonic() + timeout
    latest: tuple[int, dict[str, object]] = (0, {})
    while time.monotonic() < deadline:
        latest = _request(container.base_url, "POST", path, document)
        if latest[0] == expected_status:
            return latest
        assert latest[0] == 503 and latest[1]["code"] == "NOT_READY"
        time.sleep(0.01)
    raise AssertionError(f"motion slot did not resolve: {latest}")


def _bundle_identity(container: _Container) -> tuple[str, str]:
    status, catalog = _request(container.base_url, "GET", "/v1/catalog")
    assert status == 200
    return str(catalog["bundleVersion"]), str(catalog["bundleDigest"])


def _refresh_child(container: _Container) -> _Container:
    """Resolve the sole live child after an autonomous terminal/replacement."""
    parent_pid, descendants = _container_pids(container.name)
    assert parent_pid == container.parent_pid
    assert len(descendants) == 1
    return _Container(
        container.name,
        container.image,
        container.base_url,
        container.state_dir,
        container.parent_pid,
        descendants[0],
    )


def _walk_request(container: _Container, task_id: str, lease_ms: int) -> dict[str, object]:
    version, digest = _bundle_identity(container)
    return {
        "schema": "MICRODUCK_SIM_TASK_V1",
        "taskId": task_id,
        "actionCode": "WALK_VELOCITY",
        "bundleVersion": version,
        "bundleDigest": digest,
        "parameters": {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
        "scenario": {"terrain": "flat", "seed": 7},
        "leaseMs": lease_ms,
        "requestedBy": "container-release-gate",
    }


def _stand_request(container: _Container, task_id: str) -> dict[str, object]:
    version, digest = _bundle_identity(container)
    return {
        "schema": "MICRODUCK_SIM_TASK_V1",
        "taskId": task_id,
        "actionCode": "STAND",
        "bundleVersion": version,
        "bundleDigest": digest,
        "parameters": {},
        "scenario": {"terrain": "flat", "seed": 7},
        "requestedBy": "container-release-gate",
    }


def _wait_task(
    container: _Container, task_id: str, expected: set[str], timeout: float = 15.0
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        status, latest = _request(
            container.base_url, "GET", f"/v1/tasks/{task_id}"
        )
        if status == 200 and latest.get("state") in expected:
            return latest
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not reach {expected}: {latest}")


def _wait_ready(container: _Container, expected: bool, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        status, latest = _request(container.base_url, "GET", "/v1/ready")
        if status == 200 and latest.get("ready") is expected:
            return
        time.sleep(0.01)
    raise AssertionError(f"readiness did not become {expected}: {latest}")


def _wait_event(
    container: _Container, task_id: str, event_type: str, timeout: float = 10.0
) -> None:
    deadline = time.monotonic() + timeout
    latest: dict[str, object] = {}
    while time.monotonic() < deadline:
        status, latest = _request(
            container.base_url,
            "GET",
            f"/v1/tasks/{task_id}/events?afterSequence=-1&pageSize=100",
        )
        if status == 200 and any(
            event.get("eventType") == event_type
            for event in latest.get("events", [])  # type: ignore[union-attr]
        ):
            return
        time.sleep(0.01)
    raise AssertionError(f"event {event_type} was not observed: {latest}")


def _assert_pidfd_dead(pidfd: int, timeout: float = 20.0) -> None:
    try:
        assert select.select([pidfd], [], [], timeout)[0] == [pidfd]
    finally:
        os.close(pidfd)


def _assert_proc_absent(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    path = Path(f"/proc/{pid}")
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not path.exists(), f"PID {pid} exited but was not reaped"


def _stop_and_assert_exact_reap(
    container: _Container, *, child_pidfd: int | None = None
) -> int:
    parent_pidfd = os.pidfd_open(container.parent_pid)
    owned_child_pidfd = (
        os.pidfd_open(container.child_pid) if child_pidfd is None else child_pidfd
    )
    stopped = _run("docker", "stop", "--timeout", "60", container.name)
    assert stopped.stdout.strip() == container.name
    _assert_pidfd_dead(owned_child_pidfd)
    _assert_pidfd_dead(parent_pidfd)
    _assert_proc_absent(container.child_pid)
    _assert_proc_absent(container.parent_pid)
    inspected = json.loads(_run("docker", "inspect", container.name).stdout)[0]
    assert inspected["State"]["Running"] is False
    return int(inspected["State"]["ExitCode"])


def _remove_container(container: _Container) -> None:
    _run("docker", "rm", "--force", container.name, check=False)
    _restore_state_ownership(container.image, container.state_dir)


def _remove_container_name(name: str) -> None:
    _run("docker", "rm", "--force", name, check=False)


def _restore_state_ownership(image: str, state_dir: Path) -> None:
    """Return bind-mount files created by UID 10001 to the pytest host user."""
    _run(
        "docker",
        "run",
        "--rm",
        "--user",
        "0:0",
        "--mount",
        f"type=bind,src={state_dir},dst=/state",
        "--entrypoint",
        "/bin/chown",
        image,
        "-R",
        f"{os.getuid()}:{os.getgid()}",
        "/state",
    )


def _capture_request(
    result: dict[str, object], key: str, *args: object, **kwargs: object
) -> None:
    try:
        result[key] = _request(*args, **kwargs)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - records a deliberately severed request
        result[key] = type(exc).__name__


def _hold_state_write_lock(container: _Container) -> tuple[subprocess.Popen[str], int]:
    """Hold SQLite's writer lock and return the exact added container PID."""
    before = set(_container_pids(container.name)[1])
    locker = subprocess.Popen(
        [
            "docker",
            "exec",
            "--interactive",
            "--user",
            "10001:10001",
            container.name,
            "/app/.venv/bin/python",
            "-P",
            "-c",
            (
                "import sqlite3,sys;"
                "connection=sqlite3.connect('/state/tasks.sqlite3');"
                "connection.execute('PRAGMA journal_mode=WAL');"
                "connection.execute('BEGIN IMMEDIATE');"
                "print('LOCKED',flush=True);"
                "sys.stdin.read()"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert locker.stdout is not None
    readable, _, _ = select.select([locker.stdout], [], [], 5.0)
    assert readable == [locker.stdout]
    assert locker.stdout.readline().strip() == "LOCKED"
    after = set(_container_pids(container.name)[1])
    added = after - before
    assert len(added) == 1
    return locker, added.pop()


def _host_copy_sources(dockerfile: str) -> set[str]:
    logical = dockerfile.replace("\\\n", " ")
    sources: set[str] = set()
    for raw_line in logical.splitlines():
        line = raw_line.strip()
        if not line.startswith("COPY "):
            continue
        parts = shlex.split(line)
        if any(part.startswith("--from=") for part in parts[1:]):
            continue
        operands = [part for part in parts[1:] if not part.startswith("--")]
        sources.update(operands[:-1])
    return sources


def test_docker_context_rejects_unknown_rom_python_module() -> None:
    """A broad ROM wildcard would silently ship an unrelated debug or secret module."""
    repository = Path(__file__).parents[1]
    policies = (
        repository / ".dockerignore",
        repository / "docker/rom-simulator/Dockerfile.dockerignore",
    )
    for policy_path in policies:
        policy = policy_path.read_text()
        for unknown in (
            "src/mjlab_microduck/rom/debug_secret.py",
            "src/mjlab_microduck/rom/untracked_secret.py",
        ):
            assert not _docker_context_includes(policy, unknown)


def test_docker_context_is_the_literal_runtime_inventory() -> None:
    """Adding a repository or synthetic path cannot silently expand build input."""
    repository = Path(__file__).parents[1]
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    representatives = {
        *tracked,
        *_REPOSITORY_FILES,
        ".env",
        "output/checkpoint.pt",
        "src/mjlab_microduck/robot/microduck/assets/body.stl",
        "src/mjlab_microduck/tasks/new_training.py",
        "src/mjlab_microduck/rom/debug_secret.py",
        "src/mjlab_microduck/rom/untracked_secret.py",
        "tests/secret_fixture.bin",
    }
    policies = (
        repository / ".dockerignore",
        repository / "docker/rom-simulator/Dockerfile.dockerignore",
    )

    for policy_path in policies:
        policy = policy_path.read_text()
        included = {
            path for path in representatives if _docker_context_includes(policy, path)
        }
        assert included == _REPOSITORY_FILES


def test_dockerfile_copies_only_literal_host_files() -> None:
    """A directory or wildcard COPY would defeat review of the image inventory."""
    dockerfile = (
        Path(__file__).parents[1] / "docker/rom-simulator/Dockerfile"
    ).read_text()
    assert _host_copy_sources(dockerfile) == _REPOSITORY_FILES


def test_container_metadata_declares_linux_signal_and_nonroot_contract() -> None:
    """Removing explicit stop/user metadata would make operator lifecycle ambiguous."""
    dockerfile = (
        Path(__file__).parents[1] / "docker/rom-simulator/Dockerfile"
    ).read_text()
    assert "USER 10001:10001" in dockerfile
    assert "STOPSIGNAL SIGTERM" in dockerfile
    assert "PYTHONPATH=" not in dockerfile
    assert "install -d -o 0 -g 0 -m 0755 /usr/local/libexec" in dockerfile
    entrypoint = (
        Path(__file__).parents[1] / "docker/rom-simulator/entrypoint.sh"
    ).read_text()
    assert "exec python -P /usr/local/libexec/microduck-rom-pid1.py" in entrypoint
    bootstrap = (
        Path(__file__).parents[1] / "docker/rom-simulator/pid1_bootstrap.py"
    ).read_text()
    assert bootstrap.index("signal.signal(signal.SIGTERM") < bootstrap.index(
        "from mjlab_microduck.rom.main import main"
    )
    assert bootstrap.index(".microduck-pid1-sigterm-ready") < bootstrap.index(
        "from mjlab_microduck.rom.main import main"
    )


def test_built_image_contains_only_literal_runtime_source_inventory(
    tmp_path: Path,
) -> None:
    """The release gate sets the image variable so final layers, not mocks, are audited."""
    image = os.environ.get("MICRODUCK_ROM_CONTAINER_TEST_IMAGE")
    if not image:
        pytest.skip("set MICRODUCK_ROM_CONTAINER_TEST_IMAGE after the Docker build")
    state = tmp_path / "state"
    shadow = state / "mjlab_microduck"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text(
        "raise RuntimeError('writable-state package shadow executed')\n"
    )
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--mount",
            f"type=bind,src={state},dst=/state",
            "--workdir",
            "/state",
            "--entrypoint",
            "/app/.venv/bin/python",
            image,
            "-P",
            "-c",
            (
                "import importlib.metadata,json,pathlib,re,sys;"
                "import mjlab_microduck;"
                "root=pathlib.Path('/app');"
                "files=sorted(p.relative_to(root).as_posix() "
                "for p in root.rglob('*') if p.is_file() "
                "and '.venv' not in p.relative_to(root).parts);"
                "dists=sorted({re.sub(r'[-_.]+','-',d.metadata['Name']).lower() "
                "for d in importlib.metadata.distributions()});"
                "entry=pathlib.Path('/usr/local/bin/rom-entrypoint');"
                "print(json.dumps({'appFiles':files,'distributions':dists,"
                "'entrypointMode':entry.stat().st_mode&0o777,"
                "'packageOrigin':pathlib.Path(mjlab_microduck.__file__).resolve()"
                ".as_posix(),'sysPath':sys.path},separators=(',',':')))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = json.loads(completed.stdout)
    assert frozenset(inventory["appFiles"]) == _IMAGE_RUNTIME_FILES
    assert frozenset(inventory["distributions"]) == _IMAGE_DISTRIBUTIONS
    assert inventory["entrypointMode"] == 0o755
    assert str(inventory["packageOrigin"]).startswith("/app/src/mjlab_microduck/")
    assert "/state" not in inventory["sysPath"]

    metadata = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    inspected = json.loads(metadata.stdout)[0]
    assert inspected["Os"] == "linux"
    assert inspected["Config"]["User"] == "10001:10001"
    assert inspected["Config"]["StopSignal"] == "SIGTERM"
    assert inspected["Config"]["Entrypoint"] == [
        "/usr/local/bin/rom-entrypoint"
    ]
    assert not any(
        item.startswith("PYTHONPATH=") for item in inspected["Config"]["Env"]
    )


def test_immediate_docker_stop_is_caught_before_application_import(
    tmp_path: Path,
) -> None:
    """The PID-1 bootstrap must own SIGTERM from the first runnable instruction."""
    image, bundle = _release_inputs(tmp_path)
    name, parent_pid, descendants = _launch_without_readiness_wait(
        image=image,
        bundle=bundle,
        state_dir=tmp_path / "state",
        suffix="immediate-stop",
    )
    parent_pidfd = os.pidfd_open(parent_pid)
    descendant_handles = [(pid, os.pidfd_open(pid)) for pid in descendants]
    try:
        stopped = _run("docker", "stop", "--timeout", "60", name)
        assert stopped.stdout.strip() == name
        _assert_pidfd_dead(parent_pidfd)
        _assert_proc_absent(parent_pid)
        for pid, pidfd in descendant_handles:
            _assert_pidfd_dead(pidfd)
            _assert_proc_absent(pid)
        inspected = json.loads(_run("docker", "inspect", name).stdout)[0]
        assert inspected["State"]["ExitCode"] == 0
    finally:
        _remove_container_name(name)
        _restore_state_ownership(image, tmp_path / "state")


def test_real_read_only_container_api_and_child_replacement_matrix(
    tmp_path: Path,
) -> None:
    """Exercise the release image and actual native child; mocks are not evidence."""
    image, bundle = _release_inputs(tmp_path)
    container = _launch_container(
        image=image, bundle=bundle, state_dir=tmp_path / "state", suffix="matrix"
    )
    try:
        unauthenticated = urllib.request.Request(container.base_url + "/v1/ready")
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(unauthenticated, timeout=5)
        assert denied.value.code == 401

        status, catalog = _request(container.base_url, "GET", "/v1/catalog")
        assert status == 200
        actions = {item["actionCode"]: item for item in catalog["actions"]}  # type: ignore[index]
        assert len(actions) == 15
        assert actions["STAND"]["availability"] == "AVAILABLE"
        assert actions["WALK_VELOCITY"]["availability"] == "AVAILABLE"
        assert actions["SPIN"]["availability"] == "UNAVAILABLE"

        parent_internal_pid = _namespace_pid(container.parent_pid)
        child_internal_pid = _namespace_pid(container.child_pid)
        parent_maps = _container_proc_maps(container.name, parent_internal_pid)
        child_maps = _container_proc_maps(container.name, child_internal_pid)
        assert "libmujoco" not in parent_maps
        assert "onnxruntime" not in parent_maps
        assert "libmujoco" in child_maps
        assert "onnxruntime" in child_maps

        stand_id = "1" * 32
        created_status, _created = _request(
            container.base_url,
            "POST",
            "/v1/tasks",
            _stand_request(container, stand_id),
        )
        assert created_status == 202
        stand = _wait_task(container, stand_id, {"SUCCEEDED"})
        assert stand["stopReason"] == "STAND_POSE_SETTLED"
        event_status, event_page = _request(
            container.base_url,
            "GET",
            f"/v1/tasks/{stand_id}/events?afterSequence=-1&pageSize=100",
        )
        assert event_status == 200
        assert event_page["events"][0]["sequence"] == 0  # type: ignore[index]

        walk_timeout_id = "2" * 32
        created_status, _created = _request(
            container.base_url,
            "POST",
            "/v1/tasks",
            _walk_request(container, walk_timeout_id, 200),
        )
        assert created_status == 202
        timed_out = _wait_task(container, walk_timeout_id, {"TIMED_OUT"})
        assert timed_out["stopReason"] == "LEASE_EXPIRED"

        spin_id = "3" * 32
        spin = _stand_request(container, spin_id) | {"actionCode": "SPIN"}
        rejected_status, rejected = _post_after_slot_resolution(
            container,
            "/v1/tasks",
            spin,
            expected_status=400,
        )
        assert rejected_status == 400
        assert rejected["code"] == "ACTION_UNAVAILABLE"

        oversized_status, oversized = _request(
            container.base_url,
            "POST",
            "/v1/tasks",
            raw=b"{" + b"x" * 65_536,
        )
        assert oversized_status == 413
        assert oversized["code"] == "REQUEST_BODY_TOO_LARGE"

        blocked_id = "4" * 32
        container = _refresh_child(container)
        created_status, _created = _request(
            container.base_url,
            "POST",
            "/v1/tasks",
            _walk_request(container, blocked_id, 5_000),
        )
        assert created_status == 202
        _wait_task(container, blocked_id, {"RUNNING"})
        _signal_container_pid(container, container.child_pid, signal.SIGSTOP)
        blocked_result: dict[str, object] = {}
        canceller = threading.Thread(
            target=_capture_request,
            args=(
                blocked_result,
                "cancel",
                container.base_url,
                "POST",
                f"/v1/tasks/{blocked_id}/cancel",
            ),
            daemon=True,
        )
        canceller.start()
        _wait_event(container, blocked_id, "TASK_CANCEL_REQUESTED")
        _signal_container_pid(container, container.child_pid, signal.SIGCONT)
        canceller.join(timeout=15)
        assert not canceller.is_alive()
        assert blocked_result["cancel"][0] == 200  # type: ignore[index]
        cancelled = _wait_task(container, blocked_id, {"CANCELLED"})
        assert cancelled["stopReason"] == "CANCELLED"

        killed_id = "5" * 32
        created_status, _created = _request(
            container.base_url,
            "POST",
            "/v1/tasks",
            _walk_request(container, killed_id, 5_000),
        )
        assert created_status == 202
        killed_pid = container.child_pid
        killed_pidfd = os.pidfd_open(killed_pid)
        _signal_container_pid(container, killed_pid, signal.SIGKILL)
        _assert_pidfd_dead(killed_pidfd)
        _assert_proc_absent(killed_pid)
        failed = _wait_task(container, killed_id, {"FAILED"})
        assert failed["stopReason"] == "RUNTIME_UNRESPONSIVE"

        fresh_id = "6" * 32
        created_status, _created = _request(
            container.base_url,
            "POST",
            "/v1/tasks",
            _walk_request(container, fresh_id, 5_000),
        )
        assert created_status == 202
        _parent_pid, descendants = _container_pids(container.name)
        assert len(descendants) == 1 and descendants[0] != killed_pid
        cancel_status, _cancelled = _request(
            container.base_url, "POST", f"/v1/tasks/{fresh_id}/cancel"
        )
        assert cancel_status == 200
        _wait_task(container, fresh_id, {"CANCELLED"})
        container = _Container(
            container.name,
            container.image,
            container.base_url,
            container.state_dir,
            container.parent_pid,
            descendants[0],
        )
        assert _stop_and_assert_exact_reap(container) == 0
    finally:
        try:
            _signal_container_pid(
                container,
                container.child_pid,
                signal.SIGCONT,
                check=False,
            )
        except (FileNotFoundError, StopIteration):
            pass
        _remove_container(container)


def test_container_shutdown_surfaces_blocked_terminal_callback_and_reaps_all(
    tmp_path: Path,
) -> None:
    """Shutdown failure stays visible while child and callback locker are contained."""
    image, bundle = _release_inputs(tmp_path)
    container = _launch_container(
        image=image,
        bundle=bundle,
        state_dir=tmp_path / "state",
        suffix="callback-delivery",
    )
    locker: subprocess.Popen[str] | None = None
    locker_pidfd: int | None = None
    locker_pid: int | None = None
    child_pidfd: int | None = None
    try:
        task_id = "b" * 32
        baseline_busy_threads = sum(
            "hrtimer_nanosleep" in item for item in _parent_wchans(container)
        )
        status, _body = _request(
            container.base_url,
            "POST",
            "/v1/tasks",
            _walk_request(container, task_id, 1_000),
        )
        assert status == 202
        _wait_task(container, task_id, {"RUNNING"})
        locker, locker_pid = _hold_state_write_lock(container)
        locker_pidfd = os.pidfd_open(locker_pid)
        child_pidfd = os.pidfd_open(container.child_pid)
        assert select.select([child_pidfd], [], [], 5.0)[0] == [child_pidfd]
        assert Path(f"/proc/{container.parent_pid}").exists()
        _wait_parent_wchan_count(
            container, "hrtimer_nanosleep", baseline_busy_threads + 1
        )
        exit_code = _stop_and_assert_exact_reap(
            container, child_pidfd=child_pidfd
        )
        child_pidfd = None
        assert exit_code == 70
        _assert_pidfd_dead(locker_pidfd)
        locker_pidfd = None
        _assert_proc_absent(locker_pid)
        locker.wait(timeout=10)
    finally:
        if child_pidfd is not None:
            os.close(child_pidfd)
        if locker_pidfd is not None:
            os.close(locker_pidfd)
        if locker is not None and locker.poll() is None:
            locker.kill()
            locker.wait(timeout=5)
        _remove_container(container)


@pytest.mark.parametrize(
    "phase", ["START", "RUNNING", "STOPPING", "QUARANTINED"]
)
def test_container_sigterm_reaps_exact_child_and_restart_reconciles_unknown(
    tmp_path: Path, phase: str
) -> None:
    """PID1 shutdown contains the exact child in every motion lifecycle phase."""
    image, bundle = _release_inputs(tmp_path)
    state_dir = tmp_path / "state"
    container = _launch_container(
        image=image,
        bundle=bundle,
        state_dir=state_dir,
        suffix=f"shutdown-{phase.lower()}",
    )
    task_id = {
        "START": "7",
        "RUNNING": "8",
        "STOPPING": "9",
        "QUARANTINED": "a",
    }[phase] * 32
    request_result: dict[str, object] = {}
    operation: threading.Thread | None = None
    child_pidfd: int | None = None
    try:
        if phase == "START":
            _signal_container_pid(container, container.child_pid, signal.SIGSTOP)
            operation = threading.Thread(
                target=_capture_request,
                args=(
                    request_result,
                    "create",
                    container.base_url,
                    "POST",
                    "/v1/tasks",
                    _walk_request(container, task_id, 5_000),
                ),
                daemon=True,
            )
            operation.start()
            _wait_task(container, task_id, {"VALIDATING"})
            _wait_ready(container, False)
        else:
            status, _body = _request(
                container.base_url,
                "POST",
                "/v1/tasks",
                _walk_request(container, task_id, 5_000),
            )
            assert status == 202
            _wait_task(container, task_id, {"RUNNING"})
            if phase == "STOPPING":
                _signal_container_pid(
                    container, container.child_pid, signal.SIGSTOP
                )
                operation = threading.Thread(
                    target=_capture_request,
                    args=(
                        request_result,
                        "cancel",
                        container.base_url,
                        "POST",
                        f"/v1/tasks/{task_id}/cancel",
                    ),
                    daemon=True,
                )
                operation.start()
                _wait_event(container, task_id, "TASK_CANCEL_REQUESTED")
            elif phase == "QUARANTINED":
                child_pidfd = os.pidfd_open(container.child_pid)
                _signal_container_pid(
                    container, container.child_pid, signal.SIGSTOP
                )
                operation = threading.Thread(
                    target=_capture_request,
                    args=(
                        request_result,
                        "command",
                        container.base_url,
                        "POST",
                        f"/v1/tasks/{task_id}/command",
                        {
                            "commandSequence": 1,
                            "parameters": {
                                "vxMps": 0.0,
                                "vyMps": 0.0,
                                "yawRateRadps": 0.0,
                            },
                            "leaseMs": 5_000,
                        },
                    ),
                    daemon=True,
                )
                operation.start()
                assert child_pidfd is not None
                assert select.select([child_pidfd], [], [], 30.0)[0] == [
                    child_pidfd
                ]

        assert _stop_and_assert_exact_reap(
            container, child_pidfd=child_pidfd
        ) == 0
        child_pidfd = None
        if operation is not None:
            operation.join(timeout=20)
            assert not operation.is_alive()

        restarted = _launch_container(
            image=image,
            bundle=bundle,
            state_dir=state_dir,
            suffix=f"restart-{phase.lower()}",
        )
        try:
            expected_state = (
                {"FAILED"}
                if phase in {"START", "STOPPING", "QUARANTINED"}
                else {"UNKNOWN"}
            )
            reconciled = _wait_task(restarted, task_id, expected_state)
            if expected_state == {"FAILED"}:
                assert reconciled["stopReason"] == "RUNTIME_UNRESPONSIVE"
            else:
                assert reconciled["stopReason"] is None
            event_status, event_page = _request(
                restarted.base_url,
                "GET",
                f"/v1/tasks/{task_id}/events?afterSequence=-1&pageSize=100",
            )
            assert event_status == 200
            expected_event = (
                "TASK_FAILED"
                if expected_state == {"FAILED"}
                else "TASK_INTERRUPTED"
            )
            assert event_page["events"][-1]["eventType"] == expected_event  # type: ignore[index]
            assert _stop_and_assert_exact_reap(restarted) == 0
        finally:
            _remove_container(restarted)
    finally:
        try:
            _signal_container_pid(
                container,
                container.child_pid,
                signal.SIGCONT,
                check=False,
            )
        except (FileNotFoundError, StopIteration):
            pass
        _remove_container(container)
