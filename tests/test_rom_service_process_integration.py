"""Structural integration gates for process-owned ROM task execution."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from mjlab_microduck.rom.process_service import SimulatorTaskService

ROM_ROOT = Path(__file__).parents[1] / "src" / "mjlab_microduck" / "rom"


def test_public_service_has_no_parent_runtime_ownership_symbols():
    source = (ROM_ROOT / "service.py").read_text()
    implementation = (ROM_ROOT / "process_service.py").read_text()

    assert "_RuntimeDispatcher" not in source + implementation
    assert "_RuntimeOperation" not in source + implementation
    assert "_StartLifecycle" not in source + implementation
    assert "RuntimeHandle" not in source + implementation
    assert "SimulationRuntime" not in source + implementation


def test_parent_rom_modules_do_not_import_native_runtime_libraries():
    violations = []
    for name in (
        "service.py",
        "process_service.py",
        "api.py",
        "main.py",
        "process_supervisor.py",
    ):
        path = ROM_ROOT / name
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".", 1)[0]}
            else:
                continue
            if names & {"mujoco", "onnxruntime"}:
                violations.append(path.name)
    assert violations == []


def test_service_requires_a_supervisor_factory_not_a_runtime_handle():
    signature = inspect.signature(SimulatorTaskService)

    assert "supervisor_factory" in signature.parameters
    assert "runtime" not in signature.parameters


def test_main_composes_the_process_supervisor_without_native_runtime_imports():
    source = (ROM_ROOT / "main.py").read_text()

    assert "RuntimeProcessSupervisor" in source
    assert "MicroduckMujocoRuntime" not in source
    assert "from .runtime import" not in source
