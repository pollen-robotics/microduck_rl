import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export_policy_rollout.py"


def _cli_module():
    spec = importlib.util.spec_from_file_location("export_policy_rollout", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arguments_use_approved_defaults(monkeypatch, tmp_path):
    module = _cli_module()
    policy = tmp_path / "policy.onnx"
    output = tmp_path / "rollout.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT_PATH), str(policy), "--output", str(output)],
    )

    args = module._arguments()

    assert args.policy == policy
    assert args.output == output
    assert args.duration == 4.0
    assert args.lin_vel_x == 0.30
    assert args.lin_vel_y == 0.0
    assert args.ang_vel_z == 0.0
    assert args.seed == 0


def test_missing_policy_returns_cli_error(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(tmp_path / "missing.onnx"),
            "--output",
            str(tmp_path / "rollout.npz"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "policy file does not exist" in result.stderr
