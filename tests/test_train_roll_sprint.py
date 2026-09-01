from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_roll_sprint.py"
SPEC = importlib.util.spec_from_file_location("train_roll_sprint", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_short_burst_save_interval_is_configurable(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--num-envs",
            "4096",
            "--iterations",
            "10",
            "--save-interval",
            "1",
            "--seed",
            "7",
            "--learning-rate",
            "5e-6",
            "--exploration-std",
            "0.35",
            "--reset-optimizer",
            "--run-name",
            "short_burst",
        ],
    )

    args = MODULE._parse_args()

    assert args.num_envs == 4096
    assert args.iterations == 10
    assert args.save_interval == 1
    assert args.seed == 7
    assert args.learning_rate == 5e-6
    assert args.exploration_std == 0.35
    assert args.reset_optimizer
    assert args.run_name == "short_burst"


def test_seed_defaults_to_mjlab_default_and_accepts_zero(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH)])
    defaults = MODULE._parse_args()

    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--seed", "0"])
    zero_seed = MODULE._parse_args()

    assert defaults.seed == 42
    assert zero_seed.seed == 0


def test_negative_seed_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--seed", "-1"])

    with pytest.raises(SystemExit):
        MODULE._parse_args()


@pytest.mark.parametrize("value", ["0", "-1e-6"])
def test_nonpositive_learning_rate_is_rejected(monkeypatch, value: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT_PATH), "--learning-rate", value],
    )

    with pytest.raises(SystemExit):
        MODULE._parse_args()


def test_training_command_forwards_seed_exactly_once(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint")
    train_exe = tmp_path / ".venv" / (
        "Scripts/train.exe" if MODULE.os.name == "nt" else "bin/train"
    )
    train_exe.parent.mkdir(parents=True)
    train_exe.write_bytes(b"executable")
    captured: dict[str, object] = {}
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path)
    staged: dict[str, object] = {}

    def capture_stage(*args, **kwargs) -> None:
        staged["args"] = args
        staged["kwargs"] = kwargs

    monkeypatch.setattr(MODULE, "stage_checkpoint", capture_stage)

    def capture_call(command, **kwargs) -> int:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(MODULE.subprocess, "call", capture_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--source-checkpoint",
            str(source),
            "--seed",
            "31415",
            "--learning-rate",
            "5e-6",
        ],
    )

    assert MODULE.main() == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == str(train_exe)
    assert command.count("--agent.seed") == 1
    seed_index = command.index("--agent.seed")
    assert command[seed_index : seed_index + 2] == ["--agent.seed", "31415"]
    assert command.count("--agent.algorithm.learning-rate") == 1
    rate_index = command.index("--agent.algorithm.learning-rate")
    assert command[rate_index : rate_index + 2] == [
        "--agent.algorithm.learning-rate",
        "5e-06",
    ]
    assert staged["kwargs"] == {
        "exploration_std": None,
        "reset_optimizer": False,
        "optimizer_learning_rate": 5e-6,
        "neutralize_direction_cue": False,
    }


def test_reverse_skill_staging_can_reopen_exploration_and_reset_adam(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "staged.pt"
    torch.save(
        {
            "actor_state_dict": {
                "distribution.std_param": torch.tensor([0.01, 0.12, 0.18]),
                "mean.weight": torch.tensor([[1.0]]),
            },
            "critic_state_dict": {"value.weight": torch.tensor([[2.0]])},
            "optimizer_state_dict": {
                "state": {0: {"step": torch.tensor(1500.0)}},
                "param_groups": [{"lr": 1.0e-5, "params": [0]}],
            },
            "iter": 1500,
            "infos": {"env_state": {"common_step_counter": 36000}},
        },
        source,
    )

    MODULE.stage_checkpoint(
        source,
        destination,
        exploration_std=0.35,
        reset_optimizer=True,
        optimizer_learning_rate=2.5e-5,
    )

    staged = torch.load(destination, map_location="cpu", weights_only=False)
    assert torch.equal(
        staged["actor_state_dict"]["mean.weight"], torch.tensor([[1.0]])
    )
    assert torch.equal(
        staged["critic_state_dict"]["value.weight"], torch.tensor([[2.0]])
    )
    assert torch.equal(
        staged["actor_state_dict"]["distribution.std_param"],
        torch.full((3,), 0.35),
    )
    assert staged["optimizer_state_dict"]["state"] == {}
    assert staged["optimizer_state_dict"]["param_groups"][0]["lr"] == 2.5e-5
    assert staged["iter"] == 0
    assert staged["infos"]["env_state"]["common_step_counter"] == 0


def test_reverse_skill_staging_neutralizes_only_new_direction_cue_column(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "staged.pt"
    first_layer = torch.arange(4 * 61, dtype=torch.float32).reshape(4, 61)
    torch.save(
        {
            "actor_state_dict": {
                "mlp.0.weight": first_layer.clone(),
                "distribution.std_param": torch.tensor([0.2]),
            },
            "critic_state_dict": {
                "mlp.0.weight": torch.arange(2 * 90, dtype=torch.float32).reshape(2, 90),
                "value.weight": torch.tensor([[2.0]]),
            },
            "optimizer_state_dict": {"state": {}, "param_groups": []},
        },
        source,
    )

    MODULE.stage_checkpoint(source, destination, neutralize_direction_cue=True)

    staged = torch.load(destination, map_location="cpu", weights_only=False)
    weight = staged["actor_state_dict"]["mlp.0.weight"]
    assert torch.count_nonzero(weight[:, MODULE.ACTOR_DIRECTION_CUE_OBS_INDEX]) == 0
    assert torch.equal(
        weight[:, MODULE.ACTOR_DIRECTION_CUE_OBS_INDEX - 1], first_layer[:, 54]
    )
    assert torch.equal(
        weight[:, MODULE.ACTOR_DIRECTION_CUE_OBS_INDEX + 1], first_layer[:, 56]
    )
    critic_weight = staged["critic_state_dict"]["mlp.0.weight"]
    assert torch.count_nonzero(
        critic_weight[:, MODULE.CRITIC_DIRECTION_CUE_OBS_INDEX]
    ) == 0
    assert torch.equal(
        critic_weight[:, MODULE.CRITIC_DIRECTION_CUE_OBS_INDEX - 1],
        torch.arange(2, dtype=torch.float32) * 90 + 67,
    )
    assert torch.equal(
        critic_weight[:, MODULE.CRITIC_DIRECTION_CUE_OBS_INDEX + 1],
        torch.arange(2, dtype=torch.float32) * 90 + 69,
    )
