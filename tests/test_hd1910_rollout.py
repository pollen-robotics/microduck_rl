"""Exercise the actual Feetech BAM control law, including episode resets."""

from dataclasses import replace

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp needs CUDA")
def test_hd1910_rollout_and_partial_reset():
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck-HD1910")
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        obs, _ = env.reset(seed=7)
        assert obs["actor"].shape == (4, 61)
        assert env.action_manager.total_action_dim == 14
        robot = env.scene["robot"]
        actuator = robot.actuators[0]
        command = actuator.get_command(robot.data)
        # A large old target must be discarded only in the world being reset.
        actuator._bam_model.actuator.q_target_smooth = command.pos.clone() + 2.0
        actuator.reset(torch.tensor([0], device=env.device))
        actuator.compute(replace(command, position_target=command.pos.clone()))
        history = actuator._bam_model.actuator.q_target_smooth
        torch.testing.assert_close(history[0], command.pos[0])
        assert torch.all(history[1:] > command.pos[1:])
        env.reset(seed=7)
        for step in range(20):
            action = torch.zeros((4, 14), device=env.device)
            action[1, 0] = 0.1
            obs, reward, _, _, _ = env.step(action)
            assert torch.isfinite(obs["actor"]).all()
            assert torch.isfinite(reward).all()
            assert torch.isfinite(env.scene["robot"].data.joint_pos).all()
            if step == 9:
                # Reset one world while the others continue their control histories.
                env.reset(env_ids=torch.tensor([0], device=env.device))
    finally:
        env.close()
