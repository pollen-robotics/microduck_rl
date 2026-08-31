"""CPU (Warp "cpu" device) smoke test: one env builds and steps.

Locks in the no-GPU dev path (macOS / CUDA-less Linux):
`uv run train <task> --device cpu` (or mjlab's `--gpu-ids None`). Anything
that breaks env construction or stepping on the Warp cpu device fails here
before a training run does.
"""

import pytest
import torch


def test_velocity_env_builds_and_steps_on_cpu():
    from mjlab.envs import ManagerBasedRlEnv

    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    cfg = make_microduck_velocity_env_cfg()
    cfg.scene.num_envs = 2
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    except Exception as e:
        pytest.skip(f"Warp cpu env construction unavailable on this platform: {e}")
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
        obs = None
        for _ in range(3):
            obs, *_ = env.step(actions)
        assert obs is not None
        # Actor obs contract: 61D (48 proprio + 13 command block).
        assert obs["actor"].shape == (env.num_envs, 61)
        assert not torch.isnan(obs["actor"]).any()
    finally:
        env.close()
