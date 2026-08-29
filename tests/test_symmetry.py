import torch
from tensordict import TensorDict

from mjlab_microduck.tasks.symmetry import microduck_vel_symmetry


def test_growbot_symmetry_preserves_67d_observations_and_swaps_elbows():
    actor = torch.arange(67, dtype=torch.float32).unsqueeze(0)
    critic = torch.arange(80, dtype=torch.float32).unsqueeze(0)
    actions = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    obs = TensorDict({"actor": actor, "critic": critic}, batch_size=[1])

    mirrored_obs, mirrored_actions = microduck_vel_symmetry(None, obs, actions)

    assert mirrored_obs["actor"].shape == (2, 67)
    assert mirrored_obs["critic"].shape == (2, 80)
    assert mirrored_actions.shape == (2, 16)
    assert mirrored_actions[1, 14].item() == actions[0, 15].item()
    assert mirrored_actions[1, 15].item() == actions[0, 14].item()
    # Elbow values occupy the final two entries of each joint block.
    assert mirrored_obs["actor"][1, 6 + 14].item() == actor[0, 6 + 15].item()
    assert mirrored_obs["actor"][1, 6 + 15].item() == actor[0, 6 + 14].item()


def test_microduck_symmetry_keeps_legacy_14_joint_shape():
    actor = torch.zeros((2, 61))
    critic = torch.zeros((2, 74))
    actions = torch.zeros((2, 14))
    obs = TensorDict({"actor": actor, "critic": critic}, batch_size=[2])

    mirrored_obs, mirrored_actions = microduck_vel_symmetry(None, obs, actions)

    assert mirrored_obs["actor"].shape == (4, 61)
    assert mirrored_actions.shape == (4, 14)
