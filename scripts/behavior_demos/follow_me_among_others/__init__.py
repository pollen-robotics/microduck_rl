"""Follow-me-among-others: a CPU MuJoCo behavior demo.

Drives the stock exported walking policy with a geometric twist controller so
the robot can search a moving crowd for a requested shirt color, follow that
person, and stop. No training, no new weights, no GPU.

``camera`` and ``run_demo`` import ``mujoco``; ``crowd`` and ``metrics`` are
pure Python + numpy so the state machine, gating and acceptance thresholds can
be tested without a model or a policy.
"""
