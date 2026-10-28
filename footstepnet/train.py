"""Train FootstepNet (CrossQ, sbx) on microduck's footstep contract — mirrors footstepnet_envs/train_sbx.py.

  uv run python train.py [--provisional-limits FWD BWD DY DTHETA_DEG] [rl_zoo3 train args...]

Without --provisional-limits the limits come from src/mjlab_microduck/footsteps.py, which refuses until they are
measured on the trained footstep policy. Extra args go to rl_zoo3 (e.g. -n 20000 for a pipeline check).
"""

import sys

import rl_zoo3
import rl_zoo3.train
from sbx import CrossQ, TD3

import microduck_env

args = sys.argv[1:]
if "--provisional-limits" in args:
    i = args.index("--provisional-limits")
    microduck_env.set_limits(*map(float, args[i + 1:i + 5]))
    del args[i:i + 5]
print("footstep limits:", microduck_env.contract.footstepnet_options())

rl_zoo3.ALGOS["crossq"] = CrossQ
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

# Each env subprocess re-imports this module via --gym-packages, which re-applies the provisional limits only in
# THIS process: run with --vec-env dummy (default) so all envs live here.
sys.argv = ["train", "--algo", "crossq", "--env", microduck_env.ENV_ID, "--conf", "hyperparams.yml",
            "--gym-packages", "microduck_env", *args]
rl_zoo3.train.train()
