"""Export a trained microduck FootstepNet (CrossQ actor) to ONNX, deterministic action.

  uv run python export_onnx.py [MODEL.zip | RUN_DIR] --provisional-limits FWD BWD DY DTHETA_DEG [--out planner.onnx]

Eval-mode CrossQ actor = BatchRenorm(running stats) → [Dense → ReLU → BatchRenorm] x 2 → Dense(mean) → tanh, scaled to
the action box (symmetric: ± max_dx_forward, max_dy, max_dtheta). Each BatchRenorm is an affine map, folded into the
next Dense, so the graph is 3 Gemm + 2 Relu + Tanh + Mul. Input "obs" (N, 8) = FootstepsPlanningEnv.get_observation()
(right-support convention), output "step" (N, 3) = neutral-relative step BEFORE the env's ellipsoid clip and left
mirroring. Metadata carries the limits and the env geometry; checked against model.predict on random observations.
"""

import argparse
import json
import math
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default=None)
ap.add_argument("--provisional-limits", type=float, nargs=4, metavar=("FWD", "BWD", "DY", "DTHETA_DEG"))
ap.add_argument("--out", default="footstepnet.onnx")
A = ap.parse_args()
os.environ["JAX_PLATFORMS"] = "cpu"  # GPU jax runs float32 matmuls in TF32: ~4e-3 off; the check needs exact

import microduck_env  # noqa: E402
from models import find_model  # noqa: E402
from sbx import CrossQ  # noqa: E402

if A.provisional_limits:
    microduck_env.set_limits(*A.provisional_limits)
options = microduck_env.contract.footstepnet_options()

path = find_model(A.model)
model = CrossQ.load(path)
st = model.policy.actor_state
p, bs = st.params, st.batch_stats
EPS = 1e-3  # sbx BatchRenorm default epsilon


def brn(i):
    """BatchRenorm_i in eval mode as y = x * a + b."""
    a = np.asarray(p[f"BatchRenorm_{i}"]["scale"]) / np.sqrt(np.asarray(bs[f"BatchRenorm_{i}"]["var"]) + EPS)
    b = np.asarray(p[f"BatchRenorm_{i}"]["bias"]) - np.asarray(bs[f"BatchRenorm_{i}"]["mean"]) * a
    return a, b


def dense(name):
    return np.asarray(p[name]["kernel"], dtype=np.float64), np.asarray(p[name]["bias"], dtype=np.float64)


# fold BatchRenorm_i (on the input of Dense_i) into Dense_i: (x*a + b) W + c = x (a[:,None] W) + (b W + c)
layers = []
for i, name in enumerate(("Dense_0", "Dense_1", "Dense_2")):  # Dense_2 = mean head (Dense_3 = log_std, unused)
    a, b = brn(i)
    W, c = dense(name)
    layers.append(((a[:, None] * W).astype(np.float32), (b @ W + c).astype(np.float32)))

high = np.asarray(model.action_space.high, dtype=np.float32)
assert np.allclose(high, -np.asarray(model.action_space.low)), "action box must be symmetric"

nodes, inits = [], []
x = "obs"
for i, (W, c) in enumerate(layers):
    inits += [numpy_helper.from_array(W, f"W{i}"), numpy_helper.from_array(c, f"b{i}")]
    out = f"h{i}" if i < 2 else "mean"
    nodes.append(helper.make_node("Gemm", [x, f"W{i}", f"b{i}"], [out]))
    if i < 2:
        nodes.append(helper.make_node("Relu", [out], [f"r{i}"]))
        x = f"r{i}"
inits.append(numpy_helper.from_array(high, "high"))
nodes += [helper.make_node("Tanh", ["mean"], ["t"]), helper.make_node("Mul", ["t", "high"], ["step"])]
graph = helper.make_graph(
    nodes, "footstepnet", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["N", 8])],
    [helper.make_tensor_value_info("step", TensorProto.FLOAT, ["N", 3])], inits)
onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx_model.ir_version = 8
meta = {
    "source": str(path),
    "env_id": microduck_env.ENV_ID,
    "options": json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in options.items()}),
    "obstacle_x": str(microduck_env.OBSTACLE_X),
    "area": str(microduck_env.AREA),
    "obs": "support->target [x, y, cos, sin] (left support: y, sin negated), target foot is support (0/1), "
           "support->obstacle [x, y] (left: y negated), obstacle radius",
    "step": "neutral-relative (dx, dy, dtheta), right-support convention; apply ellipsoid clip, mirror for left",
}
for k, v in meta.items():
    onnx_model.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, A.out)

# check against the trained model
rng = np.random.default_rng(0)
lo, hi = model.observation_space.low, model.observation_space.high
obs = rng.uniform(lo, hi, size=(512, 8)).astype(np.float32)
obs[:, :2] = rng.uniform(-1.5, 1.5, size=(512, 2))
th = rng.uniform(-np.pi, np.pi, 512)
obs[:, 2], obs[:, 3], obs[:, 4] = np.cos(th), np.sin(th), rng.integers(0, 2, 512)
ref, _ = model.predict(obs, deterministic=True)
got = ort.InferenceSession(A.out).run(None, {"obs": obs})[0]
err = np.abs(ref - got).max()
print(f"model: {path}\n→ {A.out}   max |onnx - predict| = {err:.2e} (action high {high})")
assert err < 1e-4, err

# check the runtime planner (src/mjlab_microduck/footstep_planner.py) against the gym env rolled out with the model
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "footstep_planner", Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "footstep_planner.py")
fp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp)
planner = fp.FootstepPlanner(A.out)
env = microduck_env.MicroduckFootstepsEnv()
worst, n_steps, agree = 0.0, 0, 0
for ep in range(50):
    obs, _ = env.reset(seed=1000 + ep)
    sim = env.simulator
    start, foot = np.array(sim.support_pose()), sim.support_foot
    goal, goal_foot = np.array(env.target_foot_pose), env.target_support_foot
    obstacle = (*env.options["obstacle_position"], env.obstacle_radius)
    plan, reached = planner.plan(start, foot, goal, goal_foot, obstacle, max_steps=microduck_env.MAX_EPISODE_STEPS)
    env_poses, done = [], False
    for _ in range(microduck_env.MAX_EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        env_poses.append(np.array(sim.support_pose()))
        if done:
            break
    m = min(len(plan), len(env_poses))
    for (p, _), q in zip(plan[:m], env_poses[:m]):
        worst = max(worst, np.abs(p[:2] - q[:2]).max(), abs(math.remainder(p[2] - q[2], 2 * math.pi)))
    n_steps += m
    agree += (len(plan) == len(env_poses)) and (reached == done)
print(f"planner vs gym env: {agree}/50 episodes same length and outcome, max pose diff {worst:.1e} over {n_steps} steps")
