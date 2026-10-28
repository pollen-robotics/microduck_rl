"""Sample footsteps (dx, dy, dθ) vs velocity commands from a walking ONNX policy (FootstepNet groundwork).

  uv run scripts/footsteps_sample.py [--onnx velstand.onnx] [--num-envs 2048] [--T 12] [--out footsteps.npz]

Port of Sigmaban_playground's footsteps_sampler to the batched mjlab env. A touchdown = a foot's ground contact
turning on after >= --min-air control steps airborne while the other foot is (or was, one step ago) down; the footstep is the landing
foot's planar pose (site xy + yaw) expressed in the support foot's planar frame. Two regimes, half the envs each:
  steady: one random command held for the whole episode (repeatability of the step per command)
  switch: a new random command at every touchdown (what a step-by-step planner does: how fast the gait responds)
Each record stores the command active during the swing that just ended (`cmd`) and the one before it (`cmd_prev`).
Default policy = the deployed Hub velstand on the backlash model it was trained on, flat ground, no pushes.
"""

import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from mjlab.envs import ManagerBasedRlEnv

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_velstand_env_cfg import make_microduck_velstand_env_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default=None, help="walking ONNX (default: Hub pollen-robotics/microduck-policies velstand.onnx)")
ap.add_argument("--num-envs", type=int, default=2048)
ap.add_argument("--T", type=float, default=12.0, help="episode length (s)")
ap.add_argument("--warmup", type=float, default=2.0, help="touchdowns before this time are dropped (s)")
ap.add_argument("--min-air", type=int, default=2, help="control steps airborne before a contact counts as a touchdown")
ap.add_argument("--no-backlash", action="store_true")
ap.add_argument("--no-dr", action="store_true", help="drop the model DR events (friction, CoM, mass, encoder bias, ...)")
ap.add_argument("--repeat", type=int, default=1,
                help="R > 1: every command is given to R envs, all steady (multi-attractor / hysteresis check)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="footsteps.npz")
A = ap.parse_args()
DEV = "cuda:0"
N = A.num_envs
torch.manual_seed(A.seed)
np.random.seed(A.seed)

# Policy: the exported graph has a fixed batch of 1; relabel the batch dim so all envs run in one call (bit-exact).
model = onnx.load(A.onnx or hf_hub_download("pollen-robotics/microduck-policies", "velstand.onnx"))
for v in list(model.graph.input) + list(model.graph.output):
    v.type.tensor_type.shape.dim[0].dim_param = "N"
sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def policy(obs):
    return torch.as_tensor(sess.run(None, {"obs": obs.cpu().numpy().astype(np.float32)})[0], device=DEV)


cfg = make_microduck_velstand_env_cfg(play=True)
if not A.no_backlash:
    cfg = make_backlash_variant(cfg, MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG)
cfg.scene.num_envs = N
for k in ("topple_push", "push_robot"):
    cfg.events.pop(k, None)
if A.no_dr:
    for k in [k for k in cfg.events if k.startswith("randomize_") or k in ("foot_friction", "encoder_bias", "base_com")]:
        cfg.events.pop(k)
    for k in [k for k, c in cfg.curriculum.items() if c.params.get("event_name", "") not in ("", *cfg.events)]:
        cfg.curriculum.pop(k)  # curricula of the dropped DR events
cfg.curriculum.pop("prone_init_prob", None)
for k in list(cfg.terminations):
    if k != "nan_state":
        cfg.terminations.pop(k)
cfg.episode_length_s = A.T + 10.0
tw_cfg = cfg.commands["twist"]
tw_cfg.resampling_time_range = (1e4, 1e4)
tw_cfg.rel_standing_envs = 0.0
for k in ("head_pose", "body_pose"):
    if k in cfg.commands:
        cfg.commands[k].resampling_time_range = (1e4, 1e4)
RANGES = [tw_cfg.ranges.lin_vel_x, tw_cfg.ranges.lin_vel_y, tw_cfg.ranges.ang_vel_z]

env = ManagerBasedRlEnv(cfg=cfg, device=DEV)
prone = env.event_manager.get_term_cfg("random_prone_init").params
prone.update({k: 0.0 for k in prone if k.endswith("_prob")})  # every spawn standing
robot = env.scene["robot"]
twist = env.command_manager._terms["twist"]
contact = env.scene.sensors["feet_ground_contact"]
site_ids = [robot.site_names.index(n) for n in ("left_foot", "right_foot")]
dt = env.step_dt
lo = torch.tensor([r[0] for r in RANGES], device=DEV)
hi = torch.tensor([r[1] for r in RANGES], device=DEV)
switch_env = (torch.arange(N, device=DEV) >= N // 2) & (A.repeat == 1)


def sample_cmd(n):
    return lo + (hi - lo) * torch.rand(n, 3, device=DEV)


cmd = sample_cmd(N // A.repeat).repeat_interleave(A.repeat, 0)
cmd_prev = cmd.clone()


def apply_cmds():
    twist.vel_command_b[:] = cmd
    twist.is_standing_env[:] = False
    for name in ("head_pose", "body_pose"):
        t = env.command_manager._terms.get(name)
        for attr in ("command", "_raw", "_command"):
            v = getattr(t, attr, None) if t is not None else None
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                v.zero_()


def foot_planar():
    """(N, 2 feet, 3) planar pose [x, y, yaw] of each foot site, yaw of its forward axis FWD[foot]."""
    pos = robot.data.site_pos_w[:, site_ids, :2]
    q = robot.data.site_quat_w[:, site_ids]  # wxyz
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )  # (N, 2, 3, 3), columns = site axes in world
    f = torch.einsum("nfij,fj->nfi", R, FWD)
    return torch.cat([pos, torch.atan2(f[..., 1], f[..., 0]).unsqueeze(-1)], -1)


def wrap(a):
    return (a + torch.pi) % (2 * torch.pi) - torch.pi


obs, _ = env.reset()
apply_cmds()
obs, *_ = env.step(torch.zeros(N, robot.num_actuators, device=DEV))
apply_cmds()

# Each foot site's local axis that points along the trunk's forward direction in the standing spawn pose.
# Expressed per env in the site frame, THEN averaged (spawn yaw is random: world-frame averages are meaningless).
def quat_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


trunk_fwd = quat_mat(robot.data.root_link_quat_w)[:, :, 0]  # (N, 3)
local = torch.einsum("nfji,nj->nfi", quat_mat(robot.data.site_quat_w[:, site_ids]), trunk_fwd).mean(0)  # (2, 3)
FWD = torch.zeros(2, 3, device=DEV)
for f in range(2):
    i = int(local[f].abs().argmax())
    FWD[f, i] = torch.sign(local[f, i])
print("foot forward axes (site frame):", FWD.tolist(), "alignment", local.tolist())

fell = torch.zeros(N, dtype=torch.bool, device=DEV)
was_down = contact.data.found > 0
air = torch.zeros(N, 2, dtype=torch.long, device=DEV)
last_td_t = torch.full((N,), -1.0, device=DEV)
rec = {k: [] for k in ("env", "t", "landing", "footstep", "cmd", "cmd_prev", "since_last", "switch")}

with torch.no_grad():
    for step in range(int(A.T / dt)):
        obs, *_ = env.step(policy(obs["actor"]))
        apply_cmds()
        t = (step + 1) * dt
        g = robot.data.projected_gravity_b
        fell |= g[:, 2] > -np.cos(np.radians(40))
        down = contact.data.found > 0
        # landing foot; the other foot is the support if it was down this step OR the previous one (the gait often
        # lifts the support on the very control step the swing foot lands: requiring overlap dropped ~half the steps)
        td = down & ~was_down & (air >= A.min_air) & (down | was_down).flip(-1)
        air = torch.where(down, torch.zeros_like(air), air + 1)
        hit = td.any(-1) & ~fell & ~(td.all(-1))
        if hit.any():
            P = foot_planar()
            ids = hit.nonzero().squeeze(-1)
            land = td[ids].float().argmax(-1)  # 0 = left landed, 1 = right landed
            sup = 1 - land
            Pl = P[ids, land]
            Ps = P[ids, sup]
            dxy = Pl[:, :2] - Ps[:, :2]
            c, s = torch.cos(Ps[:, 2]), torch.sin(Ps[:, 2])
            fs = torch.stack([c * dxy[:, 0] + s * dxy[:, 1], -s * dxy[:, 0] + c * dxy[:, 1], wrap(Pl[:, 2] - Ps[:, 2])], -1)
            if t >= A.warmup:
                rec["env"].append(ids.cpu())
                rec["t"].append(torch.full((len(ids),), t))
                rec["landing"].append(land.cpu())
                rec["footstep"].append(fs.cpu())
                rec["cmd"].append(cmd[ids].cpu())
                rec["cmd_prev"].append(cmd_prev[ids].cpu())
                rec["since_last"].append((t - last_td_t[ids]).cpu())
                rec["switch"].append(switch_env[ids].cpu())
            last_td_t[ids] = t
            sw = ids[switch_env[ids]]
            cmd_prev[sw] = cmd[sw]
            cmd[sw] = sample_cmd(len(sw))
            apply_cmds()
        was_down = down
        if step % 100 == 0:
            n = sum(len(e) for e in rec["env"])
            print(f"t={t:5.1f}s  touchdowns={n}  fell={fell.float().mean().item() * 100:.1f}%", flush=True)

out = {k: torch.cat(v).numpy() for k, v in rec.items()}
# Drop touchdowns from envs that fell at any point (the record before the fall can be a stumble).
keep = ~fell.cpu().numpy()[out["env"]]
out = {k: v[keep] for k, v in out.items()}
out["ranges"] = np.array(RANGES)
out["dt"] = dt
np.savez(A.out, **out)
print(f"saved {len(out['env'])} touchdowns from {int((~fell).sum())}/{N} upright envs → {A.out}")
