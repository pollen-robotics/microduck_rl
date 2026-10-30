"""Reproduce the auxiliary-actuator observation failure with a released policy.

Headless XML position control, using only the pre-existing PolicyInference API.
The extra world-mounted hinge is a routing fixture, not a stock beak model.
See docs/independent-mouth-replay.md for the pinned policy and baseline commands.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime

REPO = Path(__file__).resolve().parents[1]
POLICY_SHA256 = "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"


def probe(ip, policy_path, auxiliary):
    case = {"auxiliary": auxiliary, "passed": False}
    try:
        spec = mujoco.MjSpec.from_file(str(REPO / ip.MICRODUCK_XML))
        if auxiliary:
            for key in list(spec.keys):
                spec.delete(key)
            body = spec.worldbody.add_body(name="routing_fixture", pos=[2, 0, 1])
            body.add_joint(
                name="mouth",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                limited=True,
                range=[-0.4, 0.4],
            )
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                mass=0.001,
                contype=0,
                conaffinity=0,
            )
            actuator = spec.add_actuator(
                name="mouth_motor", target="mouth", trntype=mujoco.mjtTrn.mjTRN_JOINT
            )
            actuator.set_to_position(kp=0.55)
        model = spec.compile()
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)
        case["model_actuators"] = model.nu
        # The unchanged scene's first fourteen actuators retain their XML order;
        # the fixture above appends one actuator without reordering those joints.
        data.qpos[model.jnt_qposadr[model.actuator_trnid[:14, 0]]] = ip.DEFAULT_POSE
        adr = int(model.joint("trunk_base_freejoint").qposadr[0])
        data.qpos[adr : adr + 7] = [0, 0, 0.125, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        policy = ip.PolicyInference(
            model,
            data,
            walking_onnx_path=str(policy_path),
            new_cmd_obs=True,
            use_projected_gravity=True,
        )
        case["phase"] = "observations"
        obs = policy.get_observations()
        case["phase"] = "inference_and_targets"
        action = policy.infer()
        policy.apply_action(action)
        case["phase"] = "physics"
        for _ in range(10):
            mujoco.mj_step(model, data)
        case.update(
            observations=list(obs.shape),
            actions=list(action.shape),
            physics_steps=10,
            warnings=data.warning.number.tolist(),
            finite=all(
                np.isfinite(x).all()
                for x in (obs, action, data.ctrl, data.qpos, data.qvel)
            ),
        )
        case["passed"] = bool(
            obs.shape == (61,)
            and action.shape == (14,)
            and case["finite"]
            and not any(case["warnings"])
        )
    except Exception as exc:
        case.update(error_type=type(exc).__name__, error=str(exc))
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--inference-script", type=Path, default=REPO / "scripts/infer_policy.py"
    )
    args = parser.parse_args()
    policy_hash = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    if policy_hash != POLICY_SHA256:
        parser.error(
            "Policy hash does not match the release pinned in the reproduction guide"
        )
    spec = importlib.util.spec_from_file_location(
        "replay_under_test", args.inference_script
    )
    ip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ip)
    # Keep normal replay chatter out of the machine-readable result.
    with contextlib.redirect_stdout(io.StringIO()):
        cases = [probe(ip, args.policy, auxiliary) for auxiliary in (False, True)]
    print(
        json.dumps(
            {
                "inference_sha256": hashlib.sha256(
                    args.inference_script.read_bytes()
                ).hexdigest(),
                "policy_sha256": policy_hash,
                "versions": {
                    "mujoco": mujoco.__version__,
                    "numpy": np.__version__,
                    "onnxruntime": onnxruntime.__version__,
                },
                "cases": cases,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if all(case["passed"] for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
