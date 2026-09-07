"""Build a 61->14 ONNX that outputs a CONSTANT action: a static pose for robotd.

robotd applies `targets = DEFAULT_POSITION + scale * onnx_output` for whatever
network is loaded, so a network that ignores its input and returns a fixed
offset IS a held pose -- delivered through the daemon's own gain, scale, limits
and safety, with no rebuild and no raw bus writes. Load it into a slot,
trigger it, and the robot holds the pose.

Offsets are radians from HOME_FRAME in the 14-joint policy order, with the
right leg's sagittal joints MIRRORED (negated), exactly as the sim's action
convention. Metadata is copied from a reference export so robotd's load-time
validation sees what it expects.

    uv run python scripts/make_pose_onnx.py --from-npz stable_home_best.npz \
        --ref exports/hop_k3344_59yiy9h6.onnx --out exports/pose_stable_home.onnx
"""
from __future__ import annotations
import argparse
import numpy as np, onnx, torch

L_HIP, L_KNEE, L_ANKLE, NECK, HEAD = 2, 3, 4, 5, 6
R_HIP, R_KNEE, R_ANKLE = 11, 12, 13
MIRROR = -1.0


class ConstPose(torch.nn.Module):
    def __init__(self, a):
        super().__init__(); self.register_buffer("a", torch.tensor(a, dtype=torch.float32).reshape(1, 14))
    def forward(self, obs):
        # Depend on obs so the input is kept in the graph, but contribute zero.
        return self.a + 0.0 * obs[:, :1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-npz", help="stable_home_best.npz (5 sagittal offsets)")
    ap.add_argument("--ref", default="exports/hop_k3344_59yiy9h6.onnx")
    ap.add_argument("--out", default="exports/pose_stable_home.onnx")
    ap.add_argument("--zero", action="store_true", help="HOME itself (all-zero offsets)")
    ap.add_argument("--sagittal", nargs=5, type=float, metavar=("HIP", "KNEE", "ANKLE", "NECK", "HEAD"),
                    help="Explicit sagittal offsets in radians (legs mirrored), for hardware trimming.")
    args = ap.parse_args()

    a = np.zeros(14, np.float32)
    if args.sagittal:
        p = np.array(args.sagittal, np.float32)
        a[L_HIP], a[L_KNEE], a[L_ANKLE], a[NECK], a[HEAD] = p[0], p[1], p[2], p[3], p[4]
        a[R_HIP], a[R_KNEE], a[R_ANKLE] = MIRROR*p[0], MIRROR*p[1], MIRROR*p[2]
    elif not args.zero:
        d = np.load(args.from_npz, allow_pickle=True); p = d["params"].astype(np.float32)
        a[L_HIP], a[L_KNEE], a[L_ANKLE], a[NECK], a[HEAD] = p[0], p[1], p[2], p[3], p[4]
        a[R_HIP], a[R_KNEE], a[R_ANKLE] = MIRROR*p[0], MIRROR*p[1], MIRROR*p[2]
    m = ConstPose(a).eval()
    torch.onnx.export(m, torch.zeros(1, 61), args.out, input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    ref = onnx.load(args.ref); out = onnx.load(args.out)
    del out.metadata_props[:]
    for prop in ref.metadata_props:
        e = out.metadata_props.add(); e.key, e.value = prop.key, prop.value
    onnx.save(out, args.out)
    import onnxruntime as ort
    s = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    y = s.run(None, {"obs": np.random.randn(1, 61).astype(np.float32)})[0][0]
    names = dict(kv.split("=", 1) for kv in []) if False else None
    jn = next(p.value for p in ref.metadata_props if p.key == "joint_names").split(",")
    print(f"RESULT wrote {args.out}: {s.get_inputs()[0].shape} -> {s.get_outputs()[0].shape}")
    for n, v in zip(jn, y):
        if abs(v) > 1e-6: print(f"RESULT   {n:16s} {v:+.4f} rad ({np.degrees(v):+.1f} deg)")


if __name__ == "__main__":
    main()
