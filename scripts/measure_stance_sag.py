"""Measure the real robot's joint tracking error while standing at HOME.

WHAT THIS IS FOR. In simulation the robot cannot hold HOME_FRAME: it tips 5.85
deg in 1.2 s and is on its body by 1.9 s. Decomposing that showed it is NOT
vertical sag -- the sagittal chain holds to ~3 deg and the boots compress only
0.42 mm -- but the HIP ROLL joints giving way, commanded 5.0 deg and achieving
0.6 deg, symmetrically on both legs. The legs splay, the robot tips, the trunk
lowers geometrically.

If the real hip_roll holds to under a degree, the actuator model is too
compliant on that axis and the simulated robot is fighting a lateral
instability the real one does not have. That would depress every hop number in
the campaign, and would specifically penalise the two-footed symmetric launch,
which needs a stable lateral stance more than a skip does -- a plausible reason
symmetry has cost height in every sweep.

Model predictions, for comparison (boots on, 893 g):

    hip_roll   4.4 deg     ankle      0.9 deg
    knee       3.2 deg     hip_pitch  0.05 deg
    tilt after 1.2 s: 5.85 deg      spring compression: 0.42 mm/foot

THE ROBOT MAY WELL NOT STAND. That is a result, not a failed run: it is what
the model does. Everything is logged from before `init`, so a topple is
captured rather than lost -- and `--seconds` defaults to a short window for
exactly that reason.

    # on the robot, or over ssh:
    python3 scripts/measure_stance_sag.py --seconds 6 --out sag.csv

Reads robotd's socket directly (NDJSON JSON-RPC 2.0, `robot.subscribe` ->
`robot.state`), so it needs no extra dependency and does not disturb the
control loop. It does NOT command anything: run `robotctl robot init` yourself,
with hands ready.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import time

# duck_ipc_proto::JOINT_NAMES -- the wire order every positional vector uses.
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
# What the sim predicts for |target - measured|, degrees. Sign is not compared:
# the two legs mirror, so the interesting quantity is magnitude.
MODEL = {"hip_roll": 4.4, "knee": 3.2, "ankle": 0.9, "hip_pitch": 0.05, "hip_yaw": 0.3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/run/robotd.sock")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default="stance_sag.csv")
    ap.add_argument("--settle", type=float, default=1.2,
                    help="Seconds to skip before averaging, matching the sim window.")
    args = ap.parse_args()

    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(args.socket)
    except OSError as e:
        sys.exit(f"cannot connect to {args.socket}: {e}\n"
                 "Is robotd running? (systemctl is-active robotd)")
    # `params` is REQUIRED: robot.subscribe deserialises SubscribeParams and a
    # null params is rejected with -32602 "expected struct SubscribeParams".
    # `hz: null` inside it means every tick, which is what we want -- the whole
    # point is to catch a topple that takes under two seconds.
    s.sendall(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "robot.subscribe",
                          "params": {"hz": None}}).encode() + b"\n")

    f = s.makefile("r", encoding="utf-8")
    rows = []
    t0 = time.time()
    print(f"[sag] subscribed; logging {args.seconds:.1f} s. "
          "Run `robotctl robot init` now if you have not already.", flush=True)
    while time.time() - t0 < args.seconds:
        line = f.readline()
        if not line:
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        st = msg.get("params") if msg.get("method") == "robot.state" else None
        if st is None:
            st = (msg.get("result") or {}) if "result" in msg else None
        if not isinstance(st, dict) or "joints" not in st:
            continue
        j, tg = st.get("joints") or [], st.get("targets") or []
        if len(j) < 15 or len(tg) < 15:
            continue
        g = (st.get("safety") or {}).get("gravity") or [0, 0, -1]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
        row = {"t": st.get("t", time.time() - t0), "tilt_deg": tilt,
               "fallen": (st.get("safety") or {}).get("fallen"),
               "limp": (st.get("safety") or {}).get("limp"),
               "policy": st.get("policy")}
        for i, nm in enumerate(JOINT_NAMES):
            row[f"{nm}_target"] = math.degrees(tg[i])
            row[f"{nm}_meas"] = math.degrees(j[i])
            row[f"{nm}_err"] = math.degrees(tg[i] - j[i])
        rows.append(row)

    if not rows:
        sys.exit("no robot.state frames received -- is the policy/loop running?")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    t_start = rows[0]["t"]
    late = [r for r in rows if r["t"] - t_start >= args.settle] or rows
    print(f"\n[sag] {len(rows)} frames over "
          f"{rows[-1]['t'] - t_start:.2f} s; averaging the last "
          f"{len(late)} (after {args.settle:.1f} s)")
    print(f"[sag] tilt: first {rows[0]['tilt_deg']:.2f} deg -> "
          f"last {rows[-1]['tilt_deg']:.2f} deg   (sim: 0.8 -> 5.85 in 1.2 s)")
    if any(r["fallen"] for r in rows):
        print("[sag] NOTE: `fallen` went true during the run -- the robot went down. "
              "That is the sim's behaviour too; the numbers below are still the "
              "tracking error on the way there.")
    print(f"\n{'joint':18s} {'target':>8s} {'meas':>8s} {'|err|':>7s} {'model':>7s}")
    for nm in JOINT_NAMES:
        if nm == "mouth":
            continue
        tgt = sum(r[f"{nm}_target"] for r in late) / len(late)
        mea = sum(r[f"{nm}_meas"] for r in late) / len(late)
        err = sum(abs(r[f"{nm}_err"]) for r in late) / len(late)
        key = next((k for k in MODEL if nm.endswith(k)), None)
        exp = f"{MODEL[key]:.2f}" if key else "-"
        flag = ""
        if key and MODEL[key] > 0.5:
            flag = "  <-- MODEL SAYS MORE" if err < MODEL[key] / 2 else ""
        print(f"{nm:18s} {tgt:+8.2f} {mea:+8.2f} {err:7.2f} {exp:>7s}{flag}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
