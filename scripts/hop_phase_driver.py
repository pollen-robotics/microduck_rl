"""Drive the hop policy's phase from outside robotd: stand, then hop on demand.

The hop policy reads its phase as [cos(2*pi*phi), sin(2*pi*phi), 0] in the
twist slots, and robotd's `robot.move` sets exactly those slots. So with the hop
network in the WALK slot and `cmd_alpha = 1.0` (no EMA on the command), this
script IS the phase generator:

  * FROZEN phase in the recovery half  -> the policy holds a stance: a stand.
  * ADVANCING at 1/HOP_PERIOD          -> the policy hops.
  * back to frozen                     -> it lands and stands again.

One network, one action space, no rigid-foot policy anywhere in the loop, and
no separate stand to train -- IF the frozen-phase stance holds, which
`frozen.py` checks in sim first.

Magnitude is always 1.0, above the 0.05 standing threshold, so the walk slot is
selected regardless of the pad. Sent at 50 Hz as notifications; the 500 ms
deadman needs it continuous.

    python3 hop_phase_driver.py --hold 0.65                # stand, indefinitely
    python3 hop_phase_driver.py --hold 0.65 --hops 1       # stand 3 s, one hop cycle, stand
    python3 hop_phase_driver.py --hold 0.65 --hops 3 --pre 3 --post 5
"""
from __future__ import annotations
import argparse, json, math, socket, time

HOP_PERIOD = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/run/robotd.sock")
    ap.add_argument("--hold", type=float, default=0.65, help="frozen phase for standing (recovery half: 0.5-1.0)")
    ap.add_argument("--hops", type=int, default=0, help="number of full 1 Hz cycles to run; 0 = stand forever")
    ap.add_argument("--pre", type=float, default=3.0, help="seconds standing before the hops")
    ap.add_argument("--post", type=float, default=5.0, help="seconds standing after the hops")
    ap.add_argument("--hz", type=float, default=50.0)
    args = ap.parse_args()

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(args.socket)
    dt = 1.0 / args.hz

    def send(phi):
        a = 2 * math.pi * phi
        msg = {"jsonrpc": "2.0", "method": "robot.move",
               "params": {"vx": math.cos(a), "vy": math.sin(a), "vyaw": 0.0}}
        s.sendall(json.dumps(msg).encode() + b"\n")

    print(f"[phase] hold={args.hold}  hops={args.hops}  pre={args.pre}s post={args.post}s", flush=True)
    t0 = time.time(); phi = args.hold; mode = "stand"
    hop_start = None; hops_done = 0
    try:
        while True:
            t = time.time() - t0
            if mode == "stand" and args.hops and t >= args.pre and hops_done == 0:
                mode = "hop"; hop_start = t; print("[phase] HOP", flush=True)
            if mode == "hop":
                # Advance from the hold phase so the first cycle starts where the
                # stance is, and passes through launch (sin > 0) once per period.
                phi = (args.hold + (t - hop_start) / HOP_PERIOD) % 1.0
                if (t - hop_start) >= args.hops * HOP_PERIOD:
                    mode = "post"; phi = args.hold; hops_done = args.hops
                    print("[phase] back to stand", flush=True)
            if mode == "post" and (t - hop_start) >= args.hops * HOP_PERIOD + args.post:
                break
            send(phi); time.sleep(dt)
    except KeyboardInterrupt:
        pass
    # Leave the robot standing: robotd's deadman zeroes the velocity when we stop,
    # which would hand the policy an out-of-distribution [0,0,0]. Say so.
    print("[phase] done -- NOTE: once this exits the deadman zeroes twist; relax or re-run promptly.")


if __name__ == "__main__":
    main()
