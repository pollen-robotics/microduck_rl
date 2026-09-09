"""Drive the hop policy's phase AND read the pad: stand, hop on demand, stop.

The hop policy reads its phase as [cos(2*pi*phi), sin(2*pi*phi), 0] in the
twist slots, and robotd's `robot.move` sets exactly those slots. With HopPause
in the WALK slot and `cmd_alpha = 1.0`, this script is the phase generator:
frozen phase = stand, advancing = hop.

WHY THIS ALSO READS THE PAD, AND WHY padd MUST BE STOPPED. padd sends
`robot.move` from the sticks every tick in Drive mode -- zeros included -- so
with both running the applied twist strobes between the pad's [0,0,0] and this
script's phase at ~25 Hz (measured: 52/48 over 100 ticks). robotd then flips
between the walk and stand slots every tick and the policy sees a command
alternating between "stand" and "nothing". On the robot that was violent
flailing and a fall. Two writers to one intent cannot share the robot; this one
owns it, and reads the pad from evdev directly (pure Python, no dependency).

    Start   toggle the policy (robot.enable) -- ON = stand, OFF = joints hold
    A       one hop cycle, then back to standing
    B       RELAX -- torque off, immediately. The emergency stop.

    sudo systemctl stop padd
    python3 hop_phase_driver.py --device /dev/input/event4 --hold 0.65
    sudo systemctl start padd          # when done

Magnitude is always 1.0, above the 0.05 standing threshold, so the walk slot is
selected. Sent at 50 Hz; the 500 ms deadman needs it continuous.
"""
from __future__ import annotations
import argparse, json, math, os, select, socket, struct, time

HOP_PERIOD = 1.0
# linux/input-event-codes.h
EV_KEY = 1
BTN_SOUTH, BTN_EAST, BTN_START = 0x130, 0x131, 0x13B
EVENT_FMT = "llHHi"           # struct input_event on 64-bit: timeval(2 x long), type, code, value
EVENT_SIZE = struct.calcsize(EVENT_FMT)


class Robot:
    def __init__(self, path):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); self.s.connect(path)
        self.f = self.s.makefile("r", encoding="utf-8"); self.id = 0
    def notify(self, method, params):
        self.s.sendall(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}).encode() + b"\n")
    def call(self, method, params):
        self.id += 1
        self.s.sendall(json.dumps({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params}).encode() + b"\n")
        while True:
            line = self.f.readline()
            if not line: return None
            m = json.loads(line)
            if m.get("id") == self.id: return m
    def move_phase(self, phi, enable_bit=None):
        """Write the phase into the twist slots, and optionally the enable bit.

        `enable_bit` is the vyaw slot. HopFree-and-later policies read it as
        "hopping is wanted" (1.0) versus "stand" (0.0), because a frozen phase
        and an advancing phase are indistinguishable at the instant the clock
        passes through the hold point. Leave it None for every policy trained
        before 2026-09-09: those saw a hard zero there and read a non-zero as a
        YAW-RATE COMMAND -- measured, llu5t00x spins and topples when fed 1.0.
        """
        a = 2 * math.pi * phi
        self.notify("robot.move", {"vx": math.cos(a), "vy": math.sin(a),
                                   "vyaw": 0.0 if enable_bit is None else float(enable_bit)})
    def enable(self, on):
        r = self.call("robot.enable", {"on": on, "toggle": False})
        print(f"[pad] enable({on}) -> {(r or {}).get('result', r)}", flush=True)
    def relax(self):
        r = self.call("robot.relax", {})
        print(f"[pad] RELAX -> {(r or {}).get('result', r)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/run/robotd.sock")
    ap.add_argument("--device", default="/dev/input/event4")
    ap.add_argument("--hold", type=float, default=0.65)
    ap.add_argument("--hops", type=int, default=1, help="cycles per A press")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--enable-bit", action="store_true",
                    help="write 1.0 into the vyaw slot while the phase advances and 0.0 while it "
                         "is frozen (HopFree and later). REQUIRED for those policies and WRONG "
                         "for earlier ones, which read that slot as a yaw-rate command.")
    ap.add_argument("--hop-slot", action="store_true",
                    help="A triggers the ground-pick slot (a separate hop network) instead of "
                         "advancing this driver's phase. Use when the walk slot holds a stand-only policy.")
    args = ap.parse_args()

    robot = Robot(args.socket)
    try:
        pad = os.open(args.device, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        raise SystemExit(f"cannot open pad {args.device}: {e}  (is padd still running? "
                         "it does not block us, but its robot.move will fight this script -- stop it)")

    dt = 1.0 / args.hz
    phi = args.hold; hop_t0 = None
    # Sync "enabled" from robotd rather than assuming off: this driver restarts
    # more often than robotd does, and a stale False would turn a standing
    # robot OFF on the first Start press and refuse A until then.
    enabled = False
    try:
        sub = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); sub.connect(args.socket)
        sub.sendall(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "robot.subscribe",
                                "params": {"hz": None}}).encode() + b"\n")
        sf = sub.makefile("r", encoding="utf-8")
        for _ in range(20):
            m = json.loads(sf.readline() or "{}")
            if m.get("method") == "robot.state":
                enabled = m["params"].get("policy", "held") != "held"; break
        sub.close()
    except Exception as e:  # noqa: BLE001
        print(f"[pad] could not read robot state ({e}); assuming policy OFF", flush=True)
    print(f"[pad] policy currently {'ON' if enabled else 'OFF'}", flush=True)
    print(f"[pad] hold={args.hold}  Start=stand on/off  A=hop x{args.hops}  B=RELAX", flush=True)
    t_prev = time.time()
    while True:
        # --- pad events (non-blocking) ---
        r, _, _ = select.select([pad], [], [], 0)
        if r:
            try:
                data = os.read(pad, EVENT_SIZE * 64)
            except BlockingIOError:
                data = b""
            for i in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                _, _, kind, code, value = struct.unpack(EVENT_FMT, data[i:i + EVENT_SIZE])
                if kind != EV_KEY or value != 1:      # key-down edges only
                    continue
                if code == BTN_START:
                    enabled = not enabled; robot.enable(enabled)
                elif code == BTN_SOUTH and enabled and hop_t0 is None:
                    if args.hop_slot:
                        # TWO-NETWORK MODE: the hop lives in the ground-pick slot
                        # (a network that actually hops), the walk slot holds the
                        # stand. The daemon drives the hop's phase itself for
                        # `ground_pick_period` and hands back to the walk slot,
                        # which is still receiving our frozen phase. Advancing
                        # OUR phase instead would run the STAND network's phase
                        # -- and if that network does not hop (HopPauseR-S50:
                        # 3.2 mm), A produces a head twitch and nothing else.
                        import subprocess
                        subprocess.Popen(["robotctl", "robot", "do", "ground-pick"],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        print("[pad] HOP via ground-pick slot", flush=True)
                    else:
                        hop_t0 = time.time(); print("[pad] HOP (phase advance)", flush=True)
                elif code == BTN_EAST:
                    enabled = False; hop_t0 = None; robot.relax()
        # --- phase ---
        now = time.time()
        if hop_t0 is not None:
            el = now - hop_t0
            if el >= args.hops * HOP_PERIOD:
                hop_t0 = None; phi = args.hold; print("[pad] back to stand", flush=True)
            else:
                phi = (args.hold + el / HOP_PERIOD) % 1.0
        # hop_t0 is not None exactly while the phase is advancing, which is what
        # the enable bit means. None keeps the slot at a hard zero for policies
        # that predate it.
        robot.move_phase(phi, (1.0 if hop_t0 is not None else 0.0) if args.enable_bit else None)
        # --- pace ---
        sleep = dt - (time.time() - t_prev)
        if sleep > 0: time.sleep(sleep)
        t_prev = time.time()


if __name__ == "__main__":
    main()
