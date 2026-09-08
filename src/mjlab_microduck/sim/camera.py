"""What a duck sees, in the format `mediad` captures.

MuJoCo renders RGB; `mediad` pins its pipeline to UYVY because that is what `v4l2src` can drive at
full rate off the rkisp — so the conversion happens here, on the side that knows it is a simulator.

**Frames do not go down the JSON link.** 640x360 UYVY is 460,800 bytes, and at 15 fps that is 6.9
MB/s — JSON would be absurd. So a camera is its own TCP port carrying length-prefixed raw frames:
four bytes of little-endian length, then the bytes, forever. No handshake, because there is nothing
to negotiate that both ends do not already have to agree on to be useful.

**Opt in, per duck.** Rendering is the most expensive thing in the simulator by a wide margin —
12.2 ms per 640x360 frame, measured, against 0.3 ms to step four ducks' physics. Four ducks with
cameras at 15 fps is most of a core; four without is nothing. Most sessions do not need one.
"""

from __future__ import annotations

import socket
import socketserver
import struct
import threading

import mujoco
import numpy as np

# 16:9 at a size the default offscreen framebuffer can hold — MuJoCo caps offscreen rendering at
# the model's `<global offwidth/offheight>`, which is 640x480 unless the scene says otherwise.
WIDTH = 640
HEIGHT = 360

# The sensor's rate is 30, but a rendered frame costs 12 ms and a duck that is being watched is
# usually being watched rather than raced. 15 halves the cost for something nobody can see.
FPS = 15

# BT.601, the same coefficients `duck_detect`'s one-pass sampler uses on the robot.
_Y = np.array([0.299, 0.587, 0.114])
_U = np.array([-0.168736, -0.331264, 0.5])
_V = np.array([0.5, -0.418688, -0.081312])


def to_uyvy(rgb: np.ndarray) -> bytes:
    """RGB to packed UYVY: `U Y0 V Y1` per pixel pair, chroma averaged across the pair.

    Averaged rather than dropped, because a subsampler that takes the left pixel's chroma puts a
    half-pixel colour shift into every frame — invisible on a duck and not invisible to a detector
    trained on a real camera.
    """
    frame = rgb.astype(np.float32)
    luma = frame @ _Y
    chroma_u = frame @ _U + 128.0
    chroma_v = frame @ _V + 128.0

    pairs = frame.shape[1] // 2
    packed = np.empty((frame.shape[0], pairs, 4), dtype=np.uint8)
    packed[:, :, 0] = np.clip((chroma_u[:, 0::2] + chroma_u[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 1] = np.clip(luma[:, 0::2], 0, 255)
    packed[:, :, 2] = np.clip((chroma_v[:, 0::2] + chroma_v[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 3] = np.clip(luma[:, 1::2], 0, 255)
    return packed.tobytes()


class Camera:
    """One duck's head camera, rendered on demand.

    The renderer is not thread-safe and is expensive to make, so one lives here and only the frame
    loop touches it.
    """

    def __init__(self, model: mujoco.MjModel, name: str, width: int = WIDTH, height: int = HEIGHT):
        self.camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.camera < 0:
            raise SystemExit(f"the model has no camera {name!r}")

        # **The model's head camera faces backwards.** Measured against the duck's own forward axis
        # and the ToF site's: the camera's view direction is -x where both of those are +x, exactly
        # 180 degrees out. On screen that is a duck apparently seeing what is behind it — a cyan cube
        # it is walking away from, sitting in frame.
        #
        # Turned 180 degrees about the camera's own **right** axis — not its up axis, which was the
        # first attempt and came out upside down. Both turns fix the direction; only this one leaves
        # the image the same way up. What the console has to undo is set by where the camera's right
        # axis points: the original camera has it along the world's *down*, and a yaw turn moves it
        # to *up*, so the quarter turn the console applies lands 180 degrees out.
        #
        # A roll turn keeps right pointing down, so a rendered frame comes out on its side exactly as
        # the original did — which is correct, because the real head camera is mounted a quarter turn
        # off and every consumer already expects that. `mediad --rotate 90` stays true of a simulated
        # duck for the same reason it is true of a real one.
        #
        # Done here rather than in the MJCF, because that file belongs to the RL work and a camera
        # nothing in training uses is not worth a change they have to review.
        turn = np.array([0.0, 1.0, 0.0, 0.0])  # 180 degrees about x, scalar-first
        fixed = np.zeros(4)
        mujoco.mju_mulQuat(fixed, model.cam_quat[self.camera], turn)
        model.cam_quat[self.camera] = fixed
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.width = width
        self.height = height
        self.latest: bytes | None = None
        self.lock = threading.Lock()

    def render(self, world) -> None:
        """Render one frame, reading `MjData` only while holding the world's lock.

        **`update_scene` reads the whole of `MjData`, and it runs on the step loop's thread while
        sensor reads run on socket threads.** Unlocked, a ToF read caught a site orientation
        mid-write and got a zero-length ray direction — which MuJoCo answers with
        `mj_ray: vector length is too small` and an abort, taking the simulator down with it. The
        lock is held for the scene copy, which is a millisecond, and released for the render, which
        is twelve and touches no shared state.
        """
        with world.lock:
            self.renderer.update_scene(world.data, camera=self.camera)
        packed = to_uyvy(self.renderer.render())
        with self.lock:
            self.latest = packed

    def frame(self) -> bytes | None:
        with self.lock:
            return self.latest


class FrameHandler(socketserver.BaseRequestHandler):
    """Length-prefixed frames, at the camera's rate, until the reader goes away."""

    def handle(self) -> None:
        camera: Camera = self.server.camera
        fps: int = self.server.fps
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"== camera: a reader connected from {self.client_address}", flush=True)
        period = 1.0 / max(1, fps)
        import time

        next_frame = time.perf_counter()
        try:
            while True:
                frame = camera.frame()
                if frame is not None:
                    self.request.sendall(struct.pack("<I", len(frame)) + frame)
                next_frame += period
                slack = next_frame - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_frame = time.perf_counter()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        print("== camera: the reader went away", flush=True)


class FrameServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
