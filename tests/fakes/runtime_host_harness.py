"""Real-process RuntimeChildHost with deterministic blocking fake native calls."""

from __future__ import annotations

import argparse
import socket
from dataclasses import replace
from types import SimpleNamespace

from fake_microduck_runtime import FakeMicroduckRuntime

import mjlab_microduck.rom.runtime_child as child_module
from mjlab_microduck.rom.runtime import RuntimeHandle
from mjlab_microduck.rom.runtime_child import RuntimeChildHost


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-fd", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("blocked-sample", "blocked-completion-cleanup"), required=True
    )
    args = parser.parse_args()
    control = socket.socket(fileno=args.control_fd)
    runtime = FakeMicroduckRuntime()
    host = RuntimeChildHost(control, fatal_cleanup_timeout_s=0.05)
    host._runtime = runtime
    host._bundle = SimpleNamespace(
        bundleVersion="1.0.0",
        bundleDigest="sha256:" + "a" * 64,
        model=SimpleNamespace(digest="sha256:" + "b" * 64),
        actions=[
            SimpleNamespace(
                actionCode="STAND", policyRef="stand", availability="AVAILABLE"
            )
        ],
        policies=[SimpleNamespace(policyRef="stand", digest="sha256:" + "c" * 64)],
    )
    host._handle = RuntimeHandle(taskId="1" * 32)
    runtime.active_handle = host._handle
    host._generation = 7
    host._task_id = "1" * 32
    host._active_action_code = "STAND"
    original_template = child_module.action_template

    def short_template(code: str):
        template = original_template(code)
        return replace(
            template,
            completion=template.completion.model_copy(update={"maxDurationMs": 50}),
        )

    child_module.action_template = short_template
    if args.mode == "blocked-sample":
        runtime.sample_release.clear()
    else:
        runtime.safe_stop_release.clear()
        runtime.complete_next(state="SUCCEEDED", metrics={})
    host._start_runtime_monitor()
    return host.run()


if __name__ == "__main__":
    raise SystemExit(main())
