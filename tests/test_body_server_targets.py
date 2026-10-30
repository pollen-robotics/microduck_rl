"""Bad TCP write frames must leave the previous motor targets intact."""

import json
import socket
import threading

import numpy as np
import pytest

from mjlab_microduck.sim.body_server import (
    DEFAULT_SCENE,
    JOINT_NAMES,
    MOUTH_INDEX,
    Body,
    Handler,
    Server,
    World,
)


@pytest.fixture(scope="module")
def body():
    return Body(World(DEFAULT_SCENE), 0)


def dispatch(body, targets):
    # Exercise the same JSON decode and dispatch path without binding a port.
    request = json.loads(json.dumps({"op": "write", "targets": targets}))
    return Handler.dispatch(None, body, request)


def test_valid_targets_preserve_wire_mapping(body):
    targets = [index / 100 for index in range(len(JOINT_NAMES))]
    targets[MOUTH_INDEX] = 0.25
    assert dispatch(body, targets) == {}
    np.testing.assert_array_equal(
        body.world.data.ctrl[body.actuator_slice],
        np.array(targets)[body.to_wire],
    )
    assert MOUTH_INDEX not in body.to_wire


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "bad",
        "0.1",
        True,
        [],
        {},
        float("nan"),
        float("inf"),
        -float("inf"),
        10**400,
    ],
)
@pytest.mark.parametrize("index", [0, MOUTH_INDEX, len(JOINT_NAMES) - 1])
def test_bad_target_rejects_the_whole_write(body, invalid, index):
    dispatch(body, [0.125] * len(JOINT_NAMES))
    before = body.world.data.ctrl.copy()
    targets = [0.375] * len(JOINT_NAMES)
    targets[index] = invalid
    with pytest.raises(ValueError):
        dispatch(body, targets)
    np.testing.assert_array_equal(body.world.data.ctrl, before)


@pytest.mark.parametrize(
    "targets", [None, {}, "0" * len(JOINT_NAMES), [], [0] * 14, [0] * 16]
)
def test_bad_target_container_leaves_controls_unchanged(body, targets):
    dispatch(body, [0.125] * len(JOINT_NAMES))
    before = body.world.data.ctrl.copy()
    with pytest.raises((TypeError, ValueError)):
        dispatch(body, targets)
    np.testing.assert_array_equal(body.world.data.ctrl, before)


def test_numeric_overflow_from_json_is_rejected(body):
    dispatch(body, [0.125] * len(JOINT_NAMES))
    before = body.world.data.ctrl.copy()
    request = json.loads(
        '{"op":"write","targets":[' + ",".join(["0"] * 14 + ["1e400"]) + "]}"
    )
    with pytest.raises(ValueError):
        Handler.dispatch(None, body, request)
    np.testing.assert_array_equal(body.world.data.ctrl, before)


def test_tcp_error_reply_preserves_targets_and_connection(body):
    dispatch(body, [0.125] * len(JOINT_NAMES))
    before = body.world.data.ctrl.copy()
    with Server(("127.0.0.1", 0), Handler) as server:
        server.body = body
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with (
                socket.create_connection(
                    server.server_address, timeout=2
                ) as connection,
                connection.makefile("rwb") as stream,
            ):
                targets = [0.375] * (len(JOINT_NAMES) - 1) + [None]
                stream.write(
                    (json.dumps({"op": "write", "targets": targets}) + "\n").encode()
                )
                stream.flush()
                assert "error" in json.loads(stream.readline())
                np.testing.assert_array_equal(body.world.data.ctrl, before)
                stream.write(
                    (
                        json.dumps(
                            {"op": "write", "targets": [0.25] * len(JOINT_NAMES)}
                        )
                        + "\n"
                    ).encode()
                )
                stream.flush()
                assert json.loads(stream.readline()) == {}
                np.testing.assert_array_equal(
                    body.world.data.ctrl[body.actuator_slice], 0.25
                )
        finally:
            server.shutdown()
            worker.join(timeout=2)
