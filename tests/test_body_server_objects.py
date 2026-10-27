"""The simulator's read-only `objects` op: where the loose things in a scene are.

A viewer that mirrors the world (a browser drawing the duck and its room) gets the duck from `read`,
but balls and boxes have no other way out of the simulator — `read` deliberately carries only what a
robot could sense.
"""

from mjlab_microduck.sim.body_server import SCENES, Body, Handler, World

APARTMENT = SCENES / "scene_apartment.xml"


def test_objects_lists_the_loose_bodies_and_not_the_duck():
    objects = World(APARTMENT).objects()
    assert {"ball_0", "ball_1", "ball_2", "obj_0", "obj_1", "obj_2", "obj_3"} <= set(objects)
    assert all(len(pose) == 7 for pose in objects.values())  # x y z qw qx qy qz
    assert not any("trunk" in name for name in objects)


def test_objects_reads_the_live_state():
    world = World(APARTMENT)
    adr = dict(world.loose)["ball_0"]
    before = world.objects()["ball_0"][0]
    world.data.qpos[adr] += 0.5
    assert world.objects()["ball_0"][0] == before + 0.5


def test_the_op_is_served_and_changes_nothing():
    world = World(APARTMENT)
    body = Body(world, 0)
    qpos = world.data.qpos.copy()
    answer = Handler.dispatch(None, body, {"op": "objects"})
    assert answer == {"objects": world.objects()}
    assert (world.data.qpos == qpos).all()


def test_a_scene_without_loose_things_answers_empty():
    assert World(SCENES / "scene.xml").objects() == {}
