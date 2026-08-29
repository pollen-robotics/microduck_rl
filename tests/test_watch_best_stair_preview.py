from scripts.watch_best_stair_preview import physics_improves


def _report(successes=0, x=0.60, z=0.13):
    return {
        "successes": successes,
        "best_route_x_m": x,
        "best_root_height_m": z,
    }


def test_physics_gate_prefers_success_then_material_progress_or_height():
    baseline = _report()
    assert physics_improves(_report(successes=1), baseline)
    assert physics_improves(_report(x=0.63), baseline)
    assert physics_improves(_report(x=0.58, z=0.16), baseline)
    assert not physics_improves(_report(x=0.61, z=0.14), baseline)
    assert not physics_improves(_report(x=0.55, z=0.20), baseline)
