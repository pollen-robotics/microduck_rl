"""Static checks for the jump task cfg.

``mjlab`` / ``mujoco_warp`` / ``jax`` are not installed in the CPU dev
environment, so `microduck_jump_env_cfg` cannot be imported here — and neither
can any of the neighbouring ``test_*_cfg.py`` files. That does not excuse
shipping an unverified cfg: the most likely defect in a hand-written cfg is a
symbol that does not exist (typo, renamed helper, term referenced from the wrong
module), and that is checkable WITHOUT the dependency by parsing the sources.

What this file canNOT check, and what therefore still needs a CUDA box:
  * that the env actually constructs (every ``.params[...]`` key is real),
  * that the reward weights produce a hop rather than a squat-and-rise,
  * that the 61D obs layout really matches the runtime slot.
Those are listed in runs/JUMP_TASK.md as the first-run checklist.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "mjlab_microduck"
CFG_PATH = SRC / "tasks" / "microduck_jump_env_cfg.py"
MDP_PATH = SRC / "tasks" / "mdp.py"
INIT_PATH = SRC / "tasks" / "__init__.py"
VELOCITY_CFG_PATH = SRC / "tasks" / "microduck_velocity_env_cfg.py"

JUMP_TERMS = (
    "jump_crouch_depth",
    "jump_launch",
    "jump_apex",
    "jump_airborne",
    "jump_impact_speed",
)

# The cfg's reward KEYS are not the mdp function names — the cfg registers
# jump_crouch_depth under the key "jump_crouch", and so on. Keeping the two
# lists separate is deliberate: conflating them is a mistake this test already
# caught once.
JUMP_REWARD_KEYS = (
    "jump_crouch",
    "jump_launch",
    "jump_airborne",
    "jump_apex",
    "jump_impact",
)


def _module_level_names(path: Path) -> set:
    """Names a module defines at top level (functions, classes, assignments)."""
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _attr_refs(path: Path, base: str) -> set:
    """Every ``<base>.<attr>`` referenced anywhere in the module."""
    refs = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == base):
            refs.add(node.attr)
    return refs


def _const(name: str):
    """Read a module-level literal constant without importing the module."""
    for node in ast.parse(MDP_PATH.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found at module level in {MDP_PATH.name}")


# ------------------------------------------------------------------ syntax
def test_cfg_and_mdp_both_parse():
    for p in (CFG_PATH, MDP_PATH, INIT_PATH):
        compile(p.read_text(encoding="utf-8"), str(p), "exec")


# -------------------------------------------------- symbol references are real
def test_jump_terms_are_defined_in_mdp():
    defined = _module_level_names(MDP_PATH)
    missing = [n for n in JUMP_TERMS if n not in defined]
    assert not missing, f"jump terms missing from tasks/mdp.py: {missing}"


def test_every_microduck_mdp_symbol_used_by_the_cfg_exists():
    """The failure mode this catches: a cfg that imports fine and blows up only
    when the manager builds the reward term, weeks later, on a GPU box."""
    defined = _module_level_names(MDP_PATH)
    missing = sorted(_attr_refs(CFG_PATH, "microduck_mdp") - defined)
    assert not missing, (
        f"jump cfg references microduck_mdp symbols that do not exist: {missing}")


def test_unknown_reward_keys_are_not_introduced():
    """Every reward key the cfg defines must be either a jump term or a key the
    velocity base cfg already knows about — otherwise `cfg.rewards[...]` raises
    KeyError at construction and only on a CUDA box."""
    unknown = sorted(_cfg_reward_funcs().keys() - set(_KNOWN_REWARD_KEYS))
    assert not unknown, f"cfg sets unknown reward keys: {unknown}"


_KNOWN_REWARD_KEYS = (
    set(JUMP_REWARD_KEYS)
    | {"upright", "body_ang_vel", "action_rate_l2", "self_collisions",
       "feet_flat", "neck_action_rate_l2", "neck_joint_pos_l2",
       "joint_torques_l2"}
)


def _cfg_reward_funcs() -> dict:
    """Map ``cfg.rewards["key"] = RewardTermCfg(func=<x>, ...)`` to the mdp
    function name it points at (``None`` for functions from the external mdp)."""
    out = {}
    for node in ast.walk(ast.parse(CFG_PATH.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Subscript)
                and isinstance(tgt.value, ast.Attribute)
                and tgt.value.attr == "rewards"
                and isinstance(tgt.slice, ast.Constant)):
            continue
        func = None
        for kw in node.value.keywords:
            if kw.arg == "func" and isinstance(kw.value, ast.Attribute):
                if (isinstance(kw.value.value, ast.Name)
                        and kw.value.value.id == "microduck_mdp"):
                    func = kw.value.attr
        out[tgt.slice.value] = func
    return out


def test_jump_reward_keys_are_wired_to_the_jump_terms():
    """Catches the copy-paste bug that a KeyError check cannot see: a reward
    registered under the right key but pointing at the WRONG function (e.g.
    jump_apex wired to jump_launch) builds fine and trains the wrong thing."""
    funcs = _cfg_reward_funcs()
    expected = {
        "jump_crouch": "jump_crouch_depth",
        "jump_launch": "jump_launch",
        "jump_airborne": "jump_airborne",
        "jump_apex": "jump_apex",
        "jump_impact": "jump_impact_speed",
    }
    for key, fn in expected.items():
        assert funcs.get(key) == fn, (
            f"cfg.rewards[{key!r}] is wired to {funcs.get(key)!r}, expected {fn!r}")


# ------------------------------------------------------------ registration
def test_jump_task_is_registered():
    text = INIT_PATH.read_text(encoding="utf-8")
    assert '"Mjlab-Jump-Flat-MicroDuck"' in text
    assert "make_microduck_jump_env_cfg" in text
    assert "MicroduckJumpRlCfg" in text


def test_registration_imports_what_it_uses():
    """A registration block whose import line was forgotten is an ImportError on
    plugin load — i.e. it takes down EVERY task, not just jump."""
    assert "make_microduck_jump_env_cfg" in _imported_names(INIT_PATH)
    assert "MicroduckJumpRlCfg" in _imported_names(INIT_PATH)


def test_backlash_variant_is_registered_with_the_walk_robot():
    """Jump must ship a -Backlash variant like every other family (servo gear
    play is a real deployment property), and it must use the WALK backlash
    robot: jump is a flat-ground walk-model task. Grabbing the allcollisions
    robot would silently swap the collision model out from under the task."""
    text = INIT_PATH.read_text(encoding="utf-8")
    assert '"Mjlab-Jump-Flat-Backlash-MicroDuck"' in text, (
        "jump has no -Backlash variant; the other 15 registered families have one")
    row = [ln for ln in text.splitlines()
           if "Mjlab-Jump-Flat-Backlash-MicroDuck" in ln]
    assert len(row) == 1, f"expected exactly one jump backlash row, got {len(row)}"
    assert "_BL_WALK" in row[0], (
        f"jump backlash must use the walk robot (_BL_WALK), got: {row[0].strip()}")
    assert "make_microduck_jump_env_cfg" in row[0]
    assert "MicroduckJumpRlCfg" in row[0]
    # The walk-backlash cfg must actually be imported, or plugin load dies.
    assert "MICRODUCK_WALK_BACKLASH_ROBOT_CFG" in _imported_names(INIT_PATH)


def _imported_names(path: Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
    return names


# ------------------------------------------------------- phase-window sanity
def test_phase_windows_are_ordered_and_inside_the_cycle():
    crouch_end = _const("JUMP_CROUCH_END")
    launch_end = _const("JUMP_LAUNCH_END")
    flight_end = _const("JUMP_FLIGHT_END")
    assert 0.0 < crouch_end < launch_end < flight_end <= 1.0, (
        f"jump windows must be strictly ordered inside [0,1): "
        f"{crouch_end} / {launch_end} / {flight_end}")


def test_crouch_target_is_reachable_and_above_the_sit_height():
    """A crouch target below the sit height would be the same unsatisfiable-
    reward trap as a tracked flight curve: unreachable, and a policy's cheapest
    response to an unreachable target is to stop trying."""
    stand_z = _const("JUMP_STAND_Z")
    crouch_z = _const("JUMP_CROUCH_Z")
    assert 0.06 < crouch_z < stand_z, (
        f"crouch_z={crouch_z} must sit above the ~0.06 sit height and below "
        f"stand_z={stand_z}")


def test_jump_period_shorter_than_the_ground_pick_period():
    """A hop is a fast gesture; at the 4 s ground-pick period most of an
    episode is spent standing and the launch/landing windows get too few
    samples to learn from."""
    source = CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    period = None
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "JUMP_PERIOD"
                        for t in node.targets)):
            period = ast.literal_eval(node.value)
    assert period is not None, "JUMP_PERIOD not found"
    assert 0.5 <= period < 4.0, f"JUMP_PERIOD={period} s is outside the useful range"


# --------------------------------------------------- first-run script sanity
FIRST_RUN_PATH = REPO / "scripts" / "jump_first_run.py"


def test_first_run_script_parses_and_has_its_checks():
    """The CUDA checklist must run on a box we cannot reach from here, so its
    only static guarantee is that it parses and still contains every check.
    A silently dropped check would make a half-verified run look complete."""
    compile(FIRST_RUN_PATH.read_text(encoding="utf-8"), str(FIRST_RUN_PATH), "exec")
    text = FIRST_RUN_PATH.read_text(encoding="utf-8")
    for fn in ("check_imports", "check_construct", "check_backlash_variant",
               "check_phase_terms"):
        assert f"def {fn}(" in text, f"first-run script lost {fn}()"


def test_first_run_script_reports_failure_without_crashing():
    """Run it here, where the GPU stack is absent: it must degrade to a clean
    non-zero exit with a readable summary, not a traceback — because that is
    exactly the first thing the user will see on the CUDA box if a dep is
    missing, and a stack dump there would look like a bug in the task."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(FIRST_RUN_PATH), "--skip-rollout"],
        capture_output=True, text=True, timeout=120,
    )
    combined = proc.stdout + proc.stderr
    assert "SUMMARY" in combined, f"no summary printed:\n{combined}"
    assert proc.returncode != 0, "expected non-zero exit when deps are absent"
    assert "NOT READY" in combined, combined
    # A ModuleNotFoundError is an expected, handled outcome here — but it must
    # be reported as a check failure, not escape as an unhandled traceback.
    assert "Traceback (most recent call last)" not in proc.stderr, (
        f"the script crashed instead of reporting:\n{proc.stderr}")
