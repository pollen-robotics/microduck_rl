#!/usr/bin/env python3
"""Refresh the copied parts of cmu/ from the private robotics working tree (Shawn's Mac only).

  python3 cmu/sync_from_robotics.py [--robotics ~/code/robotics] [--check]

Everything under cmu/mac, cmu/pi, cmu/whole, cmu/real-logs, cmu/model/sole, cmu/simulator and the copied files of cmu/hardware is a generated copy: edit the
source in the robotics tree, rerun this script, commit the result. The rewrite rules below are the only
difference between a source file and its copy (path hops between the two layouts, no private venv re-exec,
relative manifest paths); each rule must match or the sync stops, so a drifted source is noticed, not silently
copied. --check reports what would change without writing.
"""
import argparse
import difflib
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM_REL = "microduck-sim"
MAC = "guide/firmware/mac"
PI = "guide/firmware/pi"
WHOLE = "hardware-design-20260912/whole-robot-20260912"

# (source relative to microduck-sim, destination relative to cmu/)
FILES = []


def add(src_dir, dst_dir, names):
    for n in names:
        FILES.append((f"{src_dir}/{n}", f"{dst_dir}/{n}"))


add(MAC, "mac", [
    "walk_brain.py", "walk_wired.py", "brain_sim_check.py", "brain_sim_batch.py", "jaw_ctl.py", "fake_pi.py",
    "pickup.py", "pickup_sim.py", "target_ground.py", "orange_target.py", "duck_cam.py", "stance_switch.py",
    "sim_walk_check.py", "measure_gap.py", "walk_trial.py", "pose_snapshot.py",
    "test_pickup.py", "test_pickup_modes.py", "test_stance_switch.py", "test_target_ground.py", "test_walk_wired_battery.py",
    "PICKUP.md", "CAMERA.md", "stance_poses.json", "stance_poses_big.json", "stance_poses_small.json",
    "brain-sim-check.json", "battery-stop-sim.json",
    "duck_sound.py",  # walk_wired.py imports it (quack)
    "duck_sense.py",  # the live page's read-only standby reader on the A7Z
    "test_teleop_modes.py", "teleop_sim_batch.py", "teleop-sim-check.json",
    "test_walk_wired_touch.py",  # Present Current and head touch pads in the tick log
    "hold.py", "test_hold.py",  # graded hold: walk_wired.py and brain_sim_check.py import it
])
add(f"{MAC}/jitter-20261018", "mac/jitter-20261018", ["jitter_sim.py", "compare.py", "osc_metrics.py", "band.py", "summ.sh"])
add(f"{MAC}/pickup-sim-20261016", "mac/pickup-sim-20261016", [])  # filled below from the directory listing
add(f"{MAC}/robot-runs-20261018", "real-logs/robot-runs-20261018", [])
add(PI, "pi", [
    "stand_pose.py", "battery.py", "dxl_fast.py", "jaw_limits.py", "jaw_limits.json", "jaw_pose_caps.json",
    "jaw_calibration.json", "test_battery.py", "test_jaw_limits.py", "pi_imu_uart.py", "pi_bridge.py", "stand.py",
    "JAW-LIMIT.md", "hip_yaw_check.py", "set_shutdown.py", "check_voltage_limit.py", "joint_signs.py",
    "set_jaw_limit.py",  # imported by test_jaw_limits.py (its pure planning part is under test)
    "scan_bus.py",  # stand.py imports its ROLES table; test_battery.py imports stand.py with a mocked dynamixel_sdk
    # A7Z head computer: 10 Hz temperature and per-cluster clock log, full-CPU load test, CPU setting test (2026-10-21)
    "thermal_log.py", "a7z_load_test.sh", "a7z_cpu_compare.py", "A7Z-CPU-COMPARE.md", "test_thermal_log.py",
    "test_a7z_cpu_compare.py",
])
add(f"{PI}/microduck_rt", "pi/microduck_rt", ["constants.py", "imu_reader.py"])
add(f"{WHOLE}/dynamics", "whole/dynamics", ["evaluate.py"])
add(f"{WHOLE}/dynamics/asbuilt-20261015", "whole/dynamics/asbuilt-20261015",
    ["asbuilt-mass-manifest.json", "asbuilt.config.json", "build_asbuilt_manifest.py"])
add(f"{WHOLE}/dynamics/v19-20261014", "whole/dynamics/v19-20261014", ["v19-mass-manifest.json"])
add(f"{WHOLE}/dynamics/v19-20261014/runs", "whole/dynamics/v19-20261014/runs", ["const-bat90-sensor0-nocomp-seed0.config.json"])
add(f"{WHOLE}/dynamics/realgap-20261014", "whole/dynamics/realgap-20261014", [
    "README.md", "static_gap.py", "policy_test.py", "margins.py", "margins.json", "volt_margins.py", "volt_margins.json",
    "trims.py", "trims.json", "ramp_test.py", "scan_lean.py", "head_test.py", "sitstand_volt.py", "literature-note.md", "batch1.py",
])
add(f"{WHOLE}/delay-repair-20260912", "whole/delay-repair-20260912", ["evaluate_delay.py", "compensation.py", "README.md"])
add(f"{WHOLE}/runtime-overlay", "whole/runtime-overlay", ["source-lock.json"])  # README.md is a Pi-install note for the old community runtime, not copied
add(f"{WHOLE}/runtime-overlay/models", "whole/runtime-overlay/models", ["walk61.onnx", "stand61.onnx", "groundpick61.onnx",
    "sitstand61.onnx", "kick_left61.onnx", "kick_right61.onnx"])  # the remote-control policies (Pollen's, source-lock.json)
add(f"{WHOLE}/runtime-overlay/models-ft3-it999", "whole/runtime-overlay/models-ft3-it999", [])
add(f"{WHOLE}/electrical", "whole/electrical", ["existing-model-mass-audit.json"])
add("build-pack-20260913/procurement/feet-sole-20261018", "model/sole", ["README.md", "sole_thickness_sim.py", "sole_template.py"])
add("build-pack-20260913/procurement/feet-sole-20261018/sweep", "model/sole/sweep", ["summary.json"])
DIR_COPIES = [  # (source dir, dest dir): every regular file, no recursion
    (f"{MAC}/pickup-sim-20261016", "mac/pickup-sim-20261016"),
    (f"{MAC}/robot-runs-20261018", "real-logs/robot-runs-20261018"),
    (f"{WHOLE}/runtime-overlay/models-ft3-it999", "whole/runtime-overlay/models-ft3-it999"),
]

# cmu/hardware: the battery backpack design materials and the RoboClub print pipeline (2026-10-19, Shawn's ask: teammates
# redesign and print the backpack themselves). Generated copies like everything else here; the hand-written pages are
# cmu/hardware/README.md, battery-backpack/README.md and printing/README.md.
BP = "guide/print/back-plate/battery-backpack-20261015"
HWB = "hardware/battery-backpack"
PIPE = "build-pack-20260913/redesign-20260913/printing/13-RELEASE-PIPELINE-20261009"
ELEGOO = "build-pack-20260913/redesign-20260913/printing-elegoo-20260929"
REL = "guide/print/release-V1.20"
HWP = "hardware/printing"
FILES += [
    (f"{BP}/README.md", f"{HWB}/DESIGN-D010.md"),
    ("build-pack-20260913/power/battery-backpack-20261015/README.md", f"{HWB}/POWER-PLAN.md"),
    ("bringup/BATTERY-BACKPACK-TEAM.html", f"{HWB}/BATTERY-BACKPACK-TEAM.html"),
    (f"{PIPE}/README.md", f"{HWP}/RELEASE-PIPELINE.md"),
    ("guide/print/orientation-study/PRINT-ORIENTATION-PRACTICE.md", f"{HWP}/PRINT-ORIENTATION-PRACTICE.md"),
    ("guide/print/orientation-study/orientation_check.py", f"{HWP}/scripts/orientation_check.py"),
    ("guide/print/PRINT-LESSONS.md", f"{HWP}/PRINT-LESSONS.md"),
    (f"{ELEGOO}/filament.json", f"{HWP}/profiles/elegoo-filament.json"),
    (f"{ELEGOO}/process.json", f"{HWP}/profiles/elegoo-process.json"),
    (f"{ELEGOO}/machine.json", f"{HWP}/profiles/elegoo-machine.json"),
    (f"{REL}/manifest-V1.20.json", f"{HWP}/ready/V1.20/manifest-V1.20.json"),
    (f"{REL}/PARTS-SHEET-V1.20.html", f"{HWP}/ready/V1.20/PARTS-SHEET-V1.20.html"),
    (f"{REL}/run/Orange_Bambu_editable.3mf", f"{HWP}/ready/V1.20/Orange_Bambu_editable.3mf"),
    (f"{REL}/run/Orange_Elegoo_editable.3mf", f"{HWP}/ready/V1.20/Orange_Elegoo_editable.3mf"),
    (f"{REL}/run/delivery-folder/zhengxia_MicroDuck_V1.20/Orange-plate.png", f"{HWP}/ready/V1.20/Orange-plate.png"),
    (f"{REL}/run/delivery-folder/zhengxia_MicroDuck_V1.20/X1C/zhengxia_V1.20_Orange_Bambu-X1C_2h20m.gcode.3mf",
     f"{HWP}/ready/V1.20/X1C/zhengxia_V1.20_Orange_Bambu-X1C_2h20m.gcode.3mf"),
    (f"{REL}/run/delivery-folder/zhengxia_MicroDuck_V1.20/Elegoo/zhengxia_V1.20_Orange_Elegoo-Centauri-Carbon_2h22m.gcode",
     f"{HWP}/ready/V1.20/Elegoo/zhengxia_V1.20_Orange_Elegoo-Centauri-Carbon_2h22m.gcode"),
]
# hardware/battery-backpack/stl is owned here since the battery-on-the-cap redesign replaced the V1.20 parts
# (hardware/battery-redesign/build_cap_cradle.py writes them; the V1.20 world STLs it starts from are in its v120-source/).
add(f"{BP}/build", f"{HWB}/envelopes", ["battery-world.stl", "dfr1202-world.stl", "dfr_pads-world.stl", "switch_swann-world.stl",
                                        "switch-world.stl", "plugs-world.stl", "swap_plugs-world.stl", "wires.json", "build.json"])
add(BP, f"{HWB}/checks", ["bp_checks.json", "bp_mass.json", "bp_hip_sweep.json", "release-rows-orange.json"])
add(f"{BP}/renders", f"{HWB}/renders", ["BP-sheet.png", "lead-bend-explainer.png"])
add(f"{BP}/renders/team", f"{HWB}/renders", ["TEAM-sheet.png"])
add(BP, f"{HWB}/cad-source", ["build_bp.py", "concepts.py", "bp_checks.py", "bp_mass.py", "render_bp.py", "render_team.py", "render_concepts.py",
                              "swann_switch.py", "release_rows.py"])
add("guide/print/world", f"{HWB}/mating", ["rear_pack.stl", "rear_frame.stl", "rear_lid.stl", "shell_left.stl", "shell_right.stl",
                                           "openrb-board-upright.stl"])
add(PIPE, f"{HWP}/scripts", ["release_pipeline_v2.py", "verify_release_v2.py", "review_brim.py",
                             "wall_check.py", "verify_release.py", "release_pipeline.py"])  # the last three: imported by the v2 scripts
# (3mf, member, destination): a JSON member of a project archive written out as a readable profile
EXTRACTS = [
    (f"{REL}/run/Orange_Bambu_editable.3mf", "Metadata/project_settings.config", f"{HWP}/profiles/x1c-orange-project_settings.json"),
    (f"{REL}/run/Orange_Elegoo_editable.3mf", "Metadata/project_settings.config", f"{HWP}/profiles/elegoo-orange-project_settings.json"),
]

# cmu/simulator: the browser simulator and the Microduck macOS app, from the microduck-simulator checkout (Pollen's Hugging
# Face Space with our commits on top). Committed content only: a file with uncommitted edits is copied as it is at HEAD,
# so half-done work never lands here, and files no longer in the checkout are removed from the copy. Left out:
# app/test-results (about 1.5 GB of September household result archives) and the household research records (docs,
# experiments, tutorials, sources, scripts). rl/ is kept: the app's ground-pick tests import its review helpers, and it
# holds the jaw PPO policy. .gitattributes is left out too: this repository does not use Git LFS.
SIMULATOR = "microduck-simulator"
SIM_DST = "simulator"
SIM_KEEP = ("app/", "desktop/", "rl/", "README.md", "Dockerfile", ".gitignore", "Play-Jaw-RL.command", "Play-MicroDuck-Course.command")
SIM_SKIP = ("app/test-results/",)
LFS_POINTER = b"version https://git-lfs.github.com/spec/"


def simulator_files(sim: Path) -> dict:
    """{destination relative to cmu/: bytes} for the simulator's committed files."""
    import subprocess
    root = sim / SIMULATOR
    git = lambda *args: subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout
    tracked = [f for f in git("ls-files", "-z").decode().split("\0") if f]
    dirty = {f for f in git("diff", "--name-only", "-z", "HEAD").decode().split("\0") if f}
    out = {}
    for rel in tracked:
        if not rel.startswith(SIM_KEEP) or rel.startswith(SIM_SKIP):
            continue
        data = git("show", f"HEAD:{rel}") if rel in dirty else (root / rel).read_bytes()
        if data.startswith(LFS_POINTER):  # an LFS file whose content is not in the checkout (run git lfs pull there)
            sys.exit(f"sync: {SIMULATOR}/{rel} is a Git LFS pointer, not its content")
        if Path(rel).suffix in TEXT_SUFFIXES:  # recorded absolute paths of the private tree, as for the other copies
            data = data.replace(ABS_SIM.encode(), f"{SIM_REL}/".encode())
        out[f"{SIM_DST}/{rel}"] = data
    return out


WHOLE_HOP = '"..", "..", "..", "hardware-design-20260912", "whole-robot-20260912"'
EXEC_BLOCK = re.compile(
    r'SIM_PYTHON = os\.path\.expanduser\("~/\.venvs/microduck-dyn/bin/python"\)\n'
    r'if os\.path\.realpath\(sys\.prefix\) != os\.path\.realpath\(os\.path\.dirname\(os\.path\.dirname\(SIM_PYTHON\)\)\):\n'
    r'    os\.execv\(SIM_PYTHON, \[SIM_PYTHON, os\.path\.abspath\(__file__\)\] \+ sys\.argv\[1:\]\)\n')
EXEC_NOTE = "# cmu/ copy: runs in the repository's uv environment (uv run <script>); no private venv re-exec.\n"
ABS_SIM = os.path.expanduser("~/code/robotics/") + SIM_REL + "/"  # the private tree's absolute path, stripped from copied JSON
DUCK_IP = re.compile(r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")  # the robot's private Tailscale address (100.64/10); the copies carry the placeholder DUCK_IP

# destination -> list of (kind, pattern, replacement, min_matches)
RULES = {
    "mac/walk_brain.py": [("str", WHOLE_HOP, '"..", "whole"', 1), ("re", DUCK_IP, "DUCK_IP", 1)],
    "mac/pickup.py": [("re", DUCK_IP, "DUCK_IP", 1)],
    "mac/duck_cam.py": [("re", DUCK_IP, "DUCK_IP", 1)],
    "mac/CAMERA.md": [("re", DUCK_IP, "DUCK_IP", 1)],
    "whole/runtime-overlay/source-lock.json": [("str", ABS_SIM, f"{SIM_REL}/", 1)],
    "whole/electrical/existing-model-mass-audit.json": [("str", ABS_SIM, f"{SIM_REL}/", 1)],
    "mac/brain_sim_check.py": [("str", WHOLE_HOP, '"..", "whole"', 1), ("re", EXEC_BLOCK, EXEC_NOTE, 1),
        ("str", "# The as-built model needs the patched bam in ~/.venvs/microduck-dyn (the pip release lacks vin_drop_resistance);\n",
         "# The as-built model uses bam at the commit locked in uv.lock (the one Pollen trains with).\n", 1)],
    "mac/pickup_sim.py": [("str", WHOLE_HOP, '"..", "whole"', 1), ("re", EXEC_BLOCK, EXEC_NOTE, 1)],
    "mac/target_ground.py": [("str", WHOLE_HOP, '"..", "whole"', 1)],
    "mac/jitter-20261018/jitter_sim.py": [("re", EXEC_BLOCK, EXEC_NOTE, 1)],
    "whole/delay-repair-20260912/evaluate_delay.py": [
        ("str", "SIM / 'microduck_rl/src/mjlab_microduck/robot/microduck'", "SIM / 'src/mjlab_microduck/robot/microduck'", 1),
        ("str", "SIM / 'microduck-simulator/app/public/policies'", "SIM / 'cmu/whole/runtime-overlay/models'", 1),
        # The fork keeps Pollen's bam pin (62bd8ce: sag = vin_drop_gain in V per Nm; the private venv runs a newer bam with
        # vin_drop_resistance in Ohm). No sim-check config sets a sag term, so both read 0.0 and the models are identical.
        ("str", "vin_drop_resistance=config.get('wire_resistance',0.0),",
         "vin_drop_gain=config.get('vin_drop_gain',0.0),  # cmu/ copy: bam 62bd8ce API, see sync_from_robotics.py", 1),
    ],
    "whole/dynamics/realgap-20261014/static_gap.py": [
        ("str", "    c.update(remove_old_battery_kg=battery, **kw)\n    return c\n",
         "    c.update(remove_old_battery_kg=battery, **kw)\n"
         "    if c.get('mass_manifest') and not Path(c['mass_manifest']).is_absolute():  # cmu/ copy: paths relative to the config\n"
         "        c['mass_manifest'] = str(MODELS[model].parent / c['mass_manifest'])\n"
         "    return c\n", 1),
    ],
    "whole/dynamics/asbuilt-20261015/asbuilt.config.json": [
        ("str", f'"{ABS_SIM}{WHOLE}/dynamics/asbuilt-20261015/asbuilt-mass-manifest.json"', '"asbuilt-mass-manifest.json"', 1)],
    "whole/dynamics/v19-20261014/runs/const-bat90-sensor0-nocomp-seed0.config.json": [
        ("str", f'"{ABS_SIM}{WHOLE}/dynamics/v19-20261014/v19-mass-manifest.json"', '"../v19-mass-manifest.json"', 1)],
    "whole/dynamics/asbuilt-20261015/asbuilt-mass-manifest.json": [("str", ABS_SIM, f"{SIM_REL}/", 1)],
    "whole/dynamics/v19-20261014/v19-mass-manifest.json": [("str", ABS_SIM, f"{SIM_REL}/", 0)],
    "whole/dynamics/asbuilt-20261015/build_asbuilt_manifest.py": [("str", "/private/tmp/claude-501/", "/private/tmp/", 0)],
    # cmu/hardware copies: private absolute paths become paths relative to the robotics checkout; scratch paths are named.
    "hardware/battery-backpack/checks/release-rows-orange.json": [("str", ABS_SIM, f"{SIM_REL}/", 12)],
    "hardware/battery-backpack/cad-source/bp_checks.py": [("str", ABS_SIM, f"{SIM_REL}/", 1)],
    "hardware/battery-backpack/cad-source/concepts.py": [
        ("re", re.compile(r"Path\('/private/tmp/[^']*rev_1\.stl'\)"),
         "Path('parts/rev_1.stl')  # cmu/ copy: the REV-31-1387 STEP export lived in a scratch directory; parts/rev_31_1387_switch-local.stl is its cached copy", 1)],
    "hardware/printing/PRINT-ORIENTATION-PRACTICE.md": [("str", ABS_SIM, f"{SIM_REL}/", 2)],
    "hardware/printing/ready/V1.20/manifest-V1.20.json": [("str", ABS_SIM, f"{SIM_REL}/", 34)],
    "hardware/printing/scripts/release_pipeline_v2.py": [
        ("str", "fp=REDESIGN/'printing-elegoo-20260929/filament.json'",
         "fp=HERE.parent/'profiles/elegoo-filament.json'  # cmu/ copy: the Elegoo filament profile ships beside the script", 1)],
    "hardware/battery-backpack/POWER-PLAN.md": [
        ("re", re.compile(r" \(DigiKey order \d+\)"), "", 2), ("str", " (same order)", "", 1),
        ("str", " (same or next day to 15213)", " (same or next day to Pittsburgh)", 1)],
}
TEXT_SUFFIXES = {".py", ".json", ".md", ".sh", ".txt", ".jsonl", ".csv"}


def rewrite(dst_rel, data: bytes) -> bytes:
    rules = RULES.get(dst_rel)
    if not rules:
        return data
    text = data.decode()
    for kind, pat, rep, n_min in rules:
        if kind == "str":
            n = text.count(pat)
            text = text.replace(pat, rep)
        else:
            text, n = pat.subn(rep, text)
        if n < n_min:
            sys.exit(f"sync: rule for {dst_rel} matched {n} times (expected at least {n_min}): {pat!r}")
    return text.encode()


def missing_local_imports(sim: Path, pairs) -> list:
    """(source, module) for every import in a copied .py file of a module that sits beside it in the robotics tree but
    is not copied: the copy would fail to import (walk_wired.py imported hold.py, unsynced, from 2026-10-23)."""
    import ast
    copied = {src for src, _ in pairs}
    missing = []
    for src_rel, _ in pairs:
        if not src_rel.endswith(".py"):
            continue
        src = sim / src_rel
        for node in ast.walk(ast.parse(src.read_bytes())):
            if isinstance(node, ast.Import):
                mods = [n.name.split(".")[0] for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            else:
                continue
            for mod in mods:
                rel = f"{Path(src_rel).parent.as_posix()}/{mod}.py"
                if (sim / rel).is_file() and rel not in copied:
                    missing.append((src_rel, mod))
    return sorted(set(missing))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robotics", default=os.path.expanduser("~/code/robotics"))
    ap.add_argument("--check", action="store_true", help="report differences, write nothing")
    a = ap.parse_args()
    sim = Path(a.robotics) / SIM_REL
    if not sim.is_dir():
        sys.exit(f"sync: {sim} is not a directory; this script runs on the machine that holds the robotics tree")
    pairs = list(FILES)
    for sdir, ddir in DIR_COPIES:
        for p in sorted((sim / sdir).iterdir()):
            if p.is_file() and not p.name.startswith("."):
                pairs.append((f"{sdir}/{p.name}", f"{ddir}/{p.name}"))
    missing = missing_local_imports(sim, pairs)
    if missing:
        sys.exit("sync: copied files import local modules that are not synced (add them to FILES): "
                 + "; ".join(f"{src} imports {mod}" for src, mod in missing))
    sim_copy = simulator_files(sim)
    changed = 0
    items = [(src_rel, dst_rel, None) for src_rel, dst_rel in pairs] + \
        [(f"{SIMULATOR}/{d[len(SIM_DST) + 1:]}", d, data) for d, data in sim_copy.items()]
    for src_rel, dst_rel, data in items:
        src, dst = sim / src_rel, HERE / dst_rel
        if data is None:
            if not src.exists():
                sys.exit(f"sync: missing source {src}")
            data = rewrite(dst_rel, src.read_bytes())
        old = dst.read_bytes() if dst.exists() else None
        if old == data:
            continue
        changed += 1
        if a.check:
            if dst.suffix in TEXT_SUFFIXES and old is not None:
                diff = difflib.unified_diff(old.decode().splitlines(), data.decode().splitlines(), str(dst), str(src), lineterm="", n=1)
                print("\n".join(list(diff)[:40]))
            else:
                print(f"would write {dst_rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        print(f"wrote {dst_rel}")
    import json, zipfile
    for src_rel, member, dst_rel in EXTRACTS:
        with zipfile.ZipFile(sim / src_rel) as z:
            data = (json.dumps(json.loads(z.read(member)), indent=1, sort_keys=True) + "\n").encode()
        dst = HERE / dst_rel
        if dst.exists() and dst.read_bytes() == data:
            continue
        changed += 1
        if a.check:
            print(f"would write {dst_rel} (from {src_rel}:{member})")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        print(f"wrote {dst_rel}")
    for stale in sorted(p for p in (HERE / SIM_DST).rglob("*") if p.is_file()) if (HERE / SIM_DST).is_dir() else []:
        rel = stale.relative_to(HERE).as_posix()
        if rel in sim_copy or "/node_modules/" in f"/{rel}" or "/dist/" in f"/{rel}":
            continue
        changed += 1
        if a.check:
            print(f"would remove {rel}")
            continue
        stale.unlink()
        print(f"removed {rel}")
    unused = set(RULES) - {d for _, d in pairs}
    if unused:
        sys.exit(f"sync: rules for files that are not synced: {sorted(unused)}")
    print(f"sync: {len(pairs) + len(EXTRACTS) + len(sim_copy)} files, {changed} {'would change' if a.check else 'written'}")


if __name__ == "__main__":
    main()
