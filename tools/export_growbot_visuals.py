"""Export the approved Growbot shell pieces from robot.blend for MuJoCo.

Run with Blender in background mode.  The source .blend is never saved or
modified; meshes are evaluated with their current modifiers and written in the
MuJoCo axis convention (Blender -Y forward, X left, Z up).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import bpy
from mathutils import Vector


OBJECTS = {
    "head": ("jaw_soft", (0.94, 0.94, 0.94, 1.0)),
    "eyes": ("jaw_soft", (0.02, 0.03, 0.04, 1.0)),
    "Cylinder.002": ("jaw_soft", (0.05, 0.05, 0.06, 1.0)),
    "Cylinder.003": ("jaw_soft", (0.05, 0.05, 0.06, 1.0)),
    "Cylinder.004": ("jaw_soft", (0.05, 0.05, 0.06, 1.0)),
    "Cylinder.007": ("jaw_soft", (0.05, 0.05, 0.06, 1.0)),
    "Cylinder.008": ("jaw_soft", (0.05, 0.05, 0.06, 1.0)),
    "Cylinder.009": ("jaw_soft", (0.05, 0.05, 0.06, 1.0)),
    "Mesh_0.001": ("left_upper_arm", (0.94, 0.94, 0.94, 1.0)),
    "arm_lower.001": ("left_forearm", (0.94, 0.94, 0.94, 1.0)),
    "hand_r.001": ("left_forearm", (0.80, 0.84, 0.88, 1.0)),
    "Mesh_0.004": ("right_upper_arm", (0.94, 0.94, 0.94, 1.0)),
    "arm_lower": ("right_forearm", (0.94, 0.94, 0.94, 1.0)),
    "hand_r": ("right_forearm", (0.80, 0.84, 0.88, 1.0)),
}


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _axis_map(v: Vector) -> tuple[float, float, float]:
    return (-float(v.y), float(v.x), float(v.z))


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 1:
        raise SystemExit("usage: blender -b robot.blend --python SCRIPT -- OUTPUT_DIR")
    output_dir = Path(args[0]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    root = bpy.data.objects["trunk_base"].matrix_world.translation.copy()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    manifest: dict[str, dict] = {}

    for object_name, (body_name, rgba) in OBJECTS.items():
        obj = bpy.data.objects.get(object_name)
        if obj is None or obj.type != "MESH":
            raise RuntimeError(f"required mesh object missing: {object_name}")

        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        linear = obj.matrix_world.to_3x3()
        file_stem = _safe_name(object_name)
        output_path = output_dir / f"{file_stem}.obj"

        with output_path.open("w", encoding="utf-8") as stream:
            stream.write(f"# Exported from {bpy.data.filepath}: {object_name}\n")
            for vertex in mesh.vertices:
                x, y, z = _axis_map(linear @ vertex.co)
                stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for triangle in mesh.loop_triangles:
                a, b, c = (index + 1 for index in triangle.vertices)
                stream.write(f"f {a} {b} {c}\n")

        delta = obj.matrix_world.translation - root
        manifest[file_stem] = {
            "source_object": object_name,
            "file": output_path.name,
            "body": body_name,
            "reference_pos": list(_axis_map(delta)),
            "rgba": list(rgba),
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.loop_triangles),
        }
        evaluated.to_mesh_clear()

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(manifest)} Growbot visual meshes to {output_dir}")


if __name__ == "__main__":
    main()
