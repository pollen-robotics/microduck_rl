"""Scene assets owned by the optional visualization/simulation extension."""

from pathlib import Path

SCENES = Path(__file__).resolve().parent
VELSTAND_SCENE = SCENES / "scene_velstand.xml"


def resolve_scene(scene: Path) -> Path:
    """Resolve an extension scene name without leaving a file in robot assets.

    Explicit existing paths always win. The extension's basename remains usable
    by older launch configurations that supplied the former robot-directory path.
    """
    scene = Path(scene)
    if scene.is_file():
        return scene
    candidate = SCENES / scene.name
    return candidate if candidate.is_file() else scene
