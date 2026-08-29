from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from mjlab_microduck.rom.action_catalog import ACTION_TEMPLATES
from mjlab_microduck.rom.bundle import BundleBuildRequest, build_bundle
from mjlab_microduck.rom.contracts import sha256_prefixed

WALK_ONNX = "walk.onnx"


def _export_module():
    script = Path(__file__).parents[1] / "scripts" / "export.py"
    spec = importlib.util.spec_from_file_location("microduck_export", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_minimal_onnx(path: Path) -> Path:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["observation"], ["action"])],
        "microduck-test-policy",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 14])],
    )
    onnx.save(helper.make_model(graph), path)
    return path


def sha256_prefixed_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def minimal_request(
    tmp_path: Path, *, artifacts: dict[str, Path]
) -> BundleBuildRequest:
    model_dir = tmp_path / "model"
    assets_dir = model_dir / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "mesh.stl").write_text("solid mesh\nendsolid mesh\n")
    (assets_dir / "texture.png").write_bytes(b"texture")
    model = model_dir / "robot.xml"
    model.write_text(
        '<mujoco><include file="extra.xml"/><asset>'
        '<mesh name="mesh" file="assets/mesh.stl"/>'
        '<texture name="texture" file="assets/texture.png"/>'
        "</asset></mujoco>"
    )
    (model_dir / "extra.xml").write_text("<mujoco/>")
    qualification = tmp_path / "qualification.txt"
    qualification.write_text("qualified\n")
    license_file = tmp_path / "LICENSE.txt"
    license_file.write_text("Apache-2.0\n")
    return BundleBuildRequest(
        release="1.0.0",
        output_zip=tmp_path / "microduck-bundle-1.0.0.zip",
        artifacts=artifacts,
        model_path=model,
        source_repository="microduck-rl",
        source_commit="a" * 40,
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
        checkpoint="model_100.pt",
        experiment_ref="mjlab_microduck/test-run",
        qualification_files=(qualification,),
        license_files=(license_file,),
    )


def test_catalog_covers_every_user_intent_once():
    """Dropping or adding an action would make the ROM's public intent catalog incomplete."""
    assert {action.action_code for action in ACTION_TEMPLATES} == {
        "WALK_VELOCITY",
        "VELSTAND_VELOCITY",
        "ROLLER_VELOCITY",
        "SWIZZLE",
        "ROLLER_SLOPE",
        "STAND_UP",
        "SIT",
        "STAND",
        "GROUND_PICK",
        "KICK_LEFT",
        "KICK_RIGHT",
        "ROULADE",
        "ROLLER_CROUCH",
        "ROLLER_STAND_UP",
        "SPIN",
    }


def test_missing_artifact_is_explicitly_unavailable(tmp_path: Path):
    """Treating a missing policy as available could send ROM to an undefined artifact."""
    policy = write_minimal_onnx(tmp_path / WALK_ONNX)
    bundle = build_bundle(
        minimal_request(tmp_path, artifacts={"WALK_VELOCITY": policy})
    )

    spin = next(
        action for action in bundle.manifest.actions if action.actionCode == "SPIN"
    )
    assert spin.availability == "UNAVAILABLE"
    assert spin.unavailableReason == "POLICY_ARTIFACT_MISSING"


def test_bundle_digest_uses_unsigned_manifest_and_declared_artifact_hashes(
    tmp_path: Path,
):
    """Including bundleDigest in its own hash would make the immutable manifest unverifiable."""
    policy = write_minimal_onnx(tmp_path / WALK_ONNX)
    built = build_bundle(minimal_request(tmp_path, artifacts={"WALK_VELOCITY": policy}))
    unsigned_manifest = built.manifest.model_dump(
        mode="json", by_alias=True, exclude={"bundleDigest"}
    )

    assert built.manifest.bundleDigest == sha256_prefixed(
        {"manifest": unsigned_manifest, "artifacts": built.artifact_digests}
    )


def test_bundle_contains_complete_declared_model_and_supporting_file_closure(
    tmp_path: Path,
):
    """Omitting an MJCF include, mesh, texture, qualification, or license breaks offline replay."""
    policy = write_minimal_onnx(tmp_path / WALK_ONNX)
    built = build_bundle(minimal_request(tmp_path, artifacts={"WALK_VELOCITY": policy}))

    with zipfile.ZipFile(built.output_zip) as archive:
        paths = archive.namelist()
        assert paths == sorted(paths)
        assert len(paths) == len(set(paths))
        assert all(not Path(path).is_absolute() and "\\" not in path for path in paths)
        assert "microduck-policy-bundle.json" in paths
        assert any(
            path.startswith("policies/") and path.endswith(".onnx") for path in paths
        )
        assert {
            "models/robot.xml",
            "models/extra.xml",
            "models/assets/mesh.stl",
            "models/assets/texture.png",
        } <= set(paths)
        assert any(path.startswith("qualification/") for path in paths)
        assert any(path.startswith("licenses/") for path in paths)
        manifest = json.loads(archive.read("microduck-policy-bundle.json"))

    declared_paths = {
        manifest["model"]["path"],
        *(policy["path"] for policy in manifest["policies"]),
        *(artifact["path"] for artifact in manifest["qualification"]["artifacts"]),
        *(artifact["path"] for artifact in manifest["license"]["artifacts"]),
    }
    declared_paths.update(
        item["path"] for item in manifest["qualification"]["modelClosure"]
    )
    assert set(paths) == {"microduck-policy-bundle.json", *declared_paths}
    assert manifest["model"]["digest"] == sha256_prefixed_file(
        tmp_path / "model" / "robot.xml"
    )


def test_bundle_zip_is_byte_deterministic_for_a_fixed_request(tmp_path: Path):
    """Using archive clock or host metadata would make identical releases produce different evidence."""
    policy = write_minimal_onnx(tmp_path / WALK_ONNX)
    first = minimal_request(tmp_path / "one", artifacts={"WALK_VELOCITY": policy})
    second = minimal_request(tmp_path / "two", artifacts={"WALK_VELOCITY": policy})

    first_bundle = build_bundle(first)
    second_bundle = build_bundle(second)

    assert first_bundle.output_zip.read_bytes() == second_bundle.output_zip.read_bytes()
    assert first_bundle.manifest == second_bundle.manifest
    assert first_bundle.artifact_digests == second_bundle.artifact_digests


def test_existing_bundle_is_not_overwritten(tmp_path: Path):
    """Overwriting an existing release archive would destroy immutable release evidence."""
    policy = write_minimal_onnx(tmp_path / WALK_ONNX)
    request = minimal_request(tmp_path, artifacts={"WALK_VELOCITY": policy})
    build_bundle(request)

    import pytest

    with pytest.raises(FileExistsError, match="bundle output already exists"):
        build_bundle(request)


def test_declared_kick_mirror_can_reuse_only_the_named_opposite_side(tmp_path: Path):
    """A missing kick side must not silently reuse the other side without its exact transform."""
    policy = write_minimal_onnx(tmp_path / "kick.onnx")
    request = minimal_request(tmp_path, artifacts={"KICK_LEFT": policy})
    request = BundleBuildRequest(
        **(
            request.__dict__
            | {
                "mirroring_transforms": {
                    "KICK_RIGHT": {
                        "jointPermutation": list(range(14)),
                        "signFlips": [1] * 14,
                    }
                }
            }
        )
    )
    bundle = build_bundle(request)

    left = next(
        action for action in bundle.manifest.actions if action.actionCode == "KICK_LEFT"
    )
    right = next(
        action
        for action in bundle.manifest.actions
        if action.actionCode == "KICK_RIGHT"
    )
    assert right.policyRef == left.policyRef
    assert right.safety == {
        "mirroringTransform": {
            "jointPermutation": list(range(14)),
            "signFlips": [1] * 14,
        }
    }


def test_sit_and_stand_can_share_the_same_sitstand_policy_artifact(tmp_path: Path):
    """Duplicating a shared SitStand ONNX in the manifest would defeat content-addressed artifact identity."""
    policy = write_minimal_onnx(tmp_path / "sitstand.onnx")
    bundle = build_bundle(
        minimal_request(tmp_path, artifacts={"SIT": policy, "STAND": policy})
    )

    sit = next(
        action for action in bundle.manifest.actions if action.actionCode == "SIT"
    )
    stand = next(
        action for action in bundle.manifest.actions if action.actionCode == "STAND"
    )
    assert sit.policyRef == stand.policyRef
    assert len(bundle.manifest.policies) == 1


def test_bundle_cli_writes_release_archive_from_named_artifact(tmp_path: Path):
    """Breaking argument parsing would prevent a trained policy from becoming a verifiable release."""
    policy = write_minimal_onnx(tmp_path / WALK_ONNX)
    request = minimal_request(tmp_path, artifacts={})
    output = tmp_path / "cli.zip"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "scripts/build_rom_bundle.py",
            "--release",
            "1.0.0",
            "--artifact",
            f"WALK_VELOCITY={policy}",
            "--model",
            str(request.model_path),
            "--source-repository",
            "microduck-rl",
            "--source-commit",
            "a" * 40,
            "--created-at",
            "2026-08-29T00:00:00Z",
            "--qualification-file",
            str(request.qualification_files[0]),
            "--license-file",
            str(request.license_files[0]),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()


def test_export_metadata_preserves_baked_normalizer_graph(tmp_path: Path):
    """Replacing the exported ONNX graph while adding metadata would discard baked normalization."""
    policy = write_minimal_onnx(tmp_path / "policy.onnx")
    graph_before = onnx.load(policy).graph.SerializeToString()

    _export_module().attach_microduck_metadata(
        policy,
        task_id="Mjlab-Velocity-Flat-MicroDuck",
        source_commit="b" * 40,
        checkpoint="model_100.pt",
        run_identity="entity/project/run",
    )

    exported = onnx.load(policy)
    assert exported.graph.SerializeToString() == graph_before
    properties = {item.key: item.value for item in exported.metadata_props}
    assert properties == {
        "microduck.task_id": "Mjlab-Velocity-Flat-MicroDuck",
        "microduck.source_commit": "b" * 40,
        "microduck.observation_contract": "MICRODUCK_OBS_61_V1",
        "microduck.action_contract": "MICRODUCK_ACTION_14_V1",
        "microduck.checkpoint": "model_100.pt",
        "microduck.run_identity": "entity/project/run",
    }
