from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML
from mjlab_microduck.rom.action_catalog import ACTION_TEMPLATES
from mjlab_microduck.rom.action_specs import ACTION_RUNTIME_SPECS
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


def write_normalized_onnx(path: Path) -> Path:
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["observation", "mean"], ["centered"]),
            helper.make_node("Div", ["centered", "std"], ["normalized"]),
            helper.make_node("MatMul", ["normalized", "weights"], ["action"]),
        ],
        "microduck-normalized-test-policy",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 14])],
        [
            helper.make_tensor("mean", TensorProto.FLOAT, [61], [0.0] * 61),
            helper.make_tensor("std", TensorProto.FLOAT, [61], [1.0] * 61),
            helper.make_tensor(
                "weights", TensorProto.FLOAT, [61, 14], [0.0] * (61 * 14)
            ),
        ],
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
    slope = next(
        item for item in ACTION_TEMPLATES if item.action_code == "ROLLER_SLOPE"
    )
    assert slope.execution_mode == "CONTINUOUS_LEASE"
    assert slope.lease is not None
    assert set(ACTION_RUNTIME_SPECS) == {
        template.action_code for template in ACTION_TEMPLATES
    }
    for spec in ACTION_RUNTIME_SPECS.values():
        assert spec.required_capabilities
        assert spec.reset_profile
        assert spec.command_profile
        assert spec.fall_policy
        assert spec.metric_keys
        if not spec.supported:
            assert spec.unavailable_reason == "RUNTIME_SEMANTICS_UNSUPPORTED"
    assert ACTION_RUNTIME_SPECS["GROUND_PICK"].phase_period_s == 4.0
    assert ACTION_RUNTIME_SPECS["ROLLER_CROUCH"].phase_period_s == 5.0
    assert ACTION_RUNTIME_SPECS["SPIN"].phase_period_s == 4.0
    assert ACTION_RUNTIME_SPECS["KICK_LEFT"].kick_mirror == "LEFT_RIGHT_EXACT"
    assert "BALL_FREEJOINT" in ACTION_RUNTIME_SPECS["KICK_RIGHT"].required_capabilities


def test_actions_without_exact_runtime_scenario_semantics_remain_unavailable(
    tmp_path: Path,
):
    policy = write_minimal_onnx(tmp_path / "standup.onnx")
    bundle = build_bundle(
        minimal_request(tmp_path, artifacts={"STAND_UP": policy})
    ).manifest

    standup = next(
        action for action in bundle.actions if action.actionCode == "STAND_UP"
    )
    assert standup.availability == "UNAVAILABLE"
    assert standup.unavailableReason == "RUNTIME_SEMANTICS_UNSUPPORTED"


def test_roller_policy_is_unavailable_without_roller_model_capability(tmp_path: Path):
    policy = write_minimal_onnx(tmp_path / "roller.onnx")
    bundle = build_bundle(
        minimal_request(tmp_path, artifacts={"ROLLER_VELOCITY": policy})
    ).manifest

    roller = next(
        action for action in bundle.actions if action.actionCode == "ROLLER_VELOCITY"
    )
    assert roller.availability == "UNAVAILABLE"
    assert roller.unavailableReason == "MODEL_CAPABILITY_MISSING"


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


def test_bundle_resolves_compiler_mesh_and_texture_directories_through_includes(
    tmp_path: Path,
):
    """Ignoring an included file's compiler directories would omit deploy-time mesh and texture assets."""
    model_root = tmp_path / "model"
    (model_root / "root_meshes").mkdir(parents=True)
    (model_root / "root_textures").mkdir()
    (model_root / "nested").mkdir()
    (model_root / "included_meshes").mkdir()
    (model_root / "included_textures").mkdir()
    for relative in (
        "root_meshes/root.stl",
        "root_textures/root.png",
        "included_meshes/child.stl",
        "included_textures/child.png",
    ):
        (model_root / relative).write_bytes(relative.encode())
    (model_root / "robot.xml").write_text(
        '<mujoco><compiler meshdir="root_meshes" texturedir="root_textures"/>'
        '<include file="nested/child.xml"/><asset><mesh file="root.stl"/>'
        '<texture file="root.png"/></asset></mujoco>'
    )
    (model_root / "nested" / "child.xml").write_text(
        '<mujoco><compiler meshdir="included_meshes" texturedir="included_textures"/>'
        '<asset><mesh file="child.stl"/><texture file="child.png"/></asset></mujoco>'
    )
    built = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "compiler-paths.zip",
            artifacts={},
            model_path=model_root / "robot.xml",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
        )
    )

    with zipfile.ZipFile(built.output_zip) as archive:
        assert {
            "models/root_meshes/root.stl",
            "models/root_textures/root.png",
            "models/included_meshes/child.stl",
            "models/included_textures/child.png",
        } <= set(archive.namelist())


def test_bundle_accepts_the_default_walk_mjcf_compiler_meshdir(tmp_path: Path):
    """Resolving default CLI model assets from the XML directory would make release creation fail."""
    built = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "default-walk.zip",
            artifacts={},
            model_path=MICRODUCK_WALK_XML,
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
        )
    )

    with zipfile.ZipFile(built.output_zip) as archive:
        assert "models/assets/xl330.stl" in archive.namelist()


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
                        "jointPermutation": [
                            9,
                            10,
                            11,
                            12,
                            13,
                            5,
                            6,
                            7,
                            8,
                            0,
                            1,
                            2,
                            3,
                            4,
                        ],
                        "signFlips": [
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            1,
                            1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
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
            "jointPermutation": [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4],
            "signFlips": [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
        }
    }


@pytest.mark.parametrize(
    "transform",
    [
        None,
        {},
        {"jointPermutation": list(range(14)), "signFlips": [1] * 14},
        {"jointPermutation": [9] * 14, "signFlips": [-1] * 14},
    ],
)
def test_shared_kick_artifact_requires_the_exact_declared_mirroring_transform(
    tmp_path: Path, transform: dict[str, object] | None
):
    """Deduplicating opposite kick actions without the exact transform would execute the wrong leg motion."""
    policy = write_minimal_onnx(tmp_path / "kick.onnx")
    request = minimal_request(
        tmp_path, artifacts={"KICK_LEFT": policy, "KICK_RIGHT": policy}
    )
    if transform is not None:
        request = BundleBuildRequest(
            **(request.__dict__ | {"mirroring_transforms": {"KICK_RIGHT": transform}})
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
    assert left.availability == "UNAVAILABLE"
    assert left.unavailableReason == "RUNTIME_SEMANTICS_UNSUPPORTED"
    assert right.availability == "UNAVAILABLE"
    assert right.unavailableReason == "POLICY_ARTIFACT_MISSING"


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
    policy = write_normalized_onnx(tmp_path / "policy.onnx")
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
        "microduck.normalization": "EMPIRICAL_NORMALIZATION_V1",
        "microduck.normalization_graph_sha256": hashlib.sha256(
            graph_before
        ).hexdigest(),
    }


def test_export_metadata_refuses_graph_without_baked_normalizer(tmp_path: Path):
    policy = write_minimal_onnx(tmp_path / "policy.onnx")

    with pytest.raises(ValueError, match="empirical normalizer"):
        _export_module().attach_microduck_metadata(
            policy,
            task_id="Mjlab-Velocity-Flat-MicroDuck",
            source_commit="b" * 40,
            checkpoint="model_100.pt",
            run_identity="entity/project/run",
        )
