from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML
from mjlab_microduck.rom.action_catalog import (
    ACTION_TEMPLATES,
    CODE_OWNED_ACTION_CODES,
    validate_bundle_action_envelope,
)
from mjlab_microduck.rom.action_specs import ACTION_RUNTIME_SPECS
from mjlab_microduck.rom.bundle import BundleBuildRequest, build_bundle
from mjlab_microduck.rom.contracts import PolicyBundle, sha256_prefixed
from mjlab_microduck.rom.main import load_verified_bundle

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
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
        ),
        path,
    )
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
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
        ),
        path,
    )
    return path


def write_bypassed_normalizer_onnx(path: Path) -> Path:
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["observation", "mean"], ["centered"]),
            helper.make_node("Div", ["centered", "std"], ["normalized"]),
            helper.make_node("MatMul", ["observation", "weights"], ["action"]),
        ],
        "microduck-bypassed-normalizer-policy",
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
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    metadata = {
        "microduck.task_id": "Mjlab-Velocity-Flat-MicroDuck",
        "microduck.source_commit": "a" * 40,
        "microduck.observation_contract": "MICRODUCK_OBS_61_V1",
        "microduck.action_contract": "MICRODUCK_ACTION_14_V1",
        "microduck.checkpoint": "model_100.pt",
        "microduck.run_identity": "mjlab_microduck/test-run",
        "microduck.normalization": "EMPIRICAL_NORMALIZATION_V1",
        "microduck.normalization_graph_sha256": hashlib.sha256(
            graph.SerializeToString()
        ).hexdigest(),
    }
    for key, value in sorted(metadata.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(model, path)
    return path


def write_release_onnx(path: Path, *, task_id: str) -> Path:
    write_normalized_onnx(path)
    _export_module().attach_microduck_metadata(
        path,
        task_id=task_id,
        source_commit="a" * 40,
        checkpoint="model_100.pt",
        run_identity="mjlab_microduck/test-run",
    )
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


def test_builder_emits_the_complete_code_owned_action_envelope(tmp_path: Path) -> None:
    """The builder must not make manifest data authoritative for any V1 safety field."""
    policy = write_minimal_onnx(tmp_path / WALK_ONNX)

    manifest = build_bundle(
        minimal_request(tmp_path, artifacts={"WALK_VELOCITY": policy})
    ).manifest

    assert tuple(action.actionCode for action in manifest.actions) == (
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
    )
    assert (
        tuple(action.actionCode for action in manifest.actions)
        == CODE_OWNED_ACTION_CODES
    )
    validate_bundle_action_envelope(manifest)


def test_candidate_loader_rejects_a_resigned_widened_action_envelope(
    tmp_path: Path,
) -> None:
    """Re-hashing a WALK schema change must not make a builder candidate trusted."""
    policy = write_minimal_onnx(tmp_path / WALK_ONNX)
    built = build_bundle(minimal_request(tmp_path, artifacts={"WALK_VELOCITY": policy}))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(built.output_zip) as archive:
        archive.extractall(installed)
    manifest_path = installed / "microduck-policy-bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["actions"][0]["parameterSchema"]["properties"]["vxMps"]["maximum"] = (
        1_000.0
    )
    manifest["bundleDigest"] = None
    artifact_digests = {
        item["path"]: item["digest"]
        for item in [
            manifest["model"],
            *manifest["policies"],
            *manifest["qualification"].get("artifacts", []),
            *manifest["qualification"].get("modelClosure", []),
            *manifest["license"].get("artifacts", []),
        ]
    }
    manifest["bundleDigest"] = sha256_prefixed(
        {
            "manifest": PolicyBundle.model_validate(manifest).model_dump(
                mode="json", by_alias=True, exclude={"bundleDigest"}
            ),
            "artifacts": artifact_digests,
        }
    )
    manifest_path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )

    with pytest.raises(ValueError, match="code-owned V1 action"):
        load_verified_bundle(installed)


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
    assert roller.unavailableReason == "MODEL_QUALIFICATION_INCOMPATIBLE"


def test_robot_only_mjcf_cannot_qualify_an_action_as_executable(tmp_path: Path):
    """A robot asset without a floor is not an executable deployment scene."""
    policy = write_release_onnx(
        tmp_path / WALK_ONNX, task_id="Mjlab-Velocity-Flat-MicroDuck"
    )
    request = minimal_request(tmp_path, artifacts={"WALK_VELOCITY": policy})
    qualified = BundleBuildRequest(
        **(
            request.__dict__
            | {
                "model_terrain": "flat",
                "scenario_profile": "SEEDED_SERVO_RESET_V1",
            }
        )
    )

    bundle = build_bundle(qualified).manifest

    walk = next(item for item in bundle.actions if item.actionCode == "WALK_VELOCITY")
    assert walk.availability == "UNAVAILABLE"
    assert walk.unavailableReason == "MODEL_QUALIFICATION_INCOMPATIBLE"
    assert "modelTerrain" not in bundle.qualification


def test_real_scene_floor_emits_qualified_terrain_and_available_walk(tmp_path: Path):
    """Qualification must name terrain only after the packaged scene proves it materializes it."""
    policy = write_release_onnx(
        tmp_path / WALK_ONNX, task_id="Mjlab-Velocity-Flat-MicroDuck"
    )
    scene = MICRODUCK_WALK_XML.with_name("scene_walk.xml")

    built = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "qualified-scene.zip",
            artifacts={"WALK_VELOCITY": policy},
            model_path=scene,
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    )

    assert built.manifest.qualification["modelTerrain"] == "flat"
    assert built.manifest.qualification["scenarioProfile"] == "SEEDED_SERVO_RESET_V1"
    walk = next(
        item for item in built.manifest.actions if item.actionCode == "WALK_VELOCITY"
    )
    assert walk.availability == "AVAILABLE"
    policy_artifact = next(
        item for item in built.manifest.policies if item.policyRef == walk.policyRef
    )
    assert policy_artifact.runtimeRequirements["normalizedGraphFingerprint"].startswith(
        "sha256:"
    )
    assert walk.preconditions is not None
    assert walk.preconditions["allowedTerrains"] == ["flat"]
    assert walk.preconditions["scenarioProfile"] == "SEEDED_SERVO_RESET_V1"
    assert walk.preconditions["requiredCapabilities"] == ["FLAT_TERRAIN"]


@pytest.mark.parametrize(
    "floor_attributes",
    [
        'pos="0 0 0" quat="0.70710678 0.70710678 0 0"',
        'pos="0 0 2"',
        'pos="0 0 0" contype="4" conaffinity="4"',
    ],
)
def test_builder_rejects_unusable_or_collision_incompatible_flat_floor(
    tmp_path: Path, floor_attributes: str
) -> None:
    """A declared flat terrain must be a reachable horizontal contact surface."""
    model_dir = tmp_path / "model"
    shutil.copytree(MICRODUCK_WALK_XML.parent, model_dir)
    scene = model_dir / "scene_walk.xml"
    scene.write_text(
        scene.read_text().replace(
            'pos="0 0 0" type="plane" material="groundplane"',
            f'{floor_attributes} type="plane" material="groundplane"',
        )
    )
    policy = write_release_onnx(
        tmp_path / WALK_ONNX, task_id="Mjlab-Velocity-Flat-MicroDuck"
    )

    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "unusable-floor.zip",
            artifacts={"WALK_VELOCITY": policy},
            model_path=scene,
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    walk = next(item for item in bundle.actions if item.actionCode == "WALK_VELOCITY")
    assert walk.availability == "UNAVAILABLE"
    assert walk.unavailableReason == "MODEL_QUALIFICATION_INCOMPATIBLE"
    assert "modelTerrain" not in bundle.qualification


def test_builder_rejects_roller_masks_incompatible_with_usable_floor(
    tmp_path: Path,
) -> None:
    """Fake ankle fixtures cannot replace floor-incompatible wheel contacts."""
    model_dir = tmp_path / "model"
    shutil.copytree(MICRODUCK_WALK_XML.parent, model_dir)
    robot = model_dir / "robot_allcollisions_rollers.xml"
    rewritten: list[str] = []
    for line in robot.read_text().splitlines():
        rewritten.append(
            line.replace(
                'class="collision"',
                'class="collision" contype="4" conaffinity="4"',
            )
            if 'mesh="tire"' in line and 'class="collision"' in line
            else line
        )
        for ankle in ("left_ankle", "right_ankle"):
            if "<joint" in line and f'name="{ankle}"' in line:
                rewritten.append(
                    f'<geom name="{ankle}_contact_fixture" type="sphere" '
                    'size="0.002" mass="0.001" contype="1" conaffinity="1"/>'
                )
    robot.write_text("\n".join(rewritten))
    policy = write_release_onnx(
        tmp_path / "roller.onnx",
        task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    )

    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "incompatible-rollers.zip",
            artifacts={"ROLLER_VELOCITY": policy},
            model_path=model_dir / "scene_rollers.xml",
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    roller = next(
        item for item in bundle.actions if item.actionCode == "ROLLER_VELOCITY"
    )
    assert roller.availability == "UNAVAILABLE"
    assert roller.unavailableReason == "MODEL_QUALIFICATION_INCOMPATIBLE"


def test_identical_policy_bytes_do_not_cross_deduplicate_task_identities(
    tmp_path: Path,
):
    """Digest equality cannot make a Walk actor valid for the distinct VelStand task."""
    policy = write_release_onnx(
        tmp_path / WALK_ONNX, task_id="Mjlab-Velocity-Flat-MicroDuck"
    )
    scene = MICRODUCK_WALK_XML.with_name("scene_walk.xml")
    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "identity.zip",
            artifacts={
                "WALK_VELOCITY": policy,
                "VELSTAND_VELOCITY": policy,
            },
            model_path=scene,
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    walk = next(item for item in bundle.actions if item.actionCode == "WALK_VELOCITY")
    velstand = next(
        item for item in bundle.actions if item.actionCode == "VELSTAND_VELOCITY"
    )
    assert walk.policyRef != velstand.policyRef
    assert len(bundle.policies) == 2
    assert len({item.path for item in bundle.policies}) == 2
    assert walk.availability == "AVAILABLE"
    assert velstand.availability == "UNAVAILABLE"
    assert velstand.unavailableReason == "POLICY_TASK_ID_MISMATCH"


def test_roller_mesh_and_joint_names_without_collision_topology_do_not_qualify(
    tmp_path: Path,
):
    """Names alone cannot prove that four passive wheels physically contact the terrain."""
    model_dir = tmp_path / "model"
    shutil.copytree(MICRODUCK_WALK_XML.parent, model_dir)
    robot = model_dir / MICRODUCK_WALK_XML.name
    rewritten: list[str] = []
    wheel_names = {
        "left_ankle": ("passive_LF_wheel", "passive_LR_wheel"),
        "right_ankle": ("passive_RF_wheel", "passive_RR_wheel"),
    }
    for line in robot.read_text().splitlines():
        rewritten.append(line)
        for ankle, wheels in wheel_names.items():
            if "<joint" in line and f'name="{ankle}"' in line:
                rewritten.extend(
                    f'<body name="name_only_{wheel}"><joint name="{wheel}" '
                    'type="hinge"/><geom type="sphere" size="0.002" mass="0.001" '
                    'contype="0" conaffinity="0"/></body>'
                    for wheel in wheels
                )
    robot.write_text("\n".join(rewritten))
    policy = write_release_onnx(
        tmp_path / "roller.onnx",
        task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    )
    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "name-only.zip",
            artifacts={"ROLLER_VELOCITY": policy},
            model_path=model_dir / "scene_walk.xml",
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    roller = next(
        item for item in bundle.actions if item.actionCode == "ROLLER_VELOCITY"
    )
    assert roller.availability == "UNAVAILABLE"
    assert roller.unavailableReason == "MODEL_CAPABILITY_MISSING"


def test_checked_in_roller_scene_qualifies_exact_passive_wheel_topology(
    tmp_path: Path,
):
    policy = write_release_onnx(
        tmp_path / "roller.onnx",
        task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    )
    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "rollers.zip",
            artifacts={"ROLLER_VELOCITY": policy},
            model_path=MICRODUCK_WALK_XML.with_name("scene_rollers.xml"),
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    roller = next(
        item for item in bundle.actions if item.actionCode == "ROLLER_VELOCITY"
    )
    assert roller.availability == "AVAILABLE"
    assert "ROLLER_FEET" in bundle.qualification["modelCapabilities"]


def test_builder_rejects_tiny_ankle_fixtures_when_intended_feet_are_disabled(
    tmp_path: Path,
) -> None:
    """Arbitrary ankle descendants cannot impersonate the checked-in sole surfaces."""
    model_dir = tmp_path / "model"
    shutil.copytree(MICRODUCK_WALK_XML.parent, model_dir)
    robot = model_dir / MICRODUCK_WALK_XML.name
    rewritten: list[str] = []
    for line in robot.read_text().splitlines():
        if (
            'name="left_foot_collision"' in line
            or 'name="right_foot_collision"' in line
        ):
            line = line.replace(
                'class="collision"',
                'class="collision" contype="0" conaffinity="0"',
            )
        rewritten.append(line)
        if "<joint" in line:
            for ankle in ("left_ankle", "right_ankle"):
                if f'name="{ankle}"' in line:
                    rewritten.append(
                        f'<geom name="spoof_{ankle}_contact" type="sphere" '
                        'size="0.0001" mass="0.0001" '
                        'contype="1" conaffinity="1"/>'
                    )
    robot.write_text("\n".join(rewritten))
    policy = write_release_onnx(
        tmp_path / WALK_ONNX, task_id="Mjlab-Velocity-Flat-MicroDuck"
    )

    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "spoofed-feet.zip",
            artifacts={"WALK_VELOCITY": policy},
            model_path=model_dir / "scene_walk.xml",
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    walk = next(item for item in bundle.actions if item.actionCode == "WALK_VELOCITY")
    assert walk.availability == "UNAVAILABLE"
    assert walk.unavailableReason == "MODEL_QUALIFICATION_INCOMPATIBLE"


def test_builder_rejects_tiny_mesh_spoofing_canonical_foot_names(
    tmp_path: Path,
) -> None:
    """Canonical geom/mesh names still require the checked-in physical scale."""
    model_dir = tmp_path / "model"
    shutil.copytree(MICRODUCK_WALK_XML.parent, model_dir)
    robot = model_dir / MICRODUCK_WALK_XML.name
    robot.write_text(
        robot.read_text()
        .replace(
            '<mesh file="sole_left.stl"/>',
            '<mesh file="sole_left.stl" scale="0.1 0.1 0.1"/>',
        )
        .replace(
            '<mesh file="sole_right.stl"/>',
            '<mesh file="sole_right.stl" scale="0.1 0.1 0.1"/>',
        )
    )
    policy = write_release_onnx(
        tmp_path / WALK_ONNX, task_id="Mjlab-Velocity-Flat-MicroDuck"
    )

    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "tiny-named-feet.zip",
            artifacts={"WALK_VELOCITY": policy},
            model_path=model_dir / "scene_walk.xml",
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    walk = next(item for item in bundle.actions if item.actionCode == "WALK_VELOCITY")
    assert walk.availability == "UNAVAILABLE"
    assert walk.unavailableReason == "MODEL_QUALIFICATION_INCOMPATIBLE"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("extra_passive_wheel", "MODEL_CAPABILITY_MISSING"),
        ("renamed_wheel_body", "MODEL_QUALIFICATION_INCOMPATIBLE"),
        ("tiny_wheel_mesh", "MODEL_QUALIFICATION_INCOMPATIBLE"),
    ],
)
def test_builder_rejects_noncanonical_complete_roller_topology(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    """The four code-owned wheel joints, bodies, and no extra wheels are exact."""
    model_dir = tmp_path / "model"
    shutil.copytree(MICRODUCK_WALK_XML.parent, model_dir)
    robot = model_dir / "robot_allcollisions_rollers.xml"
    contents = robot.read_text()
    if mutation == "extra_passive_wheel":
        target = '<joint axis="0 0 1" name="passive_LF_wheel" class="passive_joint"/>'
        contents = contents.replace(
            target,
            target + '<body name="spare_tire"><joint name="passive_spare_wheel" '
            'type="hinge"/><geom type="cylinder" size="0.001 0.001"/></body>',
        )
    elif mutation == "renamed_wheel_body":
        contents = contents.replace(
            '<body name="tire" pos=', '<body name="fake_tire" pos='
        )
    else:
        contents = contents.replace(
            '<mesh file="tire.stl"/>',
            '<mesh file="tire.stl" scale="0.1 0.1 0.1"/>',
        )
    robot.write_text(contents)
    policy = write_release_onnx(
        tmp_path / "roller.onnx",
        task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    )

    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / f"{mutation}.zip",
            artifacts={"ROLLER_VELOCITY": policy},
            model_path=model_dir / "scene_rollers.xml",
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    roller = next(
        item for item in bundle.actions if item.actionCode == "ROLLER_VELOCITY"
    )
    assert roller.availability == "UNAVAILABLE"
    assert roller.unavailableReason == reason


def test_floor_scene_without_exact_position_actuator_contract_is_unavailable(
    tmp_path: Path,
):
    """A scene can be executable MuJoCo while still being incompatible with 14D radian targets."""
    model_dir = tmp_path / "model"
    shutil.copytree(MICRODUCK_WALK_XML.parent, model_dir)
    robot = model_dir / MICRODUCK_WALK_XML.name
    robot.write_text(
        robot.read_text().replace(
            '<position class="chosen_actuator"',
            '<motor class="chosen_actuator" gear="1"',
        )
    )
    policy = write_release_onnx(
        tmp_path / WALK_ONNX, task_id="Mjlab-Velocity-Flat-MicroDuck"
    )
    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "no-servos.zip",
            artifacts={"WALK_VELOCITY": policy},
            model_path=model_dir / "scene_walk.xml",
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    walk = next(item for item in bundle.actions if item.actionCode == "WALK_VELOCITY")
    assert walk.availability == "UNAVAILABLE"
    assert walk.unavailableReason == "MODEL_RUNTIME_INCOMPATIBLE"


@pytest.mark.parametrize("bias", ["0.25 -0.55 0", "0 -0.55 0.25"])
def test_builder_rejects_non_position_affine_actuator_terms(
    tmp_path: Path, bias: str
) -> None:
    """Constant or velocity force terms make controls more than radian targets."""
    source_dir = MICRODUCK_WALK_XML.parent
    model_dir = tmp_path / "model"
    shutil.copytree(source_dir, model_dir)
    robot = model_dir / MICRODUCK_WALK_XML.name
    robot.write_text(
        robot.read_text().replace(
            '<position class="chosen_actuator" name=',
            '<general class="chosen_actuator" gaintype="fixed" '
            f'biastype="affine" gainprm="0.55" biasprm="{bias}" name=',
        )
    )
    policy = write_release_onnx(
        tmp_path / WALK_ONNX, task_id="Mjlab-Velocity-Flat-MicroDuck"
    )

    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "invalid-affine.zip",
            artifacts={"WALK_VELOCITY": policy},
            model_path=model_dir / "scene_walk.xml",
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    walk = next(item for item in bundle.actions if item.actionCode == "WALK_VELOCITY")
    assert walk.availability == "UNAVAILABLE"
    assert walk.unavailableReason == "MODEL_RUNTIME_INCOMPATIBLE"


@pytest.mark.parametrize(
    "replacement",
    [
        '<position class="chosen_actuator" kp="inf" name=',
        (
            '<general class="chosen_actuator" gaintype="fixed" '
            'biastype="affine" gainprm="0.55" biasprm="0 -0.55 -inf" name='
        ),
    ],
)
def test_builder_rejects_non_finite_position_actuator_semantics(
    tmp_path: Path, replacement: str
) -> None:
    """Infinite gains or bias terms cannot qualify as physical radian servos."""
    model_dir = tmp_path / "model"
    shutil.copytree(MICRODUCK_WALK_XML.parent, model_dir)
    robot = model_dir / MICRODUCK_WALK_XML.name
    robot.write_text(
        robot.read_text().replace(
            '<position class="chosen_actuator" name=', replacement
        )
    )
    policy = write_release_onnx(
        tmp_path / WALK_ONNX, task_id="Mjlab-Velocity-Flat-MicroDuck"
    )

    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "non-finite-actuator.zip",
            artifacts={"WALK_VELOCITY": policy},
            model_path=model_dir / "scene_walk.xml",
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    walk = next(item for item in bundle.actions if item.actionCode == "WALK_VELOCITY")
    assert walk.availability == "UNAVAILABLE"
    assert walk.unavailableReason == "MODEL_RUNTIME_INCOMPATIBLE"


def test_builder_marks_dead_or_bypassed_normalizer_graph_unavailable(tmp_path: Path):
    """Export-looking metadata cannot make a normalization prefix attest itself."""
    model = MICRODUCK_WALK_XML.with_name("scene_walk.xml")
    policy = write_bypassed_normalizer_onnx(tmp_path / WALK_ONNX)
    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "bypassed.zip",
            artifacts={"WALK_VELOCITY": policy},
            model_path=model,
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    walk = next(item for item in bundle.actions if item.actionCode == "WALK_VELOCITY")
    assert walk.availability == "UNAVAILABLE"
    assert walk.unavailableReason == "POLICY_NORMALIZATION_INVALID"


def test_builder_marks_non_finite_cpu_policy_inference_unavailable(tmp_path: Path):
    """A structurally normalized actor is still not executable if its dry output is NaN."""
    policy = write_normalized_onnx(tmp_path / WALK_ONNX)
    model = onnx.load(policy)
    weights = next(item for item in model.graph.initializer if item.name == "weights")
    weights.float_data[0] = float("nan")
    onnx.save(model, policy)
    _export_module().attach_microduck_metadata(
        policy,
        task_id="Mjlab-Velocity-Flat-MicroDuck",
        source_commit="a" * 40,
        checkpoint="model_100.pt",
        run_identity="mjlab_microduck/test-run",
    )
    bundle = build_bundle(
        BundleBuildRequest(
            release="1.0.0",
            output_zip=tmp_path / "nan.zip",
            artifacts={"WALK_VELOCITY": policy},
            model_path=MICRODUCK_WALK_XML.with_name("scene_walk.xml"),
            model_terrain="flat",
            scenario_profile="SEEDED_SERVO_RESET_V1",
            source_repository="microduck-rl",
            source_commit="a" * 40,
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
            checkpoint="model_100.pt",
            experiment_ref="mjlab_microduck/test-run",
        )
    ).manifest

    walk = next(item for item in bundle.actions if item.actionCode == "WALK_VELOCITY")
    assert walk.availability == "UNAVAILABLE"
    assert walk.unavailableReason == "POLICY_INFERENCE_INVALID"


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
    assert right.safety is not None
    assert right.safety["mirroringTransform"] == {
        "jointPermutation": [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4],
        "signFlips": [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
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
    """The CLI default must package the executable walk scene with explicit qualification."""
    policy = write_release_onnx(
        tmp_path / WALK_ONNX, task_id="Mjlab-Velocity-Flat-MicroDuck"
    )
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
            "--terrain",
            "flat",
            "--scenario-profile",
            "SEEDED_SERVO_RESET_V1",
            "--source-repository",
            "microduck-rl",
            "--source-commit",
            "a" * 40,
            "--created-at",
            "2026-08-29T00:00:00Z",
            "--checkpoint",
            "model_100.pt",
            "--experiment-ref",
            "mjlab_microduck/test-run",
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
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("microduck-policy-bundle.json"))
    assert manifest["model"]["path"] == "models/scene_walk.xml"
    assert manifest["qualification"]["modelTerrain"] == "flat"
    walk = next(
        item for item in manifest["actions"] if item["actionCode"] == "WALK_VELOCITY"
    )
    assert walk["availability"] == "AVAILABLE"


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
        "microduck.normalized_graph_fingerprint": (
            "sha256:81402e65d7faf69d346f1e8b9c4a0346a2de4ce01dec402aeed5fb96be333c73"
        ),
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


def test_export_metadata_refuses_dead_normalizer_that_actor_output_bypasses(
    tmp_path: Path,
):
    policy = write_bypassed_normalizer_onnx(tmp_path / "policy.onnx")

    with pytest.raises(ValueError, match="normalizer"):
        _export_module().attach_microduck_metadata(
            policy,
            task_id="Mjlab-Velocity-Flat-MicroDuck",
            source_commit="b" * 40,
            checkpoint="model_100.pt",
            run_identity="entity/project/run",
        )
