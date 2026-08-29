"""Immutable, reproducible MicroDuck ROM policy bundle builder."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import mujoco
import numpy as np
import onnx
import onnxruntime as ort

from .action_catalog import ACTION_TEMPLATES, code_owned_action_definition
from .action_specs import ACTION_RUNTIME_SPECS
from .contracts import (
    ACTION_CONTRACT,
    OBSERVATION_CONTRACT,
    ActionContract,
    ActionDefinition,
    ModelArtifact,
    ObservationContract,
    PolicyArtifact,
    PolicyBundle,
    UnsignedPolicyBundleManifest,
    publish_policy_bundle,
)
from .mirroring import (
    MICRODUCK_JOINT_MIRROR_PERMUTATION,
    MICRODUCK_JOINT_MIRROR_SIGNS,
)
from .model_semantics import (
    has_exact_deployment_frames,
    has_exact_passive_roller_topology,
    has_exact_position_actuator_topology,
    has_flat_world_floor,
)
from .onnx_policy import inspect_normalized_actor


@dataclass(frozen=True)
class BundleBuildRequest:
    release: str
    output_zip: Path
    artifacts: Mapping[str, Path]
    model_path: Path
    source_repository: str
    source_commit: str
    created_at: datetime
    model_terrain: str | None = None
    scenario_profile: str | None = None
    checkpoint: str | None = None
    experiment_ref: str | None = None
    qualification_files: tuple[Path, ...] = ()
    license_files: tuple[Path, ...] = ()
    mirroring_transforms: Mapping[str, Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class BuiltBundle:
    manifest: PolicyBundle
    output_zip: Path
    artifact_digests: dict[str, str]


@dataclass(frozen=True)
class _AssetDirectories:
    mesh_dir: Path
    texture_dir: Path


def _file_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _archive_path(prefix: str, source: Path, root: Path) -> str:
    try:
        relative = source.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"bundle source must remain inside its declared root: {source}"
        ) from exc
    return str(PurePosixPath(prefix, *relative.parts))


def _compiler_asset_directories(
    model_root: Path, tree: ET.ElementTree, inherited: _AssetDirectories
) -> _AssetDirectories:
    compiler = tree.getroot().find("compiler")
    if compiler is None:
        return inherited
    mesh_dir = compiler.get("meshdir")
    texture_dir = compiler.get("texturedir")
    return _AssetDirectories(
        mesh_dir=(model_root / mesh_dir).resolve()
        if mesh_dir is not None
        else inherited.mesh_dir,
        texture_dir=(model_root / texture_dir).resolve()
        if texture_dir is not None
        else inherited.texture_dir,
    )


def _is_exact_kick_mirroring_transform(transform: Mapping[str, Any] | None) -> bool:
    return transform is not None and dict(transform) == {
        "jointPermutation": list(MICRODUCK_JOINT_MIRROR_PERMUTATION),
        "signFlips": list(MICRODUCK_JOINT_MIRROR_SIGNS),
    }


def _model_closure(model_path: Path) -> list[Path]:
    root = model_path.parent.resolve()
    initial_directories = _AssetDirectories(mesh_dir=root, texture_dir=root)
    pending = [(model_path.resolve(), initial_directories)]
    closure: set[Path] = set()
    seen: set[tuple[Path, _AssetDirectories]] = set()
    while pending:
        source, inherited_directories = pending.pop()
        context = (source, inherited_directories)
        if context in seen:
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        _archive_path("models", source, root)
        seen.add(context)
        closure.add(source)
        if source.suffix.lower() != ".xml":
            continue
        tree = ET.parse(source)
        directories = _compiler_asset_directories(root, tree, inherited_directories)
        for element in tree.iter():
            referenced = element.get("file")
            if not referenced:
                continue
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "include":
                target = (source.parent / referenced).resolve()
            elif tag == "mesh":
                target = (directories.mesh_dir / referenced).resolve()
            elif tag == "texture":
                target = (directories.texture_dir / referenced).resolve()
            else:
                target = (source.parent / referenced).resolve()
            _archive_path("models", target, root)
            if not target.is_file():
                raise FileNotFoundError(target)
            if tag == "include":
                pending.append((target, directories))
            else:
                closure.add(target)
    return sorted(closure, key=lambda item: _archive_path("models", item, root))


def _supporting_artifacts(
    prefix: str, sources: tuple[Path, ...]
) -> tuple[list[tuple[str, Path]], list[ModelArtifact]]:
    staged: list[tuple[str, Path]] = []
    artifacts: list[ModelArtifact] = []
    names: set[str] = set()
    for source in sources:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        archive_path = str(PurePosixPath(prefix, source.name))
        if archive_path in names:
            raise ValueError(f"duplicate supporting artifact name: {archive_path}")
        names.add(archive_path)
        staged.append((archive_path, source))
        artifacts.append(ModelArtifact(path=archive_path, digest=_file_digest(source)))
    return staged, artifacts


def _contracts() -> tuple[ObservationContract, ActionContract]:
    from .contracts import CONTROLLED_SERVO_JOINTS, OBSERVATION_FIELDS

    return (
        ObservationContract(
            identifier=OBSERVATION_CONTRACT,
            dimension=61,
            fields=list(OBSERVATION_FIELDS),
            units={},
            normalization="BAKED_IN_ONNX",
        ),
        ActionContract(
            identifier=ACTION_CONTRACT,
            dimension=14,
            joints=list(CONTROLLED_SERVO_JOINTS),
            units="rad",
            scaling={},
            clipping={},
        ),
    )


_SUPPORTED_SCENARIO_PROFILE = "SEEDED_SERVO_RESET_V1"


def _qualified_model_capabilities(
    model_path: Path, terrain: str | None, scenario_profile: str | None
) -> tuple[bool, bool, set[str]]:
    """Compile the exact scene and derive only physically demonstrated capabilities."""
    if terrain is None or scenario_profile != _SUPPORTED_SCENARIO_PROFILE:
        return False, False, set()
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except ValueError:
        return False, False, set()
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if free_joints.size != 1:
        return False, False, set()
    capabilities: set[str] = set()
    if terrain == "flat":
        if has_flat_world_floor(model):
            capabilities.add("FLAT_TERRAIN")
    elif terrain in {"ramp", "slope"}:
        # No checked-in deployment scene currently materializes the procedural
        # ramp used by training, so a label alone cannot qualify it.
        return False, False, set()
    else:
        return False, False, set()
    if "FLAT_TERRAIN" not in capabilities:
        return False, False, set()
    if has_exact_passive_roller_topology(model):
        capabilities.add("ROLLER_FEET")
    from .contracts import CONTROLLED_SERVO_JOINTS

    runtime_compatible = has_exact_position_actuator_topology(
        model, CONTROLLED_SERVO_JOINTS
    ) and has_exact_deployment_frames(model)
    return True, runtime_compatible, capabilities


def _task_id_for(action_code: str) -> str:
    return next(
        template.task_ids[0]
        for template in ACTION_TEMPLATES
        if template.action_code == action_code
    )


def _policy_archive_path(task_id: str, digest: str) -> str:
    safe_task_id = "".join(
        character.lower() if character.isalnum() else "-" for character in task_id
    ).strip("-")
    return f"policies/{safe_task_id}-{digest.removeprefix('sha256:')}.onnx"


def build_bundle(request: BundleBuildRequest) -> BuiltBundle:
    """Build a policy bundle once; existing release archives are never overwritten."""
    output_zip = request.output_zip.resolve()
    if output_zip.exists():
        raise FileExistsError(f"bundle output already exists: {output_zip}")
    if not request.release:
        raise ValueError("release must not be empty")
    unknown = set(request.artifacts) - {
        template.action_code for template in ACTION_TEMPLATES
    }
    if unknown:
        raise ValueError(f"unknown action artifacts: {sorted(unknown)}")

    model_path = request.model_path.resolve()
    model_root = model_path.parent.resolve()
    model_sources = _model_closure(model_path)
    (
        model_qualified,
        model_runtime_compatible,
        model_capabilities,
    ) = _qualified_model_capabilities(
        model_path, request.model_terrain, request.scenario_profile
    )
    staged: list[tuple[str, Path]] = [
        (_archive_path("models", source, model_root), source)
        for source in model_sources
    ]
    model_path_in_archive = _archive_path("models", model_path, model_root)
    model_closure = [
        ModelArtifact(path=archive_path, digest=_file_digest(source))
        for archive_path, source in staged
        if archive_path != model_path_in_archive
    ]

    policies: list[PolicyArtifact] = []
    policy_refs: dict[str, str] = {}
    policy_refs_by_identity: dict[tuple[str, str], str] = {}
    policy_owner_by_identity: dict[tuple[str, str], str] = {}
    policy_task_matches: dict[str, bool] = {}
    policy_normalization_valid: dict[str, bool] = {}
    policy_provenance_valid: dict[str, bool] = {}
    policy_inference_valid: dict[str, bool] = {}
    mirror_transforms = request.mirroring_transforms or {}
    for action_code, source in sorted(request.artifacts.items()):
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = _file_digest(source)
        expected_task_id = _task_id_for(action_code)
        onnx_model = onnx.load(source, load_external_data=False)
        metadata = {item.key: item.value for item in onnx_model.metadata_props}
        normalized_graph_fingerprint: str | None = None
        try:
            normalized_graph = inspect_normalized_actor(onnx_model)
            graph_metadata_valid = (
                metadata.get("microduck.normalization") == "EMPIRICAL_NORMALIZATION_V1"
                and metadata.get("microduck.normalization_graph_sha256")
                == normalized_graph.graph_sha256
            )
            if graph_metadata_valid:
                normalized_graph_fingerprint = normalized_graph.fingerprint
        except ValueError:
            pass
        expected_provenance = {
            "microduck.source_commit": request.source_commit,
            "microduck.observation_contract": OBSERVATION_CONTRACT,
            "microduck.action_contract": ACTION_CONTRACT,
            "microduck.checkpoint": request.checkpoint or "",
            "microduck.run_identity": request.experiment_ref or "",
        }
        provenance_valid = all(
            metadata.get(key) == value for key, value in expected_provenance.items()
        )
        inference_valid = False
        try:
            session = ort.InferenceSession(
                source.read_bytes(), providers=["CPUExecutionProvider"]
            )
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if (
                len(inputs) == 1
                and inputs[0].type == "tensor(float)"
                and inputs[0].shape == [1, 61]
                and len(outputs) == 1
                and outputs[0].type == "tensor(float)"
                and outputs[0].shape == [1, 14]
            ):
                output = session.run(
                    [outputs[0].name],
                    {inputs[0].name: np.zeros((1, 61), dtype=np.float32)},
                )[0]
                inference_valid = output.shape == (1, 14) and bool(
                    np.isfinite(output).all()
                )
        except Exception:  # noqa: BLE001 - invalid exports remain packaged but unavailable.
            inference_valid = False
        identity = (digest, expected_task_id)
        policy_ref = policy_refs_by_identity.get(identity)
        owner_action = policy_owner_by_identity.get(identity)
        opposite_kick = (
            {action_code, owner_action} == {"KICK_LEFT", "KICK_RIGHT"}
            if owner_action is not None
            else False
        )
        if opposite_kick and not _is_exact_kick_mirroring_transform(
            mirror_transforms.get(action_code)
        ):
            continue
        if policy_ref is None:
            archive_path = _policy_archive_path(expected_task_id, digest)
            policy_ref = f"{action_code.lower()}-{digest.removeprefix('sha256:')[:12]}"
            policy_refs_by_identity[identity] = policy_ref
            policy_owner_by_identity[identity] = action_code
            staged.append((archive_path, source))
            metadata_task_id = metadata.get("microduck.task_id", "")
            runtime_requirements = {
                "observationContract": OBSERVATION_CONTRACT,
                "actionContract": ACTION_CONTRACT,
                "normalization": "BAKED_IN_ONNX",
            }
            if normalized_graph_fingerprint is not None:
                runtime_requirements["normalizedGraphFingerprint"] = (
                    normalized_graph_fingerprint
                )
            policies.append(
                PolicyArtifact(
                    policyRef=policy_ref,
                    path=archive_path,
                    digest=digest,
                    taskId=metadata_task_id or expected_task_id,
                    checkpoint=request.checkpoint,
                    experimentRef=request.experiment_ref,
                    runtimeRequirements=runtime_requirements,
                )
            )
        policy_refs[action_code] = policy_ref
        policy_task_matches[action_code] = (
            metadata.get("microduck.task_id") == expected_task_id
        )
        policy_normalization_valid[action_code] = (
            normalized_graph_fingerprint is not None
        )
        policy_provenance_valid[action_code] = provenance_valid
        policy_inference_valid[action_code] = inference_valid

    actions: list[ActionDefinition] = []
    for template in ACTION_TEMPLATES:
        policy_ref = policy_refs.get(template.action_code)
        if policy_ref is None and template.action_code in {"KICK_LEFT", "KICK_RIGHT"}:
            other = "KICK_RIGHT" if template.action_code == "KICK_LEFT" else "KICK_LEFT"
            transform = mirror_transforms.get(template.action_code)
            if _is_exact_kick_mirroring_transform(transform) and other in policy_refs:
                policy_ref = policy_refs[other]
        runtime_spec = ACTION_RUNTIME_SPECS[template.action_code]
        missing_model_capabilities = {
            capability
            for capability in runtime_spec.required_capabilities
            if capability == "ROLLER_FEET" and capability not in model_capabilities
        }
        available = (
            policy_ref is not None
            and runtime_spec.supported
            and model_qualified
            and model_runtime_compatible
            and not missing_model_capabilities
            and policy_task_matches.get(template.action_code, False)
            and policy_normalization_valid.get(template.action_code, False)
            and policy_provenance_valid.get(template.action_code, False)
            and policy_inference_valid.get(template.action_code, False)
        )
        unavailable_reason = (
            "POLICY_ARTIFACT_MISSING"
            if policy_ref is None
            else (
                runtime_spec.unavailable_reason
                if not runtime_spec.supported
                else (
                    "MODEL_QUALIFICATION_INCOMPATIBLE"
                    if not model_qualified
                    else (
                        "MODEL_CAPABILITY_MISSING"
                        if missing_model_capabilities
                        else (
                            "MODEL_RUNTIME_INCOMPATIBLE"
                            if not model_runtime_compatible
                            else (
                                "POLICY_NORMALIZATION_INVALID"
                                if not policy_normalization_valid.get(
                                    template.action_code, False
                                )
                                else (
                                    "POLICY_TASK_ID_MISMATCH"
                                    if not policy_task_matches.get(
                                        template.action_code, False
                                    )
                                    else (
                                        "POLICY_PROVENANCE_MISMATCH"
                                        if not policy_provenance_valid.get(
                                            template.action_code, False
                                        )
                                        else "POLICY_INFERENCE_INVALID"
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        actions.append(
            code_owned_action_definition(
                template.action_code,
                availability="AVAILABLE" if available else "UNAVAILABLE",
                policy_ref=policy_ref,
                unavailable_reason=None if available else unavailable_reason,
            )
        )

    qualification_staged, qualification_artifacts = _supporting_artifacts(
        "qualification", request.qualification_files
    )
    license_staged, license_artifacts = _supporting_artifacts(
        "licenses", request.license_files
    )
    staged.extend(qualification_staged)
    staged.extend(license_staged)
    if len({archive_path for archive_path, _ in staged}) != len(staged):
        raise ValueError("duplicate archive path")

    observation_contract, action_contract = _contracts()
    unsigned = UnsignedPolicyBundleManifest(
        schema="MICRODUCK_POLICY_BUNDLE_V1",
        bundleId="org.microduck.policy",
        bundleVersion=request.release,
        createdAt=request.created_at,
        sourceRepository=request.source_repository,
        sourceCommit=request.source_commit,
        robotModel="MICRODUCK",
        observationContract=observation_contract,
        actionContract=action_contract,
        model=ModelArtifact(
            path=model_path_in_archive, digest=_file_digest(model_path)
        ),
        policies=policies,
        actions=actions,
        qualification={
            "artifacts": [
                artifact.model_dump() for artifact in qualification_artifacts
            ],
            "modelClosure": [artifact.model_dump() for artifact in model_closure],
        }
        | (
            {
                "modelTerrain": request.model_terrain,
                "scenarioProfile": request.scenario_profile,
                "modelCapabilities": sorted(model_capabilities),
            }
            if model_qualified
            else {}
        ),
        license={
            "artifacts": [artifact.model_dump() for artifact in license_artifacts]
        },
    )
    artifact_digests = {
        archive_path: _file_digest(source) for archive_path, source in sorted(staged)
    }
    manifest = publish_policy_bundle(unsigned, artifact_digests)
    manifest_json = manifest.model_dump_json(
        by_alias=True, exclude_none=True, indent=None
    ).encode()

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    contents = [
        ("microduck-policy-bundle.json", manifest_json),
        *((archive_path, source.read_bytes()) for archive_path, source in staged),
    ]
    with zipfile.ZipFile(output_zip, "x", compression=zipfile.ZIP_STORED) as archive:
        for archive_path, content in sorted(contents, key=lambda item: item[0]):
            info = zipfile.ZipInfo(archive_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, content)
    return BuiltBundle(
        manifest=manifest, output_zip=output_zip, artifact_digests=artifact_digests
    )
