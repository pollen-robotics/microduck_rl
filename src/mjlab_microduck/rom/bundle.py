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

from .action_catalog import ACTION_TEMPLATES
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
    sha256_prefixed,
)
from .mirroring import (
    MICRODUCK_JOINT_MIRROR_PERMUTATION,
    MICRODUCK_JOINT_MIRROR_SIGNS,
)


@dataclass(frozen=True)
class BundleBuildRequest:
    release: str
    output_zip: Path
    artifacts: Mapping[str, Path]
    model_path: Path
    source_repository: str
    source_commit: str
    created_at: datetime
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
    model_capabilities = {
        "ROLLER_FEET"
        for source in model_sources
        if source.name == "roller_blade.stl"
        or (source.suffix == ".xml" and b"roller_blade" in source.read_bytes())
    }
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
    policy_refs_by_digest: dict[str, str] = {}
    policy_owner_by_digest: dict[str, str] = {}
    mirror_transforms = request.mirroring_transforms or {}
    for action_code, source in sorted(request.artifacts.items()):
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = _file_digest(source)
        policy_ref = policy_refs_by_digest.get(digest)
        owner_action = policy_owner_by_digest.get(digest)
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
            archive_path = f"policies/{digest.removeprefix('sha256:')}.onnx"
            policy_ref = f"{action_code.lower()}-{digest.removeprefix('sha256:')[:12]}"
            policy_refs_by_digest[digest] = policy_ref
            policy_owner_by_digest[digest] = action_code
            staged.append((archive_path, source))
            task_id = next(
                template.task_ids[0]
                for template in ACTION_TEMPLATES
                if template.action_code == action_code
            )
            policies.append(
                PolicyArtifact(
                    policyRef=policy_ref,
                    path=archive_path,
                    digest=digest,
                    taskId=task_id,
                    checkpoint=request.checkpoint,
                    experimentRef=request.experiment_ref,
                    runtimeRequirements={
                        "observationContract": OBSERVATION_CONTRACT,
                        "actionContract": ACTION_CONTRACT,
                        "normalization": "BAKED_IN_ONNX",
                    },
                )
            )
        policy_refs[action_code] = policy_ref

    actions: list[ActionDefinition] = []
    for template in ACTION_TEMPLATES:
        policy_ref = policy_refs.get(template.action_code)
        safety: dict[str, Any] | None = None
        if policy_ref is None and template.action_code in {"KICK_LEFT", "KICK_RIGHT"}:
            other = "KICK_RIGHT" if template.action_code == "KICK_LEFT" else "KICK_LEFT"
            transform = mirror_transforms.get(template.action_code)
            if _is_exact_kick_mirroring_transform(transform) and other in policy_refs:
                policy_ref = policy_refs[other]
                safety = {"mirroringTransform": dict(transform)}
        runtime_spec = ACTION_RUNTIME_SPECS[template.action_code]
        missing_model_capabilities = {
            capability
            for capability in runtime_spec.required_capabilities
            if capability == "ROLLER_FEET" and capability not in model_capabilities
        }
        available = (
            policy_ref is not None
            and runtime_spec.supported
            and not missing_model_capabilities
        )
        unavailable_reason = (
            "POLICY_ARTIFACT_MISSING"
            if policy_ref is None
            else (
                runtime_spec.unavailable_reason
                if not runtime_spec.supported
                else "MODEL_CAPABILITY_MISSING"
            )
        )
        actions.append(
            ActionDefinition(
                actionCode=template.action_code,
                executionMode=template.execution_mode,
                availability="AVAILABLE" if available else "UNAVAILABLE",
                policyRef=policy_ref,
                unavailableReason=None if available else unavailable_reason,
                parameterSchema=template.parameter_schema,
                completion=template.completion,
                lease=template.lease,
                safety=safety,
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
    unsigned = PolicyBundle(
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
        },
        license={
            "artifacts": [artifact.model_dump() for artifact in license_artifacts]
        },
    )
    artifact_digests = {
        archive_path: _file_digest(source) for archive_path, source in sorted(staged)
    }
    unsigned_mapping = unsigned.model_dump(
        mode="json", by_alias=True, exclude={"bundleDigest"}
    )
    digest = sha256_prefixed(
        {"manifest": unsigned_mapping, "artifacts": artifact_digests}
    )
    manifest = unsigned.model_copy(update={"bundleDigest": digest})
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
