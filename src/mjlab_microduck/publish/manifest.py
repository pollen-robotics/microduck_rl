"""The policy manifest (schema 2) and the checks a published policy has to pass.

One vocabulary for two shapes — a single-policy repo (fields at the top level) and the official
set (the same fields per entry under ``policies``). This module writes the first; the daemon
(`pollen-robotics/microduck`, ``updater/src/policy.rs`` and ``robotd-params``) reads both. The
contract is `docs/policy-manifest.md` over there; the numbers below are what the daemon publishes
in ``duck_ipc_proto`` and refuses a policy for disagreeing with.

Deliberately free of mjlab / torch imports so the tests run on a laptop in milliseconds and the
CLI can validate an ONNX file without a GPU.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 2
# Schema 3 adds the free-sentence (VLA) shape: the policy ONNX takes proprio
# `obs` plus a `text_emb` input, and the repo ships a separate `encoder.onnx`.
VLA_SCHEMA_VERSION = 3
# `duck_ipc_proto`: the daemon refuses a policy whose manifest disagrees with these, and refuses
# at load a network whose graph does. 61 = 48 proprioception + 13 command; 14 = the servos.
# VLA policies drop the 13D numeric command from the actor, so their ONNX `obs`
# is the 48D proprio half; the instruction arrives as `text_emb`.
MODEL_API = 1
OBS_LEN = 61
VLA_OBS_LEN = 48
ACTION_LEN = 14
TEXT_INPUT = "text_emb"
ROBOT: dict[str, Any] = {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50}

# The one `.onnx` a repo carries. VLA repos carry a second one (`encoder.onnx`).
POLICY_FILE = "policy.onnx"
ENCODER_FILE = "encoder.onnx"

# A rollout video, if the repo carries one. The Hub's replay widget picks up a file of exactly
# this name, so it is shown there and neither the manifest nor the README points at it.
REPLAY_FILE = "replay.mp4"

Kind = Literal["episodic", "perpetual"]
KINDS: tuple[str, ...] = ("episodic", "perpetual")

ZERO_TWIST: tuple[float, float, float] = (0.0, 0.0, 0.0)

# The daemon's policy slots, for a gait's `slot` hint (display-only: `robotctl policy load <slot>`).
SLOTS: tuple[str, ...] = ("walk", "stand", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade")


class ManifestError(ValueError):
    """A manifest that the daemon would refuse, or that would load and run wrongly."""


@dataclass(frozen=True)
class Provenance:
    """Where the weights came from. Display-only for the daemon; the part people skip by hand."""

    task_id: str | None = None
    repo: str = "pollen-robotics/microduck_rl"
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    run: str | None = None
    checkpoint: int | None = None
    source_file: str | None = None
    exported: str = field(default_factory=lambda: _now_utc())

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """`commit`, `branch`, `dirty` of the checkout the export ran from, or `{}` outside git."""
    root = str(repo_root or Path(__file__).resolve().parents[3])

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", root, *args], capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = git("rev-parse", "--short=9", "HEAD")
    if commit is None:
        return {}
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "branch": branch, "dirty": bool(status)}


def build_manifest(
    *,
    name: str,
    kind: str,
    description: str,
    duration_s: float | None = None,
    chain: bool = False,
    unwind_s: float | None = None,
    idle: tuple[float, float, float] = ZERO_TWIST,
    action_scale: float | None = None,
    entry_pose: str = "standing",
    slot: str | None = None,
    command_help: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    eval: dict[str, Any] | None = None,
    vla: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A single-policy manifest the daemon loads without surprises.

    Only the constant-command family is publishable from here — a skill's network is fed a fixed
    twist. Phase and posture-flag encodings are the official set's own arms and are not something
    a community policy can be.

    ``vla`` switches to schema 3: a free-sentence policy whose ONNX takes proprio
    ``obs`` + ``text_emb`` and whose repo also carries ``encoder.onnx``. The
    dict carries ``embedding_dim`` and an ``encoder`` block (file/model/version/
    sha256/tokenizer/max_length) recorded in the manifest for the runtime.
    """
    if kind not in KINDS:
        raise ManifestError(f"kind must be one of {KINDS}, not {kind!r}")
    if not name or "/" in name or name != name.strip():
        raise ManifestError(f"name must be a bare word a client can ask for, not {name!r}")
    if kind == "episodic":
        if duration_s is None or duration_s <= 0:
            raise ManifestError(
                "an episodic policy ends itself: say how long it runs with duration_s > 0"
            )
        if unwind_s:
            raise ManifestError(
                "an episodic policy is already back when duration_s is up; unwind_s is for perpetual"
            )
    else:
        # Two things are perpetual: a gait, which lives in a slot (`policy load walk <repo>`) and
        # needs nothing here, and a held pose like the flamingo, which the owner runs as a
        # one-shot with `policy add --hold` and which then needs `unwind_s` so the robot is not
        # let go of on one foot. `unwind_s` is what says which.
        if duration_s is not None:
            raise ManifestError(
                "a perpetual policy has no length of its own; leave duration_s unset "
                "(a gait runs until told otherwise; a held pose gets --hold when added as a skill)"
            )
        if unwind_s is not None and unwind_s <= 0:
            raise ManifestError("unwind_s must be > 0 when given")
        if chain:
            raise ManifestError("chain is for episodic one-shots a held button repeats")
    if slot is not None and slot not in SLOTS:
        raise ManifestError(f"slot must be one of {SLOTS}, not {slot!r}")
    if action_scale is not None and not 0 < action_scale <= 2.0:
        raise ManifestError(f"action_scale {action_scale} is outside (0, 2]")
    if len(idle) != 3:
        raise ManifestError("idle is a 3-vector twist")

    command: dict[str, Any] = {
        "encoding": "text" if vla is not None else "constant",
        "idle": [float(v) for v in idle],
        "twist": "unused (zeros)" if vla is None else "unused (text instruction)",
        "head": "unused (zeros)",
        "body": "unused (zeros)",
    }
    if command_help:
        command.update(command_help)

    is_vla = vla is not None
    manifest: dict[str, Any] = {
        "schema_version": VLA_SCHEMA_VERSION if is_vla else SCHEMA_VERSION,
        "model_api": MODEL_API,
        "obs_len": VLA_OBS_LEN if is_vla else OBS_LEN,
        "action_len": ACTION_LEN,
        "robot": dict(ROBOT),
        "name": name,
        "kind": kind,
        "entry_pose": entry_pose,
        "description": description,
        "command": command,
    }
    if is_vla:
        # Copy so the caller's dict is not mutated, and make the file pairing
        # explicit: the daemon must load policy + encoder together.
        text = dict(vla)
        text.setdefault("input", TEXT_INPUT)
        encoder = dict(text.get("encoder") or {})
        encoder.setdefault("file", ENCODER_FILE)
        text["encoder"] = encoder
        manifest["text"] = text
        manifest["artifacts"] = {
            "policy": POLICY_FILE,
            "encoder": encoder["file"],
        }
    if kind == "episodic":
        manifest["duration_s"] = float(duration_s)  # type: ignore[arg-type]
        manifest["chain"] = bool(chain)
    else:
        manifest["duration_s"] = None
        if unwind_s is not None:
            manifest["unwind_s"] = float(unwind_s)
    if slot is not None:
        manifest["slot"] = slot
    if action_scale is not None:
        manifest["action_scale"] = float(action_scale)
    if training:
        manifest["training"] = training
    if eval:
        manifest["eval"] = eval
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Refuse what the daemon would refuse, plus the mistakes it would load and run wrongly.

    Accepts both shapes and any schema version, because absence is not evidence — a repo is under
    no obligation to carry any field. Only a claim that is present and wrong fails.
    """
    if "policies" in manifest:
        for entry in manifest["policies"]:
            if "file" not in entry:
                raise ManifestError("every set entry needs a `file`")
            validate_manifest({k: v for k, v in entry.items() if k != "file"})
        return
    if (obs := manifest.get("obs_len")) is not None:
        is_vla = (
            manifest.get("schema_version", SCHEMA_VERSION) >= VLA_SCHEMA_VERSION
            or "text" in manifest
        )
        expected = VLA_OBS_LEN if is_vla else OBS_LEN
        if obs != expected:
            raise ManifestError(
                f"obs_len {obs}: this {'VLA' if is_vla else 'standard'} policy builds {expected}"
            )
    if (act := manifest.get("action_len")) is not None and act != ACTION_LEN:
        raise ManifestError(f"action_len {act}: this robot has {ACTION_LEN}")
    if (api := manifest.get("model_api")) is not None and api > MODEL_API:
        raise ManifestError(f"model_api {api}: this repo targets {MODEL_API}")
    model = (manifest.get("robot") or {}).get("model")
    if model is not None and model.lower() != ROBOT["model"]:
        raise ManifestError(f"robot.model {model!r}: this is a {ROBOT['model']} policy repo")
    kind = manifest.get("kind")
    if kind is not None and kind not in (*KINDS, "scripted"):
        raise ManifestError(f"kind {kind!r} is not one of episodic, perpetual, scripted")
    is_vla = (
        manifest.get("schema_version", SCHEMA_VERSION) >= VLA_SCHEMA_VERSION
        or "text" in manifest
    )
    encoding = (manifest.get("command") or {}).get("encoding")
    allowed_encodings = ("constant", "phase", "posture_flag") + (("text",) if is_vla else ())
    if encoding is not None and encoding not in allowed_encodings:
        raise ManifestError(f"command.encoding {encoding!r} is not one the daemon drives")
    if is_vla:
        _validate_vla(manifest)
    if kind == "episodic" and encoding in (None, "constant", "text"):
        duration = manifest.get("duration_s")
        if duration is None or duration <= 0:
            raise ManifestError("an episodic constant-command policy needs duration_s > 0")
    idle = (manifest.get("command") or {}).get("idle")
    if idle is not None and len(idle) != 3:
        raise ManifestError("command.idle is a 3-vector twist")


def _validate_vla(manifest: dict[str, Any]) -> None:
    """Schema 3 requires a resolvable text block and encoder artifact."""
    text = manifest.get("text") or {}
    if text.get("input") != TEXT_INPUT:
        raise ManifestError(f"text.input must be {TEXT_INPUT!r} for a VLA policy")
    dim = text.get("embedding_dim")
    if not isinstance(dim, int) or dim <= 0:
        raise ManifestError("text.embedding_dim must be a positive integer")
    encoder = text.get("encoder") or {}
    if not encoder.get("file"):
        raise ManifestError("text.encoder.file is required (the repo ships encoder.onnx)")
    if not encoder.get("model"):
        raise ManifestError("text.encoder.model is required (train/deploy must match)")
    artifacts = manifest.get("artifacts") or {}
    if artifacts and artifacts.get("encoder") != encoder["file"]:
        raise ManifestError("artifacts.encoder must match text.encoder.file")


# ---------------------------------------------------------------------------------------------
# The ONNX file: the shape gate the daemon applies at load, applied before the upload.


@dataclass(frozen=True)
class OnnxShape:
    input_name: str
    output_name: str
    obs_len: int
    action_len: int
    # VLA only: the second input (sentence embedding). ``text_dim == 0`` is a
    # standard single-input policy.
    text_input_name: str | None = None
    text_dim: int = 0

    @property
    def is_vla(self) -> bool:
        return self.text_dim > 0


def inspect_onnx(path: str | Path) -> OnnxShape:
    """The graph's input(s) and output widths.

    Accepts the standard one-input policy and the VLA two-input shape
    (``obs`` + ``text_emb``). Anything else is refused.
    """
    import onnx

    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]
    if not 1 <= len(inputs) <= 2 or len(graph.output) != 1:
        raise ManifestError(
            f"{path.name}: expected one or two inputs and one output, found "
            f"{[i.name for i in inputs]} -> {[o.name for o in graph.output]}"
        )

    def last_dim(value) -> int:
        dims = value.type.tensor_type.shape.dim
        if not dims:
            raise ManifestError(f"{path.name}: {value.name} has no shape")
        last = dims[-1]
        if not last.HasField("dim_value"):
            raise ManifestError(f"{path.name}: {value.name}'s last dimension is symbolic")
        return int(last.dim_value)

    text_input_name = inputs[1].name if len(inputs) == 2 else None
    text_dim = last_dim(inputs[1]) if len(inputs) == 2 else 0
    return OnnxShape(
        input_name=inputs[0].name,
        output_name=graph.output[0].name,
        obs_len=last_dim(inputs[0]),
        action_len=last_dim(graph.output[0]),
        text_input_name=text_input_name,
        text_dim=text_dim,
    )


@dataclass(frozen=True)
class EncoderShape:
    """The separate ``encoder.onnx`` contract: token ids -> sentence embedding."""

    input_names: tuple[str, ...]
    output_name: str
    embedding_dim: int


def inspect_encoder_onnx(path: str | Path) -> EncoderShape:
    """Inspect ``encoder.onnx`` (``input_ids`` + ``attention_mask`` -> ``text_emb``).

    The sequence axis is dynamic, so unlike the policy we do not require the
    last input dimension to be concrete; only the embedding width is checked.
    """
    import onnx

    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]
    if len(inputs) != 2 or len(graph.output) != 1:
        raise ManifestError(
            f"{path.name}: encoder expects two inputs and one output, found "
            f"{[i.name for i in inputs]} -> {[o.name for o in graph.output]}"
        )
    names = tuple(i.name for i in inputs)
    if set(names) != {"input_ids", "attention_mask"}:
        raise ManifestError(
            f"{path.name}: encoder inputs must be 'input_ids' + 'attention_mask', "
            f"got {names}"
        )
    dims = graph.output[0].type.tensor_type.shape.dim
    if not dims or not dims[-1].HasField("dim_value"):
        raise ManifestError(f"{path.name}: encoder output's last dimension must be concrete")
    return EncoderShape(
        input_names=names,
        output_name=graph.output[0].name,
        embedding_dim=int(dims[-1].dim_value),
    )


def check_encoder_onnx(path: str | Path, expected_dim: int | None = None) -> EncoderShape:
    """Refuse an encoder whose shape cannot feed the policy's ``text_emb`` input."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"{path}: no such file")
    shape = inspect_encoder_onnx(path)
    if expected_dim is not None and shape.embedding_dim != expected_dim:
        raise ManifestError(
            f"{path.name}: embedding width is {shape.embedding_dim}, "
            f"the policy expects {expected_dim}"
        )
    return shape


def sha256_file(path: str | Path) -> str:
    """Content hash recorded in a VLA manifest so encoder/policy cannot drift."""
    import hashlib

    path = Path(path)

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_onnx(path: str | Path) -> OnnxShape:
    """Refuse a file the daemon would refuse at load.

    Standard: one input, 61 -> 14. VLA: ``obs`` (48) + ``text_emb`` -> 14.
    """
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"{path}: no such file")
    shape = inspect_onnx(path)
    if shape.is_vla:
        if shape.obs_len != VLA_OBS_LEN:
            raise ManifestError(
                f"{path.name}: VLA observation width is {shape.obs_len}, expected "
                f"{VLA_OBS_LEN} (proprio only; the numeric command is not an actor input)"
            )
        if shape.input_name != "obs":
            raise ManifestError(f"{path.name}: proprio input must be named 'obs'")
        if shape.text_input_name != TEXT_INPUT:
            raise ManifestError(
                f"{path.name}: text input must be named {TEXT_INPUT!r}, "
                f"got {shape.text_input_name!r}"
            )
    elif shape.obs_len != OBS_LEN:
        raise ManifestError(
            f"{path.name}: observation width is {shape.obs_len}, the robot builds {OBS_LEN} "
            "(a 51-D policy is the legacy 3-value-command family, which the daemon refuses)"
        )
    if shape.action_len != ACTION_LEN:
        raise ManifestError(f"{path.name}: {shape.action_len} actions, the robot has {ACTION_LEN}")
    return shape


def smoke_run_onnx(path: Path, steps: int = 50, seed: int = 0) -> None:
    """Run the network on plausible inputs and refuse a NaN/inf or a saturated output.

    Not a physics rehearsal — `scripts/infer_policy.py` is that — but it catches a broken export
    (an un-baked normalizer producing NaNs on raw observations, a graph that will not execute)
    before anything is uploaded.
    """
    import numpy as np
    import onnxruntime as ort

    shape = inspect_onnx(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    obs = np.zeros((1, shape.obs_len), dtype=np.float32)
    fixed: dict[str, np.ndarray] = {}
    if shape.is_vla:
        # One instruction for the whole rollout, like the robot. Keep it a
        # plausible normalized embedding (the encoder L2-normalizes).
        assert shape.text_input_name is not None
        text = rng.normal(0.0, 1.0, size=(1, shape.text_dim)).astype(np.float32)
        text /= np.linalg.norm(text)
        fixed[shape.text_input_name] = text
    outputs = []
    for _ in range(steps):
        feeds = {shape.input_name: obs, **fixed}
        (out,) = session.run([shape.output_name], feeds)
        if not np.all(np.isfinite(out)):
            raise ManifestError(f"{path.name}: the network produced a non-finite action")
        outputs.append(out)
        # Feed the action back into the last-action slots and jitter the rest, the way an
        # observation evolves on the robot; enough to leave the zero point.
        obs = rng.normal(0.0, 0.05, size=obs.shape).astype(np.float32)
        if shape.is_vla:
            # Proprio layout ends with the 14 last-action slots (no 13D command).
            obs[0, -ACTION_LEN:] = np.clip(out[0], -1, 1)
        else:
            obs[0, -ACTION_LEN - 13 : -13] = np.clip(out[0], -1, 1)
    spread = float(np.std(np.stack(outputs)))
    if spread == 0.0:
        raise ManifestError(f"{path.name}: the network's output never changes; is it a real policy?")


# ---------------------------------------------------------------------------------------------
# What else goes in the repo.


def install_commands(manifest: dict[str, Any], repo_id: str) -> str:
    """The `robotctl` lines that put this policy on a robot — one story per shape.

    Episodic: a skill, length from the manifest. Perpetual with `unwind_s`: a held pose the owner
    runs as a skill with `--hold`. Perpetual without: a gait, loaded into a slot.
    """
    name = manifest["name"]
    if manifest["kind"] == "episodic":
        return f"sudo robotctl policy add {name} {repo_id}\nrobotctl robot do {name}"
    if manifest.get("unwind_s") is not None:
        return f"sudo robotctl policy add {name} {repo_id} --hold <seconds>\nrobotctl robot do {name}"
    slot = manifest.get("slot", "<slot>")
    return f"sudo robotctl policy load {slot} {repo_id}"


def _obs_description(manifest: dict[str, Any]) -> str:
    """Human-readable observation shape for the model card."""
    if "text" in manifest:
        return (
            f"{VLA_OBS_LEN}-D proprioception + "
            f"{manifest['text'].get('embedding_dim', '?')}-D text instruction"
        )
    return f"{OBS_LEN}-D observation"


def render_readme(manifest: dict[str, Any], repo_id: str, base_model: str | None = None) -> str:
    """A model card that says how to run the policy on a robot, generated so it cannot go stale.

    ``base_model`` is the Hub repo a remix was trained from; the card declares it so the Hub links
    the two (the parent lists this repo among its fine-tunes).
    """
    if base_model is not None:
        _check_repo_id(base_model, "base_model")
        if base_model == repo_id:
            raise ManifestError(f"base_model {base_model!r} is the repo being published")
    kind = manifest["kind"]
    name = manifest["name"]
    description = manifest.get("description", "")
    training = manifest.get("training", {})
    run = install_commands(manifest, repo_id)
    if kind == "episodic":
        timing = f"Runs {manifest['duration_s']} s and returns itself to a standing pose."
        if manifest.get("chain"):
            timing += " Holding the button chains another run."
    elif manifest.get("unwind_s") is not None:
        timing = (
            f"Holds until told otherwise; the daemon drives `command.idle` for "
            f"{manifest['unwind_s']} s before handing back to the gait."
        )
    else:
        slot = manifest.get("slot")
        timing = "Runs until told otherwise" + (
            f" — a gait for the `{slot}` slot." if slot else " — a gait, loaded into a policy slot."
        )
    lines = [
        "---",
        "tags:",
        "- microduck",
        "- robotics",
        "- reinforcement-learning",
        "- onnx",
        # The Hub's `robotctl policy load <slot>` snippet reads the slot from this tag.
        *([f"- microduck-slot:{manifest['slot']}"] if manifest.get("slot") else []),
        "library_name: microduck",
        "pipeline_tag: robotics",
        *([f"base_model: {base_model}", "base_model_relation: finetune"] if base_model else []),
        "---",
        "",
        f"# {name}",
        "",
        description,
        "",
        f"A **{kind}** policy for the [microduck](https://github.com/pollen-robotics/microduck) "
        f"({_obs_description(manifest)}, {ACTION_LEN} actions, {ROBOT['control_hz']} Hz). {timing}",
        "",
        "## Run it on a robot",
        "",
        "```bash",
        run,
        "```",
        "",
        "The observation normalizer is baked into `policy.onnx`; feed raw observations.",
    ]
    if "text" in manifest:
        text = manifest["text"]
        encoder = text.get("encoder", {})
        lines += [
            "",
            "## Language conditioning",
            "",
            f"This is a **schema-{VLA_SCHEMA_VERSION} VLA** policy: `{POLICY_FILE}` takes "
            f"proprio `obs` ({VLA_OBS_LEN}D) plus `text_emb` ({text['embedding_dim']}D). "
            f"Encode the instruction with `{encoder.get('file', ENCODER_FILE)}` "
            f"(`{encoder.get('model', 'unknown encoder')}`) before calling the policy. "
            "The encoder version recorded in `manifest.json` must match training.",
        ]
    lines += [
        "",
        f"`manifest.json` follows schema "
        f"{VLA_SCHEMA_VERSION if 'text' in manifest else SCHEMA_VERSION} of the "
        "microduck policy manifest (`docs/policy-manifest.md` in the daemon repo).",
    ]
    if base_model:
        lines += ["", f"A remix of [{base_model}](https://huggingface.co/{base_model})."]
    if training:
        lines += ["", "## Training", ""]
        for key in ("task_id", "repo", "branch", "commit", "run", "checkpoint", "exported", "bundle"):
            if key in training:
                lines.append(f"- **{key}**: `{training[key]}`")
        if training.get("dirty"):
            lines.append("- exported from a checkout with uncommitted changes")
        if training.get("bundle"):
            lines.append(f"- train from it: `uv run train {training.get('task_id', '<task>')} --init-checkpoint {training['bundle']}` "
                         f"(in [{training.get('repo', 'the code repo')}](https://github.com/{training.get('repo', '')}))")
    return "\n".join(lines) + "\n"


def _check_repo_id(repo_id: str, what: str) -> None:
    owner, _, name = repo_id.partition("/")
    if not owner or not name or "/" in name or repo_id != repo_id.strip():
        raise ManifestError(f"{what} must be a Hub repo id `<user-or-org>/<name>`, not {repo_id!r}")


def dump_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2) + "\n"
