"""Language-conditioned actor: proprio MLP + raw text embedding.

Why a custom model is required
------------------------------
``rsl_rl.MLPModel`` concatenates every configured observation group and then
runs the whole concatenation through ``EmpiricalNormalization``. That would
normalize the frozen sentence embedding, which destroys its semantics and makes
the deployed encoder (``encoder.onnx``) inconsistent with training. This model:

* normalizes **only** the proprio groups (``"actor"`` / ``"critic"``),
* appends the raw ``"text"`` embedding after normalization,
* only updates the normalizer from proprio statistics,
* exports two ONNX inputs: ``obs`` (proprio) and ``text_emb``.

It is a drop-in for ``MLPModel`` because it subclasses it and only overrides
the latent construction, the normalization update and the export wrappers. If
the configured obs set has no text group (e.g. the critic), it behaves exactly
like ``MLPModel``.

Plug-in point (no rsl_rl fork): ``RslRlModelCfg.class_name`` accepts a
``"module:Attr"`` path, resolved by ``rsl_rl.utils.resolve_callable``::

    actor = RslRlModelCfg(
        class_name="mjlab_microduck.vla.model:MicroduckVLAModel",
        ...
    )
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState

DEFAULT_TEXT_GROUP = "text"


class MicroduckVLAModel(MLPModel):
    """MLP on ``normalize(proprio) || raw_text``.

    Args:
        text_group: Name of the obs group carrying the sentence embedding.
            The group is expected to be 2D ``(batch, dim)``. If the obs set does
            not contain it, this degrades to a plain ``MLPModel``.
        **kwargs: Forwarded to :class:`rsl_rl.models.mlp_model.MLPModel`.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *args,
        text_group: str = DEFAULT_TEXT_GROUP,
        **kwargs,
    ) -> None:
        # Set these BEFORE super().__init__: it calls self._get_latent_dim(),
        # which needs the text dimension to size the MLP correctly.
        groups = list(obs_groups[obs_set])
        self.text_group: str | None = text_group if text_group in groups else None
        self._text_dim = (
            int(obs[text_group].shape[-1]) if self.text_group is not None else 0
        )
        if self.text_group is not None and len(obs[text_group].shape) != 2:
            raise ValueError(
                f"Text group {text_group!r} must be 2D (batch, dim), got "
                f"shape {tuple(obs[text_group].shape)}"
            )

        # Normalize only the proprio groups: hand MLPModel a filtered group
        # list so its EmpiricalNormalization is built over proprio_dim.
        proprio_groups = [g for g in groups if g != self.text_group]
        filtered_groups = dict(obs_groups)
        filtered_groups[obs_set] = proprio_groups

        super().__init__(
            obs,
            filtered_groups,
            obs_set,
            output_dim,
            *args,
            **kwargs,
        )

        if self._text_dim > 0:
            print(
                f"[vla] {type(self).__name__} ({obs_set}): proprio "
                f"{self.obs_dim}D (normalized) + text {self._text_dim}D (raw) "
                f"-> MLP {self.obs_dim + self._text_dim}D"
            )

    # -- latent ----------------------------------------------------------

    def _get_latent_dim(self) -> int:
        """MLP input width: proprio + text (super() builds the MLP with this)."""
        return self.obs_dim + self._text_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """``normalize(proprio)`` concatenated with the **raw** text embedding."""
        proprio = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        latent = self.obs_normalizer(proprio)
        if self.text_group is None:
            return latent
        return torch.cat([latent, obs[self.text_group]], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update running statistics from proprio only — never from text."""
        if not self.obs_normalization:
            return
        proprio = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        self.obs_normalizer.update(proprio)  # type: ignore[attr-defined]

    # -- export ----------------------------------------------------------

    def as_jit(self) -> nn.Module:
        return _TorchVLAModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxVLAModel(self, verbose)


class _VLAModelMixin:
    """Shared export state/forward for the JIT and ONNX wrappers.

    No class-level type annotations here: TorchScript tries to interpret them
    and rejects ``nn.Module`` annotations when scripting ``_TorchVLAModel``.
    The attributes are assigned in :meth:`_init_from`.
    """

    def _init_from(self, model: MicroduckVLAModel) -> None:
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()
        self.proprio_dim = model.obs_dim
        self.text_dim = model._text_dim
        self.has_text = model.text_group is not None

    def _forward(self, obs: torch.Tensor, text: torch.Tensor | None) -> torch.Tensor:
        latent = self.obs_normalizer(obs)
        if self.has_text:
            if text is None:
                raise ValueError(
                    "This exported VLA policy expects a text embedding input "
                    "named 'text_emb'."
                )
            latent = torch.cat([latent, text], dim=-1)
        return self.deterministic_output(self.mlp(latent))


class _TorchVLAModel(_VLAModelMixin, nn.Module):
    """TorchScript exportable VLA policy."""

    def __init__(self, model: MicroduckVLAModel) -> None:
        super().__init__()
        self._init_from(model)

    def forward(
        self, obs: torch.Tensor, text: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._forward(obs, text)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxVLAModel(_VLAModelMixin, nn.Module):
    """ONNX exportable VLA policy (``obs`` + ``text_emb`` inputs)."""

    is_recurrent: bool = False

    def __init__(self, model: MicroduckVLAModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self._init_from(model)

    def forward(
        self, obs: torch.Tensor, text: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._forward(obs, text)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        obs = torch.zeros(1, self.proprio_dim)
        if not self.has_text:
            return (obs,)
        return (obs, torch.zeros(1, self.text_dim))

    @property
    def input_names(self) -> list[str]:
        return ["obs", "text_emb"] if self.has_text else ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
