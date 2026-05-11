# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from resfit.rl_finetuning.config.rlpd import DepthAnythingV2ConditioningConfig, VitEncoderConfig
from resfit.rl_finetuning.off_policy.networks.min_vit import MinVit


_DEPTH_ANYTHING_V2_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _iter_depth_anything_v2_roots(explicit_root: str | None) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path | None) -> None:
        if path is None:
            return
        path = path.expanduser()
        if path in seen:
            return
        seen.add(path)
        roots.append(path)

    if explicit_root is not None:
        _add(Path(explicit_root))

    env_root = os.environ.get("DEPTH_ANYTHING_V2_ROOT")
    if env_root:
        _add(Path(env_root))

    repo_root = Path(__file__).resolve().parents[4]
    search_parents = [repo_root, repo_root.parent, repo_root.parent.parent]
    candidate_names = (
        "Depth-Anything-V2",
        "depth-anything-v2",
        "depth_anything_v2",
        "third_party/Depth-Anything-V2",
        "third_party/depth-anything-v2",
        "third_party/depth_anything_v2",
    )
    for parent in search_parents:
        for name in candidate_names:
            _add(parent / name)

    return roots


def _resolve_depth_anything_v2_root(cfg: DepthAnythingV2ConditioningConfig) -> Path:
    if importlib.util.find_spec("depth_anything_v2") is not None:
        # The package is already importable from the current environment.
        return Path(".")

    for root in _iter_depth_anything_v2_roots(cfg.source_root):
        if (root / "depth_anything_v2" / "dpt.py").is_file():
            return root.resolve()

    searched = ", ".join(str(path) for path in _iter_depth_anything_v2_roots(cfg.source_root))
    raise ImportError(
        "DepthAnythingV2 source tree was not found. "
        "Set `agent.depth_anything_v2_conditioning.source_root` or the `DEPTH_ANYTHING_V2_ROOT` environment "
        f"variable. Searched: {searched}"
    )


def _load_depth_anything_v2_model(cfg: DepthAnythingV2ConditioningConfig) -> nn.Module:
    root = _resolve_depth_anything_v2_root(cfg)
    root_str = str(root)
    if root_str != "." and root_str not in sys.path:
        sys.path.insert(0, root_str)

    from depth_anything_v2.dpt import DepthAnythingV2

    model_cfg = dict(_DEPTH_ANYTHING_V2_MODEL_CONFIGS[cfg.encoder])
    model = DepthAnythingV2(**model_cfg)

    weights_path: Path | None = None
    if cfg.weights is not None:
        candidate = Path(cfg.weights).expanduser()
        if not candidate.is_absolute() and root != Path("."):
            candidate = root / candidate
        weights_path = candidate
    elif root != Path("."):
        default_name = f"depth_anything_v2_{cfg.encoder}.pth"
        candidate = root / "checkpoints" / default_name
        if candidate.is_file():
            weights_path = candidate

    if weights_path is None:
        raise ValueError(
            "DepthAnythingV2 conditioning is enabled but no checkpoint was found. "
            "Either set `agent.depth_anything_v2_conditioning.weights` or place the official checkpoint under "
            f"`{root / 'checkpoints' / f'depth_anything_v2_{cfg.encoder}.pth'}`."
        )

    if not weights_path.is_file():
        raise FileNotFoundError(f"DepthAnythingV2 checkpoint not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict):
        state_dict = {
            key[len("model.") :] if key.startswith("model.") else key: value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)

    return model


class VitEncoder(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], cfg: VitEncoderConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.vit = MinVit(
            embed_style=cfg.embed_style,
            embed_dim=cfg.embed_dim,
            embed_norm=cfg.embed_norm,
            num_head=cfg.num_heads,
            depth=cfg.depth,
            image_shape=obs_shape[1:],
            patch_size=cfg.patch_size,
            stride=cfg.stride,
        )

        self.num_patch = self.vit.num_patches
        self.patch_repr_dim = self.cfg.embed_dim
        self.repr_dim = self.cfg.embed_dim * self.vit.num_patches

    def forward(
        self,
        obs,
        flatten=True,
        *,
        layer_prefix_tokens: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        if obs.max() > 5:
            obs = obs / 255.0
        obs = obs - 0.5
        feats: torch.Tensor = self.vit.forward(obs, layer_prefix_tokens=layer_prefix_tokens)
        if flatten:
            # [B, N, D] -> [B, N*D]
            feats = feats.flatten(1, 2)
        return feats


class DepthAnythingV2TokenEncoder(nn.Module):
    """Extracts encoder patch tokens and CLS token from a local DepthAnythingV2 model."""

    def __init__(self, obs_shape: tuple[int, int, int], cfg: DepthAnythingV2ConditioningConfig):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.model = _load_depth_anything_v2_model(cfg)
        self.freeze_encoder = cfg.freeze_encoder

        self.backbone = self._resolve_backbone()
        self.patch_size = int(getattr(self.backbone, "patch_size", 14))
        embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is None:
            embed_dim = int(self.backbone.blocks[0].attn.qkv.in_features)

        self.patch_repr_dim = int(embed_dim)
        self.num_patch = (obs_shape[1] // self.patch_size) * (obs_shape[2] // self.patch_size)
        self.repr_dim = self.patch_repr_dim * self.num_patch
        self.register_buffer("pixel_mean", torch.tensor(cfg.mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.std, dtype=torch.float32).view(1, 3, 1, 1))

        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.model.eval()
        return self

    def _resolve_backbone(self):
        for attr in ("pretrained", "backbone"):
            backbone = getattr(self.model, attr, None)
            if backbone is not None and hasattr(backbone, "get_intermediate_layers"):
                return backbone
        raise AttributeError(
            "Could not find a ViT backbone with `get_intermediate_layers` on the loaded DepthAnythingV2 model."
        )

    def _preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.max() > 5:
            obs = obs / 255.0
        obs = obs.float()

        if self.cfg.resize_to > 0 and (obs.shape[-2] != self.cfg.resize_to or obs.shape[-1] != self.cfg.resize_to):
            obs = F.interpolate(
                obs,
                size=(self.cfg.resize_to, self.cfg.resize_to),
                mode="bilinear",
                align_corners=False,
            )

        return (obs - self.pixel_mean) / self.pixel_std

    def _extract_tokens_and_cls(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone.get_intermediate_layers(
            obs,
            self.cfg.num_intermediate_layers,
            return_class_token=True,
        )
        patch_tokens, cls_token = outputs[self.cfg.feature_layer]
        return patch_tokens, cls_token

    def forward_patches_and_cls(
        self,
        obs: torch.Tensor,
        *,
        flatten_patches: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self._preprocess(obs)

        if self.freeze_encoder:
            with torch.no_grad():
                patch_tokens, cls_token = self._extract_tokens_and_cls(obs)
        else:
            patch_tokens, cls_token = self._extract_tokens_and_cls(obs)

        if flatten_patches:
            patch_tokens = patch_tokens.flatten(1, 2)
        return patch_tokens, cls_token

    def forward_cls(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward_patches_and_cls(obs, flatten_patches=False)[1]


if __name__ == "__main__":
    vit_encoder = VitEncoder(obs_shape=(3, 84, 84), cfg=VitEncoderConfig())
    obs = torch.rand(10, 3, 84, 84)
    feats: torch.Tensor = vit_encoder(obs)
    print(feats.size())  # (10, 10368), i.e., 81 patches * 128 dimensions
