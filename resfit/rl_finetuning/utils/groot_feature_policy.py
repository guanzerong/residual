from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F


TASK_PROMPTS = {
    "lift": "pick up the object on the table and hold it",
    "can": "pick up the coke can and place it on the correct place",
    "square": "pick a square nut and place it on a rod",
    "toolhang": "assemble a frame consisting of a base piece and hook piece by inserting the hook into the base, and hang a wrench on the hook",
}


def _normalize_min_max(values: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    min_value = np.asarray(stats["min"], dtype=np.float32)
    max_value = np.asarray(stats["max"], dtype=np.float32)
    denom = max_value - min_value
    mask = np.abs(denom) > 1e-8
    out = np.zeros_like(values, dtype=np.float32)
    out[..., mask] = ((values[..., mask] - min_value[mask]) / denom[mask]) * 2.0 - 1.0
    return out


def _quat_xyzw_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    if np.linalg.norm(quat_xyzw) < 1e-8:
        quat_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_euler("XYZ", degrees=False).astype(np.float32)


def _rotvec_to_euler_xyz(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    return Rotation.from_rotvec(rotvec).as_euler("XYZ", degrees=False).astype(np.float32)


def _euler_xyz_to_rotvec(euler_xyz: np.ndarray) -> np.ndarray:
    euler_xyz = np.asarray(euler_xyz, dtype=np.float64)
    return Rotation.from_euler("XYZ", euler_xyz, degrees=False).as_rotvec().astype(np.float32)


def _euler_xyz_to_rotation6d(euler_xyz: np.ndarray) -> np.ndarray:
    matrix = Rotation.from_euler("XYZ", euler_xyz, degrees=False).as_matrix().astype(np.float32)
    return matrix[:2, :].reshape(6).astype(np.float32)


def _center_crop_resize(image_hwc: np.ndarray, *, crop_scale: float = 0.95, size: int = 224) -> np.ndarray:
    image_hwc = np.asarray(image_hwc)
    if image_hwc.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {tuple(image_hwc.shape)}.")
    if image_hwc.dtype != np.uint8:
        if np.issubdtype(image_hwc.dtype, np.floating):
            max_value = float(image_hwc.max()) if image_hwc.size else 1.0
            scale = 255.0 if max_value <= 1.0 + 1e-6 else 1.0
            image_hwc = np.clip(image_hwc * scale, 0.0, 255.0).astype(np.uint8)
        else:
            image_hwc = np.clip(image_hwc, 0, 255).astype(np.uint8)

    height, width = image_hwc.shape[:2]
    crop_h = max(1, int(round(height * crop_scale)))
    crop_w = max(1, int(round(width * crop_scale)))
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    cropped = image_hwc[top : top + crop_h, left : left + crop_w]
    return np.asarray(Image.fromarray(cropped).resize((size, size), resample=Image.BILINEAR), dtype=np.uint8)


def _resize_image(image_hwc: np.ndarray, *, size: int = 224) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(image_hwc, dtype=np.uint8)).resize(
            (size, size),
            resample=Image.BILINEAR,
        ),
        dtype=np.uint8,
    )


def _to_bhwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4:
        raise ValueError(f"Expected image batch with 4 dims, got {tuple(image.shape)}.")
    if image.shape[1] in {1, 3}:
        image = np.transpose(image, (0, 2, 3, 1))
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            max_value = float(image.max()) if image.size else 1.0
            scale = 255.0 if max_value <= 1.0 + 1e-6 else 1.0
            image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _map_residual_state_batch_to_oxe(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 1:
        state = state[None]
    if state.ndim != 2:
        raise ValueError(f"Expected residual state with shape [B, D], got {tuple(state.shape)}.")

    batch = int(state.shape[0])
    pos = np.zeros((batch, 3), dtype=np.float32)
    euler = np.zeros((batch, 3), dtype=np.float32)
    gripper = np.zeros((batch, 1), dtype=np.float32)

    for idx, item in enumerate(state):
        if item.shape[-1] == 9:
            pos[idx] = item[:3]
            euler[idx] = _quat_xyzw_to_euler_xyz(item[3:7])
            gripper[idx, 0] = float(np.mean(item[7:9]))
        elif item.shape[-1] == 8:
            pos[idx] = item[:3]
            euler[idx] = _rotvec_to_euler_xyz(item[3:6])
            gripper[idx, 0] = float(np.mean(item[6:8]))
        else:
            raise ValueError(f"Expected residual state dim 8 or 9, got {item.shape[-1]}.")
    return pos, euler, gripper


def _map_residual_state_batch_to_libero(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 1:
        state = state[None]
    if state.ndim != 2:
        raise ValueError(f"Expected residual state with shape [B, D], got {tuple(state.shape)}.")

    batch = int(state.shape[0])
    pos = np.zeros((batch, 3), dtype=np.float32)
    rotvec = np.zeros((batch, 3), dtype=np.float32)
    gripper = np.zeros((batch, 2), dtype=np.float32)

    for idx, item in enumerate(state):
        if item.shape[-1] == 9:
            pos[idx] = item[:3]
            quat_xyzw = np.asarray(item[3:7], dtype=np.float64)
            if np.linalg.norm(quat_xyzw) < 1e-8:
                quat_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            rotvec[idx] = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
            gripper[idx] = item[7:9]
        elif item.shape[-1] == 8:
            pos[idx] = item[:3]
            rotvec[idx] = item[3:6]
            gripper[idx] = item[6:8]
        else:
            raise ValueError(f"Expected residual state dim 8 or 9, got {item.shape[-1]}.")
    return pos, rotvec, gripper


class GROOTFeaturePolicy:
    def __init__(
        self,
        *,
        groot_root: str,
        model_path: str,
        task_name: str,
        default_prompt: str | None = None,
        token_target_count: int = 32,
        base_image_key: str = "observation.images.agentview",
        wrist_image_key: str = "observation.images.robot0_eye_in_hand",
        exterior_2_image_key: str | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self._groot_root = Path(groot_root).expanduser().resolve()
        self._ensure_import_path(self._groot_root)

        from gr00t.data.schema import DatasetMetadata
        from gr00t.model.gr00t_n1 import GR00T_N1_5
        from gr00t.model.transforms import GR00TTransform

        self._device = torch.device(device)
        self._compute_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._model_path = self._resolve_model_path(model_path)
        self._task_name = task_name
        self._default_prompt = default_prompt or TASK_PROMPTS.get(task_name.lower()) or "perform the task"
        self._token_target_count = int(token_target_count)
        self._base_image_key = str(base_image_key)
        self._wrist_image_key = str(wrist_image_key)
        self._exterior_2_image_key = str(exterior_2_image_key) if exterior_2_image_key is not None else None

        metadata_path = self._model_path / "experiment_cfg" / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata_by_embodiment = json.load(f)
        if "new_embodiment" in metadata_by_embodiment:
            self._embodiment = "new_embodiment"
        elif "oxe_droid" in metadata_by_embodiment:
            self._embodiment = "oxe_droid"
        else:
            available = ", ".join(sorted(metadata_by_embodiment))
            raise ValueError(f"Unsupported GR00T checkpoint metadata; available embodiments: {available}.")
        metadata_dict = metadata_by_embodiment[self._embodiment]
        self._metadata = DatasetMetadata.model_validate(metadata_dict)
        self._state_stats = metadata_dict["statistics"]["state"]

        if self._embodiment == "new_embodiment":
            from examples.Libero.custom_data_config import LiberoDataConfig

            self._transform = LiberoDataConfig().transform()
        else:
            self._transform = GR00TTransform(
                state_horizon=1,
                action_horizon=16,
                max_state_dim=64,
                max_action_dim=32,
                default_instruction=self._default_prompt,
            )
        self._transform.set_metadata(self._metadata)
        self._transform.eval()

        self._model = GR00T_N1_5.from_pretrained(
            str(self._model_path),
            torch_dtype=self._compute_dtype,
            tune_visual=False,
            tune_llm=False,
            tune_projector=False,
            tune_diffusion_model=False,
        )
        self._model.eval()
        self._model.to(self._device)

        self.metadata = {
            "provider": "groot_remote",
            "model_path": str(model_path),
            "task_name": task_name,
            "embodiment": self._embodiment,
            "action_horizon": int(self._model.action_horizon),
            "action_dim": 7,
            "token_target_count": int(self._token_target_count),
            "token_dim": int(self._model.config.action_head_cfg["backbone_embedding_dim"]),
        }

    @staticmethod
    def _ensure_import_path(groot_root: Path) -> None:
        root_str = str(groot_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        try:
            local_path = snapshot_download(model_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            local_path = model_path
        return Path(local_path).expanduser().resolve()

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions, _ = self._infer_and_encode(obs, return_features=False)
        return {"actions": actions}

    def infer_and_encode(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions, features = self._infer_and_encode(obs, return_features=True)
        assert features is not None
        return {"actions": actions, "observation_features": features}

    def infer_and_encode_batch(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return self.infer_and_encode(obs)

    def _infer_and_encode(
        self,
        obs: dict[str, Any],
        *,
        return_features: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        inputs = self._build_model_inputs(obs)
        autocast = (
            torch.autocast(device_type="cuda", dtype=self._compute_dtype)
            if self._device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            backbone_inputs, action_inputs = self._model.prepare_input(inputs)
            backbone_output = self._model.backbone(backbone_inputs)
            backbone_output = self._model.action_head.process_backbone_output(backbone_output)
            token_tensor = backbone_output["backbone_features"]
            token_mask = backbone_output["backbone_attention_mask"]
            action_tensor = self._sample_actions_from_processed_backbone(backbone_output, action_inputs)

        action_chunks = self._decode_action_chunk(action_tensor)
        pooled_features = self._pool_tokens(token_tensor, token_mask) if return_features else None
        return action_chunks, pooled_features

    def _build_model_inputs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._embodiment == "new_embodiment":
            return self._build_libero_model_inputs(obs)

        base = _to_bhwc_uint8(obs[self._base_image_key])
        wrist = _to_bhwc_uint8(obs[self._wrist_image_key])
        if self._exterior_2_image_key is None:
            exterior_2 = base
        else:
            exterior_2 = _to_bhwc_uint8(obs[self._exterior_2_image_key])

        base = np.stack([_center_crop_resize(image) for image in base], axis=0)
        exterior_2 = np.stack([_center_crop_resize(image) for image in exterior_2], axis=0)
        wrist = np.stack([_center_crop_resize(image) for image in wrist], axis=0)

        pos, euler_xyz, gripper = _map_residual_state_batch_to_oxe(obs["observation.state"])
        state_eef_position = _normalize_min_max(pos, self._state_stats["eef_position"])
        state_eef_rotation = np.stack(
            [_euler_xyz_to_rotation6d(item) for item in euler_xyz],
            axis=0,
        ).astype(np.float32)
        state_gripper = _normalize_min_max(gripper, self._state_stats["gripper_position"])

        state = np.concatenate([state_eef_position, state_eef_rotation, state_gripper], axis=-1).astype(np.float32)
        video = np.stack([base, exterior_2, wrist], axis=1)[:, None, ...]

        prompts = obs.get("prompt")
        if prompts is None:
            prompts = [self._default_prompt] * int(video.shape[0])
        elif isinstance(prompts, str):
            prompts = [prompts] * int(video.shape[0])
        else:
            prompts = list(prompts)

        transformed = self._transform.apply(
            {
                "video": video,
                "state": state[:, None, :],
                "annotation.language.language_instruction": prompts,
            }
        )
        return transformed

    def _build_libero_model_inputs(self, obs: dict[str, Any]) -> dict[str, Any]:
        base = _to_bhwc_uint8(obs[self._base_image_key])
        wrist = _to_bhwc_uint8(obs[self._wrist_image_key])
        base = np.stack([_resize_image(image) for image in base], axis=0)
        wrist = np.stack([_resize_image(image) for image in wrist], axis=0)
        pos, rotvec, gripper = _map_residual_state_batch_to_libero(obs["observation.state"])

        prompts = obs.get("prompt")
        if prompts is None:
            prompts = [self._default_prompt] * int(base.shape[0])
        elif isinstance(prompts, str):
            prompts = [prompts] * int(base.shape[0])
        else:
            prompts = list(prompts)

        raw_inputs = {
            "video.image": base[:, None, ...],
            "video.wrist_image": wrist[:, None, ...],
            "state.x": pos[:, None, 0:1],
            "state.y": pos[:, None, 1:2],
            "state.z": pos[:, None, 2:3],
            "state.roll": rotvec[:, None, 0:1],
            "state.pitch": rotvec[:, None, 1:2],
            "state.yaw": rotvec[:, None, 2:3],
            "state.gripper": gripper[:, None, :],
            "annotation.human.action.task_description": np.asarray(prompts, dtype=object)[:, None],
        }
        return self._transform(raw_inputs)

    def _sample_actions_from_processed_backbone(
        self,
        backbone_output,
        action_input,
    ) -> torch.Tensor:
        action_head = self._model.action_head
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = action_head.state_encoder(action_input.state, embodiment_id)

        batch_size = int(vl_embs.shape[0])
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = int(action_head.num_inference_timesteps)
        dt = 1.0 / float(num_steps)

        for step in range(num_steps):
            t_discretized = int((step / float(num_steps)) * action_head.num_timestep_buckets)
            timesteps_tensor = torch.full((batch_size,), fill_value=t_discretized, device=device)
            action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_head.action_horizon :]
            actions = actions + dt * pred_velocity
        return actions

    def _decode_action_chunk(self, action_tensor: torch.Tensor) -> np.ndarray:
        if self._embodiment == "new_embodiment":
            decoded = self._transform.unapply({"action": action_tensor.detach().cpu().float()})
            action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            return np.concatenate(
                [np.asarray(decoded[f"action.{key}"], dtype=np.float32) for key in action_keys],
                axis=-1,
            ).astype(np.float32)

        raw_actions = action_tensor.detach().cpu().float().numpy()
        pos_delta = raw_actions[..., 0:3]
        rot_axis_angle = raw_actions[..., 3:6]
        gripper_binary = (raw_actions[..., 6:7] > 0.5).astype(np.float32)
        gripper_action = gripper_binary * 2.0 - 1.0
        return np.concatenate([pos_delta, rot_axis_angle, gripper_action], axis=-1).astype(np.float32)

    def _pool_tokens(self, token_tensor: torch.Tensor, token_mask: torch.Tensor) -> np.ndarray:
        tokens = token_tensor.detach().cpu().float()
        mask = token_mask.detach().cpu().bool()

        pooled_batch = []
        for features, feature_mask in zip(tokens, mask, strict=False):
            valid_tokens = features[feature_mask]
            if valid_tokens.numel() == 0:
                raise RuntimeError("GR00T returned an empty valid token sequence.")
            if self._token_target_count <= 0:
                pooled = valid_tokens
            else:
                pooled = F.adaptive_avg_pool1d(
                    valid_tokens.T.unsqueeze(0),
                    self._token_target_count,
                ).squeeze(0).T
            pooled_batch.append(pooled.numpy())

        target_len = pooled_batch[0].shape[0]
        if any(item.shape[0] != target_len for item in pooled_batch):
            raise RuntimeError("GR00T pooled token count is inconsistent across the batch.")
        return np.stack(pooled_batch, axis=0).astype(np.float32)
