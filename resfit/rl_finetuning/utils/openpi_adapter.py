from __future__ import annotations

import dataclasses
from collections import deque
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import websockets.sync.client


_TASK_PROMPTS = {
    "lift": "pick up the object on the table and hold it",
    "can": "pick up the coke can and place it on the correct place",
    "square": "pick a square nut and place it on a rod",
    "toolhang": "assemble a frame consisting of a base piece and hook piece by inserting the hook into the base, and hang a wrench on the hook",
}


@dataclasses.dataclass(frozen=True)
class _FeatureSpec:
    shape: tuple[int, ...]


def _quat_xyzw_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def _map_residual_state_to_openpi(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] == 8:
        return state
    if state.shape[-1] != 9:
        raise ValueError(f"Expected residual state dim 8 or 9, got {state.shape[-1]}.")
    pos = state[:3]
    quat = state[3:7]
    gripper = state[7:9]
    rotvec = _quat_xyzw_to_axisangle(quat)
    return np.concatenate([pos, rotvec, gripper], axis=0).astype(np.float32, copy=False)


def _map_residual_state_batch_to_openpi(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 1:
        return _map_residual_state_to_openpi(state)
    return np.stack([_map_residual_state_to_openpi(item) for item in state], axis=0).astype(np.float32, copy=False)


class OpenPIBasePolicyAdapter(nn.Module):
    """Adapter that makes a frozen OpenPI policy look like the ACT base-policy interface."""

    _OPENPI_IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    _PATCH_GRID_SIZE = 14
    _PATCHES_PER_IMAGE = _PATCH_GRID_SIZE * _PATCH_GRID_SIZE

    def __init__(
        self,
        *,
        openpi_root: str,
        train_config_name: str,
        checkpoint_dir: str,
        state_dim: int,
        task_name: str,
        token_pool_size: int,
        base_image_key: str,
        left_wrist_image_key: str,
        right_wrist_image_key: str | None = None,
        default_prompt: str | None = None,
    ) -> None:
        super().__init__()

        if token_pool_size < 1:
            raise ValueError(f"token_pool_size must be >= 1, got {token_pool_size}")

        self._openpi_root = Path(openpi_root).expanduser().resolve()
        self._ensure_openpi_import_paths(self._openpi_root)

        from openpi.policies import policy_config as openpi_policy_config
        from openpi.training import config as openpi_training_config

        prompt = default_prompt or _TASK_PROMPTS.get(task_name.lower())
        if prompt is None:
            raise ValueError(
                "OpenPI base policy requires a default prompt for inference. "
                f"Set base_policy.openpi_default_prompt explicitly for task={task_name!r}."
            )

        self._policy = openpi_policy_config.create_trained_policy(
            openpi_training_config.get_config(train_config_name),
            checkpoint_dir,
            default_prompt=prompt,
        )
        self._token_pool_size = int(token_pool_size)
        self._camera_mapping = {
            "base_0_rgb": base_image_key,
            "left_wrist_0_rgb": left_wrist_image_key,
            "right_wrist_0_rgb": right_wrist_image_key,
        }
        self._active_slots = [
            slot_name for slot_name in self._OPENPI_IMAGE_SLOTS if self._camera_mapping.get(slot_name) is not None
        ]
        self._action_horizon = int(self._policy.metadata.get("action_horizon", 0) or 0)
        if self._action_horizon <= 0:
            # Fall back to the train config horizon when metadata is absent.
            self._action_horizon = int(openpi_training_config.get_config(train_config_name).model.action_horizon)

        pooled_tokens_per_camera = self._token_pool_size * self._token_pool_size
        self._pooled_tokens = pooled_tokens_per_camera * len(self._active_slots)
        self._token_dim: int | None = None
        self._action_queues: list[deque[torch.Tensor]] = []

        image_features = {
            camera_key: _FeatureSpec(shape=(3, 84, 84))
            for camera_key in (base_image_key, left_wrist_image_key, right_wrist_image_key)
            if camera_key is not None
        }
        self.config = SimpleNamespace(
            n_action_steps=self._action_horizon,
            image_features=image_features,
            robot_state_feature=_FeatureSpec(shape=(state_dim,)),
            env_state_feature=None,
        )

    def clone_for_eval(self) -> "OpenPIBasePolicyAdapter":
        clone = self.__class__.__new__(self.__class__)
        nn.Module.__init__(clone)
        clone._openpi_root = self._openpi_root
        clone._policy = self._policy
        clone._token_pool_size = self._token_pool_size
        clone._camera_mapping = dict(self._camera_mapping)
        clone._active_slots = list(self._active_slots)
        clone._action_horizon = self._action_horizon
        clone._pooled_tokens = self._pooled_tokens
        clone._token_dim = self._token_dim
        clone._action_queues = []
        clone.config = SimpleNamespace(
            n_action_steps=self.config.n_action_steps,
            image_features=dict(self.config.image_features),
            robot_state_feature=self.config.robot_state_feature,
            env_state_feature=self.config.env_state_feature,
        )
        return clone

    @staticmethod
    def _ensure_openpi_import_paths(openpi_root: Path) -> None:
        candidates = [
            openpi_root / "src",
            openpi_root / "packages" / "openpi-client" / "src",
        ]
        for path in candidates:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("OpenPIBasePolicyAdapter is an inference-only wrapper.")

    def to(self, *args, **kwargs):
        # The wrapped OpenPI JAX policy manages its own device placement.
        return self

    def eval(self):
        super().eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        if env_ids is None:
            self._action_queues = []
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        for env_idx in env_ids:
            if 0 <= int(env_idx) < len(self._action_queues):
                self._action_queues[int(env_idx)].clear()

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = self._batch_size_from_obs(batch)
        self._ensure_action_queues(batch_size)
        outputs: list[torch.Tensor] = []
        for env_idx in range(batch_size):
            queue = self._action_queues[env_idx]
            if not queue:
                chunk = self.select_action_chunk(self._slice_batch(batch, env_idx), n_action_steps=self.config.n_action_steps)
                for action in chunk[0]:
                    queue.append(action.detach())
            outputs.append(queue.popleft())
        return torch.stack(outputs, dim=0)

    def select_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        n_action_steps: int | None = None,
    ) -> torch.Tensor:
        batch_size = self._batch_size_from_obs(batch)
        requested_steps = int(n_action_steps or self.config.n_action_steps)
        outputs = []
        for env_idx in range(batch_size):
            obs_np = self._build_openpi_input(self._slice_batch(batch, env_idx))
            action_chunk = self._policy.infer(obs_np)["actions"]
            action_chunk = np.asarray(action_chunk[:requested_steps], dtype=np.float32)
            outputs.append(torch.from_numpy(action_chunk))
        return torch.stack(outputs, dim=0).to(device=self._obs_device(batch))

    @torch.no_grad()
    def infer_action_chunk_and_observation_features(
        self,
        batch: dict[str, torch.Tensor],
        n_action_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = self._batch_size_from_obs(batch)
        requested_steps = int(n_action_steps or self.config.n_action_steps)
        if batch_size > 1 and hasattr(self._policy, "infer_and_encode_batch"):
            result = self._policy.infer_and_encode_batch(self._build_openpi_batch_input(batch))
            action_chunks = np.asarray(result["actions"][:, :requested_steps], dtype=np.float32)
            encoded_batch = np.asarray(result["observation_features"], dtype=np.float32)
            pooled_batch = np.stack([self._pool_visual_tokens(encoded) for encoded in encoded_batch], axis=0)
            device = self._obs_device(batch)
            actions = torch.from_numpy(action_chunks).to(device=device)
            tokens = torch.from_numpy(pooled_batch).to(device=device)
            if self._token_dim is None:
                self._token_dim = int(tokens.shape[-1])
            return actions, tokens
        action_outputs = []
        feature_outputs = []
        for env_idx in range(batch_size):
            obs_np = self._build_openpi_input(self._slice_batch(batch, env_idx))
            result = self._policy.infer_and_encode(obs_np)
            action_chunk = np.asarray(result["actions"][:requested_steps], dtype=np.float32)
            encoded = np.asarray(result["observation_features"], dtype=np.float32)
            pooled = self._pool_visual_tokens(encoded)
            action_outputs.append(torch.from_numpy(action_chunk))
            feature_outputs.append(torch.from_numpy(pooled))
        device = self._obs_device(batch)
        actions = torch.stack(action_outputs, dim=0).to(device=device)
        tokens = torch.stack(feature_outputs, dim=0).to(device=device)
        if self._token_dim is None:
            self._token_dim = int(tokens.shape[-1])
        return actions, tokens

    @torch.no_grad()
    def encode_observation_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = self._batch_size_from_obs(batch)
        outputs = []
        for env_idx in range(batch_size):
            obs_np = self._build_openpi_input(self._slice_batch(batch, env_idx))
            encoded = np.asarray(self._policy.encode_observation_features(obs_np), dtype=np.float32)
            pooled = self._pool_visual_tokens(encoded)
            outputs.append(torch.from_numpy(pooled))
        tokens = torch.stack(outputs, dim=0).to(device=self._obs_device(batch))
        if self._token_dim is None:
            self._token_dim = int(tokens.shape[-1])
        return tokens

    def _ensure_action_queues(self, batch_size: int) -> None:
        while len(self._action_queues) < batch_size:
            self._action_queues.append(deque())

    @staticmethod
    def _batch_size_from_obs(batch: dict[str, torch.Tensor]) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0]) if value.dim() > 0 else 1
        raise ValueError("Observation batch is empty.")

    @staticmethod
    def _obs_device(batch: dict[str, torch.Tensor]) -> torch.device:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")

    @staticmethod
    def _slice_batch(batch: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
        sliced: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                sliced[key] = value[index : index + 1]
            else:
                sliced[key] = value
        return sliced

    def _build_openpi_input(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        state = batch["observation.state"]
        if state.dim() == 2:
            state = state[0]
        result: dict[str, Any] = {
            "observation/image": self._to_hwc_uint8(batch[self._camera_mapping["base_0_rgb"]]),
            "observation/wrist_image": self._to_hwc_uint8(batch[self._camera_mapping["left_wrist_0_rgb"]]),
            "observation/state": _map_residual_state_to_openpi(state.detach().cpu().float().numpy()),
        }
        return result

    def _build_openpi_batch_input(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {
            "observation/image": self._to_bhwc_uint8(batch[self._camera_mapping["base_0_rgb"]]),
            "observation/wrist_image": self._to_bhwc_uint8(batch[self._camera_mapping["left_wrist_0_rgb"]]),
            "observation/state": _map_residual_state_batch_to_openpi(
                batch["observation.state"].detach().cpu().float().numpy()
            ),
        }

    @staticmethod
    def _to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
        if image.dim() == 4:
            image = image[0]
        image = image.detach().cpu()
        if image.dim() != 3:
            raise ValueError(f"Expected image tensor with 3 dims, got shape {tuple(image.shape)}")
        if image.shape[0] in {1, 3}:
            image = image.permute(1, 2, 0)
        np_image = image.numpy()
        if np.issubdtype(np_image.dtype, np.floating):
            max_value = float(np_image.max()) if np_image.size else 1.0
            scale = 255.0 if max_value <= 1.0 + 1e-6 else 1.0
            np_image = np.clip(np_image * scale, 0.0, 255.0).astype(np.uint8)
        else:
            np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    @staticmethod
    def _to_bhwc_uint8(image: torch.Tensor) -> np.ndarray:
        image = image.detach().cpu()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4:
            raise ValueError(f"Expected image tensor with 4 dims, got shape {tuple(image.shape)}")
        if image.shape[1] in {1, 3}:
            image = image.permute(0, 2, 3, 1)
        np_image = image.numpy()
        if np.issubdtype(np_image.dtype, np.floating):
            max_value = float(np_image.max()) if np_image.size else 1.0
            scale = 255.0 if max_value <= 1.0 + 1e-6 else 1.0
            np_image = np.clip(np_image * scale, 0.0, 255.0).astype(np.uint8)
        else:
            np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    def _pool_visual_tokens(self, encoded_tokens: np.ndarray) -> np.ndarray:
        pooled_per_camera = []
        token_dim = int(encoded_tokens.shape[-1])
        for slot_idx, slot_name in enumerate(self._OPENPI_IMAGE_SLOTS):
            if slot_name not in self._active_slots:
                continue
            start = slot_idx * self._PATCHES_PER_IMAGE
            end = start + self._PATCHES_PER_IMAGE
            camera_tokens = encoded_tokens[start:end]
            if camera_tokens.shape[0] != self._PATCHES_PER_IMAGE:
                raise ValueError(
                    "Unexpected OpenPI token layout. "
                    f"Expected {self._PATCHES_PER_IMAGE} visual tokens per camera, got {camera_tokens.shape[0]} for {slot_name}."
                )
            pooled_per_camera.append(self._adaptive_pool_tokens(camera_tokens, token_dim))
        return np.concatenate(pooled_per_camera, axis=0).astype(np.float32, copy=False)

    def _adaptive_pool_tokens(self, camera_tokens: np.ndarray, token_dim: int) -> np.ndarray:
        token_tensor = torch.from_numpy(camera_tokens).reshape(
            self._PATCH_GRID_SIZE,
            self._PATCH_GRID_SIZE,
            token_dim,
        )
        token_tensor = token_tensor.permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(token_tensor, (self._token_pool_size, self._token_pool_size))
        pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(-1, token_dim)
        return pooled.numpy()


class OpenPIRemoteBasePolicyAdapter(nn.Module):
    """Client-side adapter for an OpenPI websocket server that returns both actions and observation features."""

    _OPENPI_IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    _PATCH_GRID_SIZE = 14
    _PATCHES_PER_IMAGE = _PATCH_GRID_SIZE * _PATCH_GRID_SIZE

    def __init__(
        self,
        *,
        openpi_root: str,
        host: str,
        port: int,
        state_dim: int,
        token_pool_size: int,
        base_image_key: str,
        left_wrist_image_key: str,
        right_wrist_image_key: str | None = None,
    ) -> None:
        super().__init__()
        if token_pool_size < 1:
            raise ValueError(f"token_pool_size must be >= 1, got {token_pool_size}")

        self._openpi_root = Path(openpi_root).expanduser().resolve()
        self._ensure_openpi_client_import_paths(self._openpi_root)
        from openpi_client import msgpack_numpy

        self._packer = msgpack_numpy.Packer()
        self._host = host
        self._port = int(port)
        self._token_pool_size = int(token_pool_size)
        self._camera_mapping = {
            "base_0_rgb": base_image_key,
            "left_wrist_0_rgb": left_wrist_image_key,
            "right_wrist_0_rgb": right_wrist_image_key,
        }
        self._active_slots = [
            slot_name for slot_name in self._OPENPI_IMAGE_SLOTS if self._camera_mapping.get(slot_name) is not None
        ]
        self._lock = threading.Lock()
        self._ws, metadata = self._connect()
        self._metadata = dict(metadata)
        self._action_horizon = int(self._metadata.get("action_horizon", 0) or 0)
        if self._action_horizon <= 0:
            raise ValueError("Remote OpenPI server metadata must include a positive action_horizon.")

        pooled_tokens_per_camera = self._token_pool_size * self._token_pool_size
        self._pooled_tokens = pooled_tokens_per_camera * len(self._active_slots)
        self._token_dim: int | None = None
        self._action_queues: list[deque[torch.Tensor]] = []

        image_features = {
            camera_key: _FeatureSpec(shape=(3, 84, 84))
            for camera_key in (base_image_key, left_wrist_image_key, right_wrist_image_key)
            if camera_key is not None
        }
        self.config = SimpleNamespace(
            n_action_steps=self._action_horizon,
            image_features=image_features,
            robot_state_feature=_FeatureSpec(shape=(state_dim,)),
            env_state_feature=None,
        )

    @staticmethod
    def _ensure_openpi_client_import_paths(openpi_root: Path) -> None:
        candidates = [openpi_root / "packages" / "openpi-client" / "src"]
        for path in candidates:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    def _connect(self):
        from openpi_client import msgpack_numpy

        uri = f"ws://{self._host}:{self._port}"
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    def clone_for_eval(self) -> "OpenPIRemoteBasePolicyAdapter":
        return OpenPIRemoteBasePolicyAdapter(
            openpi_root=str(self._openpi_root),
            host=self._host,
            port=self._port,
            state_dim=int(self.config.robot_state_feature.shape[0]),
            token_pool_size=self._token_pool_size,
            base_image_key=self._camera_mapping["base_0_rgb"],
            left_wrist_image_key=self._camera_mapping["left_wrist_0_rgb"],
            right_wrist_image_key=self._camera_mapping["right_wrist_0_rgb"],
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("OpenPIRemoteBasePolicyAdapter is an inference-only wrapper.")

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        super().eval()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        if env_ids is None:
            self._action_queues = []
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        for env_idx in env_ids:
            if 0 <= int(env_idx) < len(self._action_queues):
                self._action_queues[int(env_idx)].clear()

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = self._batch_size_from_obs(batch)
        self._ensure_action_queues(batch_size)
        outputs: list[torch.Tensor] = []
        for env_idx in range(batch_size):
            queue = self._action_queues[env_idx]
            if not queue:
                chunk = self.select_action_chunk(self._slice_batch(batch, env_idx), n_action_steps=self.config.n_action_steps)
                for action in chunk[0]:
                    queue.append(action.detach())
            outputs.append(queue.popleft())
        return torch.stack(outputs, dim=0)

    def select_action_chunk(self, batch: dict[str, torch.Tensor], n_action_steps: int | None = None) -> torch.Tensor:
        batch_size = self._batch_size_from_obs(batch)
        requested_steps = int(n_action_steps or self.config.n_action_steps)
        outputs = []
        for env_idx in range(batch_size):
            obs_np = self._build_openpi_input(self._slice_batch(batch, env_idx))
            response = self._request({"op": "infer", "obs": obs_np})
            action_chunk = np.asarray(response["actions"][:requested_steps], dtype=np.float32)
            outputs.append(torch.from_numpy(action_chunk))
        return torch.stack(outputs, dim=0).to(device=self._obs_device(batch))

    @torch.no_grad()
    def infer_action_chunk_and_observation_features(
        self,
        batch: dict[str, torch.Tensor],
        n_action_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = self._batch_size_from_obs(batch)
        requested_steps = int(n_action_steps or self.config.n_action_steps)
        if batch_size > 1:
            response = self._request({"op": "infer_and_encode_batch", "obs": self._build_openpi_batch_input(batch)})
            action_chunks = np.asarray(response["actions"][:, :requested_steps], dtype=np.float32)
            encoded_batch = np.asarray(response["observation_features"], dtype=np.float32)
            pooled_batch = np.stack([self._pool_visual_tokens(encoded) for encoded in encoded_batch], axis=0)
            device = self._obs_device(batch)
            actions = torch.from_numpy(action_chunks).to(device=device)
            tokens = torch.from_numpy(pooled_batch).to(device=device)
            if self._token_dim is None:
                self._token_dim = int(tokens.shape[-1])
            return actions, tokens
        action_outputs = []
        feature_outputs = []
        for env_idx in range(batch_size):
            obs_np = self._build_openpi_input(self._slice_batch(batch, env_idx))
            response = self._request({"op": "infer_and_encode", "obs": obs_np})
            action_chunk = np.asarray(response["actions"][:requested_steps], dtype=np.float32)
            encoded = np.asarray(response["observation_features"], dtype=np.float32)
            pooled = self._pool_visual_tokens(encoded)
            action_outputs.append(torch.from_numpy(action_chunk))
            feature_outputs.append(torch.from_numpy(pooled))
        device = self._obs_device(batch)
        actions = torch.stack(action_outputs, dim=0).to(device=device)
        tokens = torch.stack(feature_outputs, dim=0).to(device=device)
        if self._token_dim is None:
            self._token_dim = int(tokens.shape[-1])
        return actions, tokens

    @torch.no_grad()
    def encode_observation_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = self._batch_size_from_obs(batch)
        outputs = []
        for env_idx in range(batch_size):
            obs_np = self._build_openpi_input(self._slice_batch(batch, env_idx))
            response = self._request({"op": "infer_and_encode", "obs": obs_np})
            encoded = np.asarray(response["observation_features"], dtype=np.float32)
            pooled = self._pool_visual_tokens(encoded)
            outputs.append(torch.from_numpy(pooled))
        tokens = torch.stack(outputs, dim=0).to(device=self._obs_device(batch))
        if self._token_dim is None:
            self._token_dim = int(tokens.shape[-1])
        return tokens

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        from openpi_client import msgpack_numpy

        with self._lock:
            self._ws.send(self._packer.pack(payload))
            response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in remote OpenPI server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def _ensure_action_queues(self, batch_size: int) -> None:
        while len(self._action_queues) < batch_size:
            self._action_queues.append(deque())

    @staticmethod
    def _batch_size_from_obs(batch: dict[str, torch.Tensor]) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0]) if value.dim() > 0 else 1
        raise ValueError("Observation batch is empty.")

    @staticmethod
    def _obs_device(batch: dict[str, torch.Tensor]) -> torch.device:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")

    @staticmethod
    def _slice_batch(batch: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
        sliced: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                sliced[key] = value[index : index + 1]
            else:
                sliced[key] = value
        return sliced

    def _build_openpi_input(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        state = batch["observation.state"]
        if state.dim() == 2:
            state = state[0]
        return {
            "observation/image": self._to_hwc_uint8(batch[self._camera_mapping["base_0_rgb"]]),
            "observation/wrist_image": self._to_hwc_uint8(batch[self._camera_mapping["left_wrist_0_rgb"]]),
            "observation/state": _map_residual_state_to_openpi(state.detach().cpu().float().numpy()),
        }

    def _build_openpi_batch_input(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {
            "observation/image": self._to_bhwc_uint8(batch[self._camera_mapping["base_0_rgb"]]),
            "observation/wrist_image": self._to_bhwc_uint8(batch[self._camera_mapping["left_wrist_0_rgb"]]),
            "observation/state": _map_residual_state_batch_to_openpi(
                batch["observation.state"].detach().cpu().float().numpy()
            ),
        }

    @staticmethod
    def _to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
        if image.dim() == 4:
            image = image[0]
        image = image.detach().cpu()
        if image.dim() != 3:
            raise ValueError(f"Expected image tensor with 3 dims, got shape {tuple(image.shape)}")
        if image.shape[0] in {1, 3}:
            image = image.permute(1, 2, 0)
        np_image = image.numpy()
        if np.issubdtype(np_image.dtype, np.floating):
            max_value = float(np_image.max()) if np_image.size else 1.0
            scale = 255.0 if max_value <= 1.0 + 1e-6 else 1.0
            np_image = np.clip(np_image * scale, 0.0, 255.0).astype(np.uint8)
        else:
            np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    @staticmethod
    def _to_bhwc_uint8(image: torch.Tensor) -> np.ndarray:
        image = image.detach().cpu()
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4:
            raise ValueError(f"Expected image tensor with 4 dims, got shape {tuple(image.shape)}")
        if image.shape[1] in {1, 3}:
            image = image.permute(0, 2, 3, 1)
        np_image = image.numpy()
        if np.issubdtype(np_image.dtype, np.floating):
            max_value = float(np_image.max()) if np_image.size else 1.0
            scale = 255.0 if max_value <= 1.0 + 1e-6 else 1.0
            np_image = np.clip(np_image * scale, 0.0, 255.0).astype(np.uint8)
        else:
            np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    def _pool_visual_tokens(self, encoded_tokens: np.ndarray) -> np.ndarray:
        pooled_per_camera = []
        token_dim = int(encoded_tokens.shape[-1])
        for slot_idx, slot_name in enumerate(self._OPENPI_IMAGE_SLOTS):
            if slot_name not in self._active_slots:
                continue
            start = slot_idx * self._PATCHES_PER_IMAGE
            end = start + self._PATCHES_PER_IMAGE
            camera_tokens = encoded_tokens[start:end]
            if camera_tokens.shape[0] != self._PATCHES_PER_IMAGE:
                raise ValueError(
                    "Unexpected OpenPI token layout. "
                    f"Expected {self._PATCHES_PER_IMAGE} visual tokens per camera, got {camera_tokens.shape[0]} for {slot_name}."
                )
            pooled_per_camera.append(self._adaptive_pool_tokens(camera_tokens, token_dim))
        return np.concatenate(pooled_per_camera, axis=0).astype(np.float32, copy=False)

    def _adaptive_pool_tokens(self, camera_tokens: np.ndarray, token_dim: int) -> np.ndarray:
        token_tensor = torch.from_numpy(camera_tokens).reshape(
            self._PATCH_GRID_SIZE,
            self._PATCH_GRID_SIZE,
            token_dim,
        )
        token_tensor = token_tensor.permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(token_tensor, (self._token_pool_size, self._token_pool_size))
        pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(-1, token_dim)
        return pooled.numpy()
