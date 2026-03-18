from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_OPENPI_CLIENT_SRC = Path(__file__).resolve().parents[4] / "openpi" / "packages" / "openpi-client" / "src"
if _OPENPI_CLIENT_SRC.exists() and str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

from openpi_client import websocket_client_policy as _websocket_client_policy


@dataclass
class OpenPIChunkPolicyConfig:
    chunk_size: int
    image_keys: tuple[str, ...]

    @property
    def image_features(self) -> dict[str, object]:
        return {key: object() for key in self.image_keys}


class OpenPIChunkPolicy:
    """Base-policy adapter that queries an OpenPI websocket server."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        task_prompt: str,
        chunk_size: int = 20,
        image_key: str = "observation.images.agentview",
        wrist_image_key: str = "observation.images.robot0_eye_in_hand",
        api_key: str | None = None,
    ) -> None:
        self._client = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port, api_key=api_key)
        self._task_prompt = task_prompt
        self._image_key = image_key
        self._wrist_image_key = wrist_image_key
        self.config = OpenPIChunkPolicyConfig(
            chunk_size=chunk_size,
            image_keys=(image_key, wrist_image_key),
        )

    def to(self, device: str | torch.device) -> OpenPIChunkPolicy:
        return self

    def eval(self) -> OpenPIChunkPolicy:
        return self

    def reset(self) -> None:
        return None

    def select_action_chunk(self, batch: dict[str, torch.Tensor], n_steps: int | None = None) -> torch.Tensor:
        batch_device = self._infer_batch_device(batch)
        payload = self._build_openpi_observation(batch)
        actions = self._client.infer(payload)["actions"]
        actions_np = np.asarray(actions, dtype=np.float32)
        if actions_np.ndim != 2:
            raise ValueError(f"Expected OpenPI action chunk with shape (T, A), got {actions_np.shape}.")

        requested_steps = n_steps if n_steps is not None else actions_np.shape[0]
        if requested_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {requested_steps}.")
        if actions_np.shape[0] < requested_steps:
            if actions_np.shape[0] == 0:
                raise ValueError("OpenPI server returned an empty action chunk.")
            pad = np.repeat(actions_np[-1:, :], requested_steps - actions_np.shape[0], axis=0)
            actions_np = np.concatenate([actions_np, pad], axis=0)
        else:
            actions_np = actions_np[:requested_steps]

        return torch.from_numpy(actions_np).to(batch_device).unsqueeze(0)

    def _build_openpi_observation(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "observation.state" not in batch:
            raise KeyError("OpenPIChunkPolicy requires observation.state in the input batch.")
        if self._image_key not in batch:
            raise KeyError(f"OpenPIChunkPolicy requires {self._image_key} in the input batch.")
        if self._wrist_image_key not in batch:
            raise KeyError(f"OpenPIChunkPolicy requires {self._wrist_image_key} in the input batch.")

        state = self._extract_state(batch["observation.state"])
        image = self._extract_image(batch[self._image_key])
        wrist_image = self._extract_image(batch[self._wrist_image_key])
        return {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "prompt": self._task_prompt,
        }

    def _extract_state(self, value: Any) -> np.ndarray:
        state = self._to_numpy(value)
        if state.ndim == 2:
            if state.shape[0] != 1:
                raise ValueError(f"OpenPIChunkPolicy only supports batch size 1, got state batch {state.shape}.")
            state = state[0]
        if state.shape != (8,) and state.shape != (9,):
            raise ValueError(
                "OpenPIChunkPolicy expects single-arm state with 8 dims (already converted) or "
                f"9 dims (pos + quat + gripper). Got {state.shape}."
            )
        if state.shape == (8,):
            return state.astype(np.float32, copy=False)

        quat = state[3:7].astype(np.float32, copy=True)
        axis_angle = self._quat_to_axis_angle(quat)
        return np.concatenate([state[:3], axis_angle, state[7:9]], axis=0).astype(np.float32, copy=False)

    def _extract_image(self, value: Any) -> np.ndarray:
        image = self._to_numpy(value)
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(f"OpenPIChunkPolicy only supports batch size 1, got image batch {image.shape}.")
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dims, got {image.shape}.")

        if image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            max_value = float(np.max(image)) if image.size > 0 else 0.0
            if max_value <= 1.0:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _infer_batch_device(batch: dict[str, Any]) -> torch.device:
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")

    @staticmethod
    def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
        quat = quat.astype(np.float32, copy=True)
        quat[3] = np.clip(quat[3], -1.0, 1.0)
        den = math.sqrt(max(1.0 - float(quat[3] * quat[3]), 0.0))
        if math.isclose(den, 0.0):
            return np.zeros(3, dtype=np.float32)
        return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32, copy=False)
