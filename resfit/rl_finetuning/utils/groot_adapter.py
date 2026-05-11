from __future__ import annotations

import dataclasses
from collections import deque
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import msgpack
import numpy as np
import torch
from torch import nn
try:
    import zmq
except ModuleNotFoundError:
    groot_venv_site = Path("/data_all/gzr1/code/Isaac-GR00T-n1.5/.venv/lib/python3.10/site-packages")
    if groot_venv_site.exists() and str(groot_venv_site) not in sys.path:
        sys.path.append(str(groot_venv_site))
    import zmq


@dataclasses.dataclass(frozen=True)
class _FeatureSpec:
    shape: tuple[int, ...]


class _MsgSerializer:
    @staticmethod
    def to_bytes(data: dict[str, Any]) -> bytes:
        return msgpack.packb(data, default=_MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> dict[str, Any]:
        return msgpack.unpackb(data, object_hook=_MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj: dict[str, Any]) -> Any:
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class _BaseInferenceClient:
    def __init__(self, *, host: str, port: int, timeout_ms: int = 120000) -> None:
        self._host = str(host)
        self._port = int(port)
        self._timeout_ms = int(timeout_ms)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def call_endpoint(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        *,
        requires_input: bool = True,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data if data is not None else {}
        self._socket.send(_MsgSerializer.to_bytes(request))
        response = _MsgSerializer.from_bytes(self._socket.recv())
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response


class GROOTRemoteBasePolicyAdapter(nn.Module):
    def __init__(
        self,
        *,
        groot_root: str,
        host: str,
        port: int,
        state_dim: int,
        base_image_key: str,
        wrist_image_key: str,
        image_size: int = 84,
    ) -> None:
        super().__init__()
        self._groot_root = Path(groot_root).expanduser().resolve()
        self._client = _BaseInferenceClient(host=host, port=int(port))
        self._host = str(host)
        self._port = int(port)
        self._metadata = self._client.call_endpoint("get_metadata", requires_input=False)

        self._base_image_key = str(base_image_key)
        self._wrist_image_key = str(wrist_image_key)
        self._image_size = int(image_size)
        self._action_horizon = int(self._metadata["action_horizon"])
        self._token_count = int(self._metadata["token_target_count"])
        self._token_dim = int(self._metadata["token_dim"])
        self._action_queues: list[deque[torch.Tensor]] = []

        self.config = SimpleNamespace(
            n_action_steps=self._action_horizon,
            image_features={
                self._base_image_key: _FeatureSpec(shape=(3, self._image_size, self._image_size)),
                self._wrist_image_key: _FeatureSpec(shape=(3, self._image_size, self._image_size)),
            },
            robot_state_feature=_FeatureSpec(shape=(state_dim,)),
            env_state_feature=None,
        )

    def clone_for_eval(self) -> "GROOTRemoteBasePolicyAdapter":
        return GROOTRemoteBasePolicyAdapter(
            groot_root=str(self._groot_root),
            host=self._host,
            port=self._port,
            state_dim=int(self.config.robot_state_feature.shape[0]),
            base_image_key=self._base_image_key,
            wrist_image_key=self._wrist_image_key,
            image_size=self._image_size,
        )

    @property
    def observation_feature_shape(self) -> tuple[int, int]:
        return (self._token_count, self._token_dim)

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def token_dim(self) -> int:
        return self._token_dim

    def forward(self, *args, **kwargs):
        raise NotImplementedError("GROOTRemoteBasePolicyAdapter is an inference-only wrapper.")

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
                chunk = self.select_action_chunk(
                    self._slice_batch(batch, env_idx),
                    n_action_steps=self.config.n_action_steps,
                )
                for action in chunk[0]:
                    queue.append(action.detach())
            outputs.append(queue.popleft())
        return torch.stack(outputs, dim=0)

    def select_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        n_action_steps: int | None = None,
    ) -> torch.Tensor:
        requested_steps = int(n_action_steps or self.config.n_action_steps)
        payload = self._build_payload(batch)
        response = self._client.call_endpoint("get_action", payload)
        actions = np.asarray(response["actions"], dtype=np.float32)[:, :requested_steps]
        return torch.from_numpy(actions).to(device=self._obs_device(batch))

    @torch.no_grad()
    def infer_action_chunk_and_observation_features(
        self,
        batch: dict[str, torch.Tensor],
        n_action_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        requested_steps = int(n_action_steps or self.config.n_action_steps)
        payload = self._build_payload(batch)
        response = self._client.call_endpoint("get_action_and_features", payload)
        actions = np.asarray(response["actions"], dtype=np.float32)[:, :requested_steps]
        tokens = np.asarray(response["observation_features"], dtype=np.float32)
        device = self._obs_device(batch)
        return torch.from_numpy(actions).to(device=device), torch.from_numpy(tokens).to(device=device)

    @torch.no_grad()
    def encode_observation_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        payload = self._build_payload(batch)
        response = self._client.call_endpoint("encode_observation_features", payload)
        tokens = np.asarray(response["observation_features"], dtype=np.float32)
        return torch.from_numpy(tokens).to(device=self._obs_device(batch))

    def _build_payload(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {
            self._base_image_key: self._to_bhwc_uint8(batch[self._base_image_key]),
            self._wrist_image_key: self._to_bhwc_uint8(batch[self._wrist_image_key]),
            "observation.state": batch["observation.state"].detach().cpu().float().numpy(),
        }

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
