from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT.parent
_OPENPI_LIBERO_IMPORT_ROOT = _WORKSPACE_ROOT / "openpi" / "third_party" / "libero"
_OPENPI_ONLYRGBD_LIBERO_IMPORT_ROOT = _WORKSPACE_ROOT / "openpi_onlyrgbd" / "third_party" / "libero"
_LIBERO_CONFIG_DIR = _REPO_ROOT / ".cache" / "libero_openpi"
_LIBERO_IMPORT_ERROR: Exception | None = None


def _get_libero_benchmark_root() -> Path | None:
    candidate_roots = [
        _OPENPI_LIBERO_IMPORT_ROOT / "libero" / "libero",
        _OPENPI_ONLYRGBD_LIBERO_IMPORT_ROOT / "libero" / "libero",
    ]
    for path in candidate_roots:
        if path.exists():
            return path
    return None


def _configure_libero_config() -> None:
    benchmark_root = _get_libero_benchmark_root()
    if benchmark_root is None:
        return

    dataset_root = _WORKSPACE_ROOT / "datasets" / "libero"
    if not dataset_root.exists():
        dataset_root = benchmark_root.parent / "datasets"

    config_text = "\n".join(
        [
            f"benchmark_root: {benchmark_root}",
            f"bddl_files: {benchmark_root / 'bddl_files'}",
            f"init_states: {benchmark_root / 'init_files'}",
            f"datasets: {dataset_root}",
            f"assets: {benchmark_root / 'assets'}",
            "",
        ]
    )
    _LIBERO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_file = _LIBERO_CONFIG_DIR / "config.yaml"
    if not config_file.exists() or config_file.read_text() != config_text:
        config_file.write_text(config_text)
    os.environ["LIBERO_CONFIG_PATH"] = str(_LIBERO_CONFIG_DIR)


def _configure_libero_import_paths() -> None:
    candidate_paths = [
        _REPO_ROOT / "deps" / "robosuite",
        _REPO_ROOT / "third_party",
        _WORKSPACE_ROOT / "robosuite",
        _OPENPI_LIBERO_IMPORT_ROOT,
        _OPENPI_ONLYRGBD_LIBERO_IMPORT_ROOT,
    ]
    # Insert in reverse so earlier entries in candidate_paths end up with higher import priority.
    for path in reversed(candidate_paths):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_configure_libero_config()
_configure_libero_import_paths()
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except Exception as exc:  # noqa: BLE001
    benchmark = None
    get_libero_path = None
    OffScreenRenderEnv = None
    _LIBERO_IMPORT_ERROR = exc


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def ensure_libero_available() -> None:
    if _LIBERO_IMPORT_ERROR is None:
        return
    raise ImportError(
        "Failed to import LIBERO/robosuite. "
        "Tried the local vendored paths under residual-offpolicy-rl and the sibling openpi repos. "
        "You likely need a compatible robosuite + LIBERO installation in the same Python environment. "
        f"Original import error: {_LIBERO_IMPORT_ERROR!r}"
    ) from _LIBERO_IMPORT_ERROR


def get_libero_task_suite(task_suite_name: str):
    ensure_libero_available()
    assert benchmark is not None
    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown LIBERO task suite {task_suite_name!r}. "
            f"Available suites: {sorted(benchmark_dict.keys())}"
        )
    return benchmark_dict[task_suite_name]()


def get_libero_max_steps(task_suite_name: str) -> int:
    if task_suite_name not in LIBERO_MAX_STEPS:
        raise ValueError(
            f"Unknown LIBERO task suite {task_suite_name!r}. "
            f"Expected one of: {sorted(LIBERO_MAX_STEPS)}"
        )
    return LIBERO_MAX_STEPS[task_suite_name]


def create_libero_env(task: Any, *, resolution: int = LIBERO_ENV_RESOLUTION, seed: int = 0):
    ensure_libero_available()
    assert get_libero_path is not None
    assert OffScreenRenderEnv is not None
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
        seed=seed,
    )
    return env


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(1.0 - float(quat[3] * quat[3]), 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32, copy=False)


def _resize_with_pad(image: np.ndarray, target_size: int) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}.")
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    src_w, src_h = pil_image.size
    scale = min(float(target_size) / float(src_w), float(target_size) / float(src_h))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = pil_image.resize((resized_w, resized_h), resampling)
    canvas = Image.new("RGB", (target_size, target_size))
    offset_x = (target_size - resized_w) // 2
    offset_y = (target_size - resized_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return np.asarray(canvas, dtype=np.uint8)


class LiberoEvalVecEnvWrapper:
    """Single-environment LIBERO wrapper with vector-env-like batched torch outputs."""

    def __init__(
        self,
        *,
        task: Any,
        initial_states: np.ndarray,
        task_description: str,
        resize_size: int,
        max_steps: int,
        num_steps_wait: int = 10,
        seed: int = 0,
        device: str | torch.device = "cpu",
        env_resolution: int = LIBERO_ENV_RESOLUTION,
        render_size: tuple[int, int] | int | None = None,
        video_key: str = "observation.images.agentview",
    ) -> None:
        ensure_libero_available()

        if resize_size <= 0:
            raise ValueError(f"resize_size must be positive, got {resize_size}.")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")
        if num_steps_wait < 0:
            raise ValueError(f"num_steps_wait must be non-negative, got {num_steps_wait}.")

        self.env = create_libero_env(task, resolution=env_resolution, seed=seed)
        self.initial_states = np.asarray(initial_states)
        self.task_description = str(task_description)
        self.resize_size = int(resize_size)
        self.max_steps = int(max_steps)
        self.num_steps_wait = int(num_steps_wait)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.num_envs = 1
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 20, "horizon": self.max_steps}
        self.spec = None
        self.render_mode = "rgb_array"
        self.video_key = video_key
        self._render_camera_name = self._camera_name_from_video_key(video_key)
        self._episode_cursor = 0
        self._episode_steps = 0
        self._last_raw_obs: dict[str, np.ndarray] | None = None

        if render_size is None:
            self.render_size = (resize_size, resize_size)
        elif isinstance(render_size, int):
            self.render_size = (render_size, render_size)
        else:
            self.render_size = render_size

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 7), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation.images.agentview": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, 3, self.resize_size, self.resize_size),
                    dtype=np.float32,
                ),
                "observation.images.robot0_eye_in_hand": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, 3, self.resize_size, self.resize_size),
                    dtype=np.float32,
                ),
                "observation.state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1, 8),
                    dtype=np.float32,
                ),
            }
        )

    @property
    def fps(self) -> int:
        return int(self.metadata["render_fps"])

    def set_video_key(self, video_key: str) -> None:
        self.video_key = video_key
        self._render_camera_name = self._camera_name_from_video_key(video_key)

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        episode_index = kwargs.get("episode_index", self._episode_cursor)
        self._episode_cursor = int(episode_index) + 1
        self._episode_steps = 0

        init_state = self.initial_states[int(episode_index) % len(self.initial_states)]

        self.env.reset()
        raw_obs = self.env.set_init_state(init_state)
        for _ in range(self.num_steps_wait):
            raw_obs, _, done, _ = self.env.step(LIBERO_DUMMY_ACTION)
            if done:
                raise RuntimeError("LIBERO episode terminated during stabilization wait steps.")

        self._last_raw_obs = raw_obs
        processed_obs = self._process_obs(raw_obs, add_batch_dim=True)
        info = {
            "episode_index": int(episode_index),
            "initial_state_index": int(episode_index) % len(self.initial_states),
            "task_description": self.task_description,
        }
        return self._to_torch_obs(processed_obs), info

    def step(
        self, action: torch.Tensor | np.ndarray | list[float]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        env_action = self._to_numpy_action(action)
        raw_obs, reward, done, info = self.env.step(env_action.tolist())
        self._episode_steps += 1
        self._last_raw_obs = raw_obs

        terminated = bool(done)
        truncated = bool((not terminated) and self._episode_steps >= self.max_steps)
        processed_obs = self._process_obs(raw_obs, add_batch_dim=True)

        step_info: dict[str, Any] = {
            **dict(info),
            "success": bool(terminated),
            "episode_steps": self._episode_steps,
            "task_description": self.task_description,
        }

        if terminated or truncated:
            final_obs = self._process_obs(raw_obs, add_batch_dim=False)
            reset_obs, reset_info = self.reset()
            step_info.update(reset_info)
            step_info["final_obs"] = [final_obs]
            return (
                reset_obs,
                torch.tensor([float(reward)], device=self.device, dtype=torch.float32),
                torch.tensor([terminated], device=self.device, dtype=torch.bool),
                torch.tensor([truncated], device=self.device, dtype=torch.bool),
                step_info,
            )

        return (
            self._to_torch_obs(processed_obs),
            torch.tensor([float(reward)], device=self.device, dtype=torch.float32),
            torch.tensor([False], device=self.device, dtype=torch.bool),
            torch.tensor([False], device=self.device, dtype=torch.bool),
            step_info,
        )

    def render(self) -> np.ndarray:
        frame = self.env.sim.render(
            camera_name=self._render_camera_name,
            height=self.render_size[0],
            width=self.render_size[1],
        )[::-1]
        return np.expand_dims(frame, axis=0)

    def close(self) -> None:
        self.env.close()

    def get_wrapper_attr(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def set_wrapper_attr(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def _to_numpy_action(self, action: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
        if isinstance(action, torch.Tensor):
            action_np = action.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim == 2:
            if action_np.shape[0] != 1:
                raise ValueError(f"Expected a single batched action, got shape {action_np.shape}.")
            action_np = action_np[0]
        if action_np.shape != (7,):
            raise ValueError(f"Expected primitive action shape (7,), got {action_np.shape}.")
        return action_np

    def _to_torch_obs(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {key: torch.from_numpy(value).to(self.device) for key, value in obs.items()}

    def _process_obs(self, raw_obs: dict[str, Any], *, add_batch_dim: bool) -> dict[str, np.ndarray]:
        agentview = self._preprocess_image(raw_obs["agentview_image"])
        wrist = self._preprocess_image(raw_obs["robot0_eye_in_hand_image"])
        state = np.concatenate(
            (
                np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
                _quat_to_axis_angle(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32)),
                np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
            ),
            axis=0,
        ).astype(np.float32, copy=False)

        if state.shape != (8,):
            raise ValueError(f"Expected LIBERO state shape (8,), got {state.shape}.")

        processed = {
            "observation.images.agentview": agentview,
            "observation.images.robot0_eye_in_hand": wrist,
            "observation.state": state,
        }
        if add_batch_dim:
            return {key: np.expand_dims(value, axis=0) for key, value in processed.items()}
        return processed

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        image = np.ascontiguousarray(image[::-1, ::-1])
        image = _resize_with_pad(image, self.resize_size)
        image = image.astype(np.float32) / 255.0
        return np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)

    @staticmethod
    def _camera_name_from_video_key(video_key: str) -> str:
        if "." in video_key:
            return video_key.split(".")[-1]
        return video_key
