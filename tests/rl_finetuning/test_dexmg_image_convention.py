from __future__ import annotations

import numpy as np

from resfit.dexmg.environments import dexmg


class _FakeRobosuiteEnv:
    action_dim = 7

    def reset(self):
        return {
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            "agentview_image": np.zeros((84, 84, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((84, 84, 3), dtype=np.uint8),
        }

    def close(self):
        return None


def test_robosuite_wrapper_restores_global_image_convention(monkeypatch):
    captured: dict[str, str] = {}
    original_convention = dexmg.macros.IMAGE_CONVENTION
    dexmg.macros.IMAGE_CONVENTION = "opengl"

    monkeypatch.setattr(dexmg, "load_composite_controller_config", lambda robot: {})

    def _fake_make(**kwargs):
        captured["during_make"] = dexmg.macros.IMAGE_CONVENTION
        return _FakeRobosuiteEnv()

    monkeypatch.setattr(dexmg.robosuite, "make", _fake_make)

    env = dexmg.RobosuiteGymWrapper(env_name="Lift", camera_size=84)
    try:
        assert captured["during_make"] == "opencv"
        assert dexmg.macros.IMAGE_CONVENTION == "opengl"
    finally:
        env.close()
        dexmg.macros.IMAGE_CONVENTION = original_convention
