# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import wandb

def download_policy_from_wandb(
    run_id: str,
    *,
    step: str | None = None,
    artifact_version: str = "latest",
) -> tuple[Path, str]:
    """Download a policy checkpoint logged on W&B and return its folder.

    The policy is expected to have been created with the training utilities in
    `train_hf.py` and therefore to contain a `config.json` in the root of the
    downloaded artifact.
    """
    api = wandb.Api()
    project, id_ = run_id.split("/")

    if step is None or str(step).lower() == "latest":
        artifact_name = f"run_{id_}_latest:{artifact_version}"
        checkpoint_step = "latest"
    elif str(step).lower() == "best":
        artifact_name = f"run_{id_}_best:{artifact_version}"
        checkpoint_step = "best"
    else:
        artifact_name = f"run_{id_}_model_step_{step}:{artifact_version}"
        checkpoint_step = str(step)

    artifact_path = f"{project}/{artifact_name}"
    artifact = api.artifact(artifact_path)

    art_dir = Path(artifact.download())
    policy_dir = art_dir / "policy"  # The artifact root already contains the policy files.

    if not (policy_dir / "config.json").exists():
        raise FileNotFoundError(f"Policy directory not found inside downloaded artifact: {policy_dir}")

    return policy_dir, checkpoint_step


def load_policy(policy_dir: Path, **policy_kwargs: Any) -> Any:
    """Infer policy type from `config.json` and load weights."""

    with (policy_dir / "config.json").open() as f:
        cfg_dict = json.load(f)

    policy_name_field = str(cfg_dict.get("type", "")).lower()

    if "diffusion" in policy_name_field:
        from resfit.lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy.from_pretrained(
            policy_dir,
            dataset_stats=policy_kwargs.get("dataset_stats"),
        )
    if "pi05" in policy_name_field:
        from resfit.lerobot.policies.pi05.adapter import PI05PolicyAdapter

        return PI05PolicyAdapter.from_pretrained(policy_dir, **policy_kwargs)
    if "use_vae" in cfg_dict:
        from resfit.lerobot.policies.act.modeling_act import ACTPolicy

        return ACTPolicy.from_pretrained(
            policy_dir,
            dataset_stats=policy_kwargs.get("dataset_stats"),
        )
    if "act" in policy_name_field:
        from resfit.lerobot.policies.act.modeling_act import ACTPolicy

        return ACTPolicy.from_pretrained(
            policy_dir,
            dataset_stats=policy_kwargs.get("dataset_stats"),
        )

    raise ValueError(f"Unknown policy type: {policy_name_field}")


def save_checkpoint(ckpt_dir: Path, step: int, policy, optimizer, scheduler=None) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save model weights + config
    policy.save_pretrained(ckpt_dir / "policy")
    # Save optimizer & misc state
    trainer_state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        trainer_state["scheduler"] = scheduler.state_dict()
    torch.save(
        trainer_state,
        ckpt_dir / "trainer_state.pt",
    )


def load_checkpoint(ckpt_dir: Path, policy, optimizer, scheduler=None, **policy_kwargs: Any):
    state_pth = ckpt_dir / "trainer_state.pt"
    if not state_pth.exists():
        raise FileNotFoundError(state_pth)
    state = torch.load(state_pth, map_location="cpu")
    policy_loaded = policy.from_pretrained(ckpt_dir / "policy", **policy_kwargs)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return state["step"], policy_loaded, optimizer, scheduler
