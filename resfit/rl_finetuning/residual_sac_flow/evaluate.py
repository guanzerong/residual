from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
import torch

import wandb
from resfit.rl_finetuning.residual_sac_flow.agent import ResidualSACFlowAgent


def run_residual_sac_flow_evaluation(
    *,
    env,
    agent: ResidualSACFlowAgent,
    num_episodes: int = 20,
    device: torch.device | str = "cpu",
    global_step: int | None = None,
    save_video: bool = False,
    run_name: str | None = None,
    output_dir: str | Path | None = "outputs",
) -> dict[str, float]:
    device = torch.device(device)
    agent.eval()

    num_envs = env.num_envs if hasattr(env, "num_envs") else 1
    returns = [[] for _ in range(num_envs)]
    episode_returns: list[float] = []
    lengths = [0 for _ in range(num_envs)]
    successes: list[bool] = []
    successful_lengths: list[int] = []
    frames = [] if save_video else None

    done_episodes = 0
    obs, _ = env.reset()

    while done_episodes < num_episodes:
        with torch.no_grad():
            residual_action = agent.act(obs, eval_mode=True, cpu=False)

        next_obs, reward, terminated, truncated, _ = env.step(residual_action)
        done_flags = terminated | truncated

        if save_video and frames is not None:
            rendered = env.render()
            frames.extend(list(rendered))

        for env_idx in range(num_envs):
            returns[env_idx].append(float(reward[env_idx].item()))
            lengths[env_idx] += 1

            if done_flags[env_idx]:
                episode_return = float(sum(returns[env_idx]))
                is_success = bool(reward[env_idx].item() == 1.0)
                episode_returns.append(episode_return)
                successes.append(is_success)
                if is_success:
                    successful_lengths.append(lengths[env_idx])
                returns[env_idx].clear()
                lengths[env_idx] = 0
                done_episodes += 1
                if done_episodes >= num_episodes:
                    break

        obs = next_obs

    success_rate = float(np.mean(successes)) if successes else 0.0
    mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    mean_successful_episode_length = float(np.mean(successful_lengths)) if successful_lengths else 0.0

    metrics = {
        "eval/success_rate": success_rate,
        "eval/mean_return": mean_return,
        "eval/mean_successful_episode_length": mean_successful_episode_length,
    }

    if wandb.run is not None:
        wandb.log(metrics, step=global_step)

    if save_video and frames and run_name is not None:
        parent = Path(str(output_dir or "outputs")) / run_name.split("__")[0]
        parent.mkdir(parents=True, exist_ok=True)
        video_path = parent / f"eval_{run_name}_step_{global_step if global_step is not None else 'NA'}.mp4"
        with imageio.get_writer(video_path, fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)
        if wandb.run is not None:
            wandb.log({"eval/video": wandb.Video(str(video_path), format="mp4")}, step=global_step)

    return metrics
