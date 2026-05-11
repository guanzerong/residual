# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

import wandb
from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


def run_dexmg_evaluation(
    *,
    env: VectorizedEnvWrapper,
    agent: QAgent,
    num_episodes: int = 20,
    device: torch.device | str = "cpu",
    global_step: int | None = None,
    save_video: bool = False,
    save_q_plots: bool = False,
    run_name: str | None = None,
    output_dir: str | Path | None = "outputs",
) -> tuple[dict[str, float], float]:
    """Extended evaluation to match the richer functionality available in
    the *residual_td3_dexmg* evaluator.  In particular, this version:

    1. Annotates every rendered frame with useful metadata (env index,
       episode counter, step counter, predicted Q-value and SUCCESS/FAIL).
    2. Caches frames per-episode and flushes them into a single video file
       at the end of the evaluation.
    3. Keeps the original simple success-rate / return metrics so existing
       training code continues to work unchanged.
    """

    def _safe_float(value) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 0.0
            return float(value.detach().cpu().item())
        return float(value)

    def _collect_stage_metrics(eval_env: VectorizedEnvWrapper, expected_envs: int) -> list[dict[str, float]]:
        try:
            raw_metrics = eval_env.vec_env.call("get_stage_metrics")
        except Exception:
            return [{} for _ in range(expected_envs)]

        metrics_list: list[dict[str, float]] = []
        for item in raw_metrics:
            if isinstance(item, dict):
                metrics_list.append({str(k): _safe_float(v) for k, v in item.items()})
            else:
                metrics_list.append({})

        if len(metrics_list) < expected_envs:
            metrics_list.extend({} for _ in range(expected_envs - len(metrics_list)))
        return metrics_list

    # ------------------------------------------------------------------
    # Helper functions (local to avoid polluting module namespace)
    # ------------------------------------------------------------------
    def _annotate_frame(
        frame: np.ndarray,
        *,
        env_idx: int,
        episode_num: int,
        total_episodes: int,
        step_idx: int,
        is_success: bool,
        q_value: float,
        font=None,
    ) -> np.ndarray:
        """Overlay evaluation metadata onto *frame* (H, W, C)."""

        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        # Status label ---------------------------------------------------
        status_text = "SUCCESS" if is_success else "FAIL"
        status_color = (0, 255, 0) if is_success else (255, 0, 0)

        y = 10
        dy = 15
        draw.text((10, y), f"Env {env_idx + 1}", fill=(255, 255, 255), font=font)
        y += dy
        draw.text((10, y), f"Episode {episode_num}/{total_episodes}", fill=(255, 255, 255), font=font)
        y += dy
        draw.text((10, y), f"Step {step_idx}", fill=(255, 255, 255), font=font)
        y += dy
        draw.text((10, y), status_text, fill=status_color, font=font)
        y += dy
        draw.text((10, y), f"Q = {q_value:.2f}", fill=(255, 255, 255), font=font)

        return np.asarray(pil_img)

    def _create_q_trajectory_plots(
        trajectories: list[list[float]],
        episode_lengths: list[int],
        successes: list[bool],
        output_path: Path,
        global_step: int | None = None,
    ) -> None:
        """Create Q-value trajectory plots for all episodes."""
        if not trajectories:
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: All Q-trajectories over time
        # Separate successful and failed episodes
        successful_trajs = [traj for i, traj in enumerate(trajectories) if successes[i]]
        failed_trajs = [traj for i, traj in enumerate(trajectories) if not successes[i]]

        # Plot all trajectories with different colors for success/failure
        for i, traj in enumerate(successful_trajs):
            steps = list(range(len(traj)))
            ax1.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label="Success" if i == 0 else "")

        for i, traj in enumerate(failed_trajs):
            steps = list(range(len(traj)))
            ax1.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label="Failure" if i == 0 else "")

        ax1.set_xlabel("Episode Step")
        ax1.set_ylabel("Q-Value")
        ax1.set_title(f"Q-Value Trajectories Over Time (Step {global_step or 'N/A'})")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Q-value distribution at different episode progress points
        progress_points = [0.25, 0.5, 0.75, 1.0]  # 25%, 50%, 75%, 100% of episode
        q_values_at_progress = {f"{int(p * 100)}%": [] for p in progress_points}

        for traj in trajectories:
            traj_len = len(traj)
            for p in progress_points:
                step_idx = min(int(p * traj_len), traj_len - 1)
                if step_idx < len(traj):
                    q_values_at_progress[f"{int(p * 100)}%"].append(traj[step_idx])

        # Create box plot
        box_data = [q_values_at_progress[f"{int(p * 100)}%"] for p in progress_points]
        box_labels = [f"{int(p * 100)}%" for p in progress_points]

        ax2.boxplot(box_data, labels=box_labels)
        ax2.set_xlabel("Episode Progress")
        ax2.set_ylabel("Q-Value")
        ax2.set_title("Q-Value Distribution at Different Episode Progress Points")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved Q-trajectory plots to: {output_path}")

    # ------------------------------------------------------------------
    # Initial setup -----------------------------------------------------
    # ------------------------------------------------------------------
    device = torch.device(device)
    agent.eval()

    num_envs: int = env.num_envs if hasattr(env, "num_envs") else 1

    # Per-environment episode buffers ----------------------------------
    ep_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    ep_q_preds: list[list[float]] = [[] for _ in range(num_envs)]
    ep_stage_maxima: list[dict[str, float]] = [{} for _ in range(num_envs)]
    ep_trace_rows: list[list[dict[str, float | int | bool]]] = [[] for _ in range(num_envs)]

    successes: list[bool] = []  # episode-level success flags
    returns: list[float] = []  # episode-level undiscounted returns

    # Q-trajectory data for plotting ------------------------------------
    all_q_trajectories: list[list[float]] = []  # Store Q-trajectories for all episodes
    all_episode_lengths: list[int] = []  # Store episode lengths for plotting
    trace_rows: list[dict[str, float | int | bool]] = []
    episode_summary_rows: list[dict[str, float | int | bool]] = []

    # Video buffers -----------------------------------------------------
    frame_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None
    all_frames: list[np.ndarray] | None = [] if save_video else None

    done_episodes = 0
    obs, _ = env.reset()

    # Initialize progress display with dots
    progress_dots = ["."] * num_episodes
    print(f"Evaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

    while done_episodes < num_episodes:
        # --------------------------------------------------------------
        # 0. Stage metrics at the current decision state ---------------
        # --------------------------------------------------------------
        stage_metrics_per_env = _collect_stage_metrics(env, num_envs)

        # --------------------------------------------------------------
        # 1. Policy inference + Q-value prediction ---------------------
        # --------------------------------------------------------------
        q_by_horizon = None
        horizon_scores = None
        horizon_score_aux_cpu = {}
        with torch.no_grad():
            # Build features on-the-fly to obtain Q-predictions --------
            obs_q = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            obs_q["feat"] = agent._encode(obs_q, augment=False)
            policy_output = agent._forward_actor_policy(obs_q, stddev=0.0, use_target=False)

            horizon_onehot = None
            horizon_probs = None
            horizon_actions = None
            if agent.uses_adaptive_horizons:
                if policy_output.horizon_logits is None:
                    raise RuntimeError("Adaptive horizon policy is enabled but evaluation received no horizon logits.")
                horizon_onehot, horizon_probs = agent._sample_horizon_onehot(policy_output.horizon_logits, eval_mode=True)
                horizon_actions = agent._sample_horizon_actions_from_policy_output(
                    policy_output,
                    eval_mode=True,
                    clip=None,
                )
                if horizon_actions is not None:
                    residual_action = agent._select_horizon_candidate(horizon_actions, horizon_onehot)
                else:
                    residual_action = agent._sample_action_from_policy_output(policy_output, eval_mode=True, clip=None)
            else:
                residual_action = agent._sample_action_from_policy_output(policy_output, eval_mode=True, clip=None)

            actions = agent._build_env_action(residual_action, horizon_onehot)
            if agent.uses_adaptive_horizons:
                assert horizon_onehot is not None
                if horizon_actions is not None:
                    candidate_q_actions = agent._build_critic_actions_from_horizon_actions(obs_q, horizon_actions)
                    score_actions = horizon_actions
                else:
                    candidate_q_actions = agent._build_critic_actions(obs_q, residual_action)
                    score_actions = residual_action
                q_by_horizon_tensor = agent._evaluate_critic_actions(
                    agent.critic,
                    obs_q["feat"],
                    obs_q["observation.state"],
                    candidate_q_actions,
                    for_policy=False,
                )
                horizon_scores_tensor, horizon_score_aux = agent._score_adaptive_horizons(q_by_horizon_tensor, score_actions)
                horizon_index = torch.argmax(horizon_onehot, dim=-1)
                batch_index = torch.arange(q_by_horizon_tensor.shape[0], device=q_by_horizon_tensor.device)
                q_pred = q_by_horizon_tensor[batch_index, horizon_index].detach().cpu()
                q_by_horizon = q_by_horizon_tensor.detach().cpu()
                horizon_scores = horizon_scores_tensor.detach().cpu()
                horizon_score_aux_cpu = {
                    key: value.detach().cpu()
                    for key, value in horizon_score_aux.items()
                    if isinstance(value, torch.Tensor)
                }
            else:
                q_actions = agent._build_critic_actions(obs_q, residual_action, horizon_onehot=horizon_onehot)
                q_pred = agent.critic.q_value(obs_q["feat"], obs_q["observation.state"], q_actions).detach().cpu().squeeze(-1)

        # --------------------------------------------------------------
        # 2. Environment step ------------------------------------------
        # --------------------------------------------------------------
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done_flags = terminated | truncated

        # Capture frames ------------------------------------------------
        if save_video and frame_buffer is not None:
            frame = env.render()
            for env_idx in range(num_envs):
                frame_buffer[env_idx].append(frame[env_idx])

        # --------------------------------------------------------------
        # 3. Per-environment bookkeeping -------------------------------
        # --------------------------------------------------------------
        for env_idx in range(num_envs):
            stage_metrics = stage_metrics_per_env[env_idx]
            for metric_name, metric_value in stage_metrics.items():
                prev_best = ep_stage_maxima[env_idx].get(metric_name, float("-inf"))
                ep_stage_maxima[env_idx][metric_name] = max(prev_best, metric_value)

            chosen_horizon = 1
            executed_horizon = 1
            correction_l1 = float(torch.mean(torch.abs(residual_action[env_idx])).item())
            correction_rms = float(torch.sqrt(torch.mean(torch.square(residual_action[env_idx]))).item())
            correction_l2 = float(torch.linalg.vector_norm(residual_action[env_idx]).item())
            base_rms = float(torch.sqrt(torch.mean(torch.square(obs["observation.base_action"][env_idx]))).item())
            normalized_correction = correction_rms
            relative_correction = correction_rms / (base_rms + 1e-6)
            horizon_entropy = 0.0

            if agent.uses_adaptive_horizons:
                assert horizon_onehot is not None
                assert horizon_probs is not None
                horizon_idx = int(torch.argmax(horizon_onehot[env_idx]).item())
                chosen_horizon = int(agent.adaptive_horizons[horizon_idx])
                if "executed_horizon" in info:
                    executed_horizon = int(info["executed_horizon"][env_idx].item())
                else:
                    executed_horizon = chosen_horizon

                residual_chunk = residual_action[env_idx].reshape(agent.max_action_horizon, agent.primitive_action_dim)
                base_chunk = obs["observation.base_action"][env_idx].reshape(
                    agent.max_action_horizon, agent.primitive_action_dim
                )
                residual_prefix = residual_chunk[:chosen_horizon]
                base_prefix = base_chunk[:chosen_horizon]
                correction_l1 = float(torch.mean(torch.abs(residual_prefix)).item())
                correction_rms = float(torch.sqrt(torch.mean(torch.square(residual_prefix))).item())
                correction_l2 = float(torch.linalg.vector_norm(residual_prefix).item())
                base_rms = float(torch.sqrt(torch.mean(torch.square(base_prefix))).item())
                action_scale = float(getattr(agent.actor, "action_scale", 1.0))
                normalized_correction = correction_rms / max(action_scale, 1e-6)
                relative_correction = correction_rms / (base_rms + 1e-6)
                probs = horizon_probs[env_idx]
                horizon_entropy = float((-(probs * torch.log(probs.clamp_min(1e-8))).sum()).item())

            if "undiscounted_reward" in info:
                ep_rewards[env_idx].append(info["undiscounted_reward"][env_idx].item())
            else:
                ep_rewards[env_idx].append(reward[env_idx].item())
            ep_q_preds[env_idx].append(q_pred[env_idx].item())

            trace_row: dict[str, float | int | bool] = {
                "global_step": int(global_step or 0),
                "eval_episode_id": -1,
                "decision_step": len(ep_trace_rows[env_idx]),
                "episode_decision_len": -1,
                "normalized_step": 0.0,
                "normalized_bin": -1,
                "chosen_horizon": chosen_horizon,
                "executed_horizon": executed_horizon,
                "q_pred": float(q_pred[env_idx].item()),
                "correction_l1": correction_l1,
                "correction_rms": correction_rms,
                "correction_l2": correction_l2,
                "normalized_correction": normalized_correction,
                "base_rms": base_rms,
                "relative_correction": relative_correction,
                "horizon_entropy": horizon_entropy,
                "success": False,
            }
            for metric_name, metric_value in stage_metrics.items():
                trace_row[metric_name] = metric_value
            if agent.uses_adaptive_horizons and horizon_probs is not None:
                policy_argmax_idx = int(torch.argmax(horizon_probs[env_idx]).item())
                trace_row["policy_argmax_horizon"] = int(agent.adaptive_horizons[policy_argmax_idx])
                if q_by_horizon is not None:
                    q_values = q_by_horizon[env_idx]
                    q_argmax_idx = int(torch.argmax(q_values).item())
                    q_sorted = torch.sort(q_values, descending=True).values
                    trace_row["q_argmax_horizon"] = int(agent.adaptive_horizons[q_argmax_idx])
                    trace_row["q_best_margin"] = float((q_sorted[0] - q_sorted[1]).item()) if len(q_sorted) > 1 else 0.0
                    trace_row["q_selected_minus_best"] = float((q_values[horizon_idx] - q_sorted[0]).item())
                if horizon_scores is not None:
                    score_values = horizon_scores[env_idx]
                    score_argmax_idx = int(torch.argmax(score_values).item())
                    score_sorted = torch.sort(score_values, descending=True).values
                    trace_row["score_argmax_horizon"] = int(agent.adaptive_horizons[score_argmax_idx])
                    trace_row["score_best_margin"] = (
                        float((score_sorted[0] - score_sorted[1]).item()) if len(score_sorted) > 1 else 0.0
                    )
                    trace_row["score_selected_minus_best"] = float((score_values[horizon_idx] - score_sorted[0]).item())
                for horizon_idx, horizon in enumerate(agent.adaptive_horizons):
                    trace_row[f"horizon_prob_{horizon}"] = float(horizon_probs[env_idx, horizon_idx].item())
                    if q_by_horizon is not None:
                        trace_row[f"q_horizon_{horizon}"] = float(q_by_horizon[env_idx, horizon_idx].item())
                    if horizon_scores is not None:
                        trace_row[f"q_score_horizon_{horizon}"] = float(horizon_scores[env_idx, horizon_idx].item())
                    for aux_name, aux_values in horizon_score_aux_cpu.items():
                        trace_row[f"{aux_name}_horizon_{horizon}"] = float(aux_values[env_idx, horizon_idx].item())
            ep_trace_rows[env_idx].append(trace_row)

            if done_flags[env_idx]:
                # Episode finished -- aggregate results ----------------
                ep_return = float(sum(ep_rewards[env_idx]))
                if "macro_success" in info:
                    is_success = bool(info["macro_success"][env_idx].item())
                else:
                    is_success = bool(reward[env_idx].item() == 1.0)

                # Update progress display
                progress_dots[done_episodes] = "✓" if is_success else "✗"
                print(f"\rEvaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

                successes.append(is_success)
                returns.append(ep_return)

                episode_rows = ep_trace_rows[env_idx]
                episode_len = len(episode_rows)
                episode_id = done_episodes
                for step_idx, row in enumerate(episode_rows):
                    normalized_step = step_idx / max(episode_len - 1, 1)
                    row["eval_episode_id"] = episode_id
                    row["decision_step"] = step_idx
                    row["episode_decision_len"] = episode_len
                    row["normalized_step"] = normalized_step
                    row["normalized_bin"] = int(np.clip(np.floor(normalized_step * 100), 0, 99))
                    row["success"] = is_success
                trace_rows.extend(episode_rows)

                episode_summary: dict[str, float | int | bool] = {
                    "global_step": int(global_step or 0),
                    "eval_episode_id": episode_id,
                    "success": is_success,
                    "episode_return": ep_return,
                    "episode_decision_len": episode_len,
                }
                for metric_name, metric_value in ep_stage_maxima[env_idx].items():
                    episode_summary[f"max_{metric_name}"] = metric_value
                episode_summary_rows.append(episode_summary)

                # Store Q-trajectory data for plotting ------------------
                if save_q_plots:
                    all_q_trajectories.append(ep_q_preds[env_idx].copy())

                # Always track episode length for successful episodes logging
                all_episode_lengths.append(len(ep_q_preds[env_idx]))

                # Annotate and flush frames ---------------------------
                if save_video and frame_buffer is not None and all_frames is not None:
                    episode_frames = frame_buffer[env_idx]
                    episode_qs = ep_q_preds[env_idx]

                    episode_global_idx = done_episodes + 1  # 1-based

                    for step_idx, fr in enumerate(episode_frames):
                        annotated_fr = _annotate_frame(
                            fr,
                            env_idx=env_idx,
                            episode_num=episode_global_idx,
                            total_episodes=num_episodes,
                            step_idx=step_idx + 1,
                            is_success=is_success,
                            q_value=episode_qs[step_idx],
                        )
                        all_frames.append(annotated_fr)

                    # Clear per-episode frame buffer
                    frame_buffer[env_idx].clear()

                # Reset per-env caches --------------------------------
                ep_rewards[env_idx].clear()
                ep_q_preds[env_idx].clear()
                ep_stage_maxima[env_idx] = {}
                ep_trace_rows[env_idx] = []

                done_episodes += 1

                if done_episodes == num_episodes:
                    break

        # Prepare for next loop ----------------------------------------
        obs = next_obs

    print("Done")

    # ------------------------------------------------------------------
    # 4. Aggregate metrics ---------------------------------------------
    # ------------------------------------------------------------------
    # Sanity check: episode lengths must align 1:1 with successes
    if len(all_episode_lengths) != len(successes):
        raise RuntimeError(
            f"Episode length/success misalignment: lengths={len(all_episode_lengths)} successes={len(successes)}"
        )

    success_rate: float = float(np.mean(successes)) if successes else 0.0
    mean_return: float = float(np.mean(returns)) if returns else 0.0

    # Calculate mean episode length among successful episodes
    successful_episode_lengths = [length for length, is_success in zip(all_episode_lengths, successes) if is_success]
    mean_successful_episode_length: float = (
        float(np.mean(successful_episode_lengths)) if successful_episode_lengths else 0.0
    )

    metrics: dict[str, float] = {
        "eval/success_rate": success_rate,
        "eval/mean_return": mean_return,
        "eval/mean_successful_episode_length": mean_successful_episode_length,
        "eval/trace_rows": float(len(trace_rows)),
    }

    if episode_summary_rows:
        align_keys = [key for key in episode_summary_rows[0].keys() if key.startswith("max_stage/align")]
        if align_keys:
            align_key = align_keys[0]
            metrics["eval/mean_max_stage_align"] = float(np.mean([row[align_key] for row in episode_summary_rows]))
        lift_keys = [key for key in episode_summary_rows[0].keys() if key.startswith("max_stage/lift")]
        if lift_keys:
            lift_key = lift_keys[0]
            metrics["eval/mean_max_stage_lift"] = float(np.mean([row[lift_key] for row in episode_summary_rows]))

    if wandb.run is not None:
        wandb.log(metrics, step=global_step)
        wandb.summary["paper/latest_task_success_rate"] = success_rate

    # ------------------------------------------------------------------
    # 5. Q-trajectory plots --------------------------------------------
    # ------------------------------------------------------------------
    parent = Path(str(output_dir or "outputs"))
    if run_name is not None:
        parent = parent / run_name.split("__")[0]
    parent.mkdir(parents=True, exist_ok=True)

    if trace_rows:
        trace_path = parent / f"eval_horizon_trace_step_{global_step if global_step is not None else 'NA'}.csv"
        trace_df = pd.DataFrame(trace_rows)
        trace_df.to_csv(trace_path, index=False)
        print(f"Saved adaptive horizon trace to: {trace_path}")
        if wandb.run is not None:
            wandb.save(str(trace_path), base_path=str(parent))

    if episode_summary_rows:
        episode_path = parent / f"eval_episode_summary_step_{global_step if global_step is not None else 'NA'}.csv"
        episode_df = pd.DataFrame(episode_summary_rows)
        episode_df.to_csv(episode_path, index=False)
        print(f"Saved episode summary trace to: {episode_path}")
        if wandb.run is not None:
            wandb.save(str(episode_path), base_path=str(parent))

    if save_q_plots and all_q_trajectories and run_name is not None:
        plot_name = f"eval_q_trajectories_{run_name}_step_{global_step if global_step is not None else 'NA'}.png"
        plot_path = parent / plot_name

        _create_q_trajectory_plots(
            trajectories=all_q_trajectories,
            episode_lengths=all_episode_lengths,
            successes=successes,
            output_path=plot_path,
            global_step=global_step,
        )

        # Log to W&B if available
        if wandb.run is not None:
            wandb.log({"value/q_trajectories": wandb.Image(str(plot_path))}, step=global_step)

    # ------------------------------------------------------------------
    # 6. Video dump + W&B logging --------------------------------------
    # ------------------------------------------------------------------
    if save_video and all_frames is not None and run_name is not None:
        vid_name = f"eval_{run_name}_step_{global_step if global_step is not None else 'NA'}.mp4"
        video_path = parent / vid_name

        fps_val = getattr(env, "fps", 20)

        writer = imageio.get_writer(video_path, fps=fps_val)
        for fr in all_frames:
            writer.append_data(fr)
        writer.close()

        if wandb.run is not None:
            wandb.log({"eval/video": wandb.Video(str(video_path), format="mp4")}, step=global_step)

    # Restore training mode --------------------------------------------
    agent.train(True)

    return metrics
