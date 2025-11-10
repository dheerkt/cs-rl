# src/train.py
"""
Training script for PPO with Centralized Critic on Overcooked layouts
"""

import argparse
import os
import random
import numpy as np
import torch

from .env_builder import build_overcooked_env
from .models import ActorNetwork, CentralizedCritic
from .ppo import PPO
from .reward_shaping import RewardShaper
from .utils import Logger, CollaborationMetrics, save_checkpoint, print_training_stats
from configs.hyperparameters import HyperParams


def train(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}\n")

    # Build environment
    print(f"Building Overcooked environment: {args.layout}")
    env = build_overcooked_env(args.layout, horizon=400, seed=args.seed)
    print(f"Observation shape: {HyperParams.obs_dim}")
    print(f"Action space: {HyperParams.action_dim}")
    print(f"Random seed: {args.seed}\n")

    # Create networks
    print("Initializing networks...")
    actors = [
        ActorNetwork(
            HyperParams.obs_dim,
            HyperParams.action_dim,
            HyperParams.hidden_size,
            HyperParams.num_layers,
        ).to(device),
        ActorNetwork(
            HyperParams.obs_dim,
            HyperParams.action_dim,
            HyperParams.hidden_size,
            HyperParams.num_layers,
        ).to(device),
    ]
    critic = CentralizedCritic(
        HyperParams.joint_obs_dim, HyperParams.hidden_size, HyperParams.num_layers
    ).to(device)
    ppo = PPO(actors, critic, HyperParams, device=device)

    # Logger
    logger = Logger(args.log_dir, args.layout)

    # Load checkpoint if resuming
    start_episode = 0
    if args.resume:
        ckpt = os.path.join(args.checkpoint_dir, f"{args.layout}_latest.pt")
        if os.path.exists(ckpt):
            from .utils import load_checkpoint

            start_episode = load_checkpoint(
                actors, critic, ppo.actor_optimizer, ppo.critic_optimizer, ckpt, device
            )
            logger.load()

    # Training loop
    print(f"\nStarting training from episode {start_episode}...")
    print(f"Target episodes: {args.episodes}")
    print(f"Batch size: {HyperParams.batch_size}\n")

    last_update_stats = None  # cache last update stats for logging

    base_entropy = ppo.hp.entropy_coef
    for episode in range(start_episode, args.episodes):
        # Replace entropy schedule block
        if episode < 15000:
            ppo.hp.entropy_coef = base_entropy * 2.5
        elif episode < 20000:
            ppo.hp.entropy_coef = base_entropy * 1.5
        else:
            ppo.hp.entropy_coef = base_entropy

        obs = env.reset()
        state = env.state

        # Reward shaper
        progress = episode / args.episodes
        weights = HyperParams.get_shaped_reward_weights(progress)
        shaper = RewardShaper(env, weights, args.layout)
        shaper.reset(state)

        # Collaboration metrics
        collab = CollaborationMetrics()
        collab.reset()

        episode_reward = 0.0
        episode_length = 0
        num_soups = 0
        done = False
        idle_time = [0, 0]
        prev_state = state

        # Episode rollout
        while not done:
            o0 = np.concatenate([obs["both_agent_obs"], np.array([1.0, 0.0], dtype=np.float32)])
            o1 = np.concatenate([obs["both_agent_obs"], np.array([0.0, 1.0], dtype=np.float32)])
            observations = [o0, o1]

            actions, log_probs, entropies, value, joint_obs = ppo.select_actions(
                observations
            )

            for i in range(2):
                if actions[i] == HyperParams.ACTION_STAY:
                    idle_time[i] += 1

            next_obs, sparse_rewards, done, info = env.step(actions)
            next_state = env.state

            collab.update(next_state, prev_state, actions)
            shaped_rewards, shaping_info = shaper.compute_shaped_rewards(
                next_state, sparse_rewards, done
            )

            if sum(sparse_rewards) > 0:
                num_soups += int(sum(sparse_rewards) // 20)

            ppo.buffer.add(
                obs_pair=observations,
                joint_obs=joint_obs,
                actions=actions,
                log_probs=log_probs,
                rewards=shaped_rewards,
                value=value,
                done=done,
            )

            prev_state = state
            obs = next_obs
            state = next_state
            episode_reward += sum(shaped_rewards)
            episode_length += 1

            # PPO update when buffer full
            if len(ppo.buffer) >= HyperParams.batch_size:
                next_observations = [
                    next_obs["both_agent_obs"][0],
                    next_obs["both_agent_obs"][1],
                ]
                last_update_stats = ppo.update(next_observations)

        # Log episode
        collab_metrics = collab.get_episode_metrics()
        collab_info = {
            "idle_time_agent0": (
                idle_time[0] / episode_length if episode_length > 0 else 0.0
            ),
            "idle_time_agent1": (
                idle_time[1] / episode_length if episode_length > 0 else 0.0
            ),
            "pot_handoffs": collab_metrics["pot_handoffs"],
        }
        logger.log_episode(
            episode + 1, episode_reward, num_soups, episode_length, collab_info
        )

        # Write JSONL log every log_interval
        if (episode + 1) % HyperParams.log_interval == 0:
            recent = logger.get_recent_stats()
            rec = {
                "episode": int(episode + 1),
                "avg_soups_last_100": float(recent["avg_soups"]),
                "max_soups_last_100": int(recent["max_soups"]),
                "avg_reward_last_100": float(recent["avg_reward"]),
            }
            # Add shaping event counts from shaper
            if hasattr(shaper, "event_counts"):
                for k, v in shaper.event_counts.items():
                    rec[k + "_total"] = int(v)
            # Add PPO stats if available
            if last_update_stats is not None:
                rec.update(
                    {
                        "update_step": int(last_update_stats["update_step"]),
                        "actor_loss": float(last_update_stats["actor_loss"]),
                        "critic_loss": float(last_update_stats["critic_loss"]),
                        "entropy": float(last_update_stats["entropy"]),
                    }
                )
                print_training_stats(episode + 1, last_update_stats, recent)
            logger.log_jsonl(rec)

        # Save checkpoint periodically
        if (episode + 1) % HyperParams.save_interval == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"{args.layout}_ep{episode+1}.pt"
            )
            save_checkpoint(
                actors,
                critic,
                ppo.actor_optimizer,
                ppo.critic_optimizer,
                episode + 1,
                ckpt_path,
            )
            latest = os.path.join(args.checkpoint_dir, f"{args.layout}_latest.pt")
            save_checkpoint(
                actors,
                critic,
                ppo.actor_optimizer,
                ppo.critic_optimizer,
                episode + 1,
                latest,
            )
            logger.save()

        # Plot periodically
        if (episode + 1) % (HyperParams.save_interval * 2) == 0:
            plot_path = os.path.join(args.graph_dir, f"{args.layout}_training.png")
            logger.plot_training_curves(save_path=plot_path)

    # Final save
    print("\nTraining complete!")
    final = os.path.join(args.checkpoint_dir, f"{args.layout}_final.pt")
    save_checkpoint(
        actors, critic, ppo.actor_optimizer, ppo.critic_optimizer, args.episodes, final
    )
    logger.save()
    logger.close()
    plot_path = os.path.join(args.graph_dir, f"{args.layout}_training_final.png")
    logger.plot_training_curves(save_path=plot_path)

    # Final stats
    recent = logger.get_recent_stats()
    print(f"\nFinal Statistics:")
    print(f"Average Soups (last 100 episodes): {recent['avg_soups']:.2f}")
    print(f"Max Soups (last 100 episodes):     {recent['max_soups']:.0f}")
    print(f"Target Performance:                ≥7 soups/episode")
    print(f"Performance {'MET ✓' if recent['avg_soups'] >= 7 else 'NOT MET ✗'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO with Centralized Critic on Overcooked"
    )
    parser.add_argument(
        "--layout",
        type=str,
        required=True,
        choices=["cramped_room", "coordination_ring", "counter_circuit_o_1order"],
    )
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_dir", type=str, default="results/logs")
    parser.add_argument("--checkpoint_dir", type=str, default="results/models")
    parser.add_argument("--graph_dir", type=str, default="results/graphs")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.graph_dir, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
