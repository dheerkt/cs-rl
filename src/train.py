"""
Training script for PPO with Centralized Critic on Overcooked layouts
"""

import argparse
import os
import sys
import numpy as np
import torch

# Import Overcooked environment
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# Import our modules
from models import ActorNetwork, CentralizedCritic
from ppo import PPO
from reward_shaping import RewardShaper
from utils import Logger, save_checkpoint, print_training_stats
from configs.hyperparameters import HyperParams


def build_overcooked_env(layout_name, horizon=400, seed=None):
    """
    Build Overcooked environment

    Args:
        layout_name: Layout name (e.g., 'cramped_room')
        horizon: Episode horizon (default 400, do not change per spec)
        seed: Random seed for environment (for reproducibility)

    Returns:
        OvercookedEnv instance
    """
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)

    # CRITICAL FIX: Seed environment for reproducibility
    if seed is not None:
        try:
            # Try to seed the environment if it supports it
            if hasattr(env, 'seed'):
                env.seed(seed)
            # Also try to seed the MDP
            if hasattr(env, 'mdp') and hasattr(env.mdp, 'seed'):
                env.mdp.seed(seed)
        except Exception as e:
            # Some versions might not support seeding, that's okay
            print(f"Warning: Could not seed environment: {e}")

    return env


def train(args):
    """
    Main training loop

    Args:
        args: Command line arguments
    """
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # Build environment with seeding
    print(f"\nBuilding Overcooked environment: {args.layout}")
    env = build_overcooked_env(args.layout, horizon=400, seed=args.seed)
    print(f"Observation shape: {HyperParams.obs_dim}")
    print(f"Action space: {HyperParams.action_dim}")
    print(f"Random seed: {args.seed}")

    # Create networks
    print("\nInitializing networks...")
    actors = [
        ActorNetwork(
            obs_dim=HyperParams.obs_dim,
            action_dim=HyperParams.action_dim,
            hidden_size=HyperParams.hidden_size,
            num_layers=HyperParams.num_layers
        ).to(device),
        ActorNetwork(
            obs_dim=HyperParams.obs_dim,
            action_dim=HyperParams.action_dim,
            hidden_size=HyperParams.hidden_size,
            num_layers=HyperParams.num_layers
        ).to(device)
    ]

    critic = CentralizedCritic(
        joint_obs_dim=HyperParams.joint_obs_dim,
        hidden_size=HyperParams.hidden_size,
        num_layers=HyperParams.num_layers
    ).to(device)

    # Create PPO agent
    ppo = PPO(actors, critic, HyperParams, device=device)

    # Logger
    logger = Logger(args.log_dir, args.layout)

    # Load checkpoint if resuming
    start_episode = 0
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.layout}_latest.pt")
        if os.path.exists(checkpoint_path):
            from utils import load_checkpoint
            start_episode = load_checkpoint(
                actors, critic,
                ppo.actor_optimizer, ppo.critic_optimizer,
                checkpoint_path, device
            )
            logger.load()

    # Training loop
    print(f"\nStarting training from episode {start_episode}...")
    print(f"Target episodes: {args.episodes}")
    print(f"Batch size: {HyperParams.batch_size}")

    for episode in range(start_episode, args.episodes):
        # Reset environment
        obs = env.reset()
        state = env.state

        # Initialize reward shaper
        training_progress = episode / args.episodes
        shape_weights = HyperParams.get_shaped_reward_weights(training_progress)
        reward_shaper = RewardShaper(env, shape_weights)
        reward_shaper.reset(state)

        # Episode statistics
        episode_reward = 0
        episode_length = 0
        num_soups = 0
        done = False

        # Tracking for collaboration metrics
        pot_states = {}
        idle_time = [0, 0]
        last_actions = [None, None]

        # Episode rollout
        while not done:
            # Get observations for both agents
            observations = [obs['both_agent_obs'][0], obs['both_agent_obs'][1]]

            # Select actions
            actions, log_probs, entropies, value = ppo.select_actions(observations)

            # Track idle time using action constant
            for i in range(2):
                if actions[i] == HyperParams.ACTION_STAY:
                    idle_time[i] += 1

            # Step environment
            next_obs, sparse_rewards, done, info = env.step(actions)
            next_state = env.state

            # Apply reward shaping with agent-index swap correction
            shaped_rewards, shaping_info = reward_shaper.compute_shaped_rewards(
                next_state, sparse_rewards, done
            )

            # Track soups delivered
            if sum(sparse_rewards) > 0:
                num_soups += sum(sparse_rewards) // 20  # Each soup is +20

            # Store transition
            joint_obs = np.concatenate(observations)
            ppo.buffer.add(
                observations, joint_obs,
                actions, log_probs,
                shaped_rewards, value, done
            )

            # Update state
            obs = next_obs
            state = next_state
            episode_reward += sum(shaped_rewards)
            episode_length += 1

            # PPO update when buffer is full
            if len(ppo.buffer) >= HyperParams.batch_size:
                next_observations = [next_obs['both_agent_obs'][0], next_obs['both_agent_obs'][1]]
                update_stats = ppo.update(next_observations)
                logger.log_update(update_stats)

        # Log episode
        collab_info = {
            'idle_time_agent0': idle_time[0] / episode_length if episode_length > 0 else 0,
            'idle_time_agent1': idle_time[1] / episode_length if episode_length > 0 else 0,
        }
        logger.log_episode(episode + 1, episode_reward, num_soups, episode_length, collab_info)

        # Print stats periodically
        if (episode + 1) % HyperParams.log_interval == 0:
            recent_stats = logger.get_recent_stats()
            if ppo.actor_losses:
                update_stats = {
                    'actor_loss': ppo.actor_losses[-1],
                    'critic_loss': ppo.critic_losses[-1],
                    'entropy': ppo.entropies[-1],
                }
                print_training_stats(episode + 1, update_stats, recent_stats)

        # Save checkpoint periodically
        if (episode + 1) % HyperParams.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.layout}_ep{episode+1}.pt")
            save_checkpoint(
                actors, critic,
                ppo.actor_optimizer, ppo.critic_optimizer,
                episode + 1, checkpoint_path
            )

            # Save latest checkpoint
            latest_path = os.path.join(args.checkpoint_dir, f"{args.layout}_latest.pt")
            save_checkpoint(
                actors, critic,
                ppo.actor_optimizer, ppo.critic_optimizer,
                episode + 1, latest_path
            )

            # Save metrics
            logger.save()

        # Plot training curves periodically
        if (episode + 1) % (HyperParams.save_interval * 2) == 0:
            plot_path = os.path.join(args.graph_dir, f"{args.layout}_training.png")
            logger.plot_training_curves(save_path=plot_path)

    # Final save
    print("\nTraining complete!")
    final_path = os.path.join(args.checkpoint_dir, f"{args.layout}_final.pt")
    save_checkpoint(
        actors, critic,
        ppo.actor_optimizer, ppo.critic_optimizer,
        args.episodes, final_path
    )
    logger.save()

    # Final plot
    plot_path = os.path.join(args.graph_dir, f"{args.layout}_training_final.png")
    logger.plot_training_curves(save_path=plot_path)

    # Print final stats
    recent_stats = logger.get_recent_stats()
    print(f"\nFinal Statistics:")
    print(f"Average Soups (last 100 episodes): {recent_stats['avg_soups']:.2f}")
    print(f"Max Soups (last 100 episodes):     {recent_stats['max_soups']:.0f}")
    print(f"Target Performance:                ≥7 soups/episode")
    print(f"Performance {'MET ✓' if recent_stats['avg_soups'] >= 7 else 'NOT MET ✗'}")


def main():
    parser = argparse.ArgumentParser(description='Train PPO with Centralized Critic on Overcooked')

    # Environment
    parser.add_argument('--layout', type=str, required=True,
                        choices=['cramped_room', 'coordination_ring', 'counter_circuit_o_1order'],
                        help='Overcooked layout name')
    parser.add_argument('--episodes', type=int, default=50000,
                        help='Number of training episodes')

    # Training
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU available')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')

    # Directories
    parser.add_argument('--log_dir', type=str, default='results/logs',
                        help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, default='results/models',
                        help='Directory for model checkpoints')
    parser.add_argument('--graph_dir', type=str, default='results/graphs',
                        help='Directory for graphs')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.graph_dir, exist_ok=True)

    # Train
    train(args)


if __name__ == '__main__':
    main()
