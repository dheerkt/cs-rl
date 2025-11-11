"""
Evaluation script for trained PPO agents
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from .env_builder import build_overcooked_env
from .models import ActorNetwork, CentralizedCritic
from configs.hyperparameters import HyperParams


def load_trained_agents(checkpoint_path, device='cpu'):
    """
    Load trained agents from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to

    Returns:
        actors: List of loaded actor networks
    """
    print(f"[DEBUG] Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint first to infer obs_dim
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer obs_dim from checkpoint's first Linear layer weight shape
    in_features = checkpoint['actor0_state_dict']['shared.0.weight'].shape[1]
    print(f"[DEBUG] Inferred obs_dim from checkpoint: {in_features}")
    
    # Create networks with inferred dimension
    actors = [
        ActorNetwork(
            obs_dim=in_features,
            action_dim=HyperParams.action_dim,
            hidden_size=HyperParams.hidden_size,
            num_layers=HyperParams.num_layers
        ).to(device),
        ActorNetwork(
            obs_dim=in_features,
            action_dim=HyperParams.action_dim,
            hidden_size=HyperParams.hidden_size,
            num_layers=HyperParams.num_layers
        ).to(device)
    ]

    actors[0].load_state_dict(checkpoint['actor0_state_dict'])
    actors[1].load_state_dict(checkpoint['actor1_state_dict'])

    # Set to eval mode
    actors[0].eval()
    actors[1].eval()

    print(f"[DEBUG] Successfully loaded checkpoint with obs_dim={in_features}")
    return actors


def evaluate(env, actors, num_episodes=100, device='cpu', verbose=True):
    """
    Evaluate trained agents

    Args:
        env: Overcooked environment
        actors: List of actor networks
        num_episodes: Number of evaluation episodes
        device: Device to run on
        verbose: Print progress

    Returns:
        results: Dictionary of evaluation results
    """
    print(f"[DEBUG] Starting evaluation for {num_episodes} episodes...")
    
    episode_soups = []
    episode_rewards = []
    episode_lengths = []

    # Collaboration metrics
    idle_times = [[], []]

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        num_soups = 0
        idle_count = [0, 0]

        while not done:
            # Get observations
            observations = [obs['both_agent_obs'][0], obs['both_agent_obs'][1]]

            # Get actions (deterministic for evaluation)
            actions = []
            with torch.no_grad():
                for i, actor in enumerate(actors):
                    obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0).to(device)
                    action, _, _ = actor.get_action(obs_tensor, deterministic=True)
                    actions.append(action.item())

                    # Track idle time using action constant
                    if action.item() == HyperParams.ACTION_STAY:
                        idle_count[i] += 1

            # Step environment
            obs, rewards, done, info = env.step(actions)

            episode_reward += sum(rewards)
            episode_length += 1

            # Count soups
            if sum(rewards) > 0:
                num_soups += sum(rewards) // 20

        # Log episode results
        episode_soups.append(num_soups)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        for i in range(2):
            idle_times[i].append(idle_count[i] / episode_length if episode_length > 0 else 0)

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Soups = {num_soups}, Reward = {episode_reward:.1f}")

    # Compile results
    results = {
        'mean_soups': np.mean(episode_soups),
        'std_soups': np.std(episode_soups),
        'min_soups': np.min(episode_soups),
        'max_soups': np.max(episode_soups),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'episode_soups': episode_soups,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'idle_time_agent0': np.mean(idle_times[0]),
        'idle_time_agent1': np.mean(idle_times[1]),
    }

    return results


def print_results(layout_name, results):
    """Print evaluation results"""
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {layout_name}")
    print(f"{'='*60}")
    print(f"Soups Delivered:")
    print(f"  Mean:  {results['mean_soups']:.2f} ± {results['std_soups']:.2f}")
    print(f"  Min:   {results['min_soups']:.0f}")
    print(f"  Max:   {results['max_soups']:.0f}")
    print(f"\nRewards:")
    print(f"  Mean:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"\nEpisode Length:")
    print(f"  Mean:  {results['mean_length']:.1f}")
    print(f"\nCollaboration Metrics:")
    print(f"  Agent 0 Idle Time: {results['idle_time_agent0']:.2%}")
    print(f"  Agent 1 Idle Time: {results['idle_time_agent1']:.2%}")
    print(f"\nTarget Performance: ≥7 soups/episode")
    print(f"Status: {'✓ MET' if results['mean_soups'] >= 7 else '✗ NOT MET'}")
    print(f"{'='*60}\n")


def plot_evaluation_results(results_dict, save_path=None):
    """
    Plot evaluation results for all layouts

    Args:
        results_dict: Dictionary mapping layout_name -> results
        save_path: Path to save figure
    """
    layouts = list(results_dict.keys())
    num_layouts = len(layouts)

    fig, axes = plt.subplots(1, num_layouts, figsize=(6*num_layouts, 5))
    if num_layouts == 1:
        axes = [axes]

    for i, layout in enumerate(layouts):
        results = results_dict[layout]
        episode_soups = results['episode_soups']

        # Plot histogram
        axes[i].hist(episode_soups, bins=range(int(min(episode_soups)), int(max(episode_soups)) + 2),
                     alpha=0.7, edgecolor='black')
        axes[i].axvline(x=results['mean_soups'], color='red', linestyle='--',
                        label=f"Mean: {results['mean_soups']:.2f}")
        axes[i].axvline(x=7, color='green', linestyle='--', label='Target: 7')
        axes[i].set_xlabel('Soups Delivered')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{layout}\n({len(episode_soups)} episodes)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved evaluation plot to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    print("="*60)
    print("EVALUATION SCRIPT STARTED")
    print("="*60)
    
    parser = argparse.ArgumentParser(description='Evaluate trained PPO agents')

    # Evaluation settings
    parser.add_argument('--layout', type=str, default=None,
                        choices=['cramped_room', 'coordination_ring', 'counter_circuit_o_1order'],
                        help='Layout to evaluate (if None, evaluate all)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (if None, use default)')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage')

    # Directories
    parser.add_argument('--checkpoint_dir', type=str, default='results/models',
                        help='Directory containing checkpoints')
    parser.add_argument('--results_dir', type=str, default='results/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--graph_dir', type=str, default='results/graphs',
                        help='Directory to save graphs')

    args = parser.parse_args()
    
    print(f"[DEBUG] Arguments parsed:")
    print(f"  Layout: {args.layout}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Num episodes: {args.num_episodes}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[DEBUG] Using device: {device}")

    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.graph_dir, exist_ok=True)
    print(f"[DEBUG] Created output directories")

    # Determine layouts to evaluate
    if args.layout:
        layouts = [args.layout]
    else:
        layouts = ['cramped_room', 'coordination_ring', 'counter_circuit_o_1order']
    
    print(f"[DEBUG] Will evaluate layouts: {layouts}")

    # Evaluate each layout
    all_results = {}

    for layout in layouts:
        print(f"\n{'='*60}")
        print(f"Evaluating {layout}")
        print(f"{'='*60}\n")

        # Get checkpoint path
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{layout}_final.pt")
        
        print(f"[DEBUG] Checkpoint path: {checkpoint_path}")
        print(f"[DEBUG] Checkpoint exists: {os.path.exists(checkpoint_path)}")
        
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            print(f"[DEBUG] Available checkpoints in {args.checkpoint_dir}:")
            if os.path.exists(args.checkpoint_dir):
                for f in os.listdir(args.checkpoint_dir):
                    print(f"  - {f}")
            continue

        # Build environment with fixed seed for reproducibility
        print(f"[DEBUG] Building environment...")
        env = build_overcooked_env(layout, horizon=400, seed=42)
        print(f"[DEBUG] Environment built successfully")

        # Load agents
        print(f"[DEBUG] Loading agents...")
        try:
            actors = load_trained_agents(checkpoint_path, device)
            print(f"[DEBUG] Agents loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load agents: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Evaluate
        print(f"[DEBUG] Starting evaluation...")
        try:
            results = evaluate(env, actors, args.num_episodes, device)
            all_results[layout] = results
            print(f"[DEBUG] Evaluation completed successfully")
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Print results
        print_results(layout, results)

        # Save results
        results_path = os.path.join(args.results_dir, f"{layout}_eval.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {k: (v.tolist() if isinstance(v, np.ndarray) else
                               float(v) if isinstance(v, (np.floating, np.integer)) else v)
                           for k, v in results.items()}
            json.dump(json_results, f, indent=2)
        print(f"[DEBUG] Saved results to {results_path}")

    # Plot combined results
    if len(all_results) > 0:
        plot_path = os.path.join(args.graph_dir, 'evaluation_results.png')
        plot_evaluation_results(all_results, save_path=plot_path)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for layout, results in all_results.items():
        status = '✓' if results['mean_soups'] >= 7 else '✗'
        print(f"{status} {layout:25s}: {results['mean_soups']:.2f} ± {results['std_soups']:.2f} soups")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
