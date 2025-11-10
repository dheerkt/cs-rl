"""
Utility functions for logging, saving, and metrics tracking
"""

import os
import json
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt


class CollaborationMetrics:
    """
    Tracks collaboration metrics between agents
    Specifically: pot handoffs (agents working on same pot)
    """

    def __init__(self):
        """Initialize collaboration metrics tracker"""
        self.pot_last_interactor = {}  # pot_position -> agent_id
        self.handoffs_this_episode = 0
        self.total_handoffs = 0

    def reset(self):
        """Reset for new episode"""
        self.pot_last_interactor.clear()
        self.handoffs_this_episode = 0

    def update(self, state, prev_state, agent_actions):
        """
        Update metrics based on state transition

        Args:
            state: Current environment state
            prev_state: Previous environment state
            agent_actions: List of actions taken [action0, action1]
        """
        from configs.hyperparameters import HyperParams

        # Get pot states
        pots = self._get_pot_info(state)
        prev_pots = self._get_pot_info(prev_state)

        # Check each pot for changes indicating interaction
        for pot_pos, pot_info in pots.items():
            if pot_pos not in prev_pots:
                continue

            prev_info = prev_pots[pot_pos]

            # Detect if pot state changed (someone interacted with it)
            pot_changed = (
                pot_info['num_items'] != prev_info['num_items'] or
                pot_info['is_cooking'] != prev_info['is_cooking'] or
                pot_info['is_ready'] != prev_info['is_ready']
            )

            if not pot_changed:
                continue

            # Determine which agent likely caused the change
            # Agent must be adjacent and have taken an INTERACT action
            interacting_agent = None

            for agent_id in range(2):
                agent_pos = state.players[agent_id].position
                action = agent_actions[agent_id]

                # Check if agent is adjacent to pot and interacted
                if action == HyperParams.ACTION_INTERACT:
                    distance = abs(agent_pos[0] - pot_pos[0]) + abs(agent_pos[1] - pot_pos[1])
                    if distance == 1:  # Adjacent (Manhattan distance = 1)
                        interacting_agent = agent_id
                        break

            if interacting_agent is None:
                continue

            # Check if this is a handoff (different agent than last interaction)
            if pot_pos in self.pot_last_interactor:
                last_agent = self.pot_last_interactor[pot_pos]
                if last_agent != interacting_agent:
                    # Handoff detected!
                    self.handoffs_this_episode += 1
                    self.total_handoffs += 1

            # Update last interactor
            self.pot_last_interactor[pot_pos] = interacting_agent

    def get_episode_metrics(self):
        """Get metrics for current episode"""
        return {
            'pot_handoffs': self.handoffs_this_episode,
        }

    def _get_pot_info(self, state):
        """Extract pot information from state"""
        pots = {}

        if hasattr(state, 'objects'):
            for pos, obj in state.objects.items():
                obj_name = getattr(obj, 'name', '')
                if 'soup' in str(obj_name).lower() or 'pot' in str(obj_name).lower():
                    num_items = 0
                    if hasattr(obj, 'ingredients'):
                        num_items = len(obj.ingredients)
                    elif hasattr(obj, '_ingredients'):
                        num_items = len(obj._ingredients)
                    elif hasattr(obj, 'num_items'):
                        num_items = obj.num_items

                    pots[pos] = {
                        'is_cooking': getattr(obj, 'is_cooking', False) or getattr(obj, '_cooking', False),
                        'is_ready': getattr(obj, 'is_ready', False) or getattr(obj, '_ready', False),
                        'num_items': num_items,
                    }

        return pots


class Logger:
    """
    Simple logger for tracking training metrics
    """

    def __init__(self, log_dir, layout_name):
        """
        Initialize logger

        Args:
            log_dir: Directory to save logs
            layout_name: Name of layout being trained
        """
        self.log_dir = log_dir
        self.layout_name = layout_name
        os.makedirs(log_dir, exist_ok=True)

        # JSONL file for real-time logging
        self.jsonl_path = os.path.join(log_dir, f"{layout_name}.jsonl")
        print(f"Logging to: {os.path.abspath(self.jsonl_path)}")
        # Line-buffered file for immediate writes
        self.jsonl_file = open(self.jsonl_path, "a", buffering=1)

        # Metrics
        self.episode_rewards = []
        self.episode_soups = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []

        # Collaboration metrics
        self.pot_handoffs = []
        self.idle_times = [[], []]  # Per agent

        # Moving averages
        self.window_size = 100
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_soups = deque(maxlen=self.window_size)

    def log_episode(self, episode, total_reward, num_soups, episode_length, info=None):
        """
        Log episode statistics

        Args:
            episode: Episode number
            total_reward: Total reward for episode
            num_soups: Number of soups delivered
            episode_length: Length of episode
            info: Additional info dictionary
        """
        self.episode_rewards.append(total_reward)
        self.episode_soups.append(num_soups)
        self.episode_lengths.append(episode_length)

        self.recent_rewards.append(total_reward)
        self.recent_soups.append(num_soups)

        # Log collaboration metrics if available
        if info:
            if 'pot_handoffs' in info:
                self.pot_handoffs.append(info['pot_handoffs'])
            if 'idle_time_agent0' in info:
                self.idle_times[0].append(info['idle_time_agent0'])
            if 'idle_time_agent1' in info:
                self.idle_times[1].append(info['idle_time_agent1'])

    def log_update(self, stats):
        """
        Log training update statistics

        Args:
            stats: Dictionary with 'actor_loss', 'critic_loss', 'entropy'
        """
        self.actor_losses.append(stats['actor_loss'])
        self.critic_losses.append(stats['critic_loss'])
        self.entropies.append(stats['entropy'])

    def log_jsonl(self, record):
        """
        Write a JSON line to the log file with immediate flush

        Args:
            record: Dictionary to log as JSON
        """
        import time
        record['timestamp'] = time.time()
        self.jsonl_file.write(json.dumps(record) + '\n')
        self.jsonl_file.flush()
        # Force write to disk for real-time monitoring
        os.fsync(self.jsonl_file.fileno())

    def get_recent_stats(self):
        """
        Get recent statistics

        Returns:
            Dictionary of recent stats
        """
        return {
            'avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0,
            'avg_soups': np.mean(self.recent_soups) if self.recent_soups else 0,
            'max_soups': np.max(self.recent_soups) if self.recent_soups else 0,
        }

    def close(self):
        """Close JSONL file"""
        try:
            self.jsonl_file.flush()
            os.fsync(self.jsonl_file.fileno())
            self.jsonl_file.close()
        except Exception:
            pass

    def save(self):
        """Save all metrics to disk"""
        save_path = os.path.join(self.log_dir, f'{self.layout_name}_metrics.json')

        data = {
            'episode_rewards': self.episode_rewards,
            'episode_soups': self.episode_soups,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropies': self.entropies,
            'pot_handoffs': self.pot_handoffs,
            'idle_times_agent0': self.idle_times[0],
            'idle_times_agent1': self.idle_times[1],
        }

        with open(save_path, 'w') as f:
            json.dump(data, f)

        print(f"Saved metrics to {save_path}")

    def load(self):
        """Load metrics from disk"""
        load_path = os.path.join(self.log_dir, f'{self.layout_name}_metrics.json')

        if not os.path.exists(load_path):
            print(f"No saved metrics found at {load_path}")
            return

        with open(load_path, 'r') as f:
            data = json.load(f)

        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_soups = data.get('episode_soups', [])
        self.episode_lengths = data.get('episode_lengths', [])
        self.actor_losses = data.get('actor_losses', [])
        self.critic_losses = data.get('critic_losses', [])
        self.entropies = data.get('entropies', [])
        self.pot_handoffs = data.get('pot_handoffs', [])
        self.idle_times[0] = data.get('idle_times_agent0', [])
        self.idle_times[1] = data.get('idle_times_agent1', [])

        print(f"Loaded metrics from {load_path}")

    def plot_training_curves(self, save_path=None):
        """
        Plot training curves

        Args:
            save_path: Path to save figure (if None, just display)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Soups delivered
        axes[0, 0].plot(self.episode_soups, alpha=0.3, label='Raw')
        if len(self.episode_soups) > self.window_size:
            smoothed = self._moving_average(self.episode_soups, self.window_size)
            axes[0, 0].plot(smoothed, label=f'{self.window_size}-episode MA')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Soups Delivered')
        axes[0, 0].set_title(f'Training Progress - {self.layout_name}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Episode rewards
        axes[0, 1].plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) > self.window_size:
            smoothed = self._moving_average(self.episode_rewards, self.window_size)
            axes[0, 1].plot(smoothed, label=f'{self.window_size}-episode MA')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].set_title('Episode Rewards')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Losses
        if self.actor_losses:
            axes[1, 0].plot(self.actor_losses, label='Actor Loss', alpha=0.7)
            axes[1, 0].plot(self.critic_losses, label='Critic Loss', alpha=0.7)
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Entropy
        if self.entropies:
            axes[1, 1].plot(self.entropies, alpha=0.7)
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].set_title('Policy Entropy')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training curves to {save_path}")
        else:
            plt.show()

        plt.close()

    def _moving_average(self, data, window):
        """Compute moving average"""
        return np.convolve(data, np.ones(window) / window, mode='valid')


def save_checkpoint(actors, critic, optimizer_actor, optimizer_critic, episode, save_path):
    """
    Save model checkpoint

    Args:
        actors: List of actor networks
        critic: Critic network
        optimizer_actor: Actor optimizer
        optimizer_critic: Critic optimizer
        episode: Current episode number
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'episode': episode,
        'actor0_state_dict': actors[0].state_dict(),
        'actor1_state_dict': actors[1].state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(actors, critic, optimizer_actor, optimizer_critic, load_path, device='cpu'):
    """
    Load model checkpoint

    Args:
        actors: List of actor networks
        critic: Critic network
        optimizer_actor: Actor optimizer
        optimizer_critic: Critic optimizer
        load_path: Path to load checkpoint from
        device: Device to load to

    Returns:
        Episode number from checkpoint
    """
    if not os.path.exists(load_path):
        print(f"No checkpoint found at {load_path}")
        return 0

    checkpoint = torch.load(load_path, map_location=device)

    actors[0].load_state_dict(checkpoint['actor0_state_dict'])
    actors[1].load_state_dict(checkpoint['actor1_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
    optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

    episode = checkpoint['episode']
    print(f"Loaded checkpoint from {load_path} (episode {episode})")

    return episode


def print_training_stats(episode, stats, logger_stats):
    """
    Print training statistics

    Args:
        episode: Episode number
        stats: Training stats from PPO update
        logger_stats: Recent stats from logger
    """
    print(f"\n{'='*60}")
    print(f"Episode {episode}")
    print(f"{'='*60}")
    print(f"Average Soups (last 100):  {logger_stats['avg_soups']:.2f}")
    print(f"Max Soups (last 100):      {logger_stats['max_soups']:.0f}")
    print(f"Average Reward (last 100): {logger_stats['avg_reward']:.2f}")
    print(f"Actor Loss:                {stats['actor_loss']:.4f}")
    print(f"Critic Loss:               {stats['critic_loss']:.4f}")
    print(f"Entropy:                   {stats['entropy']:.4f}")
    print(f"{'='*60}\n")
