"""
PPO (Proximal Policy Optimization) with Centralized Critic
Implements CTDE (Centralized Training, Decentralized Execution)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class RolloutBuffer:
    """
    Buffer for storing trajectories during rollout
    """

    def __init__(self):
        self.observations = [[], []]  # Per agent
        self.actions = [[], []]
        self.log_probs = [[], []]
        self.rewards = [[], []]
        self.values = []  # Centralized value
        self.dones = []
        self.joint_observations = []  # For centralized critic

    def add(self, obs, joint_obs, actions, log_probs, rewards, value, done):
        """
        Add a transition to buffer

        Args:
            obs: List of observations [obs_agent0, obs_agent1]
            joint_obs: Concatenated observation
            actions: List of actions [action0, action1]
            log_probs: List of log probs [log_prob0, log_prob1]
            rewards: List of rewards [reward0, reward1]
            value: Centralized value estimate
            done: Done flag
        """
        for i in range(2):
            self.observations[i].append(obs[i])
            self.actions[i].append(actions[i])
            self.log_probs[i].append(log_probs[i])
            self.rewards[i].append(rewards[i])

        self.joint_observations.append(joint_obs)
        self.values.append(value)
        self.dones.append(done)

    def get(self):
        """
        Get all data as numpy arrays

        Returns:
            Dictionary of numpy arrays
        """
        return {
            'observations': [np.array(self.observations[i]) for i in range(2)],
            'joint_observations': np.array(self.joint_observations),
            'actions': [np.array(self.actions[i]) for i in range(2)],
            'log_probs': [np.array(self.log_probs[i]) for i in range(2)],
            'rewards': [np.array(self.rewards[i]) for i in range(2)],
            'values': np.array(self.values),
            'dones': np.array(self.dones),
        }

    def clear(self):
        """Clear buffer"""
        self.observations = [[], []]
        self.actions = [[], []]
        self.log_probs = [[], []]
        self.rewards = [[], []]
        self.values = []
        self.dones = []
        self.joint_observations = []

    def __len__(self):
        """Return buffer size"""
        return len(self.dones)


class PPO:
    """
    PPO algorithm with centralized critic for multi-agent learning
    """

    def __init__(self, actors, critic, hyperparams, device='cpu'):
        """
        Initialize PPO

        Args:
            actors: List of actor networks [actor0, actor1]
            critic: Centralized critic network
            hyperparams: Hyperparameter object
            device: Device to run on
        """
        self.actors = actors
        self.critic = critic
        self.hp = hyperparams
        self.device = device

        # Optimizers
        actor_params = list(actors[0].parameters()) + list(actors[1].parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=self.hp.lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=self.hp.lr)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Statistics
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []

    def select_actions(self, observations):
        """
        Select actions for both agents

        Args:
            observations: List of observations [obs0, obs1]

        Returns:
            actions: List of actions
            log_probs: List of log probabilities
            entropies: List of entropies
            value: Centralized value estimate
        """
        actions = []
        log_probs = []
        entropies = []

        # Get actions from each actor
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                action, log_prob, entropy = actor.get_action(obs_tensor)
                actions.append(action.item())
                log_probs.append(log_prob.item())
                entropies.append(entropy.item())

            # Get centralized value
            joint_obs = np.concatenate(observations)
            joint_obs_tensor = torch.FloatTensor(joint_obs).unsqueeze(0).to(self.device)
            value = self.critic(joint_obs_tensor).item()

        return actions, log_probs, entropies, value

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            rewards: Reward array [timesteps]
            values: Value array [timesteps]
            dones: Done array [timesteps]
            next_value: Value of next state

        Returns:
            advantages: Advantage estimates [timesteps]
            returns: Return estimates [timesteps]
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0

        # Compute advantages backward
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # TD error
            delta = rewards[t] + self.hp.gamma * next_val * (1 - dones[t]) - values[t]

            # GAE
            advantages[t] = last_gae = delta + self.hp.gamma * self.hp.gae_lambda * (1 - dones[t]) * last_gae

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def update(self, next_obs):
        """
        Update policy and value function using PPO

        Args:
            next_obs: Next observation for computing final value

        Returns:
            Dictionary of training statistics
        """
        # Get buffer data
        data = self.buffer.get()

        # Compute next value for GAE
        with torch.no_grad():
            joint_next_obs = np.concatenate(next_obs)
            joint_next_obs_tensor = torch.FloatTensor(joint_next_obs).unsqueeze(0).to(self.device)
            next_value = self.critic(joint_next_obs_tensor).item()

        # Compute advantages and returns for each agent
        all_advantages = []
        all_returns = []

        for i in range(2):
            advantages, returns = self.compute_gae(
                data['rewards'][i],
                data['values'],
                data['dones'],
                next_value
            )
            all_advantages.append(advantages)
            all_returns.append(returns)

        # Normalize advantages (combined from both agents)
        combined_advantages = np.concatenate(all_advantages)
        adv_mean = combined_advantages.mean()
        adv_std = combined_advantages.std() + 1e-8
        all_advantages = [(adv - adv_mean) / adv_std for adv in all_advantages]

        # Convert to tensors
        obs_tensors = [torch.FloatTensor(data['observations'][i]).to(self.device) for i in range(2)]
        joint_obs_tensor = torch.FloatTensor(data['joint_observations']).to(self.device)
        action_tensors = [torch.LongTensor(data['actions'][i]).to(self.device) for i in range(2)]
        old_log_prob_tensors = [torch.FloatTensor(data['log_probs'][i]).to(self.device) for i in range(2)]
        advantage_tensors = [torch.FloatTensor(all_advantages[i]).to(self.device) for i in range(2)]
        return_tensors = [torch.FloatTensor(all_returns[i]).to(self.device) for i in range(2)]

        # PPO update for multiple epochs
        batch_size = len(data['dones'])
        indices = np.arange(batch_size)

        actor_losses = []
        critic_losses = []
        entropies = []

        for epoch in range(self.hp.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.hp.minibatch_size):
                end = start + self.hp.minibatch_size
                mb_indices = indices[start:end]

                # Update actors
                total_actor_loss = 0
                total_entropy = 0

                for i in range(2):
                    mb_obs = obs_tensors[i][mb_indices]
                    mb_actions = action_tensors[i][mb_indices]
                    mb_old_log_probs = old_log_prob_tensors[i][mb_indices]
                    mb_advantages = advantage_tensors[i][mb_indices]

                    # Get current log probs and entropy
                    new_log_probs, entropy = self.actors[i].evaluate_actions(mb_obs, mb_actions)

                    # Ratio for PPO
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)

                    # Clipped surrogate objective
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.hp.clip_epsilon, 1 + self.hp.clip_epsilon) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    total_actor_loss += actor_loss
                    total_entropy += entropy.mean()

                # Entropy bonus (encourage exploration)
                total_actor_loss -= self.hp.entropy_coef * total_entropy

                # Update actors
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actors[0].parameters()) + list(self.actors[1].parameters()),
                    self.hp.max_grad_norm
                )
                self.actor_optimizer.step()

                # Update centralized critic
                mb_joint_obs = joint_obs_tensor[mb_indices]
                # Use average of both agents' returns for centralized critic
                mb_returns = (return_tensors[0][mb_indices] + return_tensors[1][mb_indices]) / 2.0

                values = self.critic(mb_joint_obs).squeeze(-1)
                critic_loss = nn.MSELoss()(values, mb_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.hp.max_grad_norm)
                self.critic_optimizer.step()

                # Record stats
                actor_losses.append(total_actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(total_entropy.item() / 2.0)  # Average over agents

        # Clear buffer
        self.buffer.clear()

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
        }
