"""
Neural network models for PPO with Centralized Critic

Actor: Decentralized policy network (one per agent)
Critic: Centralized value network (observes both agents)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """
    Decentralized policy network for one agent
    Input: Single agent observation (96 dims)
    Output: Action probabilities (6 actions)
    """

    def __init__(self, obs_dim=96, action_dim=6, hidden_size=256, num_layers=2):
        super(ActorNetwork, self).__init__()

        layers = []
        input_dim = obs_dim

        # Build hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        self.shared = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_size, action_dim)

    def forward(self, obs):
        """
        Forward pass

        Args:
            obs: Observation tensor [batch, obs_dim]

        Returns:
            action_probs: Action probability distribution [batch, action_dim]
        """
        features = self.shared(obs)
        logits = self.action_head(features)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs

    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy

        Args:
            obs: Observation tensor [batch, obs_dim]
            deterministic: If True, take argmax action

        Returns:
            action: Sampled action [batch]
            log_prob: Log probability of action [batch]
            entropy: Policy entropy [batch]
        """
        action_probs = self.forward(obs)
        dist = Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_actions(self, obs, actions):
        """
        Evaluate log probabilities and entropy of given actions

        Args:
            obs: Observation tensor [batch, obs_dim]
            actions: Action tensor [batch]

        Returns:
            log_probs: Log probabilities of actions [batch]
            entropy: Policy entropy [batch]
        """
        action_probs = self.forward(obs)
        dist = Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized value network
    Input: Joint observation from both agents (192 dims = 96 * 2)
    Output: State value
    """

    def __init__(self, joint_obs_dim=192, hidden_size=256, num_layers=2):
        super(CentralizedCritic, self).__init__()

        layers = []
        input_dim = joint_obs_dim

        # Build hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        self.shared = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, joint_obs):
        """
        Forward pass

        Args:
            joint_obs: Concatenated observations [batch, joint_obs_dim]

        Returns:
            value: State value [batch, 1]
        """
        features = self.shared(joint_obs)
        value = self.value_head(features)
        return value


class PPOAgent:
    """
    Container for actor and critic networks with utility methods
    """

    def __init__(self, agent_id, obs_dim=96, joint_obs_dim=192, action_dim=6,
                 hidden_size=256, num_layers=2, device='cpu'):
        """
        Initialize PPO agent

        Args:
            agent_id: Agent identifier (0 or 1)
            obs_dim: Observation dimension for single agent
            joint_obs_dim: Joint observation dimension for centralized critic
            action_dim: Action space dimension
            hidden_size: Hidden layer size
            num_layers: Number of hidden layers
            device: Device to run on
        """
        self.agent_id = agent_id
        self.device = device

        # Actor network (decentralized)
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_size, num_layers).to(device)

        # Note: Critic is shared and passed separately

    def get_action(self, obs, deterministic=False):
        """Get action from policy"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, entropy = self.actor.get_action(obs_tensor, deterministic)
            return action.item(), log_prob.item(), entropy.item()

    def save(self, path):
        """Save actor network"""
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        """Load actor network"""
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
