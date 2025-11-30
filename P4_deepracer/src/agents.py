import abc
import numpy as np
import torch
import torch.nn as nn

from src.transforms import EncodeObservation


class Agent(nn.Module, abc.ABC):
    '''Boilerplate class for providing interface'''
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    @abc.abstractmethod
    def get_action(self, x):
        raise NotImplementedError


class RandomAgent(Agent):
    '''
    A random agent for demonstrating usage of the environment
    '''
    def __init__(self, environment, name='random'):
        super().__init__(name=name)
        self.action_space = environment.action_space        

    def get_action(self, x):
        return self.action_space.sample()


class MyFancyAgent(Agent):
    '''
    Actor-Critic agent for DeepRacer using PPO/A2C algorithm.
    
    Architecture:
        Encoder: EncodeObservation (camera + lidar -> 128-dim)
        Actor: MLP(128 -> 64 -> action_dim) with Tanh activation
        Critic: MLP(128 -> 64 -> 1) with Tanh activation
    '''
    def __init__(self, observation_space, action_space, name='my_fancy_agent'):
        super().__init__(name=name)
        
        # Observation encoder (camera + lidar -> 128-dim latent)
        self.encoder = EncodeObservation()
        
        # Extract action space size
        self.action_dim = action_space.n
        
        # Import initialize from transforms
        from src.transforms import initialize
        
        # Policy network (actor) - outputs action logits
        self.actor = nn.Sequential(
            initialize(nn.Linear(128, 64)),
            nn.Tanh(),
            initialize(nn.Linear(64, self.action_dim)),
        )
        
        # Value network (critic) - estimates state value V(s)
        self.critic = nn.Sequential(
            initialize(nn.Linear(128, 64)),
            nn.Tanh(),
            initialize(nn.Linear(64, 1)),
        )
    
    def get_action(self, x):
        '''
        Sample an action from the policy for inference/deployment.
        
        Args:
            x: Raw observation (numpy array or torch tensor)
                Shape: [obs_dim] or [batch_size, obs_dim]
        
        Returns:
            action: Integer action index or tensor [batch_size]
        '''
        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(
                x, dtype=torch.float32, device=next(self.parameters()).device
            )
        
        # Add batch dimension if single observation
        single_obs = x.dim() == 1
        if single_obs:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            # Encode observation
            encoded = self.encoder(x)
            
            # Get action logits
            logits = self.actor(encoded)
            
            # Sample action from categorical distribution
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
        
        # Return as Python int if single observation, else tensor
        if single_obs:
            return action.item()
        return action
    
    def get_value(self, x):
        '''
        Estimate the value V(s) of a given state.
        
        Args:
            x: Raw observation
                Shape: [batch_size, obs_dim] or [obs_dim]
        
        Returns:
            value: Estimated state value [batch_size, 1] or [1]
        '''
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(
                x, dtype=torch.float32, device=next(self.parameters()).device
            )
        
        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Encode and estimate value
        encoded = self.encoder(x)
        value = self.critic(encoded)
        
        return value
    
    def get_action_and_value(self, x, action=None):
        '''
        Get action and value simultaneously for training efficiency.
        Encodes observation once and uses for both actor and critic.
        
        Args:
            x: Batched observations [batch_size, obs_dim]
            action: Optional action tensor for computing log_prob
                If None: sample new action
                If provided: compute log_prob of given action
        
        Returns:
            action: Sampled or provided action [batch_size]
            log_prob: Log probability of action [batch_size]  
            entropy: Policy entropy for exploration bonus [scalar]
            value: State value estimate [batch_size, 1]
        '''
        # Ensure tensor + device (consistent with get_action/get_value)
        if isinstance(x, np.ndarray):
            x = torch.tensor(
                x, dtype=torch.float32, device=next(self.parameters()).device
            )
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Encode observation (single pass for efficiency)
        encoded = self.encoder(x)
        
        # Actor: Get action distribution
        logits = self.actor(encoded)
        distribution = torch.distributions.Categorical(logits=logits)
        
        # Sample action if not provided, otherwise use given action
        if action is None:
            action = distribution.sample()
        
        # Compute log probability of action
        log_prob = distribution.log_prob(action)
        
        # Compute entropy for exploration bonus
        entropy = distribution.entropy()
        
        # Critic: Get value estimate
        value = self.critic(encoded)
        
        return action, log_prob, entropy, value