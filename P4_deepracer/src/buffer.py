import torch


class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO training.
    
    Storage layout: [num_steps, num_envs, *feature_shape]
    Flattens to: [num_steps * num_envs, *feature_shape] for minibatch sampling
    """
    
    def __init__(self, num_steps, num_envs, obs_dim, device):
        """
        Initialize rollout buffer.
        
        Args:
            num_steps: Steps per rollout per env (e.g., 512)
            num_envs: Number of parallel environments (e.g., 4)
            obs_dim: Flattened observation dimension
            device: torch device (cpu or cuda)
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = device
        self.total_samples = num_steps * num_envs
        
        # Storage: [num_steps, num_envs, *]
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        
        # Computed during GAE
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        
        # Tracking
        self.step_idx = 0
    
    def add(self, obs, action, log_prob, reward, done, value):
        """
        Add a single step of data from all envs.
        
        Args:
            obs: [num_envs, obs_dim] or [obs_dim] if num_envs=1
            action: [num_envs] or scalar
            log_prob: [num_envs] or scalar
            reward: [num_envs] or scalar
            done: [num_envs] or scalar (bool)
            value: [num_envs, 1] or [num_envs] or scalar
        """
        # Ensure batch dimension for single-env case
        if self.num_envs == 1:
            if isinstance(obs, torch.Tensor) and obs.dim() == 1:
                obs = obs.unsqueeze(0)
            
            if isinstance(action, (int, float)):
                action = torch.tensor([action], dtype=torch.long, device=self.device)
            elif isinstance(action, torch.Tensor) and action.dim() == 0:
                action = action.unsqueeze(0)
            
            if isinstance(log_prob, torch.Tensor) and log_prob.dim() == 0:
                log_prob = log_prob.unsqueeze(0)
            
            if not isinstance(reward, torch.Tensor):
                reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
            elif reward.dim() == 0:
                reward = reward.unsqueeze(0)
            
            if not isinstance(done, torch.Tensor):
                done = torch.tensor([done], dtype=torch.bool, device=self.device)
            elif done.dim() == 0:
                done = done.unsqueeze(0)
            
            if isinstance(value, torch.Tensor) and value.dim() == 0:
                value = value.unsqueeze(0)
        
        # Squeeze value if [num_envs, 1] -> [num_envs]
        if isinstance(value, torch.Tensor) and value.dim() == 2:
            value = value.squeeze(-1)
        
        # Store at current step
        self.obs[self.step_idx] = obs
        self.actions[self.step_idx] = action
        self.log_probs[self.step_idx] = log_prob
        self.rewards[self.step_idx] = reward
        self.dones[self.step_idx] = done
        self.values[self.step_idx] = value
        
        self.step_idx += 1
    
    def compute_gae(self, next_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            next_value: Value estimate for state after last step [num_envs] or [num_envs, 1]
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        # Squeeze next_value if needed
        if isinstance(next_value, torch.Tensor) and next_value.dim() == 2:
            next_value = next_value.squeeze(-1)
        
        # Bootstrap from next_value
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0
        
        # Backward pass through time
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_val = self.values[t + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            
            # GAE: A_t = δ_t + γ * λ * (1 - done) * A_{t+1}
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        self.advantages = advantages
        self.returns = advantages + self.values
    
    def get_batches(self, batch_size):
        """
        Generate random minibatches for training.
        
        Args:
            batch_size: Size of each minibatch (e.g., 64)
        
        Yields:
            Dictionary with minibatch data
        """
        # Flatten: [num_steps, num_envs, *] -> [total_samples, *]
        obs_flat = self.obs.reshape(-1, self.obs_dim)
        actions_flat = self.actions.reshape(-1)
        log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)
        
        # Normalize advantages (important for PPO stability)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # Random permutation
        indices = torch.randperm(self.total_samples, device=self.device)
        
        # Yield minibatches
        for start_idx in range(0, self.total_samples, batch_size):
            end_idx = min(start_idx + batch_size, self.total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'obs': obs_flat[batch_indices],
                'actions': actions_flat[batch_indices],
                'log_probs': log_probs_flat[batch_indices],
                'advantages': advantages_flat[batch_indices],
                'returns': returns_flat[batch_indices],
            }
    
    def reset(self):
        """Reset buffer for next rollout."""
        self.step_idx = 0
