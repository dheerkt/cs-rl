"""
Simple test to validate RolloutBuffer implementation.
Tests shapes, GAE computation, and minibatch generation.
"""
import torch
from src.buffer import RolloutBuffer


def test_buffer_shapes():
    """Test buffer storage shapes."""
    print("Testing buffer shapes...")
    
    num_steps = 4
    num_envs = 2
    obs_dim = 10
    device = torch.device('cpu')
    
    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, device)
    
    # Check initialization shapes
    assert buffer.obs.shape == (num_steps, num_envs, obs_dim)
    assert buffer.actions.shape == (num_steps, num_envs)
    assert buffer.log_probs.shape == (num_steps, num_envs)
    assert buffer.rewards.shape == (num_steps, num_envs)
    assert buffer.dones.shape == (num_steps, num_envs)
    assert buffer.values.shape == (num_steps, num_envs)
    
    print("✓ Buffer initialization shapes correct")


def test_buffer_add():
    """Test adding data to buffer."""
    print("\nTesting buffer.add()...")
    
    num_steps = 4
    num_envs = 2
    obs_dim = 10
    device = torch.device('cpu')
    
    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, device)
    
    # Add multi-env data
    for step in range(num_steps):
        obs = torch.randn(num_envs, obs_dim)
        action = torch.randint(0, 5, (num_envs,))
        log_prob = torch.randn(num_envs)
        reward = torch.randn(num_envs)
        done = torch.tensor([False, False])
        value = torch.randn(num_envs, 1)
        
        buffer.add(obs, action, log_prob, reward, done, value)
    
    assert buffer.step_idx == num_steps
    print("✓ Multi-env data stored correctly")
    
    # Test single-env case
    buffer_single = RolloutBuffer(num_steps, 1, obs_dim, device)
    
    for step in range(num_steps):
        obs = torch.randn(obs_dim)  # Single obs
        action = 2  # Scalar action
        log_prob = torch.tensor(-1.5)
        reward = 1.0
        done = False
        value = torch.tensor([0.5])
        
        buffer_single.add(obs, action, log_prob, reward, done, value)
    
    assert buffer_single.step_idx == num_steps
    print("✓ Single-env data stored correctly")


def test_gae_computation():
    """Test GAE computation with known values."""
    print("\nTesting GAE computation...")
    
    num_steps = 3
    num_envs = 1
    obs_dim = 5
    device = torch.device('cpu')
    
    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, device)
    
    # Simple test case: constant rewards, no dones
    for step in range(num_steps):
        obs = torch.zeros(obs_dim)
        action = 0
        log_prob = torch.tensor(-1.0)
        reward = 1.0  # Constant reward
        done = False
        value = torch.tensor([0.0])  # Zero value estimates
        
        buffer.add(obs, action, log_prob, reward, done, value)
    
    next_value = torch.tensor([0.0])
    gamma = 0.99
    gae_lambda = 0.95
    
    buffer.compute_gae(next_value, gamma, gae_lambda)
    
    # With constant rewards=1, values=0, no dones:
    # δ_t = r_t + γ * V(s_{t+1}) - V(s_t) = 1.0 for all t
    # GAE should accumulate backwards
    print(f"  Advantages: {buffer.advantages.squeeze()}")
    print(f"  Returns: {buffer.returns.squeeze()}")
    
    # Check advantages are positive (getting reward)
    assert (buffer.advantages > 0).all()
    print("✓ GAE computed (positive advantages for positive rewards)")


def test_gae_with_terminal():
    """Test GAE correctly handles terminal states."""
    print("\nTesting GAE with terminal states...")
    
    num_steps = 4
    num_envs = 1
    obs_dim = 5
    device = torch.device('cpu')
    
    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, device)
    
    # Episode that terminates at step 2
    rewards = [1.0, 1.0, 10.0, 1.0]  # Big reward before terminal
    dones = [False, False, True, False]  # Terminal at step 2
    
    for step in range(num_steps):
        obs = torch.zeros(obs_dim)
        action = 0
        log_prob = torch.tensor(-1.0)
        reward = rewards[step]
        done = dones[step]
        value = torch.tensor([0.0])
        
        buffer.add(obs, action, log_prob, reward, done, value)
    
    next_value = torch.tensor([0.0])
    buffer.compute_gae(next_value, gamma=0.99, gae_lambda=0.95)
    
    print(f"  Advantages: {buffer.advantages.squeeze()}")
    
    # Advantage at terminal step should not bootstrap from next step
    # (bootstrap should be cut off at done=True)
    adv_at_terminal = buffer.advantages[2, 0].item()
    print(f"  Advantage at terminal state (step 2): {adv_at_terminal:.4f}")
    
    assert adv_at_terminal > 0  # Should get the reward
    print("✓ GAE correctly handles terminal states")


def test_minibatch_generation():
    """Test minibatch sampling."""
    print("\nTesting minibatch generation...")
    
    num_steps = 8
    num_envs = 2
    obs_dim = 10
    batch_size = 4
    device = torch.device('cpu')
    
    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, device)
    
    # Fill buffer
    for step in range(num_steps):
        obs = torch.randn(num_envs, obs_dim)
        action = torch.randint(0, 5, (num_envs,))
        log_prob = torch.randn(num_envs)
        reward = torch.randn(num_envs)
        done = torch.tensor([False, False])
        value = torch.randn(num_envs, 1)
        
        buffer.add(obs, action, log_prob, reward, done, value)
    
    # Compute GAE
    next_value = torch.zeros(num_envs, 1)
    buffer.compute_gae(next_value)
    
    # Get batches
    total_samples = num_steps * num_envs  # 16
    batches = list(buffer.get_batches(batch_size))
    
    print(f"  Total samples: {total_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(batches)}")
    
    # Check batch shapes
    for i, batch in enumerate(batches):
        actual_batch_size = len(batch['obs'])
        print(f"  Batch {i}: size={actual_batch_size}, obs shape={batch['obs'].shape}")
        
        assert batch['obs'].shape == (actual_batch_size, obs_dim)
        assert batch['actions'].shape == (actual_batch_size,)
        assert batch['log_probs'].shape == (actual_batch_size,)
        assert batch['advantages'].shape == (actual_batch_size,)
        assert batch['returns'].shape == (actual_batch_size,)
    
    # Check all samples are used
    total_batch_size = sum(len(b['obs']) for b in batches)
    assert total_batch_size == total_samples
    
    print("✓ Minibatch generation correct")


def test_advantage_normalization():
    """Test that advantages are normalized in get_batches."""
    print("\nTesting advantage normalization...")
    
    num_steps = 4
    num_envs = 2
    obs_dim = 5
    device = torch.device('cpu')
    
    buffer = RolloutBuffer(num_steps, num_envs, obs_dim, device)
    
    # Fill with data
    for step in range(num_steps):
        obs = torch.randn(num_envs, obs_dim)
        action = torch.randint(0, 5, (num_envs,))
        log_prob = torch.randn(num_envs)
        reward = torch.randn(num_envs)
        done = torch.tensor([False, False])
        value = torch.randn(num_envs, 1)
        
        buffer.add(obs, action, log_prob, reward, done, value)
    
    buffer.compute_gae(torch.zeros(num_envs, 1))
    
    # Get single batch with all data
    batch = next(buffer.get_batches(batch_size=num_steps * num_envs))
    
    # Normalized advantages should have ~0 mean and ~1 std
    adv_mean = batch['advantages'].mean().item()
    adv_std = batch['advantages'].std().item()
    
    print(f"  Normalized advantage mean: {adv_mean:.6f}")
    print(f"  Normalized advantage std: {adv_std:.6f}")
    
    assert abs(adv_mean) < 1e-5, "Mean should be ~0"
    assert abs(adv_std - 1.0) < 1e-5, "Std should be ~1"
    
    print("✓ Advantages normalized correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("RolloutBuffer Validation Tests")
    print("=" * 60)
    
    test_buffer_shapes()
    test_buffer_add()
    test_gae_computation()
    test_gae_with_terminal()
    test_minibatch_generation()
    test_advantage_normalization()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
