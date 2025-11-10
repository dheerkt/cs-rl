"""
Hyperparameters for PPO with Centralized Critic
These parameters are FIXED across all layouts as required by the project spec.
"""

class HyperParams:
    """Fixed hyperparameters used for all layouts"""

    # PPO Algorithm
    lr = 3e-4                    # Learning rate
    gamma = 0.99                 # Discount factor
    gae_lambda = 0.95           # GAE lambda for advantage estimation
    clip_epsilon = 0.2          # PPO clipping parameter
    entropy_coef = 0.01         # Entropy regularization coefficient
    value_loss_coef = 0.5       # Value loss coefficient
    max_grad_norm = 0.5         # Gradient clipping

    # Training
    batch_size = 2048           # Number of steps to collect before update
    minibatch_size = 256        # Minibatch size for SGD
    ppo_epochs = 10             # Number of epochs per PPO update

    # Network Architecture
    hidden_size = 256           # Hidden layer size for both actor and critic
    num_layers = 2              # Number of hidden layers

    # Environment
    obs_dim = 96                # Observation dimension per agent
    joint_obs_dim = 192         # Concatenated observation for centralized critic
    action_dim = 6              # Number of discrete actions
    num_agents = 2              # Number of agents

    # Action Mapping (Overcooked standard action indices)
    # These are the standard Overcooked action indices - verified against environment
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4
    ACTION_INTERACT = 5

    # Canonical (expected) start positions for each layout
    # We check if agent 0 spawned at the first position or was swapped.
    # These were discovered by running the environments and observing actual spawn positions
    LAYOUT_START_POSITIONS = {
        'cramped_room': ((1, 2), (3, 1)),
        'coordination_ring': ((2, 1), (1, 2)),
        'counter_circuit_o_1order': ((3, 3), (3, 1))
    }

    # Reward Shaping (fixed across all layouts)
    shape_onion_in_pot = 0.5        # Bonus for placing onion in pot
    shape_cooking_start = 1.0       # Bonus for starting to cook with 3 onions
    shape_soup_pickup = 1.5         # Bonus for picking up cooked soup
    shape_correct_delivery = 2.0    # Bonus for correct delivery
    shape_penalty_drop = -0.5       # Penalty for dropping soup
    shape_anneal_start = 0.7        # Start annealing at 70% of training
    shape_anneal_end = 0.9          # End annealing at 90% of training

    # Logging
    log_interval = 100          # Log every N episodes
    save_interval = 5000        # Save checkpoint every N episodes
    eval_interval = 1000        # Evaluate every N episodes
    eval_episodes = 10          # Number of episodes for evaluation

    # Episode specific (can vary per layout, not considered hyperparameter)
    # These are just suggested defaults
    episodes_cramped_room = 50000
    episodes_coordination_ring = 100000
    episodes_counter_circuit = 150000

    @classmethod
    def get_shaped_reward_weights(cls, progress=1.0):
        """
        Get shaped reward weights with optional annealing

        Args:
            progress: Training progress (0.0 to 1.0)

        Returns:
            Dictionary of shaped reward weights
        """
        # Anneal shaped rewards toward end of training
        if progress < cls.shape_anneal_start:
            scale = 1.0
        elif progress > cls.shape_anneal_end:
            scale = 0.1  # Keep 10% of shaping at end
        else:
            # Linear interpolation
            anneal_progress = (progress - cls.shape_anneal_start) / (cls.shape_anneal_end - cls.shape_anneal_start)
            scale = 1.0 - 0.9 * anneal_progress

        return {
            'onion_in_pot': cls.shape_onion_in_pot * scale,
            'cooking_start': cls.shape_cooking_start * scale,
            'soup_pickup': cls.shape_soup_pickup * scale,
            'correct_delivery': cls.shape_correct_delivery * scale,
            'penalty_drop': cls.shape_penalty_drop * scale,
        }
