# configs/hyperparameters.py
"""
Hyperparameters for PPO with Centralized Critic.
These parameters are FIXED across all layouts per project requirements.
"""


class HyperParams:
    """Fixed hyperparameters used for all layouts."""

    # PPO Algorithm
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5

    # Training
    batch_size = 1024  # Reduced from 2048 for faster early updates
    minibatch_size = 256
    ppo_epochs = 10

    # Network Architecture
    hidden_size = 256
    num_layers = 2

    # Environment
    obs_dim = 96
    joint_obs_dim = 192
    action_dim = 6
    num_agents = 2

    # Action Mapping (Overcooked standard indices)
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4
    ACTION_INTERACT = 5

    # Canonical start positions per layout for swap detection
    LAYOUT_START_POSITIONS = {
        "cramped_room": ((1, 2), (3, 1)),
        "coordination_ring": ((2, 1), (1, 2)),
        "counter_circuit_o_1order": ((3, 3), (3, 1)),
    }

    # Reward Shaping (boosted to provide stronger gradients vs sparse +20)
    shape_onion_in_pot = 0.05  # was 0.5, now 0.05 to stay << +20
    shape_cooking_start = 0.15  # was 1.0, now 0.15
    shape_soup_ready = 0.30  # new milestone event
    shape_soup_pickup = 0.10  # was 1.5, now 0.10
    shape_correct_delivery = 0.50  # was 2.0, now 0.50
    shape_penalty_drop = 0.10  # was 0.5, now 0.10
    shape_anneal_start = 0.7
    shape_anneal_end = 0.9

    # Logging
    log_interval = 100
    save_interval = 5000
    eval_interval = 1000
    eval_episodes = 10

    # Episode budgets (suggested defaults)
    episodes_cramped_room = 50000
    episodes_coordination_ring = 100000
    episodes_counter_circuit = 150000

    @classmethod
    def get_shaped_reward_weights(cls, progress=0.0):
        """
        Get shaped reward weights with optional annealing.

        Args:
            progress: Training progress (0.0 to 1.0)

        Returns:
            Dictionary of shaped reward weights.
        """
        if progress < cls.shape_anneal_start:
            scale = 1.0
        elif progress > cls.shape_anneal_end:
            scale = 0.1
        else:
            anneal_progress = (progress - cls.shape_anneal_start) / (
                cls.shape_anneal_end - cls.shape_anneal_start
            )
            scale = 1.0 - 0.9 * anneal_progress

        return {
            "onion_in_pot": cls.shape_onion_in_pot * scale,
            "cooking_start": cls.shape_cooking_start * scale,
            "soup_ready": cls.shape_soup_ready * scale,
            "soup_pickup": cls.shape_soup_pickup * scale,
            "correct_delivery": cls.shape_correct_delivery * scale,
            "penalty_drop": cls.shape_penalty_drop * scale,
        }
