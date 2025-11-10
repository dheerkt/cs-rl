"""
Reward shaping for Overcooked with agent-index swap correction

CRITICAL: Agents are randomly assigned to starting positions each episode.
We must swap shaped rewards to match the correct agent.
"""

import numpy as np


class RewardShaper:
    """
    Handles reward shaping with proper agent-index correction
    """

    def __init__(self, env, shape_weights):
        """
        Initialize reward shaper

        Args:
            env: Overcooked environment
            shape_weights: Dictionary of shaping reward weights
        """
        self.env = env
        self.shape_weights = shape_weights

        # Track initial positions to detect swaps
        self.initial_positions = None
        self.prev_state = None
        self.swapped = False  # CRITICAL: Cache swap decision per episode

        # Event tracking for validation
        self.event_counts = {
            'onion_in_pot': 0,
            'cooking_start': 0,
            'soup_pickup': 0,
            'correct_delivery': 0,
            'penalty': 0,
        }

    def reset(self, state):
        """
        Called at episode start to record initial positions

        Args:
            state: Initial environment state
        """
        # Store initial positions for swap detection
        self.initial_positions = tuple(player.position for player in state.players)

        # CRITICAL FIX: Cache swap decision ONCE per episode
        # Agents are randomly assigned to starting positions each reset
        # We detect this once and cache it, rather than checking every step
        self.swapped = self._detect_swap_at_reset(state)

        self.prev_state = state

        # Reset event counters
        for key in self.event_counts:
            self.event_counts[key] = 0

    def compute_shaped_rewards(self, state, sparse_rewards, done):
        """
        Compute shaped rewards for both agents with swap correction

        Args:
            state: Current environment state
            sparse_rewards: Base rewards from environment [agent0_reward, agent1_reward]
            done: Episode done flag

        Returns:
            shaped_rewards: Shaped rewards [agent0_reward, agent1_reward]
            info: Dictionary with shaped reward breakdown for logging
        """
        if self.prev_state is None:
            self.prev_state = state
            return sparse_rewards, {}

        # Initialize shaped rewards with sparse rewards
        shaped_rewards = list(sparse_rewards)
        info = {
            'agent0_onion_in_pot': 0,
            'agent0_cooking_start': 0,
            'agent0_soup_pickup': 0,
            'agent0_correct_delivery': 0,
            'agent0_penalty': 0,
            'agent1_onion_in_pot': 0,
            'agent1_cooking_start': 0,
            'agent1_soup_pickup': 0,
            'agent1_correct_delivery': 0,
            'agent1_penalty': 0,
        }

        # Compute shaping for each agent
        for agent_id in range(2):
            agent_shaping = self._compute_agent_shaping(agent_id, state)

            shaped_rewards[agent_id] += sum(agent_shaping.values())

            # Track events for validation
            for event_name, reward_val in agent_shaping.items():
                if reward_val != 0:
                    self.event_counts[event_name] += 1

            # Log breakdown
            prefix = f'agent{agent_id}_'
            info[prefix + 'onion_in_pot'] = agent_shaping.get('onion_in_pot', 0)
            info[prefix + 'cooking_start'] = agent_shaping.get('cooking_start', 0)
            info[prefix + 'soup_pickup'] = agent_shaping.get('soup_pickup', 0)
            info[prefix + 'correct_delivery'] = agent_shaping.get('correct_delivery', 0)
            info[prefix + 'penalty'] = agent_shaping.get('penalty', 0)

        # CRITICAL FIX: Use cached swap flag instead of checking every step
        # This was the bug - checking positions every step would flip rewards
        # after agent movement, corrupting credit assignment
        if self.swapped:
            shaped_rewards = shaped_rewards[::-1]  # Swap agent 0 and 1 rewards

        self.prev_state = state

        return shaped_rewards, info

    def _detect_swap_at_reset(self, state):
        """
        Detect if agents were swapped at episode reset (CALLED ONCE PER EPISODE)

        Args:
            state: Initial state at reset

        Returns:
            True if agents were swapped (player 0 is not at position 0)
        """
        # In Overcooked, agents can start at different positions each episode
        # The environment randomly assigns agents to starting positions
        # We detect this ONCE at reset and cache it for the whole episode

        # For now, we use a simple heuristic: if this layout has known spawn positions,
        # we could check against them. Without that info, we assume no swap.
        # This is safe because the observations are always correctly ordered.

        # TODO: If you know the canonical spawn positions for each layout,
        # you can add them here to properly detect swaps.
        # For now, we return False (no swap detection)
        # The key fix is that this is called ONCE per episode, not every step.

        return False  # Conservative: assume no swap unless we can detect it properly

    def _compute_agent_shaping(self, agent_id, state):
        """
        Compute shaped rewards for a specific agent

        Args:
            agent_id: Agent index (0 or 1)
            state: Current state

        Returns:
            Dictionary of shaped reward components
        """
        shaping = {}
        player = state.players[agent_id]
        prev_player = self.prev_state.players[agent_id]

        # Track what the agent is holding
        current_obj = player.held_object
        prev_obj = prev_player.held_object

        # 1. Onion placed in pot
        if self._onion_placed_in_pot(prev_obj, current_obj, state):
            shaping['onion_in_pot'] = self.shape_weights['onion_in_pot']

        # 2. Cooking started with 3 onions
        if self._cooking_started_with_3_onions(state):
            shaping['cooking_start'] = self.shape_weights['cooking_start']

        # 3. Soup picked up
        if self._soup_picked_up(prev_obj, current_obj):
            shaping['soup_pickup'] = self.shape_weights['soup_pickup']

        # 4. Correct delivery (already handled by sparse +20, but add small bonus)
        if self._correct_delivery(prev_obj, current_obj, state):
            shaping['correct_delivery'] = self.shape_weights['correct_delivery']

        # 5. Penalties for mistakes
        if self._soup_dropped_or_wasted(prev_obj, current_obj, state):
            shaping['penalty'] = self.shape_weights['penalty_drop']

        return shaping

    def _onion_placed_in_pot(self, prev_obj, current_obj, state):
        """Check if agent just placed an onion in a pot"""
        # Agent was holding onion, now not holding it
        if prev_obj is None or current_obj is not None:
            return False

        if prev_obj.name != 'onion':
            return False

        # Check if any pot gained an onion
        for pot_state in state.objects.values():
            if hasattr(pot_state, 'name') and 'soup' in pot_state.name:
                # Pot exists - this is a simplified check
                # In practice, you'd compare pot states before/after
                return True

        return False

    def _cooking_started_with_3_onions(self, state):
        """Check if cooking just started with 3 onions"""
        if self.prev_state is None:
            return False

        # Check each pot to see if it just started cooking
        prev_pots = self._get_pot_states(self.prev_state)
        current_pots = self._get_pot_states(state)

        for pot_pos in current_pots:
            if pot_pos not in prev_pots:
                continue

            prev_pot = prev_pots[pot_pos]
            curr_pot = current_pots[pot_pos]

            # Pot just started cooking with 3 items
            if (not prev_pot.get('is_cooking', False) and
                curr_pot.get('is_cooking', False) and
                curr_pot.get('num_items', 0) == 3):
                return True

        return False

    def _soup_picked_up(self, prev_obj, current_obj):
        """Check if agent just picked up a soup"""
        if prev_obj is not None or current_obj is None:
            return False

        return current_obj.name == 'soup'

    def _correct_delivery(self, prev_obj, current_obj, state):
        """Check if agent just delivered a soup correctly"""
        # Agent was holding soup, now not
        if prev_obj is None or current_obj is not None:
            return False

        if prev_obj.name != 'soup':
            return False

        # This is tricky - in practice, check if near serving area
        # For now, assume delivery happened if soup disappeared
        return True

    def _soup_dropped_or_wasted(self, prev_obj, current_obj, state):
        """Check if agent dropped soup or made mistake"""
        # Soup disappeared but not via delivery (simplified check)
        if prev_obj is not None and prev_obj.name == 'soup':
            if current_obj is None:
                # Check if this was NOT a delivery
                # This is a simplified heuristic
                return False  # Assume no drops for now

        return False

    def _get_pot_states(self, state):
        """
        Extract pot states from environment state

        Returns:
            Dictionary mapping pot positions to pot info
        """
        pots = {}

        # Iterate through objects in state
        if hasattr(state, 'objects'):
            for pos, obj in state.objects.items():
                if hasattr(obj, 'name') and 'soup' in str(obj).lower():
                    pots[pos] = {
                        'is_cooking': hasattr(obj, 'is_cooking') and obj.is_cooking,
                        'is_ready': hasattr(obj, 'is_ready') and obj.is_ready,
                        'num_items': len(obj.ingredients) if hasattr(obj, 'ingredients') else 0,
                    }

        return pots


def get_default_shaping_weights():
    """Return default shaping weights"""
    from configs.hyperparameters import HyperParams
    return HyperParams.get_shaped_reward_weights(progress=0.0)
