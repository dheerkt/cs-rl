"""
Reward shaping for Overcooked with agent-index swap correction

CRITICAL: Agents are randomly assigned to starting positions each episode.
We must swap shaped rewards to match the correct agent.
"""

import numpy as np
from configs.hyperparameters import HyperParams


class RewardShaper:
    """
    Handles reward shaping with proper agent-index correction
    """

    def __init__(self, env, shape_weights, layout_name):
        """
        Initialize reward shaper

        Args:
            env: Overcooked environment
            shape_weights: Dictionary of shaping reward weights
            layout_name: Name of the layout (for swap detection)
        """
        self.env = env
        self.shape_weights = shape_weights
        self.layout_name = layout_name

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
        # CRITICAL FIX: Use canonical start positions to detect swaps
        if self.layout_name not in HyperParams.LAYOUT_START_POSITIONS:
            print(f"Warning: No canonical start positions found for {self.layout_name}. Assuming no swap.")
            return False

        # Get the canonical "first" start position for this layout
        canonical_pos_0 = HyperParams.LAYOUT_START_POSITIONS[self.layout_name][0]

        # Get the actual position agent 0 spawned at
        # self.initial_positions was already set in reset()
        actual_pos_0 = self.initial_positions[0]

        # If agent 0's actual start pos is not the canonical start pos, they were swapped.
        swapped = (actual_pos_0 != canonical_pos_0)

        if swapped:
            # Verify they are at the *other* start position, just to be safe
            canonical_pos_1 = HyperParams.LAYOUT_START_POSITIONS[self.layout_name][1]
            if actual_pos_0 != canonical_pos_1:
                print(f"Warning: Agent 0 at {actual_pos_0}, not at either canonical start pos.")
                return False  # Fallback

        return swapped

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
        """Check if agent just placed an onion in a pot (IMPROVED with state diff)"""
        # Agent was holding onion, now not holding it
        if prev_obj is None or current_obj is not None:
            return False

        if prev_obj.name != 'onion':
            return False

        # IMPROVED: Actually compare pot states before/after
        prev_pots = self._get_pot_states(self.prev_state)
        current_pots = self._get_pot_states(state)

        # Check if any pot gained an ingredient
        for pot_pos in current_pots:
            if pot_pos not in prev_pots:
                continue

            prev_num = prev_pots[pot_pos].get('num_items', 0)
            curr_num = current_pots[pot_pos].get('num_items', 0)

            # Pot gained exactly 1 ingredient
            if curr_num == prev_num + 1:
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
        Extract pot states from environment state (IMPROVED)

        Returns:
            Dictionary mapping pot positions to pot info
        """
        pots = {}

        # Iterate through objects in state
        if hasattr(state, 'objects'):
            for pos, obj in state.objects.items():
                # Check if this is a pot/soup object
                obj_name = getattr(obj, 'name', '')
                if 'soup' in str(obj_name).lower() or 'pot' in str(obj_name).lower():
                    # Extract detailed pot state
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
                        'position': pos,
                    }

        return pots


def get_default_shaping_weights():
    """Return default shaping weights"""
    from configs.hyperparameters import HyperParams
    return HyperParams.get_shaped_reward_weights(progress=0.0)
