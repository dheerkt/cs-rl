# reward_shaping.py
"""
Reward shaping for Overcooked - Team-based pot events, per-agent pickups/drops.
"""

from dataclasses import dataclass
import numpy as np
from configs.hyperparameters import HyperParams


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class RewardShaper:
    env: object
    shape_weights: dict
    layout_name: str

    def __post_init__(self):
        self.initial_positions = None
        self.prev_state = None
        self.prev_pots = {}
        self.swapped = False
        self.event_counts = {
            "onion_in_pot": 0,
            "cooking_start": 0,
            "soup_pickup": 0,
            "soup_ready": 0,
            "correct_delivery": 0,
            "penalty": 0,
        }

    def reset(self, state):
        self.initial_positions = tuple(p.position for p in state.players)
        self.swapped = self._detect_swap_at_reset(state)
        self.prev_pots = self._get_pot_states(state)
        self.prev_state = state
        for k in self.event_counts:
            self.event_counts[k] = 0

    def compute_shaped_rewards(self, state, sparse_rewards, done):
        if self.prev_state is None:
            self.prev_pots = self._get_pot_states(state)
            self.prev_state = state
            return self._coerce_rewards(sparse_rewards), {}

        base = self._coerce_rewards(sparse_rewards)
        shaped = base[:]
        info = {
            "agent0_onion_in_pot": 0.0,
            "agent0_cooking_start": 0.0,
            "agent0_soup_pickup": 0.0,
            "agent0_soup_ready": 0.0,
            "agent0_correct_delivery": 0.0,
            "agent0_penalty": 0.0,
            "agent1_onion_in_pot": 0.0,
            "agent1_cooking_start": 0.0,
            "agent1_soup_pickup": 0.0,
            "agent1_soup_ready": 0.0,
            "agent1_correct_delivery": 0.0,
            "agent1_penalty": 0.0,
        }

        # Compute pot events once per step
        curr_pots = self._get_pot_states(state)
        pot_events = self._diff_pots_dict(self.prev_pots, curr_pots)

        for agent_id in range(2):
            s = self._compute_agent_shaping(agent_id, state, pot_events)
            shaped[agent_id] += sum(s.values())
            prefix = f"agent{agent_id}_"
            for k, v in s.items():
                info[prefix + k] = float(v)
                if v != 0:
                    self.event_counts[k] = self.event_counts.get(k, 0) + 1

        if self.swapped:
            shaped = shaped[::-1]

        self.prev_pots = curr_pots
        self.prev_state = state
        return shaped, {"shaping_events": self.event_counts.copy()}

    def _detect_swap_at_reset(self, state):
        if self.layout_name not in HyperParams.LAYOUT_START_POSITIONS:
            return False
        canon0, canon1 = HyperParams.LAYOUT_START_POSITIONS[self.layout_name]
        actual0 = self.initial_positions[0]
        return actual0 == canon1

    def _get_pot_states(self, state):
        pots = {}
        objs = getattr(state, "objects", None)
        if objs is None:
            try:
                pot_states = self.env.mdp.get_pot_states(state)
                for pos, ps in pot_states.items():
                    pots[pos] = {
                        "num_items": int(getattr(ps, "num_items", 0)),
                        "is_cooking": bool(getattr(ps, "is_cooking", False)),
                        "is_ready": bool(getattr(ps, "is_ready", False)),
                    }
                return pots
            except Exception:
                return {}
        iterable = objs.items() if hasattr(objs, "items") else enumerate(objs)
        for key, obj in iterable:
            if obj is None:
                continue
            pos = getattr(obj, "position", key if isinstance(key, tuple) else None)
            if pos is None:
                continue
            name = getattr(obj, "name", "")
            if "soup" not in name.lower():
                continue
            num = getattr(obj, "num_items", None)
            if num is None:
                ing = getattr(obj, "_ingredients", getattr(obj, "ingredients", []))
                num = len(ing)
            pots[pos] = {
                "num_items": int(num),
                "is_cooking": bool(
                    getattr(obj, "is_cooking", getattr(obj, "_cooking", False))
                ),
                "is_ready": bool(
                    getattr(obj, "is_ready", getattr(obj, "_ready", False))
                ),
            }
        return pots

    def _diff_pots_dict(self, prev, curr):
        events = {"onion_added": 0, "cooking_started": 0, "soup_ready": 0}
        for pos in curr.keys():
            if pos not in prev:
                continue
            p, c = prev[pos], curr[pos]
            if c["num_items"] == p["num_items"] + 1:
                events["onion_added"] += 1
            if not p["is_cooking"] and c["is_cooking"] and c["num_items"] >= 3:
                events["cooking_started"] += 1
            if not p["is_ready"] and c["is_ready"]:
                events["soup_ready"] += 1
        return events

    def _serving_positions(self):
        pos = []
        terrain = getattr(self.env.mdp, "terrain_mtx", None)
        if terrain:
            for y, row in enumerate(terrain):
                for x, cell in enumerate(row):
                    if cell == "S":
                        pos.append((x, y))
        return pos

    def _correct_delivery_event(self, state):
        served_curr = getattr(state, "served", None)
        served_prev = getattr(self.prev_state, "served", None)
        if isinstance(served_curr, int) and isinstance(served_prev, int):
            return int(served_curr == served_prev + 1)
        serving = self._serving_positions()
        for aid in range(2):
            p, pp = state.players[aid], self.prev_state.players[aid]
            if (
                pp.held_object
                and getattr(pp.held_object, "name", "") == "soup"
                and p.held_object is None
            ):
                if any(_manhattan(p.position, s) == 1 for s in serving):
                    return 1
        return 0

    def _waste_event(self, state):
        served_curr = getattr(state, "served", None)
        served_prev = getattr(self.prev_state, "served", None)
        if (
            isinstance(served_curr, int)
            and isinstance(served_prev, int)
            and served_curr == served_prev + 1
        ):
            return 0
        serving = self._serving_positions()
        for aid in range(2):
            p, pp = state.players[aid], self.prev_state.players[aid]
            if (
                pp.held_object
                and getattr(pp.held_object, "name", "") == "soup"
                and p.held_object is None
            ):
                return 0 if any(_manhattan(p.position, s) == 1 for s in serving) else 1
        return 0

    def _compute_agent_shaping(self, agent_id, state, pot_events):
        w = self.shape_weights
        s = {
            "onion_in_pot": 0.0,
            "cooking_start": 0.0,
            "soup_pickup": 0.0,
            "soup_ready": 0.0,
            "correct_delivery": 0.0,
            "penalty": 0.0,
        }
        player, prev_player = state.players[agent_id], self.prev_state.players[agent_id]
        curr_obj, prev_obj = player.held_object, prev_player.held_object

        # Team pot events
        if pot_events["onion_added"] > 0:
            s["onion_in_pot"] = float(w.get("onion_in_pot", 0.01))
        if pot_events["cooking_started"] > 0:
            s["cooking_start"] = float(w.get("cooking_start", 0.05))
        if pot_events["soup_ready"] > 0:
            s["soup_ready"] = float(w.get("soup_ready", 0.10))

        # Per-agent pickup/delivery/penalty
        if prev_obj is None and curr_obj and getattr(curr_obj, "name", "") == "soup":
            s["soup_pickup"] = float(w.get("soup_pickup", 0.02))
        if self._correct_delivery_event(state):
            s["correct_delivery"] = float(w.get("correct_delivery", 0.25))
        if self._waste_event(state):
            s["penalty"] = -float(w.get("penalty_drop", 0.05))

        return s

    @staticmethod
    def _coerce_rewards(sparse_rewards):
        if isinstance(sparse_rewards, (int, float)):
            r = float(sparse_rewards)
            return [r, r]
        if isinstance(sparse_rewards, (list, tuple, np.ndarray)):
            if len(sparse_rewards) == 1:
                r = float(sparse_rewards[0])
                return [r, r]
            return [float(sparse_rewards[0]), float(sparse_rewards[1])]
        return [0.0, 0.0]


def get_default_shaping_weights():
    return HyperParams.get_shaped_reward_weights(progress=0.0)
