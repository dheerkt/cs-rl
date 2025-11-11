# reward_shaping.py
"""
Reward shaping for Overcooked - Exploit-proof with bootstrap gating and spatial guidance.
"""

from dataclasses import dataclass
import numpy as np
from configs.hyperparameters import HyperParams

# Debug print control
DEBUG_POT_EVENTS = False
DEBUG_DELIVERY_EVENTS = False

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class RewardShaper:
    env: object
    shape_weights: dict
    layout_name: str
    episode_count: int = 0  # Track global episode for bootstrap gating

    def __post_init__(self):
        self.initial_positions = None
        self.prev_state = None
        self.prev_pots = {}
        self.swapped = False
        self.event_counts = {
            "onion_in_pot": 0,
            "cooking_start": 0,
            "soup_ready": 0,
            "soup_pickup": 0,
            "correct_delivery": 0,
            "penalty": 0,
        }
        self.soups_delivered_this_episode = 0
        # Soup ID tracking to prevent pickup-drop exploits
        self.picked_up_soups = set()
        self.delivered_soups = set()
        # Pot tracking to prevent repeated inference credits
        self._credited_ready_pots = set()
        # Flicker protection: track pot positions credited this episode
        self._pots_credited_this_episode = set()
        # Approach serving tracking: one-time nudge per carried soup
        self._carried_soup_ids = [None, None]           # Track current soup ID per agent
        self._approach_credit_given = [True, True]      # Whether nudge given for current soup

    def update_weights(self, new_weights, episode):
        """Update shaping weights and episode count for annealing."""
        self.shape_weights = new_weights
        self.episode_count = episode

    def reset(self, state):
        self.initial_positions = tuple(p.position for p in state.players)
        self.swapped = self._detect_swap_at_reset(state)
        self.prev_pots = self._get_pot_states(state)
        self.prev_state = state
        # Reset soup tracking for new episode
        self.picked_up_soups = set()
        self.delivered_soups = set()
        self.soups_delivered_this_episode = 0
        # Reset pot tracking to prevent repeated inference credits
        self._credited_ready_pots.clear()
        # Reset flicker protection for new episode
        self._pots_credited_this_episode.clear()
        # Reset approach serving tracking for new episode
        self._carried_soup_ids = [None, None]
        self._approach_credit_given = [True, True]
        for k in self.event_counts:
            self.event_counts[k] = 0

        # Debug: print serving positions once
        serving = self._serving_positions()
        if not hasattr(self, "_serving_debug_printed"):
            print(f"[RESET_DEBUG] Serving positions: {serving}")
            self._serving_debug_printed = True

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
            "agent0_soup_ready": 0.0,
            "agent0_soup_pickup": 0.0,
            "agent0_correct_delivery": 0.0,
            "agent0_approach_serving": 0.0,
            "agent0_penalty": 0.0,
            "agent1_onion_in_pot": 0.0,
            "agent1_cooking_start": 0.0,
            "agent1_soup_ready": 0.0,
            "agent1_soup_pickup": 0.0,
            "agent1_correct_delivery": 0.0,
            "agent1_approach_serving": 0.0,
            "agent1_penalty": 0.0,
        }

        curr_pots = self._get_pot_states(state)
        pot_events = self._diff_pots_dict(self.prev_pots, curr_pots)
        delivery_occurred = self._correct_delivery_event(state, sparse_rewards)

        # Safety validation: cap onion counts to prevent any remaining exploits
        if pot_events["onion_added"] > 20:
            print(f"[ERROR] Onion inference bug detected: {pot_events['onion_added']} onions in single timestep!")
            print(f"  prev_pots: {self.prev_pots}")
            print(f"  curr_pots: {curr_pots}")
            print(f"  Capping to 3 onions (max per pot)")
            pot_events["onion_added"] = min(pot_events["onion_added"], 3)
        elif pot_events["onion_added"] > 10:
            print(f"[WARNING] High onion count: {pot_events['onion_added']} in single timestep")

        # Track actual event magnitudes (not per-timestep occurrences)
        self.event_counts["onion_in_pot"] += pot_events["onion_added"]
        self.event_counts["cooking_start"] += pot_events["cooking_started"]
        self.event_counts["soup_ready"] += pot_events["soup_ready"]

        # Track per-agent pickup events (count actual pickups, not timesteps)
        # Also track soup IDs for approach_serving credit
        for agent_id in range(2):
            p = state.players[agent_id]
            pp = self.prev_state.players[agent_id]
            prev_held = getattr(pp.held_object, "name", None) if pp.held_object else None
            curr_held = getattr(p.held_object, "name", None) if p.held_object else None
            
            if prev_held is None and curr_held == "soup":
                soup_id = id(p.held_object) if p.held_object else None
                if soup_id and soup_id not in self.picked_up_soups:
                    self.event_counts["soup_pickup"] += 1
                # Track soup ID and enable approach nudge
                self._carried_soup_ids[agent_id] = soup_id
                self._approach_credit_given[agent_id] = False
            elif prev_held == "soup" and curr_held is None:
                # Drop/delivery: clear tracking
                self._carried_soup_ids[agent_id] = None
                self._approach_credit_given[agent_id] = True

        # Track delivery events (count actual deliveries, not timesteps)
        if delivery_occurred:
            self.event_counts["correct_delivery"] += 1

        # Track penalty events
        if self._waste_event(state):
            self.event_counts["penalty"] += 1

        # Compute shaped rewards for each agent
        for agent_id in range(2):
            s = self._compute_agent_shaping(
                agent_id, state, curr_pots, pot_events, delivery_occurred
            )
            shaped[agent_id] += sum(s.values())
            prefix = f"agent{agent_id}_"
            for k, v in s.items():
                info[prefix + k] = float(v)

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
        """
        Detect pot state changes using direct state transitions ONLY.
        NO inference - only credit exact +1 transitions per official Overcooked-AI style.
        Flicker protection: credit pot initial contents ONCE per episode.
        """
        events = {"onion_added": 0, "cooking_started": 0, "soup_ready": 0}
        
        for pos in curr.keys():
            if pos not in prev:
                # New pot appeared - credit ONCE per episode to prevent flicker exploit
                if pos not in self._pots_credited_this_episode:
                    if curr[pos]["num_items"] > 0:
                        events["onion_added"] += curr[pos]["num_items"]
                        if DEBUG_POT_EVENTS:
                            print(f"[POT_DEBUG] ✓ onion_added {curr[pos]['num_items']} at new pot {pos} (first seen this episode)")
                    self._pots_credited_this_episode.add(pos)
                elif DEBUG_POT_EVENTS:
                    print(f"[POT_DEBUG] ⚠ pot at {pos} already credited this episode, skipping flicker")
                continue
            
            p, c = prev[pos], curr[pos]
            
            # Debug: Show all state changes
            if DEBUG_POT_EVENTS and (
                c["num_items"] != p["num_items"]
                or c["is_cooking"] != p["is_cooking"]
                or c["is_ready"] != p["is_ready"]
            ):
                print(
                    f"[POT_DEBUG] {pos}: items {p['num_items']}→{c['num_items']}, cooking {p['is_cooking']}→{c['is_cooking']}, ready {p['is_ready']}→{c['is_ready']}"
                )

            # CRITICAL: Only credit DIRECT num_items increases (no inference!)
            # This prevents double-counting when soup becomes ready
            items_added = c["num_items"] - p["num_items"]
            if items_added > 0:
                events["onion_added"] += items_added
                if DEBUG_POT_EVENTS:
                    print(f"[POT_DEBUG] ✓ onion_added {items_added} at {pos} (direct transition)")
            
            # Soup became ready - credit ONCE per pot using tracking set
            if not p["is_ready"] and c["is_ready"]:
                pot_id = pos
                
                if pot_id not in self._credited_ready_pots:
                    events["soup_ready"] += 1
                    self._credited_ready_pots.add(pot_id)
                    
                    if DEBUG_POT_EVENTS:
                        print(f"[POT_DEBUG] ✓ soup_ready at {pos} (first time)")
                elif DEBUG_POT_EVENTS:
                    print(f"[POT_DEBUG] ⚠ soup_ready at {pos} already credited, skipping")
            
            # Cooking started
            if not p["is_cooking"] and c["is_cooking"]:
                events["cooking_started"] += 1
                if DEBUG_POT_EVENTS:
                    print(f"[POT_DEBUG] ✓ cooking_started at {pos}")
        
        # Clear tracking for pots that were picked up (no longer in curr_pots)
        pots_to_clear = self._credited_ready_pots - set(curr.keys())
        if pots_to_clear and DEBUG_POT_EVENTS:
            print(f"[POT_DEBUG] Clearing credited pots: {pots_to_clear}")
        self._credited_ready_pots -= pots_to_clear
        
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

    def _correct_delivery_event(self, state, sparse_rewards):
        """Bootstrap gating: position-based until ep 40k, then strict sparse +20 gate."""
        base = self._coerce_rewards(sparse_rewards)
        total_sparse = sum(base)

        # Safety cap deliveries per episode (raised to 20 to avoid blocking legitimate learning)
        if self.soups_delivered_this_episode >= 20:
            print(f"[DELIVERY_DEBUG] ✗ Delivery cap (20) reached this episode")
            return 0

        serving = self._serving_positions()

        # Bootstrap phase (first 40k episodes): accept position-based for re-exploration
        if self.episode_count < 40000:
            # Debug print mode change
            if self.episode_count % 1000 == 0 and not hasattr(self, f"_printed_bootstrap_{self.episode_count}"):
                print(f"[DELIVERY_DEBUG] Episode {self.episode_count}: Using POSITION-BASED delivery detection (bootstrap mode)")
                setattr(self, f"_printed_bootstrap_{self.episode_count}", True)
            for aid in range(2):
                p, pp = state.players[aid], self.prev_state.players[aid]
                prev_held = (
                    getattr(pp.held_object, "name", None) if pp.held_object else None
                )
                curr_held = (
                    getattr(p.held_object, "name", None) if p.held_object else None
                )

                if prev_held == "soup" and curr_held is None:
                    dist_to_serving = [_manhattan(p.position, s) for s in serving]
                    min_dist = min(dist_to_serving) if dist_to_serving else 999
                    print(
                        f"[DELIVERY_DEBUG] Agent {aid} dropped soup at {p.position}, min_dist: {min_dist} (bootstrap phase)"
                    )

                    if min_dist <= 2:
                        self.soups_delivered_this_episode += 1
                        print(
                            f"[DELIVERY_DEBUG] ✓ Delivery #{self.soups_delivered_this_episode} (position-based)"
                        )
                        return 1
            return 0

        # Strict phase (after 40k): require sparse +20 reward ONLY
        if total_sparse < 19.0:
            return 0

        for aid in range(2):
            p, pp = state.players[aid], self.prev_state.players[aid]
            prev_held = (
                getattr(pp.held_object, "name", None) if pp.held_object else None
            )
            curr_held = getattr(p.held_object, "name", None) if p.held_object else None

            if prev_held == "soup" and curr_held is None:
                dist_to_serving = [_manhattan(p.position, s) for s in serving]
                min_dist = min(dist_to_serving) if dist_to_serving else 999
                print(
                    f"[DELIVERY_DEBUG] Agent {aid} dropped soup at {p.position}, min_dist: {min_dist}, sparse: {total_sparse:.1f}"
                )

                if min_dist <= 2:
                    self.soups_delivered_this_episode += 1
                    print(
                        f"[DELIVERY_DEBUG] ✓ Delivery #{self.soups_delivered_this_episode} (verified)"
                    )
                    return 1
                else:
                    print(
                        f"[DELIVERY_DEBUG] ⚠ Sparse reward but position check failed (dist={min_dist})"
                    )

        if total_sparse >= 15.0:
            print(
                f"[DELIVERY_DEBUG] ⚠ Sparse {total_sparse:.1f} but no soup drop observed"
            )
        return 0

    def _waste_event(self, state):
        serving = self._serving_positions()
        for aid in range(2):
            p, pp = state.players[aid], self.prev_state.players[aid]
            if (
                pp.held_object
                and getattr(pp.held_object, "name", "") == "soup"
                and p.held_object is None
            ):
                if not any(_manhattan(p.position, s) <= 2 for s in serving):
                    print(f"[DELIVERY_DEBUG] ✗ Agent {aid} wasted soup at {p.position}")
                    return 1
        return 0

    def _compute_agent_shaping(self, agent_id, state, curr_pots, pot_events, delivery_occurred):
        w = self.shape_weights
        s = {
            "onion_in_pot": 0.0,
            "cooking_start": 0.0,
            "soup_ready": 0.0,
            "soup_pickup": 0.0,
            "correct_delivery": 0.0,
            "approach_serving": 0.0,
            "penalty": 0.0,
        }

        # Team pot events (NO BUDGETS - allow continuous learning)
        if pot_events["onion_added"] > 0:
            s["onion_in_pot"] = float(w.get("onion_in_pot", 0.25))
        if pot_events["cooking_started"] > 0:
            s["cooking_start"] = float(w.get("cooking_start", 0.30))
        if pot_events["soup_ready"] > 0:
            s["soup_ready"] = float(w.get("soup_ready", 0.40))

        # Individual soup pickup with ID tracking (prevents pickup-drop exploit)
        p = state.players[agent_id]
        pp = self.prev_state.players[agent_id]
        prev_held = getattr(pp.held_object, "name", None) if pp.held_object else None
        curr_held = getattr(p.held_object, "name", None) if p.held_object else None
        
        if prev_held is None and curr_held == "soup":
            soup_id = id(p.held_object) if p.held_object else None
            if soup_id and soup_id not in self.picked_up_soups:
                s["soup_pickup"] = float(w.get("soup_pickup", 0.35))
                self.picked_up_soups.add(soup_id)
                if DEBUG_POT_EVENTS:
                    print(f"[POT_DEBUG] ✓ soup_pickup by agent {agent_id} (soup_id: {soup_id})")
            elif soup_id in self.picked_up_soups and DEBUG_POT_EVENTS:
                print(f"[POT_DEBUG] ⚠ soup already picked up (soup_id: {soup_id})")

        # One-time approach_serving nudge per carried soup
        if p.held_object and getattr(p.held_object, "name", "") == "soup":
            serving = self._serving_positions()
            if serving and not self._approach_credit_given[agent_id]:
                curr_dist = min(_manhattan(p.position, s_pos) for s_pos in serving)
                prev_dist = min(_manhattan(pp.position, s_pos) for s_pos in serving)
                if curr_dist < prev_dist:
                    s["approach_serving"] = float(w.get("approach_serving", 0.0))
                    self._approach_credit_given[agent_id] = True

        # Team delivery with ID tracking (prevents duplicate delivery credits)
        if delivery_occurred:
            # Track which soup was delivered to prevent duplicate credits
            delivered_soup_id = None
            for aid in range(2):
                pp = self.prev_state.players[aid]
                prev_held = getattr(pp.held_object, "name", None) if pp.held_object else None
                if prev_held == "soup":
                    delivered_soup_id = id(pp.held_object) if pp.held_object else None
                    break
            
            if delivered_soup_id and delivered_soup_id not in self.delivered_soups:
                s["correct_delivery"] = float(w.get("correct_delivery", 0.50))
                self.delivered_soups.add(delivered_soup_id)
                if DEBUG_DELIVERY_EVENTS:
                    print(f"[DELIVERY_DEBUG] ✓ correct_delivery credited (soup_id: {delivered_soup_id})")
            elif delivered_soup_id in self.delivered_soups and DEBUG_DELIVERY_EVENTS:
                print(f"[DELIVERY_DEBUG] ⚠ soup already delivered (soup_id: {delivered_soup_id})")
            elif not delivered_soup_id:
                # Fallback: credit delivery even without soup ID (shouldn't happen often)
                s["correct_delivery"] = float(w.get("correct_delivery", 0.50))

        # Penalty
        if self._waste_event(state):
            s["penalty"] = -float(w.get("penalty_drop", 0.10))

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
