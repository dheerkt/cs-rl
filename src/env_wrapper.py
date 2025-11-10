# src/env_wrapper.py

import numpy as np
from overcooked_ai_py.mdp.actions import Action
from .featurizer import featurize_state_mlam_free


class OvercookedEnvWrapper:
    """
    MLAM‑free wrapper:
      - Uses custom 96‑dim featurizer (never calls featurize_state_mdp)
      - Ensures a concrete state exists at reset/step
      - Maps discrete ints to Overcooked actions
      - Normalizes rewards to a 2‑vector of floats
      - Handles Gym (4‑tuple) and Gymnasium (5‑tuple) step signatures
      - Seeds via MDP and common RNGs
    """

    def __init__(self, env, expect_obs_dim=96):
        self.env = env
        self.state = None
        self.expect_obs_dim = expect_obs_dim

    # ---------- Internal helpers ----------

    def _ensure_concrete_state(self):
        if self.state is None or not hasattr(self.state, "players"):
            mdp = getattr(self.env, "mdp", None)
            if mdp is not None and hasattr(mdp, "get_standard_start_state"):
                self.state = mdp.get_standard_start_state()
                if hasattr(self.env, "state"):
                    self.env.state = self.state
            else:
                raise RuntimeError(
                    "Cannot initialize Overcooked state: no state from env.reset "
                    "and mdp lacks get_standard_start_state()."
                )

    def _obs_dict(self):
        obs_list = featurize_state_mlam_free(self.env)
        for i, obs in enumerate(obs_list):
            if not isinstance(obs, np.ndarray):
                raise TypeError(f"Obs[{i}] is {type(obs)}; expected np.ndarray")
            if obs.shape != (self.expect_obs_dim,):
                raise ValueError(
                    f"Obs[{i}] shape {obs.shape} != ({self.expect_obs_dim},)"
                )
            if obs.dtype != np.float32:
                obs_list[i] = obs.astype(np.float32, copy=False)
        return {"both_agent_obs": obs_list}

    def _normalize_rewards(self, rewards):
        if isinstance(rewards, (int, float, np.floating)):
            r = float(rewards)
            return (r, r)
        if isinstance(rewards, (list, tuple, np.ndarray)):
            if len(rewards) == 1:
                r = float(rewards[0])
                return (r, r)
            if len(rewards) >= 2:
                return (float(rewards[0]), float(rewards[1]))
        raise ValueError(f"Unexpected rewards format: {type(rewards)} {rewards}")

    # ---------- Public API ----------

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)

        out = self.env.reset()
        if isinstance(out, tuple):
            self.state = out[0] if len(out) > 0 and hasattr(out[0], "players") else None
        else:
            self.state = out if hasattr(out, "players") else None

        if hasattr(self.env, "state") and self.state is not None:
            self.env.state = self.state

        self._ensure_concrete_state()
        return self._obs_dict()

    def step(self, actions):
        a0 = int(actions[0])
        a1 = int(actions[1])
        joint_actions = (Action.INDEX_TO_ACTION[a0], Action.INDEX_TO_ACTION[a1])

        out = self.env.step(joint_actions)

        if not isinstance(out, tuple):
            raise RuntimeError(f"Unexpected env.step return: {type(out)}")
        if len(out) == 4:
            next_state, rewards, done, info = out
        elif len(out) == 5:
            next_state, rewards, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            raise RuntimeError(f"Unexpected env.step tuple length: {len(out)}")

        rewards = self._normalize_rewards(rewards)

        if next_state is not None and hasattr(next_state, "players"):
            self.state = next_state
            if hasattr(self.env, "state"):
                self.env.state = self.state
        else:
            self._ensure_concrete_state()

        obs = self._obs_dict()
        return obs, rewards, done, info

    def seed(self, seed=None):
        if seed is None:
            return
        try:
            if hasattr(self.env, "mdp") and hasattr(self.env.mdp, "seed"):
                self.env.mdp.seed(seed)
        except Exception:
            pass
        try:
            import random, torch

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        except Exception:
            pass

    @property
    def mdp(self):
        return getattr(self.env, "mdp", None)
