# src/env_builder.py
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from .env_wrapper import OvercookedEnvWrapper

# Complete, planning-safe params covering keys accessed in various versions
DISABLE_MLAM_PARAMS = {
    "compute_mlam_actions": False,
    "use_pickled_mlam": False,
    "force_compute": False,
    "cache_type": "none",
    # Keys frequently accessed unconditionally
    "counter_pickup": False,
    "counter_goals": [],  # <- add this to avoid KeyError
    "same_motion_goals": False,
    "wait_allowed": False,
    "num_goals": 0,
    "goal_info": "none",
}


class _NoOpMLAM:
    motion_planner = None


class _NoOpMP:
    """No-op MotionPlanner stand-in."""

    def __init__(self):
        pass


def build_overcooked_env(layout, horizon=400, seed=None):
    mdp = OvercookedGridworld.from_layout_name(layout)
    base_env = OvercookedEnv.from_mdp(
        mdp=mdp,
        horizon=horizon,
        info_level=0,
        mlam_params=DISABLE_MLAM_PARAMS,
    )

    # Short-circuit MLAM/MP properties used in some env.step paths
    try:
        base_env._mlam = _NoOpMLAM()
        base_env._mp = _NoOpMP()
        base_env.mlam_params = DISABLE_MLAM_PARAMS
    except Exception:
        pass

    # Seed reproducibly
    if seed is not None:
        try:
            if hasattr(base_env, "mdp") and hasattr(base_env.mdp, "seed"):
                base_env.mdp.seed(seed)
        except Exception:
            pass

    env = OvercookedEnvWrapper(base_env, expect_obs_dim=96)

    # Tripwire: fail fast if real planners were created
    if hasattr(base_env, "_mlam"):
        from overcooked_ai_py.planning.planners import MediumLevelActionManager

        if isinstance(base_env._mlam, MediumLevelActionManager):
            raise RuntimeError(
                "Real MLAM constructed; remove any featurize_state_mdp/mlam usage."
            )
    if hasattr(base_env, "_mp"):
        try:
            from overcooked_ai_py.mdp.overcooked_environment import (
                MotionPlanner,
            )  # alt path in some forks
        except Exception:
            from overcooked_ai_py.planning.planners import MotionPlanner  # typical path
        if isinstance(base_env._mp, MotionPlanner):
            raise RuntimeError(
                "Real MotionPlanner constructed; ensure DISABLE_MLAM_PARAMS and no-op _mp are set."
            )
    return env
