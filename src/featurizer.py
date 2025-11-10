# src/featurizer.py

import numpy as np


def one_hot(index, n):
    vec = np.zeros((n,), dtype=np.float32)
    try:
        idx = int(index)
    except Exception:
        idx = 0
    idx = max(0, min(n - 1, idx))
    vec[idx] = 1.0
    return vec


def orientation_to_idx(ori):
    # Handles tuple/list/np.array unit vectors, strings, and ints
    if isinstance(ori, (tuple, list, np.ndarray)):
        try:
            t = tuple(int(x) for x in np.asarray(ori).tolist())
        except Exception:
            t = (0, 1)
        mapping = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
        return mapping.get(t, 0)
    if isinstance(ori, str):
        mapping = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}
        return mapping.get(ori.upper(), 0)
    if isinstance(ori, (int, np.integer)):
        return int(ori) % 4
    return 0


def get_agent_features(player, mdp):
    x, y = player.position
    ori_idx = orientation_to_idx(getattr(player, "orientation", 0))
    holding = getattr(player, "held_object", None)
    name = getattr(holding, "name", "") if holding is not None else ""
    has_onion = 1.0 if name == "onion" else 0.0
    has_soup = 1.0 if name == "soup" else 0.0
    has_dish = 1.0 if name == "dish" else 0.0
    return np.concatenate(
        [
            np.array([float(x), float(y)], dtype=np.float32),  # 2
            one_hot(ori_idx, 4),  # 4
            np.array([has_onion, has_soup, has_dish], dtype=np.float32),  # 3
        ],
        axis=0,
    )  # total 9 dims


# Wherever you build obs, keep the total at 96 dims as designed
def featurize_state_mlam_free(env):
    state = env.state
    mdp = env.mdp
    # ... your grid_feats -> 72 dims ...
    g = grid_feats(state, mdp.width, mdp.height)  # 72
    p0 = get_agent_features(state.players[0], mdp)  # 9
    p1 = get_agent_features(state.players[1], mdp)  # 9
    pad = np.zeros((6,), dtype=np.float32)  # 6
    obs0 = np.concatenate([g, p0, p1, pad], axis=0)  # 72+9+9+6 = 96
    obs1 = np.concatenate([g, p1, p0, pad], axis=0)  # 96
    return [obs0, obs1]


# src/featurizer.py (add this above featurize_state_mlam_free)


def grid_feats(state, width, height):
    import numpy as np

    # 6 binary channels: onions, pots(cooking), pots(ready), dishes, serving, counters
    onions = np.zeros((height, width), np.float32)
    pots_cook = np.zeros_like(onions)
    pots_ready = np.zeros_like(onions)
    dishes = np.zeros_like(onions)
    serving = np.zeros_like(onions)
    counters = np.zeros_like(onions)

    # Populate channels based on objects in state
    for obj in getattr(state, "objects", []):
        if not hasattr(obj, "position"):
            continue
        x, y = obj.position
        if not (0 <= x < width and 0 <= y < height):
            continue
        name = getattr(obj, "name", "")
        if name == "onion":
            onions[y, x] = 1.0
        elif name == "pot":
            if bool(getattr(obj, "is_ready", False)):
                pots_ready[y, x] = 1.0
            elif bool(getattr(obj, "is_cooking", False)):
                pots_cook[y, x] = 1.0
        elif name == "dish":
            dishes[y, x] = 1.0
        elif name == "serving":
            serving[y, x] = 1.0
        elif name == "counter":
            counters[y, x] = 1.0

    # Downsample/flatten to fixed 12 values per channel -> 6 * 12 = 72
    def pool12(mat):
        flat = mat.reshape(-1)
        if flat.size >= 12:
            return flat[:12]
        out = np.zeros((12,), np.float32)
        out[: flat.size] = flat
        return out

    channels = [onions, pots_cook, pots_ready, dishes, serving, counters]
    pooled = [pool12(c) for c in channels]
    return np.concatenate(pooled, axis=0).astype(np.float32)  # shape (72,)
