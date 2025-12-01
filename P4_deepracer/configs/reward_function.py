import math
import math


def reward_function(params):
    """Obstacle-aware reward encouraging avoidance behavior."""
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    objects_location = params['objects_location']
    agent_x = params['x']
    agent_y = params['y']
    _, next_object_index = params['closest_objects']
    is_crashed = params['is_crashed']

    # 1. Immediate failure checks
    if is_crashed or not all_wheels_on_track:
        return 1e-3

    # 2. Distance to next obstacle
    next_object_loc = objects_location[next_object_index]
    dx = agent_x - next_object_loc[0]
    dy = agent_y - next_object_loc[1]
    distance_to_object = math.sqrt(dx**2 + dy**2)

    # 3. Repulsion zones
    if distance_to_object < 0.5:
        return 1e-3

    avoidance_factor = 1.0
    if distance_to_object < 0.8:
        avoidance_factor = 0.5

    # 4. Base centerline reward
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    reward = 1e-3
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1

    return float(max(reward * avoidance_factor, 1e-3))
