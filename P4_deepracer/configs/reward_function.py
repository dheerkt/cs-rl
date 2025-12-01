import math

# Tunable coefficients for clarity
CENTERLINE_SHARPNESS = 8.0  # higher -> stricter center following
STEERING_THRESHOLD_DEGREES = 15.0
STEERING_PENALTY_FACTOR = 0.5  # multiplier applied when steering beyond threshold
MIN_REWARD = 1e-3


def reward_function(params):
    """Stateless reward encouraging fast, centered, smooth driving."""
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    all_wheels_on_track = params['all_wheels_on_track']
    steering_angle = abs(params['steering_angle'])

    # Hard gate: minimal reward if any wheel leaves the track
    if not all_wheels_on_track:
        return MIN_REWARD

    # Base reward is proportional to speed (proxy for progress rate)
    reward = max(speed, MIN_REWARD)

    # Exponential centerline factor with smooth gradients
    normalized_distance = distance_from_center / track_width
    centerline_factor = math.exp(-CENTERLINE_SHARPNESS * normalized_distance ** 2)

    # Steering smoothness encourages efficient racing lines
    steering_factor = 1.0
    if steering_angle > STEERING_THRESHOLD_DEGREES:
        steering_factor = STEERING_PENALTY_FACTOR

    reward *= centerline_factor * steering_factor
    return float(max(reward, MIN_REWARD))
