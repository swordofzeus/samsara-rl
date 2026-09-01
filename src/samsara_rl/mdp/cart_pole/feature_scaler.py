from typing import Any

import numpy as np
import torch


def scale_angle_only(x: np.ndarray) -> np.ndarray:
    """Extract and normalize pole angle to [-1, 1].

    Args:
        x: Raw CartPole observation [cart_pos, cart_vel, angle, angular_vel].

    Returns:
        Single-element array with normalized angle.
    """
    angle = 2 * (x[2] - (-0.418)) / (0.418 - (-0.418)) - 1
    return np.array([angle])


def scale_angle_angvel(x: np.ndarray) -> Any:
    """Extract and normalize pole angle and angular velocity to [-1, 1].

    Angular velocity is clipped to [-5, 5] before scaling.

    Args:
        x: Raw CartPole observation [cart_pos, cart_vel, angle, angular_vel].

    Returns:
        Two-element array with normalized angle and angular velocity.
    """

    features = x[2:4] if x.ndim == 1 else x[:, 2:4]
    mins = np.array([-0.418, -5])
    maxs = np.array([0.418, 5])
    features = np.clip(features, mins, maxs)
    scaled_features = 2 * (features - mins) / (maxs - mins) - 1
    return torch.from_numpy(scaled_features) if isinstance(scaled_features, np.ndarray) else scaled_features
