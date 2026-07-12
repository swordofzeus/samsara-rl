from gymnasium import spaces
from gymnasium.spaces.utils import flatdim


def action_output_dim(action_space: spaces.Space) -> int:
    """Return the dimensionality of the agent's action output.

    Discrete spaces output a single integer, so dim = 1.
    Box/MultiDiscrete spaces output a vector matching their shape.
    """
    if isinstance(action_space, spaces.Discrete):
        return 1
    return flatdim(action_space)


def state_output_dim(observation_space: spaces.Space) -> int:
    """Return the dimensionality of a single state observation.

    Discrete spaces output a single integer, so dim = 1.
    Box/MultiDiscrete spaces output a vector matching their shape.
    """
    if isinstance(observation_space, spaces.Discrete):
        return 1
    return flatdim(observation_space)
