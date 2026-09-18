import numpy as np


def discounted_cum_trajectory(gamma: float, reward: np.ndarray) -> np.ndarray:
    """Compute discounted returns for every time step, vectorized.

    Avoids the standard O(T) reverse loop by factoring out discount
    weights from a cumulative sum:

    1. Divide each reward by its positional gamma power to normalize.
    2. Reverse and cumsum so earlier states accumulate future rewards.
    3. Multiply back by gamma powers to restore correct discounting.

    Args:
        reward: Array of shape ``(T,)`` containing rewards.

    Returns:
        Array of shape ``(T,)`` with the discounted return G_t for
        each time step.
    """
    discount_ratio = gamma ** np.arange(0, len(reward))[::-1]
    reversed_reward = reward
    reversed_reward = reversed_reward / discount_ratio
    reversed_reward_cum = reversed_reward[::-1].cumsum()
    reversed_reward_cum = reversed_reward_cum * discount_ratio[::-1]
    result: np.ndarray = reversed_reward_cum[::-1]
    return result
