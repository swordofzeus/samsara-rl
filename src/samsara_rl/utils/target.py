import numpy as np
from samsara_rl.control.function_approximation.functions.linear import LinearFunction
from typing import Any

def td_target(policy: np.ndarray, state: int, q: np.ndarray) -> float:
    result: float = q[state].max()
    return result


def sarsa_target(
    history: np.ndarray, q: LinearFunction, mdp: Any, terminal
) -> float:
    """Compute TD target: 0 for terminal states, Q(s', a') otherwise."""
    S_prime: float = history.past_states()[-1]
    A_prime: int = int(history.past_actions()[-1].item())
    return (
        0
        if terminal
        else q(S_prime)[A_prime]
    )


def qlearning_target(
    history: np.ndarray, q: LinearFunction, mdp: Any, terminal
) -> float:
    """Compute TD target: 0 for terminal states, Q(s', a') otherwise."""
    S_prime: float = history.past_states()[-1]
    return (
        0
        if terminal
        else q(S_prime).max(axis=0)
    )
