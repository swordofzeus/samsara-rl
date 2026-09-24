import numpy as np
import torch

from samsara_rl.control.policy_gradient.monte_carlo_policy_gradient import (
    MonteCarloPolicyGradient,
)


def test_monte_carlo_policy_gradient_convergence(grid_world_mdp, fully_connected_one_hot_network):
    """MC Gradient should learn a shortest path from state 10 to terminal 15.

    Two optimal paths exist: RIGHT then DOWN (states 10->11->15),
    or DOWN then RIGHT (states 10->14->15). The assert checks for either.
    """
    monte_carlo_agent = MonteCarloPolicyGradient(
        grid_world_mdp,
        gamma=0.9,
        alpha=0.004,
        policy_network=fully_connected_one_hot_network,
        batch_size=32,
    )
    monte_carlo_agent.evaluate(max_iter=5000)
    with torch.no_grad():
        action_values = np.array([
            monte_carlo_agent.policy_network(x).log_softmax(dim=0).argmax() for x in range(0, 16)
        ]).reshape(4, 4)
        assert np.array_equal(action_values[2, 2:], np.array([3, 1])) or np.array_equal(
            action_values[2:, 2], np.array([1, 3])
        )
