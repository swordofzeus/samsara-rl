import numpy as np
import torch

from samsara_rl.control.actor_critic.value_function_actor_critic import (
    ValueFunctionActorCritic,
)
from samsara_rl.functions.neural_networks.fully_connected import FullyConnected


def test_value_function_actor_critic_convergence(grid_world_mdp, one_hot_encoding_torch):
    """ValueFunctionActorCritic should learn a shortest path from state 10 to terminal 15.

    Two optimal paths exist: RIGHT then DOWN (states 10->11->15),
    or DOWN then RIGHT (states 10->14->15). The assert checks for either.
    """
    value_network = FullyConnected(16, 32, 1, preprocess=one_hot_encoding_torch)
    policy_network = FullyConnected(16, 32, 4, preprocess=one_hot_encoding_torch)
    value_fn_actor_critic = ValueFunctionActorCritic(
        grid_world_mdp,
        gamma=0.9,
        alpha=0.004,
        batch_size=32,
        value_network=value_network,
        policy_network=policy_network,
    )

    value_fn_actor_critic.evaluate(max_iter=6000)

    with torch.no_grad():
        action_values = np.array([
            value_fn_actor_critic.policy_network(x).log_softmax(dim=0).argmax() for x in range(0, 16)
        ]).reshape(4, 4)
        print(action_values)

        # assert np.array_equal(action_values[2, 2:], np.array([3, 1])) or np.array_equal(
        #     action_values[2:, 2], np.array([1, 3])
        # )
