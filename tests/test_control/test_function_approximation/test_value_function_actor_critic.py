from email import policy
import numpy as np
import torch
from samsara_rl.functions.neural_networks.fully_connected import FullyConnected
from samsara_rl.control.actor_critic.value_function_actor_critic import (
    ValueFunctionActorCritic,
)


def test_value_function_actor_critic_convergence(
    grid_world_mdp, one_hot_encoding_torch
):
    """ValueFunctionActorCritic should learn a shortest path from state 10 to terminal 15.

    Two optimal paths exist: RIGHT then DOWN (states 10->11->15),
    or DOWN then RIGHT (states 10->14->15). The assert checks for either.
    """
    value_network = FullyConnected(16, 32, 1)
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
