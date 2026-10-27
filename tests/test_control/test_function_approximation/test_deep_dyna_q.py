import numpy as np
import torch

from samsara_rl.control.dyna.deep_dyna_q import (
    DeepDynaQ,
)
from samsara_rl.functions.neural_networks.fully_connected import FullyConnected


def test_dyna_q_network_convergence_grid_world(
    grid_world_mdp,
    fully_connected_one_hot_network,
    one_hot_encoding_torch,
    grid_world_model_interpretor,
):
    """Q-Learning with linear one-hot features should learn a reasonable policy on GridWorld.

    After sufficient episodes the value of terminal states should remain
    at zero and non-terminal states should have negative values (every
    step costs -1).
    """
    world_model = FullyConnected(20, 32, 18)
    ql = DeepDynaQ(
        mdp=grid_world_mdp,
        world_model=world_model,
        world_model_interpreter=grid_world_model_interpretor,
        model_plan_steps=4,
        count_planning_updates=True,
        model_train_steps=10,
        gamma=1,
        q=fully_connected_one_hot_network,
        alpha=0.01,
        preprocess=one_hot_encoding_torch,
        target_update_freq=1000,
        state_loss=torch.nn.CrossEntropyLoss(),
        epsilon_schedule="exponential",
    )
    ql.evaluate(max_iter=60)
    q_values = np.array([ql.get_q_values(x).max() for x in range(0, 16)]).reshape(4, 4)
    q_values[0][0] = 0
    q_values[3][3] = 0
    expected_q = np.array([
        [0, -1, -2, -3],
        [-1, -2, -3, -2],
        [-2, -3, -2, -1],
        [-3, -2, -1, 0],
    ])
    assert np.allclose(q_values, expected_q, atol=0.05)
