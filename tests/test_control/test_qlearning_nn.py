from samsara_rl.control.function_approximation.functions.neural_networks.linear import LinearNetwork
from samsara_rl.control.function_approximation.functions.manual.linear import LinearFunction
import torch
from samsara_rl.control.function_approximation.online.td.td import TemporalDifferenceGradient
import gymnasium as gym


def test_linear_nn(grid_world_mdp, random_policy):
    """Q-Learning should converge close to expected Q values for the grid world."""
    lf = LinearFunction(4, 2)

    state = torch.tensor([1.2, 2.1, 4.1, 4.5])
    # out_ln = ln(state)
    # out_lf = lf(state.numpy())

    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)

    # print(out_ln)
    # print(out_lf)

    def X(S: torch.tensor) -> torch.tensor:
        arr = torch.zeros(16, dtype=torch.float64)
        arr[int(S)] = 1
        return arr

    ln = LinearNetwork(16, 4, False, preprocess=X)

    nn_agent = TemporalDifferenceGradient(grid_world_mdp, random_policy, q = ln, gamma=0.999,alpha=0.01,auto_grad=True, _lambda=0)
    nn_agent.evaluate(max_iter=20000)
    import numpy as np
    # Collect V(s) = max_a Q(s, a) for each state
    with torch.no_grad():
        v = np.array([ln(torch.tensor(s)).max() for s in range(16)]).reshape(4, 4)
        print(v)

