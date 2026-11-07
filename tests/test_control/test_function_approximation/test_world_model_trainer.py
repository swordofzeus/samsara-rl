from samsara_rl.policy.stochastic_policy import StochasticPolicy
from samsara_rl.search.monte_carlo_search import MonteCarloSearch
from samsara_rl.functions.neural_networks.fully_connected import FullyConnected
from samsara_rl.models.world_model_trainer import WorldModelAgent
import torch


def test_world_model_trainer(
    grid_world_mdp_one_hot, grid_world_model_interpretor, grid_world_random_policy
):

    world_model = FullyConnected(20, 32, 18)
    wm_agent = WorldModelAgent(
        mdp=grid_world_mdp_one_hot,
        alpha=0.01,
        gamma=1,
        loss_fn=torch.nn.CrossEntropyLoss(),
        batch_size=32,
        world_model=world_model,
        policy=grid_world_random_policy,
        world_model_interpreter=grid_world_model_interpretor,
        # save_dir="grid_world_model.pt"
    )
    wm_agent.evaluate(1000)
    assert wm_agent.metrics["World Model Loss"] < 0.001
    print("testing mc")
