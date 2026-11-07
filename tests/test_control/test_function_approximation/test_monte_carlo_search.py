import torch
import torch.nn.functional as F

from samsara_rl.search.monte_carlo_search import MonteCarloSearch


def test_monte_carlo_search(
    grid_world_mdp_one_hot,
    grid_world_model_interpretor,
    grid_world_random_policy,
    grid_world_trained_model,
):
    start_state = F.one_hot(torch.tensor(2), num_classes=16)
    mc_search = MonteCarloSearch(grid_world_random_policy, grid_world_trained_model, grid_world_model_interpretor)

    expected_rewards = mc_search.plan(
        state=start_state,
        depth=8,
        budget=30,
        actions=grid_world_mdp_one_hot.action_space,
        gamma=0.99,
    )
    best_action = torch.argmax(expected_rewards).item()
    assert best_action == 2
