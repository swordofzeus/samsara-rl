import gymnasium
import torch
import torch.nn.functional as F

from samsara_rl.search.search import Search


class MonteCarloSearch(Search):
    """Vectorized Monte Carlo forward search using a learned world model.

    Evaluates each candidate action by performing multiple stochastic
    rollouts through the world model, averaging the discounted returns.
    All actions and rollouts are evaluated in parallel using batched
    tensor operations. After the initial candidate action, subsequent
    actions are chosen by the rollout policy.
    """

    def plan(
        self,
        state: torch.Tensor,
        actions: gymnasium.spaces.Discrete,
        depth: int,
        budget: int,
        gamma: float = 1,
    ) -> torch.Tensor:
        """Evaluate all actions via Monte Carlo rollouts.

        For each action in the action space, runs ``budget`` rollouts
        of length ``depth`` and averages the discounted returns.

        Args:
            state: The current state to plan from.
            actions: The discrete action space to search over.
            depth: Maximum simulation depth per rollout.
            budget: Number of rollouts to average per action.
            gamma: Discount factor for future rewards (0 < gamma <= 1).

        Returns:
            Tensor of average estimated values, one per action.
        """
        self._validate_params(depth, budget, gamma)
        n_actions: int = int(actions.n)
        one_hot_action = F.one_hot(torch.arange(0, n_actions), num_classes=n_actions).to(
            dtype=state.dtype, device=state.device
        )

        # create all state x action rollout combinations for parallel evaluation
        state_action = state.unsqueeze(0).repeat(one_hot_action.shape[0], 1)
        state_action = torch.hstack((state_action, one_hot_action))
        state_action_batch = state_action.repeat_interleave(budget, dim=0)

        # keep track of which rollouts terminated & total reward
        rollout_cum_reward = torch.zeros((n_actions, budget), dtype=state.dtype, device=state.device)
        terminated_rollouts = torch.zeros((n_actions * budget,), dtype=torch.bool, device=state.device)
        with torch.no_grad():
            for step in range(depth):
                # predict next state, parse world model output
                prediction = self.model(state_action_batch[~terminated_rollouts])
                _, predicted_reward, terminal_logits = self.world_model_interpreter.parse(prediction)
                predicted_next_state = self.world_model_interpreter.predicted_next_state(prediction)
                predicted_is_terminal = self.world_model_interpreter.is_terminal(terminal_logits, threshold=0.5)

                # update reward, remove terminated states from next rollout
                rollout_cum_reward[~terminated_rollouts.view(n_actions, -1)] = rollout_cum_reward[
                    ~terminated_rollouts.view(n_actions, -1)
                ] + predicted_reward * (gamma**step)
                predicted_is_terminal = predicted_is_terminal.bool()
                terminated_rollouts[~terminated_rollouts] = (
                    terminated_rollouts[~terminated_rollouts] | predicted_is_terminal
                )
                predicted_next_state = predicted_next_state[~predicted_is_terminal]

                if not state_action_batch[~terminated_rollouts].numel():
                    break  # all rollouts termianted, end early

                # sample policy, encode state + action vector for next rollout
                action = self.policy(predicted_next_state)
                one_hot_action = F.one_hot(torch.as_tensor(action), num_classes=n_actions).to(
                    dtype=state.dtype, device=state.device
                )
                state_action = torch.hstack((predicted_next_state, one_hot_action))
                state_action_batch[~terminated_rollouts] = state_action

        return rollout_cum_reward.mean(dim=1)
