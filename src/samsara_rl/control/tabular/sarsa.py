from typing import Any

from samsara_rl.control.tabular.tabular_agent import TabularAgent
from samsara_rl.utils.memory.episode import Episode


class Sarsa(TabularAgent):
    def __init__(self, mdp: Any, **kwargs: Any):
        super().__init__(mdp, **kwargs)

    def td_target(self, history: Episode) -> float:
        state = history.past_states()[-1].astype(int)
        A_prime = history.past_actions()[-1].astype(int)
        result: float = self.q[state][A_prime]
        return result

    def run_episode(self) -> Episode:
        """Generate a complete episode using the SARSA loop.

        Selects A' before ``post_visit`` so the next action is recorded
        in the episode history for use in the SARSA TD target.

        Returns:
            The episode history.
        """
        curr_state, _ = self.mdp.reset()
        episode_history = Episode.from_gym(self.mdp, curr_state)
        curr_action = self.select_action(curr_state)
        terminated = False

        while not terminated:
            next_state, reward, terminated, truncated, _ = self.mdp.step(curr_action)
            next_action = self.select_action(next_state)
            episode_history.record(curr_action, reward, next_state, a_prime=next_action)
            curr_action = next_action
            curr_state = next_state

            terminated = terminated or truncated
            self.post_visit(episode_history, terminated)
            [hook(self) for hook in self.post_visit_hooks]

        return episode_history
