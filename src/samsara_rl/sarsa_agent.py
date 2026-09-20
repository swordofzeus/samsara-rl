from samsara_rl.agent import Agent
from samsara_rl.utils.memory.episode import Episode


class SARSAAgent(Agent):
    """Agent with a SARSA-style episode loop (S, A, R, S', A').

    Overrides ``run_episode`` to select the next action *before*
    ``post_visit``, so that A' is available in the episode history
    when the TD target Q(S', A') is computed. This is required for
    on-policy SARSA algorithms where the bootstrap target depends
    on the action actually taken under the current policy.

    All other agents use the standard loop in ``Agent``, which selects
    the action *after* ``post_visit`` to ensure it reflects the most
    recent policy update.
    """

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
            curr_state = episode_history.past_states()[-1]
            next_state, reward, terminated, truncated, _ = self.mdp.step(curr_action)
            next_action = self.select_action(next_state)
            episode_history.record(curr_action, reward, next_state, a_prime=next_action)
            curr_action = next_action
            curr_state = next_state

            terminated = terminated or truncated
            self.post_visit(episode_history, terminated)
            [hook(self) for hook in self.post_visit_hooks]

        return episode_history
