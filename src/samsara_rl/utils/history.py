import numpy as np
from samsara_rl.utils.gym_utils import state_output_dim, action_output_dim


class History:
    def __init__(self, observation_space, action_space, initial_state):
        self.states = (
            np.zeros((1000, observation_space))
            if observation_space > 2
            else np.zeros(
                1000,
            )
        )
        self.actions = (
            np.zeros((1000, action_space))
            if action_space > 2
            else np.zeros(
                1000,
            )
        )
        self.rewards = np.zeros(1000)
        self.states[0] = initial_state
        self.curr_index = 0

    def current_state(self):
        return self.past_states()[-1]

    def past_states(self):
        return self.states[0 : self.curr_index + 1]

    def past_actions(self):
        return self.actions[0 : self.curr_index + 1]

    def past_rewards(self):
        return self.rewards[0 : self.curr_index + 1]

    def record(self, action, reward, s_prime, a_prime=None):
        self.states[self.curr_index + 1] = s_prime
        self.actions[self.curr_index] = action
        self.rewards[self.curr_index] = reward
        if a_prime:
            self.actions[self.curr_index + 1] = a_prime
        self.curr_index += 1

    @classmethod
    def from_gym(cls, env, s):
        action_space = action_output_dim(env.action_space)
        state_space = state_output_dim(env.observation_space)
        return History(state_space, action_space, s)
