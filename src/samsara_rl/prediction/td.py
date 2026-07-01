from samsara_rl.prediction.policy_evaluation import PolicyEvaluation
import numpy as np


class TemporalDifference(PolicyEvaluation):

    def __init__(self, mdp, policy, alpha=0.01, gamma=0.9, _lambda=0.4):
        super().__init__(mdp, policy, alpha, gamma)
        self.elibility = np.zeros((mdp.STATE_COUNT, self.mdp.ACTION_COUNT))
        self._lambda = _lambda

    def post_episode(self, trajectory):
        self.elibility = np.zeros((self.mdp.STATE_COUNT, self.mdp.ACTION_COUNT))

    def post_visit(self, trajectory):
        if trajectory.shape[0] < 2:
            return
        self.elibility = self.elibility * self._lambda
        visited_state = trajectory[-2][0].astype(np.int8)
        el_action = trajectory[-2][1]
        self.elibility[visited_state][el_action.astype(np.int8)] = 1

        R_prime = trajectory[-2][2]
        S_prime = trajectory[-1][0].astype(np.int8)  # state we ended up in after A
        Q_expectation = self.q_table[S_prime].dot(
            self.policy[S_prime]
        )  # optimal known action from next step

        S = trajectory[-2][0].astype(np.int8)  # state before taking A
        A = trajectory[-2][1].astype(np.int8)  # action we took to end up in S'
        Q = self.q_table[S][A]  # Estimate of how good S,A is before taking

        td_error = (R_prime + self.gamma * Q_expectation) - Q
        self.q_table = self.q_table + self.alpha * self.elibility * td_error
