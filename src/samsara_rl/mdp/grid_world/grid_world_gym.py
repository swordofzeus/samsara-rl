import gymnasium as gym
from enum import Enum
from gymnasium import spaces
import numpy as np


class GridWorldActions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorldMDP(gym.Env):
    """
    Sample 4x4 Grid world as described in David Silvers 3rd RL Lecture.
    Consists of 4 actions, UP,DOWN,LEFT,RIGHT. Fully deterministic, transition probability is 1.0 for an action (i.e. UP will always
    move up.). Default initial policy is uniform random.
    @author ashish juneja
    """

    ACTION_COUNT = 4
    STATE_COUNT = 16

    def __init__(self) -> None:
        self.terminal_states: list[tuple[int, int] | int] = [(0, 0), (3, 3), 0, 15]
        self.state_action_transition_matrix = np.zeros((16, 4, 16))
        self.actions = {
            GridWorldActions.UP.value: self.up,
            GridWorldActions.DOWN.value: self.down,
            GridWorldActions.LEFT.value: self.left,
            GridWorldActions.RIGHT.value: self.right,
        }
        self.reward_matrix = np.full((16, 4, 16), -1)
        self.curr_state = self.initial_state()
        self.init_transition_probabilities()
        self.init_reward_matrix()


        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(16)

    def move(
        self, state: tuple[int, int], x_direction: int, y_direction: int
    ) -> list[tuple[int, int]]:
        next_step = (state[0] + x_direction, state[1] + y_direction)
        if state in self.terminal_states or not self.in_bounds(
            next_step[0], next_step[1]
        ):
            return [state]
        else:
            return [next_step]

    def up(self, state: tuple[int, int]) -> list[tuple[int, int]]:
        return self.move(state, x_direction=-1, y_direction=0)

    def down(self, state: tuple[int, int]) -> list[tuple[int, int]]:
        return self.move(state, x_direction=1, y_direction=0)

    def left(self, state: tuple[int, int]) -> list[tuple[int, int]]:
        return self.move(state, x_direction=0, y_direction=-1)

    def right(self, state: tuple[int, int]) -> list[tuple[int, int]]:
        return self.move(state, x_direction=0, y_direction=1)

    def step(self, action: int) -> tuple[int, int]:
        curr_state_transitions = self.state_action_transition_matrix[self.curr_state][
            action
        ]
        next_state = np.random.choice(
            len(curr_state_transitions), p=curr_state_transitions
        )
        reward = self.reward_matrix[self.curr_state][action][next_state]
        self.curr_state = next_state
        # return reward, next_state
        return (
            self.curr_state,
            reward,
            self.is_terminal_state(next_state),
            False,
            {},
        )

    def init_reward_matrix(self) -> None:
        self.reward_matrix[0, :, 0] = 0
        self.reward_matrix[15, :, 15] = 0

    def init_transition_probabilities(self) -> None:
        """
        Initialize a 16x4x16 numpy matrix that captures the GridWorld
        Deterministic Dynamics. 16 possible starting states, 4 actions from each of those states
        and 16 possible states the agent can end up in after invoking one of the 4 UP,DOWN,LEFT,RIGHT
        actions. All actions are deterministic, ex: UP has 100% probability of upward movement and 0% into any
        of the other 15 states. At boarder collisions agent ends up in current state before action
        """
        for curr_state_next_action in np.ndindex(
            self.state_action_transition_matrix.shape[:-1]
        ):
            curr_state, action = curr_state_next_action
            curr_row, curr_col = self._unflatten(curr_state)
            next_state = self.actions[action]((curr_row, curr_col))[0]
            flattened_next_state = self._flatten(next_state[0], next_state[1])
            self.state_action_transition_matrix[
                curr_state, action, flattened_next_state
            ] = 1

    def in_bounds(self, x: int, y: int) -> bool:
        return x in range(0, 4) and y in range(0, 4)

    def _unflatten(self, state: int) -> tuple[int, int]:
        """
        Folds the tensor addressable 0-15 state into a 4x4 human readable coordinate
        """
        row = state // 4
        col = state % 4
        return row, col

    def _flatten(self, row: int, col: int) -> int:
        """
        Folds the human readable row,col coordinate into a machine addressable 0..15
        """
        return 4 * row + col

    def initial_state(self) -> int:
        return self._flatten(2, 2)

    def is_terminal_state(self, state: int) -> bool:
        return state in self.terminal_states

    def reset(self):
        self.curr_state = self.initial_state()
        return self.curr_state, {'reward_matrix': self.reward_matrix, 'state_action_transition_matrix': self.state_action_transition_matrix}
