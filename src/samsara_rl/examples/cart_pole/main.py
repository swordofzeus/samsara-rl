"""CartPole experiment driver.

Runs a grid search over feature configs, bias, and learning rates
using linear function approximation with Q-learning.
"""

import os
import pickle
from typing import Any

import gymnasium as gym
import numpy as np
import structlog

from samsara_rl.control.function_approximation.functions.manual.linear import LinearFunction
from samsara_rl.control.function_approximation.online.td.qlearning import QLearningGradient
from samsara_rl.examples.cart_pole.cart_pole_logger import CartPoleLogger
from samsara_rl.mdp.terminal_penalty_wrapper import TerminalPenaltyWrapper

log = structlog.get_logger()

MAX_EPISODES = 100_000
EVAL_EPISODES = 20
BASE_DIR = "log5"
ALPHAS = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]


def scale_angle_only(x: np.ndarray) -> np.ndarray:
    """Extract and normalize pole angle to [-1, 1].

    Args:
        x: Raw CartPole observation [cart_pos, cart_vel, angle, angular_vel].

    Returns:
        Single-element array with normalized angle.
    """
    angle = 2 * (x[2] - (-0.418)) / (0.418 - (-0.418)) - 1
    return np.array([angle])


def scale_angle_angvel(x: np.ndarray) -> Any:
    """Extract and normalize pole angle and angular velocity to [-1, 1].

    Angular velocity is clipped to [-5, 5] before scaling.

    Args:
        x: Raw CartPole observation [cart_pos, cart_vel, angle, angular_vel].

    Returns:
        Two-element array with normalized angle and angular velocity.
    """
    clipped_angvel = np.clip(x[3], -5, 5)
    features = np.array([x[2], clipped_angvel])
    mins = np.array([-0.418, -5])
    maxs = np.array([0.418, 5])
    return 2 * (features - mins) / (maxs - mins) - 1


FEATURE_CONFIGS = [
    {"name": "angle_only", "scale_fn": scale_angle_only, "feature_count": 1},
    {"name": "angle_angvel", "scale_fn": scale_angle_angvel, "feature_count": 2},
]

BIAS_CONFIGS = [False]


def save_model(lf: LinearFunction, run_dir: str) -> None:
    """Persist learned weights to disk as a pickle file.

    Args:
        lf: LinearFunction whose W and B parameters will be saved.
        run_dir: Directory path where weights.pkl will be written.
    """
    path = os.path.join(run_dir, "weights.pkl")
    with open(path, "wb") as f:
        pickle.dump({"W": lf.W.data, "B": lf.B.data}, f)



def run_experiment(env: TerminalPenaltyWrapper, feat_config: dict[str, Any], use_bias: bool, alpha: float) -> None:
    """Train and evaluate a single configuration.

    Creates a LinearFunction and QLearningGradient agent, trains for
    MAX_EPISODES episodes, saves the learned weights, and runs greedy
    evaluation.

    Args:
        env: Gymnasium CartPole environment.
        feat_config: Dict with 'name', 'scale_fn', and 'feature_count' keys.
        use_bias: Whether the linear function includes a bias term.
        alpha: Learning rate for the Q-learning update.
    """
    run_name = f"{feat_config['name']}_bias={use_bias}_alpha={alpha}"
    run_dir = os.path.join(BASE_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log.info("starting_run", run=run_name)

    lf = LinearFunction(
        feature_count=feat_config["feature_count"],
        action_count=2,
        use_bias=use_bias,
        X=feat_config["scale_fn"],
    )

    agent = QLearningGradient(
        mdp=env,
        policy=None,
        gamma=1,
        q=lf,
        log_dir=run_dir,
        experiment_name=run_name,
        epsilon_decay=0.99995,
        epsilon=1,
        alpha=alpha,
    )
    agent.register(CartPoleLogger())

    agent.evaluate(max_iter=MAX_EPISODES)
    save_model(lf, run_dir)

    if agent.tensorboard:
        agent.tensorboard.flush()


def main() -> None:
    """Run grid search over feature configs, bias, and learning rates.

    Iterates over all combinations of FEATURE_CONFIGS, BIAS_CONFIGS,
    and ALPHAS, training and evaluating each one.
    """
    env = TerminalPenaltyWrapper(gym.make("CartPole-v1"))

    for feat_config in FEATURE_CONFIGS:
        for use_bias in BIAS_CONFIGS:
            for alpha in ALPHAS:
                run_experiment(env, feat_config, use_bias, alpha)


if __name__ == "__main__":
    main()
