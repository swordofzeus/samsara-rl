import multiprocessing
import os
from collections.abc import Callable
from multiprocessing import Pool
from typing import Any

import gymnasium as gym
import numpy as np
import structlog
import torch
import torch.nn as nn
from cart_pole_logger_dyna import CartPoleLoggerDyna

from samsara_rl.control.dyna.deep_dyna_q import DeepDynaQ
from samsara_rl.functions.neural_networks.fully_connected import FullyConnected
from samsara_rl.functions.neural_networks.world_model_interpreter import WorldModelInterpreter
from samsara_rl.mdp.cart_pole.scaled_cart_pole import ScaledCartPole

log = structlog.get_logger()


class CartPoleInterpreter(WorldModelInterpreter):
    def __init__(self):
        self.state_len = 4

    def next_state_logits(self, output):
        return output[:, 0 : self.state_len]

    def predicted_next_state(self, output):
        return output[:, 0 : self.state_len]

    def predicted_reward(self, output):
        return output[:, self.state_len]

    def predicted_terminal(self, output):
        return output[:, self.state_len + 1]

    def parse(self, output):
        return (
            self.next_state_logits(output),
            self.predicted_reward(output),
            self.predicted_terminal(output),
        )


class FullyConnected2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        output_dim: int,
        preprocess: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)
        self.preprocess = preprocess

    def forward(self, x: Any) -> torch.Tensor:
        out: torch.Tensor = self.preprocess(x) if self.preprocess else x
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out).to(torch.float32)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        return self.fc3(out)


MAX_EPISODES = 2000
BASE_DIR = "logs/deep_dyna_q"

ALPHA = 0.0005
GAMMA = 0.99
MODEL_PLAN_STEPS = [0, 2, 5]


def run_experiment(model_plan_steps: int) -> None:
    model_name = "deep_dyna_q"
    run_name = f"{model_name}_plan_steps={model_plan_steps}"
    run_dir = os.path.join(BASE_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log.info("starting_run", run=run_name)

    env = ScaledCartPole(gym.make("CartPole-v1"))
    cart_pole_wm = FullyConnected(6, 32, 6)
    q = FullyConnected2(4, 32, 16, 2)
    cart_pole_wm_interpreter = CartPoleInterpreter()

    agent = DeepDynaQ(
        mdp=env,
        world_model=cart_pole_wm,
        world_model_interpreter=cart_pole_wm_interpreter,
        model_plan_steps=model_plan_steps,
        count_planning_updates=True,
        model_train_steps=10,
        warmup=800,
        gamma=GAMMA,
        q=q,
        alpha=ALPHA,
        state_loss=torch.nn.MSELoss(),
        target_update_freq=500,
        epsilon_episodes=800,
        epsilon_min=0.01,
        lr_step_size=100,
        lr_gamma=0.3,
        log_dir=run_dir,
        experiment_name=run_name,
        batch_size=128,
        buffer_size=10000,
    )
    agent.register(CartPoleLoggerDyna())
    agent.evaluate(max_iter=MAX_EPISODES)

    if agent.tensorboard:
        agent.tensorboard.flush()

    log.info("finished_run", run=run_name)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    num_workers = min(len(MODEL_PLAN_STEPS), os.cpu_count() or 4)
    log.info("starting_parallel_training", total_runs=len(MODEL_PLAN_STEPS), workers=num_workers)

    with Pool(num_workers) as pool:
        pool.map(run_experiment, MODEL_PLAN_STEPS)

    log.info("all_runs_complete")
