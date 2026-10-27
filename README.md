# Samsara RL

<p align="center">
  <img src="img/samsara-rl.png" alt="Samsara RL" width="200">
</p>

[![CI](https://github.com/swordofzeus/samsara-rl/actions/workflows/main.yml/badge.svg)](https://github.com/swordofzeus/samsara-rl/actions/workflows/main.yml)

A vectorized NumPy implementation of foundational reinforcement learning algorithms, following David Silver's RL lecture series, Sutton and Barto book and other papers cited when referenced. Built for clarity, learning as well as speed.

Applications of RL include robotic manipulation, LLM fine-tuning, financial portfolio management, and control systems.

---

## Algorithms

| Algorithm | Category | How It Works | Good For | Limitations |
|-----------|----------|-------------|----------|-------------|
| [Policy Iteration](#policy-iteration) | Planning | Alternates policy evaluation (Bellman expectation) and greedy improvement until convergence | Small MDPs with known dynamics | Requires full model (transition + reward matrices) |
| [Value Iteration](#value-iteration) | Planning | Applies Bellman optimality equation directly; extracts policy at convergence | Small MDPs with known dynamics | Requires full model; slower per-iteration than policy iteration for large state spaces |
| [Monte Carlo](#monte-carlo) | Tabular Prediction | Estimates Q(s, a) from sampled episode returns using constant-alpha updates | Episodic tasks; unbiased value estimates | High variance; must wait until episode end |
| [TD(λ)](#tdλ) | Tabular Prediction | One-step bootstrapping with eligibility traces for online Q updates | Online learning; continuous tasks | Biased estimates from bootstrapping |
| [SARSA](#sarsa) | Tabular Control | On-policy TD control; bootstraps from action actually taken under ε-greedy | Safe exploration; risk-sensitive tasks | Learns ε-greedy value, not optimal value |
| [Q-Learning](#q-learning) | Tabular Control | Off-policy TD control; bootstraps from max Q(S', a) regardless of action taken | Learning optimal policy while exploring | Maximization bias; can overestimate Q values |
| [Linear Semi-Gradient TD(λ)](#semi-gradient-tdλ-control) | Value Approximation | TD(λ) with linear function approximation and eligibility traces | Large/continuous state spaces with known features | Linear capacity; requires manual feature engineering |
| [DQN](#deep-q-network) | Value Approximation | Neural network Q-function with replay buffer and target network | High-dimensional continuous state spaces | Maximization bias; training instability |
| [Double DQN](#deep-q-network) | Value Approximation | DQN with decoupled action selection and evaluation to reduce overestimation | Same as DQN with more stable Q estimates | Still sensitive to hyperparameters |
| [Monte Carlo Policy Gradient](#monte-carlo-policy-gradient) | Policy Gradient | Accumulates policy gradients over a batch of episodes using discounted returns, then updates the policy network | Episodic tasks; continuous or large state spaces | High variance; must wait until episode end; sensitive to baseline choice |
| [REINFORCE](#reinforce) | Policy Gradient | Special case of MC Policy Gradient with batch_size=1; updates after every episode | Simple episodic tasks; learning/prototyping | Highest variance; no gradient averaging across episodes |
| [TD Actor-Critic](#actor-critic) | Actor Critic | Separate actor (policy) and critic (value) networks; critic provides TD-based advantage signal to guide policy updates | Continuous state spaces; lower variance than REINFORCE | Sensitive to learning rate ratio between actor and critic; value function can have stale estimates for unvisited states |
| [Deep Dyna-Q](#deep-dyna-q) | Dyna (Model-Based + Model-Free) | Learns a world model from real transitions, then generates simulated experience to accelerate Q-network training via Dyna-style planning | Sample-efficient learning; environments where real interactions are expensive | World model errors compound during planning; sensitive to model accuracy |
| Monte Carlo Tree Search | Forward Search | *Next release* | *Next release* | *Next release* |

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Planning](#planning)
   - [MDP Structure](#mdp-structure)
   - [Policy Iteration](#policy-iteration)
   - [Value Iteration](#value-iteration)
4. [Model-Free Prediction](#model-free-prediction)
   - [Monte Carlo](#monte-carlo)
   - [TD Learning](#td-learning)
5. [Model-Free Control](#model-free-control)
   - [SARSA](#sarsa)
   - [Q-Learning](#q-learning)
6. [Function Approximation](#function-approximation)
   - [Linear Function Approximation](#linear-function-approximation)
   - [Semi-Gradient TD(λ) Control](#semi-gradient-tdλ-control)
   - [SARSA (Function Approximation)](#sarsa-function-approximation)
   - [Q-Learning (Function Approximation)](#q-learning-function-approximation)
7. [Deep Q-Network](#deep-q-network)
8. [Deep Dyna-Q](#deep-dyna-q)
9. [Policy Gradient](#policy-gradient)
   - [Monte Carlo Policy Gradient](#monte-carlo-policy-gradient)
   - [REINFORCE](#reinforce)
10. [Actor-Critic](#actor-critic)

---

## Installation

```bash
pip install samsara-rl
```

---

## Quick Start

```python
from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
from samsara_rl.planning.policy_iteration import PolicyIteration

mdp = GridWorldMDP()
pi = PolicyIteration(mdp)
policy = pi.find_optimal_policy()
```

---

## Planning

Planning algorithms assume full knowledge of environment dynamics (transition probabilities and reward function). While not "true RL" — agents never have access to dynamics in practice — planning provides the theoretical foundation all RL algorithms build on.

### MDP Structure

MDPs are represented as NumPy arrays. The included `GridWorldMDP` implements the 4x4 grid world from David Silver's Lecture 3.

| Attribute                        | Shape        | Description                                      |
|----------------------------------|--------------|--------------------------------------------------|
| `state_action_transition_matrix` | `(S, A, S')` | T(s, a, s') — transition probabilities           |
| `reward_matrix`                  | `(S, A, S')` | R(s, a, s') — reward for each transition         |

### Policy Iteration

Alternates between evaluating the current policy using the Bellman expectation equation and improving it greedily until the policy stops changing.

**`PolicyIteration(mdp, bellman_tolerance)`**

| Argument            | Type    | Default | Description                                 |
|---------------------|---------|---------|---------------------------------------------|
| `mdp`               | MDP     |         | MDP instance to solve                       |
| `bellman_tolerance` | `float` | `0.01`  | Convergence threshold for policy evaluation |

**`find_optimal_policy(max_iter)`**

| Argument   | Type  | Default | Description                              |
|------------|-------|---------|------------------------------------------|
| `max_iter` | `int` | `99`    | Maximum number of policy iteration steps |

### Value Iteration

Applies the Bellman optimality equation directly each iteration. Equivalent to policy iteration with k=1 evaluation steps per improvement. Policy is extracted once at convergence.

**`ValueIteration(mdp, bellman_tolerance)`**

| Argument            | Type    | Default | Description                               |
|---------------------|---------|---------|-------------------------------------------|
| `mdp`               | MDP     |         | MDP instance to solve                     |
| `bellman_tolerance` | `float` | `0.01`  | Convergence threshold for value iteration |

**Examples**

```python
from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
from samsara_rl.planning.policy_iteration import PolicyIteration
from samsara_rl.planning.value_iteration import ValueIteration

mdp = GridWorldMDP()

policy = PolicyIteration(mdp, bellman_tolerance=0.001).find_optimal_policy(max_iter=50)
policy = ValueIteration(mdp, bellman_tolerance=0.001).find_optimal_policy()
```

---

## Model-Free Prediction

Model-free methods learn value functions directly from experience (sampled episodes) without access to the MDP's transition or reward dynamics.

### Monte Carlo

Every-visit Monte Carlo prediction estimates Q(s, a) from sampled returns. After each episode, the return G_t (discounted cumulative reward from time step t onward) is computed for every visited state-action pair, and the Q-table is updated using constant-alpha learning:

Q(s, a) <- Q(s, a) + α (G_t - Q(s, a))

If the same (s, a) pair appears multiple times in an episode, each occurrence triggers an update. Returns are computed in a fully vectorized pass using a cumulative-sum trick that avoids the standard reverse loop over time steps.

**`MonteCarloPolicyEvaluation(mdp, policy, alpha, gamma)`**

| Argument | Type    | Default | Description                              |
|----------|---------|---------|------------------------------------------|
| `mdp`    | MDP     |         | MDP instance to sample episodes from     |
| `policy` | `array` |         | Stochastic policy of shape `(S, A)`      |
| `alpha`  | `float` | `0.01`  | Learning rate for incremental Q updates  |
| `gamma`  | `float` | `1`     | Discount factor                          |

**`evaluate(max_iter)`**

| Argument   | Type  | Default | Description                          |
|------------|-------|---------|--------------------------------------|
| `max_iter` | `int` | `10000` | Number of episodes to sample from    |

### TD(λ)

TD(λ) learns Q(s, a) online using one-step bootstrapping with eligibility traces. After each step, the TD error is computed against the expected Q-value of the next state under the current policy, and all previously visited state-action pairs are updated proportionally to their eligibility:

δ = R + γ E_π[Q(S', ·)] - Q(S, A)

Q(s, a) <- Q(s, a) + α δ e(s, a)

Eligibility traces use the replacing variant — on each visit to (s, a), the trace is set to 1 rather than incremented. All traces decay by γλ at each time step. Traces are reset to zero between episodes.

**`TDPolicyEvaluation(mdp, policy, alpha, gamma, _lambda)`**

| Argument  | Type    | Default | Description                                    |
|-----------|---------|---------|------------------------------------------------|
| `mdp`     | MDP     |         | MDP instance to sample episodes from           |
| `policy`  | `array` |         | Stochastic policy of shape `(S, A)`            |
| `alpha`   | `float` | `0.01`  | Learning rate for incremental Q updates        |
| `gamma`   | `float` | `0.9`   | Discount factor                                |
| `_lambda` | `float` | `0.4`   | Trace decay parameter (0 = TD(0), 1 = TD(1))  |

**`evaluate(max_iter)`**

| Argument   | Type  | Default | Description                          |
|------------|-------|---------|--------------------------------------|
| `max_iter` | `int` | `1000`  | Number of episodes to sample from    |

**Examples**

```python
from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
from samsara_rl.prediction.monte_carlo import MonteCarloPolicyEvaluation
from samsara_rl.prediction.td import TDPolicyEvaluation
from samsara_rl.utils.policy.policy_utils import init_uniform_random

mdp = GridWorldMDP()
policy = init_uniform_random(mdp)

mc = MonteCarloPolicyEvaluation(mdp, policy, alpha=0.01, gamma=0.9)
mc.evaluate(max_iter=10000)

td = TDPolicyEvaluation(mdp, policy, alpha=0.01, gamma=0.9, _lambda=0.4)
td.evaluate(max_iter=10000)

# V(s) for the random policy (expected value over actions)
v_mc = mc.q.mean(axis=1).reshape(4, 4)
v_td = td.q.mean(axis=1).reshape(4, 4)
```

---

## Model-Free Control

Control algorithms learn an optimal policy by interleaving evaluation and improvement on every step. Both SARSA and Q-Learning build on the TD(λ) engine, using ε-greedy exploration to balance exploitation with discovery of new state-action pairs.

### SARSA

On-policy TD control. Bootstraps from a sampled next action A' drawn from the current policy — the name comes from the quintuple (S, A, R, S', A'). Because the bootstrap target reflects the exploratory policy, SARSA's Q values account for the cost of occasional random actions.

δ = R + γ Q(S', A') - Q(S, A)

**`Sarsa(mdp, alpha, gamma)`**

| Argument | Type    | Default | Description                              |
|----------|---------|---------|------------------------------------------|
| `mdp`    | MDP     |         | MDP instance to sample episodes from     |
| `alpha`  | `float` | `0.01`  | Learning rate for incremental Q updates  |
| `gamma`  | `float` | `0.9`   | Discount factor                          |

**`evaluate(max_iter)`**

| Argument   | Type  | Default | Description                          |
|------------|-------|---------|--------------------------------------|
| `max_iter` | `int` | `5000`  | Number of episodes to run            |

### Q-Learning

Off-policy TD control. Bootstraps from the greedy action max_a Q(S', a) regardless of the action actually taken. This means Q-Learning converges to the optimal Q* even while following an exploratory ε-greedy policy.

δ = R + γ max_a Q(S', a) - Q(S, A)

**`QLearning(mdp, alpha, gamma)`**

| Argument | Type    | Default | Description                              |
|----------|---------|---------|------------------------------------------|
| `mdp`    | MDP     |         | MDP instance to sample episodes from     |
| `alpha`  | `float` | `0.01`  | Learning rate for incremental Q updates  |
| `gamma`  | `float` | `0.9`   | Discount factor                          |

**`evaluate(max_iter)`**

| Argument   | Type  | Default | Description                          |
|------------|-------|---------|--------------------------------------|
| `max_iter` | `int` | `5000`  | Number of episodes to run            |

**Examples**

```python
from samsara_rl.mdp.grid_world.grid_world_gym import GridWorldMDP
from samsara_rl.control.tabular.sarsa import Sarsa
from samsara_rl.control.tabular.q_learning import QLearning

mdp = GridWorldMDP()

sarsa = Sarsa(mdp, alpha=0.01, gamma=0.9)
sarsa.evaluate(max_iter=5000)

ql = QLearning(mdp, alpha=0.01, gamma=0.9)
ql.evaluate(max_iter=5000)

# Optimal value per state (best action)
v_sarsa = sarsa.q.max(axis=1).reshape(4, 4)
v_ql = ql.q.max(axis=1).reshape(4, 4)
```

---

## Function Approximation

Tabular methods store one value per state-action pair — this breaks down when the state space is large or continuous (e.g. CartPole's 4D observation vector). Function approximation replaces the Q-table with a parameterized function Q(s, a; **w**) that generalizes across states.

### Linear Function Approximation

`LinearFunction` implements Q(s) = X(s)^T **W**, where X is a user-provided feature extraction function and **W** is a learned weight matrix. It exposes a PyTorch-style interface: forward pass via `__call__`, gradient computation via `backward()`, and parameter access via `params`.

For discrete environments, a one-hot encoding X(s) gives the linear approximator the same representational power as a tabular method — useful as a sanity check before moving to richer feature representations.

**`LinearFunction(feature_count, action_count, X, use_bias)`**

| Argument        | Type       | Default    | Description                                         |
|-----------------|------------|------------|-----------------------------------------------------|
| `feature_count` | `int`      |            | Number of input features (output dimension of X)    |
| `action_count`  | `int`      |            | Number of discrete actions                          |
| `X`             | `Callable` | `identity` | Feature extraction function: state → feature vector |
| `use_bias`      | `bool`     | `False`    | Whether to include a bias term per action           |

### Semi-Gradient TD(λ) Control

`TemporalDifferenceGradient` implements semi-gradient TD(λ) control with eligibility traces. On each step, the TD error is computed and used to update the function approximator's parameters in the direction of the gradient, scaled by eligibility traces that assign credit to recently visited state-action pairs.

The TD target function is configurable — SARSA and Q-Learning are implemented as thin subclasses that fix the target.

**`TemporalDifferenceGradient(mdp, alpha, gamma, q, _lambda)`**

| Argument  | Type             | Default  | Description                                    |
|-----------|------------------|----------|------------------------------------------------|
| `mdp`     | `gym.Env`        |          | Gymnasium-compatible environment               |
| `alpha`   | `float`          | `0.001`  | Learning rate                                  |
| `gamma`   | `float`          | `1`      | Discount factor                                |
| `q`       | `LinearFunction` |          | Function approximator                          |
| `_lambda` | `float`          | `0.2`    | Eligibility trace decay (0 = TD(0), 1 = MC)   |

### SARSA (Function Approximation)

On-policy control. Bootstraps from Q(S', A') where A' is the action actually taken under the current ε-greedy policy.

**`SarsaGradient(**kwargs)`** — accepts the same arguments as `TemporalDifferenceGradient`.

### Q-Learning (Function Approximation)

Off-policy control. Bootstraps from max_a Q(S', a), learning the optimal policy regardless of exploration behavior.

**`QLearningGradient(**kwargs)`** — accepts the same arguments as `TemporalDifferenceGradient`.

**Examples**

```python
import numpy as np
from samsara_rl.mdp.grid_world.grid_world_gym import GridWorldMDP
from samsara_rl.control.function_approximation.functions.linear import LinearFunction
from samsara_rl.control.function_approximation.sarsa import SarsaGradient

mdp = GridWorldMDP()

# One-hot encoding gives tabular-equivalent capacity
def one_hot(s):
    arr = np.zeros(16)
    arr[int(s)] = 1
    return arr

q_fn = LinearFunction(16, 4, one_hot)

# SARSA with function approximation
sarsa = SarsaGradient(mdp=mdp, gamma=0.999, q=q_fn, alpha=0.01)
sarsa.evaluate(max_iter=20000)

# Learned value per state
v = np.array([q_fn.W.value[s].max() for s in range(16)]).reshape(4, 4)
```

---

## Deep Q-Network

Deep Q-Networks (DQN) replace the linear function approximator with a neural network, enabling learning in high-dimensional continuous state spaces. Two key innovations stabilize training:

- **Experience Replay** — transitions are stored in a replay buffer and sampled in random mini-batches, breaking temporal correlations in the training data.
- **Target Network** — a frozen copy of the Q-network provides stable TD targets. It is periodically updated to match the online network.

The target computation strategy is configurable. Standard DQN uses the target network for both action selection and evaluation. **Double DQN** decouples these — the online network selects the action, and the target network evaluates it — reducing the maximization bias that causes Q-value overestimation and training instability.

**`DQNAgent(mdp, alpha, gamma, q, target, target_update_freq, epsilon, epsilon_decay, loss_fn, batch_size)`**

| Argument             | Type         | Default      | Description                                         |
|----------------------|--------------|--------------|-----------------------------------------------------|
| `mdp`                | `gym.Env`    |              | Gymnasium-compatible environment                    |
| `alpha`              | `float`      | `0.001`      | Learning rate for Adam optimizer                    |
| `gamma`              | `float`      | `1`          | Discount factor                                     |
| `q`                  | `nn.Module`  |              | Neural network mapping states to Q values           |
| `target`             | `object`     | `DQNTarget`  | Target computation strategy (`DQNTarget` or `DoubleDQNTarget`) |
| `target_update_freq` | `int`        | `1000`       | Training steps between target network swaps         |
| `epsilon`            | `float`      | `1`          | Initial exploration rate for ε-greedy               |
| `epsilon_decay`      | `float`      | `0.999`      | Multiplicative decay applied to epsilon each episode |
| `loss_fn`            | `Callable`   | `MSELoss`    | Loss function (e.g. `MSELoss`, `HuberLoss`)        |
| `batch_size`         | `int`        | `128`        | Number of transitions per training mini-batch       |

**Examples**

```python
import gymnasium as gym
from samsara_rl.control.function_approximation.batch.deep_q_network.q_network import DQNAgent
from samsara_rl.control.function_approximation.batch.deep_q_network.targets.double_d_target import DoubleDQNTarget
from samsara_rl.control.function_approximation.functions.neural_networks.fully_connected import FullyConnected
from samsara_rl.mdp.cart_pole.scaled_cart_pole import ScaledCartPole

env = ScaledCartPole(gym.make("CartPole-v1"))
network = FullyConnected(4, 32, 2)

# Standard DQN
agent = DQNAgent(
    mdp=env, gamma=0.99, q=network, alpha=0.0001,
    log_dir="logs/dqn", experiment_name="cartpole_dqn",
)
agent.evaluate(max_iter=3000)

# Double DQN — swap the target strategy
agent = DQNAgent(
    mdp=env, gamma=0.99, q=network, alpha=0.0001,
    target=DoubleDQNTarget(),
    log_dir="logs/double_dqn", experiment_name="cartpole_double_dqn",
)
agent.evaluate(max_iter=3000)
```

For a full walkthrough with TensorBoard logging, decision surface visualization, and hyperparameter tuning, see the [Deep Q-Network tutorial notebook](examples/deep_q_networks/deep_q_tutorial.ipynb).

---

## Deep Dyna-Q

Adapted from [Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning](https://arxiv.org/abs/1801.06176) (Peng et al., 2018). The original paper couples the Dyna-Q framework to a dialogue task with domain-specific components. This implementation extracts the core idea — learning a world model from real transitions and using it to generate simulated experience for planning — and generalizes it to work with any Gymnasium environment.

The agent extends DQN with three additional components:

1. **World Model** — a neural network trained to predict (next_state, reward, terminal) given (state, action). The `WorldModelInterpreter` abstraction decouples the model's raw output from how it is parsed, allowing the same agent to work with discrete state spaces (GridWorld) and continuous ones (CartPole).

2. **Simulated Experience Buffer** — after the world model is sufficiently trained (controlled by the `warmup` parameter), the agent generates simulated episodes by rolling out the world model from states sampled from the real replay buffer.

3. **Planning Loop** — the Q-network is trained on mini-batches from the simulated buffer, providing additional learning signal beyond real experience. The number of planning steps per real transition is controlled by `model_plan_steps`.

**`DeepDynaQ(mdp, world_model, world_model_interpreter, ...)`**

| Argument                  | Type                    | Default      | Description                                                        |
|---------------------------|-------------------------|--------------|--------------------------------------------------------------------|
| `mdp`                     | `gym.Env`               |              | Gymnasium-compatible environment                                   |
| `world_model`             | `nn.Module`             |              | Neural network predicting (next_state, reward, terminal)           |
| `world_model_interpreter` | `WorldModelInterpreter` |              | Parses world model output into state, reward, and terminal signals |
| `alpha`                   | `float`                 | `0.001`      | Learning rate for both Q-network and world model                   |
| `gamma`                   | `float`                 | `1`          | Discount factor                                                    |
| `q`                       | `nn.Module`             |              | Q-network mapping states to action values                          |
| `model_plan_steps`        | `int`                   | `5`          | Q-network training steps on simulated data per real transition     |
| `model_train_steps`       | `int`                   | `20`         | World model gradient steps per real transition                     |
| `warmup`                  | `int`                   | `20`         | Episodes before planning begins                                    |
| `wm_batch_size`           | `int`                   | `32`         | Mini-batch size for world model training                           |
| `state_loss`              | `Callable`              | `MSELoss`    | Loss for next-state prediction (use `CrossEntropyLoss` for discrete states) |

Inherits all parameters from `DQNAgent` (epsilon, target network, replay buffer, LR scheduling, etc.).

**Examples**

```python
import gymnasium as gym
import torch
from samsara_rl.control.dyna.deep_dyna_q import DeepDynaQ
from samsara_rl.functions.neural_networks.fully_connected import FullyConnected
from samsara_rl.functions.neural_networks.world_model_interpreter import WorldModelInterpreter
from samsara_rl.mdp.cart_pole.scaled_cart_pole import ScaledCartPole

env = ScaledCartPole(gym.make("CartPole-v1"))

# Q-network: 4 observations -> 2 actions
q_network = FullyConnected(4, 32, 2)

# World model: 4 observations + 2 one-hot action -> 6 outputs (4 next_state + reward + terminal)
world_model = FullyConnected(6, 32, 6)

# Interpreter for continuous state spaces
class CartPoleInterpreter(WorldModelInterpreter):
    def next_state_logits(self, output):
        return output[:, 0:4]
    def predicted_next_state(self, output):
        return output[:, 0:4]
    def predicted_reward(self, output):
        return output[:, 4]
    def predicted_terminal(self, output):
        return output[:, 5]
    def parse(self, output):
        return self.next_state_logits(output), self.predicted_reward(output), self.predicted_terminal(output)

agent = DeepDynaQ(
    mdp=env, gamma=0.99, alpha=0.0005, q=q_network,
    world_model=world_model,
    world_model_interpreter=CartPoleInterpreter(),
    model_plan_steps=5, model_train_steps=10, warmup=300,
    state_loss=torch.nn.MSELoss(),
    epsilon_episodes=800, epsilon_min=0.01,
    buffer_size=10000, target_update_freq=500,
    log_dir="logs/dyna_q", experiment_name="cartpole_dyna_q",
)
agent.evaluate(max_iter=2000)
```

For a full walkthrough with Q-surface visualization, world model analysis, and hyperparameter tuning, see the [Deep Dyna-Q tutorial notebook](examples/deep_dyna_q/DeepDynaQTutorial.ipynb).

---

## Policy Gradient

Policy gradient methods learn a parameterized policy directly, rather than deriving it from a value function. The policy network outputs action probabilities via softmax, and gradient ascent maximizes the expected return. Unlike value-based methods (Q-Learning, DQN), policy gradients can naturally represent stochastic policies and scale to continuous action spaces.

### Monte Carlo Policy Gradient

Accumulates policy gradients over a batch of episodes before performing an optimizer step. For each episode, discounted returns are computed for every time step, and the policy gradient loss weights the log-probability of each taken action by its return. An optional running average baseline reduces variance by centering returns around their expected value.

**`MonteCarloPolicyGradient(mdp, alpha, gamma, policy_network, optimizer, batch_size, use_advantage)`**

| Argument         | Type             | Default | Description                                                     |
|------------------|------------------|---------|-----------------------------------------------------------------|
| `mdp`            | `gym.Env`        |         | Gymnasium-compatible environment                                |
| `alpha`          | `float`          |         | Learning rate                                                   |
| `gamma`          | `float`          |         | Discount factor                                                 |
| `policy_network` | `nn.Module`      |         | Neural network that maps states to action logits                |
| `optimizer`      | `Optimizer`      | `Adam`  | Optimizer for the policy network                                |
| `batch_size`     | `int`            | `1`     | Number of episodes to accumulate gradients over before stepping |
| `use_advantage`  | `bool`           | `True`  | Subtract a running average baseline to reduce variance          |

### REINFORCE

REINFORCE (Williams, 1992) is a special case of Monte Carlo Policy Gradient where the optimizer steps after every episode (batch_size=1).

**`Reinforce(mdp, alpha, gamma, policy_network, optimizer, use_advantage)`**

Accepts the same arguments as `MonteCarloPolicyGradient`, without `batch_size`.

**Examples**

```python
import gymnasium as gym
from samsara_rl.control.function_approximation.batch.monte_carlo_policy_gradient.monte_carlo_policy_gradient import MonteCarloPolicyGradient
from samsara_rl.control.function_approximation.online.reinforce.reinforce import Reinforce
from samsara_rl.control.function_approximation.functions.neural_networks.fully_connected import FullyConnected
from samsara_rl.mdp.cart_pole.scaled_cart_pole import ScaledCartPole

env = ScaledCartPole(gym.make("CartPole-v1"))
network = FullyConnected(4, 32, 2)

# Monte Carlo Policy Gradient with batch of 32 episodes
agent = MonteCarloPolicyGradient(
    mdp=env, gamma=0.99, alpha=0.002, policy_network=network, batch_size=32,
    log_dir="logs/mc_pg", experiment_name="cartpole_mcpg",
)
agent.evaluate(max_iter=3000)

# REINFORCE — updates every episode
agent = Reinforce(
    mdp=env, gamma=0.99, alpha=0.002, policy_network=network,
    log_dir="logs/reinforce", experiment_name="cartpole_reinforce",
)
agent.evaluate(max_iter=3000)
```

For a full walkthrough, see the [REINFORCE tutorial notebook](examples/reinforce/reinforce_cart_pole.ipynb).

---

## Actor-Critic

Actor-Critic methods maintain two separate function approximators that work together:

- **Actor (Policy Network)** — outputs a probability distribution over actions given a state, parameterized as π(a|s; θ). Actions are sampled from this distribution during training.
- **Critic (Value Network)** — estimates the state value V(s; w). Used to compute the advantage signal that guides policy updates.

On each episode, the critic computes TD targets and updates toward them via MSE loss:

**TD Target**: δ = R + γ V(S'; w)

**Critic Loss**: L_critic = (V(S; w) - δ)²

The advantage estimate tells the actor whether an action was better or worse than expected:

**Advantage**: A(S) = R + γ V(S'; w) - V(S; w)

The actor is updated using the policy gradient, scaled by the advantage:

**Policy Loss**: L_actor = -log(π(a|s; θ)) · A(S)

Positive advantage reinforces the action (makes it more likely), negative advantage discourages it. The negative sign converts gradient ascent into gradient descent for the optimizer.

V(S') must be **detached** from the computation graph when computing the TD target. Otherwise gradients flow through both V(S) and V(S'), causing the critic to push V(S) toward the target AND push V(S') away from it simultaneously. The advantage must also be detached before computing the policy loss — it is a fixed signal for the actor, not something to differentiate through.

**`ValueFunctionActorCritic(mdp, alpha, gamma, policy_network, value_network, policy_optimizer, value_optimizer, batch_size)`**

| Argument           | Type        | Default | Description                                                     |
|--------------------|-------------|---------|-----------------------------------------------------------------|
| `mdp`              | `gym.Env`   |         | Gymnasium-compatible environment                                |
| `alpha`            | `float`     | `0.01`  | Learning rate (used as default if no optimizer provided)        |
| `gamma`            | `float`     | `0.9`   | Discount factor                                                 |
| `policy_network`   | `nn.Module` |         | Neural network that maps states to action logits                |
| `value_network`    | `nn.Module` |         | Neural network that maps states to a single value estimate      |
| `policy_optimizer` | `Optimizer` | `Adam`  | Optimizer for the policy network                                |
| `value_optimizer`  | `Optimizer` | `Adam`  | Optimizer for the value network                                 |
| `batch_size`       | `int`       | `32`    | Number of episodes to accumulate gradients over before stepping |

The critic learning rate should be faster than the actor (1.5–3x) so the critic can provide accurate advantage estimates before the policy shifts too far.

**Examples**

```python
import gymnasium as gym
import torch
from samsara_rl.control.actor_critic.value_function_actor_critic import ValueFunctionActorCritic
from samsara_rl.functions.neural_networks.fully_connected import FullyConnected
from samsara_rl.mdp.cart_pole.scaled_cart_pole import ScaledCartPole

env = ScaledCartPole(gym.make("CartPole-v1"))
value_network = FullyConnected(4, 32, 1)
policy_network = FullyConnected(4, 32, 2)

agent = ValueFunctionActorCritic(
    mdp=env, gamma=0.9, alpha=0.025, batch_size=32,
    value_network=value_network, policy_network=policy_network,
    value_optimizer=torch.optim.Adam(value_network.parameters(), lr=0.025 * 1.75),
    policy_optimizer=torch.optim.Adam(policy_network.parameters(), lr=0.025),
    log_dir="logs/actor_critic", experiment_name="cartpole_ac",
)
agent.evaluate(max_iter=3000)
```

For a full walkthrough with learning phase analysis and value surface visualization, see the [Actor-Critic tutorial notebook](examples/actor_critic/actor_critic.ipynb).
