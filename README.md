# Samsara RL

<p align="center">
  <img src="img/samsara-rl.png" alt="Samsara RL" width="200">
</p>

[![CI](https://github.com/swordofzeus/samsara-rl/actions/workflows/main.yml/badge.svg)](https://github.com/swordofzeus/samsara-rl/actions/workflows/main.yml)

A vectorized NumPy implementation of foundational reinforcement learning algorithms, following David Silver's RL lecture series, Sutton and Barto book and other papers cited when referenced. Built for clarity, learning as well as speed.

Applications of RL include robotic manipulation, LLM fine-tuning, financial portfolio management, and control systems.

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

Q(s, a) ← Q(s, a) + α (G_t − Q(s, a))

If the same (s, a) pair appears multiple times in an episode, each occurrence triggers an update. Returns are computed in a fully vectorized pass using a cumulative-sum trick that avoids the standard reverse loop over time steps.

**`MonteCarloPrediction(mdp, policy, alpha, gamma)`**

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

δ = R + γ E_π[Q(S', ·)] − Q(S, A)

Q(s, a) ← Q(s, a) + α δ e(s, a)

Eligibility traces use the replacing variant — on each visit to (s, a), the trace is set to 1 rather than incremented. All traces decay by γλ at each time step. Traces are reset to zero between episodes.

**`TemporalDifference(mdp, policy, alpha, gamma, _lambda)`**

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
from samsara_rl.prediction.monte_carlo import MonteCarloPrediction
from samsara_rl.prediction.td import TemporalDifference
from samsara_rl.utils.policy.policy_utils import init_uniform_random

mdp = GridWorldMDP()
policy = init_uniform_random(mdp)

mc = MonteCarloPrediction(mdp, policy, alpha=0.01, gamma=0.9)
mc.evaluate(max_iter=10000)

td = TemporalDifference(mdp, policy, alpha=0.01, gamma=0.9, _lambda=0.4)
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

δ = R + γ Q(S', A') − Q(S, A)

**`Sarsa(mdp, policy, alpha, gamma)`**

| Argument | Type    | Default | Description                              |
|----------|---------|---------|------------------------------------------|
| `mdp`    | MDP     |         | MDP instance to sample episodes from     |
| `policy` | `array` |         | Initial stochastic policy of shape `(S, A)` |
| `alpha`  | `float` | `0.01`  | Learning rate for incremental Q updates  |
| `gamma`  | `float` | `0.9`   | Discount factor                          |

**`evaluate(max_iter)`**

| Argument   | Type  | Default | Description                          |
|------------|-------|---------|--------------------------------------|
| `max_iter` | `int` | `5000`  | Number of episodes to run            |

### Q-Learning

Off-policy TD control. Bootstraps from the greedy action max_a Q(S', a) regardless of the action actually taken. This means Q-Learning converges to the optimal Q* even while following an exploratory ε-greedy policy.

δ = R + γ max_a Q(S', a) − Q(S, A)

**`QLearning(mdp, policy, alpha, gamma)`**

| Argument | Type    | Default | Description                              |
|----------|---------|---------|------------------------------------------|
| `mdp`    | MDP     |         | MDP instance to sample episodes from     |
| `policy` | `array` |         | Initial stochastic policy of shape `(S, A)` |
| `alpha`  | `float` | `0.01`  | Learning rate for incremental Q updates  |
| `gamma`  | `float` | `0.9`   | Discount factor                          |

**`evaluate(max_iter)`**

| Argument   | Type  | Default | Description                          |
|------------|-------|---------|--------------------------------------|
| `max_iter` | `int` | `5000`  | Number of episodes to run            |

**Examples**

```python
from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
from samsara_rl.control.tabular.sarsa import Sarsa
from samsara_rl.control.tabular.q_learning import QLearning
from samsara_rl.utils.policy.policy_utils import init_uniform_random

mdp = GridWorldMDP()
policy = init_uniform_random(mdp)

sarsa = Sarsa(mdp, policy, alpha=0.01, gamma=0.9)
sarsa.evaluate(max_iter=5000)

ql = QLearning(mdp, policy, alpha=0.01, gamma=0.9)
ql.evaluate(max_iter=5000)

# Optimal value per state (best action)
v_sarsa = sarsa.agent.q.max(axis=1).reshape(4, 4)
v_ql = ql.agent.q.max(axis=1).reshape(4, 4)
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

**`TemporalDifferenceGradient(mdp, policy, alpha, gamma, q, _lambda)`**

| Argument  | Type             | Default  | Description                                    |
|-----------|------------------|----------|------------------------------------------------|
| `mdp`     | `gym.Env`        |          | Gymnasium-compatible environment               |
| `policy`  | `array`          |          | Stochastic policy of shape `(S, A)`            |
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
from samsara_rl.utils.policy.policy_utils import init_uniform_random

mdp = GridWorldMDP()
policy = init_uniform_random(mdp)

# One-hot encoding gives tabular-equivalent capacity
def one_hot(s):
    arr = np.zeros(16)
    arr[int(s)] = 1
    return arr

q_fn = LinearFunction(16, 4, one_hot)

# SARSA with function approximation
sarsa = SarsaGradient(mdp=mdp, policy=policy, gamma=0.999, q=q_fn, alpha=0.01)
sarsa.evaluate(max_iter=20000)

# Learned value per state
v = np.array([q_fn.W.value[s].max() for s in range(16)]).reshape(4, 4)
```
