# Samsara RL

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
v_mc = mc.q_table.mean(axis=1).reshape(4, 4)
v_td = td.q_table.mean(axis=1).reshape(4, 4)
```

---

## Model-Free Control

TODO

---

## Function Approximation

TODO
