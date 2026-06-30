# Samsara RL

A vectorized NumPy implementation of foundational reinforcement learning algorithms, following David Silver's RL lecture series. Built for clarity and learning — each algorithm maps directly to the equations in the lectures.

Applications of RL include robotic manipulation, LLM fine-tuning, financial portfolio management, and adaptive control systems.

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

Monte Carlo prediction estimates Q(s, a) by averaging sampled returns. After each episode, the return G_t (discounted cumulative reward from time step t onward) is computed for every visited state-action pair, and the Q-table is updated using constant-alpha learning:

Q(s, a) ← Q(s, a) + α (G_t − Q(s, a))

This is an every-visit implementation — if the same (s, a) pair appears multiple times in an episode, each occurrence triggers an update. Both every-visit and first-visit MC converge to the true Q^π.

Returns are computed in a fully vectorized pass using a cumulative-sum trick that avoids the standard reverse loop over time steps.

**`MonteCarloPrediction(mdp, policy, alpha, gamma)`**

| Argument | Type    | Default | Description                              |
|----------|---------|---------|------------------------------------------|
| `mdp`    | MDP     |         | MDP instance to sample episodes from     |
| `policy` | `array` |         | Stochastic policy of shape `(S, A)`      |
| `alpha`  | `float` | `0.01`  | Learning rate for incremental Q updates  |
| `gamma`  | `float` | `0.9`   | Discount factor                          |

**`evaluate(max_iter)`**

| Argument   | Type  | Default | Description                          |
|------------|-------|---------|--------------------------------------|
| `max_iter` | `int` | `40000` | Number of episodes to sample from    |

**Example**

```python
from samsara_rl.mdp.grid_world.grid_world_mdp import GridWorldMDP
from samsara_rl.prediction.monte_carlo import MonteCarloPrediction

mdp = GridWorldMDP()
policy = mdp.random_policy()

mc = MonteCarloPrediction(mdp, policy, alpha=0.01, gamma=0.9)
mc.evaluate(max_iter=10000)

# V(s) for the random policy (expected value over actions)
v = mc.q_table.mean(axis=1).reshape(4, 4)
```

---

## Model-Free Control

TODO

---

## Function Approximation

TODO
