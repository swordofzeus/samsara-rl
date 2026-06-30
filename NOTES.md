# Samsara RL

This library provides a vectorized implementation and explainations of foundational reinforcement learning algorithms. Reinforcement learning deals with the science of decision making; a learning paradigm of repeated trial and error used in many areas such as:
    I.  Robotic manipulation
    II. Fine tuning LLMs
    III. Financial portfolio management
    IV. Adaptive electric grids
There is a close intersection but often lesser studied intersection between information theory and reinforcement learning that I try and highlight in the algorithm explainations. Rate distortion theory for example an area of information theory studies the exact problem of optimal control but through the lense of noisy compression. Compression and learning are two sides of the same problem studied from different lenses.

The algorithms and notes were based off my readings of the Sutton and Barto book, the david silver deep mind RL lectures, as well as numerous other papers and resources that I cite in appropriate sections where they show up.

---

## Table of Contents


1. [Concept and Intuition](docs/01_concept_and_intuition.md)
2. [Bellman Equations](docs/02_kraft_and_prefix_codes.md)
3. [Planning Algorithms](docs/03_formalizing_surprise.md)
4. [Optimal Substructure & Unknown Dynamics](docs/04_surprise_and_code_length.md)
5. [Noisy Estimators I: Monte Carlo](docs/05_binomial_derivation.md)
6. [Noisy Estimators II: TD Learning](docs/05_binomial_derivation.md)
7. [Balancing error with eligibility Traces](docs/05_binomial_derivation.md)
8. [Optimal Control with SARSA](docs/05_binomial_derivation.md)
9. [Optimal Control with Q Learning](docs/05_binomial_derivation.md)