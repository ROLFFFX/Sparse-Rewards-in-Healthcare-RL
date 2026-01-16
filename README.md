# Between Life and Death: Examining Sparse Reward Designs in Healthcare RL

This repository contains the code and experiments for the paper **"Between Life and Death: Examining Sparse Reward Designs in Healthcare RL"**.

## Repository Structure

The codebase is organized into three main components corresponding to the experimental sections of the paper:

### 1. `DAG-Based-MDP-MRP/` (Synthetic Validation)
Procedurally generated Directed Acyclic Graph (DAG) environments used to valid theoretical propositions.
* **`mdp/`**: Code for creating random DAG MDPs and running Policy Iteration.
    * `run_large_mdp.py`: Verifies value function identities on large state spaces.
    * `break_*.py`: Experiments that systematically violate assumptions (discounting $\gamma < 1$, non-absorbing states, dense rewards).
* **`mrp/`**: Simplified Markov Reward Process experiments to test state-value ($V$) differences.

### 2. `LifeGate-Gridworld/` & `Deterministic/` (Interpretability)
A clinically-inspired $10 \times 10$ grid world ("LifeGate") variant visualized in the paper. Originally from: https://arxiv.org/abs/2110.04186
* **`gridworld.py`**: The main environment definition.
* **`*_policy.pdf`**: Visualizations of the learned optimal policies under different reward designs.
* **`cross_validation.py`**: Scripts to evaluate policy robustness (in the deterministic folder).

### 3. `mimic-sepsis/` (Clinical Application)
Real-world sepsis treatment experiments using the MIMIC-III dataset.
* **`pipeline/`**: The core analysis notebooks.
    * `1-env_model_variants.ipynb`: Construction of the MDP from patient data.
    * `2-value_iter.ipynb` & `3-policy_iter.ipynb`: Solving for optimal policies under $\mathcal{M}_+, \mathcal{M}_-, \mathcal{M}_{\pm}$.
    * `4-check_propositions.ipynb`: Empirical verification of value function identities on real clinical data.
    * `6-viz.ipynb`: Generates the policy disagreement and density plots used in the paper.

---

## Getting Started

### Prerequisites
* Python 3.10+
* Standard RL and Data Science libraries (see `requirements.txt`)

```bash
pip install -r requirements.txt
