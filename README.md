# Between Life and Death: Examining Failure Modes of Sparse Reward Designs in Healthcare RL

This repository contains the code and experiments for the paper titled above.

## Repository Structure

The codebase is organized into five components:

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
* **`gridworld_drift.py`**, **`run_drift.py`**: LifeGate with the original dead-end dynamics (in a dead-end state, every action moves right with probability 0.7 until death).

### 3. `mimic-sepsis/` (Clinical Application)
Real-world sepsis treatment experiments using the MIMIC-III dataset.
* **`pipeline/`**: The core analysis notebooks.
    * `1-env_model_variants.ipynb`: Construction of the MDP from patient data.
    * `2-value_iter.ipynb` & `3-policy_iter.ipynb`: Solving for optimal policies under $\mathcal{M}_+, \mathcal{M}_-, \mathcal{M}_{\pm}$.
    * `4-check_propositions.ipynb`: Empirical verification of value function identities on real clinical data.
    * `6-viz.ipynb`: Generates the policy disagreement and density plots used in the paper.

### 4. `Theory-Experiments/` (Terminal Defect)
Numerical checks of the theorems *Hidden incentives* and *Mass transfer* and their corollaries, on the setups of components 1–3.
* **`code/exp_original_setups.py`**: Both theorems on the original setups (main theory figure).
* **`code/exp_terminal_defect.py`**, **`code/exp_mass_transfer.py`**: The two theorems on LifeGate and MIMIC-III in detail.
* **`code/supplementary/`**: Corollaries and propositions.
* **`run_all.py`**: Runs every experiment. See [Theory-Experiments/README.md](Theory-Experiments/README.md).

### 5. `Continuous-LifeGate/` (Learned Continuous-Control Policies)
LifeGate in continuous space with PPO policies learned under survival, death-penalty and step-cost rewards.
* **`code/lifegate_continuous.py`**: The environment.
* **`code/plan_check.py`**: Optimal behavior under each design by value iteration.
* **`code/train.py`**, **`code/plot_figure.py`**: PPO training (five seeds) and the trajectory figure. See [Continuous-LifeGate/README.md](Continuous-LifeGate/README.md).

---

## Getting Started

### Prerequisites
* Python 3.10+
* Standard RL and Data Science libraries (see `requirements.txt`)

```bash
pip install -r requirements.txt
