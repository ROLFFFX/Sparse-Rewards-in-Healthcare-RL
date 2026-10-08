# Theory experiments

Numerical checks of the paper's results on the terminal defect $Z^\pi=1-V^\pi_++V^\pi_-$: the theorems
*Hidden incentives* and *Mass transfer*, and their corollaries and propositions. Optimal policies are computed
exactly (float64 value/policy iteration and linear solves). Every inequality is checked at every non-terminal
state, for every pair of designs on a grid of $\lambda=\beta/(\alpha+\beta)$ and for three rules for breaking ties
among optimal actions (uniform, lowest index, highest index).

## Main figure

`code/exp_original_setups.py` checks both theorems on the setups of the original experiments, built with their
own code, parameters and seeds (`DAG-Based-MDP-MRP/mdp`, `LifeGate-Gridworld`, `mimic-sepsis`): 58 settings,
147 values of $\lambda$, all pass. It draws `plots/original_setups_row.pdf`:

- **(a)** terminal mass of the survival-only, mixed and death-only optimal policies: $V_+$, $-V_-$ and the
  defect $Z$, split into late survival, late death and non-termination;
- **(b)** the exchange rate $\rho$ of every adjacent pair of designs, as its position in $[\lambda,\lambda']$.

## Result → experiment

| Result | Script | Figure (`plots/supplementary/`) |
|---|---|---|
| Thm. *Hidden incentives*, Thm. *Mass transfer* on the original setups | `code/exp_original_setups.py` | `../original_setups_row.pdf` |
| Prop. *Value difference and terminal defect*; Thm. *Hidden incentives* | `code/exp_terminal_defect.py` | `theorems/terminal_defect_{lifegate,mimic}` |
| Thm. *Mass transfer*; Cor. *Zero defect and classical equivalence* ($\gamma=1$) | `code/exp_mass_transfer.py` | `theorems/mass_transfer_{lifegate,mimic}[_gamma1]` |
| Approximate optimality and sharpness of the transfer bound | `code/exp_mass_transfer.py --robustness` | `theorems/mass_transfer_robustness` |
| Cor. *Exact invariance* (iii) | `code/supplementary/exp_equal_length_dag.py` | `propositions/equal_length_dag` |
| Cor. *Exact invariance* (i), (ii) | `code/supplementary/exp_random_mdp_diagnostic.py` | `propositions/random_mdp_diagnostic` |
| Cor. *Robustness certificate* | `code/supplementary/exp_certificate.py` | `propositions/certificate_{lifegate,mimic}` |
| Prop. *Terminal scale versus step cost* | `code/supplementary/exp_terminal_scale.py` | `propositions/terminal_scale_{lifegate,mimic}` |
| Sensitivity of the MIMIC-III results to the estimated model | `code/supplementary/exp_mimic_sensitivity.py` | `propositions/mimic_sensitivity` |

Environments: random DAG MDPs (`DAG-Based-MDP-MRP/mdp`), LifeGate (`LifeGate-Gridworld/gridworld.py`, slip 0.2,
$\gamma=0.9$), MIMIC-III sepsis (750 states, 25 actions, clinician action mask, $\gamma=0.99$), and batches of
random DAGs and random MDPs.

## Running

```bash
cd Theory-Experiments
python run_all.py                  # everything
python run_all.py --skip-mimic     # without MIMIC-III
python run_all.py --theorems-only  # the two theorems (main figure and their supplementary figures)
python run_all.py --plot-only      # redraw every figure from results/
python run_all.py --only mass_transfer terminal_defect
```

Every script also takes `--plot-only`. The MIMIC-III experiments need the MDP data in `mimic-sepsis/data/`
(MIMIC-III requires credentialed access); derived per-state arrays are written to
`mimic-sepsis/data/theory_results/` and never to this folder. Results (`results/`), logs (`logs/`) and figures
(`plots/`) are generated and not tracked.

## Numerical conventions

- For $\gamma<1$, actions within $10^{-13}$ of the best normalized Q-value are optimal: tied actions differ by
  round-off ($<10^{-15}$), distinct ones by more than $10^{-11}$.
- At $\gamma=1$, planning is exact when every policy terminates almost surely. Otherwise Blackwell-optimal
  policies are approximated at $\gamma=1-10^{-10}$; since a deterministic choice among tied optimal actions can
  then cycle forever, only policies verified optimal on values enter the checks.
- Agreement within these tolerances is numerical evidence, not an exact-arithmetic proof (the sharpness
  constructions of the transfer bound are checked in rational arithmetic).

## Layout

```
Theory-Experiments/
├── run_all.py
├── code/
│   ├── exp_original_setups.py   exp_terminal_defect.py   exp_mass_transfer.py
│   ├── tabular.py               solvers, policy evaluation, tie rules
│   ├── envs.py                  DAG, LifeGate and MIMIC-III MDPs
│   ├── common.py                plotting style and output paths
│   └── supplementary/           corollaries and propositions (batched.py: batched random MDPs)
└── plots/                       generated
    ├── original_setups_row.pdf
    └── supplementary/{theorems,propositions}/
```
