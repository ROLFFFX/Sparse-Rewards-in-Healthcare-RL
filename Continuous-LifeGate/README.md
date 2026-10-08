# Continuous LifeGate

The $10\times10$ LifeGate gridworld (`LifeGate-Gridworld/gridworld.py`) in continuous space, with PPO policies
learned under three reward designs. Patients can stall forever and $\gamma<1$, so the terminal defect does not
vanish and the designs lead to different behavior: a survival reward discharges patients, a death penalty
keeps them alive without discharging them, and a step cost sends patients far from discharge into the dead-ends.

## Environment (`code/lifegate_continuous.py`)

Arena $[0,10]^2$, regions as in LifeGate:

| Region | Position | Effect |
|---|---|---|
| exit | $x\in[5,8)$, $y\ge9$ | discharge, episode ends |
| death | $x\ge9$, or $x\ge8$ and $y\ge9$ | death, episode ends |
| dead-ends | $x\in[5,9)$, $y<5$ | death on entry |
| walls | $x<5$ and $y\ge9$; $x\in[1,5)$ and $y\in[4,5)$ | not entered |

Patients below the wall reach the exit only through the gap $x<1$. A step moves the patient by
$0.5\,a+(0.1,0)+0.05\,\varepsilon$, with the action $a$ clipped to the unit disk, a rightward drift and
$\varepsilon\sim\mathcal N(0,I_2)$; a move into a wall or out of the arena is not made. $\gamma=0.97$, horizon 200
steps; initial positions are uniform on the free space. Observation: $(x/10,\ y/10,\ \text{remaining time})$.

| Design | Exit | Death | Per step |
|---|---|---|---|
| `survival` | $+1$ | $0$ | $0$ |
| `death` | $0$ | $-1$ | $0$ |
| `stepcost` | $+1$ | $0$ | $-0.04$ |

## Scripts

| Script | Purpose |
|---|---|
| `code/plan_check.py` | value iteration on a $0.1$ grid: the optimal behavior under each design, before any learning |
| `code/train.py` | PPO (Stable-Baselines3), 20M transitions, entropy coefficient 0.01, seeds 31000–31004; 5,000 deterministic test episodes per seed (test seeds 94000–94004, paired across designs) |
| `code/plot_figure.py` | the figure `plots/continuous_lifegate.pdf` and `results/summary.json` (outcome rates over seeds, for all test patients and for those starting below the wall) |

```bash
cd Continuous-LifeGate/code
python plan_check.py
for d in survival death stepcost; do for k in 0 1 2 3 4; do python train.py --design $d --fold $k; done; done
python plot_figure.py                   # starts on the whole free space
python plot_figure.py --under-wall --out continuous_lifegate_under_wall
```

`python train.py --design survival --fold 0 --ent 0 --tag _no_entropy` trains without the entropy bonus; the
policy then does not learn the detour around the wall. Trained models, evaluations (`results*/`) and figures
(`plots/`) are generated and not tracked. The figure shows training seed 2 (the behavior of most seeds) with
deterministic actions, drift and noise; paths are smoothed for display only.

## Results (5 seeds, mean ± standard deviation, %)

| Design | Exit | Death | Unresolved after 200 steps |
|---|---|---|---|
| survival | 99.04 ± 0.13 | 0.96 ± 0.13 | 0.00 ± 0.01 |
| death | 1.57 ± 3.18 | 0.06 ± 0.13 | 98.37 ± 3.16 |
| stepcost | 65.23 ± 0.55 | 34.58 ± 0.51 | 0.18 ± 0.07 |

For patients starting below the wall: the survival design discharges 99.99%, the death design leaves 99.81%
unresolved, and the step-cost design sends all of them into the dead-ends. The exact optimum of the step-cost
design (`plan_check.py`) abandons only the patients farthest from the gap (10.2% of all patients); PPO abandons
every patient below the wall.
