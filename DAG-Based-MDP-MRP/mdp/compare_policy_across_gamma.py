import matplotlib.pyplot as plt
import pandas as pd
from lifegate_mdp import LifeGate_MDP, compare_pis
import numpy as np

# Parameters
gammas = [1.0, 0.9, 0.8, 0.7]

seeds = [20, 28, 18, 97, 42, 1018]

records = []

# Run comparisons
for seed in seeds:
    for gamma in gammas:
        mdp_plus = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=seed, mode="+")
        mdp_minus = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=seed, mode="-")
        mdp_mixed = LifeGate_MDP(max_num_states=200, p_terminal=0.3, max_actions=2, seed=seed, mode="+-")

        _, pi_plus = mdp_plus.policy_iter_deter(gamma=gamma)
        _, pi_minus = mdp_minus.policy_iter_deter(gamma=gamma)
        _, pi_mixed = mdp_mixed.policy_iter_deter(gamma=gamma)

        records.append({
            "Seed": str(seed),
            "Gamma": gamma,
            "π+ vs π−": compare_pis(pi_plus, pi_minus, mdp_plus, mdp_minus),
            "π+ vs π±": compare_pis(pi_plus, pi_mixed, mdp_plus, mdp_mixed),
            "π− vs π±": compare_pis(pi_minus, pi_mixed, mdp_minus, mdp_mixed)
        })


print(pd.DataFrame(records))

df = pd.DataFrame(records)

fig, axes = plt.subplots(2, 3, figsize=(9, 5))
axes = axes.flatten()

for idx, seed in enumerate(seeds):
    ax = axes[idx]
    df_seed = df[df["Seed"] == str(seed)].sort_values("Gamma", ascending=False)
    x = np.arange(len(gammas))
    width = 0.25

    ax.bar(x - width, df_seed["π+ vs π−"], width=width, label="π+ vs π−", color='#66c2a5')
    ax.bar(x, df_seed["π+ vs π±"], width=width, label="π+ vs π±", color='#fc8d62')
    ax.bar(x + width, df_seed["π− vs π±"], width=width, label="π− vs π±", color= '#8da0cb')

    ax.set_xticks(x)
    ax.set_xticklabels(df_seed["Gamma"])
    ax.set_title(f"Seed {seed}")
    ax.set_ylabel("Mismatch Count")
    ax.set_xlabel("Gamma")
    ax.legend()

plt.tight_layout()
plt.show()