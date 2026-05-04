import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from gridworld import get_gridworld, value_iter

plt.rcParams.update({
    'font.size': 18,          # Base global font size
    'axes.titlesize': 22,     # Main title size
    'axes.labelsize': 18,     # X and Y axis label size
    'xtick.labelsize': 16,    # X axis tick labels
    'ytick.labelsize': 16,    # Y axis tick labels
    'legend.fontsize': 14,    # Legend text size
    'lines.linewidth': 3,     # Thicker lines so they don't vanish when scaled
    'figure.autolayout': True # Prevents labels from getting cut off at the edges
})

PLOT_DIR = "plots_scaling"
os.makedirs(PLOT_DIR, exist_ok=True)

def count_death_preferences(policy, n=10):
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    mortalities = [(0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]
    invalids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
    survivals = [(0, 5), (0, 6), (0, 7)]
    deadends_x, deadends_y = range(5, 10), range(5, 9)

    def dfs(s, visited):
        x, y = int(s / n), int(s % n)
        state_coord = (x, y)
        if state_coord in mortalities: return True
        if state_coord in survivals or state_coord in invalids or s in visited: return False
        
        visited.add(s)
        if x in deadends_x and y in deadends_y:
            leads_to_death = dfs(99, visited)
            visited.remove(s)
            return leads_to_death

        max_prob = np.max(policy[s])
        if np.isclose(max_prob, 0):
            visited.remove(s)
            return False

        best_actions = [a for a, p in enumerate(policy[s]) if np.isclose(p, max_prob)]
        leads_to_death = False
        for a in best_actions:
            dx, dy = actions[a]
            target_x, target_y = x + dx, y + dy
            if target_x < 0 or target_x >= n or target_y < 0 or target_y >= n or (target_x, target_y) in invalids: continue 
            if dfs(target_x * n + target_y, visited):
                leads_to_death = True
                break 
        visited.remove(s)
        return leads_to_death

    death_preference_count = 0
    for s in range(n * n):
        x, y = int(s / n), int(s % n)
        if (x, y) in invalids or (x, y) in survivals or (x, y) in mortalities or (x in deadends_x and y in deadends_y): continue
        if dfs(s, set()): death_preference_count += 1
    return death_preference_count

def gridworld_scaling():
    gammas = np.linspace(0.5, 0.99, 20) 
    step_costs = np.linspace(-0.15, 0.0, 20) 
    
    deaths_plus = np.zeros((len(gammas), len(step_costs)))
    deaths_minus = np.zeros((len(gammas), len(step_costs)))
    deaths_mixed = np.zeros((len(gammas), len(step_costs)))
    
    print("Running Scaling Sweep (this may take a minute)...")
    for i, g in enumerate(gammas):
        for j, r in enumerate(step_costs):
            env_plus = get_gridworld(mode="survival", slip_prob=0.2, intermediate_reward=r)
            env_minus = get_gridworld(mode="avoidance", slip_prob=0.2, intermediate_reward=r)
            env_mixed = get_gridworld(mode="mixed", slip_prob=0.2, intermediate_reward=r)
            
            _, pi_plus = value_iter(env_plus, gamma=g)
            _, pi_minus = value_iter(env_minus, gamma=g)
            _, pi_mixed = value_iter(env_mixed, gamma=g)
            
            deaths_plus[i, j] = count_death_preferences(pi_plus)
            deaths_minus[i, j] = count_death_preferences(pi_minus)
            deaths_mixed[i, j] = count_death_preferences(pi_mixed)

    def plot_heatmap(data, title, filename):
        plt.figure(figsize=(7, 5))
        x_labels = [f"{val:.3f}" for val in step_costs]
        y_labels = [f"{val:.2f}" for val in gammas]
        
        min_val, max_val = 0, 57 
        cmap = plt.get_cmap("magma", max_val - min_val + 1)
        bounds = np.arange(min_val, max_val + 2) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        ax = sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, norm=norm, annot=False)
        ax.invert_yaxis()
        
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_xticklabels(x_labels[::2], rotation=45)
        ax.set_yticks(ax.get_yticks()[::2])
        ax.set_yticklabels(y_labels[::2], rotation=0)

        cbar = ax.collections[0].colorbar
        cbar.set_ticks(np.arange(min_val, max_val + 1, 5))
        cbar.set_label('States Transitioning to Death')

        plt.title(title)
        plt.xlabel("Intermediate Reward (Step Cost)")
        plt.ylabel("Gamma")
        
        # tight_layout is also handled globally now, but keeping it here is safe
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/{filename}.pdf")
        plt.close()

    plot_heatmap(deaths_plus, "Death Preference Count (Survival)", "scaling_deaths_plus")
    plot_heatmap(deaths_minus, "Death Preference Count (Avoidance)", "scaling_deaths_minus")
    plot_heatmap(deaths_mixed, "Death Preference Count (Mixed)", "scaling_deaths_mixed")
    print("Scaling Sweeps Complete.\n")

if __name__ == "__main__":
    gridworld_scaling()
