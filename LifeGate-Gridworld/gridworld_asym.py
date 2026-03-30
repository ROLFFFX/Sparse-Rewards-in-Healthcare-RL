import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from gridworld import get_gridworld, value_iter

PLOT_DIR = "plots_asym"
os.makedirs(PLOT_DIR, exist_ok=True)

def calculate_policy_similarity(pi_current, pi_anchor, n=10):
    invalids = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
    survivals = [(0, 5), (0, 6), (0, 7)]
    mortalities = [(0, 8), (0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9)]
    deadends_x, deadends_y = range(5, 10), range(5, 9)

    matching_states, valid_state_count = 0, 0
    for s in range(n * n):
        x, y = int(s / n), int(s % n)
        if (x, y) in invalids or (x, y) in survivals or (x, y) in mortalities or (x in deadends_x and y in deadends_y): continue
        valid_state_count += 1

        max_p_current, max_p_anchor = np.max(pi_current[s]), np.max(pi_anchor[s])
        if np.isclose(max_p_current, 0) and np.isclose(max_p_anchor, 0):
            matching_states += 1
            continue

        if set(np.where(np.isclose(pi_current[s], max_p_current))[0]) == set(np.where(np.isclose(pi_anchor[s], max_p_anchor))[0]):
            matching_states += 1
    return matching_states / valid_state_count if valid_state_count > 0 else 0

def plot_similarity_trajectory(similarities, steps, exp_suffix=""):
    plt.figure(figsize=(10, 6))
    x_axis = np.linspace(0, 100, steps + 1)
    
    plt.plot(x_axis, similarities['seek'], label='Similarity to (1, 0) [Survival]', color='blue', linewidth=2)
    plt.plot(x_axis, similarities['mixed'], label='Similarity to (1, -1) [Mixed]', color='purple', linewidth=2)
    plt.plot(x_axis, similarities['avoid'], label='Similarity to (0, -1) [Avoidance]', color='red', linewidth=2)

    plt.axvline(x=50, color='gray', linestyle='--', label='Transition (1, -1)')
    plt.title("Policy Evolution under Asymmetric Rewards", fontsize=14)
    plt.xlabel("Interpolation Progress (%)", fontsize=12)
    plt.ylabel("Policy Similarity (Exact Action Match)", fontsize=12)
    plt.xticks(ticks=[0, 25, 50, 75, 100], labels=["(1, 0)", "(1, -0.5)", "(1, -1)", "(0.5, -1)", "(0, -1)"])
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    
    save_path = f"{PLOT_DIR}/asym{exp_suffix}.pdf"
    plt.savefig(save_path, format='pdf')
    plt.close()

def run_asym(gamma=1.0, slip_prob=0.2, intermediate_reward=0.00, steps=1000, exp_suffix=""):
    print(f"Running Asym Sweep: gamma={gamma}, slip={slip_prob}, step_cost={intermediate_reward} (Steps: {steps})")
    theta = 1e-15

    env_seek = get_gridworld(mode="mixed", slip_prob=slip_prob, pos_reward=1.0, neg_reward=0.0, intermediate_reward=intermediate_reward)
    _, pi_seek = value_iter(env_seek, theta=theta, gamma=gamma)

    env_mixed = get_gridworld(mode="mixed", slip_prob=slip_prob, pos_reward=1.0, neg_reward=-1.0, intermediate_reward=intermediate_reward)
    _, pi_mixed = value_iter(env_mixed, theta=theta, gamma=gamma)

    env_avoid = get_gridworld(mode="mixed", slip_prob=slip_prob, pos_reward=0.0, neg_reward=-1.0, intermediate_reward=intermediate_reward)
    _, pi_avoid = value_iter(env_avoid, theta=theta, gamma=gamma)

    similarities = {'seek': [], 'mixed': [], 'avoid': []}

    for i in range(steps + 1):
        alpha = i / steps
        if alpha <= 0.5:
            pos_r, neg_r = 1.0, -2.0 * alpha 
        else:
            pos_r, neg_r = 1.0 - 2.0 * (alpha - 0.5), -1.0

        env_current = get_gridworld(mode="mixed", slip_prob=slip_prob, pos_reward=pos_r, neg_reward=neg_r, intermediate_reward=intermediate_reward)
        _, pi_current = value_iter(env_current, theta=theta, gamma=gamma)

        similarities['seek'].append(calculate_policy_similarity(pi_current, pi_seek))
        similarities['mixed'].append(calculate_policy_similarity(pi_current, pi_mixed))
        similarities['avoid'].append(calculate_policy_similarity(pi_current, pi_avoid))

    plot_similarity_trajectory(similarities, steps, exp_suffix)

def run_all_asym_sweeps():
    print("Starting Comprehensive Asymmetric Sweeps...\n")
    
    # Define the parameter spaces for the sweep
    gammas = [1.0, 0.9]
    slip_probs = [0.2, 0.0]
    intermediate_rewards = [0.0, -0.04]
    
    # Generate all 8 combinations
    experiments = list(itertools.product(gammas, slip_probs, intermediate_rewards))
    print(f"Total Asymmetric Combinations to run: {len(experiments)}\n")
    
    for gamma, slip, reward in experiments:
        suffix = f"_g{gamma}_slip{slip}_step{reward}"
        
        # Determine step count based on the discount factor
        current_steps = 10000 if gamma < 1.0 else 10
        
        run_asym(
            gamma=gamma, 
            slip_prob=slip, 
            intermediate_reward=reward, 
            steps=current_steps, 
            exp_suffix=suffix
        )
        
    print("\nAll asymmetric configurations complete! Check the 'plots_asym' folder.\n")

if __name__ == "__main__":
    run_all_asym_sweeps()