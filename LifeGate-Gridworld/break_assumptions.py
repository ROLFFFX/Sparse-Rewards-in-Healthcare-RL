import os
import itertools
from gridworld import (
    get_gridworld, value_iter, compute_Q, plot_highlighted_policies, 
    plot_value_heatmap, add_Vs, subtract_Vs, combine_Qs, visualize_gridworld
)

PLOT_DIR = "plots_break_asms"
os.makedirs(PLOT_DIR, exist_ok=True)

def run_experiment(gamma, slip_prob, intermediate_reward, pos_reward=1.0, neg_reward=-1.0, exp_suffix=""):
    theta = 1e-15
    print(f"Running Base Exp: r={intermediate_reward}, gamma={gamma}, slip={slip_prob}")
    
    survival = get_gridworld(mode="survival", slip_prob=slip_prob, pos_reward=pos_reward, neg_reward=neg_reward, intermediate_reward=intermediate_reward)
    avoidance = get_gridworld(mode="avoidance", slip_prob=slip_prob, pos_reward=pos_reward, neg_reward=neg_reward, intermediate_reward=intermediate_reward)
    mixed = get_gridworld(mode="mixed", slip_prob=slip_prob, pos_reward=pos_reward, neg_reward=neg_reward, intermediate_reward=intermediate_reward)

    s_V, s_pi = value_iter(survival, theta=theta, gamma=gamma)
    s_Q = compute_Q(s_pi, survival, gamma, theta)

    a_V, a_pi = value_iter(avoidance, theta=theta, gamma=gamma)
    a_Q = compute_Q(a_pi, avoidance, gamma, theta)

    m_V, m_pi = value_iter(mixed, theta=theta, gamma=gamma)
    m_Q = compute_Q(m_pi, mixed, gamma, theta)

    path_s = f'{PLOT_DIR}/plus_policy_sp{slip_prob:.1f}g{gamma:.1f}{exp_suffix}.pdf'
    path_a = f'{PLOT_DIR}/minus_policy_sp{slip_prob:.1f}g{gamma:.1f}{exp_suffix}.pdf'
    path_m = f"{PLOT_DIR}/mixed_policy_sp{slip_prob:.1f}g{gamma:.1f}{exp_suffix}.pdf"

    plot_highlighted_policies([s_pi, a_pi, m_pi], [path_s, path_a, path_m])

    s_plus_a = add_Vs(s_V, a_V)
    sq_plus_aq = combine_Qs(s_Q, a_Q, "+")

    subtracted = subtract_Vs(m_V, s_plus_a)
    plot_value_heatmap(subtracted, save_path=f"{PLOT_DIR}/gridworld_add_sp{slip_prob:.1f}g{gamma:.1f}{exp_suffix}.pdf")

    s_minus_a = subtract_Vs(s_V, a_V)
    plot_value_heatmap(s_minus_a, save_path=f"{PLOT_DIR}/gridworld_subtract_sp{slip_prob:.1f}g{gamma:.1f}{exp_suffix}.pdf")

def gridworld_break_asms():
    visualize_gridworld(f"{PLOT_DIR}/gridworld_layout.pdf")
    
    gammas = [1.0, 0.9]
    slip_probs = [0.2, 0.0]
    intermediate_rewards = [-0.04, 0.00]
    
    experiments = list(itertools.product(gammas, slip_probs, intermediate_rewards))
    print(f"Total Base Combinations: {len(experiments)}")
    
    for gamma, slip, reward in experiments:
        suffix = f"_g{gamma}_slip{slip}_step{reward}"
        run_experiment(gamma=gamma, slip_prob=slip, intermediate_reward=reward, exp_suffix=suffix)
    print("Base Assumptions Complete.\n")

if __name__ == "__main__":
    gridworld_break_asms()