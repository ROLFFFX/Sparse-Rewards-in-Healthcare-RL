import os
import itertools
import matplotlib.pyplot as plt
from gridworld import (
    get_gridworld, value_iter, compute_Q, plot_highlighted_policies, 
    plot_value_heatmap, add_Vs, subtract_Vs, combine_Qs, visualize_gridworld
)

PLOT_DIR = "plots_break_asms"
os.makedirs(PLOT_DIR, exist_ok=True)

def run_experiment(gamma, slip_prob, intermediate_reward, pos_reward=1.0, neg_reward=-1.0):
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

    # --- CLEANEST NAMING CONVENTION ---
    suffix = f"_sp{slip_prob:.1f}_g{gamma:.1f}_step{intermediate_reward:.2f}.pdf"

    path_s = f'{PLOT_DIR}/plus_policy{suffix}'
    path_a = f'{PLOT_DIR}/minus_policy{suffix}'
    path_m = f"{PLOT_DIR}/mixed_policy{suffix}"

    plot_highlighted_policies(
        [s_pi, a_pi, m_pi], 
        [path_s, path_a, path_m],
        titles=[r"$\pi_{+}^{*}$", r"$\pi_{-}^{*}$", r"$\pi_{\pm}^{*}$"]
    )

    s_plus_a = add_Vs(s_V, a_V)
    sq_plus_aq = combine_Qs(s_Q, a_Q, "+")

    subtracted = subtract_Vs(m_V, s_plus_a)
    plot_value_heatmap(
        subtracted, 
        title=r"$V_{\pm}(s) - (V_{+}(s) + V_{-}(s))$", 
        save_path=f"{PLOT_DIR}/gridworld_add{suffix}"
    )

    s_minus_a = subtract_Vs(s_V, a_V)
    plot_value_heatmap(
        s_minus_a, 
        title=r"$V_{+}(s) - V_{-}(s)$", 
        save_path=f"{PLOT_DIR}/gridworld_subtract{suffix}"
    )

def gridworld_break_asms():
    plt.rcdefaults() 
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm'
    })
    
    visualize_gridworld(f"{PLOT_DIR}/gridworld_layout.pdf")
    
    gammas = [1.0, 0.9]
    slip_probs = [0.2, 0.0]
    intermediate_rewards = [-0.04, 0.00]
    
    experiments = list(itertools.product(gammas, slip_probs, intermediate_rewards))
    print(f"Total Base Combinations: {len(experiments)}")
    
    for gamma, slip, reward in experiments:
        run_experiment(gamma=gamma, slip_prob=slip, intermediate_reward=reward)
    print("Base Assumptions Complete.\n")

if __name__ == "__main__":
    gridworld_break_asms()
