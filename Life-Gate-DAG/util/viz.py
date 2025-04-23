import numpy as np
import matplotlib.pyplot as plt
import pprint

def plot_v_plus_minus(v_plus, v_minus, title):
    """
    Visualizes V+ and V- as a scatter plot.

    Parameters:
    v_plus (np.ndarray): Array of V+ values.
    v_minus (np.ndarray): Array of V- values.

    Note:
    V+ and V- should satisfy the property V+ - V- = 1.
    Each dot would represent a state in the MRP such that: (x, y) = (V-(s), V+(s)).
    """
    if not isinstance(v_plus, np.ndarray) or not isinstance(v_minus, np.ndarray):
        raise ValueError("Both V+ and V- must be numpy arrays.")
    
    if v_plus.shape != v_minus.shape:
        raise ValueError("V+ and V- must have the same shape.")
    
    # exluding terminal states
    v_plus = v_plus[2:]
    v_minus = v_minus[2:]
    
    # pprint.pp(v_plus)
    # pprint.pp(v_minus)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(v_minus, v_plus, color='#4965b0', label='V+ vs V-')
    plt.plot(v_minus, v_minus - 1, color='#f36f43', linestyle='--', label='V+ - V- = 1')
    plt.xlabel('V-')
    plt.ylabel('V+')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()