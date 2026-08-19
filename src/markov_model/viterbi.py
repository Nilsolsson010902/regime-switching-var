import numpy as np

def viterbi(init_prob: np.ndarray, transition_matrix: np.ndarray, log_emission_matrix: np.ndarray) -> np.ndarray:
    """
    Run the Viterbi algorithm to find the most likely sequence of hidden states.

    Parameters
    ----------
    init_prob:
        Initial state probabilities, shape (K,).
    transition_matrix:
        State transition probabilities, shape (K, K).
    log_emission_matrix:
        Log emission probabilities, shape (T, K).

    Returns
    -------
    np.ndarray
        The most likely sequence of hidden states, shape (n_obs,).
    """
    log_trans = np.log(transition_matrix)
    log_init = np.log(init_prob)
    
    n_obs, states = log_emission_matrix.shape

    #log prob of best path thus far
    delta = np.zeros((n_obs, states))
    #Previous state that produced best path
    psi = np.empty((n_obs, states), dtype=int)

    # Initialization
    delta[0]= log_init + log_emission_matrix[0]

    for t in range(1, n_obs):
        for j in range(states):
                candidates = delta[t-1] + log_trans[:, j]
                delta[t, j] = np.max(candidates) + log_emission_matrix[t, j]
                psi[t, j] = np.argmax(candidates)

    # Backtracking
    best_path = np.empty(n_obs, dtype=int)
    best_path[-1] = np.argmax(delta[-1])

    for t in range(n_obs - 1, 0, -1):
        best_path[t - 1] = psi[t, best_path[t]]
        
    return best_path