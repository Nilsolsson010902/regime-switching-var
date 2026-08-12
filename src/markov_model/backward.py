import numpy as np
from scipy.special import logsumexp

def backward_filtering(transition_matrix: np.ndarray, log_emission_matrix: np.ndarray) -> np.ndarray:
    """
    Run the backward algorithm in log-space.

    Parameters
    ----------
    transition_matrix:
        State transition matrix, shape (K, K).

    log_emission_matrix:
        Log emission densities, shape (T, K).

    Returns
    -------
    np.ndarray
        Log backward messages, shape (T, K).
    """

    n_obs, n_states = log_emission_matrix.shape

    log_trans = np.log(transition_matrix)

    log_beta = np.empty((n_obs, n_states))

    # At the final observation there is no future left to explain:
    # beta_T(i) = 1  ->  log(beta_T(i)) = 0
    log_beta[-1] = 0.0

    for t in range(n_obs - 2, -1, -1):
        for j in range(n_states):

            log_beta[t, j] = logsumexp(log_trans[j, :] + log_emission_matrix[t + 1, :] + log_beta[t + 1, :])  

        log_beta[t] -= logsumexp(log_beta[t])

    return log_beta
        