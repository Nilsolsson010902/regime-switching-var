import numpy as np
from scipy.special import logsumexp


def forward_filtering(initial_probabilities: np.ndarray, transition_matrix: np.ndarray, log_emission_matrix: np.ndarray) -> np.ndarray:
    """
    Run the forward filtering algorithm in log-space.

    Parameters
    ----------
    initial_probabilities:
        Initial state probabilities, shape (K,).

    transition_matrix:
        State transition matrix, shape (K, K).

    log_emission_matrix:
        Log emission densities, shape (T, K).

    Returns
    -------
    np.ndarray
        Filtered log state probabilities, shape (T, K).
    """

    n_obs, n_states = log_emission_matrix.shape

    log_trans = np.log(transition_matrix)
    log_initial = np.log(initial_probabilities)

    log_alpha = np.empty((n_obs, n_states))

    # Initial step: t = 0
    log_alpha[0] = (
        log_initial
        + log_emission_matrix[0]
    )

    # Normalize
    log_alpha[0] -= logsumexp(log_alpha[0])

    for t in range(1, n_obs):
        for j in range(n_states):

            log_prediction = logsumexp(log_alpha[t-1] + log_trans[:, j])

            log_alpha[t, j] = (log_prediction + log_emission_matrix[t, j])          

        log_alpha[t] -= logsumexp(log_alpha[t])

    return log_alpha
        