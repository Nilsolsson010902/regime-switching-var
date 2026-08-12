import numpy as np
from forward import forward_filtering
from backward import backward_filtering
from scipy.special import logsumexp
from dataclasses import dataclass



@dataclass
class BaumWelchOutput:
    """
    Dataclass to hold the output of the Baum-Welch algorithm.
    """
    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    log_gamma: np.ndarray
    log_xi: np.ndarray

def baum_welch_step(
                initial_probabilities: np.ndarray, 
                transition_matrix: np.ndarray, 
                log_emission_matrix: np.ndarray, 
                feature_matrix: np.ndarray) -> BaumWelchOutput:
    """
    Run the baum-welch algorithm in log-space.

    Parameters
    ----------
    initial_probabilities:
        Initial state probabilities, shape (K,).

    transition_matrix:
        State transition matrix, shape (K, K).

    log_emission_matrix:
        Log emission densities, shape (T, K).
    
    feature_matrix:
        Feature observations, shape (T, K)

    Returns
    -------
    BaumWelchOutput
        A dataclass containing the updated parameters and log probabilities.
    """
    n_obs, n_states = log_emission_matrix.shape
    log_trans = np.log(transition_matrix)
    log_obs = np.log(feature_matrix)

    log_alpha = forward_filtering(initial_probabilities=initial_probabilities, transition_matrix=transition_matrix, log_emission_matrix=log_emission_matrix).log_alpha
    log_beta = backward_filtering(transition_matrix=transition_matrix, log_emission_matrix=log_emission_matrix)
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

    log_xi = np.empty((n_obs -1, n_states, n_states))
    for t in range(n_obs - 1):
        for i in range(n_states):
            for j in range (n_states):
                log_xi[t][i][j] = log_alpha[t][i] + log_trans[i][j]+log_emission_matrix[t +1][j] + log_beta[t+1][j]
        log_xi[t] -= logsumexp(log_xi[t])


    gamma = np.exp(log_gamma)
    xi = np.exp(log_xi)
    
    #updating initial probabilities for the HMM
    new_initial_prob = gamma[0]
  
    #updating new transition probabilities for HMM
    new_transition_matrix = (np.sum(xi, axis=0)/  np.sum(gamma[:-1], axis=0))
    
    new_means = np.empty((n_states, feature_matrix.shape[1]))
    #updating new means for the HMM
    for state in range(n_states):
        weights = gamma[:, state][:, None]
        weighted_sum = np.sum(weights * feature_matrix, axis=0)
        total_weight = np.sum(gamma[:, state])

        new_means[state] = weighted_sum / total_weight


    new_covariances = np.empty((n_states, feature_matrix.shape[1], feature_matrix.shape[1]))
    #updating new covariance matrix for the HMM
    for state in range(n_states):
        weights = gamma[:, state]
        residuals = feature_matrix - new_means[state]

        outer_products = (residuals[:, :, None] * residuals[:, None, :])
        weighted_outer_products = (weights[:, None, None] * outer_products)
        total_weight = np.sum(weights)

        new_covariances[state] = (np.sum(weighted_outer_products, axis=0)/ total_weight)


    
    return BaumWelchOutput(
        initial_probabilities=new_initial_prob,
        transition_matrix=new_transition_matrix,
        means=new_means,
        covariances=new_covariances,
        log_gamma=log_gamma,
        log_xi=log_xi
    )