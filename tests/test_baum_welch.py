import numpy as np
from src.markov_model.baum_welch import baum_welch_step

import numpy as np

# Initial probabilities
initial_probabilities = np.array([
    0.5,
    0.5
])

# Transition matrix
transition_matrix = np.array([
    [0.95, 0.05],
    [0.20, 0.80]
])

#emission matrix
emission_matrix = np.array([
    [0.90, 0.10],
    [0.20, 0.80],
    [0.10, 0.90]
])

log_emission_matrix = np.log(emission_matrix)

#feature matrix 
feature_matrix = np.array([
    [-0.005,  0.010, 0.015],
    [-0.025, -0.020, 0.040],
    [-0.030, -0.035, 0.050]
])

output = baum_welch_step(
        initial_probabilities=initial_probabilities,
        transition_matrix=transition_matrix,
        log_emission_matrix=log_emission_matrix,
        feature_matrix=feature_matrix
    )

def test_baum_welch_probability_constraints():
    """
    Test that the output of the Baum-Welch algorithm satisfies probability constraints.
    """
    assert np.isclose(output.initial_probabilities.sum(),1.0)
    assert np.allclose(output.transition_matrix.sum(axis=1), 1.0)

    gamma = np.exp(output.log_gamma)
    xi = np.exp(output.log_xi)

    assert np.allclose(gamma.sum(axis=1),1.0)
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0)

def test_baum_welch_output_shapes():
    """ 
    Test that the output of the Baum-Welch algorithm has the correct shapes.
    """
    n_obs, n_features = feature_matrix.shape
    n_states = len(initial_probabilities)

    assert output.initial_probabilities.shape == (n_states,)
    assert output.transition_matrix.shape == (n_states, n_states)
    assert output.means.shape == (n_states, n_features)
    assert output.covariances.shape == (
        n_states,
        n_features,
        n_features
    )
    assert output.log_gamma.shape == (n_obs, n_states)
    assert output.log_xi.shape == (
        n_obs - 1,
        n_states,
        n_states
    )

def test_baum_welch_outputs_are_finite():
    """
    Test that all outputs of the Baum-Welch algorithm are finite."""
    arrays = [
        output.initial_probabilities,
        output.transition_matrix,
        output.means,
        output.covariances,
        output.log_gamma,
        output.log_xi
    ]

    for array in arrays:
        assert np.all(np.isfinite(array))


def test_xi_gamma_consistency():
    """
    Test that the xi and gamma outputs of the Baum-Welch algorithm are consistent.
    """
    gamma = np.exp(output.log_gamma)
    xi = np.exp(output.log_xi)

    assert np.allclose(
        xi.sum(axis=2),
        gamma[:-1],
        atol=1e-8
    )

    assert np.allclose(
        xi.sum(axis=1),
        gamma[1:],
        atol=1e-8
    )