import numpy as np
from src.markov_model.forward import forward_filtering
import pytest


@pytest.fixture
def  initial_probability():
    return np.array([0.5, 0.5])

@pytest.fixture
def transition_matrix():
    return np.array([
        [0.95, 0.05],
        [0.20, 0.80]
    ])

@pytest.fixture
def log_emission_matrix():
    return np.log(np.array([
        [0.90, 0.10],
        [0.20, 0.80],
        [0.10, 0.90]
    ]))

@pytest.fixture
def forward_output(initial_probability, transition_matrix, log_emission_matrix):
    return forward_filtering(initial_probability, transition_matrix, log_emission_matrix)

def test_forward_filtering(forward_output):
    """
    Test the forward filtering algorithm with a simple example.
    """
    alpha = np.exp(forward_output.log_alpha)
    expected = np.array([
        [0.9000, 0.1000],
        [0.6364, 0.3636],
        [0.1889, 0.8111]
    ])

    assert np.allclose(alpha, expected, atol=1e-3)

def test_forward_probabilities_sum_to_one(forward_output):
    """
    Test that the forward probabilities sum to 1 at each time step.
    """

    alpha = np.exp(forward_output.log_alpha)

    assert np.allclose(
        alpha.sum(axis=1),
        1.0
    )

def test_forward_output_shape(forward_output):
    """
    Test that the output of the forward filtering algorithm has the correct shape.
    """
    assert forward_output.log_alpha.shape == (3, 2)
    assert isinstance(forward_output.log_likelihood, float)


def test_forward_values_are_finite(forward_output):
    """
    Test that the forward probabilities are finite.
    """

    assert np.all(np.isfinite(forward_output.log_alpha))
    assert np.isfinite(forward_output.log_likelihood)

def test_forward_log_likelihood_matches_bruteforce(initial_probability, transition_matrix, log_emission_matrix, forward_output):
    total_probability = 0.0
    emission_matrix = np.exp(log_emission_matrix)
    # Enumerate all 2^3 possible hidden-state paths
    for s0 in range(2):
        for s1 in range(2):
            for s2 in range(2):

                path_probability = (
                    initial_probability[s0]
                    * emission_matrix[0, s0]
                    * transition_matrix[s0, s1]
                    * emission_matrix[1, s1]
                    * transition_matrix[s1, s2]
                    * emission_matrix[2, s2]
                )

                total_probability += path_probability

    expected_log_likelihood = np.log(total_probability)

    assert np.isclose(
        forward_output.log_likelihood,
        expected_log_likelihood,
        atol=1e-10)