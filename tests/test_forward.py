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
def log_alpha():
    return forward_filtering(initial_probability, transition_matrix, log_emission_matrix).log_alpha

def test_forward_filtering(log_alpha):
    """
    Test the forward filtering algorithm with a simple example.
    """
    alpha = np.exp(log_alpha)
    expected = np.array([
        [0.9000, 0.1000],
        [0.6364, 0.3636],
        [0.1889, 0.8111]
    ])

    assert np.allclose(alpha, expected, atol=1e-3)

def test_forward_probabilities_sum_to_one(log_alpha):
    """
    Test that the forward probabilities sum to 1 at each time step.
    """

    alpha = np.exp(log_alpha)

    assert np.allclose(
        alpha.sum(axis=1),
        1.0
    )

def test_forward_output_shape(log_alpha):
    """
    Test that the output of the forward filtering algorithm has the correct shape.
    """
    assert log_alpha.shape == (3, 2)



def test_forward_values_are_finite(log_alpha):
    """
    Test that the forward probabilities are finite.
    """

    assert np.all(np.isfinite(log_alpha))