import numpy as np
from src.markov_model.backward import backward_filtering
import pytest

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

def test_backward_filtering( transition_matrix, log_emission_matrix):
    """
    Test the backward filtering algorithm with a simple example.
    """
    log_beta = backward_filtering(
            transition_matrix=transition_matrix,
            log_emission_matrix=log_emission_matrix
        )

    beta = np.exp(log_beta)

    expected_day_2 = np.array([
            0.14 / (0.14 + 0.74),
            0.74 / (0.14 + 0.74)
        ])

    assert np.allclose(
            beta[1],
            expected_day_2,
            atol=1e-3
        )
def test_backward_final_row_is_one(log_emission_matrix, transition_matrix):

    log_beta = backward_filtering(
        transition_matrix,
        log_emission_matrix
    )

    beta = np.exp(log_beta)

    assert np.allclose(
        beta[-1],
        np.ones(2)
    )

def test_backward_output_shape(transition_matrix,log_emission_matrix):
    """
    Test that the output of the backward filtering algorithm has the correct shape.
    """
    log_alpha = backward_filtering(
        transition_matrix,
        log_emission_matrix
    )

    assert log_alpha.shape == (3, 2)



def test_forward_values_are_finite( transition_matrix, log_emission_matrix):
    """
    Test that the backward probabilities are finite.
    """
    log_alpha = backward_filtering(
        transition_matrix,
        log_emission_matrix
    )

    assert np.all(np.isfinite(log_alpha))